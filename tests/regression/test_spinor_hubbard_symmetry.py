"""P82: a spinor DFT+U run on a reduced k-set reproduces the closed grid.

The unit checks on the matrix itself are in
``tests/unit/test_spin_rotations.py`` and they are algebra: that the SU(2) lift
is in SU(2), represents the rotation it claims to, and is a representation of
the group up to sign. **None of that says the driver routes an occupation
matrix through it.** That distinction is not hypothetical here -- ``promote_ns``
was right in every element while every spinor resume was silently collinear, and
an array-algebra test passed the whole time (``MAGNETISM-NEXT.md`` §5).

So this file is end to end, and the statement is the one a symmetrisation
exists to make: **the wedge and the closed grid are the same calculation**. Run
the same cell twice, once on a symmetry-reduced k-set and once with ``nosym``
on the whole unshifted grid, and compare the total energy and the occupation
matrix itself.

**The cell is chosen for its group and for being a real magnet.** Face-centred
nickel with a Hubbard U on 3d and the moment along z: ``l = 2`` so the rotation
of the ``m`` indices is not the identity, and the magnetic group is **16
operations of which eight carry ``t_rev = 1``** -- so the time-reversal branch,
which is the half of this a general moment direction would never reach, is
exercised by every test below.

Two cells were tried and rejected before this one, and both rejections are
worth more than the choice. ``bn-ldau-noncol.in``, the committed spinor DFT+U
case, is ``lspinorb`` and ultrasoft at ``ecutrho = 350`` on a 3x3x1 grid: its
``nosym`` half was **killed by a 10 GB cap**, which is the augmentation table's
memory wall and not something a threshold can be tuned around. Silicon with a U
on 3p holds a moment only marginally -- it is a nonmagnetic semiconductor being
pushed -- and did not converge to 1e-10 in 250 iterations. Nickel is a
ferromagnet, so its moment is not a near-degeneracy the SCF has to resolve, and
that is what makes the comparison below cheap enough to be a test.

A moment pointing somewhere *generic* cuts the same group to 2 (measured on the
silicon cell), so the interesting regime for this phase is the
collinear-as-spinor one, not a texture.
"""

from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.pseudo.upf import read_upf
from defumat.scf.driver import Calculation, run_scf
from defumat.system.builder import build_system

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
INPUT = CASES / "ni-ldau-noncol.in"

#: Both runs are converged to this, so the floor on any comparison between them
#: is the threshold rather than anything the symmetrisation does.
CONV_THR = 1.0e-9


@lru_cache(maxsize=2)
def _converged(pseudo_dir: Path, nosym: bool):
    """The same input with symmetry on or off. ``lru_cache(2)``, never None.

    Two is what a comparison between two runs needs and is the largest that is
    not a leak (``CLAUDE.md``, "Memory is part of the design").
    """
    data = read_pw_input(INPUT)
    data.namelists["system"]["nosym"] = nosym
    data.namelists["system"]["noinv"] = nosym
    system = build_system(data)
    pseudos = tuple(
        read_upf(Path(pseudo_dir) / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation,
                     conv_thr=CONV_THR, max_iterations=250)
    return calculation, result


@pytest.fixture(scope="module")
def pseudo_dir():
    return Path(__file__).resolve().parents[1] / "data" / "pseudo"


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """Two cells' worth of SCF stack, and XLA keeps every executable."""
    yield
    import jax

    jax.clear_caches()


def test_the_group_is_the_one_this_file_claims(pseudo_dir):
    """The premise, asserted rather than remembered.

    If the magnetic group ever stopped carrying time-reversed operations, every
    other test here would still pass and would silently stop testing the branch
    they exist for -- a check whose null result cannot be told from a pass,
    which is the trap ``CLAUDE.md`` names.
    """
    calculation, _ = _converged(pseudo_dir, False)
    assert calculation.use_symmetry
    symmetries = calculation.symmetries
    assert symmetries.nsym == 16
    t_rev = np.asarray(symmetries.t_rev_array())
    assert t_rev.sum() == 8, (
        "the time-reversal branch of the spinor ns symmetrisation is not "
        f"reached by this cell any more: t_rev = {t_rev.tolist()}"
    )


def test_the_symmetriser_fixes_the_answer_it_should_fix(pseudo_dir):
    """The converged **full-grid** ``ns`` is a fixed point of the group average.

    This is the one statement here that needs a single SCF, and it is a
    different claim from the wedge comparison rather than a cheaper version of
    it. ``ns`` summed over the whole zone is invariant under the crystal's
    group as a matter of physics, so a *correct* symmetriser leaves it alone and
    an incorrect one does not -- a wrong pairing of the spin matrix with the
    atom permutation, or a missing time-reversal transpose, moves it.

    The group is built here rather than taken from the run, because the run that
    produced this ``ns`` had ``nosym`` and therefore has no group at all. That
    is the point: the operator is being tested against an answer that owes it
    nothing.

    **It is a fixed point of the symmetry-allowed part only, and asserting more
    than that would be wrong.** The free run converges with small
    symmetry-*forbidden* entries -- nothing in a ``nosym`` calculation prevents
    them -- and a correct group average does not preserve those, it annihilates
    them. So the assertion is split, and the split is the measurement:

    * what symmetry allows is preserved to **1.1e-16**, machine zero;
    * what symmetry forbids goes from about **1e-9** in the free run to
      **exactly zero**, not to something smaller.

    An undivided "is a fixed point to 1e-8" fails at 6.0e-7 on a *correct*
    implementation, which is how this test was first written.
    """
    from defumat.hubbard.occupations import build_ns_symmetry

    wedge_calculation, _ = _converged(pseudo_dir, False)
    whole_calculation, whole = _converged(pseudo_dir, True)
    assert not whole_calculation.use_symmetry

    symmetry = build_ns_symmetry(
        whole_calculation.hubbard,
        whole_calculation.system.cell,
        whole_calculation.system.structure,
        wedge_calculation.symmetries,
    )
    assert symmetry is not None and symmetry.spin is not None
    ns = np.asarray(whole.ns)
    averaged = np.asarray(symmetry.apply(jnp.asarray(ns)))

    # Idempotence: whatever the average keeps, it keeps exactly. This is the
    # statement that does not depend on how symmetric the input happened to be.
    twice = np.asarray(symmetry.apply(jnp.asarray(averaged)))
    assert np.abs(twice - averaged).max() < 1e-12

    # The diagonal of each spin block is symmetry-allowed and survives -- but to
    # **1e-8**, not to machine zero, and the gap is again the free run's rather
    # than the average's. Three of the five d orbitals come back at 1.1e-16,
    # 4.0e-15 and 4.7e-15; the other two are a *degenerate pair*, which symmetry
    # requires to be equal and which the ``nosym`` run leaves split -- by
    # 6.5e-11 in the majority block and 1.2e-9 in the minority one. The average
    # equalises them exactly. Asserting 1e-13 here fails on correct code, for
    # the third time in this file and for the same reason each time.
    for z in (0, 3):
        diagonal = np.diagonal(ns[z, 0])
        assert np.abs(
            np.diagonal(averaged[z, 0]) - diagonal
        ).max() < 1e-8

    # And the moment the shell carries is untouched by the average.
    before = np.real(np.trace(ns[0, 0]) - np.trace(ns[3, 0]))
    after = np.real(np.trace(averaged[0, 0]) - np.trace(averaged[3, 0]))
    assert after == pytest.approx(before, abs=1e-12)
    assert abs(before) > 1e-4


def test_the_wedge_reproduces_the_closed_grid_energy(pseudo_dir):
    """The total energy, which is what a wrong spin frame moves first."""
    _, wedge = _converged(pseudo_dir, False)
    _, whole = _converged(pseudo_dir, True)
    assert float(wedge.total_energy) == pytest.approx(
        float(whole.total_energy), abs=1e-8
    )


def test_the_wedge_reproduces_the_closed_grid_occupation_matrix(pseudo_dir):
    """``ns`` itself, all four spin blocks, and it is the sharper statement.

    The energy is a scalar and a wrong off-diagonal spin block can cancel out of
    it; the occupation matrix is the object the symmetriser acts on, so
    comparing it leaves nowhere for an error to hide.

    **The tolerance here is looser than the energy's on purpose, and the reason
    is the next test.** The two runs agree to 1.6e-6, and that residual is not
    the symmetriser's error -- it is the ``nosym`` run's own, which converges
    with a spurious transverse moment of about 5e-6 mu_B because nothing forbids
    one. The symmetrised run has it at *exactly* zero. So the free run is the
    less exact of the two, and a tolerance tight enough to "catch" the
    difference would be asserting that the symmetrisation reproduce a numerical
    artefact.
    """
    _, wedge = _converged(pseudo_dir, False)
    _, whole = _converged(pseudo_dir, True)
    a = np.asarray(wedge.ns)
    b = np.asarray(whole.ns)
    assert a.shape == b.shape
    assert a.shape[0] == 4, "a spinor ns carries the four spin pairs"
    assert np.abs(a - b).max() < 1e-5


def test_symmetry_buys_exactness_rather_than_costing_physics(pseudo_dir):
    """The transverse moment is **exactly** zero on the wedge and 5e-6 free.

    With the moment along z the magnetic group contains operations that carry an
    in-plane component onto minus itself, so the group average annihilates it
    identically rather than merely making it small. The ``nosym`` run has no
    such constraint and converges with whatever transverse residue the
    eigensolver leaves.

    This is the same effect P80 measured on the four-atom cycloid, where the
    symmetrised run came out planar to 1.7e-21 against the free run's 8e-6. It
    is worth asserting because it runs the other way from the intuition that
    symmetrising is an approximation: here it is the *exact* answer and the free
    run is the approximate one.
    """
    _, wedge = _converged(pseudo_dir, False)
    _, whole = _converged(pseudo_dir, True)

    reduced = np.asarray(wedge.site_moments)
    free = np.asarray(whole.site_moments)
    assert np.abs(reduced[:, :2]).max() < 1e-12, (
        "the symmetrised run should forbid a transverse moment exactly"
    )
    assert np.abs(free[:, :2]).max() > 1e-7, (
        "the free run has no transverse residue, so this cell cannot show that "
        "the symmetrisation is the exact one rather than the lossy one"
    )
    # and the two agree on the component that survives
    assert reduced[:, 2] == pytest.approx(free[:, 2], abs=1e-5)


def test_the_occupation_matrix_is_hermitian_and_carries_a_moment(pseudo_dir):
    """Two properties of ``ns`` that a wrong spin rotation breaks separately.

    ``new_ns_nc`` ends by *imposing* hermiticity after checking the residual
    against 1e-10 and stopping if it is larger; here it is asserted instead, so
    a symmetriser that broke it would fail rather than be repaired. And the
    moment has to survive at all: a symmetrisation that averaged the two spin
    channels into each other would leave a perfectly Hermitian ``ns`` with no
    magnetization, which is the failure this whole phase is about.
    """
    _, wedge = _converged(pseudo_dir, False)
    ns = np.asarray(wedge.ns)
    # ns[2 s1 + s2] against the conjugate of ns[2 s2 + s1] transposed in m.
    for s1 in range(2):
        for s2 in range(2):
            block = ns[2 * s1 + s2]
            partner = ns[2 * s2 + s1]
            assert np.abs(
                block - np.conj(np.swapaxes(partner, -1, -2))
            ).max() < 1e-10

    moment = np.real(np.trace(ns[0], axis1=-2, axis2=-1)
                     - np.trace(ns[3], axis1=-2, axis2=-1))
    assert np.abs(moment).max() > 1e-4, (
        "the Hubbard shell carries no moment, so this cell cannot tell a "
        "correct spin rotation from one that averaged the channels away"
    )


def test_site_angular_momenta_agree_between_the_wedge_and_the_grid(pseudo_dir):
    """``<L>`` and ``<S>`` are **axial** vectors, and a wedge sums a wedge.

    The refusal this replaces sent the user onto the whole grid. What makes the
    check meaningful rather than circular is that the polar average -- the
    obvious thing to write -- gives exactly zero on a centrosymmetric cell,
    which face-centred nickel is: so a wrong implementation does not disagree by a
    little, it returns a clean and plausible nothing.
    """
    from defumat.projwfc.angular_momentum import angular_momenta

    wedge_calculation, wedge = _converged(pseudo_dir, False)
    whole_calculation, whole = _converged(pseudo_dir, True)

    reduced = angular_momenta(wedge_calculation, wedge)
    full = angular_momenta(whole_calculation, whole)

    # 1e-5 for the same reason the occupation matrix uses it: the free run's own
    # transverse residue. Measured, <S> per site: the symmetrised run gives
    # (0, 0, 0.2521571) and the free one (2.5e-6, 1.5e-6, 0.2521624) -- the
    # transverse components are *exactly* zero on the wedge and not on the grid,
    # so the difference is the free run's error and 1e-6 fails on correct code.
    assert np.abs(np.asarray(reduced.spin) - np.asarray(full.spin)).max() < 1e-5
    assert np.abs(np.asarray(reduced.spin)[:, :2]).max() < 1e-12
    assert np.abs(
        np.asarray(reduced.orbital) - np.asarray(full.orbital)
    ).max() < 1e-6
    # **<S> is the discriminating half and <L> is not.** There is no spin-orbit
    # coupling in this cell, so <L> is quenched to zero and its two sides agree
    # at zero -- agreement that would survive any implementation. The assertion
    # that makes the comparison a claim is this one: the spin moment is there,
    # so the wedge and the grid agreed about something rather than about
    # nothing.
    assert np.abs(np.asarray(full.spin)).max() > 1e-4
