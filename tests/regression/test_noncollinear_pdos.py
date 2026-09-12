"""The projected density of states of a noncollinear run **without** spin-orbit
coupling -- ``partialdos_nc``'s ``nspin0 = 2`` branch.

The regime is the ordinary one for a magnetic system with scalar-relativistic
datasets: two-component spinors, so one Hamiltonian on a space twice as large,
but no term that mixes ``s_z`` into the orbital motion. ``s_z`` is therefore
still a good quantum number, ``atomic_wfc_nc`` builds an up and a down copy of
every harmonic, and QE reports the projection as **two densities of states**
rather than as twice as many columns.

**There is no ``projwfc.x`` reference for this regime in the repository and none
was generated**, so the check is an identity instead -- and it is a sharper one
than a reference would be. Without spin-orbit coupling a state polarized along
``z`` block-diagonalises the noncollinear Hamiltonian into the two collinear
ones, so a noncollinear run with its moment along ``+z`` must reproduce an LSDA
run of the same cell channel for channel. The LSDA route is itself validated
against ``projwfc.x`` (``reference.projwfc.pw_lsda-lsda``), and the two share the
ground state and nothing else: different orbitals, different projector
construction, different column count, different integration layout.

**This identity found a defect outside the projection**, which is why the cell is
a saturated one. A hydrogen atom carries one electron, so ``zeta = 1`` wherever
there is any density at all, and the minority exchange-correlation potential was
discontinuous in the last bit of ``zeta`` there -- ``jnp.clip`` splits a tangent
evenly at a tie, so the boundary received half the interior's and the
``de_c/dzeta`` term of ``v_down`` was halved, worth 0.162 Ry. The two runs here
land on opposite sides of that boundary by rounding alone, so this file is also
where that fix is checked on a whole calculation
(``tests/unit/test_xc_spin_kernel.py`` has it pointwise).

The axis is the **global** ``z`` and not the local moment, exactly as QE's is:
an in-plane moment therefore reports equal up and down channels, which is a
statement about what the decomposition means rather than a defect. That case is
here too, because a check whose null result cannot be told from a pass is not a
check.
"""

import tempfile
from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest

from defumat import Calculator

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: Both runs use the same window and smearing, so the two grids coincide point
#: for point and the curves can be subtracted rather than interpolated.
#:
#: **The window stops below the ``2p`` manifold, and that is not a convenience.**
#: The two runs hold the same number of *states* -- 4 + 4 against 8 -- but they
#: are not the same eight: the up channel's third ``p`` state lies below the
#: down channel's first, so the eight lowest spinor states are five up and three
#: down where the collinear run takes four of each. A comparison that reached
#: into the manifold would be comparing one run's truncation against the other's.
#: At 0.08 Ry a Gaussian of width 0.02 centred on the nearest excluded state
#: (0.1459 Ry) contributes 2e-5 of its peak.
GRID = dict(emin=-1.2, emax=0.08, delta_e=0.01, degauss=0.02, scheme="gaussian")


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """``CLAUDE.md``'s rule for a file that runs several cells: keep the results
    and drop the executables."""
    yield
    jax.clear_caches()


#: **The empty states have to be converged for this comparison to mean
#: anything.** ``c_bands.f90`` holds an unoccupied band only to
#: ``empty_ethr = max(5 ethr, 1e-5)`` Ry, and the minority channel of a
#: saturated ferromagnet is *entirely* empty -- so without this the two runs'
#: down-channel eigenvalues sit 5e-4 Ry apart for a reason that has nothing to
#: do with the projection, which a Gaussian of width ``degauss`` turns into
#: about 2 per cent of the peak. ``diago_full_acc`` is QE's own switch for it,
#: and it is what ``PLAN.md`` P14 used to separate the same two effects.
ACCURATE = dict(diago_full_acc=True, conv_thr=1.0e-12)


@lru_cache(maxsize=2)
def _converged(case: str):
    calculator = Calculator.from_file(CASES / f"{case}.in", pseudo_dir=PSEUDO,
                                      **ACCURATE)
    calculator.get_scf()
    return calculator


@lru_cache(maxsize=1)
def _in_plane():
    """The same atom with its moment along ``x`` rather than ``z``.

    A variant rather than a committed input: it differs from
    ``h-atom-noncolin.in`` in one number, and what it is for is the statement
    that the decomposition is taken along the laboratory ``z``.
    """
    text = (CASES / "h-atom-noncolin.in").read_text()
    assert "angle1(1) = 0.0" in text
    path = Path(tempfile.mkdtemp()) / "variant.in"
    path.write_text(text.replace("angle1(1) = 0.0", "angle1(1) = 90.0"))
    calculator = Calculator.from_file(path, pseudo_dir=PSEUDO, **ACCURATE)
    calculator.get_scf()
    return calculator


def _pdos(calculator):
    return calculator.get_pdos(symmetrize=False, **GRID)


# --------------------------------------------------------------------------
# the identity: the collinear limit
# --------------------------------------------------------------------------


def test_the_moment_along_z_reproduces_the_lsda_projection():
    """The check that validates the split, and it is an identity rather than a
    tolerance.

    A hydrogen atom in a box carries one electron, so its LSDA ground state is
    fully polarized: the ``1S`` channel holds ~1 electron in the up channel and
    ~0 in the down one. The noncollinear run of the same cell with its moment
    along ``+z`` is the *same* Hamiltonian in a doubled basis, so the two
    channels have to come back in the same order with the same weights.

    What this catches: routing the columns by position rather than by ``s_z``,
    swapping the two channels, halving the weight of each -- the ``for_spin``
    trap, a spinor band holding one electron where a collinear one holds two --
    and losing the spin axis altogether, which would report one channel of ~1
    electron where there are two.
    """
    collinear = _pdos(_converged("h-atom-lsda"))
    spinor = _pdos(_converged("h-atom-noncolin"))

    assert collinear.nspin == 2
    assert spinor.nspin == 2, "a noncollinear run without spin-orbit coupling " \
        "reports two channels, which is partialdos_nc's nspin0 = 2"
    assert len(spinor.channels) == len(collinear.channels)
    for ours, theirs in zip(spinor.channels, collinear.channels):
        assert (ours.atom, ours.wfc, ours.l, ours.m) == \
            (theirs.atom, theirs.wfc, theirs.l, theirs.m)
        assert ours.s_z is None, "after the split the spin is an axis, not a label"

    assert np.allclose(spinor.energies, collinear.energies)
    scale = np.abs(collinear.pdos_by_spin).max()
    difference = np.abs(spinor.pdos_by_spin - collinear.pdos_by_spin)

    # **The two channels are held to different bounds, and the reason is
    # arithmetic rather than physics.** The majority channel is a comparison of
    # two ways of writing the same thing and comes back at 2e-5 of the peak. The
    # minority one cannot: the noncollinear branch reaches ``rho_down`` as
    # ``(n - |m|)/2``, a cancellation of two numbers of order 0.1 that leaves
    # round-off where the collinear branch carries ``rho_down`` itself, and the
    # minority *potential* is what the empty minority state sits in. That is
    # worth 1.1e-4 Ry on its eigenvalue -- against 1e-6 on every occupied one --
    # and a Gaussian of width ``degauss`` turns 1.1e-4 Ry into 0.4 per cent of
    # the peak. Before the exchange-correlation clamp was fixed the same number
    # was **7e-2 Ry and 85 per cent**, which is the size of what this bound is
    # actually guarding.
    assert difference[0].max() / scale < 1.0e-4, difference[0].max() / scale
    assert difference[1].max() / scale < 5.0e-3, difference[1].max() / scale

    # The polarization is the discriminator: a code that binned both spins into
    # one channel, or that split the weight evenly, passes every sum rule and
    # fails this.
    ours = spinor.charges.polarization
    theirs = collinear.charges.polarization
    assert float(theirs[0]) > 0.5, theirs
    assert np.abs(ours - theirs).max() < 5.0e-3, (ours, theirs)


def test_the_split_conserves_what_the_unsplit_columns_carried():
    """Every column keeps its own weight: the two channels add back up.

    The split is a **relabelling** of ``2 natomwfc`` columns into two sets of
    ``natomwfc``, so what it cannot touch is any quantity summed over both --
    the charge on each atom, the spilling, and the sum rule against the
    unprojected density of states. Checked against the same cell's LSDA run,
    where those three are already validated against ``projwfc.x``.
    """
    collinear = _pdos(_converged("h-atom-lsda"))
    spinor = _pdos(_converged("h-atom-noncolin"))

    assert spinor.charges.spilling == pytest.approx(collinear.charges.spilling,
                                                    abs=5.0e-3)
    assert np.abs(np.asarray(spinor.charges.total)
                  - np.asarray(collinear.charges.total)).max() < 5.0e-3
    # The sum rule, on the spinor side: the projected channels cannot carry more
    # than the density of states they are a decomposition of.
    summed = spinor.pdos_by_spin.sum(axis=(0, 1))
    assert np.all(summed <= spinor.total.total_dos + 1.0e-8)
    # An m-resolved charge exists here where a j-resolved projection has none:
    # these columns are harmonics times a spin, so p_x would mean something
    # again. ``print_lowdin`` allocates it only for nspin /= 4 and so prints
    # none for either noncollinear branch; the reason it gives -- a spin-angle
    # function has no m -- is true of the spin-orbit branch alone.
    assert spinor.charges.charges_lm is not None


# --------------------------------------------------------------------------
# what the decomposition means: the axis is the global z
# --------------------------------------------------------------------------


def test_an_in_plane_moment_reports_no_polarization():
    """The axis is the global ``z``, and this is the case that says so.

    ``atomic_wfc_nc`` builds ``|up>`` and ``|dn>`` copies of each harmonic in
    the *laboratory* frame, so the decomposition answers "how much ``s_z``" and
    not "how much along the local moment". A moment along ``x`` is an equal
    mixture of the two, so the two channels come back identical -- the same
    physics as the ``z`` run, reported in a frame that does not resolve it.

    This is here **beside** the ``z`` case rather than alone: on its own it is a
    null that a code binning every column into one channel and halving it would
    also pass.
    """
    plane = _pdos(_in_plane())
    axis = _pdos(_converged("h-atom-noncolin"))

    assert abs(float(plane.charges.polarization[0])) < 5.0e-3, \
        plane.charges.polarization
    up, down = plane.pdos_by_spin
    assert np.abs(up - down).max() / np.abs(up).max() < 1.0e-3

    # And the same state along z does resolve it, which is what makes the lines
    # above a measurement rather than a tautology.
    assert float(axis.charges.polarization[0]) > 0.5
