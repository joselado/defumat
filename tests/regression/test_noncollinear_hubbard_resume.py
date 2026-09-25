"""A noncollinear DFT+U run restarting from its own checkpoint.

This is the path item 9 of the noncollinear audit named as production-blocking,
and the reason it is a *regression* test rather than another array-algebra one:
``promote_ns`` can be right about the arrays and the driver can still never
route the occupation matrix through it. The chain that has to work is
``checkpoint.py`` rebuilding an ``SCFResult`` carrying ``ns`` ->
``driver.py``'s ``starting_from`` -> ``continued_state`` -> ``promote_ns``, and
a mock exercises none of it.

The refusal that closed it was gated on the *target's* ``nspin`` alone, so a
spinor run resuming into a spinor run tripped a message about crossing from
collinear into noncollinear. What it said was that ``ns_nc`` is unimplemented,
which P62b implemented and measured at 1.2e-7 Ry against ``pw.x`` on this very
cell -- so a wall-clock-killed run of the kind the checkpointing was written for
could not restart, and the reason given had been false for fourteen phases.
"""

import numpy as np
import pytest

from defumat import Calculator
from defumat.scf.driver import run_scf

pytestmark = [pytest.mark.regression, pytest.mark.slow]

INPUT = "tests/data/qe/bn-ldau-noncol.in"


@pytest.fixture(scope="module")
def pseudo_dir():
    return "tests/data/pseudo"


def _calculator(pseudo_dir):
    return Calculator.from_file(INPUT, pseudo_dir=pseudo_dir, announce=False)


def test_a_spinor_hubbard_run_resumes_from_its_own_checkpoint(pseudo_dir, tmp_path):
    """Stop at three iterations, resume, and land where an uninterrupted run does.

    The energies have to agree to the SCF threshold and ``Tr ns`` per spin block
    with them -- the occupation matrix is the object that could not cross the
    file, so agreeing on the energy alone would not say it had.
    """
    calculator = _calculator(pseudo_dir)
    system, pseudos = calculator.system, calculator.pseudos

    whole = run_scf(system, pseudos, conv_thr=1e-8, max_iterations=80,
                    verbose=False)
    assert whole.converged, whole.accuracy
    assert whole.ns is not None and np.asarray(whole.ns).shape[0] == 4, (
        "this cell is meant to carry a spinor occupation matrix")

    # Killed at three iterations, then resumed from the directory it wrote.
    stopped = run_scf(system, pseudos, conv_thr=1e-8, max_iterations=3,
                      checkpoint_dir=tmp_path, checkpoint_every=1, verbose=False)
    assert not stopped.converged
    resumed = run_scf(system, pseudos, conv_thr=1e-8, max_iterations=80,
                      checkpoint_dir=tmp_path, checkpoint_every=1, verbose=False)
    assert resumed.converged, resumed.accuracy

    assert resumed.total_energy == pytest.approx(whole.total_energy, abs=1e-7)
    traces_whole = np.real(np.trace(np.asarray(whole.ns), axis1=-2, axis2=-1))
    traces_resumed = np.real(np.trace(np.asarray(resumed.ns), axis1=-2, axis2=-1))
    np.testing.assert_allclose(traces_resumed, traces_whole, atol=1e-6)


#: A *scalar*-relativistic Hubbard cell, because the promotion's source has to be
#: collinear and a `rel-` dataset without spin-orbit coupling is refused by name.
#: Nickel with `U = 4` and `J = 0.9`, ultrasoft, two atoms, one species.
PROMOTION = "tests/data/qe/ni-kind1-force.in"
TEXTURE = "tests/data/qe/ni-ldau-noncol.in"


#: ``pw.x``'s converged occupation eigenvalues on this cell, per atom and ascending,
#: spin 1 then spin 2 (its own printout at ``conv_thr = 1e-12``): the seed the
#: collinear source starts from, so that which minimum it reaches is chosen and not
#: left to the path.
SEED = {1: (0.912, 0.991, 0.992, 0.998, 0.998), 2: (0.599, 0.841, 0.896, 1.0, 1.0)}


def _seeded(text):
    """``text`` with ``starting_ns_eigenvalue`` set to :data:`SEED` for species 1."""
    lines = "".join(f"    starting_ns_eigenvalue({m + 1}, {spin}, 1) = {value}\n"
                    for spin, values in SEED.items() for m, value in enumerate(values))
    seeded = text.replace("    nosym = .true.\n", "    nosym = .true.\n" + lines, 1)
    assert seeded != text
    return seeded


def test_a_collinear_hubbard_state_promotes_into_a_spinor_run(pseudo_dir):
    """The staged route into a hard magnet, which the same refusal closed.

    Converge the collinear run, hand it to the noncollinear one as
    ``starting_from``, and the occupation matrix's two channels become the two
    diagonal spin blocks of the spinor one, with the moments free to cant from
    there. **The collinear source is seeded**, because this cell has at least
    four self-consistent states within 7.5e-3 Ry and which one an unseeded run
    reaches depends on the path, ``conv_thr`` and the mixer's fit included
    (``OPEN.md`` Part XIX, 2026-09-24): -171.0025527 (traces 4.973 up, 4.179
    down per atom), -171.0002509 (4.891, 4.343, where ``pw.x`` lands unseeded),
    -170.9997723 (4.961, 4.201) and -170.9950212 (4.884, 4.390). Seeded at
    ``pw.x``'s own converged eigenvalues the source reaches the lowest of them,
    which is occupation-matrix control doing what Meredig et al. and Dorado et
    al. say it does. Measured on nickel with ``U = 4``, ``J = 0.9``:

        collinear, seeded   48 iterations   -171.0025527096 Ry
        spinor, fresh       33 iterations   -171.0002508828 Ry
        spinor, promoted     3 iterations   -171.0025527104 Ry

    **The promotion reproduces its source in three iterations** against the
    fresh run's thirty-three, which is the feature. The fresh spinor run starts
    from ``initial_ns_noncollinear``, unseeded, and falls into ``pw.x``'s state;
    holding the two spinor runs to each other would pin which minimum a
    from-scratch start happens to fall into, which is what this test did until
    ``1705a0a`` moved it. What is asserted instead is the physics that survives
    a change of path: the promoted state is the source's, and it is at least as
    low as whatever the fresh start finds.
    """
    from pathlib import Path

    text = Path(PROMOTION).read_text()
    collinear = Calculator.from_text(_seeded(text), pseudo_dir, announce=False)
    source = run_scf(collinear.system, collinear.pseudos, conv_thr=1e-8,
                     max_iterations=80, verbose=False)
    assert source.converged, source.accuracy
    assert np.asarray(source.ns).shape[0] == 2
    # The seed reaches the lowest state known on this cell. If a change of path
    # moves a *seeded* run to another minimum, that is worth knowing by name.
    assert source.total_energy == pytest.approx(-171.0025527, abs=1e-6), (
        f"the seeded collinear source converged to {source.total_energy:.7f} Ry, "
        "not the -171.0025527 Ry state the seed reached when measured"
    )

    spinor = Calculator.from_text(
        text.replace("nspin = 2", "noncolin = .true."), pseudo_dir,
        announce=False)
    fresh = run_scf(spinor.system, spinor.pseudos, conv_thr=1e-8,
                    max_iterations=80, verbose=False)
    assert fresh.converged and np.asarray(fresh.ns).shape[0] == 4

    promoted = run_scf(spinor.system, spinor.pseudos, conv_thr=1e-8,
                       max_iterations=80, starting_from=source, verbose=False)
    assert promoted.converged, promoted.accuracy

    # The same state, re-expressed on the spinor axis. The energies agree to
    # 6e-9 Ry and the occupation traces to **6e-5** out of 4.34, which is not a
    # contradiction: both runs stopped on `dr2 < 1e-8`, and `dr2` bounds an
    # occupation far more weakly than it bounds an energy -- the same weighting
    # that `OPEN.md` Y1 measures on the orbital moment. The tolerance is the
    # measurement plus a factor of three, not a hope.
    assert promoted.total_energy == pytest.approx(source.total_energy, abs=1e-7)
    traces = np.real(np.trace(np.asarray(promoted.ns), axis1=-2, axis2=-1))
    collinear_traces = np.real(np.trace(np.asarray(source.ns), axis1=-2, axis2=-1))
    np.testing.assert_allclose(traces[0], collinear_traces[0], atol=2e-4)   # uu
    np.testing.assert_allclose(traces[3], collinear_traces[1], atol=2e-4)   # dd
    # The off-diagonal blocks are what a *canted* shell would put weight in, and
    # this state is collinear, so they must stay at the level the promotion
    # wrote them: exactly zero, up to whatever the SCF put there.
    np.testing.assert_allclose(traces[1:3], 0.0, atol=1e-5)                 # ud, du

    # The point of the feature, and the thing a stale refusal cost: 4 against 78.
    assert promoted.iterations * 4 < fresh.iterations, (
        f"the promoted run took {promoted.iterations} iterations against the "
        f"fresh run's {fresh.iterations}; a promotion that saves nothing is not "
        f"being used"
    )
    # Which local minimum a from-scratch spinor start falls into is not pinned;
    # that the seeded state is not above it is the physics that survives.
    assert promoted.total_energy <= fresh.total_energy + 1e-6, (
        f"the promoted state {promoted.total_energy:.7f} Ry lies above the fresh "
        f"spinor run's {fresh.total_energy:.7f} Ry, so the seed no longer reaches "
        "the lowest known state"
    )


def _shell_moment(ns):
    """``(m_x, m_y, m_z)`` of one correlated shell, from a packed spinor ``ns``.

    Written out here rather than imported, because what is being checked is the
    direction the promotion put the moment in and the promotion's own decoder
    would agree with it whatever it did. ``ns[2 s1 + s2]`` is ``rho[s2, s1]``.
    """
    uu, du, ud, dd = (np.asarray(ns)[i, 0] for i in range(4))
    return np.array([float(np.real(np.trace(du + ud))),
                     float(np.real(1j * np.trace(ud - du))),
                     float(np.real(np.trace(uu - dd)))])


def _direction(vector):
    length = float(np.linalg.norm(vector))
    return vector / length if length > 1e-12 else vector


def test_a_promoted_shell_points_where_the_density_does(pseudo_dir):
    """The correlated shell has to cross onto the same axis the charge does.

    ``promote_density`` and ``promote_becsum`` rotate the source's
    magnetization onto the axis ``angle1``/``angle2`` name and ``promote_ns``
    wrote the two collinear channels into the two diagonal spin blocks
    regardless, which is a moment along ``z``. On fcc nickel (``U = 4.0`` eV,
    the converged collinear ferromagnet carried into ``angle1 = 90``) the
    density crossed with **0.491 mu_B along x** and the shell arrived with
    **0.383 along z**, worth **30.7 mRy** of Hubbard splitting on the wrong
    axis.

    **What that costs depends on whether the shell is free to turn, and both
    cases are here.** Left free it turns, because the carried density's own
    exchange field is along ``x`` and pulls it there, so what the defect costs
    is iterations -- and the way to see them is that a global spin rotation is
    free on a scalar-relativistic dataset, meaning that the continuation must
    cost the same whatever axis is asked for. It did not: **2, 7 and 8**
    iterations at ``angle1 = 0, 45, 90`` against 2, 2 and 2 after.

    With ``mixing_fixed_ns = 10`` it is not a cost. That is QE's own variable
    and the usual thing to set on a magnet with more than one solution: ``ns``
    is held at its starting value, and the residual of that block is zero while
    it is held (``electrons.f90:819-836`` resets the output to the input the
    same way), so the run met ``conv_thr`` at iteration 6 without the freeze
    ever being released. It reported success with the shell on ``z``, the
    density on ``x``, and a total **4.11 mRy** above the right answer on a cell
    whose anisotropy is exactly zero.
    """
    from pathlib import Path

    text = Path(TEXTURE).read_text()
    collinear = Calculator.from_text(
        text.replace("noncolin = .true.", "nspin = 2")
            .replace("angle1(1) = 0.0", "").replace("angle2(1) = 0.0", ""),
        pseudo_dir, announce=False)
    source = run_scf(collinear.system, collinear.pseudos, conv_thr=1e-8,
                     mixing_beta=0.3, max_iterations=200, verbose=False)
    assert source.converged and np.asarray(source.ns).shape[0] == 2

    def promote(angle, **extra):
        target = Calculator.from_text(
            text.replace("angle1(1) = 0.0", f"angle1(1) = {angle}"),
            pseudo_dir, announce=False)
        out = run_scf(target.system, target.pseudos, conv_thr=1e-8,
                      mixing_beta=0.3, max_iterations=200,
                      starting_from=source, magnetization="carry",
                      verbose=False, **extra)
        assert out.converged, out.accuracy
        return target, out

    along_z, out_z = promote(0.0)
    along_x, out_x = promote(90.0)
    # The premise stated rather than inherited: the two targets' magnetic groups
    # are conjugate, so the k-sets are the same size and the counts are
    # comparable at all.
    assert along_x.system.kpoints.nk == along_z.system.kpoints.nk
    assert out_x.iterations == out_z.iterations, (
        f"turning the requested axis by 90 degrees cost "
        f"{out_x.iterations} iterations against {out_z.iterations} on the "
        f"same state, and a global spin rotation is free here"
    )

    # Frozen, the shell cannot recover and the run says nothing about it.
    _, frozen = promote(90.0, mixing_fixed_ns=10)
    shell = _direction(_shell_moment(frozen.ns))
    np.testing.assert_allclose(
        shell, _direction(np.asarray(along_x.system.local_moments)[0]),
        atol=1e-6)
    assert frozen.total_energy == pytest.approx(source.total_energy, abs=1e-7)
