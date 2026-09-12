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


def test_a_collinear_hubbard_state_promotes_into_a_spinor_run(pseudo_dir):
    """The staged route into a hard magnet, which the same refusal closed.

    Converge the collinear run, hand it to the noncollinear one as
    ``starting_from``, and the occupation matrix's two channels become the two
    diagonal spin blocks of the spinor one, with the moments free to cant from
    there. Measured on nickel with ``U = 4``, ``J = 0.9``:

        collinear         42 iterations   -171.0002508525 Ry
        spinor, fresh     78 iterations   -171.0002443437 Ry
        spinor, promoted   4 iterations   -171.0002508585 Ry

    **Four iterations against seventy-eight** is the feature. The assertion on
    the energy is against the *collinear* run and not the fresh spinor one,
    because those two are not the same state: no cant develops here, so the
    promoted run is P62b's collinear-as-spinor identity reached through the
    continuation instead of from scratch, and it agrees to **6e-9 Ry**. The
    fresh spinor run lands 6.5e-6 Ry **higher**, with off-diagonal spin traces
    of -1e-5 -- a slightly canted neighbouring minimum that 78 iterations from
    ``initial_ns_noncollinear`` found and the promotion stepped over. Holding
    the two spinor runs to each other would be pinning which minimum a
    from-scratch start happens to fall into.
    """
    from pathlib import Path

    text = Path(PROMOTION).read_text()
    collinear = Calculator.from_text(text, pseudo_dir, announce=False)
    source = run_scf(collinear.system, collinear.pseudos, conv_thr=1e-8,
                     max_iterations=80, verbose=False)
    assert source.converged, source.accuracy
    assert np.asarray(source.ns).shape[0] == 2

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
    # Both are the same physics; which local minimum a from-scratch spinor start
    # falls into is not something to pin.
    assert promoted.total_energy == pytest.approx(fresh.total_energy, abs=1e-4)
