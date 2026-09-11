"""Restarting an SCF from the middle, and the three numbers that make it exact.

`OPEN.md` item 3: `run_scf` wrote a state only after the driver returned, so a
job killed at its wall clock lost every iteration it had run. A 45-atom NiBr2
helix on Triton was cancelled at 7:33:29 having completed zero SCF iterations,
and a second GPU allocation had to be submitted purely as insurance -- two jobs
doing one calculation, which is the direct cost of there being no resume.

**A restart is three things, and dropping any one of them is silent.**

* the *state*: the density, the wavefunctions, `becsum`, `ns`, `tau`;
* the *mixer's history*, `_densities` and `_residuals`, which is a different
  object and needs its own file -- a resume that restores only the density hands
  Anderson an empty history;
* the *loop state*: `iter`, `dr2` and `ethr`, which is what
  `save_in_electrons.f90` writes and for exactly this reason. `next_ethr` is QE's
  schedule from `electrons.f90` and is indexed on the **iteration number** -- it
  keeps the incoming threshold at iteration 1, resets to `ETHR_INIT` at iteration
  2, and only ever decreases. A resume that re-enters at 1 with a fresh threshold
  therefore converges on a different schedule than the one it left. Measured
  before `ethr` was carried, on the silicon benchmark at `conv_thr = 1e-12`:
  **5 + 8 = 13 iterations against 17 uninterrupted**, which looks like a saving
  and is a different calculation.

With all three, the restart is exact, and that is what the end-to-end test
asserts: the same total iteration count as an uninterrupted run, which is P67's
"2 + 4 steps, not 2 + 6" one level down. Reaching the same energy says only that
the minimum is a minimum; the count says the state, the history and the schedule
all crossed the file.

Both reference codes carry the same three parts, which is the corroboration that
this is the shape rather than a choice: QE writes ``iter, dr2, ethr`` into
``restart_scf``, and Elk's ``gndstate.f90`` reads back the mixer work array *and*
the starting loop index under ``mixsave``.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

from defumat.calculator import Calculator
from defumat.scf.checkpoint import (
    load_mixer, save_mixer, unhandled_mixer_fields,
)
from defumat.scf.driver import SCF_CHECKPOINT, SCF_MIXER, run_scf
from defumat.scf.mixing import get_mixer

pytestmark = pytest.mark.unit

QE_SILICON = "quantum_espresso/qe-7.5-ReleasePack/qe-7.5/test-suite/pw_scf/scf.in"


@pytest.fixture
def qe_silicon():
    """The canonical two-atom cell, or a skip where the vendored tree is not.

    ``quantum_espresso/`` is gitignored -- 285 MB of reference does not belong
    in history -- so a checkout that has not fetched it has no input here. Every
    other file that reads the tree goes through ``conftest``'s ``qe_testsuite``
    fixture and skips; this one named the path directly and raised
    ``FileNotFoundError`` instead, which reads as eight broken tests rather than
    as a missing download.
    """
    path = Path(QE_SILICON)
    if not path.is_file():
        pytest.skip(f"QE reference tree not present at {path}")
    return str(path)


def _exercised(mode, steps=4, size=24, seed=20260911):
    """A mixer with a history in it, from a few mixes of random densities."""
    mixer = get_mixer(mode, beta=0.7)
    rng = np.random.default_rng(seed)
    density = rng.normal(size=size)
    for _ in range(steps):
        density = np.asarray(mixer.mix(density, density + 0.1 * rng.normal(size=size)))
    return mixer


@pytest.mark.parametrize("mode", ["linear", "anderson"])
def test_a_new_mixer_attribute_is_stored_or_declared_derived(mode):
    """The coverage check, the same one ``BFGS`` has.

    A mixer that grows a third piece of state nobody saves would restart with
    part of its history, which is worse than restarting with none: it would be
    wrong rather than slow, and nothing downstream would say so.
    """
    assert unhandled_mixer_fields(get_mixer(mode, beta=0.7)) == set()
    assert unhandled_mixer_fields(_exercised(mode)) == set()


@pytest.mark.parametrize("mode", ["linear", "anderson"])
def test_the_mixer_history_crosses_the_file_exactly(mode, tmp_path):
    """This is the claim a restart actually makes, so it is asserted exactly.

    Not "the resume is faster" -- that is an iteration count, and the docstring
    above measures it going both ways. What must be true is that the object on
    the far side of the file is the one that went in.
    """
    original = _exercised(mode)
    save_mixer(original, tmp_path / SCF_MIXER)
    restored = load_mixer(get_mixer(mode, beta=0.7), tmp_path / SCF_MIXER)

    assert set(vars(restored)) == set(vars(original))
    for name, before in vars(original).items():
        after = getattr(restored, name)
        if isinstance(before, list):
            assert len(after) == len(before)
            for one, two in zip(before, after):
                np.testing.assert_array_equal(np.asarray(one), np.asarray(two))
        elif isinstance(before, np.ndarray):
            np.testing.assert_array_equal(before, np.asarray(after))
        else:
            assert after == before


def test_an_empty_history_is_not_confused_with_no_history(tmp_path):
    """A mixer that has never mixed round-trips to empty lists, not to absent ones."""
    fresh = get_mixer("anderson", beta=0.7)
    save_mixer(fresh, tmp_path / SCF_MIXER)
    restored = load_mixer(get_mixer("anderson", beta=0.7), tmp_path / SCF_MIXER)
    assert restored._densities == [] and restored._residuals == []


def test_a_cadence_of_zero_is_refused(qe_silicon, pseudo_dir, tmp_path):
    """``checkpoint_every`` is how many iterations pass between writes."""
    calculator = Calculator.from_file(qe_silicon, pseudo_dir=pseudo_dir,
                                      announce=False)
    with pytest.raises(ValueError, match="not a cadence"):
        run_scf(calculator.system, calculator.pseudos, max_iterations=1,
                checkpoint_dir=tmp_path, checkpoint_every=0, verbose=False)


@pytest.mark.parametrize("beta,stop", [(0.25, 5), (0.2, 6), (0.3, 4)])
def test_an_interrupted_scf_costs_the_same_as_an_uninterrupted_one(qe_silicon, 
    pseudo_dir, tmp_path, beta, stop
):
    """The iteration count is the assertion, and it is exact.

    Reaching the same energy says only that the minimum is a minimum -- a resume
    that threw away the mixer history or the threshold would still get there.
    The **count** is what says all three parts crossed the file, and it is
    checked at three mixing parameters because a single one can agree by
    accident: Anderson's history is not monotonically helpful, and a resume
    without it converged in 11 iterations where the uninterrupted run took 13.

    ``conv_thr`` is tight on purpose. At the default the run converges in five
    iterations and there is not enough schedule left for a dropped ``ethr`` to
    show.
    """
    options = dict(conv_thr=1.0e-12, mixing_beta=beta, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_file(qe_silicon, pseudo_dir=pseudo_dir,
                                          announce=False)
        whole = run_scf(calculator.system, calculator.pseudos, **options)

        stopped = run_scf(calculator.system, calculator.pseudos,
                          max_iterations=stop, checkpoint_dir=tmp_path,
                          checkpoint_every=1, **options)
        assert not stopped.converged
        assert stopped.iterations == stop
        assert {p.name for p in tmp_path.iterdir()} == {SCF_CHECKPOINT, SCF_MIXER}

        # The same call again, which is what a resubmitted sbatch is.
        resumed = run_scf(calculator.system, calculator.pseudos,
                          checkpoint_dir=tmp_path, checkpoint_every=1, **options)

    assert whole.converged and resumed.converged
    # ``iterations`` is absolute across a resume, so this is the total.
    assert resumed.iterations == whole.iterations
    assert resumed.total_energy == pytest.approx(whole.total_energy, abs=1.0e-10)


def test_the_threshold_schedule_crosses_the_file(qe_silicon, pseudo_dir, tmp_path):
    """``ethr`` is loop state, so it is on the result and in the checkpoint.

    It is the third of the three and the one with no obvious home: it is neither
    the state nor the mixer, and QE keeps it in ``restart_scf`` beside ``iter``
    and ``dr2`` for exactly that reason.
    """
    from defumat.scf.checkpoint import load_state

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_file(qe_silicon, pseudo_dir=pseudo_dir,
                                          announce=False)
        stopped = run_scf(calculator.system, calculator.pseudos,
                          max_iterations=4, conv_thr=1.0e-12,
                          checkpoint_dir=tmp_path, checkpoint_every=1,
                          verbose=False)
        reloaded = load_state(tmp_path / SCF_CHECKPOINT,
                              system=calculator.system)

    assert stopped.ethr is not None and stopped.ethr > 0.0
    assert reloaded.ethr == pytest.approx(stopped.ethr, rel=1.0e-12)
    assert reloaded.accuracy == pytest.approx(stopped.accuracy, rel=1.0e-12)
    assert reloaded.iterations == 4


def test_max_seconds_stops_the_loop_and_leaves_a_checkpoint(qe_silicon, pseudo_dir, tmp_path):
    """QE's ``check_stop_now``: the loop stops itself before the scheduler does.

    A wall clock is the one deadline a library can honour without installing a
    signal handler, which would change the host process's behaviour for pytest,
    notebooks and every other caller. Zero seconds is already past, so the first
    iteration is the one that stops -- which is what makes this cheap to assert.
    """
    import defumat.scf.driver as driver

    # A fake clock rather than a real deadline, so the stop lands after exactly
    # three iterations instead of after however long the machine took.
    ticks = iter([0.0] + [1.0, 1.0, 1.0] + [99.0] * 100)
    monkeypatch_time = lambda: next(ticks)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_file(qe_silicon, pseudo_dir=pseudo_dir,
                                          announce=False)
        real_time = driver.time.time
        driver.time.time = monkeypatch_time
        try:
            stopped = run_scf(calculator.system, calculator.pseudos,
                              max_iterations=50, conv_thr=1.0e-12,
                              checkpoint_dir=tmp_path, checkpoint_every=1000,
                              max_seconds=10.0, verbose=False)
        finally:
            driver.time.time = real_time

    assert not stopped.converged
    # ``checkpoint_every`` is 1000, so nothing was written on the cadence: the
    # files that exist are the ones the stop wrote on the way out.
    assert (tmp_path / SCF_CHECKPOINT).exists()
    assert (tmp_path / SCF_MIXER).exists()


def test_a_deadline_already_past_still_runs_one_iteration(qe_silicon, pseudo_dir, tmp_path):
    """One iteration always runs, because otherwise there is nothing to save.

    The loop's arrays -- eigenvalues, weights, wavefunctions -- do not exist
    before the first body, so a deadline honoured at the top of the first pass
    would have neither a result to return nor a state to checkpoint. A run with
    no time for one iteration has no time for a restart either.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_file(qe_silicon, pseudo_dir=pseudo_dir,
                                          announce=False)
        stopped = run_scf(calculator.system, calculator.pseudos,
                          max_iterations=20, conv_thr=1.0e-12,
                          checkpoint_dir=tmp_path, checkpoint_every=1000,
                          max_seconds=0.0, verbose=False)

    assert not stopped.converged
    assert stopped.iterations == 1
    assert (tmp_path / SCF_CHECKPOINT).exists()


def test_an_explicit_starting_from_is_not_overridden(qe_silicon, pseudo_dir, tmp_path):
    """A checkpoint on disk does not silently win over an argument the caller passed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_file(qe_silicon, pseudo_dir=pseudo_dir,
                                          announce=False)
        converged = run_scf(calculator.system, calculator.pseudos, verbose=False)
        run_scf(calculator.system, calculator.pseudos, max_iterations=2,
                checkpoint_dir=tmp_path, checkpoint_every=1, verbose=False)
        assert (tmp_path / SCF_CHECKPOINT).exists()

        # Both a checkpoint and an explicit state: the argument wins, and the
        # run converges immediately because it was handed a converged state.
        resumed = run_scf(calculator.system, calculator.pseudos,
                          checkpoint_dir=tmp_path, starting_from=converged,
                          verbose=False)
    assert resumed.converged
    assert resumed.iterations <= 2
