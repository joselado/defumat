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
    """A mixer with a history in it, from a few mixes of random densities.

    ``beta`` is left unset so every mode gets its own default: 0.7 means a step
    length to the QE-family mixers and an *increment* to ``adaptive``, where
    forcing 0.7 both warns and saturates the scheme it is meant to exercise.
    """
    mixer = get_mixer(mode)
    rng = np.random.default_rng(seed)
    density = rng.normal(size=size)
    for _ in range(steps):
        density = np.asarray(mixer.mix(density, density + 0.1 * rng.normal(size=size)))
    return mixer


@pytest.mark.parametrize("mode", ["linear", "anderson", "adaptive"])
def test_a_new_mixer_attribute_is_stored_or_declared_derived(mode):
    """The coverage check, the same one ``BFGS`` has.

    A mixer that grows a third piece of state nobody saves would restart with
    part of its history, which is worse than restarting with none: it would be
    wrong rather than slow, and nothing downstream would say so.
    """
    assert unhandled_mixer_fields(get_mixer(mode)) == set()
    assert unhandled_mixer_fields(_exercised(mode)) == set()


@pytest.mark.parametrize("mode", ["linear", "anderson", "adaptive"])
def test_the_mixer_history_crosses_the_file_exactly(mode, tmp_path):
    """This is the claim a restart actually makes, so it is asserted exactly.

    Not "the resume is faster" -- that is an iteration count, and the docstring
    above measures it going both ways. What must be true is that the object on
    the far side of the file is the one that went in.
    """
    original = _exercised(mode)
    save_mixer(original, tmp_path / SCF_MIXER)
    # Rebuilt the way ``run_scf`` rebuilds it: the *settings* come from the
    # input and only the evolved state comes from the file, which is why
    # ``_MIXER_DERIVED`` exists and why a resume may change ``mixing_beta``.
    restored = load_mixer(get_mixer(mode), tmp_path / SCF_MIXER)

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


def test_the_adaptive_mixers_evolved_steps_resume_where_they_stopped(tmp_path):
    """The state that crosses the file is per component, not a history list.

    ``AdaptiveMixer`` is the first mixer here whose state is neither empty nor a
    list of past densities: it is one step length and one previous residual per
    component, and both have to arrive for the next step to be the step the
    uninterrupted run would have taken. Asserting the *next mix* rather than the
    arrays is what says that, because it is the thing the run depends on.
    """
    original = _exercised("adaptive")
    save_mixer(original, tmp_path / SCF_MIXER)
    restored = load_mixer(get_mixer("adaptive"), tmp_path / SCF_MIXER)

    rng = np.random.default_rng(4)
    rho_in = rng.normal(size=24)
    rho_out = rho_in + 0.05 * rng.normal(size=24)
    np.testing.assert_array_equal(
        np.asarray(original.mix(rho_in, rho_out)),
        np.asarray(restored.mix(rho_in, rho_out)),
    )
    # And it is not trivially equal because the state was empty either way.
    assert np.ptp(np.asarray(restored._betas)) > 0.0


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
@pytest.mark.slow
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


# --- the mixer's ns block, in both precisions -------------------------------


@pytest.mark.parametrize("mode", ["anderson", "adaptive", "linear"])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_a_mixer_does_not_promote_the_densitys_precision(dtype, mode):
    """The **density** block is where a hardcoded float64 inside a mixer shows.

    Not the ``ns`` block, which is what the test below looks at: ``_mix`` casts
    that one back to ``ns``'s own real type explicitly, so it comes out right
    whatever the mixer did and an assertion there passes either way. The density
    is unpacked with a plain ``jnp.asarray`` and carries whatever the mixer
    returned, so this is the one place the convention is observable.

    Checked to fail rather than assumed to: an ``AdaptiveMixer`` subclassed to
    cast its arguments with ``dtype=float`` returns ``float64`` here from a
    ``float32`` density, which is what the first draft of this mixer did.
    """
    import numpy as np

    from defumat.scf.driver import _mix

    rho = np.random.default_rng(3).normal(size=(1, 4, 4, 4)).astype(dtype)
    mixer = get_mixer(mode, **({"beta": 0.4} if mode != "adaptive" else {}))
    mixed, _, _ = _mix(mixer, rho, rho + 0.01, (), ())
    assert np.asarray(mixed).dtype == np.dtype(dtype)


@pytest.mark.parametrize("mode", ["anderson", "adaptive"])
@pytest.mark.parametrize("dtype", ["complex128", "complex64", "float64", "float32"])
def test_the_ns_block_survives_the_mixer_in_either_precision(dtype, mode):
    """``ns`` is packed into the mixer's one real vector and unpacked from it.

    The pack was an unconditional ``.view(float)`` and the unpack a
    ``!= np.complex128`` test, so three of the four cases below were wrong:
    ``.view(float)`` is float64 *by name*, which on a real float32 ``ns``
    reinterprets pairs of numbers as one, and a complex64 ``ns`` failed the
    ``complex128`` test and came back as reals. Silently garbage rather than an
    error, and invisible while only the x64 path is run -- which is also what
    makes it a hardcoded-dtype violation of the standing convention, and what
    made it findable.

    Mixing something with *itself* is the identity for any mixer, so what this
    isolates is exactly the packing.
    """
    import numpy as np

    from defumat.scf.driver import _mix

    rng = np.random.default_rng(20260911)
    shape = (2, 1, 5, 5)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        ns = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype)
    else:
        ns = rng.normal(size=shape).astype(dtype)

    rho = rng.normal(size=(1, 4, 4, 4))
    # ``beta`` is left to each mode: 1.0 is the identity for Anderson and an
    # out-of-range increment for the adaptive mixer, where the identity comes
    # from mixing a vector with itself instead.
    mixer = get_mixer(mode, **({"beta": 1.0} if mode == "anderson" else {}))
    _, _, mixed = _mix(mixer, rho, rho, (), (), ns_in=ns, ns_out=ns)

    mixed = np.asarray(mixed)
    assert mixed.shape == ns.shape
    assert np.iscomplexobj(mixed) == np.iscomplexobj(ns)
    assert mixed.dtype == ns.dtype
    assert mixed == pytest.approx(ns, rel=1e-6, abs=1e-7)


# --------------------------------------------------------------------------
# the fourth thing that crosses the file: the field, and the scale on it
# --------------------------------------------------------------------------

#: The cheapest cell that holds a field: one hydrogen atom, LSDA, with a
#: ``LOCAL_MAGNETIC_FIELDS`` card. The field is what the test is about, so the
#: physics is deliberately the smallest that can carry one.
def _with_field(pseudo_dir, tmp_path, extra=""):
    from tests.conftest import GENERATED

    text = (GENERATED / "h-atom-lsda.in").read_text()
    marker = text.lower().index("&system") + len("&system")
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "field.in"
    path.write_text(text[:marker] + extra + text[marker:]
                    + "LOCAL_MAGNETIC_FIELDS\n 0.0 0.0 0.10\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Calculator.from_file(path, pseudo_dir=pseudo_dir, announce=False)


def test_a_run_holding_an_applied_field_still_checkpoints(pseudo_dir, tmp_path):
    """The refusal used to be "carries a field at all", and it was too wide.

    Long, magnetic and unable to restart is the class of run checkpointing
    exists for, so a blanket refusal on the field took the feature away from
    exactly the calculations that need it. An applied field is the input's from
    beginning to end: the resume rebuilds the calculator from ``scf.in`` and
    gets the identical object back.
    """
    calculator = _with_field(pseudo_dir, tmp_path / "cell")
    out = tmp_path / "ckpt"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_scf(calculator.system, calculator.pseudos,
                         max_iterations=2, conv_thr=1.0e-12,
                         checkpoint_dir=out, checkpoint_every=1, verbose=False)

    assert result.magnetic_field is not None, "the cell must actually hold one"
    assert {p.name for p in out.iterdir()} == {SCF_CHECKPOINT, SCF_MIXER}


def test_reducebf_is_picked_up_where_the_resume_left_it(pseudo_dir, tmp_path):
    """``field_scale`` was written into the file and then reset to 1.0 on load.

    Elk's ``reducebf`` multiplies the external field down towards zero after
    every iteration, so the scale *is* loop state -- and a run whose field had
    faded over forty iterations came back at full field and converged somewhere
    else without a word. The saved scale is what makes (input field, scale)
    reproduce the faded field exactly.
    """
    from defumat.scf.checkpoint import load_state

    calculator = _with_field(pseudo_dir, tmp_path / "cell",
                             extra="\n    reducebf = 0.5\n")
    out = tmp_path / "ckpt"
    options = dict(conv_thr=1.0e-12, verbose=False,
                   checkpoint_dir=out, checkpoint_every=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stopped = run_scf(calculator.system, calculator.pseudos,
                          max_iterations=3, **options)
        reloaded = load_state(out / SCF_CHECKPOINT, system=calculator.system)
        resumed = run_scf(calculator.system, calculator.pseudos,
                          max_iterations=2, **options)

    assert stopped.field_scale == pytest.approx(0.5 ** 3)
    assert reloaded.field_scale == pytest.approx(stopped.field_scale)
    # The resume continues the decay rather than restarting it. Before the scale
    # was restored this came back at 1.0, 0.5 or 0.25 -- every one of them above
    # the 0.125 the run had reached.
    assert resumed.field_scale <= stopped.field_scale


def test_a_driven_field_refuses_the_mid_scf_checkpoint_too(pseudo_dir, tmp_path,
                                                            capsys):
    """The half that was missing, and it failed silently in the other direction.

    ``_InProgressState`` hardcoded ``magnetic_field = None`` under a comment
    saying that was what made the refusal fire -- ``None`` is exactly what makes
    it pass. So a fixed-spin-moment run, whose field is the controller's state
    and is replaced after every iteration, wrote a checkpoint every cadence with
    that field dropped in silence. The state carries the loop's field now, so
    the mid-SCF write asks the same question the converged one does.
    """
    from tests.conftest import GENERATED

    path = GENERATED / "fe-fsm.in"
    if not path.is_file():
        pytest.skip(f"{path} is not present")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_file(path, pseudo_dir=pseudo_dir,
                                          announce=False)
        assert calculator.calculation.magnetic_field.constraint == "fsm"
        out = tmp_path / "ckpt"
        run_scf(calculator.system, calculator.pseudos, max_iterations=2,
                checkpoint_dir=out, checkpoint_every=1, verbose=True)

    assert not out.exists() or list(out.iterdir()) == [], (
        "a driven field must not be half-saved")
    # **Say which refusal fired.** An empty directory is also what a crash, a
    # full disk or a cadence that never came round leaves, and the whole reason
    # this test exists is that a silent no-op read as a working feature.
    printed = capsys.readouterr().out
    assert "checkpointing is off for this run" in printed
    assert "fixed-spin-moment" in printed


def test_a_checkpoint_beats_a_seed_passed_on_the_same_line(tmp_path, pseudo_dir):
    """The recovery this feature advertises is "resubmit the same line".

    A cluster script's line carries its seed on every submission, so the
    mutual-exclusion check fired on exactly the run that was meant to be
    rescued -- `ValueError: starting_from already carries the density`. The
    checkpoint is strictly later state than any seed, so it wins, and it says
    so rather than ignoring an argument silently.
    """
    out = tmp_path / "ckpt"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_file(
            "tests/data/qe/h-atom-lsda.in", pseudo_dir=pseudo_dir, announce=False)
        first = run_scf(calculator.system, calculator.pseudos, max_iterations=2,
                        checkpoint_dir=out, checkpoint_every=1)
        seed = np.asarray(first.density)

    # the resubmitted line: same arguments, seed included, checkpoint present
    with pytest.warns(RuntimeWarning, match="ignoring starting_density"):
        again = run_scf(calculator.system, calculator.pseudos, max_iterations=2,
                        checkpoint_dir=out, checkpoint_every=1,
                        starting_density=seed)
    assert again.iterations > first.iterations, "the resume did not continue"


def test_an_unreadable_mixer_costs_the_history_and_not_the_run(tmp_path, pseudo_dir):
    """The two halves of a restart are not equally recoverable.

    A state that will not load means there is nothing to resume. A *mixer* that
    will not load costs the Anderson history -- some iterations of plain mixing
    and nothing else. Letting the cheap failure raise turns "the resume is
    slower" into "the resume is dead", on the run least able to afford it.
    """
    from defumat.scf.driver import SCF_MIXER

    out = tmp_path / "ckpt"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_file(
            "tests/data/qe/h-atom-lsda.in", pseudo_dir=pseudo_dir, announce=False)
        run_scf(calculator.system, calculator.pseudos, max_iterations=2,
                checkpoint_dir=out, checkpoint_every=1)

    (out / SCF_MIXER).write_bytes(b"not an npz")
    with pytest.warns(RuntimeWarning, match="could not restore the mixer history"):
        resumed = run_scf(calculator.system, calculator.pseudos, max_iterations=2,
                          checkpoint_dir=out, checkpoint_every=1)
    assert resumed.iterations > 2, "the resume did not survive a broken mixer"


@pytest.mark.slow
def test_a_resume_does_not_re_tighten_the_bands_it_does_not_need(
    qe_silicon, pseudo_dir, tmp_path
):
    """The occupations are loop state, and the count that shows it is Davidson's.

    The sibling above asserts the *SCF* iteration count and passes with this
    defect present, because it runs at the default ``nbnd = 4``, where every
    band is occupied and a flat threshold is the right one anyway. The empty
    bands are what see it: ``band_thresholds`` reads ``wg = None`` as "the first
    iteration of a fresh run", which is ``pw.x``'s ``btype`` all ones out of
    ``init_run.f90:149``, and holds every band to ``ethr`` -- harmless at
    ``ETHR_INIT`` and not at the converged ``ethr`` a checkpoint restores, where
    thirty-six empty states are asked for an accuracy a steady-state iteration
    holds them to ``max(5 ethr, 1e-5)`` at. Measured before the fix, comparing
    the resumed run's first iteration with the same iteration of the
    uninterrupted run: **1.0 steps against 5.5** here, and the whole 100-step
    budget on a 45-atom slab at ``nbnd = 403`` (``OPEN.md`` Part VIII item 3).

    So the assertion is on **Davidson steps** rather than on SCF iterations, and
    at a tolerance rather than exactly: restoring ``wg`` changes the resumed
    eigenvalues in their last digits, and the count is a mean over k-points and
    channels.
    """
    stop = 5
    options = dict(conv_thr=1.0e-12, nbnd=40, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_file(qe_silicon, pseudo_dir=pseudo_dir,
                                          announce=False)
        whole = run_scf(calculator.system, calculator.pseudos, **options)

        stopped = run_scf(calculator.system, calculator.pseudos,
                          max_iterations=stop, checkpoint_dir=tmp_path,
                          checkpoint_every=1, **options)
        assert not stopped.converged and stopped.iterations == stop

        resumed = run_scf(calculator.system, calculator.pseudos,
                          checkpoint_dir=tmp_path, checkpoint_every=1,
                          **options)

    assert whole.converged and resumed.converged
    back = resumed.history[0]
    assert back["iteration"] == stop + 1
    uninterrupted, = [entry for entry in whole.history
                      if entry["iteration"] == stop + 1]
    assert back["davidson_iterations"] == pytest.approx(
        uninterrupted["davidson_iterations"], abs=1.0
    ), (f"the resumed iteration took {back['davidson_iterations']} Davidson "
        f"steps where the uninterrupted one took "
        f"{uninterrupted['davidson_iterations']}: the empty bands are being "
        f"held to the checkpoint's converged ethr")
    assert resumed.total_energy == pytest.approx(whole.total_energy, abs=1.0e-10)


def test_a_resume_that_changes_nbnd_drops_the_occupations_and_says_so(
    qe_silicon, pseudo_dir, tmp_path
):
    """The guard is tested by a case that trips it, not by a clean pass.

    A resume is allowed to change ``nbnd``: nothing upstream stops it, since the
    fingerprint compares the loaded state against itself and the grid check is
    on the density. The checkpoint's occupations are then about a different set
    of bands, and feeding them to ``band_thresholds`` would raise on the
    reshape -- which would make carrying them across a resume *break* a case
    that worked. So they are dropped, with a warning, back to the behaviour
    every resume had before: one iteration of full accuracy on the empty bands.
    """
    options = dict(conv_thr=1.0e-12, verbose=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calculator = Calculator.from_file(qe_silicon, pseudo_dir=pseudo_dir,
                                          announce=False)
        run_scf(calculator.system, calculator.pseudos, nbnd=8,
                max_iterations=3, checkpoint_dir=tmp_path, checkpoint_every=1,
                **options)
        resumed = run_scf(calculator.system, calculator.pseudos, nbnd=12,
                          checkpoint_dir=tmp_path, checkpoint_every=1,
                          **options)

    assert resumed.converged
    assert [w for w in caught if "nbnd has changed" in str(w.message)]
