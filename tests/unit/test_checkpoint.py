"""A checkpoint that loses a field reloads as a different state.

That is the failure mode this file is mostly about. A state missing ``becsum``
or ``tau`` does not fail to load -- it converges to something plausible and
slightly wrong, which is the hardest kind of bug to notice on a calculation big
enough to need checkpointing in the first place. So the coverage of
:mod:`defumat.scf.checkpoint` over ``SCFResult``'s fields is asserted, and the
round trip is checked to be exact rather than close.
"""

import gc
import warnings
import weakref

import numpy as np
import pytest

from defumat.calculator import Calculator
from defumat.scf import checkpoint as checkpoint_module
from defumat.scf.checkpoint import load_state, save_state, unhandled_fields
from defumat.scf.driver import (SCF_CHECKPOINT, Calculation, SCFResult,
                                run_scf)

pytestmark = pytest.mark.unit


SILICON = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1, ecutwfc = 12.0
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""

#: PAW, so the state carries a ``becsum`` -- the field whose silent loss is the
#: whole reason the coverage assertion exists.
SILICON_PAW = SILICON.replace(
    "Si 28.086 Si.pz-vbc.UPF", "Si 28.086 Si.pz-n-kjpaw_psl.0.1.UPF"
).replace("ecutwfc = 12.0", "ecutwfc = 20.0, ecutrho = 120.0")


#: DFT+U on the same two-atom cell: a ``U`` on silicon's 3p is not physics
#: anybody wants, and it is a manifold, a projector set and an ``ns`` for a
#: fraction of the cost of a transition-metal oxide.
SILICON_HUBBARD = SILICON + "HUBBARD {atomic}\n U Si-3p 2.0\n"


def _converged(text, pseudo_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_text(text, pseudo_dir, announce=False)
        return calculator, calculator.get_scf()


def test_every_field_of_the_result_is_accounted_for():
    """A field added to ``SCFResult`` must be classified, not forgotten.

    This is the test that matters most in the file: it fails when someone adds
    state to the result and does not decide whether a checkpoint carries it.
    It has already earned its keep -- ``tau``, ``meta_c``, ``stress``,
    ``solver`` and ``history`` were all missing from the first draft, and
    ``tau`` is genuine state.
    """
    assert unhandled_fields() == set()


@pytest.mark.parametrize("text", [SILICON, SILICON_PAW], ids=["nc", "paw"])
@pytest.mark.slow
def test_the_round_trip_is_exact(text, pseudo_dir, tmp_path):
    """Bit for bit, not merely close.

    A checkpoint is not a lossy summary: anything that changed in the round trip
    would show up later as a run that converges somewhere else.
    """
    calculator, result = _converged(text, pseudo_dir)
    path = save_state(result, tmp_path / "state.npz")
    back = load_state(path, system=calculator.system,
                      calculation=calculator.calculation)

    assert back.total_energy == result.total_energy
    assert back.converged == result.converged
    assert back.nspin == result.nspin and back.nspin_mag == result.nspin_mag
    for name in ("density", "wavefunctions", "potential", "eigenvalues",
                 "occupations"):
        np.testing.assert_array_equal(
            np.asarray(getattr(back, name)), np.asarray(getattr(result, name))
        )
    assert len(back.becsum) == len(result.becsum)
    for saved, original in zip(back.becsum, result.becsum):
        np.testing.assert_array_equal(np.asarray(saved), np.asarray(original))


def test_a_reloaded_state_is_one_a_run_can_continue_from(pseudo_dir, tmp_path):
    """The point of the file: ``starting_from`` must accept what came back.

    A round trip that preserved the arrays but produced an object the
    continuation refuses would be useless, and the two are checked separately
    because they fail separately.
    """
    calculator, result = _converged(SILICON, pseudo_dir)
    path = save_state(result, tmp_path / "state.npz")
    back = load_state(path, system=calculator.system)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resumed = run_scf(
            calculator.system, calculator.pseudos,
            calculation=calculator.calculation, starting_from=back,
            conv_thr=1.0e-10,
        )
    assert resumed.converged
    assert resumed.total_energy == pytest.approx(result.total_energy, abs=1e-6)


def test_a_resume_refuses_to_be_told_how_the_magnetization_crosses(
        pseudo_dir, tmp_path):
    """``magnetization=`` is for a seed from another run, not for this run's own.

    It is refused here rather than ignored because of what a resume is for: the
    recovery the checkpoint advertises is "resubmit the same line", so whatever
    is on that line arrives again after every wall-clock kill. A
    ``magnetization='seed'`` on it would throw the converged moment away and
    restart from the atomic superposition each time, on a run whose moment is
    usually the slow variable and whose charge is the part that was expensive.
    """
    calculator, result = _converged(SILICON, pseudo_dir)
    save_state(result, tmp_path / SCF_CHECKPOINT)

    with pytest.raises(ValueError, match="nothing to decide"):
        run_scf(calculator.system, calculator.pseudos,
                calculation=calculator.calculation,
                checkpoint_dir=tmp_path, magnetization="seed")


def test_loading_against_the_wrong_system_is_refused(pseudo_dir, tmp_path):
    """A fingerprint mismatch must raise, not be discovered as a wrong answer.

    The cheapest way to get this wrong on a cluster is to resume one job's
    checkpoint into another's directory.
    """
    calculator, result = _converged(SILICON, pseudo_dir)
    path = save_state(result, tmp_path / "state.npz")

    other, _ = _converged(
        SILICON.replace("celldm(1) = 10.20", "celldm(1) = 10.60"), pseudo_dir
    )
    with pytest.raises(ValueError, match="does not describe this system|has changed"):
        load_state(path, system=other.system, calculation=other.calculation)


def _field(constraint):
    """A :class:`MagneticField` of one flavour, with nothing else switched on."""
    import jax.numpy as jnp

    from defumat.scf.fields import MagneticField

    return MagneticField(
        regions=None, uniform=jnp.asarray([0.0, 0.0, 0.1]), atomic=None,
        targets=None if constraint == "none" else jnp.asarray([0.0, 0.0, 1.0]),
        penalty=0.2, constraint=constraint,
    )


def test_a_hubbard_state_round_trips_rather_than_being_refused(pseudo_dir,
                                                               tmp_path):
    """The setup is the caller's to rebuild, so ``ns`` is all the file owes.

    This used to be refused outright, on the grounds that ``ns`` without the
    setup is an array of numbers about nothing. That is true of the *file* and
    the file is not what a resume reads it against: a state is loaded against a
    system, and ``run_scf`` rebuilds the manifold from the ``HUBBARD`` card
    before it looks at ``ns``. So the assertions are the two halves of that --
    the occupation matrix comes back bit for bit, and the resume reaches the
    same energy -- plus the setup itself, which ``load_state`` takes off the
    calculation the way it takes ``system`` off the caller.

    The mid-SCF path was the standing evidence and it is why this was an
    inference rather than a suspicion: ``_InProgressState`` has never carried a
    setup, so every DFT+U run with ``checkpoint_dir`` has been reloading one of
    these correctly (``OPEN.md`` Part VII item 3).
    """
    calculator, result = _converged(SILICON_HUBBARD, pseudo_dir)
    assert result.ns is not None and result.hubbard_setup is not None

    path = save_state(result, tmp_path / "hubbard.npz")
    back = load_state(path, system=calculator.system,
                      calculation=calculator.calculation)

    assert np.array_equal(np.asarray(back.ns), np.asarray(result.ns))
    assert back.hubbard_setup is calculator.calculation.hubbard
    assert back.hubbard_occupations == result.hubbard_occupations

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resumed = run_scf(
            calculator.system, calculator.pseudos,
            calculation=calculator.calculation, starting_from=back,
            conv_thr=1.0e-10,
        )
    assert resumed.converged
    # The resume runs at a *tighter* ``conv_thr`` than the state was converged
    # at, so it settles a little further: 1.3e-8 Ry here, which is the default
    # threshold's own slack rather than anything the file lost. Same tolerance
    # as the round-trip test above, and for the same reason.
    assert resumed.total_energy == pytest.approx(result.total_energy, abs=1e-6)


def test_a_loaded_hubbard_state_without_a_calculation_has_no_setup(pseudo_dir,
                                                                   tmp_path):
    """And it says so by being ``None`` rather than by being wrong.

    ``build_hubbard_setup`` resolves the card against the structure *and* the
    datasets, and a bare ``system`` carries file names rather than datasets, so
    there is nothing to rebuild from. The state is still a state -- ``ns`` is
    there and a resume rebuilds the manifold itself -- and what is missing is
    the label, which is what ``hubbard_occupations`` reads.
    """
    calculator, result = _converged(SILICON_HUBBARD, pseudo_dir)
    path = save_state(result, tmp_path / "hubbard.npz")
    back = load_state(path, system=calculator.system)

    assert back.hubbard_setup is None
    assert back.ns is not None


@pytest.mark.parametrize(
    "constraint", ["fsm", "atomic fsm", "atomic fsm direction"])
def test_a_driven_field_is_refused_because_no_input_determines_it(
    pseudo_dir, tmp_path, constraint
):
    """The fixed-spin-moment schemes: the field *is* the controller's state.

    ``MagneticField.feedback`` replaces the field after every iteration, so what
    a run reached is not what any input file says. A checkpoint without it
    resumes at the input field and converges somewhere else in silence.
    """
    _, result = _converged(SILICON, pseudo_dir)
    result.magnetic_field = _field(constraint)
    with pytest.raises(NotImplementedError, match="fixed-spin-moment"):
        save_state(result, tmp_path / "refused.npz")


@pytest.mark.parametrize(
    "constraint", ["none", "atomic", "total", "atomic direction"])
def test_an_applied_field_and_a_penalty_are_saved_rather_than_refused(
    pseudo_dir, tmp_path, constraint
):
    """The refusal used to be "carries a field at all", and that was too wide.

    Neither an applied ``LOCAL_MAGNETIC_FIELDS`` card nor a penalty constraint
    changes the field object: ``feedback`` returns ``self`` for anything outside
    :data:`~defumat.scf.fields.FEEDBACK`, so the resume rebuilds the identical
    field from ``scf.in``. Refusing these left long, constrained, magnetic runs
    -- the class checkpointing exists for -- with no checkpoint at all.
    """
    calculator, result = _converged(SILICON, pseudo_dir)
    result.magnetic_field = _field(constraint)
    path = save_state(result, tmp_path / "kept.npz")
    reloaded = load_state(path, system=calculator.system)
    assert np.asarray(reloaded.density) == pytest.approx(
        np.asarray(result.density))


def test_reducebf_survives_the_round_trip_as_a_scale(pseudo_dir, tmp_path):
    """Elk's ``reducebf`` multiplies a scalar, and the scalar is state.

    The field object is untouched by it -- what fades is ``field_scale`` -- so
    the pair (input field, saved scale) reproduces the faded field exactly.
    A resume that reset the scale to 1.0 came back at full field, which is why
    ``run_scf`` restores it beside ``ethr``.
    """
    calculator, result = _converged(SILICON, pseudo_dir)
    result.magnetic_field = _field("none")
    result.field_scale = 0.125
    path = save_state(result, tmp_path / "faded.npz")
    assert load_state(path, system=calculator.system).field_scale == 0.125


def test_an_interrupted_write_leaves_no_readable_file(pseudo_dir, tmp_path):
    """The file appears whole or not at all.

    A resume that reads a half-written checkpoint is worse than one that finds
    no checkpoint, so the write goes to a scratch name and is moved into place.
    """
    _, result = _converged(SILICON, pseudo_dir)
    path = save_state(result, tmp_path / "state.npz")
    assert path.is_file()
    assert not list(tmp_path.glob("*.partial*"))


# --- the relaxation's own restart -------------------------------------------


_RELAX = """
&control
  calculation = 'relax', etot_conv_thr = 1.0d-5, forc_conv_thr = 1.0d-4
/
&system
  ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1, ecutwfc = 12.0,
  nosym = .true.
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.32 0.28 0.22
K_POINTS automatic
 2 2 2 0 0 0
"""


def test_the_optimizer_state_covers_every_attribute():
    """A new attribute on ``BFGS`` must be stored or declared derived."""
    import numpy as np

    from defumat.relax.bfgs import BFGS
    from defumat.scf.checkpoint import unhandled_optimizer_fields

    assert unhandled_optimizer_fields(BFGS(at=np.eye(3) * 10.2)) == set()


@pytest.mark.slow
def test_an_interrupted_relaxation_resumes_where_it_stopped(pseudo_dir, tmp_path):
    """Stopping at step 2 and resuming must cost the same total as not stopping.

    **The step count is the assertion that matters.** Reaching the same geometry
    only says the minimum is a minimum -- a resume that threw away the inverse
    Hessian would still get there, just by taking its next step as if it were
    the first. ``2 + 4 == 6`` says the history crossed the file.
    """
    from defumat.workflows.relax import run_relax

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_text(_RELAX, pseudo_dir, announce=False)
        whole = run_relax(calculator.system, calculator.pseudos, nstep=20)

        stopped = run_relax(calculator.system, calculator.pseudos, nstep=2,
                            checkpoint_dir=tmp_path)
        assert not stopped.converged
        assert {p.name for p in tmp_path.iterdir()} == {
            "scf_state.npz", "optimizer.npz", "relax_step.json"
        }
        resumed = run_relax(calculator.system, calculator.pseudos, nstep=20,
                            checkpoint_dir=tmp_path)

    assert whole.converged and resumed.converged
    assert len(stopped.steps) + len(resumed.steps) == len(whole.steps)
    assert resumed.scf.total_energy == pytest.approx(
        whole.scf.total_energy, abs=1.0e-9
    )
    np.testing.assert_allclose(
        np.asarray(resumed.system.structure.positions),
        np.asarray(whole.system.structure.positions),
        atol=1.0e-5,
    )


def test_a_resume_does_not_pin_the_checkpoint_for_the_whole_run(
        pseudo_dir, tmp_path, monkeypatch):
    """A resumed run must not hold the loaded state beside the live one.

    ``load_state`` returns an ``SCFResult`` whose arrays are ``jnp.asarray``
    and so device-resident, and the wavefunctions dominate it. Two names in
    ``run_scf`` used to keep it for the life of the call -- the ``starting_from``
    parameter, rebound to the loaded state, and ``resumed_state``, an alias
    taken 300 lines away for four scalars -- so a resumed run carried a second
    wavefunction set from start to finish where a fresh one frees its starting
    guess after the first solve. On the 45-atom NiBr2 slab that is 12.10 GB and
    it is why every resumed arm died several iterations before a fresh one.

    The assertion is on **reachability rather than on bytes**, which is what
    makes it run on any backend: ``memory_stats()`` returns ``None`` on the CPU
    client, and the defect is a live reference, not a size. It is checked at the
    *second* solve because the pin only exists inside the call, so a weakref
    taken after ``run_scf`` returns is dead whatever the code does -- which is
    the version of this test that passes on the unfixed driver.

    Deleting either name alone leaves the other holding the same object and
    frees nothing, so this fails until both go.
    """
    calculator, result = _converged(SILICON, pseudo_dir)
    save_state(result, tmp_path / SCF_CHECKPOINT)

    seen = {}
    real_load = checkpoint_module.load_state

    def watching_load(*args, **kwargs):
        loaded = real_load(*args, **kwargs)
        seen["ref"] = weakref.ref(loaded)
        return loaded

    monkeypatch.setattr(checkpoint_module, "load_state", watching_load)

    solves = []
    real_diagonalize = Calculation.diagonalize

    def counting_diagonalize(self, *args, **kwargs):
        solves.append(1)
        if len(solves) == 2:
            gc.collect()
            seen["alive_at_second_solve"] = seen["ref"]() is not None
        return real_diagonalize(self, *args, **kwargs)

    monkeypatch.setattr(Calculation, "diagonalize", counting_diagonalize)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_scf(
            calculator.system, calculator.pseudos,
            calculation=calculator.calculation,
            checkpoint_dir=tmp_path, conv_thr=1.0e-12, max_iterations=3,
        )

    assert seen.get("ref") is not None, "the run did not take the resume path"
    assert len(solves) >= 2, (
        "the resumed run converged in one solve, so the check never ran -- "
        "loosen the seed or tighten conv_thr"
    )
    assert seen["alive_at_second_solve"] is False, (
        "the loaded checkpoint is still reachable during the SCF loop, so a "
        "resume is carrying a second wavefunction set for the whole run"
    )


def test_the_checkpoints_field_scale_belongs_to_the_density_beside_it(
    pseudo_dir, tmp_path
):
    """The write sat one ``reducebf`` step before the density it was saving.

    ``reducebf`` and the fixed-spin-moment feedback act **between** iterations:
    the loop mixes the density, and only then multiplies ``field_scale`` down
    and steps the field. The cadence write was placed between those two, so it
    paired iteration ``i + 1``'s density -- the mixer's output, which is what a
    resume re-enters with -- against iteration ``i``'s scale. A resume then
    entered with a field a factor ``1/reducebf`` too strong, and because the
    decay is cumulative it stayed that way for the whole of the rest of the
    run. The unconverged-exit write already sat *after* the step, so the two
    sites disagreed with each other about what a checkpoint means.

    The assertion is the pair rather than the number: the saved scale has to be
    the one the saved iteration count implies, ``reducebf ** iterations``, and
    it has to be the scale the run itself ended on. Asserting only the second
    would pass on the defect at the last cadence boundary of a run that stops
    there anyway.
    """
    from tests.conftest import GENERATED

    text = (GENERATED / "h-atom-lsda.in").read_text()
    marker = text.lower().index("&system") + len("&system")
    reducebf = 0.5
    text = (
        text[:marker] + f"\n    reducebf = {reducebf}\n" + text[marker:]
        + "LOCAL_MAGNETIC_FIELDS\n 0.0 0.0 0.10\n"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_text(text, pseudo_dir, announce=False)
        result = calculator.get_scf(
            max_iterations=3, checkpoint_dir=tmp_path, checkpoint_every=1
        )

    state = load_state(tmp_path / SCF_CHECKPOINT, system=calculator.system)
    assert state.iterations == 3
    assert state.field_scale == pytest.approx(reducebf ** state.iterations)
    assert state.field_scale == pytest.approx(float(result.field_scale))
    # ...and it is not the value the defect wrote, which is a clean factor of
    # 1/reducebf away and is the number a run resumed from here would apply.
    assert state.field_scale != pytest.approx(reducebf ** (state.iterations - 1))
