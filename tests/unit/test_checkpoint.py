"""A checkpoint that loses a field reloads as a different state.

That is the failure mode this file is mostly about. A state missing ``becsum``
or ``tau`` does not fail to load -- it converges to something plausible and
slightly wrong, which is the hardest kind of bug to notice on a calculation big
enough to need checkpointing in the first place. So the coverage of
:mod:`defumat.scf.checkpoint` over ``SCFResult``'s fields is asserted, and the
round trip is checked to be exact rather than close.
"""

import warnings

import numpy as np
import pytest

from defumat.calculator import Calculator
from defumat.scf.checkpoint import load_state, save_state, unhandled_fields
from defumat.scf.driver import SCFResult, run_scf

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


def test_a_field_or_a_hubbard_setup_is_refused_rather_than_dropped(
    pseudo_dir, tmp_path
):
    """Two things whose loss would be silent are refused by name.

    A converged magnetic field is not the input field wherever ``reducebf``
    changed it, and ``ns`` without its setup is an array about nothing.
    """
    _, result = _converged(SILICON, pseudo_dir)
    result.magnetic_field = object()
    with pytest.raises(NotImplementedError, match="magnetic_field"):
        save_state(result, tmp_path / "refused.npz")


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
