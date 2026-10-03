"""A relaxation starts each later ionic step from the previous step's states, at 1e-6.

``OPEN.md`` Part XXIII item 11. ``pw.x`` runs ``wfcinit`` once and keeps the
converged states across ionic steps (``update_pot.f90`` extrapolates them only
when ``wfc_order > 0``, whose default is 0), and ``run_pwscf.f90:331-334``
diagonalises the first iteration of every later step at ``ethr = 1e-6``. Every
step here used to start from atomic orbitals at 1e-2.

What is asserted is the hand-over and its lifetime rather than a count of
iterations, which belongs to the measurement against ``pw.x``:

* the first step starts from atomic orbitals at the default threshold, and every
  later one is handed a span and ``diago_thr_init = 1e-6``, and its first
  iteration ran at that threshold or tighter (tighter after a redo);
* the relaxation's own frame holds no second reference while the step runs
  (the states are popped into the call, ``result`` is dropped), and the span is
  gone from ``run_scf``'s frame by its first diagonalisation, which is the
  ``MEMORY-AUDIT.md`` A2 lifetime ``test_retention.py`` guards for the result;
* a variable-cell relaxation carries them in its frozen basis and not when the
  basis is rebuilt at every step (``treinit_gvectors``);
* and the one that guards the physics: a step whose occupied manifold changes
  symmetry sector still finds the state a start from atomic orbitals finds,
  which the bare previous states do not (``pw.x``'s ``atomic+random`` factor
  on them is what makes it pass).

The old code fails the hand-over tests, since no step was handed states, and
passes the last, which fails instead on the bare carried states, by 2.6e-2 Ry.
"""

import inspect

import numpy as np
import pytest

import defumat.scf.driver as driver
import defumat.workflows.relax as relax_module
import defumat.workflows.vc_relax as vc_module
from defumat.calculator import Calculator

pytestmark = pytest.mark.unit

#: ``ethr = 1.0D-6`` in ``run_pwscf.f90:331-334``, written here rather than
#: imported so that the test pins ``pw.x``'s number and not the module's.
LATER_STEP_ETHR = 1.0e-6

DISPLACED = """
&control
  calculation = 'relax'
/
&system
  ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1, ecutwfc = 12.0
/
&electrons
  conv_thr = 1.0d-8
/
&ions
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.27 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""

COMPRESSED = DISPLACED.replace("'relax'", "'vc-relax'").replace(
    "celldm(1) = 10.20", "celldm(1) = 10.00").replace("&ions\n/", "&ions\n/\n&cell\n/")


def _frame(name):
    for record in inspect.stack(0):
        if record.function == name:
            return record.frame
    raise AssertionError(f"{name} is not on the stack")


def _watch(monkeypatch, module, caller):
    """Record what each inner ``run_scf`` was handed and what was alive around it."""
    calls, spans_at_diagonalize = [], []
    run_scf = module.run_scf

    def watched_run_scf(*args, **kwargs):
        outer = _frame(caller).f_locals
        calls.append({
            "span": kwargs.get("starting_wavefunctions") is not None,
            "diago_thr_init": kwargs.get("diago_thr_init"),
            "carried_in_caller": len(outer.get("carried", [])),
            "result_in_caller": outer.get("result") is not None,
        })
        result = run_scf(*args, **kwargs)
        calls[-1]["first_ethr"] = result.history[0]["ethr"]
        return result

    diagonalize = driver.Calculation.diagonalize

    def watched_diagonalize(self, *args, **kwargs):
        spans_at_diagonalize.append(_frame("run_scf").f_locals["starting_wavefunctions"])
        return diagonalize(self, *args, **kwargs)

    monkeypatch.setattr(module, "run_scf", watched_run_scf)
    monkeypatch.setattr(driver.Calculation, "diagonalize", watched_diagonalize)
    return calls, spans_at_diagonalize


def _assert_carried(calls, spans_at_diagonalize):
    assert len(calls) >= 2, "needs a second ionic step to say anything"
    first, later = calls[0], calls[1:]
    assert all(call["span"] for call in later), calls
    assert all(call["diago_thr_init"] == LATER_STEP_ETHR for call in later), calls
    assert all(call["first_ethr"] <= LATER_STEP_ETHR for call in later), calls
    assert not first["span"] and first["diago_thr_init"] is None
    assert first["first_ethr"] > LATER_STEP_ETHR
    assert all(call["carried_in_caller"] == 0 and not call["result_in_caller"]
               for call in calls), (
        "the relaxation still held the previous step's states while the next "
        "one ran")
    assert all(span is None for span in spans_at_diagonalize), (
        "the carried span was still bound in run_scf at a diagonalisation")


def test_a_relaxation_hands_each_later_step_the_last_states(pseudo_dir, monkeypatch):
    calls, spans = _watch(monkeypatch, relax_module, "run_relax")
    relax = Calculator.from_text(DISPLACED, pseudo_dir, announce=False).get_relax(nstep=3)
    assert relax.nsteps == len(calls)
    _assert_carried(calls, spans)


#: ``test_checkpoint.py``'s relaxation: two-atom silicon, ``nosym``, ``nbnd = 4``,
#: a large first step. The Hamiltonian keeps symmetries the run does not use,
#: and at the second geometry a state of another symmetry sector enters the
#: occupied manifold at Gamma.
CROSSING = """
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


def test_a_carried_start_does_not_lose_a_state_that_changes_sector(pseudo_dir):
    """The second step's energy is the one a start from atomic orbitals finds.

    The bare previous states cannot reach a state of a symmetry sector they
    have no component in, and the Davidson iteration does not add one, so this
    step converged to -15.5689 Ry where the atomic start and ``pw.x`` give
    -15.5954, with the fourth band at Gamma 0.10 Ry too high. ``pw.x``'s
    ``atomic+random`` factor on the carried states is what this pins; it fails
    without it by 2.6e-2 Ry.
    """
    calculator = Calculator.from_text(CROSSING, pseudo_dir, announce=False)
    relax = calculator.get_relax(nstep=2)
    second = relax.steps[1]
    # A run of its own at that geometry, from the atomic superposition and the
    # atomic orbitals: ``Calculator.with_positions`` would seed it from a
    # converged state, which is the hand-over under test.
    moved = calculator.system.with_cell(calculator.system.cell.at, second.positions)
    fresh = driver.run_scf(moved, calculator.pseudos, conv_thr=second.conv_thr)
    assert second.total_energy == pytest.approx(fresh.total_energy, abs=1.0e-5)


@pytest.mark.parametrize("treinit", [False, True], ids=["frozen-basis", "rebuilt-basis"])
def test_a_variable_cell_relaxation_carries_them_in_its_frozen_basis_only(
        pseudo_dir, monkeypatch, treinit):
    calls, spans = _watch(monkeypatch, vc_module, "run_vc_relax")
    Calculator.from_text(COMPRESSED, pseudo_dir, announce=False).get_relax(
        variable_cell=True, nstep=2, final_scf=False, treinit_gvectors=treinit)
    if not treinit:
        _assert_carried(calls, spans)
        return
    assert len(calls) == 2
    assert not any(call["span"] or call["diago_thr_init"] for call in calls), calls
    assert np.all([call["first_ethr"] > LATER_STEP_ETHR for call in calls])
