"""What the SCF's per-iteration orientation torque costs on PAW (``OPEN.md`` Part XXII item 1).

Every noncollinear magnetic SCF can record the torque on a rigid turn of its
input texture, and on PAW the one-centre field's share of it, which is three
``jax.jvp`` an iteration and, with spin-orbit coupling, a walk through every
output state, because the fully-relativistic ``becsum`` is not a vector under a
spin rotation. Its reverse-mode form stopped the fifteen-cell NiBr2 source at
``soc_scale = 0`` before its first iteration (19.50 GiB). Since 2026-10-10 (the
user's decision) a PAW run computes it only when something reads it, Route C's
step or the printed ``|tau|``, and records ``None`` otherwise; the tests hold a
run that reads nothing to computing nothing, and a run that prints it to the
k-chunked forward walk.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

from defumat.calculator import Calculator

pytestmark = [pytest.mark.regression, pytest.mark.slow]

PSEUDO = "tests/data/pseudo"


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    yield
    jax.clear_caches()


def _oxygen_pair():
    """Two PAW oxygen atoms with a texture, small enough for two iterations."""
    text = open("tests/data/qe/o2-paw-texture.in").read()
    text = text.replace("celldm(1) = 14.0", "celldm(1) = 9.0").replace(
        "ecutwfc = 30, ecutrho = 240", "ecutwfc = 25, ecutrho = 200").replace(
        "K_POINTS {gamma}", "K_POINTS {automatic}\n 2 1 1 0 0 0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Calculator.from_text(text, PSEUDO, announce=False)


def test_a_paw_run_that_reads_no_torque_computes_none(monkeypatch):
    """No one-centre torque, no spin turn and no history entry, with or without the coupling.

    Without ``rotate_moments`` and without the printed line nothing reads the
    torque, so a PAW run must not call ``_onecenter_torque`` at all and records
    ``None``. Both cases called it once an iteration on ``a1ba166``, and the
    coupled one turned every output state in it.
    """
    from defumat.scf import driver, orientation
    from defumat.scf.driver import run_scf

    calls, turns = [], []
    original_torque, original_turn = driver._onecenter_torque, orientation.spin_turned

    def counted_torque(*args, **kwargs):
        calls.append(True)
        return original_torque(*args, **kwargs)

    def counted_turn(psi, omega):
        turns.append(np.shape(psi))
        return original_turn(psi, omega)

    monkeypatch.setattr(driver, "_onecenter_torque", counted_torque)
    monkeypatch.setattr(orientation, "spin_turned", counted_turn)
    for calculator in (_oxygen_pair(), _iodine_on_two_k_points()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_scf(calculator.system, calculator.pseudos, conv_thr=1.0e-14,
                             max_iterations=2, k_batch=1)
        assert len(result.history) == 2
        assert all("orientation_torque" not in entry for entry in result.history)
        assert result.orientation_torque is None
    assert calls == [] and turns == []


def _iodine_on_two_k_points():
    """A fully-relativistic PAW iodine atom, the coupling on, two k-points."""
    text = open("tests/data/qe/i-atom-soc-paw.in").read().replace(
        "K_POINTS (automatic)\n 1 1 1 0 0 0", "K_POINTS (automatic)\n 2 1 1 0 0 0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_text(text, PSEUDO, announce=False)
    assert calculator.system.lspinorb and calculator.system.kpoints.nk == 2
    return calculator


def test_with_the_coupling_the_states_are_turned_a_k_chunk_at_a_time(monkeypatch):
    """With the coupling the states are needed, and are walked, not taped whole.

    A fully-relativistic PAW iodine atom on two k-points, ``k_batch = 1``,
    printing its iterations, so the torque is read: every spin turn it takes is
    of one k-point's states, in forward mode. The whole-set reverse pass it
    replaces turned both at once, under ``jax.grad``, and kept the tape of every
    output state (19.50 GiB on the fifteen-cell NiBr2 source; 4.5 GiB, twice
    the estimated peak, on nine atoms of NiI2).
    """
    from defumat.scf import orientation
    from defumat.scf.driver import run_scf

    calculator = _iodine_on_two_k_points()
    shapes = []
    original = orientation.spin_turned

    def counted(psi, omega):
        shapes.append(tuple(np.shape(psi)))
        return original(psi, omega)

    monkeypatch.setattr(orientation, "spin_turned", counted)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_scf(calculator.system, calculator.pseudos, conv_thr=1.0e-14,
                         max_iterations=2, k_batch=1, verbose=True)

    assert shapes, "the coupled torque no longer turns the states at all"
    assert all(shape[1] == 1 for shape in shapes), shapes
    torques = [entry.get("orientation_torque") for entry in result.history]
    assert all(t is not None and np.all(np.isfinite(t)) for t in torques)
