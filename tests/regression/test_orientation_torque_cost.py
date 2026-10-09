"""What the SCF's per-iteration orientation torque costs on PAW (``OPEN.md`` Part XXII item 1).

Every noncollinear magnetic SCF records the torque on a rigid turn of its input
texture, and on PAW the one-centre field's share of it. With spin-orbit coupling
that share is taken through the output states, a reverse pass through
``becsum(U(w) psi)`` over every one of them, because the fully-relativistic
``becsum`` is not a vector under a spin rotation. Without the coupling it is one,
exactly, and the vector turn of the input ``becsum`` is the derivative itself: the
states are not needed, and their tape stopped the fifteen-cell NiBr2 source at
``soc_scale = 0`` before its first iteration (19.50 GiB). The test holds the run
without the coupling to the wavefunction-free branch while it still records the
torque.
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


def test_without_the_coupling_the_paw_torque_does_not_differentiate_the_states(
        monkeypatch):
    """No spin turn of any output state, and a torque in every history entry.

    ``spin_turned`` is what both the whole-set and the streamed branch of
    ``_onecenter_torque`` differentiate; counted here, it was called once an
    iteration before the fix and is called never after it.
    """
    from defumat.scf import orientation
    from defumat.scf.driver import run_scf

    calculator = _oxygen_pair()
    assert calculator.system.noncolin and not calculator.system.lspinorb
    calls = []
    original = orientation.spin_turned

    def counted(psi, omega):
        calls.append(np.shape(psi))
        return original(psi, omega)

    monkeypatch.setattr(orientation, "spin_turned", counted)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_scf(calculator.system, calculator.pseudos, conv_thr=1.0e-14,
                         max_iterations=2)

    assert calls == []
    torques = [entry.get("orientation_torque") for entry in result.history]
    assert len(torques) == 2 and all(t is not None and len(t) == 3 for t in torques)
    assert all(np.all(np.isfinite(t)) for t in torques)


def test_with_the_coupling_the_states_are_turned_a_k_chunk_at_a_time(monkeypatch):
    """With the coupling the states are needed, and are walked, not taped whole.

    A fully-relativistic PAW iodine atom on two k-points, ``k_batch = 1``: every
    spin turn the torque takes is of one k-point's states, in forward mode. The
    whole-set reverse pass it replaces turned both at once, under ``jax.grad``,
    and kept the tape of every output state (19.50 GiB on the fifteen-cell NiBr2
    source; 4.5 GiB, twice the estimated peak, on nine atoms of NiI2).
    """
    from defumat.scf import orientation
    from defumat.scf.driver import run_scf

    text = open("tests/data/qe/i-atom-soc-paw.in").read().replace(
        "K_POINTS (automatic)\n 1 1 1 0 0 0", "K_POINTS (automatic)\n 2 1 1 0 0 0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_text(text, PSEUDO, announce=False)
    assert calculator.system.lspinorb and calculator.system.kpoints.nk == 2
    shapes = []
    original = orientation.spin_turned

    def counted(psi, omega):
        shapes.append(tuple(np.shape(psi)))
        return original(psi, omega)

    monkeypatch.setattr(orientation, "spin_turned", counted)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_scf(calculator.system, calculator.pseudos, conv_thr=1.0e-14,
                         max_iterations=2, k_batch=1)

    assert shapes, "the coupled torque no longer turns the states at all"
    assert all(shape[1] == 1 for shape in shapes), shapes
    torques = [entry.get("orientation_torque") for entry in result.history]
    assert all(t is not None and np.all(np.isfinite(t)) for t in torques)
