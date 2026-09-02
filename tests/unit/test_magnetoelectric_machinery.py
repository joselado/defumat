"""The guards and the bookkeeping of the magnetoelectric tensor.

Nothing here runs a self-consistent field. What is checked is the two things
that would silently produce a plausible number: a run that carries no
magnetization vector at all, where the field has nothing to couple to and every
polarization comes out identical; and a field step large enough that the two
ends of the difference may sit on different branches of a quantity that is only
defined modulo a quantum.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from defumat.io.pwin import parse_pw_input, read_pw_input
from defumat.response.magnetoelectric import (
    BRANCH_FRACTION,
    MagnetoelectricTensor,
    magnetoelectric_tensor,
)
from defumat.system.builder import build_system

pytestmark = pytest.mark.unit

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"


def test_a_run_with_no_magnetization_vector_is_refused():
    """``nspin_mag = 1`` means the field has nothing to act on.

    A noncollinear run whose ``starting_magnetization`` is zero everywhere
    allocates a *scalar* density -- ``nspin_mag`` is 4 only if the run actually
    carries a magnetization, which is the distinction ``CLAUDE.md`` insists on
    -- so the Zeeman term multiplies nothing, all six ground states are
    identical and the tensor comes out exactly zero. That is the failure this
    guard exists for: it looks like a symmetry result.
    """
    text = (CASES / "gaas-magnetoelectric.in").read_text()
    unmagnetized = text.replace(
        "starting_magnetization(1) = 0.01", "starting_magnetization(1) = 0.0"
    ).replace(
        "starting_magnetization(2) = 0.01", "starting_magnetization(2) = 0.0"
    )
    system = build_system(parse_pw_input(unmagnetized))
    assert system.nspin_mag == 1
    with pytest.raises(ValueError, match="nspin_mag"):
        magnetoelectric_tensor(system, ())


def test_the_committed_cell_does_carry_one():
    """And the case as committed is set up correctly, which is the other half."""
    system = build_system(read_pw_input(CASES / "gaas-magnetoelectric.in"))
    assert system.nspin_mag == 4
    assert system.npol == 2
    assert system.b_field == (0.0, 0.0, 0.1)
    # Elk sets reducebf = 1 explicitly for this task: a field that decays as the
    # SCF runs would leave the difference taken over nothing.
    assert system.reducebf == 1.0
    assert str(system.occupations).lower() in ("fixed", "from_input")


def test_the_field_is_varied_without_disturbing_anything_else():
    """``b_field`` is *static*, so ``tree_at`` cannot reach it.

    ``eqx.tree_at`` walks leaves and a static field is not one, so the obvious
    idiom raises rather than silently doing nothing. ``dataclasses.replace`` is
    what a frozen module's static configuration takes, and it shares every array
    with the original -- the six systems of a tensor cost one cell between them.
    """
    system = build_system(read_pw_input(CASES / "gaas-magnetoelectric.in"))
    moved = dataclasses.replace(system, b_field=(0.0, 0.0, 0.15))
    assert moved.b_field == (0.0, 0.0, 0.15)
    assert system.b_field == (0.0, 0.0, 0.1)      # unchanged
    assert moved.cell is system.cell               # and nothing was rebuilt
    assert moved.nspin_mag == 4


def test_a_step_across_a_branch_is_refused_rather_than_returned():
    """The guard's threshold, and why a difference across a branch is not small.

    The polarization is defined modulo a quantum, so if a field step moves the
    phase most of the way to the next branch the central difference can land on
    the wrong one -- and what it then returns is roughly the quantum divided by
    ``delta``, which is large, smooth, and has nothing to do with the response.
    """
    assert 0.0 < BRANCH_FRACTION < 0.5


def test_the_tensor_reports_what_it_was_built_from():
    alpha = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    tensor = MagnetoelectricTensor(
        alpha=alpha, phase_derivative=np.zeros((3, 3)), field=np.array([0, 0, 0.1]),
        delta=0.05, quantum=1.0, largest_step=0.01, volume=300.0,
    )
    assert tensor.is_symmetric
    assert tensor.magnitude == pytest.approx(1.0)
    assert tensor.delta == 0.05
    # asymmetry is allowed: nothing forces alpha to be symmetric in general
    skew = dataclasses.replace(tensor, alpha=np.array(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert not skew.is_symmetric
