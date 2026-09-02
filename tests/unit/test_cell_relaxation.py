"""P29: the cell as nine coordinates, checked without a plane-wave code.

The variable-cell BFGS and the gradient it consumes are arithmetic over a
``(3, 3)`` matrix, so they can be validated against an energy that is *written
down* rather than computed -- which separates a bug in the optimizer from a bug
in the stress it is handed. Everything expensive is in
``tests/regression/test_vc_relax.py``.
"""

import numpy as np
import pytest

from defumat.relax.bfgs import BFGS
from defumat.relax.cell import cell_dofree_mask, cell_force
from defumat.units import RY_TO_KBAR

pytestmark = pytest.mark.unit

#: A target metric with no symmetry at all, so nothing about the answer is
#: enforced by the shape of the problem.
TARGET = np.diag([25.0, 36.0, 49.0]) + 0.5 * (np.ones((3, 3)) - np.eye(3))
STIFFNESS = 0.002


def model_energy(at: np.ndarray) -> float:
    """``E = (k/2) |g - g0|^2`` with ``g = at at^T``: minimised at ``g = g0``."""
    g = np.asarray(at) @ np.asarray(at).T
    return 0.5 * STIFFNESS * float(((g - TARGET) ** 2).sum())


def model_stress(at: np.ndarray, step: float = 1.0e-6) -> np.ndarray:
    """``sigma = -(1/Omega) dE/d(epsilon)`` by central differences.

    Deliberately a finite difference rather than the analytic derivative: what
    is being tested is that the optimizer finds the stationary point of the
    energy it is *given*, and a hand-derived stress would put the same algebra
    on both sides of that statement.
    """
    at = np.asarray(at, dtype=float)
    omega = abs(np.linalg.det(at))
    out = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            e = np.zeros((3, 3))
            e[a, b] = step
            plus = model_energy(at @ (np.eye(3) + e).T)
            minus = model_energy(at @ (np.eye(3) - e).T)
            out[a, b] = -(plus - minus) / (2 * step) / omega
    return 0.5 * (out + out.T)


def _minimise(pressure: float, cell_thr: float = 1.0e-6, steps: int = 80):
    bfgs = BFGS(
        np.diag([5.6, 5.7, 6.9]), energy_thr=1e-4, grad_thr=1e-3,
        variable_cell=True, pressure=pressure, cell_thr=cell_thr,
    )
    tau, converged = np.zeros((1, 3)), False
    for _ in range(steps):
        tau, converged = bfgs.step(
            tau, model_energy(bfgs.at), np.zeros((1, 3)),
            stress=model_stress(bfgs.at),
        )
        if converged:
            break
    return bfgs, converged


def test_the_cell_relaxes_to_zero_stress_at_zero_pressure():
    bfgs, converged = _minimise(0.0)
    assert converged and not bfgs.failed
    assert np.abs(model_stress(bfgs.at)).max() < 1e-6


def test_the_relaxed_cell_carries_the_applied_pressure():
    """``sigma = P I``, not ``sigma = 0`` -- the whole content of the enthalpy.

    A cell block that minimised the *energy* would converge just as happily and
    land on the zero-pressure answer, so the check that separates the two is
    that a finite pressure moves the result: the volume has to fall.
    """
    pressure = 1.0e-4  # Ry/bohr^3, about 14.7 kbar
    relaxed, converged = _minimise(pressure)
    free, _ = _minimise(0.0)
    assert converged and not relaxed.failed
    residue = np.abs(model_stress(relaxed.at) - pressure * np.eye(3)).max()
    assert residue < 1e-6, f"{residue * RY_TO_KBAR:.4f} kbar from the target"
    assert relaxed.omega < free.omega, "pressure has to compress the cell"


def test_a_fixed_cell_step_is_untouched_by_the_cell_block():
    """``variable_cell = False`` must be the same arithmetic it always was.

    ``n = 3 nat`` is a special case of ``n = 3 (nat + 3)`` only if the extra
    block changes nothing when it is absent, and the metric being rebuilt every
    step is the part of P29 that reaches a fixed-cell run.
    """
    at = np.diag([5.0, 5.0, 5.0])
    bfgs = BFGS(at, energy_thr=1e-10, grad_thr=1e-10)
    positions = np.array([[0.1, 0.0, 0.0], [2.6, 2.5, 2.5]])
    force = np.array([[-0.05, 0.0, 0.0], [0.05, 0.0, 0.0]])
    moved, converged = bfgs.step(positions, -1.0, force)
    assert not converged
    assert bfgs.metric_blocks.shape == (2, 3, 3)
    assert np.allclose(bfgs.at, at), "a fixed-cell step must not move the cell"
    # The step is -H grad with H the inverse metric, i.e. the cartesian force
    # divided by nothing at all on a cubic cell of side 5.
    assert np.allclose(moved - positions, force * bfgs.trust_radius / 0.05, atol=1e-12)


def test_the_cell_gradient_is_the_stress_and_comes_back_again():
    """``dH/dh = Omega (P I - sigma) h^-T``, contracted back with ``h^T``."""
    at = np.array([[5.1, 0.0, 0.0], [0.4, 6.2, 0.0], [0.2, 0.3, 7.0]])
    h, omega = at.T, abs(np.linalg.det(at))
    sigma = np.array([[0.003, 0.0004, 0.0002],
                      [0.0004, 0.005, 0.0001],
                      [0.0002, 0.0001, 0.007]])
    pressure = 0.001
    recovered = cell_force(sigma, h, omega, pressure) @ h.T / omega
    assert np.allclose(recovered, pressure * np.eye(3) - sigma, atol=1e-14)


def test_a_positive_stress_makes_the_cell_want_to_expand():
    """The sign, which QE's double negative in ``cell_force`` makes easy to lose.

    A compressed crystal has a positive ``tr sigma / 3`` in this package's
    convention, and at zero applied pressure it must lower its enthalpy by
    growing -- so the BFGS step ``-H grad`` has to be outward.
    """
    at = np.diag([5.0, 5.0, 5.0])
    h, omega = at.T, 125.0
    gradient = cell_force(np.eye(3) * 0.002, h, omega)
    step = -gradient  # the inverse Hessian is positive definite, so this is the sign
    assert np.all(np.diag(step) > 0.0)


def test_cell_dofree_masks_what_it_says():
    assert np.array_equal(cell_dofree_mask(None), np.ones((3, 3)))
    assert np.array_equal(cell_dofree_mask("z"), np.diag([0.0, 0.0, 1.0]))
    assert np.array_equal(cell_dofree_mask("xyz"), np.eye(3))
    # 'fixc' holds a whole lattice vector, which is a *column* of h.
    expected = np.ones((3, 3))
    expected[:, 2] = 0.0
    assert np.array_equal(cell_dofree_mask("fixc"), expected)
    # 'xy' frees two diagonal entries; '2Dxy' frees the whole block, and QE
    # leaves the difference as three commented-out lines in ``init_dofree``.
    assert cell_dofree_mask("xy")[0, 1] == 0.0
    assert cell_dofree_mask("2Dxy")[0, 1] == 1.0


def test_a_frozen_cell_component_stays_frozen_through_the_hessian():
    """The mask is re-applied after *every* product with the inverse Hessian.

    ``inv_hess`` stops being block diagonal after the first update, so a mask
    applied once to the gradient leaks a free component into a frozen one. The
    check is that a run whose cell may only stretch along ``z`` produces a cell
    that differs from its starting one in ``h(3,3)`` and nowhere else, after
    enough steps for the Hessian to have accumulated off-diagonal weight.
    """
    start = np.diag([5.6, 5.7, 6.9])
    bfgs = BFGS(start, energy_thr=1e-12, grad_thr=1e-12, variable_cell=True,
                cell_thr=1e-9, cell_mask=cell_dofree_mask("z"))
    tau = np.zeros((1, 3))
    for _ in range(12):
        tau, converged = bfgs.step(
            tau, model_energy(bfgs.at), np.zeros((1, 3)),
            stress=model_stress(bfgs.at),
        )
        if converged:
            break
    assert bfgs.accepted_steps > 2, "the Hessian needs updates to leak through"
    difference = np.abs(bfgs.at - start)
    assert difference[2, 2] > 1e-3, "the free component should have moved"
    difference[2, 2] = 0.0
    assert difference.max() < 1e-14, "a frozen component moved"


@pytest.mark.parametrize("name", ["shape", "2Dshape", "volume", "ibrav", "ibrav+all"])
def test_a_constraint_beyond_its_mask_is_refused(name):
    """Not run as the mask alone, which is a different calculation entirely."""
    with pytest.raises(NotImplementedError, match="cell_dofree"):
        cell_dofree_mask(name)


def test_an_unknown_cell_dofree_is_rejected():
    with pytest.raises(ValueError, match="cell_dofree"):
        cell_dofree_mask("sideways")


@pytest.mark.parametrize("name", ["damp-w", "damp-pr", "pr", "w", "sd"])
def test_a_cell_dynamics_that_is_a_different_optimizer_is_refused(name):
    """``vc-relax1`` and ``vc-relax2`` in QE's suite ask for ``damp-w``.

    Wentzcovitch and Parrinello-Rahman damped dynamics carry a fictitious cell
    mass (``wmass``) and take a different path down the same surface. Running
    them as BFGS would reach a similar answer and report it under a name that
    did not happen.
    """
    from defumat.workflows.vc_relax import _check_cell_dynamics

    with pytest.raises(NotImplementedError, match="cell_dynamics"):
        _check_cell_dynamics(name)


def test_bfgs_is_the_cell_dynamics_that_runs():
    from defumat.workflows.vc_relax import _check_cell_dynamics

    _check_cell_dynamics("bfgs")
    _check_cell_dynamics(None)
    with pytest.raises(ValueError, match="cell_dynamics"):
        _check_cell_dynamics("quenched")
