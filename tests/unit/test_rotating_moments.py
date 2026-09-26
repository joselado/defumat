"""Route C's pieces, before any SCF: the stepper, the spinor turn, the mixer's history.

``ORIENTATION-NEXT.md`` Route C turns the input density after every mix. What
can be checked without a Hamiltonian is that each piece turns what it should and
nothing else: the spin-1/2 rotation turns a spinor's magnetization by the same
matrix the density is turned by, the mixer is equivariant under a common turn of
its whole history, and the step converges on a quadratic it is handed.
"""

import numpy as np
import pytest

from defumat.forces.torque import rotate_texture
from defumat.scf.mixing import get_mixer
from defumat.scf.orientation import (
    OrientationStepper,
    rotate_spinors,
    rotation_matrix,
    spinor_rotation,
)

pytestmark = pytest.mark.unit

PAULI = (
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]]),
    np.array([[1, 0], [0, -1]], dtype=complex),
)


def test_the_spinor_turn_turns_the_magnetization_by_the_density_s_matrix():
    rng = np.random.default_rng(1)
    omega = np.array([0.3, -0.5, 0.7])
    npwx = 6
    psi = rng.normal(size=(2, 3, 2 * npwx)) + 1j * rng.normal(size=(2, 3, 2 * npwx))
    turned = rotate_spinors(psi, omega)
    for state, image in ((psi[1, 2], turned[1, 2]),):
        pair = np.stack([state[:npwx], state[npwx:]])
        pair_turned = np.stack([image[:npwx], image[npwx:]])
        m = np.array([np.real(np.einsum("sg,st,tg->", pair.conj(), s, pair)) for s in PAULI])
        m2 = np.array([np.real(np.einsum("sg,st,tg->", pair_turned.conj(), s, pair_turned))
                       for s in PAULI])
        np.testing.assert_allclose(m2, rotation_matrix(omega) @ m, atol=1e-12)
    u = spinor_rotation(omega)
    np.testing.assert_allclose(u.conj().T @ u, np.eye(2), atol=1e-15)


def _field(rng, shape=(4, 3, 3, 3)):
    field = rng.normal(size=shape)
    field[0] += 3.0
    return field


def test_the_mixer_is_equivariant_under_a_common_turn_of_its_history():
    """Turning the history and the new pair is turning the mixed density.

    Anderson's coefficients come from inner products of residuals, which a
    common spin rotation leaves alone, so ``mix(R a, R b)`` after
    ``rotate_history(R)`` must be ``R mix(a, b)``: the property that lets Route C
    keep the history across a step instead of paying for a reset.
    """
    rng = np.random.default_rng(2)
    turn = rotation_matrix(np.array([0.4, 0.2, -0.9]))

    def rotated(vector):
        return np.asarray(rotate_texture(vector.reshape(4, 3, 3, 3), turn)).ravel()

    plain = get_mixer("anderson", beta=0.3, history=4)
    turned = get_mixer("anderson", beta=0.3, history=4)
    pairs = [(_field(rng).ravel(), _field(rng).ravel()) for _ in range(4)]
    for rho_in, rho_out in pairs[:3]:
        plain.mix(rho_in, rho_out)
        turned.mix(rho_in, rho_out)
    turned.rotate_history(rotated)
    rho_in, rho_out = pairs[3]
    expected = rotated(plain.mix(rho_in, rho_out))
    got = turned.mix(rotated(rho_in), rotated(rho_out))
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_the_step_converges_on_a_quadratic_and_respects_its_bounds():
    """``E = w . K w / 2`` in the accumulated rotation vector, one null direction.

    The first step is ``first_step`` long along ``-G``, no step is longer than
    ``trust``, nothing moves while ``dr2`` is above ``start``, and BFGS then
    finds the minimum of a curvature two orders below one, the size of an
    anisotropy, in a handful of steps. The null direction (a collinear
    texture's turn about its moment) is never moved along.
    """
    curvature = np.diag([8.0e-5, 8.0e-5, 0.0])
    stepper = OrientationStepper(trust=0.1, first_step=0.05, start=1.0e-5)
    w = np.array([0.5, -0.3, 0.2])
    assert stepper.propose(curvature @ w, accuracy=1.0e-3) is None
    first = stepper.propose(curvature @ w, accuracy=1.0e-6)
    assert np.linalg.norm(first) == pytest.approx(0.05)
    w = w + first
    for _ in range(12):
        step = stepper.propose(curvature @ w, accuracy=1.0e-6)
        if step is None:
            break
        assert np.linalg.norm(step) <= 0.1 + 1e-15
        w = w + step
    assert np.linalg.norm(w[:2]) < 1.0e-6
    assert w[2] == pytest.approx(0.2)


def test_the_first_order_spin_turn_is_the_exact_one_to_first_order():
    """``spin_turned`` is what the PAW torque differentiates through the states."""
    from defumat.scf.orientation import spin_turned

    rng = np.random.default_rng(4)
    psi = rng.normal(size=(2, 3, 10)) + 1j * rng.normal(size=(2, 3, 10))
    for size in (1.0e-4, 1.0e-6):
        omega = size * np.array([0.3, -0.5, 0.7])
        gap = np.abs(np.asarray(spin_turned(psi, omega)) - rotate_spinors(psi, omega)).max()
        assert gap < 10 * size ** 2 * np.abs(psi).max()
