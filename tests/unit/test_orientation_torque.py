"""The rotation of a whole texture, before any Hamiltonian sees it.

``ORIENTATION-NEXT.md`` Route A turns every spin by one rotation ``R`` and takes
the torque for the three generators at once. What can be checked without a
diagonalisation is checked here: that the rotation keeps what it must (the
charge, ``|m|`` at every point, the angles between moments), that on a collinear
texture it is the rotation P60 already validated, that its derivative at the
origin exists where Rodrigues' formula has none, and that the system is turned
with the density.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.forces.torque import (
    cross_matrix,
    rotate_texture,
    rotated_density,
    rotation_near,
)
from defumat.system.builder import system_from_file
from defumat.workflows.anisotropy import (
    _checked_rotation,
    _reference_axis,
    _rotation_taking,
    _with_quantization_axis,
    _with_rotation,
    euler_from_rotation,
    rotation_from_euler,
)
from tests.conftest import GENERATED

pytestmark = pytest.mark.unit


def _texture(seed=0, shape=(4, 5, 3)):
    """A four-channel field whose moments point every which way."""
    rng = np.random.default_rng(seed)
    charge = 1.0 + rng.random(shape)
    moment = rng.normal(size=(3,) + shape)
    return np.concatenate([charge[None], moment])


def test_the_rotation_keeps_the_charge_and_every_moment_length():
    field = _texture()
    rotation = rotation_from_euler(0.4, 1.2, -2.1)
    turned = np.asarray(rotate_texture(field, rotation))
    assert np.array_equal(turned[0], field[0])
    np.testing.assert_allclose(np.linalg.norm(turned[1:], axis=0),
                               np.linalg.norm(field[1:], axis=0), rtol=1e-14)
    # ... and it is R m, point by point, not some other map with those two
    # properties: the angle between two points' moments is kept too.
    np.testing.assert_allclose(turned[1:, 1, 2, 0], rotation @ field[1:, 1, 2, 0],
                               rtol=1e-14)
    a, b = field[1:, 0, 0, 0], field[1:, 3, 4, 2]
    ta, tb = turned[1:, 0, 0, 0], turned[1:, 3, 4, 2]
    assert ta @ tb == pytest.approx(a @ b, rel=1e-13)


def test_on_a_collinear_texture_it_is_the_plane_torque_s_rotation():
    """The rotation P60 validated, reached through a matrix instead of an angle.

    A collinear density laid along ``z`` and turned by ``R`` must be the same
    field as the collinear density laid along ``R z`` directly, signed regions
    included, since an antiferromagnet keeps its antiparallel parts only through
    that sign.
    """
    rng = np.random.default_rng(3)
    up = 1.0 + rng.random((4, 4, 4))
    down = up + rng.normal(scale=0.5, size=(4, 4, 4))  # both signs of m
    collinear = np.stack([up, down])
    along_z = np.asarray(rotated_density(collinear, (0.0, 0.0, 1.0)))
    rotation = rotation_from_euler(0.9, 0.7, 0.0)
    direction = rotation @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        np.asarray(rotate_texture(along_z, rotation)),
        np.asarray(rotated_density(collinear, tuple(direction))),
        rtol=0, atol=1e-14,
    )


def test_a_two_channel_field_is_refused():
    with pytest.raises(ValueError, match="four-channel"):
        rotate_texture(np.zeros((2, 3, 3, 3)), np.eye(3))


def test_the_derivative_at_the_origin_is_the_generator_and_is_finite():
    """No ``0/0`` where every call evaluates it.

    ``d/dw (R(w) R0 v)`` at ``w = 0`` is ``e_a x (R0 v)``. It is checked against
    a central difference of the **exact** rotation, so ``rotation_near`` is
    seen to agree with ``exp([w]x)`` to first order and not merely to be finite.
    """
    base = rotation_from_euler(0.3, 0.8, 1.9)
    vector = jnp.asarray([0.2, -1.1, 0.7])
    probe = jnp.asarray([0.5, 0.1, -0.3])

    def value(omega):
        return probe @ (rotation_near(omega, base) @ vector)

    gradient = np.asarray(jax.grad(value)(jnp.zeros(3)))
    assert np.all(np.isfinite(gradient))

    def exact(omega):
        angle = np.linalg.norm(omega)
        axis = omega / angle
        k = np.asarray(cross_matrix(axis))
        turn = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k
        return float(np.asarray(probe) @ (turn @ base @ np.asarray(vector)))

    step = 1.0e-5
    central = np.array([
        (exact(step * e) - exact(-step * e)) / (2 * step) for e in np.eye(3)
    ])
    np.testing.assert_allclose(gradient, central, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(np.asarray(rotation_near(jnp.zeros(3), base)), base,
                               rtol=0, atol=0)


def test_euler_angles_round_trip_and_name_the_tilt():
    for angles in [(0.3, 1.1, -0.7), (-2.0, 0.4, 2.9), (1.0, 2.5, 0.0)]:
        rotation = rotation_from_euler(*angles)
        np.testing.assert_allclose(euler_from_rotation(rotation), angles, atol=1e-12)
    # ``beta`` is the tilt of z, and at ``beta = 0`` only ``alpha + gamma`` is
    # a rotation at all, which is why nothing is differentiated in these angles.
    rotation = rotation_from_euler(0.0, 0.6, 0.0)
    assert np.arccos(rotation[2, 2]) == pytest.approx(0.6)
    np.testing.assert_allclose(rotation_from_euler(0.2, 0.0, 0.5),
                               rotation_from_euler(0.7, 0.0, 0.0), atol=1e-15)


def test_an_improper_or_skewed_rotation_is_refused():
    with pytest.raises(ValueError, match="determinant -1"):
        _checked_rotation(-np.eye(3))
    with pytest.raises(ValueError, match="not orthogonal"):
        _checked_rotation(np.diag([1.0, 1.0, 1.1]))
    np.testing.assert_array_equal(_checked_rotation(None), np.eye(3))


def _canted():
    """Two iron moments along ``x`` and ``y``: a texture a turn about one moment moves."""
    return system_from_file(GENERATED / "fe2-canted-soc.in")


def test_the_whole_rotation_is_the_axis_turn_where_both_apply():
    """``_with_rotation`` of the smallest rotation is ``_with_quantization_axis``.

    Angle for angle, on a canted cell, so the refactor that split one out of the
    other is seen to have changed nothing the force theorem and P60 depend on.
    """
    system = _canted()
    own = np.asarray(_reference_axis(system))
    wanted = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    by_axis = _with_quantization_axis(system, tuple(wanted))
    by_rotation = _with_rotation(system, _rotation_taking(own, wanted))
    np.testing.assert_allclose(by_rotation.angle1, by_axis.angle1, atol=1e-12)
    np.testing.assert_allclose(by_rotation.angle2, by_axis.angle2, atol=1e-12)


def test_a_turn_about_the_reference_axis_moves_the_other_moment():
    """The degree of freedom a direction cannot express and a rotation can.

    Turning the canted pair by 90 degrees about the first moment (``x``) leaves
    that moment where it is and carries the second from ``y`` to ``z``; the
    angle between them stays 90 degrees. ``_with_quantization_axis`` has no way
    to ask for this, since the direction it is given does not change.
    """
    from defumat.scf.continuation import direction_from_angles

    system = _canted()
    about_x = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    turned = _with_rotation(system, about_x)
    first = np.asarray(direction_from_angles(turned.angle1[0], turned.angle2[0]))
    second = np.asarray(direction_from_angles(turned.angle1[1], turned.angle2[1]))
    np.testing.assert_allclose(first, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(second, [0.0, 0.0, 1.0], atol=1e-12)


def test_the_identity_returns_the_system_untouched():
    system = _canted()
    assert _with_rotation(system, np.eye(3)) is system
