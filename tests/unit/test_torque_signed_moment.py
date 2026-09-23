"""The torque's rotated density keeps the sign of the moment.

``rotated_density`` builds the potential the torque differentiates, and the
states it is evaluated in come from ``nc_magnetization_from_lsda``. The two
must lay the moment down identically, point by point, or the potential is not
the one the states were diagonalised in. On an antiferromagnet the difference
is the whole answer: a per-point ``|m|`` turns it into a ferromagnet with the
same absolute moment, while the signed projection keeps its zero net moment.

Host-side only: no SCF, one small synthetic grid.
"""

from __future__ import annotations

import numpy as np
import pytest

from defumat.forces.torque import rotated_density
from defumat.scf.continuation import nc_magnetization_from_lsda

pytestmark = pytest.mark.unit


def _antiferromagnet(axis, shape=(4, 4, 4)):
    """A ``(4, ...)`` density with ``m = +-0.5 axis`` on two sublattices."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    sign = np.where(np.indices(shape).sum(axis=0) % 2 == 0, 1.0, -1.0)
    density = np.zeros((4,) + shape)
    density[0] = 1.0
    density[1:4] = 0.5 * axis.reshape(3, 1, 1, 1) * sign[None]
    return density


@pytest.mark.parametrize("axis", [(0.0, 0.0, 1.0), (1.0, 1.0, 0.0)])
def test_an_antiferromagnet_stays_antiferromagnetic(axis):
    """Rotated onto ``x``, the net moment is zero and the local one is kept."""
    density = _antiferromagnet(axis)
    rotated = np.asarray(rotated_density(density, np.array([1.0, 0.0, 0.0])))
    np.testing.assert_allclose(rotated[0], density[0], atol=1e-14)
    # The net moment: zero for the antiferromagnet, 32 for the ferromagnet a
    # per-point modulus makes of it (64 points at 0.5).
    assert abs(float(rotated[1:4].sum())) < 1e-12
    # The staggered pattern survives: each point carries its own sign.
    np.testing.assert_allclose(np.abs(rotated[1]), 0.5, atol=1e-14)
    np.testing.assert_allclose(rotated[2:4], 0.0, atol=1e-14)


@pytest.mark.parametrize("axis", [(0.0, 0.0, 1.0), (0.3, -0.4, 0.866)])
@pytest.mark.parametrize("direction", [(1.0, 0.0, 0.0), (0.0, 0.6, 0.8)])
def test_the_traceable_rotation_is_the_one_the_states_came_from(axis, direction):
    """``rotated_density`` equals ``nc_magnetization_from_lsda`` on four channels.

    The antiferromagnet is the case the two disagreed on; a ferromagnet is kept
    beside it so the agreement is not only about the sign.
    """
    for density in (_antiferromagnet(axis), np.abs(_antiferromagnet(axis))):
        density = density.copy()
        density[0] = 1.0
        ours = np.asarray(rotated_density(density, np.array(direction)))
        states = np.asarray(nc_magnetization_from_lsda(density, direction))
        np.testing.assert_allclose(ours, states, atol=1e-13)


def test_the_collinear_branch_is_unchanged():
    """Two channels: ``(n_up + n_dn, (n_up - n_dn) n)``, signed, as before."""
    rng = np.random.default_rng(3)
    density = rng.random((2, 3, 3, 3))
    direction = np.array([0.0, 0.6, 0.8])
    rotated = np.asarray(rotated_density(density, direction))
    scalar = density[0] - density[1]
    np.testing.assert_array_equal(rotated[0], density[0] + density[1])
    np.testing.assert_allclose(
        rotated[1:4], direction.reshape(3, 1, 1, 1) * scalar[None], atol=1e-15
    )
    np.testing.assert_allclose(
        rotated, np.asarray(nc_magnetization_from_lsda(density, direction)),
        atol=1e-14,
    )
