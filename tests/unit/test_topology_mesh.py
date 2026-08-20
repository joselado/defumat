"""The k-meshes and the reciprocal-lattice bookkeeping that closes them.

Nothing here needs a wavefunction. It is all integers and fractions, and it is
where the wrap convention is pinned: the neighbour of the last point along a
closed direction is the *first* point plus one reciprocal lattice vector, and
that vector is what every overlap downstream shifts the plane waves by.
"""

import numpy as np
import pytest

from pypresso.topology.mesh import PLANE_AXES, plane_mesh, pumping_mesh, trim_points

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_plane_mesh_spans_the_right_two_directions(axis):
    mesh = plane_mesh((3, 4), axis=axis, offset=0.25)
    d1, d2 = PLANE_AXES[axis]
    points = mesh.points
    assert points.shape == (3, 4, 3)
    assert np.allclose(points[..., axis], 0.25)
    assert np.allclose(points[:, 0, d1], np.arange(3) / 3)
    assert np.allclose(points[0, :, d2], np.arange(4) / 4)


def test_plane_mesh_excludes_the_endpoint():
    """A repeated endpoint would double-count a whole row of plaquettes."""
    mesh = plane_mesh((5, 5))
    assert mesh.points[..., 0].max() < 1.0
    assert mesh.nk == 25


def test_neighbour_wraps_with_a_reciprocal_lattice_vector():
    mesh = plane_mesh((3, 3))
    interior, shift = mesh.neighbour(0, 0, 0)
    assert interior == mesh.index(1, 0)
    assert np.array_equal(shift, [0, 0, 0])

    edge, shift = mesh.neighbour(2, 0, 0)
    assert edge == mesh.index(0, 0)
    assert np.array_equal(shift, mesh.span1)
    # The wrap must be an integer triple: that is what makes it a shift of the
    # Miller index rather than an interpolation.
    assert shift.dtype.kind == "i"


def test_a_wrapped_neighbour_is_the_geometric_next_point():
    mesh = plane_mesh((4, 4))
    target, shift = mesh.neighbour(3, 1, 0)
    physical = mesh.flat()[target] + shift
    assert np.allclose(physical, mesh.points[3, 1] + np.array([0.25, 0.0, 0.0]))


def test_pumping_mesh_ends_exactly_at_one_half():
    """Both ends must be TRI planes; ``0.4999`` is not one."""
    mesh = pumping_mesh(6, 5, axis=2)
    pump = mesh.points[0, :, 1]
    assert np.allclose(pump, [0.0, 0.125, 0.25, 0.375, 0.5])
    assert not mesh.closed[1]
    with pytest.raises(IndexError):
        mesh.neighbour(0, 4, 1)


def test_pumping_mesh_loop_direction_is_closed():
    mesh = pumping_mesh(6, 5)
    target, shift = mesh.neighbour(5, 2, 0)
    assert target == mesh.index(0, 2)
    assert np.array_equal(shift, mesh.span1)


def test_trim_enumeration_order_matches_the_reference():
    """First component slowest -- the order ``elkpy``'s delta tables are in."""
    points = trim_points(3)
    assert points.shape == (8, 3)
    assert np.allclose(points[0], [0, 0, 0])
    assert np.allclose(points[1], [0, 0, 0.5])
    assert np.allclose(points[4], [0.5, 0, 0])
    assert np.allclose(points[7], [0.5, 0.5, 0.5])


def test_two_dimensional_trim_lie_in_the_plane():
    points = trim_points(2, axis=2)
    assert points.shape == (4, 3)
    assert np.allclose(points[:, 2], 0.0)
    assert {tuple(p[:2]) for p in points} == {(0, 0), (0, 0.5), (0.5, 0), (0.5, 0.5)}


def test_a_single_point_mesh_is_rejected_only_when_it_cannot_pump():
    with pytest.raises(ValueError):
        pumping_mesh(4, 1)
