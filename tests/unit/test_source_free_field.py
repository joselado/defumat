"""``projsbf``: the longitudinal part of a vector field, removed.

Elk's ``nosource`` makes the exchange-correlation magnetic field divergence-free
by projecting out its longitudinal component, which on a plane-wave grid is

    B(G) -> B(G) - G (G . B(G)) / |G|^2,

with ``G = 0`` left alone. Everything here is about that one operation, on
fields built by hand where the answer is known in advance, because the run-level
consequences (what it does to a moment, and to the torque that is identically
zero without it) belong to a cell rather than to an array.

**The two cases that have to be told apart.** A gradient is *all* longitudinal
and must be removed entirely; a curl is *all* transverse and must come back
untouched. A projector that did nothing at all would pass the second on its own,
which is why they are tested as a pair and not one at a time.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.basis.gradients import divergence, gradient
from defumat.basis.gvectors import generate_gvectors
from defumat.scf.sourcefree import longitudinal_field, project_source_free
from defumat.system.cell import Cell

pytestmark = [pytest.mark.unit]


@pytest.fixture(scope="module")
def grid():
    """A cubic cell and its dense G set: nothing here depends on the crystal."""
    cell = Cell.from_vectors(np.eye(3) * 10.0, alat=10.0)
    return cell, generate_gvectors(cell, 40.0)


def _phase(gvectors, cell, index):
    """``exp(i G . r)`` as a real field: one plane wave, on the grid."""
    n = gvectors.grid
    coords = np.stack(np.meshgrid(
        np.arange(n[0]) / n[0], np.arange(n[1]) / n[1], np.arange(n[2]) / n[2],
        indexing="ij",
    ))
    return np.cos(2.0 * np.pi * sum(index[d] * coords[d] for d in range(3)))


def test_a_gradient_is_removed_entirely(grid):
    """``B = grad f`` has no transverse part, so the projection must return zero."""
    cell, gvectors = grid
    scalar = jnp.asarray(_phase(gvectors, cell, (1, 2, 0))
                         + 0.5 * _phase(gvectors, cell, (2, -1, 1)))
    from defumat.basis.fft import r_to_g
    field = gradient(r_to_g(scalar, gvectors.fft_index), gvectors, cell)

    transverse = field - longitudinal_field(field, gvectors, cell)
    assert np.abs(np.asarray(transverse)).max() < 1.0e-10 * np.abs(
        np.asarray(field)).max()


def test_a_divergence_free_field_is_untouched(grid):
    """The other half of the pair, and the one a do-nothing projector passes.

    ``B = G x e`` for a single ``G`` is transverse by construction, so nothing
    may be subtracted from it -- and *nothing* means to round-off, not to the
    accuracy of a round trip through the sphere, which is why the projection is
    written as a subtraction of the longitudinal part rather than as a rebuild
    of the whole field.
    """
    cell, gvectors = grid
    index = (1, 0, 2)
    g = np.asarray(gvectors.cartesian(cell))
    # The cartesian G of that Miller index, from the reciprocal cell directly.
    gvec = 2.0 * np.pi * np.asarray(index) @ np.linalg.inv(np.asarray(cell.at))
    direction = np.cross(gvec, [0.0, 0.0, 1.0])
    if np.linalg.norm(direction) < 1.0e-12:
        direction = np.cross(gvec, [1.0, 0.0, 0.0])
    profile = _phase(gvectors, cell, index)
    field = jnp.asarray(direction[:, None, None, None] * profile[None])

    removed = longitudinal_field(field, gvectors, cell)
    assert np.abs(np.asarray(removed)).max() < 1.0e-10 * np.abs(
        np.asarray(field)).max()


def test_the_projection_kills_the_divergence(grid):
    """A general field, and the quantity the whole thing is named after."""
    cell, gvectors = grid
    rng = np.random.default_rng(0)
    scalar = _phase(gvectors, cell, (1, 1, 0)) + 0.3 * _phase(gvectors, cell, (0, 2, 1))
    field = np.stack([scalar, 0.7 * np.roll(scalar, 3, axis=0),
                      -0.4 * np.roll(scalar, 5, axis=1)])
    v_xc = jnp.asarray(np.concatenate([np.zeros((1,) + field.shape[1:]), field]))

    before = divergence(jnp.asarray(field), gvectors, cell)
    projected = project_source_free(v_xc, gvectors, cell)
    after = divergence(jnp.real(projected[1:]), gvectors, cell)

    assert float(jnp.sum(after ** 2)) < 1.0e-16 * float(jnp.sum(before ** 2))
    # The scalar component is not a field and must not move.
    assert np.abs(np.asarray(projected[0] - v_xc[0])).max() == 0.0


def test_the_uniform_component_survives(grid):
    """``G = 0`` is excluded, and it has to be: a uniform field is already
    divergence-free, and a Poisson equation cannot move a constant."""
    cell, gvectors = grid
    uniform = jnp.asarray(np.broadcast_to(
        np.array([0.3, -0.2, 0.5])[:, None, None, None],
        (3,) + tuple(gvectors.grid),
    ).copy())
    removed = longitudinal_field(uniform, gvectors, cell)
    assert np.abs(np.asarray(removed)).max() < 1.0e-12


def test_the_tangent_is_finite_at_g_zero(grid):
    """The ``0/0`` at ``G = 0`` is masked rather than divided and repaired, so
    a derivative taken through the potential has to come back finite.

    This is the trap ``CLAUDE.md`` lists first, and the failure it produces is
    not an error: ``jnp.where`` after a division leaves the *primal* right and
    the *tangent* a nan, which then reaches every response built on the
    potential and nothing before it complains.
    """
    import jax

    cell, gvectors = grid
    profile = _phase(gvectors, cell, (1, 0, 1))
    field = jnp.asarray(np.stack([profile, 0.5 * profile, -profile]))

    def total(scale):
        removed = longitudinal_field(scale * field, gvectors, cell)
        return jnp.sum(removed ** 2)

    tangent = jax.grad(total)(1.0)
    assert np.isfinite(float(tangent))
