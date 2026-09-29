"""A Chern number and an orbital magnetization walked across the mesh.

``GPU-MEMORY-NEXT.md`` item 5. Both quantities are built from overlaps between
neighbouring k-points, and both used to diagonalise their whole mesh at once.
Streamed, a plane mesh is diagonalised a column at a time and a volume mesh a
plane at a time, with only the link phases (or the per-k terms) kept.

The standard is the whole-mesh route, and on a model the two must agree to
round-off: the states are the same ``eigh`` at the same points and the per-k
arithmetic is the same function. **Each test also checks that the streamed
route ran** -- that the source was asked for one column or one plane at a
time and never for the mesh -- because the values alone would pass with the
switch doing nothing.

The plane-wave counterparts, where each column is its own diagonalisation on
its own sphere and the agreement is the eigensolver's threshold, are in
``tests/regression/test_topology.py``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.topology.invariants import ModelSource, chern_number
from defumat.topology.mesh import volume_mesh
from defumat.topology.orbital_magnetization import (
    orbital_magnetization_sums,
    streamed_orbital_magnetization_sums,
)
from defumat.topology.states import ModelStates
from tests.models import haldane

pytestmark = pytest.mark.unit


class _Counting(ModelSource):
    """A model source that records how many points each call asked for."""

    def __init__(self, hamiltonian, nocc):
        super().__init__(hamiltonian=hamiltonian, nocc=nocc)
        self.calls = []

    def states(self, points, **kwargs):
        self.calls.append(len(np.asarray(points).reshape(-1, 3)))
        return super().states(points, **kwargs)


def _layered(gap: float = 0.3, modulation: float = 0.3):
    """Two Haldane copies, the second raised, with a mass that follows ``k_3``.

    The ``k_3`` dependence is what makes the third direction's derivative --
    and so the neighbours *across* the planes the walk cuts -- carry something,
    and two occupied bands make each overlap a matrix rather than a phase.
    Both copies stay topological (``|mass| < 3 sqrt(3) t2 = 1.04``).
    """
    block = haldane(t2=0.2, mass=0.0)
    sz = jnp.diag(jnp.array([1.0, -1.0], dtype=complex))

    def hamiltonian(k):
        k = jnp.asarray(k)
        small = block(k) + modulation * jnp.cos(2.0 * jnp.pi * k[2]) * sz
        out = jnp.zeros((4, 4), dtype=complex)
        out = out.at[:2, :2].set(small)
        return out.at[2:, 2:].set(small + gap * jnp.eye(2, dtype=complex))

    return hamiltonian


def test_a_streamed_chern_number_is_the_whole_plane_one():
    """Column by column, the same flux to round-off and the same integer."""
    model = haldane(t2=0.2, mass=0.0)
    whole = chern_number(ModelSource(hamiltonian=model, nocc=1), shape=(7, 6),
                         stream=False)
    source = _Counting(model, 1)
    streamed = chern_number(source, shape=(7, 6), stream=True)

    assert source.calls == [6] * 7, "the plane was not walked a column at a time"
    assert whole.chern_number == pytest.approx(-1.0, abs=1e-12)
    assert streamed.chern_number == pytest.approx(whole.chern_number, abs=1e-12)
    assert np.max(np.abs(whole.flux)) > 1e-2  # a flux, not a null
    np.testing.assert_allclose(streamed.flux, whole.flux, atol=1e-13)


def test_a_model_source_does_not_stream_by_default():
    """``streams`` is absent on a model, so the default is the whole plane."""
    source = _Counting(haldane(t2=0.2, mass=0.0), 1)
    chern_number(source, shape=(4, 5))
    assert source.calls == [20]


@pytest.mark.parametrize("divisions", [
    (6, 4, 3),
    # About 4.5 s each, nearly all of it compiling a new plane shape.
    pytest.param((3, 6, 1), marks=pytest.mark.slow),
    pytest.param((5, 5, 5), marks=pytest.mark.slow),
])
def test_a_streamed_orbital_magnetization_is_the_whole_mesh_one(divisions):
    """Plane by plane, the same three zone sums and the same worst overlap.

    ``(6, 4, 3)`` cuts along the first direction with six planes, so the
    middle ones are let go during the walk; ``(3, 6, 1)`` cuts along the
    second, with the third flat; ``(5, 5, 5)`` is a cube.
    """
    model = _layered()
    mesh = volume_mesh(divisions)
    whole = orbital_magnetization_sums(
        ModelStates.solve(model, mesh.flat(), nocc=2), mesh)
    source = _Counting(model, 2)
    streamed = streamed_orbital_magnetization_sums(source, mesh)

    cut = int(np.argmax(divisions))
    assert source.calls == [mesh.nk // divisions[cut]] * divisions[cut], (
        "the mesh was not walked a plane at a time, each plane once")
    for key in ("lc", "ic", "curvature"):
        assert np.max(np.abs(whole[key])) > 1e-3, key  # not a null
        np.testing.assert_allclose(streamed[key], whole[key], atol=1e-12,
                                   err_msg=key)
    assert streamed["determinant"] == pytest.approx(whole["determinant"],
                                                    rel=1e-12)
    assert streamed["flat_directions"] == whole["flat_directions"]
