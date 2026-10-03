"""The group averages walked over the operations are the batched gather's.

:data:`~defumat.system.symmetry.GATHER_BUDGET_BYTES` decides whether an average
gathers the field for every operation at once (``nsym`` copies of it) or walks
the operations in a scan. The two are the same sum in a different order, so each
walked average is held to its batched one on random fields and maps, with the
budget forced to zero so the walk is taken.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import defumat.system.symmetry as symmetry

pytestmark = [pytest.mark.unit]

NSYM, NAT, NGM = 6, 3, 50


def _maps(seed=0):
    rng = np.random.default_rng(seed)
    permutations = jnp.asarray(np.stack([rng.permutation(NGM) for _ in range(NSYM)]))
    phases = jnp.exp(1j * jnp.asarray(rng.random((NSYM, NGM))))
    rotations = jnp.asarray(rng.standard_normal((NSYM, 3, 3)))
    mapping = np.stack([rng.permutation(NAT) for _ in range(NSYM)])
    return rng, permutations, phases, rotations, mapping


def _field(rng, shape):
    return jnp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def _both(monkeypatch, average):
    batched = np.asarray(average())
    monkeypatch.setattr(symmetry, "GATHER_BUDGET_BYTES", 0)
    walked = np.asarray(average())
    return batched, walked


@pytest.mark.parametrize("kind", ["scalar", "vector", "spin", "displacement", "tensor"])
def test_the_walked_average_is_the_batched_one(monkeypatch, kind):
    rng, permutations, phases, rotations, mapping = _maps()
    spin = jnp.asarray(rng.standard_normal((NSYM, 3, 3)))
    if kind == "scalar":
        field = _field(rng, (NGM,))
        average = lambda: symmetry.apply_symmetry_maps(field, permutations, phases)
    elif kind == "vector":
        field = _field(rng, (3, NGM))
        average = lambda: symmetry.symmetrize_vector_density(
            field, permutations, phases, rotations)
    elif kind == "spin":
        field = _field(rng, (3, 3, NGM))
        average = lambda: symmetry.symmetrize_spin_vector_density(
            field, permutations, phases, rotations, spin)
    elif kind == "displacement":
        field = _field(rng, (NAT, 3, NGM))
        average = lambda: symmetry.symmetrize_atom_displacement_density(
            field, permutations, phases, rotations, mapping)
    else:
        field = _field(rng, (3, 3, NGM))
        average = lambda: symmetry.symmetrize_tensor_density(
            field, permutations, phases, rotations)
    batched, walked = _both(monkeypatch, average)
    assert batched.shape == walked.shape
    assert np.abs(walked - batched).max() <= 1e-13 * np.abs(batched).max()


def test_the_budget_decides_by_the_batched_working_set():
    assert not symmetry._walks_operations(48, np.zeros((3, 1000)))
    # Eight atoms' displacement responses on 30000 G-vectors under 48 operations:
    # the gathered array is 1.1 GB, which is what the walk exists for.
    assert symmetry._walks_operations(48, np.zeros((8, 3, 30000)))
    # A strain field on eight-atom silicon at 20 Ry: 42.5 MB of gather, under the
    # budget, and four times that once the double rotation is counted.
    strain = np.zeros((3, 3, 12893))
    assert not symmetry._walks_operations(24, strain, copies=1)
    assert symmetry._walks_operations(24, strain, copies=4)


def test_the_budget_has_a_dial(monkeypatch):
    """``DEFUMAT_GATHER_BUDGET`` is read at every decision, which is what lets
    ``monkeypatch.setenv`` reach it; a cache of the budget would have to keep
    that or this test would read the first value only."""
    field = np.zeros((3, 1000))
    monkeypatch.setenv("DEFUMAT_GATHER_BUDGET", "0")
    assert symmetry._walks_operations(48, field)
    monkeypatch.setenv("DEFUMAT_GATHER_BUDGET", "100000")
    assert not symmetry._walks_operations(48, np.zeros((8, 3, 30000)))
