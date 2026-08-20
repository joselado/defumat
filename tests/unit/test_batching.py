"""The k-point chunking, which must be invisible in every result.

``pypresso.batching`` is the dial between QE's ``k_loop`` -- one k-point
resident, ``c_bands.f90`` -- and a single ``vmap`` over the whole k axis. It
exists for memory, so the one thing it must never buy is a different answer.
The chunk size is not a physical parameter: the same per-k function runs
whatever it is, and only the order the contributions are *added* in changes, so
the two ends of the dial agree to round-off.

The checks here are on the primitives and on the pieces the SCF is assembled
from. ``tests/regression/test_batching_scf.py`` closes the loop on a whole
self-consistent run.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from pypresso.batching import DEFAULT_K_BATCH, map_k, resolve_k_batch, sum_k

pytestmark = pytest.mark.unit

#: Chunk sizes exercised against ``nk = 7``: divides exactly, leaves a
#: remainder, exceeds the axis, and the two ends of the dial.
CHUNKS = [1, 2, 3, 7, 9, None]


def test_default_is_qes_loop():
    """One k-point at a time unless something says otherwise."""
    assert DEFAULT_K_BATCH == 1 or DEFAULT_K_BATCH is None  # env may override
    assert resolve_k_batch(1) == 1
    assert resolve_k_batch(4) == 4
    assert resolve_k_batch(None) is None
    assert resolve_k_batch("all") is None
    assert resolve_k_batch("default") == DEFAULT_K_BATCH


def test_a_chunk_size_must_be_positive():
    with pytest.raises(ValueError, match="k_batch"):
        resolve_k_batch(-2)


@pytest.mark.parametrize("batch", CHUNKS)
def test_map_k_matches_vmap(batch):
    """Stacked results, whatever the chunking -- including a ragged one."""
    key = jax.random.PRNGKey(0)
    xs = jax.random.normal(key, (7, 5))

    def fn(row):
        return jnp.sin(row) @ jnp.cos(row), jnp.cumsum(row)

    want = jax.vmap(fn)(xs)
    got = map_k(fn, xs, batch=batch)
    for a, b in zip(got, want):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-14)


@pytest.mark.parametrize("batch", CHUNKS)
def test_sum_k_matches_a_full_reduction(batch):
    """Accumulation, whatever the chunking, on a pytree with a ``None`` in it."""
    key = jax.random.PRNGKey(1)
    xs = (jax.random.normal(key, (7, 4)), jax.random.normal(key, (7,)))

    def fn(arrays):
        vector, scalar = arrays
        return {"field": scalar * jnp.abs(vector) ** 2, "trace": jnp.sum(vector), "none": None}

    want = jax.tree_util.tree_map(lambda a: jnp.sum(a, axis=0), jax.vmap(fn)(xs))
    got = sum_k(fn, xs, batch=batch)
    np.testing.assert_allclose(got["field"], want["field"], rtol=0, atol=1e-13)
    np.testing.assert_allclose(got["trace"], want["trace"], rtol=0, atol=1e-13)
    assert got["none"] is None


@pytest.mark.parametrize("batch", CHUNKS)
def test_chunking_stays_differentiable(batch):
    """The dial must not cost the gradient, which is why JAX is here at all."""

    def total(scale):
        return sum_k(lambda x: jnp.sum(jnp.sin(scale * x)), jnp.arange(7.0), batch=batch)

    value, gradient = jax.value_and_grad(total)(0.3)
    expected = float(np.sum(np.cos(0.3 * np.arange(7.0)) * np.arange(7.0)))
    assert float(value) == pytest.approx(float(np.sum(np.sin(0.3 * np.arange(7.0)))))
    assert float(gradient) == pytest.approx(expected)


@pytest.mark.parametrize("batch", [1, 3, None])
def test_sum_band_is_the_same_density(batch):
    """``sum_band``'s accumulation, which is where the second-largest working set is."""
    from pypresso.scf.density import sum_band
    from pypresso.system.cell import Cell

    grid = (6, 6, 6)
    nk, nbnd, npwx = 5, 3, 11
    key = jax.random.PRNGKey(2)
    keys = jax.random.split(key, 3)
    psi = (jax.random.normal(keys[0], (1, nk, nbnd, npwx))
           + 1j * jax.random.normal(keys[1], (1, nk, nbnd, npwx)))
    fft_index = jnp.asarray(
        np.stack([np.arange(npwx) + 3 * k for k in range(nk)]) % int(np.prod(grid))
    )
    weights = jnp.abs(jax.random.normal(keys[2], (1, nk, nbnd)))
    cell = Cell(at=jnp.eye(3), alat=10.0)

    reference = sum_band(psi, fft_index, grid, weights, cell, None)
    got = sum_band(psi, fft_index, grid, weights, cell, batch)
    np.testing.assert_allclose(got, reference, rtol=0, atol=1e-12)
