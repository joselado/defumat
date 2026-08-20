"""The k-point chunking, which must be invisible in every result.

``pypresso.batching`` is the dial between QE's ``k_loop`` -- one k-point
resident, ``c_bands.f90`` -- and a single ``vmap`` over the whole k axis, and
the same dial on the band axis, between ``vloc_psi_k``'s ``DO ibnd`` and one
transform of the whole block. Both
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

from pypresso.batching import (DEFAULT_BAND_BATCH, DEFAULT_K_BATCH, map_bands,
                               map_k, resolve_k_batch, sum_bands, sum_k)

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


# --- the band axis ------------------------------------------------------------
#
# The same contract as the k axis, and a stricter one in the mapping case: a
# chunk of bands is put through exactly the same transform whatever the chunk,
# with nothing summed across bands, so ``map_bands`` must agree *bit for bit*
# rather than to round-off. Only ``sum_bands`` reorders an addition.


def test_the_band_default_is_qes_loop():
    """One band at a time, as ``vloc_psi_k`` and ``sum_band`` walk them."""
    assert DEFAULT_BAND_BATCH == 1 or DEFAULT_BAND_BATCH is None  # env may override


@pytest.mark.parametrize("batch", CHUNKS)
def test_map_bands_is_exact_whatever_the_chunk(batch):
    """No accumulation, so no round-off either: the answers must be identical."""
    key = jax.random.split(jax.random.PRNGKey(5), 2)
    states = (jax.random.normal(key[0], (7, 4))
              + 1j * jax.random.normal(key[1], (7, 4)))

    def block(x):
        return jnp.fft.fft(x, axis=-1) * 2.0 - 1.0

    reference = map_bands(block, states, batch=None)
    np.testing.assert_array_equal(map_bands(block, states, batch=batch), reference)


@pytest.mark.parametrize("batch", CHUNKS)
def test_map_bands_flattens_leading_axes(batch):
    """A caller with a spin or k index does not have to know how many it has."""
    key = jax.random.split(jax.random.PRNGKey(6), 2)
    states = (jax.random.normal(key[0], (2, 3, 4))
              + 1j * jax.random.normal(key[1], (2, 3, 4)))
    block = lambda x: jnp.fft.fft(x, axis=-1)
    got = map_bands(block, states, batch=batch)
    assert got.shape == states.shape
    np.testing.assert_array_equal(got, map_bands(block, states, batch=None))


def test_map_bands_passes_a_single_state_straight_through():
    """``(ndim,)`` is one band already -- there is no axis to walk."""
    state = jnp.arange(4.0)
    np.testing.assert_array_equal(map_bands(lambda x: x + 1, state), state + 1)


@pytest.mark.parametrize("batch", CHUNKS)
def test_sum_bands_matches_a_full_reduction(batch):
    """Here the chunk does reorder an addition, so round-off is all it may cost."""
    values = jnp.asarray(np.sin(0.3 * np.arange(7.0)))
    total = sum_bands(lambda v: jnp.stack([v, v ** 2]), values, batch=batch)
    expected = np.stack([np.sum(np.sin(0.3 * np.arange(7.0))),
                         np.sum(np.sin(0.3 * np.arange(7.0)) ** 2)])
    np.testing.assert_allclose(total, expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("batch", [1, 3, None])
def test_h_psi_is_the_same_operator_whatever_the_band_chunk(batch):
    """The change this dial was introduced for: ``h_psi``'s local term.

    It is the whole point of the exercise that walking the bands is a
    scheduling decision and nothing else, so the operator it applies has to be
    the same one to the last bit the transforms allow.
    """
    from pypresso.batching import _resolve_band_batch

    key = jax.random.split(jax.random.PRNGKey(7), 2)
    grid = (8, 8, 8)
    npwx, nbnd = 9, 5
    psi = (jax.random.normal(key[0], (nbnd, npwx))
           + 1j * jax.random.normal(key[1], (nbnd, npwx)))
    fft_index = jnp.arange(npwx)
    potential = jnp.asarray(np.cos(np.arange(int(np.prod(grid))), dtype=float)).reshape(grid)

    from pypresso.basis.fft import g_to_r, gather_from_box

    n = int(np.prod(grid))

    def local(states, band_batch):
        def block(x):
            box = jnp.fft.fftn(g_to_r(x, fft_index, grid) * potential, axes=(-3, -2, -1)) / n
            return gather_from_box(box, fft_index)

        return map_bands(block, states, batch=band_batch)

    reference = local(psi, None)
    np.testing.assert_allclose(local(psi, batch), reference, rtol=0, atol=1e-13)
    assert _resolve_band_batch("default") == DEFAULT_BAND_BATCH
