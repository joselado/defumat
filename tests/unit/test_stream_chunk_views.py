"""A streamed store's chunk crosses to the device as a view of its rows, not a copy.

``OPEN.md`` Part XXIII item 25, the bit-identical half. Every chunk
``k_chunks`` yields is a run of rows except a padded last one, so the walks take
it as a slice of the store and hand that view to ``device_put``, where an
integer index array made NumPy copy the chunk first; the padded chunk repeats a
row and keeps the index. What is checked here is what ``device_put`` is handed
(a view of the store, or a fresh array) and that the values and the write-back
are the same; that no number moves is the SCF, force and stress run against the
old code bit for bit, which is not repeated here.

The last test pins the behaviour the views rest on, on a CPU backend: there
``device_put`` of an aligned contiguous array is zero-copy, so a chunk can be
the store's own memory, and a solve donates its starting block. PjRt must copy
such a buffer rather than write the solve's output through it, or a retry would
start from the new states instead of the old ones.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.forces.chunked import _rows_of
from defumat.scf.streaming import stream_diagonalize

pytestmark = pytest.mark.unit


def _store(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


@pytest.fixture
def handed(monkeypatch):
    """Every array ``jax.device_put`` is given, in order."""
    seen = []
    real = jax.device_put

    def spy(value, *args, **kwargs):
        seen.append(value)
        return real(value, *args, **kwargs)

    monkeypatch.setattr(jax, "device_put", spy)
    return seen


def test_a_run_of_rows_crosses_as_a_view_and_a_padded_chunk_through_an_index(handed):
    store = _store((1, 6, 3, 4))
    run = _rows_of(store, np.array([1, 2, 3]))
    padded = _rows_of(store, np.array([4, 5, 4]))

    assert len(handed) == 2
    assert np.shares_memory(handed[0], store), "a run of rows was copied on the host"
    assert not np.shares_memory(handed[1], store)
    assert np.array_equal(np.asarray(run), store[:, 1:4])
    assert np.array_equal(np.asarray(padded), store[:, [4, 5, 4]])


def test_two_channels_cross_one_at_a_time_and_are_stacked_on_the_device(handed):
    """At ``nspin = 2`` the pair of channels is strided; each channel's rows are not."""
    store = _store((2, 6, 3, 4))
    chunk = _rows_of(store, np.array([2, 3, 4]))

    assert len(handed) == 2
    assert all(np.shares_memory(h, store) and h.flags.c_contiguous for h in handed)
    assert chunk.shape == (2, 3, 3, 4)
    assert np.array_equal(np.asarray(chunk), store[:, 2:5])


def test_a_band_slice_of_the_store_crosses_in_one_packing_copy(handed):
    """A store that is itself strided gains nothing from the channel split."""
    store = _store((2, 6, 5, 4))[:, :, :3]
    chunk = _rows_of(store, np.array([1, 2]))

    assert len(handed) == 1 and handed[0].flags.c_contiguous
    assert np.array_equal(np.asarray(chunk), store[:, 1:3])


def test_the_solve_walk_reads_views_and_writes_every_row_back(handed):
    """Five k-points at two a chunk: two runs and a padded chunk, per channel."""
    store = _store((2, 5, 3, 4))
    original = store.copy()
    retried = []

    def eigensolver(hamiltonian, nbnd, psi0, threshold, *, indices, psi0_again, **_):
        # The store still holds the starting block while the solve runs, which
        # is what a robust retry is handed.
        again = np.asarray(psi0_again())
        retried.append(np.array_equal(again, np.asarray(psi0)))
        rows = np.asarray(indices)
        energies = jnp.asarray(np.tile(rows[:, None], (1, nbnd)).astype(float))
        return (energies, 2.0 * psi0 + hamiltonian, jnp.zeros(len(rows), int),
                jnp.zeros(len(rows), int))

    calculation = SimpleNamespace(
        david=None, k_batch=2, eigensolver=eigensolver,
        system=SimpleNamespace(cell=SimpleNamespace(
            precision=SimpleNamespace(real=np.float64))))
    eigenvalues, steps, unsettled = stream_diagonalize(
        calculation, [0.0, 1.0], 3, store)

    assert all(retried)
    expected = 2.0 * original + np.array([0.0, 1.0])[:, None, None, None]
    assert np.array_equal(store, expected)
    assert np.array_equal(eigenvalues[0, :, 0], np.arange(5.0))
    # Per channel: the two runs cross as views, the padded chunk is copied, and
    # each psi0_again reads the same way.
    views = [np.shares_memory(h, store) for h in handed]
    assert len(handed) == 12 and sum(views) == 8, views


def _aligned(shape, dtype=np.complex128, align=64):
    dtype = np.dtype(dtype)
    size = int(np.prod(shape)) * dtype.itemsize
    raw = np.empty(size + align, np.uint8)
    offset = (-raw.ctypes.data) % align
    return raw[offset:offset + size].view(dtype).reshape(shape)


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="host and device memory are one only on a CPU")
def test_a_donated_view_of_the_store_is_copied_not_written_through():
    store = _aligned((2, 8, 4, 16))
    store[...] = 1.0
    view = store[0, 2:5]
    assert view.ctypes.data % 64 == 0
    block = jax.device_put(view)
    assert block.unsafe_buffer_pointer() == view.ctypes.data, (
        "device_put no longer aliases an aligned view on a CPU; the note in "
        "defumat.scf.streaming._to_device describes a behaviour that is gone")

    solve = jax.jit(lambda psi: 2.0 * psi, donate_argnums=0)
    out = solve(block)
    out.block_until_ready()

    assert np.all(store == 1.0), "the donated solve wrote through into the store"
    pointer = out.unsafe_buffer_pointer()
    assert not store.ctypes.data <= pointer < store.ctypes.data + store.nbytes
    assert np.all(np.asarray(out) == 2.0)
