"""The wavefunction store streamed through the device one k-chunk at a time.

``wfc_store = 'stream'`` (:mod:`defumat.batching`): the store is a numpy array
in host memory for the whole SCF, and each pass that reads it walks
:func:`~defumat.batching.k_chunks` in a Python loop -- one chunk to the device,
the same compiled kernel the whole-set path runs, the result back. This is
``c_bands.f90``'s ``k_loop`` with ``get_buffer``/``save_buffer`` around each
k-point and ``io_level`` pointing at host RAM, and ``sum_band.f90``'s second
walk over the buffer for the density.

Three passes read the store, and they are the three functions here:

* :func:`stream_start` -- ``wfcinit``: the atomic orbitals and their
  Rayleigh-Ritz, per chunk, so the start is never whole on the device either.
* :func:`stream_diagonalize` -- the Davidson calls, per chunk, each chunk's
  states written back into the store in place.
* :func:`stream_densities` -- everything built from the states and the
  occupations in one walk: ``becsum``, the smooth-grid density, and when the
  run needs them ``tau`` and DFT+U's ``ns``. Each is a sum over k, accumulated
  raw and finished once -- ``becsum`` and ``ns`` symmetrised, the density lifted,
  augmented and symmetrised, ``tau`` lifted and symmetrised -- which is exact
  because every finishing step is linear.

**Why a Python loop and not a** ``lax.scan``: the store must never reach a
kernel whole, and inside a scan it would be an operand of the loop. On jax
0.11.1 the host-offloading route that would keep it in ``pinned_host`` inside a
scan fails at the gather (``batching``'s module docstring has the two
measurements). A loop of calls to one compiled kernel pays one dispatch and one
synchronisation per chunk, which is what QE's loop pays too.

**What the device holds.** One chunk's states in and out, its Davidson
subspace, its projectors when they are rebuilt, plus the accumulators -- one
density on the smooth grid and one ``becsum`` -- and the resident basis
bookkeeping (``|k+G|^2``, the FFT index and mask, the projector core), which is
the part that still grows with the mesh.

The weights of a padded chunk's repeated rows are **zero**, so the padding
contributes nothing to a sum; a padded solve is computed and discarded.

**How a chunk crosses** (:func:`rows_to_device`, ``OPEN.md`` Part XXIII item
25). Every chunk :func:`~defumat.batching.k_chunks` yields is a run of
consecutive rows except a padded last one, so every other chunk is taken as a
slice of the store, a view, and handed to ``device_put`` as it is, where an
integer index array made NumPy copy the chunk first. The padded chunk repeats a
row, which no view expresses, and keeps the index. A solve reads one channel at
a time, which is always contiguous; the density pass reads both, and at
``nspin = 2`` with fewer rows than the store the pair is strided, so each
channel's rows cross on their own and are stacked on the device. That stack is
a device copy, and for its duration the two halves and the stacked block are
all live, one more two-channel block than a single transfer holds; on a CPU
backend the stack is a copy in host memory, so there the two-channel read costs
what the index did. The new states go back by slice assignment, since the rows
a chunk writes (``positions[:live]``) are always a run; an indexed assignment
made no extra copy either, so the gain is on the reads.

**A k-point pool walks only its own rows** (:mod:`defumat.parallel`). Its
store is a :class:`~defumat.parallel.PoolStore`, whose ``rows`` name the global
k index of each of its rows: :func:`stream_start` builds one when handed
``rows``, and :func:`stream_diagonalize` and :func:`stream_densities` walk its
positions and address every per-k table, weight and threshold by the global
row. The eigenvalues come back for the pool's rows only, and the raw sums over
k are handed to ``reduce`` before they are finished, which is where the pools
add their shares; a single pool passes the identity, and a plain ``ndarray``
store is the whole set with ``rows = arange(nk)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from defumat.batching import k_chunks
from defumat.parallel import PoolStore

__all__ = ["stream_start", "stream_diagonalize", "stream_densities",
           "stream_eigenvalues", "stream_states", "stream_becsum",
           "is_host_store", "rows_to_device"]


def _to_device(array):
    """One chunk across: a numpy array made contiguous, then ``device_put``.

    ``np.ascontiguousarray`` copies only an array that is not contiguous
    already; a run of rows of one channel, or of every channel at
    ``nspin = 1``, is a contiguous view of the store and crosses as it is.

    **On a CPU backend the device array can be the store itself.** jax 0.11.0's
    ``device_put`` of a contiguous numpy array whose data is 64-byte aligned
    is zero-copy there, whatever ``may_alias`` says (``False`` included,
    measured 2026-10-04), so a chunk taken as a view aliases the store's rows
    whenever their address happens to be aligned; before, it aliased the
    index's private copy on the same condition. That is safe for three
    reasons, and they are the rule any walk that reads a host store this way
    (here, :func:`~defumat.forces.chunked._rows_of`, and through it the chunked
    responses) has to keep. A store's rows are written only with the output of
    the computation that read them, after ``np.asarray`` of that output, so
    the read has finished: the solve's write-back follows its own eigenvalues'
    fetch, every response pass writes its ``dpsi`` the same way, and a density
    pass, which writes nothing, feeds the Hamiltonian that the next solve, the
    next writer, needs first. A solve's starting block is donated, and PjRt
    **copies** a donated zero-copy buffer rather than writing through it, so
    the store still holds the block a retry starts from
    (``tests/unit/test_stream_chunk_views.py`` pins it). And nothing keeps a
    chunk's device array past its walk. A walk that wrote rows while a
    computation reading them was still in flight would race on a CPU, where
    the index's private copy never could.
    """
    return jax.device_put(np.ascontiguousarray(array))


def _run(positions):
    """``slice(a, b)`` when ``positions`` is the run ``a, a + 1, ..., b - 1``, else ``None``."""
    positions = np.asarray(positions)
    if positions.ndim != 1 or positions.size == 0:
        return None
    start = int(positions[0])
    if np.array_equal(positions, np.arange(start, start + positions.size)):
        return slice(start, start + positions.size)
    return None


def _rows(positions):
    """``positions`` as a slice where it is a run, for an index that is a view."""
    run = _run(positions)
    return positions if run is None else run


def rows_to_device(array: np.ndarray, positions, spin: int | None = None):
    """One chunk of a host store on the device: ``array[spin, positions]``, or ``array[:, positions]``.

    The rows cross as a view of the store wherever they are a run (module
    docstring, "How a chunk crosses"), and through an index, which copies, only
    for a padded chunk. The values are the same bytes either way.
    """
    run = _run(positions)
    if run is None:
        return _to_device(array[:, positions] if spin is None
                          else array[spin, positions])
    if spin is not None:
        return _to_device(array[spin, run])
    block = array[:, run]
    if block.flags.c_contiguous or not array[0, run].flags.c_contiguous:
        # Contiguous as it stands; or a store that is itself strided (a band
        # slice, as the response stores are read), whose channels are no more
        # contiguous than the pair, so one packing copy is the least there is.
        return _to_device(block)
    return jnp.stack([_to_device(array[channel, run])
                      for channel in range(array.shape[0])])


def is_host_store(wavefunctions) -> bool:
    """Whether ``wavefunctions`` is a streamed store: a ``(nspin, nk, nbnd, ndim)`` numpy array.

    What a streamed SCF hands back as ``SCFResult.wavefunctions``, and what a
    streamed fixed-density solve that keeps its states returns. A consumer that
    reads the states whole should walk it with :func:`~defumat.batching.k_chunks`
    rather than ``jnp.asarray`` it, which puts the whole set back on the device
    (``GPU-MEMORY-NEXT.md`` item 4).
    """
    return isinstance(wavefunctions, np.ndarray) and wavefunctions.ndim == 4


def _unwrap(store):
    """``(array, global rows)`` of a store: a pool's rows, or the whole set."""
    if isinstance(store, PoolStore):
        return store.array, store.rows
    return store, np.arange(store.shape[1])


def _local_chunks(rows, batch):
    """``(positions, global rows, live)`` for each chunk of a store's rows."""
    for positions, live in k_chunks(len(rows), batch):
        yield positions, rows[positions], live


def _chunk_weights(weights: np.ndarray, rows, live: int):
    """One chunk's weights, with a padded chunk's repeated rows weighted zero."""
    w = weights[:, rows].copy()
    w[:, live:] = 0.0
    return jnp.asarray(w)


def stream_start(calculation, hamiltonians, nbnd: int, span=None, rows=None):
    """``(nspin, nk, nbnd, ndim)`` starting states, built chunk by chunk into host memory.

    ``rows`` builds a k-point pool's share only, and returns it as a
    :class:`~defumat.parallel.PoolStore`; every piece of the start is per k-point
    (``Calculation.starting_wavefunctions``), so the shares are the whole set's
    rows.
    """
    nk = hamiltonians[0].nk
    pooled = rows is not None
    rows = np.arange(nk) if rows is None else np.asarray(rows)
    store = None
    for positions, chunk_rows, live in _local_chunks(rows, calculation.k_batch):
        chunk = np.asarray(calculation.starting_wavefunctions(
            hamiltonians, nbnd, span=span, rows=chunk_rows))
        if store is None:
            store = np.empty((chunk.shape[0], len(rows)) + chunk.shape[2:], chunk.dtype)
        store[:, _rows(positions[:live])] = chunk[:, :live]
    return PoolStore(store, rows, nk) if pooled else store


def stream_diagonalize(calculation, hamiltonians, nbnd: int, store: np.ndarray,
                       ethr=None):
    """The Davidson solve of every k-point of every channel, one chunk at a time.

    ``store`` is the previous states and is **overwritten in place** with the
    new ones. Returns ``(eigenvalues, steps, unsettled)`` shaped as
    :meth:`~defumat.scf.driver.Calculation.diagonalize` returns them with
    ``return_steps``, as numpy arrays. ``ethr`` is a scalar or the
    ``(nspin, nk, nbnd)`` per-band array, as there. A
    :class:`~defumat.parallel.PoolStore` is solved at its own rows only, and the
    three arrays then carry those rows, ``(nspin, len(store.rows), ...)``.

    **Chunks are settled one at a time, and launching the next one early was
    measured not to help.** On a GTX 1060 (jax 0.11.1) the call that launches a
    chunk's solve *returns only when the solve is done*: XLA runs Davidson's
    ``while_loop`` by reading its predicate back to the host each step, so the
    execution happens on the calling thread. Timed on eight-atom silicon at 64
    k-points, the launch was 8.04 s of a 9.2 s streamed solve (512 calls,
    15.7 ms each) and waiting on its result afterwards 0.26 s; the transfers
    both ways were 0.23 s. A look-ahead that launched chunk ``i + 1`` before
    collecting chunk ``i`` measured 9.34 s against 9.22 s -- nothing to overlap
    -- while holding a second chunk on the device, so it is not here. What does
    buy time is a larger chunk (``k_batch``), which amortises the loop over a
    batch: 12.1 s at one k-point, 9.65 s at four, 7.66 s for the whole axis
    (``speed``), at 58, 141 and 2174 MB.
    """
    rank = 0 if ethr is None else jnp.ndim(ethr)
    if rank not in (0, 3):
        raise ValueError(
            f"ethr must be a scalar or a (nspin, nk, nbnd) array, got rank {rank}")
    extra = {} if calculation.david is None else {"david": calculation.david}
    array, rows = _unwrap(store)
    nspin, nlocal = len(hamiltonians), len(rows)
    eigenvalues = steps = unsettled = None
    for spin, hamiltonian in enumerate(hamiltonians):
        # A per-band threshold stays the whole set's: the solve picks its rows
        # through ``indices``, which are global.
        threshold = ethr[spin] if rank == 3 else ethr
        for positions, chunk_rows, live in _local_chunks(rows, calculation.k_batch):
            # The chunk's starting block is donated to the solve, which writes
            # its states into the same buffer; the host store still holds the
            # block, which is what a robust retry is handed instead
            # (``GPU-MEMORY-NEXT.md`` item 11).
            energies, states, taken, stuck = calculation.eigensolver(
                hamiltonian, nbnd, rows_to_device(array, positions, spin), threshold,
                k_batch=calculation.k_batch, return_steps=True,
                indices=jnp.asarray(chunk_rows),
                psi0_again=lambda spin=spin, positions=positions: rows_to_device(
                    array, positions, spin),
                **extra,
            )
            energies = np.asarray(energies)
            if eigenvalues is None:
                eigenvalues = np.empty((nspin, nlocal, nbnd), np.promote_types(
                    energies.dtype, calculation.system.cell.precision.real))
                steps = np.empty((nspin, nlocal), np.asarray(taken).dtype)
                unsettled = np.empty((nspin, nlocal), np.asarray(stuck).dtype)
            # A run in every chunk, the padded last one included: the padding
            # follows the live rows.
            written = _rows(positions[:live])
            array[spin, written] = np.asarray(states)[:live]
            eigenvalues[spin, written] = energies[:live]
            steps[spin, written] = np.asarray(taken)[:live]
            unsettled[spin, written] = np.asarray(stuck)[:live]
    return eigenvalues, steps, unsettled


def stream_eigenvalues(calculation, hamiltonians, nbnd: int, ethr):
    """A solve from scratch at every k-point, keeping the eigenvalues only.

    QE's ``c_bands_nscf``: a band structure, an NSCF grid or a density of
    states wants the energies, and each k-point's states are discarded as soon
    as its solve returns -- so the device holds one chunk's solve and never
    the ``(nspin, nk, nbnd, ndim)`` set the whole-set call stacks. Every chunk
    starts from Davidson's own random vectors (``psi0 = None``), which are
    drawn per k-point, so the chunks together are the whole-set solve.

    Returns ``(eigenvalues, steps, unsettled)`` as numpy arrays shaped as
    :meth:`~defumat.scf.driver.Calculation.diagonalize` returns them with
    ``return_steps``.
    """
    eigenvalues, _, steps, unsettled = _stream_from_scratch(
        calculation, hamiltonians, nbnd, ethr, keep=False)
    return eigenvalues, steps, unsettled


def stream_states(calculation, hamiltonians, nbnd: int, ethr):
    """:func:`stream_eigenvalues` for a caller that keeps the states, in host memory.

    ``c_bands_nscf`` with ``save_buffer`` after each k-point: the solve is the
    same chunk walk, and each chunk's states are written into a numpy store
    rather than dropped, so a projected density of states or an STM image on a
    long mesh holds one chunk on the device and the set in host RAM -- the
    store a streamed SCF keeps (:func:`is_host_store`).

    Returns ``(eigenvalues, store, steps, unsettled)``, the order
    :meth:`~defumat.scf.driver.Calculation.diagonalize` uses.
    """
    return _stream_from_scratch(calculation, hamiltonians, nbnd, ethr, keep=True)


def _stream_from_scratch(calculation, hamiltonians, nbnd, ethr, *, keep):
    extra = {} if calculation.david is None else {"david": calculation.david}
    nspin, nk = len(hamiltonians), hamiltonians[0].nk
    eigenvalues = store = steps = unsettled = None
    for spin, hamiltonian in enumerate(hamiltonians):
        for rows, live in k_chunks(nk, calculation.k_batch):
            energies, states, taken, stuck = calculation.eigensolver(
                hamiltonian, nbnd, None, ethr, k_batch=calculation.k_batch,
                return_steps=True, indices=jnp.asarray(rows), **extra,
            )
            energies = np.asarray(energies)
            if eigenvalues is None:
                eigenvalues = np.empty((nspin, nk, nbnd), energies.dtype)
                steps = np.empty((nspin, nk), np.asarray(taken).dtype)
                unsettled = np.empty((nspin, nk), np.asarray(stuck).dtype)
            live_rows = _rows(rows[:live])
            if keep:
                states = np.asarray(states)
                if store is None:
                    store = np.empty((nspin, nk) + states.shape[1:], states.dtype)
                store[spin, live_rows] = states[:live]
            del states
            eigenvalues[spin, live_rows] = energies[:live]
            steps[spin, live_rows] = np.asarray(taken)[:live]
            unsettled[spin, live_rows] = np.asarray(stuck)[:live]
    return eigenvalues, store, steps, unsettled


def _add(total, part):
    return part if total is None else jax.tree_util.tree_map(jnp.add, total, part)


def stream_becsum(calculation, store: np.ndarray, weights) -> tuple:
    """``becsum`` from the streamed store, symmetrised once: ``()`` on a norm-conserving run."""
    if not calculation.is_ultrasoft:
        return ()
    weights = np.asarray(weights)
    total = None
    for rows, live in k_chunks(store.shape[1], calculation.k_batch):
        total = _add(total, calculation.becsum(
            rows_to_device(store, rows), _chunk_weights(weights, rows, live),
            rows=rows, symmetrize=False))
    return calculation.finish_becsum(total)


def stream_densities(calculation, store, weights, *,
                     kinetic: bool = False, hubbard: bool = False,
                     becsum_=None, reduce=None, ldos_weights=None):
    """``(becsum, rho, tau, ns)`` from the streamed store, each finished once.

    ``tau`` and ``ns`` are ``None`` unless asked for. ``becsum`` is ``()`` on a
    norm-conserving run, as :meth:`~defumat.scf.driver.Calculation.becsum`
    returns it; a caller that already has it passes it as ``becsum_``, and it
    is then used for the augmentation charge and handed back rather than
    accumulated again. The values are those of the whole-set calls to
    round-off: the chunks change only the order the k contributions are added
    in.

    With ``ldos_weights`` (:meth:`~defumat.scf.driver.Calculation.ldos_weights`)
    the LDOS at the Fermi level comes back as a fifth member, the charge channel
    on the dense grid, from the same walk of the store: each chunk's states are
    read once and the two weight sets go through them under ``jax.vmap``, as
    :meth:`~defumat.scf.driver.Calculation.density_and_ldos` does for the whole
    set, and the pools' ``reduce`` carries the two extra sums with the rest.
    """
    weights = np.asarray(weights)
    with_ldos = ldos_weights is not None
    if with_ldos:
        ldos_weights = np.asarray(ldos_weights)
    array, store_rows = _unwrap(store)
    accumulate = becsum_ is None
    rho = tau = ns = ldos = ldos_becsum = None
    for positions, rows, live in _local_chunks(store_rows, calculation.k_batch):
        w = _chunk_weights(weights, rows, live)
        psi = rows_to_device(array, positions)
        if with_ldos:
            stacked = jnp.stack([w, _chunk_weights(ldos_weights, rows, live)])
            smooth = jax.vmap(
                lambda ws: calculation.smooth_density(psi, ws, rows=rows))(stacked)
            rho, ldos = _add(rho, smooth[0]), _add(ldos, smooth[1])
            if calculation.is_ultrasoft:
                raw = jax.vmap(lambda ws: calculation.becsum(
                    psi, ws, rows=rows, symmetrize=False))(stacked)
                if accumulate:
                    becsum_ = _add(becsum_, jax.tree_util.tree_map(lambda a: a[0], raw))
                ldos_becsum = _add(ldos_becsum,
                                   jax.tree_util.tree_map(lambda a: a[1], raw))
        else:
            if accumulate:
                becsum_ = _add(becsum_, calculation.becsum(psi, w, rows=rows,
                                                           symmetrize=False))
            rho = _add(rho, calculation.smooth_density(psi, w, rows=rows))
        if kinetic:
            tau = _add(tau, calculation.kinetic_energy_density(
                psi, w, rows=rows, finish=False))
        if hubbard:
            ns = _add(ns, calculation.occupation_matrix(
                psi, w, rows=rows, symmetrize=False))
        del psi
    if reduce is not None:
        if with_ldos:
            raw_becsum, rho, tau, ns, ldos, ldos_becsum = reduce(
                (becsum_ if accumulate else (), rho, tau, ns, ldos,
                 ldos_becsum if ldos_becsum is not None else ()))
        else:
            raw_becsum, rho, tau, ns = reduce(
                (becsum_ if accumulate else (), rho, tau, ns))
        if accumulate:
            becsum_ = raw_becsum
    if accumulate and becsum_:
        becsum_ = calculation.finish_becsum(becsum_)
    rho = calculation.finish_density(rho, becsum_)
    if kinetic:
        tau = calculation.finish_kinetic_energy_density(tau)
    if hubbard:
        ns = calculation.finish_occupation_matrix(ns)
    if not with_ldos:
        return becsum_, rho, tau, ns
    if ldos_becsum:
        ldos_becsum = calculation.finish_becsum(ldos_becsum)
    ldos = calculation.finish_density(ldos, ldos_becsum if ldos_becsum else ())
    return becsum_, rho, tau, ns, calculation.charge_channel(ldos)
