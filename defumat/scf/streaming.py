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
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from defumat.batching import k_chunks

__all__ = ["stream_start", "stream_diagonalize", "stream_densities"]


def _to_device(array):
    """One chunk across: a numpy slice made contiguous, then ``device_put``."""
    return jax.device_put(np.ascontiguousarray(array))


def stream_start(calculation, hamiltonians, nbnd: int, span=None) -> np.ndarray:
    """``(nspin, nk, nbnd, ndim)`` starting states, built chunk by chunk into host memory."""
    nk = hamiltonians[0].nk
    store = None
    for rows, live in k_chunks(nk, calculation.k_batch):
        chunk = np.asarray(calculation.starting_wavefunctions(
            hamiltonians, nbnd, span=span, rows=rows))
        if store is None:
            store = np.empty((chunk.shape[0], nk) + chunk.shape[2:], chunk.dtype)
        store[:, rows[:live]] = chunk[:, :live]
    return store


def stream_diagonalize(calculation, hamiltonians, nbnd: int, store: np.ndarray,
                       ethr=None):
    """The Davidson solve of every k-point of every channel, one chunk at a time.

    ``store`` is the previous states and is **overwritten in place** with the
    new ones. Returns ``(eigenvalues, steps, unsettled)`` shaped as
    :meth:`~defumat.scf.driver.Calculation.diagonalize` returns them with
    ``return_steps``, as numpy arrays. ``ethr`` is a scalar or the
    ``(nspin, nk, nbnd)`` per-band array, as there.

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
    nspin, nk = len(hamiltonians), hamiltonians[0].nk
    eigenvalues = steps = unsettled = None
    for spin, hamiltonian in enumerate(hamiltonians):
        threshold = ethr[spin] if rank == 3 else ethr
        for rows, live in k_chunks(nk, calculation.k_batch):
            energies, states, taken, stuck = calculation.eigensolver(
                hamiltonian, nbnd, _to_device(store[spin, rows]), threshold,
                k_batch=calculation.k_batch, return_steps=True,
                indices=jnp.asarray(rows), **extra,
            )
            energies = np.asarray(energies)
            if eigenvalues is None:
                eigenvalues = np.empty((nspin, nk, nbnd), energies.dtype)
                steps = np.empty((nspin, nk), np.asarray(taken).dtype)
                unsettled = np.empty((nspin, nk), np.asarray(stuck).dtype)
            live_rows = rows[:live]
            store[spin, live_rows] = np.asarray(states)[:live]
            eigenvalues[spin, live_rows] = energies[:live]
            steps[spin, live_rows] = np.asarray(taken)[:live]
            unsettled[spin, live_rows] = np.asarray(stuck)[:live]
    return eigenvalues, steps, unsettled


def _add(total, part):
    return part if total is None else jax.tree_util.tree_map(jnp.add, total, part)


def stream_densities(calculation, store: np.ndarray, weights, *,
                     kinetic: bool = False, hubbard: bool = False):
    """``(becsum, rho, tau, ns)`` from the streamed store, each finished once.

    ``tau`` and ``ns`` are ``None`` unless asked for. ``becsum`` is ``()`` on a
    norm-conserving run, as :meth:`~defumat.scf.driver.Calculation.becsum`
    returns it. The values are those of the whole-set calls to round-off: the
    chunks change only the order the k contributions are added in.
    """
    weights = np.asarray(weights)
    nk = store.shape[1]
    becsum_ = rho = tau = ns = None
    for rows, live in k_chunks(nk, calculation.k_batch):
        w = weights[:, rows].copy()
        w[:, live:] = 0.0
        w = jnp.asarray(w)
        psi = _to_device(store[:, rows])
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
    becsum_ = calculation.finish_becsum(becsum_) if becsum_ else becsum_
    rho = calculation.finish_density(rho, becsum_)
    if kinetic:
        tau = calculation.finish_kinetic_energy_density(tau)
    if hubbard:
        ns = calculation.finish_occupation_matrix(ns)
    return becsum_, rho, tau, ns
