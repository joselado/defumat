"""The sum-over-states assemblies, a k-chunk at a time.

``OPEN.md`` Part XXIII item 24, second half. The optical conductivity, the
second harmonic, the shift current, the TDDFT ``chi_0`` and the transverse spin
``chi_0`` are each a sum over k of per-k terms built from one k-point's states:
its velocity matrix elements (``dH/dk`` from the velocity operator's ``jvp``,
whose sphere and projectors are that k-point's own), or its pair densities. They
used to take the whole k axis of the store onto the device and contract it in
one program, which put a streamed store (a numpy array in host memory) back on
the card whole, with every k-point's matrix elements beside it.

Here each walks :func:`~defumat.batching.k_chunks` at the calculation's
``k_batch``, the field response's arrangement (:mod:`defumat.response.chunked`):
per chunk, the chunk's rows of the store cross to the device
(:func:`store_rows`), a calculation restricted to those rows is rebuilt inside
the compiled pass from :func:`~defumat.forces.chunked.row_leaves`, so every
k-indexed table the pass reads -- the sphere, ``|k+G|^2``, the FFT indices, the
projectors, ``wfcU`` -- is the chunk's, and the pass returns the chunk's share
of every sum the assembly accumulates. The shares are added in the order the
chunks come, so a result moves from the whole-k one at round-off, through the
order of the k sum and nothing else.

**One program per stage and call, not one per chunk.** A chunk's work is a
short sequence of stages (the velocity matrix elements, then their contraction),
each traced once at the first chunk and kept by its structure
(:func:`~defumat.eager.compiled_function`, the torque's arrangement), and every
chunk is padded to one shape by
:func:`~defumat.batching.k_chunks`, so a later call on another calculation of
the same shapes compiles nothing either. The padded rows of a short last chunk
are a repeat of a real row and carry **zero weight**, which is exact because
every sum here is linear in its per-k weights; each assembly says which weight
that is.

**One chunk in flight.** The accumulator is waited on after every chunk.
Dispatch is asynchronous, so a loop that only enqueued could make the next
chunks' uploads before the work on the previous ones had finished, and hold
several chunks' states at once; the wait is what bounds it at one. Inside a
chunk the velocity operator's directions are walked one at a time
(``sequential=True``), since a pass that writes them out holds their
temporaries side by side (:func:`~defumat.response.velocity.
_one_direction_at_a_time` has the measurement).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from defumat.batching import k_chunks, resolve_k_batch
from defumat.eager import compiled_function
from defumat.forces.chunked import row_leaves, with_rows
from defumat.scf.streaming import rows_to_device

__all__ = ["chunk_size", "store_rows", "padded", "walk", "row_leaves", "with_rows"]


def chunk_size(calculation, k_batch="default") -> int | None:
    """The chunk the walk takes: the calculation's ``k_batch`` unless one is given.

    ``"default"`` is the calculation's own dial, which is what every workflow
    hands the assemblies anyway; an explicit value, or ``None`` for the whole
    axis in one chunk, is resolved as :func:`~defumat.batching.resolve_k_batch`
    resolves it.
    """
    if isinstance(k_batch, str) and k_batch == "default":
        return calculation.k_batch
    return resolve_k_batch(k_batch)


def store_rows(store, rows, spin: int | None = None):
    """One chunk of a ``(nspin, nk, ...)`` store on the device.

    ``store[spin, rows]``, or ``store[:, rows]`` with ``spin = None``. A host
    store (a numpy array, or a band slice of one) crosses as a view of its rows
    wherever they are a run (:func:`~defumat.scf.streaming.rows_to_device`); a
    device array is gathered where it is.
    """
    if isinstance(store, np.ndarray):
        return rows_to_device(store, rows, spin)
    index = jnp.asarray(np.asarray(rows))
    return store[:, index] if spin is None else store[spin, index]


def padded(values, rows, live: int, axis: int = 0) -> np.ndarray:
    """``values[rows]`` along ``axis`` on the host, with the padded rows past ``live`` zeroed.

    The weight a padded chunk's repeated rows are given, so that they add
    nothing to a sum that is linear in it.
    """
    taken = np.take(np.asarray(values), np.asarray(rows), axis=axis)
    index = [slice(None)] * taken.ndim
    index[axis] = slice(live, None)
    taken[tuple(index)] = 0.0
    return taken


def walk(nk: int, batch: int | None, arguments, *stages, shares: int = 1):
    """The sum over the chunks of ``nk`` k-points of the last stages' outputs.

    ``arguments(rows, live)`` builds one chunk's arguments, every array of them
    with the chunk's rows only: one tuple per stage. Stage ``i`` is called with
    the outputs of the stages before it and then its own tuple, and the last
    ``shares`` stages return the chunk's shares -- one pytree, or a tuple of
    ``shares`` of them, a band sum truncated at two places being two stages
    rather than one, for the reason that follows. **Each stage is its own
    program**, traced once at the first chunk and kept: a pass that built the
    matrix elements and contracted them in one program held both stages'
    temporaries side by side, where in sequence the peak is the larger stage's
    (measured compile only on AlAs's second harmonic at a chunk of seven, 49.5
    MB of temporaries in one program against 35.2, 13.0 and 7.8 MB as the
    matrix elements and the two band sums apart). What crosses between them is
    a chunk's matrix elements, ``nbnd^2`` per k-point.
    The sum is accumulated on the device and waited on after each chunk, so one
    chunk's states are in flight at a time.
    """
    runs = [None] * len(stages)
    total = None
    for rows, live in k_chunks(nk, batch):
        outputs = []
        for index, (stage, own) in enumerate(zip(stages, arguments(rows, live))):
            call = tuple(outputs) + tuple(own)
            if runs[index] is None:
                runs[index] = compiled_function(stage, *call)
            outputs.append(runs[index](*call))
        part = outputs[-1] if shares == 1 else tuple(outputs[-shares:])
        del outputs
        total = part if total is None else jax.tree_util.tree_map(jnp.add, total, part)
        jax.block_until_ready(total)
    return total
