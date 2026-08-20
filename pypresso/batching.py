"""Chunking the k-point axis: QE's ``k_loop``, kept differentiable and jitted.

``PW/src/c_bands.f90`` diagonalises **one k-point at a time** --
``k_loop: DO ik = ik_+1, nks``, with ``get_buffer(evc, nwordwfc, iunwfc, ik)``
reading that k-point's wavefunctions in and ``save_buffer`` writing them back --
and ``sum_band.f90`` accumulates the density inside the same loop. So the
working set QE holds is *one* k-point's, whatever ``nks`` is; the whole set of
``evc`` lives in a buffer that is memory or disk according to ``io_level``. Its
parallelism over k comes from MPI pools, not from batching.

This code batches the k axis with ``vmap`` instead -- k is the leading
independent axis of every wavefunction-shaped array precisely so that it can be
(rule R6) -- and that trades memory for throughput: ``nk`` separate kernel
launches per operation was what made the first Python-loop version slow, and a
GPU wants the batch. The cost is that every k-point's Davidson subspace, and
every k-point's band-by-band real-space field, are live at once. On a cell with
many irreducible k-points and two-component spinors that is tens of gigabytes
where QE needs one k-point's worth.

This module is the dial between the two, and the default is QE's end of it:

* ``batch = 1`` -- QE's loop, one k-point resident.
* ``batch = n`` -- n k-points at a time, the compromise.
* ``batch = None`` -- one ``vmap`` over every k-point, the largest working set
  and the fastest per iteration.

Two properties matter for how it is written. The loop is a ``lax.scan`` (or
``lax.map``, which is a scan underneath) rather than a Python ``for``: a Python
loop over k unrolls ``nk`` copies of ``h_psi`` into the jaxpr, so compilation
time grows with the k-point count and nothing is saved at runtime. And the
chunk size is not a physical parameter: every k-point is solved by exactly the
same function whatever it is, so the only difference is the order the
contributions are *added* in -- one tree reduction over the whole axis against a
sequential accumulation over chunks. That is a round-off difference and nothing
else (~1e-15 Ry on the silicon reference cell), which is what
``tests/unit/test_batching.py`` pins.
"""

from __future__ import annotations

import os
import warnings

import jax
import jax.numpy as jnp
from jax import lax

__all__ = ["DEFAULT_K_BATCH", "resolve_k_batch", "map_k", "sum_k"]


def _default_from_environment() -> int | None:
    """``PYPRESSO_K_BATCH``: an integer, or ``all``/``0`` for one ``vmap``."""
    setting = os.environ.get("PYPRESSO_K_BATCH", "").strip().lower()
    if not setting:
        return 1
    if setting in ("all", "0", "off", "none"):
        return None
    try:
        value = int(setting)
    except ValueError:
        warnings.warn(f"ignoring PYPRESSO_K_BATCH={setting!r}: not a number",
                      RuntimeWarning, stacklevel=2)
        return 1
    if value < 1:
        return None
    return value


#: How many k-points are processed at once when nothing says otherwise. One, as
#: QE does it. ``PYPRESSO_K_BATCH`` overrides it for a whole process; every
#: entry point takes a ``k_batch`` argument that overrides it for one call.
DEFAULT_K_BATCH = _default_from_environment()


def resolve_k_batch(requested: int | None | str = "default") -> int | None:
    """Turn what a caller passed into a chunk size.

    The sentinel is the string ``"default"`` rather than ``None``, because
    ``None`` is a meaningful value here -- it asks for every k-point at once --
    and a caller that has one must be able to pass it through.
    """
    if isinstance(requested, str):
        if requested == "default":
            return DEFAULT_K_BATCH
        return _named(requested)
    if requested is None:
        return None
    value = int(requested)
    if value < 1:
        raise ValueError(f"k_batch must be a positive integer or None, got {requested!r}")
    return value


def _named(setting: str) -> int | None:
    if setting.strip().lower() in ("all", "0", "off", "none"):
        return None
    return resolve_k_batch(int(setting))


def _leading(xs) -> int:
    leaves = jax.tree_util.tree_leaves(xs)
    if not leaves:
        raise ValueError("nothing to map over: the k-axis pytree is empty")
    return int(leaves[0].shape[0])


def map_k(fn, xs, *, batch: int | None):
    """``fn`` at every k-point, results stacked on a leading k axis.

    ``xs`` is a pytree whose leaves all have ``nk`` as their leading axis, and
    ``fn`` takes one k-point's slice of it. With ``batch=None`` this is exactly
    ``jax.vmap(fn)(xs)``; otherwise ``lax.map`` runs it ``batch`` k-points at a
    time, which handles a remainder itself.
    """
    nk = _leading(xs)
    if batch is None or batch >= nk:
        return jax.vmap(fn)(xs)
    return lax.map(fn, xs, batch_size=batch)


def sum_k(fn, xs, *, batch: int | None):
    """``sum_k fn(k)`` -- the same chunking, accumulated instead of stacked.

    This is the shape ``sum_band`` and ``sum_bec`` need: the per-k contribution
    is a whole density or a whole ``becsum``, so stacking ``nk`` of them and
    summing afterwards would defeat the point of chunking at all.
    """
    nk = _leading(xs)
    if batch is None or batch >= nk:
        return _chunk_sum(fn, xs)

    full = nk // batch
    head = jax.tree_util.tree_map(
        lambda a: a[: full * batch].reshape((full, batch) + a.shape[1:]), xs
    )
    template = jax.eval_shape(
        lambda chunk: _chunk_sum(fn, chunk),
        jax.tree_util.tree_map(lambda a: a[0], head),
    )
    zero = jax.tree_util.tree_map(lambda s: jnp.zeros(s.shape, s.dtype), template)

    def step(carry, chunk):
        return jax.tree_util.tree_map(jnp.add, carry, _chunk_sum(fn, chunk)), None

    total, _ = lax.scan(step, zero, head)

    remainder = nk - full * batch
    if remainder:
        tail = jax.tree_util.tree_map(lambda a: a[full * batch :], xs)
        total = jax.tree_util.tree_map(jnp.add, total, _chunk_sum(fn, tail))
    return total


def _chunk_sum(fn, chunk):
    """One chunk's contributions, summed over the k-points in it."""
    return jax.tree_util.tree_map(
        lambda a: jnp.sum(a, axis=0), jax.vmap(fn)(chunk)
    )
