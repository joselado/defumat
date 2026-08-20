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

**A batch of one is not a batch.** The obvious way to write "one k-point at a
time" is a batch axis of width one -- ``jax.vmap`` when ``nk`` is 1, and
``lax.map(..., batch_size=1)``, which is *defined* as a scan over ``vmap(fn)``
of one-element chunks, when it is not. Both are wrong, and by a wide margin:
XLA lowers a width-one batch dimension into batched matrix products, a batched
``eigh`` and scatters rather than into the unbatched kernels, and on the eight-
atom silicon cell at 30 Ry that costs **37% of a whole Davidson solve** (110 ms
unbatched against 150 ms under either form of width-one batching). The FFTs are
not what pays -- ``h_psi`` measures the same either way -- it is the subspace
linear algebra, which is a third of a Davidson step and every part of it a small
dense operation.

So ``batch = 1`` here means *no batch axis at all*: a direct call when there is
one k-point, and a plain ``lax.map`` -- a scan whose body is ``fn`` itself --
when there are several. ``vmap`` is used only where a batch is genuinely asked
for. This costs nothing in generality: the body is compiled once either way, and
the accumulation order is the same sequential one, so the answers are unchanged
to the last digit.

**The band axis is the same story, and it is worth more.** ``vloc_psi_k`` walks
its bands one at a time -- ``DO ibnd = 1, m`` around a single ``invfft`` -- and
so does ``sum_band``. Transforming a whole block instead is the obvious
vectorisation and it is a large loss on any cell big enough to matter, for a
reason that has nothing to do with JAX: **a band's real-space box is the working
set, and a block of them is not**. On the sixteen-atom silicon cell at 30 Ry one
band's box is 1.5 MB and thirty-two of them are 48 MB, so the batched transform
streams the whole array from memory twice per pass while the looped one stays in
cache. Measured on the local term of ``h_psi``, single core, one band at a time
against all of them:

===================  ===========  ============  =======
case                 all bands    one at a time
===================  ===========  ============  =======
``si16-1k-ecut30``      153.9 ms       62.0 ms   2.48x
``si16-1k``              23.5 ms       13.9 ms   1.69x
``si8-1k-ecut30``        23.1 ms       13.9 ms   1.66x
``si8-1k``                4.2 ms        3.7 ms   1.14x
``si-1k``                 0.31 ms       0.34 ms  0.91x
===================  ===========  ============  =======

The gain grows with the box, and the one case that loses is the 180-plane-wave
cell where the whole calculation is fixed overhead. So the band axis is a dial
too, with the same default as the k axis -- QE's loop -- and the same escape
hatch for a GPU, which wants the batch that a cache does not.
"""

from __future__ import annotations

import os
import warnings

import jax
import jax.numpy as jnp
from jax import lax

__all__ = ["DEFAULT_K_BATCH", "resolve_k_batch", "map_k", "sum_k",
           "DEFAULT_BAND_BATCH", "map_bands", "sum_bands"]


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


def _band_default_from_environment() -> int | None:
    """``PYPRESSO_BAND_BATCH``: an integer, or ``all``/``0`` for one block."""
    setting = os.environ.get("PYPRESSO_BAND_BATCH", "").strip().lower()
    if not setting:
        return 1
    if setting in ("all", "0", "off", "none"):
        return None
    try:
        value = int(setting)
    except ValueError:
        warnings.warn(f"ignoring PYPRESSO_BAND_BATCH={setting!r}: not a number",
                      RuntimeWarning, stacklevel=2)
        return 1
    return None if value < 1 else value


#: How many bands are transformed at once. One, as ``vloc_psi_k`` and
#: ``sum_band`` do it -- see the module docstring. ``PYPRESSO_BAND_BATCH``
#: overrides it for a whole process.
DEFAULT_BAND_BATCH = _band_default_from_environment()


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
    ``fn`` takes one k-point's slice of it. ``batch=None`` asks for the whole
    axis at once, which is ``jax.vmap(fn)(xs)``; otherwise the k axis is walked
    ``batch`` at a time. A single k-point is the same computation under every
    setting, and is done without a batch axis whatever was asked for.

    **One k-point at a time means no ``vmap`` at all** -- see the module
    docstring's "A batch of one is not a batch".
    """
    nk = _leading(xs)
    if nk == 1:
        # A single k-point is not a batch. Calling ``fn`` on the squeezed
        # pytree and putting the axis back is the same computation without the
        # width-one batch dimension, which is what costs.
        one = jax.tree_util.tree_map(lambda a: a[0], xs)
        return jax.tree_util.tree_map(lambda a: a[None], fn(one))
    if batch == 1:
        return lax.map(fn, xs)  # a plain scan: one k-point, no batch axis
    if batch is None or batch >= nk:
        return jax.vmap(fn)(xs)
    return lax.map(fn, xs, batch_size=batch)


def sum_k(fn, xs, *, batch: int | None):
    """``sum_k fn(k)`` -- the same chunking, accumulated instead of stacked.

    This is the shape ``sum_band`` and ``sum_bec`` need: the per-k contribution
    is a whole density or a whole ``becsum``, so stacking ``nk`` of them and
    summing afterwards would defeat the point of chunking at all.

    ``batch = 1`` accumulates through a ``lax.scan`` with no ``vmap`` around the
    body, for the reason the module docstring gives.
    """
    nk = _leading(xs)
    if nk == 1:
        return fn(jax.tree_util.tree_map(lambda a: a[0], xs))
    if batch == 1:
        template = jax.eval_shape(fn, jax.tree_util.tree_map(lambda a: a[0], xs))
        zero = jax.tree_util.tree_map(lambda s: jnp.zeros(s.shape, s.dtype), template)

        def add(carry, one):
            return jax.tree_util.tree_map(jnp.add, carry, fn(one)), None

        total, _ = lax.scan(add, zero, xs)
        return total
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


def map_bands(fn, states, *, batch: int | None | str = "default"):
    """``fn`` over the leading axis of a block of states, ``batch`` bands at a time.

    ``states`` is ``(..., m, ndim)`` and ``fn`` maps a block of bands to a block
    of the same shape; the leading axes are flattened together first, so a
    caller with a spin or k index does not have to know how many there are.

    Unlike :func:`map_k` this is not a dial between two *algorithms* -- every
    band is transformed by the same code whatever the chunk -- so the answer is
    identical to the last bit rather than to round-off (2.8e-15 on the
    sixteen-atom cell is the difference in the *inputs* of a random test, not in
    the operation). It is a dial between two working sets, and the default is
    QE's: see the module docstring for why one band wins by 2.5x where the box
    is large, and by nothing at all where it is small.
    """
    batch = _resolve_band_batch(batch)
    if states.ndim == 1:
        return fn(states)

    shape = states.shape
    flat = states.reshape((-1,) + shape[-1:])
    m = flat.shape[0]
    if batch is None or batch >= m:
        return fn(states)

    full = m // batch
    head = flat[: full * batch].reshape((full, batch, shape[-1]))
    done = lax.map(fn, head).reshape((full * batch, shape[-1]))
    if m > full * batch:
        done = jnp.concatenate([done, fn(flat[full * batch :])], axis=0)
    return done.reshape(shape)


def _resolve_band_batch(requested: int | None | str = "default") -> int | None:
    if isinstance(requested, str):
        if requested == "default":
            return DEFAULT_BAND_BATCH
        return _named(requested)
    if requested is None:
        return None
    value = int(requested)
    if value < 1:
        raise ValueError(
            f"band batch must be a positive integer or None, got {requested!r}")
    return value


def sum_bands(fn, xs, *, batch: int | None | str = "default"):
    """``sum_b fn(b)`` over the band axis, ``batch`` bands at a time.

    ``sum_band.f90`` accumulates one band's ``|psi(r)|^2`` into ``rho`` inside
    ``DO ibnd = 1, nbnd`` and never holds more than one band's real-space field.
    That is the same working-set argument :func:`map_bands` records, applied to
    the density instead of to ``h_psi``, so it shares :func:`sum_k`'s machinery
    and only the default differs.

    The chunk changes the order the band contributions are added in and nothing
    else, so it moves the density by round-off and no more.
    """
    return sum_k(fn, xs, batch=_resolve_band_batch(batch))
