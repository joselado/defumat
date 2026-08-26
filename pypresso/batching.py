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

This module is the dial between the two, and on a CPU the default is QE's
end of it:

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

**Both defaults are per platform, because both arguments above are arguments
about a cache and an accelerator has none.** On a CPU each dial defaults to 1
and every number in this docstring says why; on anything else both default to
``None``, the whole axis at once. That is not a preference, it is what was
measured on the device (``PERFORMANCE.md``, "First contact with a GPU"):
``al10-metal`` runs at **801 ms/iteration on the CPU defaults and 177 at
``k=all, b=all``** -- **4.5x given up** by inheriting them -- 16-atom silicon
runs at **0.20x**, an outright loss against four CPU cores, and at 32 atoms
``band_batch = 1`` did not finish two SCF runs in the fifteen minutes four CPU
cores need eleven seconds for. **The two axes move together or not at all**:
``k=all, b=1`` is *worse than either end* (2075 ms), because batching k while
looping bands multiplies the per-band launches by ``nk``, so ``_platform_default``
answers for both dials at once rather than per axis.

``GPU.md`` §5's rule is that a platform-dependent choice is "a dial with a
per-platform default and both settings tested", never a rewrite -- so nothing
here changes on a CPU, both ends stay reachable everywhere, and the order of
precedence is unchanged: an explicit argument beats
``PYPRESSO_K_BATCH``/``PYPRESSO_BAND_BATCH``, which beat the platform. And the
default is not visible in a result under any of them: the chunk size changes
only the order the k contributions are *added* in (~1e-15 Ry), which is what
``tests/unit/test_batching.py`` pins.

**A single k-point is unaffected either way.** :func:`map_k` and :func:`sum_k`
short-circuit ``nk == 1`` to a direct call before they look at the chunk size,
so the new default does not put a width-one batch axis on the single-k
benchmark cells -- the case "a batch of one is not a batch" above is about.
"""

from __future__ import annotations

import functools
import os
import warnings

import jax
import jax.numpy as jnp
from jax import lax

__all__ = ["DEFAULT_K_BATCH", "resolve_k_batch", "map_k", "sum_k",
           "DEFAULT_BAND_BATCH", "map_bands", "sum_bands"]


_UNSET = object()


def _from_environment(name: str) -> int | None | object:
    """``name`` read as a chunk size: an integer, or ``all``/``0`` for a ``vmap``.

    Returns :data:`_UNSET` when the variable is not set, which is *not* the same
    as ``None``: ``None`` is a meaningful setting here -- every k-point at once
    -- so it cannot double as "nothing was said" and let the platform decide.
    """
    setting = os.environ.get(name, "").strip().lower()
    if not setting:
        return _UNSET
    if setting in ("all", "0", "off", "none"):
        return None
    try:
        value = int(setting)
    except ValueError:
        warnings.warn(f"ignoring {name}={setting!r}: not a number",
                      RuntimeWarning, stacklevel=2)
        return _UNSET
    return None if value < 1 else value


@functools.cache
def _backend() -> str:
    """The platform JAX will run on -- ``cpu``, ``gpu``, ``tpu``.

    Asked lazily and cached, because asking *initialises* the backend and on a
    GPU node that allocates device memory: importing this module must not do
    that as a side effect. Anything that is not ``cpu`` counts as an
    accelerator, so a name this was never tested against (``cuda``, ``rocm``)
    lands on the accelerator default rather than on neither.
    """
    try:
        return jax.default_backend()
    except Exception:       # no backend at all: there is nothing to batch for
        return "cpu"


def _platform_default() -> int | None:
    """QE's loop on a CPU, the whole axis at once on an accelerator.

    This is the one place the two ends of both dials are chosen, and it is a
    *default* rather than a rule: every entry point takes ``k_batch`` and
    ``band_batch``, and ``PYPRESSO_K_BATCH``/``PYPRESSO_BAND_BATCH`` override a
    whole process. See the module docstring for what each end costs where.
    """
    return 1 if _backend() == "cpu" else None


def _k_default() -> int | None:
    setting = _from_environment("PYPRESSO_K_BATCH")
    return _platform_default() if setting is _UNSET else setting


def _band_default() -> int | None:
    setting = _from_environment("PYPRESSO_BAND_BATCH")
    return _platform_default() if setting is _UNSET else setting


def __getattr__(name: str):
    """``DEFAULT_K_BATCH`` and ``DEFAULT_BAND_BATCH``, resolved when read.

    Attributes (PEP 562) rather than module constants, because the platform
    half of the answer costs a backend initialisation to obtain and the
    environment half may be set after this module is imported.
    """
    if name == "DEFAULT_K_BATCH":
        return _k_default()
    if name == "DEFAULT_BAND_BATCH":
        return _band_default()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def resolve_k_batch(requested: int | None | str = "default") -> int | None:
    """Turn what a caller passed into a chunk size.

    The sentinel is the string ``"default"`` rather than ``None``, because
    ``None`` is a meaningful value here -- it asks for every k-point at once --
    and a caller that has one must be able to pass it through.
    """
    if isinstance(requested, str):
        if requested == "default":
            return _k_default()
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
            return _band_default()
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
