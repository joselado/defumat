"""Transforms between the G-vector sphere and the real-space FFT box.

Two conventions are fixed here and used everywhere downstream.

**Scaling.** QE's ``invfft`` (G -> r) applies no factor and its ``fwfft``
(r -> G) divides by the number of grid points, so that

    f(r) = sum_G c_G exp(i G . r),      c_G = (1/N) sum_r f(r) exp(-i G . r)

NumPy/JAX put the ``1/N`` on the inverse transform instead, hence the explicit
rescaling below. The pair is exact inverses of each other, which is what the
round-trip test checks.

**Volume factors are not applied here.** Plane waves are normalised as
``exp(iGr)/sqrt(omega)`` in the formalism, but carrying ``sqrt(omega)`` inside
the transform makes every caller guess whether it has been applied. It is
applied where densities and matrix elements are formed instead.

The sphere is much smaller than the box (about 4% of it at a typical cutoff), so
these gather/scatter steps -- not the FFT itself -- are where a careless
implementation loses time on GPU.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["scatter_to_box", "gather_from_box", "g_to_r", "r_to_g",
           "sticks_to_r", "r_to_sticks",
           "scatter_to_box_gamma", "g_to_r_gamma", "r_to_g_gamma",
           "gamma_inner", "force_real_g0"]


def scatter_to_box(coefficients: jnp.ndarray, fft_index: jnp.ndarray, grid) -> jnp.ndarray:
    """Place sphere coefficients into a full FFT box.

    Args:
        coefficients: ``(..., npw)``. **Padding entries must already be zero**
            -- they share the index of G = 0, and this uses an accumulating
            scatter so that zeros contribute nothing. A plain ``.set`` would let
            padding overwrite the G = 0 coefficient.
        fft_index: ``(npw,)`` flat indices into the box.
        grid: ``(n1, n2, n3)``.
    """
    n1, n2, n3 = grid
    flat = jnp.zeros(coefficients.shape[:-1] + (n1 * n2 * n3,), dtype=coefficients.dtype)
    flat = flat.at[..., fft_index].add(coefficients)
    return flat.reshape(coefficients.shape[:-1] + (n1, n2, n3))


def gather_from_box(box: jnp.ndarray, fft_index: jnp.ndarray) -> jnp.ndarray:
    """Read sphere coefficients back out of a full FFT box."""
    n1, n2, n3 = box.shape[-3:]
    flat = box.reshape(box.shape[:-3] + (n1 * n2 * n3,))
    return flat[..., fft_index]


def g_to_r(coefficients: jnp.ndarray, fft_index: jnp.ndarray, grid) -> jnp.ndarray:
    """Sphere coefficients -> the field on the real-space grid (QE's invfft)."""
    box = scatter_to_box(coefficients, fft_index, grid)
    return jnp.fft.ifftn(box, axes=(-3, -2, -1)) * (grid[0] * grid[1] * grid[2])


def r_to_g(field: jnp.ndarray, fft_index: jnp.ndarray) -> jnp.ndarray:
    """A field on the real-space grid -> sphere coefficients (QE's fwfft)."""
    n1, n2, n3 = field.shape[-3:]
    box = jnp.fft.fftn(field, axes=(-3, -2, -1)) / (n1 * n2 * n3)
    return gather_from_box(box, fft_index)


# --- gamma-only: half the sphere, and the other half by conjugation -----------
#
# At ``k = 0`` a Kohn-Sham state can be chosen real, and then
# ``c(-G) = conj(c(G))``: half the sphere carries all the information and QE
# stores only that half (``ggen``'s ``gamma_only`` branch, reproduced in
# :func:`~defumat.basis.gvectors._half_sphere`). Everything below is what it
# costs to *consume* that storage rather than merely produce it.
#
# **``G = 0`` is the whole difficulty and it is not symmetric with the rest.**
# Every other stored G stands for a pair and is worth twice its stored value;
# ``G = 0`` is its own partner and is worth once. So a sum over the full sphere
# becomes ``2 Re(sum over the stored half) - (the G = 0 term)``, and the
# scatter that rebuilds the box must not write ``G = 0`` twice. QE says the same
# thing with ``gstart``: ``gstart == 2`` means "this rank holds G = 0, so skip
# entry 1 and correct for it", and `regterg.f90` spends four `MYDGER` calls and
# two explicit ``CMPLX(DBLE(psi(1,:)), 0)`` on exactly that.


def scatter_to_box_gamma(coefficients, fft_index, fft_index_minus, grid):
    """Rebuild the whole box from the stored half, by conjugation.

    ``c(-G) = conj(c(G))``, which makes the transformed field real. ``G = 0`` is
    written once -- it is its own conjugate partner, and
    ``fft_index_minus[0] == fft_index[0]``, so adding both would double it.

    The reality of the result also needs ``Im c(0) = 0``. That is *not* imposed
    here, because a caller whose ``c(0)`` has drifted complex has a bug upstream
    that silently zeroing it would hide -- see :func:`force_real_g0`, which is
    where QE imposes it and where this code imposes it too.
    """
    box = scatter_to_box(coefficients, fft_index, grid)
    n1, n2, n3 = grid
    flat = box.reshape(coefficients.shape[:-1] + (n1 * n2 * n3,))
    flat = flat.at[..., fft_index_minus[1:]].add(coefficients[..., 1:].conj())
    return flat.reshape(coefficients.shape[:-1] + (n1, n2, n3))


def g_to_r_gamma(coefficients, fft_index, fft_index_minus, grid):
    """Half-sphere coefficients -> the **real** field on the grid.

    The imaginary part is discarded rather than returned: it is zero for a
    consistent input and round-off otherwise, and every consumer of this wants a
    real field. Callers that need to *check* the reality should compare against
    the full-sphere transform, which is what the tests do.
    """
    box = scatter_to_box_gamma(coefficients, fft_index, fft_index_minus, grid)
    n = grid[0] * grid[1] * grid[2]
    return (jnp.fft.ifftn(box, axes=(-3, -2, -1)) * n).real


def r_to_g_gamma(field, fft_index):
    """A real field on the grid -> the stored half of its coefficients.

    The same gather as :func:`r_to_g`; only half the indices are asked for. No
    factor: the stored coefficients *are* the field's coefficients, and the
    doubling belongs to sums over them, not to the transform.
    """
    return r_to_g(field, fft_index)


def gamma_inner(a, b, gamma_only: bool, keepdims: bool = False):
    """``<a|b>`` over a plane-wave axis, on a half sphere or a whole one.

    For ``gamma_only`` the stored half is doubled and ``G = 0`` -- which stands
    for itself rather than for a pair -- is counted once::

        <a|b> = 2 Re sum_stored conj(a) b  -  Re(conj(a_0) b_0)

    and the result is **real**, which is the point: `regterg` works with real
    subspace matrices, so a complex overlap here is round-off that
    ``generalised_eigh`` would turn into an arbitrary phase per eigenvector and
    hence a complex ``c(0)`` and a field that is no longer real.

    One helper rather than the same three lines at each site, because the
    ``G = 0`` correction is exactly the term that gets dropped in one place out
    of ten and only an energy comparison notices.
    """
    product = jnp.sum(a.conj() * b, axis=-1, keepdims=keepdims)
    if not gamma_only:
        return product
    zero = a[..., :1].conj() * b[..., :1]
    if not keepdims:
        zero = zero[..., 0]
    return 2.0 * product.real - zero.real


def force_real_g0(coefficients, gamma_only: bool):
    """``Im c(G = 0) = 0``, which a real field requires and round-off breaks.

    ``regterg.f90:174`` and ``:375``: ``psi(1,k) = CMPLX(DBLE(psi(1,k)), 0)``,
    applied every time a vector enters the subspace. It is not cosmetic -- an
    imaginary part at ``G = 0`` makes the rebuilt field complex, and the run
    converges to a plausible wrong answer rather than failing.

    **Written as a select rather than as ``.at[..., 0].set(...)``**, which is
    the same value and not the same buffer. A scatter is opaque to XLA's loop
    fusion, and this sits in the middle of the Davidson correction chain --
    precondition, mask, *this*, normalise -- where every link is otherwise
    elementwise over a ``(nbnd, npwx)`` block. The scatter split that chain in
    two and each half materialised a block of its own; on a 157-atom slab a
    block is 5.8 GiB. A select fuses with its neighbours and costs nothing.
    """
    if not gamma_only:
        return coefficients
    at_zero = jnp.arange(coefficients.shape[-1]) == 0
    return jnp.where(at_zero, coefficients.real.astype(coefficients.dtype),
                     coefficients)


# --- the stick layout ---------------------------------------------------------
#
# The pair below is the same transform as g_to_r / r_to_g, done QE's way: the z
# pass on the sticks the wavefunction sphere actually occupies, then the xy
# passes on the box. The field they produce is laid out ``(n3, n1, n2)``, with
# the xy plane contiguous, because that is what makes the 2D pass cheap -- see
# defumat.basis.sticks. Anything multiplying such a field, the local potential
# above all, has to be stored in the same order.


def sticks_to_r(coefficients: jnp.ndarray, sticks, columns, index) -> jnp.ndarray:
    """Sphere coefficients -> the field on the grid, as ``(..., n3, n1, n2)``.

    QE's ``invfft('Wave')``: ``cft_1z`` over the sticks, then ``cft_2xy``.
    """
    n1, n2, n3 = sticks.grid
    lead = coefficients.shape[:-1]

    compact = jnp.zeros(lead + (sticks.nsticks * n3,), coefficients.dtype)
    compact = compact.at[..., index].add(coefficients)
    compact = jnp.fft.ifft(compact.reshape(lead + (sticks.nsticks, n3)), axis=-1) * n3

    box = jnp.zeros(lead + (n3, n1 * n2), coefficients.dtype)
    box = box.at[..., columns].set(jnp.moveaxis(compact, -1, -2))
    return jnp.fft.ifftn(box.reshape(lead + (n3, n1, n2)), axes=(-2, -1)) * (n1 * n2)


def sticks_local(coefficients: jnp.ndarray, sticks, columns, index, apply,
                 potential: jnp.ndarray, plane_chunk: int) -> jnp.ndarray:
    """``r_to_sticks(apply(sticks_to_r(c), V))``, a chunk of ``z`` planes at a time.

    Between the two ``z`` passes, which run on the sticks only, everything the
    local term does is **plane-local**: for one ``z`` the ``xy`` transform, the
    pointwise product with that plane of ``V`` and the transform back touch that
    plane and nothing else. So instead of taking the whole ``(n3, n1, n2)`` box
    through each step in turn -- the ``xy`` pass as two full passes over the
    box, then the product, then two more -- a chunk of ``plane_chunk`` planes is
    taken through all of it while it sits in a core's private cache, and only
    the stick arrays cross the box. ``cft_2xy`` in QE does the plane transforms
    on the whole ``psic``; this goes one step further and fuses the product in.

    **It is a memory measure, and it is worth most when cores share memory.**
    On an i5-12600K the local term of 64 bands on si32's ``36x36x144`` box took
    111 ms whole and 99 ms in 16-plane chunks on one core, and with six copies
    running at once 232 ms whole against 99 ms chunked: the whole-box path
    streams each band's 3 MB box through the shared L3 several times and slows
    2.1x under load, the chunked one does not slow at all (``PERFORMANCE.md``,
    "The local term a chunk of planes at a time"). The result is the same to the
    last bit: the same transforms on the same numbers, issued in a different
    order.

    ``apply(field, v)`` is the pointwise product for one chunk -- ``field * v``
    for a scalar potential, :func:`~defumat.hamiltonian.noncollinear.spin_multiply`
    for a 2x2 one -- with ``field`` shaped ``(..., chunk, n1, n2)`` and ``v`` the
    matching planes of ``potential``, whose last three axes are the stick
    layout's ``(n3, n1, n2)``. ``coefficients`` may carry any leading axes (a
    block of bands, the two spinor components); they ride along untouched. The
    loop over chunks is a ``lax.map``, so it compiles once and differentiates.
    """
    n1, n2, n3 = sticks.grid
    nsticks = sticks.nsticks
    lead = coefficients.shape[:-1]
    chunk = max(1, min(int(plane_chunk), n3))
    nchunks = -(-n3 // chunk)
    padded = nchunks * chunk

    compact = jnp.zeros(lead + (nsticks * n3,), coefficients.dtype)
    compact = compact.at[..., index].add(coefficients)
    compact = jnp.fft.ifft(compact.reshape(lead + (nsticks, n3)), axis=-1) * n3

    # (..., nsticks, n3) -> (nchunks, ..., chunk, nsticks): the planes of each chunk.
    planes = jnp.moveaxis(compact, -1, -2)
    planes = jnp.pad(planes, [(0, 0)] * len(lead) + [(0, padded - n3), (0, 0)])
    planes = jnp.moveaxis(planes.reshape(lead + (nchunks, chunk, nsticks)), -3, 0)
    vlead = potential.shape[:-3]
    v = jnp.pad(potential, [(0, 0)] * len(vlead) + [(0, padded - n3), (0, 0), (0, 0)])
    v = jnp.moveaxis(v.reshape(vlead + (nchunks, chunk, n1, n2)), -4, 0)

    def one_chunk(pair):
        sticks_here, v_here = pair
        box = jnp.zeros(lead + (chunk, n1 * n2), coefficients.dtype)
        box = box.at[..., columns].set(sticks_here)
        field = jnp.fft.ifftn(box.reshape(lead + (chunk, n1, n2)), axes=(-2, -1)) * (n1 * n2)
        back = jnp.fft.fftn(apply(field, v_here), axes=(-2, -1)) / (n1 * n2)
        return back.reshape(lead + (chunk, n1 * n2))[..., columns]

    out = jax.lax.map(one_chunk, (planes, v))
    out = jnp.moveaxis(out, 0, -3).reshape(lead + (padded, nsticks))[..., :n3, :]
    out = jnp.fft.fft(jnp.moveaxis(out, -1, -2), axis=-1) / n3
    return out.reshape(lead + (nsticks * n3,))[..., index]


def sticks_density(states: jnp.ndarray, weights: jnp.ndarray, sticks, columns, index,
                   combine, ncomp: int, plane_chunk: int) -> jnp.ndarray:
    """``sum_b w_b combine(psi_b(r))`` on the grid, a chunk of ``z`` planes at a time.

    ``sum_band``'s band loop done the way :func:`sticks_local` does the local
    term: each band goes through the ``z`` transform on its sticks, and then a
    chunk of planes at a time through the ``xy`` transform, and its
    contribution is added straight into those planes of the density, so no
    band's box is ever built whole. ``states`` is ``(nbnd, ..., npwx)`` -- any
    axes between the band and the plane wave (the two spinor components) ride
    along -- and ``weights`` is ``(nbnd,)``. ``combine(field)`` maps one chunk's
    field, ``(..., chunk, n1, n2)``, to its ``(ncomp, chunk, n1, n2)`` real
    contribution: ``|psi|^2`` for a scalar density, the Pauli components for a
    spinor. Returns ``(ncomp, n1, n2, n3)``, not yet divided by the volume.

    The bands are walked one at a time and the chunks inside each, both as
    loops that compile once and differentiate; the density is carried through
    both and updated in place. The transforms are the ones the whole-box path
    makes, issued in a different order, so the density changes by round-off.
    """
    n1, n2, n3 = sticks.grid
    nsticks = sticks.nsticks
    lead = states.shape[1:-1]
    chunk = max(1, min(int(plane_chunk), n3))
    nchunks = -(-n3 // chunk)
    padded = nchunks * chunk
    real = jnp.zeros((), states.dtype).real.dtype
    total = jnp.zeros((ncomp, padded, n1 * n2), real)

    def one_band(rho, pair):
        state, weight = pair
        compact = jnp.zeros(lead + (nsticks * n3,), states.dtype).at[..., index].add(state)
        compact = jnp.fft.ifft(compact.reshape(lead + (nsticks, n3)), axis=-1) * n3
        planes = jnp.moveaxis(compact, -1, -2)
        planes = jnp.pad(planes, [(0, 0)] * len(lead) + [(0, padded - n3), (0, 0)])

        def one_chunk(j, rho):
            start = j * chunk
            here = jax.lax.dynamic_slice_in_dim(planes, start, chunk, axis=-2)
            box = jnp.zeros(lead + (chunk, n1 * n2), states.dtype).at[..., columns].set(here)
            field = jnp.fft.ifftn(box.reshape(lead + (chunk, n1, n2)), axes=(-2, -1)) * (n1 * n2)
            add = weight * combine(field).reshape(ncomp, chunk, n1 * n2)
            now = jax.lax.dynamic_slice_in_dim(rho, start, chunk, axis=1)
            return jax.lax.dynamic_update_slice_in_dim(rho, now + add, start, axis=1)

        return jax.lax.fori_loop(0, nchunks, one_chunk, rho), None

    total, _ = jax.lax.scan(one_band, total, (states, weights))
    total = total[:, :n3].reshape(ncomp, n3, n1, n2)
    return jnp.moveaxis(total, 1, -1)


def r_to_sticks(field: jnp.ndarray, sticks, columns, index) -> jnp.ndarray:
    """The inverse of :func:`sticks_to_r`, back to sphere coefficients.

    QE's ``fwfft('Wave')``. Only the sticks are transformed back along z, since
    everything outside them is discarded by the sphere anyway.
    """
    n1, n2, n3 = sticks.grid
    lead = field.shape[:-3]

    box = jnp.fft.fftn(field, axes=(-2, -1)) / (n1 * n2)
    compact = jnp.moveaxis(box.reshape(lead + (n3, n1 * n2))[..., columns], -1, -2)
    compact = jnp.fft.fft(compact, axis=-1) / n3
    return compact.reshape(lead + (sticks.nsticks * n3,))[..., index]
