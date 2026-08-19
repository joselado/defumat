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

import jax.numpy as jnp

__all__ = ["scatter_to_box", "gather_from_box", "g_to_r", "r_to_g",
           "sticks_to_r", "r_to_sticks"]


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


# --- the stick layout ---------------------------------------------------------
#
# The pair below is the same transform as g_to_r / r_to_g, done QE's way: the z
# pass on the sticks the wavefunction sphere actually occupies, then the xy
# passes on the box. The field they produce is laid out ``(n3, n1, n2)``, with
# the xy plane contiguous, because that is what makes the 2D pass cheap -- see
# pypresso.basis.sticks. Anything multiplying such a field, the local potential
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
