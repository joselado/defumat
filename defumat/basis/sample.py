"""Evaluating a grid field at arbitrary points, by its own Fourier series.

A field carried on the FFT grid -- a density, a potential, the tunnelling local
density of states -- is known everywhere, not only at the grid points: it is a
finite sum of plane waves inside the cutoff sphere, so

    f(r) = sum_G c_G e^{iG.r},    c_G = (1/N) sum_r f(r) e^{-iG.r}

is an *exact* statement about the field rather than an interpolation of it, and
evaluating it at a point the grid does not carry costs one sum over the sphere.
Elk's ``rfpts`` does the same job for a muffin-tin function and has to do real
work inside the spheres; in a plane-wave code the whole of it is this sum.

**The sphere, not the box.** The coefficients come from
:func:`~defumat.basis.fft.r_to_g`, which gathers the ``ecutrho`` sphere and
drops the box corners -- and the corners have to be dropped, for the reason
:func:`defumat.workflows.sfac._require_a_representable_cutoff` gives at length:
what sits there is aliasing rather than a Fourier component of the field, and it
is small, smooth and entirely plausible. Summing the box instead reproduces the
grid values exactly (it is the inverse transform) and is wrong everywhere
between them.

**Peak working set** is ``chunk x ngm`` complex numbers for the phase table,
which is why the points are chunked at all: a 40x40 plotting plane against a
30000-vector sphere is 768 MB in one block and 25 MB in blocks of fifty. The
default chunk targets ~32 MB and the argument is there to override it.
"""

from __future__ import annotations

import numpy as np

from defumat.basis.fft import r_to_g
from defumat.basis.gvectors import GVectors

__all__ = ["sample_field", "sample_coefficients"]

#: Complex entries in one phase block -- ~32 MB of complex128.
_PHASE_BLOCK = 2_000_000


def sample_field(field, gvectors: GVectors, points, chunk: int | None = None):
    """``f(r)`` at ``points``, for a real field sampled on the dense grid.

    Args:
        field: ``(..., n1, n2, n3)`` real. Leading axes are carried through, so
            the channels of a spin density go in one call.
        gvectors: the :class:`~defumat.basis.gvectors.GVectors` the field lives
            on -- the *dense* set for anything built from a density.
        points: ``(np, 3)`` **crystal** coordinates, Elk's convention for
            ``vclp2d`` and the one a surface is described in. They need not lie
            in the unit cell: the series is periodic and evaluating outside it
            is the same thing as wrapping.
        chunk: points per phase block.

    Returns ``(..., np)`` real, in whatever units the field carried.
    """
    coefficients = np.asarray(r_to_g(np.asarray(field), gvectors.fft_index))
    return sample_coefficients(coefficients, gvectors, points, chunk=chunk)


def sample_coefficients(coefficients, gvectors: GVectors, points,
                        chunk: int | None = None):
    """:func:`sample_field` for a field already transformed to the sphere.

    Separate because a caller that samples the same field on many planes -- a
    constant-current scan is exactly that -- should transform it once.
    """
    coefficients = np.asarray(coefficients)
    miller = np.asarray(gvectors.miller, dtype=float)
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if points.shape[-1] != 3:
        raise ValueError(f"points must be (np, 3) crystal coordinates, got {points.shape}")

    ngm = miller.shape[0]
    if coefficients.shape[-1] != ngm:
        raise ValueError(
            f"the coefficients carry {coefficients.shape[-1]} G-vectors and the "
            f"G set has {ngm}: they are not the same sphere"
        )
    if chunk is None:
        chunk = max(1, int(_PHASE_BLOCK // max(ngm, 1)))

    leading = coefficients.shape[:-1]
    flat = coefficients.reshape((-1, ngm))
    out = np.empty(flat.shape[:1] + points.shape[:1])
    for start in range(0, points.shape[0], chunk):
        block = points[start:start + chunk]
        # exp(iG.r) = exp(2 pi i h.s) with h the Miller indices and s the
        # crystal coordinate: the reciprocal and direct bases cancel, so the
        # cell never enters and no cartesian transform is needed.
        phases = np.exp(2.0j * np.pi * (block @ miller.T))
        out[:, start:start + chunk] = np.real(flat @ phases.T)
    return out.reshape(leading + points.shape[:1])
