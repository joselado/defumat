"""Expanding a product of two real spherical harmonics back into harmonics.

    Y_{l1 m1}(r) Y_{l2 m2}(r) = sum_LM  ap(LM, l1m1, l2m2) Y_LM(r)

These are Gaunt coefficients in the real-harmonic basis, and they are what turns
the augmentation charge from a radial function into something a plane-wave code
can add to the density: ``Q_ij(G)`` needs the product ``Y_i Y_j`` resolved into
the ``Y_LM`` the radial transforms ``Q^L_ij(|G|)`` are indexed by (``qvan2.f90``).

**They are computed, not tabulated.** ``upflib/uspp.f90``'s ``aainit`` does the
same thing: evaluate both sides at ``llx = (2 lmax + 1)^2`` directions, invert
the matrix of harmonics, and read the coefficients off. The reason to keep that
approach rather than substituting a closed-form Gaunt formula is that it can only
ever be consistent with :mod:`pypresso.pseudo.harmonics` -- the coefficients are
defined *by* whatever ordering and normalisation that module uses, so a formula
transcribed from a table would have to be re-derived if either changed. The
product is exactly representable in the finite basis, so nothing is approximated
by solving numerically; the result is exact to round-off, which a test checks.

QE draws its directions at random. Here they are the Fibonacci spiral, which is
deterministic, near-uniform on the sphere, and gives a better-conditioned matrix
than a random draw -- a run should not depend on a random number generator, and
an ill-conditioned draw would show up as noise in the augmentation charge.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from pypresso.pseudo.harmonics import real_spherical_harmonics

__all__ = ["harmonic_products", "coupled_channels"]


@lru_cache(maxsize=None)
def harmonic_products(lmax: int) -> np.ndarray:
    """``ap[LM, i, j]``, shaped ``((2 lmax + 1)^2, (lmax+1)^2, (lmax+1)^2)``.

    Args:
        lmax: the largest projector angular momentum (QE's ``lmaxkb``). The
            product of two harmonics of that ``l`` reaches ``L = 2 lmax``, hence
            the leading dimension.
    """
    if lmax < 0:
        raise ValueError(f"lmax must be non-negative, got {lmax}")

    nlm = (lmax + 1) ** 2
    nlmq = (2 * lmax + 1) ** 2

    directions = _fibonacci_sphere(nlmq)
    ylm_q = np.asarray(real_spherical_harmonics(directions, 2 * lmax))  # (nlmq, nlmq)
    ylm = ylm_q[:, :nlm]

    inverse = np.linalg.inv(ylm_q)  # (nlmq, nlmq): LM <- direction
    return np.einsum("Lr,ri,rj->Lij", inverse, ylm, ylm)


def coupled_channels(lmax: int, tolerance: float = 1.0e-8):
    """The ``(LM, i, j)`` triples with a non-vanishing coefficient.

    Most of ``ap`` is zero -- parity and the triangle rule between ``l_i``,
    ``l_j`` and ``L`` kill all but a few percent of it -- so the sum in
    ``Q_ij(G)`` runs over these, as ``qvan2``'s ``lpl``/``lpx`` lists do.
    """
    ap = harmonic_products(lmax)
    return np.argwhere(np.abs(ap) > tolerance)


def _fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` near-uniformly spread unit vectors."""
    index = np.arange(n) + 0.5
    z = 1.0 - 2.0 * index / n
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = np.pi * (1.0 + 5.0**0.5) * index
    return np.stack([radius * np.cos(phi), radius * np.sin(phi), z], axis=1)
