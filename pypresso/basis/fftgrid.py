"""FFT grid dimensions, chosen the way Quantum ESPRESSO chooses them.

The grid is not a free parameter: it must be large enough that the G-vector
sphere of radius ``sqrt(gcut)`` fits inside the box without its periodic images
overlapping, and it must factorise into sizes the FFT library likes. QE's answer
to both is in ``FFTXlib/src/fft_types.f90`` (``realspace_grid_init`` and
``grid_set``) and ``FFTXlib/src/fft_support.f90`` (``allowed``,
``good_fft_order``); this module reproduces it exactly, because the grid
dimensions decide ``ngm`` and therefore every subsequent number.

Everything here works in QE's scaled units: lattice vectors in units of ``alat``,
reciprocal vectors in units of ``2*pi/alat``, and cutoffs as ``gcut = ecut /
tpiba^2`` in Ry with ``hbar^2/2m = 1``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["allowed_fft_size", "good_fft_order", "fft_grid_dimensions", "gcut_from_ecut"]

#: Prime factors an FFT size may contain. QE's ``allowed`` additionally rejects
#: 7 and 11 for every build except IBM ESSL, so the practical set is 2, 3, 5.
_FFT_FACTORS = (2, 3, 5, 7, 11)


def gcut_from_ecut(ecut: float, alat: float) -> float:
    """Cutoff in units of ``(2*pi/alat)^2``, which is how QE stores ``gcutm``."""
    tpiba2 = (2.0 * np.pi / alat) ** 2
    return ecut / tpiba2


def allowed_fft_size(n: int) -> bool:
    """Whether ``n`` is an FFT size QE would accept (a 2-3-5 product)."""
    if n < 1:
        return False
    remainder = n
    powers = {}
    for factor in _FFT_FACTORS:
        powers[factor] = 0
        while remainder % factor == 0:
            remainder //= factor
            powers[factor] += 1
    if remainder != 1:
        return False  # a prime factor above 11: no good in any case
    # FFTW and everything except IBM ESSL: no factors of 7 or 11 either.
    return powers[7] == 0 and powers[11] == 0


def good_fft_order(n: int, multiple_of: int | None = None) -> int:
    """The smallest acceptable FFT size that is at least ``n``."""
    candidate = int(n)
    while not allowed_fft_size(candidate) or (multiple_of and candidate % multiple_of):
        candidate += 1
    return candidate


def fft_grid_dimensions(
    at_alat: np.ndarray,
    bg_2pi_alat: np.ndarray,
    gcut: float,
    fft_factors: tuple[int, int, int] = (1, 1, 1),
) -> tuple[int, int, int]:
    """The dense FFT grid for a cutoff, following ``realspace_grid_init``.

    Args:
        at_alat: lattice vectors in units of alat, rows ``a1, a2, a3``.
        bg_2pi_alat: reciprocal vectors in units of ``2*pi/alat``, rows.
        gcut: cutoff in units of ``(2*pi/alat)^2``.
        fft_factors: each dimension must be a multiple of its entry. These come
            from the crystal's fractional translations -- see
            ``Symmetries.fft_factors``. Diamond silicon's are ``(4, 4, 4)``, and
            leaving them at 1 gives grids one FFT size below QE's.

    The two-step structure is QE's and matters: a first estimate bounds the
    Miller indices via ``|G . a_i| = |n_i| <= |G_max| |a_i|``, then ``grid_set``
    scans that box for the largest index actually inside the sphere, and the grid
    is ``2*nb + 1`` so the sphere just touches its periodic image. Only then is
    the size rounded up to a factorisable one.
    """
    at_alat = np.asarray(at_alat, dtype=float)
    bg_2pi_alat = np.asarray(bg_2pi_alat, dtype=float)

    # First estimate: an upper bound on the Miller indices.
    estimate = (np.sqrt(gcut) * np.linalg.norm(at_alat, axis=1)).astype(int) + 1

    nb = _largest_indices_inside(bg_2pi_alat, gcut, estimate)
    return tuple(
        good_fft_order(2 * int(n) + 1, factor)
        for n, factor in zip(nb, fft_factors)
    )


def _largest_indices_inside(bg: np.ndarray, gcut: float, bound: np.ndarray) -> np.ndarray:
    """``grid_set``: the largest |Miller index| of any G inside the sphere."""
    ranges = [np.arange(-int(b), int(b) + 1) for b in bound]
    i, j, k = np.meshgrid(*ranges, indexing="ij")
    miller = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1)

    g2 = np.sum((miller @ bg) ** 2, axis=1)
    inside = miller[g2 < gcut]  # QE's grid_set uses a strict inequality here
    if len(inside) == 0:
        raise ValueError("no G-vectors inside the cutoff sphere; is the cutoff far too small?")
    return np.abs(inside).max(axis=0)
