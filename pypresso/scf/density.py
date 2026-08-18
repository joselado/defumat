"""Building the electron density from occupied Kohn-Sham states.

``sum_band``: transform each occupied state to the grid and accumulate
``|psi(r)|^2`` with its weight. Following ``PW/src/sum_band.f90``.

Normalisation: a state is normalised as ``sum_G |c_G|^2 = 1``, so on the grid
``(1/N) sum_r |u(r)|^2 = 1`` and the density carries a ``1/Omega``. The weights
``wg`` already include both the k-point weight and the occupation, and they sum
to the number of electrons -- which is what makes ``integral rho = nelec`` an
exact identity rather than something to renormalise.
"""

from __future__ import annotations

import jax.numpy as jnp

from pypresso.basis.fft import g_to_r
from pypresso.system.cell import Cell

__all__ = ["sum_band", "band_density"]


def band_density(psi: jnp.ndarray, fft_index: jnp.ndarray, grid, weights: jnp.ndarray, cell: Cell):
    """Contribution of one k-point's bands to the density.

    Args:
        psi: ``(nbnd, npwx)`` wavefunctions at this k-point.
        fft_index: ``(npwx,)`` box indices for this k-point.
        grid: FFT dimensions.
        weights: ``(nbnd,)`` occupation weights ``wg`` for these bands.
        cell: for the cell volume.
    """
    field = g_to_r(psi, fft_index, grid)  # (nbnd, n1, n2, n3)
    return jnp.einsum("b,b...->...", weights, jnp.abs(field) ** 2) / cell.volume


def sum_band(psi, fft_index, grid, weights, cell: Cell) -> jnp.ndarray:
    """The density from every k-point, ``(n1, n2, n3)`` and real.

    Args:
        psi: ``(nk, nbnd, npwx)``.
        fft_index: ``(nk, npwx)``.
        weights: ``(nk, nbnd)`` occupation weights.
    """
    total = jnp.zeros(grid, dtype=psi.real.dtype)
    for ik in range(psi.shape[0]):
        total = total + band_density(psi[ik], fft_index[ik], grid, weights[ik], cell)
    return total
