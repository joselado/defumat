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

import jax
import jax.numpy as jnp

from pypresso.basis.fft import g_to_r
from pypresso.system.cell import Cell

__all__ = ["sum_band", "band_density", "becsum"]


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

    Batched over k with ``vmap`` rather than accumulated in a Python loop: k is
    the leading axis of every wavefunction-shaped array precisely so that this
    is available (rule R6). Inside ``jit`` the sum over the batch fuses with the
    transforms, so the intermediate ``(nk, nbnd, n1, n2, n3)`` field is not
    materialised in full.
    """
    contributions = jax.vmap(band_density, in_axes=(0, 0, None, 0, None))(
        psi, fft_index, grid, weights, cell
    )
    return jnp.sum(contributions, axis=0)


def becsum(psi, vkb, weights, species_channels) -> tuple:
    """The projector occupation matrices ``becsum``, per species.

    ``PW/src/sum_band.f90``'s ``sum_bec``:

        becsum^a_ij = sum_k sum_b w_kb <psi_kb|beta_i^a> <beta_j^a|psi_kb>

    It is what the augmentation charge is built from -- the density outside the
    projector subspace comes from ``|psi|^2``, and everything inside it from
    these numbers.

    Args:
        psi: ``(nk, nbnd, npwx)`` wavefunctions.
        vkb: ``(nk, npwx, nkb)`` projectors.
        weights: ``(nk, nbnd)`` occupation weights.
        species_channels: for each species, the ``(nat_t, nh_t)`` array of
            channel columns belonging to each of its atoms, or ``None`` when the
            species is norm-conserving.

    Returns one real ``(nat_t, nh_t, nh_t)`` array per species. QE stores only
    the upper triangle, with the off-diagonal entries doubled; the full
    symmetric matrix is carried here instead, which is the same contraction
    against a ``Q_ij`` symmetric in the same pair of indices and avoids a packed
    index that nothing else in this code uses.
    """
    projections = jnp.einsum("kgc,kbg->kbc", vkb.conj(), psi)  # <beta_c|psi_kb>
    return tuple(
        None if channels is None else _becsum_species(projections, weights, channels)
        for channels in species_channels
    )


@jax.jit
def _becsum_species(projections, weights, channels):
    """One species' ``becsum``, gathering its atoms' channels in one go."""
    columns = projections[:, :, channels]  # (nk, nbnd, nat, nh)
    return jnp.real(
        jnp.einsum("kb,kbai,kbaj->aij", weights.astype(columns.dtype),
                   columns.conj(), columns)
    )
