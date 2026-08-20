"""Building the electron density from occupied Kohn-Sham states.

``sum_band``: transform each occupied state to the grid and accumulate
``|psi(r)|^2`` with its weight. Following ``PW/src/sum_band.f90``.

Normalisation: a state is normalised as ``sum_G |c_G|^2 = 1``, so on the grid
``(1/N) sum_r |u(r)|^2 = 1`` and the density carries a ``1/Omega``. The weights
``wg`` already include both the k-point weight and the occupation, and they sum
to the number of electrons -- which is what makes ``integral rho = nelec`` an
exact identity rather than something to renormalise.

**Spin.** Wavefunctions, weights and the density all carry a leading channel
axis, and the accumulation is per channel: ``sum_band`` writes into
``rho%of_r(:,current_spin)``, with ``isk(ik)`` saying which. QE flattens the two
channels into one k-list of length ``2 nks`` and lets that index decide; here
the channel is a separate axis, which keeps ``k`` the leading *independent* axis
of every wavefunction-shaped array -- the property the batching and the eventual
sharding rest on.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from pypresso.basis.fft import g_to_r
from pypresso.batching import resolve_k_batch, sum_bands, sum_k
from pypresso.system.cell import Cell

__all__ = ["sum_band", "band_density", "becsum", "spinor_sum_band",
           "spinor_band_density", "spinor_becsum"]


def band_density(psi: jnp.ndarray, fft_index: jnp.ndarray, grid, weights: jnp.ndarray, cell: Cell):
    """Contribution of one k-point's bands to the density.

    Args:
        psi: ``(nbnd, npwx)`` wavefunctions at this k-point.
        fft_index: ``(npwx,)`` box indices for this k-point.
        grid: FFT dimensions.
        weights: ``(nbnd,)`` occupation weights ``wg`` for these bands.
        cell: for the cell volume.
    """
    def one_band(arrays):
        state, weight = arrays
        return weight * jnp.abs(g_to_r(state, fft_index, grid)) ** 2

    # One band at a time, as ``sum_band.f90`` accumulates them: a band's
    # real-space box is the working set (:mod:`pypresso.batching`).
    return sum_bands(one_band, (psi, weights)) / cell.volume


def sum_band(psi, fft_index, grid, weights, cell: Cell,
             k_batch: int | None | str = "default") -> jnp.ndarray:
    """The density from every k-point, ``(nspin, n1, n2, n3)`` and real.

    Args:
        psi: ``(nspin, nk, nbnd, npwx)``.
        fft_index: ``(nk, npwx)`` -- shared by the channels, since the plane-wave
            basis at a k-point does not depend on spin.
        weights: ``(nspin, nk, nbnd)`` occupation weights.

    Accumulated ``k_batch`` k-points at a time (:mod:`pypresso.batching`), which
    is ``sum_band.f90``'s own structure -- it adds each k-point's bands into
    ``rho%of_r`` inside ``k_loop`` and never holds more than one k-point's
    real-space fields. The batched end of the dial holds ``(nk, nbnd, n1, n2,
    n3)`` complex numbers in flight, which on a large cell is the second-largest
    working set in the code after the Davidson subspace.
    """
    batch = resolve_k_batch(k_batch)

    def channel(states, occupations):
        def one_k(arrays):
            state, index, occupation = arrays
            return band_density(state, index, grid, occupation, cell)

        return sum_k(one_k, (states, fft_index, occupations), batch=batch)

    return jax.vmap(channel)(psi, weights)


def becsum(psi, vkb, weights, species_channels,
           k_batch: int | None | str = "default") -> tuple:
    """The projector occupation matrices ``becsum``, per species.

    ``PW/src/sum_band.f90``'s ``sum_bec``:

        becsum^a_ij = sum_k sum_b w_kb <psi_kb|beta_i^a> <beta_j^a|psi_kb>

    It is what the augmentation charge is built from -- the density outside the
    projector subspace comes from ``|psi|^2``, and everything inside it from
    these numbers.

    Args:
        psi: ``(nspin, nk, nbnd, npwx)`` wavefunctions.
        vkb: ``(nk, npwx, nkb)`` projectors -- the same in both channels.
        weights: ``(nspin, nk, nbnd)`` occupation weights.
        species_channels: for each species, the ``(nat_t, nh_t)`` array of
            channel columns belonging to each of its atoms, or ``None`` when the
            species is norm-conserving.

    Returns one real ``(nspin, nat_t, nh_t, nh_t)`` array per species. QE stores
    only the upper triangle, with the off-diagonal entries doubled; the full
    symmetric matrix is carried here instead, which is the same contraction
    against a ``Q_ij`` symmetric in the same pair of indices and avoids a packed
    index that nothing else in this code uses.

    The spin index is ``becsum``'s third in QE (``becsum(ijh, na, nspin)``) and
    it is not decoration: with two channels the augmentation charge, the
    self-consistent ``D_ij`` and PAW's one-centre terms all become per-channel
    quantities, and they are all built from this one.
    """
    batch = resolve_k_batch(k_batch)

    def channel(states, occupations):
        def one_k(arrays):
            projectors, state, occupation = arrays
            projections = jnp.einsum("gc,bg->bc", projectors.conj(), state)
            return tuple(
                None if channels is None
                else _becsum_species(projections, occupation, channels)
                for channels in species_channels
            )

        return sum_k(one_k, (vkb, states, occupations), batch=batch)

    # One channel at a time rather than a spin axis through the accumulation:
    # QE has no spin axis here either -- ``sum_bec`` writes into
    # ``becsum(:,:,current_spin)`` and its k-list runs over both channels.
    per_channel = [channel(psi[spin], weights[spin]) for spin in range(psi.shape[0])]
    return tuple(
        None if channels is None
        else jnp.stack([values[species] for values in per_channel])
        for species, channels in enumerate(species_channels)
    )


@jax.jit
def _becsum_species(projections, weights, channels):
    """One species' ``becsum`` at one k-point, gathering its atoms' channels."""
    columns = projections[:, channels]  # (nbnd, nat, nh)
    return jnp.real(
        jnp.einsum("b,bai,baj->aij", weights.astype(columns.dtype),
                   columns.conj(), columns)
    )


def spinor_band_density(psi, fft_index, grid, weights, cell: Cell, nspin_mag: int):
    """One k-point's contribution to a noncollinear density.

    Args:
        psi: ``(nbnd, 2 npwx)`` spinors -- the two components stored one after
            the other, as :class:`~pypresso.hamiltonian.noncollinear.SpinorHamiltonian`
            holds them.
        nspin_mag: 1 for the charge alone, 4 for ``(n, m_x, m_y, m_z)``.

    ``get_rho_k`` and ``get_rho_domag`` in ``sum_band.f90``. The charge is the
    sum of the two components' densities and the magnetization is the Pauli
    expectation value ``psi^dagger sigma psi`` -- so the *same* wavefunctions
    give one number or four depending only on whether the calculation carries a
    magnetization, and a nonmagnetic spin-orbit run keeps a scalar density.
    """
    npwx = psi.shape[-1] // 2
    components = psi.reshape(psi.shape[:-1] + (2, npwx))
    field = g_to_r(components, fft_index, grid)  # (nbnd, 2, n1, n2, n3)
    up, down = field[:, 0], field[:, 1]

    charge = jnp.abs(up) ** 2 + jnp.abs(down) ** 2
    if nspin_mag == 1:
        stacked = charge[None]
    else:
        cross = jnp.conj(up) * down
        stacked = jnp.stack([
            charge,
            2.0 * jnp.real(cross),
            2.0 * jnp.imag(cross),
            jnp.abs(up) ** 2 - jnp.abs(down) ** 2,
        ])
    return jnp.einsum("b,cb...->c...", weights, stacked) / cell.volume


def spinor_sum_band(psi, fft_index, grid, weights, cell: Cell, nspin_mag: int,
                    k_batch: int | None | str = "default"):
    """A noncollinear density from every k-point, ``(nspin_mag, n1, n2, n3)``.

    Args:
        psi: ``(nk, nbnd, 2 npwx)``.
        weights: ``(nk, nbnd)``.
    """
    def one_k(arrays):
        state, index, occupation = arrays
        return spinor_band_density(state, index, grid, occupation, cell, nspin_mag)

    return sum_k(one_k, (psi, fft_index, weights), batch=resolve_k_batch(k_batch))


def spinor_becsum(psi, vkb, weights, species_channels,
                  k_batch: int | None | str = "default") -> tuple:
    """The spinor projector occupations, per species, before the spin transform.

        becsum_nc^a_{i s1, j s2} = sum_kb w_kb <psi_kb|beta_i^a s1>
                                              <beta_j^a s2|psi_kb>

    ``sum_bec``'s noncollinear branch. It is *not* the ``becsum`` the
    augmentation charge is built from: that one has ``nspin_mag`` real
    components and comes from contracting this with the spin-orbit coefficients
    (:meth:`pypresso.pseudo.spinorbit.SpinOrbitCoupling.becsum_so`). Keeping the
    two apart is QE's structure too -- ``aux_nc`` then ``add_becsum_so`` -- and
    it is what lets the same accumulation serve a spin-orbit species and a
    scalar-relativistic one in the same cell.

    Args:
        psi: ``(nk, nbnd, 2 npwx)``.
        vkb: ``(nk, npwx, nkb)``.
        weights: ``(nk, nbnd)``.

    Returns one complex ``(nat_t, nh_t, 2, nh_t, 2)`` array per species.
    """
    npwx = vkb.shape[-2]

    def one_k(arrays):
        projectors, state, occupation = arrays
        components = state.reshape(state.shape[:-1] + (2, npwx))
        projections = jnp.einsum("gc,bag->bac", projectors.conj(), components)
        return tuple(
            None if channels is None
            else _spinor_becsum_species(projections, occupation, channels)
            for channels in species_channels
        )

    return sum_k(one_k, (vkb, psi, weights), batch=resolve_k_batch(k_batch))


@jax.jit
def _spinor_becsum_species(projections, weights, channels):
    """One species' spinor ``becsum`` at one k-point, gathering its channels."""
    columns = projections[:, :, channels]  # (nbnd, 2, nat, nh)
    return jnp.einsum(
        "b,bani,bcnj->niajc",
        weights.astype(columns.dtype),
        columns.conj(),
        columns,
    )
