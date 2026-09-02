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

from defumat.basis.fft import g_to_r
from defumat.batching import resolve_k_batch, sum_bands, sum_k
from defumat.system.cell import Cell

__all__ = ["sum_band", "band_density", "becsum", "spinor_sum_band",
           "spinor_band_density", "spinor_becsum",
           "band_kinetic_density", "kinetic_energy_density",
           "spinor_band_kinetic_density", "spinor_kinetic_energy_density"]


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
        field = g_to_r(state, fft_index, grid)
        # ``|psi|^2`` as ``Re(conj(psi) psi)`` and not as ``abs(psi)**2``. The two
        # are the same number to a rounding, and only one of them is
        # differentiable: ``abs``'s derivative is ``Re(conj z t)/|z|``, which is
        # ``0/0`` wherever the field vanishes, and a wavefunction has nodes *on
        # grid points* by symmetry. One of them poisons the whole density with
        # NaN. It is :func:`defumat.basis.gvectors.modulus`'s trap in a second
        # place, and it is only reachable once the density is differentiated with
        # respect to the *states* -- which is what a linear response does
        # (:mod:`defumat.response.sternheimer`).
        return weight * jnp.real(jnp.conj(field) * field)

    # One band at a time, as ``sum_band.f90`` accumulates them: a band's
    # real-space box is the working set (:mod:`defumat.batching`).
    return sum_bands(one_band, (psi, weights)) / cell.volume


def sum_band(psi, fft_index, grid, weights, cell: Cell,
             k_batch: int | None | str = "default") -> jnp.ndarray:
    """The density from every k-point, ``(nspin, n1, n2, n3)`` and real.

    Args:
        psi: ``(nspin, nk, nbnd, npwx)``.
        fft_index: ``(nk, npwx)`` -- shared by the channels, since the plane-wave
            basis at a k-point does not depend on spin.
        weights: ``(nspin, nk, nbnd)`` occupation weights.

    Accumulated ``k_batch`` k-points at a time (:mod:`defumat.batching`), which
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


def band_kinetic_density(psi, fft_index, grid, weights, cell: Cell, kplusg):
    """One k-point's contribution to ``tau``, in **Rydberg**.

    Args:
        psi: ``(nbnd, npwx)``.
        kplusg: ``(npwx, 3)`` in 1/bohr, zero on padding.

    ``tau(r) = sum_i w_i |grad psi_i(r)|^2`` -- three more transforms per band
    than the density costs, since ``grad psi`` has to be built one cartesian
    direction at a time in G space (``i(k+G) c_G``) and brought back. This is
    ``sum_band.f90``'s meta-GGA branch exactly: it forms ``kplusgi *
    evc(i,ibnd)`` for ``j = 1, 3`` and calls ``get_rho`` on each, accumulating
    into ``rho%kin_r`` with the *same* weights the density uses.

    **Rydberg, and no factor of one half.** QE's ``rho%kin_r`` is this sum as
    written and ``v_xc_meta`` divides it by ``e2`` to get the Hartree ``tau``
    the functionals want; the same division happens here, in
    :func:`defumat.scf.potential.meta_exchange`, so that what flows through the
    SCF is in the package's units and only the functional sees Hartree.

    ``|grad psi|^2`` is ``Re(conj z z)`` summed over the three directions and
    not ``abs(z)**2``, for :func:`band_density`'s reason and with more of an
    edge: a *derivative* of a state has nodes wherever the state has extrema,
    which on a symmetric cell is a great many grid points exactly.
    """
    def one_band(arrays):
        state, weight = arrays
        components = 1j * kplusg.T * state[None, :]  # (3, npwx)
        field = g_to_r(components, fft_index, grid)  # (3, n1, n2, n3)
        return weight * jnp.sum(jnp.real(jnp.conj(field) * field), axis=0)

    return sum_bands(one_band, (psi, weights)) / cell.volume


def kinetic_energy_density(psi, fft_index, grid, weights, cell: Cell, kplusg,
                           k_batch: int | None | str = "default") -> jnp.ndarray:
    """``tau`` from every k-point, ``(nspin, n1, n2, n3)`` and real, Ry.

    Args:
        psi: ``(nspin, nk, nbnd, npwx)``.
        kplusg: ``(nk, npwx, 3)``.

    The channel axis means what it means in the density -- QE's
    ``rho%kin_r(:, current_spin)``, and this package's ``(up, down)`` for
    ``nspin = 2``. **QE's own storage differs from its density's there**, since
    ``sum_band`` converts ``rho`` to ``(total, magnetization)`` at the end and
    leaves ``kin_r`` alone; ``potinit.f90`` says so in a comment ("for LSDA rho
    is (tot,magn), rho_kin is (up,down)"). Here the two agree, so nothing
    converts between them -- and a transcribed conversion would be a bug.
    """
    batch = resolve_k_batch(k_batch)

    def channel(states, occupations):
        def one_k(arrays):
            state, index, vectors, occupation = arrays
            return band_kinetic_density(state, index, grid, occupation, cell, vectors)

        return sum_k(one_k, (states, fft_index, kplusg, occupations), batch=batch)

    return jax.vmap(channel)(psi, weights)


def spinor_band_kinetic_density(psi, fft_index, grid, weights, cell: Cell,
                                kplusg, nspin_mag: int):
    """One k-point's contribution to a **noncollinear** ``tau``, Ry.

    Args:
        psi: ``(nbnd, 2 npwx)`` spinors, the two components stored one after
            the other, as :func:`spinor_band_density` takes them.
        kplusg: ``(npwx, 3)`` in 1/bohr.
        nspin_mag: 1 for the trace alone, 4 for ``(tau, tau_x, tau_y, tau_z)``.

    **The kinetic energy density of a spinor is a 2x2 matrix**, not a number and
    not two numbers:

        tau_ab(r) = sum_i w_i grad psi_ia^* . grad psi_ib,

    and it decomposes on the Pauli basis exactly as the density does -- a trace
    and an axial three-vector. This is :func:`spinor_band_density` with
    ``grad psi`` in place of ``psi``, and the *only* difference beyond that is
    that every product is a dot product over the three cartesian directions
    before the spin algebra happens. Getting that order wrong -- taking the spin
    structure of each direction and summing afterwards -- gives the same trace
    and a different vector part.

    The sign conventions are the density's, so that the two arrays can be
    resolved onto the same local axis: ``m = psi^dagger sigma psi`` with
    ``cross = conj(up) down``, ``m_x = 2 Re(cross)``, ``m_y = 2 Im(cross)``.
    """
    npwx = psi.shape[-1] // 2
    fft_index = jnp.asarray(fft_index)
    if fft_index.ndim != 1:
        raise NotImplementedError(
            "a spin spiral's two spinor components live on different "
            "plane-wave spheres, so their gradients do not add to a "
            "lattice-periodic tau; meta-GGA with spiral_q is refused"
        )

    def one_band(arrays):
        state, weight = arrays
        components = state.reshape((2, npwx))
        # ``(2, 3, npwx)`` -> ``(2, 3, n1, n2, n3)``: the spin component, then
        # the cartesian direction of the gradient.
        gradients = 1j * kplusg.T[None, :, :] * components[:, None, :]
        field = g_to_r(gradients, fft_index, grid)
        up, down = field[0], field[1]

        # Summed over the cartesian axis *first*: these are dot products of
        # gradients, and the spin algebra acts on the result.
        up_density = jnp.sum(jnp.real(jnp.conj(up) * up), axis=0)
        down_density = jnp.sum(jnp.real(jnp.conj(down) * down), axis=0)
        trace = up_density + down_density
        if nspin_mag == 1:
            return weight * trace[None]
        cross = jnp.sum(jnp.conj(up) * down, axis=0)
        return weight * jnp.stack([
            trace,
            2.0 * jnp.real(cross),
            2.0 * jnp.imag(cross),
            up_density - down_density,
        ])

    return sum_bands(one_band, (psi, weights)) / cell.volume


def spinor_kinetic_energy_density(psi, fft_index, grid, weights, cell: Cell,
                                  kplusg, nspin_mag: int,
                                  k_batch: int | None | str = "default"):
    """Noncollinear ``tau`` from every k-point, ``(nspin_mag, n1, n2, n3)``, Ry.

    Args:
        psi: ``(nk, nbnd, 2 npwx)``.
        kplusg: ``(nk, npwx, 3)``.
        weights: ``(nk, nbnd)``.
    """
    def one_k(arrays):
        state, index, vectors, occupation = arrays
        return spinor_band_kinetic_density(
            state, index, grid, occupation, cell, vectors, nspin_mag
        )

    return sum_k(one_k, (psi, fft_index, kplusg, weights),
                 batch=resolve_k_batch(k_batch))


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
            the other, as :class:`~defumat.hamiltonian.noncollinear.SpinorHamiltonian`
            holds them.
        nspin_mag: 1 for the charge alone, 4 for ``(n, m_x, m_y, m_z)``.

    ``get_rho_k`` and ``get_rho_domag`` in ``sum_band.f90``. The charge is the
    sum of the two components' densities and the magnetization is the Pauli
    expectation value ``psi^dagger sigma psi`` -- so the *same* wavefunctions
    give one number or four depending only on whether the calculation carries a
    magnetization, and a nonmagnetic spin-orbit run keeps a scalar density.

    ``fft_index`` is ``(npwx,)`` normally and ``(2, npwx)`` for a spin spiral,
    whose components live on different spheres. In the spiral case what this
    returns is the **rotated-frame** density: the charge and ``m_z`` are what
    they always were, and the transverse pair ``(m_x, m_y)`` is measured in the
    frame that turns with the spiral, which is the frame the potential is built
    in. The laboratory-frame spiral is recovered on output and nowhere else.
    """
    npwx = psi.shape[-1] // 2
    components = psi.reshape(psi.shape[:-1] + (2, npwx))
    fft_index = jnp.asarray(fft_index)
    if fft_index.ndim == 1:
        field = g_to_r(components, fft_index, grid)  # (nbnd, 2, n1, n2, n3)
    else:
        # A spin spiral: the two components are on different spheres, so they
        # are transformed with different index maps. What comes back is the pair
        # of *periodic* parts ``U_up``, ``U_dn``, and every quantity below is
        # built from those -- which is the point of the generalized Bloch
        # theorem: the density it produces is lattice periodic even though the
        # magnetization it describes turns from cell to cell.
        field = jnp.stack(
            [g_to_r(components[:, spin], fft_index[spin], grid) for spin in range(2)],
            axis=1,
        )
    up, down = field[:, 0], field[:, 1]

    # ``Re(conj(z) z)`` rather than ``abs(z)**2``, for the reason in
    # :func:`band_density`: the two are the same number and only one has a
    # derivative at a node.
    up_density = jnp.real(jnp.conj(up) * up)
    down_density = jnp.real(jnp.conj(down) * down)
    charge = up_density + down_density
    if nspin_mag == 1:
        stacked = charge[None]
    else:
        cross = jnp.conj(up) * down
        stacked = jnp.stack([
            charge,
            2.0 * jnp.real(cross),
            2.0 * jnp.imag(cross),
            up_density - down_density,
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
    (:meth:`defumat.pseudo.spinorbit.SpinOrbitCoupling.becsum_so`). Keeping the
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
