"""Building the self-consistent potential from the density.

``v_of_rho``: given the electron density, produce the Hartree and
exchange-correlation potentials and their energies. Following
``PW/src/v_of_rho.f90``.

The Hartree term is diagonal in G space -- ``V_H(G) = 4 pi e^2 rho(G) / G^2`` --
with the ``G = 0`` component set to zero. That divergence is not an error: it
cancels against the corresponding divergences in the Ewald sum and in the local
pseudopotential, and the three ``G = 0`` terms are only finite together, for a
neutral cell.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from pypresso.basis.fft import g_to_r, r_to_g
from pypresso.basis.gvectors import GVectors
from pypresso.system.cell import Cell
from pypresso.units import E2, FPI
from pypresso.xc.lda import xc_energy_density, xc_potential

__all__ = ["Potential", "v_of_rho", "hartree", "exchange_correlation", "scf_accuracy"]


class Potential(eqx.Module):
    """The self-consistent potential and the energies that go with it.

    ``v_scf`` excludes the local pseudopotential: that part is fixed, while this
    is what changes from iteration to iteration and what ``deband`` subtracts.
    """

    v_scf: jnp.ndarray  # (n1, n2, n3), Ry -- Hartree + XC
    ehart: jnp.ndarray  # Ry
    etxc: jnp.ndarray  # Ry


def hartree(rho_g: jnp.ndarray, gvectors: GVectors, cell: Cell):
    """Hartree potential in G space and its energy.

    Returns ``(v_hartree(G), E_hartree)``.
    """
    g2 = gvectors.kinetic(cell)  # |G|^2 in 1/bohr^2
    inverse = jnp.where(g2 > 1e-12, 1.0 / jnp.where(g2 > 1e-12, g2, 1.0), 0.0)

    v = E2 * FPI * rho_g * inverse
    energy = 0.5 * cell.volume * E2 * FPI * jnp.sum(jnp.abs(rho_g) ** 2 * inverse)
    if gvectors.gamma_only:
        # Only half the sphere is stored; every G != 0 stands for a +-G pair.
        energy = 2.0 * energy
    return v, energy


def scf_accuracy(residual_r: jnp.ndarray, gvectors: GVectors, cell: Cell) -> jnp.ndarray:
    """QE's ``dr2``: how far from self-consistency the density still is, in Ry.

    ``rho_ddot`` in ``PW/src/scf_mod.f90``, evaluated on the residual with
    itself. It is the *Hartree energy of the density error* -- literally the same
    expression as :func:`hartree`'s energy, applied to ``rho_out - rho_in`` --
    which is why it is an estimate of the error in the total energy rather than
    just a size of the density change: it weights long-wavelength errors by
    ``1/G^2``, and those are the ones that cost energy.

    This is the quantity QE compares against ``conv_thr`` and the quantity its
    diagonalisation threshold is scheduled from, so both of those now mean the
    same thing here as they do there.
    """
    residual_g = r_to_g(residual_r, gvectors.fft_index)
    return hartree(residual_g, gvectors, cell)[1]


def exchange_correlation(rho_r: jnp.ndarray, cell: Cell):
    """XC potential on the grid and its energy."""
    density = jnp.real(rho_r)
    v = xc_potential(density)
    n = density.size
    energy = cell.volume / n * jnp.sum(density * xc_energy_density(density))
    return v, energy


def v_of_rho(rho_r: jnp.ndarray, gvectors: GVectors, cell: Cell) -> Potential:
    """The full self-consistent potential from a real-space density."""
    rho_g = r_to_g(rho_r, gvectors.fft_index)

    v_hartree_g, ehart = hartree(rho_g, gvectors, cell)
    v_hartree_r = jnp.real(g_to_r(v_hartree_g, gvectors.fft_index, gvectors.grid))

    v_xc, etxc = exchange_correlation(rho_r, cell)

    return Potential(v_scf=v_hartree_r + v_xc, ehart=ehart, etxc=etxc)
