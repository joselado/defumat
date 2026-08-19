"""Building the self-consistent potential from the density.

``v_of_rho``: given the electron density, produce the Hartree and
exchange-correlation potentials and their energies. Following
``PW/src/v_of_rho.f90``, with the gradient correction of ``gradcorr.f90``.

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
from pypresso.basis.gradients import divergence, gradient
from pypresso.basis.gvectors import GVectors
from pypresso.system.cell import Cell
from pypresso.units import E2, FPI
from pypresso.xc.functional import Functional, get_functional

__all__ = ["Potential", "v_of_rho", "hartree", "exchange_correlation",
           "gradient_correction", "scf_accuracy", "DEFAULT_FUNCTIONAL"]

#: What a calculation uses when nothing names a functional. QE has no such
#: default -- it takes the functional from the pseudopotentials and stops if
#: they disagree -- so this is only for callers that construct a potential
#: directly, and every path through the SCF driver passes one explicitly.
DEFAULT_FUNCTIONAL = "PZ"


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


def exchange_correlation(
    rho_r: jnp.ndarray,
    cell: Cell,
    rho_core: jnp.ndarray | None = None,
    functional: Functional | None = None,
):
    """The local part of the XC potential on the grid, and its energy.

    ``rho_core`` is the nonlinear core correction (``PP_NLCC``): the frozen core
    charge a pseudopotential leaves out of the valence density but which the
    exchange-correlation functional is nonlinear in, so that evaluating ``e_xc``
    at the valence density alone is wrong wherever the core overlaps the
    valence. QE's ``v_xc`` handles it in the way transcribed here, and the two
    halves are deliberately asymmetric (``PW/src/v_of_rho.f90``):

    * the functional -- both ``e_xc`` and ``v_xc`` -- is evaluated at the
      **total** density ``rho + rho_core``;
    * the energy integrates ``(rho + rho_core) e_xc`` over the total density,
      but ``vtxc`` and hence ``deband`` integrate ``v_xc`` against the
      **valence** density only, because it is only the valence density that the
      SCF varies.

    Getting the second point backwards leaves the total energy self-consistent
    and wrong, which is the same failure mode as the ``vloc`` ``G = 0`` trap.

    With a gradient-corrected functional this is only half the story: QE calls
    ``v_xc`` for the local part and then ``gradcorr`` for the rest, and so does
    :func:`v_of_rho`.
    """
    functional = functional or get_functional(DEFAULT_FUNCTIONAL)
    valence = jnp.real(rho_r)
    density = valence if rho_core is None else valence + jnp.real(rho_core)
    v = functional.potential(density)
    n = density.size
    energy = cell.volume / n * jnp.sum(density * functional.energy_density(density))
    return v, energy


def gradient_correction(
    density_r: jnp.ndarray,
    density_g: jnp.ndarray,
    gvectors: GVectors,
    cell: Cell,
    functional: Functional,
):
    """``gradcorr``: what a GGA adds to the potential and to the energy.

    The functional derivative of an energy that depends on ``grad rho`` has two
    terms,

        v = d e / d rho  -  div ( d e / d grad rho )
          = v1           -  div ( v2 grad rho ),

    and the second is why a gradient-corrected potential costs four more
    transforms per iteration than a local one: three to build ``grad rho`` and
    one to take the divergence back. QE assembles exactly this, storing the
    vector field ``h = v2 grad rho`` and calling ``fft_graddot`` on it.

    Args:
        density_r: ``rho + rho_core`` on the grid -- the density the functional
            sees, as in :func:`exchange_correlation`.
        density_g: the same density on the G-vector sphere. Passed in rather
            than transformed here because the caller already holds ``rho(G)``
            for the Hartree term, and the core charge's own G components come
            straight from the pseudopotential.

    Returns ``(v, energy)`` in Ry and Ry/bohr^3 respectively -- the energy
    already integrated over the cell, so that it adds to ``etxc``.
    """
    grad = gradient(density_g, gvectors, cell)  # (3, n1, n2, n3)
    sigma = jnp.sum(grad * grad, axis=0)

    v1, v2 = functional.gradient_potentials(density_r, sigma)
    v = v1 - divergence(v2[None, ...] * grad, gvectors, cell)

    n = density_r.size
    energy = cell.volume / n * jnp.sum(functional.gradient_energy(density_r, sigma))
    return v, energy


def v_of_rho(
    rho_r: jnp.ndarray,
    gvectors: GVectors,
    cell: Cell,
    rho_core: jnp.ndarray | None = None,
    functional: Functional | None = None,
    rho_core_g: jnp.ndarray | None = None,
) -> Potential:
    """The full self-consistent potential from a real-space density.

    The Hartree term sees the valence density alone -- the core charge is a
    device for the exchange-correlation functional and carries no Hartree
    energy, since the pseudopotential already contains the core's electrostatics.

    ``rho_core_g`` is the core charge on the G-vector sphere, needed only by a
    gradient-corrected functional: its gradient is taken in G space along with
    the valence density's, which is why ``set_rhoc`` keeps ``rhog_core`` around
    rather than only its transform.
    """
    functional = functional or get_functional(DEFAULT_FUNCTIONAL)
    rho_g = r_to_g(rho_r, gvectors.fft_index)

    v_hartree_g, ehart = hartree(rho_g, gvectors, cell)
    v_hartree_r = jnp.real(g_to_r(v_hartree_g, gvectors.fft_index, gvectors.grid))

    v_xc, etxc = exchange_correlation(rho_r, cell, rho_core, functional)

    if functional.is_gradient:
        density_r = jnp.real(rho_r)
        density_g = rho_g
        if rho_core is not None:
            density_r = density_r + jnp.real(rho_core)
            density_g = density_g + (
                r_to_g(jnp.real(rho_core), gvectors.fft_index)
                if rho_core_g is None
                else rho_core_g
            )
        v_gradient, e_gradient = gradient_correction(
            density_r, density_g, gvectors, cell, functional
        )
        v_xc = v_xc + v_gradient
        etxc = etxc + e_gradient

    return Potential(v_scf=v_hartree_r + v_xc, ehart=ehart, etxc=etxc)
