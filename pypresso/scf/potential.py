"""Building the self-consistent potential from the density.

``v_of_rho``: given the electron density, produce the Hartree and
exchange-correlation potentials and their energies. Following
``PW/src/v_of_rho.f90``, with the gradient correction of ``gradcorr.f90``.

The Hartree term is diagonal in G space -- ``V_H(G) = 4 pi e^2 rho(G) / G^2`` --
with the ``G = 0`` component set to zero. That divergence is not an error: it
cancels against the corresponding divergences in the Ewald sum and in the local
pseudopotential, and the three ``G = 0`` terms are only finite together, for a
neutral cell.

**The spin axis.** Densities and potentials here are ``(nspin, ...)`` and each
channel is that spin's own density -- so ``nspin = 1`` is the total density in a
single channel and needs no special case anywhere. QE instead stores the pair as
(total, magnetization) and converts back and forth (``rhoz_or_updw``); the two
conventions meet where its formulas are written in one or the other, which is
exactly three places: the Hartree term and ``dr2`` want the total, and the
exchange-correlation functional wants the total and ``zeta``.

The physics of the split is the whole of LSDA: **Hartree is a functional of the
total density alone** -- an electron does not care about the spin of the charge
repelling it -- while exchange-correlation is not, and that asymmetry is why the
two channels see different potentials at all.
"""

from __future__ import annotations

import equinox as eqx
import jax
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

    v_scf: jnp.ndarray  # (nspin, n1, n2, n3), Ry -- Hartree + XC
    ehart: jnp.ndarray  # Ry
    etxc: jnp.ndarray  # Ry

    @property
    def nspin(self) -> int:
        return self.v_scf.shape[0]


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

    With two spin channels ``rho_ddot`` gains a second piece, and it is not a
    second Hartree energy: the magnetization enters with a **G-independent**
    weight ``e2 4 pi / (2 pi)^2`` (QE's comment says ``lambda = 1 a.u.``) and
    with its ``G = 0`` component *included*, where the Hartree half excludes it.
    An error in the total charge is expensive in proportion to its wavelength; an
    error in the magnetization is expensive at every wavelength equally, and a
    uniform shift of the magnetization is a real error where a uniform shift of
    the charge is forbidden by neutrality.
    """
    residual_g = r_to_g(residual_r, gvectors.fft_index)
    total = hartree(jnp.sum(residual_g, axis=0), gvectors, cell)[1]
    if residual_g.shape[0] == 1:
        return total

    magnetization = residual_g[0] - residual_g[1]
    weight = E2 * FPI / (2.0 * jnp.pi) ** 2
    contribution = jnp.sum(jnp.abs(magnetization) ** 2)
    if gvectors.gamma_only:
        # Only half the sphere is stored, and unlike the Hartree half the G = 0
        # term is counted here -- so the doubling applies to the rest of it.
        contribution = 2.0 * contribution - jnp.abs(magnetization[0]) ** 2
    return total + 0.5 * cell.volume * weight * contribution


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
    nspin = valence.shape[0]
    # The core charge is unpolarized, so it is shared equally between the
    # channels -- which is the same thing as adding all of it to the total and
    # none of it to the magnetization, the form ``v_xc`` writes it in.
    density = (
        valence if rho_core is None else valence + jnp.real(rho_core) / nspin
    )
    n = density[0].size

    if nspin == 1:
        v = functional.potential(density[0])[None]
        energy = (
            cell.volume / n
            * jnp.sum(density[0] * functional.energy_density(density[0]))
        )
        return v, energy

    v = functional.spin_potential(density)
    energy = (
        cell.volume
        / n
        * jnp.sum(jnp.sum(density, axis=0) * functional.spin_energy_density(density))
    )
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

    With two channels there is a third term. Correlation depends on the
    **total** density's gradient, so ``d e / d(grad rho_up)`` picks up
    ``grad rho_dw`` as well; QE calls that cross term ``v2c_ud`` and adds it by
    hand. Here ``h`` is the derivative of the energy with respect to the
    gradient *field* rather than with respect to ``|grad rho|^2``, so the cross
    term is simply part of it and its pairing with the two channels cannot be
    got the wrong way round.

    Args:
        density_r: ``rho + rho_core/nspin`` on the grid, ``(nspin, ...)`` -- the
            density the functional sees, as in :func:`exchange_correlation`.
        density_g: the same density on the G-vector sphere. Passed in rather
            than transformed here because the caller already holds ``rho(G)``
            for the Hartree term, and the core charge's own G components come
            straight from the pseudopotential.

    Returns ``(v, energy)`` in Ry and Ry/bohr^3 respectively -- the energy
    already integrated over the cell, so that it adds to ``etxc``.
    """
    nspin = density_r.shape[0]
    grad = jax.vmap(gradient, in_axes=(0, None, None))(density_g, gvectors, cell)
    n = density_r[0].size

    if nspin == 1:
        sigma = jnp.sum(grad[0] * grad[0], axis=0)
        v1, v2 = functional.gradient_potentials(density_r[0], sigma)
        v = v1 - divergence(v2[None, ...] * grad[0], gvectors, cell)
        energy = (
            cell.volume / n * jnp.sum(functional.gradient_energy(density_r[0], sigma))
        )
        return v[None], energy

    v1, h = functional.spin_gradient_terms(density_r, grad)
    v = v1 - jax.vmap(divergence, in_axes=(0, None, None))(h, gvectors, cell)
    energy = (
        cell.volume / n * jnp.sum(functional.spin_gradient_energy(density_r, grad))
    )
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
    rho_r = jnp.asarray(rho_r)
    if rho_r.ndim == 3:
        # A bare grid means one spin channel. Accepted so that callers holding a
        # plain density -- the tests, and the band-structure workflow -- do not
        # have to know about the axis.
        rho_r = rho_r[None]
    nspin = rho_r.shape[0]
    rho_g = jax.vmap(r_to_g, in_axes=(0, None))(rho_r, gvectors.fft_index)

    # The Hartree term sees the total charge and nothing else, so it is the same
    # potential in both channels -- ``v_h`` is called on ``rho%of_g(:,1)`` and
    # added to every component of ``v%of_r``.
    v_hartree_g, ehart = hartree(jnp.sum(rho_g, axis=0), gvectors, cell)
    v_hartree_r = jnp.real(g_to_r(v_hartree_g, gvectors.fft_index, gvectors.grid))

    v_xc, etxc = exchange_correlation(rho_r, cell, rho_core, functional)

    if functional.is_gradient:
        density_r = jnp.real(rho_r)
        density_g = rho_g
        if rho_core is not None:
            density_r = density_r + jnp.real(rho_core) / nspin
            core_g = (
                r_to_g(jnp.real(rho_core), gvectors.fft_index)
                if rho_core_g is None
                else rho_core_g
            )
            density_g = density_g + core_g[None] / nspin
        v_gradient, e_gradient = gradient_correction(
            density_r, density_g, gvectors, cell, functional
        )
        v_xc = v_xc + v_gradient
        etxc = etxc + e_gradient

    return Potential(v_scf=v_hartree_r[None] + v_xc, ehart=ehart, etxc=etxc)
