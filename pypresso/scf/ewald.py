"""Ewald summation: the electrostatic energy of the point ion cores.

The ion-ion interaction is a lattice sum of ``1/r`` terms, conditionally
convergent in both real and reciprocal space. Ewald's method splits it with a
Gaussian screening parameter ``alpha``: the short-ranged remainder is summed in
real space and the smooth part in reciprocal space, and the total is independent
of ``alpha``.

Transcribed from ``PW/src/ewald.f90``, including its choice of ``alpha`` (start
at 2.9 and reduce until the reciprocal-space truncation error is below 1e-7) and
its real-space cutoff of ``4/sqrt(alpha)``, at which the neglected terms are
``erfc(4) ~ 2e-8``. Matching those choices is not required for correctness --
the sum converges either way -- but it makes agreement with QE exact rather than
approximate, which is what makes the comparison a real test.

The result is one of the terms QE prints in its energy decomposition, so it can
be checked on its own long before there is an SCF to put it in.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erfc
from scipy.special import erfc as erfc_host

from pypresso.basis.gvectors import GVectors
from pypresso.system.cell import Cell, lattice_translations, pair_separation_bound
from pypresso.system.structure import Structure
from pypresso.units import E2, TPI

__all__ = ["ewald_energy", "ewald_alpha", "EwaldSum", "build_ewald"]

#: The neighbour-list helpers were general enough that a second pair sum wanted
#: them (:mod:`pypresso.vdw.grimme`), so they live in :mod:`pypresso.system.cell`
#: -- they are lattice geometry and know nothing about Ewald's split. Re-exported
#: under their old private names because this is where they were written.
_position_independent_radius = pair_separation_bound
_lattice_translations = lattice_translations


def ewald_alpha(charge: float, gcut: float, tpiba2: float, tolerance: float = 1.0e-7) -> float:
    """QE's choice of screening parameter: the largest that still converges.

    ``upperbound`` is a bound on the error made by truncating the reciprocal
    sum at ``gcut``; ``alpha`` is stepped down from 2.9 until it is small enough.
    """
    alpha = 2.9
    while alpha > 0.0:
        alpha -= 0.1
        if alpha <= 0.0:
            break
        upperbound = (
            2.0
            * charge**2
            * np.sqrt(2.0 * alpha / TPI)
            * float(erfc_host(np.sqrt(tpiba2 * gcut / 4.0 / alpha)))
        )
        if upperbound <= tolerance:
            return alpha
    raise ValueError("optimal Ewald alpha not found; is the G cutoff absurdly small?")


class EwaldSum(eqx.Module):
    """The Ewald sum with everything but the positions decided in advance.

    Two host-side quantities have to be fixed before the sum can be a
    differentiable function of the atomic positions: the screening parameter
    ``alpha``, which depends on the total charge and the G cutoff, and the list
    of lattice translations the real-space part runs over, which is integer
    bookkeeping over the cell. Both are settled here, once, so that
    :meth:`energy` is pure JAX and ``grad`` of it is the ionic part of the force.

    The translation list is built for the **whole cell** rather than for the
    separation of the atoms it was constructed with (see
    :func:`_position_independent_radius`), so the same object stays valid as the
    atoms move during a relaxation. That costs a larger neighbour list -- a few
    dozen translations rather than a few -- and buys a sum that cannot silently
    lose an image when an atom moves.
    """

    charges: jnp.ndarray  # (nat,), valence charge of each atom
    alpha: float = eqx.field(static=True)
    rmax: float = eqx.field(static=True)
    translations: jnp.ndarray  # (ntrans, 3), cartesian bohr
    gamma_factor: float = eqx.field(static=True)

    def energy(self, cell: Cell, positions: jnp.ndarray, gvectors: GVectors) -> jnp.ndarray:
        """Electrostatic energy of the ion cores at ``positions``, in Ry."""
        reciprocal = _reciprocal_kernel(
            gvectors.cartesian(cell), positions, self.charges,
            self.alpha, cell.volume, self.gamma_factor,
        )
        real = _real_kernel(
            positions, self.charges, self.translations, self.alpha, self.rmax
        )
        return 0.5 * E2 * (reciprocal + real)


def build_ewald(
    cell: Cell,
    structure: Structure,
    gvectors: GVectors,
    charges: np.ndarray,
) -> EwaldSum:
    """Fix ``alpha`` and the neighbour list for this cell and charge set."""
    charges = np.asarray(charges, dtype=float)
    if len(charges) != structure.nat:
        raise ValueError(f"{len(charges)} charges for {structure.nat} atoms")

    tpiba2 = cell.tpiba**2
    gcut = gvectors.ecut / tpiba2
    alpha = ewald_alpha(float(charges.sum()), gcut, tpiba2)
    rmax = 4.0 / np.sqrt(alpha)

    at = np.asarray(cell.at)
    radius = rmax + _position_independent_radius(at, np.asarray(structure.positions))
    return EwaldSum(
        charges=jnp.asarray(charges),
        alpha=float(alpha),
        rmax=float(rmax),
        translations=jnp.asarray(_lattice_translations(at, radius)),
        gamma_factor=2.0 if gvectors.gamma_only else 1.0,
    )


def ewald_energy(
    cell: Cell,
    structure: Structure,
    gvectors: GVectors,
    charges: np.ndarray,
) -> jnp.ndarray:
    """Electrostatic energy of the ion cores, in Ry.

    Args:
        cell: the unit cell.
        structure: atomic positions (cartesian, bohr).
        gvectors: the dense G-vector set, whose cutoff sets ``alpha``.
        charges: valence charge ``Z`` of each *atom* (not each species).
    """
    return build_ewald(cell, structure, gvectors, charges).energy(
        cell, structure.positions, gvectors
    )


@jax.jit
def _reciprocal_kernel(g, tau, charges, alpha, volume, factor):
    # rho(G) = sum_a Z_a conj(S_a(G)); only |rho|^2 is used.
    rho = jnp.sum(charges * jnp.exp(1j * (g @ tau.T)), axis=1)
    g2 = jnp.sum(g**2, axis=1)

    charge = jnp.sum(charges)
    total = -(charge**2) / alpha / 4.0

    # G = 0 is index 0 by construction and is excluded from the sum.
    nonzero = g2[1:]
    # ``Re(conj(rho) rho)`` and **not** ``abs(rho)**2``: ``abs``'s derivative is
    # ``Re(conj(z) dz)/|z|``, which is ``0/0`` where the structure factor
    # vanishes -- and in a cell that is a supercell it vanishes *exactly*, not
    # to round-off. The conventional cubic cell of fcc aluminium puts its atoms
    # at 0 and 1/2, so every phase is exactly +-1 and the fcc extinction rule
    # (mixed-parity Miller indices) is exact in floating point: **92 of its 3287
    # G-vectors have |rho| == 0.0**, where two-atom aluminium and diamond
    # silicon reach 4e-16 and never zero. The energy is right either way and the
    # *second* derivative is not -- ``jvp(grad)`` disagreed with a finite
    # difference of the same gradient by 3.0e-4 Ry/bohr^2 at that geometry while
    # agreeing to 3e-8 as soon as the atoms were displaced off it, and it was
    # worth 1.3 cm^-1 on the cell's phonon spectrum.
    #
    # This is :func:`pypresso.scf.density.band_density`'s trap, and
    # :func:`pypresso.basis.gvectors.modulus`'s, and
    # :mod:`pypresso.forces.energy`'s, in a **fourth** place. What is new here is
    # the way in: the other three are ``0/0`` at a node of a wavefunction, which
    # is a measure-zero accident, and this one is forced by the crystal's own
    # symmetry on every cell that is a supercell.
    total = total + factor * jnp.sum(
        jnp.real(rho[1:] * jnp.conj(rho[1:])) * jnp.exp(-nonzero / alpha / 4.0) / nonzero
    )

    total = 2.0 * TPI / volume * total

    # The self-interaction of each Gaussian, removed once per atom.
    return total - jnp.sum(charges**2) * jnp.sqrt(8.0 / TPI * alpha)


#: The real-space part sums ``erfc(sqrt(alpha) r)/r`` over neighbouring images.
#: The set of lattice translations is enumerated on the host -- it is integer
#: bookkeeping over a fixed cell, the definition of setup work -- and the sum
#: over ``(atom, atom, translation)`` is then one broadcast kernel rather than a
#: Python double loop over atom pairs. Beyond the speed, this is what makes the
#: term differentiable with respect to the atomic positions, which the forces
#: need: the neighbour list is a constant, the distances computed from it are not.


@jax.jit
def _real_kernel(tau, charges, translations, alpha, rmax):
    # (nat, nat, ntrans, 3): every pair, every image.
    separations = tau[:, None, None, :] - tau[None, :, None, :] + translations[None, None, :, :]
    square = jnp.sum(separations**2, axis=-1)

    # The self term (r = 0) and images past the cutoff are dropped by weight, not
    # by indexing, so the shape stays static. The *squared* distance is what is
    # sanitised, before the square root rather than after it: masking the result
    # of `sqrt(0)` still leaves an infinite derivative to be multiplied by zero,
    # and `0 * inf` is NaN. That NaN appears only in the gradient -- the energy
    # is correct either way -- so it is exactly the kind of thing that survives
    # until the day the forces are written.
    keep = (square > 1.0e-16) & (square <= rmax**2)
    distances = jnp.sqrt(jnp.where(keep, square, 1.0))
    terms = jnp.where(keep, erfc(jnp.sqrt(alpha) * distances) / distances, 0.0)

    pairs = charges[:, None] * charges[None, :]
    return jnp.sum(pairs * jnp.sum(terms, axis=-1))
