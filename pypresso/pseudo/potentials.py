"""Species radial data summed onto the crystal: local potential and charges.

Each atomic quantity enters the crystal multiplied by its structure factor,

    V_loc(G) = sum_t vloc_t(|G|) S_t(G),    S_t(G) = sum_{a in t} e^{-i G . tau_a}

so this module is where per-species radial tables (``formfactors``) become
crystal quantities on the dense G grid. Following ``PW/src/setlocal.f90`` and
``upflib/rhoat_mod.f90``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from pypresso.basis.gvectors import GVectors
from pypresso.pseudo.formfactors import (
    atomic_charge_of_g,
    core_charge_of_g,
    local_potential_of_g,
)
from pypresso.pseudo.upf import Pseudopotential
from pypresso.system.cell import Cell
from pypresso.system.structure import Structure

__all__ = ["structure_factors", "local_potential", "starting_charge", "core_charge"]


def structure_factors(
    structure: Structure, cell: Cell, gvectors: GVectors
) -> jnp.ndarray:
    """``S_t(G)`` for every species, shaped ``(ntyp, ngm)``."""
    membership = _membership(structure)
    return _structure_factors(gvectors.cartesian(cell), structure.positions, membership)


def _membership(structure: Structure) -> jnp.ndarray:
    """``(ntyp, nat)``, one where the atom belongs to the species.

    A matrix product against this sums the per-atom phases into per-species
    structure factors in one operation, rather than one boolean gather and one
    reduction per species. It is also what keeps the result a smooth function of
    the positions, since nothing about it depends on their values.
    """
    types = np.asarray(structure.types)
    return jnp.asarray(
        np.equal(types[None, :], np.arange(structure.ntyp)[:, None]).astype(float)
    )


@jax.jit
def _structure_factors(g, positions, membership):
    phases = jnp.exp(-1j * (g @ positions.T))  # (ngm, nat)
    return membership @ phases.T  # (ntyp, ngm)


def _sum_over_species(radial, structure, cell, gvectors) -> jnp.ndarray:
    """Combine per-species radial transforms with their structure factors.

    The radial transforms are evaluated first and stacked, so that the
    combination with the structure factors is a single compiled contraction
    instead of one dispatch per species.
    """
    gmod = _gmod(gvectors.cartesian(cell))
    factors = structure_factors(structure, cell, gvectors)
    values = tuple(radial(t, gmod) for t in range(structure.ntyp))
    return _contract_species(values, factors)


@jax.jit
def _gmod(g):
    return jnp.sqrt(jnp.sum(g**2, axis=1))


@jax.jit
def _contract_species(values, factors):
    """``sum_t f_t(|G|) S_t(G)``. ``values`` arrives as a tuple, and stacking it
    inside the compiled unit keeps the stack from being a dispatch of its own."""
    return jnp.sum(jnp.stack(values, axis=0).astype(factors.dtype) * factors, axis=0)


def local_potential(
    pseudos: tuple[Pseudopotential, ...],
    structure: Structure,
    cell: Cell,
    gvectors: GVectors,
) -> jnp.ndarray:
    """The local pseudopotential on the dense grid, ``V_loc(G)`` in Ry."""
    volume = float(cell.volume)
    return _sum_over_species(
        lambda t, gmod: local_potential_of_g(pseudos[t], gmod, volume),
        structure,
        cell,
        gvectors,
    )


def starting_charge(
    pseudos: tuple[Pseudopotential, ...],
    structure: Structure,
    cell: Cell,
    gvectors: GVectors,
    nelec: float | None = None,
    magnetization=None,
):
    """Superposition of atomic charges, ``rho(G)``, as the SCF starting guess.

    QE renormalises this to the exact electron count (``atomic_rho``): the
    tabulated atomic charges are integrated on a mesh truncated at 10 bohr, so
    their sum misses a small fraction of an electron. Without the rescaling the
    first SCF iteration starts from the wrong total charge.

    ``magnetization`` -- ``starting_magnetization`` per species -- asks for the
    LSDA pair. ``atomic_rho_g`` builds the second component from the *same*
    radial charges weighted by each species' value, so an atom with
    ``starting_magnetization = 1`` starts fully polarized and one with 0 starts
    unpolarized. Both components are then scaled by the one factor that fixes
    the total charge, which is ``potinit``'s ``rho%of_g = rho%of_g/charge*nelec``
    applied to the whole array rather than to its first component: the *ratio*
    of magnetization to charge is what the input asked for and rescaling only
    one of them would change it.

    Returns ``rho(G)``, or ``(rho(G), m(G))`` when ``magnetization`` is given.
    """
    volume = float(cell.volume)
    rho = _sum_over_species(
        lambda t, gmod: atomic_charge_of_g(pseudos[t], gmod, volume),
        structure,
        cell,
        gvectors,
    )

    if nelec is None:
        nelec = sum(pseudos[t].z_valence for t in structure.types)

    if magnetization is None:
        return _renormalise(rho, cell.volume, nelec)

    weights = jnp.asarray(magnetization, dtype=rho.real.dtype)
    polarized = _sum_over_species(
        lambda t, gmod: weights[t] * atomic_charge_of_g(pseudos[t], gmod, volume),
        structure,
        cell,
        gvectors,
    )
    scale = _renormalisation(rho, cell.volume, nelec)
    return rho * scale, polarized * scale


@jax.jit
def _renormalisation(rho, volume, nelec):
    charge = jnp.real(rho[0]) * volume  # rho(G=0) * Omega = electron count
    return nelec / charge


@jax.jit
def _renormalise(rho, volume, nelec):
    return rho * _renormalisation(rho, volume, nelec)


def core_charge(
    pseudos: tuple[Pseudopotential, ...],
    structure: Structure,
    cell: Cell,
    gvectors: GVectors,
) -> jnp.ndarray | None:
    """The nonlinear core-correction charge, or ``None`` if no species has one."""
    if not any(p.has_nlcc for p in pseudos):
        return None

    volume = float(cell.volume)

    def radial(t, gmod):
        if not pseudos[t].has_nlcc:
            return jnp.zeros_like(gmod)
        return core_charge_of_g(pseudos[t], gmod, volume)

    return _sum_over_species(radial, structure, cell, gvectors)
