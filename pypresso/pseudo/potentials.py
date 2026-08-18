"""Species radial data summed onto the crystal: local potential and charges.

Each atomic quantity enters the crystal multiplied by its structure factor,

    V_loc(G) = sum_t vloc_t(|G|) S_t(G),    S_t(G) = sum_{a in t} e^{-i G . tau_a}

so this module is where per-species radial tables (``formfactors``) become
crystal quantities on the dense G grid. Following ``PW/src/setlocal.f90`` and
``upflib/rhoat_mod.f90``.
"""

from __future__ import annotations

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
    g = gvectors.cartesian(cell)  # (ngm, 3)
    phases = jnp.exp(-1j * (g @ structure.positions.T))  # (ngm, nat)

    types = np.asarray(structure.types)
    return jnp.stack(
        [jnp.sum(phases[:, types == t], axis=1) for t in range(structure.ntyp)], axis=0
    )


def _sum_over_species(radial, structure, cell, gvectors) -> jnp.ndarray:
    """Combine per-species radial transforms with their structure factors."""
    gmod = jnp.sqrt(gvectors.kinetic(cell))
    factors = structure_factors(structure, cell, gvectors)
    total = jnp.zeros(gvectors.ngm, dtype=cell.precision.complex)
    for t in range(structure.ntyp):
        total = total + radial(t, gmod) * factors[t]
    return total


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
) -> jnp.ndarray:
    """Superposition of atomic charges, ``rho(G)``, as the SCF starting guess.

    QE renormalises this to the exact electron count (``atomic_rho``): the
    tabulated atomic charges are integrated on a mesh truncated at 10 bohr, so
    their sum misses a small fraction of an electron. Without the rescaling the
    first SCF iteration starts from the wrong total charge.
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
    charge = jnp.real(rho[0]) * cell.volume  # rho(G=0) * Omega = electron count
    return rho * (nelec / charge)


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
