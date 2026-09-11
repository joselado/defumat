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

from defumat.basis.gvectors import GVectors, modulus
from defumat.pseudo.formfactors import (
    atomic_charge_of_g,
    core_charge_of_g,
    local_potential_of_g,
)
from defumat.pseudo.upf import Pseudopotential
from defumat.system.cell import Cell
from defumat.system.structure import Structure

__all__ = ["structure_factors", "local_potential", "local_potential_at_q",
           "starting_charge", "core_charge",
           "species_local_potential", "species_core_charge", "species_atomic_charge",
           "combine_species"]


def structure_factors(
    structure: Structure, cell: Cell, gvectors: GVectors
) -> jnp.ndarray:
    """``S_t(G)`` for every species, shaped ``(ntyp, ngm)``."""
    membership = _membership(structure)
    return _structure_factors_at(
        gvectors.cartesian(cell), structure.positions, membership
    )


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



def _weighted_structure_factors(structure, cell, gvectors, weights) -> jnp.ndarray:
    """``sum_{a in t} w_a e^{-iG.tau_a}`` for every species, ``(ntyp, ngm)``.

    :func:`structure_factors` with the atoms *counted* replaced by the atoms
    *weighted*, which is all a per-atom magnetization needs: the radial charge
    is still a property of the species, so only the phase sum changes. Contract
    this with the same per-species radial transforms and the result is
    ``sum_a w_a rho^at_{t(a)}(G) e^{-iG.tau_a}``.
    """
    # No dtype literal: ``_membership`` is the real mask and the product follows
    # it, so the phase sum's precision stays the one the G vectors and the
    # positions set rather than one written down here.
    membership = _membership(structure) * jnp.asarray(weights)[None, :]
    return _structure_factors_at(
        gvectors.cartesian(cell), structure.positions, membership
    )


def _sum_over_species(radial, structure, cell, gvectors) -> jnp.ndarray:
    """Combine per-species radial transforms with their structure factors.

    The radial transforms are evaluated first and stacked, so that the
    combination with the structure factors is a single compiled contraction
    instead of one dispatch per species.
    """
    gmod = _gmod(gvectors.cartesian(cell))
    values = tuple(radial(t, gmod) for t in range(structure.ntyp))
    return combine_species(values, structure, cell, gvectors)


def combine_species(values, structure: Structure, cell: Cell, gvectors: GVectors):
    """``sum_t f_t(|G|) S_t(G)``, given the per-species radial transforms.

    Separated from :func:`_sum_over_species` because the radial transforms do
    not depend on where the atoms are and the structure factors are all that
    does: a moved geometry reuses the tables and pays only for this contraction
    (:meth:`defumat.scf.driver.Calculation.at_positions`).
    """
    factors = structure_factors(structure, cell, gvectors)
    return _contract_species(tuple(values), factors)


def species_local_potential(
    pseudos: tuple[Pseudopotential, ...], cell: Cell, gvectors: GVectors
) -> tuple[jnp.ndarray, ...]:
    """``vloc_t(|G|)`` for each species, before any structure factor."""
    gmod = _gmod(gvectors.cartesian(cell))
    volume = cell.volume
    return tuple(local_potential_of_g(p, gmod, volume) for p in pseudos)


def species_atomic_charge(
    pseudos: tuple[Pseudopotential, ...], cell: Cell, gvectors: GVectors
) -> tuple[jnp.ndarray, ...]:
    """``rho_atomic_t(|G|)`` for each species, before any structure factor.

    The same radial transform :func:`starting_charge` sums into the SCF's
    starting guess, kept as a per-species table because two other things want
    it: nothing about it depends on where the atoms are, and QE tabulates it
    once for exactly that reason (``init_tab_rhoat`` in ``upflib/rhoat_mod.f90``,
    read back by ``interp_rhoat``). Evaluating it per *atom* instead costs the
    multiplicity of the species -- eight radial integrations over 36257
    G-vectors for one silicon species, which was 91% of the analytic force.
    """
    gmod = _gmod(gvectors.cartesian(cell))
    volume = cell.volume
    return tuple(atomic_charge_of_g(p, gmod, volume) for p in pseudos)


def species_core_charge(
    pseudos: tuple[Pseudopotential, ...], cell: Cell, gvectors: GVectors
) -> tuple[jnp.ndarray, ...] | None:
    """``rho_core_t(|G|)`` for each species, or ``None`` if no species has one."""
    if not any(p.has_nlcc for p in pseudos):
        return None
    gmod = _gmod(gvectors.cartesian(cell))
    volume = cell.volume
    return tuple(
        core_charge_of_g(p, gmod, volume) if p.has_nlcc else jnp.zeros_like(gmod)
        for p in pseudos
    )


def _gmod(g):
    """``|G|``, guarded at the origin -- see :func:`defumat.basis.gvectors.modulus`."""
    return modulus(g)


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
    volume = cell.volume
    return _sum_over_species(
        lambda t, gmod: local_potential_of_g(pseudos[t], gmod, volume),
        structure,
        cell,
        gvectors,
    )


def local_potential_at_q(
    pseudos: tuple[Pseudopotential, ...],
    structure: Structure,
    cell: Cell,
    gvectors: GVectors,
    q_cart,
) -> jnp.ndarray:
    """``sum_t vloc_t(|G+q|) sum_{a in t} e^{-i (G+q) . tau_a}``.

    :func:`local_potential` with the argument shifted, and it exists for one
    reason: **the derivative of this is the bare local perturbation of a phonon
    at** ``q``. Displace atom ``a`` by ``u e^{i q R}`` and the change in the
    local potential is ``e^{i q r}`` times a lattice-periodic function whose
    Fourier coefficients are

        dV_a(G) = -i (G+q)_alpha vloc_t(|G+q|) e^{-i (G+q) . tau_a} u_alpha

    which is exactly ``d/dtau_a`` of the expression above -- the phase carries
    ``G+q`` rather than ``G``, so differentiating it brings down ``-i(G+q)``
    and the radial table is already evaluated at the shifted argument. That is
    ``compute_dvloc.f90``'s ``vlocq(ig,nt) * gu * fact * gtau``, with its
    ``fact = tpiba (-i) eigqts(na)`` and ``gtau = e^{-i G . tau}`` recombined
    into the one phase ``e^{-i(G+q) . tau}``, and its ``gu = (xq+g) . u``.

    So the phonon's bare local term stays what ``PLAN.md`` P24 made it -- one
    ``jvp`` through the positions of code that builds a potential -- rather
    than a second, hand-derived expression. What ``q`` changes is the *code
    being differentiated*, not the way the derivative is taken.

    The radial table is QE's own ``init_vlocq``: ``vloc_of_g`` evaluated at
    ``|q+G|^2`` and nothing else, so ``q = 0`` returns :func:`local_potential`
    to round-off and that is the regression this pair is checked by.

    ``q_cart`` is in 1/bohr, like :meth:`defumat.basis.gvectors.GVectors.cartesian`.
    """
    g = gvectors.cartesian(cell) + jnp.asarray(q_cart)[None, :]
    gmod = modulus(g)
    volume = cell.volume
    values = tuple(
        local_potential_of_g(pseudos[t], gmod, volume)
        for t in range(structure.ntyp)
    )
    membership = _membership(structure)
    factors = _structure_factors_at(g, structure.positions, membership)
    return _contract_species(tuple(values), factors)


@jax.jit
def _structure_factors_at(g, positions, membership):
    """``S_t(G)`` on a G set handed in rather than derived from the cell.

    One function for both callers: :func:`structure_factors` passes ``G`` and
    :func:`local_potential_at_q` passes ``G+q``.
    """
    phases = jnp.exp(-1j * (g @ positions.T))  # (ngm, nat)
    return membership @ phases.T  # (ntyp, ngm)


def starting_charge(
    pseudos: tuple[Pseudopotential, ...],
    structure: Structure,
    cell: Cell,
    gvectors: GVectors,
    nelec: float | None = None,
    magnetization=None,
    per_atom=None,
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
    unpolarized.

    ``per_atom`` gives that weight **per atom** instead, which is what the
    ``STARTING_MOMENTS`` card asks for: two atoms of one species pointing
    opposite ways is an antiferromagnet, and no per-species number can say it.
    It costs nothing -- the radial transform is still per species, and the only
    change is that the structure factor each one is contracted with carries the
    atoms' weights instead of counting them. Given together with
    ``magnetization``, ``per_atom`` wins; that is the precedence the card is
    documented with. Both components are then scaled by the one factor that fixes
    the total charge, which is ``potinit``'s ``rho%of_g = rho%of_g/charge*nelec``
    applied to the whole array rather than to its first component: the *ratio*
    of magnetization to charge is what the input asked for and rescaling only
    one of them would change it.

    Returns ``rho(G)``, or ``(rho(G), m(G))`` when ``magnetization`` is given.
    """
    volume = cell.volume
    rho = _sum_over_species(
        lambda t, gmod: atomic_charge_of_g(pseudos[t], gmod, volume),
        structure,
        cell,
        gvectors,
    )

    if nelec is None:
        nelec = sum(pseudos[t].z_valence for t in structure.types)

    if magnetization is None and per_atom is None:
        return _renormalise(rho, cell.volume, nelec)

    if per_atom is not None:
        gmod = _gmod(gvectors.cartesian(cell))
        values = tuple(
            atomic_charge_of_g(pseudos[t], gmod, volume)
            for t in range(structure.ntyp)
        )
        polarized = _contract_species(
            values, _weighted_structure_factors(structure, cell, gvectors, per_atom)
        )
    else:
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

    volume = cell.volume

    def radial(t, gmod):
        if not pseudos[t].has_nlcc:
            return jnp.zeros_like(gmod)
        return core_charge_of_g(pseudos[t], gmod, volume)

    return _sum_over_species(radial, structure, cell, gvectors)
