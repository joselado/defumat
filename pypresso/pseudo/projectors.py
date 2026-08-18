"""Nonlocal pseudopotential projectors in the plane-wave basis.

The nonlocal part of the pseudopotential is a sum of separable terms,

    V_NL = sum_{a,ij} |beta_i^a> D_ij^a <beta_j^a|

and in the plane-wave basis each projector is

    <k+G| beta_i^a> = 4 pi / sqrt(Omega) * Y_lm(k+G) f_l(|k+G|) (-i)^l e^{-i(k+G).tau_a}

following ``upflib/init_us_2_acc.f90``. The three factors are the angular part,
the radial form factor of the previous module, and the structure factor placing
the projector on its atom.

**The whole expression is a differentiable function of k.** That is the point of
computing the form factors rather than interpolating them: for a nonlocal
pseudopotential the velocity operator is not ``p`` but involves ``[V_NL, r]``,
which QE hand-codes in ``commutator_Hx_psi.f90``. Here it will fall out of
``jacfwd`` of ``H(k)`` with respect to ``k`` (rule D2), provided nothing along
this path is a table lookup.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from pypresso.basis.gvectors import GVectors
from pypresso.basis.planewaves import PlaneWaveBasis
from pypresso.pseudo.formfactors import projector_form_factors
from pypresso.pseudo.harmonics import real_spherical_harmonics
from pypresso.pseudo.upf import Pseudopotential
from pypresso.system.cell import Cell
from pypresso.system.kpoints import KPoints
from pypresso.system.structure import Structure

__all__ = ["Projectors", "build_projectors", "projector_channels"]


def projector_channels(pseudo: Pseudopotential) -> list[tuple[int, int, int]]:
    """The ``(beta index, l, lm column)`` of every projector channel of a species.

    QE's ``indv`` / ``nhtol`` / ``nhtolm``: each radial projector ``beta_nb`` with
    angular momentum ``l`` contributes ``2l+1`` channels, one per ``m``, and the
    ``lm`` column indexes the spherical harmonics in the ordering of
    :mod:`pypresso.pseudo.harmonics`.
    """
    channels = []
    for nb, projector in enumerate(pseudo.projectors):
        l = projector.l
        for m in range(2 * l + 1):
            channels.append((nb, l, l * l + m))
    return channels


class Projectors(eqx.Module):
    """The projectors ``<k+G|beta>`` and their coefficients ``D``.

    ``vkb`` is ``(nk, npwx, nkb)``: for each k-point, every plane wave against
    every projector channel of every atom. ``dij`` is the ``(nkb, nkb)``
    coefficient matrix, block-diagonal over atoms.
    """

    vkb: jnp.ndarray  # (nk, npwx, nkb), complex
    dij: jnp.ndarray  # (nkb, nkb), Ry
    atom_of_channel: tuple[int, ...] = eqx.field(static=True)

    @property
    def nkb(self) -> int:
        return self.vkb.shape[-1]

    def project(self, psi: jnp.ndarray, ik: int) -> jnp.ndarray:
        """``<beta|psi>`` for wavefunctions ``psi`` of shape ``(..., npwx)``."""
        return jnp.einsum("...g,gk->...k", psi.conj(), self.vkb[ik]).conj()


def build_projectors(
    pseudos: tuple[Pseudopotential, ...],
    structure: Structure,
    cell: Cell,
    gvectors: GVectors,
    planewaves: PlaneWaveBasis,
    kpoints: KPoints,
) -> Projectors:
    """Assemble ``<k+G|beta>`` for every k-point, atom and channel."""
    channels_by_species = [projector_channels(p) for p in pseudos]
    nkb = sum(len(channels_by_species[t]) for t in structure.types)
    if nkb == 0:
        empty = jnp.zeros((kpoints.nk, planewaves.npwx, 0), dtype=cell.precision.complex)
        return Projectors(vkb=empty, dij=jnp.zeros((0, 0)), atom_of_channel=())

    lmax = max(p.lmax for p in pseudos)
    g = gvectors.cartesian(cell)[planewaves.indices]  # (nk, npwx, 3), 1/bohr
    kg = kpoints.cartesian(cell)[:, None, :] + g  # (nk, npwx, 3)
    kg_norm = jnp.sqrt(jnp.sum(kg**2, axis=-1))

    ylm = real_spherical_harmonics(kg, lmax)  # (nk, npwx, (lmax+1)^2)

    # Radial form factors, per species: (nbeta, nk * npwx) -> (nk, npwx, nbeta)
    shape = kg_norm.shape
    flat = kg_norm.reshape(-1)
    form_factors = [
        projector_form_factors(p, flat, float(cell.volume)).reshape((-1,) + shape).transpose(1, 2, 0)
        for p in pseudos
    ]

    tau = structure.positions
    columns, atom_of_channel, dij_blocks = [], [], []

    for atom, species in enumerate(structure.types):
        phase = jnp.exp(-1j * jnp.einsum("kgc,c->kg", kg, tau[atom]))  # (nk, npwx)
        pseudo = pseudos[species]
        for nb, l, lm in channels_by_species[species]:
            radial = form_factors[species][..., nb]
            # (-i)^l, the standard phase of a spherical-wave expansion.
            columns.append(((-1j) ** l) * ylm[..., lm] * radial * phase)
            atom_of_channel.append(atom)
        dij_blocks.append(_expand_dij(pseudo, channels_by_species[species]))

    vkb = jnp.stack(columns, axis=-1)
    vkb = jnp.where(planewaves.mask[..., None], vkb, 0.0)

    return Projectors(
        vkb=vkb.astype(cell.precision.complex),
        dij=jnp.asarray(_block_diagonal(dij_blocks)),
        atom_of_channel=tuple(atom_of_channel),
    )


def _expand_dij(pseudo: Pseudopotential, channels) -> np.ndarray:
    """``D`` in the channel basis: ``D_ij`` is diagonal in ``lm``, not in ``nb``.

    Two projectors of the same ``l`` (common in ultrasoft and multi-projector
    norm-conserving sets) couple through the off-diagonal ``D_ij``; different
    ``lm`` never couple, by rotational invariance.
    """
    n = len(channels)
    block = np.zeros((n, n))
    if pseudo.dij is None:
        return block
    for i, (nb_i, _, lm_i) in enumerate(channels):
        for j, (nb_j, _, lm_j) in enumerate(channels):
            if lm_i == lm_j:
                block[i, j] = pseudo.dij[nb_i, nb_j]
    return block


def _block_diagonal(blocks) -> np.ndarray:
    total = sum(b.shape[0] for b in blocks)
    out = np.zeros((total, total))
    offset = 0
    for block in blocks:
        n = block.shape[0]
        out[offset : offset + n, offset : offset + n] = block
        offset += n
    return out
