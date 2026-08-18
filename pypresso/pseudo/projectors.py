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

from functools import partial

import equinox as eqx
import jax
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
    kg, kg_norm, ylm = _angular_part(
        gvectors.cartesian(cell), planewaves.indices, kpoints.cartesian(cell), lmax
    )

    # Radial form factors, per species: (nbeta, nk * npwx) -> (nk, npwx, nbeta),
    # concatenated over species so that a channel selects a column by one index.
    shape = kg_norm.shape
    flat = kg_norm.reshape(-1)
    form_factors = tuple(
        projector_form_factors(p, flat, float(cell.volume)) for p in pseudos
    )
    radial = _radial_table(form_factors, shape)
    beta_offset = np.cumsum([0] + [f.shape[0] for f in form_factors])

    # One row per projector channel, in QE's order: atoms outermost, then the
    # channels of that atom's species. Everything the assembly needs is an index
    # into an already-computed array, so it is three gathers and two products
    # rather than four operations per channel.
    beta_of, lm_of, l_of, atom_of = [], [], [], []
    dij_blocks = []
    for atom, species in enumerate(structure.types):
        for nb, l, lm in channels_by_species[species]:
            beta_of.append(beta_offset[species] + nb)
            lm_of.append(lm)
            l_of.append(l)
            atom_of.append(atom)
        dij_blocks.append(_expand_dij(pseudos[species], channels_by_species[species]))

    vkb = _assemble(
        kg,
        ylm,
        radial,
        structure.positions,
        planewaves.mask,
        jnp.asarray(beta_of),
        jnp.asarray(lm_of),
        jnp.asarray(atom_of),
        jnp.asarray((-1j) ** np.asarray(l_of)),
    )

    return Projectors(
        vkb=vkb.astype(cell.precision.complex),
        dij=jnp.asarray(_block_diagonal(dij_blocks)),
        atom_of_channel=tuple(atom_of),
    )


@partial(jax.jit, static_argnames=("lmax",))
def _angular_part(gcart, indices, kcart, lmax):
    """``k+G``, its modulus, and the spherical harmonics on it, in one kernel."""
    kg = kcart[:, None, :] + gcart[indices]  # (nk, npwx, 3), 1/bohr
    kg_norm = jnp.sqrt(jnp.sum(kg**2, axis=-1))
    return kg, kg_norm, real_spherical_harmonics(kg, lmax)


@partial(jax.jit, static_argnames=("shape",))
def _radial_table(form_factors, shape):
    """Per-species ``(nbeta, nk*npwx)`` tables -> one ``(nk, npwx, nbeta_total)``."""
    reshaped = [f.reshape((-1,) + shape).transpose(1, 2, 0) for f in form_factors]
    return jnp.concatenate(reshaped, axis=-1)


@jax.jit
def _assemble(kg, ylm, radial, tau, mask, beta_of, lm_of, atom_of, l_phase):
    """``<k+G|beta>`` for every channel: angular times radial times structure."""
    phases = jnp.exp(-1j * jnp.einsum("kgc,ac->akg", kg, tau))  # (nat, nk, npwx)
    columns = (
        l_phase[:, None, None]
        * jnp.take(ylm, lm_of, axis=-1).transpose(2, 0, 1)
        * jnp.take(radial, beta_of, axis=-1).transpose(2, 0, 1)
        * phases[atom_of]
    )
    vkb = jnp.transpose(columns, (1, 2, 0))  # (nk, npwx, nkb)
    return jnp.where(mask[..., None], vkb, 0.0)


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
