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

from defumat.basis.gvectors import GVectors, modulus
from defumat.basis.planewaves import PlaneWaveBasis
from defumat.pseudo.formfactors import projector_form_factors
from defumat.pseudo.harmonics import real_spherical_harmonics
from defumat.pseudo.upf import Pseudopotential
from defumat.system.cell import Cell
from defumat.system.kpoints import KPoints
from defumat.system.structure import Structure

__all__ = ["Projectors", "ProjectorCore", "build_projectors", "build_projector_core",
           "projector_channels"]


def projector_channels(pseudo: Pseudopotential) -> list[tuple[int, int, int]]:
    """The ``(beta index, l, lm column)`` of every projector channel of a species.

    QE's ``indv`` / ``nhtol`` / ``nhtolm``: each radial projector ``beta_nb`` with
    angular momentum ``l`` contributes ``2l+1`` channels, one per ``m``, and the
    ``lm`` column indexes the spherical harmonics in the ordering of
    :mod:`defumat.pseudo.harmonics`.
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
    #: ``q_ij``, the integral of the augmentation charge, block diagonal over
    #: atoms like ``dij``. ``None`` for a purely norm-conserving calculation,
    #: which is what makes ``S`` the identity there.
    qq: jnp.ndarray | None = None

    @property
    def nkb(self) -> int:
        return self.vkb.shape[-1]

    def project(self, psi: jnp.ndarray, ik: int) -> jnp.ndarray:
        """``<beta|psi>`` for wavefunctions ``psi`` of shape ``(..., npwx)``."""
        return jnp.einsum("...g,gk->...k", psi.conj(), self.vkb[ik]).conj()


class ProjectorCore(eqx.Module):
    """``<k+G|beta>`` with the structure factor left out.

    The angular part, the radial form factor and the ``(-i)^l`` phase depend on
    the *species* of an atom and not on where it is; only ``e^{-i(k+G).tau}``
    does. Holding the two apart is what lets a moved geometry rebuild the
    projectors for the cost of one complex exponential per atom
    (:meth:`at_positions`), and it is what makes the projectors a
    differentiable function of the positions without recomputing the radial
    integrals inside the gradient.

    **Memory.** ``columns`` is ``(nk, npwx, sum_t nh_t)`` complex -- one entry
    per *species* channel, where ``vkb`` has one per *atom* channel. For a cell
    with several atoms of the same species it is therefore smaller than the
    ``vkb`` it builds, by the multiplicity of that species.
    """

    #: ``(nk, npwx, ncs)``: the phase-free columns, one per species channel.
    columns: jnp.ndarray
    #: ``(nk, npwx, 3)``: ``k + G``, which the phase needs.
    kg: jnp.ndarray
    mask: jnp.ndarray  # (nk, npwx), which plane waves exist at each k
    dij: jnp.ndarray  # (nkb, nkb), Ry
    #: For each of the ``nkb`` channels, which atom it sits on and which column
    #: of :attr:`columns` it takes its species-dependent part from.
    atom_of_channel: tuple[int, ...] = eqx.field(static=True)
    column_of_channel: jnp.ndarray = eqx.field(converter=jnp.asarray)
    complex_dtype: object = eqx.field(static=True, default=None)

    def at_positions(self, positions: jnp.ndarray, qq=None) -> Projectors:
        """The projectors for atoms at ``positions`` (cartesian, bohr)."""
        vkb = _apply_phases(
            self.columns, self.kg, positions, self.mask,
            jnp.asarray(self.atom_of_channel), self.column_of_channel,
        )
        return Projectors(
            vkb=vkb.astype(self.complex_dtype),
            dij=self.dij,
            atom_of_channel=self.atom_of_channel,
            qq=qq,
        )


def build_projector_core(
    pseudos: tuple[Pseudopotential, ...],
    structure: Structure,
    cell: Cell,
    gvectors: GVectors,
    planewaves: PlaneWaveBasis,
    kpoints: KPoints,
    kcart: jnp.ndarray | None = None,
) -> ProjectorCore:
    """Everything in ``<k+G|beta>`` except where the atoms are.

    ``kcart`` replaces the k-points' cartesian coordinates (``(nk, 3)`` in
    1/bohr) while keeping ``planewaves`` -- the sphere each point was selected
    with. It exists so that the projectors can be a *traced* function of a
    k-point: the whole of ``<k+G|beta>`` is differentiable in ``k`` (the radial
    form factors are integrated rather than interpolated for exactly that
    reason), and the only host-side step is choosing which plane waves are in
    the sphere. A spin spiral's ``dE/dq`` is the caller
    (:mod:`defumat.forces.spiral`).
    """
    channels_by_species = [projector_channels(p) for p in pseudos]
    nkb = sum(len(channels_by_species[t]) for t in structure.types)
    if nkb == 0:
        # ``planewaves`` and not ``kpoints`` decides how many rows there are:
        # ``kcart`` is free to replace the k-points' own coordinates, and a
        # caller differentiating a *subset* of them (a spin spiral's chunked
        # ``dE/dq``) passes a basis with fewer rows than ``kpoints`` has.
        nk_rows = planewaves.nk
        empty = jnp.zeros((nk_rows, planewaves.npwx, 0), dtype=cell.precision.complex)
        return ProjectorCore(
            columns=empty,
            kg=jnp.zeros((nk_rows, planewaves.npwx, 3)),
            mask=planewaves.mask,
            dij=jnp.zeros((0, 0)),
            atom_of_channel=(),
            column_of_channel=jnp.zeros((0,), dtype=int),
            complex_dtype=cell.precision.complex,
        )

    lmax = max(p.lmax for p in pseudos)
    kg, kg_norm, ylm = _angular_part(
        gvectors.cartesian(cell),
        planewaves.indices,
        kpoints.cartesian(cell) if kcart is None else kcart,
        lmax,
    )

    # Radial form factors, per species: (nbeta, nk * npwx) -> (nk, npwx, nbeta),
    # concatenated over species so that a channel selects a column by one index.
    shape = kg_norm.shape
    flat = kg_norm.reshape(-1)
    form_factors = tuple(
        projector_form_factors(p, flat, cell.volume) for p in pseudos
    )
    radial = _radial_table(form_factors, shape)
    beta_offset = np.cumsum([0] + [f.shape[0] for f in form_factors])

    # One column per *species* channel, in the order the species are declared;
    # an atom's channels then select from it by index.
    beta_of, lm_of, l_of = [], [], []
    column_offset = [0]
    for species, channels in enumerate(channels_by_species):
        for nb, l, lm in channels:
            beta_of.append(beta_offset[species] + nb)
            lm_of.append(lm)
            l_of.append(l)
        column_offset.append(len(beta_of))

    columns = _species_columns(
        ylm, radial,
        jnp.asarray(beta_of), jnp.asarray(lm_of),
        jnp.asarray((-1j) ** np.asarray(l_of)),
    )

    # One row per projector channel, in QE's order: atoms outermost, then the
    # channels of that atom's species.
    atom_of, column_of, dij_blocks = [], [], []
    for atom, species in enumerate(structure.types):
        for index in range(len(channels_by_species[species])):
            atom_of.append(atom)
            column_of.append(column_offset[species] + index)
        dij_blocks.append(_expand_dij(pseudos[species], channels_by_species[species]))

    return ProjectorCore(
        columns=columns,
        kg=kg,
        mask=planewaves.mask,
        dij=jnp.asarray(_block_diagonal(dij_blocks)),
        atom_of_channel=tuple(atom_of),
        column_of_channel=jnp.asarray(column_of),
        complex_dtype=cell.precision.complex,
    )


def build_projectors(
    pseudos: tuple[Pseudopotential, ...],
    structure: Structure,
    cell: Cell,
    gvectors: GVectors,
    planewaves: PlaneWaveBasis,
    kpoints: KPoints,
) -> Projectors:
    """Assemble ``<k+G|beta>`` for every k-point, atom and channel."""
    core = build_projector_core(
        pseudos, structure, cell, gvectors, planewaves, kpoints
    )
    return core.at_positions(structure.positions)


@partial(jax.jit, static_argnames=("lmax",))
def _angular_part(gcart, indices, kcart, lmax):
    """``k+G``, its modulus, and the spherical harmonics on it, in one kernel."""
    kg = kcart[:, None, :] + gcart[indices]  # (nk, npwx, 3), 1/bohr
    # Guarded at the origin: a k-point at Gamma has ``k + G = 0`` in its sphere,
    # and ``sqrt``'s derivative there is what makes a strain gradient NaN.
    kg_norm = modulus(kg)
    return kg, kg_norm, real_spherical_harmonics(kg, lmax)


@partial(jax.jit, static_argnames=("shape",))
def _radial_table(form_factors, shape):
    """Per-species ``(nbeta, nk*npwx)`` tables -> one ``(nk, npwx, nbeta_total)``."""
    reshaped = [f.reshape((-1,) + shape).transpose(1, 2, 0) for f in form_factors]
    return jnp.concatenate(reshaped, axis=-1)


@jax.jit
def _species_columns(ylm, radial, beta_of, lm_of, l_phase):
    """The angular times radial part of every species channel, ``(nk, npwx, ncs)``."""
    columns = (
        jnp.take(ylm, lm_of, axis=-1)
        * jnp.take(radial, beta_of, axis=-1)
    )
    return columns * l_phase


@jax.jit
def _apply_phases(columns, kg, tau, mask, atom_of, column_of):
    """``<k+G|beta>``: each channel's column times its atom's structure factor.

    The only place the atomic positions enter the nonlocal pseudopotential, and
    therefore the only place ``grad`` with respect to them has to reach.
    """
    phases = jnp.exp(-1j * jnp.einsum("kgc,ac->kga", kg, tau))  # (nk, npwx, nat)
    vkb = jnp.take(columns, column_of, axis=-1) * jnp.take(phases, atom_of, axis=-1)
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
