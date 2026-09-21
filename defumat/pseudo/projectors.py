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

from defumat.basis.gvectors import ORIGIN_TOL, GVectors, modulus
from defumat.basis.planewaves import PlaneWaveBasis
from defumat.pseudo.formfactors import (
    _origin_integrals, projector_form_factors)
from defumat.pseudo.harmonics import real_spherical_harmonics
from defumat.pseudo.upf import Pseudopotential
from defumat.system.cell import Cell
from defumat.system.kpoints import KPoints
from defumat.system.structure import Structure
from defumat.units import FPI

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

    #: ``(nk, npwx, nkb)`` complex, or ``None`` when this set is **lazy** and
    #: rebuilds one k-point at a time from :attr:`core` -- see :meth:`at_k` and
    #: the class docstring. Read it through the :attr:`vkb` property, never
    #: directly: on a lazy set the property materialises the whole-k array,
    #: which is exactly what a lazy set exists not to hold.
    stored: jnp.ndarray | None
    dij: jnp.ndarray  # (nkb, nkb), Ry
    atom_of_channel: tuple[int, ...] = eqx.field(static=True)
    #: ``q_ij``, the integral of the augmentation charge, block diagonal over
    #: atoms like ``dij``. ``None`` for a purely norm-conserving calculation,
    #: which is what makes ``S`` the identity there.
    qq: jnp.ndarray | None = None
    #: The phase-free core and the positions, kept only by a **lazy** set. They
    #: are what :meth:`at_k` rebuilds from, and they are ``(nk, npwx, ncs)``
    #: against :attr:`stored`'s ``(nk, npwx, nkb)`` -- smaller by the
    #: multiplicity of each species, which on a 45-atom cell of two species is
    #: about twenty.
    core: "ProjectorCore | None" = None
    positions: jnp.ndarray | None = None

    @property
    def is_lazy(self) -> bool:
        """Whether this set rebuilds per k rather than holding the whole array."""
        return self.stored is None

    @property
    def dtype(self):
        """The complex dtype, **without materialising anything.**

        ``projectors.vkb.dtype`` on a lazy set would build the whole-k array to
        read a five-byte attribute off it, which is the silent version of this
        feature not working. Every dtype read goes through here.
        """
        return (self.core.complex_dtype if self.stored is None
                else self.stored.dtype)

    @property
    def nk(self) -> int:
        return (self.core.columns.shape[0] if self.stored is None
                else self.stored.shape[0])

    @property
    def vkb(self) -> jnp.ndarray:
        """``(nk, npwx, nkb)``, materialising a lazy set if it has to.

        Every consumer that wants the whole k-axis at once -- the forces, the
        response stack, the topology overlaps -- reads this and is unchanged by
        laziness. Inside the eigensolver, where the whole-k array *is* the
        memory problem, use :meth:`at_k` instead.
        """
        if self.stored is not None:
            return self.stored
        return _apply_phases(
            self.core.columns, self.core.kg, self.positions, self.core.mask,
            jnp.asarray(self.core.atom_of_channel), self.core.column_of_channel,
        ).astype(self.core.complex_dtype)

    def at_k(self, ik) -> jnp.ndarray:
        """``(npwx, nkb)`` for one k-point, built rather than sliced.

        This is ``init_us_2`` called inside ``c_bands.f90``'s ``k_loop``: QE
        holds one k-point's projectors however many there are, and this is the
        same trade. ``ik`` may be a tracer -- it is, everywhere this is called
        from, since :func:`~defumat.batching.map_k` walks the axis with
        ``jnp.arange``.

        **Rebuilding costs less memory than hoisting, which is the opposite of
        the obvious worry.** Compiled inside a ``lax.while_loop`` body with
        loop-invariant inputs at slab-like shapes, the scratch is one ``vkb``
        (297.8 MB) where lifting the build out of the loop by hand is 1.5 of
        them (446.4 MB): XLA does not hoist it, and the loop-carried array it
        would otherwise hold is the larger cost. What rebuilding does cost is
        ``nat npwx`` complex exponentials per call.
        """
        if self.stored is not None:
            return self.stored[ik]
        return _apply_phases(
            self.core.columns[ik], self.core.kg[ik], self.positions,
            self.core.mask[ik], jnp.asarray(self.core.atom_of_channel),
            self.core.column_of_channel,
        ).astype(self.core.complex_dtype)

    @property
    def nkb(self) -> int:
        if self.stored is not None:
            return self.stored.shape[-1]
        return len(self.atom_of_channel)

    def project(self, psi: jnp.ndarray, ik: int) -> jnp.ndarray:
        """``<beta|psi>`` for wavefunctions ``psi`` of shape ``(..., npwx)``."""
        return jnp.einsum("...g,gk->...k", psi.conj(), self.at_k(ik)).conj()


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

    def at_positions(self, positions: jnp.ndarray, qq=None,
                     lazy: bool = False) -> Projectors:
        """The projectors for atoms at ``positions`` (cartesian, bohr).

        ``lazy`` returns a set that holds *this core* and the positions instead
        of the ``(nk, npwx, nkb)`` array, and rebuilds one k-point at a time on
        demand (:meth:`Projectors.at_k`). The arithmetic is identical -- same
        expression, same order -- so nothing about a result moves; what changes
        is that the largest resident array in a many-k run stops being resident.
        See :mod:`defumat.batching` for the dial that chooses.
        """
        if lazy:
            return Projectors(
                stored=None, dij=self.dij,
                atom_of_channel=self.atom_of_channel, qq=qq,
                core=self, positions=positions,
            )
        vkb = _apply_phases(
            self.columns, self.kg, positions, self.mask,
            jnp.asarray(self.atom_of_channel), self.column_of_channel,
        )
        return Projectors(
            stored=vkb.astype(self.complex_dtype),
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
    origin_tangent: bool = True,
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

    ``origin_tangent`` carries the ``l = 1`` tangent at ``k + G = 0``, which is
    the default and is the derivative the operator actually has: the product
    ``f_1(q) Y_1m(qhat)`` goes to ``c sqrt(3/4pi) q_m``, linear in the vector
    ``q``, so its gradient at the origin is ``c sqrt(3/4pi) delta_m,alpha``
    rather than zero (see :func:`_with_origin_tangent`).

    **``origin_tangent=False`` is Quantum ESPRESSO's convention and is there so
    that a ``ph.x`` comparison stays exact.** QE zeroes that row twice over --
    ``PW/src/commutator_Hx_psi.f90:113-118`` sets ``gk_vpol = 0`` where
    ``g2k < 1.0d-10``, killing the ``gen_us_dj`` term whatever ``djl`` reads,
    and ``upflib/dylmr2.f90:88-92`` sets ``dg = 0`` where ``gg <= eps``, so
    ``dylm`` and the ``gen_us_dy`` term go with it -- which makes QE's
    ``dH/dk`` discontinuous at exactly Gamma, since a ``k`` of 1e-4 off it
    computes the term. The row exists only where ``k + G = 0``, so the flag
    changes nothing on a shifted mesh and everything on a Gamma-only cell:
    measured on ``o2-fixed-lsda`` against ``reference.out.ph-o2-fixed-lsda``,
    ``eps_xx`` reads 1.11644639 here and 1.11091517 with the flag off against
    ``ph.x``'s 1.110915996, and ``Z*_xx`` 0.10110 against 0.13372 and 0.13367
    (`PLAN.md` P24).
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
    # The identity, owning the ``l = 1`` tangent at ``k + G = 0`` that the two
    # origin guards drop between them. See :func:`_with_origin_tangent`.
    # ``origin_tangent=False`` is QE's convention and drops it again, which is
    # what a ``ph.x`` comparison on a Gamma-containing mesh is held to.
    axes, slopes = _origin_slopes(pseudos, channels_by_species, cell.volume)
    if not origin_tangent:
        axes = ()
    columns = _with_origin_tangent(columns, kg, slopes, axes)

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
    origin_tangent: bool = True,
) -> Projectors:
    """Assemble ``<k+G|beta>`` for every k-point, atom and channel."""
    core = build_projector_core(
        pseudos, structure, cell, gvectors, planewaves, kpoints,
        origin_tangent=origin_tangent,
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


#: For an ``l = 1`` harmonic, the cartesian axis it is proportional to and the
#: sign, in this module's ``lm`` ordering. Read off
#: :func:`~defumat.pseudo.harmonics.real_spherical_harmonics` rather than
#: derived: at the unit vectors it returns ``Y_1 = +c z``, ``Y_2 = -c x`` and
#: ``Y_3 = -c y`` with ``c = sqrt(3/4pi)``, and the two minus signs are
#: ``ylmr2``'s ``-sent/sqrt(2)`` surviving into the ``m = 1`` pair.
_P_AXIS = {1: (2, 1.0), 2: (0, -1.0), 3: (1, -1.0)}


@partial(jax.custom_jvp, nondiff_argnums=(3,))
def _origin_tangent_rule(columns, kg, slopes, axes):
    """``columns`` itself, carrying the tangent the origin guards lose.

    A projector column is ``Y_lm(qhat) f_l(|q|)`` and **both factors guard the
    origin by zeroing** -- :func:`~defumat.basis.gvectors.modulus` because
    ``sqrt`` has an infinite derivative there, and
    :func:`~defumat.pseudo.harmonics.real_spherical_harmonics` because a zero
    vector has no direction. Each guard is right about its own factor and the
    primal is right too, since ``f_l(0) = 0`` kills the finite harmonic. **The
    product is what carries the derivative.** For ``l = 1``, ``f_1(q) -> c q``
    and ``Y_1m(qhat) = sqrt(3/4pi) q_alpha/q``, so the product is
    ``sqrt(3/4pi) c q_alpha`` -- a linear function of the *vector* ``q``, whose
    derivative is ``sqrt(3/4pi) c`` and not zero. The chain rule computes
    ``Y df + dY f`` with both terms zero and returns zero. ``l = 0`` is
    genuinely flat (``f_0`` is even in ``q``) and ``l >= 2`` genuinely vanishes
    (the product goes as ``q^l``), so ``l = 1`` is the only channel affected,
    and it is in almost every dataset.

    Measured before this correction, on ``si2-nosym.in`` at Gamma against a
    central difference of the same operator at a frozen sphere: the
    ``Gamma_1``-by-``Gamma_15`` block of ``<psi|dH/dk|psi>`` came out at
    **0.3695** of its value (0.16957 against 0.45892, Frobenius over the three
    axes), the worst entry being 0.13245 Ry bohr out of 1.0775, and it did not
    move with the step size. ``OPEN.md`` Part XIV has the controls.

    **The primal is the identity**, returned unchanged rather than added to, so
    that no value can move for a structural reason rather than an arithmetic
    one -- and so that the whole column array is not allocated a second time on
    a path a 157-atom slab takes. The rule fires on the rows
    :data:`~defumat.basis.gvectors.ORIGIN_TOL` selects, which is the same test
    ``modulus`` uses, so a row is corrected if and only if it was guarded. A
    **strain** derivative reaches it and gets nothing, correctly: ``k + G = 0``
    scales to ``0`` under any strain, so ``dkg`` is zero on exactly those rows.

    ``axes`` is the cartesian axis of each column, packed as a static tuple;
    ``slopes`` carries ``(-i) sign sqrt(3/4pi) f_1'(0)`` and is **zero for every
    column that is not an ``l = 1`` channel**, so the arithmetic is uniform and
    the branch lives in the data rather than in a mask.
    """
    return columns


@_origin_tangent_rule.defjvp
def _origin_tangent_jvp(axes, primals, tangents):
    columns, kg, slopes = primals
    dcolumns, dkg, _ = tangents
    if not axes:
        return columns, dcolumns
    at_origin = jnp.sum(kg * kg, axis=-1) <= ORIGIN_TOL
    along = jnp.take(dkg, jnp.asarray(axes), axis=-1)
    correction = jnp.where(at_origin[..., None], along * slopes, 0.0)
    return columns, dcolumns + correction.astype(dcolumns.dtype)


@partial(jax.jit, static_argnums=(3,))
def _with_origin_tangent(columns, kg, slopes, axes):
    """:func:`_origin_tangent_rule` under ``jit``.

    **What this costs, measured rather than assumed**, because
    ``build_projector_core`` runs eagerly and is rebuilt once per cartesian
    direction inside a ``jvp``. One ``VelocityOperator.matrix_elements`` call,
    median of 15 warm, on ``si-epsilon-unshifted`` (8 k-points, ``npwx = 360``):
    **343 ms** with no correction at all, 389 ms with the ``custom_jvp``
    boundary present and its rule trivial, 469 ms with the rule active, and
    **465 ms** compiled -- so the cost is the boundary and its four array
    operations rather than the dispatch, and ``jit`` buys 4 ms of it.

    **It is a fixed cost per call and does not scale with the cell**, which is
    what decides whether it matters: the same measurement on ``si2-nosym``
    (64 k-points) is **1122.9 ms against 1149.7**, an overhead of 27 ms and
    **2.4 per cent** where the eight-point cell paid 35. Precomputing the
    slopes entirely -- the other candidate -- is worth 5 ms of the 125, so they
    are not where the time goes.
    """
    return _origin_tangent_rule(columns, kg, slopes, axes)


def _origin_slopes(pseudos, channels_by_species, volume):
    """``(axes, slopes)`` for :func:`_with_origin_tangent`, one per column.

    ``slopes`` is ``(-i) sign sqrt(3/4pi) lim_{q->0} f_1(q)/q`` on an ``l = 1``
    channel and **zero on every other**, so the correction is inert wherever
    the guarded product's tangent was right to begin with and the rule needs no
    mask over channels.

    **The signs are host arithmetic and the radial integrals are one batched
    product**, since ``at_kcart`` rebuilds this inside a ``jvp`` once per
    velocity call: a loop of JAX scalars here cost 1.30 ms per rebuild on
    ``si-epsilon-unshifted`` against 1.01 ms like this. That is not where the
    correction's cost is -- precomputing the whole thing saves 5 ms of 125 --
    and it is written this way because the flat form is also the clearer one.
    The integrals cannot go on the host, although every number in them is
    tabulated: a stress derivative traces the whole calculation,
    pseudopotentials included (see
    :func:`~defumat.pseudo.formfactors._origin_integrals`).
    """
    root = float(np.sqrt(3.0 / (4.0 * np.pi)))
    offsets = np.cumsum([0] + [len(p.projectors) for p in pseudos])
    axes, picks, coefficients = [], [], []
    for species, channels in enumerate(channels_by_species):
        for nb, l, lm in channels:
            if l != 1:
                # A column the guards never cost anything: zero coefficient, and
                # the index is a placeholder that the zero annihilates.
                axes.append(0)
                picks.append(0)
                coefficients.append(0.0)
                continue
            axis, sign = _P_AXIS[lm]
            axes.append(axis)
            picks.append(int(offsets[species]) + nb)
            coefficients.append(-1j * sign * root)
    if not axes:
        return (), jnp.zeros((0,), dtype=complex)
    table = jnp.concatenate([_origin_integrals(p) for p in pseudos])
    slopes = (jnp.take(table, jnp.asarray(picks))
              * jnp.asarray(np.asarray(coefficients, dtype=complex)))
    return tuple(axes), slopes * (FPI / jnp.sqrt(volume))


@jax.jit
def _apply_phases(columns, kg, tau, mask, atom_of, column_of):
    """``<k+G|beta>``: each channel's column times its atom's structure factor.

    The only place the atomic positions enter the nonlocal pseudopotential, and
    therefore the only place ``grad`` with respect to them has to reach.
    """
    # ``...`` rather than ``k``: the same expression serves the whole k-axis
    # and one k-point's slice of it, which is what lets a lazy ``Projectors``
    # rebuild through this without a second implementation of the phase.
    phases = jnp.exp(-1j * jnp.einsum("...gc,ac->...ga", kg, tau))
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
