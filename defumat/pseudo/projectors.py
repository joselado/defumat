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

import hashlib
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from defumat.basis.gvectors import ORIGIN_TOL, GVectors, modulus
from defumat.basis.planewaves import PlaneWaveBasis
from defumat.pseudo.formfactors import origin_series, projector_form_factors
from defumat.pseudo.harmonics import real_solid_harmonics, real_spherical_harmonics
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


def _content_digest(array) -> tuple | None:
    """A fingerprint of one tabulated array: its shape, its dtype and a blake2b
    of its bytes. ``None`` for ``None``, so an absent array and an empty one
    are told apart.

    The hashing is :func:`defumat.pseudo.augmentation._dataset_key`'s, and it is
    here rather than there so that this module and :mod:`defumat.paw.onecenter`
    can key their own setup on it -- ``augmentation`` imports this module, so
    the helper cannot live on that side of the dependency. The shape and dtype
    go into the fingerprint as well as the bytes, which costs nothing and means
    that no key built from these has to carry the lengths separately to be
    safe.

    **Only ever called on a UPF file's own NumPy arrays**, never on anything
    built from the cell or the G set. Those are host constants on every path,
    including inside :meth:`~defumat.scf.driver.Calculation.at_strain`'s trace,
    where ``build_augmentation`` already hashes ``pseudo.r`` the same way.
    """
    if array is None:
        return None
    values = np.ascontiguousarray(np.asarray(array))
    return (
        values.shape,
        values.dtype.str,
        hashlib.blake2b(values.tobytes(), digest_size=16).digest(),
    )


def _projector_dataset_key(pseudo: Pseudopotential) -> tuple:
    """A fingerprint of everything a species' projector columns are built from.

    Two species that name the same UPF file are the same dataset, and the
    phase-free half of ``<k+G|beta>`` depends on the dataset and on ``k + G``
    -- never on which label an atom carries. One species per magnetic site is
    the standard way to write a noncollinear texture (``angle1``/``angle2`` are
    per species), and site-resolved DFT+U is the same pattern, so a fifteen-site
    helix arrives here as fifteen identical datasets and would otherwise build
    fifteen identical blocks of :attr:`ProjectorCore.columns`.

    **The key is exactly what the columns read and nothing else**:
    :func:`~defumat.pseudo.formfactors.projector_form_factors` and
    :func:`~defumat.pseudo.formfactors.origin_series` integrate each
    ``beta[:kkbeta]`` against ``r[:kkbeta]`` with Simpson weights from
    ``rab[:kkbeta]``, and the channels fix the ``l``, the ``lm`` column and the
    ``(-i)^l`` phase. ``D_ij`` is not in it, because :func:`_expand_dij` is
    still called once per atom on that atom's own species, so two species with
    identical projectors and different ``D`` would share their columns and keep
    their own coefficients -- which is right.

    **It is not ``augmentation._dataset_key``, which does not hash ``beta``.**
    That key fingerprints what ``Q_ij(G)`` is built from -- ``r``, ``rab`` and
    ``qfuncl`` -- and for a norm-conserving dataset ``qfuncl`` is empty, so two
    norm-conserving datasets on one radial grid with the same ``l`` structure
    would collide there and share the wrong projectors here. Its ``nl_species``
    has no meaning for a projector either.
    """
    kkbeta = pseudo.kkbeta
    return (
        kkbeta,
        tuple(projector_channels(pseudo)),
        _content_digest(pseudo.r[:kkbeta]),
        _content_digest(pseudo.rab[:kkbeta]),
        tuple(_content_digest(projector.beta[:kkbeta])
              for projector in pseudo.projectors),
    )


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
    #: multiplicity of each distinct dataset, which on a 45-atom cell of two
    #: datasets is about twenty however many species labels name them.
    core: "ProjectorCore | None" = None
    positions: jnp.ndarray | None = None
    #: ``(nk_basis,)`` positions in :attr:`stored` of each basis row, ``-1``
    #: where the row is not held, or ``None`` when every row is. Set only on a
    #: **k-point pool's** set (:meth:`ProjectorCore.at_positions` with
    #: ``rows``), which stores the rows its own k-points read and rebuilds any
    #: other from :attr:`core`, so a consumer that asks for another pool's row
    #: gets the right array rather than a neighbour's.
    row_map: jnp.ndarray | None = None

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
        if self.row_map is not None:
            return self.row_map.shape[0]
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
        if self.stored is not None and self.row_map is None:
            return self.stored
        return _apply_phases(
            self.core.columns, self.core.kg, self.positions, self.core.mask,
            jnp.asarray(self.core.atom_of_channel), self.core.column_of_channel,
            self.core.phase_of_column,
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
        if self.stored is not None and self.row_map is None:
            return self.stored[ik]
        if self.row_map is not None:
            # A pool's set: its own rows from the store, any other rebuilt. With
            # one k-point per chunk ``ik`` is a scalar and the ``cond`` takes one
            # branch; under a ``vmap`` it evaluates both, which is the rebuild's
            # cost and never a wrong row.
            local = self.row_map[ik]
            return jax.lax.cond(local >= 0,
                                lambda: self.stored[jnp.maximum(local, 0)],
                                lambda: self._rebuilt(ik))
        return self._rebuilt(ik)

    def _rebuilt(self, ik) -> jnp.ndarray:
        return _apply_phases(
            self.core.columns[ik], self.core.kg[ik], self.positions,
            self.core.mask[ik], jnp.asarray(self.core.atom_of_channel),
            self.core.column_of_channel, self.core.phase_of_column,
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

    **Memory.** ``columns`` is ``(nk, npwx, sum_d nh_d)`` real -- one entry
    per channel of each distinct *dataset* ``d``, where ``vkb`` has one per
    *atom* channel. For a cell with several atoms of the same dataset it is
    therefore smaller than the ``vkb`` it builds, by the multiplicity of that
    dataset. **A dataset and not a species label**: two labels naming one UPF
    file share one block of columns (:func:`_projector_dataset_key`), which is
    what a noncollinear texture written one species per site needs. Before
    that the sum ran over labels, and fifteen nickel labels built fifteen
    identical blocks, which is the defect P73 closed for the augmentation
    charge (``OPEN.md`` Part III, M2).
    """

    #: ``(nk, npwx, ncs)``: the phase-free columns, one per channel of each
    #: distinct dataset. Several species may point at the same ones through
    #: :attr:`column_of_channel`. **Real**: a column is ``Y_lm`` times the
    #: radial transform, both real, and its ``(-i)^l`` is kept apart in
    #: :attr:`phase_of_column` -- half the bytes of the largest per-k-point
    #: array a memory-mode run keeps resident (``GPU-MEMORY-NEXT.md`` item 6).
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
    #: ``(ncs,)`` complex, ``(-i)^l`` of each column's channel, applied by
    #: :func:`_apply_phases` before the structure factor. **Multiplying by it
    #: is exact** -- it is ``1``, ``-i``, ``-1`` or ``i``, so a product with it
    #: only moves and negates the real one -- which is why the column can be
    #: stored without it and every ``vkb`` is the one it was when the phase
    #: was inside the column.
    phase_of_column: jnp.ndarray | None = None

    def rows(self, rows) -> "ProjectorCore":
        """The same core restricted to the k-points ``rows``: the per-k leaves sliced.

        ``columns``, ``kg`` and ``mask`` carry the k index; everything else is a
        property of the datasets and is shared. Selected rather than rebuilt,
        so the columns are the whole set's own, bit for bit.
        """
        return eqx.tree_at(
            lambda core: (core.columns, core.kg, core.mask), self,
            (self.columns[rows], self.kg[rows], self.mask[rows]),
        )

    def at_positions(self, positions: jnp.ndarray, qq=None,
                     lazy: bool = False, rows=None) -> Projectors:
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
        if rows is not None:
            # **A k-point pool stores its own basis rows only.** Every pool held
            # every k-point's ``(npwx, nkb)`` in ``store`` mode, which was the
            # largest array a pool replicated: 1.9 GiB of a 4.4 GiB rank on
            # ultrasoft Si40 at 27 k-points, and 14 GB a rank on the NiBr2 slab.
            # Rebuilding instead (``projectors = 'rebuild'``) frees it at 6 to 11
            # per cent of an iteration; holding the pool's rows frees it at none.
            rows = np.asarray(rows, dtype=int)
            stored = _apply_phases(
                self.columns[rows], self.kg[rows], positions, self.mask[rows],
                jnp.asarray(self.atom_of_channel), self.column_of_channel,
                self.phase_of_column,
            ).astype(self.complex_dtype)
            row_map = np.full(self.columns.shape[0], -1, dtype=np.int32)
            row_map[rows] = np.arange(len(rows), dtype=np.int32)
            return Projectors(
                stored=stored, dij=self.dij,
                atom_of_channel=self.atom_of_channel, qq=qq,
                core=self, positions=positions, row_map=jnp.asarray(row_map),
            )
        vkb = _apply_phases(
            self.columns, self.kg, positions, self.mask,
            jnp.asarray(self.atom_of_channel), self.column_of_channel,
            self.phase_of_column,
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

    ``origin_tangent`` carries the derivatives at ``k + G = 0``, to every
    order, which is the default and is the operator the projectors actually
    are: the product ``f_l(q) Y_lm(qhat)`` is ``S_lm(q) g_l(q^2)``, a solid
    harmonic times a function analytic in ``q^2``, smooth at the origin where
    both of its guarded factors are not. Its first derivative is the ``l = 1``
    tangent ``c sqrt(3/4pi) delta_m,alpha``, and its second and higher ones are
    the curvature of ``l = 0``, the ``q_a q_b`` of ``l = 2`` and the cubic part
    of ``l = 1`` (see :func:`_with_origin_rows`).

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

    # One entry per distinct *dataset*, not per species label: two labels
    # naming one UPF file get one block of columns, and the second label's
    # atoms select from the first's (see :func:`_projector_dataset_key`). The
    # datasets keep the order in which their first label is declared, so on a
    # cell with one label per dataset this is the list of species unchanged
    # and every array below is what it was before the sharing existed.
    datasets, slot_of, seen = [], [], {}
    for species, pseudo in enumerate(pseudos):
        key = _projector_dataset_key(pseudo)
        if key not in seen:
            seen[key] = len(datasets)
            datasets.append(species)
        slot_of.append(seen[key])
    dataset_pseudos = tuple(pseudos[species] for species in datasets)
    dataset_channels = [channels_by_species[species] for species in datasets]

    # Radial form factors, per dataset: (nbeta, nk * npwx) -> (nk, npwx, nbeta),
    # concatenated over datasets so that a channel selects a column by one index.
    shape = kg_norm.shape
    flat = kg_norm.reshape(-1)
    form_factors = tuple(
        projector_form_factors(p, flat, cell.volume) for p in dataset_pseudos
    )
    radial = _radial_table(form_factors, shape)
    beta_offset = np.cumsum([0] + [f.shape[0] for f in form_factors])

    # One column per *dataset* channel, in the order the datasets were first
    # declared; an atom's channels then select from it by index.
    beta_of, lm_of, l_of = [], [], []
    column_offset = [0]
    for slot, channels in enumerate(dataset_channels):
        for nb, l, lm in channels:
            beta_of.append(beta_offset[slot] + nb)
            lm_of.append(lm)
            l_of.append(l)
        column_offset.append(len(beta_of))

    columns = _species_columns(ylm, radial, jnp.asarray(beta_of), jnp.asarray(lm_of))
    phase_of_column = jnp.asarray((-1j) ** np.asarray(l_of))
    # The rows at ``k + G = 0``, where both factors of a column are guarded and
    # every derivative of the product is lost, rewritten as a polynomial in the
    # vector ``q`` that has them (:func:`_with_origin_rows`).
    # ``origin_tangent=False`` is QE's convention and keeps the guarded rows,
    # which is what a ``ph.x`` comparison on a Gamma-containing mesh is held to.
    if origin_tangent and _has_origin_rows(kg):
        series = _origin_columns(dataset_pseudos, beta_of, l_of, cell.volume)
        columns = _with_origin_rows(columns, kg, series, jnp.asarray(lm_of), lmax)

    # One row per projector channel, in QE's order: atoms outermost, then the
    # channels of that atom's species. The column comes from the species'
    # *dataset*, and ``D`` from the species itself. Every column is computed
    # element by element from its own dataset's radial table, and
    # ``_apply_phases`` gathers columns by index, so a shared column holds the
    # values each label's own copy would have held and ``vkb`` is unchanged:
    # a two-label cell gives the same bytes as the same cell written with one
    # label (``tests/unit/test_dataset_dedupe.py``). What does move is the
    # order of a reverse-mode sum: the cotangent of a shared column is
    # accumulated over both labels' atoms before it is carried back to
    # ``k + G``, where it used to be carried back twice and added there, so a
    # stress or a spiral ``dE/dq`` on such a cell is reassociated at round-off.
    atom_of, column_of, dij_blocks = [], [], []
    for atom, species in enumerate(structure.types):
        for index in range(len(channels_by_species[species])):
            atom_of.append(atom)
            column_of.append(column_offset[slot_of[species]] + index)
        dij_blocks.append(_expand_dij(pseudos[species], channels_by_species[species]))

    return ProjectorCore(
        columns=columns,
        kg=kg,
        mask=planewaves.mask,
        dij=jnp.asarray(_block_diagonal(dij_blocks)),
        atom_of_channel=tuple(atom_of),
        column_of_channel=jnp.asarray(column_of),
        complex_dtype=cell.precision.complex,
        phase_of_column=phase_of_column,
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
def _species_columns(ylm, radial, beta_of, lm_of, l_phase=None):
    """The angular times radial part of every species channel, ``(nk, npwx, ncs)``.

    Real without ``l_phase``, which is how the projector core keeps them; the
    atomic orbitals pass their ``i^l`` and get the phased columns.
    """
    columns = (
        jnp.take(ylm, lm_of, axis=-1)
        * jnp.take(radial, beta_of, axis=-1)
    )
    return columns if l_phase is None else columns * l_phase


def _has_origin_rows(kg) -> bool:
    """Whether any row of ``kg`` can be at ``k + G = 0``.

    A traced ``kg`` -- a moved k-point under ``at_kcart``, a strained cell --
    can put any row there, so the answer is yes without looking. A concrete one
    is read, and a mesh with no such row (any shifted one) skips
    :func:`_with_origin_rows` entirely, so its projectors are the bytes they
    were before the rows were rewritten.
    """
    if isinstance(kg, jax.core.Tracer):
        return True
    return bool(jnp.any(jnp.sum(kg * kg, axis=-1) <= ORIGIN_TOL))


def _origin_columns(pseudos, beta_of, l_of, volume):
    """The Taylor coefficients of ``g_l(q^2)`` for every column, ``(ncols, terms)``.

    :func:`~defumat.pseudo.formfactors.origin_series` for each dataset, with the
    ``4 pi / sqrt(Omega)`` of :func:`~defumat.pseudo.formfactors.projector_form_factors`
    (traced under a strain), gathered by each column's radial index. **The
    constant term of an** ``l = 0`` **column is removed**, because the guarded
    column already holds ``Y_00 f_0(0)`` there: what :func:`_with_origin_rows`
    adds is the rest of the series, which is exactly zero at ``q = 0``, so the
    value at the origin is the transform's own and not a second evaluation of
    it.
    """
    table = jnp.concatenate([origin_series(p) for p in pseudos])
    series = jnp.take(table, jnp.asarray(beta_of), axis=0) * (FPI / jnp.sqrt(volume))
    keep_constant = np.asarray([l != 0 for l in l_of], dtype=float)
    return series.at[:, 0].multiply(jnp.asarray(keep_constant))


@partial(jax.jit, static_argnames=("lmax",))
def _with_origin_rows(columns, kg, series, lm_of, lmax):
    """``columns`` with the rows at ``k + G = 0`` written as ``S_lm(q) g_l(q^2)``.

    A projector column is ``Y_lm(qhat) f_l(|q|)`` and **both factors guard the
    origin by zeroing** -- :func:`~defumat.basis.gvectors.modulus` because
    ``sqrt`` has an infinite derivative there, and
    :func:`~defumat.pseudo.harmonics.real_spherical_harmonics` because a zero
    vector has no direction. Each guard is right about its own factor and the
    value is right too, since ``f_l(0) = 0`` kills the finite harmonic. **The
    product is what carries the derivatives**, and the chain rule through two
    guarded factors returns zero for every one of them.

    The product is ``S_lm(q) g_l(q^2)``, with ``S_lm = |q|^l Y_lm`` the regular
    solid harmonic, a polynomial in the components of ``q``
    (:func:`~defumat.pseudo.harmonics.real_solid_harmonics`), and
    ``g_l = f_l/q^l`` the transform's own Taylor series in ``q^2``
    (:func:`~defumat.pseudo.formfactors.origin_series`). That is differentiable
    to every order at the origin. On the rows
    :data:`~defumat.basis.gvectors.ORIGIN_TOL` selects, the same test
    ``modulus`` uses, the guarded column is replaced by itself **plus** the
    series without the ``l = 0`` constant, which the guarded column already
    holds: at ``q = 0`` exactly the sum is that column's own value (the added
    terms are a zero, of either sign), and away from it it is the function.

    **What was wrong without it, measured** on ``si2-nosym.in`` at Gamma. To
    first order, against a central difference of the same operator at a frozen
    sphere, the ``Gamma_1``-by-``Gamma_15`` block of ``<psi|dH/dk|psi>`` came
    out at **0.3695** of its value (0.16957 against 0.45892, Frobenius over the
    three axes) and did not move with the step; a ``custom_jvp`` that put back
    the ``l = 1`` tangent alone repaired it (``OPEN.md`` Part XIV). To second
    order ``second_matrix_elements`` ``xx`` was off by **3.46e-2 Ry bohr^2** in
    one element of a block of norm 5.15, the same at ``h = 1e-3`` and
    ``3e-4``, while two other k-points sat at the stencil's 2e-7 and 2e-8
    (``HARMONICS-NEXT.md``, "The row at k + G = 0"), and the shift current
    consumes that operator. This form covers both and every order above, where
    a second custom rule would have covered order two only.

    The primal on a row *inside* the guard but off the origin, ``0 < |q|^2 <=
    1e-8``, also changes: the guarded ``l = 1`` column there was zero, and it
    is now ``c q``, which a finite difference with a step below 1e-4 around
    Gamma reaches and a field ``kappa(t)`` passing through zero reaches once
    per crossing.

    **What it costs** is a solid harmonic and a four-term polynomial per row
    and column, beside a Bessel transform of a few hundred mesh points per row
    and radial channel, and nothing on a mesh with no such row
    (:func:`_has_origin_rows`). A **strain** derivative reaches it and gets
    nothing, correctly: ``k + G = 0`` scales to ``0`` under any strain, so
    ``dkg`` is zero on exactly those rows, and the ``Omega`` in the series
    multiplies terms that vanish there.
    """
    s = jnp.sum(kg * kg, axis=-1)
    at_origin = s <= ORIGIN_TOL
    solid = jnp.take(real_solid_harmonics(kg, lmax), lm_of, axis=-1)
    g = jnp.broadcast_to(series[:, -1], solid.shape)
    for j in range(series.shape[1] - 2, -1, -1):
        g = g * s[..., None] + series[:, j]
    added = (solid * g).astype(columns.dtype)
    return jnp.where(at_origin[..., None], columns + added, columns)


@jax.jit
def _apply_phases(columns, kg, tau, mask, atom_of, column_of, column_phase=None):
    """``<k+G|beta>``: each channel's column times its atom's structure factor.

    The only place the atomic positions enter the nonlocal pseudopotential, and
    therefore the only place ``grad`` with respect to them has to reach.

    ``column_phase`` is the ``(-i)^l`` a real column was stored without, and it
    multiplies the gathered column **before** the structure factor, so the
    product is the one a phased column gave: the phase is exact, the order of
    the one rounding multiply is unchanged.
    """
    # ``...`` rather than ``k``: the same expression serves the whole k-axis
    # and one k-point's slice of it, which is what lets a lazy ``Projectors``
    # rebuild through this without a second implementation of the phase.
    phases = jnp.exp(-1j * jnp.einsum("...gc,ac->...ga", kg, tau))
    gathered = jnp.take(columns, column_of, axis=-1)
    if column_phase is not None:
        gathered = gathered * jnp.take(column_phase, column_of)
    vkb = gathered * jnp.take(phases, atom_of, axis=-1)
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
