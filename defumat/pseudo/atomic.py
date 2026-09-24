"""Pseudo-atomic orbitals in the plane-wave basis: the starting wavefunctions.

An SCF has to start its eigensolver somewhere. Starting from random vectors
works and is what this code did, but it wastes the first diagonalisation
discovering that electrons sit near atoms -- something the pseudopotential file
already knows, because it carries the orbitals of the isolated atom in
``PP_PSWFC``. QE starts from a superposition of those (``wfcinit`` ->
``atomic_wfc``), and its first SCF iteration costs two Davidson steps where a
random start costs eight.

The expression is the projectors' expression with a different radial function,

    <k+G| chi_lm^a> = 4 pi / sqrt(Omega) i^l Y_lm(k+G) chi_l(|k+G|) e^{-i(k+G).tau_a}

following ``Modules/atomic_wfc_mod.f90``. **The phase is** ``i^l``, not the
``(-i)^l`` of the projectors, and the Fortran says why in a comment: it is what
makes the k = 0 wavefunctions real in real space. Getting it wrong does not
fail loudly -- it produces a starting guess that is merely worse.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from defumat.basis.gvectors import GVectors
from defumat.basis.planewaves import PlaneWaveBasis
from defumat.pseudo.formfactors import atomic_form_factors
from defumat.pseudo.projectors import (
    _angular_part,
    _apply_phases,
    _content_digest,
    _radial_table,
    _species_columns,
)
from defumat.pseudo.spinorbit import LMAXX, rot_ylm, sph_ind, spinor
from defumat.pseudo.upf import Pseudopotential
from defumat.system.cell import Cell
from defumat.system.kpoints import KPoints
from defumat.system.structure import Structure

__all__ = [
    "atomic_channels",
    "atomic_wavefunctions",
    "count_atomic_wavefunctions",
    "spinor_orbital_blocks",
    "spinor_atomic_wavefunctions",
    "count_spinor_wavefunctions",
]


def atomic_channels(pseudo: Pseudopotential) -> list[tuple[int, int, int]]:
    """``(radial index, l, lm column)`` for every orbital channel of a species.

    Orbitals with negative occupation are skipped, as QE skips them, and the
    radial index counts only the kept ones so that it indexes
    :func:`~defumat.pseudo.formfactors.atomic_form_factors` directly.
    """
    channels, kept = [], 0
    for orbital in pseudo.orbitals:
        if orbital.occupation < 0.0:
            continue
        for m in range(2 * orbital.l + 1):
            channels.append((kept, orbital.l, orbital.l * orbital.l + m))
        kept += 1
    return channels


def _atomic_dataset_key(pseudo: Pseudopotential) -> tuple:
    """A fingerprint of everything a species' atomic-orbital columns are built from.

    The counterpart of :func:`~defumat.pseudo.projectors._projector_dataset_key`
    for ``<k+G|chi>``, and for the same reason: two species labels naming one
    UPF file are one dataset, the phase-free half of an atomic orbital depends
    on the dataset and on ``k + G`` and never on the label, and one label per
    magnetic site (``angle1``/``angle2``, a site-resolved ``HUBBARD`` card)
    would otherwise pay one radial transform per label for identical tables.

    **It cannot be the projector key, which never reads an orbital.** That key
    hashes ``beta`` over ``r[:kkbeta]``, while
    :func:`~defumat.pseudo.formfactors.atomic_form_factors` integrates
    ``PP_PSWFC`` over QE's 10-bohr ``msh``, so under it two datasets with the
    same projectors and different orbitals would share the wrong columns, and a
    difference in ``r`` or ``rab`` between ``kkbeta`` and ``msh`` would not be
    seen at all. It is written here rather than beside the projector key
    because this is the one build that reads it.

    **The key is exactly what the columns read**: the cut ``msh``, ``r`` and
    ``rab`` up to it, every kept orbital's ``chi`` up to it, and the channels,
    which fix the ``l``, the ``lm`` column, the ``i^l`` phase and which orbitals
    a negative occupation drops. The occupations themselves are not in it and
    neither is ``j``, because nothing in the columns reads them: the spinor
    blocks that read ``j`` (:func:`spinor_orbital_blocks`, and DFT+U's
    ``_spinor_channels``) are built atom by atom from each atom's own species,
    on the output of :func:`atomic_wavefunctions` rather than on its columns.
    """
    msh = pseudo.msh
    return (
        msh,
        tuple(atomic_channels(pseudo)),
        _content_digest(pseudo.r[:msh]),
        _content_digest(pseudo.rab[:msh]),
        # ``atomic_form_factors`` slices ``chi[: len(r[:msh])]``, which is
        # ``chi[:msh]`` because ``msh`` never exceeds the mesh.
        tuple(_content_digest(orbital.chi[:msh])
              for orbital in pseudo.orbitals if orbital.occupation >= 0.0),
    )


def count_atomic_wavefunctions(
    pseudos: tuple[Pseudopotential, ...], structure: Structure
) -> int:
    """QE's ``natomwfc``: how many atomic orbitals the crystal has in total."""
    return sum(len(atomic_channels(pseudos[t])) for t in structure.types)


def atomic_wavefunctions(
    pseudos: tuple[Pseudopotential, ...],
    structure: Structure,
    cell: Cell,
    gvectors: GVectors,
    planewaves: PlaneWaveBasis,
    kpoints: KPoints,
    kcart: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """``(nk, natomwfc, npwx)`` pseudo-atomic orbitals at every k-point.

    Bands-first, matching how wavefunctions are carried everywhere else, so the
    result can be handed straight to an eigensolver as a starting guess.
    """
    channels_by_species = [atomic_channels(p) for p in pseudos]
    natomwfc = sum(len(channels_by_species[t]) for t in structure.types)
    if natomwfc == 0:
        return jnp.zeros((kpoints.nk, 0, planewaves.npwx), dtype=cell.precision.complex)

    lmax = max((l for channels in channels_by_species for _, l, _ in channels), default=0)
    kg, kg_norm, ylm = _angular_part(
        gvectors.cartesian(cell), planewaves.indices,
        kpoints.cartesian(cell) if kcart is None else kcart, lmax
    )

    # One entry per distinct *dataset*, not per species label, exactly as
    # ``build_projector_core`` does it: two labels naming one UPF file get one
    # radial transform and one block of columns, and the second label's atoms
    # select from the first's (see :func:`_atomic_dataset_key`). The datasets
    # keep the order in which their first label is declared, so on a cell with
    # one label per dataset this is the list of species unchanged.
    datasets, slot_of, seen = [], [], {}
    for species, pseudo in enumerate(pseudos):
        key = _atomic_dataset_key(pseudo)
        if key not in seen:
            seen[key] = len(datasets)
            datasets.append(species)
        slot_of.append(seen[key])

    shape = kg_norm.shape
    flat = kg_norm.reshape(-1)
    form_factors = tuple(
        atomic_form_factors(pseudos[species], flat, cell.volume)
        for species in datasets
    )
    radial = _radial_table(form_factors, shape)
    chi_offset = np.cumsum([0] + [f.shape[0] for f in form_factors])

    # One column per *dataset* channel, in the order the datasets were first
    # declared. This is narrower than the build it replaces on every cell and
    # not only on one with repeated labels: that build made one column per
    # *atom* channel and handed ``_apply_phases`` ``arange``, so two atoms of
    # one species already had two copies of each column.
    chi_of, lm_of, l_of = [], [], []
    column_offset = [0]
    for slot, species in enumerate(datasets):
        for nb, l, lm in channels_by_species[species]:
            chi_of.append(chi_offset[slot] + nb)
            lm_of.append(lm)
            l_of.append(l)
        column_offset.append(len(chi_of))

    # One row per atomic orbital, atoms outermost and then the channels of that
    # atom's species, which is QE's ``natomwfc`` order and the order every
    # consumer indexes (``spinor_orbital_blocks``' ``start``, DFT+U's manifold
    # columns, ``projwfc``'s labels). The column comes from the atom's dataset.
    atom_of, column_of = [], []
    for atom, species in enumerate(structure.types):
        for index in range(len(channels_by_species[species])):
            atom_of.append(atom)
            column_of.append(column_offset[slot_of[species]] + index)

    # The same assembly as the projectors, with i^l in place of (-i)^l. Every
    # column is computed element by element from its own dataset's radial
    # table and ``_apply_phases`` gathers columns by index, so the output holds
    # the bytes the per-atom build held, and a cell written with two labels on
    # one dataset gives the bytes of the same cell written with one
    # (``tests/unit/test_atomic_columns_per_dataset.py`` checks both).
    #
    # What moves is the order of a reverse-mode sum, and **its scope is wider
    # than the projectors'**, because the build this replaces was per atom and
    # not per label: on every cell with two or more atoms on one dataset, one
    # label or several, those atoms' cotangents are now summed at the gather in
    # ``_apply_phases`` and multiplied by the column's other factors once, where
    # each used to be multiplied by them and scatter-added into ``radial`` and
    # ``ylm`` on its own -- ``(g_0 + g_1) b`` against ``g_0 b + g_1 b``, the
    # same sum at round-off. The one reverse-mode consumer is the DFT+U stress:
    # ``Calculation.at_strain`` rebuilds ``wfcU`` at a traced ``kcart`` through
    # ``_build_hubbard_projectors``, under ``stress/autodiff.py``'s
    # ``_energy_gradient``, which is ``jax.jit(jax.grad(...))``. The one DFT+U
    # stress a test asserts, ``ni-ldau-stress``, has one atom and cannot see it.
    # Three things do not move at all: the primal; the DFT+U force, since the
    # positions enter only through the phases and the gathered columns are the
    # same bytes; and a forward-mode derivative (``at_kcart``'s ``jvp``,
    # ``_term_gradients``' ``jacfwd``), whose tangent is carried through the
    # gather element by element. A spin spiral, the other rebuild at a traced
    # ``k`` that is differentiated in reverse, refuses DFT+U outright
    # (``Calculation._setup_hubbard``).
    columns = _species_columns(
        ylm,
        radial,
        jnp.asarray(chi_of),
        jnp.asarray(lm_of),
        jnp.asarray((1j) ** np.asarray(l_of)),
    )
    wfc = _apply_phases(
        columns,
        kg,
        structure.positions,
        planewaves.mask,
        jnp.asarray(atom_of),
        jnp.asarray(column_of),
    )
    return jnp.transpose(wfc, (0, 2, 1)).astype(cell.precision.complex)


# ---------------------------------------------------------------------------
# The spinor projector set: |l, j, m_j> rather than up/down copies of Y_lm.
# ---------------------------------------------------------------------------
#
# ``projwfc.x`` reaches these through ``atomic_wfc_nc_proj``, which calls
# ``atomic_wfc_acc`` with ``starting_spin_angle = .TRUE.`` and ``updown =
# .TRUE.``. That combination is **not** the one a noncollinear SCF or DFT+U
# uses: those take ``starting_spin_angle = .FALSE.`` and land in
# ``atomic_wfc_so_mag``, which averages the two ``j`` radial functions of a
# shell and fills pure up and down spinors with the result
# (:func:`defumat.hubbard.projectors._spinor_channels`). A projection wants the
# spin-angle functions themselves, so the two sets are different bases of the
# same space and both are needed.
#
# Every branch is a **fixed complex matrix acting on the scalar orbitals of one
# radial channel**, which is what keeps this small: the radial function, the
# structure factor and the ``i^l`` phase are all in
# :func:`atomic_wavefunctions` already, and what a spin-angle function adds is a
# Clebsch-Gordan coefficient and a change of harmonic basis. So the map is
# built once per species in numpy and applied as a contraction.


def _spin_angle_matrix(l: int, j: float) -> np.ndarray:
    """``(2j+1, 2, 2l+1)``: ``atomic_wfc_so``'s map onto ``|l j m_j>``.

    Row ``c`` is one spin-angle function, ``[c, is]`` its ``is``-th spinor
    component expressed in the *real* spherical harmonics of the shell:

        Omega_{l j m_j}[is] = spinor(l, j, m, is) * sum_m' rot_ylm[m'', m'] Y_lm'

    with ``m'' = sph_ind(l, j, m, is)`` and ``m_j = m + 1/2``. The ``m`` loop
    runs ``-l-1 .. l``, one value more than ``2l+1``, and the rows where both
    Clebsch-Gordan coefficients vanish are dropped -- which is how the two ``j``
    shells of one ``l`` come out with ``2l+2`` and ``2l`` members.

    ``rot_ylm`` is indexed relative to its centre row exactly as
    :func:`defumat.projwfc.angular_momentum.orbital_matrices` indexes it, so
    one matrix built at ``LMAXX`` serves every shell.
    """
    u = rot_ylm(LMAXX)
    width = 2 * l + 1
    rows = []
    for m in range(-l - 1, l + 1):
        factors = (spinor(l, j, m, 0), spinor(l, j, m, 1))
        if abs(factors[0]) <= 1.0e-8 and abs(factors[1]) <= 1.0e-8:
            continue
        block = np.zeros((2, width), dtype=complex)
        for component, factor in enumerate(factors):
            if abs(factor) <= 1.0e-8:
                continue
            block[component] = factor * u[LMAXX + sph_ind(l, j, m, component), :width]
        rows.append(block)
    return np.stack(rows) if rows else np.zeros((0, 2, width), dtype=complex)


def _updown_matrix(l: int) -> np.ndarray:
    """``(2 (2l+1), 2, 2l+1)``: ``atomic_wfc_nc``'s map, with ``updown``.

    Each real harmonic appears twice, once as a pure spin-up spinor and once as
    a pure spin-down one, and the two copies of a *shell* are emitted as blocks
    -- every ``m`` up, then every ``m`` down -- which is the order
    ``fill_nlmchi`` labels with ``ind = m`` and ``ind = m + 2l + 1``.
    """
    width = 2 * l + 1
    matrix = np.zeros((2 * width, 2, width), dtype=complex)
    for m in range(width):
        matrix[m, 0, m] = 1.0
        matrix[width + m, 1, m] = 1.0
    return matrix


def spinor_orbital_blocks(
    pseudos: tuple[Pseudopotential, ...], structure: Structure, lspinorb: bool
) -> list[tuple[int, int, np.ndarray]]:
    """``(start, width, matrix)`` for every spinor shell of the crystal.

    ``start`` and ``width`` address the *scalar* columns
    :func:`atomic_wavefunctions` builds -- one radial channel of one atom -- and
    ``matrix`` is ``(ncolumns, 2, width)`` complex, so the spinor columns of the
    shell are

        out[c] = concat( matrix[c, 0] @ phi, matrix[c, 1] @ phi )

    with ``phi`` that scalar block. Every branch of ``atomic_wfc_acc`` that
    ``atomic_wfc_nc_proj`` can reach is one of these, and which one is chosen
    follows the Fortran's own test -- the *dataset* decides, not the input:

    * a fully-relativistic dataset (``has_so``, i.e. ``PP_RELWFC`` carried a
      ``jchi``) gives ``atomic_wfc_so``, the genuine ``|l j m_j>`` shells;
    * a scalar dataset under ``lspinorb`` gives ``atomic_wfc_so2``, which builds
      *both* ``j = l +- 1/2`` shells out of the one radial function it has;
    * a scalar dataset without it gives ``atomic_wfc_nc``, an up and a down copy
      of each real harmonic.

    **A relativistic dataset with** ``lspinorb = .false.`` **is refused.** QE
    dispatches the orbitals on ``has_so`` and the *labels* on ``lspinorb``
    (``fill_nlmchi``), so that combination builds ``j``-resolved columns and
    labels them as up/down ones. The counts agree, nothing fails, and every
    label is wrong; it is not reproduced here.
    """
    blocks: list[tuple[int, int, np.ndarray]] = []
    start = 0
    for species in structure.types:
        pseudo = pseudos[species]
        kept = [o for o in pseudo.orbitals if o.occupation >= 0.0]
        has_so = any(getattr(o, "j", None) is not None for o in kept)
        if has_so and not lspinorb:
            raise NotImplementedError(
                "a fully-relativistic dataset with lspinorb = .false. is not "
                "implemented for the projection: QE dispatches the orbitals on "
                "has_so and their labels on lspinorb, so it builds j-resolved "
                "columns and calls them up/down ones. Run with lspinorb = "
                ".true., which is what a relativistic dataset is for"
            )
        for orbital in kept:
            l, width = orbital.l, 2 * orbital.l + 1
            if not lspinorb:
                blocks.append((start, width, _updown_matrix(l)))
            elif has_so:
                blocks.append((start, width, _spin_angle_matrix(l, orbital.j)))
            else:
                # ``atomic_wfc_so2``: both j shells off one radial function.
                for n2 in (l, l + 1):
                    j = n2 - 0.5
                    if j > 0.0:
                        blocks.append((start, width, _spin_angle_matrix(l, j)))
            start += width
    return blocks


def count_spinor_wavefunctions(
    pseudos: tuple[Pseudopotential, ...], structure: Structure, lspinorb: bool
) -> int:
    """``natomwfc`` for a spinor projection: ``n_atom_wfc`` with ``noncolin``.

    **It is not twice the scalar count on a fully-relativistic dataset**, which
    is the trap worth stating: such a file carries the two ``j`` of a shell as
    two separate ``PP_CHI`` entries, so the *scalar* count already has ``2l+1``
    for each of them where the spinor one has ``2j+1``. Platinum's
    ``5D(j=3/2), 5D(j=5/2), 6S`` is 11 scalar columns and **12** spinor ones,
    not 22. Doubling is right only for the two branches that build both ``j``
    shells (or both spins) out of one radial function.

    Counted from the blocks rather than by a rule, so that a mistake in one of
    them shows up here rather than downstream.
    """
    return sum(
        matrix.shape[0] for _, _, matrix in spinor_orbital_blocks(pseudos, structure, lspinorb)
    )


def spinor_atomic_wavefunctions(
    pseudos: tuple[Pseudopotential, ...],
    structure: Structure,
    cell: Cell,
    gvectors: GVectors,
    planewaves: PlaneWaveBasis,
    kpoints: KPoints,
    lspinorb: bool,
    kcart: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """``(nk, natomwfc_spinor, 2 npwx)``: the spin-angle atomic orbitals.

    ``atomic_wfc_nc_proj``. The scalar orbitals are built once by
    :func:`atomic_wavefunctions` -- which carries the radial function, the
    structure factor and the ``i^l`` phase, and is validated against ``pw.x``
    through every calculation that starts from them -- and each spinor shell is
    a fixed complex contraction of one of their blocks
    (:func:`spinor_orbital_blocks`).
    """
    scalar = atomic_wavefunctions(
        pseudos, structure, cell, gvectors, planewaves, kpoints, kcart
    )  # (nk, natomwfc, npwx)
    blocks = spinor_orbital_blocks(pseudos, structure, lspinorb)
    npwx = planewaves.npwx
    complex_dtype = cell.precision.complex
    if not blocks:
        return jnp.zeros((scalar.shape[0], 0, 2 * npwx), dtype=complex_dtype)

    columns = []
    for start, width, matrix in blocks:
        phi = scalar[:, start:start + width, :]           # (nk, width, npwx)
        m = jnp.asarray(matrix, dtype=complex_dtype)      # (ncol, 2, width)
        # (nk, ncol, 2, npwx) -- the spin axis is folded into the coefficient
        # vector afterwards, because a spinor here is one vector of length
        # ``2 npwx`` and not a (2, npwx) array.
        block = jnp.einsum("csw,kwg->kcsg", m, phi)
        columns.append(jnp.reshape(block, block.shape[:2] + (2 * npwx,)))
    return jnp.concatenate(columns, axis=1).astype(complex_dtype)
