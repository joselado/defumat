"""The augmentation charge of an ultracell: one displaced table per Q-difference.

``PLAN.md`` P88 stage 5. An ultrasoft or PAW dataset puts a charge inside the
projector spheres that no wavefunction carries, and everything an ultracell does
-- the matrix element, the density, PAW's one-centre terms -- gains a term
because of it. There is exactly **one** new object behind all three, and it is
the resident table displaced by a Q-vector.

**Where the displacement comes from.** A matrix element of the difference
potential between two ultracell basis functions runs over the whole ultracell,
so it sums over the ``N`` copies of every atom:

    <psi_{k0+Q,m}| dV |psi_{k0+Q',n}>
        += sum_{a,R} sum_ij [int dV(r) Q_ij^a(r - tau_a - R) dr]
                            conj(B^{a,Q}_i) B^{a,Q'}_j e^{i(Q'-Q).R}

with ``B^{a,Q}_i = <beta_i^{a}|psi_{k0+Q}>`` the unit cell's own projection and
the Bloch phase ``e^{i(k0+Q).R}`` of each copy's projector left over as
``e^{i(Q'-Q).R}``. Summing the lattice sum inside the integral instead,

    sum_R e^{i(Q'-Q).R} Q_ij^a(r - tau_a - R),

is precisely what :func:`~defumat.pseudo.augmentation.build_augmentation`
builds with ``shift = Q' - Q``, the **ket's** wavevector minus the bra's. So the
term is one integral against a displaced table, contracted with the two
projections, and there is no second term to find.

**The sign is the spiral's and is checked against it rather than derived twice.**
A spin spiral is this formula's special case with the up component at ``Q =
+q/2`` (the bra) and the down at ``Q' = -q/2`` (the ket), so ``Q' - Q`` is
``-q`` -- which is the displacement ``Calculation.at_spiral_q`` passes
(``shift = -qcart``) and the 90-degree supercell measures. The opposite spelling
gives ``+q``, is real, is correctly Hermitian and is wrong, which is this
repository's "index order in a transposed pair reads as a sign" trap in the one
line where it lives.

**What does *not* happen is the thing the old refusal was written around.** The
overlap operator does not enter: the augmentation part of
``<psi_{k0+Q}|S|psi_{k0+Q'}>`` carries the same ``sum_R e^{i(Q'-Q).R}``, and
``Q - Q'`` is a reciprocal vector of the ultracell, so that sum is
``N delta_{QQ'}``. The frozen states are exactly S-orthonormal across ``Q`` and
the eigenproblem stays an ordinary one.

**The box does the umklapp, as it does everywhere else here.** The ultracell
box's reciprocal grid *is* the ``{G + Q}`` set
(:mod:`defumat.ultracell.grid`), so the displaced table wanted at difference
``Q_d`` is read against the box coefficients at ``G + Q_d``, gathered with the
same ``J = (n G + q) mod (n Nd)`` map the wavefunctions are scattered with.
Nothing carries a phase and nothing is aligned by Miller index.

**Cost and peak.** ``N`` tables of ``nh^2 x ngm`` complex per species, which is
``N`` times the resident one. Measured on the PAW silicon this was validated on
(``ngm = 2277`` at ``ecutrho = 64`` Ry, ``nh = 8``): **4.66 MB at ``N = 2``,
9.33 at 4 and 18.65 at 8**, plus ``nat x ngm`` structure factors per table, which
is another 0.15 MB per two cells. Linear in ``N`` and in ``ngm``, so the number
to size before a slab is ``nh^2 ngm N x 16`` bytes per species. The gate that
chooses between a stored table and QE's radial one
(:data:`~defumat.pseudo.augmentation.AUG_MAX_BYTES`) is asked about
``max_bytes / N`` here rather than about ``max_bytes``, so the *total* stays
under it and a large ultracell falls to the tabulated route the way a large cell
does. Per iteration the new work is ``N`` ``newd`` integrals and one
``(N nbnd)^2`` contraction over projector channels, both far below the ``N^2``
ultracell transforms the matrix already pays.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from defumat.pseudo.augmentation import _aug_max_bytes, build_augmentation
from defumat.system.cell import Cell
from defumat.ultracell.grid import Ultracell

__all__ = [
    "UltracellAugmentation",
    "build_ultracell_augmentation",
    "ultracell_projections",
    "ultracell_deeq",
    "augmentation_matrix",
    "ultracell_becsum",
    "ultracell_augmentation_charge",
    "becsum_per_copy",
    "coefficients_per_difference",
    "as_onecentre_becsum",
    "from_onecentre_coefficients",
    "onecentre_deeq",
]


@dataclasses.dataclass(frozen=True, eq=False)
class UltracellAugmentation:
    """The ``N`` displaced tables, and the box map each one is read through.

    A plain frozen dataclass rather than an ``eqx.Module``, for
    :class:`~defumat.ultracell.grid.Ultracell`'s reason: nothing in this
    subpackage crosses a ``jit`` or a ``grad`` boundary, the atoms do not move
    and there is no force to take. The tables it holds *are* ``eqx.Module``
    instances, which is what lets them be passed straight into the jitted
    kernels below.
    """

    #: ``N`` :class:`~defumat.pseudo.augmentation.AugmentationCharge`, entry
    #: ``d`` displaced by ``Q_d`` -- the ``d``-th vector of the Q-set, in the
    #: same C-order over the integer triple :class:`Ultracell` uses.
    tables: tuple
    #: ``(N, ngm)`` flat ultracell box index of the unit cell's **dense** sphere
    #: at each difference block, ``J = (n G + q_d) mod (n Nd)``.
    box_index: jnp.ndarray
    #: ``(N, N)`` integer: entry ``[bra, ket]`` is the difference index of
    #: ``Q_ket - Q_bra``, which is the table that pair reads. It is the
    #: **transpose** of :attr:`Ultracell.difference_index`, whose ``[i, j]`` is
    #: ``Q_i - Q_j``, and the transpose is the whole sign of this module.
    difference: np.ndarray
    #: ``(N, N)`` integer: entry ``[q, d]`` is the Q index of ``Q_q + Q_d``,
    #: which is the partner ``becsum`` pairs ``Q_q`` with at difference ``d``.
    sum_index: np.ndarray
    #: Per species, the ``(nat_t, nh_t)`` projector columns of each of its
    #: atoms, or ``None`` for a norm-conserving species --
    #: ``Calculation._species_channels``.
    species_channels: tuple
    #: The ultracell's ``(n1, n2, n3)``, which is the shape the ``R``-transform
    #: of ``becsum`` runs over.
    shape: tuple
    nkb: int

    @property
    def cells(self) -> int:
        return int(len(self.tables))


def build_ultracell_augmentation(
    calculation, ultracell: Ultracell, cell: Cell
) -> UltracellAugmentation | None:
    """The displaced tables for every Q-difference. ``None`` if nothing is augmented.

    ``calculation`` is the one
    :func:`~defumat.workflows.nscf.fixed_density_states` returned, so its dense
    G set and its atoms are the ones every table has to agree with.

    The table at ``d = 0`` is built with an explicit zero shift rather than
    reusing ``calculation.augmentation``: the two hold the same numbers, but a
    table built with ``shift = None`` reports ``shift is None`` and its
    :meth:`~defumat.pseudo.augmentation.AugmentationCharge.integrals` silently
    takes a real part, so keeping the whole set uniform means one code path
    below instead of a branch that is exercised by exactly one of ``N`` terms.
    """
    if not calculation.is_ultrasoft:
        return None

    dense = calculation.basis.dense
    cells = ultracell.cells
    q_crystal = ultracell.q_crystal
    budget = max(1, _aug_max_bytes() // cells)

    tables = []
    for d in range(cells):
        shift = np.asarray(
            cell.k_to_cartesian(jnp.asarray(q_crystal[d]))
        ) * float(cell.tpiba)
        tables.append(build_augmentation(
            calculation.pseudos, calculation.system.structure, cell, dense,
            max_bytes=budget, shift=jnp.asarray(shift),
        ))

    miller = np.asarray(dense.miller)
    box_index = np.stack(
        [ultracell.box_index(miller, d) for d in range(cells)], axis=0
    )
    triples = ultracell.q_triples
    sum_index = ultracell.q_index(triples[:, None, :] + triples[None, :, :])

    return UltracellAugmentation(
        tables=tuple(tables),
        box_index=jnp.asarray(box_index),
        difference=np.asarray(ultracell.difference_index).T.copy(),
        sum_index=np.asarray(sum_index),
        species_channels=tuple(calculation.species_channels),
        shape=tuple(int(n) for n in ultracell.shape),
        nkb=int(calculation.augmentation.nkb),
    )


# -- the projections the whole module contracts against -----------------------


def ultracell_projections(coefficients, projectors, batch: int | None = 1):
    """``B^{a,Q}_i = <beta_i|psi_{k0+Q,n}>`` for every frozen state.

    Args:
        coefficients: ``(blocks, nk0, N, nbnd, npwx)`` frozen states, padding
            already zeroed.
        projectors: the folded-k :class:`~defumat.pseudo.projectors.Projectors`,
            whose k-list is ordered ``ik = ik0 * N + iq`` exactly as
            :func:`~defumat.ultracell.grid.folded_kpoints` builds it.

    Returns ``(blocks, nk0, N, nbnd, nkb)`` complex.

    Built one folded k-point at a time through
    :meth:`~defumat.pseudo.projectors.Projectors.at_k` rather than off the whole
    ``vkb`` array, which is ``nk0 N npwx nkb`` complex and is the largest thing
    this method would otherwise hold -- 1.5 GB on a 21-cell slab where the
    result is 12 MB.
    """
    blocks, nk0, cells, nbnd, npwx = coefficients.shape
    out = []
    for block in range(int(blocks)):
        rows = []
        for ik0 in range(int(nk0)):
            per_q = []
            for iq in range(int(cells)):
                vkb = projectors.at_k(int(ik0) * int(cells) + int(iq))
                per_q.append(jnp.einsum(
                    "gc,bg->bc", vkb.conj(), coefficients[block, ik0, iq]
                ))
            rows.append(jnp.stack(per_q, axis=0))
        out.append(jnp.stack(rows, axis=0))
    return jnp.stack(out, axis=0)


# -- D_ij, per Q-difference ---------------------------------------------------


def ultracell_deeq(delta_v, augmentation: UltracellAugmentation, grid):
    """``int dV(r) q~_ij(r) dr`` for every atom and every Q-difference.

    Args:
        delta_v: ``(nspin_mag, *box)`` real difference potential on the
            ultracell box, which is the same array the smooth part of the matrix
            is built from. It is *not* masked to the tiled dense sphere here and
            does not need to be: the gather below reads exactly the ``N ngm``
            components that sphere holds, which is the set ``newd`` integrates
            over in the unit cell.

    Returns ``(nspin_mag, N, nkb, nkb)`` complex, entry ``[s, d]`` the block
    matrix a bra at ``Q`` and a ket at ``Q + Q_d`` reads.

    **The conjugate is where the spiral puts it.**
    :meth:`~defumat.pseudo.augmentation.AugmentationCharge.cross_integrals` is
    ``Omega sum_G conj(q~_ij(G)) V(G)``, and what a matrix element multiplying
    ``conj(B_i) B_j`` wants is the conjugate of that -- ``newd_so``'s cross
    block is written the same way (``scf/driver.py``'s
    ``_noncollinear_coefficients``), and the other spelling is Hermitian and
    wrong.
    """
    delta_v = jnp.asarray(delta_v)
    points = int(grid[0]) * int(grid[1]) * int(grid[2])
    box = jnp.fft.fftn(delta_v, axes=(-3, -2, -1)) / points
    flat = box.reshape(delta_v.shape[0], -1)

    out = []
    for d, table in enumerate(augmentation.tables):
        gathered = flat[:, augmentation.box_index[d]]
        out.append(jnp.stack([
            jnp.conj(table.block_matrix(table.cross_integrals(gathered[s])))
            for s in range(delta_v.shape[0])
        ]))
    return jnp.stack(out, axis=1)


def coefficients_per_difference(ddd_r, shape):
    """A per-copy one-centre ``ddd`` as its ``N`` Q-difference components.

    ``(1/N) sum_R e^{i Q_d . R} ddd^R``, which is an inverse discrete transform
    over the three cell axes: the operator a bra at ``Q`` and a ket at
    ``Q + Q_d`` sees is that combination of the copies' coefficients, for the
    same reason the potential's term is the displaced table.

    ``ddd_r`` has the ``N`` copies on its leading axis, in the C-order over the
    integer triple :class:`~defumat.ultracell.grid.Ultracell` enumerates ``R``
    and ``Q`` in -- the same order, which is what makes the pair of transforms
    here and in :func:`becsum_per_copy` inverses of each other.
    """
    ddd_r = jnp.asarray(ddd_r)
    rest = ddd_r.shape[1:]
    folded = ddd_r.reshape(tuple(shape) + rest)
    return jnp.fft.ifftn(folded, axes=(0, 1, 2)).reshape((-1,) + rest)


def becsum_per_copy(becsum_q, shape):
    """A Q-resolved ``becsum`` as the ``N`` copies' own real occupations.

    ``becsum^R = sum_{Q_d} e^{i Q_d . R} becsum(Q_d)``, the inverse of
    :func:`coefficients_per_difference` up to the factor ``N`` the two
    conventions differ by. What comes back is **real** to round-off -- a
    projector occupation matrix is Hermitian on every copy -- and a complex
    result is a wrong index order caught for nothing, so the imaginary part is
    returned beside it rather than discarded.

    Returns ``(values, residual)`` with ``values`` real and ``residual`` the
    largest imaginary part that was dropped.
    """
    becsum_q = jnp.asarray(becsum_q)
    cells = int(np.prod(shape))
    rest = becsum_q.shape[1:]
    folded = becsum_q.reshape(tuple(shape) + rest)
    copies = jnp.fft.ifftn(folded, axes=(0, 1, 2)).reshape((-1,) + rest) * cells
    return jnp.real(copies), float(jnp.max(jnp.abs(jnp.imag(copies))))


# -- the matrix element -------------------------------------------------------


@jax.jit
def augmentation_matrix(becp, deeq, difference):
    """``sum_{a,ij} conj(B^{Q,m}_i) D_ij(Q' - Q) B^{Q',n}_j``.

    Args:
        becp: ``(N, nbnd, nkb)`` the projections of one ``k0``'s frozen states.
        deeq: ``(N, nkb, nkb)`` this spin channel's coefficients, indexed by
            difference.
        difference: ``(N, N)`` the difference index of ``Q_ket - Q_bra``.

    Returns ``(N nbnd, N nbnd)``, in the ``(Q, n)`` C-order
    :func:`~defumat.ultracell.hamiltonian.ultracell_matrix` lays its rows out
    in, so it is added to that matrix and nothing is reordered.

    Hermitian by construction and not only by symmetrisation: the displaced
    tables obey ``D(-Q_d)_ji = conj(D(Q_d)_ij)`` because ``Q_ij(r)`` is real
    and symmetric in its channel pair. That is worth knowing and is **not** a
    check of the sign -- the opposite index order is Hermitian too, which is why
    only a supercell can tell them apart.
    """
    gathered = deeq[difference]
    return jnp.einsum(
        "qmk,qQkl,Qnl->qmQn", becp.conj(), gathered, becp, optimize=True,
    ).reshape(becp.shape[0] * becp.shape[1], -1)


# -- becsum and the charge it puts on the box ---------------------------------


def ultracell_becsum(becp, vectors, weights, augmentation: UltracellAugmentation):
    """``becsum`` resolved by Q-difference, per species.

    Args:
        becp: ``(N, nbnd, nkb)`` the projections at one ``k0``.
        vectors: ``(N nbnd, nstate)`` the envelope amplitudes, column ``j`` the
            state, in the ``(Q, n)`` C-order the matrix uses.
        weights: ``(nstate,)`` ``w_k0 f_j / N`` -- the same weights
            :func:`~defumat.ultracell.density.ultracell_density` divides by
            ``N``, because a copy's occupation matrix is built from the
            **ultracell-normalised** state and the frozen states are normalised
            over the unit cell.

    Returns one ``(N, nat_t, nh_t, nh_t)`` complex array per species, or
    ``None`` where the species is norm-conserving.

    The per-copy occupations are ``becsum^R = sum_d e^{i Q_d . R} becsum(Q_d)``
    (:func:`becsum_per_copy`), so this is the envelope of the projector
    occupations exactly as the density on the box is the envelope of the
    charge, and at ``Q_d = 0`` it is the unit cell's own ``becsum``.

    **Only the symmetric part of each block survives**, so it is imposed here.
    ``Q_ij`` is symmetric in its channel pair and the one-centre tensors are
    too, so the antisymmetric part contracts to nothing downstream; imposing it
    is what makes the per-copy matrix come back real rather than merely nearly
    real.
    """
    cells, nbnd, _ = becp.shape
    amplitudes = jnp.asarray(vectors).reshape(cells, nbnd, -1)
    weights = jnp.asarray(weights).astype(becp.dtype)
    sum_index = augmentation.sum_index

    values = []
    for channels in augmentation.species_channels:
        if channels is None:
            values.append(None)
            continue
        columns = becp[:, :, jnp.asarray(channels)]         # (N, nbnd, nat, nh)
        mixed = jnp.einsum("qnj,qnai->qjai", amplitudes, columns)
        blocks = []
        for d in range(cells):
            partner = mixed[jnp.asarray(sum_index[:, d])]
            block = jnp.einsum(
                "j,qjai,qjak->aik", weights, mixed.conj(), partner, optimize=True,
            )
            blocks.append(0.5 * (block + jnp.swapaxes(block, -1, -2)))
        values.append(jnp.stack(blocks, axis=0))
    return tuple(values)


def ultracell_augmentation_charge(
    becsum_q, augmentation: UltracellAugmentation, grid, nspin: int
):
    """The augmentation charge of every atom copy, on the ultracell box.

    Args:
        becsum_q: per species, ``(N, nspin, nat_t, nh_t, nh_t)`` complex, or
            ``None``.

    Returns ``(field, residual)``: ``(nspin, *grid)`` real, to be added to the
    smooth density, and the largest imaginary part that was dropped relative to
    the field's own scale.

    ``addusdens`` with the ultracell's index map in place of the unit cell's:
    the charge at difference ``Q_d`` is the displaced table contracted with
    ``becsum(Q_d)``, and it lands at the box slots ``n G + q_d``. Summed over
    ``d`` the result is real, because ``becsum(-Q_d)`` is the conjugate of
    ``becsum(Q_d)`` on a block symmetric in its channel pair and the tables obey
    the matching relation.

    **The imaginary part is reported rather than merely dropped**, which
    :func:`_addusdens` can afford not to do and this cannot: there the relation
    that makes it zero is ``Q_ij(-G) = conj(Q_ij(G))`` on one table, and here it
    is a relation *between* ``N`` different displaced tables and ``N``
    components of ``becsum``. Dropping it silently would make a wrong pairing
    across ``d`` look like a slightly different density, which is the shape of
    error this repository calls a check whose null result cannot be told from a
    pass.
    """
    points = int(grid[0]) * int(grid[1]) * int(grid[2])
    dtype = augmentation.tables[0].phases.dtype
    box = jnp.zeros((int(nspin), points), dtype=dtype)
    for d, table in enumerate(augmentation.tables):
        index = augmentation.box_index[d]
        charge = jnp.stack([
            table.charge(tuple(
                None if b is None else b[d, s] for b in becsum_q
            ))
            for s in range(int(nspin))
        ])
        box = box.at[:, index].add(charge)
    field = jnp.fft.ifftn(
        box.reshape((int(nspin),) + tuple(grid)), axes=(-3, -2, -1)
    ) * points
    real = jnp.real(field)
    scale = float(jnp.max(jnp.abs(real)))
    residual = float(jnp.max(jnp.abs(jnp.imag(field)))) / max(scale, 1.0e-30)
    return real, residual


# -- PAW: the one-centre terms, one set per atom copy --------------------------


def as_onecentre_becsum(per_copy) -> tuple:
    """The copies' occupations in the layout the one-centre routine takes.

    ``(N, nspin, nat_t, nh, nh)`` per species becomes
    ``(nspin, N nat_t, nh, nh)``: the one-centre machinery reads its atom count
    off ``becsum.shape[1]`` and knows nothing about positions
    (``paw/onecenter.py``'s ``build_paw`` takes the structure only for its
    species labels), so ``N`` copies of every atom are simply ``N nat_t`` atoms
    of that species and nothing in it changes.

    The copy index is the **slower** of the two, so entry ``R nat_t + a`` is
    atom ``a`` in cell ``R``, and :func:`from_onecentre_coefficients` undoes
    exactly this. The pair is written together because the ordering is the one
    thing a wrong answer here would not announce: ``ddd`` landing in another
    copy's channels is a converged, plausible and wrong modulation.
    """
    out = []
    for values in per_copy:
        if values is None:
            out.append(None)
            continue
        values = jnp.asarray(values)
        cells, nspin, nat, nh = values.shape[0], values.shape[1], values.shape[2], values.shape[3]
        out.append(jnp.moveaxis(values, 0, 1).reshape(nspin, cells * nat, nh, nh))
    return tuple(out)


def from_onecentre_coefficients(ddd, cells: int) -> tuple:
    """:func:`as_onecentre_becsum` inverted: back to ``(N, nspin, nat_t, nh, nh)``."""
    out = []
    for values in ddd:
        if values is None:
            out.append(None)
            continue
        values = jnp.asarray(values)
        nspin, total, nh = values.shape[0], values.shape[1], values.shape[2]
        out.append(jnp.moveaxis(
            values.reshape(nspin, int(cells), total // int(cells), nh, nh), 1, 0
        ))
    return tuple(out)


def _blocks_like(table, values, dtype) -> tuple:
    """Per-species blocks with a uniform dtype, zero where a species has none.

    :meth:`~defumat.pseudo.augmentation.AugmentationCharge.block_matrix` reads
    its dtype off the *first* block, so a norm-conserving species sitting first
    would decide it. ``cross_integrals`` solves this by returning complex zeros
    of the right shape and this does the same, which is why the two can be
    added before either is assembled.
    """
    out = []
    for t, (q, atoms) in enumerate(zip(table.qgm, table.species_atoms)):
        nh = int(q.shape[0])
        entry = None if values is None else values[t]
        if entry is None or nh == 0 or not atoms:
            out.append(jnp.zeros((len(atoms), nh, nh), dtype=dtype))
        else:
            out.append(jnp.asarray(entry).astype(dtype))
    return tuple(out)


def onecentre_deeq(ddd_per_copy, reference_ddd, augmentation, nspin: int):
    """PAW's one-centre coefficients as a ``(nspin, N, nkb, nkb)`` difference.

    Args:
        ddd_per_copy: per species, ``(N, nspin, nat_t, nh, nh)`` real -- the
            **full** one-centre coefficients of every copy, at the ``becsum``
            the matrix is being built from.
        reference_ddd: the unit cell's own, per species ``(nspin, nat_t, nh,
            nh)``. It is subtracted at ``Q_d = 0`` and nowhere else, because
            that is the only difference component a lattice-periodic quantity
            has, and the frozen eigenvalues carry it already.

    The transform to Q-difference space is
    :func:`coefficients_per_difference`, and what comes out is complex even
    though every copy's coefficients are real -- a modulation of a real
    quantity has complex Fourier components, and the ``d`` and ``-d`` entries
    are conjugate transposes of each other, which is what keeps the matrix
    Hermitian.
    """
    table = augmentation.tables[0]
    cells = augmentation.cells
    dtype = table.phases.dtype
    per_difference = tuple(
        None if values is None else coefficients_per_difference(
            values, augmentation.shape)
        for values in ddd_per_copy
    )

    out = []
    for d in range(cells):
        rows = []
        for s in range(int(nspin)):
            values = tuple(
                None if entry is None else entry[d, s] for entry in per_difference
            )
            blocks = _blocks_like(table, values, dtype)
            if d == 0 and reference_ddd is not None:
                subtract = _blocks_like(
                    table,
                    tuple(None if e is None else e[s] for e in reference_ddd),
                    dtype,
                )
                blocks = tuple(a - b for a, b in zip(blocks, subtract))
            rows.append(table.block_matrix(blocks))
        out.append(jnp.stack(rows))
    return jnp.stack(out, axis=1)
