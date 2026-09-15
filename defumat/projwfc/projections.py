"""``<phi_i|S|psi_nk>``: the Kohn-Sham states on the pseudo-atomic basis.

``PP/src/projwfc.f90``'s ``projwave``. Everything a projected density of states
or a Löwdin charge is made of is this one matrix,

    proj0[i, n] = <phi_i| S |psi_n>,   proj[i, n] = |proj0[i, n]|^2

with ``phi`` the Löwdin-orthogonalised pseudo-atomic orbitals of the crystal.
The construction of ``phi`` is not repeated here -- it is
:func:`defumat.hubbard.projectors.build_atomic_projectors`, the same function
DFT+U's ``wfcU`` comes out of, because ``orthoUwfc`` and ``projwave`` build the
same object and having two of them is how they come to disagree.

**``S`` is applied even for** ``atomic`` **projectors.** ``projwave`` calls
``s_psi`` on ``wfcatom`` unconditionally, before it does anything else with it;
so does ``orthoUwfc``. This is the same silent trap P20 records, and it has the
same tell: with a norm-conserving dataset ``S`` is the identity and nothing
distinguishes the two, so a test on silicon cannot find it.

**The default projector set is** ``ortho-atomic``, because that is the *only*
one ``projwfc.x`` has -- it diagonalises ``O_ij = <phi_i|S|phi_j>`` over all
``natomwfc`` orbitals and projects onto ``O^{-1/2} S phi``. ``atomic`` and
``norm-atomic`` are offered here as well (they are ``pw.x``'s
``Hubbard_projectors`` choices), and they are a *different* decomposition: they
do not sum to one over a complete shell and their "spilling" is not
Sanchez-Portal's.

**Symmetrisation.** ``lsym = .true.`` is ``projwfc.x``'s default and it is not
cosmetic on a reduced k-set: what it averages is

    proj[i] = 1/nsym sum_S | sum_m' D^l_S[m', m] proj0[S(a), n, l, m'] |^2

(``sym_proj_k``), which is the group average of the *squared* projection, atom
index following ``irt`` and the ``m`` mixing given by the matrices that rotate
real spherical harmonics -- :func:`defumat.paw.symmetry.harmonic_rotations`,
which is this project's ``d_matrix``. Without it silicon's three ``p`` channels
come out unequal at a single k-point, exactly as ``becsum`` does (P12), and the
per-``m`` Löwdin charges are wrong while their sum is right.

Memory: the projector functions are ``(nk, npwx, natomwfc)`` complex, the same
shape and the same argument as DFT+U's ``wfcU``; the projections themselves are
``(nspin, nk, natomwfc, nbnd)`` real, which is negligible beside the
wavefunctions they are made from.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from defumat.basis.gvectors import refuse_gamma_storage

from defumat.batching import map_k
from defumat.hubbard.projectors import build_atomic_projectors
from defumat.paw.symmetry import harmonic_rotations
from defumat.projwfc.channels import AtomicChannel, projection_channels
from defumat.system.symmetry import atom_mapping, spin_rotations

__all__ = [
    "ProjectionSymmetry",
    "build_projection_symmetry",
    "atomic_projections",
    "calculation_channels",
    "PROJECTION_KINDS",
]


def calculation_channels(calculation) -> tuple[AtomicChannel, ...]:
    """The projection's label table for whichever spin regime this run is in.

    One function rather than the two lines repeated, because the labels and the
    *orbitals* have to agree column for column and they are built in different
    modules: a second copy of the regime test is how they come to disagree.
    """
    system = calculation.system
    return projection_channels(
        calculation.pseudos,
        system.structure,
        bool(system.noncolin),
        bool(getattr(system, "lspinorb", False)),
    )

#: The projector sets a projection can be made onto. ``projwfc.x`` has only the
#: first; the other two are ``pw.x``'s ``Hubbard_projectors`` spellings and
#: reach the same code path.
PROJECTION_KINDS = ("ortho-atomic", "atomic", "norm-atomic")


class ProjectionSymmetry(eqx.Module):
    """``sym_proj_k``, precomputed as a gather and a set of coefficients.

    ``indices[s, c, j]`` is the projection column that operation ``s`` draws on
    for output column ``c``, and ``coefficients[s, c, j]`` is ``D^l_s[m', m]``.
    Columns of a shell shorter than ``2 lmax + 1`` are padded with a zero
    coefficient, so every ``l`` runs through the same contraction and no shape
    depends on which shells a crystal happens to have (rule R7).
    """

    indices: jnp.ndarray  # (nsym, natomwfc, mmax), int
    coefficients: jnp.ndarray  # (nsym, natomwfc, mmax): real, or complex
    # on a spinor run, where the coefficient is ``D^l x U`` and ``mmax``
    # covers both spin halves of a shell.
    nsym: int = eqx.field(static=True)

    def apply(self, proj0: jnp.ndarray) -> jnp.ndarray:
        """``(natomwfc, nbnd)`` complex in, ``(natomwfc, nbnd)`` real out."""
        # The sum over ``m'`` is walked rather than gathered in one go: the
        # gathered array would be (nsym, natomwfc, mmax, nbnd) complex, and
        # ``mmax`` steps of (nsym, natomwfc, nbnd) is the same arithmetic with
        # ``mmax`` times less of it resident.
        work = jnp.zeros(
            (self.nsym,) + proj0.shape, dtype=proj0.dtype
        )
        for j in range(self.indices.shape[-1]):
            work = work + (
                self.coefficients[:, :, j, None] * proj0[self.indices[:, :, j]]
            )
        return jnp.sum(jnp.abs(work) ** 2, axis=0) / self.nsym


def build_projection_symmetry(
    channels: tuple[AtomicChannel, ...], cell, structure, symmetries
) -> ProjectionSymmetry | None:
    """The tables :class:`ProjectionSymmetry` contracts through.

    ``None`` when the group is trivial, in which case the symmetrisation is the
    identity and is skipped rather than multiplied out.

    **The spinor branch is the same contraction on a longer index.** A
    noncollinear column without spin-orbit coupling is ``|l m> x |sigma>``, and
    the operator ``sym_proj_nc`` averages over is the tensor product
    ``D^l x U`` -- the harmonic rotation this function already builds, times the
    2x2 spinor rotation :func:`~defumat.system.symmetry.spin_rotations`
    validated in P82. So the shell's block grows from ``2l+1`` columns to
    ``2 (2l+1)``, in the order the orbitals are built (every ``m`` up, then
    every ``m`` down), the coefficients become complex, and nothing else about
    the contraction changes.

    **The spin factor is taken from** ``spin_rotations`` **rather than from**
    ``d_matrix_nc``, for the reason P82 gives about ``d_spin_ldau``, and here
    there is a second reason: ``d_matrix_nc`` is not consistent with itself.
    Its ``l = 0`` block is ``conjg(s_spin(n1, m1))`` where every ``l > 0`` block
    is ``s_spin(m1, n1)``, so the two differ by a conjugation of the spinor
    matrix, and a cell with only ``s`` channels cannot tell them apart. What
    pins the convention here is a property instead: on a closed grid the
    projection at ``S k`` must equal the rotated projection at ``k``, multiplet
    by multiplet (``tests/unit/test_spinor_projection_symmetry.py``).

    **Time reversal needs no index relabelling.** ``sym_proj_nc`` swaps the
    output column between the two spin halves by hand
    (``ind = 2 m - ind0 + 2 l + 1``); here that swap is already inside
    ``spin_rotations``, which carries ``i sigma_y D*`` for an operation that is
    a symmetry only with time reversal, and the conjugation the antiunitary
    half asks for is applied to the matrix rather than to the projection, which
    is the same thing under ``|.|^2``.
    """
    if symmetries is None or symmetries.nsym <= 1 or not channels:
        return None

    spinor = any(channel.s_z is not None for channel in channels)
    lmax = max(channel.l for channel in channels)
    rotations = harmonic_rotations(cell, symmetries, lmax)
    mapping = atom_mapping(cell, structure, symmetries)
    nsym = symmetries.nsym
    npol = 2 if spinor else 1
    mmax = npol * (2 * lmax + 1)
    spins = spin_rotations(cell, symmetries) if spinor else None

    # Where each (atom, wfc, l) shell starts among the columns, so that the
    # image shell can be found by its key rather than by ``sym_proj_k``'s linear
    # search for "the same atom, n and l with m = 1".
    first = {}
    for channel in channels:
        first.setdefault((channel.atom, channel.wfc, channel.l), channel.index)

    indices = np.zeros((nsym, len(channels), mmax), dtype=int)
    coefficients = np.zeros(
        (nsym, len(channels), mmax), dtype=complex if spinor else float
    )
    for channel in channels:
        block = rotations[channel.l]  # (nsym, 2l+1, 2l+1)
        width = 2 * channel.l + 1
        # ``s_z`` is +1/2 on the first half of a shell's columns and -1/2 on the
        # second, which is the order the orbitals themselves are built in.
        sigma = 0 if (channel.s_z is None or channel.s_z > 0.0) else 1
        for s in range(nsym):
            image = first[(int(mapping[s, channel.atom]), channel.wfc, channel.l)]
            for sigma1 in range(npol):
                # ``conj(U)`` rather than ``U``, and it is not a taste: the
                # harmonic factor is contracted on its *first* index, so what
                # multiplies the projection is ``D^T = D^{-1}`` -- the inverse
                # operation -- and the spin factor has to be the inverse of the
                # same operation, which for a unitary ``U`` contracted the same
                # way is ``conj(U)``, since ``conj(U)^T = U^dagger``. Measured,
                # not derived and hoped for: on nickel with its moment along a
                # three-fold axis the other three arrangements are wrong by
                # 2.1e-2 to 3.9e-2 electrons where this one is 1.1e-5.
                weight = 1.0 if spins is None else np.conj(spins[s, sigma1, sigma])
                for m1 in range(width):
                    column = m1 + width * sigma1
                    indices[s, channel.index, column] = image + column
                    coefficients[s, channel.index, column] = (
                        block[s, m1, channel.m] * weight
                    )
            # The padding columns gather from the shell's own first index with a
            # zero weight: a valid index keeps the gather in bounds and the zero
            # keeps it out of the answer.
            indices[s, channel.index, npol * width :] = image

    return ProjectionSymmetry(
        indices=jnp.asarray(indices),
        coefficients=jnp.asarray(
            coefficients,
            dtype=cell.precision.complex if spinor else cell.precision.real,
        ),
        nsym=nsym,
    )


def atomic_projections(
    calculation,
    wavefunctions: jnp.ndarray,
    kind: str = "ortho-atomic",
    symmetrize: bool = True,
) -> np.ndarray:
    """``(nspin, nk, natomwfc, nbnd)``: ``|<phi|S|psi>|^2``, symmetrised.

    ``wavefunctions`` is ``(nspin, nk, nbnd, npwx)`` -- an
    :class:`~defumat.scf.driver.SCFResult`'s or an NSCF run's, on the k-points
    ``calculation`` was built with. Nothing is diagonalised here: ``projwfc.x``
    reads the states a ``pw.x`` run left behind and so does this.

    ``symmetrize`` asks for ``sym_proj_k``'s average over the point group, and
    it is **and**-ed with :attr:`~defumat.scf.driver.Calculation.use_symmetry`:
    a run that set ``nosym`` did not use those operations, so averaging over
    them here averages a quantity the states do not share. ``projwfc.x`` gets
    this for free -- it reaches ``sym_proj_k`` through ``nsym``, which
    ``setup.f90`` has already collapsed to 1 -- and this code has the group
    whole beside a switch, so it has to make the test. It is the shape of the
    ``dielectric_tensor``-symmetrising-a-``nosym``-run defect ``PLAN.md`` P28b
    found, in a second place; the failure is silent both times, because an
    average over the wrong group is still a smooth, normalised, plausible
    projection.
    """
    if kind not in PROJECTION_KINDS:
        raise ValueError(
            f"unknown projector set {kind!r}; expected one of {PROJECTION_KINDS}"
        )
    refuse_gamma_storage(
        bool(getattr(calculation, "gamma_only", False)),
        "the projected density of states",
        "<phi|S|psi> here is a plain sum over the stored k + G list, so a "
        "Loewdin charge comes out at roughly a quarter of its value",
    )
    system = calculation.system
    noncolin = bool(system.noncolin)
    lspinorb = bool(getattr(system, "lspinorb", False))
    if lspinorb and symmetrize and calculation.use_symmetry and (
        calculation.symmetries is not None and calculation.symmetries.nsym > 1
    ):
        # A spin-angle function carries a spin frame that the operation turns,
        # so the group average needs a spin matrix beside the rotation of the
        # harmonics; averaging the ``m`` indices alone mixes columns across a
        # frame that has moved, which is a smooth, normalised, plausible and
        # wrong projection.
        #
        # **The two regimes need two different matrices, and only one of them is
        # still missing.** Without spin-orbit coupling the columns are
        # ``|l m> x |sigma>`` and ``sym_proj_nc``'s operator is the tensor
        # product ``D^l x U`` (``PP/src/d_matrix_nc.f90`` builds exactly
        # ``dy_l(m,n) * s_spin(m1,n1)``), whose two factors are both here --
        # :func:`~defumat.paw.symmetry.harmonic_rotations` and
        # :func:`~defumat.system.symmetry.spin_rotations` -- and
        # :func:`build_projection_symmetry` assembles them.
        #
        # With spin-orbit coupling the columns are ``|j m_j>`` instead, and
        # ``sym_proj_so`` contracts ``d_matrix_so``'s ``D^j`` for
        # ``j = 1/2, 3/2, 5/2, 7/2`` -- a **different** matrix that nothing here
        # builds, and it is not the tensor product above. It is not reachable
        # by relabelling the one above either: the ``j`` shells of one ``l``
        # have different dimensions and mix under a rotation only within
        # themselves.
        raise NotImplementedError(
            "a symmetrised projection is not implemented for a spin-orbit "
            "run: the columns are |j m_j> and sym_proj_so contracts "
            "d_matrix_so's D^j, which is a different matrix from the D^l x U "
            "the noncollinear branch uses and is not built here. Run with "
            "nosym = .true. and the whole k-grid, which is the same physics, "
            "or pass symmetrize=False"
        )
    channels = calculation_channels(calculation)
    if not channels:
        raise ValueError(
            "none of the pseudopotentials carries an atomic orbital to project "
            "on -- projwave refuses the same way ('Cannot project on zero "
            "atomic wavefunctions')"
        )

    projectors = build_atomic_projectors(
        calculation.pseudos,
        system.structure,
        system.cell,
        calculation.basis.smooth,
        calculation.basis.planewaves,
        calculation.basis_kpoints,
        # ``s_psi`` written against the projectors alone, exactly as the Hubbard
        # projectors reach it -- there is no Hamiltonian in a projection. The
        # spinor branch is a **different operator** and not the same one on a
        # longer vector: ``_spinor_overlap`` carries ``qq_so``, whose off-
        # diagonal spin blocks are exactly what tells the two ``j`` channels
        # apart, so contracting each component against the scalar ``qq`` would
        # give the j-averaged overlap. ``_build_hubbard_projectors`` picks
        # between them the same way (``scf/driver.py:1512``).
        calculation._spinor_overlap if noncolin else calculation._overlap,
        kind=kind,
        noncolin=noncolin,
        # ``atomic_wfc_nc_proj``'s ``starting_spin_angle = .TRUE.``: the
        # projection is onto the spin-angle functions themselves, where the SCF
        # and DFT+U start from the j-averaged up/down set. Without spin-orbit
        # coupling the two coincide -- there is no j to average.
        spinor_basis="jmj" if lspinorb else "updown",
    )  # (nk, npol npwx, natomwfc)

    symmetry = (
        build_projection_symmetry(
            channels, system.cell, system.structure, calculation.symmetries
        ) if symmetrize and calculation.use_symmetry else None
    )

    def one_kpoint(state):
        phi, psi = state
        proj0 = jnp.einsum("gi,bg->ib", jnp.conj(phi), psi)
        if symmetry is None:
            return jnp.abs(proj0) ** 2
        return symmetry.apply(proj0)

    # One spin channel at a time, and the k axis walked by the calculation's own
    # batching dial inside each -- the same shape ``sum_band`` has (rule R6).
    return np.stack([
        np.asarray(map_k(one_kpoint, (projectors, states), batch=calculation.k_batch))
        for states in jnp.asarray(wavefunctions)
    ])
