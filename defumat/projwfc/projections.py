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

from defumat.batching import map_k
from defumat.hubbard.projectors import build_atomic_projectors
from defumat.paw.symmetry import harmonic_rotations
from defumat.projwfc.channels import AtomicChannel, projection_channels
from defumat.system.symmetry import atom_mapping

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
    coefficients: jnp.ndarray  # (nsym, natomwfc, mmax), real
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
    """
    if symmetries is None or symmetries.nsym <= 1 or not channels:
        return None

    lmax = max(channel.l for channel in channels)
    rotations = harmonic_rotations(cell, symmetries, lmax)
    mapping = atom_mapping(cell, structure, symmetries)
    nsym = symmetries.nsym
    mmax = 2 * lmax + 1

    # Where each (atom, wfc, l) shell starts among the columns, so that the
    # image shell can be found by its key rather than by ``sym_proj_k``'s linear
    # search for "the same atom, n and l with m = 1".
    first = {}
    for channel in channels:
        first.setdefault((channel.atom, channel.wfc, channel.l), channel.index)

    indices = np.zeros((nsym, len(channels), mmax), dtype=int)
    coefficients = np.zeros((nsym, len(channels), mmax))
    for channel in channels:
        block = rotations[channel.l]  # (nsym, 2l+1, 2l+1)
        for s in range(nsym):
            image = first[(int(mapping[s, channel.atom]), channel.wfc, channel.l)]
            for m1 in range(2 * channel.l + 1):
                indices[s, channel.index, m1] = image + m1
                coefficients[s, channel.index, m1] = block[s, m1, channel.m]
            # The padding columns gather from the shell's own first index with a
            # zero weight: a valid index keeps the gather in bounds and the zero
            # keeps it out of the answer.
            indices[s, channel.index, 2 * channel.l + 1 :] = image

    return ProjectionSymmetry(
        indices=jnp.asarray(indices),
        coefficients=jnp.asarray(coefficients, dtype=cell.precision.real),
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
    system = calculation.system
    noncolin = bool(system.noncolin)
    lspinorb = bool(getattr(system, "lspinorb", False))
    if noncolin and symmetrize and calculation.use_symmetry and (
        calculation.symmetries is not None and calculation.symmetries.nsym > 1
    ):
        # ``sym_proj_so`` averages the projection over the group with the
        # **SU(2)** representation of each operation beside the rotation of the
        # harmonics, because a spin-angle function carries a spin frame that the
        # operation turns. Nothing here builds those matrices -- DFT+U with
        # noncolin refuses in the same place and for the same reason
        # (``scf/driver.py``, ``d_spin_ldau``) -- and averaging the ``m``
        # indices alone would mix ``m_j`` across a frame that has moved, which
        # is a smooth, normalised, plausible and wrong projection.
        raise NotImplementedError(
            "a symmetrised projection is not implemented for a noncollinear or "
            "spin-orbit run: sym_proj_so needs the SU(2) representation of each "
            "point-group operation beside the rotation of the harmonics, and "
            "nothing here builds those. Run with nosym = .true. and the whole "
            "k-grid, which is the same physics, or pass symmetrize=False"
        )
    if noncolin and not lspinorb:
        # The *orbitals* for this branch are built (``atomic_wfc_nc``, an up and
        # a down copy of each harmonic -- ``_updown_matrix``), and the labels
        # carry their ``s_z``. What is not here is ``partialdos_nc``'s layout for
        # it: that branch has ``nspin0 = 2`` and routes each column into an up or
        # a down channel by ``ind <= 2l+1``, where this package's ``compute_pdos``
        # would bin all of them as one. No reference was generated for it either,
        # so it is refused rather than shipped as a plausible decomposition --
        # the same rule the rest of the package follows.
        raise NotImplementedError(
            "a projected density of states for a noncollinear run without "
            "spin-orbit coupling is not implemented: the spin-angle orbitals are "
            "built, but partialdos_nc splits such a run's columns into up and "
            "down densities of states (nspin0 = 2) and nothing here does that. "
            "lspinorb = .true. is implemented and validated against projwfc.x"
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
