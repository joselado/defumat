"""``<L>``, ``<S>`` and ``<J>`` per atom: where the orbital moment actually sits.

Elk's tasks 15 and 16 (``writelsj.f90``, ``dmatls.f90``, ``gendmat.f90``).
``pw.x`` has the *cell's* orbital magnetization by the modern theory
(``PW/src/orbm_kubo.f90``, reached by ``lorbm``) and nothing resolved by atom,
so this is the site decomposition rather than a second route to the same number.

The construction is one site density matrix per atom and per shell,

    rho^a_{(m s),(m' s')} = sum_{n k} wg_{nk} c^a_{m s,nk} conj(c^a_{m' s',nk}),
    c^a_{m s,nk} = <phi^a_m | S | psi_{nk}>^s,

and then the two expectation values ``dmatls`` takes of it,

    <L_i> = sum_s Tr_m ( L_i rho_{ss} ),      <S_i> = (1/2) sum_m Tr_s ( sigma_i rho_{mm} ),

with ``J = L + S``. Everything before the traces already existed: the orbitals
are :func:`defumat.hubbard.projectors.build_atomic_projectors`'s -- the same
Löwdin-orthogonalised set ``projwfc.x`` projects on and DFT+U measures its
occupations with -- and ``S`` is the calculation's own overlap operator.

**``L`` has to be written in the basis the code actually uses, and that is the
real one.** ``L_z`` is diagonal on ``Y_lm`` and is not diagonal on the real
harmonics ``ylmr2`` builds, so the matrices come from conjugating the complex
ones with ``rot_ylm`` -- the unitary
:mod:`defumat.pseudo.spinorbit` already builds for ``fcoef``:

    L^real = A^T L^complex conj(A),   A = rot_ylm restricted to the shell.

The result is purely imaginary and antisymmetric, which is not a convention but
a consequence -- ``L`` is Hermitian and the real harmonics are real -- and is
therefore the cheapest test that the transform is right. ``[L_x, L_y] = i L_z``
is the other one.

**What the three spin regimes contribute.** ``nspin = 1`` has no spin structure
in the wavefunction at all, so ``<S>`` is identically zero and the weights carry
the factor of two; ``nspin = 2`` fills the two diagonal spin blocks from the two
channels and has no cross terms, so ``<S>`` is ``(n_up - n_down)/2`` along the
quantisation axis; ``nspin = 4`` fills the whole ``2 x 2``, which is the only
regime where ``<S>`` can point anywhere. **``<L>`` is quenched to zero without
spin-orbit coupling** -- there is nothing in a scalar-relativistic Hamiltonian
to lock the orbital moment to the lattice, and the states can be chosen real --
so a vanishing ``<L>`` on a collinear magnet is the headline check here, in the
"vanishes pointwise" style of P47's silicon curvature, rather than a tolerance
someone chose.

**A projector expectation value is not a muffin-tin one, and the difference is
definitional.** Elk integrates over a sphere of a stated radius; this projects
onto an orbital set. They are the same kind of quantity in the sense Löwdin and
muffin-tin charges are the same kind of quantity, and they do not agree
digit for digit for the same reason. Trends, sum rules and the analytic limits
are what this is validated on; a comparison against Elk's number is a comparison
of two decompositions.

**Refused by name.** A *symmetry-reduced* k-set: ``<L>`` and ``<S>`` are
vectors, and a sum over a wedge is a sum over a wedge -- the axial-vector
symmetrisation P24 records for a response, one index up and with ``det(R)`` on
top, is not written here. An unshifted whole grid is closed under the point
group and is the escape, the same one ``dielectric_tensor`` documents. And a
**fully-relativistic ultrasoft or PAW** dataset, because the spinor overlap's
off-diagonal spin blocks are ``qq_so`` and the projection built here applies the
*scalar* ``S`` to each component -- which is the validated ``projwfc.x`` path in
every other regime and is missing a term in that one.

Memory: the orbitals are ``(nk, npwx, natomwfc)`` complex, the same array a
projected density of states or a Hubbard ``U`` already holds; the site matrices
are ``(natom, nshell, 2l+1, 2, 2l+1, 2)``, which is nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from defumat.pseudo.spinorbit import LMAXX, rot_ylm

__all__ = [
    "SiteAngularMomentum",
    "AtomicMoments",
    "angular_momenta",
    "orbital_matrices",
    "PAULI",
]

#: ``sigma_x``, ``sigma_y``, ``sigma_z``, in the order the vectors are reported.
PAULI = np.array([
    [[0.0, 1.0], [1.0, 0.0]],
    [[0.0, -1.0j], [1.0j, 0.0]],
    [[1.0, 0.0], [0.0, -1.0]],
], dtype=complex)


def orbital_matrices(l: int) -> np.ndarray:
    """``(3, 2l+1, 2l+1)``: ``L_x``, ``L_y``, ``L_z`` on QE's real harmonics.

    In units of ``hbar``. Built by conjugating the complex-basis matrices with
    :func:`defumat.pseudo.spinorbit.rot_ylm`, so the ordering of the ``m``
    index is ``ylmr2``'s -- ``m = 0`` first, then the cosine and sine of each
    ``|m|`` -- which is the ordering
    :func:`defumat.projwfc.channels.projection_channels` labels.
    """
    if l < 0 or l > LMAXX:
        raise ValueError(f"l must be in 0..{LMAXX}, got {l}")
    size = 2 * l + 1
    if l == 0:
        return np.zeros((3, 1, 1), dtype=complex)

    m = np.arange(-l, l + 1)
    lz = np.diag(m.astype(float)).astype(complex)
    # ``L_+ |l,m> = sqrt(l(l+1) - m(m+1)) |l,m+1>`` -- the raised state sits one
    # row up, so the coefficient goes on the sub-diagonal of the column it came
    # from.
    plus = np.zeros((size, size), dtype=complex)
    minus = np.zeros((size, size), dtype=complex)
    for index, mm in enumerate(m):
        if index + 1 < size:
            plus[index + 1, index] = np.sqrt(l * (l + 1) - mm * (mm + 1))
        if index - 1 >= 0:
            minus[index - 1, index] = np.sqrt(l * (l + 1) - mm * (mm - 1))
    complex_basis = np.stack([
        0.5 * (plus + minus),
        (plus - minus) / (2.0j),
        lz,
    ])

    # ``rot_ylm`` is built once at ``LMAXX`` and indexed relative to its centre
    # row, which is what lets one matrix serve every shell.
    u = rot_ylm(LMAXX)
    a = u[LMAXX - l:LMAXX + l + 1, :size]
    return np.stack([a.T @ matrix @ a.conj() for matrix in complex_basis])


@dataclass(frozen=True)
class AtomicMoments:
    """``<L>``, ``<S>`` and ``<J>`` on one atom, in units of ``hbar``."""

    atom: int  # 0-based
    species: str
    #: ``(3,)`` cartesian, summed over every shell the pseudopotential carries.
    l: np.ndarray
    s: np.ndarray
    #: ``(natomwfc_of_this_atom,)``: the Löwdin occupation of each column, whose
    #: sum is the atom's charge on this projector set. Reported because it is
    #: what says whether the projection captured the electron it is decomposing.
    charge: float

    @property
    def j(self) -> np.ndarray:
        return self.l + self.s

    def __str__(self) -> str:
        return (
            f"{self.species}{self.atom + 1}  "
            f"L = ({self.l[0]: .5f},{self.l[1]: .5f},{self.l[2]: .5f})  "
            f"S = ({self.s[0]: .5f},{self.s[1]: .5f},{self.s[2]: .5f})  "
            f"J = ({self.j[0]: .5f},{self.j[1]: .5f},{self.j[2]: .5f})"
        )


@dataclass
class SiteAngularMomentum:
    """What ``writelsj`` writes, per atom, plus the cell totals."""

    atoms: tuple[AtomicMoments, ...]
    #: ``(natom, 3)`` stacked, for arithmetic.
    orbital: np.ndarray
    spin: np.ndarray
    #: The projector set the decomposition is against.
    kind: str
    nspin: int = 1

    @property
    def total_orbital(self) -> np.ndarray:
        """``sum_a <L>_a``: the cell's orbital moment on this projector set."""
        return self.orbital.sum(axis=0)

    @property
    def total_spin(self) -> np.ndarray:
        return self.spin.sum(axis=0)

    @property
    def total(self) -> np.ndarray:
        return self.total_orbital + self.total_spin

    def table(self) -> str:
        """``LSJ.OUT``'s block, which is what this is for."""
        lines = [
            f"Angular momenta on the {self.kind} projectors (units of hbar)",
            "",
        ]
        lines += [f"  {atom}" for atom in self.atoms]
        lines += [
            "",
            f"  total L = ({self.total_orbital[0]: .5f},"
            f"{self.total_orbital[1]: .5f},{self.total_orbital[2]: .5f})",
            f"  total S = ({self.total_spin[0]: .5f},"
            f"{self.total_spin[1]: .5f},{self.total_spin[2]: .5f})",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------


def angular_momenta(
    calculation,
    result,
    kind: str = "ortho-atomic",
) -> SiteAngularMomentum:
    """``<L>``, ``<S>`` and ``<J>`` on every atom, from a converged run.

    Args:
        calculation: the :class:`~defumat.scf.driver.Calculation` the states
            belong to.
        result: the converged :class:`~defumat.scf.driver.SCFResult`; its
            wavefunctions and its ``wg`` occupations are what is contracted.
        kind: the projector set, one of
            :data:`defumat.projwfc.projections.PROJECTION_KINDS`. The default
            is Löwdin-orthogonalised, which is ``projwfc.x``'s only choice and
            the one a charge decomposition is defined against.

    Returns:
        :class:`SiteAngularMomentum`.
    """
    from defumat.hubbard.projectors import build_atomic_projectors
    from defumat.projwfc.channels import projection_channels
    from defumat.projwfc.projections import PROJECTION_KINDS

    if kind not in PROJECTION_KINDS:
        raise ValueError(
            f"unknown projector set {kind!r}; expected one of {PROJECTION_KINDS}"
        )
    _refuse_what_is_not_written(calculation)

    system = calculation.system
    channels = projection_channels(calculation.pseudos, system.structure)
    if not channels:
        raise ValueError(
            "none of the pseudopotentials carries an atomic orbital to project "
            "on, so there is no site to resolve an angular momentum onto"
        )

    projectors = np.asarray(build_atomic_projectors(
        calculation.pseudos, system.structure, system.cell,
        calculation.basis.smooth, calculation.basis.planewaves,
        calculation.basis_kpoints, calculation._overlap, kind=kind,
    ))  # (nk, npwx, natomwfc)

    density = _site_density_matrix(calculation, result, projectors, channels)
    return _contract(density, channels, system, calculation.nspin, kind)


def _refuse_what_is_not_written(calculation) -> None:
    from defumat.tddft.chi0 import _kpoints_are_reduced

    if calculation.spiral:
        raise NotImplementedError(
            "site angular momenta on a spin spiral are not implemented: the two "
            "spinor components live on spheres centred at k + q/2 and k - q/2, "
            "so a single set of atomic orbitals does not project both"
        )
    if _kpoints_are_reduced(calculation) and calculation.use_symmetry:
        raise NotImplementedError(
            "<L> and <S> on a symmetry-reduced k-set are not implemented: they "
            "are vectors, and summing a wedge sums a wedge -- an axial vector "
            "needs the group average with det(R) and the time-reversal sign, "
            "which is not written here. Run the whole grid (nosym = .true.), "
            "unshifted so that it is closed under the point group; that is the "
            "same escape dielectric_tensor documents"
        )
    relativistic = any(
        getattr(pseudo, "has_so", False) for pseudo in calculation.pseudos
    )
    if relativistic and calculation.projectors.qq is not None:
        raise NotImplementedError(
            "site angular momenta with a fully-relativistic ultrasoft or PAW "
            "dataset are not implemented: the spinor overlap's off-diagonal "
            "spin blocks are qq_so (transform_qq_so), and the projection here "
            "applies the scalar S to each component -- which is projwfc.x's "
            "validated path in every other regime and is missing that term in "
            "this one. A fully-relativistic norm-conserving dataset has S = 1 "
            "and is exact"
        )


def _site_density_matrix(calculation, result, projectors, channels):
    """``(natomwfc, 2, natomwfc, 2)``: ``sum_nk wg c c*``, over the whole basis.

    The spin axis is always two wide, whatever the regime, so that the traces
    below have one form. ``nspin = 1`` splits its (already spin-degenerate)
    weight equally between the two diagonal blocks, which is what makes ``<S>``
    come out as the zero it is rather than as an artefact of the layout.
    """
    import jax.numpy as jnp

    psi = jnp.asarray(result.wavefunctions)          # (nspin, nk, nbnd, ndim)
    weights = np.asarray(result.occupations)          # wg, k-weights folded in
    if weights.ndim == 2:
        weights = weights[None]
    nspin = calculation.nspin
    natomwfc = projectors.shape[2]

    coefficients = []  # one (nk, nbnd, natomwfc) per spin component
    if nspin == 4:
        npwx = calculation.basis.planewaves.npwx
        for component in range(2):
            block = psi[0, :, :, component * npwx:(component + 1) * npwx]
            coefficients.append(
                np.asarray(jnp.einsum("kgi,kbg->kbi", projectors.conj(), block))
            )
        band_weights = [np.asarray(weights[0]), np.asarray(weights[0])]
    else:
        for spin in range(2):
            channel = min(spin, nspin - 1)
            coefficients.append(np.asarray(
                jnp.einsum("kgi,kbg->kbi", projectors.conj(), psi[channel])
            ))
        if nspin == 1:
            # ``wg`` already carries ``degspin = 2``; half of it belongs to each
            # component, and the two are the same state.
            band_weights = [0.5 * np.asarray(weights[0])] * 2
        else:
            band_weights = [np.asarray(weights[0]), np.asarray(weights[1])]

    density = np.zeros((natomwfc, 2, natomwfc, 2), dtype=complex)
    for a in range(2):
        for b in range(2):
            if nspin != 4 and a != b:
                # Two independent collinear channels have no coherence between
                # them; a spinor is the only regime that does.
                continue
            # **One weight, not the geometric mean of two.** ``wg`` is a
            # property of the band and not of the spin component, and the two
            # arrays are the same object wherever ``a != b`` contributes at all
            # (a spinor's two components are one state). Writing it as
            # ``sqrt(w_a w_b)`` gives ``|w|``, which is the same number until a
            # smearing makes an occupation **negative** -- Methfessel-Paxton
            # routinely does and Marzari-Vanderbilt can -- and then that band
            # enters rho with the wrong sign. Silent, small, and invisible to a
            # rotation check, because it is systematic across orientations.
            density[:, a, :, b] = np.einsum(
                "kb,kbi,kbj->ij",
                band_weights[a], coefficients[a], coefficients[b].conj(),
            )
    return density


def _contract(density, channels, system, nspin, kind):
    """``dmatls``: ``Tr(L rho)`` per atom, and ``(1/2) Tr(sigma rho)`` beside it."""
    natom = len(system.structure.positions)
    orbital = np.zeros((natom, 3))
    spin = np.zeros((natom, 3))
    charge = np.zeros(natom)

    for indices, l in _shells(channels):
        block = density[np.ix_(indices, [0, 1], indices, [0, 1])]
        atom = channels[indices[0]].atom
        matrices = orbital_matrices(l)
        for axis in range(3):
            # ``<L_i> = sum_s Tr_m (L_i rho_ss)`` -- L is diagonal in spin.
            orbital[atom, axis] += float(np.real(np.einsum(
                "mn,nsms->", matrices[axis], block
            )))
            # ``<S_i> = (1/2) sum_m Tr_s (sigma_i rho_mm)``.
            spin[atom, axis] += 0.5 * float(np.real(np.einsum(
                "st,mtms->", PAULI[axis], block
            )))
        charge[atom] += float(np.real(np.einsum("msms->", block)))

    atoms = tuple(
        AtomicMoments(
            atom=index,
            species=system.structure.species[system.structure.types[index]].name,
            l=orbital[index],
            s=spin[index],
            charge=charge[index],
        )
        for index in range(natom)
    )
    return SiteAngularMomentum(
        atoms=atoms, orbital=orbital, spin=spin, kind=kind, nspin=nspin,
    )


def _shells(channels):
    """Group the projection columns into ``(indices, l)`` per ``(atom, shell)``.

    ``L`` connects neither two ``l`` nor two radial shells of the same ``l`` --
    on an orthonormal projector set the radial overlap is a Kronecker delta --
    so the block structure is exactly one shell, and an atom's moment is the sum
    over its shells.
    """
    groups, current, key = [], [], None
    for channel in channels:
        this = (channel.atom, channel.wfc, channel.l)
        if this != key:
            if current:
                groups.append((current, key[2]))
            current, key = [], this
        current.append(channel.index)
    if current:
        groups.append((current, key[2]))
    return groups
