"""The self-consistent field loop.

Following ``PW/src/electrons.f90``: from a starting density, build the
potential, diagonalise the Hamiltonian, rebuild the density from the occupied
states, and mix -- until the density stops changing.

The loop itself stays in Python because its termination test depends on the
values being computed (rule: data-dependent control flow does not belong inside
``jit``). The work inside one iteration is what gets compiled.

The total energy is assembled exactly as QE prints it, term by term, because
comparing the decomposition rather than only the total is what localises an
error to one physical contribution:

    E = (eband + deband) + E_Hartree + E_xc + E_Ewald [+ smearing (-TS)]

``eband`` is the sum of occupied eigenvalues, which double-counts the Hartree
and exchange-correlation energies; ``deband = -int rho v_scf`` removes exactly
that double counting, so their sum is the kinetic plus external-potential
energy that QE labels "one-electron contribution".

**Spin.** With ``nspin = 2`` the density, the potential, ``becsum``, ``D_ij``,
the eigenvalues and the wavefunctions all grow a leading channel axis, and one
SCF iteration diagonalises a *different Hamiltonian per channel* -- same kinetic
term and same local pseudopotential, different self-consistent potential and
different ``D_ij``. QE arranges this by storing both channels in a single
k-point list of length ``2 nks`` with ``isk(ik)`` saying which; here the channel
is an explicit axis and the loop over it is a Python loop, since ``nspin`` is
static and two Hamiltonians are two compiled kernels either way. What must not
change is that ``k`` stays the leading *independent* axis inside each channel,
because that is the axis the batching and the eventual sharding use.

**Noncollinear.** With ``nspin = 4`` there is again *one* Hamiltonian, not two:
the spin channels are not separate problems but the two components of a single
spinor, and the loop below therefore runs once. What grows is the state itself
-- ``2 npwx`` numbers instead of ``npwx``
(:class:`~pypresso.hamiltonian.noncollinear.SpinorHamiltonian`) -- and, when the
calculation carries a magnetization, the density and the potential, which become
``(n, m_x, m_y, m_z)``. The spin-channel axis of every array is kept at length
one so that occupations, mixing and the result objects need no third case; it is
``npol`` and ``nspin_mag`` that carry the noncollinear shapes, not ``nspin``.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from pypresso.basis.builder import Basis, build_basis
from pypresso.basis.planewaves import build_plane_wave_basis
from pypresso.basis.interpolate import to_dense, to_smooth
from pypresso.basis.sticks import build_sticks
from pypresso.basis.fft import g_to_r, r_to_g
from pypresso.hamiltonian.noncollinear import SpinorHamiltonian
from pypresso.hamiltonian.operator import Hamiltonian
from pypresso.pseudo.atomic import atomic_wavefunctions
from pypresso.paw.onecenter import build_paw
from pypresso.paw.symmetry import build_becsum_symmetry
from pypresso.pseudo.augmentation import build_augmentation
from pypresso.pseudo.potentials import (
    combine_species,
    species_atomic_charge,
    species_core_charge,
    species_local_potential,
    starting_charge,
)
from pypresso.pseudo.projectors import build_projector_core, projector_channels
from pypresso.pseudo.upf import Pseudopotential
from pypresso.pseudo.spinorbit import becsum_transform, build_spin_orbit
from pypresso.batching import map_k, resolve_k_batch
from pypresso.scf.density import becsum, spinor_becsum, spinor_sum_band, sum_band
from pypresso.scf.ewald import build_ewald
from pypresso.scf.mixing import get_mixer
from pypresso.scf.occupations import (
    fixed_occupations,
    input_occupations,
    smeared_occupations,
    smearing_entropy,
    spin_electron_counts,
    tetrahedra_for,
    tetrahedron_occupations_spin,
)
from pypresso.scf.potential import as_potential_components, scf_accuracy, v_of_rho
from pypresso.xc.functional import resolve_functional
from pypresso.solvers import get_eigensolver
from pypresso.solvers.davidson import ETHR_MIN, starting_vectors
from pypresso.solvers.subspace import rayleigh_ritz
from pypresso.system.builder import System
from pypresso.system.kpoints import KPoints
from pypresso.system.symmetry import apply_symmetry_maps, find_symmetries, symmetry_maps
from pypresso.units import RY_TO_EV

__all__ = ["SCFResult", "Calculation", "run_scf", "default_nbnd"]


# The iteration body is compiled in three units rather than one, because the
# occupation weights that separate them are decided on the host: the Fermi level
# is a bisection whose bracket is data. Everything on either side of that
# decision is one compiled kernel each, so a whole SCF iteration costs three
# dispatches instead of the hundreds the eager version issued.

_potential_of_rho = jax.jit(v_of_rho)
_accuracy = jax.jit(scf_accuracy)


#: Where ``ethr`` starts, from ``PW/src/setup.f90``: the starting potential is a
#: superposition of atomic charges and is nowhere near self-consistent, so there
#: is nothing to be gained by diagonalising against it accurately.
ETHR_INIT = 1.0e-2


@partial(jax.jit, static_argnames=("grid",))
def _symmetrize(rho_r, fft_index, grid, maps):
    """Impose the crystal symmetry on a real-space density, in one kernel.

    Per channel: a collinear symmetry operation acts on positions and leaves
    the spin alone, so it symmetrises the two densities independently. (QE
    symmetrises in the ``(up, down)`` representation too -- ``sym_rho`` runs
    before ``rhoz_or_updw`` converts.)
    """
    permutations, phases = maps

    def channel(rho):
        rho_g = apply_symmetry_maps(r_to_g(rho, fft_index), permutations, phases)
        return jnp.real(g_to_r(rho_g, fft_index, grid))

    return jax.vmap(channel)(rho_r)


@partial(jax.jit, static_argnames=("grid", "k_batch"))
def _density_of_bands(psi, fft_index, grid, weights, cell, k_batch):
    """``sum_band`` on the smooth grid, in one kernel.

    The symmetrisation used to be fused in here. It cannot be any more: it acts
    on the dense grid, and with a double grid the density has to be lifted there
    first.
    """
    return sum_band(psi, fft_index, grid, weights, cell, k_batch)


@jax.jit
def _paw_deband(ddd_paw, augmentation, becsum_):
    """``sum_s sum_a sum_ij ddd_paw^a_ij becsum^a_ij``, out of the block matrix.

    Reading the blocks back out of the assembled ``(nspin, nkb, nkb)`` matrix
    rather than keeping them separately is deliberate: it is the *same* array
    the Hamiltonian used, so the two cannot drift apart.
    """
    total = jnp.asarray(0.0)
    nspin = ddd_paw.shape[0]
    for values, atoms in zip(becsum_, augmentation.species_atoms):
        if values is None or not atoms:
            continue
        nh = values.shape[-1]
        for n, atom in enumerate(atoms):
            start = augmentation.channel_offsets[atom]
            block = jax.lax.dynamic_slice(
                ddd_paw, (0, start, start), (nspin, nh, nh)
            )
            total = total + jnp.sum(block * values[:, n])
    return total


def _mix(mixer, rho, rho_out, becsum_in, becsum_out):
    """One mixing step over the density and, for PAW, ``becsum`` with it.

    The two are packed into a single vector so that the extrapolation
    coefficients Anderson computes from the density residual are applied to both
    -- they are two views of the same fixed point, and mixing them with
    different histories makes the iteration inconsistent rather than merely
    slower.
    """
    flat = [np.asarray(rho).ravel()]
    flat_out = [np.asarray(rho_out).ravel()]
    for old, new in zip(becsum_in, becsum_out):
        if old is None:
            continue
        flat.append(np.asarray(old).ravel())
        flat_out.append(np.asarray(new).ravel())

    mixed = mixer.mix(np.concatenate(flat), np.concatenate(flat_out))

    offset = rho.size
    rho_mixed = jnp.asarray(mixed[:offset].reshape(rho.shape))
    becsum_mixed = []
    for old in becsum_in:
        if old is None:
            becsum_mixed.append(None)
            continue
        becsum_mixed.append(jnp.asarray(mixed[offset : offset + old.size].reshape(old.shape)))
        offset += old.size
    return rho_mixed, tuple(becsum_mixed)


@jax.jit
def _paw_onecenter(paw, becsum_):
    """``PAW_potential``: the one-centre energy and its ``ddd``, in one kernel."""
    return paw.energy_and_coefficients(becsum_)


@jax.jit
def _newd(potential, fft_index, dij, augmentation):
    """``D^(0)_ij + int V_eff Q_ij``, as the block matrix the Hamiltonian takes.

    One block matrix per spin channel: ``newq_acc`` integrates ``v%of_r(:,is)``
    against the same ``Q_ij`` for each ``is``, so the augmentation charge is
    shared and only the potential differs.
    """
    blocks = [
        augmentation.block_matrix(augmentation.integrals(r_to_g(channel, fft_index)))
        for channel in potential
    ]
    return dij[None] + jnp.stack(blocks)


@partial(jax.jit, static_argnames=("grid",))
def _addusdens(rho_r, fft_index, grid, augmentation, becsum_):
    """The augmentation charge, added to the density on the dense grid.

    Per channel, from that channel's ``becsum``: ``addusdens`` is called on
    ``rho%of_g(:,:)`` while the density is still in the ``(up, down)``
    representation, so each channel gets its own augmentation charge.
    """
    charge = jnp.stack([
        augmentation.charge(
            tuple(None if b is None else b[spin] for b in becsum_)
        )
        for spin in range(rho_r.shape[0])
    ])
    return rho_r + jnp.real(g_to_r(charge, fft_index, grid))


@partial(jax.jit, static_argnames=("nbnd", "k_batch"))
def _rotate_all(hamiltonian, vectors, nbnd: int, k_batch):
    """Rayleigh-Ritz at ``k_batch`` k-points at a time.

    ``wfcinit`` does this inside its own ``DO ik`` loop, and the working set is
    the same one the eigensolver has: the atomic orbitals of every k-point, plus
    ``H`` applied to them.
    """
    return map_k(
        lambda pair: rayleigh_ritz(hamiltonian, pair[0], pair[1], nbnd),
        (jnp.arange(hamiltonian.nk), vectors),
        batch=resolve_k_batch(k_batch),
    )


@partial(jax.jit, static_argnames=("grid", "nspin_mag", "k_batch"))
def _spinor_density_of_bands(psi, fft_index, grid, weights, cell, nspin_mag, k_batch):
    """``sum_band`` for spinors, in one kernel."""
    return spinor_sum_band(psi, fft_index, grid, weights, cell, nspin_mag, k_batch)


@jax.jit
def _newd_noncollinear(deeq_components, dvan_so, fcoef):
    """``newd_so``/``newd_nc``: the scalar integrals as a 2x2 spin matrix.

    ``deeq_components`` is ``(nspin_mag, nkb, nkb)`` -- one integral of the
    augmentation charge against each component of the potential -- and the
    recombination is the same one the local potential undergoes:
    ``sum_a d_a sigma^a``, then sandwiched between the spin-orbit coefficients.

    ``fcoef`` is block diagonal over atoms, so the full ``nkb x nkb`` matrix
    products below never mix two atoms and the sandwich is the per-species one
    of ``newd_so_acc`` written once. For a scalar-relativistic species ``fcoef``
    is the identity on each diagonal spin block, and the sandwich collapses to
    the plain recombination of ``newd_nc_acc``.
    """
    nspin_mag = deeq_components.shape[0]
    charge = deeq_components[0]
    zero = jnp.zeros_like(charge)
    if nspin_mag == 1:
        blocks = jnp.stack([jnp.stack([charge, zero]), jnp.stack([zero, charge])])
    else:
        mx, my, mz = deeq_components[1], deeq_components[2], deeq_components[3]
        blocks = jnp.stack([
            jnp.stack([charge + mz, mx - 1j * my]),
            jnp.stack([mx + 1j * my, charge - mz]),
        ])
    blocks = blocks.astype(fcoef.dtype)
    return dvan_so + jnp.einsum("asij,stjk,tbkl->abil", fcoef, blocks, fcoef, optimize=True)


@jax.jit
def _noncollinear_magnetization(rho_r, volume):
    """``(m_x, m_y, m_z, |m|)`` integrated over the cell, in Bohr magnetons.

    ``report_mag`` prints the three components of the total moment and, as in
    the collinear case, the integral of the modulus. The two say different
    things: an antiferromagnet or a spiral integrates to zero and still has a
    large absolute moment, and with spin-orbit coupling the *direction* of the
    total moment is a result rather than an input, which is why all three
    components are worth printing.
    """
    magnetization = rho_r[1:]
    scale = volume / magnetization[0].size
    modulus = jnp.sqrt(jnp.sum(magnetization**2, axis=0))
    return jnp.concatenate([
        scale * jnp.sum(magnetization, axis=(1, 2, 3)),
        jnp.asarray([scale * jnp.sum(modulus)]),
    ])


@jax.jit
def _magnetization(rho_r, volume):
    """``(total, absolute)`` magnetization in Bohr magnetons per cell.

    ``report_mag``/``electrons.f90`` print both: the integral of ``rho_up -
    rho_dw`` and the integral of its absolute value. They differ whenever the
    magnetization changes sign inside the cell -- an antiferromagnet has zero
    total and a large absolute moment -- so printing only the first would hide
    exactly the states LSDA exists to describe.
    """
    magnetization = rho_r[0] - rho_r[1]
    scale = volume / magnetization.size
    return jnp.stack([
        scale * jnp.sum(magnetization),
        scale * jnp.sum(jnp.abs(magnetization)),
    ])


@jax.jit
def _iteration_scalars(eigenvalues, weights, rho_in, rho_out, v_scf, volume):
    """``(eband, deband, residual)`` as one array, so the loop syncs once.

    Each ``float()`` on a device array is a host round trip; computing the three
    together and transferring them in one go is the difference between one
    synchronisation per iteration and one per printed number.

    ``deband`` integrates the **output** density against the **input**
    potential, which is where ``delta_e()`` sits in ``electrons.f90``'s
    ordering -- see the note in :func:`run_scf`.
    """
    eband = jnp.sum(weights * eigenvalues)
    # ``delta_e``: one integral per channel of that channel's own density
    # against its own potential. QE writes the same number in the (total,
    # magnetization) representation as ``0.5[(v_up + v_dw) rho + (v_up - v_dw)
    # m]``, which is the same contraction rearranged.
    deband = -volume / rho_out[0].size * jnp.sum(rho_out * v_scf)
    residual = jnp.max(jnp.abs(rho_out - rho_in))
    return jnp.stack([eband, deband, residual])


def _without_gamma_storage(system: System) -> System:
    """Turn ``K_POINTS gamma`` into an explicit single k-point at the origin.

    ``ggen`` keeps only half the G sphere when the calculation is at k = 0,
    because a real wavefunction has ``c(-G) = conj(c(G))``. That storage is
    generated here (``basis/gvectors.py``) but it is not *consumed* anywhere:
    ``vloc_psi`` would need QE's ``vloc_psi_gamma`` packing, the eigensolver
    QE's real ``regterg`` overlaps, ``calbec``/``addusdens``/``newd`` their
    ``fact = 2`` doubling, and the symmetry maps a way to follow a rotation that
    leaves the stored half. None of that exists, so running a gamma-only basis
    through the SCF would be silently wrong (where it does not simply fail in
    ``symmetry_maps``).

    The substitution below is exact rather than approximate: the same cell at
    the same cutoffs has the same FFT dimensions and the same physics, and the
    full sphere is what every routine here already expects. What it costs is
    the factor of two in storage and transforms that the trick exists to save,
    which is a performance matter and not a correctness one.

    The k-point weights are carried over rather than rebuilt, because
    :meth:`KPoints.from_cartesian` renormalises and reapplies ``DEGSPIN`` --
    which would undo the halving an ``nspin = 2`` run has already applied.
    """
    kpoints = system.kpoints
    if not kpoints.gamma_only:
        return system
    warnings.warn(
        "K_POINTS gamma asks for the half-sphere storage of the gamma-point "
        "trick, which is not implemented; running at an explicit k = 0 with the "
        "full G sphere instead. The result is the same, the cost is twice the "
        "plane waves",
        stacklevel=3,
    )
    replacement = KPoints(
        coords=kpoints.coords,
        weights=kpoints.weights,
        gamma_only=False,
        precision=kpoints.precision,
    )
    return eqx.tree_at(lambda s: s.kpoints, system, replacement)


def next_ethr(ethr: float, accuracy: float, nelec: float, iteration: int) -> float:
    """QE's diagonalisation-threshold schedule (``PW/src/electrons.f90``).

    The eigenvalues never need to be more accurate than the density they are
    computed from. ``ethr`` therefore tracks ``dr2``, the estimated error in the
    density, and only tightens -- 1e-2 while the density is still wrong in the
    second decimal, 1e-13 by the time it is converged.

    Three details are QE's and matter: the threshold is *reset* to 1e-2 at the
    second iteration rather than carried over from the first, it can only ever
    decrease (``MIN``), and it is floored at 1e-13 because an iterative
    diagonalisation asked for more than that becomes unstable rather than more
    accurate.
    """
    if iteration <= 1:
        return ethr
    if iteration == 2:
        ethr = ETHR_INIT
    return max(min(ethr, 0.1 * accuracy / max(1.0, nelec)), ETHR_MIN)


def default_nbnd(
    nelec: float, occupations: str, nelup=None, neldw=None, noncolin: bool = False
) -> int:
    """QE's default band count (``PW/src/setup.f90``).

    An insulator needs exactly the occupied bands; a smeared calculation needs
    20% more so there are empty states for the Fermi level to sit among.

    ``IF (noncolin) nbnd = 2 * nbnd``, applied last and to whichever count the
    lines above arrived at: a spinor band holds one electron rather than two, so
    a noncollinear run of the same crystal needs twice as many. QE says why in a
    comment -- "bands are NOT twofold degenerate" -- and the doubling is not
    optional even when spin-orbit coupling leaves them degenerate anyway.

    With two channels the count is ``MAX(nelec/degspin, nelup, neldw)`` and
    ``degspin`` is 1, so what decides it is the *fuller* channel: eight
    electrons split 5/3 need five bands, not four. Getting this wrong is
    immediately visible -- QE prints "number of Kohn-Sham states" -- but only if
    it is looked at, and a run with too few bands converges to a wrong energy
    rather than failing.
    """
    occupied = int(round(nelec / 2.0))
    if nelup is not None:
        occupied = max(occupied, int(round(nelup)), int(round(neldw)))
    degeneracy = 2 if noncolin else 1
    if occupations == "fixed":
        return degeneracy * max(occupied, 1)
    if nelup is not None:
        return max(
            int(round(1.2 * nelec / 2.0)),
            int(round(1.2 * nelup)),
            int(round(1.2 * neldw)),
            occupied + 4,
        )
    return degeneracy * max(int(round(1.2 * nelec / 2.0)), occupied + 4)


@dataclass
class SCFResult:
    """Everything an SCF produces, with the energy split QE's way."""

    converged: bool
    iterations: int
    total_energy: float
    energy_terms: dict
    #: ``(nk, nbnd)`` for an unpolarized run and ``(2, nk, nbnd)`` for LSDA, in
    #: Ry. The spin axis is *squeezed away* when there is only one channel, so
    #: that everything written against the unpolarized shape keeps working and
    #: a polarized result cannot be mistaken for one. Use
    #: :attr:`eigenvalues_by_spin` where the axis must always be there.
    eigenvalues: np.ndarray
    occupations: np.ndarray  # same shape as eigenvalues -- the weights wg
    wavefunctions: jnp.ndarray  # (nspin, nk, nbnd, npwx)
    density: jnp.ndarray  # (nspin, n1, n2, n3), electrons/bohr^3
    potential: jnp.ndarray  # (nspin, n1, n2, n3), Ry -- the total local potential
    #: ``V[rho_out] - V[rho_in]`` at the last iteration -- QE's ``vnew``, the
    #: self-consistency the run did not reach. Zero to the extent that it
    #: converged, and the only input to ``force_corr``.
    potential_change: jnp.ndarray | None = None
    fermi_energy: float | None = None
    homo: float | None = None
    lumo: float | None = None
    #: QE's estimated scf accuracy at the last iteration, in Ry.
    accuracy: float | None = None
    nspin: int = 1
    #: ``int (rho_up - rho_dw)`` and ``int |rho_up - rho_dw|``, in Bohr
    #: magnetons per cell -- the two numbers QE prints. ``None`` unpolarized.
    magnetization: float | None = None
    absolute_magnetization: float | None = None
    #: Only when ``tot_magnetization`` constrained the two channels separately.
    fermi_energy_up: float | None = None
    fermi_energy_down: float | None = None
    #: ``int m(r)`` as a cartesian vector, in Bohr magnetons per cell. Only a
    #: noncollinear run has one -- a collinear one has :attr:`magnetization`,
    #: which is the same quantity when the axis is fixed by construction.
    magnetization_vector: tuple | None = None
    #: 1, 2 or 4: how many components :attr:`density` and :attr:`potential`
    #: have. It is 1 for a *nonmagnetic* spin-orbit run, where ``nspin`` is 4.
    nspin_mag: int = 1
    history: list = field(default_factory=list)

    @property
    def eigenvalues_ev(self) -> np.ndarray:
        return self.eigenvalues * RY_TO_EV

    @property
    def eigenvalues_by_spin(self) -> np.ndarray:
        """``(nspin, nk, nbnd)`` whatever ``nspin`` is."""
        return self.eigenvalues if self.nspin == 2 else self.eigenvalues[None]

    @property
    def total_density(self) -> jnp.ndarray:
        """``(n1, n2, n3)``: the charge density.

        The two channels summed when they are ``(up, down)``, and the first
        component alone when they are ``(n, m_x, m_y, m_z)``.
        """
        from pypresso.scf.potential import total_charge

        return total_charge(self.density)


def _spin_block_diagonal(per_atom) -> np.ndarray:
    """Per-atom ``(nh, nh, 2, 2)`` blocks -> one ``(2, 2, nkb, nkb)`` matrix.

    The same block-diagonal assembly ``build_projectors`` does for ``D_ij``,
    with a spin pair in front. Laying the spin indices outermost is what lets
    the sandwich in :func:`_newd_noncollinear` be four ordinary ``nkb x nkb``
    matrix products: the block structure keeps the atoms from mixing, so nothing
    downstream has to loop over them.
    """
    sizes = [block.shape[0] for block in per_atom]
    nkb = sum(sizes)
    out = np.zeros((2, 2, nkb, nkb), dtype=complex)
    offset = 0
    for block, nh in zip(per_atom, sizes):
        out[:, :, offset : offset + nh, offset : offset + nh] = np.transpose(
            block, (2, 3, 0, 1)
        )
        offset += nh
    return out


class Calculation:
    """Everything that stays fixed while the density changes.

    Separating this from the loop keeps the setup work (rule R2: host-side,
    dynamic shapes, done once) out of the iteration, and makes a band-structure
    run reuse exactly the same objects with a different k-point set.
    """

    def __init__(
        self,
        system: System,
        pseudos: tuple[Pseudopotential, ...],
        basis: Basis | None = None,
        diagonalization: str | None = None,
        k_batch: int | None | str = "default",
    ):
        system = _without_gamma_storage(system)
        self.system = system
        self.eigensolver = get_eigensolver(diagonalization)
        # How many k-points are in flight at once, everywhere this calculation
        # touches the k axis. One -- QE's ``k_loop`` -- unless asked otherwise;
        # ``None`` is a single ``vmap`` over all of them. See
        # :mod:`pypresso.batching`.
        self.k_batch = resolve_k_batch(k_batch)
        self.pseudos = tuple(pseudos)
        self.basis = basis if basis is not None else build_basis(system)
        if self.basis.dense.gamma_only:
            raise NotImplementedError(
                "the gamma-only half-sphere storage is not implemented in h_psi, "
                "sum_band or the eigensolver; pass a basis built with "
                "gamma_only=False"
            )

        # Which exchange-correlation functional this run uses is decided once,
        # here, from the pseudopotentials and the input -- not defaulted to
        # anywhere downstream. Every consumer takes it as an argument, so a PBE
        # dataset cannot end up running under LDA by omission.
        self.functional = resolve_functional(
            [pseudo.functional for pseudo in self.pseudos], system.input_dft
        )

        # QE's three spin numbers, kept apart (``set_spin_vars``): how many
        # regimes there are, how many components a *state* has, and how many a
        # *density* has. See :class:`pypresso.system.builder.System`.
        self.nspin = int(system.nspin)
        self.npol = int(system.npol)
        self.nspin_mag = int(system.nspin_mag)
        self.noncolin = bool(system.noncolin)
        self.lspinorb = bool(system.lspinorb)
        if self.nspin == 2 or self.nspin_mag == 4:
            # Refused here rather than where it would first divide by something:
            # a functional whose correlation has no polarized parameterisation
            # would otherwise run with the unpolarized one and converge to a
            # number that is wrong and looks right. A noncollinear magnetization
            # needs it for the same reason -- the functional is evaluated along
            # the local spin axis, which is the polarized one.
            self.functional.require_spin()
        if self.lspinorb and not any(p.has_so for p in self.pseudos):
            raise ValueError(
                "lspinorb = .true. but no pseudopotential is fully relativistic; "
                "a spin-orbit run needs a rel- dataset with a PP_SPIN_ORB section"
            )
        if self.noncolin and not self.lspinorb:
            for pseudo in self.pseudos:
                if pseudo.has_so:
                    # ``average_pp``: QE j-averages the projectors back to the
                    # scalar-relativistic ones. That is a different
                    # pseudopotential from the one in the file, so it is refused
                    # rather than done silently.
                    raise NotImplementedError(
                        f"{pseudo.element}: a fully-relativistic pseudopotential "
                        "with lspinorb = .false. asks for QE's j-averaging "
                        "(average_pp), which is not implemented; use "
                        "lspinorb = .true. or a scalar-relativistic dataset"
                    )

        self.nelec = sum(self.pseudos[t].z_valence for t in system.structure.types)
        # ``set_nelup_neldw``. Only used when the magnetization is constrained,
        # but computed always because ``default_nbnd`` needs both counts.
        self.nelup, self.neldw = spin_electron_counts(
            self.nelec, system.tot_magnetization
        )
        self.charges = np.array([self.pseudos[t].z_valence for t in system.structure.types])

        # Two grids, and which quantity lives on which is QE's split: the
        # wavefunctions and everything built from them are on the *smooth* grid,
        # the density and the potential on the *dense* one. They are the same
        # object unless the input asks for a dual above 4.
        dense, smooth = self.basis.dense, self.basis.smooth
        planewaves = self.basis.planewaves
        self.kinetic = planewaves.kinetic(smooth, system.kpoints, system.cell)
        self.fft_index = planewaves.fft_index(smooth)

        # QE's FFT layout for the wavefunction transforms; see basis/sticks.py.
        self.sticks = build_sticks(self.fft_index, planewaves.mask, smooth.grid)

        # The projectors are built in two halves -- the species-dependent
        # columns once, the structure factor per geometry -- so that moving the
        # atoms costs one exponential per atom and so that ``grad`` with respect
        # to the positions never reaches the radial integrals. See
        # :class:`pypresso.pseudo.projectors.ProjectorCore` and
        # :meth:`at_positions`.
        self.projector_core = build_projector_core(
            self.pseudos, system.structure, system.cell, smooth, planewaves, system.kpoints
        )
        self.projectors = self.projector_core.at_positions(system.structure.positions)

        # The augmentation charge lives on the *dense* grid: it is sharply
        # peaked, and holding it is what the second grid is for. Everything
        # ultrasoft hangs off this object being non-None.
        self.augmentation = build_augmentation(
            self.pseudos, system.structure, system.cell, dense
        )
        self.species_channels = ()
        self.paw = None
        if self.augmentation is not None:
            self.projectors = eqx.tree_at(
                lambda p: p.qq,
                self.projectors,
                self._per_atom(self.augmentation.qq),
                is_leaf=lambda x: x is None,
            )
            self.species_channels = self._species_channels()

        # PAW adds the one-centre corrections on top of everything ultrasoft
        # does. They depend on ``becsum`` and on nothing else that changes, so
        # like ``newd`` they are rebuilt once per SCF iteration.
        self.paw = build_paw(self.pseudos, system.structure, self.functional)

        # The spin-orbit coefficients: ``fcoef`` per species and, assembled over
        # the atoms, the two block matrices the spinor Hamiltonian takes --
        # ``dvan_so`` (which *is* the nonlocal potential for a norm-conserving
        # run) and ``qq_so``. Built once: they depend on the pseudopotentials
        # and on nothing that changes during the SCF.
        self.spin_orbit = ()
        self.dvan_so = None
        self.fcoef_matrix = None
        self.qq_so = None
        if self.noncolin:
            self.spin_orbit = build_spin_orbit(self.pseudos)
            types = system.structure.types
            self.dvan_so = jnp.asarray(_spin_block_diagonal(
                [self.spin_orbit[t].dvan_so for t in types]
            ))
            self.fcoef_matrix = jnp.asarray(_spin_block_diagonal(
                [self.spin_orbit[t].fcoef for t in types]
            ))
            if self.augmentation is not None:
                # ``qq`` is empty for a norm-conserving species even inside an
                # ultrasoft calculation, while its projector block is not: the
                # atom has projectors and no augmentation charge. Padding it to
                # ``nh`` here is what ``block_matrix`` does implicitly by
                # writing nothing, and what keeps the assembled matrix square.
                def species_qq(t: int) -> np.ndarray:
                    nh = self.pseudos[t].nh
                    values = np.asarray(self.augmentation.qq[t])
                    return values if values.shape == (nh, nh) else np.zeros((nh, nh))

                self.qq_so = jnp.asarray(_spin_block_diagonal(
                    [self.spin_orbit[t].qq_so(species_qq(t)) for t in types]
                ))

        # Per-species radial transforms, which the structure factor multiplies:
        # cached so a moved geometry re-contracts them instead of re-integrating
        # them (``setlocal``/``set_rhoc`` are called afresh by QE each ionic
        # step, but their ``vloc``/``rhoc`` tables are not rebuilt either).
        self.vloc_species = species_local_potential(self.pseudos, system.cell, dense)
        self.rho_core_species = species_core_charge(self.pseudos, system.cell, dense)
        # The atomic charge, on the same footing: it is what the SCF starts from
        # and what ``force_corr`` pairs with the potential the last iteration
        # did not apply. QE tabulates it once (``init_tab_rhoat``) for the same
        # reason -- the radial integration does not depend on the geometry.
        self.rho_atomic_species = species_atomic_charge(self.pseudos, system.cell, dense)

        vloc_g = combine_species(self.vloc_species, system.structure, system.cell, dense)
        self.vltot = jnp.real(g_to_r(vloc_g, dense.fft_index, dense.grid))

        # The nonlinear core correction, on the dense grid (``set_rhoc``). It is
        # ``None`` when no species has a PP_NLCC section, and ``None`` is an
        # empty pytree, so the two cases compile separately with no runtime
        # branch in the potential.
        rho_core_g = (
            None if self.rho_core_species is None
            else combine_species(
                self.rho_core_species, system.structure, system.cell, dense
            )
        )
        self.rho_core = (
            None if rho_core_g is None
            else jnp.real(g_to_r(rho_core_g, dense.fft_index, dense.grid))
        )
        # Kept as well as its transform: a gradient-corrected functional needs
        # the core charge's gradient, which is taken in G space alongside the
        # valence density's (``gradcorr`` adds ``rhog_core`` to ``rhogaux``).
        self.rho_core_g = rho_core_g

        # ``alpha`` and the neighbour list are fixed here and the sum is then a
        # differentiable function of the positions -- which is what makes the
        # ionic part of the force ``grad`` of it rather than a second expression.
        self.ewald_sum = build_ewald(system.cell, system.structure, dense, self.charges)
        self.ewald = float(
            self.ewald_sum.energy(system.cell, system.structure.positions, dense)
        )

        # The density built from a symmetry-reduced k-point set is not itself
        # symmetric; QE restores the symmetry explicitly and so must we, or
        # degenerate levels split by tens of meV and the energy is wrong in the
        # third decimal. See pypresso.system.symmetry.
        # QE's default dual is 4, which is exactly the condition for the density
        # grid to resolve every G - G' between two wavefunction plane waves --
        # see Hamiltonian.matrix. An input with a smaller dual is legal, so the
        # question is asked rather than assumed. Gamma-only storage keeps half
        # the sphere, so a difference of two stored G's need not be a stored G at
        # all and the gather does not apply however large the dual is.
        # The gather reads the potential stored on the *smooth* grid, since that
        # is the grid H|psi> runs on, so the question is whether the smooth
        # cutoff reaches 4*ecutwfc -- which it does by construction whenever the
        # dual exceeds 4, and only then depends on ecutrho.
        self.resolves_differences = bool(
            smooth.ecut >= 4.0 * system.ecutwfc - 1e-8 and not smooth.gamma_only
        )

        # ``nosym`` is not a speed switch. An input can ask for a state that
        # does not have the crystal's symmetry -- one of an atom's three p
        # channels occupied and the others empty, which is what
        # ``pw_pawatom/paw-atom_spin*.in`` does -- and symmetrising the density
        # or ``becsum`` anyway averages the three and converges somewhere else
        # entirely. QE requires ``nosym = .true.`` for those inputs; honouring it
        # is a correctness matter.
        self.symmetries = find_symmetries(system.cell, system.structure)
        use_symmetry = not system.nosym and self.symmetries.nsym > 1
        # PAW's projector occupations need the symmetry imposed on them
        # explicitly -- see pypresso.paw.symmetry. Built after the symmetry
        # search, which is why it is not up with the rest of the PAW setup.
        self._becsum_symmetry = (
            build_becsum_symmetry(self.pseudos, system.structure, system.cell,
                                  self.symmetries)
            if self.paw is not None and use_symmetry else None
        )
        self._symmetry_maps = symmetry_maps(dense, self.symmetries) if use_symmetry else None

    def _per_atom(self, per_species) -> jnp.ndarray:
        """A per-species ``(nh, nh)`` quantity, as the ``(nkb, nkb)`` block matrix."""
        blocks = tuple(
            jnp.broadcast_to(block, (len(atoms),) + block.shape)
            for block, atoms in zip(per_species, self.augmentation.species_atoms)
        )
        return self.augmentation.block_matrix(blocks)

    def _species_channels(self) -> tuple:
        """For each species, the projector columns of each of its atoms.

        ``None`` for a norm-conserving species: it has projectors, but no
        augmentation charge, so no ``becsum`` is needed from it.
        """
        offsets = self.augmentation.channel_offsets
        channels = []
        for t, atoms in enumerate(self.augmentation.species_atoms):
            nh = self.pseudos[t].nh
            if not self.pseudos[t].is_ultrasoft or nh == 0 or not atoms:
                channels.append(None)
                continue
            channels.append(
                jnp.asarray([[offsets[a] + i for i in range(nh)] for a in atoms])
            )
        return tuple(channels)

    def at_positions(self, positions: jnp.ndarray) -> "Calculation":
        """The same calculation with the atoms somewhere else.

        Everything that does not depend on where the atoms are is shared with
        ``self`` rather than rebuilt: the plane-wave basis and both FFT grids,
        the radial tables of every species, the symmetry group, the PAW and
        spin-orbit coefficients. What is rebuilt is exactly what the structure
        factor multiplies -- the local potential, the core charge, the
        projectors, the augmentation charge's phases -- plus the Ewald sum.

        The method is **traceable**: given traced ``positions`` it returns a
        calculation whose position-dependent arrays are traced, which is what
        makes the force ``grad`` of an energy evaluated through it
        (:mod:`pypresso.forces`). It is also what a relaxation step uses to move
        the atoms.

        **Two things are deliberately not recomputed**, both following
        ``setup.f90``, which runs once whatever the ion dynamics does:

        * the **FFT grid**. Its dimensions must be a multiple of the
          denominators of the crystal's fractional translations, so a geometry
          that breaks a symmetry would be given a *different grid* -- and the
          exchange-correlation energy is evaluated pointwise on it, so the
          energy would jump by ~1e-6 Ry in the middle of a relaxation for a
          reason that has nothing to do with the physics.
        * the **symmetry group**. QE finds it once and afterwards only *checks*
          it (``checkallsym``); re-searching it here would symmetrise a
          distorted structure with operations it no longer has, or -- worse --
          quietly stop symmetrising and change the answer between two steps of
          the same relaxation. :func:`pypresso.system.symmetry.check_symmetry`
          is the check to run instead.
        """
        moved = copy.copy(self)
        moved.system = eqx.tree_at(
            lambda sys: sys.structure.positions, self.system, positions
        )
        dense = self.basis.dense
        cell = self.system.cell
        structure = moved.system.structure

        qq = None if self.projectors.qq is None else self.projectors.qq
        moved.projectors = self.projector_core.at_positions(positions, qq=qq)
        if self.augmentation is not None:
            moved.augmentation = self.augmentation.at_positions(
                positions, dense.cartesian(cell)
            )

        vloc_g = combine_species(self.vloc_species, structure, cell, dense)
        moved.vltot = jnp.real(g_to_r(vloc_g, dense.fft_index, dense.grid))
        if self.rho_core_species is None:
            moved.rho_core_g = moved.rho_core = None
        else:
            moved.rho_core_g = combine_species(
                self.rho_core_species, structure, cell, dense
            )
            moved.rho_core = jnp.real(
                g_to_r(moved.rho_core_g, dense.fft_index, dense.grid)
            )

        moved.ewald = self.ewald_sum.energy(cell, positions, dense)
        return moved

    def at_kpoints(self, kpoints) -> "Calculation":
        """The same calculation on a different k-point list.

        The counterpart of :meth:`at_positions` on the other axis, and it exists
        for the same reason: **almost nothing depends on which k-points are
        asked for.** The G-vector sets of both grids, the FFT dimensions, the
        local potential, the core charge, the augmentation charge, the Ewald
        sum, the symmetry group, the radial tables, PAW's one-centre setup and
        the spin-orbit coefficients are all properties of the cell and the
        atoms. What a new k-list changes is only what carries a ``k`` index --
        the plane-wave spheres and their padding width, ``|k+G|^2``, the box
        indices, the stick layout, and the projectors ``vkb(k)``.

        This is what makes a Berry phase affordable. Every quantity in
        :mod:`pypresso.topology` is built from states on a k-mesh or a loop, and
        the natural way to write that is one call per row -- which, when each
        call built a whole :class:`Calculation`, meant rebuilding the dense
        G set and ``Q_ij(G)`` every time: **~1 GB and 70 s per call** on
        bismuthene, more than the states of the entire mesh cost to hold. The
        choice was then between streaming (cheap in memory, ruinous in time) and
        holding the whole mesh (the opposite). Sharing the setup removes the
        choice.

        The k-independent arrays are *shared*, not copied, so ``n`` calls hold
        one copy between them.
        """
        system = eqx.tree_at(lambda sys: sys.kpoints, self.system, kpoints)
        moved = copy.copy(self)
        moved.system = system

        smooth, cell = self.basis.smooth, self.system.cell
        planewaves = build_plane_wave_basis(smooth, kpoints, cell, system.ecutwfc)
        moved.basis = Basis(
            dense=self.basis.dense, smooth=smooth, planewaves=planewaves
        )
        moved.kinetic = planewaves.kinetic(smooth, kpoints, cell)
        moved.fft_index = planewaves.fft_index(smooth)
        moved.sticks = build_sticks(moved.fft_index, planewaves.mask, smooth.grid)

        # The projectors are rebuilt whole: their radial half is tabulated
        # against ``|k+G|``, so unlike a change of position this is not a matter
        # of a new structure factor over a cached core.
        moved.projector_core = build_projector_core(
            self.pseudos, system.structure, cell, smooth, planewaves, kpoints
        )
        moved.projectors = moved.projector_core.at_positions(
            system.structure.positions, qq=self.projectors.qq
        )
        return moved

    @property
    def is_ultrasoft(self) -> bool:
        return self.augmentation is not None

    @property
    def is_paw(self) -> bool:
        return self.paw is not None

    def potential(self, rho_r: jnp.ndarray):
        """``v_of_rho`` for this calculation: Hartree plus exchange-correlation.

        Everything the potential needs and the density does not carry -- the
        core charge on both grids, and which functional is in use -- comes from
        here rather than from a default, so that the same density cannot produce
        two different potentials depending on which call site built it.
        """
        return _potential_of_rho(
            rho_r,
            self.basis.dense,
            self.system.cell,
            self.rho_core,
            self.functional,
            self.rho_core_g,
        )

    def onecenter(self, becsum_):
        """``(epaw, ddd_paw)`` for the current ``becsum``, as a block matrix.

        ``(0, None)`` when no species is PAW. ``ddd_paw`` is
        ``(nspin_mag, nkb, nkb)``: the one-centre potential differs between the
        components exactly as the grid potential does, and there are as many of
        them as the density has -- one for a nonmagnetic spin-orbit run, whose
        spin structure comes entirely from ``fcoef`` afterwards and not from
        the one-centre terms.
        """
        if self.paw is None:
            return jnp.asarray(0.0), None
        energy, blocks = _paw_onecenter(self.paw, becsum_)
        return energy, jnp.stack([
            self.augmentation.block_matrix(
                tuple(None if b is None else b[spin] for b in blocks)
            )
            for spin in range(self.nspin_mag)
        ])

    def becsum(self, wavefunctions, weights) -> tuple:
        """``becsum`` for every ultrasoft species, or ``()`` when there are none."""
        if not self.is_ultrasoft:
            return ()
        if self.noncolin:
            values = self._noncollinear_becsum(wavefunctions, weights)
        else:
            values = becsum(
                wavefunctions, self.projectors.vkb, weights, self.species_channels,
                self.k_batch,
            )
        if self._becsum_symmetry is not None:
            values = self._becsum_symmetry.apply(values)
        return values

    def _noncollinear_becsum(self, wavefunctions, weights) -> tuple:
        """``sum_bec`` then ``add_becsum_so``, per species.

        The projector occupations are accumulated as a spin-density *matrix* and
        only then contracted with the spin-orbit coefficients into the
        ``nspin_mag`` real components everything downstream wants. Doing it in
        that order is not an implementation detail: the intermediate is the only
        place the two spin components of the spinor still appear separately, and
        it is what a fully-relativistic species needs in order to know which
        ``j`` shell its occupation belongs to.
        """
        spinors = spinor_becsum(
            wavefunctions[0], self.projectors.vkb, weights[0], self.species_channels,
            self.k_batch,
        )
        values = []
        for t, block in enumerate(spinors):
            if block is None:
                values.append(None)
                continue
            values.append(
                becsum_transform(
                    jnp.asarray(self.spin_orbit[t].fcoef).astype(block.dtype),
                    block,
                    self.nspin_mag,
                )
            )
        return tuple(values)

    def coefficients(self, potential: jnp.ndarray, ddd_paw=None) -> jnp.ndarray | None:
        """``newd``: the ``D_ij`` the Hamiltonian should use with this potential.

        ``PW/src/newd_acc.f90``. ``D_ij^a = D_ij^(0) + int V_eff(r) Q_ij^a(r) dr``
        with ``V_eff`` the **total** local potential -- ``vltot`` included, which
        ``newq_acc`` folds in through its ``skip_vltot = .false.`` argument. The
        integral is done on the dense grid in G space, where ``Q_ij(G)`` already
        is. ``None`` means "nothing to rebuild": the norm-conserving case, where
        the file's ``D_ij`` is the answer for the whole run.
        """
        if self.noncolin:
            return self._noncollinear_coefficients(potential, ddd_paw)
        if not self.is_ultrasoft:
            return None
        dense = self.basis.dense
        deeq = _newd(
            potential, dense.fft_index, self.projectors.dij, self.augmentation
        )
        # ``add_paw_to_deeq``: the one-centre coefficients enter the nonlocal
        # term in exactly the same place the ultrasoft integral does.
        return deeq if ddd_paw is None else deeq + ddd_paw

    def _noncollinear_coefficients(self, potential, ddd_paw=None):
        """``deeq_nc``: the ``(2, 2, nkb, nkb)`` coefficients of a spinor run.

        Three things happen in the order ``newd_us`` does them, and the order is
        the point:

        1. the augmentation charge is integrated against each component of the
           potential, giving one *scalar* matrix per component;
        2. ``add_paw_to_deeq`` adds PAW's one-centre coefficients to those
           scalars -- **before** the spin transform, not after, because they are
           an addition to the same integral;
        3. only then are the components recombined into spin blocks and
           sandwiched between the spin-orbit coefficients.

        Adding ``ddd_paw`` after step 3 instead would put the one-centre term
        into the wrong spin structure, which converges perfectly well to the
        wrong answer.

        A norm-conserving spin-orbit run has no step 1 or 2 at all -- there is
        no augmentation charge to integrate -- and its coefficients are
        ``dvan_so``, fixed for the whole run.
        """
        if not self.is_ultrasoft:
            return self.dvan_so
        dense = self.basis.dense
        components = jnp.stack([
            self.augmentation.block_matrix(
                self.augmentation.integrals(r_to_g(channel, dense.fft_index))
            )
            for channel in potential
        ])
        if ddd_paw is not None:
            components = components + ddd_paw
        return _newd_noncollinear(components, self.dvan_so, self.fcoef_matrix)

    def augmented(self, rho_r: jnp.ndarray, becsum_) -> jnp.ndarray:
        """``addusdens``: the augmentation charge added to a real-space density."""
        if not self.is_ultrasoft:
            return rho_r
        dense = self.basis.dense
        return _addusdens(rho_r, dense.fft_index, dense.grid, self.augmentation, becsum_)

    def symmetrize(self, rho_r: jnp.ndarray) -> jnp.ndarray:
        """Impose the crystal symmetry on a real-space density."""
        if self._symmetry_maps is None:
            return rho_r
        if self.nspin_mag == 4:
            # ``sym_rho`` treats the magnetization as an axial vector: a
            # symmetry operation permutes the grid *and* rotates the three
            # components into each other, with a sign from the determinant and
            # a further one from time reversal on a magnetic operation. None of
            # that is written here, and symmetrising the components
            # independently -- which is what the collinear path would do -- is
            # not an approximation but a different, wrong, symmetry.
            raise NotImplementedError(
                "symmetrising a noncollinear magnetization is not implemented; "
                "run with nosym = .true., or with zero starting_magnetization "
                "(a nonmagnetic spin-orbit run has a scalar density and is "
                "symmetrised normally)"
            )
        gvectors = self.basis.dense
        return _symmetrize(rho_r, gvectors.fft_index, gvectors.grid, self._symmetry_maps)

    def density(self, wavefunctions, weights, becsum_=None) -> jnp.ndarray:
        """The symmetrised output density from the occupied states.

        ``sum_band`` runs on the smooth grid, where the wavefunctions are; the
        result is lifted to the dense grid before it is symmetrised, because the
        dense grid is where the density is mixed, where the potential is built
        from it, and -- once there is an augmentation charge -- where the rest
        of it is added.
        """
        dense, smooth = self.basis.dense, self.basis.smooth
        if becsum_ is None:
            becsum_ = self.becsum(wavefunctions, weights)
        if self.noncolin:
            rho = _spinor_density_of_bands(
                wavefunctions[0], self.fft_index, smooth.grid, weights[0],
                self.system.cell, self.nspin_mag, self.k_batch,
            )
        else:
            rho = _density_of_bands(
                wavefunctions, self.fft_index, smooth.grid, weights, self.system.cell,
                self.k_batch,
            )
        return self.symmetrize(self.augmented(to_dense(rho, smooth, dense), becsum_))

    def hamiltonian(self, v_scf: jnp.ndarray, ddd_paw=None) -> tuple:
        """One :class:`Hamiltonian` per spin channel.

        The kinetic term, the projectors and the local pseudopotential are
        shared; the self-consistent potential and ``D_ij`` are not. Returned as
        a tuple rather than one object with a spin axis because the eigensolver
        is written for a single Hamiltonian and a channel is simply another
        problem to solve -- which is also how QE sees it, its ``2 nks`` k-list
        differing only in which ``vrs(:, isk)`` each point reads.
        """
        # ``set_vrs`` adds the fixed local pseudopotential to the self-consistent
        # part on the dense grid, and ``interpolate`` hands the wavefunction
        # transforms a smooth-grid copy. ``newd`` reads the *dense* one, since
        # the augmentation charge it integrates against is only representable
        # there.
        # ``set_vrs``: the local pseudopotential is felt in full by both
        # channels of an (up, down) potential and only by the charge component
        # of an (n, m) one. That is *not* the rule an unpolarized density
        # follows -- see as_potential_components -- and using the density's rule
        # here halves the local pseudopotential in an LSDA run.
        total = v_scf + as_potential_components(self.vltot, self.nspin_mag)
        deeq = self.coefficients(total, ddd_paw)
        if self.noncolin:
            return (self._spinor_hamiltonian(total, deeq),)
        hamiltonians = []
        for spin in range(self.nspin):
            potential = to_smooth(total[spin], self.basis.dense, self.basis.smooth)
            hamiltonians.append(Hamiltonian(
                kinetic=self.kinetic,
                potential=potential,
                # the same potential with its xy plane contiguous, which is the
                # layout the stick transforms hold the field in
                potential_wave=jnp.moveaxis(potential, -1, -3),
                sticks=self.sticks,
                fft_index=self.fft_index,
                mask=self.basis.planewaves.mask,
                projectors=self.projectors,
                grid=self.basis.smooth.grid,
                resolves_differences=self.resolves_differences,
                deeq=None if deeq is None else deeq[spin],
            ))
        return tuple(hamiltonians)

    def _spinor_hamiltonian(self, total: jnp.ndarray, deeq) -> SpinorHamiltonian:
        """The single noncollinear Hamiltonian, at the given total potential."""
        potential = jnp.stack([
            to_smooth(component, self.basis.dense, self.basis.smooth)
            for component in total
        ])
        return SpinorHamiltonian(
            kinetic=self.kinetic,
            potential=potential,
            potential_wave=jnp.moveaxis(potential, -1, -3),
            sticks=self.sticks,
            fft_index=self.fft_index,
            mask=self.basis.planewaves.mask,
            projectors=self.projectors,
            deeq=deeq,
            grid=self.basis.smooth.grid,
            resolves_differences=self.resolves_differences,
            qq=self.qq_so,
        )

    def starting_density(self) -> jnp.ndarray:
        """The superposition of atomic charges, split by ``starting_magnetization``.

        ``atomic_rho_g``: the total is the sum of the species' tabulated atomic
        charges, and the magnetization is the same sum weighted by each species'
        ``starting_magnetization``. Nothing else in the calculation breaks the
        spin symmetry -- with every ``starting_magnetization`` at zero the two
        channels start identical, stay identical, and the run converges to the
        unpolarized solution, which for a symmetric crystal is always a
        stationary point. That is why QE demands the input variable rather than
        finding the magnetic state on its own.
        """
        dense = self.basis.dense
        if self.nspin_mag == 1:
            rho_g = starting_charge(
                self.pseudos, self.system.structure, self.system.cell, dense,
                self.nelec,
            )
            return jnp.real(g_to_r(rho_g, dense.fft_index, dense.grid))[None]

        if self.noncolin:
            return self._noncollinear_starting_density()

        rho_g, magnetization_g = starting_charge(
            self.pseudos, self.system.structure, self.system.cell, dense, self.nelec,
            magnetization=self.spin_weights[0] - self.spin_weights[1],
        )
        channels = jnp.stack([rho_g + magnetization_g, rho_g - magnetization_g]) / 2.0
        return jnp.real(g_to_r(channels, dense.fft_index, dense.grid))

    def _noncollinear_starting_density(self) -> jnp.ndarray:
        """``atomic_rho_g`` with ``nspin = 4``: charge, and a magnetization vector.

        The magnetization is the same superposition of atomic charges the
        collinear case builds, once per cartesian component, with each species'
        contribution weighted by its ``starting_magnetization`` *and* by the
        direction ``(angle1, angle2)`` points in. Three sums rather than one is
        the whole difference -- and it is why the angles are per species: two
        sublattices pointing different ways is exactly what a noncollinear
        calculation exists to describe, and a single common axis would make it a
        collinear one in disguise.
        """
        dense = self.basis.dense
        directions = self.magnetization_directions  # (ntyp, 3)
        magnitudes = self.starting_magnetization
        components = []
        rho_g = None
        for axis in range(3):
            rho_g, component = starting_charge(
                self.pseudos, self.system.structure, self.system.cell, dense,
                self.nelec, magnetization=magnitudes * directions[:, axis],
            )
            components.append(component)
        channels = jnp.stack([rho_g, *components])
        return jnp.real(g_to_r(channels, dense.fft_index, dense.grid))

    @property
    def starting_magnetization(self) -> np.ndarray:
        """``starting_magnetization`` per species, padded to ``ntyp``."""
        ntyp = self.system.structure.ntyp
        values = np.zeros(ntyp)
        given = np.asarray(self.system.starting_magnetization, dtype=float)
        values[: given.size] = given[:ntyp]
        return values

    @property
    def magnetization_directions(self) -> np.ndarray:
        """``(ntyp, 3)`` unit vectors from ``angle1``/``angle2``, in degrees.

        ``input.f90`` converts the input's degrees to radians and
        ``angle1``/``angle2`` are the polar and azimuthal angles of that
        species' moment; both default to zero, which points along ``z`` and is
        what makes an unspecified noncollinear run start out collinear.
        """
        ntyp = self.system.structure.ntyp
        theta = np.zeros(ntyp)
        phi = np.zeros(ntyp)
        for target, given in ((theta, self.system.angle1), (phi, self.system.angle2)):
            values = np.asarray(given, dtype=float)
            target[: values.size] = np.radians(values[:ntyp])
        return np.stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ], axis=1)

    @property
    def spin_weights(self) -> np.ndarray:
        """``0.5 (1 +- starting_magnetization)`` per species, or ``[1]``.

        The split ``PAW_atomic_becsum`` applies to the reference occupations,
        which is the same split :meth:`starting_density` applies to the atomic
        charge -- the two starting guesses have to agree about how polarized the
        atom is or the first iteration contradicts itself.
        """
        ntyp = self.system.structure.ntyp
        magnetization = np.zeros(ntyp)
        given = np.asarray(self.system.starting_magnetization, dtype=float)
        magnetization[: given.size] = given[:ntyp]
        if self.nspin == 1:
            return np.ones((1, ntyp))
        return 0.5 * np.stack([1.0 + magnetization, 1.0 - magnetization])

    def starting_becsum(self) -> tuple:
        """The projector occupations of isolated atoms (``PAW_atomic_becsum``).

        PAW needs a ``becsum`` before there are any wavefunctions, because the
        one-centre potential enters the very first Hamiltonian. QE takes it from
        the reference occupations the pseudopotential file records, spread evenly
        over the ``2l+1`` channels of each projector -- the atom's own ground
        state, which is the same information the starting density comes from.
        """
        if not self.is_ultrasoft:
            return ()
        becsum = []
        for t, atoms in enumerate(self.augmentation.species_atoms):
            channels = self.species_channels[t]
            if channels is None or not atoms:
                becsum.append(None)
                continue
            pseudo = self.pseudos[t]
            occupations = (
                pseudo.paw.occupations if pseudo.paw is not None
                else np.zeros(pseudo.nbeta)
            )
            diagonal = np.array([
                occupations[nb] / (2 * l + 1) for nb, l, _ in projector_channels(pseudo)
            ])
            per_spin = self._becsum_split(t) [:, None] * diagonal[None, :]
            becsum.append(
                jnp.broadcast_to(
                    jnp.stack([jnp.diag(jnp.asarray(row)) for row in per_spin])[:, None],
                    (self.nspin_mag, len(atoms), pseudo.nh, pseudo.nh),
                )
            )
        return tuple(becsum)

    def _becsum_split(self, t: int) -> np.ndarray:
        """How one species' reference occupations divide over ``nspin_mag``.

        ``PAW_atomic_becsum``. Collinear: the ``(1 +- m)/2`` pair. Noncollinear
        with a magnetization: the occupation itself, then ``m`` times the
        direction it points in -- which is the same information written in the
        other representation, and has to agree with what
        :meth:`_noncollinear_starting_density` does to the charge or the first
        iteration contradicts itself.
        """
        if not self.noncolin:
            return self.spin_weights[:, t]
        if self.nspin_mag == 1:
            return np.ones(1)
        moment = self.starting_magnetization[t] * self.magnetization_directions[t]
        return np.concatenate([[1.0], moment])

    def starting_wavefunctions(self, hamiltonians, nbnd: int) -> jnp.ndarray:
        """The first guess at the wavefunctions, from the atomic orbitals.

        QE's ``wfcinit``: build the pseudo-atomic orbitals of every atom, then
        diagonalise the Hamiltonian inside their span. What comes out is not the
        answer, but it is close enough that the first Davidson call costs two
        steps instead of eight -- the atoms already know roughly where their
        electrons are, and the pseudopotential file carries that knowledge.

        Falls back to random vectors for a pseudopotential with no ``PP_PSWFC``
        section, and tops up with them when a species has fewer orbitals than
        the calculation has bands.
        """
        atomic = atomic_wavefunctions(
            self.pseudos, self.system.structure, self.system.cell,
            self.basis.smooth, self.basis.planewaves, self.system.kpoints,
        )
        if self.noncolin:
            atomic = self._as_spinors(atomic)

        missing = nbnd - atomic.shape[1]
        if missing > 0:
            # Aluminium has four atomic orbitals and a smeared calculation asks
            # for six bands; the rest are random, exactly as QE tops up.
            ndim = self.npol * self.basis.npwx
            tiled = jnp.tile(self.basis.planewaves.mask, (1, self.npol))
            extra = map_k(
                lambda arrays: starting_vectors(
                    None, missing, ndim, arrays[0], arrays[1], atomic.dtype
                ),
                (jnp.tile(self.kinetic, (1, self.npol)), tiled),
                batch=self.k_batch,
            )
            atomic = jnp.concatenate([atomic, extra], axis=1)

        # The same atomic orbitals seed both channels; what differs is the
        # Hamiltonian they are then diagonalised inside, which is already
        # spin-split at the first iteration because the starting density is.
        return jnp.stack([
            _rotate_all(hamiltonian, atomic, nbnd, self.k_batch)[1]
            for hamiltonian in hamiltonians
        ])

    def _as_spinors(self, atomic: jnp.ndarray) -> jnp.ndarray:
        """Scalar atomic orbitals -> twice as many spinors, ``(nk, 2 n, 2 npwx)``.

        Each orbital is used twice, once in each spin component, which spans the
        same space as QE's ``atomic_wfc_nc``. It is *not* the same set of
        vectors: with spin-orbit coupling QE builds the ``j``-resolved
        spin-angle functions (``atomic_wfc_so``), which are already close to the
        eigenstates. The difference is entirely one of convergence -- both spans
        are then diagonalised by ``rotate_wfc``, and what comes out of that is
        what the SCF starts from -- so this is a slower start, not a different
        calculation.
        """
        nk, count, npwx = atomic.shape
        zero = jnp.zeros_like(atomic)
        up = jnp.concatenate([atomic, zero], axis=-1)
        down = jnp.concatenate([zero, atomic], axis=-1)
        return jnp.concatenate([up, down], axis=1)

    def diagonalize(self, hamiltonians, nbnd: int, psi0=None, ethr=None):
        """Solve at every k-point of every channel.

        Returns ``(eigenvalues, wavefunctions)`` shaped ``(nspin, nk, nbnd)``
        and ``(nspin, nk, nbnd, npwx)``.

        ``psi0`` is the previous iteration's wavefunctions. Passing them is what
        makes an iterative solver cheap after the first SCF iteration: the
        density moves a little, so the eigenvectors do too, and Davidson starts
        one step away from the answer instead of from a random guess.

        ``ethr`` is how accurately to converge each eigenvalue. A direct solver
        ignores both.
        """
        solved = [
            self.eigensolver(
                hamiltonian, nbnd, None if psi0 is None else psi0[spin], ethr,
                k_batch=self.k_batch,
            )
            for spin, hamiltonian in enumerate(hamiltonians)
        ]
        return (
            jnp.stack([values for values, _ in solved]),
            jnp.stack([vectors for _, vectors in solved]),
        )

    @property
    def two_fermi_energies(self) -> bool:
        """Whether the magnetization is constrained (QE's ``two_fermi_energies``).

        Set by ``tot_magnetization`` being given at all, not by its value:
        ``input.f90`` turns the sentinel into the flag, and the flag decides
        whether the channels share a Fermi level or each gets its own.
        """
        return self.nspin == 2 and self.system.tot_magnetization is not None

    def occupations(self, eigenvalues):
        """Occupation weights, plus whichever level statistic applies.

        ``eigenvalues`` is ``(nspin, nk, nbnd)`` and the weights come back the
        same shape.
        """
        weights = self.system.kpoints.weights
        scheme = self.system.occupations

        # How many electrons one band holds. It is the factor the k-point
        # weights already carry, and it appears here only where a *count* of
        # bands is taken -- a spinor band holds one electron, not two.
        degeneracy = 1 if self.noncolin else 2

        if scheme == "fixed":
            wg, homo, lumo = fixed_occupations(
                eigenvalues, weights, self.nelec, degeneracy
            )
            return wg, {"homo": float(homo), "lumo": None if lumo is None else float(lumo)}

        if scheme == "from_input":
            if self.system.input_occupations is None:
                raise ValueError("occupations='from_input' needs an OCCUPATIONS card")
            return input_occupations(
                self.system.input_occupations, eigenvalues, weights, degeneracy
            ), {}

        if scheme.startswith("tetrahedra"):
            # The tetrahedra are a property of the k-grid and the symmetry, not
            # of the density, so they are built once and kept. There is no
            # ``-TS`` term: the method integrates the true step function, which
            # is why QE prints no "smearing contrib." for a tetrahedron run.
            if getattr(self, "_tetrahedra", None) is None:
                self._tetrahedra = tetrahedra_for(
                    scheme, self.system.kpoints, self.symmetries, self.system.cell
                )
            counts = (self.nelup, self.neldw) if self.two_fermi_energies else None
            wg, ef = tetrahedron_occupations_spin(
                self._tetrahedra, eigenvalues, weights, self.nelec, counts=counts
            )
            if counts is None:
                return wg, {"fermi_energy": float(ef)}
            return wg, {
                "fermi_energy_up": float(ef[0]),
                "fermi_energy_down": float(ef[1]),
                "fermi_energy": 0.5 * (float(ef[0]) + float(ef[1])),
            }

        counts = (self.nelup, self.neldw) if self.two_fermi_energies else None
        wg, ef = smeared_occupations(
            eigenvalues, weights, self.nelec, self.system.degauss,
            self.system.smearing, counts=counts,
        )
        entropy = float(
            smearing_entropy(
                eigenvalues, weights, ef, self.system.degauss, self.system.smearing
            )
        )
        levels = {"smearing": entropy}
        if self.two_fermi_energies:
            levels["fermi_energy_up"] = float(ef[0])
            levels["fermi_energy_down"] = float(ef[1])
            # QE prints the mean as "the" Fermi energy in this case, with a
            # comment saying it is only to keep the printed value from being NaN.
            levels["fermi_energy"] = 0.5 * (float(ef[0]) + float(ef[1]))
        else:
            levels["fermi_energy"] = float(ef)
        return wg, levels


def run_scf(
    system: System,
    pseudos: tuple[Pseudopotential, ...],
    nbnd: int | None = None,
    conv_thr: float = 1.0e-6,
    max_iterations: int = 100,
    mixing_mode: str = "anderson",
    mixing_beta: float = 0.7,
    calculation: Calculation | None = None,
    diagonalization: str | None = None,
    verbose: bool = False,
    k_batch: int | None | str = "default",
    starting_density: jnp.ndarray | None = None,
    starting_becsum: tuple | None = None,
) -> SCFResult:
    """Run the self-consistent field loop to convergence.

    ``conv_thr`` is compared against the estimated self-consistency error --
    QE's ``dr2``, the Hartree energy of the density residual, in Ry -- so it
    means the same thing here as in a ``pw.x`` input.

    ``k_batch`` is how many k-points are held in flight at once: 1 is QE's
    ``k_loop``, ``None`` is one ``vmap`` over the whole axis, and it trades
    memory against speed without touching the answer
    (:mod:`pypresso.batching`). It is ignored when ``calculation`` is given,
    which already carries its own.

    ``starting_density`` replaces the superposition of atomic charges the run
    would otherwise start from. It is what a relaxation hands the next geometry
    (``PW/src/update_pot.f90``): the density of the previous ionic step is a far
    better guess than the atomic one and costs several SCF iterations less.
    ``starting_becsum`` is its ultrasoft/PAW counterpart, and the two belong
    together -- the mixed state is the pair, and giving one without the other
    starts the run from two different geometries at once.
    """
    calculation = calculation or Calculation(
        system, pseudos, diagonalization=diagonalization, k_batch=k_batch
    )
    nbnd = nbnd or system.nbnd or default_nbnd(
        calculation.nelec,
        system.occupations,
        *((calculation.nelup, calculation.neldw) if system.nspin == 2 else (None, None)),
        noncolin=system.noncolin,
    )

    mixer = get_mixer(mixing_mode, beta=mixing_beta)
    rho = (
        calculation.starting_density() if starting_density is None
        else jnp.asarray(starting_density)
    )
    # ``becsum`` is mixed alongside the density, not derived from it. For an
    # ultrasoft run it could be recomputed from the wavefunctions at any point,
    # but for PAW the one-centre potential is built from it *before* the
    # Hamiltonian exists, so it has to be part of the mixed state -- which is
    # why ``mix_rho.f90`` says it mixes "rho in g-space ... and becsum (for
    # paw)". The starting value is the isolated atoms', matching the starting
    # density.
    becsum_state = (
        calculation.starting_becsum() if starting_becsum is None else starting_becsum
    )

    previous_energy, history = None, []
    potential_change = None
    converged = False
    wavefunctions = None
    ethr, accuracy = ETHR_INIT, None

    for iteration in range(1, max_iterations + 1):
        ethr = next_ethr(ethr, accuracy, calculation.nelec, iteration)

        potential = calculation.potential(rho)
        epaw, ddd_paw = calculation.onecenter(becsum_state)
        hamiltonians = calculation.hamiltonian(potential.v_scf, ddd_paw)

        # QE's threshold for judging the *first* diagonalisation after the fact:
        # if the density turns out to be better than the eigenvalues, the loose
        # starting ethr was a false economy and the iteration is redone.
        floor = ethr * max(1.0, calculation.nelec)
        if wavefunctions is None:
            wavefunctions = calculation.starting_wavefunctions(hamiltonians, nbnd)

        for attempt in range(2):
            eigenvalues, wavefunctions = calculation.diagonalize(
                hamiltonians, nbnd, wavefunctions, ethr
            )
            wg, levels = calculation.occupations(eigenvalues)
            becsum_out = calculation.becsum(wavefunctions, wg)
            rho_out = calculation.density(wavefunctions, wg, becsum_out)
            # On the dense grid, which is the grid the residual lives on. QE
            # sums rho_ddot over the *smooth* set instead (and says so, in a
            # comment noting the change from ngm to ngms); the difference is the
            # residual's high-G tail, which can only make dr2 larger, so this is
            # the conservative direction. Using the smooth GVectors here would
            # be a silent error whenever they differ: their fft_index addresses
            # a smaller box than the array being gathered from.
            accuracy = float(_accuracy(rho_out - rho, calculation.basis.dense, system.cell))

            if iteration > 1 or attempt > 0 or accuracy >= floor:
                break
            ethr = max(0.1 * accuracy / max(1.0, calculation.nelec), ETHR_MIN)
            if verbose:
                print(f"  iteration {iteration:3d}   ethr was too large; "
                      f"diagonalising again at {ethr:.2e}")

        # Which density each term is evaluated at is QE's convention, and it is
        # not uniform (``electrons.f90``, and the comment there justifying
        # ``descf``):
        #
        #   * ``eband``  -- the eigenvalues, hence the potential of the *input*
        #     density, the one the Hamiltonian was built from;
        #   * ``deband`` -- ``delta_e()``, which runs *before* ``v_of_rho`` is
        #     called again, so it pairs the **output** density with the
        #     **input** potential;
        #   * ``ehart``/``etxc`` -- ``v_of_rho`` on the density that will be
        #     used next, which at convergence is the unmixed **output** one.
        #
        # ``descf`` is QE's first-order correction for that mismatch, and it is
        # identically zero at convergence, which is the only iteration whose
        # terms are compared. Evaluating all of them at the input density
        # instead leaves each one ~1e-5 Ry away from QE's while the total --
        # being variational -- still agrees to 1e-9.
        eband, deband, residual = (
            float(x) for x in
            _iteration_scalars(
                eigenvalues, wg, rho, rho_out, potential.v_scf, system.cell.volume
            )
        )

        converged = accuracy < conv_thr
        if converged:
            # QE's ``vnew``: the potential the last step did *not* apply,
            # V[rho_out] - V[rho_in]. It is zero at exact self-consistency and it
            # is what ``force_corr`` pairs with the atomic charges to correct a
            # force for a run that stopped short (``PW/src/force_corr.f90``).
            v_in = potential.v_scf
            potential = calculation.potential(rho_out)
            potential_change = potential.v_scf - v_in
            # ... and the one-centre energy with it. ``ddd_paw`` is deliberately
            # *not* refreshed: ``deband`` below pairs it with the output becsum
            # exactly as ``delta_e`` does, which runs before QE recomputes it.
            epaw, _ = calculation.onecenter(becsum_out)

        # PAW's contribution to ``deband``: ``delta_e`` subtracts
        # ``sum ddd_paw * becsum`` for the same reason it subtracts
        # ``int rho v_scf`` -- the one-centre potential is already inside every
        # eigenvalue through ``deeq``, and ``eband`` would double-count it.
        if calculation.is_paw:
            deband -= float(_paw_deband(ddd_paw, calculation.augmentation, becsum_out))

        terms = {
            "one-electron": eband + deband,
            "hartree": float(potential.ehart),
            "xc": float(potential.etxc),
            "ewald": float(calculation.ewald),
        }
        if calculation.is_paw:
            terms["one_center_paw"] = float(epaw)
        if "smearing" in levels:
            terms["smearing"] = levels["smearing"]
        total = sum(terms.values())

        change = None if previous_energy is None else abs(total - previous_energy)
        magnetization = moment = None
        if calculation.nspin == 2:
            magnetization = [
                float(x) for x in _magnetization(rho_out, system.cell.volume)
            ]
        elif calculation.nspin_mag == 4:
            values = [
                float(x) for x in _noncollinear_magnetization(rho_out, system.cell.volume)
            ]
            moment, magnetization = tuple(values[:3]), [None, values[3]]

        entry = {"iteration": iteration, "total_energy": total,
                 "accuracy": accuracy, "ethr": ethr,
                 "residual": residual, "change": change}
        if magnetization is not None:
            entry["magnetization"] = magnetization[0]
            entry["absolute_magnetization"] = magnetization[1]
        if moment is not None:
            entry["magnetization_vector"] = moment
        history.append(entry)
        if verbose:
            if moment is not None:
                extra = (
                    f"   m = ({moment[0]:6.3f}, {moment[1]:6.3f}, {moment[2]:6.3f})"
                    f" [{magnetization[1]:6.3f}] mu_B"
                )
            elif magnetization is None:
                extra = ""
            else:
                extra = f"   m = {magnetization[0]:7.4f} ({magnetization[1]:7.4f}) mu_B"
            print(f"  iteration {iteration:3d}   E = {total:16.8f} Ry"
                  f"   accuracy = {accuracy:.2e}   ethr = {ethr:.2e}"
                  f"   |drho| = {residual:.2e}{extra}")

        if converged:
            rho = rho_out
            break

        previous_energy = total
        rho, becsum_state = _mix(mixer, rho, rho_out, becsum_state, becsum_out)

    nspin = calculation.nspin
    return SCFResult(
        converged=converged,
        iterations=iteration,
        total_energy=total,
        energy_terms=terms,
        # The spin axis is dropped when there is only one channel: an
        # unpolarized result then has exactly the shape it always had.
        eigenvalues=np.asarray(eigenvalues if nspin == 2 else eigenvalues[0]),
        occupations=np.asarray(wg if nspin == 2 else wg[0]),
        wavefunctions=wavefunctions,
        density=rho,
        potential=calculation.vltot[None] + potential.v_scf,
        potential_change=potential_change,
        accuracy=accuracy,
        nspin=nspin,
        nspin_mag=calculation.nspin_mag,
        magnetization=None if magnetization is None else magnetization[0],
        absolute_magnetization=None if magnetization is None else magnetization[1],
        magnetization_vector=moment,
        fermi_energy=levels.get("fermi_energy"),
        fermi_energy_up=levels.get("fermi_energy_up"),
        fermi_energy_down=levels.get("fermi_energy_down"),
        homo=levels.get("homo"),
        lumo=levels.get("lumo"),
        history=history,
    )
