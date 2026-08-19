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
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from pypresso.basis.builder import Basis, build_basis
from pypresso.basis.interpolate import to_dense, to_smooth
from pypresso.basis.sticks import build_sticks
from pypresso.basis.fft import g_to_r, r_to_g
from pypresso.hamiltonian.operator import Hamiltonian
from pypresso.pseudo.atomic import atomic_wavefunctions
from pypresso.paw.onecenter import build_paw
from pypresso.paw.symmetry import build_becsum_symmetry
from pypresso.pseudo.augmentation import build_augmentation
from pypresso.pseudo.potentials import core_charge, local_potential, starting_charge
from pypresso.pseudo.projectors import build_projectors, projector_channels
from pypresso.pseudo.upf import Pseudopotential
from pypresso.scf.density import becsum, sum_band
from pypresso.scf.ewald import ewald_energy
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
from pypresso.scf.potential import scf_accuracy, v_of_rho
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


@partial(jax.jit, static_argnames=("grid",))
def _density_of_bands(psi, fft_index, grid, weights, cell):
    """``sum_band`` on the smooth grid, in one kernel.

    The symmetrisation used to be fused in here. It cannot be any more: it acts
    on the dense grid, and with a double grid the density has to be lifted there
    first.
    """
    return sum_band(psi, fft_index, grid, weights, cell)


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


@partial(jax.jit, static_argnames=("nbnd",))
def _rotate_all(hamiltonian, vectors, nbnd: int):
    """Rayleigh-Ritz at every k-point at once."""
    return jax.vmap(lambda ik, v: rayleigh_ritz(hamiltonian, ik, v, nbnd))(
        jnp.arange(hamiltonian.nk), vectors
    )


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


def default_nbnd(nelec: float, occupations: str, nelup=None, neldw=None) -> int:
    """QE's default band count (``PW/src/setup.f90``).

    An insulator needs exactly the occupied bands; a smeared calculation needs
    20% more so there are empty states for the Fermi level to sit among.

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
    if occupations == "fixed":
        return max(occupied, 1)
    if nelup is not None:
        return max(
            int(round(1.2 * nelec / 2.0)),
            int(round(1.2 * nelup)),
            int(round(1.2 * neldw)),
            occupied + 4,
        )
    return max(int(round(1.2 * nelec / 2.0)), occupied + 4)


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
        """``(n1, n2, n3)``: the charge density, both channels summed."""
        return jnp.sum(self.density, axis=0)


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
    ):
        system = _without_gamma_storage(system)
        self.system = system
        self.eigensolver = get_eigensolver(diagonalization)
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

        self.nspin = int(system.nspin)
        if self.nspin == 2:
            # Refused here rather than where it would first divide by something:
            # a functional whose correlation has no polarized parameterisation
            # would otherwise run with the unpolarized one and converge to a
            # number that is wrong and looks right.
            self.functional.require_spin()

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

        self.projectors = build_projectors(
            self.pseudos, system.structure, system.cell, smooth, planewaves, system.kpoints
        )

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

        vloc_g = local_potential(self.pseudos, system.structure, system.cell, dense)
        self.vltot = jnp.real(g_to_r(vloc_g, dense.fft_index, dense.grid))

        # The nonlinear core correction, on the dense grid (``set_rhoc``). It is
        # ``None`` when no species has a PP_NLCC section, and ``None`` is an
        # empty pytree, so the two cases compile separately with no runtime
        # branch in the potential.
        rho_core_g = core_charge(self.pseudos, system.structure, system.cell, dense)
        self.rho_core = (
            None if rho_core_g is None
            else jnp.real(g_to_r(rho_core_g, dense.fft_index, dense.grid))
        )
        # Kept as well as its transform: a gradient-corrected functional needs
        # the core charge's gradient, which is taken in G space alongside the
        # valence density's (``gradcorr`` adds ``rhog_core`` to ``rhogaux``).
        self.rho_core_g = rho_core_g

        self.ewald = float(
            ewald_energy(system.cell, system.structure, dense, self.charges)
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
        ``(nspin, nkb, nkb)``: the one-centre potential differs between the
        channels exactly as the grid potential does.
        """
        if self.paw is None:
            return jnp.asarray(0.0), None
        energy, blocks = _paw_onecenter(self.paw, becsum_)
        return energy, jnp.stack([
            self.augmentation.block_matrix(
                tuple(None if b is None else b[spin] for b in blocks)
            )
            for spin in range(self.nspin)
        ])

    def becsum(self, wavefunctions, weights) -> tuple:
        """``becsum`` for every ultrasoft species, or ``()`` when there are none."""
        if not self.is_ultrasoft:
            return ()
        values = becsum(
            wavefunctions, self.projectors.vkb, weights, self.species_channels
        )
        if self._becsum_symmetry is not None:
            values = self._becsum_symmetry.apply(values)
        return values

    def coefficients(self, potential: jnp.ndarray, ddd_paw=None) -> jnp.ndarray | None:
        """``newd``: the ``D_ij`` the Hamiltonian should use with this potential.

        ``PW/src/newd_acc.f90``. ``D_ij^a = D_ij^(0) + int V_eff(r) Q_ij^a(r) dr``
        with ``V_eff`` the **total** local potential -- ``vltot`` included, which
        ``newq_acc`` folds in through its ``skip_vltot = .false.`` argument. The
        integral is done on the dense grid in G space, where ``Q_ij(G)`` already
        is. ``None`` means "nothing to rebuild": the norm-conserving case, where
        the file's ``D_ij`` is the answer for the whole run.
        """
        if not self.is_ultrasoft:
            return None
        dense = self.basis.dense
        deeq = _newd(
            potential, dense.fft_index, self.projectors.dij, self.augmentation
        )
        # ``add_paw_to_deeq``: the one-centre coefficients enter the nonlocal
        # term in exactly the same place the ultrasoft integral does.
        return deeq if ddd_paw is None else deeq + ddd_paw

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
        rho = _density_of_bands(
            wavefunctions, self.fft_index, smooth.grid, weights, self.system.cell
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
        total = self.vltot[None] + v_scf
        deeq = self.coefficients(total, ddd_paw)
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
        if self.nspin == 1:
            rho_g = starting_charge(
                self.pseudos, self.system.structure, self.system.cell, dense,
                self.nelec,
            )
            return jnp.real(g_to_r(rho_g, dense.fft_index, dense.grid))[None]

        rho_g, magnetization_g = starting_charge(
            self.pseudos, self.system.structure, self.system.cell, dense, self.nelec,
            magnetization=self.spin_weights[0] - self.spin_weights[1],
        )
        channels = jnp.stack([rho_g + magnetization_g, rho_g - magnetization_g]) / 2.0
        return jnp.real(g_to_r(channels, dense.fft_index, dense.grid))

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
            per_spin = self.spin_weights[:, t][:, None] * diagonal[None, :]
            becsum.append(
                jnp.broadcast_to(
                    jnp.stack([jnp.diag(jnp.asarray(row)) for row in per_spin])[:, None],
                    (self.nspin, len(atoms), pseudo.nh, pseudo.nh),
                )
            )
        return tuple(becsum)

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
        missing = nbnd - atomic.shape[1]
        if missing > 0:
            # Aluminium has four atomic orbitals and a smeared calculation asks
            # for six bands; the rest are random, exactly as QE tops up.
            extra = jax.vmap(
                lambda kinetic, mask: starting_vectors(
                    None, missing, self.basis.npwx, kinetic, mask, atomic.dtype
                )
            )(self.kinetic, self.basis.planewaves.mask)
            atomic = jnp.concatenate([atomic, extra], axis=1)

        # The same atomic orbitals seed both channels; what differs is the
        # Hamiltonian they are then diagonalised inside, which is already
        # spin-split at the first iteration because the starting density is.
        return jnp.stack([
            _rotate_all(hamiltonian, atomic, nbnd)[1] for hamiltonian in hamiltonians
        ])

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
                hamiltonian, nbnd, None if psi0 is None else psi0[spin], ethr
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

        if scheme == "fixed":
            wg, homo, lumo = fixed_occupations(eigenvalues, weights, self.nelec)
            return wg, {"homo": float(homo), "lumo": None if lumo is None else float(lumo)}

        if scheme == "from_input":
            if self.system.input_occupations is None:
                raise ValueError("occupations='from_input' needs an OCCUPATIONS card")
            return input_occupations(self.system.input_occupations, eigenvalues, weights), {}

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
) -> SCFResult:
    """Run the self-consistent field loop to convergence.

    ``conv_thr`` is compared against the estimated self-consistency error --
    QE's ``dr2``, the Hartree energy of the density residual, in Ry -- so it
    means the same thing here as in a ``pw.x`` input.
    """
    calculation = calculation or Calculation(system, pseudos, diagonalization=diagonalization)
    nbnd = nbnd or system.nbnd or default_nbnd(
        calculation.nelec,
        system.occupations,
        *((calculation.nelup, calculation.neldw) if system.nspin == 2 else (None, None)),
    )

    mixer = get_mixer(mixing_mode, beta=mixing_beta)
    rho = calculation.starting_density()
    # ``becsum`` is mixed alongside the density, not derived from it. For an
    # ultrasoft run it could be recomputed from the wavefunctions at any point,
    # but for PAW the one-centre potential is built from it *before* the
    # Hamiltonian exists, so it has to be part of the mixed state -- which is
    # why ``mix_rho.f90`` says it mixes "rho in g-space ... and becsum (for
    # paw)". The starting value is the isolated atoms', matching the starting
    # density.
    becsum_state = calculation.starting_becsum()

    previous_energy, history = None, []
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
            potential = calculation.potential(rho_out)
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
            "ewald": calculation.ewald,
        }
        if calculation.is_paw:
            terms["one_center_paw"] = float(epaw)
        if "smearing" in levels:
            terms["smearing"] = levels["smearing"]
        total = sum(terms.values())

        change = None if previous_energy is None else abs(total - previous_energy)
        magnetization = None
        if calculation.nspin == 2:
            magnetization = [
                float(x) for x in _magnetization(rho_out, system.cell.volume)
            ]

        entry = {"iteration": iteration, "total_energy": total,
                 "accuracy": accuracy, "ethr": ethr,
                 "residual": residual, "change": change}
        if magnetization is not None:
            entry["magnetization"] = magnetization[0]
            entry["absolute_magnetization"] = magnetization[1]
        history.append(entry)
        if verbose:
            extra = (
                "" if magnetization is None
                else f"   m = {magnetization[0]:7.4f} ({magnetization[1]:7.4f}) mu_B"
            )
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
        accuracy=accuracy,
        nspin=nspin,
        magnetization=None if magnetization is None else magnetization[0],
        absolute_magnetization=None if magnetization is None else magnetization[1],
        fermi_energy=levels.get("fermi_energy"),
        fermi_energy_up=levels.get("fermi_energy_up"),
        fermi_energy_down=levels.get("fermi_energy_down"),
        homo=levels.get("homo"),
        lumo=levels.get("lumo"),
        history=history,
    )
