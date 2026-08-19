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
"""

from __future__ import annotations

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
)
from pypresso.scf.potential import scf_accuracy, v_of_rho
from pypresso.xc.functional import resolve_functional
from pypresso.solvers import get_eigensolver
from pypresso.solvers.davidson import ETHR_MIN, starting_vectors
from pypresso.solvers.subspace import rayleigh_ritz
from pypresso.system.builder import System
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
    """Impose the crystal symmetry on a real-space density, in one kernel."""
    permutations, phases = maps
    rho_g = apply_symmetry_maps(r_to_g(rho_r, fft_index), permutations, phases)
    return jnp.real(g_to_r(rho_g, fft_index, grid))


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
    """``sum_a sum_ij ddd_paw^a_ij becsum^a_ij``, out of the block matrix.

    Reading the blocks back out of the assembled ``(nkb, nkb)`` matrix rather
    than keeping them separately is deliberate: it is the *same* array the
    Hamiltonian used, so the two cannot drift apart.
    """
    total = jnp.asarray(0.0)
    for values, atoms in zip(becsum_, augmentation.species_atoms):
        if values is None or not atoms:
            continue
        nh = values.shape[-1]
        for n, atom in enumerate(atoms):
            start = augmentation.channel_offsets[atom]
            block = jax.lax.dynamic_slice(ddd_paw, (start, start), (nh, nh))
            total = total + jnp.sum(block * values[n])
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
    """``D^(0)_ij + int V_eff Q_ij``, as the block matrix the Hamiltonian takes."""
    potential_g = r_to_g(potential, fft_index)
    return dij + augmentation.block_matrix(augmentation.integrals(potential_g))


@partial(jax.jit, static_argnames=("grid",))
def _addusdens(rho_r, fft_index, grid, augmentation, becsum_):
    """The augmentation charge, added to the density on the dense grid."""
    charge = augmentation.charge(becsum_)
    return rho_r + jnp.real(g_to_r(charge, fft_index, grid))


@partial(jax.jit, static_argnames=("nbnd",))
def _rotate_all(hamiltonian, vectors, nbnd: int):
    """Rayleigh-Ritz at every k-point at once."""
    return jax.vmap(lambda ik, v: rayleigh_ritz(hamiltonian, ik, v, nbnd))(
        jnp.arange(hamiltonian.nk), vectors
    )


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
    deband = -volume / rho_out.size * jnp.sum(rho_out * v_scf)
    residual = jnp.max(jnp.abs(rho_out - rho_in))
    return jnp.stack([eband, deband, residual])


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


def default_nbnd(nelec: float, occupations: str) -> int:
    """QE's default band count (``PW/src/setup.f90``).

    An insulator needs exactly the occupied bands; a smeared calculation needs
    20% more so there are empty states for the Fermi level to sit among.
    """
    occupied = int(round(nelec / 2.0))
    if occupations == "fixed":
        return max(occupied, 1)
    return max(int(round(1.2 * nelec / 2.0)), occupied + 4)


@dataclass
class SCFResult:
    """Everything an SCF produces, with the energy split QE's way."""

    converged: bool
    iterations: int
    total_energy: float
    energy_terms: dict
    eigenvalues: np.ndarray  # (nk, nbnd), Ry
    occupations: np.ndarray  # (nk, nbnd), the weights wg
    wavefunctions: jnp.ndarray  # (nk, nbnd, npwx)
    density: jnp.ndarray  # (n1, n2, n3), electrons/bohr^3
    potential: jnp.ndarray  # (n1, n2, n3), Ry -- the total local potential
    fermi_energy: float | None = None
    homo: float | None = None
    lumo: float | None = None
    #: QE's estimated scf accuracy at the last iteration, in Ry.
    accuracy: float | None = None
    history: list = field(default_factory=list)

    @property
    def eigenvalues_ev(self) -> np.ndarray:
        return self.eigenvalues * RY_TO_EV


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
        self.system = system
        self.eigensolver = get_eigensolver(diagonalization)
        self.pseudos = tuple(pseudos)
        self.basis = basis if basis is not None else build_basis(system)

        # Which exchange-correlation functional this run uses is decided once,
        # here, from the pseudopotentials and the input -- not defaulted to
        # anywhere downstream. Every consumer takes it as an argument, so a PBE
        # dataset cannot end up running under LDA by omission.
        self.functional = resolve_functional(
            [pseudo.functional for pseudo in self.pseudos], system.input_dft
        )

        self.nelec = sum(self.pseudos[t].z_valence for t in system.structure.types)
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

        self.symmetries = find_symmetries(system.cell, system.structure)
        # PAW's projector occupations need the symmetry imposed on them
        # explicitly -- see pypresso.paw.symmetry. Built after the symmetry
        # search, which is why it is not up with the rest of the PAW setup.
        self._becsum_symmetry = (
            build_becsum_symmetry(self.pseudos, system.structure, system.cell,
                                  self.symmetries)
            if self.paw is not None else None
        )
        self._symmetry_maps = (
            symmetry_maps(dense, self.symmetries) if self.symmetries.nsym > 1 else None
        )

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

        ``(0, None)`` when no species is PAW.
        """
        if self.paw is None:
            return jnp.asarray(0.0), None
        energy, blocks = _paw_onecenter(self.paw, becsum_)
        return energy, self.augmentation.block_matrix(blocks)

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

    def hamiltonian(self, v_scf: jnp.ndarray, ddd_paw=None) -> Hamiltonian:
        # ``set_vrs`` adds the fixed local pseudopotential to the self-consistent
        # part on the dense grid, and ``interpolate`` hands the wavefunction
        # transforms a smooth-grid copy. ``newd`` reads the *dense* one, since
        # the augmentation charge it integrates against is only representable
        # there.
        total = self.vltot + v_scf
        deeq = self.coefficients(total, ddd_paw)
        potential = to_smooth(total, self.basis.dense, self.basis.smooth)
        return Hamiltonian(
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
            deeq=deeq,
        )

    def starting_density(self) -> jnp.ndarray:
        rho_g = starting_charge(
            self.pseudos, self.system.structure, self.system.cell, self.basis.dense, self.nelec
        )
        return jnp.real(g_to_r(rho_g, self.basis.dense.fft_index, self.basis.dense.grid))

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
            becsum.append(
                jnp.broadcast_to(jnp.diag(jnp.asarray(diagonal)),
                                 (len(atoms), pseudo.nh, pseudo.nh))
            )
        return tuple(becsum)

    def starting_wavefunctions(self, hamiltonian: Hamiltonian, nbnd: int) -> jnp.ndarray:
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

        _, vectors = _rotate_all(hamiltonian, atomic, nbnd)
        return vectors

    def diagonalize(self, hamiltonian: Hamiltonian, nbnd: int, psi0=None, ethr=None):
        """Solve at every k-point. Returns ``(eigenvalues, wavefunctions)``.

        ``psi0`` is the previous iteration's wavefunctions. Passing them is what
        makes an iterative solver cheap after the first SCF iteration: the
        density moves a little, so the eigenvectors do too, and Davidson starts
        one step away from the answer instead of from a random guess.

        ``ethr`` is how accurately to converge each eigenvalue. A direct solver
        ignores both.
        """
        return self.eigensolver(hamiltonian, nbnd, psi0, ethr)

    def occupations(self, eigenvalues):
        """Occupation weights, plus whichever level statistic applies."""
        weights = self.system.kpoints.weights
        scheme = self.system.occupations

        if scheme == "fixed":
            wg, homo, lumo = fixed_occupations(eigenvalues, weights, self.nelec)
            return wg, {"homo": float(homo), "lumo": None if lumo is None else float(lumo)}

        if scheme == "from_input":
            if self.system.input_occupations is None:
                raise ValueError("occupations='from_input' needs an OCCUPATIONS card")
            return input_occupations(self.system.input_occupations, eigenvalues, weights), {}

        wg, ef = smeared_occupations(
            eigenvalues, weights, self.nelec, self.system.degauss, self.system.smearing
        )
        entropy = float(
            smearing_entropy(
                eigenvalues, weights, ef, self.system.degauss, self.system.smearing
            )
        )
        return wg, {"fermi_energy": float(ef), "smearing": entropy}


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
    nbnd = nbnd or system.nbnd or default_nbnd(calculation.nelec, system.occupations)

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
        hamiltonian = calculation.hamiltonian(potential.v_scf, ddd_paw)

        # QE's threshold for judging the *first* diagonalisation after the fact:
        # if the density turns out to be better than the eigenvalues, the loose
        # starting ethr was a false economy and the iteration is redone.
        floor = ethr * max(1.0, calculation.nelec)
        if wavefunctions is None:
            wavefunctions = calculation.starting_wavefunctions(hamiltonian, nbnd)

        for attempt in range(2):
            eigenvalues, wavefunctions = calculation.diagonalize(
                hamiltonian, nbnd, wavefunctions, ethr
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
        history.append({"iteration": iteration, "total_energy": total,
                        "accuracy": accuracy, "ethr": ethr,
                        "residual": residual, "change": change})
        if verbose:
            print(f"  iteration {iteration:3d}   E = {total:16.8f} Ry"
                  f"   accuracy = {accuracy:.2e}   ethr = {ethr:.2e}"
                  f"   |drho| = {residual:.2e}")

        if converged:
            rho = rho_out
            break

        previous_energy = total
        rho, becsum_state = _mix(mixer, rho, rho_out, becsum_state, becsum_out)

    return SCFResult(
        converged=converged,
        iterations=iteration,
        total_energy=total,
        energy_terms=terms,
        eigenvalues=np.asarray(eigenvalues),
        occupations=np.asarray(wg),
        wavefunctions=wavefunctions,
        density=rho,
        potential=calculation.vltot + potential.v_scf,
        accuracy=accuracy,
        fermi_energy=levels.get("fermi_energy"),
        homo=levels.get("homo"),
        lumo=levels.get("lumo"),
        history=history,
    )
