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

import jax
import jax.numpy as jnp
import numpy as np

from pypresso.basis.builder import Basis, build_basis
from pypresso.basis.fft import g_to_r, r_to_g
from pypresso.hamiltonian.operator import Hamiltonian
from pypresso.pseudo.potentials import local_potential, starting_charge
from pypresso.pseudo.projectors import build_projectors
from pypresso.pseudo.upf import Pseudopotential
from pypresso.scf.density import sum_band
from pypresso.scf.ewald import ewald_energy
from pypresso.scf.mixing import get_mixer
from pypresso.scf.occupations import (
    fixed_occupations,
    input_occupations,
    smeared_occupations,
    smearing_entropy,
)
from pypresso.scf.potential import scf_accuracy, v_of_rho
from pypresso.solvers import get_eigensolver
from pypresso.solvers.davidson import ETHR_MIN
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
def _density_of_bands(psi, fft_index, grid, weights, cell, dense_index, maps):
    """The output density: ``sum_band`` and the symmetrisation, fused.

    ``maps`` is ``None`` when the cell has no symmetry beyond the identity;
    ``None`` is an empty pytree, so the two cases compile separately and neither
    carries a runtime branch.
    """
    rho = sum_band(psi, fft_index, grid, weights, cell)
    if maps is None:
        return rho
    permutations, phases = maps
    rho_g = apply_symmetry_maps(r_to_g(rho, dense_index), permutations, phases)
    return jnp.real(g_to_r(rho_g, dense_index, grid))


@jax.jit
def _iteration_scalars(eigenvalues, weights, rho_in, rho_out, v_scf, volume):
    """``(eband, deband, residual)`` as one array, so the loop syncs once.

    Each ``float()`` on a device array is a host round trip; computing the three
    together and transferring them in one go is the difference between one
    synchronisation per iteration and one per printed number.
    """
    eband = jnp.sum(weights * eigenvalues)
    deband = -volume / rho_in.size * jnp.sum(rho_in * v_scf)
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

        self.nelec = sum(self.pseudos[t].z_valence for t in system.structure.types)
        self.charges = np.array([self.pseudos[t].z_valence for t in system.structure.types])

        gvectors, planewaves = self.basis.dense, self.basis.planewaves
        self.kinetic = planewaves.kinetic(gvectors, system.kpoints, system.cell)
        self.fft_index = planewaves.fft_index(gvectors)

        self.projectors = build_projectors(
            self.pseudos, system.structure, system.cell, gvectors, planewaves, system.kpoints
        )

        vloc_g = local_potential(self.pseudos, system.structure, system.cell, gvectors)
        self.vltot = jnp.real(g_to_r(vloc_g, gvectors.fft_index, gvectors.grid))

        self.ewald = float(
            ewald_energy(system.cell, system.structure, gvectors, self.charges)
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
        self.resolves_differences = bool(
            system.ecutrho >= 4.0 * system.ecutwfc - 1e-8
            and not self.basis.dense.gamma_only
        )

        self.symmetries = find_symmetries(system.cell, system.structure)
        self._symmetry_maps = (
            symmetry_maps(gvectors, self.symmetries) if self.symmetries.nsym > 1 else None
        )

    def symmetrize(self, rho_r: jnp.ndarray) -> jnp.ndarray:
        """Impose the crystal symmetry on a real-space density."""
        if self._symmetry_maps is None:
            return rho_r
        gvectors = self.basis.dense
        return _symmetrize(rho_r, gvectors.fft_index, gvectors.grid, self._symmetry_maps)

    def density(self, wavefunctions, weights) -> jnp.ndarray:
        """The symmetrised output density from the occupied states."""
        gvectors = self.basis.dense
        return _density_of_bands(
            wavefunctions,
            self.fft_index,
            gvectors.grid,
            weights,
            self.system.cell,
            gvectors.fft_index,
            self._symmetry_maps,
        )

    def hamiltonian(self, v_scf: jnp.ndarray) -> Hamiltonian:
        return Hamiltonian(
            kinetic=self.kinetic,
            potential=self.vltot + v_scf,
            fft_index=self.fft_index,
            mask=self.basis.planewaves.mask,
            projectors=self.projectors,
            grid=self.basis.dense.grid,
            resolves_differences=self.resolves_differences,
        )

    def starting_density(self) -> jnp.ndarray:
        rho_g = starting_charge(
            self.pseudos, self.system.structure, self.system.cell, self.basis.dense, self.nelec
        )
        return jnp.real(g_to_r(rho_g, self.basis.dense.fft_index, self.basis.dense.grid))

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

    previous_energy, history = None, []
    converged = False
    wavefunctions = None
    ethr, accuracy = ETHR_INIT, None

    for iteration in range(1, max_iterations + 1):
        ethr = next_ethr(ethr, accuracy, calculation.nelec, iteration)

        potential = _potential_of_rho(rho, calculation.basis.dense, system.cell)
        hamiltonian = calculation.hamiltonian(potential.v_scf)

        # QE's threshold for judging the *first* diagonalisation after the fact:
        # if the density turns out to be better than the eigenvalues, the loose
        # starting ethr was a false economy and the iteration is redone.
        floor = ethr * max(1.0, calculation.nelec)
        for attempt in range(2):
            eigenvalues, wavefunctions = calculation.diagonalize(
                hamiltonian, nbnd, wavefunctions, ethr
            )
            wg, levels = calculation.occupations(eigenvalues)
            rho_out = calculation.density(wavefunctions, wg)
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

        eband, deband, residual = (
            float(x) for x in
            _iteration_scalars(eigenvalues, wg, rho, rho_out, potential.v_scf, system.cell.volume)
        )

        terms = {
            "one-electron": eband + deband,
            "hartree": float(potential.ehart),
            "xc": float(potential.etxc),
            "ewald": calculation.ewald,
        }
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

        if accuracy < conv_thr:
            converged = True
            # One more density from the converged potential, without mixing.
            rho = rho_out
            break

        previous_energy = total
        rho = jnp.asarray(mixer.mix(np.asarray(rho), np.asarray(rho_out)))

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
