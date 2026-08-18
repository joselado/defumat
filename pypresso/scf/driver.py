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
from pypresso.scf.potential import v_of_rho
from pypresso.solvers.dense import dense_eigensolver
from pypresso.system.builder import System
from pypresso.system.symmetry import find_symmetries, symmetry_maps, symmetrize_density
from pypresso.units import RY_TO_EV

__all__ = ["SCFResult", "Calculation", "run_scf", "default_nbnd"]


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

    def __init__(self, system: System, pseudos: tuple[Pseudopotential, ...], basis: Basis | None = None):
        self.system = system
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
        self.symmetries = find_symmetries(system.cell, system.structure)
        self._symmetry_maps = (
            symmetry_maps(gvectors, self.symmetries) if self.symmetries.nsym > 1 else None
        )

    def symmetrize(self, rho_r: jnp.ndarray) -> jnp.ndarray:
        """Impose the crystal symmetry on a real-space density."""
        if self.symmetries.nsym <= 1:
            return rho_r
        gvectors = self.basis.dense
        rho_g = r_to_g(rho_r, gvectors.fft_index)
        rho_g = symmetrize_density(rho_g, gvectors, self.symmetries, self._symmetry_maps)
        return jnp.real(g_to_r(rho_g, gvectors.fft_index, gvectors.grid))

    def hamiltonian(self, v_scf: jnp.ndarray) -> Hamiltonian:
        return Hamiltonian(
            kinetic=self.kinetic,
            potential=self.vltot + v_scf,
            fft_index=self.fft_index,
            mask=self.basis.planewaves.mask,
            projectors=self.projectors,
            grid=self.basis.dense.grid,
        )

    def starting_density(self) -> jnp.ndarray:
        rho_g = starting_charge(
            self.pseudos, self.system.structure, self.system.cell, self.basis.dense, self.nelec
        )
        return jnp.real(g_to_r(rho_g, self.basis.dense.fft_index, self.basis.dense.grid))

    def diagonalize(self, hamiltonian: Hamiltonian, nbnd: int):
        """Solve at every k-point. Returns ``(eigenvalues, wavefunctions)``."""
        eigenvalues, wavefunctions = [], []
        for ik in range(self.system.kpoints.nk):
            values, vectors = dense_eigensolver(hamiltonian, ik, nbnd)
            eigenvalues.append(values)
            wavefunctions.append(vectors)
        return jnp.stack(eigenvalues), jnp.stack(wavefunctions)

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
    verbose: bool = False,
) -> SCFResult:
    """Run the self-consistent field loop to convergence.

    ``conv_thr`` is compared against the change in total energy between
    iterations, in Ry -- QE's criterion is its estimated scf accuracy, which is a
    tighter bound on the same thing.
    """
    calculation = calculation or Calculation(system, pseudos)
    nbnd = nbnd or system.nbnd or default_nbnd(calculation.nelec, system.occupations)

    mixer = get_mixer(mixing_mode, beta=mixing_beta)
    rho = calculation.starting_density()

    previous_energy, history = None, []
    converged = False

    for iteration in range(1, max_iterations + 1):
        potential = v_of_rho(rho, calculation.basis.dense, system.cell)
        hamiltonian = calculation.hamiltonian(potential.v_scf)

        eigenvalues, wavefunctions = calculation.diagonalize(hamiltonian, nbnd)
        wg, levels = calculation.occupations(eigenvalues)

        rho_out = calculation.symmetrize(
            sum_band(
                wavefunctions, calculation.fft_index, calculation.basis.dense.grid, wg, system.cell
            )
        )

        eband = float(jnp.sum(wg * eigenvalues))
        n = rho.size
        deband = -float(system.cell.volume / n * jnp.sum(rho * potential.v_scf))

        terms = {
            "one-electron": eband + deband,
            "hartree": float(potential.ehart),
            "xc": float(potential.etxc),
            "ewald": calculation.ewald,
        }
        if "smearing" in levels:
            terms["smearing"] = levels["smearing"]
        total = sum(terms.values())

        residual = float(jnp.max(jnp.abs(rho_out - rho)))
        change = None if previous_energy is None else abs(total - previous_energy)
        history.append({"iteration": iteration, "total_energy": total,
                        "residual": residual, "change": change})
        if verbose:
            print(f"  iteration {iteration:3d}   E = {total:16.8f} Ry"
                  f"   dE = {'-' if change is None else f'{change:.2e}'}"
                  f"   |drho| = {residual:.2e}")

        if change is not None and change < conv_thr:
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
        fermi_energy=levels.get("fermi_energy"),
        homo=levels.get("homo"),
        lumo=levels.get("lumo"),
        history=history,
    )
