"""Band structure: diagonalise at a k-path with the density held fixed.

A band structure is not a self-consistent calculation. The density comes from a
converged SCF run, the potential built from it is frozen, and the Hamiltonian is
then diagonalised at whatever k-points the band path asks for -- which is why
those k-points may be anywhere in the zone and carry no integration weights.

Following ``PW/src/non_scf.f90``. The same routine serves an NSCF run on a
denser grid, which is what a density of states needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from pypresso.pseudo.upf import Pseudopotential
from pypresso.scf.driver import Calculation, default_nbnd
from pypresso.scf.potential import v_of_rho
from pypresso.solvers.davidson import ETHR_MIN
from pypresso.system.builder import System
from pypresso.system.kpoints import KPoints
from pypresso.units import RY_TO_EV

__all__ = ["BandStructure", "run_bands"]


@dataclass
class BandStructure:
    """Eigenvalues along a path, with everything needed to plot them."""

    kpoints: KPoints
    eigenvalues: np.ndarray  # (nk, nbnd), Ry
    fermi_energy: float | None = None
    homo: float | None = None

    @property
    def eigenvalues_ev(self) -> np.ndarray:
        return self.eigenvalues * RY_TO_EV

    @property
    def path_length(self) -> np.ndarray:
        """Cumulative distance along the path -- the x-axis of a band plot."""
        if self.kpoints.path_length is not None:
            return np.asarray(self.kpoints.path_length)
        coords = np.asarray(self.kpoints.coords)
        steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(steps)])

    def gap(self, nelec: float) -> float:
        """Direct-plus-indirect band gap in eV, for a system with fixed filling."""
        occupied = int(round(nelec / 2))
        return float(
            self.eigenvalues_ev[:, occupied].min() - self.eigenvalues_ev[:, occupied - 1].max()
        )


def run_bands(
    system: System,
    pseudos: tuple[Pseudopotential, ...],
    density: jnp.ndarray,
    kpoints: KPoints | None = None,
    nbnd: int | None = None,
    conv_thr: float = 1.0e-6,
    fermi_energy: float | None = None,
    homo: float | None = None,
) -> BandStructure:
    """Diagonalise at a k-path with the density fixed.

    Args:
        system: the system the density was converged for.
        pseudos: its pseudopotentials.
        density: the converged real-space density from an SCF run.
        kpoints: the path to evaluate on. Defaults to ``system.kpoints``, which
            is what an input file with ``calculation='bands'`` already carries.
        nbnd: number of bands; a band structure normally wants more than the
            occupied ones.
        conv_thr: the accuracy the density was converged to, which is what sets
            how accurately the bands are worth computing.

    The potential is built once from the given density and never updated -- that
    is the whole content of "non self-consistent".
    """
    if kpoints is not None:
        system = eqx.tree_at(lambda s: s.kpoints, system, kpoints)

    calculation = Calculation(system, pseudos)
    nbnd = nbnd or system.nbnd or default_nbnd(calculation.nelec, system.occupations)

    potential = v_of_rho(density, calculation.basis.dense, system.cell)
    hamiltonian = calculation.hamiltonian(potential.v_scf)

    # There is no SCF here to tighten the threshold over, so ``setup.f90`` picks
    # one up front from the accuracy of the density the bands are computed in.
    ethr = max(ETHR_MIN, 0.1 * min(1.0e-2, conv_thr / max(1.0, calculation.nelec)))
    eigenvalues, _ = calculation.diagonalize(hamiltonian, nbnd, None, ethr)

    return BandStructure(
        kpoints=system.kpoints,
        eigenvalues=np.asarray(eigenvalues),
        fermi_energy=fermi_energy,
        homo=homo,
    )
