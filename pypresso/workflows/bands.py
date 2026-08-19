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
from pypresso.solvers.davidson import ETHR_MIN
from pypresso.system.builder import System
from pypresso.system.kpoints import KPoints
from pypresso.units import RY_TO_EV

__all__ = ["BandStructure", "run_bands"]


@dataclass
class BandStructure:
    """Eigenvalues along a path, with everything needed to plot them."""

    kpoints: KPoints
    #: ``(nk, nbnd)`` unpolarized, ``(2, nk, nbnd)`` for LSDA -- the same
    #: squeeze-when-there-is-one-channel convention as :class:`SCFResult`.
    eigenvalues: np.ndarray
    fermi_energy: float | None = None
    homo: float | None = None
    nspin: int = 1

    @property
    def eigenvalues_by_spin(self) -> np.ndarray:
        """``(nspin, nk, nbnd)`` whatever ``nspin`` is."""
        return self.eigenvalues if self.nspin == 2 else self.eigenvalues[None]

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
        levels = self.eigenvalues_ev
        if self.nspin == 2:
            raise NotImplementedError(
                "a fixed-filling gap is not defined channel by channel; take it "
                "from eigenvalues_by_spin with the occupation of each channel"
            )
        return float(
            levels[:, occupied].min() - levels[:, occupied - 1].max()
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
        density: the converged real-space density from an SCF run, shaped
            ``(nspin, n1, n2, n3)``.
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
    nbnd = nbnd or system.nbnd or default_nbnd(
        calculation.nelec,
        system.occupations,
        *((calculation.nelup, calculation.neldw) if system.nspin == 2 else (None, None)),
    )

    if calculation.is_paw:
        # A PAW Hamiltonian's nonlocal coefficients are D^(0) + int V Q + ddd_paw,
        # and only the first two can be rebuilt from the density: ddd_paw comes
        # from ``becsum``, which is a property of the *wavefunctions* and is not
        # recoverable from the density that ``run_bands`` is handed. Building the
        # Hamiltonian without it converges perfectly well and gives eigenvalues
        # that are wrong by tenths of an eV -- the failure mode this codebase
        # refuses rather than risks. Threading becsum through ``SCFResult`` is
        # the fix; it is not written yet.
        raise NotImplementedError(
            "band structures with a PAW pseudopotential need the converged becsum "
            "as well as the density, which run_bands does not yet take; "
            "ultrasoft and norm-conserving band structures work"
        )

    potential = calculation.potential(density)
    hamiltonians = calculation.hamiltonian(potential.v_scf)

    # There is no SCF here to tighten the threshold over, so ``setup.f90`` picks
    # one up front from the accuracy of the density the bands are computed in.
    ethr = max(ETHR_MIN, 0.1 * min(1.0e-2, conv_thr / max(1.0, calculation.nelec)))
    eigenvalues, _ = calculation.diagonalize(hamiltonians, nbnd, None, ethr)

    nspin = calculation.nspin
    return BandStructure(
        kpoints=system.kpoints,
        eigenvalues=np.asarray(eigenvalues if nspin == 2 else eigenvalues[0]),
        fermi_energy=fermi_energy,
        homo=homo,
        nspin=nspin,
    )
