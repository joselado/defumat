"""Band structure: diagonalise at a k-path with the density held fixed.

A band structure is not a self-consistent calculation. The density comes from a
converged SCF run, the potential built from it is frozen, and the Hamiltonian is
then diagonalised at whatever k-points the band path asks for -- which is why
those k-points may be anywhere in the zone and carry no integration weights.

Following ``PW/src/non_scf.f90``. The diagonalisation itself lives in
:mod:`pypresso.workflows.nscf`, because the same routine serves an NSCF run on a
denser grid, which is what a density of states needs; what is left here is the
band-path presentation of it -- the path length, the gap, the plotting abscissa.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from pypresso.pseudo.upf import Pseudopotential
from pypresso.system.builder import System
from pypresso.system.kpoints import KPoints
from pypresso.units import RY_TO_EV
from pypresso.workflows.nscf import fixed_density_bands

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
        """Fundamental band gap in eV, for a system with fixed filling.

        The lowest conduction level anywhere on the path minus the highest
        valence level anywhere on it -- the *indirect* gap, which is the smaller
        of the two and equals the direct one when both extrema sit at the same
        k-point.

        How many bands are filled depends on how many electrons a band holds:
        two for a spin-degenerate calculation, one for a noncollinear one, where
        each band is a spinor. Getting that wrong halves or doubles the count of
        occupied bands and reports the gap between the wrong pair.
        """
        occupied = int(round(nelec / (1 if self.nspin == 4 else 2)))
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
    k_batch: int | None | str = "default",
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
    is the whole content of "non self-consistent", and it is why this is a thin
    wrapper around :func:`pypresso.workflows.nscf.fixed_density_bands`. A band
    path carries no integration weights, so unlike an NSCF grid run it stops at
    the eigenvalues: any Fermi level or HOMO must come from the SCF that
    produced the density, which is what the two arguments are for.
    """
    calculation, system, eigenvalues = fixed_density_bands(
        system, pseudos, density, kpoints, nbnd, conv_thr, k_batch
    )
    nspin = calculation.nspin
    return BandStructure(
        kpoints=system.kpoints,
        eigenvalues=np.asarray(eigenvalues if nspin == 2 else eigenvalues[0]),
        fermi_energy=fermi_energy,
        homo=homo,
        nspin=nspin,
    )
