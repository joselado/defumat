"""Non-self-consistent diagonalisation at a fixed density.

``PW/src/non_scf.f90``. The density comes from a converged SCF run, the
potential built from it is frozen, and the Hamiltonian is diagonalised wherever
the caller asks -- which is the same operation whether those k-points form a
band path (:mod:`pypresso.workflows.bands`) or a denser uniform grid for a
density of states (:mod:`pypresso.workflows.dos`). This module is that shared
core; the two workflows differ only in what they do with the eigenvalues
afterwards.

A denser grid is the whole reason a DOS is an NSCF run rather than a by-product
of the SCF: the density converges on a coarse grid, but a density of states
resolved to a few tens of meV needs an order of magnitude more k-points, and
paying for those inside the SCF loop would be waste.
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
from pypresso.system.cell import Cell
from pypresso.system.kpoints import KPoints
from pypresso.units import RY_TO_EV

__all__ = ["NSCFResult", "fixed_density_bands", "run_nscf", "denser_grid"]


@dataclass
class NSCFResult:
    """Eigenvalues on a fixed density, with whatever occupation statistic applies."""

    kpoints: KPoints
    eigenvalues: np.ndarray  # (nk, nbnd), Ry
    occupations: np.ndarray | None = None  # (nk, nbnd), QE's wg
    fermi_energy: float | None = None  # Ry
    homo: float | None = None  # Ry
    lumo: float | None = None  # Ry

    @property
    def eigenvalues_ev(self) -> np.ndarray:
        return self.eigenvalues * RY_TO_EV


def fixed_density_bands(
    system: System,
    pseudos: tuple[Pseudopotential, ...],
    density: jnp.ndarray,
    kpoints: KPoints | None = None,
    nbnd: int | None = None,
    conv_thr: float = 1.0e-6,
):
    """Diagonalise once at every k-point of ``system`` with ``density`` fixed.

    Returns ``(calculation, system, eigenvalues)``: the caller usually needs the
    :class:`~pypresso.scf.driver.Calculation` too, because the electron count and
    the symmetry group live on it and both are needed to turn eigenvalues into
    occupations.

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
        # recoverable from the density this function is handed. Building the
        # Hamiltonian without it converges perfectly well and gives eigenvalues
        # that are wrong by tenths of an eV -- the failure mode this codebase
        # refuses rather than risks. Threading becsum through ``SCFResult`` is
        # the fix; it is not written yet.
        raise NotImplementedError(
            "a fixed-density run with a PAW pseudopotential needs the converged "
            "becsum as well as the density, which is not yet carried across; "
            "ultrasoft and norm-conserving pseudopotentials work"
        )

    potential = calculation.potential(density)
    hamiltonians = calculation.hamiltonian(potential.v_scf)

    # There is no SCF here to tighten the threshold over, so ``setup.f90`` picks
    # one up front from the accuracy of the density the bands are computed in.
    ethr = max(ETHR_MIN, 0.1 * min(1.0e-2, conv_thr / max(1.0, calculation.nelec)))
    eigenvalues, _ = calculation.diagonalize(hamiltonians, nbnd, None, ethr)
    return calculation, system, np.asarray(eigenvalues)


def run_nscf(
    system: System,
    pseudos: tuple[Pseudopotential, ...],
    density: jnp.ndarray,
    kpoints: KPoints | None = None,
    nbnd: int | None = None,
    conv_thr: float = 1.0e-6,
) -> NSCFResult:
    """A full NSCF run: diagonalise, then occupy by the system's own scheme.

    ``system.occupations`` decides how -- fixed, smeared or tetrahedron -- so an
    input asking for ``occupations='tetrahedra'`` gets a tetrahedron Fermi level
    here exactly as it would from an SCF run, which is what makes the DOS of a
    metal consistent with the calculation that produced its density.
    """
    calculation, system, eigenvalues = fixed_density_bands(
        system, pseudos, density, kpoints, nbnd, conv_thr
    )
    wg, levels = calculation.occupations(jnp.asarray(eigenvalues))
    return NSCFResult(
        kpoints=system.kpoints,
        eigenvalues=eigenvalues,
        occupations=np.asarray(wg),
        fermi_energy=levels.get("fermi_energy"),
        homo=levels.get("homo"),
        lumo=levels.get("lumo"),
    )


def denser_grid(
    system: System,
    grid: tuple[int, int, int],
    shift: tuple[int, int, int] | None = None,
    cell: Cell | None = None,
    rotations: np.ndarray | None = None,
) -> KPoints:
    """The same crystal's irreducible wedge on a finer Monkhorst-Pack grid.

    Convenience for "SCF on the input's grid, DOS on a denser one": the symmetry
    used to reduce it must be the crystal's, so it is taken from the system
    rather than rediscovered, and the shift defaults to the input's own.
    """
    from pypresso.system.symmetry import find_symmetries

    cell = cell if cell is not None else system.cell
    if rotations is None:
        rotations = find_symmetries(cell, system.structure).rotation_array()
    if shift is None:
        shift = system.kpoints.shift or (0, 0, 0)
    return KPoints.automatic(
        tuple(int(n) for n in grid),
        tuple(int(s) for s in shift),
        cell,
        precision=system.kpoints.precision,
        rotations=rotations,
    )
