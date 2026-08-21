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
from pypresso.system.kpoints import KPoints, for_spin as kpoints_for_spin
from pypresso.units import RY_TO_EV

__all__ = [
    "NSCFResult",
    "fixed_density_bands",
    "fixed_density_states",
    "run_nscf",
    "denser_grid",
]


@dataclass
class NSCFResult:
    """Eigenvalues on a fixed density, with whatever occupation statistic applies.

    ``eigenvalues`` and ``occupations`` are ``(nk, nbnd)`` for an unpolarized run
    and ``(2, nk, nbnd)`` for LSDA -- the spin axis is squeezed away when there
    is only one channel, the same convention
    :class:`~pypresso.scf.driver.SCFResult` uses, so that everything written
    against the unpolarized shape keeps working and a polarized result cannot be
    mistaken for one. :attr:`eigenvalues_by_spin` always has the axis.
    """

    kpoints: KPoints
    eigenvalues: np.ndarray  # (nk, nbnd) or (2, nk, nbnd), Ry
    occupations: np.ndarray | None = None  # same shape, QE's wg
    fermi_energy: float | None = None  # Ry
    homo: float | None = None  # Ry
    lumo: float | None = None  # Ry
    nspin: int = 1
    #: Only when ``tot_magnetization`` constrained the channels separately.
    fermi_energy_up: float | None = None
    fermi_energy_down: float | None = None

    @property
    def eigenvalues_ev(self) -> np.ndarray:
        return self.eigenvalues * RY_TO_EV

    @property
    def eigenvalues_by_spin(self) -> np.ndarray:
        """``(nspin, nk, nbnd)`` whatever ``nspin`` is."""
        return self.eigenvalues if self.nspin == 2 else self.eigenvalues[None]

    @property
    def occupations_by_spin(self) -> np.ndarray:
        return self.occupations if self.nspin == 2 else self.occupations[None]


def fixed_density_states(
    system: System,
    pseudos: tuple[Pseudopotential, ...],
    density: jnp.ndarray,
    kpoints: KPoints | None = None,
    nbnd: int | None = None,
    conv_thr: float = 1.0e-6,
    k_batch: int | None | str = "default",
    ns: jnp.ndarray | None = None,
):
    """Diagonalise once at every k-point of ``system`` with ``density`` fixed.

    Returns ``(calculation, system, eigenvalues, wavefunctions)``: the caller
    usually needs the :class:`~pypresso.scf.driver.Calculation` too, because the
    electron count and the symmetry group live on it and both are needed to turn
    eigenvalues into occupations. :func:`fixed_density_bands` is this without the
    wavefunctions, which is all a band structure or a density of states wants;
    a projection onto atomic orbitals is what needs them kept.

    The potential is built once from the given density and never updated -- that
    is the whole content of "non self-consistent".

    ``ns`` is the converged Hubbard occupation matrix (``SCFResult.ns``), needed
    for the same reason PAW's ``becsum`` is below: it is a property of the
    *wavefunctions*, so it cannot be rebuilt from the density this is handed,
    and the Hubbard potential is built from it.
    """
    if kpoints is not None:
        system = eqx.tree_at(lambda s: s.kpoints, system, kpoints)

    calculation = Calculation(system, pseudos, k_batch=k_batch)
    nbnd = nbnd or system.nbnd or default_nbnd(
        calculation.nelec,
        system.occupations,
        *((calculation.nelup, calculation.neldw) if system.nspin == 2 else (None, None)),
        noncolin=system.noncolin,
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

    hubbard_terms = None
    if calculation.is_hubbard:
        if ns is None:
            raise ValueError(
                "a fixed-density run with a Hubbard U needs the converged "
                "occupation matrix as well as the density: pass ns = "
                "scf_result.ns. It cannot be rebuilt from the density, and "
                "leaving the term out gives eigenvalues that look plausible "
                "and are wrong by the whole Hubbard shift"
            )
        _, _, hubbard_terms = calculation.hubbard_terms(jnp.asarray(ns))

    potential = calculation.potential(density)
    hamiltonians = calculation.hamiltonian(potential.v_scf, None, hubbard_terms)

    # There is no SCF here to tighten the threshold over, so ``setup.f90`` picks
    # one up front from the accuracy of the density the bands are computed in.
    ethr = max(ETHR_MIN, 0.1 * min(1.0e-2, conv_thr / max(1.0, calculation.nelec)))
    eigenvalues, wavefunctions = calculation.diagonalize(hamiltonians, nbnd, None, ethr)
    return calculation, system, np.asarray(eigenvalues), wavefunctions


def fixed_density_bands(*args, **kwargs):
    """:func:`fixed_density_states` without the wavefunctions.

    The wavefunctions are ``(nspin, nk, nbnd, npwx)`` complex -- the largest
    array a run holds -- so the caller that does not need them says so by
    calling this, and the buffer is free as soon as this returns.
    """
    calculation, system, eigenvalues, _ = fixed_density_states(*args, **kwargs)
    return calculation, system, eigenvalues


def run_nscf(
    system: System,
    pseudos: tuple[Pseudopotential, ...],
    density: jnp.ndarray,
    kpoints: KPoints | None = None,
    nbnd: int | None = None,
    conv_thr: float = 1.0e-6,
    k_batch: int | None | str = "default",
    ns: jnp.ndarray | None = None,
) -> NSCFResult:
    """A full NSCF run: diagonalise, then occupy by the system's own scheme.

    ``system.occupations`` decides how -- fixed, smeared or tetrahedron -- so an
    input asking for ``occupations='tetrahedra'`` gets a tetrahedron Fermi level
    here exactly as it would from an SCF run, which is what makes the DOS of a
    metal consistent with the calculation that produced its density.
    """
    calculation, system, eigenvalues = fixed_density_bands(
        system, pseudos, density, kpoints, nbnd, conv_thr, k_batch, ns
    )
    wg, levels = calculation.occupations(jnp.asarray(eigenvalues))
    nspin = calculation.nspin
    return NSCFResult(
        kpoints=system.kpoints,
        eigenvalues=eigenvalues if nspin == 2 else eigenvalues[0],
        occupations=np.asarray(wg if nspin == 2 else wg[0]),
        fermi_energy=levels.get("fermi_energy"),
        homo=levels.get("homo"),
        lumo=levels.get("lumo"),
        nspin=nspin,
        fermi_energy_up=levels.get("fermi_energy_up"),
        fermi_energy_down=levels.get("fermi_energy_down"),
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

    The weights go through :func:`pypresso.system.kpoints.for_spin`, because
    every constructor applies the spin degeneracy unconditionally and an LSDA
    run wants it halved. Skipping that step counts every electron twice on the
    denser grid, which does not fail -- it moves the Fermi level and integrates
    to the right electron count at the wrong energy.
    """
    from pypresso.system.symmetry import find_symmetries

    cell = cell if cell is not None else system.cell
    if rotations is None:
        rotations = find_symmetries(cell, system.structure).rotation_array()
    if shift is None:
        shift = system.kpoints.shift or (0, 0, 0)
    return kpoints_for_spin(
        KPoints.automatic(
            tuple(int(n) for n in grid),
            tuple(int(s) for s in shift),
            cell,
            precision=system.kpoints.precision,
            rotations=rotations,
        ),
        system.nspin,
    )
