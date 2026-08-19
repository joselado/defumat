"""The density of states, behind a name registry of integration schemes.

``PP/src/dos.f90``, which is a thin driver over two ways of turning a set of
eigenvalues into ``D(E)``:

* **smearing** (``PP/src/dosg.f90``) -- replace each level by a normalised
  delta of width ``degauss`` and sum over the k-points. Cheap, works on any
  k-set including one with no grid behind it, and broadens sharp structure by
  construction.
* **tetrahedra** (:mod:`pypresso.scf.tetrahedra`) -- interpolate the bands
  linearly inside the tetrahedra of the k-grid and integrate exactly. No width,
  so a gap really is empty, but it needs the uniform grid the points came from.

Both are written the same way here: only the **integrated** density of states
``N(E)`` is coded, and ``D(E)`` is ``jax.grad`` of it. That is not a shortcut --
it is what makes ``int D dE = N`` an identity rather than a test, and for the
smearing case it also guarantees the delta is exactly the derivative of the
occupation function the SCF used (see ``w0gauss`` in ``scf/occupations.py``).

A DOS is an NSCF calculation: the density converges on a coarse grid, but
resolving structure to a few tens of meV takes an order of magnitude more
k-points, so the sequence is SCF -> NSCF on a denser grid -> integrate.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from pypresso.pseudo.upf import Pseudopotential
from pypresso.scf.occupations import SMEARING_ORDER, w0gauss, wgauss
from pypresso.scf.tetrahedra import (
    ENERGY_CHUNK,
    TETRAHEDRON_KINDS,
    Tetrahedra,
    tetrahedra_for,
    tetrahedron_dos,
)
from pypresso.system.builder import System
from pypresso.system.kpoints import KPoints
from pypresso.units import RY_TO_EV
from pypresso.workflows.nscf import denser_grid, run_nscf

__all__ = [
    "DensityOfStates",
    "DOS_SCHEMES",
    "get_dos_scheme",
    "energy_grid",
    "compute_dos",
    "run_dos",
]

#: ``dos.x``'s default energy step, 0.01 eV, in Ry.
DEFAULT_DELTA_E = 0.01 / RY_TO_EV


@dataclass
class DensityOfStates:
    """``D(E)`` and ``N(E)`` on an energy grid, in Rydberg atomic units.

    ``dos`` is states per Ry per cell and ``integrated`` is states per cell, both
    already carrying the spin degeneracy -- so ``integrated`` reaches the number
    of valence electrons at the Fermi level, which is the sum rule worth
    checking. ``dos.x`` writes eV instead; that conversion belongs to
    :mod:`pypresso.io.output`.
    """

    energies: np.ndarray  # (nE,), Ry
    dos: np.ndarray  # (nE,), states/Ry
    integrated: np.ndarray  # (nE,), states
    scheme: str
    fermi_energy: float | None = None  # Ry

    @property
    def energies_ev(self) -> np.ndarray:
        return self.energies * RY_TO_EV

    @property
    def dos_ev(self) -> np.ndarray:
        """States per eV, which is what a DOS is conventionally plotted in."""
        return self.dos / RY_TO_EV

    def states_below(self, energy: float) -> float:
        """``N(E)`` at an arbitrary energy, interpolated from the grid."""
        return float(np.interp(energy, self.energies, self.integrated))

    def at(self, energy: float) -> float:
        """``D(E)`` at an arbitrary energy, in states/Ry."""
        return float(np.interp(energy, self.energies, self.dos))


# --------------------------------------------------------------------------
# The schemes
# --------------------------------------------------------------------------


def _smearing_scheme(ngauss: int):
    """``dos_g``: a normalised delta per level, summed with the k-point weights.

    ``N(E) = sum_k w_k sum_n wgauss((E - e_nk)/degauss)`` is written down and
    ``D(E)`` is its derivative, which is ``w0gauss`` over ``degauss`` -- exactly
    ``dos_g``'s expression, but reached by differentiating rather than by
    transcribing a second formula. The intermediate is ``(nE, nk, nbnd)``, five
    orders of magnitude smaller than the tetrahedron one, so it is not chunked.
    """

    def scheme(eigenvalues, weights, energies, *, degauss=None, **_):
        if not degauss:
            raise ValueError("a smearing density of states needs a positive degauss")
        x = (energies[:, None, None] - eigenvalues[None, :, :]) / degauss
        dos = jnp.einsum("k,ekb->e", weights, w0gauss(x, ngauss)) / degauss
        integrated = jnp.einsum("k,ekb->e", weights, wgauss(x, ngauss))
        return dos, integrated

    scheme.__name__ = f"smearing_dos_ngauss_{ngauss}"
    return scheme


def _tetrahedron_scheme(
    eigenvalues, weights, energies, *, tetrahedra=None, chunk=ENERGY_CHUNK, **_
):
    """``tetra_dos_t`` / ``opt_tetra_dos_t``, via :mod:`pypresso.scf.tetrahedra`."""
    if tetrahedra is None:
        raise ValueError(
            "a tetrahedron density of states needs the tetrahedra of the k-grid; "
            "pass tetrahedra=... or use run_dos, which builds them"
        )
    return tetrahedron_dos(tetrahedra, eigenvalues, weights, energies, chunk=chunk)


#: Scheme name -> implementation. Every entry has the signature
#: ``scheme(eigenvalues, weights, energies, **options) -> (dos, integrated)``
#: with ``eigenvalues`` ``(nk, nbnd)`` and everything in Rydberg atomic units.
#: Adding a scheme is a registration, not a branch in the workflow (rule R4).
DOS_SCHEMES = {name: _smearing_scheme(ngauss) for name, ngauss in SMEARING_ORDER.items()}
DOS_SCHEMES.update({name: _tetrahedron_scheme for name in TETRAHEDRON_KINDS})
#: ``dos.x`` spells the smearing family this way in its ``bz_sum`` variable.
DOS_SCHEMES["smearing"] = DOS_SCHEMES["gaussian"]


def get_dos_scheme(name: str):
    """Look up an integration scheme by the name an input file would use."""
    try:
        return DOS_SCHEMES[name.lower()]
    except KeyError as error:
        raise ValueError(
            f"unknown density-of-states scheme {name!r}; expected one of {sorted(DOS_SCHEMES)}"
        ) from error


def is_tetrahedron_scheme(name: str) -> bool:
    return name.lower() in TETRAHEDRON_KINDS


# --------------------------------------------------------------------------
# The energy grid
# --------------------------------------------------------------------------


def energy_grid(
    eigenvalues: np.ndarray,
    emin: float | None = None,
    emax: float | None = None,
    delta_e: float = DEFAULT_DELTA_E,
    degauss: float = 0.0,
) -> np.ndarray:
    """``dos.x``'s energy grid, in Ry.

    Its defaults are the bottom of the *lowest* band and the top of the
    *highest* one -- ``MINVAL(et(1,:))`` and ``MAXVAL(et(nbnd,:))``, not the
    extremes of the whole array, which differ whenever the bands cross -- padded
    by ``3*degauss`` when there is a smearing, since a delta of that width puts
    weight outside the band. The point count is
    ``nint((Emax - Emin)/DeltaE + 0.500001)``, whose odd constant is Fortran
    rounding half away from zero.
    """
    eigenvalues = np.asarray(eigenvalues)
    if emin is None:
        emin = float(eigenvalues[:, 0].min()) - (3.0 * degauss if degauss > 0.0 else 0.0)
    if emax is None:
        emax = float(eigenvalues[:, -1].max()) + (3.0 * degauss if degauss > 0.0 else 0.0)
    if delta_e <= 0.0:
        raise ValueError(f"the energy step must be positive, got {delta_e}")
    ndos = int(np.floor((emax - emin) / delta_e + 0.500001 + 0.5))
    return emin + np.arange(max(ndos, 1)) * delta_e


# --------------------------------------------------------------------------
# The calculation
# --------------------------------------------------------------------------


def compute_dos(
    eigenvalues,
    weights,
    energies,
    scheme: str,
    degauss: float | None = None,
    tetrahedra: Tetrahedra | None = None,
    fermi_energy: float | None = None,
    chunk: int = ENERGY_CHUNK,
) -> DensityOfStates:
    """Integrate a set of eigenvalues into a density of states.

    Per spin channel: ``eigenvalues`` is ``(nk, nbnd)`` and ``weights`` ``(nk,)``,
    whose sum carries the spin degeneracy exactly as everywhere else.
    """
    energies = jnp.asarray(energies)
    dos, integrated = get_dos_scheme(scheme)(
        jnp.asarray(eigenvalues),
        jnp.asarray(weights),
        energies,
        degauss=degauss,
        tetrahedra=tetrahedra,
        chunk=chunk,
    )
    return DensityOfStates(
        energies=np.asarray(energies),
        dos=np.asarray(dos),
        integrated=np.asarray(integrated),
        scheme=scheme.lower(),
        fermi_energy=fermi_energy,
    )


def run_dos(
    system: System,
    pseudos: tuple[Pseudopotential, ...],
    density: jnp.ndarray,
    grid: tuple[int, int, int] | None = None,
    shift: tuple[int, int, int] | None = None,
    kpoints: KPoints | None = None,
    nbnd: int | None = None,
    scheme: str | None = None,
    degauss: float | None = None,
    emin: float | None = None,
    emax: float | None = None,
    delta_e: float = DEFAULT_DELTA_E,
    conv_thr: float = 1.0e-6,
    chunk: int = ENERGY_CHUNK,
):
    """SCF density in, ``(DensityOfStates, NSCFResult)`` out.

    ``grid`` asks for a denser Monkhorst-Pack grid than the one the density was
    converged on, reduced with the same crystal symmetries; without it the
    calculation's own k-points are reused, which is rarely enough for a DOS.

    ``scheme`` defaults to whatever the calculation itself used to occupy its
    bands -- the tetrahedron variant if it ran with one, otherwise its smearing
    -- so the states counted here are the states the SCF counted. Failing both,
    ``dos.x``'s own fallback: a Gaussian of width ``DeltaE``.
    """
    if kpoints is None and grid is not None:
        kpoints = denser_grid(system, grid, shift)

    nscf = run_nscf(system, pseudos, density, kpoints, nbnd, conv_thr)

    if scheme is None:
        if is_tetrahedron_scheme(system.occupations):
            scheme = system.occupations
        elif system.occupations == "smearing" and system.degauss > 0.0:
            scheme, degauss = system.smearing, degauss or system.degauss
        else:
            # dos.f90's last resort when the run carried no broadening at all.
            scheme, degauss = "gaussian", degauss or delta_e

    tetrahedra = None
    if is_tetrahedron_scheme(scheme):
        from pypresso.system.symmetry import find_symmetries

        symmetries = find_symmetries(system.cell, system.structure)
        tetrahedra = tetrahedra_for(scheme, nscf.kpoints, symmetries, system.cell)
        degauss = 0.0
    elif degauss is None:
        degauss = system.degauss or delta_e

    energies = energy_grid(nscf.eigenvalues, emin, emax, delta_e, degauss or 0.0)
    dos = compute_dos(
        nscf.eigenvalues,
        nscf.kpoints.weights,
        energies,
        scheme,
        degauss=degauss,
        tetrahedra=tetrahedra,
        fermi_energy=nscf.fermi_energy if nscf.fermi_energy is not None else nscf.homo,
        chunk=chunk,
    )
    return dos, nscf
