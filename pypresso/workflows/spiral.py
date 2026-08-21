"""``E(q)``: a spin-spiral energy surface, and the exchange constants in it.

One SCF run per wavevector, and the point of collecting them is that ``E(q)``
*is* the magnon dispersion of a Heisenberg model. Mapping a classical Heisenberg
Hamiltonian ``H = - sum_ij J_ij e_i . e_j`` onto a flat spiral of unit moments
gives

    E(q) - E(0) = - m^2 sum_R J(R) [cos(q . R) - 1] = m^2 [J(0) - J(q)],

so a scan over ``q`` is the Fourier transform of the exchange constants
(Sandratskii's frozen-magnon method), and the curvature at ``q = 0`` is the spin
stiffness. That is what makes a spiral worth computing rather than a curiosity:
a handful of SCF runs give the parameters of a spin model that a supercell
calculation would need one cell per period to reach.

**Every point is an independent SCF and they share almost everything.** The
cell, the atoms, the pseudopotentials, the dense G set, the local potential and
the Ewald sum do not depend on ``q``; only the plane-wave spheres, ``|k+G|^2``,
the stick layout and ``vkb`` do. :meth:`~pypresso.scf.driver.Calculation.at_spiral_q`
rebuilds exactly those and shares the rest, the way ``at_kpoints`` does for a
k-list (P16 measured that sharing at 29.8x on a large cell).

**Reading the result.** The energies are per unit cell and include no
contribution from the field or constraint machinery (:mod:`pypresso.scf.fields`),
so differences between points are directly the magnetic energy. ``E(q)`` is even
in ``q`` and periodic under ``q -> q + 2G``; it is periodic under ``q -> q + G``
only on a k-grid invariant under a shift by ``G/2``, which is what
:mod:`pypresso.system.spiral` documents and what an even Monkhorst-Pack grid
gives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pypresso.pseudo.upf import Pseudopotential
from pypresso.scf.driver import Calculation, run_scf
from pypresso.system.builder import System

__all__ = ["SpiralScan", "run_spiral_scan", "heisenberg_exchange"]


@dataclass
class SpiralScan:
    """``E(q)`` over a list of spiral wavevectors."""

    #: ``(nq, 3)`` in lattice coordinates, as they were given.
    wavevectors: np.ndarray
    #: ``(nq,)`` total energies in Ry, per unit cell.
    energies: np.ndarray
    #: ``(nq, 3)`` the rotated-frame moment of each converged state, in Bohr
    #: magnetons -- the amplitude of the moment that turns, not a net moment.
    moments: np.ndarray
    converged: tuple
    results: tuple = field(default_factory=tuple)

    @property
    def relative(self) -> np.ndarray:
        """Energies measured from the first point, in mRy."""
        return 1.0e3 * (self.energies - self.energies[0])


def run_spiral_scan(
    system: System,
    pseudos: tuple[Pseudopotential, ...],
    wavevectors,
    keep_results: bool = False,
    **scf_options,
) -> SpiralScan:
    """One SCF per wavevector, sharing everything that does not depend on ``q``.

    Args:
        system: a spiral system -- ``noncolin``, ``nosym``, and a ``spiral_q``
            that this scan overrides point by point.
        wavevectors: ``(nq, 3)`` in lattice coordinates.
        keep_results: hold every :class:`~pypresso.scf.driver.SCFResult`. Off by default: each one
            carries its wavefunctions, which is the largest array in the run.
    """
    wavevectors = np.asarray(wavevectors, dtype=float).reshape(-1, 3)
    if not system.spiral:
        raise ValueError(
            "run_spiral_scan needs a system with spiral_q set: it is what makes "
            "the run noncollinear, symmetry-free and two-sphered"
        )

    base = Calculation(system, pseudos, k_batch=scf_options.pop("k_batch", "default"))
    energies, moments, converged, results = [], [], [], []
    for q in wavevectors:
        calculation = base.at_spiral_q(q)
        result = run_scf(
            calculation.system, pseudos, calculation=calculation, **scf_options
        )
        energies.append(result.total_energy)
        moments.append(result.magnetization_vector or (0.0, 0.0, 0.0))
        converged.append(bool(result.converged))
        if keep_results:
            results.append(result)

    return SpiralScan(
        wavevectors=wavevectors,
        energies=np.array(energies),
        moments=np.array(moments),
        converged=tuple(converged),
        results=tuple(results),
    )


def heisenberg_exchange(scan: SpiralScan, cell, shells) -> np.ndarray:
    """Fit ``E(q) - E(0) = m^2 sum_R J(R) [1 - cos(q . R)]`` for the ``J(R)``.

    Args:
        shells: ``(nshell, 3)`` lattice vectors in *crystal* coordinates, one
            per neighbour shell to fit. The moment is taken from the scan's own
            converged states, so the ``J`` come out in Ry per pair of unit
            vectors -- the convention in which ``H = -sum_ij J_ij e_i . e_j``.

    A least-squares fit rather than an inversion: the number of ``q`` points is
    usually larger than the number of shells, and the residual is the honest
    statement of how well a Heisenberg model describes the surface. A large one
    means the moments' *magnitude* is changing with ``q``, which is exactly what
    the ``moments`` column of the scan is there to show.
    """
    q = np.asarray(scan.wavevectors, dtype=float)
    shells = np.asarray(shells, dtype=float).reshape(-1, 3)
    # q is in lattice (reciprocal) coordinates and R in crystal coordinates, so
    # q . R is 2 pi times their dot product -- no metric needed, which is the
    # convenience those two conventions exist for.
    phase = 1.0 - np.cos(2.0 * np.pi * (q @ shells.T))
    amplitude = np.linalg.norm(np.asarray(scan.moments), axis=1)
    magnitude = float(np.mean(amplitude[amplitude > 0.0])) if np.any(amplitude) else 1.0
    energies = scan.energies - scan.energies[0]
    solution, *_ = np.linalg.lstsq(phase * magnitude**2, energies, rcond=None)
    return solution
