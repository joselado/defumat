"""Assemble the full basis for a system: dense grid, smooth grid, plane waves.

QE keeps two G-vector sets. The **dense** grid holds the charge density and
potential out to ``ecutrho``; the **smooth** grid holds wavefunction-related
quantities out to ``4*ecutwfc``. For norm-conserving pseudopotentials
``ecutrho = 4*ecutwfc`` and the two coincide, which is why a norm-conserving run
prints only one grid. Ultrasoft and PAW need a larger density cutoff (``dual``
of 8 to 12) to represent the augmentation charges, and then the two differ --
``doublegrid`` in ``PW/src/setup.f90``.

**A caveat for the ultrasoft phase.** ``build_plane_wave_basis`` currently selects
from, and indexes into, the *dense* set. That is exact for norm-conserving
pseudopotentials, where the two sets are the same object, and so it is correct
for the whole first milestone. QE, however, keeps wavefunctions on the *smooth*
grid: once ``doublegrid`` is true, the wavefunction FFTs must use the smooth
grid's dimensions and index map, and ``PlaneWaveBasis.indices`` has to be rebased
onto the smooth set. Anything choosing an FFT grid for ``vloc_psi`` should read
this first.
"""

from __future__ import annotations

import equinox as eqx

from pypresso.basis.gvectors import GVectors, generate_gvectors
from pypresso.basis.planewaves import PlaneWaveBasis, build_plane_wave_basis
from pypresso.system.builder import System

__all__ = ["Basis", "build_basis"]

#: QE treats dual > 4 (within eps8) as needing a separate smooth grid.
_DOUBLEGRID_THRESHOLD = 4.0 + 1.0e-8


class Basis(eqx.Module):
    """Everything the basis set consists of, built once and then fixed."""

    dense: GVectors
    smooth: GVectors
    planewaves: PlaneWaveBasis

    @property
    def doublegrid(self) -> bool:
        """Whether the smooth grid is genuinely distinct from the dense one."""
        return self.smooth.grid != self.dense.grid or self.smooth.ngm != self.dense.ngm

    @property
    def npwx(self) -> int:
        return self.planewaves.npwx


def build_basis(system: System) -> Basis:
    """Generate the G-vectors and per-k plane waves for a system."""
    gamma_only = system.kpoints.gamma_only

    dense = generate_gvectors(system.cell, system.ecutrho, gamma_only=gamma_only)

    dual = system.ecutrho / system.ecutwfc
    if dual > _DOUBLEGRID_THRESHOLD:
        smooth = generate_gvectors(system.cell, 4.0 * system.ecutwfc, gamma_only=gamma_only)
    else:
        smooth = dense

    planewaves = build_plane_wave_basis(dense, system.kpoints, system.cell, system.ecutwfc)
    return Basis(dense=dense, smooth=smooth, planewaves=planewaves)
