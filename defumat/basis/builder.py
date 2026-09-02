"""Assemble the full basis for a system: dense grid, smooth grid, plane waves.

QE keeps two G-vector sets. The **dense** grid holds the charge density and
potential out to ``ecutrho``; the **smooth** grid holds wavefunctions and
everything derived from them, out to ``4*ecutwfc``. For norm-conserving
pseudopotentials ``ecutrho = 4*ecutwfc`` and the two coincide, which is why a
norm-conserving run prints only one grid. Ultrasoft and PAW need a larger
density cutoff (``dual`` of 8 to 12) to represent the augmentation charge, and
then the two differ -- ``doublegrid`` in ``PW/src/setup.f90``.

**The smooth set is a prefix of the dense one, and that is load-bearing.** Both
are sorted by ``|G|^2``, so the G-vectors inside the smaller sphere are exactly
the first ``ngms`` entries of the larger list. QE relies on the same fact
(``ggens`` in ``Modules/recvec_subs.f90`` sets ``ngms`` and reuses the dense
``g`` array), and it is what makes moving a field between the grids a truncation
in one direction and a zero-pad in the other -- see
:mod:`defumat.basis.interpolate` -- rather than a search for matching Miller
indices. :func:`smooth_subset` builds the smooth set that way and checks the
property rather than assuming it.

The wavefunctions select from and index into the **smooth** set, as QE's do.
Only the density, the potential and the augmentation charge live on the dense
one.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from defumat.basis.fftgrid import fft_grid_dimensions, gcut_from_ecut
from defumat.basis.gvectors import GVectors, generate_gvectors
from defumat.basis.planewaves import PlaneWaveBasis, build_plane_wave_basis
from defumat.system.builder import System
from defumat.system.symmetry import find_symmetries

__all__ = ["Basis", "build_basis", "smooth_subset"]

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
    def ngms(self) -> int:
        """Number of G-vectors on the smooth grid -- QE's ``ngms``."""
        return self.smooth.ngm

    @property
    def npwx(self) -> int:
        return self.planewaves.npwx


def smooth_subset(
    dense: GVectors, cell, ecut: float, fft_factors=(1, 1, 1)
) -> GVectors:
    """The G-vectors of ``dense`` inside a smaller cutoff, on their own FFT box.

    Args:
        dense: the dense set, sorted by ``|G|^2``.
        cell: the unit cell, for the reciprocal lattice.
        ecut: the smooth cutoff in Ry, normally ``4*ecutwfc``.

    The returned set holds the *same* Miller indices as the first ``ngms``
    entries of ``dense`` -- not an independently enumerated set that happens to
    agree -- so a field's G-space coefficients can be moved between the two by
    slicing. Its ``grid`` is the smaller FFT box those vectors fit in, which is
    the box the wavefunction transforms then run on.
    """
    gcut = gcut_from_ecut(ecut, cell.alat)
    miller = np.asarray(dense.miller)
    g2 = np.sum((miller @ np.asarray(cell.bg_2pi_alat)) ** 2, axis=1)

    inside = g2 <= gcut
    ngms = int(inside.sum())
    if not inside[:ngms].all():
        # Only possible if the dense list were not sorted by |G|^2; the prefix
        # property is what interpolate.py's slicing rests on, so it is asserted
        # rather than trusted.
        raise AssertionError("the dense G-vector list is not sorted by |G|^2")

    grid = fft_grid_dimensions(
        np.asarray(cell.at_alat), np.asarray(cell.bg_2pi_alat), gcut, fft_factors
    )
    return GVectors(
        miller=jnp.asarray(miller[:ngms], dtype=jnp.int32),
        grid=tuple(int(n) for n in grid),
        ecut=float(ecut),
        gamma_only=dense.gamma_only,
    )


def build_basis(system: System) -> Basis:
    """Generate the G-vectors and per-k plane waves for a system."""
    gamma_only = system.kpoints.gamma_only

    # The FFT box must be commensurate with the crystal's fractional
    # translations, which is a property of the structure rather than of the
    # basis -- hence the symmetry search here. See ``Symmetries.fft_factors``.
    # With ``nosym`` there are no symmetry operations to be commensurate with:
    # QE sets ``nsym = 1`` and its ``fft_fact`` is then 1 along every axis.
    factors = (
        (1, 1, 1) if system.nosym
        else find_symmetries(system.cell, system.structure).fft_factors()
    )

    dense = generate_gvectors(
        system.cell, system.ecutrho, gamma_only=gamma_only, fft_factors=factors
    )

    dual = system.ecutrho / system.ecutwfc
    if dual > _DOUBLEGRID_THRESHOLD:
        smooth = smooth_subset(dense, system.cell, 4.0 * system.ecutwfc, factors)
    else:
        smooth = dense

    planewaves = build_plane_wave_basis(smooth, system.kpoints, system.cell, system.ecutwfc)
    return Basis(dense=dense, smooth=smooth, planewaves=planewaves)
