"""Per-k-point plane-wave basis sets.

The number of plane waves inside the wavefunction cutoff differs from k-point to
k-point, which is exactly the ragged shape JAX cannot tolerate. The resolution
(rule R7) is QE's own: allocate to ``npwx = max_k npw_k`` and carry a boolean
mask. Nothing downstream may branch on ``npw_k``; it multiplies by the mask
instead, so every k-point traces to the same program and the k-axis stays
``vmap``-able and sortable across devices.

Selection follows ``PW/src/gk_sort.f90``: keep G with ``|k+G|^2 <= ecutwfc``,
ordered by ``|k+G|^2``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from pypresso.basis.fftgrid import gcut_from_ecut
from pypresso.basis.gvectors import GVectors
from pypresso.system.cell import Cell
from pypresso.system.kpoints import KPoints

__all__ = ["PlaneWaveBasis", "build_plane_wave_basis"]

#: QE snaps |k+G|^2 below this to zero before comparing with the cutoff
#: (``eps8`` in gk_sort), so that a k-point at a zone boundary does not gain or
#: lose a plane wave to rounding.
_EPS = 1.0e-8


class PlaneWaveBasis(eqx.Module):
    """The plane waves retained at each k-point, padded to a common width.

    ``indices[ik, n]`` selects a G-vector for the n-th plane wave at k-point
    ``ik``; entries beyond ``npw[ik]`` are padding, flagged by ``mask``. Padding
    entries point at G = 0 rather than at an invalid index, so a gather is always
    in bounds and only the mask decides what counts.
    """

    indices: jnp.ndarray  # (nk, npwx) int32, into GVectors.miller
    mask: jnp.ndarray  # (nk, npwx) bool
    npw: tuple[int, ...] = eqx.field(static=True)
    ecutwfc: float = eqx.field(static=True)

    @property
    def nk(self) -> int:
        return self.indices.shape[0]

    @property
    def npwx(self) -> int:
        return self.indices.shape[1]

    def miller(self, gvectors: GVectors) -> jnp.ndarray:
        """(nk, npwx, 3) Miller indices, padded entries repeating G = 0."""
        return gvectors.miller[self.indices]

    def kinetic(self, gvectors: GVectors, kpoints: KPoints, cell: Cell) -> jnp.ndarray:
        """``|k+G|^2`` in Ry for every (k, plane wave), zero on padding.

        This is the kinetic energy operator in the plane-wave basis: diagonal,
        and the cheapest part of ``H|psi>``.
        """
        return _kinetic(gvectors.cartesian(cell), kpoints.cartesian(cell),
                        self.indices, self.mask)

    def fft_index(self, gvectors: GVectors) -> jnp.ndarray:
        """(nk, npwx) flat FFT-box index for each retained plane wave."""
        return gvectors.fft_index[self.indices]


# Setup arrays are built once, but each one built from a handful of eager
# operations costs a separate XLA compilation -- which on a small cell is far
# more than the arithmetic. One jitted function is one compilation.
@jax.jit
def _kinetic(gcart, kcart, indices, mask):
    kinetic = jnp.sum((kcart[:, None, :] + gcart[indices]) ** 2, axis=-1)
    return jnp.where(mask, kinetic, 0.0)


def build_plane_wave_basis(
    gvectors: GVectors, kpoints: KPoints, cell: Cell, ecutwfc: float
) -> PlaneWaveBasis:
    """Select and pad the plane waves for every k-point.

    Args:
        gvectors: the dense G-vector set to select from.
        kpoints: the k-points, in units of ``2*pi/alat``.
        cell: the unit cell.
        ecutwfc: wavefunction cutoff in Ry.
    """
    gcutw = gcut_from_ecut(ecutwfc, cell.alat)
    g = np.asarray(gvectors.reduced(cell))  # (ngm, 3) in 2*pi/alat
    k = np.asarray(kpoints.coords)  # (nk, 3) in 2*pi/alat

    selected = []
    for ik in range(len(k)):
        kg2 = np.sum((k[ik] + g) ** 2, axis=1)
        kg2 = np.where(kg2 <= _EPS, 0.0, kg2)
        (indices,) = np.nonzero(kg2 <= gcutw)
        # Order by |k+G|^2, as gk_sort does; ties broken by G index so the
        # result does not depend on the sorting algorithm.
        selected.append(indices[np.lexsort((indices, np.round(kg2[indices], 12)))])

    npw = tuple(len(s) for s in selected)
    if min(npw) == 0:
        raise ValueError("a k-point retained no plane waves; ecutwfc is far too small")
    npwx = max(npw)

    indices = np.zeros((len(selected), npwx), dtype=np.int32)
    mask = np.zeros((len(selected), npwx), dtype=bool)
    for ik, chosen in enumerate(selected):
        indices[ik, : len(chosen)] = chosen
        mask[ik, : len(chosen)] = True

    return PlaneWaveBasis(
        indices=jnp.asarray(indices),
        mask=jnp.asarray(mask),
        npw=npw,
        ecutwfc=float(ecutwfc),
    )
