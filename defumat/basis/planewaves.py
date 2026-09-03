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

from defumat.basis.fftgrid import gcut_from_ecut
from defumat.basis.gvectors import GVectors, _half_sphere
from defumat.system.cell import Cell
from defumat.system.kpoints import KPoints

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
    #: Whether only one plane wave of each ``(G, -G)`` pair is stored. The
    #: **dense** G-vector set is whole either way -- see
    #: :func:`~defumat.basis.builder.build_basis` for why the trick is confined
    #: to the wavefunctions.
    gamma_only: bool = eqx.field(static=True, default=False)

    @property
    def nk(self) -> int:
        return self.indices.shape[0]

    @property
    def npwx(self) -> int:
        return self.indices.shape[1]

    def miller(self, gvectors: GVectors) -> jnp.ndarray:
        """(nk, npwx, 3) Miller indices, padded entries repeating G = 0."""
        return gvectors.miller[self.indices]

    def kinetic(
        self,
        gvectors: GVectors,
        kpoints: KPoints,
        cell: Cell,
        kcart: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """``|k+G|^2`` in Ry for every (k, plane wave), zero on padding.

        This is the kinetic energy operator in the plane-wave basis: diagonal,
        and the cheapest part of ``H|psi>``.

        ``kcart`` overrides the k-points' own cartesian coordinates (1/bohr)
        while keeping *this* basis -- the sphere, its padding and its mask. The
        selection of plane waves is a host-side decision and cannot be traced;
        the arithmetic on top of it can, and separating the two is what lets
        ``|k+G|^2`` be differentiated with respect to a k-point. A spin
        spiral's ``dE/dq`` is the caller (:mod:`defumat.forces.spiral`).
        """
        if kcart is None:
            kcart = kpoints.cartesian(cell)
        return _kinetic(gvectors.cartesian(cell), kcart, self.indices, self.mask)

    def kplusg(
        self,
        gvectors: GVectors,
        kpoints: KPoints,
        cell: Cell,
        kcart: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """``k + G`` in 1/bohr for every (k, plane wave), zero on padding.

        ``(nk, npwx, 3)``. The vector whose square :meth:`kinetic` returns, kept
        separately because a *gradient* of a state needs the vector and not its
        modulus: ``grad psi = sum_G i(k+G) c_G e^{i(k+G)r}``, which is how the
        kinetic energy density is built
        (:func:`defumat.scf.density.kinetic_energy_density`) and how
        ``sum_band.f90`` builds it (``kplusgi``, three times per band).

        Zeroed on padding for the same reason ``kinetic`` is: a padded entry
        points at ``G = 0`` and would otherwise contribute ``k`` itself to every
        band's gradient.
        """
        if kcart is None:
            kcart = kpoints.cartesian(cell)
        return _kplusg(gvectors.cartesian(cell), kcart, self.indices, self.mask)

    def fft_index(self, gvectors: GVectors) -> jnp.ndarray:
        """(nk, npwx) flat FFT-box index for each retained plane wave."""
        return gvectors.fft_index[self.indices]

    def fft_index_minus(self, gvectors: GVectors) -> jnp.ndarray:
        """``(nk, npwx)`` flat index of ``-(k+G)`` -- QE's ``nlm``.

        For ``gamma_only`` only, where ``k = 0`` and the sphere is a half.
        """
        return gvectors.fft_index_minus[self.indices]


# Setup arrays are built once, but each one built from a handful of eager
# operations costs a separate XLA compilation -- which on a small cell is far
# more than the arithmetic. One jitted function is one compilation.
@jax.jit
def _kinetic(gcart, kcart, indices, mask):
    kinetic = jnp.sum((kcart[:, None, :] + gcart[indices]) ** 2, axis=-1)
    return jnp.where(mask, kinetic, 0.0)


@jax.jit
def _kplusg(gcart, kcart, indices, mask):
    vectors = kcart[:, None, :] + gcart[indices]
    return jnp.where(mask[..., None], vectors, 0.0)


def build_plane_wave_basis(
    gvectors: GVectors, kpoints: KPoints, cell: Cell, ecutwfc: float,
    gamma_only: bool = False,
) -> PlaneWaveBasis:
    """Select and pad the plane waves for every k-point.

    Args:
        gvectors: the dense G-vector set to select from.
        kpoints: the k-points, in units of ``2*pi/alat``.
        cell: the unit cell.
        ecutwfc: wavefunction cutoff in Ry.
        gamma_only: keep one plane wave of each ``(G, -G)`` pair. At ``k = 0``
            the state can be chosen real, so the other half is
            ``c(-G) = conj(c(G))`` and storing it is storing the same numbers
            twice. Halves ``npwx`` and with it every array a band lives in.
    """
    gcutw = gcut_from_ecut(ecutwfc, cell.alat)
    g = np.asarray(gvectors.reduced(cell))  # (ngm, 3) in 2*pi/alat
    k = np.asarray(kpoints.coords)  # (nk, 3) in 2*pi/alat

    selected = []
    for ik in range(len(k)):
        kg2 = np.sum((k[ik] + g) ** 2, axis=1)
        kg2 = np.where(kg2 <= _EPS, 0.0, kg2)
        keep = kg2 <= gcutw
        if gamma_only:
            keep = keep & _half_sphere(np.asarray(gvectors.miller))
        (indices,) = np.nonzero(keep)
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
        gamma_only=gamma_only,
        npw=npw,
        ecutwfc=float(ecutwfc),
    )
