"""The G-vector set: reciprocal lattice points inside the density cutoff sphere.

What is stored is the **Miller indices** -- integers, fixed for a run -- and not
cartesian components. Cartesian G is then a function of the cell, computed on
demand, so a strain derivative flows through it (rule D2/D3). Storing cartesian
components would silently freeze the cell and make stress-by-differentiation
impossible.

Generation follows ``Modules/recvec_subs.f90`` (``ggen``): enumerate Miller
indices over the FFT box, keep those with ``|G|^2 <= gcut``, sort by ``|G|^2``.
The ordering QE produces comes from ``hpsort_eps`` and is *not* reproduced
exactly -- nothing physical depends on the order within a shell of equal
``|G|^2``, and only the *set* has to match. What is reproduced is that G = 0 is
first, which many formulas rely on.
"""

from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from defumat.basis.fftgrid import fft_grid_dimensions, gcut_from_ecut
from defumat.system.cell import Cell

__all__ = ["GVectors", "generate_gvectors", "modulus"]


class GVectors(eqx.Module):
    """Reciprocal lattice vectors inside a cutoff sphere, sorted by ``|G|^2``.

    ``miller[n]`` are the integer coefficients of G_n on the reciprocal basis;
    ``miller[0]`` is always ``(0, 0, 0)``.
    """

    miller: jnp.ndarray  # (ngm, 3), integers
    grid: tuple[int, int, int] = eqx.field(static=True)
    ecut: float = eqx.field(static=True)
    gamma_only: bool = eqx.field(static=True, default=False)

    @property
    def ngm(self) -> int:
        return self.miller.shape[0]

    def reduced(self, cell: Cell) -> jnp.ndarray:
        """G in units of ``2*pi/alat`` -- QE's ``g`` array."""
        return _transform(self.miller, cell.bg_2pi_alat)

    def cartesian(self, cell: Cell) -> jnp.ndarray:
        """G in 1/bohr, which is what ``|k+G|^2`` in Ry needs."""
        return _transform(self.miller, cell.bg)

    def g2(self, cell: Cell) -> jnp.ndarray:
        """``|G|^2`` in units of ``(2*pi/alat)^2`` -- QE's ``gg`` array."""
        return jnp.sum(self.reduced(cell) ** 2, axis=1)

    def kinetic(self, cell: Cell) -> jnp.ndarray:
        """``|G|^2`` in Ry (``hbar^2/2m = 1``), i.e. in 1/bohr^2."""
        return jnp.sum(self.cartesian(cell) ** 2, axis=1)

    @property
    def fft_index(self) -> jnp.ndarray:
        """Flat index of each G in the FFT box, for scatter/gather.

        Negative Miller indices wrap to the top of each axis, which is the
        standard FFT frequency layout and matches QE's ``nl`` map.
        """
        return _fft_index(self.miller, self.grid)

    @property
    def fft_index_minus(self) -> jnp.ndarray:
        """Flat index of **-G** in the FFT box -- QE's ``nlm``.

        Only meaningful for a :attr:`gamma_only` set, where the stored half
        sphere carries one G of each ``(G, -G)`` pair and the other half is
        recovered from ``c(-G) = conj(c(G))`` -- a field with real values in
        real space, which for ``gamma_only`` every field is.

        ``fft_index_minus[0] == fft_index[0]``, both being ``G = 0``, and every
        caller has to skip that entry rather than add it twice. It is entry 0
        because :func:`generate_gvectors` sorts ``G = 0`` first and asserts it.
        """
        return _fft_index(-self.miller, self.grid)

    def shell_boundaries(self, cell: Cell, tolerance: float = 1e-8) -> np.ndarray:
        """Indices where a new ``|G|^2`` shell starts. Useful for radial tables."""
        g2 = np.asarray(self.g2(cell))
        return np.flatnonzero(np.diff(g2) > tolerance) + 1


@jax.jit
def _transform(miller, matrix):
    return miller.astype(matrix.dtype) @ matrix


#: Below this ``|G|^2`` a G-vector counts as the origin. QE's own ``eps8``.
_TINY = 1.0e-8


@jax.jit
def modulus(vectors):
    """``|v|`` for a stack of cartesian vectors, differentiable at the origin.

    The mask goes on ``|v|^2`` **before** the square root, not on the result
    after it. ``sqrt`` has an infinite derivative at zero, so a guard placed
    afterwards leaves the value right and the gradient ``0 * inf = NaN`` -- and
    ``G = 0`` is the first entry of every G-vector set, so the whole stress
    tensor comes back NaN with every value on the way to it correct. It is P15's
    Ewald trap in a different module, and it only appears once the cell is what
    is being differentiated: with respect to the atomic positions ``|G|`` is a
    constant and no derivative ever reaches this.
    """
    norm2 = jnp.sum(vectors**2, axis=-1)
    return jnp.where(norm2 > _TINY, jnp.sqrt(jnp.where(norm2 > _TINY, norm2, 1.0)), 0.0)


@partial(jax.jit, static_argnames=("grid",))
def _fft_index(miller, grid):
    n1, n2, n3 = grid
    i = jnp.mod(miller[:, 0], n1)
    j = jnp.mod(miller[:, 1], n2)
    k = jnp.mod(miller[:, 2], n3)
    return (i * n2 + j) * n3 + k


def generate_gvectors(
    cell: Cell,
    ecut: float,
    grid: tuple[int, int, int] | None = None,
    gamma_only: bool = False,
    fft_factors: tuple[int, int, int] = (1, 1, 1),
) -> GVectors:
    """All G with ``|G|^2 <= ecut`` (Ry), sorted by magnitude.

    Args:
        cell: the unit cell.
        ecut: cutoff in Ry -- ``ecutrho`` for the dense grid, ``4*ecutwfc`` for
            the smooth one.
        grid: FFT dimensions to enumerate within. Computed from the cutoff when
            omitted, which is what a caller normally wants; passing it lets the
            smooth grid be generated inside the dense box.
        fft_factors: divisibility constraint on the FFT dimensions, from the
            crystal's fractional translations. It changes the box, never the
            G-vector set: a larger box only reaches vectors that the cutoff
            rejects anyway.
        gamma_only: keep only half the sphere. At k = 0 a real wavefunction
            satisfies ``c(-G) = conj(c(G))``, so storing one G of each pair
            halves both memory and work. QE does this whenever the k-point set
            is Gamma alone, and reports the halved ``ngm``, so reproducing the
            count means reproducing the selection.

    The enumeration range is QE's: Miller indices in ``[-(n-1)//2, (n-1)//2]``
    for each axis of the FFT grid, which is exactly the set of frequencies the
    box can represent without aliasing.
    """
    at_alat = np.asarray(cell.at_alat)
    bg = np.asarray(cell.bg_2pi_alat)
    gcut = gcut_from_ecut(ecut, cell.alat)

    if grid is None:
        grid = fft_grid_dimensions(at_alat, bg, gcut, fft_factors)

    ranges = [np.arange(-((n - 1) // 2), (n - 1) // 2 + 1) for n in grid]
    i, j, k = np.meshgrid(*ranges, indexing="ij")
    miller = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1)

    g2 = np.sum((miller @ bg) ** 2, axis=1)
    inside = g2 <= gcut  # ggen keeps the boundary; grid_set excludes it
    if gamma_only:
        inside &= _half_sphere(miller)
    miller, g2 = miller[inside], g2[inside]

    # Sort by |G|^2, breaking ties by Miller index so the result is reproducible
    # across machines and runs. QE's hpsort_eps orders ties differently; nothing
    # physical depends on the order inside a shell.
    order = np.lexsort((miller[:, 2], miller[:, 1], miller[:, 0], np.round(g2, 12)))
    miller = miller[order]

    if not np.all(miller[0] == 0):
        raise AssertionError("G = 0 must sort first")

    return GVectors(
        miller=jnp.asarray(miller, dtype=jnp.int32),
        grid=tuple(int(n) for n in grid),
        ecut=float(ecut),
        gamma_only=gamma_only,
    )


def _half_sphere(miller: np.ndarray) -> np.ndarray:
    """One G from each (G, -G) pair, choosing QE's representative.

    ``ggen`` walks ``i`` from 0, then ``j`` from 0 only on the ``i = 0`` plane,
    then ``k`` from 0 only on the ``i = j = 0`` line -- that is, the closed
    half-space ``x > 0``, plus the half-plane ``x = 0, y > 0``, plus the
    half-line ``x = y = 0, z >= 0``. G = 0 belongs to the last of these and is
    kept exactly once, which is why the count is ``(ngm_full + 1) / 2``.
    """
    i, j, k = miller[:, 0], miller[:, 1], miller[:, 2]
    return (i > 0) | ((i == 0) & (j > 0)) | ((i == 0) & (j == 0) & (k >= 0))
