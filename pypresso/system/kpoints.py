"""k-point sets: Monkhorst-Pack grids, explicit lists, and band paths.

Coordinates follow QE's convention: cartesian, in units of ``2*pi/alat``. That
is what pw.x prints, so a reference comparison needs no conversion, and it is
what the Hamiltonian consumes once multiplied by ``tpiba``.

Weights follow QE too: normalised to sum to 1 and then multiplied by 2 for a
spin-degenerate calculation (``degspin`` in ``PW/src/setup.f90``), which is why
an unpolarised run prints weights summing to 2.

No symmetry reduction happens here. Grids are generated complete; reducing them
to the irreducible wedge is the symmetry phase's job, and until then a run is
compared against QE with ``nosym=.true., noinv=.true.``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from pypresso.config import DEFAULT_PRECISION, Precision
from pypresso.system.cell import Cell

__all__ = ["KPoints", "monkhorst_pack", "expand_band_path"]

#: Spin degeneracy factor applied to weights for an unpolarised calculation.
DEGSPIN = 2.0


def monkhorst_pack(
    grid: tuple[int, int, int], shift: tuple[int, int, int] = (0, 0, 0)
) -> tuple[np.ndarray, np.ndarray]:
    """The complete (unreduced) Monkhorst-Pack grid in crystal coordinates.

    Transcribed from ``PW/src/kpoint_grid.f90``, including its point ordering
    (last index fastest) and the ``x - nint(x)`` fold into the first Brillouin
    zone. Ordering matters because QE's reduced list keeps the *first* member of
    each symmetry-equivalent set, so a later symmetry phase can only reproduce
    the same representatives if it starts from the same sequence.

    Args:
        grid: ``(nk1, nk2, nk3)``, the grid dimensions.
        shift: ``(k1, k2, k3)``, each 0 or 1; 1 offsets that axis by half a step.

    Returns:
        ``(points, weights)`` with points ``(nk, 3)`` in crystal coordinates and
        equal weights summing to 1.
    """
    nk1, nk2, nk3 = (int(n) for n in grid)
    if min(nk1, nk2, nk3) < 1:
        raise ValueError(f"k-point grid must be positive, got {grid}")
    k1, k2, k3 = (int(s) for s in shift)
    if not all(s in (0, 1) for s in (k1, k2, k3)):
        raise ValueError(f"k-point shift components must be 0 or 1, got {shift}")

    i = np.arange(nk1)[:, None, None]
    j = np.arange(nk2)[None, :, None]
    k = np.arange(nk3)[None, None, :]

    xkg = np.empty((nk1, nk2, nk3, 3))
    xkg[..., 0] = i / nk1 + k1 / 2 / nk1
    xkg[..., 1] = j / nk2 + k2 / 2 / nk2
    xkg[..., 2] = k / nk3 + k3 / 2 / nk3

    points = xkg.reshape(-1, 3)  # C order == QE's (k fastest, then j, then i)
    points = points - _fortran_nint(points)  # back into the first BZ, as QE does

    nk = points.shape[0]
    return points, np.full(nk, 1.0 / nk)


def expand_band_path(
    vertices: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Expand a ``*_b`` band path into individual k-points.

    Transcribed from ``Modules/generate_k_along_lines.f90``: ``counts[i]`` points
    are placed on the segment from vertex ``i`` to vertex ``i+1``, the endpoint
    included and the start point counted once at the previous segment. A count of
    zero inserts a discontinuity -- the next vertex is added without advancing
    the path length, which is how disconnected paths are written.

    The final count is ignored, so the total is ``1 + sum(counts[:-1])``.

    Returns:
        ``(points, path_length)``: the k-points, and the cumulative distance
        along the path, which is the natural x-axis of a band structure plot.
    """
    vertices = np.asarray(vertices, dtype=float)
    counts = np.asarray(counts)
    if vertices.shape[0] != counts.shape[0]:
        raise ValueError("a band path needs one count per vertex")
    if np.any(counts < 0):
        raise ValueError("band path counts must be non-negative")

    points = [vertices[0]]
    lengths = [0.0]
    for i in range(1, len(vertices)):
        n = int(counts[i - 1])
        if n > 0:
            delta = 1.0 / n
            for step in range(1, n + 1):
                point = vertices[i - 1] + delta * step * (vertices[i] - vertices[i - 1])
                lengths.append(lengths[-1] + float(np.linalg.norm(point - points[-1])))
                points.append(point)
        else:  # discontinuity: jump to the next vertex without adding length
            points.append(vertices[i])
            lengths.append(lengths[-1])
    return np.array(points), np.array(lengths)


class KPoints(eqx.Module):
    """A set of k-points with integration weights.

    ``coords`` is cartesian in units of ``2*pi/alat``, matching QE's ``xk``.
    ``path_length`` is set only for band paths, where it is the plotting abscissa.
    """

    coords: jnp.ndarray  # (nk, 3), units 2*pi/alat
    weights: jnp.ndarray  # (nk,)
    path_length: jnp.ndarray | None = None
    gamma_only: bool = eqx.field(static=True, default=False)
    grid: tuple[int, int, int] | None = eqx.field(static=True, default=None)
    shift: tuple[int, int, int] | None = eqx.field(static=True, default=None)
    precision: Precision = eqx.field(static=True, default=DEFAULT_PRECISION)

    @property
    def nk(self) -> int:
        return self.coords.shape[0]

    def cartesian(self, cell: Cell) -> jnp.ndarray:
        """k-points in 1/bohr, which is what ``|k+G|`` needs."""
        return self.coords * cell.tpiba

    def crystal(self, cell: Cell) -> jnp.ndarray:
        return cell.k_to_crystal(self.coords)

    @classmethod
    def from_crystal(
        cls,
        points,
        weights,
        cell: Cell,
        precision: Precision = DEFAULT_PRECISION,
        **kwargs,
    ) -> "KPoints":
        """Build from crystal coordinates, converting to QE's cartesian units."""
        coords = np.asarray(cell.k_to_cartesian(np.asarray(points, dtype=float)))
        return cls(
            coords=precision.as_real(coords),
            weights=precision.as_real(_normalise(weights)),
            precision=precision,
            **kwargs,
        )

    @classmethod
    def from_cartesian(
        cls, points, weights, precision: Precision = DEFAULT_PRECISION, **kwargs
    ) -> "KPoints":
        """Build from cartesian coordinates already in units of ``2*pi/alat``."""
        return cls(
            coords=precision.as_real(np.asarray(points, dtype=float)),
            weights=precision.as_real(_normalise(weights)),
            precision=precision,
            **kwargs,
        )

    @classmethod
    def gamma(cls, precision: Precision = DEFAULT_PRECISION) -> "KPoints":
        """The single point k=0, flagged so the real-wavefunction trick applies."""
        return cls(
            coords=precision.as_real(np.zeros((1, 3))),
            weights=precision.as_real(np.array([DEGSPIN])),
            gamma_only=True,
            precision=precision,
        )

    @classmethod
    def automatic(
        cls,
        grid: tuple[int, int, int],
        shift: tuple[int, int, int],
        cell: Cell,
        precision: Precision = DEFAULT_PRECISION,
    ) -> "KPoints":
        """A complete Monkhorst-Pack grid (no symmetry reduction)."""
        points, weights = monkhorst_pack(grid, shift)
        return cls.from_crystal(
            points,
            weights,
            cell,
            precision=precision,
            grid=tuple(int(n) for n in grid),
            shift=tuple(int(s) for s in shift),
        )

    @classmethod
    def band_path(
        cls,
        vertices,
        counts,
        cell: Cell,
        crystal: bool,
        precision: Precision = DEFAULT_PRECISION,
    ) -> "KPoints":
        """A ``tpiba_b`` / ``crystal_b`` path.

        Weights are uniform: a band path is not a Brillouin-zone integration, and
        QE likewise prints uniform weights for these runs.
        """
        points, lengths = expand_band_path(np.asarray(vertices, dtype=float), counts)
        if crystal:
            points = np.asarray(cell.k_to_cartesian(points))
            lengths = np.asarray(
                np.concatenate(
                    [[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
                )
            )
        weights = np.full(len(points), 1.0)
        return cls(
            coords=precision.as_real(points),
            weights=precision.as_real(_normalise(weights)),
            path_length=precision.as_real(lengths),
            precision=precision,
        )


def _fortran_nint(x: np.ndarray) -> np.ndarray:
    """Fortran ``NINT``: round half *away from zero*.

    NumPy's ``rint`` rounds half to even, so ``rint(0.5) == 0`` where Fortran
    gives 1. Exact halves are not a corner case here -- an even grid without a
    shift puts k-points at exactly 0.5 -- and the difference decides whether a
    point is folded to -0.5 or left at +0.5. The two are equivalent by a
    reciprocal lattice vector, but only one of them matches QE's printed list and
    its subsequent G-vector bookkeeping.
    """
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _normalise(weights) -> np.ndarray:
    """Normalise to sum 1, then apply the spin degeneracy factor, as QE does."""
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0.0:
        raise ValueError("k-point weights must sum to a positive number")
    return weights / total * DEGSPIN
