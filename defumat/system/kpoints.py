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

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from defumat.config import DEFAULT_PRECISION, Precision
from defumat.system.cell import Cell

#: Tolerance for deciding that a rotated k-point lands on the grid (QE's ``eps``).
_GRID_EPS = 1.0e-5

__all__ = ["KPoints", "monkhorst_pack", "irreducible_wedge", "grid_equivalence",
           "expand_to_subgroup", "expand_band_path", "for_spin"]

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


def irreducible_wedge(
    grid: tuple[int, int, int],
    shift: tuple[int, int, int],
    rotations: np.ndarray,
    time_reversal: bool = True,
    t_rev: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The symmetry-reduced Monkhorst-Pack grid: QE's ``kpoint_grid.f90``.

    Two k-points related by a symmetry of the crystal give the same eigenvalues
    and contribute the same thing to the density, so only one of each orbit has
    to be diagonalised -- the other members are recovered by symmetrising the
    density afterwards, which this code already does. For silicon on an 8x8x8
    grid that is 60 k-points instead of 512.

    The algorithm is QE's, and so is the choice of representative: walk the grid
    in order, and whenever a point has not already been marked equivalent to an
    earlier one, keep it and mark everything its orbit reaches. Keeping the
    *first* member of each orbit is what makes the reduced list agree with QE's
    point for point rather than merely orbit for orbit.

    Args:
        grid: ``(nk1, nk2, nk3)``.
        shift: ``(k1, k2, k3)``, each 0 or 1.
        rotations: ``(nsym, 3, 3)`` integer rotations in crystal axes -- the
            crystal's symmetries, not the lattice point group, so that a
            structure with fewer symmetries than its lattice is not over-reduced.
        time_reversal: whether ``-k`` is equivalent to ``k``. ``setup.f90`` sets
            it to ``.NOT. noinv .AND. .NOT. magnetic_sym``: a magnetic
            noncollinear run has no ``-k = k`` of its own, and what it has
            instead is the individual operations flagged in ``t_rev``.
        t_rev: ``(nsym,)`` of 0/1. Where it is 1 the operation is a symmetry
            only together with time reversal, so it sends ``k`` to ``-S k``
            rather than to ``S k`` (``kpoint_grid.f90``: ``IF (t_rev(ns) == 1)
            xkr = -xkr``).

    Returns:
        ``(points, weights)``, points in crystal coordinates folded into the
        first Brillouin zone and weights summing to 1.
    """
    nk1, nk2, nk3 = (int(n) for n in grid)
    k1, k2, k3 = (int(x) for x in shift)
    offsets = np.array([k1, k2, k3]) / 2.0
    counts = np.array([nk1, nk2, nk3])

    i, j, k = np.meshgrid(np.arange(nk1), np.arange(nk2), np.arange(nk3), indexing="ij")
    integers = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1)
    xkg = (integers + offsets) / counts  # QE's xkg, before folding

    equivalent = np.arange(len(xkg))
    multiplicity = np.zeros(len(xkg))
    reversed_ops = (np.zeros(len(rotations), dtype=bool) if t_rev is None
                    else np.asarray(t_rev, dtype=int) == 1)

    def grid_index(rotated):
        """Which grid point ``rotated`` is, or None if it is not on the grid."""
        scaled = rotated * counts - offsets
        nearest = np.rint(scaled)
        if np.any(np.abs(scaled - nearest) > _GRID_EPS):
            return None
        a, b, c = (int(x) % n for x, n in zip(nearest, counts))
        return (a * nk2 + b) * nk3 + c

    for n in range(len(xkg)):
        if equivalent[n] != n:
            continue
        multiplicity[n] = 1.0
        for s_index, rotation in enumerate(rotations):
            rotated = xkg[n] @ rotation.T
            if reversed_ops[s_index]:
                rotated = -rotated
            rotated = rotated - np.rint(rotated)
            images = [rotated, -rotated] if time_reversal else [rotated]
            for image in images:
                other = grid_index(image)
                if other is None or other <= n:
                    continue
                if equivalent[other] == other:
                    equivalent[other] = n
                    multiplicity[n] += 1.0

    keep = equivalent == np.arange(len(xkg))
    points = xkg[keep]
    points = points - _fortran_nint(points)  # into the first Brillouin zone
    weights = multiplicity[keep]
    return points, weights / weights.sum()


def grid_equivalence(
    grid: tuple[int, int, int],
    shift: tuple[int, int, int],
    rotations: np.ndarray,
    time_reversal: bool = True,
    t_rev: np.ndarray | None = None,
) -> np.ndarray:
    """Which irreducible point each point of the *complete* grid reduces to.

    :func:`irreducible_wedge` throws this map away -- it only needs the
    representatives and their multiplicities -- but the tetrahedron method needs
    it: the tetrahedra are built on the full ``nk1*nk2*nk3`` grid, where a
    microcell has eight well-defined corners, and every corner is then looked up
    in the reduced list. That is exactly what ``tetra_init``'s ``equiv`` array in
    ``PW/src/tetra.f90`` is, and QE builds it the same way for the same reason.

    QE recomputes the map by rotating every irreducible point by every symmetry
    and matching it against the grid. Here the orbit walk of
    :func:`irreducible_wedge` is repeated instead, which reaches the same answer
    by construction rather than by a second search -- the price is that the loop
    is written twice. It is duplicated rather than factored out because
    ``irreducible_wedge`` is the function every existing k-point comparison
    against QE goes through, and a shared helper would put those comparisons at
    the mercy of an edit made for the tetrahedra.

    Returns:
        ``equiv`` of length ``nk1*nk2*nk3``: for each point of the complete grid,
        its index **into the reduced list** ``irreducible_wedge`` returns, in the
        same grid ordering ``monkhorst_pack`` uses (last index fastest).
    """
    nk1, nk2, nk3 = (int(n) for n in grid)
    k1, k2, k3 = (int(x) for x in shift)
    offsets = np.array([k1, k2, k3]) / 2.0
    counts = np.array([nk1, nk2, nk3])

    i, j, k = np.meshgrid(np.arange(nk1), np.arange(nk2), np.arange(nk3), indexing="ij")
    integers = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1)
    xkg = (integers + offsets) / counts

    equivalent = np.arange(len(xkg))
    reversed_ops = (np.zeros(len(rotations), dtype=bool) if t_rev is None
                    else np.asarray(t_rev, dtype=int) == 1)

    def grid_index(rotated):
        scaled = rotated * counts - offsets
        nearest = np.rint(scaled)
        if np.any(np.abs(scaled - nearest) > _GRID_EPS):
            return None
        a, b, c = (int(x) % n for x, n in zip(nearest, counts))
        return (a * nk2 + b) * nk3 + c

    for n in range(len(xkg)):
        if equivalent[n] != n:
            continue
        for s_index, rotation in enumerate(rotations):
            rotated = xkg[n] @ rotation.T
            if reversed_ops[s_index]:
                rotated = -rotated
            rotated = rotated - np.rint(rotated)
            images = [rotated, -rotated] if time_reversal else [rotated]
            for image in images:
                other = grid_index(image)
                if other is None or other <= n:
                    continue
                if equivalent[other] == other:
                    equivalent[other] = n

    # Representatives are kept in grid order, so an irreducible point's position
    # in the reduced list is how many representatives precede it.
    keep = equivalent == np.arange(len(xkg))
    position = np.cumsum(keep) - 1
    return position[equivalent]


def expand_to_subgroup(
    points: np.ndarray,
    weights: np.ndarray,
    lattice_rotations: np.ndarray,
    rotations: np.ndarray,
    time_reversal: bool = True,
    t_rev: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand an explicit k-list from the lattice's wedge to the crystal's.

    ``irreducible_BZ`` in ``PW/src/irrek.f90``, and it runs for **every** SCF
    run, an explicit ``K_POINTS`` card included -- QE's comment says why: "input
    k-points are assumed to be given in the IBZ of the Bravais lattice, with the
    full point symmetry of the lattice; if some symmetries of the lattice are
    missing in the crystal, irreducible_BZ computes the missing k-points".

    It does nothing at all when the crystal has the lattice's whole point group,
    which is why it went unnoticed until a magnetic run: pointing a moment along
    ``x`` in a cubic crystal is exactly the case where the crystal's group is a
    proper subgroup, and QE's own `pw_noncolin` benchmark then runs on 22
    k-points where the input lists 11.

    The construction here is the coset arithmetic of ``irrek`` done the direct
    way: take each input point's orbit under the *lattice* group, split it into
    orbits under the crystal's, and give each of those a share of the input
    weight in proportion to its size. Same points and same weights, chosen
    representative aside.

    Args:
        points: ``(nk, 3)`` in crystal coordinates.
        weights: ``(nk,)``, any normalisation.
        lattice_rotations: the point group of the Bravais lattice.
        rotations: the point group of the crystal (magnetic, if it is one).
        t_rev: as in :func:`irreducible_wedge`.

    Returns:
        ``(points, weights)`` in crystal coordinates, weights normalised to 1.
    """
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    reversed_ops = (np.zeros(len(rotations), dtype=bool) if t_rev is None
                    else np.asarray(t_rev, dtype=int) == 1)

    def key(point):
        folded = np.round(point % 1.0, 6) % 1.0
        return tuple(np.round(folded, 6))

    def orbit(point, rots, reversal, flags=None):
        found = {}
        for index, rotation in enumerate(rots):
            image = point @ rotation.T
            if flags is not None and flags[index]:
                image = -image
            candidates = [image, -image] if reversal else [image]
            for candidate in candidates:
                found.setdefault(key(candidate), candidate - np.rint(candidate))
        return found

    out_points, out_weights = [], []
    for point, weight in zip(points, weights):
        full = orbit(point, lattice_rotations, time_reversal)
        # The input point is the representative of its own sub-orbit, so a list
        # that needs no expansion comes back unchanged -- which is what QE
        # prints, and what every explicit-list comparison against it expects.
        ordered = [(key(point), point)]
        ordered += [item for item in full.items() if item[0] != key(point)]
        assigned = set()
        for member_key, member in ordered:
            if member_key in assigned:
                continue
            sub = orbit(member, rotations, time_reversal, reversed_ops)
            # Only members of the lattice orbit count: a rounding difference
            # would otherwise leave a stray point with no weight of its own.
            sub_keys = {k for k in sub if k in full}
            assigned |= sub_keys
            out_points.append(member)
            out_weights.append(weight * len(sub_keys) / len(full))

    out_weights = np.array(out_weights)
    return np.array(out_points), out_weights / out_weights.sum()


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
    #: Whether :func:`for_spin` has already divided the weights by ``degspin``.
    #: The flag exists because that division is **not** idempotent and a k-set
    #: reaches a polarized run by more than one route -- built by
    #: :func:`~defumat.system.builder.build_system`, rebuilt by
    #: ``System._recelled_kpoints``, produced by
    #: :func:`~defumat.workflows.nscf.denser_grid`, or handed in whole to
    #: :meth:`defumat.calculator.Calculator.with_kpoints`. Without it the
    #: boundary that normalises has to know which of those it is looking at,
    #: and applying the factor twice is as silent as not applying it once: the
    #: Fermi level moves and the run integrates to the right electron count at
    #: the wrong energy.
    spin_normalized: bool = eqx.field(static=True, default=False)

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
        rotations: np.ndarray | None = None,
        time_reversal: bool = True,
        t_rev: np.ndarray | None = None,
    ) -> "KPoints":
        """A Monkhorst-Pack grid, reduced to the irreducible wedge if it can be.

        ``rotations`` are the crystal's symmetry operations in crystal axes.
        Without them the complete grid is returned, which is correct but costs a
        factor of up to the size of the point group in diagonalisations.
        ``time_reversal`` and ``t_rev`` are the magnetic case; see
        :func:`irreducible_wedge`.
        """
        if rotations is not None and len(rotations):
            points, weights = irreducible_wedge(
                grid, shift, rotations, time_reversal=time_reversal, t_rev=t_rev
            )
        else:
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
            # The path length has to be recomputed in cartesian space: distances
            # in crystal coordinates are not physical for a non-orthogonal cell.
            # Segments that expand_band_path held flat (a zero count, i.e. a
            # deliberate discontinuity) must stay flat, so the recomputation is
            # masked by where the crystal-space length actually advanced.
            moved = np.diff(lengths) > 0.0
            points = np.asarray(cell.k_to_cartesian(points))
            steps = np.linalg.norm(np.diff(points, axis=0), axis=1) * moved
            lengths = np.concatenate([[0.0], np.cumsum(steps)])
        weights = np.full(len(points), 1.0)
        return cls(
            coords=precision.as_real(points),
            weights=precision.as_real(_normalise(weights)),
            path_length=precision.as_real(lengths),
            precision=precision,
        )


def for_spin(kpoints: "KPoints", nspin: int) -> "KPoints":
    """The same k-set with the weight convention ``nspin`` asks for.

    ``setup.f90`` multiplies the weights by ``degspin`` **only** in its LDA
    branch: with two channels each k-point is diagonalised twice and the weights
    sum to one *per channel*, so the two together still account for two electrons
    per band. Every constructor here applies the factor of two unconditionally,
    so this undoes it for a polarized run.

    A **noncollinear** run (``nspin = 4``) is in the same branch of
    ``setup.f90`` and for the same reason read the other way round: there is one
    k-point list, but a band is a spinor and holds *one* electron rather than
    two, so the degeneracy factor does not belong in the weights either.

    It exists as a function rather than a line in ``build_system`` because a
    k-set can be built long after the system is -- a density of states runs on a
    denser grid than the SCF did (:func:`defumat.workflows.nscf.denser_grid`),
    and a grid built there with the unpolarized convention counts every electron
    twice. That mistake does not look like one: the Fermi level simply comes out
    somewhere else, and the density of states integrates to the right number of
    electrons at the wrong energy.

    **It is idempotent**, through :attr:`KPoints.spin_normalized`, and it has to
    be: the division reaches a k-set from several directions at once and each
    one of them is a boundary that cannot see where the set came from. Applying
    it twice is the same silent error as not applying it, in the other
    direction. The flag is set **only** when the division actually happens, so a
    set built for ``nspin = 1`` and later handed to a polarized run is still
    normalised then; and nothing else about which k-set gets the factor has
    changed -- ``setup.f90`` applies ``degspin`` to ``wk`` whatever the sampling
    is, a Gamma-only run included.
    """
    if int(nspin) not in (2, 4) or kpoints.spin_normalized:
        return kpoints
    return dataclasses.replace(
        kpoints,
        weights=kpoints.weights / DEGSPIN,
        spin_normalized=True,
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
