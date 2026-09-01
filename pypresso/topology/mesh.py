"""k-meshes for topological invariants, and the reciprocal-lattice bookkeeping.

Every quantity in this subpackage is built from overlaps between Bloch states at
*neighbouring* k-points, so what a mesh has to carry is not only where its
points are but **how a step off its edge comes back onto it**. That second part
is the whole content of this module and it is where the classic bug lives: on a
closed mesh the neighbour of the last point along a direction is the first point
*plus a reciprocal lattice vector* ``b``, and the periodic gauge

    u_{k+b}(G) = u_k(G + b)

means the overlap with it is taken against **shifted** plane-wave coefficients.
Forget the shift and the plaquette product stops being a closed loop; the Chern
number comes out a smooth non-integer function of the mesh and looks plausible.

So a mesh here is a grid of points in **crystal coordinates** together with the
integer vector that closes each direction. Crystal coordinates are the right
frame for exactly that reason: the closing vector is an integer triple in them,
whatever the lattice is.

Three shapes are needed and all three are the same object:

* a **plane mesh** closed in both directions -- what the Fukui-Hatsugai-Suzuki
  plaquette sum integrates over (:mod:`pypresso.topology.berry`);
* a **pumping mesh**, closed along the loop direction and open along the other,
  which is swept over half the Brillouin zone -- what a Wilson loop needs
  (:mod:`pypresso.topology.wilson`);
* a **string mesh**, open across the transverse plane and closed along one
  reciprocal lattice vector, which is what a Berry-phase polarization is
  averaged over (:mod:`pypresso.topology.polarization`);
* the **time-reversal-invariant momenta**, a bare list of points
  (:mod:`pypresso.topology.parity`).

Nothing here allocates anything wavefunction-shaped: a mesh is ``(n1, n2, 3)``
floats, kilobytes at any size that could be run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pypresso.config import DEFAULT_PRECISION, Precision
from pypresso.system.cell import Cell
from pypresso.system.kpoints import KPoints

__all__ = [
    "PlaneMesh",
    "plane_mesh",
    "pumping_mesh",
    "string_mesh",
    "trim_points",
    "PLANE_AXES",
]

#: Which crystal directions span the plane normal to each axis, cyclically.
#: Axis ``a`` (0-based) is spanned by ``PLANE_AXES[a]`` -- 0 -> (1, 2),
#: 1 -> (2, 0), 2 -> (0, 1). The cyclic order matters: it fixes the sign of the
#: curvature, and the three-dimensional Z2 indices are read plane by plane in
#: it (``elkpy``'s ``combine_3d_invariants`` maps ``axis 1 -> (loop, pump) =
#: (2, 3)`` in its 1-based numbering, which is this).
PLANE_AXES: tuple[tuple[int, int], ...] = ((1, 2), (2, 0), (0, 1))


@dataclass(frozen=True)
class PlaneMesh:
    """A grid of k-points in a plane, with the vectors that close it.

    ``points[i, j]`` is a k-point in **crystal** coordinates. Stepping past
    ``i = n1 - 1`` returns to ``i = 0`` displaced by ``span1``, and likewise for
    the second direction -- but only where ``closed`` says so. A pumping mesh is
    closed along its loop direction and open along the direction it is pumped
    over, because no overlap is ever taken along that one.
    """

    points: np.ndarray  # (n1, n2, 3), crystal coordinates
    span1: np.ndarray  # (3,) integer: the reciprocal lattice vector closing dir 1
    span2: np.ndarray  # (3,) integer
    closed: tuple[bool, bool] = (True, True)

    @property
    def shape(self) -> tuple[int, int]:
        return self.points.shape[0], self.points.shape[1]

    @property
    def nk(self) -> int:
        return int(self.points.shape[0] * self.points.shape[1])

    def flat(self) -> np.ndarray:
        """``(n1 * n2, 3)``: the points in C order, ``j`` fastest."""
        return self.points.reshape(-1, 3)

    def index(self, i, j) -> np.ndarray:
        """Flat index of point ``(i, j)``, vectorised over both arguments."""
        n2 = self.shape[1]
        return np.asarray(i) * n2 + np.asarray(j)

    def neighbour(self, i: int, j: int, direction: int):
        """``(flat index, integer shift)`` of the next point along ``direction``.

        The shift is the reciprocal lattice vector ``b`` for which the physical
        neighbour is ``points[flat] + b``; it is zero except on the edge of a
        closed direction. Returning it rather than folding it into the point is
        deliberate: the coordinates stay inside the mesh, so the states are the
        ones already computed, and the wrap survives as an explicit index shift
        of the plane-wave coefficients.
        """
        n1, n2 = self.shape
        if direction == 0:
            wrapped = (i + 1) % n1
            shift = self.span1 if i + 1 == n1 else np.zeros(3, dtype=int)
            if i + 1 == n1 and not self.closed[0]:
                raise IndexError("direction 0 of this mesh is open; there is no wrap")
            return int(self.index(wrapped, j)), np.asarray(shift, dtype=int)
        wrapped = (j + 1) % n2
        shift = self.span2 if j + 1 == n2 else np.zeros(3, dtype=int)
        if j + 1 == n2 and not self.closed[1]:
            raise IndexError("direction 1 of this mesh is open; there is no wrap")
        return int(self.index(i, wrapped)), np.asarray(shift, dtype=int)

    def kpoints(self, cell: Cell, precision: Precision = DEFAULT_PRECISION) -> KPoints:
        """The mesh as a :class:`~pypresso.system.kpoints.KPoints` to diagonalise.

        The weights are uniform and meaningless -- nothing here integrates a
        density -- but a ``KPoints`` must have some, and equal ones are the only
        honest choice.
        """
        flat = self.flat()
        return KPoints.from_crystal(
            flat, np.full(len(flat), 1.0 / len(flat)), cell, precision=precision
        )


def plane_mesh(
    shape: tuple[int, int],
    axis: int = 2,
    offset: float = 0.0,
    origin: tuple[float, float, float] | None = None,
) -> PlaneMesh:
    """A closed uniform mesh over the plane normal to ``axis``.

    ``axis`` names the crystal reciprocal direction held fixed at ``offset``;
    the plane is spanned by the other two in the cyclic order of
    :data:`PLANE_AXES`. Neither endpoint is repeated -- the mesh has exactly
    ``n1 * n2`` points and the wrap closes it, which is what makes the plaquette
    sum a sum over the whole zone with no double counting.
    """
    d1, d2 = PLANE_AXES[axis % 3]
    n1, n2 = int(shape[0]), int(shape[1])
    if n1 < 1 or n2 < 1:
        raise ValueError(f"a plane mesh needs at least one point per direction, got {shape}")

    base = np.zeros(3)
    if origin is not None:
        base = np.asarray(origin, dtype=float)
    base = base.copy()
    base[axis % 3] = float(offset)

    points = np.zeros((n1, n2, 3))
    points[..., :] = base
    points[..., d1] += (np.arange(n1) / n1)[:, None]
    points[..., d2] += (np.arange(n2) / n2)[None, :]

    span1 = np.zeros(3, dtype=int)
    span1[d1] = 1
    span2 = np.zeros(3, dtype=int)
    span2[d2] = 1
    return PlaneMesh(points=points, span1=span1, span2=span2, closed=(True, True))


def pumping_mesh(
    nloop: int,
    npump: int,
    axis: int = 2,
    offset: float = 0.0,
) -> PlaneMesh:
    """The half-zone mesh a Z2 Wilson loop is computed on.

    The **loop** direction spans the whole zone with ``nloop`` points and is
    closed; the **pump** direction takes ``npump`` values covering *half* the
    zone, ``t_m = m / (2 (npump - 1))`` for ``m = 0 .. npump - 1``, so that it
    starts at ``t = 0`` and ends exactly at ``t = 1/2``. Both are
    time-reversal-invariant planes, which is what makes the Wannier centres at
    the two ends come back to themselves as a set and the crossing count a Z2
    quantity at all.

    Ending exactly on ``1/2`` is not a detail. It is why the pump grid is
    described as *half* of a ``2 (npump - 1)`` grid rather than as
    ``linspace(0, 0.5, npump)``: the two are the same numbers, and saying it the
    first way makes it obvious that the last row is the TRI plane and not a
    point near it.

    ``axis`` is the direction held fixed at ``offset``, as in :func:`plane_mesh`;
    the loop direction is the first of :data:`PLANE_AXES` for that axis and the
    pump direction the second.
    """
    if npump < 2:
        raise ValueError("a pumping mesh needs at least two steps, t = 0 and t = 1/2")
    loop, pump = PLANE_AXES[axis % 3]
    base = np.zeros(3)
    base[axis % 3] = float(offset)

    points = np.zeros((nloop, npump, 3))
    points[..., :] = base
    points[..., loop] += (np.arange(nloop) / nloop)[:, None]
    points[..., pump] += (np.arange(npump) / (2 * (npump - 1)))[None, :]

    span1 = np.zeros(3, dtype=int)
    span1[loop] = 1
    span2 = np.zeros(3, dtype=int)
    span2[pump] = 1
    return PlaneMesh(points=points, span1=span1, span2=span2, closed=(True, False))


def string_mesh(
    transverse: tuple[int, int],
    npoints: int,
    gdir: int = 2,
    shift: tuple[int, int, int] = (0, 0, 0),
) -> PlaneMesh:
    """The strings a Berry-phase polarization along ``gdir`` is averaged over.

    ``transverse`` is the number of points along the two crystal directions
    *other* than ``gdir``, in the cyclic order of :data:`PLANE_AXES`, and
    ``npoints`` the number of **distinct** k-points on each string. The mesh is
    open across the transverse plane -- no overlap is ever taken in that
    direction, the strings being independent -- and closed along ``gdir``, whose
    span is the unit reciprocal vector.

    ``npoints`` is one fewer than QE's ``nppstr``. ``kp_strings`` lays out
    ``nppstr`` points spanning the whole reciprocal vector, so its last point is
    the first one's periodic image; here the repeat is dropped and the wrap
    closes the string instead, which leaves the same links and one fewer
    diagonalisation per string. A run being compared against ``pw.x`` therefore
    takes ``npoints = nppstr - 1``.

    ``shift`` is QE's ``k1, k2, k3``: a half-step offset per crystal direction.
    Its ``gdir`` entry is ignored, because a string has to start where it can
    close on itself and an offset along it is a relabelling of the same loop.
    """
    axis = gdir % 3
    d1, d2 = PLANE_AXES[axis]
    n1, n2 = int(transverse[0]), int(transverse[1])
    npoints = int(npoints)
    if n1 < 1 or n2 < 1:
        raise ValueError(f"a string mesh needs at least one string per direction, got {transverse}")
    if npoints < 2:
        raise ValueError(
            f"a string needs at least two k-points to have a phase, got {npoints} "
            "(this is nppstr - 1, one fewer than the number pw.x is given)"
        )

    offsets = np.asarray(shift, dtype=float)
    first = (np.arange(n1) + 0.5 * offsets[d1]) / n1
    second = (np.arange(n2) + 0.5 * offsets[d2]) / n2

    points = np.zeros((n1, n2, npoints, 3))
    points[..., d1] = first[:, None, None]
    points[..., d2] = second[None, :, None]
    points[..., axis] = (np.arange(npoints) / npoints)[None, None, :]

    span2 = np.zeros(3, dtype=int)
    span2[axis] = 1
    return PlaneMesh(
        points=points.reshape(n1 * n2, npoints, 3),
        span1=np.zeros(3, dtype=int),
        span2=span2,
        closed=(False, True),
    )


def trim_points(dimension: int = 3, axis: int = 2, offset: float = 0.0) -> np.ndarray:
    """The time-reversal-invariant momenta, in crystal coordinates.

    Eight of them in three dimensions, in the order

        (0,0,0) (0,0,½) (0,½,0) (0,½,½) (½,0,0) (½,0,½) (½,½,0) (½,½,½)

    -- first component slowest, as ``elkpy``'s ``TRIM_3D`` generates them, so
    that a delta table can be read across the two codes without re-sorting.
    In two dimensions there are four, spanning the plane normal to ``axis``
    (again in the cyclic order of :data:`PLANE_AXES`) with that component held
    at ``offset``, which for a slab is the vacuum direction and must be zero for
    the points to be TRI at all.
    """
    if dimension == 3:
        half = (0.0, 0.5)
        return np.array([(a, b, c) for a in half for b in half for c in half])
    if dimension != 2:
        raise ValueError(f"dimension must be 2 or 3, got {dimension}")
    d1, d2 = PLANE_AXES[axis % 3]
    points = []
    for a in (0.0, 0.5):
        for b in (0.0, 0.5):
            k = np.zeros(3)
            k[d1], k[d2] = a, b
            k[axis % 3] = float(offset)
            points.append(k)
    return np.array(points)
