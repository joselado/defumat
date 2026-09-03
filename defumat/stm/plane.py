"""The plotting plane an image is sampled on -- Elk's ``plotpt2d``.

A plane is three corners in **crystal** coordinates: an origin and the two
vertices that reach along its edges, exactly Elk's ``vclp2d(:,0..2)``. Crystal
coordinates are the right ones and not merely Elk's: a surface is a lattice
plane, its periodicity is the surface unit cell, and a tip height is naturally
"a fraction of the way up the slab's c axis". A user who reaches for cartesian
bohr has to know the cell to write down a plane that tiles.

The sampling follows ``plotpt2d`` including the detail that decides whether the
image tiles: the parameters run ``i/n`` and **not** ``i/(n-1)``, so the far edge
is excluded. On a plane spanning the whole cell that makes the image periodic --
column ``n`` would have repeated column ``0`` -- and it is why an image can be
tiled to show more than one surface cell without a seam.

``coordinates`` is the pair of *in-plane* cartesian distances in bohr, which is
what an axis of the plot should be labelled with: the two edges of a hexagonal
surface cell are 60 degrees apart, so a plot against the crystal parameters
``(t1, t2)`` is sheared and a plot against these is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PlotPlane", "plot_plane"]


@dataclass(frozen=True)
class PlotPlane:
    """A parallelogram of sampling points, in crystal and in-plane coordinates."""

    #: ``(n1, n2, 3)`` crystal coordinates of every point.
    points: np.ndarray
    #: ``(n1, n2, 2)`` cartesian coordinates *within the plane*, in bohr.
    coordinates: np.ndarray
    #: The three corners, in crystal coordinates.
    origin: np.ndarray
    edge1: np.ndarray
    edge2: np.ndarray
    #: The plane's unit normal, in crystal coordinates (Elk's ``vpnl``).
    normal: np.ndarray
    #: The two edge lengths in bohr and the angle between them in radians.
    lengths: tuple[float, float]
    angle: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.points.shape[:2]

    def flat(self) -> np.ndarray:
        """``(n1 n2, 3)`` -- what :func:`defumat.basis.sample.sample_field` takes."""
        return self.points.reshape((-1, 3))

    def offset(self, displacement) -> "PlotPlane":
        """The same plane translated by a crystal-coordinate ``displacement``.

        The in-plane coordinates are unchanged, which is the point: a stack of
        offset planes is a scan, and the images have to be comparable point by
        point for a constant-current inversion to mean anything.
        """
        shift = np.asarray(displacement, dtype=float)
        return PlotPlane(
            points=self.points + shift,
            coordinates=self.coordinates,
            origin=self.origin + shift,
            edge1=self.edge1 + shift,
            edge2=self.edge2 + shift,
            normal=self.normal,
            lengths=self.lengths,
            angle=self.angle,
        )


def plot_plane(cell, origin, edge1, edge2, shape=(40, 40)) -> PlotPlane:
    """Elk's ``plotpt2d``: sample the parallelogram spanned by three corners.

    Args:
        cell: the :class:`~defumat.system.cell.Cell`, for the metric -- the
            in-plane distances and the normal are the only things it decides.
        origin, edge1, edge2: the three corners in crystal coordinates. The
            parallelogram is ``origin`` plus the two vectors to the others.
        shape: ``(n1, n2)`` sampling, Elk's ``np2d``.

    ``points[i1, i2]`` runs ``i1`` along ``edge1 - origin``. Elk's ``STM2D.OUT``
    writes the transpose of that, ``i1`` fastest; nothing here reads its files.
    """
    at = np.asarray(cell.at, dtype=float)
    origin = np.asarray(origin, dtype=float)
    edge1 = np.asarray(edge1, dtype=float)
    edge2 = np.asarray(edge2, dtype=float)
    n1, n2 = (int(shape[0]), int(shape[1]))
    if n1 < 1 or n2 < 1:
        raise ValueError(f"the plotting grid must be at least 1x1, got {shape}")

    vl1, vl2 = edge1 - origin, edge2 - origin
    vc1, vc2 = vl1 @ at, vl2 @ at
    d1, d2 = float(np.linalg.norm(vc1)), float(np.linalg.norm(vc2))
    if min(d1, d2) < 1.0e-8:
        raise ValueError(
            "a plotting edge has zero length: the three corners must span a "
            "parallelogram, not a line or a point"
        )
    cosine = float(vc1 @ vc2) / (d1 * d2)
    normal = np.cross(vc1, vc2)
    norm = float(np.linalg.norm(normal))
    if norm < 1.0e-8 * d1 * d2:
        raise ValueError("the two plotting edges are collinear: they span no plane")

    t1 = (np.arange(n1) / n1)[:, None]
    t2 = (np.arange(n2) / n2)[None, :]
    points = (origin
              + t1[..., None] * vl1
              + t2[..., None] * vl2)
    # The in-plane frame Elk writes: the first edge along x, the second at its
    # true angle to it, so distances on the plot are distances in the crystal.
    sine = np.sqrt(max(0.0, 1.0 - cosine ** 2))
    coordinates = np.stack([t1 * d1 + t2 * d2 * cosine,
                            np.broadcast_to(t2 * d2 * sine, (n1, n2))], axis=-1)
    coordinates = np.broadcast_to(coordinates, (n1, n2, 2)).copy()

    return PlotPlane(
        points=points,
        coordinates=coordinates,
        origin=origin,
        edge1=edge1,
        edge2=edge2,
        # ``vpnl``: the cartesian normal expressed back in crystal coordinates,
        # which for a non-orthogonal cell is not the crystal-coordinate cross
        # product of the edges.
        normal=(normal / norm) @ np.linalg.inv(at),
        lengths=(d1, d2),
        angle=float(np.arccos(np.clip(cosine, -1.0, 1.0))),
    )
