"""Charge and magnetization integrated in a sphere around each atom.

A plane-wave code has no muffin tins, so "the moment on this atom" is not a
quantity the basis defines -- it is a choice of region. QE makes that choice once
(``PW/src/make_pointlists.f90``) and then reuses it everywhere an atom-resolved
magnetic quantity is needed: to *report* local moments (``report_mag.f90``), to
*constrain* them (``add_bfield.f90``), and to apply a field to one atom rather
than to the whole cell.

The region is a sphere of radius ``r_m`` around each atom, with a weight that is
1 inside it and falls linearly to 0 at ``1.2 r_m``:

    w(d) = 1                                for d <= r_m
         = 1 - (d - r_m) / (0.2 r_m)        for r_m < d <= 1.2 r_m
         = 0                                beyond

and every grid point belongs to **at most one** atom -- QE assigns it to the
first atom whose (tapered) sphere contains it and stops looking. That is why
``r_m`` defaults to a little under half the nearest-neighbour distance divided by
1.2: it makes the tapered spheres disjoint, so the local charges sum to less than
the total and never double-count.

**The taper is a function of the atomic positions**, and that matters here
because forces are ``jax.grad`` of the energy (P15). Two implementations are
registered:

* ``qe`` -- the integer nearest-atom map with the linear taper, reproducing
  ``make_pointlists`` point for point. It is a *host-side* array of weights, so
  the positions are not in the autodiff graph and a constraint contributes no
  force. This is QE's behaviour, including the missing force, and it is the
  default because it is the one that can be compared against QE.
* ``smooth`` -- the same profile written as a differentiable partition of unity
  over the atoms, with the positions live. It gives the constraint's
  contribution to the force, and it does not reproduce QE bit for bit.

Which one a run used is recorded in the result rather than assumed.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from defumat.system.cell import Cell
from defumat.system.structure import Structure

__all__ = [
    "LocalRegions",
    "build_local_regions",
    "default_radii",
    "local_weights_registry",
    "get_locals",
]

#: The taper reaches zero at this multiple of ``r_m`` (``make_pointlists``).
TAPER = 1.2


def _minimum_image_distances(points: np.ndarray, positions: np.ndarray,
                             at: np.ndarray) -> np.ndarray:
    """``(npoint, nat)`` nearest-image distances, in bohr.

    ``compute_distances_SoA``: the difference is taken in crystal coordinates,
    folded into the unit cell, and then compared against the eight corners --
    which is the minimum image for any cell whose difference vector has been
    folded, and is what QE does rather than a general neighbour search.
    """
    difference = positions[None, :, :] - points[:, None, :]
    difference = difference - np.floor(difference)
    cartesian = difference @ at
    corners = np.array([
        [-i, -j, -k] for i in (0, 1) for j in (0, 1) for k in (0, 1)
    ]) @ at
    shifted = cartesian[:, :, None, :] + corners[None, None, :, :]
    return np.sqrt(np.min(np.sum(shifted**2, axis=-1), axis=-1))


def default_radii(cell: Cell, structure: Structure) -> np.ndarray:
    """``r_m`` per species in bohr, QE's rule when the input does not give one.

    Half the shortest distance from an atom of that species to any other atom,
    divided by the taper factor and shaved by 1%, so that no grid point can
    belong to two atoms. The distance falls back to the Wigner-Seitz radius of
    the cell when there is only one atom in it -- there is no neighbour to
    measure against, and the cell edge is the only length available.
    """
    at = np.asarray(cell.at, dtype=float)
    positions = np.asarray(structure.positions_crystal(cell)) % 1.0
    types = np.asarray(structure.types, dtype=int)

    images = np.array([
        [i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
        if (i, j, k) != (0, 0, 0)
    ]) @ at
    ws_radius = float(np.sqrt(np.min(np.sum(images**2, axis=1))))

    minimum = np.full(structure.ntyp, ws_radius)
    distances = _minimum_image_distances(positions, positions, at)
    for a in range(len(positions)):
        for b in range(len(positions)):
            if a == b:
                continue
            d = distances[a, b]
            minimum[types[a]] = min(minimum[types[a]], d)
            minimum[types[b]] = min(minimum[types[b]], d)
    return 0.5 * minimum / TAPER * 0.99


class LocalRegions(eqx.Module):
    """The per-atom integration weights on the dense grid.

    Two storage layouts, because the two schemes admit different ones and the
    difference is the whole cost of an atom-resolved quantity on a large cell:

    * **packed** (``qe``) -- QE's own ``pointlist`` (which atom owns the point,
      or ``-1``) and ``factlist`` (the taper there), which is possible exactly
      because a point belongs to at most one atom. It is ``(ngrid,)`` of each,
      so the cost does not grow with the number of atoms.
    * **dense** (``smooth``) -- one grid-shaped row per atom, which the
      partition of unity needs since a point contributes to several atoms.

    The layout is invisible above :meth:`integrate`, which is the only way
    either is read.

    **The number this is here for.** Dense is ``nat x ngrid x 8`` bytes and
    packed is ``ngrid x 12``, so they cross at two atoms and diverge from
    there. Measured on the committed cells, with the two estimates that decide
    the layout:

    ========================  ===  =========  ========  ========
    cell                      nat  ngrid      dense     packed
    ========================  ===  =========  ========  ========
    ``fe-kind1-noncol``         1     15 625   0.12 MB   0.19 MB
    ``h4-noncolin-force``       4    102 400   3.28 MB   1.23 MB
    ``h10-chain-noncolin``     10    256 000  20.48 MB   3.07 MB
    NiBr2 cycloid (estimate)   45  ~2 000 000  ~700 MB    ~24 MB
    157-atom slab (estimate)  157  ~4 000 000    ~5 GB    ~31 MB
    ========================  ===  =========  ========  ========

    The right-hand column is what lets the per-atom moment be reported every
    iteration on any magnetic run rather than only on the small ones.
    """

    radii: tuple = eqx.field(static=True)
    grid: tuple = eqx.field(static=True)
    nat: int = eqx.field(static=True)
    scheme: str = eqx.field(static=True, default="qe")
    #: ``(nat, n1, n2, n3)`` -- the dense layout, or ``None`` when packed.
    weights: jnp.ndarray | None = None
    #: ``(ngrid,)`` int -- which atom owns each point, ``-1`` for none.
    owner: jnp.ndarray | None = None
    #: ``(ngrid,)`` -- the taper at each owned point.
    taper: jnp.ndarray | None = None

    @property
    def packed(self) -> bool:
        return self.weights is None

    def dense_weights(self) -> jnp.ndarray:
        """``(nat, n1, n2, n3)`` whichever layout is stored.

        For the packed layout this *builds* the array the layout exists to
        avoid, so it is for tests and for comparing the two schemes, never for
        the SCF.
        """
        if self.weights is not None:
            return self.weights
        rows = jnp.zeros((self.nat, self.owner.size), dtype=self.taper.dtype)
        keep = self.owner >= 0
        rows = rows.at[jnp.where(keep, self.owner, 0), jnp.arange(self.owner.size)].set(
            jnp.where(keep, self.taper, 0.0)
        )
        return rows.reshape((self.nat,) + tuple(self.grid))

    def integrate(self, field: jnp.ndarray) -> jnp.ndarray:
        """``(nat, ncomponent)``: each component of ``field`` in each sphere.

        ``field`` is ``(ncomponent, n1, n2, n3)``. This is the *sum* over grid
        points, with no volume element -- every caller multiplies by
        ``Omega / ngrid`` itself, as ``get_locals`` does.
        """
        if self.weights is not None:
            return jnp.einsum("anmk,cnmk->ac", self.weights, field)
        flat = field.reshape(field.shape[0], -1)
        # ``segment_sum`` wants a valid index everywhere, so unowned points are
        # parked on atom 0 with a zero taper and dropped by the multiplication.
        keep = self.owner >= 0
        contribution = jnp.where(keep, self.taper, 0.0)[None, :] * flat
        return jax.ops.segment_sum(
            contribution.T,
            jnp.where(keep, self.owner, 0),
            num_segments=self.nat,
        )


def _grid_points(grid: tuple[int, int, int]) -> np.ndarray:
    """The FFT grid in crystal coordinates, in the C order the arrays use."""
    n1, n2, n3 = grid
    i, j, k = np.meshgrid(np.arange(n1), np.arange(n2), np.arange(n3), indexing="ij")
    return np.stack([i.ravel() / n1, j.ravel() / n2, k.ravel() / n3], axis=1)


def _qe_weights(distances: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """``pointlist``/``factlist``: the first atom whose sphere holds the point.

    "First" is QE's ``EXIT`` out of the atom loop, and it is a real part of the
    definition rather than an implementation detail: with the default ``r_m``
    the spheres are disjoint, and where an input radius makes them overlap the
    lower atom index wins.
    """
    npoint, nat = distances.shape
    weights = np.zeros((nat, npoint))
    taken = np.zeros(npoint, dtype=bool)
    for a in range(nat):
        radius = radii[a]
        inside = (distances[:, a] <= radius) & ~taken
        tapered = (
            (distances[:, a] > radius)
            & (distances[:, a] <= TAPER * radius)
            & ~taken
        )
        weights[a, inside] = 1.0
        weights[a, tapered] = 1.0 - (
            distances[tapered, a] - radius
        ) / ((TAPER - 1.0) * radius)
        taken |= inside | tapered
    return weights


def _smooth_weights(distances: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """The same profile, without the "first atom wins" branch.

    Every atom's weight is the taper evaluated at its own distance, so a point
    inside two spheres contributes to both. With the default radii the spheres
    are disjoint and this agrees with :func:`_qe_weights` exactly; where they
    overlap it does not, which is the price of a rule that is a smooth function
    of the positions.
    """
    radius = radii[None, :]
    d = distances
    profile = np.clip((TAPER * radius - d) / ((TAPER - 1.0) * radius), 0.0, 1.0)
    return profile.T


local_weights_registry = {"qe": _qe_weights, "smooth": _smooth_weights}


def build_local_regions(
    cell: Cell,
    structure: Structure,
    grid: tuple[int, int, int],
    radii=None,
    scheme: str = "qe",
) -> LocalRegions:
    """Assign every point of the dense grid to an atom -- ``make_pointlists``.

    Args:
        grid: the dense FFT grid, which is where the density lives.
        radii: ``r_m`` per **species** in bohr, or ``None`` for QE's default.
        scheme: ``qe`` or ``smooth``; see the module docstring.
    """
    if scheme not in local_weights_registry:
        raise NotImplementedError(
            f"local weight scheme {scheme!r} is not implemented; available: "
            f"{sorted(local_weights_registry)}"
        )
    at = np.asarray(cell.at, dtype=float)
    positions = np.asarray(structure.positions_crystal(cell)) % 1.0
    types = np.asarray(structure.types, dtype=int)
    species_radii = (
        default_radii(cell, structure) if radii is None
        else np.asarray(radii, dtype=float)
    )
    per_atom = species_radii[types]

    points = _grid_points(grid)
    distances = _minimum_image_distances(points, positions, at)
    weights = local_weights_registry[scheme](distances, per_atom)
    common = dict(
        radii=tuple(float(r) for r in species_radii),
        grid=tuple(int(n) for n in grid),
        nat=len(positions),
        scheme=scheme,
    )
    if scheme == "qe":
        # Disjoint by construction, so the whole ``(nat, ngrid)`` array is one
        # owner index and one taper per point -- ``pointlist``/``factlist``.
        # ``argmax`` picks the single nonzero row; ``max`` is its value, and a
        # point in nobody's sphere has a maximum of exactly zero.
        taper = weights.max(axis=0)
        owner = np.where(taper > 0.0, weights.argmax(axis=0), -1)
        return LocalRegions(
            owner=jnp.asarray(owner),
            taper=jnp.asarray(taper),
            **common,
        )
    return LocalRegions(
        weights=jnp.asarray(weights.reshape((len(positions),) + tuple(grid))),
        **common,
    )


def get_locals(rho_r: jnp.ndarray, regions: LocalRegions, cell: Cell):
    """``get_locals``: the charge and moment inside each atom's sphere.

    Args:
        rho_r: ``(nspin_mag, n1, n2, n3)``.

    Returns ``(charge, moment)``: ``(nat,)`` in electrons and ``(nat, nspin-1)``
    in Bohr magnetons -- one column for a collinear run, three for a
    noncollinear one, and an empty one when there is no magnetization at all.
    The moment of a *collinear* run is ``rho_up - rho_down``, which is what the
    second component holds; QE's ``magloc(ipol, na) = auxrholoc(na, ipol+1)``
    says the same thing for both regimes because both store the magnetization in
    the components after the first.
    """
    scale = cell.volume / rho_r[0].size
    if rho_r.shape[0] == 2:
        components = jnp.stack([rho_r[0] + rho_r[1], rho_r[0] - rho_r[1]])
    else:
        components = rho_r
    integrated = scale * regions.integrate(components)
    return integrated[:, 0], integrated[:, 1:]
