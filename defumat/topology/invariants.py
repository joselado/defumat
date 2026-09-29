"""The invariants as calculations: what to diagonalise, and in what order.

:mod:`~defumat.topology.berry`, :mod:`~defumat.topology.wilson` and
:mod:`~defumat.topology.parity` are algorithms on state sets. This module is
the layer above: it decides *which* k-points a given invariant needs, asks a
**state source** for the states there, and feeds them to the algorithm. Both
registered Z2 methods live here, because that is where they share a signature --
``method(source, **kwargs)`` -- and the registry is what makes them
interchangeable in :mod:`defumat.workflows.topology`.

A **state source** is anything with

    ``states(points, keep_projectors=False) -> StateSet``

for ``points`` a ``(n, 3)`` array of crystal k-points, plus an ``nocc``
attribute. One optional extension: a source that can also answer
``keep_velocity=True`` returns a state set carrying the velocity operator and
the whole diagonalised band set, which is what the ``kubo`` curvature needs and
what nothing else here does. It is passed only when it is asked for, so a
source implementing the two-argument signature alone still satisfies the
protocol. Two exist: :class:`ModelSource` here, for a tight-binding model, and
``defumat.workflows.topology.DFTSource``, which runs a fixed-density
diagonalisation. Everything in this module is written against the protocol and
neither knows the other exists.

**Why a source rather than one big state set.** Memory. A Wilson loop over a
24x13 half-zone mesh of bismuthene spinors would hold 810 MB of wavefunctions;
one loop of it holds 63 MB. So :func:`wilson_z2` calls the source once per
pumping step and drops the previous step's states: the peak is
``nloop * nocc * npol * npwx * 16`` bytes and does not grow with the pumping
resolution at all.

The plaquette mesh a Chern number needs is streamed too, a **column** at a
time (:func:`chern_number` with ``stream=True``): a plaquette joins two
neighbouring columns, so the links along a column are taken inside it, the
links across to the next column between the two, and only the link *phases*
-- ``(n1, n2)`` complex numbers -- are kept. The first column is held to the
end, because the last column's neighbour is the first one plus a reciprocal
lattice vector. At most three columns of states are ever resident (the first,
the current and the next), against the whole plane. It is the default where
the source keeps its states streamed -- memory mode on a card
(``DFTSource.streams``) -- and not on a CPU, because each column is its own
diagonalisation with its own plane-wave padding and therefore its own compiled
solve. Streaming is a dial rather than a law, and ``stream=False`` on
:func:`wilson_z2` is the same trade for the pumping mesh.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from defumat.topology.berry import BerryCurvature, berry_curvature, plaquette_flux
from defumat.topology.links import link_phase
from defumat.topology.mesh import (
    PLANE_AXES,
    PlaneMesh,
    plane_mesh,
    pumping_mesh,
    trim_points,
)
from defumat.topology.parity import (
    ParityInvariant,
    fu_kane_z2,
    parity_eigenvalues,
    trim_delta,
)
from defumat.topology.registry import get_z2_method, register_z2_method
from defumat.topology.states import ModelStates
from defumat.topology.wilson import (
    WannierFlow,
    Z2Invariant3D,
    combine_3d,
    wilson_z2_from_loops,
)

__all__ = [
    "ModelSource",
    "chern_number",
    "wilson_z2",
    "parity_z2",
    "z2_invariant",
    "z2_invariant_3d",
]


@dataclass
class ModelSource:
    """A state source backed by a model ``H(k)`` -- the tests' and notebook's.

    ``hamiltonian(k)`` takes crystal coordinates and returns a Hermitian matrix
    in pure JAX, so it is differentiable and the ``kubo`` curvature works on it.
    ``inversion`` is the matrix representing spatial inversion on the basis, for
    the parity route; ``None`` means the model has no inversion centre and the
    parity method refuses, exactly as a crystal without one does.
    """

    hamiltonian: object
    nocc: int
    orbital_positions: np.ndarray | None = None
    inversion: np.ndarray | None = None

    def states(self, points, keep_projectors: bool = False,
               keep_velocity: bool = False,
               keep_hamiltonian: bool = False) -> ModelStates:
        # ``keep_velocity`` and ``keep_hamiltonian`` are accepted and ignored: a
        # model state set already carries ``H(k)`` itself, which is what its
        # Kubo route differentiates and what an orbital magnetization applies.
        return ModelStates.solve(
            self.hamiltonian,
            points,
            self.nocc,
            self.orbital_positions,
            inversion=self.inversion,
        )


def chern_number(
    source,
    shape=(12, 12),
    axis: int = 2,
    offset: float = 0.0,
    method: str | None = None,
    k_batch="default",
    stream: bool | None = None,
    **kwargs,
) -> BerryCurvature:
    """Berry curvature and the Chern number over one plane of the zone.

    ``axis`` is the crystal direction held fixed at ``offset``; the plane is
    spanned by the other two. For a two-dimensional crystal the only meaningful
    choice is the stacking axis at ``offset = 0``, which is the default.

    ``stream`` walks the plane a column at a time instead of diagonalising it
    whole (the module docstring; ``GPU-MEMORY-NEXT.md`` item 5). ``None`` asks
    the source -- ``source.streams``, true for a plane-wave source in memory
    mode on a card and absent (so false) for a model. The two routes take the
    same overlaps on states solved at the same k-points, so a Chern number is
    the same integer on both and the flux agrees to what the eigensolver's
    threshold leaves in each state.
    """
    mesh = plane_mesh(shape, axis=axis, offset=offset)
    # The ``kubo`` route is a sum over *empty* states through a velocity
    # operator, so it needs more of the source than the occupied manifold every
    # other quantity here is a property of. Asked for by name rather than
    # always, because it doubles what a mesh of states costs to hold.
    from defumat.topology.registry import (
        DEFAULT_CURVATURE_METHOD,
        get_curvature_method,
    )

    name = (method or DEFAULT_CURVATURE_METHOD).lower()
    get_curvature_method(name)  # an unknown method is refused before any solve
    wants_velocity = name == "kubo"
    # Passed only when it is wanted, so that a source written to the protocol's
    # two-argument signature still satisfies it.
    extra = {"keep_velocity": True} if wants_velocity else {}
    if stream is None:
        stream = bool(getattr(source, "streams", False))
    if stream and wants_velocity and isinstance(source, ModelSource):
        # A model's Kubo route takes its degeneracy scale from the band width
        # over the mesh it is handed, which a column would change; and a model
        # has nothing to save by streaming. So it stays whole.
        stream = False
    if stream:
        if wants_velocity:
            return _streamed_kubo(source, mesh, extra, k_batch=k_batch, **kwargs)
        return _streamed_fhs(source, mesh, k_batch=k_batch)
    states = source.states(mesh.flat(), **extra)
    return berry_curvature(states, mesh, method=method, k_batch=k_batch, **kwargs)


def _column(mesh: PlaneMesh, i: int) -> PlaneMesh:
    """Column ``i`` of a plane mesh as a mesh of its own, open across."""
    return PlaneMesh(points=mesh.points[i:i + 1], span1=mesh.span1,
                     span2=mesh.span2, closed=(False, mesh.closed[1]))


def _streamed_fhs(source, mesh: PlaneMesh, k_batch="default") -> BerryCurvature:
    """The FHS curvature with the plane diagonalised one column at a time.

    Column ``i`` is ``mesh.points[i]``. The links along it, ``U_2(i, :)``, are
    taken inside that column's state set, the wrap at ``j = n2 - 1`` being the
    shift by ``span2`` exactly as :func:`~defumat.topology.berry.link_variables`
    takes it; the links across, ``U_1(i, :)``, between column ``i`` and column
    ``i + 1`` (``overlaps(..., other=...)``), the last one reaching back to
    column 0 through ``span1``. Everything after the phases is
    :func:`~defumat.topology.berry.plaquette_flux`, the whole-mesh route's own.
    """
    if not all(mesh.closed):
        raise ValueError(
            "the Chern number is an integral over a closed surface; this mesh "
            "is open in at least one direction"
        )
    n1, n2 = mesh.shape
    zero = np.zeros(3, dtype=int)
    along = [(j, (j + 1) % n2, mesh.span2 if j + 1 == n2 else zero)
             for j in range(n2)]
    first = source.states(mesh.points[0])
    current = first
    u1, u2 = [], []
    for i in range(n1):
        following = first if i + 1 == n1 else source.states(mesh.points[i + 1])
        u2.append(link_phase(current.overlaps(along, k_batch=k_batch)))
        across = mesh.span1 if i + 1 == n1 else zero
        u1.append(link_phase(current.overlaps(
            [(j, j, across) for j in range(n2)], k_batch=k_batch,
            other=following)))
        # Dropping the reference here is what bounds the working set: the
        # column just finished is never read again, except the first.
        current = following
    flux = plaquette_flux(jnp.stack(u1), jnp.stack(u2))
    return BerryCurvature(
        mesh=mesh, curvature=flux * n1 * n2, flux=flux, method="fhs"
    )


def _streamed_kubo(source, mesh: PlaneMesh, extra, k_batch="default",
                   **kwargs) -> BerryCurvature:
    """The Kubo curvature one column at a time: it is pointwise, so it splits.

    What does not split is the two diagnostics, and they are recombined as the
    whole-mesh route defines them: the truncation is the largest shift over
    the plane divided by the largest ``|Omega|`` over the plane, not a mean of
    per-column ratios, and the singular points are counted over every column.
    """
    n1, _ = mesh.shape
    parts = []
    for i in range(n1):
        states = source.states(mesh.points[i], **extra)
        parts.append(berry_curvature(states, _column(mesh, i), method="kubo",
                                     k_batch=k_batch, **kwargs))
        del states
    curvature = np.concatenate([np.asarray(p.curvature) for p in parts])
    by_band = (None if parts[0].curvature_by_band is None else
               np.concatenate([np.asarray(p.curvature_by_band) for p in parts]))
    shifts = [p.truncation_abs for p in parts]
    shift = None if any(s is None for s in shifts) else float(max(shifts))
    scale = float(np.max(np.abs(curvature)))
    truncation = None
    if shift is not None:
        truncation = shift / scale if scale > 0.0 else float("nan")
    singular = [p.singular_points for p in parts]
    return BerryCurvature(
        mesh=mesh,
        curvature=curvature,
        flux=None,
        method="kubo",
        curvature_by_band=by_band,
        nbnd=parts[0].nbnd,
        nocc=parts[0].nocc,
        truncation=truncation,
        truncation_abs=shift,
        singular_points=(None if any(s is None for s in singular)
                         else int(sum(singular))),
    )


def wilson_z2(
    source,
    axis: int = 2,
    offset: float = 0.0,
    nloop: int = 24,
    npump: int = 13,
    k_batch="default",
    stream: bool = True,
    **_,
) -> WannierFlow:
    """The 2D Z2 of one plane, by Wannier-charge-centre flow.

    ``stream=True`` (the default) builds and solves the loops **one at a time**:
    each pumping step's ``nloop`` k-points are diagonalised, reduced to ``nocc``
    charge-centre angles, and dropped. The working set is one loop's states and
    ``npump`` costs time rather than space -- 63 MB for a 24-point loop of
    bismuthene spinors against 810 MB for a 24x13 mesh of them.

    ``stream=False`` asks for the whole mesh in one call, which is worth having
    only where the per-call *setup* dominates the states.

    **It used to be the right default for the plane-wave case, and is not any
    more.** A :class:`~defumat.workflows.topology.DFTSource` once rebuilt a
    whole ``Calculation`` per call -- the dense G-vector set and the
    augmentation charge among it -- so a row cost ~1 GB and seconds where its
    states cost megabytes, and taking the whole mesh at once was three times
    faster at a *lower* peak because most of the setups never happened. That was
    a real measurement of an avoidable cost:
    :meth:`~defumat.scf.driver.Calculation.at_kpoints` now shares everything a
    k-list does not affect, a row is 29.8x cheaper, and streaming is simply the
    cheap option. The flag stays because the two ends still differ in *how* they
    spend, but the reason to reach for ``stream=False`` has gone.
    """
    mesh = pumping_mesh(nloop, npump, axis=axis, offset=offset)
    n1, n2 = mesh.shape
    pump = mesh.points[0, :, PLANE_AXES[axis % 3][1]]

    if not stream:
        states = source.states(mesh.flat())
        loops = (
            (states.select([int(mesh.index(i, j)) for i in range(n1)]), mesh.span1)
            for j in range(n2)
        )
        return wilson_z2_from_loops(loops, pump=pump)

    def loops():
        for j in range(n2):
            yield source.states(mesh.points[:, j, :]), mesh.span1

    return wilson_z2_from_loops(loops(), pump=pump)


def parity_z2(
    source,
    dimension: int = 3,
    axis: int = 2,
    offset: float = 0.0,
    centre=None,
    **_,
) -> ParityInvariant:
    """The Fu-Kane parity invariants, from the four or eight TRIM.

    ``centre`` is the inversion centre in crystal coordinates. A DFT source
    finds it from the space group and passes it; a model's is the origin unless
    it says otherwise.
    """
    points = trim_points(dimension, axis=axis, offset=offset)
    states = source.states(points, keep_projectors=True)
    centre = np.zeros(3) if centre is None else np.asarray(centre, dtype=float)

    deltas, eigenvalues = {}, {}
    for index, point in enumerate(points):
        matrix = states.parity_matrix(index, centre)
        values = parity_eigenvalues(matrix)
        key = tuple(float(x) for x in point)
        eigenvalues[key] = values
        deltas[key] = trim_delta(values)

    result = fu_kane_z2(deltas, dimension=dimension)
    result.eigenvalues = eigenvalues
    return result


def z2_invariant(source, method: str | None = None, **kwargs):
    """The 2D Z2 invariant of one plane, by the named method."""
    return get_z2_method(method)(source, **kwargs)


def z2_invariant_3d(
    source,
    method: str | None = None,
    nloop: int = 24,
    npump: int = 13,
    k_batch="default",
    **kwargs,
) -> Z2Invariant3D:
    """The four indices ``(nu0; nu1 nu2 nu3)``.

    The ``parity`` method computes all four at once from eight k-points. The
    ``wilson`` method runs six independent 2D calculations, one per plane
    ``k_i = 0`` and ``k_i = 1/2``, and :func:`~defumat.topology.wilson.combine_3d`
    assembles them -- including the consistency check that the three axes agree
    about ``nu0``, which they must as an identity.
    """
    name = (method or "wilson").lower()
    if name == "parity":
        result = parity_z2(source, dimension=3, **kwargs)
        return Z2Invariant3D(
            nu0=result.nu0,
            nu=result.nu,
            nu0_by_axis=(result.nu0,) * 3,
            planes={"parity": result},
        )
    planes, flows = {}, {}
    for axis in range(3):
        for offset in (0.0, 0.5):
            flow = get_z2_method(name)(
                source,
                axis=axis,
                offset=offset,
                nloop=nloop,
                npump=npump,
                k_batch=k_batch,
                **kwargs,
            )
            planes[(axis, offset)] = flow.z2
            flows[(axis, offset)] = flow
    combined = combine_3d(planes)
    combined.planes = flows
    return combined


register_z2_method("wilson", wilson_z2)
register_z2_method("parity", parity_z2)
