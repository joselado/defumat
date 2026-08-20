"""The invariants as calculations: what to diagonalise, and in what order.

:mod:`~pypresso.topology.berry`, :mod:`~pypresso.topology.wilson` and
:mod:`~pypresso.topology.parity` are algorithms on state sets. This module is
the layer above: it decides *which* k-points a given invariant needs, asks a
**state source** for the states there, and feeds them to the algorithm. Both
registered Z2 methods live here, because that is where they share a signature --
``method(source, **kwargs)`` -- and the registry is what makes them
interchangeable in :mod:`pypresso.workflows.topology`.

A **state source** is anything with

    ``states(points, keep_projectors=False) -> StateSet``

for ``points`` a ``(n, 3)`` array of crystal k-points, plus an ``nocc``
attribute. Two exist: :class:`ModelSource` here, for a tight-binding model, and
``pypresso.workflows.topology.DFTSource``, which runs a fixed-density
diagonalisation. Everything in this module is written against the protocol and
neither knows the other exists.

**Why a source rather than one big state set.** Memory. A Wilson loop over a
24x13 half-zone mesh of bismuthene spinors would hold 810 MB of wavefunctions;
one loop of it holds 63 MB. So :func:`wilson_z2` calls the source once per
pumping step and drops the previous step's states: the peak is
``nloop * nocc * npol * npwx * 16`` bytes and does not grow with the pumping
resolution at all.

Two honest qualifications. The plaquette mesh a Chern number needs is *not*
streamed -- :func:`chern_number` asks for the whole plane at once, because the
link variables along both directions are needed together and the mesh a Chern
number needs is small; splitting it is a change to make when a case demands it,
not before. And streaming is a dial rather than a law: a plane-wave source
rebuilds a gigabyte of setup on every call, which on a mesh of the sizes that
run here costs more than the states do, so ``stream=False`` takes the whole
pumping mesh in one go and is the right choice there. Both are stated where the
number is, in :func:`wilson_z2`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pypresso.topology.berry import BerryCurvature, berry_curvature
from pypresso.topology.mesh import PLANE_AXES, plane_mesh, pumping_mesh, trim_points
from pypresso.topology.parity import (
    ParityInvariant,
    fu_kane_z2,
    parity_eigenvalues,
    trim_delta,
)
from pypresso.topology.registry import get_z2_method, register_z2_method
from pypresso.topology.states import ModelStates
from pypresso.topology.wilson import (
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

    def states(self, points, keep_projectors: bool = False) -> ModelStates:
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
    **kwargs,
) -> BerryCurvature:
    """Berry curvature and the Chern number over one plane of the zone.

    ``axis`` is the crystal direction held fixed at ``offset``; the plane is
    spanned by the other two. For a two-dimensional crystal the only meaningful
    choice is the stacking axis at ``offset = 0``, which is the default.
    """
    mesh = plane_mesh(shape, axis=axis, offset=offset)
    states = source.states(mesh.flat())
    return berry_curvature(states, mesh, method=method, k_batch=k_batch, **kwargs)


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
    more.** A :class:`~pypresso.workflows.topology.DFTSource` once rebuilt a
    whole ``Calculation`` per call -- the dense G-vector set and the
    augmentation charge among it -- so a row cost ~1 GB and seconds where its
    states cost megabytes, and taking the whole mesh at once was three times
    faster at a *lower* peak because most of the setups never happened. That was
    a real measurement of an avoidable cost:
    :meth:`~pypresso.scf.driver.Calculation.at_kpoints` now shares everything a
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
    ``k_i = 0`` and ``k_i = 1/2``, and :func:`~pypresso.topology.wilson.combine_3d`
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
