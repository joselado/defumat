"""Berry curvature, Chern numbers and Z2 topological invariants.

Everything here is built from one primitive -- the overlap of the occupied
manifolds at two neighbouring k-points, through the pseudopotential's overlap
operator -- and the reason is in :mod:`pypresso.topology.states`: overlaps are
gauge invariant and degeneracy-safe where a derivative of an eigendecomposition
is neither (PLAN.md D4), and a lattice of them is *quantised* where a Riemann
sum of a pointwise curvature is not.

    ``mesh``          k-meshes and the reciprocal-lattice wrap that closes them
    ``states``        Bloch states and the overlap, for a model or a DFT run
    ``augmentation``  ultrasoft's ``S`` between two different k-points
    ``links``         determinants, polar decompositions, and the sign convention
    ``berry``         Berry curvature and the Chern number (``fhs``, ``kubo``)
    ``wilson``        Z2 from Wannier-charge-centre flow
    ``parity``        Z2 from the Fu-Kane parity products at the TRIM
    ``invariants``    which k-points each invariant needs, and the Z2 registry
    ``registry``      the names

:mod:`pypresso.workflows.topology` is the entry point a calculation uses.
"""

from __future__ import annotations

from pypresso.topology.berry import BerryCurvature, berry_curvature
from pypresso.topology.invariants import (
    ModelSource,
    chern_number,
    parity_z2,
    wilson_z2,
    z2_invariant,
    z2_invariant_3d,
)
from pypresso.topology.mesh import PlaneMesh, plane_mesh, pumping_mesh, trim_points
from pypresso.topology.parity import ParityInvariant, fu_kane_z2, inversion_centre
from pypresso.topology.registry import (
    curvature_methods,
    get_curvature_method,
    get_z2_method,
    z2_methods,
)
from pypresso.topology.states import (
    ArrayStates,
    ModelStates,
    PlaneWaveStates,
    StateSet,
    build_plane_wave_states,
)
from pypresso.topology.wilson import WannierFlow, Z2Invariant3D, combine_3d

__all__ = [
    "ArrayStates",
    "BerryCurvature",
    "ModelSource",
    "ModelStates",
    "ParityInvariant",
    "PlaneMesh",
    "PlaneWaveStates",
    "StateSet",
    "WannierFlow",
    "Z2Invariant3D",
    "berry_curvature",
    "build_plane_wave_states",
    "chern_number",
    "combine_3d",
    "curvature_methods",
    "fu_kane_z2",
    "get_curvature_method",
    "get_z2_method",
    "inversion_centre",
    "parity_z2",
    "plane_mesh",
    "pumping_mesh",
    "trim_points",
    "wilson_z2",
    "z2_invariant",
    "z2_invariant_3d",
    "z2_methods",
]
