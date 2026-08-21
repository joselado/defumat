"""DFT+U: the simplified rotationally-invariant Hubbard correction.

QE's ``lda_plus_u_kind = 0`` -- Dudarev's functional with the ``J0`` and
``beta`` extensions -- on the ``atomic``, ``ortho-atomic`` and ``norm-atomic``
projector sets, for ``nspin = 1`` and ``nspin = 2``, on norm-conserving,
ultrasoft and PAW datasets.

Four pieces, one per module:

* :mod:`~pypresso.hubbard.manifold` -- which orbitals, with which parameters
  (host-side setup);
* :mod:`~pypresso.hubbard.projectors` -- those orbitals in the plane-wave basis;
* :mod:`~pypresso.hubbard.occupations` -- the occupation matrix ``ns`` measured
  on them, its symmetrisation, and its starting value;
* :mod:`~pypresso.hubbard.energy` -- the energy, with the potential as its
  ``jax.grad``, and :mod:`~pypresso.hubbard.operator` applying that potential to
  the states.

Refused rather than approximated: the full (Liechtenstein) formulation
``lda_plus_u_kind = 1``, the intersite ``V`` (``kind = 2``), background
channels, the orbital-resolved variant, noncollinear ``ns``, and the ``wf`` and
``pseudo`` projector types.
"""

from pypresso.hubbard.energy import (
    coefficients_from_setup,
    hubbard_energy,
    hubbard_potential,
    ns_ddot,
    qe_hubbard_potential,
)
from pypresso.hubbard.manifold import (
    HubbardInput,
    HubbardSetup,
    HubbardSpecies,
    build_hubbard_setup,
    parse_manifold,
)
from pypresso.hubbard.occupations import (
    adjust_ns,
    build_ns_symmetry,
    initial_ns,
    occupation_matrix,
    projections,
)
from pypresso.hubbard.operator import HubbardTerm, block_potential
from pypresso.hubbard.projectors import build_hubbard_projectors

__all__ = [
    "HubbardInput",
    "HubbardSetup",
    "HubbardSpecies",
    "HubbardTerm",
    "adjust_ns",
    "block_potential",
    "build_hubbard_projectors",
    "build_hubbard_setup",
    "build_ns_symmetry",
    "coefficients_from_setup",
    "hubbard_energy",
    "hubbard_potential",
    "initial_ns",
    "ns_ddot",
    "occupation_matrix",
    "parse_manifold",
    "projections",
    "qe_hubbard_potential",
]
