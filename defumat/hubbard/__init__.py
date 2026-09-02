"""DFT+U: the simplified rotationally-invariant Hubbard correction.

QE's ``lda_plus_u_kind = 0`` -- Dudarev's functional with the ``J0`` and
``beta`` extensions -- and ``lda_plus_u_kind = 1``, Liechtenstein's full
rotationally-invariant functional with the Coulomb matrix built from the Slater
integrals; on the ``atomic``, ``ortho-atomic`` and ``norm-atomic`` projector
sets, for ``nspin = 1`` and ``nspin = 2``, on norm-conserving, ultrasoft and PAW
datasets. **Which of the two runs is decided by the card**, as QE decides it: a
``J``, ``B``, ``E2`` or ``E3`` selects the full functional.

Four pieces, one per module:

* :mod:`~defumat.hubbard.manifold` -- which orbitals, with which parameters
  (host-side setup);
* :mod:`~defumat.hubbard.projectors` -- those orbitals in the plane-wave basis;
* :mod:`~defumat.hubbard.occupations` -- the occupation matrix ``ns`` measured
  on them, its symmetrisation, and its starting value;
* :mod:`~defumat.hubbard.energy` -- the energy, with the potential as its
  ``jax.grad``, and :mod:`~defumat.hubbard.operator` applying that potential to
  the states;
* :mod:`~defumat.hubbard.interaction` -- the Coulomb matrix ``vee`` of the full
  (Liechtenstein) functional, from the Slater integrals.

Refused rather than approximated: the intersite ``V`` (``kind = 2``),
background channels, the orbital-resolved variant, noncollinear ``ns``, and the
``wf`` and ``pseudo`` projector types.
"""

from defumat.hubbard.energy import (
    coefficients_from_setup,
    elk_amf_potential,
    hubbard_energy,
    hubbard_potential,
    ns_ddot,
    qe_hubbard_full_potential,
    qe_hubbard_potential,
)
from defumat.hubbard.interaction import (
    coulomb_matrix,
    default_racah,
    exchange_from_slater,
    slater_integrals,
)
from defumat.hubbard.manifold import (
    HubbardInput,
    HubbardSetup,
    HubbardSpecies,
    build_hubbard_setup,
    parse_manifold,
)
from defumat.hubbard.occupations import (
    adjust_ns,
    build_ns_symmetry,
    initial_ns,
    ns_shape,
    occupation_matrix,
    projections,
    spin_averaged_ns,
    uniform_ns,
)
from defumat.hubbard.operator import HubbardTerm, block_potential
from defumat.hubbard.projectors import build_hubbard_projectors

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
    "coulomb_matrix",
    "default_racah",
    "elk_amf_potential",
    "exchange_from_slater",
    "hubbard_energy",
    "hubbard_potential",
    "initial_ns",
    "ns_ddot",
    "ns_shape",
    "occupation_matrix",
    "parse_manifold",
    "projections",
    "uniform_ns",
    "slater_integrals",
    "spin_averaged_ns",
    "qe_hubbard_full_potential",
    "qe_hubbard_potential",
]
