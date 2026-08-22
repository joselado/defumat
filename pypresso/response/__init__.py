"""Linear response: the velocity operator, the Sternheimer solve, and the
dielectric tensor.

``PLAN.md`` P11 and P22c. Everything here is one construction seen three times:
the response of a self-consistent state to a perturbation, obtained by
differentiating the code that builds it rather than by transcribing a derived
expression.

``velocity``
    ``v = dH/dk`` from a ``jvp`` of ``H(k)`` at a frozen sphere (rule D2), and
    the band velocities it gives.
``sternheimer``
    ``(H - eps S) |dpsi> = -P_c dV|psi>`` by projected conjugate gradient, and
    the independent-particle susceptibility ``chi_0 = drho/dV`` built from it.
    Ultrasoft and PAW come with it rather than needing a second implementation:
    ``dbecsum``, the augmentation charge's own response, ``int3`` and
    ``PAW_dpotential`` are all derivatives of code that already existed.
``efield``
    the response to a uniform electric field, and the dielectric tensor and Born
    effective charges it gives.
``phonon``
    the response to an atomic displacement, and the force constants at
    ``Gamma``. The second derivative of the energy is one ``jvp`` of the
    gradient the force already is, so ``dynmat0`` and ``drhodv`` are two halves
    of one tangent rather than two routines.
``strain``
    the response to a homogeneous strain -- the third perturbation, and the one
    that carries a rank-2 label. Abinit's metric-tensor formulation is not
    implemented here; it is what ``Calculation.at_strain`` already was.
``elastic``
    the elastic constants, which are the stress differentiated once more along
    that response: ``phonon``'s construction with the cell in place of the atoms.
``electrostriction``
    ``d(chi)/d(strain)`` as a **third** derivative, from the 2n+1 theorem in the
    one form it takes here -- the second-order energy is stationary in the
    first-order wavefunctions, so it may be differentiated with them held fixed.
"""

from pypresso.response.efield import DielectricTensor, dielectric_tensor
from pypresso.response.elastic import ElasticConstants, elastic_constants
from pypresso.response.electrostriction import Electrostriction, electrostriction
from pypresso.response.strain import StrainResponse, strain_response
from pypresso.response.phonon import Phonons, dynamical_matrix
from pypresso.response.sternheimer import (
    SternheimerResult,
    SternheimerSolver,
    local_perturbation,
    make_sternheimer,
)
from pypresso.response.velocity import (
    BandVelocities,
    VelocityOperator,
    band_velocities,
)

__all__ = [
    "BandVelocities",
    "DielectricTensor",
    "ElasticConstants",
    "Electrostriction",
    "Phonons",
    "SternheimerResult",
    "SternheimerSolver",
    "StrainResponse",
    "VelocityOperator",
    "band_velocities",
    "dielectric_tensor",
    "dynamical_matrix",
    "elastic_constants",
    "electrostriction",
    "local_perturbation",
    "make_sternheimer",
    "strain_response",
]
