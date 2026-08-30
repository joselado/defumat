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
    the response to a uniform electric field, and the dielectric tensor it
    gives.
``born``
    the Born effective charges, which are ``dF/dE`` -- one ``jvp`` of the force
    along that response, so an ultrasoft dataset costs one more tangent rather
    than ``zstar_eu_us.f90``'s five stages.
``phonon``
    the response to an atomic displacement, and the force constants at
    ``Gamma``. The second derivative of the energy is one ``jvp`` of the
    gradient the force already is, so ``dynmat0`` and ``drhodv`` are two halves
    of one tangent rather than two routines.
``effmass``
    the effective mass tensor ``(1/m*)_ab = (1/2) d^2 eps_n/dk_a dk_b``. The
    first derivative is ``velocity``'s ``jvp``; the second is one central
    difference of it, because an individual band's ``|dpsi/dk>`` is not what
    the Sternheimer projector produces and the band is usually empty anyway.
    Elk's eigenvalue-differencing route is kept beside it as the check that
    shares no machinery with the operator.
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
``nonlinear``
    the Raman tensors, ``d(eps)/d(tau)``: the same third derivative with the
    atoms as its geometry variable instead of the cell, so the phase is an
    assembly of tangents the two modules above already produce. ``chi^(2)`` and
    the electro-optic tensor are refused there, with the term they need named.
``spectra``
    what a spectroscopist reads: the tensors above are per *atom* and an
    experiment resolves a *mode*, so this contracts them with the phonon
    eigendisplacements into per-mode Raman and infrared activities. It solves
    nothing, and it is the one thing here with a QE reference that still works
    -- ``dynmat.x``'s ``RamanIR`` is post-processing and shares nothing with the
    third-derivative branch that regressed.
"""

from pypresso.response.born import born_effective_charges
from pypresso.response.efield import DielectricTensor, dielectric_tensor
from pypresso.response.effmass import EffectiveMass, Multiplet, effective_mass
from pypresso.response.elastic import ElasticConstants, elastic_constants
from pypresso.response.electrostriction import Electrostriction, electrostriction
from pypresso.response.nonlinear import RamanTensors, raman_tensors
from pypresso.response.strain import StrainResponse, strain_response
from pypresso.response.phonon import Phonons, dynamical_matrix
from pypresso.response.spectra import (
    VibrationalSpectrum,
    mode_activities,
    vibrational_spectrum,
)
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
    "EffectiveMass",
    "ElasticConstants",
    "Electrostriction",
    "Multiplet",
    "Phonons",
    "RamanTensors",
    "SternheimerResult",
    "SternheimerSolver",
    "StrainResponse",
    "VelocityOperator",
    "VibrationalSpectrum",
    "band_velocities",
    "born_effective_charges",
    "dielectric_tensor",
    "dynamical_matrix",
    "effective_mass",
    "elastic_constants",
    "electrostriction",
    "local_perturbation",
    "make_sternheimer",
    "mode_activities",
    "raman_tensors",
    "strain_response",
    "vibrational_spectrum",
]
