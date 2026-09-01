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
``photocurrent``
    the shift current ``sigma^abc(0; w, -w)`` -- the bulk photovoltaic effect,
    a second-order optical response by sum over states
``shg``
    second-harmonic generation ``chi^(2)(-2w; w, w)``, the same sum over states
    contracted with two resonances instead of a delta. Elk's ``nonlinopt.f90``
    is a real reference for it and ``tests/data/elk/`` carries its output
``nesting``
    the Fermi-surface nesting function ``N(q)``, which is not a response at all
    -- it is a correlation of ``delta(eps - E_F)`` over the k-grid, and it
    lives here because it is the geometric half of the susceptibility the rest
    of this package solves for. Elk writes it as a double loop over ``q`` and
    ``k``; the fold that makes ``k + q`` land on the grid makes it a cyclic
    correlation, so one FFT gives the whole ``q`` dependence.
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

from pypresso.response.magnetoelectric import (
    MagnetoelectricTensor,
    magnetoelectric_tensor,
)
from pypresso.response.born import born_effective_charges
from pypresso.response.efield import DielectricTensor, dielectric_tensor
from pypresso.response.effmass import EffectiveMass, Multiplet, effective_mass
from pypresso.response.elastic import ElasticConstants, elastic_constants
from pypresso.response.electrostriction import Electrostriction, electrostriction
from pypresso.response.nonlinear import RamanTensors, raman_tensors
from pypresso.response.conductivity import (
    OpticalConductivity,
    optical_conductivity,
)
from pypresso.response.nesting import NestingFunction, nesting_from_eigenvalues
from pypresso.response.photocurrent import (
    ShiftCurrent,
    generalized_derivative,
    shift_current,
)
from pypresso.response.piezo import PiezoelectricTensor, piezoelectric_tensor
from pypresso.response.shg import (
    SecondHarmonic,
    band_velocity_difference,
    second_harmonic,
    shg_coefficients,
)
from pypresso.response.strain import StrainResponse, strain_response
from pypresso.response.phonon import Phonons, dynamical_matrix
from pypresso.response.spectra import (
    VibrationalSpectrum,
    loto_modes,
    mode_activities,
    neutral_born_charges,
    nonanal,
    polar_mode_permittivity,
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
    "MagnetoelectricTensor",
    "magnetoelectric_tensor",
    "BandVelocities",
    "DielectricTensor",
    "EffectiveMass",
    "ElasticConstants",
    "Electrostriction",
    "Multiplet",
    "NestingFunction",
    "Phonons",
    "PiezoelectricTensor",
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
    "loto_modes",
    "mode_activities",
    "nesting_from_eigenvalues",
    "neutral_born_charges",
    "nonanal",
    "OpticalConductivity",
    "optical_conductivity",
    "ShiftCurrent",
    "generalized_derivative",
    "shift_current",
    "SecondHarmonic",
    "band_velocity_difference",
    "second_harmonic",
    "shg_coefficients",
    "piezoelectric_tensor",
    "polar_mode_permittivity",
    "raman_tensors",
    "strain_response",
    "vibrational_spectrum",
]
