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
"""

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
    "SternheimerResult",
    "SternheimerSolver",
    "VelocityOperator",
    "band_velocities",
    "local_perturbation",
    "make_sternheimer",
]
