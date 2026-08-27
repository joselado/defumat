"""Time-dependent density functional theory in the frequency domain.

``PLAN.md`` P37. What this computes is the macroscopic dielectric function of an
insulator in the optical limit, ``eps_M(omega)`` -- and with a kernel that is
not zero, so that ``Im eps_M`` carries the bound electron-hole pair that RPA and
ALDA cannot produce. The kernel is Sharma, Dewhurst, Sanna and Gross's
**bootstrap** (PRL **107**, 186401 (2011)), which is Elk's ``fxctype = 210``.

``chi0``
    the independent-particle response as a *matrix* over reciprocal lattice
    vectors and frequencies, by Adler-Wiser sum over states -- the one
    expensive object here, and the only one ``pypresso/response/`` cannot
    supply, since a Sternheimer solve gives ``chi_0`` as an operator.
``kernels``
    the exchange-correlation kernels of the Dyson equation, behind a name
    registry: ``rpa``, ``alda``, ``lrc``, ``bootstrap`` and ``bootstrap-1``.
``dyson``
    the Dyson equation itself, the bootstrap's fixed point, and the
    macroscopic tensor -- which is the inverse of the **3x3 head** of
    ``eps^-1`` and not the head of the inverse.

**There is no Quantum ESPRESSO counterpart to transcribe.** ``TDDFPT/`` is a
Liouville-Lanczos solver with RPA and ALDA; it has no bootstrap kernel and no
Dyson equation in G space. The reference is Elk (``tddftlr.f90``,
``genvchi0.f90``, ``genvfxc.f90``).
"""

from pypresso.tddft.chi0 import (
    ChiZero,
    ResponseSphere,
    independent_response,
    require_a_sum_over_states_regime,
    response_sphere,
)
from pypresso.tddft.dyson import DysonSolution, macroscopic_tensor, solve_dyson
from pypresso.tddft.kernels import (
    DEFAULT_KERNEL,
    XCKernel,
    alda_matrix,
    get_kernel,
    kernel_names,
    register_kernel,
)

__all__ = [
    "ChiZero",
    "DEFAULT_KERNEL",
    "DysonSolution",
    "ResponseSphere",
    "XCKernel",
    "alda_matrix",
    "get_kernel",
    "independent_response",
    "kernel_names",
    "macroscopic_tensor",
    "register_kernel",
    "require_a_sum_over_states_regime",
    "response_sphere",
    "solve_dyson",
]
