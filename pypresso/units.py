"""Physical constants and unit conversions, in Rydberg atomic units.

Internal units are QE's (rule R6): energies in Ry, lengths in bohr, e^2 = 2.
Values are transcribed from the reference source's ``Modules/constants.f90``
(CODATA 2018) so that pypresso and QE agree bit-for-bit on conversions; a test
asserts they still match. Conversions belong at the ``io`` boundary only.

These are plain Python floats, deliberately dtype-free: they are consumed inside
JAX expressions where the surrounding array's dtype decides the result, which is
what keeps the single-precision policy workable (see ``config``).
"""

import math

# --- geometry -----------------------------------------------------------------
PI = math.pi
TPI = 2.0 * PI
FPI = 4.0 * PI
SQRT_PI = math.sqrt(PI)

# --- SI reference values (constants.f90, CODATA 2018) --------------------------
H_PLANCK_SI = 6.62607015e-34  # J s
K_BOLTZMANN_SI = 1.380649e-23  # J / K
ELECTRON_SI = 1.602176634e-19  # C
ELECTRONVOLT_SI = 1.602176634e-19  # J
ELECTRONMASS_SI = 9.1093837015e-31  # kg
HARTREE_SI = 4.3597447222071e-18  # J
RYDBERG_SI = HARTREE_SI / 2.0  # J
BOHR_RADIUS_SI = 0.529177210903e-10  # m
AMU_SI = 1.66053906660e-27  # kg
C_SI = 2.99792458e8  # m / s

# --- conversions out of Rydberg atomic units ----------------------------------
HARTREE_TO_EV = HARTREE_SI / ELECTRONVOLT_SI
RY_TO_EV = HARTREE_TO_EV / 2.0
BOHR_TO_ANGSTROM = BOHR_RADIUS_SI * 1.0e10
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
AMU_TO_RY = (AMU_SI / ELECTRONMASS_SI) / 2.0
K_BOLTZMANN_RY = K_BOLTZMANN_SI / RYDBERG_SI
RY_TO_KELVIN = RYDBERG_SI / K_BOLTZMANN_SI

_AU_GPA = HARTREE_SI / BOHR_RADIUS_SI**3 / 1.0e9
RY_TO_KBAR = 10.0 * _AU_GPA / 2.0  # Ry/bohr^3 -> kbar, as QE prints stress

# --- the one that bites -------------------------------------------------------
#: Square of the electron charge in Rydberg atomic units. Hartree and XC terms
#: carry an explicit ``E2`` in QE; dropping it is the classic factor-of-two bug
#: when transcribing from a Hartree-unit reference.
E2 = 2.0
