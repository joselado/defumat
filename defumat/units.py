"""Physical constants and unit conversions, in Rydberg atomic units.

Internal units are QE's (rule R6): energies in Ry, lengths in bohr, e^2 = 2.
Values are transcribed from the reference source's ``Modules/constants.f90``
(CODATA 2018) so that defumat and QE agree bit-for-bit on conversions; a test
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
#: The vacuum permittivity, F/m. Wanted only where a *quadratic* response is
#: reported in SI -- Tanner et al.'s electrostriction coefficients
#: (:mod:`defumat.response.electrostriction`) -- since nothing internal to a
#: Rydberg-atomic-unit code has an ``eps0`` in it.
EPSILON0_SI = 8.8541878128e-12

# --- conversions out of Rydberg atomic units ----------------------------------
HARTREE_TO_EV = HARTREE_SI / ELECTRONVOLT_SI
RY_TO_EV = HARTREE_TO_EV / 2.0
BOHR_TO_ANGSTROM = BOHR_RADIUS_SI * 1.0e10
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
AMU_TO_RY = (AMU_SI / ELECTRONMASS_SI) / 2.0
K_BOLTZMANN_RY = K_BOLTZMANN_SI / RYDBERG_SI
RY_TO_KELVIN = RYDBERG_SI / K_BOLTZMANN_SI

# --- frequency: what a phonon is printed in ------------------------------------
#: The atomic unit of time, in seconds, and in picoseconds. ``AU_TERAHERTZ`` is
#: ``AU_PS`` under another name in ``constants.f90``, kept there because a
#: frequency in THz and a time in ps are the same number.
AU_SEC = H_PLANCK_SI / TPI / HARTREE_SI
AU_PS = AU_SEC * 1.0e12
#: A frequency in Ry (energy, with hbar = 1) as THz and as cm^-1 -- the two
#: units ``dyndia`` prints a phonon in. The ``4 pi`` is not decoration: an
#: angular frequency in Hartree a.u. is ``1/AU_SEC``, and the two factors of two
#: that take it to an ordinary frequency in Rydberg units are exactly that.
RY_TO_THZ = 1.0 / AU_PS / FPI
RY_TO_CMM1 = 1.0e10 * RY_TO_THZ / C_SI

_AU_GPA = HARTREE_SI / BOHR_RADIUS_SI**3 / 1.0e9
RY_TO_KBAR = 10.0 * _AU_GPA / 2.0  # Ry/bohr^3 -> kbar, as QE prints stress

# --- the one that bites -------------------------------------------------------
#: Square of the electron charge in Rydberg atomic units. Hartree and XC terms
#: carry an explicit ``E2`` in QE; dropping it is the classic factor-of-two bug
#: when transcribing from a Hartree-unit reference.
E2 = 2.0

#: ``e/bohr^2`` to C/m^2, the unit a piezoelectric constant is tabulated in
#: (:mod:`defumat.response.piezo`). ``e_(k)ij = d(sigma_ij)/dE_k`` comes out of
#: a Rydberg-atomic-unit code in electrons per bohr squared, and 57.2 is the
#: number that makes ``e_14`` comparable with a measurement.
E_BOHR2_TO_C_M2 = ELECTRON_SI / BOHR_RADIUS_SI**2

#: One Hartree atomic unit of conductivity, ``e^2 / (hbar a_0)``, in S/m. The
#: optical conductivity (:mod:`defumat.response.conductivity`) is assembled in
#: atomic units -- the Kubo-Greenwood sum is ``1/Omega`` times a squared
#: velocity over a squared energy, which in Rydberg units is ``bohr^2/bohr^3``
#: and therefore ``1/bohr`` -- and 4.6e6 is what takes it to the S/cm an
#: anomalous Hall conductivity is tabulated in.
AU_CONDUCTIVITY_SI = ELECTRON_SI**2 / (H_PLANCK_SI / TPI) / BOHR_RADIUS_SI
AU_TO_S_PER_CM = AU_CONDUCTIVITY_SI / 100.0
