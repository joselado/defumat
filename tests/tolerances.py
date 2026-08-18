"""Comparison tolerances, declared once per physical quantity.

Tolerances live here rather than in individual tests so that "how close to QE is
close enough" is a single, reviewable decision. A test that needs a looser bound
than these must say why in a comment at the point of use.

Values are in the units the quantity is compared in: Ry for energies, eV for
eigenvalues as pw.x prints them, Ry/bohr for forces, Ry/bohr^3 for stress.
"""

#: Total energy and every printed energy term (one-electron, Hartree, XC, Ewald,
#: smearing). QE prints 8 decimals, so this is essentially "the last digit".
TOTAL_ENERGY_RY = 1e-6
ENERGY_TERM_RY = 1e-6

#: Eigenvalues, printed by pw.x with 4 decimals in eV.
EIGENVALUE_EV = 1e-4
FERMI_EV = 1e-4

#: Forces from autodiff against QE's analytic Hellmann-Feynman + Pulay forces.
FORCE_RY_BOHR = 1e-4
STRESS_RY_BOHR3 = 1e-4

#: Densities of states, in states/eV.
DOS_STATES_EV = 1e-3

#: Structural quantities that must agree exactly in exact arithmetic but are
#: printed rounded (lattice vectors in units of alat, k-points in 2pi/alat).
GEOMETRY = 1e-6

#: Finite-difference gradient checks (D5). Central differences with a step
#: chosen per quantity; this is the relative agreement demanded of them.
FINITE_DIFFERENCE_REL = 1e-5

#: Single-precision runs cannot meet the float64 bounds; they exist to prove the
#: dtype-generic path still runs and stays physically sensible.
FLOAT32_ENERGY_RY = 1e-3
FLOAT32_EIGENVALUE_EV = 1e-2
