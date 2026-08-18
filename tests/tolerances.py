"""Comparison tolerances, declared once per physical quantity.

Tolerances live here rather than in individual tests so that "how close to QE is
close enough" is a single, reviewable decision. A test that needs a looser bound
than these must say why in a comment at the point of use.

Values are in the units the quantity is compared in: Ry for energies, eV for
eigenvalues as pw.x prints them, Ry/bohr for forces, Ry/bohr^3 for stress.
"""

#: Total energy. QE prints 8 decimals, but its own runs stop at conv_thr = 1e-6,
#: so 1e-6 Ry is the most that can meaningfully be asked of the comparison.
#: (In practice silicon agrees to 1e-8 and the metals to 3e-8.)
TOTAL_ENERGY_RY = 1e-6

#: A single energy term. Density-independent terms (Ewald) must match to this.
ENERGY_TERM_RY = 1e-6

#: Density-*dependent* terms -- one-electron, Hartree, XC -- are compared more
#: loosely, and the reason is physics rather than sloppiness: the total energy is
#: variational, so it is second-order accurate in the density error, while the
#: individual terms are first-order. Two calculations whose totals agree to 1e-8
#: legitimately differ in their one-electron term at the 1e-4 level once QE's own
#: conv_thr = 1e-6 is taken into account.
DENSITY_DEPENDENT_TERM_RY = 5e-4

#: Eigenvalues, printed by pw.x with 4 decimals in eV. Two effects put a floor
#: under the agreement, both on QE's side: its runs converge only to 1e-6 Ry, and
#: it interpolates the local potential from a dq = 0.01 table where this code
#: integrates directly, which shifts levels by a few tenths of a meV.
EIGENVALUE_EV = 2e-3
FERMI_EV = 2e-3

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
