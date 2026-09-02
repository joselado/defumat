"""The self-consistent field: potential, density, occupations, mixing, driver.

Two ways to reach the fixed point, behind ``run_scf``'s ``scf_solver`` (rule R4):
the mixing loop in :mod:`~defumat.scf.driver`, which is the default, and the
residual of :mod:`~defumat.scf.residual` solved by
:mod:`~defumat.scf.solvers`. See ``PLAN.md`` P22 for which to reach for.

:mod:`~defumat.scf.continuation` starts one run where another stopped, across a
change of spin regime -- an unpolarized density as the starting point of a
collinear run, a collinear one of a noncollinear run, and spin-orbit coupling
switched on and off without going back to the atoms (``PLAN.md`` P23).
"""

from defumat.scf.continuation import ContinuedState, continued_state
from defumat.scf.density import band_density, sum_band
from defumat.scf.driver import Calculation, SCFResult, default_nbnd, run_scf
from defumat.scf.ewald import ewald_energy
from defumat.scf.mixing import get_mixer
from defumat.scf.occupations import fermi_level, fixed_occupations, smeared_occupations
from defumat.scf.potential import Potential, v_of_rho
from defumat.scf.residual import ScfResidual, make_residual
from defumat.scf.solvers import SCF_SOLVERS, get_scf_solver, newton_krylov

__all__ = [
    "Calculation",
    "ContinuedState",
    "Potential",
    "SCF_SOLVERS",
    "ScfResidual",
    "SCFResult",
    "band_density",
    "continued_state",
    "default_nbnd",
    "ewald_energy",
    "fermi_level",
    "fixed_occupations",
    "get_mixer",
    "get_scf_solver",
    "make_residual",
    "newton_krylov",
    "run_scf",
    "smeared_occupations",
    "sum_band",
    "v_of_rho",
]
