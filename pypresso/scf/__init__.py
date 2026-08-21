"""The self-consistent field: potential, density, occupations, mixing, driver.

Two ways to reach the fixed point, behind ``run_scf``'s ``scf_solver`` (rule R4):
the mixing loop in :mod:`~pypresso.scf.driver`, which is the default, and the
residual of :mod:`~pypresso.scf.residual` solved by
:mod:`~pypresso.scf.solvers`. See ``PLAN.md`` P22 for which to reach for.
"""

from pypresso.scf.density import band_density, sum_band
from pypresso.scf.driver import Calculation, SCFResult, default_nbnd, run_scf
from pypresso.scf.ewald import ewald_energy
from pypresso.scf.mixing import get_mixer
from pypresso.scf.occupations import fermi_level, fixed_occupations, smeared_occupations
from pypresso.scf.potential import Potential, v_of_rho
from pypresso.scf.residual import ScfResidual, make_residual
from pypresso.scf.solvers import SCF_SOLVERS, get_scf_solver, newton_krylov

__all__ = [
    "Calculation",
    "Potential",
    "SCF_SOLVERS",
    "ScfResidual",
    "SCFResult",
    "band_density",
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
