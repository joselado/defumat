"""Constants must match the reference source exactly.

A silently different Bohr radius or Ry->eV factor shows up much later as a
"nearly right" total energy, so this test reads the SI values straight out of
QE's ``Modules/constants.f90`` and compares them to ours, rather than restating
numbers we typed. It skips when the reference tree is absent.
"""

import re

import pytest

from pypresso import units
from tests.conftest import QE_ROOT

pytestmark = pytest.mark.unit

# name in constants.f90 -> name in pypresso.units
_SI_CONSTANTS = {
    "H_PLANCK_SI": "H_PLANCK_SI",
    "K_BOLTZMANN_SI": "K_BOLTZMANN_SI",
    "ELECTRON_SI": "ELECTRON_SI",
    "ELECTRONVOLT_SI": "ELECTRONVOLT_SI",
    "ELECTRONMASS_SI": "ELECTRONMASS_SI",
    "HARTREE_SI": "HARTREE_SI",
    "BOHR_RADIUS_SI": "BOHR_RADIUS_SI",
    "AMU_SI": "AMU_SI",
    "C_SI": "C_SI",
}


def _fortran_constants() -> dict[str, float]:
    path = QE_ROOT / "Modules" / "constants.f90"
    if not path.is_file():
        pytest.skip(f"QE reference tree not present at {path}")
    text = path.read_text()
    values = {}
    for name in _SI_CONSTANTS:
        m = re.search(rf"{name}\s*=\s*([-+]?[\d.]+(?:[EeDd][-+]?\d+)?)_DP", text)
        assert m is not None, f"{name} not found in constants.f90"
        values[name] = float(m.group(1).replace("D", "E").replace("d", "e"))
    return values


def test_si_constants_match_quantum_espresso():
    fortran = _fortran_constants()
    for qe_name, our_name in _SI_CONSTANTS.items():
        assert getattr(units, our_name) == fortran[qe_name], qe_name


def test_derived_conversions():
    # QE's RYTOEV and BOHR_RADIUS_ANGS, to the digits it prints them with.
    assert units.RY_TO_EV == pytest.approx(13.6056931230, abs=1e-9)
    assert units.BOHR_TO_ANGSTROM == pytest.approx(0.529177210903, abs=1e-12)
    assert units.ANGSTROM_TO_BOHR * units.BOHR_TO_ANGSTROM == pytest.approx(1.0)
    # RY_KBAR, the factor QE uses to print stress in kbar.
    assert units.RY_TO_KBAR == pytest.approx(147105.07846, abs=1e-4)


def test_e2_is_rydberg_convention():
    """e^2 = 2 in Ry a.u. -- the classic factor-of-two trap when porting."""
    assert units.E2 == 2.0
