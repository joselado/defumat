"""The gradient correction of a **vector** magnetization, against ``pw.x``.

The noncollinear GGA branch (``gradcorr``'s ``nspin == 4 .AND. domag`` case,
:func:`defumat.scf.potential._noncollinear_gradient_correction`) had no measured
number of any kind anywhere in this project. Its *guard* was measured -- a bare
``|m|`` differentiated through its own nodes gave nan on 243 grid components and
``safe_modulus`` made them finite (P77a) -- but whether the branch is right was
never checked against anything: every committed PBE reference here is collinear
or nonmagnetic, and every noncollinear reference is LDA.

**It is right, and the check is a derivative.** bcc iron with its moment in the
plane, ultrasoft, PBE, ``nosym``, a shifted 4x4x4 grid
(``tests/data/qe/fe-noncolin-pbe-stress.in``):

    quantity                     defumat          pw.x 7.4.1     difference
    total energy (Ry)         -55.78872994     -55.78872993      6.7e-9
    magnetization (mu_B)        1.95101          1.95
    absolute magnetization      2.14008          2.14
    stress, diagonal          1.038232e-3       1.03807e-3       1.6e-7 Ry/bohr^3
    pressure (kbar)             152.73           152.71          0.019

The **stress** is the point: with one atom in a bcc cell the force is zero by
symmetry, so the stress is the only derivative this geometry has, and the energy
being right says nothing about it -- being stationary hides an error in the
gradient, which is how a whole supercell family and a dropped gamma-storage term
were both missed (``CLAUDE.md``'s trap list). 1.6e-7 Ry/bohr^3 is the same level
the *collinear* ultrasoft cases reach on the same quantity: 2.7e-7 for
``si2-us-pbe-stress`` (PBE, ``stres_gradcorr``) and 2.4e-7 for ``pw_lsda/lsda.in``
(``nspin = 2``, ultrasoft nickel). So the vector branch is no worse than the
scalar one it generalises.

**Which branch of the two this exercises.** ``compute_ux`` takes a fixed
quantization axis whenever the starting moments are all parallel to one
direction, and ``compute_rho`` then resolves the density as
``(n +- sign(m.ux)|m|)/2`` -- *signed*, so "up" stays up across a node where
``m`` passes through zero. One atom is trivially parallel to itself, so this is
the **signed** branch, and both codes agree on the axis: ``pw.x`` prints "Fixed
quantization axis for GGA: 1.000000 0.000000 0.000000" and
:func:`~defumat.scf.potential.fixed_quantization_axis` returns the same.

**The unsigned branch is still unmeasured, and the obstacle is not this code.**
Plain ``|m|`` with a real kink at every node is what a genuinely canted cell
takes, and two attempts on hydrogen both failed on the ``pw.x`` side: four
moments 90 degrees apart limit-cycles at an accuracy of 5e-6 Ry under PBE (100
iterations at ``mixing_beta = 0.3``, 300 at 0.1) and two moments at 5e-8, where
the same cells reach 1e-11 under LDA in 62 iterations. A cell whose canted PBE
state converges in ``pw.x`` is what that half needs.
"""

from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest

from defumat import Calculator

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
CASE = "fe-noncolin-pbe-stress"

#: The measured figures, not the suite's generic bounds. ``STRESS_RY_BOHR3`` is
#: 1e-4 and would pass on a stress three orders wrong.
ENERGY_RY = 5.0e-8
STRESS_RY_BOHR3 = 5.0e-7
PRESSURE_KBAR = 0.1


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """The SCF *and* its strain gradient compile here; XLA keeps both for the
    life of the process. The converged result stays cached."""
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _calculator(pseudo_dir):
    return Calculator.from_file(CASES / f"{CASE}.in", pseudo_dir, announce=False)


@lru_cache(maxsize=2)
def _reference():
    from defumat.io import read_qe_output

    return read_qe_output(CASES / f"reference.out.{CASE}")


def test_both_codes_take_the_same_fixed_quantization_axis(pseudo_dir):
    """``compute_ux``, which decides *which* GGA branch runs.

    If the two codes disagreed here they would be evaluating different
    functionals, and the disagreement would show up as an energy difference with
    no obvious cause. ``pw.x`` prints the axis; this asserts the same vector.
    """
    calculation = _calculator(pseudo_dir).calculation
    assert calculation.nspin_mag == 4
    axis = np.asarray(calculation.quantization_axis)
    assert axis == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    assert "Fixed quantization axis for GGA" in (
        CASES / f"reference.out.{CASE}"
    ).read_text()


def test_the_noncollinear_gga_energy_and_moment_match_pw_x(pseudo_dir):
    """The ground state first, so a stress disagreement cannot be blamed on it."""
    scf = _calculator(pseudo_dir).get_scf(max_iterations=200)
    reference = _reference()

    assert scf.converged
    assert scf.total_energy == pytest.approx(reference.total_energy, abs=ENERGY_RY)
    # QE prints the vector to two decimals, which is all this can be held to.
    moment = np.asarray(scf.magnetization_vector)
    assert moment[0] == pytest.approx(1.95, abs=5e-3)
    assert abs(moment[1]) < 5e-3 and abs(moment[2]) < 5e-3
    assert scf.absolute_magnetization == pytest.approx(2.14, abs=5e-3)


def test_the_noncollinear_gga_stress_matches_pw_x(pseudo_dir):
    """The derivative, which is what the energy above cannot vouch for.

    Isotropic on a bcc cell, so the off-diagonal entries being round-off is a
    second check that costs nothing -- and one the *energy* comparison has no
    counterpart for.
    """
    calculator = _calculator(pseudo_dir)
    calculator.get_scf(max_iterations=200)
    stress = calculator.get_stress()
    tensor = np.asarray(stress.tensor)
    reference = np.asarray(_reference().stress)

    assert tensor == pytest.approx(reference, abs=STRESS_RY_BOHR3)
    assert stress.pressure_kbar == pytest.approx(
        _reference().pressure, abs=PRESSURE_KBAR
    )
    # Isotropy, which this cell's symmetry requires and nothing imposed:
    # ``nosym`` is set, so the tensor was not symmetrised.
    assert calculator.system.nosym
    off = tensor - np.diag(np.diag(tensor))
    assert np.abs(off).max() < 1e-8
    assert np.diag(tensor).std() < 1e-9
