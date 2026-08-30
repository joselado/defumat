"""P15 check: forces against Quantum ESPRESSO, both ways of computing them.

Five cases, each adding one thing to the one before, all on the same displaced
two-atom silicon cell except the last:

* ``si2-nc-force``   -- norm-conserving LDA. Pure Hellmann-Feynman: the local
  potential and the projectors move, and nothing else does.
* ``si2-us-force``   -- ultrasoft. Adds the augmentation charge's own
  derivative (``addusforce``) and the Pulay term from the orthonormality
  constraint being position-dependent, which is what ``deff = deeq - eps qq``
  is. Also the first case with a core charge, hence with ``force_cc``.
* ``si2-paw-force``  -- PAW, whose one-centre terms reach the force the same way
  they reach the energy: through ``deeq`` and ``becsum``, with no force routine
  of their own.
* ``si2-us-pbe-force`` -- the same with PBE, so the gradient correction is
  differentiated along with everything else.
* ``o2-lsda-force``  -- an oxygen molecule, spin-polarized and magnetic. The
  case that catches the ``(up, down)`` versus ``(rho, m)`` trap in ``force_lc``.

Each is checked three ways: the autodiff force against QE, the analytic force
against QE, and the two against each other. The last comparison is the sharpest
of the three, because the two share no machinery -- one differentiates the
energy, the other evaluates six hand-derived expressions -- and because what
separates them is a known quantity: ``force_corr``, the term that exists only
because the density stopped short of self-consistency.

The references are generated with the vendored ``pw.x`` at ``conv_thr = 1e-10``
and with ``verbosity = 'high'``, which makes QE print the force term by term.
That breakdown is compared too: a total force can be right with two terms wrong
in opposite directions, and this is the one project in which that has already
happened once (the augmentation force's sign).
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from pypresso.forces import compute_forces, frozen_energy, state_from_result
from pypresso.io import read_qe_output
from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf import Calculation, run_scf
from pypresso.system import build_system
from pypresso.system.symmetry import atom_mapping, symmetrize_vector
from tests.tolerances import FORCE_RY_BOHR, TOTAL_ENERGY_RY

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"

ALL = [
    "si2-nc-force",
    "si2-us-force",
    "si2-paw-force",
    "si2-us-pbe-force",
    "o2-lsda-force",
]

#: How far the two force methods may differ. They are the same quantity computed
#: two ways *plus* ``force_corr``, which the analytic one has and the autodiff
#: one cannot: at ``conv_thr = 1e-10`` that term is ~1e-5 Ry/bohr on the
#: molecule and ~1e-6 on the crystals.
METHOD_AGREEMENT = 5e-5


@lru_cache(maxsize=None)
def _converged(case: str, pseudo_dir: Path):
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-10,
                     max_iterations=80)
    return system, calculation, result


def _reference(case: str):
    path = CASES / f"reference.out.{case}"
    if not path.is_file():
        pytest.skip(f"no generated reference for {case}; run tools/generate_reference.py")
    return read_qe_output(path)


@pytest.mark.parametrize("case", ALL)
def test_the_frozen_functional_reproduces_the_scf_total(case, pseudo_dir):
    """What is differentiated has to be the SCF's own energy, on every dataset.

    The gate on the whole autodiff path, and it is per-dataset rather than
    per-code-path: ``frozen_energy`` reassembles QE's decomposition out of
    different pieces (the bare ``D_ij`` rather than the self-consistent one, the
    augmentation charge inside the density rather than inside ``deeq``, PAW's
    one-centre terms through ``becsum``), and each pseudopotential kind
    reassembles it differently. If the two totals disagree, the gradient is the
    force of some other calculation and every comparison below it is
    coincidence.

    ``test_force_machinery.py`` makes this claim once, on norm-conserving
    silicon. This is the same claim across the five datasets, and it arrived
    here from notebook 09, whose first code cell was a four-case table of
    exactly this -- the test suite's job being done in a tutorial (P49).
    """
    system, calculation, result = _converged(case, pseudo_dir)
    energy = frozen_energy(calculation, system.structure.positions,
                           state_from_result(result))
    assert float(energy) == pytest.approx(result.total_energy, abs=1e-9)


@pytest.mark.parametrize("case", ALL)
@pytest.mark.parametrize("method", ["autodiff", "analytic"])
def test_forces_match_quantum_espresso(case, method, pseudo_dir):
    _, calculation, result = _converged(case, pseudo_dir)
    reference = _reference(case)

    assert result.total_energy == pytest.approx(
        reference.total_energy, abs=TOTAL_ENERGY_RY
    )
    forces = compute_forces(calculation, result, method=method)
    assert np.abs(forces.forces - reference.forces).max() < FORCE_RY_BOHR


@pytest.mark.parametrize("case", ALL)
def test_the_two_methods_agree(case, pseudo_dir):
    """Differentiating the energy and transcribing QE's algebra give one answer."""
    _, calculation, result = _converged(case, pseudo_dir)
    autodiff = compute_forces(calculation, result, method="autodiff")
    analytic = compute_forces(calculation, result, method="analytic")
    assert np.abs(autodiff.forces - analytic.forces).max() < METHOD_AGREEMENT


@pytest.mark.parametrize("case", ALL)
def test_the_two_methods_differ_by_the_scf_correction(case, pseudo_dir):
    """...and what separates them is the term one of them does not have.

    ``force_corr`` is the correction for a density that has not quite reached
    the fixed point, so it is exactly the piece a force obtained by
    differentiating the energy *at* the fixed point is missing. Their difference
    should be that term and nothing else.
    """
    _, calculation, result = _converged(case, pseudo_dir)
    autodiff = compute_forces(calculation, result, method="autodiff")
    analytic = compute_forces(calculation, result, method="analytic")

    correction = analytic.terms["scf_correction"]
    residue = np.abs(analytic.forces - autodiff.forces).max()
    assert residue < max(3.0 * np.abs(correction).max(), 2e-6)


@pytest.mark.parametrize("case", ALL)
def test_each_term_matches_quantum_espresso(case, pseudo_dir):
    """The breakdown, not only the total (``verbosity = 'high'``).

    QE symmetrises the nonlocal term on its own -- ``force_us`` ends with
    ``symvector`` -- and folds ``addusforce`` into it, so the comparison
    symmetrises this side's terms and adds the augmentation to the nonlocal one
    before comparing.
    """
    system, calculation, result = _converged(case, pseudo_dir)
    reference = _reference(case)
    if not reference.force_terms:
        pytest.skip(f"{case}: reference was not run with verbosity = 'high'")

    forces = compute_forces(calculation, result, method="analytic")
    mapping = atom_mapping(system.cell, system.structure, calculation.symmetries)

    terms = dict(forces.terms)
    terms["nonlocal"] = terms["nonlocal"] + terms.pop("augmentation", 0.0)
    for name, qe_name in [("ewald", "ionic"), ("local", "local"),
                          ("core", "core"), ("nonlocal", "nonlocal")]:
        ours = np.asarray(
            symmetrize_vector(
                np.asarray(terms[name]), system.cell, calculation.symmetries, mapping
            )
        )
        assert np.abs(ours - reference.force_terms[qe_name]).max() < FORCE_RY_BOHR, name


def test_the_total_force_on_the_crystal_vanishes(pseudo_dir):
    """Nothing in the energy knows that translating the crystal changes nothing.

    What survives is the discretisation error of the FFT grid and the cutoff,
    which QE removes by subtracting the average (``sumfor``) -- and which is a
    useful number before it is removed, so the result keeps it.
    """
    _, calculation, result = _converged("si2-nc-force", pseudo_dir)
    forces = compute_forces(calculation, result)
    assert np.abs(forces.forces.sum(axis=0)).max() < 1e-12
    assert np.abs(forces.total_before_correction).max() < 1e-4


def test_the_force_is_the_derivative_of_the_energy(pseudo_dir):
    """Rule D5, in its most direct form: central differences of the SCF energy.

    This is the check that does not trust any of the machinery -- not the
    Lagrangian, not QE's algebra, not the reference output. It converges the SCF
    at three geometries per coordinate and differences the total energies.

    It has to run with ``nosym``. Moving one atom along one axis breaks the
    symmetry of the starting structure, and the symmetry group is deliberately
    held fixed while the atoms move (see
    :meth:`~pypresso.scf.driver.Calculation.at_positions`), so a symmetrised run
    would compare the energy of the displaced structure against a density
    symmetrised with operations it no longer has. The agreement is limited by
    the step size: at ``h = 2e-3`` bohr the truncation error of a central
    difference is ~1e-6 Ry/bohr, which is what is asked for here.
    """
    import dataclasses

    system = build_system(read_pw_input(CASES / "si2-nc-force.in"))
    system = dataclasses.replace(system, nosym=True)
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)

    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12)
    forces = compute_forces(calculation, result)

    def energy_at(positions):
        moved = calculation.at_positions(np.asarray(positions))
        return run_scf(
            moved.system, pseudos, calculation=moved, conv_thr=1e-12
        ).total_energy

    step = 2.0e-3
    origin = np.asarray(system.structure.positions)
    for atom, direction in [(0, 0), (0, 1)]:
        plus, minus = origin.copy(), origin.copy()
        plus[atom, direction] += step
        minus[atom, direction] -= step
        difference = -(energy_at(plus) - energy_at(minus)) / (2.0 * step)
        assert difference == pytest.approx(forces.forces[atom, direction], abs=5e-6)


def test_a_shifted_grid_agrees_with_qe_when_neither_symmetrises(pseudo_dir):
    """The same k-sample, no symmetrisation, two codes: the same number.

    A *shifted* Monkhorst-Pack grid is not invariant under every operation the
    crystal has -- for this displaced cell, four of the eight map the grid onto
    itself and four do not -- so a density built from it does not have the
    symmetry the crystal does. Both codes symmetrise it anyway, with sets of
    operations that are not the same (PLAN.md, P6), and their totals differ by
    ~1e-4 Ry as a result.

    With ``nosym`` there is nothing to choose: 64 k-points, no symmetrisation,
    and the two agree to the last digit. That is what this pins -- the SCF, the
    basis and the k-point generation are identical, and the spread above is a
    decision about symmetry rather than an error in the physics. Forces are
    checked here too, since they are the quantity that first exposed it.
    """
    system = build_system(read_pw_input(CASES / "si2-nc-shifted-nosym.in"))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    reference = _reference("si2-nc-shifted-nosym")

    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-10)
    assert system.kpoints.nk == 64
    assert result.total_energy == pytest.approx(reference.total_energy, abs=1e-8)

    forces = compute_forces(calculation, result)
    assert np.abs(forces.forces - reference.forces).max() < FORCE_RY_BOHR
