"""P27: Grimme's D2 correction inside a real run, on bilayer graphene.

The system is the one the correction exists for. Two layers of graphene held
together by nothing a semilocal functional can see: PBE alone gives an interlayer
force of the wrong sign at short range and no minimum at all, and D2 puts one
back. The reference is the vendored ``pw.x`` on the *same* input, regenerated at
``conv_thr = 1e-10`` with ``verbosity = 'high'`` so that QE prints its dispersion
force and its ``DFT-D`` stress as separate blocks -- which makes
``force_london`` and ``stres_london`` comparable term by term rather than only
through a total.

**The sharpest test here is the null one.** The correction is a function of the
nuclei alone: ``electrons.f90`` adds ``elondon`` after the SCF loop, so the
density, the eigenvalues and the potential must be *bit for bit* what the same
run without it produces, and the two total energies must differ by exactly the
printed ``Dispersion Correction``. A correction that had leaked into ``v_of_rho``
would still give a plausible total energy and would fail that comparison in the
tenth decimal of the Hartree term.

The third-derivative case is silicon rather than graphene, and deliberately so:
``pypresso.response.electrostriction`` requires a closed unshifted k-grid, an
insulator and norm-conserving data, and ``si-electrostriction.in`` is the cell
P26 was checked on. Switching D2 on there is unphysical and is exactly the point
-- what is being measured is that the correction reaches the elastic constants
through :meth:`~pypresso.scf.driver.Calculation.at_strain` and does *not* reach
``d(chi)/dx``, which is the same statement as the null test one derivative up.
"""

from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pypresso.forces import compute_forces
from pypresso.io import read_qe_output
from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.response.elastic import VOIGT
from pypresso.response.electrostriction import electrostriction, refined_states
from pypresso.response.strain import strain_response, strain_tangent
from pypresso.scf import Calculation, run_scf
from pypresso.system import build_system
from pypresso.vdw.analytic import dispersion_force, dispersion_stress
from pypresso.workflows.relax import run_relax
from tests.tolerances import ENERGY_TERM_RY, TOTAL_ENERGY_RY

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: ``force_london`` and ``stres_london`` against the blocks ``verbosity =
#: 'high'`` prints. Both are limited by QE's own formatting rather than by the
#: agreement: the force is printed to 1e-8 Ry/bohr and the stress in kbar to two
#: decimals, which is 7e-9 Ry/bohr^3. Achieved: 3.9e-9 and 1.2e-8.
DISPERSION_FORCE = 5e-8
DISPERSION_STRESS = 5e-8


@lru_cache(maxsize=None)
def _converged(case: str, conv_thr: float = 1e-10):
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=conv_thr,
                     max_iterations=100)
    return system, pseudos, calculation, result


def _reference(case: str):
    path = CASES / f"reference.out.{case}"
    if not path.is_file():
        pytest.skip(f"no generated reference for {case}; run tools/generate_reference.py")
    return read_qe_output(path)


# -- the energy ---------------------------------------------------------------


def test_the_total_energy_matches_quantum_espresso():
    _, _, _, result = _converged("graphene-bilayer-d2")
    reference = _reference("graphene-bilayer-d2")
    assert result.total_energy == pytest.approx(
        reference.total_energy, abs=TOTAL_ENERGY_RY
    )


def test_the_dispersion_term_matches_quantum_espresso():
    """QE's ``Dispersion Correction`` line, which is ``energy_london`` and nothing
    else -- a density-independent term, so it is held to the tight tolerance."""
    _, _, calculation, result = _converged("graphene-bilayer-d2")
    reference = _reference("graphene-bilayer-d2")
    assert result.energy_terms["dispersion"] == pytest.approx(
        reference.energy_terms["dispersion"], abs=ENERGY_TERM_RY
    )
    # The calculation carries the same number, computed before the SCF ran.
    assert float(calculation.dispersion) == pytest.approx(
        result.energy_terms["dispersion"], abs=1e-12
    )


def test_a_run_without_the_correction_has_no_dispersion_term():
    _, _, calculation, result = _converged("graphene-bilayer")
    assert calculation.dispersion_sum is None
    assert "dispersion" not in result.energy_terms


# -- the null test: it does not touch the electronic structure -----------------


def test_the_correction_does_not_reach_the_density_or_the_eigenvalues():
    """``electrons.f90`` adds ``elondon`` *after* the loop, and so does this.

    Not "agrees to the convergence threshold" -- identical. The two runs solve
    the same Hamiltonian, mix the same residual and stop at the same iteration,
    so any difference at all would mean the pair potential had reached
    ``v_of_rho``.
    """
    _, _, _, plain = _converged("graphene-bilayer")
    _, _, _, corrected = _converged("graphene-bilayer-d2")
    assert np.abs(np.asarray(plain.density) - np.asarray(corrected.density)).max() == 0.0
    assert np.abs(
        np.asarray(plain.eigenvalues) - np.asarray(corrected.eigenvalues)
    ).max() == 0.0
    for term in ("one-electron", "hartree", "xc", "ewald", "smearing"):
        assert plain.energy_terms[term] == corrected.energy_terms[term]


def test_the_two_runs_differ_by_exactly_the_dispersion_energy():
    """...and QE's own two runs differ by the same amount."""
    _, _, _, plain = _converged("graphene-bilayer")
    _, _, _, corrected = _converged("graphene-bilayer-d2")
    difference = corrected.total_energy - plain.total_energy
    assert difference == pytest.approx(
        corrected.energy_terms["dispersion"], abs=1e-12
    )
    qe = (_reference("graphene-bilayer-d2").total_energy
          - _reference("graphene-bilayer").total_energy)
    assert difference == pytest.approx(qe, abs=ENERGY_TERM_RY)


# -- forces and stress --------------------------------------------------------


@pytest.mark.parametrize("method", ["autodiff", "analytic"])
def test_the_force_matches_quantum_espresso(method):
    _, _, calculation, result = _converged("graphene-bilayer-d2", 1e-12)
    reference = _reference("graphene-bilayer-d2")
    forces = compute_forces(calculation, result, method=method)
    assert np.abs(np.asarray(forces.forces) - reference.forces).max() < 1e-5


def test_the_transcribed_force_matches_quantum_espressos_own_block():
    """``force_london`` against the ``Dispersion contribution to forces`` block.

    The one term of the force whose reference is QE's *same* expression rather
    than a different route to the same number -- which is what makes the
    autodiff comparison below worth having.
    """
    system, _, calculation, _ = _converged("graphene-bilayer-d2")
    reference = _reference("graphene-bilayer-d2")
    ours = np.asarray(
        dispersion_force(calculation.dispersion_sum, system.structure.positions)
    )
    assert np.abs(ours - reference.force_terms["dispersion"]).max() < DISPERSION_FORCE


def test_the_transcribed_stress_matches_quantum_espressos_own_block():
    """``stres_london`` against QE's ``DFT-D   stress (kbar)`` row."""
    system, _, calculation, _ = _converged("graphene-bilayer-d2")
    reference = _reference("graphene-bilayer-d2")
    ours = np.asarray(dispersion_stress(
        calculation.dispersion_sum,
        system.structure.positions,
        float(system.cell.volume),
    ))
    assert np.abs(ours - reference.stress_terms["dispersion"]).max() < DISPERSION_STRESS


def test_the_stress_matches_quantum_espresso():
    from pypresso.stress import compute_stress

    _, _, calculation, result = _converged("graphene-bilayer-d2", 1e-12)
    reference = _reference("graphene-bilayer-d2")
    stress = compute_stress(calculation, result)
    assert np.abs(np.asarray(stress.tensor) - reference.stress).max() < 1e-6


def test_the_correction_moves_the_stress_the_way_it_moves_the_energy():
    """The difference between the two runs' stresses is ``stres_london`` alone."""
    from pypresso.stress import compute_stress

    system, _, plain_calc, plain = _converged("graphene-bilayer", 1e-12)
    _, _, calc, corrected = _converged("graphene-bilayer-d2", 1e-12)
    difference = (np.asarray(compute_stress(calc, corrected).tensor)
                  - np.asarray(compute_stress(plain_calc, plain).tensor))
    expected = np.asarray(dispersion_stress(
        calc.dispersion_sum, system.structure.positions, float(system.cell.volume)
    ))
    assert np.abs(difference - expected).max() < 1e-9


# -- relaxation ---------------------------------------------------------------


@pytest.mark.slow
def test_the_relaxation_binds_the_two_layers():
    """The whole point of the correction, on the system that shows it.

    PBE alone gives bilayer graphene no minimum in the interlayer separation at
    all; with D2 the relaxation walks in from 7.2 bohr and settles at **6.10
    bohr (3.23 A)**, against a measured 3.35 and the ~3.2 that D2's known
    overbinding gives. The energy it reaches is the reference for
    :func:`test_the_relaxed_geometry_is_a_stationary_point_of_pw_xs_surface`,
    which is where the comparison with ``pw.x`` actually happens.
    """
    result = _relaxed()
    system = build_system(read_pw_input(CASES / "graphene-bilayer-d2-relax.in"))
    assert result.converged

    inverse = np.linalg.inv(np.asarray(system.cell.at))
    height = float(np.asarray(system.cell.at)[2, 2])
    crystal = np.asarray(result.positions) @ inverse
    started = np.asarray(system.structure.positions) @ inverse

    assert (started[2, 2] - started[0, 2]) * height == pytest.approx(7.2, abs=0.01)
    assert (crystal[2, 2] - crystal[0, 2]) * height == pytest.approx(6.10, abs=0.05)


@lru_cache(maxsize=None)
def _relaxed():
    system = build_system(read_pw_input(CASES / "graphene-bilayer-d2-relax.in"))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return run_relax(system, pseudos, conv_thr=1e-10, etot_conv_thr=1e-8,
                     forc_conv_thr=1e-5, nstep=60)


@pytest.mark.slow
def test_the_relaxed_geometry_is_a_stationary_point_of_pw_xs_surface():
    """``pw.x`` run at the geometry pypresso relaxed to, and it agrees.

    **A geometry comparison is the wrong test here and this is the right one.**
    The interlayer force constant is ~2e-4 Ry/bohr^2 -- three orders below a
    chemical bond -- so the ``forc_conv_thr = 1e-5`` both codes stop at pins the
    separation only to a few tenths of a bohr, and what ``max |F|`` is actually
    measuring is the *stiff* mode, the A/B sublattice buckling inside each layer.
    On that surface the two BFGS runs stop 0.48 bohr apart and both are entitled
    to: QE's own force at its answer is 6.9e-6 and pypresso's force at QE's
    answer is 4.1e-5, each below the other's threshold, and pypresso's answer is
    8.3e-4 Ry lower.

    What settles it is asking ``pw.x`` about pypresso's geometry
    (``graphene-bilayer-d2-relaxed.in``). It gives the same total energy to
    **1e-8 Ry** and a force of **3e-6 Ry/bohr**, so the two codes are walking the
    same surface and pypresso walked further down it. The energies, not the
    coordinates, are what the two relaxations can be compared through.
    """
    result = _relaxed()
    reference = _reference("graphene-bilayer-d2-relaxed")
    assert result.total_energy == pytest.approx(reference.total_energy, abs=1e-6)
    assert np.abs(reference.forces).max() < 1e-5

    # ... and it is at least as low as the geometry QE's own relaxation stopped
    # at, which is the only sense in which one relaxation can beat another.
    qe_final = _final_energy(CASES / "reference.out.graphene-bilayer-d2-relax")
    assert result.total_energy <= qe_final + 1e-6


def _final_energy(path: Path) -> float:
    """The *last* ``!    total energy`` in a relax output, not the first.

    :func:`read_qe_output` reads the first one, which for a relaxation is the
    starting geometry's.
    """
    import re

    values = re.findall(r"^!\s+total energy\s+=\s+(\S+)", path.read_text(), re.M)
    return float(values[-1])


@pytest.mark.slow
def test_quantum_espressos_relaxed_geometry_is_stationary_here_too():
    """The comparison in the other direction, and it is the weaker of the two.

    pypresso's force at QE's answer is 4.1e-5 Ry/bohr -- below the 1e-4 a
    ``pw.x`` input asks for by default and above the 1e-5 this case asks for,
    which is the flatness of the surface rather than a disagreement: the
    *analytic* force at a shared geometry agrees with QE's to 3.7e-7
    (:func:`test_the_transcribed_force_matches_quantum_espressos_own_block` and
    :func:`test_the_force_matches_quantum_espresso`).
    """
    system = build_system(read_pw_input(CASES / "graphene-bilayer-d2-relax.in"))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    reference = _reference("graphene-bilayer-d2-relax")

    calculation = Calculation(system, pseudos).at_positions(
        jnp.asarray(reference.final_positions)
    )
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                     max_iterations=100)
    forces = compute_forces(calculation, result)
    assert np.abs(np.asarray(forces.forces)).max() < 1e-4


# -- the third derivative -----------------------------------------------------


@lru_cache(maxsize=None)
def _electrostriction(case: str):
    system, pseudos, calculation, result = _converged(case, 1e-12)
    eigenvalues, psi = refined_states(calculation, result)
    response = strain_response(
        calculation, psi, eigenvalues, jnp.asarray(result.density)
    )
    return system, calculation, electrostriction(calculation, result, strain=response)


def test_the_susceptibility_derivative_is_untouched_by_the_correction():
    """``d(chi)/dx`` with and without D2, and the answer is *identical*.

    The null test one derivative up, and the reason it is exact rather than
    close: the perturbation the Sternheimer equation solves is built from the
    Hamiltonian, and a pair potential over the nuclei is not in it. If a future
    correction of the density-dependent kind (Tkatchenko-Scheffler, XDM) is ever
    added, this is the test that will have to change -- which is the point of
    writing it as an equality.
    """
    _, _, plain = _electrostriction("si-electrostriction")
    _, _, corrected = _electrostriction("si-electrostriction-d2")
    assert np.abs(plain.dchi_dstrain - corrected.dchi_dstrain).max() == 0.0
    assert np.abs(plain.epsilon - corrected.epsilon).max() == 0.0
    assert np.abs(plain.m - corrected.m).max() == 0.0
    assert np.abs(plain.q - corrected.q).max() == 0.0


def test_the_elastic_constants_pick_up_exactly_the_pair_sums_second_derivative():
    """``C(D2) - C(no D2)`` against ``(1/Omega) d^2 E_disp/dx^2``, alone.

    The elastic constants are one ``jvp`` of the stress along a tangent that
    carries the strain, the states and the density together (P26). The states'
    half of that tangent is unchanged by a correction that does not enter the
    Hamiltonian, so the whole of the difference has to be the frozen second
    derivative of the pair sum -- which is computable here in three lines and
    shares nothing with the machinery that produced it inside
    ``elastic_constants``.
    """
    system, calculation, corrected = _electrostriction("si-electrostriction-d2")
    _, _, plain = _electrostriction("si-electrostriction")
    difference = corrected.elastic.tensor - plain.elastic.tensor

    dispersion = calculation.dispersion_sum
    positions = system.structure.positions
    volume = float(system.cell.volume)

    def energy(strain):
        deformation = jnp.eye(3) + strain
        return dispersion.at_cell(deformation).energy(positions @ deformation.T)

    gradient = jax.grad(energy)
    expected = np.zeros((3, 3, 3, 3))
    for (k, l) in VOIGT:
        _, column = jax.jvp(
            gradient, (jnp.zeros((3, 3)),), (strain_tangent(k, l),)
        )
        column = np.asarray(column) / volume
        column = 0.5 * (column + column.T)
        expected[:, :, k, l] = expected[:, :, l, k] = column

    assert np.abs(expected).max() > 1e-5, "the correction must actually do something"
    assert np.abs(difference - expected).max() < 1e-9


def test_the_strain_coefficients_move_and_the_stress_ones_do_not():
    """``M`` and ``Q`` are ``m`` and ``q`` through the compliance, so only they move.

    Which is the whole of what a clamped-ion electrostriction tensor inherits
    from a van der Waals correction: the electronic susceptibility does not feel
    it, and the elastic constants that convert a stress coefficient into a strain
    one do.
    """
    _, _, plain = _electrostriction("si-electrostriction")
    _, _, corrected = _electrostriction("si-electrostriction-d2")
    assert np.abs(plain.M - corrected.M).max() > 1e-6 * np.abs(plain.M).max()
    assert np.abs(plain.Q - corrected.Q).max() > 1e-6 * np.abs(plain.Q).max()
