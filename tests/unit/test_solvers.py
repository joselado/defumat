"""The iterative eigensolver against the exact one.

The dense solver forms the matrix and diagonalises it, so it is right by
construction on any system small enough to try. That makes it the reference for
Davidson, which is right only if it converged -- and the way an iterative solver
fails is by *quietly* not converging, returning plausible numbers that are wrong
in the fourth decimal.

Eigenvalues are compared, not eigenvectors: silicon's bands are degenerate at
the k-point used here, and any rotation within a degenerate subspace is an
equally valid answer.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from pypresso.basis.builder import build_basis
from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf.driver import Calculation, run_scf
from pypresso.scf.potential import v_of_rho
from pypresso.solvers import (
    DEFAULT_EIGENSOLVER,
    EIGENSOLVERS,
    davidson_eigensolver_all,
    dense_eigensolver_all,
    get_eigensolver,
)
from pypresso.system import build_system

pytestmark = pytest.mark.unit

BENCHMARK = Path(__file__).resolve().parents[2] / "benchmarks" / "si-1k.in"
NBND = 4


@pytest.fixture(scope="module")
def silicon(pseudo_dir):
    system = build_system(read_pw_input(BENCHMARK))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    potential = v_of_rho(calculation.starting_density(), calculation.basis.dense, system.cell)
    return system, pseudos, calculation.hamiltonian(potential.v_scf)


def test_davidson_reproduces_the_exact_eigenvalues(silicon):
    """Asked for machine precision, it delivers machine precision."""
    _, _, hamiltonian = silicon
    exact, _ = dense_eigensolver_all(hamiltonian, NBND)
    iterative, _ = davidson_eigensolver_all(
        hamiltonian, NBND, None, ethr=1e-13, residual_threshold=1e-8, max_iterations=60
    )
    assert np.asarray(iterative) == pytest.approx(np.asarray(exact), abs=1e-10)


def test_a_looser_threshold_costs_fewer_steps_and_less_accuracy(silicon):
    """The whole point of scheduling ``ethr``: pay only for what is needed."""
    _, _, hamiltonian = silicon
    exact, _ = dense_eigensolver_all(hamiltonian, NBND)

    errors = {}
    for ethr in (1e-2, 1e-6, 1e-13):
        values, _ = davidson_eigensolver_all(hamiltonian, NBND, None, ethr=ethr,
                                             max_iterations=60)
        errors[ethr] = np.abs(np.asarray(values) - np.asarray(exact)).max()

    assert errors[1e-2] > errors[1e-6] > errors[1e-13]
    assert errors[1e-13] < 1e-9


def test_davidson_returns_eigenvectors_of_the_hamiltonian(silicon):
    """``(H - e) psi`` small is the property that makes the eigenvalue right."""
    _, _, hamiltonian = silicon
    values, vectors = davidson_eigensolver_all(hamiltonian, NBND, None, max_iterations=60)

    matrix = np.asarray(hamiltonian.matrix(0))
    psi = np.asarray(vectors)[0]
    residual = psi @ matrix.T - np.asarray(values)[0][:, None] * psi
    assert np.linalg.norm(residual, axis=1).max() < 1e-6


def test_seeding_with_the_answer_converges_immediately(silicon):
    """The SCF's reason for carrying wavefunctions between iterations."""
    _, _, hamiltonian = silicon
    exact, vectors = dense_eigensolver_all(hamiltonian, NBND)
    seeded, _ = davidson_eigensolver_all(hamiltonian, NBND, vectors, max_iterations=2)
    assert np.asarray(seeded) == pytest.approx(np.asarray(exact), abs=1e-10)


def test_the_two_solvers_give_the_same_converged_scf(pseudo_dir, silicon):
    """End to end: swapping the solver must not move the total energy.

    The two bounds are deliberately different. The total energy is variational
    in the density, so it must agree tightly whatever the solver did. The
    eigenvalues are only ever converged to the threshold the SCF asked for --
    ``ethr`` is scheduled against the error in the density, exactly as QE
    schedules it, so at the end they carry an error of order the last ``ethr``
    amplified by how weakly a change in an eigenvalue bounds its error. That is
    a property of QE's method, not a defect of this transcription, and it is
    four orders of magnitude inside the tolerance the QE comparison uses.
    """
    system, pseudos, _ = silicon
    results = {
        name: run_scf(
            system,
            pseudos,
            calculation=Calculation(system, pseudos, diagonalization=name),
            conv_thr=1e-10,
        )
        for name in ("dense", "davidson")
    }
    assert results["davidson"].total_energy == pytest.approx(
        results["dense"].total_energy, abs=1e-9
    )
    assert np.asarray(results["davidson"].eigenvalues) == pytest.approx(
        np.asarray(results["dense"].eigenvalues), abs=1e-5
    )


def test_a_separate_smooth_grid_does_not_break_convergence(pseudo_dir, silicon):
    """``ecutrho > 4 ecutwfc`` gives the smooth grid its own, smaller FFT box.

    Every reference case in the suite has ``dual = 4``, where the smooth and
    dense grids are the same object, so nothing else exercises the case where
    they are not. It is worth a test because the failure mode is silent: an
    index map built for the smaller box addresses the larger one perfectly
    legally, and what comes back is simply the wrong numbers.
    """
    system, pseudos, _ = silicon
    coarse = dataclasses.replace(system, ecutrho=8.0 * system.ecutwfc)

    basis = build_basis(coarse)
    assert basis.smooth.grid != basis.dense.grid, "this case should have two grids"

    result = run_scf(coarse, pseudos, conv_thr=1e-10)
    assert result.converged
    assert result.accuracy < 1e-10
    # A finer density grid is a different (slightly better) calculation, not a
    # different answer: the two must agree to the size of that improvement.
    reference = run_scf(system, pseudos, conv_thr=1e-10)
    assert result.total_energy == pytest.approx(reference.total_energy, abs=1e-4)


def test_the_registry_covers_every_solver():
    assert set(EIGENSOLVERS) >= {"dense", "davidson"}
    assert get_eigensolver(None) is EIGENSOLVERS[DEFAULT_EIGENSOLVER]
    assert get_eigensolver("DAVIDSON") is EIGENSOLVERS["davidson"]
    with pytest.raises(ValueError, match="unknown diagonalization"):
        get_eigensolver("no-such-solver")
