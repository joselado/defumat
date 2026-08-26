"""The iterative eigensolver against an exact answer.

Davidson is right only if it converged, and the way an iterative solver fails is
by *quietly* not converging: plausible numbers, wrong in the fourth decimal. On
a cell of a couple of hundred plane waves the question is settled by forming
``H`` and handing it to ``eigh``, which is right by construction -- so that is
what these tests do (``tests/exact_reference.py``). It is a test fixture and not
a solver the package offers; see ``pypresso/solvers/__init__.py`` for why.

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
    get_eigensolver,
)
from pypresso.system import build_system
from tests.exact_reference import exact_eigenpairs_all

pytestmark = pytest.mark.unit

BENCHMARK = Path(__file__).resolve().parents[2] / "benchmarks" / "si-1k.in"
NBND = 4


@pytest.fixture(scope="module")
def silicon(pseudo_dir):
    system = build_system(read_pw_input(BENCHMARK))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    potential = v_of_rho(calculation.starting_density(), calculation.basis.dense, system.cell)
    # one Hamiltonian per spin channel; these tests are unpolarized
    return system, pseudos, calculation.hamiltonian(potential.v_scf)[0]


def test_davidson_reproduces_the_exact_eigenvalues(silicon):
    """Asked for machine precision, it delivers machine precision."""
    _, _, hamiltonian = silicon
    exact, _ = exact_eigenpairs_all(hamiltonian, NBND)
    iterative, _ = davidson_eigensolver_all(
        hamiltonian, NBND, None, ethr=1e-13, residual_threshold=1e-8, max_iterations=60
    )
    assert np.asarray(iterative) == pytest.approx(np.asarray(exact), abs=1e-10)


def test_a_looser_threshold_costs_fewer_steps_and_less_accuracy(silicon):
    """The whole point of scheduling ``ethr``: pay only for what is needed."""
    _, _, hamiltonian = silicon
    exact, _ = exact_eigenpairs_all(hamiltonian, NBND)

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
    exact, vectors = exact_eigenpairs_all(hamiltonian, NBND)
    seeded, _ = davidson_eigensolver_all(hamiltonian, NBND, vectors, max_iterations=2)
    assert np.asarray(seeded) == pytest.approx(np.asarray(exact), abs=1e-10)


def test_the_converged_scf_sits_on_the_exact_eigenvalues(pseudo_dir, silicon):
    """End to end: what the SCF converged to is what the Hamiltonian holds.

    The two bounds are deliberately different. Diagonalising the *converged*
    potential exactly must reproduce the SCF's own eigenvalues -- but only to
    the threshold the SCF asked for, since ``ethr`` is scheduled against the
    error in the density, exactly as QE schedules it, and a change in an
    eigenvalue bounds its error only weakly. That is a property of QE's method,
    not a defect of this transcription, and it is four orders of magnitude
    inside the tolerance the QE comparison uses. The total energy, being
    variational in the density, has no such excuse and is held tightly.
    """
    system, pseudos, _ = silicon
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-10)

    potential = v_of_rho(result.density, calculation.basis.dense, system.cell)
    hamiltonian = calculation.hamiltonian(potential.v_scf)[0]
    exact, _ = exact_eigenpairs_all(hamiltonian, np.asarray(result.eigenvalues).shape[-1])

    assert np.asarray(result.eigenvalues) == pytest.approx(np.asarray(exact), abs=1e-5)
    # ...and the band energy those eigenvalues carry, which is what the total
    # energy is built from, to far better than that.
    assert float(np.asarray(result.eigenvalues).sum()) == pytest.approx(
        float(np.asarray(exact).sum()), abs=1e-4
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
    assert set(EIGENSOLVERS) == {"davidson", "david"}
    # There is deliberately no dense/exact entry: O(npw^2) memory is not
    # something a calculation should be able to select by name.
    assert "dense" not in EIGENSOLVERS and "exact" not in EIGENSOLVERS
    assert get_eigensolver(None) is EIGENSOLVERS[DEFAULT_EIGENSOLVER]
    assert get_eigensolver("DAVIDSON") is EIGENSOLVERS["davidson"]
    with pytest.raises(ValueError, match="unknown diagonalization"):
        get_eigensolver("no-such-solver")


def test_a_cholesky_that_returns_nan_is_rescued_outside_the_k_batch(silicon,
                                                                    monkeypatch):
    """The 64-atom ``NaN``, end to end, with the guard where batching allows it.

    ``generalised_eigh``'s conditional cannot live inside the solve any more:
    one level down it is inside ``map_k``'s ``vmap``, where a batched predicate
    lowers to ``select_n`` and both branches run on every step (2.85x of the
    subspace solve, measured on ``si10-nc``). So the batched solve takes the
    Cholesky route unconditionally and
    :func:`~pypresso.solvers.davidson.davidson_eigensolver_all` retries the
    whole k-set with canonical orthogonalisation when the eigenvalues come back
    non-finite -- which is one scalar predicate, outside the batch, and a real
    branch again.

    Forcing the failure is the only way to test the retry: the overlap goes
    indefinite by round-off on cells far larger than anything a unit test may
    run, and which side of zero it lands on is a coin flip (see
    ``tests/unit/test_subspace_robustness.py``). Replacing the fast route with
    one that returns ``NaN`` exercises the identical path deterministically.
    """
    from pypresso.solvers import davidson, subspace

    _, _, hamiltonian = silicon
    exact, _ = exact_eigenpairs_all(hamiltonian, NBND)

    real_route = subspace._cholesky_route

    def nan_route(h, s):
        values, vectors = real_route(h, s)
        return values * np.nan, vectors * np.nan

    monkeypatch.setattr(subspace, "_cholesky_route", nan_route)
    davidson.davidson_eigensolver_all.clear_cache()
    try:
        values, _ = davidson_eigensolver_all(hamiltonian, NBND, None, ethr=1e-13,
                                             max_iterations=60)
        values = np.asarray(values)
        assert np.isfinite(values).all(), "the retry did not fire"
        assert values == pytest.approx(np.asarray(exact), abs=1e-8)

        # ... and with the retry switched off, the NaN is what comes back --
        # which is what says the rescue above was the retry and not the route.
        unguarded, _ = davidson_eigensolver_all(hamiltonian, NBND, None, ethr=1e-13,
                                                max_iterations=60, robust_retry=False)
        assert not np.isfinite(np.asarray(unguarded)).all()
    finally:
        davidson.davidson_eigensolver_all.clear_cache()
