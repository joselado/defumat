"""The iterative eigensolver against an exact answer.

Davidson is right only if it converged, and the way an iterative solver fails is
by *quietly* not converging: plausible numbers, wrong in the fourth decimal. On
a cell of a couple of hundred plane waves the question is settled by forming
``H`` and handing it to ``eigh``, which is right by construction -- so that is
what these tests do (``tests/exact_reference.py``). It is a test fixture and not
a solver the package offers; see ``defumat/solvers/__init__.py`` for why.

Eigenvalues are compared, not eigenvectors: silicon's bands are degenerate at
the k-point used here, and any rotation within a degenerate subspace is an
equally valid answer.
"""

import dataclasses
from pathlib import Path

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.basis.builder import build_basis
from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf.driver import Calculation, run_scf
from defumat.scf.potential import v_of_rho
from defumat.solvers import (
    DEFAULT_EIGENSOLVER,
    EIGENSOLVERS,
    davidson_eigensolver_all,
    get_eigensolver,
)
from defumat.system import build_system
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
    :func:`~defumat.solvers.davidson.davidson_eigensolver_all` retries the
    whole k-set with canonical orthogonalisation when the eigenvalues come back
    non-finite -- which is one scalar predicate, outside the batch, and a real
    branch again.

    Forcing the failure is the only way to test the retry: the overlap goes
    indefinite by round-off on cells far larger than anything a unit test may
    run, and which side of zero it lands on is a coin flip (see
    ``tests/unit/test_subspace_robustness.py``). Replacing the fast route with
    one that returns ``NaN`` exercises the identical path deterministically.
    """
    from defumat.solvers import davidson, subspace

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


# --------------------------------------------------------------------------
# the retry keeps what the fast route already converged
# --------------------------------------------------------------------------


class _StubHamiltonian:
    """The two attributes ``davidson_eigensolver_all`` reads before it solves.

    It broadcasts ``ethr`` to ``(nk, nbnd)`` using the kinetic term's dtype, so
    a bare ``None`` does not get as far as the stubbed ``_every_k``.
    """

    nk = 5

    class kinetic:
        dtype = np.float64


_HAMILTONIAN_IS_UNUSED = _StubHamiltonian()


def _stub_every_k(bad_index, nk=5, nbnd=3, ndim=7):
    """A fake ``_every_k``: the fast route fails at one k-point, robust at none.

    The select logic is what is under test, so the two routes return *different*
    finite numbers where both succeed. That is what makes "the fast answer was
    kept" a checkable statement rather than a coincidence.
    """
    fast_e = np.tile(np.arange(nbnd, dtype=float), (nk, 1))
    fast_psi = np.ones((nk, nbnd, ndim))
    fast_e[bad_index] = np.nan
    fast_psi[bad_index] = np.nan
    # Built from scratch, NOT from ``fast_e``: ``fast_e * 0 + 100`` keeps the
    # NaN and the test then cannot tell a working select from a broken one.
    robust_e = np.full((nk, nbnd), 100.0)
    robust_psi = np.full((nk, nbnd, ndim), 7.0)

    def stub(*_arguments, robust=False, return_steps=False):
        e, psi = (robust_e, robust_psi) if robust else (fast_e, fast_psi)
        out = (jnp.asarray(e), jnp.asarray(psi))
        if return_steps:
            out = out + (jnp.zeros(nk, dtype=int), jnp.zeros(nk, dtype=int))
        return out

    return stub, fast_e, fast_psi, robust_e, robust_psi


def test_the_retry_replaces_only_the_k_points_that_failed(monkeypatch):
    """One bad k-point used to discard every other k-point's converged answer.

    The guard was a single scalar over the whole set --
    ``bool(jnp.isfinite(...).all())`` -- and the retry it gated returned the
    robust route's result for *every* k-point. On a dense mesh one failure in
    ten is ordinary rather than exceptional, so that threw away nine converged
    Cholesky solves and recomputed them from a fresh random start; and since
    nothing downstream reads ``notcnv``, a robust pass that then hit its
    iteration budget handed back **less** converged wavefunctions than the ones
    it discarded. Measured on a 1H-NbSe2 mesh at ``ethr = 4e-9``: k-points 0-4
    finite, k-point 9 not, 509 s of solve turned into 1368 s.

    The predicate is now per k-point and the robust answer is taken only where
    the fast one has none.
    """
    from defumat.solvers import davidson

    stub, fast_e, fast_psi, robust_e, robust_psi = _stub_every_k(bad_index=2)
    monkeypatch.setattr(davidson, "_every_k", stub)

    with pytest.warns(UserWarning, match="non-finite"):
        values, vectors = davidson.davidson_eigensolver_all(
            _HAMILTONIAN_IS_UNUSED, 3, None, 1.0e-6)

    values, vectors = np.asarray(values), np.asarray(vectors)
    for k in (0, 1, 3, 4):
        assert values[k] == pytest.approx(fast_e[k]), f"k={k} was not kept"
        assert vectors[k] == pytest.approx(fast_psi[k]), f"k={k} was not kept"
    assert values[2] == pytest.approx(robust_e[2])
    assert vectors[2] == pytest.approx(robust_psi[2])


def test_no_retry_and_no_warning_when_every_k_point_is_finite(monkeypatch):
    """The clean case must be untouched -- and pay no second solve."""
    from defumat.solvers import davidson

    calls = []
    stub, fast_e, _, _, _ = _stub_every_k(bad_index=2)

    def clean(*arguments, robust=False, return_steps=False):
        calls.append(robust)
        nk, nbnd, ndim = 5, 3, 7
        return (jnp.tile(jnp.arange(nbnd, dtype=float), (nk, 1)),
                jnp.ones((nk, nbnd, ndim)))

    monkeypatch.setattr(davidson, "_every_k", clean)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any warning fails the test
        davidson.davidson_eigensolver_all(_HAMILTONIAN_IS_UNUSED, 3, None, 1.0e-6)
    assert calls == [False], "the robust route ran on a clean solve"


def test_the_warning_names_which_k_points_failed(monkeypatch):
    """"Something went non-finite" is not actionable; an index is."""
    from defumat.solvers import davidson

    stub, *_ = _stub_every_k(bad_index=3)
    monkeypatch.setattr(davidson, "_every_k", stub)
    with pytest.warns(UserWarning, match=r"1 of 5 k-points.*\[3\]"):
        davidson.davidson_eigensolver_all(_HAMILTONIAN_IS_UNUSED, 3, None, 1.0e-6)





# --------------------------------------------------------------------------
# a fixed-density solve that ran out of iterations says so
# --------------------------------------------------------------------------


def test_a_stalled_fixed_density_solve_is_reported_rather_than_returned():
    """The solver has always counted this and nothing ever read it.

    An SCF's early iterations are *meant* to be loose -- ``ethr`` tightens as
    the density settles -- but a fixed-density solve has no later iteration, so
    a k-point that exhausts its budget is simply not solved and its
    wavefunctions go into whatever asked for them. Measured on a 1H-NbSe2
    monolayer at the ``ethr = 4e-9`` that ``conv_thr = 1e-6`` produces: seven of
    ten k-points hit the 100-step budget with one to six bands unsettled, and
    the quantity built on top of them looked entirely plausible.
    """
    from defumat.workflows.nscf import _say_what_did_not_converge

    settled = np.zeros((1, 4), dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _say_what_did_not_converge(np.full((1, 4), 30), settled, 4e-9, 1e-6, 24)

    stalled = np.array([[0, 3, 0, 6]])
    with pytest.warns(UserWarning) as caught:
        _say_what_did_not_converge(np.array([[30, 100, 41, 100]]), stalled,
                                   4e-9, 1e-6, 24)
    message = str(caught[0].message)
    # The numbers a user needs to act: how many k-points, the worst band count,
    # the budget that was hit, and the threshold that caused it.
    for expected in ("2 of 4", "6 of 24", "100 Davidson steps", "4.0e-09",
                     "1.0e-06", "conv_thr"):
        assert expected in message, f"{expected!r} missing from: {message}"
