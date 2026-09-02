"""``generalised_eigh`` when the subspace overlap stops being positive definite.

The failure this pins is not hypothetical and was not caught by anything: 64
atoms at ``ecutwfc = 30`` converged to ``conv_thr = 1e-8`` on a GPU, reproducing
Quantum ESPRESSO's total energy, and returned ``NaN`` on the way to 1e-10. The
cause is one line. As Davidson's subspace fills, the vectors it expands with are
normalised residuals of roots that have *already converged* -- amplified
round-off -- so they go linearly dependent, the overlap's smallest eigenvalue
lands on the round-off floor, and its sign is then arbitrary. Measured on the
device: ``min eig(S) = -4.3e-16`` against ``max|S| = 1.0``.
``jnp.linalg.cholesky`` of that takes the square root of a negative pivot and
**returns NaN rather than raising**, and the NaN travels into the density, the
mixer and the total energy without anything reporting a problem.

The identical input converges on a CPU, because at that size the eigenvalue is
zero to round-off and which side of zero it falls on is a coin flip. That is the
whole of why the bug looked GPU-specific, and it is why these tests construct
the singular overlap directly instead of trying to reproduce a platform.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from jax.scipy.linalg import solve_triangular

from defumat.solvers.subspace import _cholesky_route, generalised_eigh

pytestmark = pytest.mark.unit


def _hermitian(n, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return jnp.asarray(a + a.conj().T)


def _positive(n, seed):
    rng = np.random.default_rng(seed)
    b = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return jnp.asarray(b @ b.conj().T + n * np.eye(n))


def _indefinite(n, seed, smallest=-1.0e-8):
    """An overlap with one **negative** eigenvalue, which is the real failure.

    A Gram matrix with a repeated column is only *semi*definite -- its smallest
    eigenvalue comes out at ``+1e-16`` as often as ``-1e-16``, and Cholesky
    survives the positive case, so that construction does not reproduce
    anything. The device measured a negative one, and a negative one is what
    makes Cholesky take the square root of a negative pivot.

    The magnitude cannot be the ``-4.3e-16`` actually measured: building a
    matrix whose smallest eigenvalue is 1e-16 of its largest is defeated by the
    round-off of building it. ``-1e-8`` exercises the identical code path with a
    value that survives construction.
    """
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    w = np.concatenate([np.linspace(1.0, 0.1, n - 1), [smallest]])
    m = q @ np.diag(w) @ q.conj().T
    return jnp.asarray(0.5 * (m + m.conj().T))


def test_the_fast_path_is_taken_bit_for_bit_when_the_overlap_is_positive():
    """No validated number may move: a working solve must be the old solve."""
    h, s = _hermitian(12, 0), _positive(12, 1)
    expected, expected_vectors = _cholesky_route(h, s)
    values, vectors = generalised_eigh(h, s)
    assert np.array_equal(np.asarray(values), np.asarray(expected))
    assert np.array_equal(np.asarray(vectors), np.asarray(expected_vectors))


def test_cholesky_alone_really_does_return_nan_here():
    """The premise of the fix, asserted rather than assumed.

    And asserted on the *whole* factor and on the lower triangle separately,
    because JAX leaves the unused triangle alone and a check that only looked at
    the whole array would pass for a reason that has nothing to do with the
    factorisation failing.
    """
    s = _indefinite(12, 7)
    factor = np.asarray(jnp.linalg.cholesky(s))
    assert not np.isfinite(factor).all()
    assert not np.isfinite(np.tril(factor)).all(), "the failure must be in the part that is used"


def test_the_old_route_would_have_returned_nan():
    """Without this, the tests above would pass on the unfixed code too.

    This is the ``generalised_eigh`` that shipped before the fix, verbatim.
    """
    h, s = _hermitian(12, 3), _indefinite(12, 7)
    factor = jnp.linalg.cholesky(s)
    reduced = solve_triangular(factor, h, lower=True)
    reduced = solve_triangular(factor, reduced.conj().T, lower=True).conj().T
    values = np.asarray(jnp.linalg.eigh(0.5 * (reduced + reduced.conj().T))[0])
    assert not np.isfinite(values).all(), "the old route was supposed to fail here"


def test_a_semidefinite_overlap_gives_finite_eigenpairs():
    """The regression itself. On the unfixed code both of these are NaN."""
    h, s = _hermitian(12, 3), _indefinite(12, 7)
    values, vectors = generalised_eigh(h, s)
    assert np.isfinite(np.asarray(values)).all()
    assert np.isfinite(np.asarray(vectors)).all()


def test_the_surviving_roots_solve_the_equation_that_is_solvable():
    """``H x = e S x`` has no solution outside ``range(S)`` when ``S`` is singular.

    So the check is the *projected* residual, plus S-orthonormality of what
    comes back -- which is what Davidson actually consumes.
    """
    n = 12
    h, s = _hermitian(n, 3), _indefinite(n, 7)
    values, vectors = map(np.asarray, generalised_eigh(h, s))
    matrix_h, matrix_s = np.asarray(h), np.asarray(s)

    w, u = np.linalg.eigh(matrix_s)
    kept = u[:, w > 1.0e-12 * w.max()]
    projector = kept @ kept.conj().T

    physical = values < values.max() / 2.0
    assert physical.sum() == kept.shape[1], "one direction should have been parked"
    assert kept.shape[1] == n - 1

    x = vectors[:, physical]
    residual = projector @ (matrix_h @ x - matrix_s @ x * values[physical])
    assert np.abs(residual).max() < 1.0e-10

    overlap = x.conj().T @ matrix_s @ x
    assert np.abs(overlap - np.eye(physical.sum())).max() < 1.0e-10


def test_the_parked_direction_sorts_above_every_physical_root():
    """Davidson takes ``values[:nbnd]``, so a dropped direction must not land there."""
    h, s = _hermitian(12, 3), _indefinite(12, 7)
    values = np.asarray(generalised_eigh(h, s)[0])
    assert values[-1] > 100.0 * np.abs(values[:-1]).max()


# ------------------------------------------------------ the guard's own cost

def test_the_static_routes_are_the_guard_taken_apart():
    """``robust=False`` is bit-for-bit what the guard returns when it passes.

    That equality is what lets the batched Davidson path drop the ``cond``
    without changing a validated number: the guard passing *is* the Cholesky
    route, and ``select_n`` chose between two computed arrays and took that one.
    """
    h, s = _hermitian(12, 3), _positive(12, 5)
    guarded = [np.asarray(a) for a in generalised_eigh(h, s)]
    fast = [np.asarray(a) for a in generalised_eigh(h, s, robust=False)]
    assert np.array_equal(guarded[0], fast[0])
    assert np.array_equal(guarded[1], fast[1])


def test_the_robust_route_survives_an_overlap_the_fast_one_does_not():
    """And the two disagree exactly where they should: on an indefinite ``S``."""
    h, s = _hermitian(12, 3), _indefinite(12, 7)
    assert not np.isfinite(np.asarray(_cholesky_route(h, s)[0])).all()
    assert np.isfinite(np.asarray(generalised_eigh(h, s, robust=True)[0])).all()


def test_a_batched_guard_has_no_branch_to_take():
    """Why the guard cannot live inside a ``vmap`` over k, as a structural fact.

    ``lax.cond`` with a *batched* predicate is lowered to ``select_n`` over the
    results of both branches -- there is no per-element branch on a device -- so
    a guarded solve inside ``map_k``'s ``vmap`` computes canonical
    orthogonalisation on every step of every k-point in addition to the Cholesky
    route it uses. Measured at ``si10-nc``'s shapes: 2.85x. This test pins the
    mechanism rather than the ratio, which is a property of the machine.
    """
    import jax

    h, s = _hermitian(12, 3), _positive(12, 5)
    stack = (jnp.broadcast_to(h, (3, 12, 12)), jnp.broadcast_to(s, (3, 12, 12)))

    guarded = str(jax.make_jaxpr(jax.vmap(generalised_eigh))(*stack))
    fast = str(jax.make_jaxpr(jax.vmap(
        lambda a, b: generalised_eigh(a, b, robust=False)))(*stack))

    assert "cond[" in guarded, "the guarded route should carry a conditional"
    assert "select_n" in guarded, "which under vmap becomes a select over both branches"
    assert "cond[" not in fast, "the fast route must have no conditional at all"
