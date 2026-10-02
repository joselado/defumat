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
import scipy.linalg

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

    # the parked direction sorts last (the next test), so the physical roots are the rest
    physical = np.arange(n) < kept.shape[1]
    assert kept.shape[1] == n - 1, "one direction should have been parked"

    x = vectors[:, physical]
    residual = projector @ (matrix_h @ x - matrix_s @ x * values[physical])
    assert np.abs(residual).max() < 1.0e-10

    overlap = x.conj().T @ matrix_s @ x
    assert np.abs(overlap - np.eye(physical.sum())).max() < 1.0e-10


def test_the_parked_direction_sorts_above_every_physical_root():
    """Davidson takes ``values[:nbnd]``, so a dropped direction must not land there.

    Above, and no further than it has to be: it is parked one above the kept
    block's Gershgorin bound, which every kept eigenvalue lies under, so the
    margin is set by that bound rather than by a factor.
    """
    h, s = _hermitian(12, 3), _indefinite(12, 7)
    values = np.asarray(generalised_eigh(h, s)[0])
    assert values[-1] > values[:-1].max() + 0.5
    assert values[-1] < 10.0 * np.abs(values[:-1]).max()


def _davidson_pair(n_live=24, n_park=8, top=30.0, park_factor=4.0, seed=3):
    """A projected pair shaped like a Davidson solve's, with one dead direction.

    The live block is a Ritz-like projected ``H`` (spectrum -0.5 to ``top``, as
    on a cell whose largest diagonal element is ``top``) in a basis whose
    overlap has one eigenvalue at -1e-15, the near-dependent correction vector
    that makes the Cholesky route fail; the solver's own inactive directions sit
    on the diagonal at ``park_factor * top + 1`` against a unit overlap, which is
    what :func:`~defumat.solvers.davidson.davidson_eigensolver` hands
    ``generalised_eigh``. Returns ``(hc, sc)`` and the live block's spectrum.
    """
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n_live, n_live))
                        + 1j * rng.standard_normal((n_live, n_live)))
    spectrum = np.concatenate([np.linspace(-0.5, 2.0, n_live - 6),
                               np.linspace(10.0, top, 6)])
    v, _ = np.linalg.qr(rng.standard_normal((n_live, n_live))
                        + 1j * rng.standard_normal((n_live, n_live)))
    w = np.concatenate([np.linspace(1.0, 0.3, n_live - 1), [-1.0e-15]])
    root = v @ np.diag(np.sqrt(np.abs(w))) @ v.conj().T
    h_live = root.conj().T @ (q @ np.diag(spectrum) @ q.conj().T) @ root
    s_live = v @ np.diag(w) @ v.conj().T
    m = n_live + n_park
    hc = np.zeros((m, m), complex)
    sc = np.zeros((m, m), complex)
    hc[:n_live, :n_live] = 0.5 * (h_live + h_live.conj().T)
    sc[:n_live, :n_live] = 0.5 * (s_live + s_live.conj().T)
    hc[n_live:, n_live:] = (park_factor * top + 1.0) * np.eye(n_park)
    sc[n_live:, n_live:] = np.eye(n_park)
    return jnp.asarray(hc), jnp.asarray(sc), spectrum


def test_the_retry_does_not_multiply_a_callers_own_parked_directions():
    """The canonical route parks at a bound of the kept block, not at 1000 times its diagonal.

    A caller that parks its idle directions on the diagonal of ``H`` itself, as
    Davidson did at 4 times the largest diagonal element of ``H``, puts them in
    the kept block, so the old rule, 1000 times the largest diagonal element of
    the reduced matrix, handed ``eigh`` a matrix of norm 4000 times ``H``'s
    diagonal: 121001 here, against the 3e4 that made a card's Davidson stall at
    the ``ethr`` floor (``PERFORMANCE.md``, "The endgame on a card is a stall").
    The largest eigenvalue in magnitude of what ``eigh`` returns is the norm of
    what it was handed. The physical roots must not care where the dead direction
    is parked, so they are held against the live block's exact spectrum.
    """
    top, factor = 30.0, 4.0
    hc, sc, spectrum = _davidson_pair(top=top, park_factor=factor)
    values = np.asarray(generalised_eigh(hc, sc, robust=True)[0])
    assert np.abs(values).max() < 1.5 * (factor * top + 1.0)

    # the 23 live roots the overlap keeps: the projected pencil's spectrum,
    # which is ``spectrum`` with its dead direction gone, interlaced
    live = np.sort(values[values < top + 1.0])
    assert live.size == spectrum.size - 1
    assert live.min() >= spectrum.min() - 1e-10 and live.max() <= spectrum.max() + 1e-10


@pytest.mark.parametrize("robust", [False, True])
def test_parked_rows_sort_last_just_above_the_live_block_on_both_routes(robust):
    """``parked`` rows come back on top, one above the live block's bound, and change no live root.

    What Davidson hands over: its idle rows decoupled in both matrices against a
    unit overlap, and nothing on their diagonal. On the fast route they are put
    one above the Gershgorin bound of the *reduced* live block; on the retry they
    are made null directions of the overlap and parked with the dropped ones. A
    refreshed block's overlap is exactly the identity, degenerate with the
    parked rows', which is the case where an ``eigh`` of the overlap could rotate
    the two together, so the live block here starts with one.
    """
    rng = np.random.default_rng(11)
    nbnd, n_live, n_park = 6, 14, 10
    m = n_live + n_park
    h_live = np.asarray(_hermitian(n_live, 5)) * 0.2
    s_live = np.eye(n_live, dtype=complex)
    b = rng.standard_normal((n_live - nbnd, n_live - nbnd)) * 0.1
    s_live[nbnd:, nbnd:] += b @ b.T            # the corrections are not orthonormal
    parked = np.zeros(m, bool)
    parked[[3, 9, 15, 17, 18, 19, 20, 21, 22, 23]] = True   # interleaved and trailing
    h = np.zeros((m, m), complex)
    s = np.zeros((m, m), complex)
    live = np.flatnonzero(~parked)
    h[np.ix_(live, live)] = h_live
    s[np.ix_(live, live)] = s_live
    s[parked, parked] = 1.0

    exact = scipy.linalg.eigh(h_live, s_live, eigvals_only=True)
    values = np.asarray(generalised_eigh(jnp.asarray(h), jnp.asarray(s),
                                         robust=robust, parked=jnp.asarray(parked))[0])
    assert np.abs(values[:n_live] - exact).max() < 1e-12
    top = values[n_live:]
    assert np.ptp(top) < 1e-12 * top[0] and top[0] > exact.max()
    assert top[0] < 3.0 * np.abs(exact).max() * np.sqrt(n_live) + 1.0


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

    single = str(jax.make_jaxpr(generalised_eigh)(h, s))
    guarded = str(jax.make_jaxpr(jax.vmap(generalised_eigh))(*stack))
    fast = str(jax.make_jaxpr(jax.vmap(
        lambda a, b: generalised_eigh(a, b, robust=False)))(*stack))

    # **Read the conditional off the unbatched graph and the dense solves off the
    # batched one.** An earlier form asserted ``"cond[" in guarded`` and passed for
    # a reason unrelated to the guard: under ``vmap`` the guard's ``cond`` is
    # already a ``select_n``, and the ``cond[`` it found was a ``platform_index``
    # conditional that ``jnp.diagonal`` brought into the canonical route.
    assert "cond[" in single, "unbatched, the guard is a real branch"
    assert "cond[" not in guarded and "select_n" in guarded, \
        "batched, it is a select over the results of both branches"
    # ... so both routes' dense solves are computed: Cholesky's one eigh and
    # canonical orthogonalisation's two, against the fast route's one
    assert guarded.count("eigh[") == 3
    assert fast.count("eigh[") == 1
    assert "cond[" not in fast, "the fast route must have no conditional at all"


def test_the_subspace_solves_ask_for_syevd_by_name():
    """jaxlib would take cuSOLVER's Jacobi solver at 32 rows or fewer, 6 to 10x slower there.

    The route is an argument of the ``eigh`` primitive, so it is read off the
    graph: every ``eigh`` in both routes names the QR route (``syevd``), and on a
    CPU, where every route is LAPACK's ``heevd``, the values are the old ones.
    """
    import jax

    h, s = _hermitian(12, 3), _indefinite(12, 7)
    for robust in (False, True):
        graph = str(jax.make_jaxpr(lambda a, b: generalised_eigh(a, b, robust=robust))(h, s))
        assert graph.count("eigh[") >= 1
        assert graph.count("eigh[") == graph.count("algorithm=EighImplementation.QR")
    values = np.asarray(generalised_eigh(h, _positive(12, 5), robust=False)[0])
    reference = scipy.linalg.eigh(np.asarray(h), np.asarray(_positive(12, 5)), eigvals_only=True)
    assert np.abs(values - reference).max() < 1e-12


def test_the_host_route_is_opt_in_and_solves_what_the_device_route_does(monkeypatch):
    """``DEFUMAT_HOST_EIGH_ROWS``: LAPACK through a callback, only on an accelerator, only when small.

    The platform is faked, since the route is chosen at trace time from
    ``jax.default_backend()``; on a CPU both routes are LAPACK, so the eigenvalues
    agree to round-off and the graph is what says which one was taken.
    """
    import jax

    from defumat.solvers import subspace

    h, s = _hermitian(12, 3), _positive(12, 5)
    device = np.asarray(generalised_eigh(h, s, robust=False)[0])
    monkeypatch.setattr(subspace.jax, "default_backend", lambda: "gpu")
    assert subspace.HOST_EIGH_ROWS == 0
    graph = str(jax.make_jaxpr(lambda a, b: generalised_eigh(a, b, robust=False))(h, s))
    assert "pure_callback" not in graph

    monkeypatch.setattr(subspace, "HOST_EIGH_ROWS", 12)
    for robust in (False, True):
        graph = str(jax.make_jaxpr(lambda a, b: generalised_eigh(a, b, robust=robust))(h, s))
        assert "pure_callback" in graph and "eigh[" not in graph
    host = np.asarray(jax.jit(lambda a, b: generalised_eigh(a, b, robust=False))(h, s)[0])
    assert np.abs(host - device).max() < 1e-12
    batched = jax.vmap(lambda a, b: generalised_eigh(a, b, robust=False)[0])(
        jnp.stack([h, h]), jnp.stack([s, s]))
    assert np.abs(np.asarray(batched) - device[None]).max() < 1e-12

    monkeypatch.setattr(subspace, "HOST_EIGH_ROWS", 11)
    graph = str(jax.make_jaxpr(lambda a, b: generalised_eigh(a, b, robust=False))(h, s))
    assert "pure_callback" not in graph
