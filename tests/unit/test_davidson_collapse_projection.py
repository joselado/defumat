"""Davidson's refresh projects the vectors it keeps, and only when it runs.

``OPEN.md`` H5. When the subspace is full, ``cegterg``'s refresh collapses it
onto the current Ritz vectors, and with an augmentation charge it needs their
projections ``<beta|evc>`` to rebuild the tracked ``becp``/``becq``. Those used
to be computed on **every** step, just above the ``lax.cond`` that is their only
consumer, and thrown away on every step that did not collapse. The call is now
inside the branch. The arithmetic is unchanged, so no eigenvalue can tell the
two apart; what can is **how many times the projections are executed**, and
that is what these tests count, with a host callback inside the operator's
``s_projections``. With no ``vmap`` and no derivative around it, a
``jax.debug.callback`` inside a ``cond`` whose predicate is a scalar fires once
each time its branch runs and not otherwise; under a ``vmap`` the ``cond``
becomes a ``select_n`` and both branches run, which is why the solve here is
called directly, with no batch axis.

**The count is exact rather than bounded, because the solver's structure fixes
it.** ``project`` runs once on the starting block, once per step on the
expansion block (``live_block``; one branch of its ``switch`` runs, so one
call), and once per refresh. A solve of ``n`` steps with ``r`` refreshes
therefore executes it ``1 + n + r`` times, where the code before the fix
executed it ``1 + 2n`` times whatever ``r`` was. A solve of fewer than
``david`` steps cannot refresh: before step ``t`` the subspace holds at most
``t nbnd`` vectors, and the refresh needs ``nbase + notcnv > david nbnd``, so
``r = 0``. This is the seeded solve an SCF iteration past the first usually
makes, and there *every* projection was discarded.

**When the refresh fires is ``cegterg``'s test, ``nbase + notcnv > nvecx``**
(``cegterg.f90:580-581``), and since 2026-10-10 that is what the solver tests;
it used to be ``nbase + nbnd > nvecx``, room for a whole block, which at
``david = 2`` refreshed on every step after the first however few roots were
left. The stub logs every application of ``H`` with its row count, and with one
band rung per band the expansion block is applied at exactly ``notcnv`` rows,
so the log is the solve's schedule: the ``notcnv`` of each step and whether a
refresh preceded it. :func:`_replay` walks it with ``cegterg``'s rule. Against
the old code the replay fails on the cold solve and on the seeded one with one
spoiled band, and the seeded one also takes ``steps - 1`` refreshes there.

**A stub operator rather than a real one.** The saving exists only where there
is an overlap, and the unit-test silicon cell is norm-conserving, where the
projections are zero-width; and a spy on ``Hamiltonian.s_projections`` would
mean patching a method onto an ``eqx.Module``. The stub is a dense generalised
problem ``H v = e S v`` with ``S = 1 + beta q beta^H``, written in exactly the
conventions of :meth:`~defumat.hamiltonian.operator.Hamiltonian.s_projections`
and ``s_correction``, and ``scipy.linalg.eigh`` is the exact answer it is
checked against.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from defumat.config import DOUBLE
from defumat.solvers.davidson import DAVID_NDIM, davidson_eigensolver

pytestmark = pytest.mark.unit

#: Toy sizes. ``NDIM`` must stay at least ``DAVID_NDIM * NBND`` so that the
#: subspace is not capped by the space (``nvecx = min(david nbnd, space)``),
#: which both regimes in the module docstring rely on.
NDIM, NBND, NKB = 40, 4, 3


class _Overlapping:
    """A dense ``H v = e S v`` with the attributes ``davidson_eigensolver`` reads.

    ``H`` is diagonally dominant, a kinetic-like diagonal ``2 g`` plus a small
    Hermitian coupling, so the ``g_psi`` preconditioner does what it does on a
    plane-wave Hamiltonian and a cold solve converges well inside the budget.
    ``S - 1 = beta q beta^H`` with ``q`` positive, so ``S`` is positive definite.
    """

    gamma_only = False
    has_overlap = True

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)

        def gaussian(*shape):
            return (rng.normal(size=shape)
                    + 1j * rng.normal(size=shape)).astype(DOUBLE.complex)

        coupling = gaussian(NDIM, NDIM)
        kinetic = 2.0 * np.arange(NDIM, dtype=DOUBLE.real)
        self.h = np.diag(kinetic).astype(DOUBLE.complex) + 0.01 * (
            coupling + coupling.conj().T)
        self.beta = 0.03 * gaussian(NDIM, NKB)
        self.q = np.diag([0.5, 0.3, 0.2]).astype(DOUBLE.complex)
        self.s = np.eye(NDIM, dtype=DOUBLE.complex) + self.beta @ self.q @ self.beta.conj().T

        self.ndim = NDIM
        self.space = NDIM
        self.dtype = DOUBLE.complex
        self.state_mask = jnp.ones((1, NDIM), dtype=bool)
        self.state_kinetic = jnp.asarray(kinetic[None, :])
        self._h = jnp.asarray(self.h)
        self._beta = jnp.asarray(self.beta)
        self._q = jnp.asarray(self.q)
        self._diagonal = jnp.asarray(np.diag(self.h).real)
        self._s_diagonal = jnp.asarray(np.diag(self.s).real)
        #: One entry per *executed* ``s_projections``, appended by the host
        #: callback, so a traced call that never runs adds nothing.
        self.projections = []
        #: Every executed ``apply`` and ``s_projections``, in order, as
        #: ``("H" or "P", rows)``: the solve's schedule (:func:`_replay`).
        self.events = []

    def diagonal(self, ik):
        return self._diagonal

    def overlap_diagonal(self, ik):
        return self._s_diagonal

    def apply(self, psi, ik):
        jax.debug.callback(partial(self._log, "H", psi.shape[0]), ordered=True)
        return psi @ self._h.T

    def s_projections(self, vectors, ik):
        jax.debug.callback(partial(self._executed, vectors.shape[0]), ordered=True)
        becp = vectors @ self._beta.conj()
        return becp, becp @ self._q.T

    def apply_projected(self, psi, ik):
        """``apply`` and ``s_projections`` of one block, as the real operators pair them.

        Written as the two calls so that the callback counts one projection
        per block exactly as it did when the solver called them separately.
        """
        return (self.apply(psi, ik), *self.s_projections(psi, ik))

    def s_correction(self, becq, ik):
        return becq @ self._beta.T

    def _executed(self, rows):
        self.projections.append(None)
        self._log("P", rows)

    def _log(self, kind, rows):
        self.events.append((kind, rows))

    def exact(self):
        """The lowest ``NBND`` generalised eigenpairs, S-orthonormal."""
        values, vectors = scipy.linalg.eigh(self.h, self.s)
        return values[:NBND], vectors[:, :NBND].T


def _solve(operator, *args, **kwargs):
    """``davidson_eigensolver`` at k-point 0, with every callback accounted for."""
    values, _, steps, unsettled = davidson_eigensolver(
        operator, 0, NBND, *args, return_steps=True, **kwargs)
    values = np.asarray(values)
    jax.effects_barrier()
    return values, int(np.asarray(steps)), int(np.asarray(unsettled))


def test_a_solve_that_never_refreshes_projects_nothing_it_discards():
    """The seeded solve: one projection per step, where there used to be two.

    Seeded with the exact eigenvectors, the solve settles in a step or two,
    far short of the ``DAVID_NDIM`` steps a refresh needs. So every projection
    of ``evc`` the old code made was discarded, and the count is
    ``1 + steps`` against its ``1 + 2 steps``.
    """
    operator = _Overlapping()
    exact, seed = operator.exact()
    # ``ethr`` at an SCF-like 1e-8 rather than the 1e-12 default. Seeded with
    # the answer, the change in each eigenvalue at the first step is round-off,
    # and 1e-8 is a margin so that no round-off can cost a second step; the
    # count asserted below needs only that no refresh happened.
    values, steps, unsettled = _solve(operator, jnp.asarray(seed), ethr=1e-8,
                                      david=DAVID_NDIM)

    assert 1 <= steps < DAVID_NDIM, (
        f"{steps} steps: a refresh is possible from step {DAVID_NDIM}, so the "
        "count below no longer pins r = 0"
    )
    assert unsettled == 0
    assert np.max(np.abs(values - exact)) < 1e-10
    assert len(operator.projections) == 1 + steps, (
        f"{len(operator.projections)} projections in {steps} steps with no "
        f"refresh; the start and one expansion per step make {1 + steps}, and "
        f"{1 + 2 * steps} is the refresh's projections taken on every step"
    )


def _replay(events, david):
    """Walk the logged schedule with ``cegterg``'s rule; the number of refreshes.

    The log is ``("H", rows)`` then ``("P", rows)`` for every block ``H`` is
    applied to (``apply_projected``), the start block first, and a lone
    ``("P", nbnd)`` for each refresh, which projects the vectors it keeps and
    applies nothing. A step is an optional refresh followed by one expansion of
    ``notcnv`` rows, and the refresh must have fired exactly when the
    expansion would not otherwise have fitted: ``nbase + notcnv > nvecx``.
    """
    nvecx = min(david * NBND, NDIM)
    assert events[:2] == [("H", NBND), ("P", NBND)], events[:2]
    nbase, refreshes, pending, i = NBND, 0, False, 2
    while i < len(events):
        kind, rows = events[i]
        if kind == "P":                       # a refresh: no H in front of it
            assert not pending and rows == NBND, f"event {i}: {events[i - 1:i + 2]}"
            pending = True
            i += 1
            continue
        assert events[i + 1] == ("P", rows), events[i:i + 2]
        notcnv = rows
        assert pending == (nbase + notcnv > nvecx), (
            f"event {i}: nbase {nbase}, notcnv {notcnv}, nvecx {nvecx}: the "
            f"solver {'refreshed' if pending else 'did not refresh'}, and "
            "cegterg's nbase + notcnv > nvecx says otherwise"
        )
        if pending:
            nbase, refreshes, pending = NBND, refreshes + 1, False
        nbase += notcnv
        i += 2
    assert not pending, "a refresh with no expansion after it"
    return refreshes


def _spoiled_seed(operator):
    """The exact eigenvectors with the top one spoiled, and the exact eigenvalues."""
    exact, seed = operator.exact()
    rng = np.random.default_rng(1)
    seed = seed.copy()
    seed[-1] += 0.3 * (rng.normal(size=NDIM) + 1j * rng.normal(size=NDIM))
    return exact, jnp.asarray(seed)


def test_a_cold_solve_refreshes_on_cegtergs_test():
    """``david = 2``, cold: refreshes where ``nbase + notcnv > nvecx``, and right.

    The count is ``1 + steps + r`` with ``r`` read off the schedule, which says
    the refresh still projects the vectors it keeps and that nothing else
    does. The eigenvalues against ``scipy`` say the refreshed ``becp``/``becq``
    are the right ones: a wrong overlap after a refresh would put every later
    Rayleigh-Ritz solve on a different problem. One band rung per band
    (``band_rungs = NBND``) so that the logged width of each expansion is its
    ``notcnv``.
    """
    operator = _Overlapping()
    exact, _ = operator.exact()
    values, steps, unsettled = _solve(operator, None, ethr=1e-12,
                                      residual_threshold=1e-9, david=2,
                                      band_rungs=NBND)

    assert steps >= 2, "no step past the first, so no refresh was exercised"
    assert unsettled == 0, f"{unsettled} roots unsettled after {steps} steps"
    assert np.max(np.abs(values - exact)) < 1e-9
    refreshes = _replay(operator.events, 2)
    assert refreshes >= 1, "the solve never refreshed, so the branch was not exercised"
    assert len(operator.projections) == 1 + steps + refreshes, (
        f"{len(operator.projections)} projections in {steps} steps with "
        f"{refreshes} refreshes; expected {1 + steps + refreshes}"
    )


def test_one_unsettled_root_keeps_its_history():
    """``david = 2``, seeded with one band spoiled: ``notcnv`` is 1 from step 2.

    ``cegterg`` then fits ``nvecx - nbnd = nbnd`` single-vector expansions
    between refreshes, where the old ``nbase + nbnd > nvecx`` refreshed on
    every step after the first, ``steps - 1`` times, which made ``david = 2`` a
    restarted steepest descent on exactly the root that needed the history.
    Fails on the code before 2026-10-10 twice: in the replay, and in the count.
    """
    operator = _Overlapping()
    exact, seed = _spoiled_seed(operator)
    values, steps, unsettled = _solve(operator, seed, ethr=1e-12,
                                      residual_threshold=1e-9, david=2,
                                      band_rungs=NBND)

    assert unsettled == 0 and np.max(np.abs(values - exact)) < 1e-9
    assert steps >= 3, f"{steps} steps: too few to tell the two rules apart"
    refreshes = _replay(operator.events, 2)
    assert refreshes < steps - 1, (
        f"{refreshes} refreshes in {steps} steps: one on every step after the "
        "first is the old nbase + nbnd > nvecx"
    )


@pytest.mark.parametrize("narrow, rungs", [(True, 1), (True, NBND), (False, 1)])
def test_the_schedule_does_not_depend_on_the_route(narrow, rungs):
    """The collapse is a property of the solve, not of how it is batched.

    ``narrow`` and ``band_rungs`` are performance dials (``narrow`` is what
    :func:`~defumat.solvers.davidson._every_k` sets from the k-batch), so the
    step count, the eigenvalues and the number of projections must be the
    same on every one of them. A collapse test written on the band rung rather
    than on ``notcnv`` fails this.
    """
    reference = _Overlapping()
    _, seed = _spoiled_seed(reference)
    base = _solve(reference, seed, ethr=1e-12, residual_threshold=1e-9,
                  david=2, band_rungs=NBND)
    other = _Overlapping()
    run = _solve(other, seed, ethr=1e-12, residual_threshold=1e-9,
                 david=2, narrow=narrow, band_rungs=rungs)
    assert run[1:] == base[1:]
    assert np.max(np.abs(run[0] - base[0])) < 1e-12
    assert len(other.projections) == len(reference.projections)


def test_a_write_past_the_end_drops_the_tail_and_keeps_the_live_rows():
    """:func:`_write_rows` against the clamp it replaces.

    Six rows, the first four live; a block of three, one live row and two zero
    ones, written at row 4. ``dynamic_update_slice`` moves the write back to
    row 3 and overwrites a live vector. The helper keeps it, puts the live row
    at 4, drops the zero row that falls off the end, and at a start that fits
    is the plain update to the bit.
    """
    # imported here, so that the rest of the file runs against the code before
    # the helper existed, which is how its failures there were checked
    from defumat.solvers.davidson import _write_rows

    buffer = jnp.arange(12.0).reshape(6, 2) + 1.0
    buffer = buffer.at[4:].set(0.0)
    block = jnp.zeros((3, 2)).at[0].set(jnp.array([-1.0, -2.0]))

    clamped = jax.lax.dynamic_update_slice(buffer, block, (4, 0))
    assert not np.array_equal(clamped[3], buffer[3]), "the clamp did not bite"

    written, at = _write_rows(buffer, block, 4)
    assert int(at) == 3
    np.testing.assert_array_equal(written[:4], buffer[:4])
    np.testing.assert_array_equal(written[4], block[0])
    np.testing.assert_array_equal(written[5], np.zeros(2))

    fits, at = _write_rows(buffer, block, 2)
    assert int(at) == 2
    np.testing.assert_array_equal(fits, jax.lax.dynamic_update_slice(buffer, block, (2, 0)))


def test_a_subspace_of_one_block_is_refused():
    """``david = 1`` leaves no room for an expansion, ``cegterg.f90:125``."""
    with pytest.raises(ValueError, match="at least 2"):
        davidson_eigensolver(_Overlapping(), 0, NBND, david=1)
