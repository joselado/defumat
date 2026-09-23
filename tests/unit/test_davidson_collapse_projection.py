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
executed it ``1 + 2n`` times whatever ``r`` was. Two regimes pin ``r`` without
reading it off the solver:

* a solve of fewer than ``david`` steps cannot refresh: the subspace grows by at
  most ``nbnd`` a step from ``nbnd``, and the refresh fires only once
  ``nbase + nbnd > david nbnd``, so ``r = 0``. This is the seeded solve an SCF
  iteration past the first usually makes, and there *every* projection was
  discarded;
* at ``david = 2`` every step after the first refreshes: a step is entered only
  with at least one root unsettled, which leaves ``nbase >= nbnd + 1`` and so
  ``nbase + nbnd > 2 nbnd``, while the first step starts at exactly ``nbnd``.
  So ``r = n - 1``, which is also the check that the branch still projects.

Against the old code both assertions on the count fail, the first by ``n`` and
the second by one.

**A stub operator rather than a real one.** The saving exists only where there
is an overlap, and the unit-test silicon cell is norm-conserving, where the
projections are zero-width; and a spy on ``Hamiltonian.s_projections`` would
mean patching a method onto an ``eqx.Module``. The stub is a dense generalised
problem ``H v = e S v`` with ``S = 1 + beta q beta^H``, written in exactly the
conventions of :meth:`~defumat.hamiltonian.operator.Hamiltonian.s_projections`
and ``s_correction``, and ``scipy.linalg.eigh`` is the exact answer it is
checked against.
"""

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

    def diagonal(self, ik):
        return self._diagonal

    def overlap_diagonal(self, ik):
        return self._s_diagonal

    def apply(self, psi, ik):
        return psi @ self._h.T

    def s_projections(self, vectors, ik):
        jax.debug.callback(self._executed)
        becp = vectors @ self._beta.conj()
        return becp, becp @ self._q.T

    def s_correction(self, becq, ik):
        return becq @ self._beta.T

    def _executed(self):
        self.projections.append(None)

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


def test_every_refresh_projects_the_vectors_it_keeps_and_nothing_else():
    """``david = 2``, cold: a refresh on every step but the first.

    The count is then ``2 steps`` exactly -- one more than a solve with no
    refresh, per refresh, which says the branch still projects; one fewer than
    the old ``1 + 2 steps``, which says the first step no longer does. The
    eigenvalues against ``scipy`` say the refreshed ``becp``/``becq`` are the
    right ones: a wrong overlap after a refresh would put every later
    Rayleigh-Ritz solve on a different problem.
    """
    operator = _Overlapping()
    exact, _ = operator.exact()
    values, steps, unsettled = _solve(operator, None, ethr=1e-12,
                                      residual_threshold=1e-9, david=2)

    assert steps >= 2, "no step past the first, so no refresh was exercised"
    assert unsettled == 0, f"{unsettled} roots unsettled after {steps} steps"
    assert np.max(np.abs(values - exact)) < 1e-9
    assert len(operator.projections) == 2 * steps, (
        f"{len(operator.projections)} projections in {steps} steps with "
        f"{steps - 1} refreshes; expected {2 * steps}, and {1 + 2 * steps} is "
        "the refresh's projections taken on the first step too"
    )
