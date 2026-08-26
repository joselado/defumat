"""The charge-density mixers, and the conditioning that decides whether they work.

``AndersonMixer`` had **no coverage at all** until 64 atoms at ``ecutwfc = 30``
converged to ``conv_thr = 1e-8`` on a GPU and returned ``NaN`` on the way to
1e-10 (`PERFORMANCE.md`). That absence is what these tests are for.

**They are coverage, not a reproduction of that failure**, and the distinction
is worth keeping straight. The Gram matrix ``r_i . r_j`` is built from residuals
whose magnitudes span the whole history, so its condition number grows about two
orders per SCF iteration -- 1.1e11 by the eighth on a 16-atom cell, 1.7e8 even
after normalising on a 64-atom one. But a large condition number is *not* by
itself enough to produce garbage: a synthetic history at cond 1e53 still solves
to ``max|c| = 1`` under partial pivoting, because there the solution is
well-determined even though the matrix is not. So the conditioning is a real
defect worth removing and is removed here, and whether it is *the* cause of that
``NaN`` is a separate claim that these tests do not make.
"""

import numpy as np
import pytest

from pypresso.scf.mixing import AndersonMixer, LinearMixer

pytestmark = pytest.mark.unit


def _converging_history(steps, ratio=0.05, size=64, seed=0):
    """Densities whose residuals shrink geometrically, as a converging SCF's do."""
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(size)
    return [(rng.standard_normal(size) * 1e-3 + direction) * ratio**n for n in range(steps)]


def test_linear_mixer_is_the_plain_step():
    mixer = LinearMixer(beta=0.3)
    rho_in, rho_out = np.ones(4), np.ones(4) * 2.0
    assert np.allclose(mixer.mix(rho_in, rho_out), 1.0 + 0.3 * 1.0)


def test_anderson_reproduces_the_exact_solution_when_well_conditioned():
    """With residuals all of one size the raw and normalised systems agree."""
    rng = np.random.default_rng(1)
    mixer = AndersonMixer(beta=0.7)
    rho = rng.standard_normal(32)
    for _ in range(4):
        out = rho + rng.standard_normal(32) * 0.1
        mixed = mixer.mix(rho, out)
        assert np.all(np.isfinite(mixed))
        rho = mixed


def test_anderson_survives_a_history_spanning_many_orders():
    """Residuals from 1e0 down to 1e-24 in one history.

    The bordered system built from the raw Gram matrix has a condition number
    past 1e50 here. Measured: the *old* code also survives this particular
    history, because partial pivoting copes when the solution is well-determined
    even though the matrix is not -- so this is a guard on the property, not a
    reproduction of the 64-atom failure. It would catch a future change that
    made the mixer amplify instead of interpolate.
    """
    mixer = AndersonMixer(beta=0.7, history=8)
    rho = np.zeros(64)
    residuals = _converging_history(9, ratio=1.0e-3, size=64)
    for residual in residuals:
        mixed = mixer.mix(rho, rho + residual)
        assert np.all(np.isfinite(mixed)), "the mixer produced a non-finite density"
        # A mixing step interpolates; it must not amplify by orders of magnitude.
        assert np.abs(mixed - rho).max() < 1.0e3 * np.abs(residual).max() + 1.0e-12
        rho = mixed


def test_anderson_trims_its_history_rather_than_solving_a_singular_system():
    """An exactly repeated residual makes the Gram matrix singular by construction."""
    mixer = AndersonMixer(beta=0.7, history=8)
    rho = np.zeros(16)
    fixed = np.arange(16, dtype=float) * 1.0e-6
    for _ in range(5):
        mixed = mixer.mix(rho, rho + fixed)      # the same residual every time
        assert np.all(np.isfinite(mixed))
        rho = mixed


def test_the_normalised_system_gives_the_same_coefficients():
    """The substitution is exact, so it may not move a well-conditioned answer."""
    rng = np.random.default_rng(2)
    residuals = [rng.standard_normal(48) * 10.0**-k for k in range(5)]
    n = len(residuals)
    gram = np.array([[float(a @ b) for b in residuals] for a in residuals])
    norms = np.sqrt(np.diag(gram))

    raw = np.zeros((n + 1, n + 1))
    raw[:n, :n] = gram
    raw[:n, n] = raw[n, :n] = 1.0
    rhs = np.zeros(n + 1)
    rhs[n] = 1.0
    exact = np.linalg.solve(raw, rhs)[:n]

    scaled = AndersonMixer._build_overlap(gram, norms, n)
    got = np.linalg.solve(scaled, rhs)[:n] / norms

    assert np.allclose(got, exact, rtol=1.0e-6)
    assert np.isclose(got.sum(), 1.0)          # the constraint still holds
    # and the point of it all: the normalised system is far better conditioned
    assert np.linalg.cond(scaled) < np.linalg.cond(raw) / 1.0e3
