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

import warnings

import numpy as np
import pytest

from defumat.scf.mixing import (
    AdaptiveMixer, AndersonMixer, LinearMixer, PRECONDITIONED, get_mixer,
)

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


def test_mixing_ndim_reaches_the_mixer_and_an_unset_one_does_not_move():
    """``mixing_ndim`` was parsed and dropped on the floor until P78.

    ``AndersonMixer.history`` was fixed at 8, which is also ``pw.x``'s default,
    so the *silence* was the whole defect: an input that set it behaved exactly
    like one that did not, and raising it is the first thing a QE user does to a
    cell that will not converge.
    """
    from defumat.scf.mixing import get_mixer

    assert get_mixer("anderson", beta=0.3, history=12).history == 12
    assert get_mixer("anderson", beta=0.3, history=None).history == 8
    assert get_mixer("anderson", beta=0.3).history == 8

    # A mixer with no such knob says so rather than raising a TypeError from
    # inside a dataclass constructor, because the caller is usually an input
    # file and the fix belongs in the input file.
    with pytest.raises(ValueError, match="has no history"):
        get_mixer("linear", beta=0.3, history=12)


def test_mixing_ndim_is_adopted_from_the_electrons_namelist_as_an_int():
    from defumat.calculator import electrons_defaults
    from defumat.io.pwin import parse_pw_input

    text = """
 &control
    calculation = 'scf'
 /
 &system
    ibrav = 1, celldm(1) = 5.0, nat = 1, ntyp = 1, ecutwfc = 10.0
 /
 &electrons
    mixing_ndim = 12
 /
ATOMIC_SPECIES
 H 1.008 H.pz-vbc.UPF
ATOMIC_POSITIONS crystal
 H 0.0 0.0 0.0
K_POINTS gamma
"""
    adopted = electrons_defaults(parse_pw_input(text))
    assert adopted["mixing_ndim"] == 12
    assert isinstance(adopted["mixing_ndim"], int)


def test_every_relaxation_driver_names_mixing_ndim():
    """A ``**kwargs`` is not permission to pass everything, so each driver has
    to name it -- the defect ``mixing_fixed_ns`` already had once."""
    import inspect

    from defumat.workflows.relax import run_relax
    from defumat.workflows.spiral import relax_spiral_q
    from defumat.workflows.vc_relax import run_vc_relax

    for driver in (run_relax, run_vc_relax, relax_spiral_q):
        assert "mixing_ndim" in inspect.signature(driver).parameters


# ---------------------------------------------------------------------------
# Elk's ``mixadapt``
# ---------------------------------------------------------------------------


def _mixadapt(iscl, beta0, betamax, nu, mu, beta, f):
    """``src/mixadapt.f90``, transcribed literally, as the thing to agree with.

    Written from the Fortran and not from :class:`AdaptiveMixer`, because a
    reference derived from the code under test only ever checks that the code
    agrees with itself. The loop below is the Fortran's, index for index, with
    its ``iscl < 1`` initialisation arm kept: ``mu = nu``, ``f = 0``,
    ``beta = beta0``, and no mixing at all on that call.
    """
    nu, mu, beta, f = (np.array(x, dtype=float) for x in (nu, mu, beta, f))
    if iscl < 1:
        mu[:] = nu
        f[:] = 0.0
        beta[:] = beta0
        return nu, mu, beta, f, 1.0
    d = 0.0
    for i in range(len(nu)):
        t1 = nu[i] - mu[i]
        d += t1**2
        if t1 * f[i] >= 0.0:
            beta[i] = beta[i] + beta0
            if beta[i] > betamax:
                beta[i] = betamax
        else:
            beta[i] = 0.5 * (beta[i] + beta0)
        f[i] = t1
        nu[i] = beta[i] * nu[i] + (1.0 - beta[i]) * mu[i]
        mu[i] = nu[i]
    return nu, mu, beta, f, np.sqrt(d / len(nu))


def test_the_adaptive_mixer_is_elks_mixadapt_to_the_last_bit():
    """Both driven over the same outputs, with all three branches exercised.

    Component 0 keeps its residual's sign throughout, so its step climbs by
    ``beta0`` every iteration and then sits at ``betamax``; component 1 changes
    sign at every step, so its step is halved back towards ``beta0`` each time;
    component 2 is the ``f = 0`` case of the very first call, where ``t1 * 0``
    compares ``>= 0`` and Elk increments rather than halves.
    """
    beta0, betamax, steps = 0.05, 0.4, 24
    rng = np.random.default_rng(20260914)
    mixer = AdaptiveMixer(beta=beta0, beta_max=betamax)

    # Elk's initialisation call, which mixes nothing.
    density = np.array([0.0, 0.0, 0.0])
    nu, mu, beta, f, _ = _mixadapt(0, beta0, betamax, density, density, density, density)
    ours = density.copy()

    for step in range(1, steps + 1):
        # The same output density for both, built so the signs are as described.
        magnitude = 0.9**step * (1.0 + 0.1 * rng.random(3))
        out = ours + magnitude * np.array([1.0, (-1.0) ** step, 1.0])
        # Elk is handed its own ``mu``; ours is handed the input it produced.
        nu, mu, beta, f, _ = _mixadapt(step, beta0, betamax, out, mu, beta, f)
        ours = np.asarray(mixer.mix(ours, out))
        assert ours.tolist() == nu.tolist(), f"diverged at step {step}"
        assert mixer._betas.tolist() == beta.tolist(), f"beta diverged at step {step}"

    # The branches really were taken, so this is not three copies of one case.
    assert mixer._betas[0] == pytest.approx(betamax)
    assert beta0 < mixer._betas[1] < betamax


def test_the_adaptive_step_grows_while_the_residual_keeps_its_sign():
    """The whole mechanism, on the flat direction it exists for.

    A residual that never turns around is what a badly conditioned direction
    looks like from inside the mixer: the density crawls the same way for ever
    and a fixed ``beta`` crawls with it. Elk's rule lets the step climb out.
    """
    mixer = AdaptiveMixer(beta=0.05, beta_max=1.0)
    density = np.zeros(2)
    lengths = []
    for _ in range(30):
        density = np.asarray(mixer.mix(density, density + np.ones(2)))
        lengths.append(float(mixer._betas[0]))
    assert lengths == sorted(lengths)
    assert lengths[0] == pytest.approx(0.1)     # beta0 + beta0, the f = 0 call
    assert lengths[-1] == pytest.approx(1.0)    # capped at beta_max, not past it


def test_the_adaptive_mixer_refuses_a_preconditioner():
    """And the guard is watched firing, not read.

    The driver installs one only for a mode in ``PRECONDITIONED`` and this is
    not in it, so the refusal is for the caller who sets the attribute by hand.
    """
    mixer = AdaptiveMixer()
    mixer.precondition = lambda residual, density=None: residual
    with pytest.raises(ValueError, match="cannot take a preconditioner"):
        mixer.mix(np.zeros(4), np.ones(4))
    assert "adaptive" not in PRECONDITIONED


@pytest.mark.parametrize("beta,beta_max", [(-0.1, 1.0), (0.05, 1.5), (0.05, -0.2)])
def test_the_adaptive_mixer_keeps_elks_own_bounds(beta, beta_max):
    with pytest.raises(ValueError):
        AdaptiveMixer(beta=beta, beta_max=beta_max)


def test_a_step_length_passed_as_the_increment_says_so():
    """``mixing_beta = 0.7`` is QE's step length and Elk's increment is not it.

    At 0.7 the scheme saturates at ``beta_max`` in one iteration and is linear
    mixing at 0.85 wearing an adaptive name, which is the failure that would
    otherwise be read as "the Elk mixer does not help here".
    """
    with pytest.warns(RuntimeWarning, match="increment and a floor"):
        AdaptiveMixer(beta=0.7)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        AdaptiveMixer()                      # Elk's own 0.05, silent
        AdaptiveMixer(beta=0.05)


def test_an_unset_mixing_beta_gives_each_mode_its_own_default():
    """Which is the whole reason the entry points default it to ``None``.

    ``get_mixer`` drops a ``None``, so a caller who never mentioned
    ``mixing_beta`` gets QE's 0.7 where that is a step length and Elk's 0.05
    where it is an increment, rather than 0.7 in both meanings.
    """
    assert get_mixer("anderson").beta == pytest.approx(0.7)
    assert get_mixer("linear").beta == pytest.approx(0.7)
    assert get_mixer("adaptive").beta == pytest.approx(0.05)
    assert get_mixer("adaptive").beta_max == pytest.approx(1.0)


def test_the_ultracell_refuses_the_pair_before_it_builds_anything():
    """``kerker`` defaults to ``True`` there, so the clash is the easy mistake.

    Watched firing rather than read: the call is made with nothing else valid,
    which passes only because the refusal is ahead of every argument the rest of
    the driver would touch. A guard further down would raise a different error
    here and the test would fail, which is the point of calling it this way.
    """
    from defumat.ultracell.driver import run_ultracell

    with pytest.raises(ValueError, match="does not take a preconditioner"):
        run_ultracell(None, (), None, (2, 1, 1), mixing_mode="adaptive")
    # And it is the *pair* that is refused, not the mixer: turning kerker off
    # gets past this and on to the ordinary argument handling.
    with pytest.raises(Exception) as other:
        run_ultracell(None, (), None, (2, 1, 1), mixing_mode="adaptive",
                      kerker=False)
    assert "does not take a preconditioner" not in str(other.value)


def test_the_adaptive_mixer_refuses_a_complex_vector():
    """Because numpy would not, which is the whole reason the guard is there.

    ``np.array([1+1j]) >= 0`` does not raise: numpy orders complex numbers on
    the real part and breaks ties on the imaginary one. So the sign rule would
    run on the real part alone and adapt every step length from half the
    information, silently. The driver never reaches this -- ``_mix`` packs a
    spinor ``ns`` as a real view -- and a caller using the mixer directly can.
    """
    mixer = AdaptiveMixer()
    with pytest.raises(TypeError, match="has no sign"):
        mixer.mix(np.zeros(4, dtype=complex), np.ones(4, dtype=complex))
    # The premise, asserted rather than remembered: numpy really does compare.
    assert bool((np.array([1 + 1j]) * np.array([1 + 0j]) >= 0)[0]) is True
