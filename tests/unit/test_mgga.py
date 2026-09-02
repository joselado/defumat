"""P30 check: the Becke-Roussel / Becke-Johnson / Tran-Blaha potential.

There is no Fortran to compare against -- QE reaches TB09 only through libxc,
and with a zero Laplacian and ``c = 1`` at that (see
:mod:`defumat.xc.mgga`) -- so the checks here are *analytic identities* the
functional has to satisfy, which is a stronger kind of test than an agreement
to somebody else's floating point:

* the **hydrogen atom**, where the Becke-Roussel potential is the exact Slater
  potential of the 1s orbital and the exchange energy is exactly -5/16 Ha. This
  is the check that pins the whole chain -- the sign of ``Q``, the branch of the
  nonlinear solve, the ``rho^(1/3)`` prefactor and the Hartree ``tau``
  convention -- and it is independent of ``gamma``, because ``D`` vanishes
  identically for a one-orbital density;
* the **uniform electron gas**, where Becke-Johnson's coefficient is chosen so
  that ``v_x^BJ`` reproduces ``v_x^LDA``. It does so to 6e-4 and not exactly,
  and the residue is not this implementation's: it is the Becke-Roussel model's
  own error against the uniform gas's Slater potential at ``gamma = 0.8``. The
  test measures it and pins ``gamma`` at the same time, since it is a steep
  function of ``gamma`` (2.8% at ``gamma = 1``);
* the **implicit derivative**, against a finite difference of the bisection it
  replaces.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from defumat.units import E2
from defumat.xc.functional import get_functional
from defumat.xc.mgga import (
    BR89_GAMMA,
    META_RHO_THRESHOLD,
    TB09_ALPHA,
    TB09_BETA,
    becke_roussel_potential_hartree,
    becke_roussel_x,
    tb09_coefficient,
    tb09_potential,
    thomas_fermi_tau,
)

pytestmark = pytest.mark.unit


def _hydrogen(r):
    """The 1s density of atomic hydrogen and everything the functional wants.

    ``rho = e^(-2r)/pi`` for a single occupied orbital ``psi = e^(-r)/sqrt(pi)``,
    so this is one spin channel with one electron in it -- the case Becke and
    Roussel's model is exact for.
    """
    rho = np.exp(-2 * r) / np.pi
    sigma = (2 * np.exp(-2 * r) / np.pi) ** 2
    laplacian = (4 - 4 / r) * np.exp(-2 * r) / np.pi
    tau = np.exp(-2 * r) / (2 * np.pi)  # Hartree: (1/2)|grad psi|^2
    return rho, sigma, laplacian, tau


def _slater_1s(r):
    """The exact Slater potential of a hydrogen 1s orbital, Hartree.

    ``-(1/r)[1 - (1 + r) e^(-2r)]``: the potential of the exchange hole, which
    for a one-electron system is minus the electron's own Hartree potential.
    """
    return -(1.0 / r) * (1.0 - (1.0 + r) * np.exp(-2 * r))


def test_becke_roussel_is_exact_for_the_hydrogen_atom():
    """``v_x^BR`` on a 1s density *is* the Slater potential, pointwise.

    Not "to a tolerance" but to **machine precision** -- 6e-13, which is what
    double-precision arithmetic through an exponential, a cube root and a
    bisection leaves. Becke and Roussel's model is *exact* for a one-orbital
    density, and this is the check that it has been transcribed exactly.

    Compared where the functional is switched on. ``rho = e^(-2r)/pi`` falls
    below :data:`META_RHO_THRESHOLD` at ``r = 6.34``, and beyond that the
    density is clamped to the threshold rather than evaluated -- so the model
    potential flattens at -0.021 where the true Slater potential goes on as
    ``-1/r``. The comparison keeps a margin off that boundary, because the
    handful of points inside it where the clamp has begun to bite disagree at
    the 1e-3 level; in a calculation they contribute nothing, and evaluating
    them ungated is how a plane-wave density's vacuum noise gets into the
    potential.
    """
    r = np.linspace(1.0e-4, 20.0, 20001)
    v = np.asarray(becke_roussel_potential_hartree(*(jnp.asarray(a) for a in _hydrogen(r))))
    inside = _hydrogen(r)[0] > 10.0 * META_RHO_THRESHOLD
    assert r[_hydrogen(r)[0] > META_RHO_THRESHOLD].max() == pytest.approx(6.34, abs=0.01)
    assert np.max(np.abs(v - _slater_1s(r))[inside]) < 1.0e-11


def test_hydrogen_exchange_energy_is_minus_five_sixteenths():
    """``E_x = (1/2) int rho v_x^BR = -5/16 Ha``, the exact value.

    The energy is a *different* functional of the same solve than the potential
    -- it weights the radial profile by ``4 pi r^2 rho`` -- so it catches an
    error in the tail that a pointwise comparison at moderate ``r`` would not.
    """
    r = np.linspace(1.0e-6, 40.0, 400001)
    rho, sigma, laplacian, tau = _hydrogen(r)
    v = np.asarray(becke_roussel_potential_hartree(
        *(jnp.asarray(a) for a in (rho, sigma, laplacian, tau))
    ))
    energy = 0.5 * np.trapezoid(4 * np.pi * r**2 * rho * v, r)
    assert energy == pytest.approx(-5.0 / 16.0, abs=1.0e-5)


@pytest.mark.parametrize("gamma", [BR89_GAMMA])
def test_hydrogen_is_independent_of_gamma(gamma):
    """``D = 2 tau - |grad rho|^2/(4 rho)`` vanishes for one orbital.

    So ``Q`` is ``lap rho / 6`` whatever ``gamma`` is, and the test above pins
    everything *except* ``gamma``. Stated as its own check because it is the
    reason the uniform-gas test below is the one that pins it.
    """
    r = np.linspace(1.0e-4, 6.0, 2001)  # inside the density gate, as above
    args = tuple(jnp.asarray(a) for a in _hydrogen(r))
    reference = becke_roussel_potential_hartree(*args, gamma=gamma)
    other = becke_roussel_potential_hartree(*args, gamma=1.0)
    assert np.max(np.abs(np.asarray(reference - other))) < 1.0e-9


def _uniform(rho_sigma):
    """``(rho, sigma, lap, tau)`` for a spin channel of a uniform gas, Hartree.

    ``tau = (3/10)(6 pi^2)^(2/3) rho_sigma^(5/3)``, which is the spin-resolved
    Thomas-Fermi value; the gradient and the Laplacian vanish.
    """
    tau = 0.3 * (6 * np.pi**2) ** (2.0 / 3.0) * rho_sigma ** (5.0 / 3.0)
    return rho_sigma, 0.0, 0.0, tau


@pytest.mark.parametrize("rho_sigma", [0.01, 0.1, 1.0, 5.0, 50.0])
def test_becke_johnson_reproduces_lda_exchange_for_a_uniform_gas(rho_sigma):
    """``v_x^BJ(c = 1) = v_x^LDA = -(6 rho_sigma/pi)^(1/3)``, to 6e-4.

    The construction: the second term of the Becke-Johnson potential evaluates
    in the uniform limit to ``+(1/2)(6 rho/pi)^(1/3)`` exactly -- the
    ``sqrt(5/12) sqrt(3/5) = 1/2`` is where the odd-looking constant comes from
    -- and the Slater potential to ``-(3/2)(6 rho/pi)^(1/3)``, so the sum is
    ``-(6 rho/pi)^(1/3)`` if and only if ``v_x^BR`` *is* the Slater potential
    there. It is not quite, and the 6e-4 is exactly that gap.
    """
    v = float(tb09_potential(*(jnp.asarray(a) for a in _uniform(rho_sigma)), 1.0)) / E2
    lda = -((6 * rho_sigma / np.pi) ** (1.0 / 3.0))
    assert v == pytest.approx(lda, rel=1.0e-3)
    # ...and it is a *systematic* 6e-4, scale-free, not a numerical wobble.
    assert abs(v - lda) / abs(lda) == pytest.approx(5.996e-4, rel=1.0e-2)


def test_gamma_is_the_uniform_gas_fit():
    """0.8 is (all but exactly) the ``gamma`` that makes the uniform limit right.

    Becke and Roussel's ``gamma`` is usually presented as an empirical fit to
    atomic exchange energies. Whatever it was fitted to, ``v_x^BR`` against the
    uniform gas's Slater potential is 1.0281 at ``gamma = 0.6``, 0.9996 at 0.8
    and 0.9745 at 1.0 -- so 0.8 sits on the crossing to four digits, and the
    default is not arbitrary. This is what makes the 6e-4 above a *property*
    rather than a bug: no ``gamma`` in the family does better than about 4e-4.
    """
    slater = -1.5 * (6.0 / np.pi) ** (1.0 / 3.0)
    ratios = {
        gamma: float(becke_roussel_potential_hartree(
            *(jnp.asarray(a) for a in _uniform(1.0)), gamma=gamma
        )) / slater
        for gamma in (0.6, 0.8, 1.0)
    }
    assert ratios[0.6] == pytest.approx(1.0281, abs=1.0e-3)
    assert ratios[0.8] == pytest.approx(1.0000, abs=1.0e-3)
    assert ratios[1.0] == pytest.approx(0.9745, abs=1.0e-3)


@pytest.mark.parametrize("q", [-3.0, -0.5, -1.0e-6, 1.0e-6, 0.5, 3.0, 100.0])
def test_becke_roussel_x_solves_its_own_equation(q):
    """``x e^(-2x/3)/(x - 2) = (2/3) pi^(2/3)/Q`` at the returned ``x``.

    Written as the residual ``x e^(-2x/3) - y(x - 2)`` and not as the ratio, so
    that a root near ``x = 2`` is tested where the ratio is a pole.
    """
    x = float(becke_roussel_x(jnp.asarray(q)))
    y = (2.0 / 3.0) * np.pi ** (2.0 / 3.0) / q
    assert abs(x * np.exp(-2 * x / 3.0) - y * (x - 2.0)) < 1.0e-12 * max(1.0, abs(y))
    # ...and on the branch libxc's bracketing puts it on.
    assert (x > 2.0) if q > 0 else (0.0 <= x <= 2.0)


@pytest.mark.parametrize("q", [-3.0, -0.5, 0.5, 3.0, 20.0])
def test_becke_roussel_x_derivative_matches_a_finite_difference(q):
    """The ``custom_jvp`` against a difference of the bisection it replaces.

    A bisection's own tangent is zero, so this is the only check that the
    implicit derivative is attached *and* right -- and it is the derivative the
    Newton-Krylov SCF solver rides on, since ``d v / d tau`` goes through it.
    """
    step = 1.0e-5 * max(1.0, abs(q))
    numerical = (
        float(becke_roussel_x(jnp.asarray(q + step)))
        - float(becke_roussel_x(jnp.asarray(q - step)))
    ) / (2 * step)
    analytic = float(jax.grad(lambda z: becke_roussel_x(z))(jnp.asarray(q)))
    assert analytic == pytest.approx(numerical, rel=1.0e-6)


def test_tb09_coefficient_on_a_density_whose_average_is_known():
    """``c = alpha + beta sqrt(<|grad rho|/rho>)`` on ``rho = e^(-2z)``.

    A one-dimensional exponential has ``|grad rho|/rho = 2`` at every point, so
    the average is 2 whatever the sampling, and ``c`` is a closed form. The
    point of the test is the *gating*: the same profile taken far enough out to
    fall below the threshold must give the same answer, because the dropped
    points leave the numerator and the count together.
    """
    z = np.linspace(0.0, 3.0, 64)
    rho = np.exp(-2 * z)[None, None, :] * np.ones((4, 4, 1))
    grad = np.zeros((3,) + rho.shape)
    grad[2] = -2 * rho
    expected = TB09_ALPHA + TB09_BETA * np.sqrt(2.0)
    assert float(tb09_coefficient(jnp.asarray(rho), jnp.asarray(grad))) == pytest.approx(
        expected, rel=1.0e-12
    )

    far = np.linspace(0.0, 12.0, 256)
    rho_far = np.exp(-2 * far)[None, None, :] * np.ones((4, 4, 1))
    grad_far = np.zeros((3,) + rho_far.shape)
    grad_far[2] = -2 * rho_far
    assert float(
        tb09_coefficient(jnp.asarray(rho_far), jnp.asarray(grad_far))
    ) == pytest.approx(expected, rel=1.0e-12)


def test_thomas_fermi_tau_matches_the_free_electron_gas():
    """``potinit.f90``'s guess, in Ry, is twice the Hartree Thomas-Fermi value."""
    rho = np.array([0.05, 0.5, 5.0])
    tau = np.asarray(thomas_fermi_tau(jnp.asarray(rho)))
    hartree = 0.3 * (3 * np.pi**2) ** (2.0 / 3.0) * rho ** (5.0 / 3.0)
    assert tau == pytest.approx(E2 * hartree, rel=1.0e-12)


def test_the_functional_registry_knows_the_meta_names():
    """``tb09`` and ``bj06`` resolve, and carry the right ``c`` policy."""
    tb09 = get_functional("tb09")
    assert tb09.is_meta and tb09.name == "TB09"
    # ``c`` is a cell average, so there is no constant to report.
    assert tb09.meta_coefficient is None
    # ...and exchange is *replaced*, not corrected: a Slater term on top would
    # count exchange twice.
    assert not tb09.is_gradient
    assert get_functional("bj06").meta_coefficient == 1.0
    assert not get_functional("PBE").is_meta


def test_the_potential_is_zero_where_the_density_is():
    """Vacuum is gated, and the gate is what keeps ``grad`` finite through it.

    Every ingredient of this functional is a ratio to a power of the density, so
    an ungated evaluation in the Fourier noise of an empty region is not merely
    inaccurate -- it is ``0/0``, and one NaN there poisons a whole SCF.
    """
    rho = jnp.asarray([1.0e-12, 1.0e-3, 0.0])
    zeros = jnp.zeros(3)
    v = tb09_potential(rho, zeros, zeros, zeros, 1.0)
    assert float(v[0]) == 0.0 and float(v[2]) == 0.0
    assert float(v[1]) != 0.0
    assert np.all(np.isfinite(np.asarray(
        jax.jacfwd(lambda r: tb09_potential(r, zeros, zeros, zeros, 1.0))(rho)
    )))
