"""The spin gradient correction to correlation where ``|zeta| > 1``.

``gcc_spin`` clamps ``zeta`` to ``1 - rho_threshold_gga`` only under
``ABS(zeta) <= 1`` (``XClib/qe_drivers_gga.f90:1082-1083``), so a raw
``|zeta| > 1`` survives the clamp and meets the test that CYCLEs with
``sc = v1c = v2c = 0`` (``:1086-1092``). A plane-wave density reaches such a
point wherever the minority channel is slightly negative, after mixing or from
an augmentation charge. The value there is zero, and every derivative with it,
since the potential is ``jax.grad`` of the value.

``|zeta| == 1`` exactly is **not** cut, because QE's test is on ``> 1``: a
saturated point whose minority channel is exactly zero is clamped and
evaluated. ``test_spin_gga_saturated_potential.py`` checks the potential there;
the boundary test below pins the value, so the ``<=`` is held by a test rather
than by a comment.

The references are the correlation slot's own ``H(n, zeta, sigma)`` evaluated
with no gate and no clamp, as in the neighbouring file. Host-side and
pointwise, no SCF.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.xc.functional import RHO_THRESHOLD_GGA, SMALL_SPIN_GGA, get_functional

pytestmark = pytest.mark.unit

ZETA_CLAMPED = 1.0 - RHO_THRESHOLD_GGA
GRAD_UP = 0.01
GRAD_DOWN = 0.002
FUNCTIONALS = ["PBE", "PBESOL"]


def _point(rho_up, rho_down, grad_up=GRAD_UP, grad_down=GRAD_DOWN, mirrored=False):
    """A single grid point, both channel gradients along x.

    ``mirrored`` swaps the two channels, which sends ``zeta`` to ``-zeta`` and
    checks the cut on the negative side too.
    """
    rho = jnp.array([[rho_up], [rho_down]])
    grad = jnp.zeros((2, 3, 1)).at[0, 0, 0].set(grad_up).at[1, 0, 0].set(grad_down)
    if mirrored:
        return rho[::-1], grad[::-1]
    return rho, grad


def _beyond_full_polarization(mirrored):
    """``rho = (0.02, -0.001)``: ``n = 0.019``, ``zeta = 0.021 / 0.019 = 1.105``.

    The point passes the density gate and the gradient gate by four orders of
    magnitude, so only the ``|zeta| > 1`` test can cut it.
    """
    return _point(0.02, -0.001, mirrored=mirrored)


def _correlation_derivatives(functional, rho, grad):
    """``(d sc/d rho, d sc/d grad)``: ``v1c`` and the correlation part of ``h``."""

    def total(r, g):
        return jnp.sum(functional._spin_correlation_energy(r, g))

    return jax.grad(total, argnums=(0, 1))(rho, grad)


@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("name", FUNCTIONALS)
def test_correlation_is_cut_beyond_full_polarization(name, mirrored):
    functional = get_functional(name)
    rho, grad = _beyond_full_polarization(mirrored)

    n = 0.02 - 0.001
    sigma = (GRAD_UP + GRAD_DOWN) ** 2
    assert n > RHO_THRESHOLD_GGA and np.sqrt(sigma) > RHO_THRESHOLD_GGA
    assert abs((0.02 + 0.001) / n) > 1.0
    # The guard must be seen to fire: what the clamp-and-keep evaluation would
    # have returned here is not small, so an exact zero is the cut and not a
    # vanishing H.
    kept = float(functional.gradient_correlation_spin(n, ZETA_CLAMPED, sigma))
    assert abs(kept) > 1e-5

    sc = functional._spin_correlation_energy(rho, grad)
    assert float(sc[0]) == 0.0


@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("name", FUNCTIONALS)
def test_the_cut_point_carries_no_correlation_potential(name, mirrored):
    functional = get_functional(name)
    rho, grad = _beyond_full_polarization(mirrored)

    v1c, hc = _correlation_derivatives(functional, rho, grad)
    assert np.all(np.isfinite(np.asarray(v1c))) and np.all(np.isfinite(np.asarray(hc)))
    assert np.all(np.asarray(v1c) == 0.0)
    assert np.all(np.asarray(hc) == 0.0)

    # The public route. The minority channel is below ``gcx_spin``'s own gate,
    # so its exchange is cut and whatever it carries is correlation alone,
    # which must now be zero: ``v1c_dw`` and the ``v2c`` that multiplies the
    # total gradient in ``h`` both vanish, as QE's CYCLE sets them.
    minority = 0 if mirrored else 1
    assert float(rho[minority, 0]) <= SMALL_SPIN_GGA
    v1, h = functional.spin_gradient_terms(rho, grad)
    assert np.all(np.isfinite(np.asarray(v1))) and np.all(np.isfinite(np.asarray(h)))
    assert float(v1[minority, 0]) == 0.0
    assert np.all(np.asarray(h[minority]) == 0.0)


@pytest.mark.parametrize("name", FUNCTIONALS)
def test_a_regular_point_is_unchanged(name):
    """``rho = (0.02, 0.001)``, ``zeta = 0.905``: nothing is gated or clamped,
    and the value and both derivatives are the slot's own expression put
    through ``n = rho_up + rho_down``, ``zeta = (rho_up - rho_down)/n`` and
    ``sigma = |grad rho_up + grad rho_down|^2``."""
    functional = get_functional(name)
    rho, grad = _point(0.02, 0.001)

    def reference(r, g):
        n = r[0] + r[1]
        gradient = g[0] + g[1]
        sigma = jnp.sum(gradient * gradient, axis=0)
        return jnp.sum(functional.gradient_correlation_spin(n, (r[0] - r[1]) / n, sigma))

    sc = functional._spin_correlation_energy(rho, grad)
    np.testing.assert_allclose(float(sc[0]), float(reference(rho, grad)), rtol=1e-12)
    assert float(sc[0]) != 0.0

    v1c, hc = _correlation_derivatives(functional, rho, grad)
    v1c_ref, hc_ref = jax.grad(reference, argnums=(0, 1))(rho, grad)
    np.testing.assert_allclose(np.asarray(v1c), np.asarray(v1c_ref), rtol=1e-10)
    # The y and z entries are exactly zero on both sides, which rtol alone
    # accepts, since assert_allclose's atol defaults to 0.
    np.testing.assert_allclose(np.asarray(hc), np.asarray(hc_ref), rtol=1e-10)


@pytest.mark.parametrize("name", FUNCTIONALS)
def test_exactly_full_polarization_is_kept(name):
    """``rho_down = 0`` exactly gives ``zeta = 0.02 / 0.02 = 1.0`` exactly, which
    ``gcc_spin`` clamps to ``1 - 1e-6`` and evaluates rather than cuts."""
    functional = get_functional(name)
    rho, grad = _point(0.02, 0.0, grad_down=0.0)
    assert (0.02 - 0.0) / (0.02 + 0.0) == 1.0

    expected = float(functional.gradient_correlation_spin(0.02, ZETA_CLAMPED, GRAD_UP**2))
    sc = functional._spin_correlation_energy(rho, grad)
    assert float(sc[0]) != 0.0
    np.testing.assert_allclose(float(sc[0]), expected, rtol=1e-12)
