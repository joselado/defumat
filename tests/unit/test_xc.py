"""The exchange-correlation functional, and the units trap inside it."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pypresso.units import E2
from pypresso.xc.lda import (
    correlation_energy_density,
    exchange_energy_density,
    wigner_seitz_radius,
    xc_energy_density,
    xc_potential,
)

pytestmark = pytest.mark.unit

DENSITIES = np.array([1e-3, 3e-3, 0.01, 0.05, 0.2, 1.0, 5.0])


def _qe_reference(rho):
    """QE's own analytic expressions, transcribed independently.

    XClib returns Hartree and ``v_of_rho`` multiplies by ``e2``; the factor is
    applied here so the comparison is in Ry, matching what this module returns.
    """
    rs = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
    f, alpha = -0.687247939924714, 2.0 / 3.0
    ex, vx = f * alpha / rs, 4.0 / 3.0 * f * alpha / rs

    a, b, c, d = 0.0311, -0.048, 0.0020, -0.0116
    gc, b1, b2 = -0.1423, 1.0529, 0.3334
    if rs < 1.0:
        lnrs = np.log(rs)
        ec = a * lnrs + b + c * rs * lnrs + d * rs
        vc = a * lnrs + (b - a / 3.0) + 2.0 / 3.0 * c * rs * lnrs + (2.0 * d - c) / 3.0 * rs
    else:
        root = np.sqrt(rs)
        ox = 1.0 + b1 * root + b2 * rs
        dox = 1.0 + 7.0 / 6.0 * b1 * root + 4.0 / 3.0 * b2 * rs
        ec = gc / ox
        vc = ec * dox / ox
    return E2 * (ex + ec), E2 * (vx + vc)


@pytest.mark.parametrize("rho", DENSITIES)
def test_energy_density_matches_quantum_espresso(rho):
    expected, _ = _qe_reference(rho)
    assert float(xc_energy_density(jnp.array(rho))) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("rho", DENSITIES)
def test_potential_from_autodiff_matches_the_hand_derived_formula(rho):
    """The point of writing only the energy density.

    QE derives ``v_xc`` by hand in a separate routine; here it is ``grad`` of
    ``rho e_xc``. That the two agree to machine precision is what licenses
    dropping the hand-derived version -- and it is checked on both sides of the
    ``rs = 1`` branch.
    """
    _, expected = _qe_reference(rho)
    assert float(xc_potential(jnp.array([rho]))[0]) == pytest.approx(expected, rel=1e-10)


def test_both_branches_of_the_correlation_functional_are_exercised():
    """rs = 1 separates the high-density expansion from the interpolation."""
    assert float(wigner_seitz_radius(jnp.array(0.2387))) == pytest.approx(1.0, rel=1e-3)
    assert np.any(np.asarray(wigner_seitz_radius(jnp.asarray(DENSITIES))) < 1.0)
    assert np.any(np.asarray(wigner_seitz_radius(jnp.asarray(DENSITIES))) > 1.0)


def test_exchange_scales_as_the_cube_root_of_the_density():
    """e_x proportional to rho^(1/3) is exact for Slater exchange."""
    ratio = float(exchange_energy_density(jnp.array(8.0)) / exchange_energy_density(jnp.array(1.0)))
    assert ratio == pytest.approx(2.0, rel=1e-12)


def test_correlation_is_negative_and_smaller_than_exchange():
    for rho in DENSITIES:
        ec = float(correlation_energy_density(jnp.array(rho)))
        ex = float(exchange_energy_density(jnp.array(rho)))
        assert ec < 0.0 and ex < 0.0 and abs(ec) < abs(ex)


def test_vacuum_is_handled_without_nan():
    """Empty regions of a cell have essentially zero density; the logarithm in
    the high-density branch must not be allowed to reach them."""
    rho = jnp.array([0.0, 1e-30, 1e-12])
    assert np.all(np.isfinite(np.asarray(xc_energy_density(rho))))
    assert np.asarray(xc_potential(rho)) == pytest.approx(np.zeros(3))


def test_second_derivative_exists():
    """The response kernel f_xc = d^2(rho e_xc)/drho^2 is what a dielectric
    response needs; it must survive a second differentiation."""
    kernel = jax.grad(jax.grad(lambda r: r * xc_energy_density(r)))(0.05)
    assert np.isfinite(float(kernel)) and float(kernel) != 0.0
