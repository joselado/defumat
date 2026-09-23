"""The spin gradient correction's potential at a saturated point.

``gcc_spin`` clamps ``|zeta|`` to ``1 - rho_threshold_gga`` and ``pbec_spin``
then evaluates its analytic ``v1c_up``/``v1c_dw`` **at the clamped zeta, with
the dzeta/drho terms in them** (``XClib/qe_drivers_gga.f90:1082-1105``,
``qe_funct_corr_gga.f90:525-535``). The clamp moves the point, and the
derivative there is the interior one. A clamp that selects a constant over
``(1 - 1e-6, 1]`` instead drops ``dH/dzeta . dzeta/drho_down`` from the
minority potential at every point of a saturated magnet, with the energy
unchanged.

The reference here is built without the gate or the clamp: the correlation
slot's own ``H(n, zeta, sigma)`` is differentiated at ``zeta = 1 - 1e-6`` and
put through the chain rule of ``zeta = (rho_up - rho_down) / n`` by hand.
Host-side and pointwise, no SCF.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.xc.functional import RHO_THRESHOLD_GGA, get_functional

pytestmark = pytest.mark.unit

ZETA_CLAMPED = 1.0 - RHO_THRESHOLD_GGA
RHO_UP = 0.02
GRAD_UP = 0.01


def _point(rho_down):
    """A single grid point: majority with a gradient along x, minority flat."""
    rho = jnp.array([[RHO_UP], [rho_down]])
    grad = jnp.zeros((2, 3, 1)).at[0, 0, 0].set(GRAD_UP)
    return rho, grad


def _qe_reference(functional, rho_down):
    """``(sc, v1c_up, v1c_down)`` as ``pbec_spin`` means them.

    ``H`` at the clamped zeta, its partial derivatives there, and
    ``dzeta/drho_up = (1 - zeta)/n``, ``dzeta/drho_down = -(1 + zeta)/n`` with
    the clamped zeta, which is what the analytic expression contains.
    """
    n = RHO_UP + rho_down
    sigma = GRAD_UP**2

    def h(density, zeta):
        return functional.gradient_correlation_spin(density, zeta, sigma)

    sc = h(n, ZETA_CLAMPED)
    dh_dn, dh_dzeta = jax.grad(h, argnums=(0, 1))(n, ZETA_CLAMPED)
    v_up = dh_dn + dh_dzeta * (1.0 - ZETA_CLAMPED) / n
    v_down = dh_dn - dh_dzeta * (1.0 + ZETA_CLAMPED) / n
    return float(sc), float(v_up), float(v_down)


@pytest.mark.parametrize("name", ["PBE", "PBESOL"])
@pytest.mark.parametrize("rho_down", [1e-12, 0.0])
def test_minority_potential_keeps_the_zeta_term_at_saturation(name, rho_down):
    functional = get_functional(name)
    rho, grad = _point(rho_down)
    v1, _ = functional.spin_gradient_terms(rho, grad)
    sc_ref, v_up_ref, v_down_ref = _qe_reference(functional, rho_down)

    # The minority channel has no gradient, so its exchange is gated out and
    # v1[1] is correlation alone; the majority's exchange depends only on its
    # own channel, so correlation is isolated by the energy's own split.
    correlation = lambda r: jnp.sum(
        functional._spin_correlation_energy(r, grad)
    )
    v1c = jax.grad(correlation)(rho)

    np.testing.assert_allclose(float(v1[1, 0]), v_down_ref, rtol=1e-5)
    np.testing.assert_allclose(float(v1c[1, 0]), v_down_ref, rtol=1e-5)
    np.testing.assert_allclose(float(v1c[0, 0]), v_up_ref, rtol=1e-5, atol=1e-12)
    # The energy is QE's clamped one, exactly as before the tangent was fixed.
    np.testing.assert_allclose(
        float(functional._spin_correlation_energy(rho, grad)[0]), sc_ref, rtol=1e-12
    )


def test_minority_potential_is_continuous_into_the_clamp():
    """Just inside the clamp, zeta = 1 - 2e-6, nothing is clamped and the
    potential is the interior derivative; the saturated value must be its
    continuation rather than a jump of order 0.5 Ry."""
    functional = get_functional("PBE")
    inside, _ = functional.spin_gradient_terms(*_point(2e-8))
    saturated, _ = functional.spin_gradient_terms(*_point(1e-12))
    # dH/dzeta carries (1 - zeta)^(-1/3), which is 1.26x larger at 1e-6 than
    # at 2e-6, so the two agree in sign and to within that factor.
    assert float(inside[1, 0]) > 0.0 and float(saturated[1, 0]) > 0.0
    ratio = float(saturated[1, 0]) / float(inside[1, 0])
    assert 0.9 < ratio < 1.5
