"""The radial Bessel transform and its derivative rule (``formfactors.bessel_transform``).

The rule says the ``q``-derivative of ``sum_m h_m r_m^n j_l^(n)(q r_m)`` is the
same transform one order up, and is what keeps a strain's second derivative from
holding the ``(chunk, mesh)`` kernel matrix. Three things are checked here:

* every order against a transform with a closed form, which shares nothing with
  the code: ``int r^(l+2) exp(-a r^2) j_l(q r) dr = sqrt(pi) q^l /
  (2^(l+2) a^(l+3/2)) exp(-q^2 / 4a)``;
* the memory claim itself, as the compiled temporaries of a ``jvp`` of a
  gradient along a strain, which the rematted scan the rule replaced held at the
  kernel matrix's size;
* that differentiating the radial mesh is refused rather than answered wrongly.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.pseudo.formfactors import bessel_transform
from defumat.pseudo.radial import simpson_weights, spherical_bessel

pytestmark = pytest.mark.unit

A = 0.7


def _mesh():
    # QE's logarithmic mesh shape, fine enough that Simpson's rule is exact to
    # 1e-13 on a Gaussian
    x = np.arange(2001)
    r = 1e-5 * np.exp(x * 0.0085)
    rab = r * 0.0085
    cut = int(np.searchsorted(r, 12.0)) | 1
    return jnp.asarray(r[:cut]), simpson_weights(rab[:cut])


def _closed_form(l, q):
    return (math.sqrt(math.pi) * q**l / (2 ** (l + 2) * A ** (l + 1.5))
            * jnp.exp(-q**2 / (4 * A)))


@pytest.mark.parametrize("l", [0, 1, 2, 3, 4])
def test_every_order_matches_the_closed_form(l):
    r, w = _mesh()
    h = w * r ** (l + 2) * jnp.exp(-A * r**2)
    q = jnp.asarray(np.concatenate([[0.0, 1e-3, 0.05], np.linspace(0.1, 9.0, 3000)]))

    def ones(x):
        return jnp.ones_like(x)

    transform = lambda x: bessel_transform(x, r, h, l)  # noqa: E731
    exact = lambda x: _closed_form(l, x)  # noqa: E731
    got, want = transform, exact
    for order in range(4):
        scale = float(jnp.max(jnp.abs(want(q))))
        error = float(jnp.max(jnp.abs(got(q) - want(q)))) / scale
        assert error < 1e-10, (l, order, error)
        got = (lambda f: lambda x: jax.jvp(f, (x,), (ones(x),))[1])(got)
        want = (lambda f: lambda x: jax.jvp(f, (x,), (ones(x),))[1])(want)


def test_the_value_is_qes_bessel_function_and_the_reverse_mode_is_the_forward():
    r, w = _mesh()
    h = w * r**3 * jnp.exp(-A * r**2)
    q = jnp.asarray(np.linspace(0.0, 8.0, 5000))
    direct = (h[None, :] * spherical_bessel(1, q[:, None] * r[None, :])).sum(-1)
    assert float(jnp.max(jnp.abs(bessel_transform(q, r, h, 1) - direct))) < 1e-14

    # each value depends on its own q alone, so the gradient of the sum is the
    # forward derivative along all ones
    f = lambda x: jnp.sin(bessel_transform(x, r, h, 1))  # noqa: E731
    reverse = jax.grad(lambda x: jnp.sum(f(x)))(q)
    forward = jax.jvp(f, (q,), (jnp.ones_like(q),))[1]
    assert float(jnp.max(jnp.abs(reverse - forward))) < 1e-15


def test_a_strains_second_derivative_holds_no_kernel_matrix():
    """The compiled temporaries of ``jvp(grad)`` along a shear stay at the vectors' size.

    14211 values of ``q`` on an 841-point mesh, AlAs's dense set at 200 Ry: one
    ``(chunk, mesh)`` kernel matrix at the default chunk is about 8 MB, and the
    rematted scan this replaced held 209 MB of temporaries here; the rule holds
    the ``(nq,)`` slopes and one block's kernel at a time.
    """
    rng = np.random.default_rng(0)
    r = jnp.asarray(np.linspace(1e-4, 4.0, 841))
    w = jnp.asarray(rng.random(841)) * 0.005
    h = w * r**3 * jnp.exp(-r**2)
    g = jnp.asarray(rng.normal(size=(14211, 3)) * 3)
    shear = jnp.asarray([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0.0]])

    def energy(strain):
        q = jnp.sqrt(jnp.sum((g @ (jnp.eye(3) + strain)) ** 2, axis=1))
        return jnp.sum(jnp.cos(bessel_transform(q, r, h, 2)))

    gradient = jax.grad(energy)
    second = jax.jit(lambda e: jax.jvp(gradient, (e,), (shear,))[1])
    temp = second.lower(jnp.zeros((3, 3))).compile().memory_analysis().temp_size_in_bytes
    assert temp < 32 * 1024**2, temp / 1024**2


def test_differentiating_the_mesh_is_refused():
    r, w = _mesh()
    h = w * r**2 * jnp.exp(-A * r**2)
    q = jnp.asarray([0.5, 1.0])
    with pytest.raises(NotImplementedError, match="radial mesh"):
        jax.jvp(lambda rr: bessel_transform(q, rr, h, 0), (r,), (jnp.ones_like(r),))


# The transcribed stress (``stress/analytic.py``) differentiates two of these
# transforms by hand, QE's ``drhoc`` and ``dvloc_of_g``, with ``j_0' = -j_1`` and
# the derivative of ``sin(qr)/q`` written out. Neither shares anything with the
# rule, so each is an identity for it on a real dataset.
NLCC_DATASET = "Al.pbe-n-rrkjus_psl.1.0.0.UPF"
OMEGA = 300.0


def _dataset(pseudo_dir):
    from defumat.pseudo.upf import read_upf

    return read_upf(pseudo_dir / NLCC_DATASET)


def _moduli():
    return jnp.asarray(np.linspace(0.05, 12.0, 2000))


def test_the_core_charges_slope_is_qes_drhoc(pseudo_dir):
    from defumat.pseudo.formfactors import core_charge_of_g
    from defumat.stress.analytic import _drhoc_of_g

    pseudo = _dataset(pseudo_dir)
    q = _moduli()
    slope = jax.jvp(lambda x: core_charge_of_g(pseudo, x, OMEGA), (q,),
                    (jnp.ones_like(q),))[1]
    reference = _drhoc_of_g(pseudo, q, OMEGA)
    error = float(jnp.max(jnp.abs(slope - reference)) / jnp.max(jnp.abs(reference)))
    assert error < 1e-12, error


def test_the_local_potentials_slope_is_qes_dvloc(pseudo_dir):
    from defumat.pseudo.formfactors import local_potential_of_g
    from defumat.stress.analytic import _dvloc_of_g2

    pseudo = _dataset(pseudo_dir)
    q = _moduli()
    slope = jax.jvp(lambda x: local_potential_of_g(pseudo, x, OMEGA), (q,),
                    (jnp.ones_like(q),))[1]
    # QE returns dV/d(q^2), which is dV/dq / 2q
    reference = _dvloc_of_g2(pseudo, q, OMEGA) * 2.0 * q
    error = float(jnp.max(jnp.abs(slope - reference)) / jnp.max(jnp.abs(reference)))
    assert error < 1e-12, error
