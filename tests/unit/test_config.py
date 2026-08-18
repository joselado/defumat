"""The precision policy is the mechanism that keeps single precision viable.

These tests exist to catch the failure mode the policy is meant to prevent:
somewhere down the line a dtype gets hardcoded and the float32 path silently
becomes float64 (or worse, x64 gets disabled and float64 silently becomes
float32).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pypresso
from pypresso.config import DOUBLE, SINGLE, precision_by_name

pytestmark = pytest.mark.unit


def test_x64_is_enabled_on_import():
    assert jax.config.jax_enable_x64 is True
    assert jnp.zeros(1).dtype == jnp.float64


@pytest.mark.parametrize(
    ("precision", "real", "complex_"),
    [(DOUBLE, np.float64, np.complex128), (SINGLE, np.float32, np.complex64)],
)
def test_policy_controls_dtypes(precision, real, complex_):
    assert precision.zeros(3).dtype == real
    assert precision.zeros(3, complex_=True).dtype == complex_
    assert precision.as_real([1, 2]).dtype == real
    assert precision.as_complex([1, 2]).dtype == complex_


def test_single_precision_is_actually_single():
    """Guards against x64 quietly promoting the float32 path back to float64."""
    x = SINGLE.as_complex(np.arange(4))
    assert (x * SINGLE.as_real(2.0)).dtype == np.complex64
    assert jnp.fft.fftn(x).dtype == np.complex64


def test_precision_is_a_static_pytree_field():
    """A Precision must not become a traced leaf: it carries no arrays."""
    leaves = jax.tree_util.tree_leaves(DOUBLE)
    assert leaves == []


def test_precision_by_name():
    assert precision_by_name("double") is DOUBLE
    assert precision_by_name("float32") is SINGLE
    with pytest.raises(ValueError, match="unknown precision"):
        precision_by_name("quadruple")


def test_equinox_module_survives_jit_and_grad():
    """The OO pattern the codebase is built on has to work end to end."""

    class Toy(eqx.Module):
        values: jnp.ndarray
        scale: float = eqx.field(static=True)

        def energy(self):
            return self.scale * jnp.vdot(self.values, self.values).real

    toy = Toy(values=DOUBLE.as_complex(np.arange(3)), scale=2.0)
    assert eqx.filter_jit(lambda t: t.energy())(toy) == pytest.approx(10.0)

    grad = jax.grad(lambda v: Toy(values=v, scale=2.0).energy())(DOUBLE.as_real(np.arange(3.0)))
    assert grad == pytest.approx([0.0, 4.0, 8.0])
    assert grad.dtype == np.float64


def test_package_exports():
    assert pypresso.__version__
    assert pypresso.DOUBLE is DOUBLE
