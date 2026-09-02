"""Sphere <-> box transforms: the conventions, and the traps in the padding."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.basis.fft import g_to_r, gather_from_box, r_to_g, scatter_to_box
from defumat.basis.gvectors import generate_gvectors
from defumat.config import DOUBLE
from defumat.system.cell import Cell

pytestmark = pytest.mark.unit

SILICON = Cell.from_ibrav(2, [10.2, 0, 0, 0, 0, 0])


@pytest.fixture(scope="module")
def gvectors():
    return generate_gvectors(SILICON, 48.0)


def _random(shape, seed=0):
    rng = np.random.default_rng(seed)
    return DOUBLE.as_complex(rng.normal(size=shape) + 1j * rng.normal(size=shape))


def test_round_trip(gvectors):
    coefficients = _random(gvectors.ngm)
    recovered = r_to_g(g_to_r(coefficients, gvectors.fft_index, gvectors.grid), gvectors.fft_index)
    assert np.asarray(recovered) == pytest.approx(np.asarray(coefficients), abs=1e-12)


def test_parseval(gvectors):
    """With QE's scaling, sum_G |c|^2 == (1/N) sum_r |f(r)|^2."""
    coefficients = _random(gvectors.ngm, seed=1)
    field = g_to_r(coefficients, gvectors.fft_index, gvectors.grid)
    n = np.prod(gvectors.grid)
    assert float(jnp.sum(jnp.abs(field) ** 2) / n) == pytest.approx(
        float(jnp.sum(jnp.abs(coefficients) ** 2))
    )


def test_constant_coefficient_gives_a_constant_field(gvectors):
    """Only G = 0 populated -> a uniform field equal to that coefficient. This
    pins the scaling convention: any stray factor of N would show up here."""
    coefficients = jnp.zeros(gvectors.ngm, dtype=jnp.complex128).at[0].set(2.5 + 0j)
    field = g_to_r(coefficients, gvectors.fft_index, gvectors.grid)
    assert np.asarray(field) == pytest.approx(np.full(gvectors.grid, 2.5))


def test_single_plane_wave_has_the_right_phase(gvectors):
    """c_G = 1 for one G -> f(r) = exp(i G . r) sampled on the grid."""
    which = 5
    coefficients = jnp.zeros(gvectors.ngm, dtype=jnp.complex128).at[which].set(1.0)
    field = np.asarray(g_to_r(coefficients, gvectors.fft_index, gvectors.grid))

    miller = np.asarray(gvectors.miller[which])
    grid = gvectors.grid
    axes = [np.arange(n) / n for n in grid]
    phase = np.exp(
        2j
        * np.pi
        * (
            miller[0] * axes[0][:, None, None]
            + miller[1] * axes[1][None, :, None]
            + miller[2] * axes[2][None, None, :]
        )
    )
    assert field == pytest.approx(phase, abs=1e-12)


def test_batching_over_leading_axes(gvectors):
    """Bands and k-points become leading axes; the transform must not care."""
    coefficients = _random((3, 4, gvectors.ngm), seed=2)
    field = g_to_r(coefficients, gvectors.fft_index, gvectors.grid)
    assert field.shape == (3, 4, *gvectors.grid)

    single = g_to_r(coefficients[1, 2], gvectors.fft_index, gvectors.grid)
    assert np.asarray(field[1, 2]) == pytest.approx(np.asarray(single))


def test_scatter_accumulates_so_padding_cannot_clobber_g_zero(gvectors):
    """Padded plane waves share the index of G = 0. With an accumulating
    scatter, zero-valued padding contributes nothing; a plain overwrite would
    silently destroy the G = 0 coefficient -- the average of the field."""
    index = jnp.concatenate([gvectors.fft_index, jnp.zeros(5, dtype=jnp.int32)])
    coefficients = jnp.concatenate(
        [jnp.ones(gvectors.ngm, dtype=jnp.complex128), jnp.zeros(5, dtype=jnp.complex128)]
    )
    box = scatter_to_box(coefficients, index, gvectors.grid)
    assert complex(box[0, 0, 0]) == pytest.approx(1.0)


def test_gather_inverts_scatter(gvectors):
    coefficients = _random(gvectors.ngm, seed=3)
    box = scatter_to_box(coefficients, gvectors.fft_index, gvectors.grid)
    assert np.asarray(gather_from_box(box, gvectors.fft_index)) == pytest.approx(
        np.asarray(coefficients)
    )
    assert box.shape == gvectors.grid


def test_transforms_are_differentiable(gvectors):
    """Response properties differentiate through the FFT, so it must not be a
    barrier to grad."""

    def energy(coefficients):
        field = g_to_r(coefficients, gvectors.fft_index, gvectors.grid)
        return jnp.sum(jnp.abs(field) ** 2).real

    coefficients = _random(gvectors.ngm, seed=4)
    gradient = jax.grad(energy)(coefficients)
    assert gradient.shape == coefficients.shape
    # By Parseval the energy is N * sum_G |c_G|^2, so the derivative has
    # magnitude 2N|c|. JAX returns the *conjugate* gradient for a real-valued
    # function of complex inputs (the steepest-ascent direction), which is the
    # convention any later response-property code has to expect.
    n = np.prod(gvectors.grid)
    assert np.asarray(gradient) == pytest.approx(2 * n * np.conj(np.asarray(coefficients)), rel=1e-9)


def test_jit_compiles_and_agrees(gvectors):
    coefficients = _random(gvectors.ngm, seed=5)
    jitted = jax.jit(lambda c: g_to_r(c, gvectors.fft_index, gvectors.grid))
    assert np.asarray(jitted(coefficients)) == pytest.approx(
        np.asarray(g_to_r(coefficients, gvectors.fft_index, gvectors.grid))
    )
