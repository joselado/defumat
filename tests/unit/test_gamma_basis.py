"""The gamma-only primitives, against the full sphere they replace.

At ``k = 0`` a state can be chosen real and ``c(-G) = conj(c(G))``, so half the
sphere carries all of it. These are the transforms and the inner product that
*consume* that storage; everything above them in the gamma path is written in
terms of these three, so they are tested on their own before anything else uses
them.

**The reference is the full sphere**, which needs no external code: the same
cell at the same cutoff, transformed the ordinary way, is the exact answer.
"""

import numpy as np
import pytest

import jax.numpy as jnp

from defumat.basis.fft import (
    force_real_g0, g_to_r, g_to_r_gamma, gamma_inner, r_to_g, r_to_g_gamma,
)
from defumat.basis.gvectors import generate_gvectors
from defumat.system.cell import Cell

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def spheres():
    cell = Cell.from_ibrav(2, celldm=[10.2, 0, 0, 0, 0, 0])
    full = generate_gvectors(cell, 40.0, gamma_only=False)
    half = generate_gvectors(cell, 40.0, gamma_only=True)
    return cell, full, half


def _band_limited(full, rng):
    """A real field that the sphere represents exactly, so a round trip is one.

    A random grid field has frequencies outside the cutoff and both spheres
    truncate them identically -- which is a fine equality test and a useless
    round-trip one. Building the field *from* the sphere removes that.
    """
    coefficients = (rng.standard_normal(full.ngm) + 1j * rng.standard_normal(full.ngm))
    box = np.zeros(int(np.prod(full.grid)), dtype=complex)
    index = np.asarray(full.fft_index)
    minus = np.asarray(full.fft_index_minus)
    box[index] = coefficients
    # Impose reality: c(-G) = conj(c(G)), and c(0) real.
    box[minus] = np.conj(coefficients)
    box[index[0]] = coefficients[0].real
    field = np.fft.ifftn(box.reshape(full.grid)) * np.prod(full.grid)
    return jnp.asarray(field.real)


def test_the_half_sphere_holds_half_the_vectors(spheres):
    """``(ngm_full + 1)/2`` -- G = 0 is kept exactly once."""
    _, full, half = spheres
    assert half.ngm == (full.ngm + 1) // 2
    assert tuple(full.grid) == tuple(half.grid)


def test_g_zero_is_its_own_conjugate_partner(spheres):
    """``nlm[0] == nl[0]``, which is why every caller has to skip it.

    Adding both would write ``G = 0`` into the box twice, and the scatter
    accumulates rather than sets.
    """
    _, _, half = spheres
    assert int(half.fft_index[0]) == int(half.fft_index_minus[0])


def test_the_gamma_transform_is_the_full_sphere_transform(spheres):
    """The whole claim of the storage, in one equality.

    A real field's half-sphere coefficients rebuild exactly the field its full
    sphere does -- so anything written against the full transform is correct
    against this one.
    """
    _, full, half = spheres
    field = _band_limited(full, np.random.default_rng(0))

    whole = g_to_r(r_to_g(field, full.fft_index), full.fft_index, full.grid)
    stored = r_to_g_gamma(field, half.fft_index)
    rebuilt = g_to_r_gamma(stored, half.fft_index, half.fft_index_minus, half.grid)

    np.testing.assert_allclose(np.asarray(rebuilt), np.asarray(whole.real), atol=1e-13)
    # ... and it is a genuine round trip, the field being band-limited.
    np.testing.assert_allclose(np.asarray(rebuilt), np.asarray(field), atol=1e-13)


def test_the_inner_product_doubles_the_half_and_counts_g_zero_once(spheres):
    """``2 Re sum - Re(conj(a_0) b_0)`` against the full-sphere sum.

    The ``G = 0`` correction is the term that gets dropped in one call site out
    of ten, and only an energy comparison notices -- hence one helper and this
    test of it.
    """
    _, full, half = spheres
    rng = np.random.default_rng(1)
    a = _band_limited(full, rng)
    b = _band_limited(full, rng)

    reference = complex(jnp.sum(r_to_g(a, full.fft_index).conj()
                                * r_to_g(b, full.fft_index)))
    gamma = float(gamma_inner(r_to_g_gamma(a, half.fft_index),
                              r_to_g_gamma(b, half.fft_index), True))
    assert reference.imag == pytest.approx(0.0, abs=1e-14)
    assert gamma == pytest.approx(reference.real, abs=1e-14)


def test_dropping_the_g_zero_correction_is_visible(spheres):
    """The correction is not round-off, so a test that ignores it would pass.

    Pinned because the failure it guards is quiet: without it every overlap is
    wrong by one G-component, which shifts an eigenvalue rather than raising.
    """
    _, full, half = spheres
    rng = np.random.default_rng(2)
    a = _band_limited(full, rng)
    stored = r_to_g_gamma(a, half.fft_index)

    correct = float(gamma_inner(stored, stored, True))
    without = float(2.0 * jnp.sum(stored.conj() * stored).real)
    assert abs(without - correct) > 1e-6 * abs(correct)


def test_the_g_zero_coefficient_is_forced_real(spheres):
    """``regterg.f90:174``. An imaginary part there makes the field complex."""
    _, _, half = spheres
    coefficients = jnp.asarray(np.random.default_rng(3).standard_normal(half.ngm)
                               + 1j * np.random.default_rng(4).standard_normal(half.ngm))
    forced = force_real_g0(coefficients, True)
    assert complex(forced[0]).imag == 0.0
    # ... and nothing else is touched.
    np.testing.assert_array_equal(np.asarray(forced[1:]), np.asarray(coefficients[1:]))
    # A non-gamma set passes through untouched, including G = 0.
    np.testing.assert_array_equal(
        np.asarray(force_real_g0(coefficients, False)), np.asarray(coefficients)
    )
