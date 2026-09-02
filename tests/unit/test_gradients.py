"""Gradients and divergences, on the plane-wave grid and on a radial mesh.

Both are exact for the functions the code actually applies them to -- a
band-limited field in the first case, a smooth radial function in the second --
so the checks here are against analytic derivatives rather than against a
tolerance pulled out of the air.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.basis.fft import g_to_r
from defumat.basis.gradients import divergence, gradient
from defumat.basis.gvectors import generate_gvectors
from defumat.paw.angular import build_angular_grid
from defumat.paw.gradient import radial_derivative
from defumat.system.cell import Cell

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def silicon():
    """The usual fcc cell, and a G-vector set on it."""
    cell = Cell.from_ibrav(2, [10.2, 0, 0, 0, 0, 0])
    return cell, generate_gvectors(cell, 24.0)


def _field(gvectors, cell, coefficients):
    """A real field with the given coefficients on the first few G vectors."""
    field_g = jnp.zeros(gvectors.ngm, dtype=complex).at[: len(coefficients)].set(
        jnp.asarray(coefficients)
    )
    return field_g, jnp.real(g_to_r(field_g, gvectors.fft_index, gvectors.grid))


def test_gradient_of_a_single_plane_wave_is_exact(silicon):
    """``grad exp(iGr) = iG exp(iGr)``, so the answer is known in closed form.

    The comparison is against the analytic gradient of the *same* band-limited
    field, which is the only fair one: a G-space derivative is exact for what the
    basis can represent and says nothing about what it cannot.
    """
    cell, gvectors = silicon
    index = 5
    g = np.asarray(gvectors.cartesian(cell))[index]

    field_g = jnp.zeros(gvectors.ngm, dtype=complex).at[index].set(1.0)
    grad = np.asarray(gradient(field_g, gvectors, cell))

    # Re part of i G exp(iGr) = -G sin(Gr); build it from the transform itself so
    # that no assumption about the grid's coordinates is smuggled in.
    wave = np.asarray(g_to_r(field_g, gvectors.fft_index, gvectors.grid))
    for axis in range(3):
        assert grad[axis] == pytest.approx(np.real(1j * g[axis] * wave), abs=1e-12)


def test_divergence_of_the_gradient_is_minus_g_squared(silicon):
    """``div grad`` on this pair is exactly ``-|G|^2``, which is what makes the
    gradient correction consistent with the Hartree term's Laplacian."""
    cell, gvectors = silicon
    g2 = np.asarray(gvectors.kinetic(cell))

    field_g, _ = _field(gvectors, cell, [0.0, 0.3, -0.2, 0.1, 0.05])
    laplacian = divergence(gradient(field_g, gvectors, cell), gvectors, cell)

    expected = np.real(
        np.asarray(g_to_r(-jnp.asarray(g2) * field_g, gvectors.fft_index, gvectors.grid))
    )
    assert np.asarray(laplacian) == pytest.approx(expected, abs=1e-10)


def test_the_gradient_of_a_constant_vanishes(silicon):
    cell, gvectors = silicon
    field_g = jnp.zeros(gvectors.ngm, dtype=complex).at[0].set(2.5)
    assert np.asarray(gradient(field_g, gvectors, cell)) == pytest.approx(0.0, abs=1e-14)


def test_integrating_a_divergence_over_the_cell_gives_zero(silicon):
    """Gauss's theorem on a periodic cell: no boundary, no net flux."""
    cell, gvectors = silicon
    field_g, _ = _field(gvectors, cell, [0.0, 0.4, 0.1, -0.3])
    total = jnp.sum(divergence(gradient(field_g, gvectors, cell), gvectors, cell))
    assert float(total) == pytest.approx(0.0, abs=1e-8)


def test_gamma_only_storage_is_refused_rather_than_halved(silicon):
    """Half a sphere is not the transform of a real field, and quietly taking
    its gradient would be wrong by a factor and an offset."""
    cell, _ = silicon
    half = generate_gvectors(cell, 24.0, gamma_only=True)
    with pytest.raises(NotImplementedError, match="gamma-only"):
        gradient(jnp.zeros(half.ngm, dtype=complex), half, cell)


# --- the radial mesh ----------------------------------------------------------


def _logarithmic_mesh(n=400, start=-7.0, dx=0.02, zmesh=14.0):
    """A UPF-style mesh: ``r = exp(x)/Z`` at equally spaced ``x``."""
    x = start + dx * np.arange(n)
    return np.exp(x) / zmesh


@pytest.mark.parametrize(
    ("name", "f", "df"),
    [
        ("gaussian", lambda r: np.exp(-2.0 * r**2), lambda r: -4.0 * r * np.exp(-2.0 * r**2)),
        ("exponential", lambda r: np.exp(-1.5 * r), lambda r: -1.5 * np.exp(-1.5 * r)),
        ("polynomial", lambda r: r**3 - 2.0 * r, lambda r: 3.0 * r**2 - 2.0),
    ],
)
def test_radial_derivative_matches_the_analytic_one(name, f, df):
    """QE's three-point formula on an unequally spaced mesh.

    The interior is a parabola through three neighbours, so it is exact for a
    quadratic and second-order otherwise; the two ends are QE's conventions and
    are excluded here, since the outer one is deliberately zero.
    """
    r = _logarithmic_mesh()
    got = np.asarray(radial_derivative(jnp.asarray(f(r)), jnp.asarray(r)))
    expected = df(r)

    # Second-order in the mesh spacing, which on a logarithmic mesh grows with
    # r -- so the bound is set by the outermost points, not the origin.
    interior = slice(1, -1)
    scale = np.abs(expected[interior]).max()
    assert got[interior] == pytest.approx(expected[interior], abs=1e-4 * scale)
    assert got[-1] == 0.0


def test_radial_derivative_is_exact_for_a_quadratic():
    """The formula is a differentiated parabola, so this is exact rather than
    approximate -- and it is the check that catches an index off by one."""
    r = _logarithmic_mesh()
    f = 3.0 * r**2 + 2.0 * r + 1.0
    got = np.asarray(radial_derivative(jnp.asarray(f), jnp.asarray(r)))
    assert got[1:-1] == pytest.approx(6.0 * r[1:-1] + 2.0, rel=1e-9)


def test_radial_derivative_works_on_a_stack_of_functions():
    """The PAW path differentiates one function per direction at once."""
    r = _logarithmic_mesh()
    stack = jnp.asarray(np.stack([np.exp(-r), np.exp(-2.0 * r), r**2]))
    got = np.asarray(radial_derivative(stack, jnp.asarray(r)))
    for row, f in enumerate([lambda x: -np.exp(-x), lambda x: -2 * np.exp(-2 * x),
                             lambda x: 2 * x]):
        single = np.asarray(radial_derivative(stack[row], jnp.asarray(r)))
        assert got[row] == pytest.approx(single, rel=1e-14)
        assert got[row][1:-1] == pytest.approx(f(r)[1:-1], abs=1e-4)


# --- the angular grid ---------------------------------------------------------


def test_the_gradient_grid_is_larger_than_the_local_one():
    """``paw_init`` adds ``xlm = 2`` to the quadrature when the functional is a
    GGA, and tabulates the harmonics two multipoles further."""
    local = build_angular_grid(2, 9, gradient=False)
    corrected = build_angular_grid(2, 9, gradient=True)

    assert local.nx == 28  # lmax = 6: 4 Gauss nodes x 7 phi points
    assert corrected.nx == 45  # lmax = 8: 5 x 9
    assert local.ylm.shape[1] == 9
    assert corrected.ylm.shape[1] == 25  # (2 + 2 + 1)^2


def test_the_quadrature_is_orthonormal_on_the_harmonics_it_carries():
    grid = build_angular_grid(2, 9, gradient=True)
    overlap = np.einsum(
        "x,xl,xm->lm", np.asarray(grid.weights), np.asarray(grid.ylm), np.asarray(grid.ylm)
    )
    assert overlap == pytest.approx(np.eye(overlap.shape[0]), abs=1e-12)


def test_the_angular_derivatives_are_the_derivatives_of_the_harmonics():
    """``dylmt`` and ``dylmp`` against finite differences in theta and phi.

    QE builds these by finite-differencing ``ylmr2`` in each cartesian direction
    and projecting onto the versors; here the cartesian derivative is exact and
    only the projection is transcribed, so the check is against the angles
    directly.
    """
    from defumat.pseudo.harmonics import real_spherical_harmonics

    grid = build_angular_grid(2, 9, gradient=True)
    lmax, nlm = 4, 25
    nphi = 9
    step = 1e-6

    def harmonics(theta, phi):
        direction = jnp.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )
        return np.asarray(real_spherical_harmonics(direction, lmax))[:nlm]

    for ix in (0, 5, 17, 40):
        theta = float(np.arccos(np.asarray(grid.cos_theta)[ix]))
        phi = 2.0 * np.pi * (ix % nphi) / nphi
        polar = (harmonics(theta + step, phi) - harmonics(theta - step, phi)) / (2 * step)
        azimuthal = (harmonics(theta, phi + step) - harmonics(theta, phi - step)) / (
            2 * step * np.sin(theta)
        )
        assert np.asarray(grid.dylmt)[ix] == pytest.approx(polar, abs=1e-7)
        assert np.asarray(grid.dylmp)[ix] == pytest.approx(azimuthal, abs=1e-7)
