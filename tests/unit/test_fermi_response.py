"""``dE_F``: the derivative of a bisection, which is not the bisection's own.

``bisect_fermi`` finds the Fermi level by halving a bracket and then refining
with Newton. Differentiating that *search* gives nothing useful -- every number
in it is a midpoint chosen by a comparison -- so the derivative is written down
from the implicit function theorem instead (``scf/occupations.py``). These are
its tests, and they matter beyond the residual solver that motivated them: the
same term is the Fermi-level shift of metallic linear response, and any future
DFPT or implicit-differentiation route through a metal goes through it.

Central differences are the reference, as ``PLAN.md`` D5 asks.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pypresso.scf.occupations import _wgauss_prime, bisect_fermi, w0gauss

SMEARINGS = [(0, "gaussian"), (1, "methfessel-paxton"), (-1, "cold"), (-99, "fermi-dirac")]


@pytest.fixture
def levels():
    """A metallic spectrum: eight bands at four k-points, six electrons."""
    rng = np.random.default_rng(0)
    return (
        jnp.asarray(np.sort(rng.normal(size=(4, 8)))),
        jnp.asarray(np.full(4, 0.5)),
        6.0,
        0.05,
    )


@pytest.mark.unit
@pytest.mark.parametrize("ngauss,name", SMEARINGS)
def test_wgauss_derivative_is_w0gauss(ngauss, name):
    """The rule uses ``jax.grad(wgauss)``; ``w0gauss`` is the transcription of
    the same derivative from ``Modules/w0gauss.f90``. They must be the same
    function, and this is what pins the sign convention that
    ``wgauss``'s own docstring warns about."""
    x = jnp.linspace(-4.0, 4.0, 41)
    assert np.allclose(_wgauss_prime(x, ngauss), w0gauss(x, ngauss), atol=1e-14)


@pytest.mark.unit
@pytest.mark.parametrize("ngauss,name", SMEARINGS)
def test_fermi_level_responds_to_the_eigenvalues(levels, ngauss, name):
    eigenvalues, weights, nelec, degauss = levels
    rng = np.random.default_rng(1)
    direction = jnp.asarray(rng.normal(size=eigenvalues.shape))

    def level(e):
        return bisect_fermi(e, weights, nelec, degauss, ngauss)

    _, tangent = jax.jvp(level, (eigenvalues,), (direction,))
    h = 1.0e-6
    difference = (level(eigenvalues + h * direction) - level(eigenvalues - h * direction)) / (2 * h)
    assert float(tangent) == pytest.approx(float(difference), abs=1e-8)
    # ...and it is not the zero the bisection would have given on its own.
    assert abs(float(tangent)) > 1e-3


@pytest.mark.unit
def test_fermi_level_responds_to_nelec_and_degauss(levels):
    """The other two tangent slots. They are not exercised by the SCF residual,
    which moves only the eigenvalues, and are written because a rule that is
    silently wrong in a slot nobody uses is a trap for whoever uses it next."""
    eigenvalues, weights, nelec, degauss = levels
    h = 1.0e-6

    def by_electrons(n):
        return bisect_fermi(eigenvalues, weights, n, degauss, 0)

    def by_smearing(g):
        return bisect_fermi(eigenvalues, weights, nelec, g, 0)

    for function, point in ((by_electrons, nelec), (by_smearing, degauss)):
        _, tangent = jax.jvp(function, (point,), (1.0,))
        difference = (function(point + h) - function(point - h)) / (2 * h)
        assert float(tangent) == pytest.approx(float(difference), abs=1e-7)


@pytest.mark.unit
def test_gapped_spectrum_has_no_fermi_response(levels):
    """With a wide gap and a narrow smearing there is no density of states at
    the Fermi level, the implicit derivative's denominator vanishes, and the
    tangent is forced to zero rather than to an infinity."""
    _, weights, nelec, _ = levels
    eigenvalues = jnp.asarray(np.tile([-5.0, -5.0, -5.0, 5.0, 5.0, 5.0, 5.0, 5.0], (4, 1)))
    rng = np.random.default_rng(2)
    direction = jnp.asarray(rng.normal(size=eigenvalues.shape))
    _, tangent = jax.jvp(
        lambda e: bisect_fermi(e, weights, nelec, 0.01, 0), (eigenvalues,), (direction,)
    )
    assert float(tangent) == 0.0
