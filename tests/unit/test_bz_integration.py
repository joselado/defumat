"""P8 unit checks: Brillouin-zone integration -- tetrahedra, and the smeared delta.

Nothing here needs a QE reference. The tetrahedron method is checked against a
band whose Brillouin-zone integral is known in closed form -- a free-electron
band, whose occupied region is a sphere -- and against the invariants the method
has to satisfy whatever the band structure is.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pypresso.scf.occupations import (
    smeared_occupations,
    w0gauss,
    wgauss,
)
from pypresso.scf.tetrahedra import (
    build_tetrahedra,
    integrated_states,
    tetrahedron_dos,
    tetrahedron_kind,
    tetrahedron_occupations,
    _sorted_corners,
)
from pypresso.system.kpoints import (
    DEGSPIN,
    grid_equivalence,
    irreducible_wedge,
    monkhorst_pack,
)
from pypresso.units import SQRT_PI

pytestmark = pytest.mark.unit

KINDS = ["bloechl", "linear", "optimized"]


def _free_electron(n: int):
    """A free-electron band on an ``n^3`` grid of a cubic cell.

    ``e(k) = |k|^2`` with ``k`` in units of ``2 pi / a``. Below ``E = 0.25`` the
    occupied region is a sphere entirely inside the zone, so the exact number of
    states is the sphere's volume -- which is what makes this a closed-form check
    on the tetrahedron geometry rather than a self-consistency check.
    """
    points, weights = monkhorst_pack((n, n, n))
    eigenvalues = np.sum(points**2, axis=1)[:, None]
    return jnp.asarray(eigenvalues), jnp.asarray(weights * DEGSPIN)


def _tetrahedra(kind: str, n: int):
    """Tetrahedra on the *unreduced* grid, so k-point ``i`` is grid point ``i``.

    ``time_reversal=False`` matters: with it on, the identity alone still folds
    ``-k`` onto ``k`` and the "irreducible" list would be half the grid.
    """
    return build_tetrahedra(
        kind, (n, n, n), (0, 0, 0), np.eye(3, dtype=int)[None], np.eye(3),
        time_reversal=False,
    )


def _exact_states(energy: float) -> float:
    """``DEGSPIN`` times the volume of the sphere ``|k|^2 < E`` inside the zone."""
    return DEGSPIN * 4.0 / 3.0 * np.pi * energy**1.5


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_grid_equivalence_agrees_with_irreducible_wedge():
    """The two copies of QE's orbit walk must not drift apart.

    ``grid_equivalence`` repeats ``irreducible_wedge``'s loop because the
    contract forbids refactoring the latter; this pins them together by checking
    that the multiplicities implied by the map are the wedge's own weights.
    """
    rotations = np.array(
        [np.eye(3, dtype=int), -np.eye(3, dtype=int), np.diag([1, -1, -1]), np.diag([-1, 1, -1])]
    )
    for grid, shift in [((4, 4, 4), (0, 0, 0)), ((6, 6, 6), (1, 1, 1)), ((3, 4, 5), (0, 1, 0))]:
        points, weights = irreducible_wedge(grid, shift, rotations)
        equiv = grid_equivalence(grid, shift, rotations)

        assert equiv.shape == (int(np.prod(grid)),)
        assert equiv.max() == len(points) - 1
        multiplicity = np.bincount(equiv, minlength=len(points))
        assert multiplicity.sum() == np.prod(grid)
        assert multiplicity / multiplicity.sum() == pytest.approx(weights)


@pytest.mark.parametrize("kind", KINDS)
def test_tetrahedra_cover_the_zone_six_times_per_microcell(kind):
    tetra = _tetrahedra(kind, 4)
    assert tetra.ntetra == 6 * 4**3
    assert tetra.nntetra == (20 if kind == "optimized" else 4)
    # Every row of the interpolation stencil is a partition of unity; that is
    # what makes the integrated DOS reach exactly one per tetrahedron.
    assert np.asarray(jnp.sum(tetra.wlsm, axis=1)) == pytest.approx(np.ones(4))
    # Each k-point sits at the corner of the same number of tetrahedra.
    counts = np.bincount(np.asarray(tetra.corners[:, :4]).ravel())
    assert set(counts.tolist()) == {24}


def test_tetrahedron_kind_maps_qe_keywords():
    assert tetrahedron_kind("tetrahedra") == "bloechl"
    assert tetrahedron_kind("tetrahedra-lin") == "linear"
    assert tetrahedron_kind("tetrahedra_opt") == "optimized"
    with pytest.raises(ValueError):
        tetrahedron_kind("tetrahedra-bloechl")


# --------------------------------------------------------------------------
# The integrated density of states
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_integrated_states_reproduce_the_free_electron_sphere(kind):
    """N(E) is the volume of a sphere, and the tetrahedra had better find it."""
    tetra = _tetrahedra(kind, 20)
    eigenvalues, weights = _free_electron(20)
    e_sorted, _ = _sorted_corners(tetra, eigenvalues)

    for energy in (0.06, 0.12, 0.2):
        got = float(
            integrated_states(e_sorted, energy, tetra.ntetra, jnp.sum(weights))
        )
        # A piecewise-linear band inscribes the sphere, so the error is one-sided
        # and set by how many grid spacings the radius spans.
        assert got == pytest.approx(_exact_states(energy), rel=4e-2)


def test_optimized_tetrahedra_are_the_most_accurate():
    """Kawamura's stencil is worth two orders of magnitude on a smooth band.

    This is the whole point of ``tetrahedra-opt``, and it is also the sharpest
    available check that the 20-point ``wlsm`` table was transcribed correctly:
    a single wrong entry breaks the cancellation and the advantage disappears.
    """
    energy = 0.12
    errors = {}
    for kind in KINDS:
        tetra = _tetrahedra(kind, 20)
        eigenvalues, weights = _free_electron(20)
        e_sorted, _ = _sorted_corners(tetra, eigenvalues)
        got = float(integrated_states(e_sorted, energy, tetra.ntetra, jnp.sum(weights)))
        errors[kind] = abs(got / _exact_states(energy) - 1.0)
    assert errors["optimized"] < 1e-3
    assert errors["optimized"] < 0.05 * errors["linear"]


@pytest.mark.parametrize("kind", KINDS)
def test_integrated_states_converge_with_the_grid(kind):
    """The error must fall as the grid is refined -- the point of the method."""
    energy = 0.06
    errors = []
    for n in (6, 12, 24):
        tetra = _tetrahedra(kind, n)
        eigenvalues, weights = _free_electron(n)
        e_sorted, _ = _sorted_corners(tetra, eigenvalues)
        got = float(integrated_states(e_sorted, energy, tetra.ntetra, jnp.sum(weights)))
        errors.append(abs(got - _exact_states(energy)))
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]


@pytest.mark.parametrize("kind", KINDS)
def test_dos_integrates_to_the_integrated_dos(kind):
    """``int D = N`` holds by construction: D is ``jax.grad`` of N."""
    tetra = _tetrahedra(kind, 8)
    eigenvalues, weights = _free_electron(8)
    energies = jnp.linspace(-0.05, 0.8, 601)
    dos, integrated = tetrahedron_dos(tetra, eigenvalues, weights, energies, chunk=32)

    cumulative = np.cumsum((np.asarray(dos)[1:] + np.asarray(dos)[:-1]) / 2.0) * float(
        energies[1] - energies[0]
    )
    assert cumulative == pytest.approx(np.asarray(integrated)[1:], abs=2e-3)
    # Beyond the top of the band every state is counted.
    assert float(integrated[-1]) == pytest.approx(DEGSPIN, abs=1e-10)


def test_dos_chunking_does_not_change_the_answer():
    tetra = _tetrahedra("bloechl", 6)
    eigenvalues, weights = _free_electron(6)
    energies = jnp.linspace(0.0, 0.7, 77)
    a = tetrahedron_dos(tetra, eigenvalues, weights, energies, chunk=7)
    b = tetrahedron_dos(tetra, eigenvalues, weights, energies, chunk=1024)
    assert np.asarray(a[0]) == pytest.approx(np.asarray(b[0]))
    assert np.asarray(a[1]) == pytest.approx(np.asarray(b[1]))


def test_gradient_is_finite_on_a_degenerate_tetrahedron():
    """The NaN-in-grad trap: dead branches divide by zero, ``grad`` propagates it.

    A flat band makes every corner energy of every tetrahedron equal, so three
    of the four branches are 0/0. The forward value survives a plain ``where``;
    the gradient does not, unless the denominators are clamped *before* the
    division.
    """
    tetra = _tetrahedra("bloechl", 3)
    nk = int(np.asarray(tetra.corners).max()) + 1
    # Two bands: one perfectly flat, one with a two-fold degeneracy at every
    # corner of half the tetrahedra.
    flat = jnp.zeros((nk, 1))
    stepped = jnp.asarray((np.arange(nk) % 2).astype(float))[:, None]
    eigenvalues = jnp.concatenate([flat, stepped], axis=1)
    weights = jnp.full((nk,), DEGSPIN / nk)

    def total(et, energy):
        e_sorted, _ = _sorted_corners(tetra, et)
        return integrated_states(e_sorted, energy, tetra.ntetra, jnp.sum(weights))

    for energy in (-0.5, 0.0, 0.5, 1.0, 1.5):
        d_energy = jax.grad(total, argnums=1)(eigenvalues, energy)
        d_eigen = jax.grad(total, argnums=0)(eigenvalues, energy)
        assert np.isfinite(float(d_energy)), energy
        assert np.all(np.isfinite(np.asarray(d_eigen))), energy


# --------------------------------------------------------------------------
# Occupation weights
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_weights_sum_to_the_electron_count(kind):
    tetra = _tetrahedra(kind, 8)
    eigenvalues, weights = _free_electron(8)
    for nelec in (0.4, 1.0, 1.6):
        wg, ef = tetrahedron_occupations(tetra, eigenvalues, weights, nelec)
        assert float(jnp.sum(wg)) == pytest.approx(nelec, abs=1e-9)
        assert float(ef) > 0.0


@pytest.mark.parametrize("kind", KINDS)
def test_fermi_level_agrees_with_small_degauss_smearing(kind):
    """Both schemes must find the same Fermi level, which here is known exactly.

    For the free-electron band ``N(E_F) = nelec`` is ``(4/3) pi E^{3/2} * 2 = 1``,
    so ``E_F = (3/(8 pi))^{2/3}``. Tetrahedra and a narrow smearing are two
    estimates of the same root and must bracket it; they cannot agree to
    arbitrary precision, because smearing sums over discrete k-points where the
    tetrahedra interpolate between them, and the two ``N(E)`` differ at O(1/N).
    """
    n = 16
    tetra = _tetrahedra(kind, n)
    eigenvalues, weights = _free_electron(n)
    nelec = 1.0
    exact = (3.0 / (8.0 * np.pi)) ** (2.0 / 3.0)

    _, ef = tetrahedron_occupations(tetra, eigenvalues, weights, nelec)
    smeared = float(smeared_occupations(eigenvalues, weights, nelec, 0.005)[1])

    assert float(ef) == pytest.approx(exact, abs=5e-3)
    assert smeared == pytest.approx(exact, abs=1e-2)
    assert float(ef) == pytest.approx(smeared, abs=1e-2)


@pytest.mark.parametrize("kind", KINDS)
def test_weights_are_the_step_function_far_from_the_fermi_level(kind):
    """Bands wholly below ``E_F`` carry the full k-point weight, ones above none."""
    n = 8
    tetra = _tetrahedra(kind, n)
    points, weights = monkhorst_pack((n, n, n))
    weights = jnp.asarray(weights * DEGSPIN)
    # Two well-separated flat-ish bands: the lower one is entirely occupied.
    low = jnp.asarray(np.sum(points**2, axis=1) * 0.01 - 1.0)
    high = low + 5.0
    eigenvalues = jnp.stack([low, high], axis=1)

    # With the lower band exactly full, N(E) has a plateau across the gap and
    # the bisection walks to its upper end -- QE's ``efermit`` does the same,
    # which is why it keeps an ``efbetter``. The upper band therefore picks up a
    # weight of order the plateau's numerical width rather than exactly zero.
    wg, _ = tetrahedron_occupations(tetra, eigenvalues, weights, 2.0)
    assert np.asarray(wg[:, 0]) == pytest.approx(np.asarray(weights), abs=1e-9)
    assert np.asarray(wg[:, 1]) == pytest.approx(np.zeros(len(weights)), abs=1e-9)


# --------------------------------------------------------------------------
# w0gauss
# --------------------------------------------------------------------------


def _w0gauss_reference(x, n):
    """``Modules/w0gauss.f90``, transcribed, so the ``jvp`` has something to match."""
    x = np.asarray(x, dtype=float)
    if n == -99:
        return np.where(np.abs(x) <= 36.0, 1.0 / (2.0 + np.exp(-x) + np.exp(x)), 0.0)
    if n == -1:
        arg = np.minimum(200.0, (x - 1.0 / np.sqrt(2.0)) ** 2)
        return np.exp(-arg) / SQRT_PI * (2.0 - np.sqrt(2.0) * x)
    arg = np.minimum(200.0, x**2)
    value = np.exp(-arg) / SQRT_PI
    hd = np.zeros_like(x)
    hp = np.exp(-arg)
    ni, a = 0, 1.0 / SQRT_PI
    for i in range(1, n + 1):
        hd = 2.0 * x * hp - 2.0 * ni * hd
        ni += 1
        a = -a / (i * 4.0)
        hp = 2.0 * x * hd - 2.0 * ni * hp
        ni += 1
        value = value + a * hp
    return value


@pytest.mark.parametrize("ngauss", [0, 1, 2, -1, -99])
def test_w0gauss_is_the_derivative_of_wgauss(ngauss):
    x = jnp.linspace(-6.0, 6.0, 241)
    assert np.asarray(w0gauss(x, ngauss)) == pytest.approx(
        _w0gauss_reference(np.asarray(x), ngauss), abs=1e-12
    )


@pytest.mark.parametrize("ngauss", [0, 1, 2, -1, -99])
def test_w0gauss_integrates_to_one(ngauss):
    """A delta function, however oddly shaped: ``int w0gauss dx = 1``."""
    x = np.linspace(-40.0, 40.0, 400001)
    values = np.asarray(w0gauss(jnp.asarray(x), ngauss))
    assert np.trapezoid(values, x) == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("ngauss", [0, 1, 2, -1, -99])
def test_w0gauss_matches_finite_differences_of_wgauss(ngauss):
    x = jnp.linspace(-4.0, 4.0, 81)
    h = 1e-5
    numeric = (wgauss(x + h, ngauss) - wgauss(x - h, ngauss)) / (2.0 * h)
    assert np.asarray(w0gauss(x, ngauss)) == pytest.approx(np.asarray(numeric), abs=1e-7)
