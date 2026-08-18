"""Pseudopotential reading and its radial-to-G transforms."""

import numpy as np
import pytest
from scipy.special import spherical_jn

from pypresso.pseudo import (
    atomic_charge_of_g,
    local_potential_of_g,
    mesh_cutoff_index,
    projector_form_factors,
    read_upf,
    simpson,
    spherical_bessel,
)
from pypresso.pseudo.radial import RCUT
from pypresso.units import E2, FPI

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def silicon(pseudo_dir):
    return read_upf(pseudo_dir / "Si.pz-vbc.UPF")


def test_header_and_arrays(silicon):
    assert silicon.element == "Si"
    assert silicon.z_valence == pytest.approx(4.0)
    assert silicon.pseudo_type == "NC"
    assert silicon.mesh == 431
    assert [p.l for p in silicon.projectors] == [0, 1]
    assert silicon.dij.shape == (2, 2)
    assert silicon.nh == 1 + 3  # one s channel, three p channels
    assert not silicon.is_ultrasoft and not silicon.has_nlcc
    assert len(silicon.r) == len(silicon.rab) == len(silicon.vloc) == silicon.mesh


def test_every_shipped_pseudopotential_parses(pseudo_dir):
    files = sorted(pseudo_dir.glob("*.UPF"))
    assert len(files) >= 10
    for path in files:
        pseudo = read_upf(path)
        assert pseudo.z_valence > 0
        assert np.all(np.diff(pseudo.r) > 0), f"{path.name}: radial grid is not increasing"
        assert pseudo.msh <= pseudo.mesh


def test_mesh_truncation_is_odd_and_within_ten_bohr(silicon):
    msh = mesh_cutoff_index(silicon.r)
    assert msh % 2 == 1, "Simpson's rule needs an odd number of points"
    assert silicon.r[msh - 1] <= RCUT
    assert silicon.r[-1] > RCUT, "this pseudopotential should extend past the cutoff"


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_spherical_bessel_matches_scipy(l):
    x = np.concatenate([[0.0], np.logspace(-8, 2, 60)])
    assert np.asarray(spherical_bessel(l, x)) == pytest.approx(spherical_jn(l, x), abs=1e-10)


def test_spherical_bessel_series_branch_is_accurate():
    """Below the switch the closed forms lose their digits to cancellation."""
    x = np.array([1e-6, 1e-4, 0.01, 0.049])
    for l in range(4):
        assert np.asarray(spherical_bessel(l, x)) == pytest.approx(spherical_jn(l, x), rel=1e-10)


def test_simpson_integrates_a_known_function(silicon):
    """int_0^inf r^2 exp(-r) dr = 2, on the pseudopotential's own log mesh."""
    r, rab = silicon.r, silicon.rab
    assert float(simpson(r**2 * np.exp(-r), rab)) == pytest.approx(2.0, rel=1e-9)


def test_atomic_charge_integrates_to_the_valence(silicon):
    """rho(q=0) * Omega is the electron count -- up to the 10-bohr truncation,
    which QE shares, and which is why the starting density is renormalised."""
    omega = 265.302
    value = float(atomic_charge_of_g(silicon, np.array([0.0]), omega)[0]) * omega
    assert value == pytest.approx(silicon.z_valence, rel=1e-3)


def test_local_potential_zero_term_is_not_the_limit_of_the_others(silicon):
    """The G = 0 term uses Z e^2, not Z e^2 erf(r); QE says so in a comment.

    They are genuinely different numbers, and using the wrong one shifts every
    eigenvalue by a constant -- a convincing calculation with the wrong answer.
    """
    omega = 265.302
    at_zero = float(local_potential_of_g(silicon, np.array([0.0]), omega)[0])
    just_above = float(local_potential_of_g(silicon, np.array([1e-4]), omega)[0])

    assert at_zero == pytest.approx(-0.01853306, abs=1e-8)
    # The q -> 0 limit of the erf-split expression diverges as -1/q^2, so the two
    # are nowhere near each other: that is the whole point of treating G = 0
    # separately, and using the wrong integrand here (Z e^2 erf(r) instead of
    # Z e^2) gives -0.11326531, which shifted every silicon eigenvalue by 2.5 eV.
    assert just_above < -1e6


def test_local_potential_approaches_the_coulomb_tail(silicon):
    """At large q the short-range part dies and only -4 pi Z e^2/(Omega q^2)
    exp(-q^2/4) survives, which itself vanishes."""
    omega = 265.302
    q = np.array([20.0, 40.0])
    values = np.asarray(local_potential_of_g(silicon, q, omega))
    assert np.all(np.abs(values) < 1e-4)


def test_projector_form_factors_vanish_at_the_origin_for_l_above_zero(silicon):
    """f_l(q) ~ q^l as q -> 0, because j_l does."""
    factors = np.asarray(projector_form_factors(silicon, np.array([0.0, 0.01]), 265.302))
    assert abs(factors[0, 0]) > 0.1  # l = 0 is finite at q = 0
    assert abs(factors[1, 0]) < 1e-12  # l = 1 vanishes
    assert abs(factors[1, 1]) > 0.0


def test_form_factors_are_differentiable_in_q(silicon):
    """Rule D2: the velocity operator comes from differentiating vkb(k), so the
    radial factors must not be a table lookup."""
    import jax

    derivative = jax.grad(
        lambda q: projector_form_factors(silicon, q[None], 265.302)[1, 0]
    )(np.array(0.5))
    assert np.isfinite(float(derivative)) and float(derivative) != 0.0
