"""The exchange-correlation functionals, and the units trap inside them.

Every test here has the same shape: QE's own analytic expressions are
transcribed independently into the test, and what the code computes by
differentiating an energy is checked against them. That is what licenses
writing only the energy down -- see :mod:`pypresso.xc.functional`.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pypresso.units import E2
from pypresso.xc.functional import get_functional, resolve_functional
from pypresso.xc.lda import wigner_seitz_radius

pytestmark = pytest.mark.unit

DENSITIES = np.array([1e-3, 3e-3, 0.01, 0.05, 0.2, 1.0, 5.0])

#: (rho, |grad rho|^2) pairs spanning what a real grid holds: dense and slowly
#: varying at a bond centre, thin and sharply varying in the tail.
GRADIENT_POINTS = [(0.05, 0.01), (0.5, 0.3), (1e-3, 1e-6), (2.0, 5.0), (1e-5, 1e-8)]

PZ = get_functional("PZ")
PBE = get_functional("PBE")


# --- the local density approximation -----------------------------------------


def _qe_lda(rho):
    """QE's ``slater`` and ``pz``, transcribed.

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


def _qe_pw(rs):
    """QE's ``pw`` with ``iflag = 1``, in Hartree: ``(ec, vc)``."""
    a, a1 = 0.031091, 0.21370
    b1, b2, b3, b4 = 7.5957, 3.5876, 1.6382, 0.49294
    rs12, rs32, rs2 = np.sqrt(rs), rs ** 1.5, rs * rs
    om = 2.0 * a * (b1 * rs12 + b2 * rs + b3 * rs32 + b4 * rs2)
    dom = 2.0 * a * (0.5 * b1 * rs12 + b2 * rs + 1.5 * b3 * rs32 + 2.0 * b4 * rs2)
    olog = np.log(1.0 + 1.0 / om)
    ec = -2.0 * a * (1.0 + a1 * rs) * olog
    vc = -2.0 * a * (1.0 + 2.0 / 3.0 * a1 * rs) * olog - 2.0 / 3.0 * a * (
        1.0 + a1 * rs
    ) * dom / (om * (om + 1.0))
    return ec, vc


@pytest.mark.parametrize("rho", DENSITIES)
def test_lda_energy_density_matches_quantum_espresso(rho):
    expected, _ = _qe_lda(rho)
    assert float(PZ.energy_density(jnp.array(rho))) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("rho", DENSITIES)
def test_lda_potential_from_autodiff_matches_the_hand_derived_formula(rho):
    """The point of writing only the energy density.

    QE derives ``v_xc`` by hand in a separate routine; here it is ``grad`` of
    ``rho e_xc``. That the two agree to machine precision is what licenses
    dropping the hand-derived version -- and it is checked on both sides of the
    ``rs = 1`` branch.
    """
    _, expected = _qe_lda(rho)
    assert float(PZ.potential(jnp.array([rho]))[0]) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("rho", DENSITIES)
def test_perdew_wang_correlation_matches_quantum_espresso(rho):
    """PBE's local correlation is Perdew-Wang, not Perdew-Zunger."""
    functional = get_functional("PW")
    rs = float(wigner_seitz_radius(jnp.array(rho)))
    ec, vc = _qe_pw(rs)
    ex = E2 * -0.687247939924714 * (2.0 / 3.0) / rs
    assert float(functional.energy_density(jnp.array(rho))) == pytest.approx(
        ex + E2 * ec, rel=1e-12
    )
    vx = 4.0 / 3.0 * -0.687247939924714 * (2.0 / 3.0) / rs
    assert float(functional.potential(jnp.array([rho]))[0]) == pytest.approx(
        E2 * (vx + vc), rel=1e-10
    )


def test_both_branches_of_the_correlation_functional_are_exercised():
    """rs = 1 separates the high-density expansion from the interpolation."""
    assert float(wigner_seitz_radius(jnp.array(0.2387))) == pytest.approx(1.0, rel=1e-3)
    assert np.any(np.asarray(wigner_seitz_radius(jnp.asarray(DENSITIES))) < 1.0)
    assert np.any(np.asarray(wigner_seitz_radius(jnp.asarray(DENSITIES))) > 1.0)


def test_exchange_scales_as_the_cube_root_of_the_density():
    """e_x proportional to rho^(1/3) is exact for Slater exchange."""
    ratio = float(PZ.exchange(jnp.array(8.0)) / PZ.exchange(jnp.array(1.0)))
    assert ratio == pytest.approx(2.0, rel=1e-12)


def test_correlation_is_negative_and_smaller_than_exchange():
    for rho in DENSITIES:
        ec = float(PZ.correlation(jnp.array(rho)))
        ex = float(PZ.exchange(jnp.array(rho)))
        assert ec < 0.0 and ex < 0.0 and abs(ec) < abs(ex)


def test_vacuum_is_handled_without_nan():
    """Empty regions of a cell have essentially zero density; the logarithm in
    the high-density branch must not be allowed to reach them."""
    rho = jnp.array([0.0, 1e-30, 1e-12])
    assert np.all(np.isfinite(np.asarray(PZ.energy_density(rho))))
    assert np.asarray(PZ.potential(rho)) == pytest.approx(np.zeros(3))


def test_second_derivative_exists():
    """The response kernel f_xc = d^2(rho e_xc)/drho^2 is what a dielectric
    response needs; it must survive a second differentiation."""
    kernel = jax.grad(jax.grad(lambda r: r * PZ.energy_density(r)))(0.05)
    assert np.isfinite(float(kernel)) and float(kernel) != 0.0


# --- the gradient correction --------------------------------------------------


def _qe_pbex(rho, grho, kappa=0.804, mu=0.2195149727645171):
    """``pbex``, transcribed: ``(sx, v1x, v2x)`` in Hartree."""
    c1, c2 = 0.75 / np.pi, 3.093667726280136
    agrho = np.sqrt(grho)
    kf = c2 * rho ** (1.0 / 3.0)
    dsg = 0.5 / kf
    s1 = agrho * dsg / rho
    f2 = 1.0 + s1 * s1 * mu / kappa
    fx = kappa - kappa / f2

    exunif = -c1 * kf
    sx_s = exunif * fx
    dfx = 2.0 * mu * s1 / (f2 * f2)
    v1x = sx_s + exunif / 3.0 * fx + exunif * dfx * (-4.0 / 3.0 * s1)
    v2x = exunif * dfx * dsg / agrho
    return sx_s * rho, v1x, v2x


def _qe_pbec(rho, grho, beta=0.06672455060314922):
    """``pbec``, transcribed: ``(sc, v1c, v2c)`` in Hartree."""
    ga, pi34 = 0.0310906908696548950, 0.6203504908994
    xkf, xks = 1.919158292677513, 1.128379167095513

    rs = pi34 / rho ** (1.0 / 3.0)
    ec, vc = _qe_pw(rs)
    kf = xkf / rs
    ks = xks * np.sqrt(kf)
    t = np.sqrt(grho) / (2.0 * ks * rho)

    expe = np.exp(-ec / ga)
    af = beta / ga * (1.0 / (expe - 1.0))
    bf = expe * (vc - ec)
    y = af * t * t
    xy = (1.0 + y) / (1.0 + y + y * y)
    qy = y * y * (2.0 + y) / (1.0 + y + y * y) ** 2
    s1 = 1.0 + beta / ga * t * t * xy
    h0 = ga * np.log(s1)
    dh0 = beta * t * t / s1 * (-7.0 / 3.0 * xy - qy * (af * bf / beta - 7.0 / 3.0))
    ddh0 = beta / (2.0 * ks * ks * rho) * (xy - qy) / s1
    return rho * h0, h0 + dh0, ddh0


@pytest.mark.parametrize(("rho", "sigma"), GRADIENT_POINTS)
def test_pbe_gradient_energy_matches_quantum_espresso(rho, sigma):
    expected = E2 * (_qe_pbex(rho, sigma)[0] + _qe_pbec(rho, sigma)[0])
    got = float(PBE.gradient_energy(jnp.array([rho]), jnp.array([sigma]))[0])
    assert got == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(("rho", "sigma"), GRADIENT_POINTS)
def test_pbe_gradient_potentials_match_the_hand_derived_formulas(rho, sigma):
    """``v1`` and ``v2`` by autodiff against ``pbex``/``pbec``'s own algebra.

    ``v2`` is the one worth stating: QE defines it as ``d(rho e)/d|grad rho|``
    divided by ``|grad rho|``, which is ``2 d(rho e)/d sigma``. Getting the
    factor of two wrong halves the gradient term and still converges.
    """
    v1_expected = E2 * (_qe_pbex(rho, sigma)[1] + _qe_pbec(rho, sigma)[1])
    v2_expected = E2 * (_qe_pbex(rho, sigma)[2] + _qe_pbec(rho, sigma)[2])

    v1, v2 = PBE.gradient_potentials(jnp.array([rho]), jnp.array([sigma]))
    assert float(v1[0]) == pytest.approx(v1_expected, rel=1e-10)
    assert float(v2[0]) == pytest.approx(v2_expected, rel=1e-10)


@pytest.mark.parametrize("name", ["REVPBE", "PBESOL"])
def test_the_pbe_family_differs_only_in_its_constants(name):
    constants = {
        "REVPBE": (dict(kappa=1.2450), dict()),
        "PBESOL": (dict(mu=0.12345679012345679), dict(beta=0.046)),
    }[name]
    functional = get_functional(name)
    rho, sigma = 0.05, 0.01
    expected = E2 * (
        _qe_pbex(rho, sigma, **constants[0])[0] + _qe_pbec(rho, sigma, **constants[1])[0]
    )
    got = float(functional.gradient_energy(jnp.array([rho]), jnp.array([sigma]))[0])
    assert got == pytest.approx(expected, rel=1e-12)


def test_the_gradient_correction_is_gated_exactly_where_quantum_espresso_gates_it():
    """``rho <= 1e-6`` or ``sigma <= 1e-10`` contributes nothing at all.

    The LDA threshold is 1e-10, four orders of magnitude smaller; using it here
    would evaluate the gradient terms in the tail of the density where QE does
    not, and no amount of convergence would recover the difference.
    """
    rho = jnp.array([1e-7, 1e-5, 1e-5, 0.0, -1e-3])
    sigma = jnp.array([1e-6, 1e-11, 1e-6, 1e-6, 1e-6])
    active = np.array([False, False, True, False, False])

    energy = np.asarray(PBE.gradient_energy(rho, sigma))
    v1, v2 = (np.asarray(x) for x in PBE.gradient_potentials(rho, sigma))
    for array in (energy, v1, v2):
        assert np.all(np.isfinite(array))
        assert np.all(array[~active] == 0.0)
    assert energy[active] != 0.0 and v1[active] != 0.0 and v2[active] != 0.0


def test_the_gradient_correction_vanishes_for_a_uniform_density():
    """PBE is exact for the uniform electron gas: at sigma = 0 it must add
    nothing, or the functional is not the one it claims to be."""
    energy = float(PBE.gradient_energy(jnp.array([0.3]), jnp.array([0.0]))[0])
    assert energy == 0.0


# --- composing and naming -----------------------------------------------------


def test_names_are_read_in_every_spelling_a_upf_file_uses():
    assert get_functional("PBE").name == "PBE"
    assert get_functional("SLA-PW-PBX-PBC").name == "PBE"
    # The legacy spelling every ``*.pbe-*.UPF`` in the test set carries.
    assert get_functional(" SLA  PW   PBE  PBE").name == "PBE"
    assert get_functional(" SLA  PZ   NOGX NOGC").name == "PZ"
    assert get_functional("LDA").name == "PZ"


def test_an_unimplemented_functional_is_refused_rather_than_ignored():
    """The failure this prevents: a BLYP pseudopotential quietly run as LDA."""
    with pytest.raises(ValueError, match="unknown exchange-correlation term"):
        get_functional("SLA LYP B88 BLYP")


def test_gradient_functionals_announce_themselves():
    assert get_functional("PBE").is_gradient
    assert get_functional("PBESOL").is_gradient
    assert not get_functional("PZ").is_gradient
    assert not get_functional("PW").is_gradient


def test_pseudopotentials_must_agree_on_the_functional():
    with pytest.raises(ValueError, match="different functionals"):
        resolve_functional([" SLA  PZ   NOGX NOGC", " SLA  PW   PBE  PBE"])


def test_input_dft_overrides_the_pseudopotentials_but_says_so():
    with pytest.warns(UserWarning, match="input_dft asks for PBE"):
        functional = resolve_functional([" SLA  PZ   NOGX NOGC"], input_dft="pbe")
    assert functional.name == "PBE"


def test_input_dft_overrides_a_functional_that_is_not_implemented():
    """An override is the way to run a dataset whose own functional is missing,
    so parsing the dataset's string must not be what stops the run."""
    assert resolve_functional(["SLA LYP B88 BLYP"], input_dft="pbe").name == "PBE"
