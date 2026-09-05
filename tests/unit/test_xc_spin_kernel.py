"""``dmxc_lsda``: the second derivative of the LSDA energy, transcribed as a check.

The kernel a spin-polarized screened response is built from is
``d v_sigma / d rho_sigma'``, and here it is one ``jvp`` of
:meth:`~defumat.xc.functional.Functional.spin_potential`, which is itself a
gradient -- so nothing in the package writes it down. QE does
(``XClib/qe_drivers_d_lda_lsda.f90``), and this file transcribes that expression
to compare against, which is the project's standing rule: the Fortran is the
*check*, never the implementation.

Two separate things need checking and only a pointwise comparison separates
them. At an **ordinary** point the kernel must be QE's, and that is the
transcription below. At a **fully polarized** point, where a channel density
reaches zero and ``rho_sigma^(4/3)``'s second derivative is infinite, QE does not
regularise anything -- it *defines* the kernel to be zero, by pre-zeroing
``dmuxc`` and then ``CYCLE``-ing past ``ABS(zeta_s) >= 1``. That convention is
:func:`~defumat.xc.functional._fully_polarized`, and it is what makes triplet
O2's dielectric constant computable at all (P70).

**The `e2` trap.** ``slater``, ``pz``, ``pz_polarized``, ``dpz`` and
``dpz_polarized`` all return Hartree-based quantities, and ``dmxc_lsda``
multiplies the **whole bracket** -- the exchange diagonal included -- by
``e2 = 2`` once, at the end. Applying it twice, or to correlation only, gives a
factor that a downstream comparison would then absorb into something else.

**The off-diagonal cannot be checked for a transpose here**, because QE assigns
``dmuxc(2,1)`` and copies it into ``dmuxc(1,2)``. What catches that is the
symmetry assertion: the kernel is a Hessian of a scalar, so its 2x2 must be
symmetric whatever the transcription says.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.xc.functional import _fully_polarized, get_functional

E2 = 2.0
#: ``qe_drivers_d_lda_lsda.f90:219`` -- **0.75/pi, not (3/4 pi)^(1/3)**. QE uses
#: the name ``pi34`` for both, and which one it holds depends on whether the file
#: writes ``rs = (pi34/rho)**third`` or ``rs = pi34/rho**third``. This file is the
#: first form.
PI34 = 0.75 / np.pi
FPI = 4.0 * np.pi
THIRD = 1.0 / 3.0
P43 = 4.0 / 3.0
P49 = 4.0 / 9.0
M23 = -2.0 / 3.0

#: ``pz``'s ``iflag = 1`` column, which is what ``dmxc_lsda`` asks for.
PZ_U = dict(a=0.0311, b=-0.048, c=0.0020, d=-0.0116,
            gc=-0.1423, b1=1.0529, b2=0.3334)
#: ``pz_polarized``.
PZ_P = dict(a=0.01555, b=-0.0269, c=0.0007, d=-0.0048,
            gc=-0.0843, b1=1.3981, b2=0.2611)


def _slater(rs):
    """``qe_funct_exch_lda_lsda.f90:35-38``. Returns ``(ex, vx)``."""
    f, alpha = -0.687247939924714, 2.0 / 3.0
    return f * alpha / rs, 4.0 / 3.0 * f * alpha / rs


def _pz(rs, p):
    """``pz`` / ``pz_polarized``: Perdew-Zunger correlation. ``(ec, vc)``."""
    if rs < 1.0:
        lnrs = np.log(rs)
        ec = p["a"] * lnrs + p["b"] + p["c"] * rs * lnrs + p["d"] * rs
        vc = (p["a"] * lnrs + (p["b"] - p["a"] / 3.0)
              + 2.0 / 3.0 * p["c"] * rs * lnrs
              + (2.0 * p["d"] - p["c"]) / 3.0 * rs)
        return ec, vc
    rs12 = np.sqrt(rs)
    ox = 1.0 + p["b1"] * rs12 + p["b2"] * rs
    dox = 1.0 + 7.0 / 6.0 * p["b1"] * rs12 + 4.0 / 3.0 * p["b2"] * rs
    ec = p["gc"] / ox
    return ec, ec * dox / ox


def _dpz(rs, p):
    """``dpz`` / ``dpz_polarized``: ``d vc / d rho``, the ``iflg`` branch and all."""
    a1, a2 = 7.0 * p["b1"] / 6.0, 4.0 * p["b2"] / 3.0
    if rs < 1.0:
        dmrs = (p["a"] / rs + 2.0 / 3.0 * p["c"] * (np.log(rs) + 1.0)
                + (2.0 * p["d"] - p["c"]) / 3.0)
    else:
        x = np.sqrt(rs)
        den = 1.0 + x * (p["b1"] + x * p["b2"])
        dmx = p["gc"] * ((a1 + 2.0 * a2 * x) * den
                         - 2.0 * (p["b1"] + 2.0 * p["b2"] * x)
                         * (1.0 + x * (a1 + x * a2))) / den ** 3
        dmrs = 0.5 * dmx / x
    return -FPI * rs ** 4 / 9.0 * dmrs


def dmxc_lsda(rho_up, rho_dw):
    """``dmxc_lsda``'s analytic branch, as a 2x2. Ry, ``e2`` applied once."""
    total = rho_up + rho_dw
    zeta = (rho_up - rho_dw) / total
    assert abs(zeta) < 1.0, "the analytic branch CYCLEs past |zeta| >= 1"

    # ... exchange: a pure diagonal, each channel at its own rs.
    dxx = _slater((PI34 / (2.0 * rho_up)) ** THIRD)[1] / (3.0 * rho_up)
    dyy = _slater((PI34 / (2.0 * rho_dw)) ** THIRD)[1] / (3.0 * rho_dw)

    # ... correlation, at the total density's rs.
    rs = (PI34 / total) ** THIRD
    ecu, vcu = _pz(rs, PZ_U)
    ecp, vcp = _pz(rs, PZ_P)
    denominator = 2.0 ** P43 - 2.0
    fz = ((1.0 + zeta) ** P43 + (1.0 - zeta) ** P43 - 2.0) / denominator
    fz1 = P43 * ((1.0 + zeta) ** THIRD - (1.0 - zeta) ** THIRD) / denominator
    fz2 = P49 * ((1.0 + zeta) ** M23 + (1.0 - zeta) ** M23) / denominator

    aa = _dpz(rs, PZ_U) + fz * (_dpz(rs, PZ_P) - _dpz(rs, PZ_U))
    bb = 2.0 * fz1 * (vcp - vcu - (ecp - ecu)) / total
    cc = fz2 * (ecp - ecu) / total

    upup = (dxx + aa + (1.0 - zeta) * bb + (1.0 - zeta) ** 2 * cc) * E2
    updw = (aa + (-zeta) * bb + (zeta ** 2 - 1.0) * cc) * E2
    dwdw = (dyy + aa - (1.0 + zeta) * bb + (1.0 + zeta) ** 2 * cc) * E2
    return np.array([[upup, updw], [updw, dwdw]])


def _kernel(functional, rho_up, rho_dw):
    """The package's own 2x2, as ``jacfwd`` of the potential at one point."""
    return _kernels(functional, np.atleast_1d(rho_up), np.atleast_1d(rho_dw))[0]


def _kernels(functional, up, dw):
    """``(n, 2, 2)`` for ``n`` points at once.

    ``vmap`` over the points rather than one ``jacfwd`` of the whole row: the
    potential is pointwise, so a jacobian of the row would be ``2n`` tangents
    each evaluated over ``n`` points and the comparison would cost ``O(n^2)``.
    """
    def one(pair):
        return jax.jacfwd(functional.spin_potential)(pair[:, None])[:, 0, :, 0]

    return np.asarray(jax.vmap(one, in_axes=1)(jnp.stack([up, dw])))


@pytest.fixture(scope="module")
def pz_functional():
    return get_functional("pz")


def test_the_regular_kernel_is_qes(pz_functional):
    """200 ordinary points, both ``rs`` branches, against the transcription.

    Both routes are analytic derivatives of the same Perdew-Zunger fit, so they
    agree to round-off or a term is wrong. ``rs`` within 1e-3 of 1 is skipped
    because PZ is piecewise there and the two sides of the join are different
    functions -- that is the fit's own discontinuity, not a disagreement.
    """
    generator = np.random.default_rng(20260905)
    points = []
    while len(points) < 200:
        # Log-uniform, so that the total density spans the ``rs = 1`` join --
        # a uniform draw over [1e-4, 1] puts 94 per cent of the points on one
        # side of it and leaves ``dpz``'s ``iflg`` branch essentially unchecked.
        up, dw = 10.0 ** generator.uniform(-3.5, 0.0, size=2)
        total = up + dw
        if abs((up - dw) / total) > 0.99:
            continue
        if abs((PI34 / total) ** THIRD - 1.0) < 1e-3:
            continue
        points.append((up, dw))
    up, dw = (np.array([p[0] for p in points]), np.array([p[1] for p in points]))
    # Both ``rs`` branches must actually be exercised, or the ``iflg`` split in
    # ``dpz`` goes unchecked and the test would pass with one of them wrong.
    small_rs = ((PI34 / (up + dw)) ** THIRD < 1.0).sum()
    assert 40 < small_rs < 160, small_rs

    mine = _kernels(pz_functional, up, dw)
    for index, (a, b) in enumerate(points):
        theirs = dmxc_lsda(a, b)
        assert np.allclose(mine[index], theirs, rtol=1e-9, atol=0.0), (
            a, b, mine[index], theirs
        )


def test_the_kernel_is_symmetric(pz_functional):
    """It is a Hessian, so it must be -- which is what checks the off-diagonal.

    QE assigns ``dmuxc(2,1)`` and copies it into ``dmuxc(1,2)``, so a transposed
    off-diagonal is invisible to the comparison above. This sees it.
    """
    generator = np.random.default_rng(7)
    for _ in range(20):
        up, dw = generator.uniform(1e-3, 1.0, size=2)
        kernel = _kernel(pz_functional, up, dw)
        assert abs(kernel[0, 1] - kernel[1, 0]) < 1e-12 * max(1.0, abs(kernel[0, 1]))


def test_the_mask_fires_exactly_where_qe_cycles():
    """``|zeta| >= 1``, and nowhere else -- including at negative total density."""
    rho = jnp.asarray([
        [0.30, 0.20, 0.00, 0.19999, 1e-30, -0.10, -0.30],
        [0.10, 0.00, 0.20, 0.00001, 1e-30, -0.20, 0.10],
    ])
    fired = np.asarray(_fully_polarized(rho))
    # The last two are the reason the predicate is written on |n| rather than on
    # QE's signed ``rhotot > small``: a negative total density is not by itself
    # a saturated point (|m| = 0.1 against |n| = 0.3), and one whose
    # magnetization does exceed its charge is, whatever the sign.
    #             regular  zeta=+1  zeta=-1  0.9999  zeta=0  n<0  n<0, |m|>|n|
    assert list(fired) == [False, True, True, False, False, False, True]


def test_the_kernel_is_finite_at_full_polarization(pz_functional):
    """Zero there, and finite in every mode the response stack differentiates in.

    ``dmxc_lsda`` pre-zeroes the block and ``CYCLE``s, so zero is the value QE
    would return; what mattered here is that it is a *number*. Reverse mode is
    included because a third derivative reaches this through a cotangent.
    """
    rho = jnp.asarray([[0.30, 0.20, 0.00], [0.10, 0.00, 0.20]])
    probe = jnp.ones_like(rho)

    _, tangent = jax.jvp(pz_functional.spin_potential, (rho,), (probe,))
    assert bool(jnp.all(jnp.isfinite(tangent)))
    assert np.allclose(np.asarray(tangent)[:, 1:], 0.0)
    assert not np.allclose(np.asarray(tangent)[:, 0], 0.0)

    compiled = jax.jit(lambda r, t: jax.jvp(pz_functional.spin_potential, (r,), (t,))[1])
    assert np.allclose(np.asarray(compiled(rho, probe)), np.asarray(tangent))

    reverse = jax.grad(lambda r: jnp.sum(pz_functional.spin_potential(r) ** 2))(rho)
    assert bool(jnp.all(jnp.isfinite(reverse)))


def test_the_mask_changes_nothing_at_an_ordinary_point(pz_functional):
    """Bit-identical, not merely close -- which is why nothing validated moved.

    At an unsaturated point the masked argument *is* the unmasked one, so the
    value and every derivative come off the same expression. Anything looser
    than exact equality here would mean the mask leaks into ordinary points.
    """
    rho = jnp.asarray([[0.30, 0.70, 0.12], [0.10, 0.20, 0.09]])
    probe = jnp.asarray([[1.0, -0.5, 0.25], [0.3, 0.8, -1.0]])
    unmasked = jax.jvp(
        lambda r: pz_functional._spin_potential_unmasked(r)
        if hasattr(pz_functional, "_spin_potential_unmasked")
        else pz_functional.spin_potential(r),
        (rho,), (probe,),
    )[1]
    masked = jax.jvp(pz_functional.spin_potential, (rho,), (probe,))[1]
    assert np.array_equal(np.asarray(masked), np.asarray(unmasked))
