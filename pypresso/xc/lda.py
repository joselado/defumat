"""Local-density exchange and correlation components.

Written in JAX rather than bound from libxc, for the reason set out in
`PLAN.md` D1: the exchange-correlation term has to be differentiated once for
the potential, twice for the response kernel, and more for higher-order
properties. A C library gives none of those and cannot run on GPU.

The payoff is visible here: only the **energy density** is written down. The
potential is

    v_xc = d(rho e_xc) / d rho

obtained by ``jax.grad`` in :mod:`pypresso.xc.functional` -- so there is no
hand-derived ``v_xc`` routine to keep consistent with the energy, and no
possibility of the two drifting apart. A test checks the result against QE's
analytic expressions.

What lives here is one function per *component*, in QE's sense: XClib composes a
functional out of four independently chosen slots (exchange, correlation, and
their gradient corrections), and a name like ``PBE`` is a shorthand for a
particular choice of the four (``qe_dft_list.f90``). The two LDA slots are
here, the two gradient slots in :mod:`pypresso.xc.gga`, and the composition in
:mod:`pypresso.xc.functional`.

Parameterisations follow ``XClib/qe_funct_exch_lda_lsda.f90`` (Slater exchange)
and ``XClib/qe_funct_corr_lda_lsda.f90`` (Perdew-Zunger and Perdew-Wang
correlation).

**Units, and the trap in them.** QE's XClib routines return *Hartree*, even
though QE is a Rydberg code throughout: ``v_of_rho.f90`` multiplies by ``e2``
when it accumulates ``etxc`` and ``v_xc``. Taking the constants from XClib
without that factor produces an exchange-correlation energy exactly half the
right size -- a plausible-looking SCF that converges to the wrong answer. The
factor is applied here instead, so everything this module returns is in Ry per
electron, with densities in electrons/bohr^3. The one exception is
:func:`pw_correlation_hartree`, which exists because PBE's correlation is built
on top of the Perdew-Wang energy *in Hartree* and needs it unconverted.
"""

from __future__ import annotations

import jax.numpy as jnp

from pypresso.units import E2

__all__ = ["slater_exchange", "pz_correlation", "pw_correlation",
           "pw_correlation_hartree", "no_exchange", "no_correlation",
           "pz_correlation_spin", "pw_correlation_spin", "pw_spin_hartree",
           "no_correlation_spin", "spin_interpolation", "wigner_seitz_radius",
           "RHO_THRESHOLD"]

#: -9/8 (3/2pi)^(2/3), the Slater exchange constant in Hartree (``f`` in QE).
_SLATER_F = -0.687247939924714
_ALPHA = 2.0 / 3.0

#: Perdew-Zunger 1981 coefficients (QE's ``iflag = 1``), in Hartree.
_PZ_A, _PZ_B, _PZ_C, _PZ_D = 0.0311, -0.048, 0.0020, -0.0116
_PZ_GC, _PZ_B1, _PZ_B2 = -0.1423, 1.0529, 0.3334

#: Perdew-Wang 1992 coefficients (QE's ``pw`` with ``iflag = 1``), in Hartree.
_PW_A, _PW_A1 = 0.031091, 0.21370
_PW_B1, _PW_B2, _PW_B3, _PW_B4 = 7.5957, 3.5876, 1.6382, 0.49294

#: Densities below this are treated as vacuum. QE uses the same threshold
#: (``rho_threshold_lda`` in ``XClib/dft_setting_params.f90``); without it
#: rs -> infinity and the logarithms in the high-density branch produce NaN in
#: empty regions of the cell.
RHO_THRESHOLD = 1.0e-10


def wigner_seitz_radius(rho: jnp.ndarray) -> jnp.ndarray:
    """``rs`` such that a sphere of that radius holds one electron.

    The **absolute value** of the density, as ``xc_lda`` takes it. A plane-wave
    density is a truncated Fourier series and goes slightly negative in vacuum
    -- QE reports how much on every iteration -- and there the local functional
    is evaluated at ``|rho|`` rather than switched off. Clamping to the
    threshold instead leaves a large low-density region with no
    exchange-correlation potential at all, which is invisible on a bulk crystal
    and worth ~1e-5 Ry per energy term on an atom in a box.
    """
    safe = jnp.maximum(jnp.abs(rho), RHO_THRESHOLD)
    return (3.0 / (4.0 * jnp.pi * safe)) ** (1.0 / 3.0)


def no_exchange(rho: jnp.ndarray) -> jnp.ndarray:
    """QE's ``NOX``: the slot left empty."""
    return jnp.zeros_like(rho)


def no_correlation(rho: jnp.ndarray) -> jnp.ndarray:
    """QE's ``NOC``."""
    return jnp.zeros_like(rho)


def slater_exchange(rho: jnp.ndarray) -> jnp.ndarray:
    """Slater exchange energy per electron, Ry (QE's ``SLA``)."""
    return E2 * _SLATER_F * _ALPHA / wigner_seitz_radius(rho)


def pz_correlation(rho: jnp.ndarray) -> jnp.ndarray:
    """Perdew-Zunger correlation energy per electron, Ry (QE's ``PZ``).

    Two branches meet at ``rs = 1``: a high-density expansion in ``log(rs)`` and
    a Pade interpolation of the Monte Carlo data. Both are evaluated on
    sanitised arguments so that the unused branch cannot contribute a NaN
    gradient through ``where``.
    """
    rs = wigner_seitz_radius(rho)
    high_density = rs < 1.0

    safe_rs = jnp.where(high_density, rs, 1.0)
    lnrs = jnp.log(safe_rs)
    high = _PZ_A * lnrs + _PZ_B + _PZ_C * safe_rs * lnrs + _PZ_D * safe_rs

    interpolation_rs = jnp.where(high_density, 1.0, rs)
    root = jnp.sqrt(interpolation_rs)
    low = _PZ_GC / (1.0 + _PZ_B1 * root + _PZ_B2 * interpolation_rs)

    return E2 * jnp.where(high_density, high, low)


def pw_correlation_hartree(rs: jnp.ndarray) -> jnp.ndarray:
    """Perdew-Wang 1992 correlation energy per electron, **Hartree**.

    QE's ``pw`` with ``iflag = 1``, which is the correlation half of PBE. Its
    high- and low-density limits are coded in XClib but deliberately not used
    for this ``iflag`` -- the comment there is "inconsistencies in PBE/PW91
    functionals" -- so the interpolation formula is evaluated everywhere, and
    reproducing that choice is what makes PBE agree with QE rather than with the
    paper.
    """
    rs12 = jnp.sqrt(rs)
    rs32 = rs * rs12
    rs2 = rs * rs
    om = 2.0 * _PW_A * (_PW_B1 * rs12 + _PW_B2 * rs + _PW_B3 * rs32 + _PW_B4 * rs2)
    return -2.0 * _PW_A * (1.0 + _PW_A1 * rs) * jnp.log(1.0 + 1.0 / om)


def pw_correlation(rho: jnp.ndarray) -> jnp.ndarray:
    """Perdew-Wang correlation energy per electron, Ry (QE's ``PW``)."""
    return E2 * pw_correlation_hartree(wigner_seitz_radius(rho))


# --- the spin-polarized halves ------------------------------------------------
#
# Exchange needs nothing here. The spin-scaling relation
#
#     E_x[rho_up, rho_dw] = ( E_x[2 rho_up] + E_x[2 rho_dw] ) / 2
#
# is exact for *any* exchange functional, and QE uses it directly -- ``gcx_spin``
# doubles each channel's density and calls the unpolarized routine. So
# :mod:`pypresso.xc.functional` derives the polarized exchange from whichever
# unpolarized slot is filled, and only correlation gets its own entries below.
#
# Correlation does need them: the polarized limit is a separate fit to the same
# Monte Carlo data, and what interpolates between the two limits is
# ``f(zeta)``, the ratio of the non-interacting kinetic energies.

#: Perdew-Zunger's ferromagnetic coefficients (``pz_polarized``), in Hartree.
_PZP_A, _PZP_B, _PZP_C, _PZP_D = 0.01555, -0.0269, 0.0007, -0.0048
_PZP_GC, _PZP_B1, _PZP_B2 = -0.0843, 1.3981, 0.2611

#: Perdew-Wang 1992's polarized and spin-stiffness fits (``pw_spin``), Hartree.
_PWP = (0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)
_PWA = (0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671)

#: ``f''(0)``, the coefficient the spin stiffness is divided by.
_FZ0 = 1.709921


def spin_interpolation(zeta: jnp.ndarray) -> jnp.ndarray:
    """``f(zeta) = ((1+z)^(4/3) + (1-z)^(4/3) - 2) / (2^(4/3) - 2)``.

    Zero for an unpolarized density and one for a fully polarized one -- the
    universal interpolation von Barth and Hedin introduced and every LSDA
    correlation functional here uses. ``zeta`` is clipped to [-1, 1] first, as
    ``xc_lsda`` clips it, so that the fractional powers never see a negative
    argument: a plane-wave magnetization can exceed the density it is divided by
    in low-density regions, and ``(1 - z)^(4/3)`` would then be a NaN rather
    than a large number.
    """
    z = jnp.clip(zeta, -1.0, 1.0)
    return ((1.0 + z) ** (4.0 / 3.0) + (1.0 - z) ** (4.0 / 3.0) - 2.0) / (
        2.0 ** (4.0 / 3.0) - 2.0
    )


def no_correlation_spin(rho: jnp.ndarray, zeta: jnp.ndarray) -> jnp.ndarray:
    """QE's ``NOC``, spin-polarized."""
    return jnp.zeros_like(rho)


def _pz_polarized_hartree(rs: jnp.ndarray) -> jnp.ndarray:
    """``pz_polarized``: the ferromagnetic Perdew-Zunger energy, Hartree.

    Same two branches as the unpolarized fit and the same meeting point at
    ``rs = 1``; only the coefficients differ.
    """
    high_density = rs < 1.0

    safe_rs = jnp.where(high_density, rs, 1.0)
    lnrs = jnp.log(safe_rs)
    high = _PZP_A * lnrs + _PZP_B + _PZP_C * safe_rs * lnrs + _PZP_D * safe_rs

    interpolation_rs = jnp.where(high_density, 1.0, rs)
    root = jnp.sqrt(interpolation_rs)
    low = _PZP_GC / (1.0 + _PZP_B1 * root + _PZP_B2 * interpolation_rs)

    return jnp.where(high_density, high, low)


def pz_correlation_spin(rho: jnp.ndarray, zeta: jnp.ndarray) -> jnp.ndarray:
    """Perdew-Zunger correlation per electron, Ry, spin-polarized (``pz_spin``).

    ``ec = ecu + f(zeta) (ecp - ecu)``: the unpolarized fit, moved towards the
    ferromagnetic one by the interpolation function. Only the energy is written
    down; ``vc_up`` and ``vc_dw`` -- QE's three-term expressions with their
    explicit ``df/dzeta`` -- come from differentiating it.
    """
    rs = wigner_seitz_radius(rho)
    ecu = pz_correlation(rho) / E2  # the unpolarized fit, in Hartree
    ecp = _pz_polarized_hartree(rs)
    return E2 * (ecu + spin_interpolation(zeta) * (ecp - ecu))


def _pw_branch(rs: jnp.ndarray, a, a1, b1, b2, b3, b4) -> jnp.ndarray:
    """One Perdew-Wang interpolation, Hartree; the sign is the caller's.

    The same expression serves the unpolarized fit, the polarized one and the
    spin stiffness -- QE writes it out three times with different constants.
    """
    rs12 = jnp.sqrt(rs)
    om = 2.0 * a * (b1 * rs12 + b2 * rs + b3 * rs * rs12 + b4 * rs * rs)
    return -2.0 * a * (1.0 + a1 * rs) * jnp.log(1.0 + 1.0 / om)


def pw_spin_hartree(rs: jnp.ndarray, zeta: jnp.ndarray) -> jnp.ndarray:
    """Perdew-Wang 1992 correlation per electron, **Hartree**, spin-polarized.

    ``ec = ec_U + alpha_c f(z) (1 - z^4)/f''(0) + (ec_P - ec_U) f(z) z^4``:
    the spin stiffness governs the small-``z`` behaviour and the polarized fit
    takes over as ``z -> 1``, which is what the ``z^4`` weighting arranges.

    QE's ``alpha`` is ``-alpha_c`` of the paper -- it is built with the same
    ``_pw_branch`` expression and then *added* rather than subtracted -- so the
    sign here is QE's, not the paper's. Unconverted, like
    :func:`pw_correlation_hartree`, because PBE's spin correlation is built on
    top of it in Hartree.
    """
    z = jnp.clip(zeta, -1.0, 1.0)
    z4 = z**4
    fz = spin_interpolation(z)

    ec_u = pw_correlation_hartree(rs)
    ec_p = _pw_branch(rs, *_PWP)
    alpha = -_pw_branch(rs, *_PWA)

    return ec_u + alpha * fz * (1.0 - z4) / _FZ0 + (ec_p - ec_u) * fz * z4


def pw_correlation_spin(rho: jnp.ndarray, zeta: jnp.ndarray) -> jnp.ndarray:
    """Perdew-Wang correlation per electron, Ry, spin-polarized (``pw_spin``)."""
    return E2 * pw_spin_hartree(wigner_seitz_radius(rho), zeta)
