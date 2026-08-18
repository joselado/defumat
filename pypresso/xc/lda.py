"""Local-density exchange and correlation functionals.

Written in JAX rather than bound from libxc, for the reason set out in
`PLAN.md` D1: the exchange-correlation term has to be differentiated once for
the potential, twice for the response kernel, and more for higher-order
properties. A C library gives none of those and cannot run on GPU.

The payoff is visible here: only the **energy density** is written down. The
potential is

    v_xc = d(rho e_xc) / d rho

obtained by ``jax.grad`` -- so there is no hand-derived ``v_xc`` routine to keep
consistent with the energy, and no possibility of the two drifting apart. A test
checks the result against QE's analytic expressions.

Parameterisations follow ``XClib/qe_funct_exch_lda_lsda.f90`` (Slater exchange)
and ``XClib/qe_funct_corr_lda_lsda.f90`` (Perdew-Zunger correlation).

**Units, and the trap in them.** QE's XClib routines return *Hartree*, even
though QE is a Rydberg code throughout: ``v_of_rho.f90`` multiplies by ``e2``
when it accumulates ``etxc`` and ``v_xc``. Taking the constants from XClib
without that factor produces an exchange-correlation energy exactly half the
right size -- a plausible-looking SCF that converges to the wrong answer. The
factor is applied here instead, so everything this module returns is in Ry per
electron, with densities in electrons/bohr^3.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from pypresso.units import E2

__all__ = ["exchange_energy_density", "correlation_energy_density", "xc_energy_density",
           "xc_potential", "wigner_seitz_radius"]

#: -9/8 (3/2pi)^(2/3), the Slater exchange constant in Hartree (``f`` in QE).
_SLATER_F = -0.687247939924714
_ALPHA = 2.0 / 3.0

#: Perdew-Zunger 1981 coefficients (QE's ``iflag = 1``), in Hartree.
_PZ_A, _PZ_B, _PZ_C, _PZ_D = 0.0311, -0.048, 0.0020, -0.0116
_PZ_GC, _PZ_B1, _PZ_B2 = -0.1423, 1.0529, 0.3334

#: Densities below this are treated as vacuum. QE uses the same threshold
#: (``small`` in the xc drivers); without it rs -> infinity and the logarithms
#: in the high-density branch produce NaN in empty regions of the cell.
_RHO_THRESHOLD = 1.0e-10


def wigner_seitz_radius(rho: jnp.ndarray) -> jnp.ndarray:
    """``rs`` such that a sphere of that radius holds one electron."""
    safe = jnp.maximum(rho, _RHO_THRESHOLD)
    return (3.0 / (4.0 * jnp.pi * safe)) ** (1.0 / 3.0)


def exchange_energy_density(rho: jnp.ndarray) -> jnp.ndarray:
    """Slater exchange energy per electron, Ry."""
    return E2 * _SLATER_F * _ALPHA / wigner_seitz_radius(rho)


def correlation_energy_density(rho: jnp.ndarray) -> jnp.ndarray:
    """Perdew-Zunger correlation energy per electron, Ry.

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


def xc_energy_density(rho: jnp.ndarray) -> jnp.ndarray:
    """Exchange plus correlation energy per electron, Ry."""
    return exchange_energy_density(rho) + correlation_energy_density(rho)


def _energy_per_volume(rho: jnp.ndarray) -> jnp.ndarray:
    return rho * xc_energy_density(rho)


#: v_xc = d(rho e_xc)/d rho, by differentiation rather than a second derivation.
_potential = jax.grad(_energy_per_volume)


def xc_potential(rho: jnp.ndarray) -> jnp.ndarray:
    """``v_xc(r)`` in Ry, obtained by differentiating the energy density.

    Zero wherever the density is below the vacuum threshold: those points do not
    contribute to the energy, and letting the derivative act there would put
    spurious structure into empty space.
    """
    rho = jnp.asarray(rho)
    potential = jax.vmap(_potential)(rho.reshape(-1)).reshape(rho.shape)
    return jnp.where(rho > _RHO_THRESHOLD, potential, 0.0)
