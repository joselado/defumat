"""Gradient corrections to exchange and correlation: the PBE family.

A generalised-gradient functional adds to the local energy a term that depends
on ``|grad rho|`` as well as on ``rho``. QE keeps the two apart -- the LDA slots
of :mod:`pypresso.xc.lda` return the local part and the routines here return the
*correction on top of it*, which is why ``pbex`` is documented as "PBE exchange
(without Slater exchange)" and ``pbec`` as "PBE correlation (without LDA part)".
Reproducing that split matters: PBE's local half is Slater exchange plus
**Perdew-Wang** correlation, not the Perdew-Zunger correlation an LDA
calculation uses, and pairing the gradient terms with the wrong local ones is a
functional that converges to a number QE never prints.

Following ``XClib/qe_funct_exch_gga.f90`` (``pbex``) and
``XClib/qe_funct_corr_gga.f90`` (``pbec``). Three members of the family differ
only in constants, and are registered separately in
:mod:`pypresso.xc.functional`:

    PBE      kappa = 0.804,  mu = 0.2195149727645171,  beta = 0.06672455060314922
    revPBE   kappa = 1.245,  mu = 0.2195149727645171,  beta = 0.06672455060314922
    PBEsol   kappa = 0.804,  mu = 10/81,               beta = 0.046

**What these functions return, and why it is not what QE returns.** QE's
routines hand back three numbers per point -- the energy ``sx`` and the two
potential pieces ``v1x = d(rho e)/d rho`` and ``v2x = d(rho e)/d|grad rho| /
|grad rho|``, each derived by hand and each a source of drift against the
energy. Here only the energy is written down, per unit volume and as a function
of ``rho`` and ``sigma = |grad rho|^2``; both potential pieces come from
``jax.grad`` of it in :mod:`pypresso.xc.functional`. The identity that makes
that a drop-in replacement is ``v2 = 2 d(rho e)/d sigma``, since ``d sigma /
d|grad rho| = 2 |grad rho|``.

Units are Ry per bohr^3, following the convention of :mod:`pypresso.xc.lda`:
XClib's constants are Hartree and the ``e2`` is applied here rather than by the
caller.
"""

from __future__ import annotations

import jax.numpy as jnp

from pypresso.units import E2
from pypresso.xc.lda import pw_correlation_hartree, pw_spin_hartree

__all__ = ["pbe_exchange", "pbe_correlation", "pbe_correlation_spin",
           "no_gradient_exchange", "no_gradient_correlation",
           "no_gradient_correlation_spin", "PBE_KAPPA", "PBE_MU", "PBESOL_MU",
           "REVPBE_KAPPA", "PBE_BETA", "PBESOL_BETA"]

#: ``k`` and ``mu`` of ``pbex``, per variant.
PBE_KAPPA = 0.804
REVPBE_KAPPA = 1.2450
PBE_MU = 0.2195149727645171
PBESOL_MU = 0.12345679012345679  # 10/81

#: ``be`` of ``pbec``, per variant.
PBE_BETA = 0.06672455060314922
PBESOL_BETA = 0.046

#: (3 pi^2)^(1/3) and 3/(4 pi), QE's ``c2`` and ``c1``.
_C2 = 3.093667726280136
_C1 = 0.75 / jnp.pi

#: ``pbec``'s constants: (3/4pi)^(1/3), (9 pi/4)^(1/3), sqrt(4/pi), and the
#: ``gamma`` of the PBE correlation paper, in Hartree.
_PI34 = 0.6203504908994
_XKF = 1.919158292677513
_XKS = 1.128379167095513
_GAMMA = 0.0310906908696548950


def no_gradient_exchange(rho: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """QE's ``NOGX``: no gradient correction to exchange."""
    return jnp.zeros_like(rho)


def no_gradient_correlation(rho: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """QE's ``NOGC``."""
    return jnp.zeros_like(rho)


def pbe_exchange(rho, sigma, kappa: float = PBE_KAPPA, mu: float = PBE_MU):
    """PBE exchange beyond Slater, Ry/bohr^3 (``pbex``, ``iflag = 1, 2, 3``).

    The reduced gradient ``s = |grad rho| / (2 k_F rho)`` measures how fast the
    density varies on the scale of its own Fermi wavelength; the enhancement
    factor is ``F(s) = 1 + kappa - kappa / (1 + mu s^2 / kappa)``, and the ``1``
    is exactly the Slater exchange already counted in the local part, so what is
    returned is ``rho e_x^unif (F - 1)``.

    ``revPBE`` raises ``kappa`` and ``PBEsol`` lowers ``mu``; nothing else about
    the expression changes, which is why one function serves all three.
    """
    kf = _C2 * rho ** (1.0 / 3.0)
    s2 = sigma / (4.0 * kf * kf * rho * rho)
    enhancement = kappa - kappa / (1.0 + mu * s2 / kappa)
    return E2 * rho * (-_C1 * kf) * enhancement


def pbe_correlation(rho, sigma, beta: float = PBE_BETA):
    """PBE correlation beyond the local part, Ry/bohr^3 (``pbec``).

    ``H(rho, t)`` of the PBE paper, with ``t = |grad rho| / (2 k_s rho)`` the
    gradient measured against the Thomas-Fermi screening length. It is built on
    the **Perdew-Wang** local correlation ``ec``, which enters through
    ``A = beta/gamma / (exp(-ec/gamma) - 1)``.

    That dependence is where writing only the energy pays off twice over: QE's
    hand-derived ``v1c`` contains ``vc = d(rho ec)/d rho`` explicitly, so its
    correctness depends on the local and gradient routines agreeing about which
    correlation parameterisation is in use. Differentiating this expression
    cannot get that wrong.
    """
    rs = _PI34 / rho ** (1.0 / 3.0)
    ec = pw_correlation_hartree(rs)

    kf = _XKF / rs
    ks = _XKS * jnp.sqrt(kf)
    t2 = sigma / (4.0 * ks * ks * rho * rho)

    a = beta / _GAMMA / jnp.expm1(-ec / _GAMMA)
    y = a * t2
    xy = (1.0 + y) / (1.0 + y + y * y)
    h0 = _GAMMA * jnp.log(1.0 + beta / _GAMMA * t2 * xy)
    return E2 * rho * h0


# --- the spin-polarized gradient correction -----------------------------------
#
# As in the local part, exchange needs nothing: ``gcx_spin`` doubles each
# channel's density *and quadruples its |grad rho|^2* before calling the
# unpolarized routine, which is the spin-scaling relation written for
# ``sigma = |grad rho|^2``. Correlation does need its own version, and its
# structure differs from exchange's in a way worth stating: it is a function of
# the **total** density, its polarization, and the gradient of the **total**
# density -- not of the two channels separately. That is why QE's ``gcc_spin``
# takes ``(rho, zeta, grho)`` and returns a single ``v2c``, while ``gcx_spin``
# takes two of everything.


#: ``pbec_spin``'s ``gamma``, which is **not** ``pbec``'s. The unpolarized
#: routine carries 0.0310906908696548950 and the polarized one the rounded
#: 0.031091 -- a relative difference of 2e-6 that QE has never reconciled. It is
#: reproduced rather than unified because the point of this code is to agree
#: with that Fortran, and 2e-6 in ``gamma`` is visible in the sixth decimal of a
#: PBE correlation energy.
_GAMMA_SPIN = 0.031091


def no_gradient_correlation_spin(rho, zeta, sigma):
    """QE's ``NOGC``, spin-polarized."""
    return jnp.zeros_like(rho)


def pbe_correlation_spin(rho, zeta, sigma, beta: float = PBE_BETA):
    """PBE correlation beyond the local part, Ry/bohr^3 (``pbec_spin``).

    The polarized ``H(rho, zeta, t)``. Two things change relative to the
    unpolarized form and both come from the same place -- the spin scaling of
    the Thomas-Fermi screening length:

    * ``phi(zeta) = ((1+z)^(2/3) + (1-z)^(2/3)) / 2`` divides the reduced
      gradient, so ``t = |grad rho| / (2 phi k_s rho)``;
    * every ``gamma`` in the expression becomes ``phi^3 gamma``, which is what
      makes ``H`` reduce to the unpolarized one at ``z = 0`` and vanish in the
      fully polarized high-density limit at the right rate.

    ``ec`` underneath is the **spin-polarized** Perdew-Wang energy, so the
    dependence QE's hand-derived ``v1c_up``/``v1c_dw`` have to carry explicitly
    (they contain ``vc_up - ec`` and a ``dh0/dzeta``) is here just a term in an
    expression that ``jax.grad`` differentiates.
    """
    z = jnp.clip(zeta, -1.0, 1.0)
    rs = _PI34 / rho ** (1.0 / 3.0)
    ec = pw_spin_hartree(rs, z)

    kf = _XKF / rs
    ks = _XKS * jnp.sqrt(kf)

    phi = 0.5 * ((1.0 + z) ** (2.0 / 3.0) + (1.0 - z) ** (2.0 / 3.0))
    phi3 = phi**3
    t2 = sigma / (4.0 * phi * phi * ks * ks * rho * rho)

    a = beta / _GAMMA_SPIN / jnp.expm1(-ec / (phi3 * _GAMMA_SPIN))
    y = a * t2
    xy = (1.0 + y) / (1.0 + y + y * y)
    h0 = phi3 * _GAMMA_SPIN * jnp.log(1.0 + beta / _GAMMA_SPIN * t2 * xy)
    return E2 * rho * h0
