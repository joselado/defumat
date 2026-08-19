"""The angular grid the PAW one-centre exchange-correlation is integrated on.

Everything else about a PAW correction is an ``l``-by-``l`` radial problem: the
density is carried as multipoles ``rho_lm(r)`` and the Hartree term stays in
that representation, because Poisson's equation does not mix ``lm``. Exchange
and correlation do -- ``e_xc(rho)`` is not linear -- so the density has to be
put back on the sphere, the functional evaluated pointwise there, and the result
projected back onto the multipoles. That needs a spherical quadrature.

``PW/src/paw_init.f90``'s ``PAW_rad_init`` builds one as a product rule:
Gauss-Legendre in ``cos(theta)`` times equally spaced points in ``phi``, which
is exact for spherical harmonics up to a chosen ``lmax``. The choice is QE's:
``lmax = 3 * lmax_rho``, three times what the density itself carries, because
``v_xc`` of a density with multipoles up to ``L`` has multipoles well past ``L``
and truncating the quadrature at ``L`` would alias them back down. For silicon
that is ``lmax = 6``, so 4 Gauss nodes in theta and 7 in phi -- 28 directions.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from pypresso.pseudo.harmonics import real_spherical_harmonics

__all__ = ["AngularGrid", "build_angular_grid", "LM_FACTOR"]

#: ``lm_fact`` in ``upflib/paw_variables.f90``: integrate the exchange-correlation
#: term up to this multiple of the density's own ``lmax``.
LM_FACTOR = 3


class AngularGrid(eqx.Module):
    """Directions, weights, and the harmonics evaluated on them.

    ``weights`` sums to ``4 pi``, so ``sum_x w_x f(x)`` is the integral of ``f``
    over the sphere and ``sum_x w_x Y_lm(x) Y_l'm'(x) = delta`` within the
    quadrature's exactness.
    """

    weights: jnp.ndarray  # (nx,)
    ylm: jnp.ndarray  # (nx, nlm) -- only the components the density carries
    weighted_ylm: jnp.ndarray  # (nx, nlm) -- QE's wwylm, weights * ylm

    @property
    def nx(self) -> int:
        return self.weights.shape[0]


def build_angular_grid(lmax_rho: int, nlm: int) -> AngularGrid:
    """The grid for a density carrying multipoles up to ``lmax_rho``.

    Args:
        lmax_rho: the density's own highest multipole (``l_max_rho`` in the UPF
            header, normally twice the highest projector ``l``).
        nlm: how many ``lm`` components to tabulate the harmonics for -- the
            density's ``(lmax_rho + 1)^2``. The *quadrature* is built for the
            larger ``LM_FACTOR * lmax_rho``; only the projection needs the
            harmonics themselves.
    """
    lmax = LM_FACTOR * max(lmax_rho, 0)
    nphi = lmax + 1 + lmax % 2
    ntheta = (lmax + 2) // 2

    nodes, weights = np.polynomial.legendre.leggauss(ntheta)
    phi = 2.0 * np.pi * np.arange(nphi) / nphi

    z = np.repeat(nodes, nphi)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z**2))
    directions = np.stack(
        [radius * np.cos(np.tile(phi, ntheta)),
         radius * np.sin(np.tile(phi, ntheta)),
         z],
        axis=1,
    )
    ww = np.repeat(weights, nphi) * 2.0 * np.pi / nphi

    ylm = np.asarray(real_spherical_harmonics(jnp.asarray(directions), lmax_rho))[:, :nlm]
    return AngularGrid(
        weights=jnp.asarray(ww),
        ylm=jnp.asarray(ylm),
        weighted_ylm=jnp.asarray(ww[:, None] * ylm),
    )
