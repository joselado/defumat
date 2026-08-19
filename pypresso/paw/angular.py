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

**A gradient-corrected functional needs a bigger grid, and more on it.**
``paw_init.f90`` adds ``xlm = 2`` to that ``lmax`` when the functional is a GGA
(``lmax_add``), taking silicon to ``lmax = 8`` and 45 directions. The reason is
in the divergence: the potential is assembled from a vector field expanded to
``l_rho + 2``, and the quadrature has to integrate *that* exactly, not just the
density. Reusing the local functional's grid converges to a slightly different
number, which is the kind of error that looks like a bug in the functional.

The same condition brings in ``dylmt`` and ``dylmp``, the derivatives of the
harmonics along the ``theta`` and ``phi`` versors, which is how an angular
gradient is taken in this representation. QE builds them by finite-differencing
``ylmr2`` in each cartesian direction and projecting; here the cartesian
derivative comes from ``jax.jacfwd`` of the harmonics themselves, which is exact
rather than accurate to ``delta^2`` and needs no step to choose. The projection
onto the versors is QE's, including the detail that ``dylmp`` carries no
``1/sin(theta)``: projecting on the ``phi`` versor already supplies it.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from pypresso.pseudo.harmonics import real_spherical_harmonics

__all__ = ["AngularGrid", "build_angular_grid", "LM_FACTOR", "XLM"]

#: ``lm_fact`` in ``upflib/paw_variables.f90``: integrate the exchange-correlation
#: term up to this multiple of the density's own ``lmax``.
LM_FACTOR = 3

#: ``xlm``: what a gradient-corrected functional adds on top of that, both to
#: the quadrature's exactness and to the multipoles the vector field is expanded
#: in (``ladd``).
XLM = 2


class AngularGrid(eqx.Module):
    """Directions, weights, and the harmonics evaluated on them.

    ``weights`` sums to ``4 pi``, so ``sum_x w_x f(x)`` is the integral of ``f``
    over the sphere and ``sum_x w_x Y_lm(x) Y_l'm'(x) = delta`` within the
    quadrature's exactness.

    The harmonic tables carry ``nlm_max`` columns: the density's own ``nlm`` for
    a local functional, and ``(l_rho + 1 + ladd)^2`` for a gradient-corrected
    one, since the vector field whose divergence gives the potential is expanded
    further than the density is. Consumers slice to what they need.
    """

    weights: jnp.ndarray  # (nx,)
    ylm: jnp.ndarray  # (nx, nlm_max)
    weighted_ylm: jnp.ndarray  # (nx, nlm_max) -- QE's wwylm, weights * ylm
    #: Derivatives along the theta and phi versors, and the two trigonometric
    #: factors the divergence needs. Empty unless the functional is a GGA.
    dylmt: jnp.ndarray
    dylmp: jnp.ndarray
    sin_theta: jnp.ndarray
    cos_theta: jnp.ndarray

    @property
    def nx(self) -> int:
        return self.weights.shape[0]


def build_angular_grid(lmax_rho: int, nlm: int, gradient: bool = False) -> AngularGrid:
    """The grid for a density carrying multipoles up to ``lmax_rho``.

    Args:
        lmax_rho: the density's own highest multipole (``l_max_rho`` in the UPF
            header, normally twice the highest projector ``l``).
        nlm: how many ``lm`` components to tabulate the harmonics for -- the
            density's ``(lmax_rho + 1)^2``. The *quadrature* is built for the
            larger ``LM_FACTOR * lmax_rho``; only the projection needs the
            harmonics themselves.
        gradient: whether the functional is gradient-corrected. Adds ``XLM`` to
            the quadrature's exactness, extends the harmonic tables to
            ``(sqrt(nlm) + XLM)^2`` components, and builds the angular
            derivatives.
    """
    lmax = LM_FACTOR * max(lmax_rho, 0) + (XLM if gradient and lmax_rho else 0)
    nphi = lmax + 1 + lmax % 2
    ntheta = (lmax + 2) // 2

    nodes, weights = np.polynomial.legendre.leggauss(ntheta)
    phi = 2.0 * np.pi * np.arange(nphi) / nphi

    z = np.repeat(nodes, nphi)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z**2))
    azimuth = np.tile(phi, ntheta)
    directions = np.stack([radius * np.cos(azimuth), radius * np.sin(azimuth), z], axis=1)
    ww = np.repeat(weights, nphi) * 2.0 * np.pi / nphi

    # The density's own l, and the one the tables are built to: a gradient
    # correction expands its vector field ``ladd`` multipoles further.
    l_rho = int(round(np.sqrt(nlm))) - 1
    lmax_table = l_rho + (XLM if gradient else 0)
    nlm_table = (lmax_table + 1) ** 2

    ylm = np.asarray(real_spherical_harmonics(jnp.asarray(directions), lmax_table))
    ylm = ylm[:, :nlm_table]

    if gradient:
        dylmt, dylmp = _angular_derivatives(directions, azimuth, lmax_table, nlm_table)
    else:
        dylmt = dylmp = np.zeros((0, 0))

    return AngularGrid(
        weights=jnp.asarray(ww),
        ylm=jnp.asarray(ylm),
        weighted_ylm=jnp.asarray(ww[:, None] * ylm),
        dylmt=jnp.asarray(dylmt),
        dylmp=jnp.asarray(dylmp),
        sin_theta=jnp.asarray(radius),
        cos_theta=jnp.asarray(z),
    )


def _angular_derivatives(directions, azimuth, lmax: int, nlm: int):
    """``dylmt`` and ``dylmp``: the harmonics' gradient along the two versors.

    The cartesian gradient of ``Y_lm`` at each direction comes from ``jacfwd``;
    projecting it onto

        phi_hat   = (-sin phi, cos phi, 0)
        theta_hat = phi_hat x r_hat

    gives the two components. QE forms exactly these two products
    (``paw_init.f90``), building ``theta_hat`` by the same cross product rather
    than from its explicit trigonometric form.
    """
    jacobian = jax.jacfwd(lambda v: real_spherical_harmonics(v, lmax))
    gradients = np.asarray(jax.vmap(jacobian)(jnp.asarray(directions)))  # (nx, nlm, 3)
    gradients = gradients[:, :nlm, :]

    phi_hat = np.stack([-np.sin(azimuth), np.cos(azimuth), np.zeros_like(azimuth)], axis=1)
    theta_hat = np.cross(phi_hat, directions)

    return (
        np.einsum("xli,xi->xl", gradients, theta_hat),
        np.einsum("xli,xi->xl", gradients, phi_hat),
    )
