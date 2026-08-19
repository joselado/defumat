"""The gradient correction to the PAW one-centre exchange-correlation.

``PAW_gcxc_potential`` in ``PW/src/paw_onecenter.f90``. The local part of the
functional only needs the density at each quadrature direction; a gradient
correction needs ``grad rho`` there too, and then the divergence of a vector
field to turn ``v2`` back into a potential -- the same two-term functional
derivative the plane-wave path takes in G space (:mod:`pypresso.scf.potential`),
but in spherical coordinates on an atom's own radial mesh.

Three things about the spherical version are not obvious and are QE's:

* **The gradient's angular part costs nothing extra.** ``rho`` is carried as
  multipoles, so its angular derivatives are the *harmonics'* angular
  derivatives contracted with the same ``rho_lm`` -- tabulated once per species
  (``dylmt``, ``dylmp`` in :mod:`pypresso.paw.angular`). Only the radial
  derivative is taken numerically, by the three-point non-uniform-mesh formula
  of ``Modules/radial_gradients.f90`` transcribed in :func:`radial_derivative`.

* **The vector field is expanded further than the density.** ``h = v2 grad
  rho`` is projected onto ``(l_rho + 2)^2`` multipoles rather than the density's
  ``(l_rho + 1)^2``, because taking a divergence loses two multipoles and QE
  wants the *output* accurate to the density's own ``l``. That is the ``ladd``
  the angular grid is enlarged for.

* **The ``theta`` component is divided by ``sin(theta)`` before it is
  projected**, and the ``sin(theta)`` reappears inside the divergence's angular
  sum. QE's comment says why: the ``lm`` expansion of ``dY/dtheta`` converges
  very slowly, while the same derivative divided by ``sin(theta)`` converges
  fast enough for a modest ``ladd``. Skipping the trick and raising ``ladd``
  instead would need an angular grid several times larger.

The components are ordered ``(r, phi, theta)`` throughout, which is the order
``PAW_gradient`` fills them in -- not the order its comment claims.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["radial_derivative", "onecenter_gradient_correction"]


def radial_derivative(f: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
    """``df/dr`` on a logarithmic mesh, over the last axis.

    ``radial_gradient`` with ``iflag = 0``, which is what ``paw_variables.f90``
    fixes ``radial_grad_style`` to. It is the three-point formula for an
    unequally spaced mesh -- a parabola through ``(i-1, i, i+1)``, differentiated
    at ``i`` -- with the two ends handled as QE handles them: the derivative
    vanishes at the outer end, where the functions this is applied to have long
    since decayed, and the first point is linearly extrapolated from the second
    and third.

    Transcribed rather than replaced by ``jnp.gradient``: that would use the same
    parabola in the interior but different end conditions, and the first few
    points of a logarithmic mesh are where a radial derivative is most delicate.
    """
    lead = f.shape[:-1]
    step_up = r[2:] - r[1:-1]
    step_down = r[:-2] - r[1:-1]

    interior = (
        step_up**2 * (f[..., :-2] - f[..., 1:-1])
        - step_down**2 * (f[..., 2:] - f[..., 1:-1])
    ) / (step_up * step_down * (r[2:] - r[:-2]))

    outer = jnp.zeros(lead + (1,), dtype=interior.dtype)
    first = interior[..., :1] + (interior[..., 1:2] - interior[..., :1]) * (
        (r[0] - r[1]) / (r[2] - r[1])
    )
    return jnp.concatenate([first, interior, outer], axis=-1)


def onecenter_gradient_correction(rho_lm, rho_rad, core, paw):
    """``(v_lm, energy)``: what a GGA adds to one on-site potential and energy.

    Args:
        rho_lm: ``(nspin, nlm, mesh)``, holding ``r^2 rho_lm`` as everything in
            :mod:`pypresso.paw.onecenter` does.
        rho_rad: ``(nspin, nx, mesh)``, the same density already put on the
            sphere -- the local part needed it too, so it is passed in rather
            than rebuilt.
        core: ``(mesh,)`` core charge, spherical, shared equally between the
            channels (``co2 = rho_core / nspin_gga`` in ``PAW_gcxc_potential``).
        paw: the species' precomputed tables.
    """
    nlm = paw.nlm
    r2 = paw.r2
    nspin = rho_lm.shape[0]
    weighted = paw.angular.weighted_ylm

    if nspin == 1:
        # ``rho_full(ixk,1) = ABS(...)``: QE takes the absolute value in the
        # unpolarized branch only, so it stays inside this one.
        density = jnp.abs(rho_rad[0] / r2 + core)  # (nx, mesh)

        grad = _gradient(rho_lm[0], density, paw)  # (3, nx, mesh)
        sigma = jnp.sum(grad * grad, axis=0)

        v1, v2 = paw.functional.gradient_potentials(density, sigma)
        energy_density = paw.functional.gradient_energy(density, sigma)

        # h = v2 grad rho, with the r^2 that the divergence expects to find in
        # its input, and the theta component divided by sin(theta) -- see the
        # module docstring.
        h = v2[None, ...] * grad * r2[None, None, :]
        h = h.at[2].divide(paw.angular.sin_theta[:, None])

        v_lm = jnp.einsum("xl,xr->lr", weighted[:, :nlm], v1)
        h_lm = jnp.einsum("xl,cxr->clr", weighted, h)
        potential = (v_lm - _divergence(h_lm, paw))[None]
    else:
        density = rho_rad / r2 + core / nspin  # (nspin, nx, mesh)
        grad = jnp.stack(
            [_gradient(rho_lm[s], density[s], paw) for s in range(nspin)]
        )  # (nspin, 3, nx, mesh)

        # ``h`` comes out of the differentiation already carrying the cross term
        # QE adds by hand: correlation depends on the *total* gradient, so
        # ``d e / d(grad rho_up)`` sees ``grad rho_dw`` too. That is ``v2cud``,
        # and here it is not a separate quantity at all.
        v1, h = paw.functional.spin_gradient_terms(density, grad)
        energy_density = paw.functional.spin_gradient_energy(density, grad)

        h = h * r2
        h = h.at[:, 2].divide(paw.angular.sin_theta[:, None])

        v_lm = jnp.einsum("xl,sxr->slr", weighted[:, :nlm], v1)
        h_lm = jnp.einsum("xl,scxr->sclr", weighted, h)
        potential = v_lm - jax.vmap(_divergence, in_axes=(0, None))(h_lm, paw)

    # The energy integrates over the sphere with the quadrature weights and over
    # the mesh with Simpson's, against r^2 -- the r^2 that ``rho_lm`` carries
    # and this density does not.
    energy = jnp.sum(
        paw.angular.weights[:, None] * energy_density * (r2 * paw.weights_full)[None, :]
    )
    return potential, energy


def _gradient(rho_lm, density, paw):
    """``PAW_gradient``: ``grad rho`` in spherical components on each direction.

    The angular components carry ``1/r^3``: one factor of ``1/r`` from the
    gradient in spherical coordinates, and ``1/r^2`` because ``rho_lm`` holds
    ``r^2 rho`` and the angular derivative passes straight through it.
    """
    nlm = paw.nlm
    radial = radial_derivative(density, paw.r)  # (nx, mesh)

    # lm = 0 is left out exactly as QE leaves it out: the derivative of a
    # constant harmonic is zero, so the spherical component of the density --
    # the core charge included -- contributes nothing to the angular gradient.
    azimuthal = jnp.einsum("xl,lr->xr", paw.angular.dylmp[:, 1:nlm], rho_lm[1:nlm])
    polar = jnp.einsum("xl,lr->xr", paw.angular.dylmt[:, 1:nlm], rho_lm[1:nlm])

    inverse_r3 = 1.0 / (paw.r * paw.r2)
    return jnp.stack([radial, azimuthal * inverse_r3, polar * inverse_r3])


def _divergence(h_lm, paw):
    """``PAW_divergence``: ``div h`` back on the density's own multipoles.

    In spherical coordinates,

        div h = (1/r^2) d(r^2 h_r)/dr
              + (1/(r sin t)) d(h_t sin t)/dt
              + (1/(r sin t)) d h_p / dp,

    and the middle term is what the ``2 Y_lm cos(theta)`` below comes from: with
    ``h_t`` expanded in harmonics *divided* by ``sin(theta)``, differentiating
    ``h_t sin(theta)`` gives ``dY/dtheta sin(theta) + 2 Y cos(theta)``.
    """
    angular = paw.angular
    polar_operator = angular.dylmt * angular.sin_theta[:, None] + 2.0 * (
        angular.ylm * angular.cos_theta[:, None]
    )
    on_sphere = jnp.einsum("xl,lr->xr", angular.dylmp, h_lm[1]) + jnp.einsum(
        "xl,lr->xr", polar_operator, h_lm[2]
    )

    # Back onto the density's multipoles, with the 1/r^3 the two angular terms
    # share (1/r from the derivative, 1/r^2 from the r^2 the field carries) ...
    nlm = paw.nlm
    divergence = jnp.einsum("xl,xr->lr", angular.weighted_ylm[:, :nlm], on_sphere)
    divergence = divergence / (paw.r * paw.r2)[None, :]

    # ... and the radial component, which is already in the right form: the
    # field carries r^2, so d(r^2 h_r)/dr is the derivative of what is stored.
    return divergence + radial_derivative(h_lm[0, :nlm], paw.r) / paw.r2[None, :]
