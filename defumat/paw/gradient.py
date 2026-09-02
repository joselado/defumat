"""The gradient correction to the PAW one-centre exchange-correlation.

``PAW_gcxc_potential`` in ``PW/src/paw_onecenter.f90``. The local part of the
functional only needs the density at each quadrature direction; a gradient
correction needs ``grad rho`` there too, and then the divergence of a vector
field to turn ``v2`` back into a potential -- the same two-term functional
derivative the plane-wave path takes in G space (:mod:`defumat.scf.potential`),
but in spherical coordinates on an atom's own radial mesh.

Three things about the spherical version are not obvious and are QE's:

* **The gradient's angular part costs nothing extra.** ``rho`` is carried as
  multipoles, so its angular derivatives are the *harmonics'* angular
  derivatives contracted with the same ``rho_lm`` -- tabulated once per species
  (``dylmt``, ``dylmp`` in :mod:`defumat.paw.angular`). Only the radial
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


def onecenter_gradient_correction(rho_lm, rho_rad, core, paw, axis=None):
    """``(v_lm, energy)``: what a GGA adds to one on-site potential and energy.

    Args:
        rho_lm: ``(nspin, nlm, mesh)``, holding ``r^2 rho_lm`` as everything in
            :mod:`defumat.paw.onecenter` does.
        rho_rad: ``(nspin, nx, mesh)``, the same density already put on the
            sphere -- the local part needed it too, so it is passed in rather
            than rebuilt.
        core: ``(mesh,)`` core charge, spherical, shared equally between the
            channels (``co2 = rho_core / nspin_gga`` in ``PAW_gcxc_potential``).
        paw: the species' precomputed tables.
        axis: the fixed quantization axis (``compute_ux``), or ``None``. Read
            only by the ``nspin = 4`` branch, and there for the reason
            :func:`defumat.scf.potential.fixed_quantization_axis` gives: the
            naive ``(n +- |m|)/2`` has a kink wherever ``m`` passes through
            zero, and a kink in the density is a divergence in its gradient.
    """
    nlm = paw.nlm
    r2 = paw.r2
    nspin = rho_lm.shape[0]
    weighted = paw.angular.weighted_ylm

    if nspin == 4:
        return _noncollinear_gradient(rho_lm, rho_rad, core, paw, axis)

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


def _noncollinear_gradient(rho_lm, rho_rad, core, paw, axis):
    """``PAW_gcxc_potential``'s ``nspin = 4`` branch, on the radial sphere.

    ``compute_rho_spin_lm`` in, ``compute_pot_nonc`` out, and between them the
    ordinary two-channel code -- the same three steps
    :func:`defumat.scf.potential._noncollinear_gradient_correction` takes on
    the plane-wave grid, which is the point: there is one local-spin-frame
    construction and the sphere calls into it rather than restating it.

    1. **Rotate.** ``rho_up/dw = (n +- s|m|)/2`` at every (direction, radius) of
       the quadrature, with ``s = sign(m . ux)`` where there is a fixed axis and
       ``+1`` where ``|m|`` vanishes. The frozen core is unpolarized, so it goes
       wholly into the charge before the split and half lands in each channel --
       QE's ``co2 = rho_core / nspin_gga``.

    2. **Project back afresh, and this is what the refusal that stood here was
       about.** The rotated channels' *multipoles* are recomputed by quadrature
       from their grid values (``PAW_rad2lm``) rather than assembled from
       ``rho_lm``: the rotation runs through ``|m|`` and is not linear in the
       components, so no combination of the stored multipoles is the expansion
       of the result. The angular part of the gradient and the divergence both
       read those.

    3. **Rotate back** on the radial grid, where the direction lives:
       ``v_0 = (v_up + v_dw)/2`` in the charge component and
       ``s (v_up - v_dw)/2 m-hat`` in the other three, then one last projection
       onto the multipoles the caller wants.

    **Not reproduced: ``add_small_mag``.** A fully-relativistic dataset's small
    component carries magnetization of its own, and QE folds it in here and in
    ``compute_pot_nonc``. The *local* part of this package's one-centre XC does
    not fold it in either, so leaving it out keeps the two halves consistent;
    putting it in one and not the other would be worse than in neither.
    """
    nlm = paw.nlm
    r2 = paw.r2
    weighted = paw.angular.weighted_ylm
    eps = VANISHING_RADIAL_MAGNETIZATION

    charge = rho_rad[0] / r2 + core
    magnetization = rho_rad[1:] / r2
    modulus = jnp.sqrt(jnp.sum(magnetization**2, axis=0))
    if axis is None:
        sign = jnp.ones_like(modulus)
    else:
        projection = jnp.tensordot(jnp.asarray(axis), magnetization, axes=(0, 0))
        sign = jnp.where(projection >= 0.0, 1.0, -1.0)
    sign = jnp.where(modulus < eps, 1.0, sign)

    signed = sign * modulus
    channels = 0.5 * jnp.stack([charge + signed, charge - signed])  # (2, nx, mesh)
    channel_lm = jnp.einsum("xl,sxr->slr", weighted[:, :nlm], channels * r2)

    grad = jnp.stack([
        _gradient(channel_lm[s], channels[s], paw) for s in range(2)
    ])  # (2, 3, nx, mesh)
    v1, h = paw.functional.spin_gradient_terms(channels, grad)
    energy_density = paw.functional.spin_gradient_energy(channels, grad)

    h = h * r2
    h = h.at[:, 2].divide(paw.angular.sin_theta[:, None])
    v_lm = jnp.einsum("xl,sxr->slr", weighted[:, :nlm], v1)
    h_lm = jnp.einsum("xl,scxr->sclr", weighted, h)
    out_lm = v_lm - jax.vmap(_divergence, in_axes=(0, None))(h_lm, paw)

    out_rad = jnp.einsum("xl,slr->sxr", paw.angular.ylm[:, :nlm], out_lm)
    v0 = 0.5 * (out_rad[0] + out_rad[1])
    vs = 0.5 * (out_rad[0] - out_rad[1])
    safe = jnp.where(modulus > 0.0, modulus, 1.0)
    direction = jnp.where(modulus > eps, magnetization / safe, 0.0)
    potential_rad = jnp.concatenate([v0[None], (sign * vs)[None] * direction])
    potential = jnp.einsum("xl,sxr->slr", weighted[:, :nlm], potential_rad)

    energy = jnp.sum(
        paw.angular.weights[:, None] * energy_density * (r2 * paw.weights_full)[None, :]
    )
    return potential, energy


#: ``eps12`` in ``compute_rho_spin_lm``: below this magnetization the local axis
#: is undefined, the sign is taken as ``+1`` and the vector part of the
#: potential is left at zero.
VANISHING_RADIAL_MAGNETIZATION = 1.0e-12


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
