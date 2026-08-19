"""Radial pseudopotential data transformed into reciprocal space.

Every pseudopotential quantity enters the plane-wave code as a function of
``|G|`` (or ``|k+G|``) obtained by a radial integral. QE precomputes each of
these on a uniform ``q`` grid with spacing ``dq = 0.01`` and interpolates with a
cubic polynomial; here the integral is evaluated directly at the ``|G|`` actually
needed.

That choice is deliberate. It is slightly more accurate (no interpolation error),
and more importantly it keeps the result a differentiable function of ``q`` --
and therefore of ``k`` and of the cell. A spline table would break the chain that
makes the velocity operator fall out of ``jacfwd`` of ``H(k)`` (rule D2). The
price is arithmetic, mitigated by chunking; if it ever matters, the replacement
is a *differentiable* interpolation, not a lookup.

Conventions follow ``upflib``: ``vloc_mod.f90``, ``rhoat_mod.f90``,
``rhoc_mod.f90`` and ``beta_mod.f90``. Units are Rydberg atomic units, ``q`` in
1/bohr, volumes in bohr^3.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf

from pypresso.pseudo.radial import simpson_weights, spherical_bessel
from pypresso.pseudo.upf import Pseudopotential
from pypresso.units import E2, FPI

__all__ = [
    "local_potential_of_g",
    "atomic_charge_of_g",
    "core_charge_of_g",
    "projector_form_factors",
    "atomic_form_factors",
    "CHUNK",
]

#: Number of q values transformed at once. The intermediate is (chunk, mesh), so
#: this bounds memory at a few tens of MB while keeping the work as large matrix
#: products.
CHUNK = 4096

# The four kernels below are module-level and jitted rather than closures defined
# per call. Each radial transform is ~30 elementwise operations on a (nq, mesh)
# intermediate; dispatched eagerly, XLA compiles and launches every one of them
# separately, which is where most of a cold run's setup time went. As one
# compiled unit they fuse into a single pass over the intermediate, and the
# compilation is cached across species and across calculations -- a closure
# would be a new callable each time and so a new compilation each time.


def _chunked(function, q: jnp.ndarray) -> jnp.ndarray:
    """Apply a vectorised transform to ``q`` in bounded-size pieces."""
    q = jnp.atleast_1d(jnp.asarray(q))
    if q.shape[0] <= CHUNK:
        return function(q)
    pieces = [function(q[start : start + CHUNK]) for start in range(0, q.shape[0], CHUNK)]
    return jnp.concatenate(pieces, axis=-1)


def _truncated(pseudo: Pseudopotential):
    """The mesh QE integrates over, with its Simpson weights."""
    msh = pseudo.msh
    r = jnp.asarray(pseudo.r[:msh])
    weights = simpson_weights(jnp.asarray(pseudo.rab[:msh]))
    return r, weights, msh


def local_potential_of_g(pseudo: Pseudopotential, q, omega: float) -> jnp.ndarray:
    """Fourier transform of the local potential, ``V_loc(q)`` in Ry.

    The bare potential is long-ranged (``-Z e^2 / r``) and its transform diverges
    as ``1/q^2``, so it cannot be integrated numerically. QE's trick, reproduced
    here, is to add ``Z e^2 erf(r)/r`` inside the integral -- making the
    integrand short-ranged -- and subtract that function's analytic transform
    ``4 pi Z e^2 exp(-q^2/4) / (Omega q^2)`` outside it.

    At ``q = 0`` the remaining integral is the ``alpha Z`` term: the average of
    the potential over the cell, finite only because the divergence cancels
    against the Hartree and Ewald ``G = 0`` terms.
    """
    r, weights, msh = _truncated(pseudo)
    vloc = jnp.asarray(pseudo.vloc[:msh])
    z = pseudo.z_valence

    # Short-ranged integrand for q > 0: r^2 [V(r) + Z e^2 erf(r) / r]
    short = r * vloc + z * E2 * erf(r)
    # The q = 0 term is *not* the q -> 0 limit of that expression, and QE says so
    # in as many words. The erf is part of the splitting that makes the q > 0
    # integral converge; at q = 0 what is wanted is the average of the potential
    # with its bare Coulomb tail removed, so the screening function is 1, not
    # erf(r). Using the erf form here shifts every eigenvalue by a constant --
    # a convincingly self-consistent calculation with the wrong absolute energy.
    at_zero = r * (r * vloc + z * E2)

    return _chunked(lambda qq: _vloc_kernel(qq, r, weights, short, at_zero, z, omega), q)


@jax.jit
def _vloc_kernel(qq, r, weights, short, at_zero, z, omega):
    qq = qq[:, None]
    small = qq[:, 0] < 1e-8
    safe = jnp.where(qq < 1e-8, 1.0, qq)
    integrand = jnp.where(small[:, None], at_zero[None, :],
                          short[None, :] * jnp.sin(safe * r[None, :]) / safe)
    value = integrand @ weights * FPI / omega

    analytic = FPI / omega * z * E2 * jnp.exp(-safe[:, 0] ** 2 * 0.25) / safe[:, 0] ** 2
    return jnp.where(small, value, value - analytic)


def atomic_charge_of_g(pseudo: Pseudopotential, q, omega: float) -> jnp.ndarray:
    """Transform of the atomic charge density used to start the SCF.

    ``PP_RHOATOM`` is tabulated as ``4 pi r^2 rho(r)``, so the transform is a
    plain ``j_0`` integral and ``rho(q=0) = Z_valence / Omega``.
    """
    if pseudo.rho_atom is None:
        raise ValueError(f"{pseudo.element}: the UPF file has no PP_RHOATOM section")

    r, weights, msh = _truncated(pseudo)
    rho = jnp.asarray(pseudo.rho_atom[:msh])

    return _chunked(lambda qq: _rhoat_kernel(qq, r, weights, rho, omega), q)


@jax.jit
def _rhoat_kernel(qq, r, weights, rho, omega):
    qq = qq[:, None]
    small = qq[:, 0] < 1e-8
    safe = jnp.where(qq < 1e-8, 1.0, qq)
    argument = safe * r[None, :]
    integrand = jnp.where(
        small[:, None], rho[None, :], rho[None, :] * spherical_bessel(0, argument)
    )
    return integrand @ weights / omega


def core_charge_of_g(pseudo: Pseudopotential, q, omega: float) -> jnp.ndarray:
    """Transform of the nonlinear core-correction charge (``PP_NLCC``).

    Unlike ``PP_RHOATOM`` this is tabulated as ``rho_c(r)`` itself, so the
    ``4 pi r^2`` measure appears explicitly.
    """
    if pseudo.rho_core is None:
        raise ValueError(f"{pseudo.element}: the UPF file has no PP_NLCC section")

    r, weights, msh = _truncated(pseudo)
    rho = jnp.asarray(pseudo.rho_core[:msh])

    return _chunked(lambda qq: _rhocore_kernel(qq, r, weights, rho, omega), q)


@jax.jit
def _rhocore_kernel(qq, r, weights, rho, omega):
    argument = qq[:, None] * r[None, :]
    integrand = FPI * r[None, :] ** 2 * rho[None, :] * spherical_bessel(0, argument)
    return integrand @ weights / omega


def projector_form_factors(pseudo: Pseudopotential, q, omega: float) -> jnp.ndarray:
    """Radial parts ``f_l(q)`` of the nonlocal projectors, shaped ``(nbeta, nq)``.

    ``PP_BETA`` is tabulated as ``r beta_l(r)``, so the transform is
    ``4 pi / sqrt(Omega) * int dr r beta_l(r) j_l(qr)``. The ``1/sqrt(Omega)``
    rather than ``1/Omega`` is because the projectors multiply wavefunctions,
    which carry their own normalisation.

    Every projector is integrated over the *same* range, the species-wide
    ``kkbeta`` of ``upflib/beta_mod.f90``, rather than each over its own
    ``cutoff_radius_index``. The two differ for a PAW dataset, where ``kkbeta``
    is widened to cover the augmentation sphere -- and the tabulated ``beta``
    is truncated abruptly rather than tapering to zero, so it is still of order
    1e-3 at its own cutoff. Integrating over the shorter range would drop a
    contribution QE keeps.
    """
    prefactor = FPI / np.sqrt(omega)
    q = jnp.atleast_1d(jnp.asarray(q))

    cutoff = pseudo.kkbeta
    r = jnp.asarray(pseudo.r[:cutoff])
    weights = simpson_weights(jnp.asarray(pseudo.rab[:cutoff]))

    rows = []
    for projector in pseudo.projectors:
        beta = jnp.asarray(projector.beta[:cutoff])
        l = projector.l

        rows.append(_chunked(
            lambda qq, r=r, weights=weights, beta=beta, l=l:
                _beta_kernel(qq, r, weights, beta, prefactor, l),
            q,
        ))

    if not rows:
        return jnp.zeros((0,) + q.shape)
    return jnp.stack(rows, axis=0)


def atomic_form_factors(pseudo: Pseudopotential, q, omega: float) -> jnp.ndarray:
    """Radial parts of the pseudo-atomic orbitals, shaped ``(nwfc, nq)``.

    ``upflib/atwfc_mod.f90``. ``PP_CHI`` is tabulated as ``r chi(r)``, exactly as
    ``PP_BETA`` is, so this is the same integral as
    :func:`projector_form_factors` with the same prefactor -- the difference is
    the mesh it runs over (QE's 10-bohr truncation rather than each projector's
    own cutoff) and that orbitals with negative occupation are skipped, as QE
    skips them.
    """
    prefactor = FPI / np.sqrt(omega)
    q = jnp.atleast_1d(jnp.asarray(q))
    r, weights, _ = _truncated(pseudo)

    rows = [
        _chunked(
            lambda qq, chi=jnp.asarray(orbital.chi[: r.shape[0]]), l=orbital.l:
                _beta_kernel(qq, r, weights, chi, prefactor, l),
            q,
        )
        for orbital in pseudo.orbitals
        if orbital.occupation >= 0.0
    ]
    if not rows:
        return jnp.zeros((0,) + q.shape)
    return jnp.stack(rows, axis=0)


@partial(jax.jit, static_argnames=("l",))
def _beta_kernel(qq, r, weights, beta, prefactor, l):
    argument = qq[:, None] * r[None, :]
    integrand = beta[None, :] * spherical_bessel(l, argument) * r[None, :]
    return integrand @ weights * prefactor
