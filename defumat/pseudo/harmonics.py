"""Real spherical harmonics, in Quantum ESPRESSO's convention and ordering.

Transcribed from ``upflib/ylmr2.f90``. Both the normalisation *and* the ordering
matter: the projector coefficients ``D_ij`` are indexed by the same ``lm``, so a
different ordering silently scrambles the nonlocal potential rather than failing
loudly.

The ordering, for each ``l``, is m = 0, then (cos, sin) pairs for m = 1..l:

    lm = l^2 + 1        -> m =  0
    lm = l^2 + 2m       -> m = +m  (cos m phi)
    lm = l^2 + 2m + 1   -> m = -m  (sin m phi)

The functions are real and orthonormal on the sphere.

This is written to be differentiable in the direction vector, because the
velocity operator comes from differentiating ``vkb(k)`` with respect to ``k``
(rule D2), and ``vkb`` contains ``Y_lm(k+G)``, and because the stress
differentiates the same expression with respect to the cell (P11).

**The azimuthal angle is never formed.** ``ylmr2`` writes ``Y_lm`` as
``sin^m(theta) * P(cos theta) * {cos, sin}(m phi)`` and gets ``phi`` from an
``atan2``, which is a parameterisation with a coordinate singularity on the
``z`` axis: ``sin theta`` has an infinite derivative there and ``phi`` is
undefined. The *function* is perfectly smooth -- ``r^l Y_lm`` is a polynomial in
``(x, y, z)`` -- and the singularity is entirely the fault of the spherical
coordinates, so a gradient taken through them is NaN rather than wrong. That is
not hypothetical: fcc silicon at ``ecutrho = 48`` has **ten** dense G-vectors
exactly on the ``z`` axis, and every one of them poisons the whole stress
tensor.

The two factors are therefore combined before they are evaluated. Writing
``rho = sqrt(x^2 + y^2)``,

    sin^m(theta) cos(m phi) = Re[(x + iy)^m] / r^m,
    sin^m(theta) sin(m phi) = Im[(x + iy)^m] / r^m,

both polynomials in ``x`` and ``y`` divided by a power of ``r``, and hence
smooth wherever ``r > 0``. What is recursed on is then ``Q(l,m) / sin^m theta``,
a polynomial in ``cos theta``, which obeys exactly ``ylmr2``'s recursion with
the one ``sent`` factor in the ``m = l`` step removed. The values are the
Fortran's; only the route to them avoids the pole.
"""

from __future__ import annotations

import jax.numpy as jnp

from defumat.units import FPI

__all__ = ["real_spherical_harmonics", "real_solid_harmonics", "lm_index"]

_EPS = 1.0e-9


def lm_index(l: int, m: int) -> int:
    """The 0-based column of ``Y_lm`` in QE's ordering.

    ``m > 0`` selects the cosine-like harmonic, ``m < 0`` the sine-like one.
    """
    if abs(m) > l:
        raise ValueError(f"|m| must not exceed l, got l={l}, m={m}")
    if m == 0:
        return l * l
    return l * l + 2 * abs(m) - (1 if m > 0 else 0)


def real_spherical_harmonics(vectors: jnp.ndarray, lmax: int) -> jnp.ndarray:
    """``Y_lm`` for every vector, shaped ``(..., (lmax+1)^2)``.

    Args:
        vectors: ``(..., 3)`` cartesian directions; only the direction matters.
        lmax: highest angular momentum required.

    The recursion follows ``ylmr2``: build ``Q(l,m) = sqrt((l-m)!/(l+m)!) P(l,m)``
    by upward recursion in ``l``, then attach the ``cos(m phi)`` / ``sin(m phi)``
    factors and the normalisation. What is stored here is ``Q(l,m)`` divided by
    ``sin^m theta``, and the missing power is put back by the azimuthal factors,
    which carry it in the smooth form described in the module docstring.
    """
    if lmax < 0:
        raise ValueError(f"lmax must be non-negative, got {lmax}")

    vectors = jnp.asarray(vectors)
    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
    norm2 = x * x + y * y + z * z

    # A zero vector has no direction; QE sets cos(theta) = 0 there. The mask is
    # applied to ``norm2`` *before* the square root, not to the quotient after
    # it: ``sqrt`` has an infinite derivative at zero, so guarding the result
    # leaves the value right and the gradient NaN. That is P15's Ewald trap in
    # a second place, and it only shows up once the cell is differentiated.
    tiny = norm2 < _EPS**2
    norm = jnp.sqrt(jnp.where(tiny, 1.0, norm2))
    cost = jnp.where(tiny, 0.0, z / norm)

    if lmax == 0:
        return jnp.full(vectors.shape[:-1] + (1,), 1.0 / jnp.sqrt(FPI))

    # --- Q(l, m) / sin^m(theta), in the column ylmr2 stores Q(l, m) in ---
    size = (lmax + 1) ** 2
    q = [None] * size
    q[0] = jnp.ones_like(cost)
    q[1] = cost
    q[3] = jnp.full_like(cost, -1.0 / jnp.sqrt(2.0))  # ylmr2's -sent/sqrt(2)

    for l in range(2, lmax + 1):
        for m in range(0, l - 1):
            lm, lm1, lm2 = l**2 + 2 * m, (l - 1) ** 2 + 2 * m, (l - 2) ** 2 + 2 * m
            denominator = jnp.sqrt(float(l * l - m * m))
            q[lm] = (
                cost * (2 * l - 1) / denominator * q[lm1]
                - jnp.sqrt(float((l - 1) ** 2 - m * m)) / denominator * q[lm2]
            )
        lm, lm1, lm2 = l**2 + 2 * l, l**2 + 2 * (l - 1), (l - 1) ** 2 + 2 * (l - 1)
        q[lm1] = cost * jnp.sqrt(float(2 * l - 1)) * q[lm2]
        # ``- sqrt((2l-1)/2l) sent Q(l-1,l-1)`` divided through by sin^l theta:
        # the one place the recursion changes power of ``sent``, and the one
        # place the division has to be accounted for.
        q[lm] = -jnp.sqrt((2 * l - 1) / (2.0 * l)) * q[lm2]

    # --- sin^m(theta) {cos, sin}(m phi) = Re, Im of ((x + i y) / r)^m ---
    # Built by the complex-multiplication recursion rather than by forming the
    # angle: it is the same arithmetic, and it has no branch cut.
    # A zero vector takes ``ylmr2``'s own values there: it sets ``cos theta = 0``
    # and leaves ``sent = 1`` and ``phi = 0``, which in this parameterisation is
    # ``(u, v) = (1, 0)``. Nothing physical reads them -- the radial factor
    # ``f_l(0)`` vanishes for every ``l > 0`` and ``Y_00`` has no direction --
    # but reproducing the Fortran there keeps the two codes' arrays comparable
    # element by element, which is how this one is tested.
    u = jnp.where(tiny, 1.0, x / norm)
    v = jnp.where(tiny, 0.0, y / norm)
    cos_m = [jnp.ones_like(cost)]
    sin_m = [jnp.zeros_like(cost)]
    for m in range(1, lmax + 1):
        cos_m.append(cos_m[-1] * u - sin_m[-1] * v)
        sin_m.append(sin_m[-1] * u + cos_m[-2] * v)

    # --- attach the azimuthal factors and normalise ---
    ylm = [jnp.broadcast_to(q[0] / jnp.sqrt(FPI), cost.shape)]
    for l in range(1, lmax + 1):
        c = jnp.sqrt((2 * l + 1) / FPI)
        ylm.append(c * q[l**2])
        for m in range(1, l + 1):
            base = c * jnp.sqrt(2.0) * q[l**2 + 2 * m]
            ylm.append(base * cos_m[m])
            ylm.append(base * sin_m[m])

    return jnp.stack(ylm, axis=-1)


def real_solid_harmonics(vectors: jnp.ndarray, lmax: int) -> jnp.ndarray:
    """``|v|^l Y_lm(vhat)``, the regular solid harmonics, shaped ``(..., (lmax+1)^2)``.

    The same functions as :func:`real_spherical_harmonics`, in the same
    ordering and with the same signs, multiplied by ``|v|^l``, which makes each
    a homogeneous polynomial of degree ``l`` in the components of ``v``. It is
    therefore smooth to every order at ``v = 0``, where the spherical ones have
    no value at all, and that is what it is for: a projector column
    ``Y_lm(qhat) f_l(|q|)`` is ``S_lm(q) g_l(q^2)`` with ``g_l = f_l/q^l``
    analytic in ``q^2``, and this is the form that can be differentiated at
    ``k + G = 0`` (``pseudo.projectors._with_origin_rows``).

    The recursion is :func:`real_spherical_harmonics`' one with ``cos theta``
    replaced by ``z`` and the ``Q(l-2, m)`` term carrying ``|v|^2``: the stored
    ``Q(l, m) / sin^m theta`` is a polynomial of degree ``l - m`` in
    ``cos theta`` with that parity, so ``|v|^(l-m)`` times it is a polynomial
    in ``z`` and ``|v|^2``, and the azimuthal factor times ``|v|^m`` is
    ``Re, Im (x + iy)^m``. No square root and no division appear.
    """
    if lmax < 0:
        raise ValueError(f"lmax must be non-negative, got {lmax}")

    vectors = jnp.asarray(vectors)
    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
    norm2 = x * x + y * y + z * z
    if lmax == 0:
        return jnp.full(vectors.shape[:-1] + (1,), 1.0 / jnp.sqrt(FPI),
                        dtype=vectors.dtype)

    size = (lmax + 1) ** 2
    q = [None] * size
    q[0] = jnp.ones_like(z)
    q[1] = z
    q[3] = jnp.full_like(z, -1.0 / jnp.sqrt(2.0))

    for l in range(2, lmax + 1):
        for m in range(0, l - 1):
            lm, lm1, lm2 = l**2 + 2 * m, (l - 1) ** 2 + 2 * m, (l - 2) ** 2 + 2 * m
            denominator = jnp.sqrt(float(l * l - m * m))
            q[lm] = (
                z * (2 * l - 1) / denominator * q[lm1]
                - jnp.sqrt(float((l - 1) ** 2 - m * m)) / denominator * norm2 * q[lm2]
            )
        lm, lm1, lm2 = l**2 + 2 * l, l**2 + 2 * (l - 1), (l - 1) ** 2 + 2 * (l - 1)
        q[lm1] = z * jnp.sqrt(float(2 * l - 1)) * q[lm2]
        q[lm] = -jnp.sqrt((2 * l - 1) / (2.0 * l)) * q[lm2]

    cos_m = [jnp.ones_like(z)]
    sin_m = [jnp.zeros_like(z)]
    for m in range(1, lmax + 1):
        cos_m.append(cos_m[-1] * x - sin_m[-1] * y)
        sin_m.append(sin_m[-1] * x + cos_m[-2] * y)

    solid = [jnp.broadcast_to(q[0] / jnp.sqrt(FPI), z.shape)]
    for l in range(1, lmax + 1):
        c = jnp.sqrt((2 * l + 1) / FPI)
        solid.append(c * q[l**2])
        for m in range(1, l + 1):
            base = c * jnp.sqrt(2.0) * q[l**2 + 2 * m]
            solid.append(base * cos_m[m])
            solid.append(base * sin_m[m])

    return jnp.stack(solid, axis=-1)
