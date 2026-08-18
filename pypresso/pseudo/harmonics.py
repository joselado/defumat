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
(rule D2), and ``vkb`` contains ``Y_lm(k+G)``. The two places where a naive
transcription produces NaN gradients rather than wrong numbers -- ``sqrt`` at the
poles and ``atan`` at the origin -- are handled with sanitised arguments.
"""

from __future__ import annotations

import jax.numpy as jnp

from pypresso.units import FPI, PI

__all__ = ["real_spherical_harmonics", "lm_index"]

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
    factors and the normalisation.
    """
    if lmax < 0:
        raise ValueError(f"lmax must be non-negative, got {lmax}")

    vectors = jnp.asarray(vectors)
    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
    norm2 = x * x + y * y + z * z

    # A zero vector has no direction; QE sets cos(theta) = 0 there. Sanitising
    # the norm keeps the gradient finite instead of NaN.
    tiny = norm2 < _EPS**2
    norm = jnp.sqrt(jnp.where(tiny, 1.0, norm2))
    cost = jnp.where(tiny, 0.0, z / norm)
    sent = jnp.sqrt(jnp.maximum(0.0, 1.0 - cost * cost))

    if lmax == 0:
        return jnp.full(vectors.shape[:-1] + (1,), 1.0 / jnp.sqrt(FPI))

    # --- Q(l, m), stored the way ylmr2 stores them: column l^2 + 1 + 2m ---
    size = (lmax + 1) ** 2
    q = [None] * size
    q[0] = jnp.ones_like(cost)
    q[1] = cost
    q[3] = -sent / jnp.sqrt(2.0)

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
        q[lm] = -jnp.sqrt((2 * l - 1) / (2.0 * l)) * sent * q[lm2]

    # --- the azimuthal angle, defined modulo pi by atan ---
    phi = jnp.where(
        jnp.abs(x) > _EPS,
        jnp.arctan2(y, jnp.where(jnp.abs(x) > _EPS, x, 1.0)),
        jnp.sign(y) * PI / 2.0,
    )

    # --- attach cos(m phi) / sin(m phi) and normalise ---
    ylm = [jnp.broadcast_to(q[0] / jnp.sqrt(FPI), cost.shape)]
    for l in range(1, lmax + 1):
        c = jnp.sqrt((2 * l + 1) / FPI)
        ylm.append(c * q[l**2])
        for m in range(1, l + 1):
            base = c * jnp.sqrt(2.0) * q[l**2 + 2 * m]
            ylm.append(base * jnp.cos(m * phi))
            ylm.append(base * jnp.sin(m * phi))

    return jnp.stack(ylm, axis=-1)
