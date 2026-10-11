"""A noncollinear potential from a collinear one, with no local axis: the parity form.

A potential-only meta-GGA (Tran-Blaha, Becke-Johnson) has no energy, so the
``grad m`` construction of the gradient correction -- write the energy in the
invariants of ``grad n`` and ``grad m_k`` and differentiate it
(:meth:`~defumat.xc.functional.Functional.noncollinear_gradient_exchange_energy`)
-- has nothing to differentiate. What it can be given instead is the same
idea one step earlier, on the potential itself.

The collinear up-channel potential is a function of four *signed* scalars,

    V(x_m, x_a, x_L, x_T),   rho = (n + x_m)/2,  sigma = (S + 2 x_a)/4,
                             lap = (lap n + x_L)/2,  tau = (tau_0 + x_T)/2,

with ``v_up = V(x)`` and ``v_dw = V(-x)``. Its sixteen parity components

    V_S = 2^-4 sum_eps (prod_{i in S} eps_i) V(eps x),   S a subset of {m, a, L, T},

are odd in the arguments in ``S`` and even in the rest. A noncollinear density
has four spin-space vectors in place of the four scalars -- ``m``,
``a_k = grad n . grad m_k``, ``L_k = lap m_k`` and ``T_k = tau_k`` -- and the
potential is assembled from the parity components at their moduli
``x_i = |X_i|`` with the cosines ``c_ij = e_i . e_j`` of their directions:

    v_0 = V_0 + sum_{i<j} c_ij V_ij + (1/3)(c_ma c_LT + c_mL c_aT + c_mT c_aL) V_maLT,
    B   = sum_i e_i V_i + sum_{ijk} (1/3)(c_ij e_k + c_ik e_j + c_jk e_i) V_ijk.

Where every ``X_i`` lies along one axis ``u`` the cosines are signs and this is
``v_0 = (v_up + v_dw)/2``, ``B = (v_up - v_dw)/2 u`` exactly, nodes included and
with no axis chosen. At a node, ``V_S`` is ``prod_{i in S} x_i`` times a smooth
function of the squares, so every term is a product of dot products of the
``X_i`` and is smooth through it -- where the rotation onto ``m-hat`` makes the
field depend on the direction the node is approached from. Off the collinear
limit the pairings of the cubic and quartic terms are a choice (the symmetric
mean), so it is a functional of its own there, as the ``grad m`` exchange is.
The plan and its review are ``plans/noncollinear-gga-grad-m-2.md``.
"""

from __future__ import annotations

import itertools

import jax.numpy as jnp

from defumat.xc.functional import safe_modulus

__all__ = ["parity_form"]

#: The triples of the four odd arguments, in the order the cubic terms are summed.
_TRIPLES = tuple(itertools.combinations(range(4), 3))


def parity_form(channel, vectors):
    """``(v_0, B)`` from a collinear channel potential and four spin-space vectors.

    Args:
        channel: ``x -> V(x)``, elementwise, with ``x`` the ``(4, ...)`` signed
            odd arguments in the order ``(m, a, L, T)`` -- the up channel's
            potential; the down channel's is ``V(-x)``.
        vectors: ``(4, 3, ...)``, the four spin-space vectors in that order.

    Returns ``v_0`` of shape ``(...)`` and ``B`` of shape ``(3, ...)``. Sixteen
    evaluations of ``channel``.
    """
    moduli = jnp.stack([safe_modulus(vectors[i]) for i in range(4)])
    unit = jnp.where(
        (moduli > 0.0)[:, None],
        vectors / jnp.where(moduli > 0.0, moduli, 1.0)[:, None],
        0.0,
    )
    cos = jnp.einsum("ik...,jk...->ij...", unit, unit)

    signs = jnp.asarray(list(itertools.product((1.0, -1.0), repeat=4)),
                        dtype=moduli.dtype)
    values = [channel(sign.reshape((4,) + (1,) * (moduli.ndim - 1)) * moduli)
              for sign in signs]

    def component(subset):
        total = 0.0
        for sign, value in zip(signs, values):
            weight = 1.0
            for i in subset:
                weight = weight * sign[i]
            total = total + weight * value
        return total / 16.0

    v0 = component(())
    for i, j in itertools.combinations(range(4), 2):
        v0 = v0 + cos[i, j] * component((i, j))
    quartic = (cos[0, 1] * cos[2, 3] + cos[0, 2] * cos[1, 3] + cos[0, 3] * cos[1, 2])
    v0 = v0 + quartic / 3.0 * component((0, 1, 2, 3))

    field = sum(unit[i] * component((i,))[None] for i in range(4))
    for i, j, k in _TRIPLES:
        pair = (cos[i, j][None] * unit[k] + cos[i, k][None] * unit[j]
                + cos[j, k][None] * unit[i])
        field = field + pair / 3.0 * component((i, j, k))[None]
    return v0, field
