"""The perturbative orders of the current by dense linear algebra: the reference.

At order ``n`` in the field amplitude and harmonic ``m`` of the drive, the
steady-state component of each occupied state under the adiabatic field
``kappa(t) = lam exp(eta t) cos(w t) e`` solves

    (eps_n + m w + i n eta - H0) c^(n)_m
        = sum_{p>=1} 1/(p! 2^p) sum_s binom(p, s) h_p c^(n-p)_(m-(2s-p)),

with ``h_p = d^p H(k + x e)/dx^p`` at ``x = 0``, and the current along ``e`` is
the sum over the components of total order ``N`` of ``c^dagger h_(p+1) c`` with
the same weights and the selection ``M = -m1 + (2s - p) + m2``
(``HARMONICS-NEXT.md``, "The third harmonic as a spectrum"). The response of
order ``N`` is then ``exp(N eta t) sum_M J_(N,M) exp(-i M w t)``, which is what
the real-time route's nested ``jvp`` projects out
(:mod:`defumat.realtime.orders`).

This is the reference for that route and nothing else. It builds ``H(k + x e)``
as an explicit matrix on the frozen sphere of each k-point
(:meth:`~defumat.hamiltonian.operator.Hamiltonian.matrix`) and its derivatives in
``x`` by nested forward differentiation, and solves the hierarchy densely with
every band of the sphere, so it costs ``npw^3`` per k-point and is meant for a
cutoff chosen small for it. The two routes share ``H(k)`` and nothing else: no
time step, no propagator, no Fourier projection.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["dense_hamiltonians", "dense_ground_states", "dense_orders"]


def dense_hamiltonians(calculation, terms, ik: int, direction, order: int,
                       kcart=None) -> list[np.ndarray]:
    """``[h_0, ..., h_order]`` at k-point ``ik``, each ``(npw, npw)`` on the sphere.

    ``h_p = d^p H(k + x e)/dx^p`` at ``x = 0`` on the frozen sphere of ``ik``,
    restricted to its real plane waves (the padding of ``npwx`` dropped).
    ``terms`` is :meth:`~defumat.scf.driver.Calculation.local_terms` of the frozen
    potential; ``direction`` is cartesian, 1/bohr per unit of ``x``.
    """
    row = calculation.at_rows([ik])
    if kcart is None:
        kcart = calculation.system.kpoints.cartesian(calculation.system.cell)
    k0 = jnp.asarray(np.asarray(kcart)[[ik]])
    e = jnp.asarray(direction, dtype=k0.dtype)
    keep = np.flatnonzero(np.asarray(calculation.basis.planewaves.mask[ik]))

    def matrix(x):
        moved = row.at_kcart(k0 + x * e[None, :])
        return moved.hamiltonian_from(terms)[0].matrix(0)

    out = []
    f = matrix
    for p in range(order + 1):
        value = np.asarray(f(jnp.zeros((), dtype=k0.dtype)))[np.ix_(keep, keep)]
        out.append(0.5 * (value + value.conj().T))
        f = _derivative(f)
    return out


def _derivative(f):
    return lambda x: jax.jvp(f, (x,), (jnp.ones_like(x),))[1]


def dense_ground_states(h0: np.ndarray, nocc: int):
    """``(eigenvalues, vectors)`` of the lowest ``nocc`` states of ``h0``, vectors as rows."""
    values, vectors = np.linalg.eigh(h0)
    return values[:nocc], vectors[:, :nocc].T


def dense_orders(hamiltonians, energies, states, weights, omega: float, eta: float,
                 nmax: int = 3) -> dict:
    """``{(N, M): J_(N,M)}`` for one k-point, ``sum_n w_n <..|h_(p+1)|..>``, Ry bohr.

    ``hamiltonians`` from :func:`dense_hamiltonians` to order ``nmax + 1``,
    ``energies`` and ``states`` the occupied eigenpairs of ``h_0`` (``states``
    as rows on the same sphere), ``weights`` their occupations times the
    k-weight, ``omega`` and ``eta`` in Ry, the energy unit of ``H``. Each band is
    its own hierarchy, as each state evolves on its own at a frozen potential,
    and the bands are summed with their weights at the end, which is where the
    denominators between two occupied states cancel.
    """
    h = [np.asarray(m) for m in hamiltonians]
    npw = h[0].shape[0]
    total: dict = {}
    for energy, state, weight in zip(energies, states, weights):
        c = {(0, 0): np.asarray(state)}
        for n in range(1, nmax + 1):
            for m in range(-n, n + 1, 2):
                rhs = np.zeros(npw, dtype=complex)
                for p in range(1, n + 1):
                    for s in range(p + 1):
                        key = (n - p, m - (2 * s - p))
                        if key in c:
                            rhs += (math.comb(p, s) / (math.factorial(p) * 2**p)
                                    * (h[p] @ c[key]))
                c[(n, m)] = np.linalg.solve(
                    (energy + m * omega + 1j * n * eta) * np.eye(npw) - h[0], rhs)
        for big_n in range(0, nmax + 1):
            for (n1, m1), c1 in c.items():
                for (n2, m2), c2 in c.items():
                    p = big_n - n1 - n2
                    if p < 0:
                        continue
                    for s in range(p + 1):
                        big_m = -m1 + (2 * s - p) + m2
                        value = (math.comb(p, s) / (math.factorial(p) * 2**p)
                                 * np.vdot(c1, h[p + 1] @ c2))
                        total[(big_n, big_m)] = total.get((big_n, big_m), 0.0) + weight * value
    return total
