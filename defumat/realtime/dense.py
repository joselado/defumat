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

:func:`dense_first_order` is the reference for the potential updated in time,
at first order: the same solve with the local potential's linear response in
it, made self-consistent over every k-point.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["dense_hamiltonians", "dense_ground_states", "dense_orders",
           "dense_first_order"]


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


def dense_first_order(calculation, terms, direction, omega: float, eta: float, nocc: int,
                      potential: str = "frozen", tolerance: float = 1e-12, kcart=None):
    """``J_(1,1)`` with the local potential's linear response, every band of each sphere.

    The first order of :func:`dense_orders` with the potential of
    :mod:`defumat.realtime.selfconsistent` updated: each occupied state's
    components at the harmonics ``m = +-1`` solve

        (e_n + m w + i eta - H0) c_m = (1/2) h_1 c_0 + dv_m c_0,

    with ``dv_+1 = K drho_+1``, ``drho_+1 = sum_nk w (conj(u_0) u_+1 + u_0 conj(u_-1))``
    and ``dv_-1 = conj(dv_+1)`` (the density is real), and ``K`` the Hartree
    kernel (``'hartree'``) or the derivative of the whole potential at the
    states' own density (``'hxc'``), where the difference form of the
    propagation linearises; ``'frozen'`` is ``K = 0``. The fixed point in
    ``dv_+1`` is solved by GMRES on its real and imaginary parts to
    ``tolerance``. The density is built from the dense coefficients by FFT
    here, the potential's local term is applied as the matrix
    ``v(G - G')``, which is exact on a grid that holds the products, and the
    current is assembled as in :func:`dense_orders`, so what this shares with
    the propagation is ``H(k)``, the potential's derivative and nothing of the
    time stepping.

    ``omega`` and ``eta`` in Ry. Returns ``(J_(1,1) in Ry bohr as dense_orders
    sums it, states (nk, nocc, npwx) on the frozen spheres, the GMRES
    residual)``; a norm-conserving dataset on one grid is assumed.
    """
    import scipy.sparse.linalg as sla

    from defumat.basis.fft import g_to_r, r_to_g
    from defumat.scf.potential import hartree

    dense, smooth = calculation.basis.dense, calculation.basis.smooth
    grid = tuple(int(n) for n in dense.grid)
    if tuple(int(n) for n in smooth.grid) != grid:
        raise NotImplementedError("the dense first order assumes one FFT grid")
    mask = np.asarray(calculation.basis.planewaves.mask)
    indices = np.asarray(calculation.basis.planewaves.indices)
    smooth_miller = np.asarray(smooth.miller)
    weights = np.asarray(calculation.system.kpoints.weights)
    volume = float(calculation.system.cell.volume)
    nk, npwx = mask.shape
    size = int(np.prod(grid))
    blocks = []
    for ik in range(nk):
        h = dense_hamiltonians(calculation, terms, ik, direction, 2, kcart=kcart)
        energies, vectors = np.linalg.eigh(h[0])
        keep = np.flatnonzero(mask[ik])
        blocks.append((h, energies, vectors, keep,
                       np.mod(smooth_miller[indices[ik][keep]], grid)))

    def field(c, miller):
        box = np.zeros(grid, dtype=complex)
        box[miller[:, 0], miller[:, 1], miller[:, 2]] = c
        return np.fft.ifftn(box) * size

    def local(v, miller):
        vg = np.fft.fftn(v) / size
        d = np.mod(miller[:, None, :] - miller[None, :, :], grid)
        return vg[d[..., 0], d[..., 1], d[..., 2]]

    occupied = [[field(vectors[:, n], miller) for n in range(nocc)]
                for (_, _, vectors, _, miller) in blocks]
    rho0 = sum(weights[ik] * np.abs(u) ** 2 for ik in range(nk) for u in occupied[ik]).real
    rho0 = rho0 / volume

    if potential == "hartree":
        def real_kernel(x):
            vg, _ = hartree(r_to_g(jnp.asarray(x), dense.fft_index), dense,
                            calculation.system.cell)
            return np.real(np.asarray(g_to_r(vg, dense.fft_index, dense.grid)))
    elif potential == "hxc":
        at = jnp.asarray(rho0)[None]

        def real_kernel(x):
            return np.asarray(jax.jvp(lambda r: calculation.potential(r).v_scf, (at,),
                                      (jnp.asarray(x)[None],))[1])[0]
    elif potential == "frozen":
        def real_kernel(x):
            return np.zeros(grid)
    else:
        raise ValueError(f"unknown potential {potential!r}")

    def kernel(drho):
        return real_kernel(drho.real) + 1j * real_kernel(drho.imag)

    def components(dv):
        out = []
        for ik, (h, energies, vectors, _, miller) in enumerate(blocks):
            plus, minus = local(dv, miller), local(np.conj(dv), miller)
            per = []
            for n in range(nocc):
                c0 = vectors[:, n]
                pair = []
                for m, vmat in ((1, plus), (-1, minus)):
                    rhs = 0.5 * h[1] @ c0 + vmat @ c0
                    z = energies[n] + m * omega + 1j * eta
                    pair.append(vectors @ ((vectors.conj().T @ rhs) / (z - energies)))
                per.append(pair)
            out.append(per)
        return out

    def density(comp):
        total = np.zeros(grid, dtype=complex)
        for ik, (_, _, _, _, miller) in enumerate(blocks):
            for n in range(nocc):
                u0 = occupied[ik][n]
                total += weights[ik] * (np.conj(u0) * field(comp[ik][n][0], miller)
                                        + u0 * np.conj(field(comp[ik][n][1], miller)))
        return total / volume

    def residual(dv):
        return dv - kernel(density(components(dv)))

    zero = np.zeros(grid, dtype=complex)
    bare = kernel(density(components(zero)))
    if potential == "frozen":
        dv, resid = zero, 0.0
    else:
        def flat(x):
            return np.concatenate([x.real.ravel(), x.imag.ravel()])

        def unflat(x):
            return (x[:size] + 1j * x[size:]).reshape(grid)

        b = flat(bare)
        operator = sla.LinearOperator((2 * size, 2 * size), dtype=float,
                                      matvec=lambda x: flat(residual(unflat(x))) + b)
        x, _ = sla.gmres(operator, b, rtol=tolerance, atol=0.0, restart=60, maxiter=200)
        dv = unflat(x)
        resid = float(np.linalg.norm(flat(residual(dv))) / np.linalg.norm(b))

    comp = components(dv)
    current = 0.0
    states = np.zeros((nk, nocc, npwx), dtype=complex)
    for ik, (h, _, vectors, keep, _) in enumerate(blocks):
        states[ik][:, keep] = vectors[:, :nocc].T
        for n in range(nocc):
            c0 = vectors[:, n]
            plus, minus = comp[ik][n]
            current += weights[ik] * (np.vdot(c0, h[1] @ plus) + np.vdot(minus, h[1] @ c0)
                                      + 0.5 * np.vdot(c0, h[2] @ c0))
    return current, states, resid
