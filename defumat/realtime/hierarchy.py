"""The perturbative orders of the current in the frequency domain, by an iterative solve.

The steady state of each occupied state under the adiabatic field
``kappa(t) = lam exp(eta t) cos(w t) e`` at a frozen potential is a sum of
components ``c^(N)_M``, order ``N`` in ``lam`` and harmonic ``M`` of the drive,
each the solution of

    (e_n + M w + i N eta - H0) c^(N)_M
        = sum_{p=1..N} 1/(p! 2^p) sum_s binom(p, s) h_p c^(N-p)_(M-(2s-p)),

``h_p = d^p H(k + x e)/dx^p`` at ``x = 0``, and the current of order ``N`` at
harmonic ``M`` is the sum over the ordered pairs of components with
``p = N - n1 - n2 >= 0`` and ``M = -m1 + (2s - p) + m2`` of
``binom(p, s)/(p! 2^p) <c_(n1,m1)| d_kappa h_p |c_(n2,m2)>``, weighted by the
occupation (``HARMONICS-NEXT.md``, "The third harmonic as a spectrum"). This is
:func:`~defumat.realtime.dense.dense_orders` with the dense solve replaced, so
that it reaches the whole plane-wave sphere at production cutoffs and a
frequency costs nine solves per band and k-point (``c^(1)_{+-1}``,
``c^(2)_{+-2}``, ``c^(2)_0``, ``c^(3)_{+-3}``, ``c^(3)_{+-1}``) where the
real-time route of :mod:`defumat.realtime.orders` propagates every frequency
afresh.

**The computed bands exactly, the rest by a Krylov solve.** Every component is
split as ``c = P c + (1 - P) c`` with ``P`` the projector on the bands the
fixed-density solve resolved (the occupied ones and the conduction bands below
the top four, which an unconverged Davidson root would occupy):
``P c = sum_j |u_j> <u_j|rhs> / (z - e_j)`` is arithmetic, and ``(1 - P) c``
solves ``(H0 - z + alpha P) x = -(1 - P) rhs``, ``cch_psi_all.f90``'s operator
with ``alpha`` past every ``M w`` of the run, whose solution stays in the
complement. The two ill-conditioned pieces of the hierarchy are both in ``P``:
the secular ``1/(2 i eta)`` of ``c^(2)_0`` along the state itself, and the
denominators between two occupied states that cancel only in the sum over
bands, so neither is ever a Krylov iterate. The occupied states are the
fixed-density solve's, whose residual at ``conv_thr = 1e-10`` (about 2e-6)
moves every component by its own size and no more (measured in review against
the dense hierarchy with a random admixture of that size: 1e-7 to 9e-7, the
same with and without a Rayleigh-Ritz rotation).

**The solver is BiCGStab, right-preconditioned with the Sternheimer stack's
kinetic preconditioner** ``1/max(1, |k+G|^2 / eprec)``, ``eprec = 1.35 <T>``,
batched over the bands and the harmonics of one order in a masked
``lax.while_loop``. ``H0 - z`` is complex shifted, normal and indefinite above
the gap, and the CG of the static response does not apply. Measured in review
on two-atom silicon to a relative residual of 1e-10 at ``eta = 0.1`` eV and
``w`` = 1 and 4 eV: with twelve computed bands in ``P``, 40 to 52 matvecs mean
and 52 at most at 40 Ry over every component, 1.3x full GMRES; with the four
occupied ones alone 36 to 82 and 117; QE's GMRES(4) of ``solve_e_fpol.f90``
(built for an imaginary frequency) 1020 mean and 3344 above the gap. **A
vector that reaches the budget is refused by name**, since an unconverged
first-order component is amplified by ``1/(2 eta)`` into the second order.

**The derivatives** come from the real-time route's Chebyshev table of the
projectors (:class:`~defumat.realtime.propagate._Chunk`): ``h_p`` for ``p >= 1``
is kinetic and nonlocal only, since the local potential carries no ``k``, so it
is ``p`` nested ``jvp`` of ``|k+G+x e|^2 x + vkb D vkb^dagger x`` with no
transform; ``<c1| d_kappa h_p |c2>`` is the ``kappa`` Jacobian of the same form,
which gives the current along all three axes at once. The table agrees with the
radial transform the dense reference uses to 5e-15 in ``h_0`` and 6e-10 in
``h_4`` at 12 Ry (measured in review).

**Memory**, per k-point: the nine components and ``c0`` (``10 nocc npwx``),
BiCGStab's six vectors for the largest order's ``4 nocc`` right-hand sides, the
computed bands (``nb npwx``), complex; the k-points are independent at a frozen
potential and are walked one at a time.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from defumat.eager import compiled_function

__all__ = ["hierarchy_orders", "HierarchyError"]


#: The smallest right-hand side, relative to the largest of its batch, that a
#: row is converged against on its own scale.
FLOOR = 1e-12


class HierarchyError(RuntimeError):
    """A component's solve did not converge within its budget."""


def _bicgstab(apply, b, precondition, tolerance, max_iterations, start=None):
    """``x`` with ``apply(x) = b`` per row, right-preconditioned BiCGStab, masked by row.

    ``b`` and ``precondition`` are ``(nv, npwx)``; ``start`` is the first iterate,
    zero when ``None``, and the shadow residual is the first residual. Returns
    ``(x, iterations, residual)``, the residual the true one, ``|b - A x| / |b|``
    per row, from one more application after the loop.
    """
    real = jnp.finfo(b.dtype).dtype

    def dot(a, c):
        return jnp.einsum("vg,vg->v", jnp.conj(a), c)

    def norm(a):
        return jnp.sqrt(jnp.real(dot(a, a)))

    # a row whose right-hand side is round-off against the batch's largest (a
    # component a symmetry forces to zero at one k-point) is solved to the
    # batch's scale rather than its own, which it could never reach
    bnorm = jnp.maximum(norm(b), FLOOR * jnp.max(norm(b)))
    target = tolerance * bnorm
    zero = jnp.zeros_like(b)
    x0 = zero if start is None else start.astype(b.dtype)
    r0 = b if start is None else b - apply(x0)
    one = jnp.ones(b.shape[0], dtype=b.dtype)
    state = (x0, r0, zero, zero, one, one, one, jnp.zeros(b.shape[0], dtype=jnp.int32),
             norm(r0) <= target, jnp.array(0))

    def safe(numerator, denominator):
        ok = jnp.abs(denominator) > 0.0
        return jnp.where(ok, numerator / jnp.where(ok, denominator, 1.0), 0.0)

    def body(state):
        x, r, p, v, rho, alpha, omega, count, done, step = state
        rho_new = dot(r0, r)  # the shadow residual is the first residual
        beta = jnp.where(step == 0, 0.0, safe(rho_new, rho) * safe(alpha, omega))
        p_new = r + beta[:, None] * (p - omega[:, None] * v)
        y = precondition * p_new
        v_new = apply(y)
        alpha_new = safe(rho_new, dot(r0, v_new))
        s = r - alpha_new[:, None] * v_new
        early = norm(s) <= target
        zz = precondition * s
        t = apply(zz)
        omega_new = jnp.where(early, 0.0, safe(dot(t, s), dot(t, t)))
        x_new = x + alpha_new[:, None] * y + omega_new[:, None] * zz
        r_new = s - omega_new[:, None] * t
        now = done | early | (norm(r_new) <= target)
        keep = done[:, None]
        return (jnp.where(keep, x, x_new), jnp.where(keep, r, r_new),
                jnp.where(keep, p, p_new), jnp.where(keep, v, v_new),
                jnp.where(done, rho, rho_new), jnp.where(done, alpha, alpha_new),
                jnp.where(done, omega, omega_new),
                count + jnp.where(done, 0, 1).astype(jnp.int32), now, step + 1)

    def going(state):
        return (state[9] < max_iterations) & ~jnp.all(state[8])

    x, _, _, _, _, _, _, count, _, _ = jax.lax.while_loop(going, body, state)
    residual = jnp.where(bnorm > 0.0, norm(b - apply(x)) / jnp.where(bnorm > 0.0, bnorm, 1.0),
                         0.0).astype(real)
    return x, count, residual


def _derivative(f, order: int):
    """``d^order f/ds^order`` at ``s = 0`` for ``f`` of one real scalar, by nested ``jvp``."""
    def nth(g, n):
        if n == 0:
            return g

        def lowered(s):
            return jax.jvp(nth(g, n - 1), (s,), (jnp.ones_like(s),))[1]
        return lowered
    return nth(f, order)


def _orders_at(chunk, basis, energies, weights, precondition, omega, eta, alpha, direction,
               starts, *, nocc: int, order: int, tolerance: float, max_iterations: int):
    """At one k-point: ``{(N, M): sum_n w <..|d_kappa h_p|..>}`` (3,), iterations, residuals.

    ``basis`` is ``(nb, npwx)``, the computed bands with the occupied first,
    ``energies`` their ``(nb,)`` energies in Ry, ``weights`` the ``(nocc,)``
    occupations times the k-weight, ``omega`` and ``eta`` in Ry. ``starts`` is
    one ``((N+1) nocc, npwx)`` first iterate per order, the complement's
    solution at the previous frequency of a sweep, and the new ones are
    returned beside the currents.
    """
    real = energies.dtype
    zero3 = jnp.zeros(3, dtype=real)
    mask = chunk.mask[0]
    ham0 = chunk.hamiltonian(zero3)
    coefficients = chunk.template.coefficients.astype(basis.dtype)
    e = direction.astype(real)

    def h0(x):
        return jnp.where(mask, ham0.apply(x, 0), 0.0)

    def kinetic_nonlocal(kappa, x):
        kinetic, projectors = chunk.moved(kappa)
        vkb = projectors.vkb[0]
        becp = jnp.einsum("gi,ng->ni", jnp.conj(vkb), x)
        out = kinetic[0][None, :] * x + jnp.einsum("gi,ij,nj->ng", vkb, coefficients, becp)
        return jnp.where(mask, out, 0.0)

    def h_p(p, x):
        return _derivative(lambda s: kinetic_nonlocal(s * e, x), p)(jnp.zeros((), real))

    def form(kappa, c1, c2):
        kinetic, projectors = chunk.moved(kappa)
        vkb = projectors.vkb[0]
        b1 = jnp.einsum("gi,ng->ni", jnp.conj(vkb), c1)
        b2 = jnp.einsum("gi,ng->ni", jnp.conj(vkb), c2)
        kin = jnp.einsum("n,ng,g,ng->", weights.astype(basis.dtype), jnp.conj(c1),
                         kinetic[0].astype(basis.dtype), c2)
        nl = jnp.einsum("n,ni,ij,nj->", weights.astype(basis.dtype), jnp.conj(b1),
                        coefficients, b2)
        return kin + nl

    def current(p, c1, c2):
        def along(kappa):
            return _derivative(lambda s: form(kappa + s * e, c1, c2), p)(jnp.zeros((), real))
        return jax.jacfwd(along, holomorphic=False)(zero3)

    occupied = basis[:nocc]
    e_occ = energies[:nocc]

    def solve(rhs, z, band, start):
        coefficient = jnp.einsum("jg,vg->vj", jnp.conj(basis), rhs)
        inside = jnp.einsum("vj,jg->vg", coefficient / (z[:, None] - energies[None, :]), basis)
        outside = -(rhs - jnp.einsum("vj,jg->vg", coefficient, basis))

        def apply(x):
            projected = jnp.einsum("vj,jg->vg", jnp.einsum("jg,vg->vj", jnp.conj(basis), x),
                                   basis)
            return h0(x) - z[:, None] * x + alpha * projected

        x, count, residual = _bicgstab(apply, outside, precondition[band], tolerance,
                                       max_iterations, start)
        return inside + x, x, count, residual

    components = {(0, 0): occupied}
    counts, residuals, outsides = [], [], []
    applied = {}

    def apply_h(p, key):
        if (p, key) not in applied:
            applied[(p, key)] = h_p(p, components[key])
        return applied[(p, key)]

    for n in range(1, order + 1):
        harmonics = list(range(-n, n + 1, 2))
        rhs = []
        for m in harmonics:
            total = jnp.zeros_like(occupied)
            for p in range(1, n + 1):
                for s in range(p + 1):
                    key = (n - p, m - (2 * s - p))
                    if key in components:
                        total = total + (math.comb(p, s) / (math.factorial(p) * 2**p)
                                         * apply_h(p, key))
            rhs.append(total)
        z = jnp.concatenate([e_occ + m * omega + 1j * n * eta for m in harmonics])
        band = jnp.tile(jnp.arange(nocc), len(harmonics))
        solution, outside, count, residual = solve(jnp.concatenate(rhs),
                                                   z.astype(basis.dtype), band, starts[n - 1])
        for i, m in enumerate(harmonics):
            components[(n, m)] = solution[i * nocc:(i + 1) * nocc]
        counts.append(count)
        residuals.append(residual)
        outsides.append(outside)

    totals = {}
    keys = list(components)
    for k1 in keys:
        for k2 in keys:
            for p in range(0, order - k1[0] - k2[0] + 1):
                value = current(p, components[k1], components[k2])
                for s in range(p + 1):
                    big = (k1[0] + k2[0] + p, -k1[1] + (2 * s - p) + k2[1])
                    weight = math.comb(p, s) / (math.factorial(p) * 2**p)
                    totals[big] = totals.get(big, 0.0) + weight * value
    return totals, jnp.concatenate(counts), jnp.concatenate(residuals), tuple(outsides)


def _preconditioner(chunk, occupied):
    """``1 / max(1, |k+G|^2 / eprec_n)``, ``eprec_n = 1.35 <c_n|T|c_n>``, ``(nocc, npwx)``."""
    kinetic = jnp.asarray(chunk.moved(jnp.zeros(3, dtype=chunk.k0.dtype))[0][0])
    expectation = jnp.real(jnp.einsum("ng,g,ng->n", jnp.conj(occupied), kinetic, occupied))
    eprec = 1.35 * expectation
    return 1.0 / jnp.maximum(1.0, kinetic[None, :] / eprec[:, None])


def hierarchy_orders(calculation, basis, energies, weights, v_scf, *, omegas, eta: float,
                     direction, order: int = 3, tolerance: float = 1e-10,
                     max_iterations: int = 500, kcart=None, symmetrise=None,
                     warm: bool = True) -> dict:
    """``J_(N,M)`` at every frequency, Hartree atomic units, the whole set of k-points.

    Args:
        calculation: the calculation the states belong to, on the k-set walked.
        basis: ``(nk, nb, npwx)``, the computed bands of every k-point, the
            occupied ones first; the projector ``P`` of the module docstring.
        energies: ``(nk, nb)`` in Ry.
        weights: ``(nk, nocc)``, occupation times k-weight, which decides how
            many of the leading bands are carried.
        v_scf: the frozen potential.
        omegas: the drive's frequencies in Hartree; ``eta`` the broadening per
            photon in Hartree.
        direction: the field's unit vector, cartesian.
        order: the highest order, at most 3.
        tolerance: BiCGStab's relative residual on each right-hand side.
        symmetrise: cartesian rotations to average the currents over, a polar
            vector each, when the k-set is the wedge of their group.
        warm: start each frequency's solves from the previous frequency's
            solutions at the same k-point; the answer is the same to the
            tolerance either way.

    Returns ``{"components": {(N, M): (nw, 3) complex}, "iterations": (nw,)
    largest BiCGStab iteration count, "residual": (nw,) largest final residual,
    "volume": ...}``; the currents are ``-(1/(2 Omega)) sum``, the real-time
    route's ``J_(N,M)`` (:meth:`~defumat.realtime.orders.OrdersResult.component`).
    """
    from defumat.realtime.propagate import _Chunk, _prepare

    basis = np.asarray(basis)
    energies = np.asarray(energies, dtype=float)
    weights = np.asarray(weights, dtype=float)
    omegas = np.atleast_1d(np.asarray(omegas, dtype=float))
    nk, nb, _ = basis.shape
    nocc = weights.shape[1]
    if order > 3:
        raise NotImplementedError("the hierarchy is written to third order")
    if nb <= nocc:
        raise ValueError("the computed bands must include more than the occupied ones")
    setup = _prepare(calculation, basis[:, :nocc], weights, v_scf, 0.0, None, "taylor4",
                     1, kcart, bounds=False)
    real = setup.real
    unit = np.asarray(direction, dtype=float)
    unit = unit / np.linalg.norm(unit)
    omegas_ry, eta_ry = 2.0 * omegas, 2.0 * float(eta)
    # past every M w of the run, so that no eigenvalue of the shifted operator on
    # P (e_j - Re z + alpha) comes near zero and amplifies what leaks into P
    alpha = 2.0 * (float(energies.max()) - float(energies[:, :nocc].min())) \
        + 3.0 * float(omegas_ry.max()) + 1.0

    def at(chunk, u, e, w, precondition, omega, eta_, alpha_, direction_, starts):
        return _orders_at(chunk, u, e, w, precondition, omega, eta_, alpha_, direction_,
                          starts, nocc=nocc, order=order, tolerance=float(tolerance),
                          max_iterations=int(max_iterations))

    totals = {}
    iterations = np.zeros(len(omegas), dtype=int)
    residual = np.zeros(len(omegas))
    run = None
    for ik in range(nk):
        chunk = setup.first if ik == 0 else _Chunk.build(calculation, np.asarray([ik]),
                                                          setup.terms, setup.table,
                                                          setup.kcart)
        u = jnp.asarray(basis[ik])
        precondition = _preconditioner(chunk, u[:nocc]).astype(real)
        fixed = (chunk, u, jnp.asarray(energies[ik], dtype=real),
                 jnp.asarray(weights[ik], dtype=real), precondition)
        # each frequency starts from the previous one's solution at this k-point,
        # which a sweep finer than the broadening makes a good guess
        starts = tuple(jnp.zeros(((n + 1) * nocc, u.shape[-1]), dtype=u.dtype)
                       for n in range(1, order + 1))
        for iw, omega in enumerate(omegas_ry):
            arguments = fixed + (jnp.asarray(omega, dtype=real), jnp.asarray(eta_ry, dtype=real),
                                 jnp.asarray(alpha, dtype=real), jnp.asarray(unit, dtype=real),
                                 starts)
            if run is None:
                run = compiled_function(at, *arguments)
            out, count, res, starts = run(*arguments)
            if not warm:
                starts = tuple(jnp.zeros_like(x) for x in starts)
            count, res = np.asarray(count), np.asarray(res)
            if res.max() > 10.0 * tolerance:
                raise HierarchyError(
                    f"a component did not converge at k-point {ik}, w = {omega / 2.0:.5f} Ha: "
                    f"relative residual {res.max():.2e} after {int(count.max())} BiCGStab "
                    f"iterations against {tolerance:.1e} (max_iterations = {max_iterations}). "
                    "An unconverged first order is amplified by 1/(2 eta) into the second; "
                    "raise max_iterations or the number of computed bands")
            iterations[iw] = max(iterations[iw], int(count.max()))
            residual[iw] = max(residual[iw], float(res.max()))
            for key, value in out.items():
                totals.setdefault(key, np.zeros((len(omegas), 3), dtype=complex))
                totals[key][iw] += np.asarray(value)
    volume = setup.volume
    components = {key: -value / (2.0 * volume) for key, value in totals.items()}
    if symmetrise is not None:
        rotations = np.asarray(symmetrise, dtype=float)
        components = {key: np.einsum("sab,wb->wa", rotations, value) / len(rotations)
                      for key, value in components.items()}
    return {"components": components, "iterations": iterations, "residual": residual,
            "volume": volume, "alpha": alpha, "nocc": nocc, "computed_bands": nb}
