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

from defumat.basis.fft import g_to_r, r_to_g
from defumat.eager import compiled_function

__all__ = ["hierarchy_orders", "hierarchy_linear_self_consistent", "HierarchyError"]


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
               starts, induced=None, *, nocc: int, order: int, tolerance: float,
               max_iterations: int):
    """At one k-point: ``{(N, M): sum_n w <..|d_kappa h_p|..>}`` (3,), iterations, residuals.

    ``basis`` is ``(nb, npwx)``, the computed bands with the occupied first,
    ``energies`` their ``(nb,)`` energies in Ry, ``weights`` the ``(nocc,)``
    occupations times the k-weight, ``omega`` and ``eta`` in Ry. ``starts`` is
    one ``((N+1) nocc, npwx)`` first iterate per order, the complement's
    solution at the previous frequency of a sweep, and the new ones are
    returned beside the currents. ``induced``, at first order only, is the
    local potential ``dv_+`` the field induces, complex on the FFT box, which
    drives ``c_+`` as ``dv_+ c0`` and ``c_-`` as ``conj(dv_+) c0``; with it the
    weighted cross density ``sum_n w (conj(u0) u_+ + u0 conj(u_-))`` of those
    components is returned last, the first-order density at ``e^{-iwt}`` times
    the cell volume.
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

    if induced is not None:
        if order != 1:
            raise NotImplementedError("an induced potential is written at first order only")
        fft_index, grid = chunk.template.fft_index[0], chunk.template.grid

        def to_r(c):
            return g_to_r(jnp.where(mask, c, 0.0), fft_index, grid)

        def local(field, c):
            return jnp.where(mask, r_to_g(field[None] * to_r(c), fft_index), 0.0)

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
            if induced is not None and n == 1:
                total = total + local(induced if m == 1 else jnp.conj(induced), occupied)
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
    if induced is None:
        return totals, jnp.concatenate(counts), jnp.concatenate(residuals), tuple(outsides)
    u0, up, um = to_r(occupied), to_r(components[(1, 1)]), to_r(components[(1, -1)])
    cross = jnp.einsum("n,nxyz->xyz", weights.astype(basis.dtype),
                       jnp.conj(u0) * up + u0 * jnp.conj(um))
    return (totals, jnp.concatenate(counts), jnp.concatenate(residuals), tuple(outsides),
            cross)


def _preconditioner(chunk, occupied):
    """``1 / max(1, |k+G|^2 / eprec_n)``, ``eprec_n = 1.35 <c_n|T|c_n>``, ``(nocc, npwx)``."""
    kinetic = jnp.asarray(chunk.moved(jnp.zeros(3, dtype=chunk.k0.dtype))[0][0])
    expectation = jnp.real(jnp.einsum("ng,g,ng->n", jnp.conj(occupied), kinetic, occupied))
    eprec = 1.35 * expectation
    return 1.0 / jnp.maximum(1.0, kinetic[None, :] / eprec[:, None])


def hierarchy_orders(calculation, basis, energies, weights, v_scf, *, omegas, eta: float,
                     direction, order: int = 3, tolerance: float = 1e-10,
                     max_iterations: int = 500, kcart=None, symmetrise=None,
                     warm: bool = True, k_batch="default") -> dict:
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
        k_batch: how many k-points are solved together, one ``vmap`` over
            them; the calculation's dial by default, one on a CPU. On a card a
            single k-point's transforms are too small to fill it.

    Returns ``{"components": {(N, M): (nw, 3) complex}, "iterations": (nw,)
    largest BiCGStab iteration count, "residual": (nw,) largest final residual,
    "volume": ...}``; the currents are ``-(1/(2 Omega)) sum``, the real-time
    route's ``J_(N,M)`` (:meth:`~defumat.realtime.orders.OrdersResult.component`).
    """
    from defumat.batching import k_chunks
    from defumat.realtime.propagate import _batch, _Chunk, _prepare

    basis = np.asarray(basis)
    energies = np.asarray(energies, dtype=float)
    weights = np.asarray(weights, dtype=float)
    omegas = np.atleast_1d(np.asarray(omegas, dtype=float))
    nk, nb, npwx = basis.shape
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
    batch = _batch(calculation, k_batch)
    batch = nk if batch is None else max(1, min(int(batch), nk))

    def at(chunk, u, e, w, precondition, omega, eta_, alpha_, direction_, starts):
        return _orders_at(chunk, u, e, w, precondition, omega, eta_, alpha_, direction_,
                          starts, nocc=nocc, order=order, tolerance=float(tolerance),
                          max_iterations=int(max_iterations))

    if batch > 1:
        # one vmap over the k-points of a chunk: each row is a one-point chunk,
        # so the program is the one-point one with a batch axis, and a padded
        # row carries zero weight
        at = jax.vmap(at, in_axes=(0, 0, 0, 0, 0, None, None, None, None, 0))
        preconditioner = jax.vmap(_preconditioner)
    else:
        preconditioner = _preconditioner

    def one(ik):
        if ik == 0:
            return setup.first
        return _Chunk.build(calculation, np.asarray([ik]), setup.terms, setup.table, setup.kcart)

    totals = {}
    iterations = np.zeros(len(omegas), dtype=int)
    residual = np.zeros(len(omegas))
    run = None
    for rows, live in k_chunks(nk, batch):
        if batch > 1:
            chunk = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[one(ik) for ik in rows])
            w = np.where(np.arange(len(rows))[:, None] < live, weights[rows], 0.0)
            u = jnp.asarray(basis[rows])
            e = jnp.asarray(energies[rows], dtype=real)
            occupied = u[:, :nocc]
            shape = (len(rows),)
        else:
            chunk = one(int(rows[0]))
            w = weights[rows[0]]
            u = jnp.asarray(basis[rows[0]])
            e = jnp.asarray(energies[rows[0]], dtype=real)
            occupied = u[:nocc]
            shape = ()
        precondition = preconditioner(chunk, occupied).astype(real)
        fixed = (chunk, u, e, jnp.asarray(w, dtype=real), precondition)
        # each frequency starts from the previous one's solution at these
        # k-points, which a sweep finer than the broadening makes a good guess
        starts = tuple(jnp.zeros(shape + ((n + 1) * nocc, npwx), dtype=u.dtype)
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
            if batch > 1:
                count, res = count[:live], res[:live]
            if res.max() > 10.0 * tolerance:
                raise HierarchyError(
                    f"a component did not converge at k-points {list(rows[:live])}, "
                    f"w = {omega / 2.0:.5f} Ha: relative residual {res.max():.2e} after "
                    f"{int(count.max())} BiCGStab iterations against {tolerance:.1e} "
                    f"(max_iterations = {max_iterations}). An unconverged first order is "
                    "amplified by 1/(2 eta) into the second; raise max_iterations or the "
                    "number of computed bands")
            iterations[iw] = max(iterations[iw], int(count.max()))
            residual[iw] = max(residual[iw], float(res.max()))
            for key, value in out.items():
                value = np.asarray(value)
                if batch > 1:
                    value = value.sum(axis=0)  # the padded rows carry zero weight
                totals.setdefault(key, np.zeros((len(omegas), 3), dtype=complex))
                totals[key][iw] += value
    volume = setup.volume
    components = {key: -value / (2.0 * volume) for key, value in totals.items()}
    if symmetrise is not None:
        rotations = np.asarray(symmetrise, dtype=float)
        components = {key: np.einsum("sab,wb->wa", rotations, value) / len(rotations)
                      for key, value in components.items()}
    return {"components": components, "iterations": iterations, "residual": residual,
            "volume": volume, "alpha": alpha, "nocc": nocc, "computed_bands": nb}


def hierarchy_linear_self_consistent(calculation, basis, energies, weights, v_scf, *, omegas,
                                     eta: float, direction, potential: str = "hxc",
                                     tolerance: float = 1e-10, outer_tolerance: float = 1e-8,
                                     max_iterations: int = 500, outer_iterations: int = 200,
                                     kcart=None, symmetrise=None, density_symmetry=None,
                                     k_batch="default") -> dict:
    """``J_(1,1)`` with the induced potential at ``w``, self-consistent, at every frequency.

    The first order of :func:`hierarchy_orders` with ``dv_+ = K drho_+`` in the
    right-hand side, ``K`` the Hartree kernel (``potential = 'hartree'``) or the
    derivative of the whole potential at the states' own density (``'hxc'``),
    which is the propagation's difference form of
    :mod:`defumat.realtime.selfconsistent` linearised. The fixed point in
    ``dv_+`` is a GMRES solve on its real and imaginary parts, each product one
    pass over every k-point with the inner solves from a zero start at the
    fixed ``tolerance`` (a scheduled or warm inner solve would change the
    operator between Krylov steps). Measured in review on two-atom silicon:
    10 to 28 products at 1 and 4 eV and on an interband transition, where the
    mixing of the static response (``ph.x``'s, history 4) took 200 and diverged
    with the Hartree kernel. ``dv_+(G = 0)`` is set to zero: the Hartree
    potential has none, and the uniform part of the exchange-correlation one
    is a global phase. The density of a wedge is completed with
    ``density_symmetry``, the field's little group, real and imaginary parts as
    two channels (the symmetrisation keeps the real part of what it is given).

    Returns ``{"components": {(1, 1): (nw, 3)}, "frozen": {(1, 1): (nw, 3)},
    "outer": (nw,) products, "outer_residual": (nw,), ...}``, the frozen
    current being the same run's at ``dv = 0``. A frequency whose outer solve
    does not reach ``outer_tolerance`` is refused by name.
    """
    import scipy.sparse.linalg as sla

    from defumat.basis.interpolate import to_dense
    from defumat.batching import k_chunks
    from defumat.realtime.propagate import _batch, _Chunk, _prepare
    from defumat.realtime.selfconsistent import POTENTIALS
    from defumat.scf.driver import _symmetrize
    from defumat.scf.potential import hartree
    from defumat.system.symmetry import symmetry_maps

    if potential not in POTENTIALS or potential == "frozen":
        raise ValueError(f"potential must be 'hartree' or 'hxc', not {potential!r}")
    if potential == "hxc" and calculation.functional.is_meta:
        raise NotImplementedError("the exchange-correlation kernel of a meta-GGA is not here")
    dense, smooth = calculation.basis.dense, calculation.basis.smooth
    grid = tuple(int(n) for n in dense.grid)
    if tuple(int(n) for n in smooth.grid) != grid:
        raise NotImplementedError(
            "the self-consistent hierarchy on two FFT grids is not implemented: the "
            "induced potential would go from the dense grid to the smooth one at every "
            "product, and nothing checks that path yet")
    if symmetrise is not None and density_symmetry is None:
        raise ValueError("a reduced k-set needs the field's little group for its density")
    basis = np.asarray(basis)
    energies = np.asarray(energies, dtype=float)
    weights = np.asarray(weights, dtype=float)
    omegas = np.atleast_1d(np.asarray(omegas, dtype=float))
    nk, nb, npwx = basis.shape
    nocc = weights.shape[1]
    setup = _prepare(calculation, basis[:, :nocc], weights, v_scf, 0.0, None, "taylor4",
                     1, kcart, bounds=False)
    real = setup.real
    volume = setup.volume
    unit = np.asarray(direction, dtype=float)
    unit = unit / np.linalg.norm(unit)
    omegas_ry, eta_ry = 2.0 * omegas, 2.0 * float(eta)
    alpha = 2.0 * (float(energies.max()) - float(energies[:, :nocc].min())) \
        + 3.0 * float(omegas_ry.max()) + 1.0
    batch = _batch(calculation, k_batch)
    batch = nk if batch is None else max(1, min(int(batch), nk))
    maps = None if density_symmetry is None else symmetry_maps(dense, density_symmetry)

    # the states' own density, where the kernel is taken, completed as the
    # propagation completes it
    flat_weights = jnp.asarray(weights, dtype=real)
    rho0 = to_dense(calculation.smooth_density(jnp.asarray(basis[:, :nocc])[None],
                                               flat_weights[None]), smooth, dense)
    if maps is not None:
        rho0 = _symmetrize(rho0, dense.fft_index, dense.grid, maps)

    if potential == "hxc":
        def kernel_real(x):
            return jax.jvp(lambda r: calculation.potential(r).v_scf, (rho0,),
                           (x[None].astype(rho0.dtype),))[1][0]
    else:
        def kernel_real(x):
            vg, _ = hartree(r_to_g(x, dense.fft_index), dense, calculation.system.cell)
            return jnp.real(g_to_r(vg, dense.fft_index, dense.grid))

    @jax.jit
    def kernel(drho):
        parts = jnp.stack([jnp.real(drho), jnp.imag(drho)])[:, None]
        if maps is not None:
            parts = jax.vmap(lambda f: _symmetrize(f, dense.fft_index, dense.grid, maps))(parts)
        dv = kernel_real(parts[0, 0]) + 1j * kernel_real(parts[1, 0])
        return dv - jnp.mean(dv)   # no uniform part: a global phase, and Hartree has none

    def at(chunk, u, e, w, precondition, omega, eta_, alpha_, direction_, starts, induced):
        return _orders_at(chunk, u, e, w, precondition, omega, eta_, alpha_, direction_,
                          starts, induced, nocc=nocc, order=1, tolerance=float(tolerance),
                          max_iterations=int(max_iterations))

    if batch > 1:
        at = jax.vmap(at, in_axes=(0, 0, 0, 0, 0, None, None, None, None, 0, None))
        preconditioner = jax.vmap(_preconditioner)
    else:
        preconditioner = _preconditioner

    def one(ik):
        if ik == 0:
            return setup.first
        return _Chunk.build(calculation, np.asarray([ik]), setup.terms, setup.table, setup.kcart)

    held = []
    for rows, live in k_chunks(nk, batch):
        if batch > 1:
            chunk = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[one(ik) for ik in rows])
            w = np.where(np.arange(len(rows))[:, None] < live, weights[rows], 0.0)
            u = jnp.asarray(basis[rows])
            e = jnp.asarray(energies[rows], dtype=real)
            occupied, shape = u[:, :nocc], (len(rows),)
        else:
            chunk = one(int(rows[0]))
            w = weights[rows[0]]
            u = jnp.asarray(basis[rows[0]])
            e = jnp.asarray(energies[rows[0]], dtype=real)
            occupied, shape = u[:nocc], ()
        precondition = preconditioner(chunk, occupied).astype(real)
        starts = (jnp.zeros(shape + (2 * nocc, npwx), dtype=u.dtype),)
        held.append(((chunk, u, e, jnp.asarray(w, dtype=real), precondition), starts, rows, live))

    run = None
    size = int(np.prod(grid))

    def sweep(omega, induced):
        nonlocal run
        totals, cross = {}, jnp.zeros(grid, dtype=basis.dtype)
        worst, count_max = 0.0, 0
        for fixed, starts, rows, live in held:
            arguments = fixed + (jnp.asarray(omega, dtype=real), jnp.asarray(eta_ry, dtype=real),
                                 jnp.asarray(alpha, dtype=real), jnp.asarray(unit, dtype=real),
                                 starts, induced)
            if run is None:
                run = compiled_function(at, *arguments)
            out, count, res, _, density = run(*arguments)
            count, res = np.asarray(count), np.asarray(res)
            if batch > 1:
                count, res, density = count[:live], res[:live], density.sum(axis=0)
                out = {key: value.sum(axis=0) for key, value in out.items()}
            worst, count_max = max(worst, float(res.max())), max(count_max, int(count.max()))
            cross = cross + density
            for key, value in out.items():
                totals[key] = totals.get(key, 0.0) + np.asarray(value)
        if worst > 10.0 * tolerance:
            raise HierarchyError(
                f"an inner solve did not converge at w = {omega / 2.0:.5f} Ha: relative "
                f"residual {worst:.2e} against {tolerance:.1e}")
        return totals, cross / volume, count_max

    def flat(x):
        x = np.asarray(x)
        return np.concatenate([x.real.ravel(), x.imag.ravel()])

    def unflat(x):
        return jnp.asarray((x[:size] + 1j * x[size:]).reshape(grid), dtype=basis.dtype)

    def finish(totals):
        components = {key: -np.asarray(value) / (2.0 * volume) for key, value in totals.items()}
        if symmetrise is not None:
            rotations = np.asarray(symmetrise, dtype=float)
            components = {key: np.einsum("sab,b->a", rotations, value) / len(rotations)
                          for key, value in components.items()}
        return components

    sc = np.zeros((len(omegas), 3), dtype=complex)
    frozen = np.zeros((len(omegas), 3), dtype=complex)
    products = np.zeros(len(omegas), dtype=int)
    outer_residual = np.zeros(len(omegas))
    inner = np.zeros(len(omegas), dtype=int)
    previous = None
    zero = jnp.zeros(grid, dtype=basis.dtype)
    for iw, omega in enumerate(omegas_ry):
        bare_totals, bare_density, inner[iw] = sweep(omega, zero)
        frozen[iw] = finish(bare_totals)[(1, 1)]
        bare = flat(kernel(bare_density))
        calls = [0]

        def linear(x):
            calls[0] += 1
            _, density, _ = sweep(omega, unflat(x))
            return x - (flat(kernel(density)) - bare)

        operator = sla.LinearOperator((2 * size, 2 * size), matvec=linear, dtype=float)
        x, _ = sla.gmres(operator, bare, x0=previous, rtol=outer_tolerance, atol=0.0,
                         restart=60, maxiter=max(1, outer_iterations // 60 + 1))
        totals, density, _ = sweep(omega, unflat(x))
        residual = float(np.linalg.norm(x - flat(kernel(density)))
                         / max(np.linalg.norm(bare), 1e-300))
        if residual > 10.0 * outer_tolerance:
            raise HierarchyError(
                f"the induced potential did not converge at w = {omega / 2.0:.5f} Ha: "
                f"relative residual {residual:.2e} after {calls[0]} products against "
                f"{outer_tolerance:.1e}")
        sc[iw] = finish(totals)[(1, 1)]
        products[iw], outer_residual[iw] = calls[0], residual
        previous = x
    return {"components": {(1, 1): sc}, "frozen": {(1, 1): frozen}, "outer": products,
            "outer_residual": outer_residual, "iterations": inner, "volume": volume,
            "alpha": alpha, "nocc": nocc, "computed_bands": nb}
