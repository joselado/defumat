"""The check behind HARMONICS-NEXT.md: perturbative orders from a real-time run.

A one-dimensional plane-wave model on a frozen sphere,

    H(k) = (k+G)^2/2 + V_loc(G-G') + beta(k+G) beta(k+G'),

with a local potential that has no inversion centre and a separable term whose form
factor depends on k+G, as a pseudopotential projector does. The field enters in the
velocity gauge, H(t) = H(k + lam a(t)), with a(t) = exp(eta t) cos(w t) on [-T, 0].

Route 1, real time: a fourth-order Taylor step at the midpoint, the current as the
derivative of <psi|H(k+kappa)|psi> with respect to kappa at frozen psi, and the
perturbative orders J^(n)(t) = (1/n!) d^n J/d lam^n at lam = 0 from nested jax.jvp
through the whole propagation. exp(-n eta t) J^(n)(t) is periodic and is projected on
exp(-i m w t) over the last period.

Route 2, frequency domain: the steady-state hierarchy at the complex frequencies
m w + i n eta, dense linear solves, no time stepping.

The two share H(k) and nothing else. Usage:

    JAX_PLATFORMS=cpu python3 tools/realtime/toy_orders.py [steps_per_period] [eta_T]

Measured 2026-10-06 (relative difference of the two routes, components (1,1), (2,2),
(2,0), (3,3), (3,1)):

    400  steps/period, eta T = 14:  2.0e-5  2.9e-5  5.1e-5  1.7e-5  2.6e-5
    1600 steps/period, eta T = 14:  1.2e-6  8.4e-6  3.2e-5  9.1e-6  5.2e-7
    1600 steps/period, eta T = 20:  1.3e-6  1.4e-6  1.2e-6  1.6e-6  1.7e-6

so the residual at eta T = 14 is the start transient, of relative size exp(-eta T)
times a resonance factor, and it goes away with a longer run.
"""
import sys
import jax, math, itertools
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

NPW = 9
G = jnp.arange(-(NPW // 2), NPW // 2 + 1).astype(jnp.float64)
# local potential without inversion symmetry (so that chi2 != 0)
def vloc():
    d = (G[:, None] - G[None, :])
    v1 = 0.35 * (jnp.abs(d) == 1) + 0.12 * (jnp.abs(d) == 2)
    v2 = 0.10j * jnp.sign(d) * (jnp.abs(d) == 1) + 0.07j * jnp.sign(d) * (jnp.abs(d) == 2)
    return (v1 + v2).astype(jnp.complex128)
VLOC = vloc()

def beta(q):                       # a projector's form factor at q = k + G
    return q * jnp.exp(-0.35 * q**2)

def H(k):
    q = k + G
    b = beta(q)
    return jnp.diag(0.5 * q**2).astype(jnp.complex128) + VLOC + 0.8 * jnp.outer(b, b)

def dH(k, order):
    f = H
    for _ in range(order):
        f = jax.jacfwd(f)
    return f(k)

NK = 12
KS = (jnp.arange(NK) + 0.37) / NK - 0.5
ETA, W = 0.05, 0.30
NPER = int(sys.argv[1]) if len(sys.argv) > 1 else 400      # steps per period
ETA_T = float(sys.argv[2]) if len(sys.argv) > 2 else 14.0
DT = (2 * math.pi / W) / NPER
NSTEP = int(round(ETA_T / ETA / DT / NPER)) * NPER
T0 = -NSTEP * DT

def a_of_t(t):
    return jnp.exp(ETA * t) * jnp.cos(W * t)

def taylor4(Hm, psi, dt):
    out, term = psi, psi
    for n in range(1, 5):
        term = (-1j * dt / n) * (Hm @ term)
        out = out + term
    return out

def current_k(k, psi0, lam):
    """J_k(t) on the last period, shape (NPER,)."""
    def step(psi, i):
        t = T0 + i * DT
        psi = taylor4(H(k + lam * a_of_t(t + 0.5 * DT)), psi, DT)
        kap = lam * a_of_t(t + DT)
        # current = d/dkappa <psi|H(k+kappa)|psi> at frozen psi
        j = jax.grad(lambda x: jnp.real(jnp.vdot(psi, H(k + x) @ psi)))(kap)
        return psi, j
    _, js = jax.lax.scan(step, psi0, jnp.arange(NSTEP))
    return js[-NPER:]

def ground(k):
    e, v = jnp.linalg.eigh(H(k))
    return e[0], v[:, 0]

E0, U0 = jax.vmap(ground)(KS)
print("gap min", float(jnp.min(jax.vmap(lambda k: jnp.linalg.eigvalsh(H(k)))(KS)[:, 1] - E0)), " dt", DT, " nstep", NSTEP, " |H|max", float(jnp.max(jnp.linalg.eigvalsh(H(0.5)))))

def J_total(lam):
    return jnp.sum(jax.vmap(lambda k, u: current_k(k, u, lam))(KS, U0), axis=0) / NK

def orders(f, x0):
    one = jnp.ones_like(x0)
    f1 = lambda x: jax.jvp(f, (x,), (one,))[1]
    f2 = lambda x: jax.jvp(f1, (x,), (one,))[1]
    f3 = lambda x: jax.jvp(f2, (x,), (one,))[1]
    return f(x0), f1(x0), f2(x0) / 2, f3(x0) / 6

import time
t = time.time()
J0, J1, J2, J3 = jax.jit(lambda: orders(J_total, jnp.float64(0.0)))()
print("real-time done in %.1f s" % (time.time() - t))
tgrid = (jnp.arange(NPER) - NPER + 1) * DT       # last period, ends at t = 0
def project(Jn, n, m):
    return complex(jnp.mean(Jn * jnp.exp(-n * ETA * tgrid) * jnp.exp(1j * m * W * tgrid)))

# ---- route 2: frequency-domain hierarchy -------------------------------------
def freq_components(k, e0, u0, nmax=3):
    h = [np.asarray(dH(k, p)) for p in range(0, nmax + 2)]
    H0 = h[0]
    c = {(0, 0): np.asarray(u0)}
    for n in range(1, nmax + 1):
        for m in range(-n, n + 1, 2):
            rhs = np.zeros(NPW, complex)
            for p in range(1, n + 1):
                for s in range(p + 1):
                    key = (n - p, m - (2 * s - p))
                    if key in c:
                        rhs += math.comb(p, s) / (math.factorial(p) * 2**p) * (h[p] @ c[key])
            c[(n, m)] = np.linalg.solve((e0 + m * W + 1j * n * ETA) * np.eye(NPW) - H0, rhs)
    Jc = {}
    for N in range(0, nmax + 1):
        for (n1, m1), c1 in c.items():
            for (n2, m2), c2 in c.items():
                p = N - n1 - n2
                if p < 0:
                    continue
                for s in range(p + 1):
                    M = -m1 + (2 * s - p) + m2
                    val = math.comb(p, s) / (math.factorial(p) * 2**p) * np.vdot(c1, h[p + 1] @ c2)
                    Jc[(N, M)] = Jc.get((N, M), 0) + val
    return Jc

tot = {}
for k, e0, u0 in zip(np.asarray(KS), np.asarray(E0), np.asarray(U0)):
    for key, v in freq_components(float(k), float(e0), u0).items():
        tot[key] = tot.get(key, 0) + v / NK

print("%-8s %-34s %-34s %s" % ("(n, m)", "real time + jvp", "frequency-domain hierarchy", "rel. diff"))
for (n, Jn) in ((1, J1), (2, J2), (3, J3)):
    for m in range(n, -1, -2):
        rt = project(Jn, n, m)
        fd = tot[(n, m)]
        print("%-8s %-34s %-34s %.2e" % ((n, m), "%+.9e%+.9ej" % (rt.real, rt.imag), "%+.9e%+.9ej" % (fd.real, fd.imag), abs(rt - fd) / abs(fd)))
print("J0 max", float(jnp.max(jnp.abs(J0))))
