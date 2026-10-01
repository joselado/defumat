"""The subspace solve's eigenvalue error against SciPy, by the overlap's condition number.

    python3 tools/gpu/replay/eigh_noise.py        # on each platform; JAX_PLATFORMS=cpu for a CPU

A 128 x 128 Hermitian pair with a spectrum from -5 to 40 and an overlap of condition number
1 to 1e6, solved by ``generalised_eigh`` (the Cholesky route the Davidson step takes) and by
``scipy.linalg.eigh``; prints the largest error in the lowest 32 eigenvalues and the spread under
1e-16 perturbations of the input. **The pairs are random, which is its limit**: on them the card and a
CPU were within a factor of two, and the Davidson pair, with its parked 3e4 diagonal, is what differed
(``GPU-SPEED-NEXT.md`` item 4 is to run this on a matrix with that structure).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, scipy.linalg as sl
import jax, jax.numpy as jnp
from defumat.solvers.subspace import generalised_eigh
n, nb = 128, 32
rng = np.random.default_rng(3)
def pair(cond):
    q, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    w = np.logspace(0, -np.log10(cond), n)
    s = (q * w) @ q.conj().T
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h0 = 0.5 * (a + a.conj().T)
    ev = np.linspace(-5.0, 40.0, n)
    u, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    h = (u * ev) @ u.conj().T
    return 0.5 * (h + h.conj().T), 0.5 * (s + s.conj().T)
f = jax.jit(lambda h, s: generalised_eigh(h, s, robust=False)[0][:nb].real)
dev = jax.default_backend()
print("backend", dev)
for cond in (1e0, 1e1, 2e1, 5e1, 1e2, 1e3, 1e4, 1e6):
    h, s = pair(cond)
    ref = sl.eigh(h.astype(np.complex128), s.astype(np.complex128), eigvals_only=True, driver="gvd")[:nb]
    got = np.asarray(f(jnp.asarray(h), jnp.asarray(s)))
    # repeat with a relative 1e-16 perturbation of the inputs: how much does the result move
    moved = [np.abs(np.asarray(f(jnp.asarray(h * (1 + 1e-16 * rng.standard_normal(h.shape))), jnp.asarray(s))) - got).max() for _ in range(5)]
    print("cond(S) %.0e   |e_jax - e_scipy| max %.2e   rerun spread (1e-16 input noise) max %.2e" % (cond, np.abs(got - ref).max(), max(moved)))
