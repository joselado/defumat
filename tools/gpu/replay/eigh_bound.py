"""Captured si16 solves (from eigh_captured.py): are the parked rows trailing, and what does the card's error become with the
parked value at one above the Gershgorin bound of the reduced live block, or with no parked rows at all."""
import json, os
import numpy as np, scipy.linalg as sl
import jax, jax.numpy as jnp
import defumat
import defumat.solvers.subspace as sub
cpu, dev = jax.devices("cpu")[0], jax.devices()[0]
solve = jax.jit(lambda h, s: sub.generalised_eigh(h, s, robust=False)[0])
z = np.load(os.environ.get("OUT", "eigh_captured") + ".npz")
def parked_rows(h, s, floor=50.0):
    off_h = np.abs(h - np.diag(np.diag(h))).sum(axis=1) == 0
    off_s = np.abs(s - np.diag(np.diag(s))).sum(axis=1) == 0
    return off_h & off_s & np.isclose(np.real(np.diag(s)), 1.0) & (np.real(np.diag(h)) > floor)
rows, trailing = [], 0
for i in range(int(z["n"])):
    h, s = z[f"h{i}"], z[f"s{i}"]
    parked = parked_rows(h, s)
    if not parked.any():
        continue
    m = len(parked); k = int((~parked).sum())
    trailing += int(parked[k:].all() and not parked[:k].any())
    hl, slv = h[:k, :k], s[:k, :k]
    hl = 0.5 * (hl + hl.conj().T); slv = 0.5 * (slv + slv.conj().T)
    exact = sl.eigh(hl, slv, eigvals_only=True)[:32]
    L = np.linalg.cholesky(slv); Li = np.linalg.inv(L)
    reduced = Li @ hl @ Li.conj().T
    bound = float(np.abs(reduced).sum(axis=1).max())
    old = float(np.real(np.diag(h))[parked][0])
    out = {"solve": i, "m": m, "parked": int(parked.sum()), "cond_s": float(np.linalg.cond(slv)),
           "live_norm": float(np.abs(sl.eigvalsh(reduced)).max()), "bound": bound}
    for label, value in (("factor4", (old - 1.0) / 1000.0 * 4.0 + 1.0), ("gershgorin", bound + 1.0)):
        hp = h.copy(); hp[parked, parked] = value
        with jax.default_device(dev):
            v = np.asarray(solve(jnp.asarray(hp), jnp.asarray(s)))[:32]
        out[label] = float(np.abs(v - exact).max())
    with jax.default_device(dev):
        v = np.asarray(solve(jnp.asarray(hl), jnp.asarray(slv)))[:32]
    out["live_only"] = float(np.abs(v - exact).max())
    rows.append(out)
print(json.dumps({"solves_with_parked": len(rows), "parked_trailing": trailing}))
for key in ("factor4", "gershgorin", "live_only"):
    x = np.array([r[key] for r in rows])
    print(json.dumps({"parked_at": key, "median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
                      "max": float(x.max()), "over_1e-13": int((x > 1e-13).sum()), "over_3e-14": int((x > 3e-14).sum())}))
g = np.array([r["bound"] for r in rows]); ln = np.array([r["live_norm"] for r in rows])
print(json.dumps({"live_norm_median": float(np.median(ln)), "live_norm_max": float(ln.max()),
                  "gershgorin_median": float(np.median(g)), "gershgorin_max": float(g.max())}))
for r in sorted(rows, key=lambda r: -r["factor4"])[:4]:
    print(json.dumps(r))
