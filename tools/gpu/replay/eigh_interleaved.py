"""The card's eigh error with Davidson's interleaving of parked directions, in place and sorted last."""
import os, json, time
import numpy as np, scipy.linalg as sl
import jax, jax.numpy as jnp
import defumat
import defumat.solvers.subspace as sub

OUT = os.environ.get("OUT", "eigh_captured")
cpu, dev = jax.devices("cpu")[0], jax.devices()[0]
solve = jax.jit(lambda h, s: sub.generalised_eigh(h, s, robust=False)[0])

def run(h, s, nbnd, exact):
    row = {}
    for name, device in (("card", dev), ("host", cpu)):
        with jax.default_device(device):
            v = np.asarray(solve(jnp.asarray(h), jnp.asarray(s)))[:nbnd]
        row[name] = float(np.abs(v - exact).max())
    return row

def last(h, s, parked):
    order = np.argsort(parked, kind="stable")
    return h[np.ix_(order, order)], s[np.ix_(order, order)]

# per-solve detail of the captured call, factor 4, in place and sorted last
z = np.load(OUT + ".npz")
def parked_rows(h, s, floor=50.0):
    off_h = np.abs(h - np.diag(np.diag(h))).sum(axis=1) == 0
    off_s = np.abs(s - np.diag(np.diag(s))).sum(axis=1) == 0
    return off_h & off_s & np.isclose(np.real(np.diag(s)), 1.0) & (np.real(np.diag(h)) > floor)
detail = []
for i in range(int(z["n"])):
    h, s = z[f"h{i}"], z[f"s{i}"]
    parked = parked_rows(h, s)
    if not parked.any():
        continue
    live = ~parked
    hl, slv = h[np.ix_(live, live)], s[np.ix_(live, live)]
    exact = sl.eigh(0.5 * (hl + hl.conj().T), 0.5 * (slv + slv.conj().T), eigvals_only=True)[:32]
    old = float(np.real(np.diag(h))[parked][0])
    for factor in (1000.0, 4.0):
        hp = h.copy(); hp[parked, parked] = (old - 1.0) / 1000.0 * factor + 1.0
        a = run(hp, s, 32, exact); b = run(*last(hp, s, parked), 32, exact)
        detail.append({"solve": i, "factor": factor, "m": int(h.shape[0]), "parked": int(parked.sum()),
                       "cond_s": float(np.linalg.cond(slv)), "in_place": a["card"], "sorted_last": b["card"], "host": a["host"]})
for factor in (1000.0, 4.0):
    sel = [d for d in detail if d["factor"] == factor]
    for key in ("in_place", "sorted_last", "host"):
        x = np.array([d[key] for d in sel])
        print(json.dumps({"stage": "captured-detail", "factor": factor, "which": key, "median": float(np.median(x)),
                          "p90": float(np.percentile(x, 90)), "max": float(x.max()),
                          "over_1e-13": int((x > 1e-13).sum()), "of": len(x)}), flush=True)
    worst = sorted(sel, key=lambda d: -d["in_place"])[:3]
    print(json.dumps({"stage": "captured-worst", "factor": factor, "worst": worst}), flush=True)

# synthetic, Davidson's interleaving: a refreshed block of nbnd, then three blocks each live in its first
# notcnv rows (nbnd/2, nbnd/4, nbnd/8) and parked in the rest, as a step at nvecx = 4 nbnd lays them out
rng = np.random.default_rng(1)
for m in (128, 512, 1024, 2048, 4096):
    nbnd = m // 4
    active = np.zeros(m, bool); active[:nbnd] = True
    for j, frac in enumerate((2, 4, 8), start=1):
        active[j * nbnd: j * nbnd + nbnd // frac] = True
    live = int(active.sum())
    q, _ = np.linalg.qr(rng.standard_normal((live, live)) + 1j * rng.standard_normal((live, live)))
    spec = np.concatenate([np.sort(rng.uniform(-0.5, 1.5, nbnd)), np.sort(rng.uniform(1.0, 8.0, live - nbnd))])
    v, _ = np.linalg.qr(rng.standard_normal((live, live)) + 1j * rng.standard_normal((live, live)))
    w = np.geomspace(1.0, 1.0 / 20.0, live)
    root = v @ np.diag(np.sqrt(w)) @ v.conj().T
    hl = root.conj().T @ (q @ np.diag(spec) @ q.conj().T) @ root; hl = 0.5 * (hl + hl.conj().T)
    sll = v @ np.diag(w) @ v.conj().T; sll = 0.5 * (sll + sll.conj().T)
    exact = sl.eigh(hl, sll, eigvals_only=True)[:nbnd]
    idx = np.flatnonzero(active)
    for maxdiag, factor in ((30.0, 1000.0), (30.0, 4.0), (100.0, 4.0)):
        shift = factor * maxdiag + 1.0
        h = np.zeros((m, m), complex); s = np.zeros((m, m), complex)
        h[np.ix_(idx, idx)] = hl; s[np.ix_(idx, idx)] = sll
        h[~active, ~active] = shift; s[~active, ~active] = 1.0
        t = time.perf_counter()
        a = run(h, s, nbnd, exact); b = run(*last(h, s, ~active), nbnd, exact)
        print(json.dumps({"stage": "interleaved", "m": m, "live": live, "maxdiag": maxdiag, "factor": factor,
                          "in_place": a["card"], "sorted_last": b["card"], "host": a["host"],
                          "seconds": round(time.perf_counter() - t, 1)}), flush=True)
