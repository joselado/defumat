"""The subspace solve's eigenvalue error on Davidson's own matrices, by where the idle directions are parked.

    CAPTURE=1 python3 tools/gpu/replay/eigh_captured.py   # on the card, at a commit with PARK_FACTOR (b418095 to cc21ad4)
    python3 tools/gpu/replay/eigh_captured.py             # then the error tables, card and host


Stage 1 (CAPTURE=1): run si16's SCF on the card at PARK_FACTOR=1000 with a host callback in the Cholesky
route recording every (hc, sc) pair it is handed; save them. Stage 2: for every recorded pair, the exact
lowest roots are those of the live block alone (the parked block is decoupled), taken by SciPy at the live
block's own norm; the error of generalised_eigh on the padded matrix, on the card and on the host, with the
parked value at 1000 and at 4 times the largest diagonal element. Stage 3: synthetic pairs of the same
structure from m = 128 to 4096.
"""
import os, sys, json, time
import numpy as np, scipy.linalg as sl
import jax, jax.numpy as jnp
import defumat
import defumat.solvers.subspace as sub
import defumat.solvers.davidson as dav

OUT = os.environ.get("OUT", "eigh_captured")
cpu = jax.devices("cpu")[0]
dev = jax.devices()[0]
solve = jax.jit(lambda h, s: sub.generalised_eigh(h, s, robust=False)[0])

def capture():
    from pathlib import Path
    from defumat.io.pwin import read_pw_input
    from defumat.pseudo import read_upf
    from defumat.scf.driver import Calculation, run_scf
    from defumat.system import build_system
    dav.PARK_FACTOR = 1000.0   # exists at b418095 to cc21ad4 only
    pairs = []
    route = sub._cholesky_route
    def spy(h, s):
        jax.debug.callback(lambda a, b: pairs.append((np.asarray(a), np.asarray(b))), h, s)
        return route(h, s)
    sub._cholesky_route = spy
    cell = "benchmarks/si16-1k-ecut30.in"
    system = build_system(read_pw_input(Path(cell)))
    pseudos = tuple(read_upf(Path("tests/data/pseudo") / s.pseudo_file) for s in system.structure.species)
    calc = Calculation(system, pseudos)
    import defumat.scf.potential as P
    r = run_scf(system, pseudos, calculation=calc, conv_thr=1e-10)
    steps = [h.get("davidson_iterations") for h in r.history]
    pot = P.v_of_rho(calc.starting_density(), calc.basis.dense, system.cell)
    maxdiag = float(jnp.max(jnp.abs(calc.hamiltonian(pot.v_scf)[0].diagonal(0))))
    arrays = {f"h{i}": p[0] for i, p in enumerate(pairs)}
    arrays.update({f"s{i}": p[1] for i, p in enumerate(pairs)})
    np.savez_compressed(OUT + ".npz", n=len(pairs), maxdiag=maxdiag, **arrays)
    print(json.dumps({"captured": len(pairs), "steps": steps, "maxdiag": maxdiag}), flush=True)

def parked_rows(h, s, floor=50.0):
    """Rows with no coupling in either matrix, unit overlap and a diagonal far above the live block."""
    off_h = np.abs(h - np.diag(np.diag(h))).sum(axis=1) == 0
    off_s = np.abs(s - np.diag(np.diag(s))).sum(axis=1) == 0
    return off_h & off_s & np.isclose(np.real(np.diag(s)), 1.0) & (np.real(np.diag(h)) > floor)

def errors(h, s, nbnd, parked, shift_to):
    """Error of the lowest nbnd roots on card and host, the parked rows moved to shift_to."""
    live = ~parked
    hl, slv = h[np.ix_(live, live)], s[np.ix_(live, live)]
    exact = sl.eigh(0.5 * (hl + hl.conj().T), 0.5 * (slv + slv.conj().T), eigvals_only=True)[:nbnd]
    hp = h.copy(); hp[parked, parked] = shift_to
    row = {"m": int(h.shape[0]), "parked": int(parked.sum()), "cond_s": float(np.linalg.cond(slv)),
           "norm": float(max(shift_to, np.abs(exact).max()))}
    for name, device in (("card", dev), ("host", cpu)):
        with jax.default_device(device):
            v = np.asarray(solve(jnp.asarray(hp), jnp.asarray(s)))[:nbnd]
        row[name] = float(np.abs(v - exact).max())
    return row

def stage2():
    z = np.load(OUT + ".npz")
    maxdiag = float(z["maxdiag"])
    H = [z[f"h{i}"] for i in range(int(z["n"]))]; S = [z[f"s{i}"] for i in range(int(z["n"]))]
    nbnd = 32
    rows = []
    for i in range(len(H)):
        parked = parked_rows(H[i], S[i])
        if not parked.any():
            continue
        old = float(np.real(np.diag(H[i]))[parked][0])
        new = (old - 1.0) / 1000.0 * 4.0 + 1.0
        for label, to in (("1000", old), ("4", new)):
            r = errors(H[i], S[i], nbnd, parked, to); r.update(solve_index=i, factor=label); rows.append(r)
    for f in ("1000", "4"):
        sel = [r for r in rows if r["factor"] == f]
        card = np.array([r["card"] for r in sel]); host = np.array([r["host"] for r in sel])
        print(json.dumps({"stage": "captured", "factor": f, "solves_with_parked": len(sel),
                          "card_median": float(np.median(card)), "card_max": float(card.max()),
                          "host_median": float(np.median(host)), "host_max": float(host.max()),
                          "norm": sel[0]["norm"], "eps_norm": 2.22e-16 * sel[0]["norm"],
                          "cond_s_max": max(r["cond_s"] for r in sel)}), flush=True)

def stage3():
    rng = np.random.default_rng(0)
    for m in (128, 512, 1024, 2048, 4096):
        nbnd = m // 4
        live = nbnd + nbnd // 2       # a step with half a block of corrections, the rest parked
        q, _ = np.linalg.qr(rng.standard_normal((live, live)) + 1j * rng.standard_normal((live, live)))
        spec = np.concatenate([np.sort(rng.uniform(-0.5, 1.5, nbnd)), np.sort(rng.uniform(1.0, 8.0, live - nbnd))])
        v, _ = np.linalg.qr(rng.standard_normal((live, live)) + 1j * rng.standard_normal((live, live)))
        w = np.geomspace(1.0, 1.0 / 20.0, live)    # cond(S) 20, as at step 3 of the stalled call
        root = v @ np.diag(np.sqrt(w)) @ v.conj().T
        hl = root.conj().T @ (q @ np.diag(spec) @ q.conj().T) @ root
        sll = v @ np.diag(w) @ v.conj().T
        for maxdiag in (30.0, 100.0):
            for factor in (1000.0, 4.0):
                shift = factor * maxdiag + 1.0
                h = np.zeros((m, m), complex); s = np.zeros((m, m), complex)
                h[:live, :live] = 0.5 * (hl + hl.conj().T); s[:live, :live] = 0.5 * (sll + sll.conj().T)
                h[live:, live:] = shift * np.eye(m - live); s[live:, live:] = np.eye(m - live)
                t = time.perf_counter()
                parked = np.arange(m) >= live
                r = errors(h, s, nbnd, parked, shift)
                r.update(stage="synthetic", maxdiag=maxdiag, factor=factor, seconds=round(time.perf_counter() - t, 1))
                print(json.dumps(r), flush=True)

if os.environ.get("CAPTURE") and not os.path.exists(OUT + ".npz"):
    capture()
else:
    stage2(); stage3()
