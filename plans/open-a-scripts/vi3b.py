"""OPEN.md Part VI item 3, second pass: the weak turning field on the Kramers-closed basis."""
import json, time
from pathlib import Path
import numpy as np
import jax
from tests.regression.test_ultracell import _noncollinear
from defumat.ultracell import run_ultracell

pseudo_dir = Path("/l/ladovj1/w-a/tests/data/pseudo")
tmp = Path("/l/ladovj1/review/open-a/vi3-tmp"); tmp.mkdir(parents=True, exist_ok=True)
n, kgrid = 4, (1, 2, 2)
shape = (n, 1, 1)
folded = tuple(a * b for a, b in zip(shape, kgrid))
out = {}
calc = _noncollinear(tmp, pseudo_dir, (1, 1, 1), folded, angle1=54.735610, angle2=45.0, tag="vi3111")
scf = calc.get_scf(conv_thr=1e-11, nbnd=8, max_iterations=300)


def helix(x):
    phase = 2 * np.pi * x[..., 0] / n
    return np.stack([np.cos(phase), np.sin(phase), np.zeros_like(phase)], axis=-1)


def field(amp):
    return lambda x: amp * helix(x)


def arm(name, iterations, beta, amp, **kw):
    t = time.time()
    r = run_ultracell(calc.system, calc.pseudos, scf, shape, kgrid, nbnd=8, conv_thr=1e-10,
                      states_conv_thr=1e-8, mixing_beta=beta, max_iterations=iterations,
                      magnetic_field=field(amp), **kw)
    m = np.asarray(r.cell_moments())
    hist = np.asarray(r.history, dtype=float)
    rec = dict(iterations=int(r.iterations), converged=bool(r.converged), energy=float(r.total_energy),
               moments=m.tolist(), net=m.mean(axis=0).tolist(), seconds=time.time() - t,
               history=hist.tolist(), energy_history=list(map(float, r.energy_history)))
    out[name] = rec
    print(name, rec["iterations"], rec["converged"], rec["energy"], np.round(rec["net"], 5).tolist(),
          m.round(4).tolist(), flush=True)
    every = max(1, len(hist) // 40)
    print("   dr2 total/charge/mag every", every, hist[::every].round(14).tolist(), flush=True)
    print("   energy", np.asarray(rec["energy_history"])[::every].round(9).tolist(), flush=True)
    jax.clear_caches()


arm("k_field0.01_b0.7", 400, 0.7, 0.01)
arm("k_field0.002_b0.7_seeded", 400, 0.7, 0.002, seed_magnetization=helix)
arm("k_field0.002_b0.7_long", 1500, 0.7, 0.002)
Path("/l/ladovj1/review/open-a/vi3b.json").write_text(json.dumps(out, indent=1))
print("VI3B-DONE", flush=True)
