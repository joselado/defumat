"""The two-cell +-y turning field of test_a_turning_field_turns_the_magnetization, both starts.

OPEN.md Part VI item 3: under the default the test's linear-response check failed, so
measure which state each start reaches and which is lower in E + field.
"""
import json
import time
from pathlib import Path

import jax
import numpy as np

from defumat.ultracell import run_ultracell
from tests.regression.test_ultracell import _noncollinear

root = Path("/l/ladovj1/w-g")
pseudo_dir = root / "tests/data/pseudo"
out_dir = Path("/l/ladovj1/review/decided-g")
tmp = out_dir / "turn2-tmp"
tmp.mkdir(parents=True, exist_ok=True)
shape, kgrid = (2, 1, 1), (2, 2, 2)
folded = tuple(n * m for n, m in zip(shape, kgrid))
calc = _noncollinear(tmp, pseudo_dir, (1, 1, 1), folded, angle1=90.0, angle2=0.0, tag="turn")
scf = calc.get_scf(conv_thr=1e-11, nbnd=16)
print("reference", scf.converged, scf.total_energy, flush=True)


def turning(x):
    sign = np.cos(2 * np.pi * x[..., 0] / shape[0] - np.pi / 2)
    zero = np.zeros_like(sign)
    return 0.01 * np.stack([zero, sign, zero], axis=-1)


base = run_ultracell(calc.system, calc.pseudos, scf, shape, kgrid, nbnd=16, conv_thr=1e-9,
                     states_conv_thr=1e-8, mixing_beta=0.3)
unperturbed = np.asarray(base.density)[0]
out = {}
for arm, kw in (("default", {}), ("tiled", {"seed_from_field": False})):
    t = time.time()
    r = run_ultracell(calc.system, calc.pseudos, scf, shape, kgrid, nbnd=16, conv_thr=1e-9,
                      states_conv_thr=1e-8, mixing_beta=0.3, magnetic_field=turning,
                      max_iterations=120, **kw)
    m = np.asarray(r.cell_moments())
    moved = float(np.abs(np.asarray(r.density)[0] - unperturbed).max() / unperturbed.max())
    rec = dict(iterations=int(r.iterations), converged=bool(r.converged),
               total=float(r.total_energy), field=float(r.field_energy),
               free=float(r.total_energy) + float(r.field_energy), moments=m.tolist(),
               moved=moved, transverse=float(abs(m[0][1]) / max(abs(m[0][0]), 1e-30)),
               origin=r.seed_origin, seconds=time.time() - t)
    out[arm] = rec
    print(arm, json.dumps(rec), flush=True)
    jax.clear_caches()
print("dE+field default - tiled", out["default"]["free"] - out["tiled"]["free"], flush=True)
(out_dir / "turn2.json").write_text(json.dumps(out, indent=1))
print("TURN2-DONE", flush=True)
