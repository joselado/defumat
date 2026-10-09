"""OPEN.md Part XX item 2, second pass: is the 1e-5 transverse part a fixed point or an unconverged soft mode?"""
import json
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
from tests.regression.test_ultracell import _noncollinear, _modulation
from defumat.scf.driver import Calculation, run_scf
from defumat.ultracell import with_external_potential

pseudo_dir = Path("/l/ladovj1/w-a/tests/data/pseudo")
tmp = Path("/l/ladovj1/review/open-a/xx2-tmp"); tmp.mkdir(parents=True, exist_ok=True)
shape, kgrid = (2, 1, 1), (2, 2, 2)
out = {}


def transverse(rho):
    m = np.asarray(rho)[1:]
    e = m.reshape(3, -1).sum(axis=1); e = e / np.linalg.norm(e)
    perp = m - np.tensordot(e, m, axes=(0, 0))[None] * e[:, None, None, None]
    return float(np.abs(np.fft.fftn(perp, axes=(1, 2, 3))).max()
                 / np.abs(np.fft.fftn(m, axes=(1, 2, 3))).max())


def site_cant(sites):
    s = np.asarray(sites)
    e = s.sum(axis=0); e = e / np.linalg.norm(e)
    perp = s - np.outer(s @ e, e)
    return float(np.linalg.norm(perp, axis=1).max() / np.linalg.norm(s, axis=1).max())


for tag, (a1, a2), kw in [
        ("z_thr14", (0.0, 0.0), dict(conv_thr=1e-14)),
        ("z_thr14_fullacc", (0.0, 0.0), dict(conv_thr=1e-14, diago_full_acc=True)),
        ("z_thr14_beta0.7", (0.0, 0.0), dict(conv_thr=1e-14, mixing_beta=0.7)),
        ("z_thr11_nomod", (0.0, 0.0), dict(conv_thr=1e-11, modulated=False)),
        ("111_thr14", (54.735610, 45.0), dict(conv_thr=1e-14))]:
    modulated = kw.pop("modulated", True)
    sup = _noncollinear(tmp, pseudo_dir, shape, kgrid, angle1=a1, angle2=a2, tag="xx2b" + tag)
    calc = Calculation(sup.system, sup.pseudos)
    grid = tuple(calc.basis.dense.grid)
    if modulated:
        axes = [np.arange(m) / m * n for m, n in zip(grid, shape)]
        coords = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        calc = with_external_potential(calc, jnp.asarray(_modulation(shape)(coords)))
    r = run_scf(sup.system, sup.pseudos, calculation=calc, nbnd=32, max_iterations=80, **kw)
    cants = [site_cant(h["site_moments"]) for h in r.history if "site_moments" in h]
    acc = [float(h["accuracy"]) for h in r.history]
    rec = dict(converged=bool(r.converged), iterations=int(r.iterations),
               transverse=transverse(r.density), site_cant_history=cants, accuracy=acc)
    out[tag] = rec
    print(tag, rec["converged"], rec["iterations"], rec["transverse"], flush=True)
    print("   cant", np.array(cants).round(9).tolist(), flush=True)
    print("   acc ", np.array(acc).tolist(), flush=True)
    jax.clear_caches()

Path("/l/ladovj1/review/open-a/xx2b.json").write_text(json.dumps(out, indent=1))
print("XX2B-DONE", flush=True)
