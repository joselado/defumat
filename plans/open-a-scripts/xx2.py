"""OPEN.md Part XX item 2: the supercell's transverse modulated magnetization, by seed."""
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
    fm = np.abs(np.fft.fftn(m, axes=(1, 2, 3)))
    fp = np.abs(np.fft.fftn(perp, axes=(1, 2, 3)))
    # The uniform (G=0) part and the modulated part, separately.
    fp0 = fp[:, 0, 0, 0].max(); fp_rest = fp.copy(); fp_rest[:, 0, 0, 0] = 0
    return dict(axis=e.tolist(), ratio_all=float(fp.max() / fm.max()),
                ratio_modulated=float(fp_rest.max() / fm.max()), g0_perp=float(fp0 / fm.max()))


for tag, (a1, a2), modulated, thr in [
        ("111_mod", (54.735610, 45.0), True, 1e-11),
        ("111_mod_thr12", (54.735610, 45.0), True, 1e-12),
        ("111exact_mod", (54.7356103172453, 45.0), True, 1e-11),
        ("z_mod", (0.0, 0.0), True, 1e-11),
        ("x_mod", (90.0, 0.0), True, 1e-11),
        ("111_nomod", (54.735610, 45.0), False, 1e-11)]:
    sup = _noncollinear(tmp, pseudo_dir, shape, kgrid, angle1=a1, angle2=a2, tag="xx2" + tag)
    calc = Calculation(sup.system, sup.pseudos)
    grid = tuple(calc.basis.dense.grid)
    if modulated:
        axes = [np.arange(m) / m * n for m, n in zip(grid, shape)]
        coords = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        calc = with_external_potential(calc, jnp.asarray(_modulation(shape)(coords)))
    r = run_scf(sup.system, sup.pseudos, calculation=calc, conv_thr=thr, nbnd=32,
                max_iterations=300)
    sites = np.asarray(r.site_moments)
    dirs = sites / np.linalg.norm(sites, axis=1)[:, None]
    cant = float(np.degrees(np.arccos(np.clip(dirs[0] @ dirs[1], -1, 1))))
    rec = dict(converged=bool(r.converged), iterations=int(r.iterations),
               sites=sites.tolist(), cant_deg=cant, **transverse(r.density))
    out[tag] = rec
    print(tag, json.dumps(rec), flush=True)
    jax.clear_caches()

Path("/l/ladovj1/review/open-a/xx2.json").write_text(json.dumps(out, indent=1))
print("XX2-DONE", flush=True)
