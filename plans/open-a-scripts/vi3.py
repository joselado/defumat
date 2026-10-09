"""OPEN.md Part VI item 3: is the flat-manifold traversal still there on the Kramers-closed basis?"""
import json, sys, time
from pathlib import Path
import numpy as np
import jax
from tests.regression.test_ultracell import _noncollinear, _pitch, _cone
from defumat.ultracell import run_ultracell

pseudo_dir = Path("/l/ladovj1/w-a/tests/data/pseudo")
tmp = Path("/l/ladovj1/review/open-a/vi3-tmp"); tmp.mkdir(parents=True, exist_ok=True)
n, kgrid = 4, (1, 2, 2)
shape = (n, 1, 1)
folded = tuple(a * b for a, b in zip(shape, kgrid))
out = {}


def helix(x):
    phase = 2 * np.pi * x[..., 0] / n
    return np.stack([np.cos(phase), np.sin(phase), np.zeros_like(phase)], axis=-1)


refs = {}
for tag, (a1, a2) in {"z": (0.0, 0.0), "111": (54.735610, 45.0)}.items():
    calc = _noncollinear(tmp, pseudo_dir, (1, 1, 1), folded, angle1=a1, angle2=a2, tag="vi3" + tag)
    scf = calc.get_scf(conv_thr=1e-11, nbnd=8, max_iterations=300)
    refs[tag] = (calc, scf)
    print(tag, "reference", scf.converged, np.asarray(scf.magnetization_vector), flush=True)


def arm(name, ref, nbnd, pairs, **kw):
    calc, scf = refs[ref]
    t = time.time()
    r = run_ultracell(calc.system, calc.pseudos, scf, shape, kgrid, nbnd=nbnd,
                      conv_thr=1e-10, states_conv_thr=1e-8, mixing_beta=kw.pop("beta", 0.3),
                      max_iterations=400, kramers_pairs=pairs, **kw)
    m = np.asarray(r.cell_moments())
    rec = dict(iterations=int(r.iterations), converged=bool(r.converged),
               energy=float(r.total_energy), moments=m.tolist(),
               lengths=np.linalg.norm(m, axis=1).tolist(),
               net=float(np.linalg.norm(m.mean(axis=0))), seconds=time.time() - t,
               history_last=str(r.history[-1])[:300])
    out[name] = rec
    print(name, rec["iterations"], rec["converged"], rec["energy"], rec["net"], m.round(5).tolist(), flush=True)
    jax.clear_caches()
    return m


def aligned(a, b):
    """Residual after the best global rotation of b onto a (Kabsch)."""
    u, s, vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(u @ vt))
    R = (u @ np.diag([1, 1, d]) @ vt).T
    return float(np.abs(a - b @ R.T).max())


mA = arm("old_ref_z_nbnd16", "z", 16, False, seed_magnetization=helix)
mB = arm("old_ref_111_nbnd16", "111", 16, False, seed_magnetization=helix)
mC = arm("kramers_ref_z_nbnd8", "z", 8, True, seed_magnetization=helix)
mD = arm("kramers_ref_111_nbnd8", "111", 8, True, seed_magnetization=helix)
out["aligned_old"] = aligned(mA, mB)
out["aligned_kramers"] = aligned(mC, mD)
out["aligned_kramers_z_vs_old_z"] = aligned(mC, mA)
print("aligned", out["aligned_old"], out["aligned_kramers"], flush=True)

# The weak field of the original entry, as the 2026-09-15 note reads it.
for amp in (0.002,):
    def turning(x, amp=amp):
        phase = 2 * np.pi * x[..., 0] / n
        return amp * np.stack([np.cos(phase), np.sin(phase), np.zeros_like(phase)], axis=-1)
    for pairs, nb in ((False, 16), (True, 8)):
        arm(f"field{amp}_ref_111_{'kramers' if pairs else 'old'}_beta0.7", "111", nb, pairs,
            magnetic_field=turning, beta=0.7)
        arm(f"field{amp}_ref_111_{'kramers' if pairs else 'old'}_beta0.3", "111", nb, pairs,
            magnetic_field=turning, beta=0.3)

Path("/l/ladovj1/review/open-a/vi3.json").write_text(json.dumps(out, indent=1))
print("VI3-DONE", flush=True)
