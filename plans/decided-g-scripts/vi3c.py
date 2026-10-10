"""OPEN.md Part VI item 3, after the decision: the field's own seed, A/B against the tiled start.

Run on D22 from /l/ladovj1/review/decided-g with PYTHONPATH at the decided-g checkout.
Arms: the plan's cell (vi3b) at 0.002 and 0.01 Ry, default against seed_from_field=False;
notebook 44's cell at 0.01 Ry, the same pair; the exact four-atom supercell under a
0.002 Ry LOCAL_MAGNETIC_FIELDS ladder from a ferromagnetic and from a helical start.
"""
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

from defumat import Calculator
from defumat.ultracell import run_ultracell
from tests.regression.test_ultracell import _noncollinear, _helix_supercell

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/l/ladovj1/w-g")
which = sys.argv[2] if len(sys.argv) > 2 else "all"
pseudo_dir = root / "tests/data/pseudo"
out_dir = Path("/l/ladovj1/review/decided-g")
tmp = out_dir / "vi3c-tmp"
tmp.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"vi3c-{which}.json"
out = json.loads(out_path.read_text()) if out_path.exists() else {}


def turning(amp, n=4):
    def field(x):
        phase = 2 * np.pi * x[..., 0] / n
        return amp * np.stack([np.cos(phase), np.sin(phase), np.zeros_like(phase)], axis=-1)
    return field


def record(name, r, started):
    m = np.asarray(r.cell_moments())
    hist = np.asarray(r.history, dtype=float)
    fe = None if r.field_energy is None else float(r.field_energy)
    rec = dict(iterations=int(r.iterations), converged=bool(r.converged),
               total_energy=float(r.total_energy), field_energy=fe,
               free=float(r.total_energy) + (fe or 0.0), moments=m.tolist(),
               lengths=np.linalg.norm(m, axis=1).tolist(), net=m.mean(axis=0).tolist(),
               seed_origin=getattr(r, "seed_origin", None),
               field_seed_skipped=getattr(r, "field_seed_skipped", None),
               seconds=time.time() - started, dr2_last=hist[-1].tolist())
    out[name] = rec
    out_path.write_text(json.dumps(out, indent=1))
    angles = np.degrees(np.arctan2(m[:, 1], m[:, 0]))
    print(f"{name}: {rec['iterations']} it, converged {rec['converged']}, E {rec['total_energy']:.10f}, "
          f"field {fe}, E+field {rec['free']:.10f}, origin {rec['seed_origin']}, "
          f"lengths {np.round(rec['lengths'], 5).tolist()}, angles {np.round(angles, 2).tolist()}, "
          f"net {np.round(rec['net'], 6).tolist()}, {rec['seconds']:.0f} s", flush=True)
    jax.clear_caches()


def aligned(a, b):
    """Largest difference of two moment sets after the best global rotation of b onto a."""
    a, b = np.asarray(a), np.asarray(b)
    u, _, vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1, 1, d]) @ vt
    return float(np.abs(b @ rot - a).max())


if which in ("all", "plan"):
    n, kgrid = 4, (1, 2, 2)
    shape = (n, 1, 1)
    folded = tuple(a * b for a, b in zip(shape, kgrid))
    calc = _noncollinear(tmp, pseudo_dir, (1, 1, 1), folded, angle1=54.735610, angle2=45.0,
                         tag="vi3c111")
    scf = calc.get_scf(conv_thr=1e-11, nbnd=8, max_iterations=300)
    out["plan_reference_energy"] = float(scf.total_energy)
    print("plan reference", scf.converged, scf.iterations, scf.total_energy, flush=True)
    for amp, label in ((0.002, "0.002"), (0.01, "0.01")):
        for arm, kw, budget in (("default", {}, 400), ("tiled", {"seed_from_field": False}, 400)):
            t = time.time()
            r = run_ultracell(calc.system, calc.pseudos, scf, shape, kgrid, nbnd=8, conv_thr=1e-10,
                              states_conv_thr=1e-8, mixing_beta=0.7, max_iterations=budget,
                              magnetic_field=turning(amp), **kw)
            record(f"plan_{label}_{arm}", r, t)
    for label in ("0.002", "0.01"):
        a, b = out.get(f"plan_{label}_default"), out.get(f"plan_{label}_tiled")
        if a and b:
            print(f"plan {label}: dE+field {a['free'] - b['free']:.3e} Ry, moments after a rotation "
                  f"{aligned(a['moments'], b['moments']):.3e}", flush=True)

if which in ("all", "nb44"):
    spinor = Calculator.from_file(root / "tests/data/qe/h-noncolin-ultracell.in", pseudo_dir=pseudo_dir)
    scf = spinor.get_scf()
    out["nb44_reference_energy"] = float(scf.total_energy)
    print("nb44 reference", scf.converged, scf.iterations, scf.total_energy, flush=True)
    for arm, kw in (("default", {}), ("tiled", {"seed_from_field": False})):
        for thr_label, thr in (("as_notebook", {}), ("1e-10", {"conv_thr": 1e-10, "max_iterations": 400})):
            t = time.time()
            r = spinor.get_ultracell(supercell=(4, 1, 1), kgrid=(1, 2, 2), nbnd=16,
                                     magnetic_field=turning(0.01), mixing_beta=0.3, **kw, **thr)
            record(f"nb44_{arm}_{thr_label}", r, t)
    for thr_label in ("as_notebook", "1e-10"):
        a, b = out.get(f"nb44_default_{thr_label}"), out.get(f"nb44_tiled_{thr_label}")
        if a and b:
            print(f"nb44 {thr_label}: dE+field {a['free'] - b['free']:.3e} Ry, moments after a "
                  f"rotation {aligned(a['moments'], b['moments']):.3e}", flush=True)

if which in ("all", "super"):
    n, kgrid, amp = 4, (1, 2, 2), 0.002
    card = "LOCAL_MAGNETIC_FIELDS\n" + "".join(
        f" {amp * np.cos(2 * np.pi * i / n):.12f} {amp * np.sin(2 * np.pi * i / n):.12f} 0.0\n"
        for i in range(n))
    for arm, build in (("ferro", lambda: _noncollinear(tmp, pseudo_dir, (n, 1, 1), kgrid,
                                                         angle1=54.735610, angle2=45.0, tag="sup_ferro")),
                       ("helix", lambda: _helix_supercell(tmp, pseudo_dir, n, kgrid))):
        plain = build()
        path = tmp / ("h_sup_ferro.in" if arm == "ferro" else f"h_helix{n}.in")
        held = path.with_name(path.stem + "_b0.002.in")
        held.write_text(path.read_text().rstrip("\n") + "\n" + card)
        sup = Calculator.from_file(held, pseudo_dir=pseudo_dir)
        t = time.time()
        r = sup.get_scf(conv_thr=1e-10, nbnd=16, max_iterations=400)
        sites = np.asarray(r.site_moments)
        fe = None if r.field_energy is None else float(r.field_energy)
        rec = dict(iterations=int(r.iterations), converged=bool(r.converged),
                   total_per_cell=float(r.total_energy) / n,
                   field_per_cell=None if fe is None else fe / n,
                   free_per_cell=(float(r.total_energy) + (fe or 0.0)) / n,
                   sites=sites.tolist(), seconds=time.time() - t)
        out[f"super_{arm}"] = rec
        out_path.write_text(json.dumps(out, indent=1))
        angles = np.degrees(np.arctan2(sites[:, 1], sites[:, 0]))
        print(f"super_{arm}: {rec['iterations']} it, converged {rec['converged']}, E/cell "
              f"{rec['total_per_cell']:.10f}, field/cell {rec['field_per_cell']}, E+field/cell "
              f"{rec['free_per_cell']:.10f}, site |m| {np.round(np.linalg.norm(sites, axis=1), 4).tolist()}, "
              f"angles {np.round(angles, 2).tolist()}, z {np.round(sites[:, 2], 5).tolist()}, "
              f"{rec['seconds']:.0f} s", flush=True)
        jax.clear_caches()

print("VI3C-DONE", which, flush=True)
