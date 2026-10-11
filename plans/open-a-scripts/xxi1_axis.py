"""OPEN.md Part XXI item 1, the PBE cobalt helix: M0, M2 and M3 of plans/open-a-xxi1-cobalt-pbe.md.

    xxi1_axis.py --root W --ecutrho 160 --q -0.25 --axis none|x|auto --arm lab|resident [--dft pz]
                 [--gga kubler|grad_m] [--conv-thr 1e-10] --out tag.json

``--axis auto`` leaves the Calculation's own rule (None for a spiral, the
starting moment's axis at q = 0); ``none`` and ``x`` override it.
"""
import argparse, json, time
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--ecutrho", type=float, default=160.0)
p.add_argument("--q", type=float, default=-0.25)
p.add_argument("--axis", default="auto", choices=("auto", "none", "x"))
p.add_argument("--arm", default="lab", choices=("lab", "resident"))
p.add_argument("--dft", default=None)
p.add_argument("--iterations", type=int, default=150)
p.add_argument("--gga", default="kubler", choices=("kubler", "grad_m"))
p.add_argument("--conv-thr", type=float, default=1e-10)
p.add_argument("--out", required=True)
a = p.parse_args()

import defumat.scf.driver as driver
from defumat import Calculator
from defumat.scf.driver import Calculation, run_scf

ROOT = Path(a.root)
if a.arm == "resident":
    driver._spiral_gvectors = (lambda basis, cell, qcart, needed:
                               (basis.dense, basis.smooth) if needed else (None, None))

text = (ROOT / "tests/data/qe/co-helix4-spiral.in").read_text()
text = text.replace("ecutrho = 160.0", f"ecutrho = {a.ecutrho}")
text = text.replace("spiral_q(1) = 0.0, spiral_q(2) = 0.0, spiral_q(3) = -0.25,",
                    "" if a.q == 0 else f"spiral_q(1) = 0.0, spiral_q(2) = 0.0, spiral_q(3) = {a.q},")
if a.dft:
    text = text.replace("occupations = 'smearing'", f"input_dft = '{a.dft}', occupations = 'smearing'")
front = Calculator.from_text(text, pseudo_dir=ROOT / "tests/data/pseudo", announce=False)
system, pseudos = front.system, front.pseudos
import dataclasses
system = dataclasses.replace(system, noncollinear_gga=a.gga)
calc = Calculation(system, pseudos)
rec = {"args": vars(a), "default_axis": calc.quantization_axis}
if a.axis == "none":
    calc.quantization_axis = None
elif a.axis == "x":
    calc.quantization_axis = (1.0, 0.0, 0.0)
rec["axis"] = calc.quantization_axis

# M0: members of the laboratory sets past the box's half range.
if getattr(calc, "spiral_dense", None) is not None:
    m0 = {}
    for name, gset, box in (("dense", calc.spiral_dense, calc.basis.dense),
                            ("smooth", calc.spiral_smooth, calc.basis.smooth)):
        mil = np.asarray(gset.miller)
        grid = np.asarray(box.grid if hasattr(box, "grid") else box.shape)
        half = (grid - 1) // 2
        m0[name] = dict(grid=grid.tolist(), max_abs=np.abs(mil).max(axis=0).tolist(),
                        past=int((np.abs(mil) > half).any(axis=1).sum()))
    rec["m0"] = m0
    print("M0", m0, flush=True)

t = time.time()
res = run_scf(system, pseudos, calculation=calc, conv_thr=a.conv_thr,
              max_iterations=a.iterations)
hist = res.history
acc = [float(h.get("accuracy", np.nan)) for h in hist]
ene = [float(h.get("total_energy", h.get("energy", np.nan))) for h in hist]
rec.update(iterations=int(res.iterations), converged=bool(res.converged),
           accuracy=float(res.accuracy), energy=float(res.total_energy),
           seconds=time.time() - t, accuracy_history=acc, energy_history=ene,
           energy_spread_last20=float(np.ptp(ene[-20:])) if len(ene) >= 2 else None,
           magnetization=np.asarray(res.total_magnetization).tolist()
           if hasattr(res, "total_magnetization") else None)
print(json.dumps({k: v for k, v in rec.items() if not k.endswith("_history")}), flush=True)
Path(a.out).write_text(json.dumps(rec, indent=1, default=str))
