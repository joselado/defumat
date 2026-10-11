"""The cobalt helix's four-cell supercell as a plain noncollinear run, by functional.

    gradm_supercell.py --root W --gga kubler|grad_m [--conv-thr 1e-11] --out tag.json

No axis: the four starting moments are not parallel, so 'kubler' takes |m|.
Its energy over four is the spiral's, by the generalized Bloch theorem.
"""
import argparse, dataclasses, json, sys, time
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--gga", default="kubler", choices=("kubler", "grad_m"))
p.add_argument("--conv-thr", type=float, default=1e-11)
p.add_argument("--iterations", type=int, default=150)
p.add_argument("--out", required=True)
a = p.parse_args()
ROOT = Path(a.root)
sys.path.insert(0, str(ROOT))
from tests.conftest import GENERATED
from defumat import Calculator
from defumat.scf.driver import Calculation, run_scf

front = Calculator.from_text((GENERATED / "co-helix4-nosoc.in").read_text(),
                             pseudo_dir=ROOT / "tests/data/pseudo", announce=False)
system = dataclasses.replace(front.system, noncollinear_gga=a.gga)
calc = Calculation(system, front.pseudos)
t = time.time()
res = run_scf(system, front.pseudos, calculation=calc, conv_thr=a.conv_thr,
              max_iterations=a.iterations)
rec = dict(args=vars(a), axis=calc.quantization_axis, iterations=int(res.iterations),
           converged=bool(res.converged), accuracy=float(res.accuracy),
           energy=float(res.total_energy), energy_per_cell=float(res.total_energy) / 4,
           seconds=time.time() - t)
print(json.dumps(rec), flush=True)
Path(a.out).write_text(json.dumps(rec, indent=1, default=str))
