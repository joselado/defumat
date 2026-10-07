"""chi^(2) and chi^(3) of AlAs from the ultrasoft and the PAW pslibrary 1.0.0 datasets (both PBE)
on one cell: alas-epsilon-us.in with only the datasets and the cutoffs changed. The hierarchy at
1.5 eV, eta = 0.3 eV, [111], the 2x2x2 mesh, orders one to three.

usage: datasets.py <us|paw> <ecutwfc> <ecutrho> [calls]
"""
import json, math, re, sys, time
from pathlib import Path

import numpy as np

from defumat import Calculator

REPO = Path("/l/ladovj1/defumat-hspin")
HERE = Path(__file__).resolve().parent
kind, ecut, ecutrho = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
text = (REPO / "tests/data/qe/alas-epsilon-us.in").read_text()
text = re.sub(r"ecutwfc\s*=\s*[0-9.dD+-]+", f"ecutwfc = {ecut}", text)
text = re.sub(r"ecutrho\s*=\s*[0-9.dD+-]+", f"ecutrho = {ecutrho}", text)
if kind == "paw":
    text = text.replace("rrkjus_psl.1.0.0", "kjpaw_psl.1.0.0")
assert text.count("kjpaw" if kind == "paw" else "rrkjus") == 2
path = HERE / f"datasets-{kind}-{ecut:g}.in"
path.write_text(text)
calculator = Calculator.from_file(path, pseudo_dir=REPO / "tests/data/pseudo", announce=False)
start = time.time()
scf = calculator.get_scf(conv_thr=1e-12)
print(kind, ecut, ecutrho, "scf", round(time.time() - start, 1), "s, energy", scf.total_energy, flush=True)
options = dict(broadening=0.3, direction=tuple(np.ones(3) / math.sqrt(3.0)), grid=(2, 2, 2),
               conv_thr=1e-12)
calls = []
for call in range(int(sys.argv[4]) if len(sys.argv) > 4 else 1):
    start = time.time()
    spectrum = calculator.get_nonlinear_spectrum([1.5], order=3, **options)
    calls.append(time.time() - start)
    print(kind, ecut, "call", call, round(calls[-1], 1), "s", flush=True)
elapsed = calls[-1]
out = {"kind": kind, "ecut": ecut, "ecutrho": ecutrho, "energy": float(scf.total_energy),
       "seconds": elapsed, "iterations": list(map(int, spectrum.iterations))}
for key in ((1, 1), (2, 2), (2, 0), (3, 3), (3, 1)):
    value = complex(spectrum.component(*key, axis=0)[0])
    out[str(key)] = [value.real, value.imag]
    print(kind, ecut, key, f"{value:.6e}", flush=True)
chi2, chi3 = complex(spectrum.chi2(0)[0]), complex(spectrum.chi3(0)[0][0])
out["chi2"], out["chi3"] = [chi2.real, chi2.imag], [chi3.real, chi3.imag]
print(kind, ecut, "hierarchy", round(elapsed, 1), "s; chi2 [111] x (pm/V)", f"{chi2:.6e}",
      "chi3 (m^2/V^2)", f"{chi3:.6e}", flush=True)
(HERE / f"datasets-{kind}-{ecut:g}-{len(calls)}.json").write_text(json.dumps(out))
