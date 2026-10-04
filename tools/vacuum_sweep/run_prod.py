"""One SCF with the production LDOS mixer: python3 run_prod.py <input> <mode> <flat|ddot> <out.json>.

LDOS_TOL and LDOS_SIGMA_MIN in the environment override the module's values.
"""
import json
import math
import os
import sys
import time

import defumat.scf.driver as driver
import defumat.scf.mixing as mixing
from defumat import Calculator

path, mode, fit, out = sys.argv[1:5]
driver.RHO_DDOT_FIT = fit == "ddot"
if os.environ.get("LDOS_TOL"):
    mixing.LDOS_TOL = float(os.environ["LDOS_TOL"])
if os.environ.get("LDOS_SIGMA_MIN"):
    mixing.LDOS_SIGMA_MIN = float(os.environ["LDOS_SIGMA_MIN"])
built = []
for name in ("ldos_preconditioner", "ldos_preconditioner_g"):
    original = getattr(driver, name)

    def build(*args, _original=original, **kwargs):
        built.append(_original(*args, **kwargs))
        return built[-1]

    setattr(driver, name, build)

calc = Calculator.from_file(path, announce=False)
record = {"input": path, "mode": mode, "fit": fit, "tol": mixing.LDOS_TOL,
          "sigma_min": mixing.LDOS_SIGMA_MIN}
start = time.perf_counter()
try:
    if os.environ.get("MIXING_SPACE"):
        # Calculator does not forward mixing_space; the module default is what
        # an unset one resolves to.
        mixing.DEFAULT_MIXING_SPACE = os.environ["MIXING_SPACE"]
    record["mixing_space"] = mixing.DEFAULT_MIXING_SPACE
    result = calc.get_scf(mixing_mode=mode)
    record.update(converged=bool(result.converged), iterations=int(result.iterations),
                  energy=float(result.total_energy),
                  energies=[float(h["total_energy"]) for h in result.history])
    if result.magnetization is not None:
        record["magnetization"] = float(result.magnetization)
except Exception as error:
    record.update(converged=False, error=f"{type(error).__name__}: {error}")
record["seconds"] = time.perf_counter() - start
if built:
    p = built[-1]
    solves = list(p.solves)
    record.update(solves=solves, states=p.states, clamped=p.clamped)
with open(out, "w") as handle:
    json.dump(record, handle, default=lambda x: str(x) if isinstance(x, float) and not math.isfinite(x) else x)
s = record.get("solves") or []
print(path, mode, fit, record.get("iterations"), record.get("converged"), record.get("energy"),
      f"{record['seconds']:.1f}s", f"cg/call {sum(a for a, _ in s) / max(len(s), 1):.1f}" if s else "",
      f"clamped {record.get('clamped', 0):.1e}", record.get("error", "")[:300], flush=True)
