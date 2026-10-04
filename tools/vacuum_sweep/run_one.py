"""One SCF of the vacuum sweep: python3 run_one.py <input> <mode> <flat|ddot> <out.json> [beta]."""
import json
import math
import sys
import time

import jax

import defumat.scf.driver as driver
from defumat import Calculator

path, mode, fit, out = sys.argv[1:5]
beta = float(sys.argv[5]) if len(sys.argv) > 5 else None
driver.RHO_DDOT_FIT = fit == "ddot"

# For the LDOS arm: keep the preconditioner to read its counters, and clock the
# extra pass over the bands that builds the LDOS.
built, ldos_clock = [], {"calls": 0, "seconds": 0.0}
if mode == "ldos":
    import defumat.scf.mixing as mixing
    mixing.LDOS_COUNT_MATVECS = True
    import os
    if os.environ.get("LDOS_TOL"):
        mixing.LDOS_TOL = float(os.environ["LDOS_TOL"])
    original_build, original_ldos = driver.ldos_preconditioner, driver._fermi_ldos

    def build(*args, **kwargs):
        built.append(original_build(*args, **kwargs))
        return built[-1]

    def timed_ldos(*args, **kwargs):
        start = time.perf_counter()
        value = jax.block_until_ready(original_ldos(*args, **kwargs))
        ldos_clock["calls"] += 1
        ldos_clock["seconds"] += time.perf_counter() - start
        return value

    driver.ldos_preconditioner, driver._fermi_ldos = build, timed_ldos

calc = Calculator.from_file(path, announce=False)
options = {"mixing_mode": mode}
if beta is not None:
    options["mixing_beta"] = beta
start = time.perf_counter()
record = {"input": path, "mode": mode, "fit": fit, "beta": beta}
try:
    result = calc.get_scf(**options)
    record.update(
        converged=bool(result.converged), iterations=int(result.iterations),
        energy=float(result.total_energy),
        energies=[float(h["total_energy"]) for h in result.history],
        accuracies=[float(h["accuracy"]) for h in result.history],
        grid=list(result.density.shape[1:]),
    )
    if result.magnetization is not None:
        record["magnetization"] = float(result.magnetization)
        record["absolute_magnetization"] = float(result.absolute_magnetization)
except Exception as error:  # a diverged run is a result, not a crash of the sweep
    record.update(converged=False, error=f"{type(error).__name__}: {error}")
record["seconds"] = time.perf_counter() - start
if built:
    record.update(precond_calls=built[-1].calls, precond_seconds=built[-1].seconds,
                  matvecs=built[-1].matvecs, ldos_calls=ldos_clock["calls"],
                  ldos_seconds=ldos_clock["seconds"])


def clean(x):
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x


with open(out, "w") as handle:
    json.dump({k: clean(v) for k, v in record.items()}, handle)
print(path, mode, fit, record.get("iterations"), record.get("converged"),
      record.get("energy"), f"{record['seconds']:.1f}s", flush=True)
