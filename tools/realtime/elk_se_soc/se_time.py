"""One core of defumat on selenium with spin-orbit coupling under Elk's Gaussian pulse.

    taskset -c 0 python3 se_time.py ECUT N DT STEPS BLOCK POTENTIAL OUT.npz [FWHM_FS] [INTENSITY]

The SCF twice (the first compiles; the second, on a fresh calculator, is timed),
then get_realtime twice in the same process: one block of BLOCK steps, which
compiles every program the long run uses, and the timed run of STEPS steps (a
multiple of BLOCK, so no padded steps), with the fixed-density solve on the
field's wedge timed apart. The current is saved for the comparison with Elk's
JTOT_TD.OUT.
"""
import json
import os
import sys
import time

import numpy as np

import defumat  # noqa: F401
from defumat import Calculator

from se_common import PSEUDO, make_input, pulse

ecut, n, dt = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
steps, block, potential, out = int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7]
fwhm = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0
intensity = float(sys.argv[9]) if len(sys.argv) > 9 else 1.0e10
assert steps % block == 0
path = make_input(ecut, (n, n, n))
p = pulse(intensity=intensity, fwhm_fs=fwhm)

t0 = time.perf_counter()
Calculator.from_file(path, pseudo_dir=PSEUDO, announce=False).get_scf()
scf_first = time.perf_counter() - t0
calc = Calculator.from_file(path, pseudo_dir=PSEUDO, announce=False)
t0 = time.perf_counter()
scf = calc.get_scf()
scf_warm = time.perf_counter() - t0
print(json.dumps({"scf_first_s": scf_first, "scf_warm_s": scf_warm,
                  "scf_iterations": scf.iterations, "nks_scf": len(calc.system.kpoints.weights),
                  "total_energy_ry": float(scf.total_energy)}), flush=True)

from defumat.workflows import realtime as workflow  # noqa: E402

marks = {}
original = workflow._occupied_states


def timed(*args, **kwargs):
    start = time.perf_counter()
    result = original(*args, **kwargs)
    marks["fixed_density_s"] = time.perf_counter() - start
    marks["states_shape"] = list(np.shape(result[1]))
    return result


workflow._occupied_states = timed
t0 = time.perf_counter()
calc.get_realtime(p, grid=(n, n, n), dt=dt, duration=dt * block, potential=potential,
                  block_steps=block)
compile_s = time.perf_counter() - t0
t0 = time.perf_counter()
run = calc.get_realtime(p, grid=(n, n, n), dt=dt, duration=dt * steps, potential=potential,
                        block_steps=block)
total = time.perf_counter() - t0
np.savez(out, times=run.times, current=run.current, kappa=run.kappa, efield=run.efield,
         volume=run.volume, energy=run.energy, energy_times=run.energy_times)
nk = marks["states_shape"][-3]
print(json.dumps({
    "potential": potential, "ecut": ecut, "grid": n, "dt": dt, "steps": steps,
    "pulse": {"amplitude": p.amplitude, "omega": p.omega, "fwhm": p.fwhm, "peak": p.peak},
    "compile_call_s": compile_s, "total_s": total,
    "fixed_density_s": marks["fixed_density_s"], "propagation_s": total - marks["fixed_density_s"],
    "states_shape": marks["states_shape"], "nk_wedge": nk,
    "ms_per_k_step": 1e3 * (total - marks["fixed_density_s"]) / (steps * nk),
    "symmetry_operations": run.symmetry_operations, "norm_drift": run.norm_drift,
    "excited": run.excited, "nelec": run.nelec, "volume": run.volume,
    "affinity": sorted(os.sched_getaffinity(0)),
}), flush=True)
