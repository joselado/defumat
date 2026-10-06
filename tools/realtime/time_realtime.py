"""One core of defumat on Elk's Si-dielectric propagation: the time it takes, and the current.

The defumat half of ``tools/realtime/time_against_elk.sh``. From the converged
ground state of ``tests/data/elk/si_rt/si-dielectric.in`` (not timed), a short
run first, which compiles every program the long one uses, then the timed run:
Elk's example's field, a constant ``A = 0.1`` along x from ``t = 0`` (a kick of
``kappa = 0.1/c``), over 800 Hartree a.u. in steps of 0.2 on the 8x8x8 grid's
wedge for that field. The fixed-density solve on the wedge and the propagation
are timed apart. The current is saved for the comparison with Elk's.

    taskset -c 0 python3 tools/realtime/time_realtime.py <out.npz>
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import defumat  # noqa: F401
from defumat import Calculator
from defumat.realtime.pulse import Kick
from defumat.units import C_AU

repo = Path(__file__).resolve().parents[2]
calc = Calculator.from_file(repo / "tests/data/elk/si_rt/si-dielectric.in",
                            pseudo_dir=repo / "tests/data/pseudo", announce=False)
calc.get_scf()
kick = Kick(strength=0.1 / C_AU, direction=(1.0, 0.0, 0.0))
calc.get_realtime(kick, grid=(8, 8, 8), dt=0.2, duration=0.2 * 400)   # compiles

from defumat.workflows import realtime as workflow  # noqa: E402

marks = {}
original = workflow._occupied_states


def timed(*args, **kwargs):
    start = time.perf_counter()
    out = original(*args, **kwargs)
    marks["fixed_density_s"] = time.perf_counter() - start
    return out


workflow._occupied_states = timed
start = time.perf_counter()
run = calc.get_realtime(kick, grid=(8, 8, 8), dt=0.2, duration=800.0)
total = time.perf_counter() - start
np.savez(sys.argv[1], times=run.times, current=run.current, volume=run.volume,
         energy=run.energy, energy_times=run.energy_times)
print(json.dumps({
    "total_s": total, "fixed_density_s": marks["fixed_density_s"],
    "propagation_s": total - marks["fixed_density_s"], "steps": len(run.times) - 1,
    "kpoints_wedge": int(len(calc.calculation.system.kpoints.weights)),
    "symmetry_operations": run.symmetry_operations, "norm_drift": run.norm_drift,
    "affinity": sorted(os.sched_getaffinity(0)),
}), flush=True)
