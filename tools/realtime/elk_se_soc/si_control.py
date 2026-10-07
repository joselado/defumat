"""A load control: the silicon frozen run of PERFORMANCE.md's Elk pair, 400 steps, warm.

On a quiet core 0 it read 1.21 ms per k-point and step (486 s for 4000 steps on the
100-point wedge); this reads the same quantity now, to say what the load costs.

    taskset -c 0 python3 si_control.py
"""
import json
import os
import time

import defumat  # noqa: F401
from defumat import Calculator
from defumat.realtime.pulse import Kick
from defumat.units import C_AU

from se_common import REPO

calc = Calculator.from_file(REPO / "tests/data/elk/si_rt/si-dielectric.in",
                            pseudo_dir=REPO / "tests/data/pseudo", announce=False)
calc.get_scf()
kick = Kick(strength=0.1 / C_AU, direction=(1.0, 0.0, 0.0))
calc.get_realtime(kick, grid=(8, 8, 8), dt=0.2, duration=80.0)   # compiles
from defumat.workflows import realtime as workflow  # noqa: E402

marks = {}
original = workflow._occupied_states


def timed(*args, **kwargs):
    start = time.perf_counter()
    out = original(*args, **kwargs)
    marks["fixed"] = time.perf_counter() - start
    return out


workflow._occupied_states = timed
t0 = time.perf_counter()
run = calc.get_realtime(kick, grid=(8, 8, 8), dt=0.2, duration=80.0)
total = time.perf_counter() - t0
print(json.dumps({"si_control_total_s": total, "steps": 400, "nk": 100,
                  "fixed_density_s": marks["fixed"],
                  "ms_per_k_step": 1e3 * (total - marks["fixed"]) / (400 * 100),
                  "quiet_reference_ms_per_k_step": 1.21,
                  "affinity": sorted(os.sched_getaffinity(0))}), flush=True)
