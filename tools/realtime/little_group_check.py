"""The current of a pulse on the whole 4x4x4 mesh against the wedge of the field's little group.

    JAX_PLATFORMS=cpu python3 tools/realtime/little_group_check.py
"""
import time, json, numpy as np, sys
import defumat
from pathlib import Path
from defumat import Calculator
from defumat.realtime.pulse import Sin2
repo = Path(__file__).resolve().parents[2]
c = Calculator.from_file(repo/"tests/data/qe/si2-symmetric.in" if (repo/"tests/data/qe/si2-symmetric.in").exists() else repo/"benchmarks/si-1k.in", pseudo_dir=repo/"tests/data/pseudo", announce=False)
c.get_scf(conv_thr=1e-10)
out = {}
for direction in ((1, 0, 0), (1, 1, 0), (1, 2, 3)):
    pulse = Sin2.from_intensity(5e11, 1.55, 2, direction)
    runs = {}
    for lg in (False, True):
        t = time.time()
        r = c.get_realtime(pulse, grid=(4, 4, 4), little_group=lg, dt=0.2)
        runs[lg] = (r, time.time() - t)
    a, b = runs[False][0].current, runs[True][0].current
    out[str(direction)] = {"ops": runs[True][0].symmetry_operations,
                           "max_diff": float(np.abs(a - b).max()), "scale": float(np.abs(a).max()),
                           "seconds_full": runs[False][1], "seconds_wedge": runs[True][1],
                           "work_minus_gain": runs[True][0].work - runs[True][0].energy_gained}
    print(direction, out[str(direction)], flush=True)
