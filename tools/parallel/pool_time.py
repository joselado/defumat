"""Warm per-iteration time of one SCF under whatever pools ``DEFUMAT_POOLS`` started.

    python3 tools/parallel/pool_time.py <input> <repeats> <label> [--max-iterations N]

Every pool runs this on the same input (``run_pools.sh`` starts them); rank 0
prints one JSON line. The first SCF compiles and is discarded, the next
``repeats`` reuse the same ``Calculation``, and the median ms per iteration is
reported with every sample beside it. Each rank's peak resident set is gathered
and printed too, since what a pool costs in memory is half of what a layout is
chosen on.

``--max-iterations N`` stops every SCF after ``N`` iterations with the threshold
tightened so that none stops earlier, as in ``time_scf.py``: the figure is then
comparable across layouts and not with a converged run's.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import time
import warnings
from pathlib import Path

import numpy as np

import defumat  # noqa: F401  (starts the pools before any array exists)
from defumat.io.pwin import read_pw_input
from defumat.parallel import current_pools
from defumat.pseudo import read_upf
from defumat.scf.driver import Calculation, run_scf
from defumat.system import build_system

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("repeats", type=int)
    parser.add_argument("label")
    parser.add_argument("--pseudo-dir", type=Path, default=REPO / "tests" / "data" / "pseudo")
    parser.add_argument("--conv-thr", type=float, default=1e-10)
    parser.add_argument("--max-iterations", type=int, default=None)
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    options = {"conv_thr": args.conv_thr}
    if args.max_iterations is not None:
        options = {"conv_thr": 1e-16, "max_iterations": args.max_iterations}

    pools = current_pools()
    system = build_system(read_pw_input(args.input))
    pseudos = tuple(read_upf(args.pseudo_dir / s.pseudo_file)
                    for s in system.structure.species)
    calculation = Calculation(system, pseudos)

    start = time.perf_counter()
    result = run_scf(system, pseudos, calculation=calculation, **options)
    cold = time.perf_counter() - start
    per_iteration, iterations = [], []
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = run_scf(system, pseudos, calculation=calculation, **options)
        wall = time.perf_counter() - start
        per_iteration.append(wall / result.iterations)
        iterations.append(result.iterations)

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # GiB
    peaks = pools.communicator.allgather(np.array([peak]))[:, 0] if pools.size > 1 else [peak]
    if pools.rank == 0:
        print(json.dumps({
            "label": args.label, "input": args.input.name, "pools": pools.size,
            "cpus_rank0": sorted(os.sched_getaffinity(0)),
            "cold_s": round(cold, 2), "iterations": iterations,
            "ms_per_iter_median": round(1e3 * statistics.median(per_iteration), 1),
            "ms_per_iter_all": [round(1e3 * t, 1) for t in per_iteration],
            "energy_ry": float(result.total_energy),
            "max_iterations": args.max_iterations,
            "peak_rss_gib": [round(float(p), 3) for p in peaks],
        }), flush=True)


if __name__ == "__main__":
    main()
