"""One warm-SCF timing in this process's affinity mask, printed as one JSON line.

    python3 tools/parallel/time_scf.py benchmarks/si16-1k-ecut30.in 5 P4

The mask is the caller's: set it with ``taskset`` *before* Python starts, and
set ``DEFUMAT_THREADS=off`` so that the package does not narrow it again on
import (``defumat/__init__.py``). ``JAX_PLATFORMS=cpu`` is needed on a machine
with a card, where the default backend is the card. The first SCF compiles and
is discarded (``CLAUDE.md``: never time a first call); the next ``repeats``
reuse the same ``Calculation`` and the median ms per iteration is reported with
every sample beside it, since a check that a number did *not* move wants the
median rather than the best of N.

``DEFUMAT_BAND_BATCH`` is read as usual, and the resolved band dial is printed
so that a typo in it cannot pass for a working setting.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import jax

from defumat.batching import resolve_band_batch
from defumat.io.pwin import read_pw_input
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
    args = parser.parse_args()

    system = build_system(read_pw_input(args.input))
    pseudos = tuple(read_upf(args.pseudo_dir / s.pseudo_file)
                    for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    jax.block_until_ready(calculation.vltot)

    start = time.perf_counter()
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=args.conv_thr)
    cold = time.perf_counter() - start

    per_iteration, iterations = [], []
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = run_scf(system, pseudos, calculation=calculation, conv_thr=args.conv_thr)
        wall = time.perf_counter() - start
        per_iteration.append(wall / result.iterations)
        iterations.append(result.iterations)

    print(json.dumps({
        "label": args.label,
        "input": args.input.name,
        "cpus": sorted(os.sched_getaffinity(0)),
        "band_batch": os.environ.get("DEFUMAT_BAND_BATCH", "default"),
        "resolved_band_batch": str(resolve_band_batch()),
        "backend": jax.default_backend(),
        "cold_s": round(cold, 3),
        "ms_per_iter_median": round(1e3 * statistics.median(per_iteration), 2),
        "ms_per_iter_all": [round(1e3 * t, 2) for t in per_iteration],
        "iterations": iterations,
        "energy_ry": float(result.total_energy),
    }), flush=True)


if __name__ == "__main__":
    main()
