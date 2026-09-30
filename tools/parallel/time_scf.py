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

``DEFUMAT_BAND_BATCH`` and ``DEFUMAT_MEMORY_MODE`` are read as usual, and the
resolved band dial and memory mode are printed so that a typo in either cannot
pass for a working setting. On an accelerator the device's peak bytes in use
are printed too; the counter has no reset, so the figure is the whole process's
high-water mark, compiling run included, and one configuration per process is
what makes it that configuration's.

``--max-iterations N`` stops every SCF after ``N`` iterations (with the
threshold tightened so that none stops earlier), which bounds the cost on a
large cell. The per-iteration figure then averages over the first ``N``
iterations, whose Davidson calls run to a looser threshold than a converged
run's last ones, so it is comparable across masks and not with a full-SCF
figure.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import jax

from defumat.batching import resolve_band_batch, resolve_memory_mode
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
    parser.add_argument("--max-iterations", type=int, default=None)
    args = parser.parse_args()
    options = {"conv_thr": args.conv_thr}
    if args.max_iterations is not None:
        options = {"conv_thr": 1e-16, "max_iterations": args.max_iterations}

    system = build_system(read_pw_input(args.input))
    pseudos = tuple(read_upf(args.pseudo_dir / s.pseudo_file)
                    for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    jax.block_until_ready(calculation.vltot)

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

    device = jax.local_devices()[0]
    stats = device.memory_stats() if device.platform != "cpu" else None
    print(json.dumps({
        "label": args.label,
        "input": args.input.name,
        "cpus": sorted(os.sched_getaffinity(0)),
        "band_batch": os.environ.get("DEFUMAT_BAND_BATCH", "default"),
        "resolved_band_batch": str(resolve_band_batch()),
        "memory_mode": resolve_memory_mode(),
        "calculation_band_batch": str(getattr(calculation, "band_batch", "n/a")),
        "backend": jax.default_backend(),
        "cold_s": round(cold, 3),
        "ms_per_iter_median": round(1e3 * statistics.median(per_iteration), 2),
        "ms_per_iter_all": [round(1e3 * t, 2) for t in per_iteration],
        "iterations": iterations,
        "energy_ry": float(result.total_energy),
        "max_iterations": args.max_iterations,
        "device_peak_gib": (round(stats["peak_bytes_in_use"] / 2**30, 3)
                            if stats and "peak_bytes_in_use" in stats else None),
    }), flush=True)


if __name__ == "__main__":
    main()
