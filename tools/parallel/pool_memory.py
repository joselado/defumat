"""What one pool holds: peak resident set at three points, gathered from every rank.

    DEFUMAT_POOLS=P ... python3 tools/parallel/pool_memory.py <input> <label> [--iterations 2]

The three points are after the import (the runtime and the distributed
client), after the ``Calculation`` is built (the replicated set and the per-k
tables), and after a short SCF (the store share, one k-point's solve, the
executables). ``ru_maxrss`` is a high-water mark, so each figure is the peak
up to that point. Rank 0 prints one JSON line with every rank's three numbers,
in GiB. ``DEFUMAT_PROJECTORS`` and ``DEFUMAT_CACHE_DIR`` are read as usual,
which is how the per-k projector store and the cache are switched between runs.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import warnings
from pathlib import Path

import numpy as np

import defumat  # noqa: F401
from defumat.io.pwin import read_pw_input
from defumat.parallel import current_pools
from defumat.pseudo import read_upf
from defumat.scf.driver import Calculation, run_scf
from defumat.system import build_system

REPO = Path(__file__).resolve().parents[2]


def peak() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("label")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--pseudo-dir", type=Path, default=REPO / "tests" / "data" / "pseudo")
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    pools = current_pools()
    marks = [peak()]
    system = build_system(read_pw_input(args.input))
    pseudos = tuple(read_upf(args.pseudo_dir / s.pseudo_file)
                    for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    marks.append(peak())
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-16,
                     max_iterations=args.iterations, verbose=False)
    marks.append(peak())
    local = np.array(marks)
    every = pools.communicator.allgather(local) if pools.size > 1 else local[None]
    if pools.rank == 0:
        print(json.dumps({
            "label": args.label, "input": args.input.name, "pools": pools.size,
            "nk": int(system.kpoints.nk),
            "projectors": calculation.projector_storage,
            "cache": os.environ.get("DEFUMAT_CACHE_DIR", "default"),
            "energy_ry": float(result.total_energy),
            "import_gib": [round(float(x), 3) for x in every[:, 0]],
            "calculation_gib": [round(float(x), 3) for x in every[:, 1]],
            "scf_gib": [round(float(x), 3) for x in every[:, 2]],
        }), flush=True)


if __name__ == "__main__":
    main()
