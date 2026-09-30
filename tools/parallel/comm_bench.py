"""Time the pools' collectives, and see what the others do when one pool dies.

    DEFUMAT_POOLS=P ... python3 tools/parallel/comm_bench.py [--sizes-mb 1.5 83 330] [--kill-after N]

Every rank runs it (``run_comm.sh`` on one node, ``srun`` across nodes). Rank 0
prints one JSON line with the warm time of an all-reduce and a broadcast of a
real array of each size, the eigenvalue gather and a scalar broadcast, the
median of ``--repeats`` after one discarded call.

``--kill-after N`` is the failure test: the last rank leaves with ``os._exit``
before its ``N``-th all-reduce, and every surviving rank prints, to stderr, how
long its next collective took to fail and with what, or nothing if it never
returns (the caller's ``timeout`` then says so). That is the case a pool
dying of an out-of-memory kill puts the others in.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

import defumat  # noqa: F401
from defumat.parallel import current_pools


def timed(fn, repeats):
    fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return round(1e3 * statistics.median(samples), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes-mb", type=float, nargs="+", default=[1.5, 83.0, 330.0])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--kill-after", type=int, default=None)
    args = parser.parse_args()
    pools = current_pools()

    if args.kill_after is not None:
        field = np.full((1 << 18,), float(pools.rank + 1))
        for step in range(args.kill_after + 5):
            if pools.rank == pools.size - 1 and step == args.kill_after:
                print(f"rank {pools.rank}: leaving before all-reduce {step}",
                      file=sys.stderr, flush=True)
                os._exit(3)
            start = time.perf_counter()
            try:
                pools.allreduce_sum(field)
            except Exception as error:  # noqa: BLE001 -- what it raises is the finding
                print(f"rank {pools.rank}: all-reduce {step} failed after "
                      f"{time.perf_counter() - start:.1f} s with "
                      f"{type(error).__name__}: {str(error)[:300]}",
                      file=sys.stderr, flush=True)
                sys.exit(2)
        print(f"rank {pools.rank}: every all-reduce returned", file=sys.stderr, flush=True)
        return

    out = {"size": pools.size, "hosts": os.environ.get("SLURM_JOB_NODELIST", "localhost")}
    for mb in args.sizes_mb:
        n = int(mb * 2**20 / 8)
        array = np.full((n,), float(pools.rank + 1))
        out[f"allreduce_{mb:g}MB_ms"] = timed(lambda: pools.allreduce_sum(array), args.repeats)
        out[f"broadcast_{mb:g}MB_ms"] = timed(lambda: pools.broadcast(array), args.repeats)
    eig = np.ones((1, 1, 200))
    out["gather_k_eig_ms"] = timed(lambda: pools.gather_k(eig, pools.size), args.repeats)
    out["broadcast_scalar_ms"] = timed(lambda: pools.broadcast_scalar(1.5), args.repeats)
    check = pools.allreduce_sum(np.ones(3))
    out["allreduce_correct"] = bool(np.allclose(np.asarray(check), pools.size))
    if pools.rank == 0:
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
