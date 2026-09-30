"""CPU lists for ``P`` processes of ``T`` cores each, packed by shared cache.

    python3 tools/parallel/cpu_layout.py 27 2        # 27 lists of 2, ";"-separated
    python3 tools/parallel/cpu_layout.py 1 16        # the first 16 cores, one list
    python3 tools/parallel/cpu_layout.py --describe  # what the ordering was built from

The order is read off ``lscpu -p=CPU,CORE,SOCKET,NODE,CACHE``: one CPU per
physical core (the lowest-numbered hardware thread), sorted by socket, NUMA
node, last-level cache and core, so that consecutive entries share an L3 where
the machine allows it. A process of ``T`` cores then takes ``T`` consecutive
entries and never straddles two caches when ``T`` divides the cache's core
count. On a Milan node (two sockets of 64 cores, eight cores per L3) a pool of
two, four or eight cores stays inside one L3, and the first 64 entries are one
socket.

``--smt`` keeps every hardware thread of a core together instead of one per
core, for a measurement of SMT itself.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def cores(smt: bool = False) -> list[list[int]]:
    """Physical cores in cache order, each the list of its hardware threads.

    Only CPUs this process may run on are listed: on a node shared with other
    jobs the Slurm allocation is a subset of what ``lscpu`` reports, and a
    ``taskset`` onto a CPU outside it fails.
    """
    text = subprocess.run(["lscpu", "-p=CPU,CORE,SOCKET,NODE,CACHE"],
                          capture_output=True, text=True, check=True).stdout
    allowed = os.sched_getaffinity(0)
    by_core: dict[tuple, list[int]] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        cpu, core, socket, node, cache = (line.split(",") + [""] * 5)[:5]
        if int(cpu) not in allowed:
            continue
        llc = cache.split(":")[-1] if cache else ""
        key = (int(socket or 0), int(node or 0), int(llc or 0), int(core))
        by_core.setdefault(key, []).append(int(cpu))
    ordered = [sorted(by_core[key]) for key in sorted(by_core)]
    return ordered if smt else [[threads[0]] for threads in ordered]


def layout(processes: int, width: int, smt: bool = False) -> list[list[int]]:
    """``processes`` lists of ``width`` cores' CPUs, consecutive in cache order."""
    available = cores(smt)
    if processes * width > len(available):
        raise SystemExit(f"cpu_layout: {processes} x {width} cores asked for, "
                         f"{len(available)} present")
    return [[cpu for core in available[p * width:(p + 1) * width] for cpu in core]
            for p in range(processes)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("processes", type=int, nargs="?")
    parser.add_argument("width", type=int, nargs="?")
    parser.add_argument("--smt", action="store_true")
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    if args.describe:
        for index, threads in enumerate(cores(smt=True)):
            print(index, threads)
        return
    if args.processes == 0:
        # ``cpu_layout.py 0 T``: how many processes of T cores the allowed CPUs hold.
        print(len(cores(args.smt)) // args.width)
        return
    if args.processes is None or args.width is None:
        parser.error("give the process count and the width")
    lists = layout(args.processes, args.width, args.smt)
    sys.stdout.write(";".join(",".join(map(str, cpus)) for cpus in lists) + "\n")


if __name__ == "__main__":
    main()
