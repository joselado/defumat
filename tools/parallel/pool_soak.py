"""A long-lived pooled process that compiles many cells: the deadlock's condition.

    DEFUMAT_POOLS=2 ... python3 tools/parallel/pool_soak.py [--cells 12]

The XLA thread-pool deadlock (``OPEN.md`` Part IV item 1) has only been seen in
processes that had already compiled many different cells, at narrow affinity
masks, and the pools' preferred width is exactly such a mask. This walks
``--cells`` silicon cells without symmetry, each at a different cutoff and so a
different set of executables, then a noncollinear iron metal, all in one
process, and prints each one's completion to stderr with a timestamp; a hang is
the caller's ``timeout`` firing with the last line naming the cell it stopped
in. Rank 0 prints one JSON line at the end.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import warnings
from pathlib import Path

import defumat  # noqa: F401
from defumat.io.pwin import read_pw_input
from defumat.parallel import current_pools
from defumat.pseudo import read_upf
from defumat.scf.driver import run_scf
from defumat.system import build_system

REPO = Path(__file__).resolve().parents[2]
SILICON = """\
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.2, nat = 2, ntyp = 1, ecutwfc = {ecut},
  nosym = .true., noinv = .true.
/
&electrons
  conv_thr = 1.0d-9
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""


def iron() -> str:
    text = (REPO / "benchmarks" / "fe-mag-1k.in").read_text()
    text = text[: text.upper().index("K_POINTS")] + "K_POINTS automatic\n 2 2 2 0 0 0\n"
    return text.replace("&system", "&system\n  nosym = .true., noinv = .true.,", 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells", type=int, default=12)
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    pools = current_pools()
    texts = [(f"si-ecut{10 + 2 * i}", SILICON.format(ecut=10 + 2 * i))
             for i in range(args.cells)] + [("fe-noncollinear", iron())]
    start = time.perf_counter()
    energies = {}
    for name, text in texts:
        with tempfile.NamedTemporaryFile("w", suffix=".in", delete=False) as handle:
            handle.write(text)
        system = build_system(read_pw_input(Path(handle.name)))
        pseudos = tuple(read_upf(REPO / "tests" / "data" / "pseudo" / s.pseudo_file)
                        for s in system.structure.species)
        result = run_scf(system, pseudos, verbose=False)
        energies[name] = float(result.total_energy)
        print(f"rank {pools.rank}: {name} done at {time.perf_counter() - start:.1f} s",
              file=sys.stderr, flush=True)
    if pools.rank == 0:
        print(json.dumps({"pools": pools.size, "cells": len(texts),
                          "wall_s": round(time.perf_counter() - start, 1),
                          "energies": energies}), flush=True)


if __name__ == "__main__":
    main()
