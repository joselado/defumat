"""The two benchmark sets: which cells are measured, and why each one is there.

A benchmark set is a list of cases, and a case is the name of a ``pw.x`` input
resolved against ``benchmarks/`` and then ``tests/data/qe/``. Nothing here runs
anything -- ``performance/sweep.py`` is the harness, and this module exists so
that the laptop run and the cluster array measure the *same* list without it
being written down twice.

**fast** is the gate: one case per kind of physics, all of them single k-point
and 2-8 atoms, sized so the whole set finishes on a laptop while someone waits.
It answers "did anything move?" and nothing else.

**complete** is the claim. It adds the size ladder -- the same silicon cell at
8, 16, 32 and 64 atoms, which is where the cost stops being fixed overhead and
starts being the physics -- and the ten-atom sweep across the pseudopotential
kinds, a GGA, a metal, collinear and noncollinear magnetism, spin-orbit
coupling and DFT+U, so that size and physics are not confounded in one number.

**The single-k cases are single-k on purpose**: both codes parallelise over k,
so a multi-k comparison against QE measures batching rather than the cost of
the physics. The ten-atom sweep cases carry a 4x4x1 grid deliberately, because
the k axis is exactly where a GPU has something to win and a set with no
multi-k case in it cannot show that. Which is which is not hardcoded here --
the harness records ``nk`` per case and the report says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = ROOT / "benchmarks"
QE_CASES = ROOT / "tests" / "data" / "qe"


@dataclass(frozen=True)
class Case:
    """One benchmark cell: the input's stem, why the set contains it, and its size.

    ``peak_gb`` is the working set this case is known to reach, and it is a
    **guard rather than a measurement**: the figures are the device peaks from
    the GPU sweep recorded in ``PERFORMANCE.md`` ("Memory is the constraint that
    bites"), used here only to keep the local runner from starting a case that
    cannot fit in the machine's RAM. A host peak is not the same number as a
    device peak; it is the right order, which is all this is asked to be.
    ``None`` means unmeasured, and unmeasured is not treated as small.

    ``max_iterations`` overrides the sweep's default where a cell is known to
    need more. Two cases do, and getting it wrong has already produced a wrong
    result once: ``ni10-ldau`` and ``h40-chain-lsda`` converge in 151 and 104
    iterations, so a cap of 100 reports them as *not converged* rather than as
    capped -- which is exactly what the first GPU sweep did before
    ``PERFORMANCE.md`` caught it.
    """

    name: str
    description: str
    peak_gb: float | None = None
    max_iterations: int | None = None

    @property
    def path(self) -> Path:
        return resolve(self.name)


def resolve(name: str) -> Path:
    """A case name as a path, looked up in ``benchmarks/`` then ``tests/data/qe/``.

    The two directories are one namespace on purpose: the size ladder and the
    feature cells live in ``benchmarks/``, the ten-atom sweep cells were added
    for the GPU work and live beside the reference outputs they are checked
    against. A case is named once and found wherever it is.
    """
    direct = Path(name)
    if direct.is_file():
        return direct
    stem = direct.stem if direct.suffix == ".in" else name
    for directory in (BENCHMARKS, QE_CASES):
        candidate = directory / f"{stem}.in"
        if candidate.is_file():
            return candidate
    raise SystemExit(f"no such case: {name!r} (looked in {BENCHMARKS} and {QE_CASES})")


#: One case per kind of physics, small enough to run while someone waits.
FAST = [
    Case("si-1k", "Si, 2 atoms, LDA, norm-conserving -- the smallest thing that runs"),
    Case("si-1k-ecut40", "the same cell at a production cutoff"),
    Case("si8-1k", "Si, 8 atoms -- where the cost is physics, not overhead"),
    Case("si2-us-1k", "ultrasoft: the augmentation charge and two grids"),
    Case("si2-paw-1k", "PAW: the one-centre terms on top"),
    Case("si8-pbe-1k", "a gradient-corrected functional"),
    Case("si8-smeared-1k", "a metal: smearing, and the collinear half of the spinor pair"),
    Case("si8-nc-1k", "the same run as spinors -- what npol = 2 costs"),
    Case("pt-so-1k", "spin-orbit: j-resolved projectors, a complex 2x2 D_ij"),
    Case("fe-mag-1k", "a magnetization that is a vector: nspin_mag = 4"),
]

#: The size ladder. Same cell, same physics, four sizes -- so a ratio that
#: changes with size can be told from one that does not.
_LADDER = [
    Case("si16-1k", "the ladder: 16 atoms"),
    Case("si32-1k", "the ladder: 32 atoms"),
    Case("si64-1k", "the ladder: 64 atoms -- the largest that stays inside a laptop"),
]

#: Ten atoms across the physics, on a 4x4x1 grid. Same size throughout, so what
#: moves between these rows is the physics and not the cell.
_SWEEP = [
    Case("si10-nc", "ten atoms, norm-conserving -- the sweep's own baseline"),
    Case("si10-us", "ten atoms, ultrasoft"),
    Case("si10-paw", "ten atoms, PAW"),
    Case("si10-nc-pbe", "ten atoms, PBE"),
    Case("al10-metal", "a metal with smearing, ten atoms"),
    Case("h10-chain-lsda", "collinear spin, nspin = 2"),
    Case("h10-chain-noncolin", "noncollinear, npol = 2", peak_gb=1.4),
    Case("bi10-soc", "spin-orbit on a heavy element", peak_gb=16.9),
    Case("ni10-ldau", "DFT+U", peak_gb=1.6, max_iterations=200),
]

#: The big ones. These are where a GPU is supposed to earn its keep and where
#: single-core QE starts to hurt; they are in `complete` and nowhere else.
_LARGE = [
    Case("h20-chain-lsda", "twenty atoms, collinear spin", peak_gb=1.1),
    Case("bi20-soc", "twenty atoms, spin-orbit -- the heaviest case here",
         peak_gb=34.7),
    Case("h40-chain-lsda", "forty atoms, collinear spin",
         peak_gb=4.1, max_iterations=200),
]

#: Everything, in the order the report tabulates it: the fast set first, then
#: size, then physics at fixed size, then the large cells.
COMPLETE = FAST + _LADDER + _SWEEP + _LARGE

SETS = {"fast": FAST, "complete": COMPLETE}


def cases(name: str) -> list[Case]:
    """The named set, or a clear error naming the ones that exist."""
    try:
        return SETS[name]
    except KeyError:
        raise SystemExit(f"no such set: {name!r} (have {', '.join(sorted(SETS))})") from None


def main(argv=None) -> int:
    """``python3 performance/sets.py fast`` -- one case name per line.

    The shell needs this: the sbatch array bakes the list in as a bash array,
    and it is generated from here rather than retyped.
    """
    import argparse

    parser = argparse.ArgumentParser(description="list the cases in a benchmark set")
    parser.add_argument("set", nargs="?", default="fast", choices=sorted(SETS))
    parser.add_argument("--describe", action="store_true", help="one case and its reason per line")
    parser.add_argument("--check", action="store_true", help="verify every input file exists")
    arguments = parser.parse_args(argv)

    missing = 0
    for case in cases(arguments.set):
        if arguments.check:
            try:
                case.path
            except SystemExit as error:
                print(f"MISSING  {case.name}: {error}")
                missing += 1
                continue
        print(f"{case.name:22s}  {case.description}" if arguments.describe else case.name)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
