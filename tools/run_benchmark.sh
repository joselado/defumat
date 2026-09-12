#!/usr/bin/env bash
# The laptop benchmark: Quantum ESPRESSO, defumat on a CPU core, and a GPU if
# there is one. The counterpart to tools/cluster/submit_benchmark.py, which puts
# the same sweep into a Slurm array.
#
#   tools/run_benchmark.sh              # the fast set, ~10 cases, both codes
#   tools/run_benchmark.sh complete     # the whole thing, hours rather than minutes
#   tools/run_benchmark.sh fast --gpu off --resume
#
# Anything after the set name is passed through to performance/sweep.py.
#
# It needs pw.x built serially once, which is the QE leg's whole requirement:
#   cd quantum_espresso/qe-7.5-ReleasePack/qe-7.5
#   ./configure --disable-parallel --disable-openmp && make -j pw
# Without it the QE leg fails and is reported as such, case by case; the defumat
# columns still land.
#
# `complete` carries cells that do not fit every machine -- bi20-soc peaks near
# 35 GB. Those are refused by name against this machine's RAM before they start,
# rather than left to the OOM killer, which here takes the terminal with it.
# --force-large overrides that if you know the machine is big enough.
#
# Do not run this beside a test run or anything else being measured. A timing
# taken next to a background suite is not a timing -- PERFORMANCE.md has the
# episode where one read 70% slow and had to be discarded.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SET="${1:-fast}"
[ $# -gt 0 ] && shift

case "$SET" in
  fast|complete) ;;
  -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "unknown set: $SET (fast or complete)" >&2; exit 2 ;;
esac

cd "$ROOT"
echo "defumat benchmark, set '$SET' -- both codes on one core, one GPU if present."
echo "results: performance/results/$SET/"
echo
exec python3 performance/sweep.py --set "$SET" "$@"
