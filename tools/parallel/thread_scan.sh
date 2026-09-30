#!/bin/bash
# Warm-SCF time per iteration over a set of CPU masks, one fresh process each.
#
#   MASKS="P1:0 P4:0,2,4,6 P4-bb4:0,2,4,6:4" tools/parallel/thread_scan.sh
#
# Each MASKS entry is label:cpus[:band_batch]. The mask is applied by taskset
# before Python starts and DEFUMAT_THREADS=off keeps the package from narrowing
# it, so the XLA pool is exactly the mask. Read the core layout off the machine
# first (lscpu -e, /sys/devices/system/cpu/cpu*/topology/thread_siblings_list,
# and on a hybrid chip /sys/devices/cpu_core/cpus and cpu_atom/cpus) and pick
# one CPU per physical core unless SMT is what is being measured.
#
# The set used on 2026-09-30 on an i5-12600K, whose performance cores are 0-11
# with siblings (2k, 2k+1) and whose efficiency cores are 12-15:
#   P1:0 P2:0,2 P4:0,2,4,6 P6:0,2,4,6,8,10 P6+SMT:0-11 E4:12-15
#   P4+E4:0,2,4,6,12-15 P6+E4:0,2,4,6,8,10,12-15 all16:0-15
#   P4-bb-all:0,2,4,6:all P6-bb-all:0,2,4,6,8,10:all
#   P2-bb2:0,2:2 P4-bb4:0,2,4,6:4 P6-bb6:0,2,4,6,8,10:6
# (PERFORMANCE.md, "Threads", has what it returned.)
#
# Environment: PYTHON (default python3), CELLS (benchmark inputs, default
# "si8-1k-ecut30.in si16-1k-ecut30.in"; a path containing a slash is used as
# given), REPEATS (default 5), MAX_ITERATIONS (unset runs every SCF to
# convergence; set, time_scf.py stops each SCF there), OUT (default
# thread-scan.jsonl in the current directory). DEFUMAT_CACHE_DIR is left to the
# caller; point it at local disk on a machine whose home is a network mount.
set -u
if [ -z "${MASKS:-}" ]; then
  echo "thread_scan: set MASKS to label:cpus[:band_batch] entries (see the header)" >&2
  exit 2
fi
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-python3}
CELLS=${CELLS:-"si8-1k-ecut30.in si16-1k-ecut30.in"}
REPEATS=${REPEATS:-5}
OUT=${OUT:-thread-scan.jsonl}
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" JAX_PLATFORMS=cpu DEFUMAT_THREADS=off \
       OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

for cell in $CELLS; do
  for entry in $MASKS; do
    IFS=: read -r label cpus band <<< "$entry"
    if [ -n "$band" ]; then export DEFUMAT_BAND_BATCH=$band; else unset DEFUMAT_BAND_BATCH; fi
    case "$cell" in */*) input=$cell ;; *) input=$REPO/benchmarks/$cell ;; esac
    taskset -c "$cpus" "$PYTHON" "$REPO/tools/parallel/time_scf.py" \
        "$input" "$REPEATS" "$label" ${MAX_ITERATIONS:+--max-iterations "$MAX_ITERATIONS"} >> "$OUT" \
      || echo "{\"label\": \"$label\", \"input\": \"$cell\", \"failed\": true}" >> "$OUT"
  done
done
