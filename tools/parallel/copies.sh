#!/bin/bash
# P independent single-process SCF timings at once, each pinned to its own T cores.
#
#   OUT=copies.jsonl tools/parallel/copies.sh <input> <label> <P> <T> [time_scf.py options]
#
# The ideal pool throughput with no communication at all: every copy is
# time_scf.py on the same input, started together, so the only thing they share
# is the node (its caches, its memory channels, its sockets). P = 1 is the
# same copy alone, the baseline each copy's slowdown is read against. Every copy
# appends its own JSON line, labelled with its index.
#
# Environment: PYTHON (default python3), REPEATS (default 3), OUT (required).
set -u
input=$1; label=$2; P=$3; T=$4; shift 4
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-python3}
: "${OUT:?set OUT to the results file}"
IFS=';' read -ra lists <<< "$("$PYTHON" "$REPO/tools/parallel/cpu_layout.py" "$P" "$T")"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" JAX_PLATFORMS=cpu DEFUMAT_THREADS=off \
       OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
pids=()
for ((c = 0; c < P; c++)); do
  taskset -c "${lists[$c]}" "$PYTHON" "$REPO/tools/parallel/time_scf.py" \
      "$input" "${REPEATS:-3}" "$label-P$P-T$T-copy$c" "$@" >> "$OUT" 2>> "${OUT%.jsonl}.err" &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
