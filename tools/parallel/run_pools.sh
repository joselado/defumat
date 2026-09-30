#!/bin/bash
# One SCF as P k-point pools of T cores each, every pool pinned to its own cores.
#
#   OUT=results.jsonl tools/parallel/run_pools.sh <input> <label> <P> <T> [pool_time.py options]
#
# The CPU lists come from cpu_layout.py (one CPU per physical core, consecutive
# in cache order), the pools meet at a localhost coordinator on a random port,
# and DEFUMAT_THREADS=off leaves each pool's XLA thread pool exactly its mask.
# P = 1 runs one process with no pools at all, the single-process reference.
# Rank 0 appends one JSON line to $OUT; the others' stderr goes to ${OUT%.jsonl}.err.
#
# Environment: PYTHON (default python3), REPEATS (default 2), OUT (required),
# DEFUMAT_CACHE_DIR is left to the caller.
set -u
input=$1; label=$2; P=$3; T=$4; shift 4
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-python3}
: "${OUT:?set OUT to the results file}"
IFS=';' read -ra lists <<< "$("$PYTHON" "$REPO/tools/parallel/cpu_layout.py" "$P" "$T")"
port=$((20000 + RANDOM % 20000))
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" JAX_PLATFORMS=cpu DEFUMAT_THREADS=off \
       OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
pids=()
for ((r = 0; r < P; r++)); do
  if [ "$P" -gt 1 ]; then
    pool=(DEFUMAT_POOLS=$P DEFUMAT_POOL_RANK=$r DEFUMAT_COORDINATOR=localhost:$port)
  else
    pool=()
  fi
  env "${pool[@]}" taskset -c "${lists[$r]}" "$PYTHON" "$REPO/tools/parallel/pool_time.py" \
      "$input" "${REPEATS:-2}" "$label" "$@" >> "$OUT" 2>> "${OUT%.jsonl}.err" &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [ $status -ne 0 ]; then
  echo "{\"label\": \"$label\", \"input\": \"$(basename "$input")\", \"pools\": $P, \"width\": $T, \"failed\": true}" >> "$OUT"
fi
