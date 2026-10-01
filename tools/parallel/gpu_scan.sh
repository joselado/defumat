#!/bin/bash
# Warm-SCF time per iteration on the card, over a set of batching dials.
#
#   CONFIGS="default:: speed:speed: bb16::16" tools/parallel/gpu_scan.sh
#   CONFIGS="sticks:speed::sticks box:speed::box" tools/parallel/gpu_scan.sh
#
# Each CONFIGS entry is label:memory_mode:band_batch[:fft_layout], an empty
# field leaving that dial to the platform default (``sticks`` for the layout,
# DEFUMAT_FFT_LAYOUT). One fresh process per entry, so the
# device's peak bytes in use (printed by time_scf.py) belongs to that entry.
# The host side is pinned by HOST_CPUS (default 0-3) with DEFUMAT_THREADS=off;
# the card is JAX's default backend, so JAX_PLATFORMS is left unset.
#
# Environment: PYTHON, CELLS, REPEATS, MAX_ITERATIONS and OUT as in
# thread_scan.sh (OUT defaults to gpu-scan.jsonl), and HOST_CPUS.
set -u
if [ -z "${CONFIGS:-}" ]; then
  echo "gpu_scan: set CONFIGS to label:memory_mode:band_batch entries (see the header)" >&2
  exit 2
fi
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-python3}
CELLS=${CELLS:-"si8-1k-ecut30.in si16-1k-ecut30.in"}
REPEATS=${REPEATS:-5}
OUT=${OUT:-gpu-scan.jsonl}
HOST_CPUS=${HOST_CPUS:-0-3}
unset JAX_PLATFORMS
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" DEFUMAT_THREADS=off \
       OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

for cell in $CELLS; do
  for entry in $CONFIGS; do
    IFS=: read -r label mode band layout <<< "$entry"
    if [ -n "$mode" ]; then export DEFUMAT_MEMORY_MODE=$mode; else unset DEFUMAT_MEMORY_MODE; fi
    if [ -n "$band" ]; then export DEFUMAT_BAND_BATCH=$band; else unset DEFUMAT_BAND_BATCH; fi
    if [ -n "$layout" ]; then export DEFUMAT_FFT_LAYOUT=$layout; else unset DEFUMAT_FFT_LAYOUT; fi
    case "$cell" in */*) input=$cell ;; *) input=$REPO/benchmarks/$cell ;; esac
    taskset -c "$HOST_CPUS" "$PYTHON" "$REPO/tools/parallel/time_scf.py" \
        "$input" "$REPEATS" "$label" ${MAX_ITERATIONS:+--max-iterations "$MAX_ITERATIONS"} >> "$OUT" \
      || echo "{\"label\": \"$label\", \"input\": \"$cell\", \"failed\": true}" >> "$OUT"
  done
done
