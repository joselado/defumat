#!/bin/bash
# The vacuum sweep on D22: every cell x every mixer x both fits, six workers,
# each pinned to one performance core. Iteration counts only; nothing is timed.
set -u
cd /l/ladovj1/review/vacuum
export JAX_PLATFORMS=cpu PYTHONPATH=/l/ladovj1/defumat DEFUMAT_CACHE_DIR=/l/ladovj1/jaxcache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DEFUMAT_THREADS=off
PY=/l/ladovj1/venv/bin/python3
mkdir -p out logs
rm -f sweep.done
# Heaviest first, so the round-robin split is roughly balanced.
CELLS="nbse2-v64 nbse2-v48 nbse2-v28 nbse2-v16 graphene-v60 graphene-v40 hbn-v60 graphene-v20 hbn-v40 hbn-v20 al-v64 al-v48 al-v32 al-v16"
jobs=()
for cell in $CELLS; do
  for mode in anderson tf local-tf; do
    for fit in flat ddot; do
      jobs+=("$cell $mode $fit")
    done
  done
done
worker() {
  local core=$1; shift
  for job in "$@"; do
    set -- $job
    local tag="$1_$2_$3"
    [ -f out/$tag.json ] && continue
    taskset -c $core $PY run_one.py inputs/$1.in $2 $3 out/$tag.json > logs/$tag.log 2>&1
  done
}
cores=(0 2 4 6 8 10)
n=${#cores[@]}
for w in $(seq 0 $((n-1))); do
  mine=()
  for i in "${!jobs[@]}"; do
    [ $((i % n)) -eq $w ] && mine+=("${jobs[$i]}")
  done
  worker ${cores[$w]} "${mine[@]}" &
done
wait
touch sweep.done
