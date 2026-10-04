#!/bin/bash
# The cobalt film at three vacuum sizes: every mixer, both fits. Iteration counts only.
set -u
cd /l/ladovj1/review/vacuum
export JAX_PLATFORMS=cpu DEFUMAT_CACHE_DIR=/l/ladovj1/jaxcache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DEFUMAT_THREADS=off
PY=/l/ladovj1/venv/bin/python3
mkdir -p out logs
rm -f sweep_co.done
jobs=()
for cell in co-c14 co-c10 co-c6; do
  for mode in ldos anderson local-tf tf; do
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
    # Every arm runs from the prototype's worktree: it differs from master only
    # by the 'ldos' mode, which the other three modes never reach.
    PYTHONPATH=/l/ladovj1/defumat-ldos taskset -c $core $PY run_one.py inputs/$1.in $2 $3 out/$tag.json > logs/$tag.log 2>&1
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
touch sweep_co.done
