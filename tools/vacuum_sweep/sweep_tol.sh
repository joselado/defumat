#!/bin/bash
# The LDOS arm at DFTK's inner tolerance, 1e-2, on every cell that has an LDOS.
set -u
cd /l/ladovj1/review/vacuum
export JAX_PLATFORMS=cpu PYTHONPATH=/l/ladovj1/defumat-ldos DEFUMAT_CACHE_DIR=/l/ladovj1/jaxcache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DEFUMAT_THREADS=off LDOS_TOL=1e-2
PY=/l/ladovj1/venv/bin/python3
rm -f sweep_tol.done
jobs=()
for cell in co-c14 co-c10 co-c6 nbse2-v64 nbse2-v28 graphene-v60 al-v64 al-v48 al-v32 al-v16; do
  for fit in flat ddot; do jobs+=("$cell $fit"); done
done
worker() {
  local core=$1; shift
  for job in "$@"; do
    set -- $job
    [ -f out/$1_ldos2_$2.json ] && continue
    taskset -c $core $PY run_one.py inputs/$1.in ldos $2 out/$1_ldos2_$2.json > logs/$1_ldos2_$2.log 2>&1
  done
}
cores=(0 2 4 6 8 10)
n=${#cores[@]}
for w in $(seq 0 $((n-1))); do
  mine=()
  for i in "${!jobs[@]}"; do [ $((i % n)) -eq $w ] && mine+=("${jobs[$i]}"); done
  worker ${cores[$w]} "${mine[@]}" &
done
wait
touch sweep_tol.done
