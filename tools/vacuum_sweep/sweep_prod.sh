#!/bin/bash
# The production LDOS mixer on D22: the sweep again, the inner tolerance, the
# Gaussian's minimum width, the G layout and a tetrahedron run. Iteration counts.
set -u
cd /l/ladovj1/review/vacuum
export JAX_PLATFORMS=cpu PYTHONPATH=/l/ladovj1/defumat-ldos DEFUMAT_CACHE_DIR=/l/ladovj1/jaxcache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DEFUMAT_THREADS=off
PY=/l/ladovj1/venv/bin/python3
mkdir -p prod prod-logs
rm -f sweep_prod.done
jobs=()
# tag cell mode fit env
for cell in co-c14 co-c10 co-c6 nbse2-v64 nbse2-v48 nbse2-v28 nbse2-v16 graphene-v60 graphene-v40 graphene-v20 hbn-v60 hbn-v40 hbn-v20 al-v64 al-v48 al-v32 al-v16; do
  for fit in flat ddot; do jobs+=("P $cell ldos $fit -"); done
done
for cell in co-c14 co-c10 al-v64; do jobs+=("T3 $cell ldos flat LDOS_TOL=1e-3"); done
jobs+=("S01 co-c10 ldos flat LDOS_SIGMA_MIN=0.01")
jobs+=("S02 co-c10 ldos flat LDOS_SIGMA_MIN=0.02")
jobs+=("G co-c10 ldos flat MIXING_SPACE=g")
jobs+=("G al-v64 ldos flat MIXING_SPACE=g")
jobs+=("TET al-v16-tetra ldos flat -")
jobs+=("TET al-v16-tetra anderson flat -")
worker() {
  local core=$1; shift
  for job in "$@"; do
    set -- $job
    local out=prod/$2_$3$1_$4.json
    [ -f $out ] && continue
    if [ "$5" = "-" ]; then
      taskset -c $core $PY run_prod.py inputs/$2.in $3 $4 $out > prod-logs/$2_$3$1_$4.log 2>&1
    else
      env $5 taskset -c $core $PY run_prod.py inputs/$2.in $3 $4 $out > prod-logs/$2_$3$1_$4.log 2>&1
    fi
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
touch sweep_prod.done
