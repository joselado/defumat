#!/bin/bash
# The production LDOS mixer on the cobalt film at the input's own vacuum, alone on
# core 0, twice, the second read; and Kerker beside it as the per-iteration base.
cd /l/ladovj1/review/vacuum
export JAX_PLATFORMS=cpu PYTHONPATH=/l/ladovj1/defumat-ldos DEFUMAT_CACHE_DIR=/l/ladovj1/jaxcache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DEFUMAT_THREADS=off
mkdir -p clean
rm -f clean_prod.done
for arm in "ldos flat" "tf flat"; do
  set -- $arm
  for rep in 1 2; do
    taskset -c 0 /l/ladovj1/venv/bin/python3 run_prod.py inputs/co-c10.in $1 $2 clean/prod-co-c10_$1_$2_r$rep.json > clean/prod-co-c10_$1_$2_r$rep.log 2>&1
  done
done
touch clean_prod.done
