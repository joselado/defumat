#!/bin/bash
# The cobalt film (the input's own vacuum), one arm at a time, alone on core 0, cache warm
# from the sweep. Each arm is run twice and the second is the one read.
cd /l/ladovj1/review/vacuum
export JAX_PLATFORMS=cpu PYTHONPATH=/l/ladovj1/defumat-ldos DEFUMAT_CACHE_DIR=/l/ladovj1/jaxcache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DEFUMAT_THREADS=off
mkdir -p clean
rm -f clean.done
for arm in "ldos flat" "local-tf flat" "local-tf ddot" "tf flat" "anderson ddot"; do
  set -- $arm
  for rep in 1 2; do
    taskset -c 0 /l/ladovj1/venv/bin/python3 run_one.py inputs/co-c10.in $1 $2 clean/co-c10_$1_$2_r$rep.json > clean/co-c10_$1_$2_r$rep.log 2>&1
  done
done
touch clean.done
