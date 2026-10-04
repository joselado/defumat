#!/bin/bash
# The inner tolerance at 1e-3 on the cells the GMRES prototype was sensitive on.
cd /l/ladovj1/review/vacuum
export JAX_PLATFORMS=cpu PYTHONPATH=/l/ladovj1/defumat-ldos DEFUMAT_CACHE_DIR=/l/ladovj1/jaxcache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DEFUMAT_THREADS=off LDOS_TOL=1e-3
rm -f sweep_t3.done
i=0
for job in "nbse2-v64 flat" "nbse2-v64 ddot" "nbse2-v28 flat" "co-c6 flat" "graphene-v60 flat" "al-v32 flat"; do
  set -- $job
  core=$((2 * (i % 6))); i=$((i + 1))
  taskset -c $core /l/ladovj1/venv/bin/python3 run_prod.py inputs/$1.in ldos $2 prod/$1_ldosT3_$2.json > prod-logs/$1_ldosT3_$2.log 2>&1 &
done
wait
touch sweep_t3.done
