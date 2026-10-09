#!/bin/bash
# usage: run.sh <script.py> [cores] [mem] [args...]; run from /l/ladovj1/review/open-a
cd /l/ladovj1/review/open-a
cores=${2:-0-3}
mem=${3:-6G}
export PYTHONPATH=/l/ladovj1/w-a DEFUMAT_CACHE_DIR=/l/ladovj1/jaxcache MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
systemd-run --user --scope -p MemoryMax=$mem -p MemorySwapMax=0 taskset -c $cores /l/ladovj1/venv/bin/python -u $1 "${@:4}"
echo "EXIT $? $(date)"
