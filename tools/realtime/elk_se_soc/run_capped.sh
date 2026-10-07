#!/bin/bash
# run_capped.sh NAME DIR CMD...   (on D22)
# Runs CMD in DIR on core 0, one thread, inside a capped scope (8G, no swap), with
# the load read before and after; NAME.log holds it, NAME.done the exit status.
name=$1; dir=$2; shift 2
cd "$dir" || exit 1
source /l/ladovj1/venv/bin/activate
export PYTHONPATH=/l/ladovj1/defumat-hspin:/l/ladovj1/review/hspin-elk
export DEFUMAT_REPO=/l/ladovj1/defumat-hspin DEFUMAT_CACHE_DIR=/l/ladovj1/jaxcache
export JAX_PLATFORMS=cpu DEFUMAT_THREADS=off
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_STACKSIZE=64M
ulimit -s unlimited 2>/dev/null || true
load() {
  echo "=== $1 $(date -Is)"; uptime
  ps -eo pid,psr,pcpu,etime,args --sort=-pcpu | head -8 | cut -c1-160
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>&1
}
unit="hspinelk-$name-$$"
{
  load before
  start=$(date +%s.%N)
  systemd-run --user --quiet --unit="$unit" --scope -p MemoryMax=8G -p MemorySwapMax=0 \
      taskset -c 0 "$@"
  rc=$?
  end=$(date +%s.%N)
  echo "=== wall $(echo "$end - $start" | bc) s, exit $rc"
  load after
} > "$name.log" 2>&1
systemctl --user reset-failed "$unit.scope" >/dev/null 2>&1
echo $rc > "$name.done"
