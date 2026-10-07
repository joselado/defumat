#!/bin/bash
# Waits for svo_queue.sh's all.done, then runs one job at a time, each in a memory-capped scope:
# the magnet as spinors through the real-time orders with the potential updated (card), the
# ultrasoft spin-orbit Kubo identity (card), the dataset sweep (card), and the cost of a step of
# the orders on one core (CPU). <tag>.done holds each exit status, all2.done marks the end.
out=/l/ladovj1/review/hspin/svo12
handoff=/l/ladovj1/review/hspin/handoff
until [ -f $out/all.done ]; do sleep 30; done
cd /l/ladovj1/defumat-hspin
source /l/ladovj1/venv/bin/activate
export PYTHONPATH=/l/ladovj1/defumat-hspin DEFUMAT_REPO=/l/ladovj1/defumat-hspin
export DEFUMAT_CACHE_DIR=/l/ladovj1/jaxcache MKL_NUM_THREADS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
run() {  # tag platform cores threads mem command...
  local tag=$1 platform=$2 cores=$3 threads=$4 mem=$5; shift 5
  if [ "$platform" = cpu ]; then export JAX_PLATFORMS=cpu; else unset JAX_PLATFORMS; fi
  local unit="svo12-$tag-$$"
  ( date; uptime; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
    OMP_NUM_THREADS=$threads DEFUMAT_THREADS=off \
    time systemd-run --user --quiet --unit="$unit" -p MemoryMax=$mem -p MemorySwapMax=0 \
         --scope taskset -c $cores "$@"; uptime ) > $out/$tag.log 2>&1
  echo $? > $out/$tag.done
  systemctl --user reset-failed "$unit.scope" >/dev/null 2>&1
}
rm -f $out/all2.done
run disguise-rt card 0-3 4 10G python3 -u $handoff/disguise.py rt
run kubo-us-soc card 0-3 4 10G python3 -u $handoff/kubo_identity.py alas-epsilon-us-soc 10 40
for e in 20 30 40; do
  for kind in us paw; do
    run datasets-$kind-$e card 0-3 4 10G python3 -u $out/datasets.py $kind $e $((8 * e))
  done
done
for order in 1 3; do
  for kind in nc us paw; do
    run cost-$kind-$order cpu 0 1 8G python3 -u $handoff/orders_cost.py $kind $order
  done
done
touch $out/all2.done
