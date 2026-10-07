#!/bin/bash
# The dataset sweep's hierarchy called twice in one process at 30 Ry, ultrasoft then PAW, on the
# card, capped: whether the ultrasoft run's extra minute is compilation. twice.done marks the end.
out=/l/ladovj1/review/hspin/svo12
cd /l/ladovj1/defumat-hspin
source /l/ladovj1/venv/bin/activate
export PYTHONPATH=/l/ladovj1/defumat-hspin DEFUMAT_REPO=/l/ladovj1/defumat-hspin
export DEFUMAT_CACHE_DIR=/l/ladovj1/jaxcache MKL_NUM_THREADS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export OMP_NUM_THREADS=4 DEFUMAT_THREADS=off
unset JAX_PLATFORMS
rm -f $out/twice.done
for kind in us paw; do
  ( date; uptime; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
    time systemd-run --user --quiet --unit="svo12-twice-$kind-$$" -p MemoryMax=10G -p MemorySwapMax=0 \
         --scope taskset -c 0-3 python3 -u $out/datasets.py $kind 30 240 2
    uptime; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader ) > $out/twice-$kind.log 2>&1
  systemctl --user reset-failed "svo12-twice-$kind-$$.scope" >/dev/null 2>&1
done
touch $out/twice.done
