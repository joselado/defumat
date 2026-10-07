#!/bin/bash
# Replaces queue3.sh after the half-step magnet run it started: waits for that run (PID given),
# then the dataset sweep again (fixed script), then the selenium parametrisation at 520 steps a
# period, on the card, capped. all3.done marks the end, which the Elk timing waits for.
out=/l/ladovj1/review/hspin/svo12
while kill -0 $1 2>/dev/null; do sleep 20; done
cd /l/ladovj1/defumat-hspin
source /l/ladovj1/venv/bin/activate
export PYTHONPATH=/l/ladovj1/defumat-hspin:$out DEFUMAT_REPO=/l/ladovj1/defumat-hspin
export DEFUMAT_CACHE_DIR=/l/ladovj1/jaxcache MKL_NUM_THREADS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export OMP_NUM_THREADS=4 DEFUMAT_THREADS=off
unset JAX_PLATFORMS
run() {  # tag command...
  local tag=$1; shift
  local unit="svo12-$tag-$$"
  ( date; uptime; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
    time systemd-run --user --quiet --unit="$unit" -p MemoryMax=10G -p MemorySwapMax=0 --scope \
         taskset -c 0-3 "$@"; uptime ) > $out/$tag.log 2>&1
  echo $? > $out/$tag.done
  systemctl --user reset-failed "$unit.scope" >/dev/null 2>&1
}
for e in 20 30 40; do
  for kind in us paw; do
    run datasets-$kind-$e python3 -u $out/datasets.py $kind $e $((8 * e))
  done
done
test=tests/regression/test_realtime_regimes.py::test_the_spectrum_is_the_propagation_at_one_frequency
git status --short > $out/state3.txt
run se2 python3 -u -m pytest -q -s -p no:cacheprovider -p svo_report "$test[se]"
touch $out/all3.done
