#!/bin/bash
# One job at a time, each in a memory-capped scope with no swap: the hierarchy against the
# real-time orders at eta_t = 12 on the card (mag, us, paw), the fast gate on the CPU, then se on
# the card. <tag>.done holds the exit status, all.done marks the end.
out=/l/ladovj1/review/hspin/svo12
cd /l/ladovj1/defumat-hspin
source /l/ladovj1/venv/bin/activate
export PYTHONPATH=/l/ladovj1/defumat-hspin:$out DEFUMAT_REPO=/l/ladovj1/defumat-hspin
export DEFUMAT_CACHE_DIR=/l/ladovj1/jaxcache MKL_NUM_THREADS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
rm -f $out/all.done
git log --oneline -1 > $out/state.txt; git status --short >> $out/state.txt
run() {  # tag platform cores mem command...
  local tag=$1 platform=$2 cores=$3 mem=$4; shift 4
  if [ "$platform" = cpu ]; then export JAX_PLATFORMS=cpu; else unset JAX_PLATFORMS; fi
  local unit="svo12-$tag-$$"
  ( date; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
    time systemd-run --user --quiet --unit="$unit" -p MemoryMax=$mem -p MemorySwapMax=0 \
         --scope taskset -c $cores "$@" ) > $out/$tag.log 2>&1
  echo $? > $out/$tag.done
  systemctl --user reset-failed "$unit.scope" >/dev/null 2>&1
}
test=tests/regression/test_realtime_regimes.py::test_the_spectrum_is_the_propagation_at_one_frequency
for v in mag us paw; do
  OMP_NUM_THREADS=4 run $v card 0-3 10G python3 -u -m pytest -q -s -p no:cacheprovider \
      -p svo_report "$test[$v]"
done
OMP_NUM_THREADS=4 DEFUMAT_TEST_MEM_MAX=11G run gate cpu 0-11 12G tools/test-fast.sh -q -p no:cacheprovider
OMP_NUM_THREADS=4 run se card 0-3 10G python3 -u -m pytest -q -s -p no:cacheprovider \
    -p svo_report "$test[se]"
touch $out/all.done
