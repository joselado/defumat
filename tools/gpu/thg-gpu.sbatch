#!/bin/bash
# The third harmonic on one H100: its time against the CPU arm (thg-cpu.sbatch), and its tests.
#
# GPU.md §2.3's metric: card defumat against CPU defumat, the same commit and the same input,
# per k-point and step, compilation its own line (tools/realtime/thg_timing.py). Two-atom
# silicon at 12 Ry, chi^(3)(-3w; w, w, w) at 1.55 eV along [100], 0.2 eV, 450 steps a period:
#
#   4^3,  eta_t 6   18 points,  3600 steps   (the CPU arm runs it too)
#   8^3,  eta_t 6   100 points, 3600 steps   (the CPU arm runs it too)
#   4^3,  eta_t 12  18 points,  6750 steps   (card only: the hierarchy check at 5e-5)
#   12^3, eta_t 6   294 points, 3600 steps   (card only)
#
# each with the frequency-domain hierarchy at the same frequency as the check on the answer,
# each case its own process so that a device peak belongs to one case. Then the real-time
# test files that carry the third order, one pytest process each, after the timings so that
# nothing else is on the card while they are taken.
#
# Not submitted by any tool here (CLAUDE.local.md): by hand, after `sbatch --test-only`.
#
#   sbatch --export=ALL,DEFUMAT_REPO=/scratch/work/ladovj1/apps/defumat-thg tools/gpu/thg-gpu.sbatch
#
# partition   one named H100 partition, so the card is the one asked for (a list gets
#             whichever card frees first, CLAUDE.local.md 2026-10-02); on 2026-10-08 at
#             10:41 `sbatch --test-only` said 12:59 on gpu-h100-80g and 14:59 on its short
#             sibling, whatever the time asked.
# time        04:00:00: the timings are minutes on a data-centre card; the regime tests
#             took about an hour on D22's A2000, which runs float64 at 1/70 of float32.
# mem         32G of host memory; the 8^3 orders hold 2^3 copies of 100 points' states.
#SBATCH --job-name=defumat-thg-gpu
#SBATCH --partition=gpu-h100-80g
#SBATCH --time=04:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-thg/out/gpu-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-thg/out/gpu-%j.err

WORK=/scratch/work/ladovj1/calculations/defumat-thg
REPO=${DEFUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-thg}
PY=/scratch/work/ladovj1/calculations/pypresso-gpu/venv/bin/python3
mkdir -p "$WORK/out" "$WORK/cache-gpu"
cd "$REPO" || exit 1
unset XLA_PYTHON_CLIENT_PREALLOCATE XLA_PYTHON_CLIENT_MEM_FRACTION JAX_PLATFORMS \
      DEFUMAT_K_BATCH DEFUMAT_BAND_BATCH DEFUMAT_MEMORY_MODE
export PYTHONPATH="$REPO" DEFUMAT_THREADS=off OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export DEFUMAT_CACHE_DIR="$WORK/cache-gpu"
OUT="$WORK/out/gpu-${SLURM_JOB_ID}.json"

echo "=== node ==="; hostname; date
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$PY" -c "import jax; print('jax', jax.__version__, jax.devices())"
git -C "$REPO" rev-parse HEAD

for case in "4 6" "8 6" "4 12" "12 6"; do
  set -- $case
  echo; echo "########## grid $1, eta_t $2 ##########"; date
  "$PY" -u tools/realtime/thg_timing.py "$OUT" "$1" "$2" --hierarchy
  echo "exit $?"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
done

echo; echo "=== the real-time tests that carry the third order, on the card ==="
for file in tests/unit/test_realtime.py tests/regression/test_realtime.py \
            tests/regression/test_realtime_orders_fd.py \
            tests/regression/test_realtime_hierarchy.py \
            tests/regression/test_realtime_regimes.py; do
  echo; echo "########## $file ##########"; date
  "$PY" -m pytest -p no:cacheprovider -q -rfEs --durations=10 "$file" \
      > "$WORK/out/gpu-${SLURM_JOB_ID}-$(basename "$file" .py)-$(basename "$(dirname "$file")").log" 2>&1
  echo "exit $?"
  tail -3 "$WORK/out/gpu-${SLURM_JOB_ID}-$(basename "$file" .py)-$(basename "$(dirname "$file")").log"
done
echo; echo "=== done ==="; date
