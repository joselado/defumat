#!/bin/bash
# NiI2 spiral, calibration: what does one SCF iteration cost on the card, and at
# which k_batch does the working set stop fitting?
#
# This job produces no physics. It runs the *production* input -- 81 k-points,
# 64 spinor bands, ecutwfc = 80, a 45 x 45 x 270 grid -- for a fixed ten
# iterations at three settings of k_batch, and reports seconds per iteration and
# peak HBM for each. The gradient is skipped (--no-gradient): dE/dq at a state
# stopped after ten iterations is not a number.
#
# Why it exists rather than going straight to the array: the sweep's --time and
# --k-batch are both guesses until one q has run on this hardware. GPU.md's rule
# is that a dial is set by hand in a measurement and the platform default is not
# inherited, and its own numbers stop at ten silicon atoms on a single k-point --
# nothing there covers 81 k-points of two-component spinors.
#
# ---- the #SBATCH values ----
#
# partition   gpu-h200-141g-short (2-day limit; this asks for one hour). 141 GB
#             is the point: 81 x 64 x 26315 x 2 complex128 is 4.4 GB of
#             wavefunctions before Davidson's subspace, and one real-space box is
#             (45 x 45 x 270) x 16 B = 8.7 MB, so a fully batched h_psi at
#             k_batch = 81 would be ~90 GB of transforms alone.
# time        01:00:00 -- three ten-iteration runs plus compile.
# cpus-per-task 4, matching the existing defumat GPU jobs; XLA pinned to it.
# mem         160G host.
#SBATCH --job-name=nii2-calib
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --time=01:00:00
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --output=/scratch/work/ladovj1/calculations/nii2-spiral/out/calib-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/nii2-spiral/out/calib-%j.err

WORK=/scratch/work/ladovj1/calculations/nii2-spiral
REPO=/scratch/work/ladovj1/apps/defumat
mkdir -p "$WORK/out" "$WORK/results" "$WORK/cache-gpu"
cd "$REPO" || exit 1
source /scratch/work/ladovj1/calculations/defumat-gpu/venv/bin/activate

export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export DEFUMAT_CACHE_DIR="$WORK/cache-gpu"
export PYTHONPATH="$REPO"

echo "=== node ==="; hostname; date
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"
git -C "$REPO" rev-parse HEAD

for kb in 4 8 16; do
  echo; echo "########## k_batch = $kb ##########"
  timeout 900 python3 "$REPO/calculations/nii2-spiral/run_spiral.py" \
      "$REPO/calculations/nii2-spiral/nii2-spiral.in" \
      --pseudo-dir "$REPO/tests/data/pseudo" \
      --path GM --npoints 11 --q-index 0 \
      --k-batch "$kb" --max-iterations 10 --conv-thr 1e-12 --no-gradient \
      --json "$WORK/results/calib-k${kb}-${SLURM_JOB_ID}.json"
  echo "exit $?"
done

echo; echo "=== done ==="; date
