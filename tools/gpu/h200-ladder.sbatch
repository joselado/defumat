#!/bin/bash
# 64 and 128 atoms on one H200, against Quantum ESPRESSO on one CPU core.
#
# **The baseline is the softest one available and the table must say so.**
# GPU.md §2.3 forbids quoting a GPU against single-core QE by default, because
# the ratio then contains the hardware difference rather than the code's. It is
# run because it was asked for; the CPU half (h200-ladder-cpu.sbatch) also runs
# pw.x at 4, 16 and 32 cores so the single-core column can be read next to what
# someone would actually launch.
#
# Both cells on ONE card, unlike the earlier 64-atom runs which landed on an
# H200 at ecutwfc=12 and a V100-16GB at 30 -- so those two numbers differed by
# card as much as by cutoff and could not be put in one column. This job fixes
# that: same H200, all four cases.
#
# 128 atoms at ecutwfc=30 is 256 bands on a 36x36x576 box, ~3.1 GB per
# band-batched real-space array, so a handful of them plus the Davidson
# subspace should sit inside 141 GB with room to spare -- but that is an
# estimate and the driver's check 4 is what settles it.
#SBATCH --job-name=pypresso-h200-ladder
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --time=03:00:00
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --output=/scratch/work/ladovj1/calculations/pypresso-gpu/out/h200lad-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/pypresso-gpu/out/h200lad-%j.err

WORK=/scratch/work/ladovj1/calculations/pypresso-gpu
REPO=/scratch/work/ladovj1/apps/pypresso
mkdir -p "$WORK/out" "$WORK/cache-gpu"
cd "$REPO" || exit 1
source "$WORK/venv/bin/activate"
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYPRESSO_CACHE_DIR="$WORK/cache-gpu"
export PYTHONPATH="$REPO"

echo "=== node ==="; hostname; date
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"
git -C "$REPO" rev-parse HEAD

# Production cutoff only: ecutwfc = 30 is the one worth comparing, and 12 Ry
# on these cells is small enough that fixed overheads move the ratio.
# 64 atoms first, because 128 at 30 Ry is the case that might not fit.
for case in si64-1k-ecut30 si128-1k-ecut30; do
  echo; echo "########## $case ##########"
  python3 tools/gpu/phase0.py $case --k-batch 1 --band-batch all --repeats 2 \
          --skip-precision-probe \
          --json "$WORK/out/h200lad-${SLURM_JOB_ID}-$case.json"
  echo "exit $?"
done
echo; echo "=== done ==="; date
