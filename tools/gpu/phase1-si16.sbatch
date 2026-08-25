#!/bin/bash
# GPU.md Phase 1 -- where the time actually goes, on the cells that show it.
#
# Phase 0 measured whole SCF iterations and found 5.44x on `al10-metal`, which
# has ten k-points. That left the question it could not answer: **how much of
# that is the k axis, and does the diagonalisation itself win?** These cells
# answer it, because they are **single k-point** -- there is no k-parallelism
# available, so whatever ratio they show is the per-k path's own.
#
# So this runs two things per case:
#   phase0.py    -- whole SCF iterations, GPU against the CPU twin (the metric)
#   benchmark.py -- the per-stage split, which already breaks out `h_psi` and
#                   `diagonalise (one k, Davidson)` separately. That is the
#                   direct measurement of the eigensolver, and the reason this
#                   job exists rather than another whole-iteration sweep.
#
# Cells: si8-1k-ecut30, si16-1k, si16-1k-ecut30 -- 16 atoms at 5900 plane waves
# and 32 bands is the largest committed benchmark, and `PERFORMANCE.md` measures
# its CPU iteration at 283 ms. The standing rule is that a two-atom cell shows
# none of this; Phase 0 confirmed it in its GPU form (si-1k lost at 0.33x).
#
# ---- #SBATCH ----
# gpus       v100:1 *by name*, not just any GPU. gpu-debug holds both V100 and
#            A100 nodes, and Phase 0 landed on a V100 (dgx8); comparing 16 atoms
#            on an A100 against 8 atoms on a V100 would confound cell size with
#            card generation, which is the whole failure this project's
#            measurement rules exist to prevent.
# time       28 min inside gpu-debug's 30. The runs are ordered decisive-first,
#            so a walltime overrun costs the least informative case rather than
#            the one the job was submitted for.
# cpus/mem   4 and 32G, matching phase0-*.sbatch exactly so the two jobs'
#            numbers sit in one table.
#SBATCH --job-name=pypresso-gpu-phase1
#SBATCH --partition=gpu-debug
#SBATCH --time=00:28:00
#SBATCH --gpus=v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/work/ladovj1/calculations/pypresso-gpu/out/p1gpu-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/pypresso-gpu/out/p1gpu-%j.err

WORK=/scratch/work/ladovj1/calculations/pypresso-gpu
REPO=/scratch/work/ladovj1/apps/pypresso
mkdir -p "$WORK/out" "$WORK/cache-gpu"
cd "$REPO" || exit 1

source "$WORK/venv/bin/activate"
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYPRESSO_CACHE_DIR="$WORK/cache-gpu"
export PYTHONPATH="$REPO"

echo "=== node ==="; hostname; date
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"
git -C "$REPO" rev-parse HEAD

iterations () {  # whole-iteration ratio: phase0.py <case> <k> <b>
  echo; echo "########## scf $1 k$2 b$3 ##########"
  python3 tools/gpu/phase0.py "$1" --k-batch "$2" --band-batch "$3" --repeats 2 \
          --skip-precision-probe \
          --json "$WORK/out/p1gpu-${SLURM_JOB_ID}-$1-k$2-b$3.json"
}

stages () {  # the per-stage split: benchmark.py inherits the dials from the env
  echo; echo "########## stages $1 band_batch=$2 ##########"
  PYPRESSO_K_BATCH=1 PYPRESSO_BAND_BATCH="$2" python3 tools/benchmark.py "benchmarks/$1.in"
}

# Decisive first: the biggest cell, split by stage, at the dial that won.
stages     si16-1k-ecut30 all
stages     si16-1k-ecut30 1
iterations si16-1k-ecut30 1 all
iterations si16-1k-ecut30 1 1

stages     si16-1k        all
iterations si16-1k        1 all
iterations si16-1k        1 1

stages     si8-1k-ecut30  all
iterations si8-1k-ecut30  1 all
iterations si8-1k-ecut30  1 1

echo; echo "=== done ==="; date
