#!/bin/bash
# The third harmonic on one H100 against the number of atoms in the cell.
#
# The same driver as thg-gpu.sbatch (tools/realtime/thg_timing.py), on the silicon
# supercells of benchmarks/ at the same 12 Ry, so the step (450 a period, dt = 0.245)
# is inside the propagator's bound for every one: the 2-atom fcc cell, the 8-atom
# cubic cell and its 1x1xN stacks of 16, 32, 64 and 128 atoms. Each cell's ground
# state is its own input's (Gamma only, the 2-atom cell 4^3), and every propagation
# runs on the field's wedge of one 2x2x1 mesh, so the number of k-points stays a few
# and what grows is the cell: what is read is the time per point and step, and the
# time of the whole call. The hierarchy runs on each as the check, written after the
# timing so that a check that refuses keeps it.
#
# Not submitted by any tool here (CLAUDE.local.md): by hand, after `sbatch --test-only`.
#
#   sbatch --export=ALL,DEFUMAT_REPO=/scratch/work/ladovj1/apps/defumat-thg tools/gpu/thg-cells-gpu.sbatch
#
# The run length is per cell: eta_t 6 (3600 steps) up to 16 atoms, where the check
# against the hierarchy means something (8e-3 at 6 on the 2-atom cell), and eta_t 1
# (600 steps) from 32 atoms, a timing only, since on D22's A2000 the 32-atom cell took
# over 50 minutes at eta_t 0.5 where the 8-atom one took 3 (37.6 ms a point and step).
# The cells run smallest first and each record is written as it finishes, so the
# time limit can only cost the largest.
#SBATCH --job-name=defumat-thg-cells
#SBATCH --partition=gpu-h100-80g
#SBATCH --time=08:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-thg/out/cells-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-thg/out/cells-%j.err

WORK=/scratch/work/ladovj1/calculations/defumat-thg
REPO=${DEFUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-thg}
PY=/scratch/work/ladovj1/calculations/pypresso-gpu/venv/bin/python3
mkdir -p "$WORK/out" "$WORK/cache-gpu"
cd "$REPO" || exit 1
unset XLA_PYTHON_CLIENT_PREALLOCATE XLA_PYTHON_CLIENT_MEM_FRACTION JAX_PLATFORMS \
      DEFUMAT_K_BATCH DEFUMAT_BAND_BATCH DEFUMAT_MEMORY_MODE
export PYTHONPATH="$REPO" DEFUMAT_THREADS=off OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export DEFUMAT_CACHE_DIR="$WORK/cache-gpu"
OUT="$WORK/out/cells-${SLURM_JOB_ID}.json"

echo "=== node ==="; hostname; date
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
git -C "$REPO" rev-parse HEAD

for case in "tests/data/qe/si2-symmetric.in 6" "benchmarks/si8-1k.in 6" "benchmarks/si16-1k.in 6" \
            "benchmarks/si32-1k.in 1" "benchmarks/si64-1k.in 1" "benchmarks/si128-1k.in 1"; do
  set -- $case
  echo; echo "########## $1, 2x2x1, eta_t $2 ##########"; date
  "$PY" -u tools/realtime/thg_timing.py "$OUT" 2 "$2" --input "$1" --grid3 2 2 1 \
      --hierarchy 2>&1 | grep --line-buffered -v '^{'
  echo "exit ${PIPESTATUS[0]}"
done
echo; echo "=== done ==="; date
