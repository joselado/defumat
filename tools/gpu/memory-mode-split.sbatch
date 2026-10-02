#!/bin/bash
# GPU-SPEED-NEXT.md section 4b: what memory mode costs on a data-centre card, and which half of it.
#
# On an H200 (job 20644012, PERFORMANCE.md "The stall check on a data-centre card") memory mode took
# 106.3 ms an iteration on 64-atom silicon against speed mode's 54.1, for 4 per cent less device memory,
# where the RTX A2000 showed 8 per cent. With one k-point the batch over k is not what differs, so it is
# the projectors rebuilt at every application or the wavefunctions streamed through host memory. The arms
# swap the two dials one at a time from memory mode (DEFUMAT_PROJECTORS, DEFUMAT_WFC_STORE), with speed
# mode beside them; time_scf.py prints the resolved dials and davidson_steps beside every time, so an arm
# whose steps differ is visible. The 27-point mesh adds memory mode at one k-point against k_batch='fit'
# and speed mode, the user's 'fit' decision measured on a fast card.
#
# Not submitted by any tool here (CLAUDE.local.md): submit it by hand from the checkout the job should run,
# after `sbatch --test-only` has accepted the partition and the limits.
#
#   sbatch --export=ALL,DEFUMAT_REPO=/scratch/work/ladovj1/apps/defumat-stall tools/gpu/memory-mode-split.sbatch
#
# ---- the #SBATCH values, and why each one ----
# partition   the stall check's four card partitions, the H200 first, where the 2x was measured; every
#             arm runs on the one card the job gets, so the comparison is within the job.
# time        00:30:00: the stall check's twelve SCF timings took 2 min 25 s on an H200; this is
#             thirteen, a 27-point mesh among them.
# mem         32G host memory, as the stall check; si64 peaked at 3.3 GiB on the card.
#SBATCH --job-name=defumat-memory-split
#SBATCH --partition=gpu-h200-141g-short,gpu-h100-80g,gpu-h100-80g-short,gpu-a100-80g
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-stall/out/split-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-stall/out/split-%j.err

WORK=/scratch/work/ladovj1/calculations/defumat-stall
REPO=${DEFUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-jobs}
PY=/scratch/work/ladovj1/calculations/pypresso-gpu/venv/bin/python3
mkdir -p "$WORK/out" "$WORK/cache"
cd "$REPO" || exit 1
unset XLA_PYTHON_CLIENT_PREALLOCATE JAX_PLATFORMS DEFUMAT_ETHR_MIN DEFUMAT_K_BATCH DEFUMAT_BAND_BATCH \
      DEFUMAT_PROJECTORS DEFUMAT_WFC_STORE DEFUMAT_MEMORY_MODE
export PYTHONPATH="$REPO" DEFUMAT_THREADS=off OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export DEFUMAT_CACHE_DIR="$WORK/cache"
OUT="$WORK/out/split-${SLURM_JOB_ID}.jsonl"

echo "=== node ==="; hostname; date
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$PY" -c "import jax; print('jax', jax.__version__, jax.devices())"
git -C "$REPO" rev-parse HEAD

arm() {  # label input repeats, then NAME=value settings for this arm only
  local label=$1 input=$2 repeats=$3; shift 3
  env "$@" timeout 600 "$PY" tools/parallel/time_scf.py "$input" "$repeats" "$label" >> "$OUT" \
    || echo "failed $label"
}

echo; echo "=== memory mode with one dial swapped at a time, one k-point, conv_thr 1e-10 ==="
for cell in si64-1k-ecut30 si16-1k-ecut30; do
  arm "$cell-memory" benchmarks/$cell.in 2 DEFUMAT_MEMORY_MODE=memory
  arm "$cell-memory-projstore" benchmarks/$cell.in 2 DEFUMAT_MEMORY_MODE=memory DEFUMAT_PROJECTORS=store
  arm "$cell-memory-wfcdevice" benchmarks/$cell.in 2 DEFUMAT_MEMORY_MODE=memory DEFUMAT_WFC_STORE=device
  arm "$cell-memory-both" benchmarks/$cell.in 2 DEFUMAT_MEMORY_MODE=memory DEFUMAT_PROJECTORS=store \
      DEFUMAT_WFC_STORE=device
  arm "$cell-speed" benchmarks/$cell.in 2 DEFUMAT_MEMORY_MODE=speed
done

echo; echo "=== a 27-point mesh: memory mode at one k-point, at k_batch='fit', and speed mode ==="
arm si8k3-memory benchmarks/si8-ecut20-nosym-k3.in 2 DEFUMAT_MEMORY_MODE=memory
arm si8k3-memory-fit benchmarks/si8-ecut20-nosym-k3.in 2 DEFUMAT_MEMORY_MODE=memory DEFUMAT_K_BATCH=fit
arm si8k3-speed benchmarks/si8-ecut20-nosym-k3.in 2 DEFUMAT_MEMORY_MODE=speed

"$PY" tools/gpu/replay/table.py "$OUT"
echo; echo "=== done ==="; date
