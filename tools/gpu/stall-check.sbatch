#!/bin/bash
# GPU-SPEED-NEXT.md item 1: the card's Davidson stall, and the float32 band tier, on a production card.
#
# Everything about the stall and its fix was measured on one float32 card, D22's RTX A2000
# (float64 at 1/70 of float32). Three questions only a data-centre card answers:
#
#   1. Does this card's dense eigh follow the norm of the matrix it is handed, as the A2000's
#      does (3.4e-12 median at a parked value of 1000x the diagonal, against 2.9e-15 for the
#      host's LAPACK)? tools/gpu/replay/eigh_bound.py on the 72 captured solves answers it with
#      no SCF: copy D22's /l/ladovj1/review/eigh_parked.npz to $WORK first. Skipped without it.
#   2. Do the SCF step counts stay equal across the dials at the committed parking (one above a
#      bound of the reduced live block)? time_scf.py prints davidson_steps beside every time.
#   3. What is a float32 band side worth per iteration on a card whose float64 is fast? The
#      A2000 gave 4.2x to 4.6x on 64 atoms at conv_thr 1e-7 (PERFORMANCE.md, "The band side in
#      single precision"); GPU.md Phase 3 ranks the tier by this number.
#
# Not submitted by any tool here (CLAUDE.local.md): submit it by hand, from the checkout the
# job should run, after `sbatch --test-only` has accepted the partition and the limits.
#
#   sbatch --export=ALL,DEFUMAT_REPO=/scratch/work/ladovj1/apps/defumat-stall tools/gpu/stall-check.sbatch
#
# ---- the #SBATCH values, and why each one ----
# partition   any of the four card partitions a committed script here has used; the job prints
#             the card it got, and a second card is a second submission. Verify with
#             `sbatch --test-only` rather than inheriting this reading.
# time        01:00:00: the eigh table is seconds, si64 at conv_thr 1e-10 is about a minute an
#             arm on the A2000, and there are eight arms plus one profile.
# mem         32G host memory; si64 peaked at 3.3 GiB on the card in speed mode.
#SBATCH --job-name=defumat-stall-check
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g,gpu-h100-80g-short,gpu-h200-141g-short
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-stall/out/stall-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-stall/out/stall-%j.err

WORK=/scratch/work/ladovj1/calculations/defumat-stall
REPO=${DEFUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-jobs}
PY=/scratch/work/ladovj1/calculations/pypresso-gpu/venv/bin/python3
mkdir -p "$WORK/out" "$WORK/cache"
cd "$REPO" || exit 1
unset XLA_PYTHON_CLIENT_PREALLOCATE JAX_PLATFORMS DEFUMAT_ETHR_MIN DEFUMAT_K_BATCH DEFUMAT_BAND_BATCH
export PYTHONPATH="$REPO" DEFUMAT_THREADS=off OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export DEFUMAT_CACHE_DIR="$WORK/cache"
OUT="$WORK/out/stall-${SLURM_JOB_ID}.jsonl"

echo "=== node ==="; hostname; date
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$PY" -c "import jax; print('jax', jax.__version__, jax.devices())"
git -C "$REPO" rev-parse HEAD

echo; echo "=== 1. the device eigh on the A2000's captured solves ==="
if [ -f "$WORK/eigh_parked.npz" ]; then
  OUT="$WORK/eigh_parked" timeout 900 "$PY" tools/gpu/replay/eigh_bound.py
else
  echo "skipped: copy D22's /l/ladovj1/review/eigh_parked.npz to $WORK first"
fi

echo; echo "=== 2. steps and times at the committed parking, conv_thr 1e-10 ==="
for mode in memory speed; do
  for cell in si16-1k-ecut30 si64-1k-ecut30; do
    DEFUMAT_MEMORY_MODE=$mode timeout 900 "$PY" tools/parallel/time_scf.py \
        benchmarks/$cell.in 2 $cell-$mode-double >> "$OUT" || echo "failed $cell $mode"
  done
done

echo; echo "=== 3. the float32 band side, conv_thr 1e-7, against double at the same threshold ==="
for mode in memory speed; do
  for prec in double single; do
    DEFUMAT_MEMORY_MODE=$mode DEFUMAT_BAND_PRECISION=$prec timeout 900 "$PY" tools/parallel/time_scf.py \
        benchmarks/si64-1k-ecut30.in 2 si64-$mode-$prec-1e-7 --conv-thr 1e-7 >> "$OUT" || echo "failed $mode $prec"
  done
done
"$PY" tools/gpu/replay/table.py "$OUT"

echo; echo "=== 4. where the kernel time goes, si64 memory mode, if nsys is here ==="
if command -v nsys > /dev/null; then
  nsys profile -t cuda -o "$WORK/out/stall-${SLURM_JOB_ID}" -f true --stats=false \
      "$PY" tools/parallel/time_scf.py benchmarks/si64-1k-ecut30.in 1 si64-nsys > /dev/null
  nsys stats -r cuda_gpu_kern_sum -f csv -o "$WORK/out/stall-${SLURM_JOB_ID}" "$WORK/out/stall-${SLURM_JOB_ID}.nsys-rep"
  "$PY" tools/gpu/replay/kern_cats.py "$WORK/out/stall-${SLURM_JOB_ID}_cuda_gpu_kern_sum.csv"
else
  echo "no nsys on this node: skipped"
fi
echo; echo "=== done ==="; date
