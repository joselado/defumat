#!/bin/bash
# OPEN.md Part VII item 1, step 0: the Davidson executable of the 45-atom NiBr2 slab, compiled
# and not run, on an A100 (plans/open-d-vii1-review.md, "A17's 11 GB card offset has a cheaper
# test"). The H100 twin is vii1-compile-h100.sbatch; the two differ in the #SBATCH lines only.
#
# What it measures. tools/gpu/davidson_memory.py builds the slab's Calculation, lowers the
# compiled _every_k at the run's shapes and reads memory_analysis(): temp, arguments, outputs.
# Nothing the size of the subspace is allocated. Two points, each its own process: the device
# store (the September runs' and the long job's store, whole-set lax.map at k_batch 1), then the
# streamed store (one k-point chunk, the memory-mode default on a card), so the unit the long job
# runs comes first and is in the JSON even if the second one meets the 30-minute limit.
#
# Prediction, written before it runs. September's runtime requests at these dials were 23.18
# GiB on the A100 and 33.44 GiB on the H100 (MEMORY-AUDIT.md A16); their difference, 10.26 GiB
# = 11.02 GB, is A17's card offset to two digits. The fit's second coefficient has fallen from
# 6.20 to 4.20 since (sizing.py), so:
#   temp on the A100 about 19.4 GiB (23.18 - 3.8), device store;
#   temp on the H100 about 29.6 GiB (33.44 - 3.8);
#   H100 minus A100 about 10.3 GiB if the offset is the executable's, about 0 if it is not
#   (then it is outside the unit and the long job's brackets have to find it);
#   the streamed point within a few per cent of the device point, since both solve one k-point
#   at a time.
#
# The slab: NiBr2_defumat/nibr2_k161_anderson.scf.in (45 atoms, 1 6 1, nosym, nbnd 403,
# npwx 156346, fully-relativistic PAW Ni and Br from NiBr2_defumat/pseudo), the input of jobs
# 20257133, 20258175 and 20258183. Dials as there: david 2, band batch 16, k_batch 1, memory
# mode pinned to memory. XLA autotuning at level 0 as nibr2.sbatch ran it in September, so the
# compile is comparable with those requests.
#
# Submitted by hand only, after `sbatch --test-only` (CLAUDE.local.md):
#   sbatch /scratch/work/ladovj1/calculations/defumat-vii1/vii1-compile-a100.sbatch
#
# partition   gpu-debug, whose A100 nodes are asked for by --gpus=a100:1 (the partition also
#             holds V100s); 30 minutes is its limit, one job per user at a time.
# mem         240G of host memory, as every September job on this cell asked.
#SBATCH --job-name=vii1-compile-a100
#SBATCH --partition=gpu-debug
#SBATCH --time=00:30:00
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-vii1/out/%x-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-vii1/out/%x-%j.err

WORK=/scratch/work/ladovj1/calculations/defumat-vii1
SLAB=/scratch/work/ladovj1/calculations/NiBr2_defumat
REPO=/scratch/work/ladovj1/apps/defumat-vii1
PY=/scratch/work/ladovj1/calculations/pypresso-gpu/venv/bin/python3
INPUT=$SLAB/nibr2_k161_anderson.scf.in
PSEUDO=$SLAB/pseudo
mkdir -p "$WORK/out" "$WORK/cache-gpu"
cd "$WORK" || exit 1

# Every DEFUMAT_* dial off, so the script's flags are the only dials (DEFUMAT_BAND_BATCH is
# set below by the tool itself, from --band-batch, which is how it reaches the compile).
for name in $(compgen -e | grep '^DEFUMAT_'); do unset "$name"; done
unset JAX_PLATFORMS XLA_PYTHON_CLIENT_PREALLOCATE
export DEFUMAT_CACHE_DIR="$WORK/cache-gpu"
export PYTHONPATH="$REPO" PYTHONWARNINGS=always OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.97
export XLA_FLAGS="--xla_gpu_autotune_level=0"

echo "=== node ==="; hostname; date
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$PY" -c "import jax, sys; print('jax', jax.__version__, jax.devices()); sys.exit(0 if jax.devices()[0].platform == 'gpu' else 4)" || exit 4
git -C "$REPO" rev-parse HEAD
git -C "$REPO" status --porcelain --untracked-files=no
env | grep -E '^(DEFUMAT_|XLA_|JAX_|PYTHONWARNINGS)' | sort

for store in device stream; do
  echo; echo "########## wfc_store $store ##########"; date
  "$PY" -u "$REPO/tools/gpu/davidson_memory.py" "$INPUT" --pseudo-dir "$PSEUDO" \
      --david 2 --band-batch 16 --nbnd 403 --k-batch 1 \
      --memory-mode memory --wfc-store "$store" \
      --json "$WORK/out/$SLURM_JOB_NAME-$SLURM_JOB_ID-$store.json"
  echo "exit $?"; date
done
echo "=== done ==="
