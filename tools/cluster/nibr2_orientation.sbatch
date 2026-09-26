#!/bin/bash
# Which plane the NiBr2 helix lies in (P122, `ORIENTATION-NEXT.md` step 5):
# `tools/cluster/nibr2_orientation.py`, Route A from two starts and Route C from
# one, on the three-cell supercell with the coupling, one GPU each.
#
# ---- the #SBATCH values ----
#
# partition     the NiBr2 ultracell project's GPU set (`uc_gpu.sbatch`), whose
#               own note puts one iteration of this nine-atom supercell at a
#               minute or two on one GPU.
# array         0-2: Route A from the plane tilted 0.5 rad and 80 degrees off
#               its seed, and Route C from the first. Each Route A task converges
#               its own source at soc_scale = 0 in its own directory, because two
#               jobs checkpointing one directory land on each other's state.
# time          12:00:00, the project's cap; the run checkpoints every
#               iteration, so a job killed at the wall resumes from its
#               directory when resubmitted identically.
# mem           120G host, as the project asks for the same cell.
#SBATCH --job-name=nibr2-orient
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g,gpu-h100-80g-short,gpu-h200-141g-short
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --output=/scratch/work/ladovj1/calculations/nibr2-orientation/out/%x-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/nibr2-orientation/out/%x-%A_%a.err

WORK=/scratch/work/ladovj1/calculations/nibr2-orientation
REPO=${DEFUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-jobs}
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv
PSEUDO=/scratch/work/ladovj1/calculations/NiBr2_ultracell/pseudo_paw

mkdir -p "$WORK/out" || exit 1
cd "$REPO" || exit 1
source "$VENV/bin/activate" || exit 1

export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$REPO"
export DEFUMAT_BAND_BATCH=64
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export DEFUMAT_WFC_STORE=device
export DEFUMAT_PROJECTORS=store
export DEFUMAT_CACHE_DIR=$WORK/cache-gpu

TASKS=(routea-tilted routea-steep routec-tilted)
TASK=${TASKS[${SLURM_ARRAY_TASK_ID:-0}]}

echo "=== node ==="; hostname; date; nvidia-smi -L
echo "commit: $(git -C "$REPO" rev-parse HEAD)"
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"

python3 -u tools/cluster/nibr2_orientation.py "$TASK" "$WORK/$TASK" \
        --pseudo-dir "$PSEUDO" --one-shot-conv-thr "${ONESHOT_CONV_THR:-1e-10}"
echo "exit $?"
echo "=== done ==="; date
