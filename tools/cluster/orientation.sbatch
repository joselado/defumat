#!/bin/bash
# The orientation torque and relaxation on the four-cell cobalt helix (P122,
# `ORIENTATION-NEXT.md` steps 2 and 3): `tools/cluster/orientation_helix.py`.
#
# Two tasks, one process each so that each peak is its own:
#   0  checks: the torque at an oblique orientation, its central difference
#      from two separate one-shots per component, the closed form
#      integral of m_out x B, and the torque with the coupling off;
#   1  relax: the relaxation from the same start, with the curvature at the end.
#
# ---- the #SBATCH values ----
#
# partition     batch-milan, as the other jobs here.
# time          04:00:00. The source SCF took 277 s on the workstation for 100
#               iterations of plain mixing (2.2 GB peak); a one-shot with the
#               coupling and its gradient are each a fraction of that, and the
#               relaxation is of order ten of them plus six for the curvature.
# cpus-per-task 4, MKL and OMP pinned to 1.
# mem           16G: seven times the SCF's measured peak, for a backward pass
#               on four atoms that has not been measured.
#SBATCH --job-name=defumat-orientation
#SBATCH --partition=batch-milan
#SBATCH --array=0-1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-audit/out/orientation-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-audit/out/orientation-%A_%a.err

WORK=/scratch/work/ladovj1/calculations/defumat-audit
REPO=${DEFUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-jobs}
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv

mkdir -p "$WORK/out" || exit 1
cd "$REPO" || exit 1
source "$VENV/bin/activate" || exit 1

export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export PYTHONPATH="$REPO"

TASKS=(checks relax)
INDEX=${SLURM_ARRAY_TASK_ID:-0}
TASK=${TASKS[$INDEX]}

echo "=== node ==="; hostname; date
echo "commit: $(git -C "$REPO" rev-parse HEAD)"
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"

python3 -u tools/cluster/orientation_helix.py "$TASK" \
        --out "$WORK/out/orientation-${SLURM_ARRAY_JOB_ID}-${TASK}.json"
echo "exit $?"

echo "=== done ==="; date
