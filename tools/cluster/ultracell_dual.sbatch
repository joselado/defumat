#!/bin/bash
# The augmented ultracell at the dual its dataset was generated for, one array
# task per case, plus the two regression files as the validation arm.
#
# **Why this wants a cluster.** Seven cases, each of which is a unit-cell SCF, a
# tiled null, a real supercell SCF and a three-rung band ladder; the spinor
# rungs go to nbnd = 96 and the platinum pair is fully relativistic at
# ecutrho = 240 Ry. Run in one process on the workstation they would accumulate
# every compiled SCF stack of seven cells that share no shape, which is the
# accumulation `CLAUDE.md` names; here each case has a process and 32 G to
# itself and the split costs nothing.
#
# ---- the #SBATCH values ----
#
# partition     batch-milan, as `piezo_regression.sbatch` and `regression.sbatch`
#               use: 128 cores and 510 GB a node, 5-day limit. Check rather than
#               trust this line (`scontrol show partition batch-milan`).
# array         0-10: nine measurement cases (the two controls at
#               ecutrho = 4 ecutwfc first, so that a rung at a dual is read
#               against a committed number rather than against a table in
#               another file) and two regression files. One array and not
#               eleven submissions, which is the cluster policy's own rule.
# time          04:00:00. The silicon collinear pair is about 10 minutes on the
#               workstation at ecutrho = 4 ecutwfc and the dual roughly triples
#               the dense box; the platinum pair is the unknown and is what the
#               four hours are for.
# cpus-per-task 4, matching the other two scripts. XLA takes every core it can
#               see and Slurm's cgroup is what bounds it, so this *is* the thread
#               count; MKL and OMP are pinned to 1 beside it.
# mem           32G. The boxes here are small (a 25^3 dense grid at the silicon
#               dual, 45^3 at platinum's) and the peak is the compiled stack
#               rather than any one array, so this is many times the worst case
#               and leaves room for the run to be read afterwards with `seff`.
#SBATCH --job-name=defumat-ulr-dual
#SBATCH --partition=batch-milan
#SBATCH --array=0-10
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-ulr-dual/out/dual-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-ulr-dual/out/dual-%A_%a.err

WORK=/scratch/work/ladovj1/calculations/defumat-ulr-dual
REPO=${DEFUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-jobs}
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv

mkdir -p "$WORK/out" "$WORK/results" || exit 1
cd "$REPO" || exit 1
source "$VENV/bin/activate" || exit 1

export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export JAX_PLATFORMS=cpu          # a CPU node has no device; say so rather than discover it
export PYTHONPATH="$REPO"
# A cache *miss* is the cheaper state in memory and the dearer one in time, and
# nothing here is a timing, so the cache is off and every task pays its own
# compilation. It also keeps nine tasks off one cache directory on NFS.
export DEFUMAT_CACHE_DIR=off
export DEFUMAT_TEST_MEM_MAX=28G

TASK=${SLURM_ARRAY_TASK_ID:-0}

echo "=== node ==="; hostname; date
echo "commit: $(git -C "$REPO" rev-parse HEAD)"
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"
python3 -c "import os; print('affinity:', len(os.sched_getaffinity(0)), 'cores')"

if [ "$TASK" -le 8 ]; then
    echo "=== task $TASK: measurement case $TASK ==="
    python3 -u tools/cluster/ultracell_dual.py "$TASK" "$WORK/results"
else
    # The validation arm: the two ultracell regression files, which is where the
    # lift's own tests live and where a number that moved would show.
    FILES=(tests/regression/test_ultracell_augmented.py
           tests/regression/test_ultracell.py)
    FILE=${FILES[$((TASK - 9))]}
    echo "=== task $TASK: $FILE ==="
    tools/run_regression.sh "$WORK/out/regression-$TASK" slow "$FILE"
    echo "=== summary ==="; cat "$WORK/out/regression-$TASK/summary.txt"
fi

echo "=== done ==="; date
