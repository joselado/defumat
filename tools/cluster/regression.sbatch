#!/bin/bash
# The whole `slow` set at one commit, split across an array.
#
# It is two hours in one process on the workstation and is not a push gate, so
# it drifts: `PLAN.md` records three phases' claims going stale unnoticed before
# P38 looked. A cluster array is the natural home for it -- the files are
# independent, the machine is not the user's, and a file that needs more memory
# than a workstation has (`test_ldau_flavours.py` is killed at 16 GB there) gets
# it here.
#
# ---- the #SBATCH values ----
#
# partition     batch-milan: 32 nodes, 128 cores and 510 GB each, 5-day limit
#               (`scontrol show partition batch-milan`). It is what this
#               project's existing CPU jobs use.
# array         0-7. Eight tasks over 91 files that carry slow tests, taken
#               round robin off a sorted list, so a heavy file is unlikely to
#               land beside another heavy one. **One array rather than 91
#               submissions**, which is the cluster policy's own rule.
# time          08:00:00. The whole set is ~2 h in one process on a faster core;
#               an eighth of it with compilation paid per file rather than from
#               a cache should be well under this, and a task that finishes
#               early releases the slot.
# cpus-per-task 4. XLA takes every core it can see and Slurm's cgroup is what
#               bounds it, so this *is* the thread count; MKL and OMP are pinned
#               to 1 beside it.
# mem           48G. The measured worst file is `test_ldau_flavours.py` at over
#               16 GB on the workstation, where it is killed; 48 G leaves room
#               for the accumulation that makes it expensive (every distinct
#               cell compiles the whole stack afresh and XLA keeps every
#               executable for the life of the process).
#SBATCH --job-name=defumat-regression
#SBATCH --partition=batch-milan
#SBATCH --array=0-7
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-audit/out/reg-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-audit/out/reg-%A_%a.err

WORK=/scratch/work/ladovj1/calculations/defumat-audit
REPO=/scratch/work/ladovj1/apps/defumat
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv

mkdir -p "$WORK/out" || exit 1
cd "$REPO" || exit 1
source "$VENV/bin/activate" || exit 1

export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export JAX_PLATFORMS=cpu          # a CPU node has no device; say so rather than discover it
export PYTHONPATH="$REPO"
# **The kernel cache is off on purpose.** It is 4 GB and growing on the
# workstation, scratch here is at 89 per cent, and its concurrent-writer
# behaviour on this filesystem has never been checked -- eight tasks writing one
# directory is exactly the case nobody has measured. What it costs is
# compilation per file; what it buys is that nothing is shared between tasks and
# that a cache *miss* is the cheaper state in memory (`CLAUDE.md`).
export DEFUMAT_CACHE_DIR=off
# Under the Slurm cap, so the in-process watchdog names the test before the
# kernel acts on the 48 G.
export DEFUMAT_TEST_MEM_MAX=40G

TASK=${SLURM_ARRAY_TASK_ID:-0}
TASKS=${SLURM_ARRAY_TASK_COUNT:-8}
FILES=$(ls tests/regression/test_*.py tests/unit/test_*.py \
        | awk -v n="$TASKS" -v i="$TASK" 'NR % n == i')

echo "=== node ==="; hostname; date
echo "commit: $(git -C "$REPO" rev-parse HEAD)"
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"
python3 -c "import os; print('affinity:', len(os.sched_getaffinity(0)), 'cores')"
echo "=== task $TASK of $TASKS, $(echo "$FILES" | wc -l) files ==="
echo "$FILES"

tools/run_regression.sh "$WORK/out/reg-$TASK" slow "$FILES"

echo "=== summary ==="; cat "$WORK/out/reg-$TASK/summary.txt"
echo "=== done ==="; date
