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
# mem           32G. Most files need a few GB; the measured worst is
#               `test_ldau_flavours.py`, killed at 16 G on the workstation and
#               still climbing at 15.3 G under a 16 G cap, because every
#               distinct cell compiles the whole stack afresh and XLA keeps
#               every executable for the life of the process. 32 G is twice
#               that and a smaller ask than the headroom would suggest: eight
#               tasks at 48 G would hold 384 GB of a shared cluster for the
#               sake of one file.
#SBATCH --job-name=defumat-regression
#SBATCH --partition=batch-milan
#SBATCH --array=0-7
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-audit/out/reg-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-audit/out/reg-%A_%a.err

WORK=/scratch/work/ladovj1/calculations/defumat-audit
# The checkout to run, overridable so that a run never has to modify the
# working tree someone else is using: this defaults to a clone of it.
REPO=${DEFUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-audit}
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
# **The vendored QE tree is gitignored**, so the checkout here does not have it
# and every test that reads a committed QE input or benchmark out of
# ``test-suite/`` would skip -- silently, and skipping is what this run exists
# to avoid. Only ``test-suite/`` is read (``tests/conftest.py``), 59 MB of it,
# so that subtree alone is mirrored to scratch and pointed at.
export DEFUMAT_QE_ROOT=/scratch/work/ladovj1/apps/qe-7.5-reference
# Under the Slurm cap, so the in-process watchdog names the test before the
# kernel acts on the 32 G.
export DEFUMAT_TEST_MEM_MAX=28G

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
