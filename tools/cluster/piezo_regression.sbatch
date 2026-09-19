#!/bin/bash
# The three piezoelectric regression files, one array task each.
#
# **Why these three want a cluster and the rest of the suite does not.** They are
# the only files here whose cost is a *single test's working set* rather than
# accumulation over a file: the taped route holds a forward-over-reverse tape of
# every radial and reciprocal-space intermediate a cell derivative rebuilds, and
# on the augmented cells that is about 10 GiB for one k-set. The workstation has
# 30 GB and `tools/run_regression.sh` caps a file at 12, so two augmented cells
# could not share a process there -- they passed together once, cold, and were
# killed at 137 the next time, because the first run had compiled the second
# cell's kernels and the second read them back off the on-disk cache and a
# loaded executable is *larger* resident than a compiled one (`CLAUDE.md`'s cache
# rule, sign reversed). Splitting the files was the workstation's answer. Here
# each task has 32 G to itself and the split costs nothing either way.
#
# ---- the #SBATCH values ----
#
# partition     batch-milan, as `regression.sbatch` uses: 128 cores and 510 GB a
#               node, 5-day limit. Look it up rather than trust this line
#               (`scontrol show partition batch-milan`).
# array         0-2, one task per file. **One array and not three submissions**,
#               which is the cluster policy's own rule.
# time          01:00:00. The three files are 5m24, 1m55 and 5m00 on the
#               workstation with a warm kernel cache; the cache is off here, so
#               every kernel is compiled once, and an hour is roughly four times
#               the worst of them.
# cpus-per-task 4, matching `regression.sbatch`. XLA takes every core it can see
#               and Slurm's cgroup is what bounds it, so this *is* the thread
#               count; MKL and OMP are pinned to 1 beside it.
# mem           32G. The measured peak is 10.1 GiB for one augmented cell with a
#               warm cache and less with a cold one, so this is three times the
#               worst and leaves room for the in-process watchdog to name a test
#               before the kernel acts.
#SBATCH --job-name=defumat-piezo
#SBATCH --partition=batch-milan
#SBATCH --array=0-2
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-piezo/out/piezo-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-piezo/out/piezo-%A_%a.err

WORK=/scratch/work/ladovj1/calculations/defumat-piezo
# The checkout to run, overridable so that a run never has to modify a working
# tree someone else is using.
REPO=${DEFUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-audit}
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv

mkdir -p "$WORK/out" || exit 1
cd "$REPO" || exit 1
source "$VENV/bin/activate" || exit 1

export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export JAX_PLATFORMS=cpu          # a CPU node has no device; say so rather than discover it
export PYTHONPATH="$REPO"
# Off for `regression.sbatch`'s reasons, and for one more that belongs to these
# files in particular: a cache *miss* is the cheaper state in memory, and the
# whole reason these three are split on the workstation is a warm cache. Running
# them cold here is the like-for-like comparison with what the workstation did
# the first time, not with what it does now.
export DEFUMAT_CACHE_DIR=off
# These files read no committed QE benchmark, so `DEFUMAT_QE_ROOT` is not needed
# -- but it is set anyway, because a test that silently skips is what a cluster
# run exists to avoid and the cost of pointing at a directory is nothing.
export DEFUMAT_QE_ROOT=/scratch/work/ladovj1/apps/qe-7.5-reference
# Under the Slurm cap, so the in-process watchdog names the test before the
# kernel acts on the 32 G.
export DEFUMAT_TEST_MEM_MAX=28G

FILES=(tests/regression/test_piezoelectric_augmented.py
       tests/regression/test_piezoelectric_wedge.py
       tests/regression/test_piezoelectric.py)
TASK=${SLURM_ARRAY_TASK_ID:-0}
FILE=${FILES[$TASK]}

echo "=== node ==="; hostname; date
echo "commit: $(git -C "$REPO" rev-parse HEAD)"
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"
python3 -c "import os; print('affinity:', len(os.sched_getaffinity(0)), 'cores')"
echo "=== task $TASK: $FILE ==="

tools/run_regression.sh "$WORK/out/piezo-$TASK" slow "$FILE"

echo "=== summary ==="; cat "$WORK/out/piezo-$TASK/summary.txt"
echo "=== done ==="; date
