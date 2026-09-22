#!/bin/bash
# The seven files the workstation's slow set could not finish, one per task.
#
# `tools/run_regression.sh` on 2026-09-21 lost seven files to SIGKILL, every one
# at **12.42 GB give or take 10 MB** -- which is the 12 G cap and not any test's
# own peak, so nothing in that run says what they need. This job is the
# measurement: give them room and read `seff` and the runner's own peak line.
#
# It is the narrow sibling of `regression.sbatch`, which runs the *whole* slow
# set eight tasks over 91 files. The difference that matters is **one file per
# task**: a shared process is what accumulates XLA executables, which is the
# mechanism behind these seven in the first place, so spreading them round robin
# beside other files would measure the wrong thing.
#
# ---- the #SBATCH values ----
#
# partition     batch-milan: 32 nodes, 128 cores and 510 GB each, 5-day limit
#               (`sinfo`). What this project's other CPU jobs use.
# array         0-6, one per file. Seven tasks is one array rather than seven
#               submissions, which is the cluster policy's own rule.
# time          12:00:00. `test_ten_site.py` is 27 slow tests and the heaviest
#               file on the workstation ran 38 minutes; with the cache off,
#               compilation is paid per file rather than read, so this is
#               headroom rather than an estimate.
# cpus-per-task 8. XLA sizes its pool from the affinity mask and Slurm's cgroup
#               is what bounds it, so this *is* the thread count; MKL and OMP
#               are pinned to 1 beside it. Eight rather than `regression.sbatch`'s
#               four because `tests/conftest.py` sets `DEFUMAT_THREADS=8` itself,
#               for the thread-pool deadlock in `OPEN.md` Part IV item 1.
# mem           64G. Five times the 12.42 G these files died at, and unlike
#               `regression.sbatch`'s 32 G the ask is justified per task rather
#               than by one outlier: every one of the seven is a known hog, and
#               `regression.sbatch` already records `test_ldau_flavours.py`
#               still climbing at 15.3 G under a 16 G cap. Seven tasks is 448 GB
#               across a shared cluster, which is why the list is seven and not
#               ninety-one.
#SBATCH --job-name=defumat-bigtests
#SBATCH --partition=batch-milan
#SBATCH --array=0-6
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-bigtests/out/big-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-bigtests/out/big-%A_%a.err

set -u
FILES=(test_ldau_flavours test_lsda test_noncollinear_hubbard_resume \
       test_piezoelectric_augmented test_piezoelectric_paw test_spinor_forces \
       test_ten_site)
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

WORK=/scratch/work/ladovj1/calculations/defumat-bigtests
REPO=${DEFUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-jobs}
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv

mkdir -p "$WORK/out" || exit 1
cd "$REPO" || exit 1
# A batch shell does not source .bashrc, so the environment is named in full.
source "$VENV/bin/activate" || exit 1

export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export DEFUMAT_THREADS=8
export JAX_PLATFORMS=cpu          # a CPU node has no device; say so rather than discover it
export PYTHONPATH="$REPO"

# The vendored QE tree is gitignored and absent from the cluster checkout, so
# `test-suite/` is shipped separately (61 MB) and the fixture pointed at it.
# Only `test_lsda` and `test_spinor_forces` read it; the other five do not.
export DEFUMAT_QE_ROOT=/scratch/work/ladovj1/apps/qe-ref

# **The kernel cache is off, for `regression.sbatch`'s reason and one more.**
# Its concurrent-writer behaviour on this filesystem has never been checked, and
# scratch is at 101 G of 200 G. The extra reason is that it decides how the
# peaks read: a miss is cheaper in memory and dearer in time than a hit, so
# these numbers are a **lower bound** against the warm workstation figures
# rather than like-for-like, and this job does not settle the warm-against-cold
# question of `OPEN.md` Part XIV item 5.
export DEFUMAT_CACHE_DIR=off

# Below the 64 G Slurm allocation, so the package's own watchdog names the
# offending test before Slurm kills the task without naming anything.
export DEFUMAT_TEST_MEM_MAX=56G

echo "=== node ==="; hostname; date
echo "file:   $FILE"
echo "commit: $(git rev-parse --short HEAD)  $(git log --oneline -1)"
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"

tools/run_regression.sh "$WORK/out/reg-$FILE" "slow" "$FILE.py"
status=$?

echo "=== summary ==="
cat "$WORK/out/reg-$FILE/summary.txt" 2>/dev/null
echo "exit $status"
date
echo "=== done ==="
