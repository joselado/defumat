#!/bin/bash
#SBATCH --job-name=defumat-bigtests
#SBATCH --partition=batch-milan
#SBATCH --array=0-6
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o /scratch/work/ladovj1/calculations/defumat-bigtests/out/big-%A_%a.out
#SBATCH -e /scratch/work/ladovj1/calculations/defumat-bigtests/out/big-%A_%a.err

# The seven files the workstation's slow set could not finish: every one was
# SIGKILLed at 12.42 GB give or take 10 MB, which is the 12 G cap and not the
# tests' own peaks, so what this job measures is what they actually need.
# One array task per file, because a shared process is what accumulates XLA
# executables in the first place.

set -u
FILES=(test_ldau_flavours test_lsda test_noncollinear_hubbard_resume \
       test_piezoelectric_augmented test_piezoelectric_paw test_spinor_forces \
       test_ten_site)
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

BASE=/scratch/work/ladovj1/calculations/defumat-bigtests
REPO=/scratch/work/ladovj1/apps/defumat-jobs
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv/bin

# A batch shell does not source .bashrc, so the interpreter is named in full.
export PATH=$VENV:$PATH

# XLA sizes its pool from the affinity mask and ignores OMP_NUM_THREADS, so the
# pin has to be the mask. tests/conftest.py sets DEFUMAT_THREADS=8 itself, for
# the deadlock reason in OPEN.md Part IV, and 8 is what is asked for here.
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export DEFUMAT_THREADS=8

# The vendored QE tree is gitignored and absent from the checkout; only
# test_lsda and test_spinor_forces read it, and they skip cleanly without it.
export DEFUMAT_QE_ROOT=/scratch/work/ladovj1/apps/qe-ref

# One compile cache per task. It is COLD, which matters for reading the peaks:
# a cache miss is cheaper in memory and dearer in time than a hit, so these
# numbers are a lower bound on the warm workstation figures rather than a
# like-for-like comparison with them.
export DEFUMAT_CACHE_DIR=$BASE/out/cache-$SLURM_ARRAY_TASK_ID

# Below the 64 G Slurm allocation, so the watchdog names the offending test
# before Slurm kills the task without naming anything.
export DEFUMAT_TEST_MEM_MAX=56G

echo "=== node ==="; hostname; date
echo "file:   $FILE"
cd $REPO || exit 1
echo "commit: $(git rev-parse --short HEAD)  $(git log --oneline -1)"
python3 -c "import jax; print('jax', jax.__version__, jax.devices())" 2>&1 | tail -1

tools/run_regression.sh $BASE/out/reg-$FILE "slow" "$FILE.py"
status=$?

echo "=== summary ==="
cat $BASE/out/reg-$FILE/summary.txt 2>/dev/null
echo "exit $status"
date
echo "=== done ==="
