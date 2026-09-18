#!/bin/bash
# The same files at two commits, which is the only thing that says whether a
# failure is the session's work or the machine's.
#
# A cluster run of the `slow` set turned up assertion failures that no change in
# the session could plausibly reach -- `test_relaxed_anisotropy.py`'s identity
# calls the functional entry point directly, not the facade that changed -- and
# the environment is not the workstation's either: JAX 0.11.1 against 0.11.0,
# NumPy 2.5.2 against 2.4.6. Reading the diff cannot settle that. Running the
# same file at the commit the session started from can.
#
# `DEFUMAT_REPO` is the checkout and `DEFUMAT_FILES` the space-separated list,
# so one script serves both halves:
#
#   sbatch --export=ALL,DEFUMAT_REPO=<clone at the old commit>,DEFUMAT_FILES="a.py b.py" \
#          tools/cluster/control.sbatch
#
# ---- the #SBATCH values ----
#
# partition     batch-milan, as the rest.
# time          04:00:00: this is a handful of files, not the whole set.
# cpus-per-task 4, MKL and OMP pinned to 1.
# mem           32G, the regression array's size, for the same reasons.
#SBATCH --job-name=defumat-control
#SBATCH --partition=batch-milan
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-audit/out/control-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-audit/out/control-%j.err

WORK=/scratch/work/ladovj1/calculations/defumat-audit
REPO=${DEFUMAT_REPO:?set DEFUMAT_REPO to the checkout to test}
FILES=${DEFUMAT_FILES:?set DEFUMAT_FILES to the files to run}
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv
LABEL=${DEFUMAT_LABEL:-control}

mkdir -p "$WORK/out" || exit 1
cd "$REPO" || exit 1
source "$VENV/bin/activate" || exit 1

export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export PYTHONPATH="$REPO"
export DEFUMAT_CACHE_DIR=off
export DEFUMAT_QE_ROOT=/scratch/work/ladovj1/apps/qe-7.5-reference
export DEFUMAT_TEST_MEM_MAX=28G

echo "=== node ==="; hostname; date
echo "repo:   $REPO"
echo "commit: $(git -C "$REPO" rev-parse HEAD)  $(git -C "$REPO" log --oneline -1)"
echo "files:  $FILES"
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"

tools/run_regression.sh "$WORK/out/$LABEL" slow "$FILES"

echo "=== summary ==="; cat "$WORK/out/$LABEL/summary.txt"
echo "=== done ==="; date
