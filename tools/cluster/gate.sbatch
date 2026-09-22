#!/bin/bash
# The push gate on a compute node, for when the workstation is in use.
#
# `tools/test-fast.sh` is everything not marked `slow` -- about 2600 tests in ten
# minutes at a 5.5 GB peak on the workstation -- and it is what runs before a
# push. It belongs on the workstation, where the timing means something and the
# cache is warm; this exists for the case the workstation is busy doing physics,
# which is exactly when a long stretch of committed and unpushed work has
# accumulated.
#
# **The cache is off**, so this pays compilation for every shape rather than
# reading it, which makes the wall clock here **not** comparable with the
# workstation's ten minutes and not a timing of anything. What it is for is the
# pass or fail.
#
# ---- the #SBATCH values ----
#
# partition     batch-milan.
# time          02:00:00. Ten minutes warm on the workstation, and this pays
#               compilation per shape with `DEFUMAT_CACHE_DIR=off`; two hours is
#               headroom over the worst that has been seen rather than an
#               estimate.
# cpus-per-task 8, because `tests/conftest.py` sets `DEFUMAT_THREADS=8` itself,
#               for the thread-pool deadlock in `OPEN.md` Part IV item 1. MKL and
#               OMP are pinned to 1 beside XLA.
# mem           48G. The gate's measured peak on the workstation is 5.5 GB, but
#               that is with a warm cache, and a cache **miss is cheaper in
#               memory and dearer in time** while a hit expands a loaded
#               executable -- so the sign of the cache's effect on memory is the
#               opposite of the obvious one and this is sized well above either.
#SBATCH --job-name=defumat-gate
#SBATCH --partition=batch-milan
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-magnetism/out/gate-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-magnetism/out/gate-%j.err

WORK=/scratch/work/ladovj1/calculations/defumat-magnetism
REPO=${DEFUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-jobs}
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv

mkdir -p "$WORK/out" || exit 1
cd "$REPO" || exit 1
source "$VENV/bin/activate" || exit 1

export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export PYTHONPATH="$REPO"
export DEFUMAT_CACHE_DIR=off

echo "=== node ==="; hostname; date
echo "commit: $(git -C "$REPO" rev-parse HEAD)  $(git -C "$REPO" log --oneline -1)"
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"

# Not through `tools/test-fast.sh`: that script pins and times for the
# workstation, and what is wanted here is the selection alone. The marker is the
# selection, so it is spelled out rather than inherited.
#
# **One file is ignored, and it is the environment rather than the code.** The
# venv these jobs run has no `matplotlib`, so `test_result_plots.py` cannot even
# be imported, and a collection error stops the whole run rather than failing one
# file -- 1243 tests deselected and nothing executed. Ignoring it here does not
# make this gate equivalent to the workstation's, which is why that one stays the
# authority: what this answers is whether anything *else* broke.
python3 -m pytest -m "not slow" -q -p no:cacheprovider \
    --ignore=tests/unit/test_result_plots.py
STATUS=$?
echo "pytest exit $STATUS"

echo "=== done ==="; date
exit $STATUS
