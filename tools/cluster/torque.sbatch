#!/bin/bash
# What the magnetic torque's k dial is worth (`AUDIT-2026-09-18.md` hole.1):
# the same torque taken with the whole k axis on the tape and one k-point at a
# time, on a bulk cell and on a slab.
#
# Four tasks rather than one script running four things, because a peak
# resident set is a **high-water mark**: two routes in one process report the
# larger of the two twice, which is the shape of measurement error this project
# has already made about memory more than once.
#
# ---- the #SBATCH values ----
#
# partition     batch-milan, as the other two jobs here.
# array         0-3: (tetragonal, slab) x (whole, chunked), in that order.
# time          08:00:00. The slab is three cobalt atoms with vacuum at
#               ecutwfc 25 / ecutrho 200 on a shifted 4x4x1 grid, converged
#               scalar-relativistically and then diagonalised once as a spinor;
#               the gradient is one pass either way.
# cpus-per-task 4, MKL and OMP pinned to 1.
# mem           64G. Twice what the whole-axis route is expected to need on the
#               slab, because the point of the job is that nobody knows what
#               that route needs -- the entry sized it structurally and said in
#               as many words that no torque memory figure exists. A task that
#               is killed for want of memory is itself the measurement, and the
#               `seff` line says so.
#SBATCH --job-name=defumat-torque
#SBATCH --partition=batch-milan
#SBATCH --array=0-3
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-audit/out/torque-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-audit/out/torque-%A_%a.err

WORK=/scratch/work/ladovj1/calculations/defumat-audit
REPO=${DEFUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-audit}
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv

mkdir -p "$WORK/out" || exit 1
cd "$REPO" || exit 1
source "$VENV/bin/activate" || exit 1

export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export PYTHONPATH="$REPO"
# **Both sides must be the same cache state or the memory figure is about the
# cache** -- `CLAUDE.md` records a 603 MB cache entry worth 6.3 GB resident, in
# the direction that flatters whichever side was the miss. Off on both.
export DEFUMAT_CACHE_DIR=off
export DEFUMAT_QE_ROOT=/scratch/work/ladovj1/apps/qe-7.5-reference

CASES=(tetragonal tetragonal slab slab)
ROUTES=(whole chunked whole chunked)
INDEX=${SLURM_ARRAY_TASK_ID:-0}
CASE=${CASES[$INDEX]}
ROUTE=${ROUTES[$INDEX]}

echo "=== node ==="; hostname; date
echo "commit: $(git -C "$REPO" rev-parse HEAD)"
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"
python3 -c "import os; print('affinity:', len(os.sched_getaffinity(0)), 'cores')"

python3 -u tools/cluster/torque_batching.py "$CASE" "$ROUTE" \
        --out "$WORK/out/torque-${SLURM_ARRAY_JOB_ID}-${CASE}-${ROUTE}.json"
echo "exit $?"

echo "=== done ==="; date
