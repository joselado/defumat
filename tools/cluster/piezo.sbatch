#!/bin/bash
# `e_14` by this package's response route and by Elk's, on a norm-conserving
# cell and on an ultrasoft one -- the measurement the piezoelectric tensor's
# dataset refusal needs before it can be lifted (`AUDIT-2026-09-18.md` drift.3).
#
# Two array tasks rather than two submissions, and they are deliberately the
# *same* script on two cells: the norm-conserving one is the calibration of the
# finite-difference harness, and without it a disagreement on the ultrasoft cell
# says nothing about which side is wrong.
#
# ---- the #SBATCH values ----
#
# partition     batch-milan, as the regression array.
# array         0-1: task 0 is the norm-conserving control, task 1 the
#               ultrasoft case. Task 1 is much the more expensive of the two
#               (ecutwfc 25 with ecutrho 200 against 10), which is why they are
#               array members rather than one script running both in sequence.
# time          12:00:00. Each task is one SCF, one Sternheimer field response,
#               and then four more converged ground states with three
#               Berry-phase string meshes each. The ultrasoft one carries the
#               augmentation charge through all of it.
# cpus-per-task 4, with MKL and OMP pinned to 1: XLA sees the cgroup's cores.
# mem           120G, and the first attempt is why. At 32 G the ultrasoft task
#               was **OOM-killed inside the response**, after an SCF that took
#               19 s -- on a two-atom cell with eight k-points, where the states
#               themselves are megabytes. What is large is not the states but
#               the augmentation table the backward pass carries, `nh^2 x ngm`
#               per atom on the **dense** grid that `ecutrho = 200` sets, which
#               is `CLAUDE.md`'s own account of where an ultrasoft derivative's
#               peak lives. The norm-conserving task shares the allocation and
#               will not use it.
#SBATCH --job-name=defumat-piezo
#SBATCH --partition=batch-milan
#SBATCH --array=0-1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-audit/out/piezo-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-audit/out/piezo-%A_%a.err

WORK=/scratch/work/ladovj1/calculations/defumat-audit
# The checkout to run, overridable so that a run never has to modify the
# working tree someone else is using: this defaults to a clone of it.
REPO=${DEFUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-audit}
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv

mkdir -p "$WORK/out" || exit 1
cd "$REPO" || exit 1
source "$VENV/bin/activate" || exit 1

export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export PYTHONPATH="$REPO"
# One process does every step here, so the on-disk cache would only help across
# *jobs*; it is off for the regression array's reasons (scratch at 89 per cent,
# concurrent writers unmeasured) and costs this job almost nothing.
export DEFUMAT_CACHE_DIR=off
export DEFUMAT_QE_ROOT=/scratch/work/ladovj1/apps/qe-7.5-reference

CASES=(nc us)
CASE=${CASES[${SLURM_ARRAY_TASK_ID:-0}]}

echo "=== node ==="; hostname; date
echo "commit: $(git -C "$REPO" rev-parse HEAD)"
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"
python3 -c "import os; print('affinity:', len(os.sched_getaffinity(0)), 'cores')"

python3 -u tools/cluster/piezo_measure.py "$CASE" \
        --out "$WORK/out/piezo-${SLURM_ARRAY_JOB_ID}-${CASE}.json"
echo "exit $?"

echo "=== done ==="; date
