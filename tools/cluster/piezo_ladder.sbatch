#!/bin/bash
# Which of the two piezoelectric routes is the unconverged one.
#
# The calibration cell disagreed (`AUDIT-2026-09-18.md` `drift.3`, `OPEN.md`
# Part XIII item 3): on norm-conserving AlAs the response route gives
# -0.763786 C/m^2 and a Berry-phase finite difference gives -0.661386, 13.4 per
# cent low, and the ultrasoft cell reads 15.7 per cent. The step is not the
# cause -- doubling it moves the answer by 0.3 per cent and away from the
# response route -- and the harness's single-cell contraction is exact for this
# component. What the two routes do *not* share is how they sample `k`: the
# response integrates the SCF's `4 4 4 0 0 0`, 64 points, and the Berry phase
# runs `nppstr x transverse` strings, 396 on that cell.
#
# **So the ladder is two-sided, and that is the whole design.** A one-sided
# ladder cannot tell a converged route from an unconverged one: whichever number
# moves toward the other is the answer, and a run that moves neither says the
# disagreement is somewhere else and rules out the k-sampling by measurement
# rather than by argument.
#
#   0, 1  the response route at the whole 6^3 and 8^3 grids, 216 and 512 points
#         against the input's 64. `--kmesh` refuses a symmetry-reduced cell, so
#         only the `nosym` calibration cell is laddered this way.
#   2     the finite difference at the *ultrasoft* cell's mesh, 7 strings over
#         4x4. This is the rung that makes the two cells' deficits comparable:
#         at present they are 13.4 and 15.7 per cent at two different meshes, so
#         the 2.3-point difference is attributable to nothing.
#   3, 4  the finite difference denser, 15 strings over 6x6 and over 8x8.
#   5     the finite difference with its *ground states* on the 6^3 grid, so that
#         the SCF mesh and the string mesh are separated from each other.
#   6     the ultrasoft cell's finite difference at the calibration cell's own
#         mesh, 11 over 6x6, which is the other half of rung 2.
#
# One shear per rung rather than two: the step was shown to be converged, so a
# second one buys nothing here and the rungs are the expensive axis.
#
# ---- the #SBATCH values ----
#
# partition     batch-milan.
# array         0-6, the seven rungs above.
# time          02:00:00. The dearest rung is 6 (the ultrasoft cell at 396
#               strings, extrapolating to ~20 minutes from its 318 s at 112)
#               and rung 4 is ~20 minutes on the cheap cell.
# cpus-per-task 4, MKL and OMP pinned to 1.
# mem           120G for every task, which is over-provisioned for the six
#               norm-conserving rungs and is what rung 6 needs: the ultrasoft
#               response was killed at 32 G by the augmentation table
#               (`nh^2 x ngm` per atom on the dense grid). One array with one
#               `--mem` is worth more than splitting the ladder to save
#               allocation on tasks that run for ten minutes.
#SBATCH --job-name=defumat-ladder
#SBATCH --partition=batch-milan
#SBATCH --array=0-6
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-audit/out/ladder-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-audit/out/ladder-%A_%a.err

WORK=/scratch/work/ladovj1/calculations/defumat-audit
REPO=${DEFUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-jobs}
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv

mkdir -p "$WORK/out" || exit 1
cd "$REPO" || exit 1
source "$VENV/bin/activate" || exit 1

export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export PYTHONPATH="$REPO"
export DEFUMAT_CACHE_DIR=off
export DEFUMAT_QE_ROOT=/scratch/work/ladovj1/apps/qe-7.5-reference

case ${SLURM_ARRAY_TASK_ID:-0} in
    0) CASE=nc; RUNG=response-k6;  ARGS="--kmesh 6 --skip-difference" ;;
    1) CASE=nc; RUNG=response-k8;  ARGS="--kmesh 8 --skip-difference" ;;
    2) CASE=nc; RUNG=berry-7-4x4;  ARGS="--skip-response --nppstr 7  --transverse 4 4 --shears 0.005" ;;
    3) CASE=nc; RUNG=berry-15-6x6; ARGS="--skip-response --nppstr 15 --transverse 6 6 --shears 0.005" ;;
    4) CASE=nc; RUNG=berry-15-8x8; ARGS="--skip-response --nppstr 15 --transverse 8 8 --shears 0.005" ;;
    5) CASE=nc; RUNG=berry-k6;     ARGS="--skip-response --kmesh 6 --nppstr 11 --transverse 6 6 --shears 0.005" ;;
    6) CASE=us; RUNG=berry-11-6x6; ARGS="--skip-response --nppstr 11 --transverse 6 6 --shears 0.005" ;;
    *) echo "no such rung"; exit 1 ;;
esac

echo "=== node ==="; hostname; date
echo "rung:   ${SLURM_ARRAY_TASK_ID:-0}  $CASE $RUNG"
echo "commit: $(git -C "$REPO" rev-parse HEAD)  $(git -C "$REPO" log --oneline -1)"
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"
python3 -c "import os; print('affinity:', len(os.sched_getaffinity(0)), 'cores')"

# shellcheck disable=SC2086
python3 -u tools/cluster/piezo_measure.py "$CASE" $ARGS \
        --out "$WORK/out/ladder-${SLURM_ARRAY_JOB_ID}-${CASE}-${RUNG}.json"
echo "exit $?"

echo "=== done ==="; date
