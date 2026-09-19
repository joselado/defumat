#!/bin/bash
# The three numbers that are left before the piezoelectric refusal can move.
#
# The ladder (`20337789`) answered its question: the response route is the
# unconverged one, reading -0.763786, -0.687475 and -0.672897 C/m^2 at `4 4 4`,
# `6 6 6` and `8 8 8` against a Berry-phase value that stays inside 0.7 per cent
# across four string meshes and two ground states. So the 15.7 per cent measured
# on the *ultrasoft* cell was taken at the same `4 4 4` and measures this
# k-error rather than the dataset, and `AUDIT-2026-09-18.md` `drift.3` cannot
# close until the ultrasoft response is run at a converged mesh.
#
#   0  the calibration cell one rung further, `10^3`, 1000 points. The response
#      route is still 1.6 per cent from the Berry value at `8 8 8` and still
#      moving, and no extrapolation of that residual is on the record because
#      three points on a moving series cannot say where it lands. This rung
#      says. Rung 1 of the ladder took four minutes at 512 points.
#   1  **the rung this array exists for**: the ultrasoft response at `6 6 6`.
#      Put against task 2 and against the ultrasoft Berry value already in hand
#      (+0.687757 at 7 strings over 4x4), it is the first number that is about
#      the dataset rather than about the mesh.
#   2  the ultrasoft Berry value at a second string length, 11 over 4x4. The
#      ladder's rung 6 tried 11 over 6x6, 36 strings, and died of mapping
#      exhaustion where the same 36 strings on the norm-conserving cell
#      finished in ten minutes (`OPEN.md` Part XIII item 2: `npwx` varies
#      string to string, so the Davidson stack recompiles once per distinct
#      shape). 16 strings is the count that is known to run on this cell, so
#      this rung lengthens the string instead of widening the mesh, and it
#      gives the ultrasoft cell the string-convergence check the calibration
#      cell already has.
#
# ---- the #SBATCH values ----
#
# partition     batch-milan; the nodes are 128 cores and 510 G, so a task this
#               size shares one rather than needing a whole one.
# array         0-2.
# time          04:00:00. The ultrasoft response took 13.5 minutes at 64 points
#               and the work is close to linear in `nk`, so task 1 is roughly
#               45 minutes; the margin is for the compilation the recompile
#               above makes unpredictable.
# cpus-per-task 4, MKL and OMP pinned to 1.
# mem           260G, sized from measurement rather than guessed. The
#               norm-conserving response peaked at 13.8 GiB at 216 points and
#               28.3 GiB at 512, so the peak is close to affine in `nk`, and the
#               ultrasoft response peaked at 49.4 GiB at 64 points
#               (`20336374_1`). Taking the pessimistic reading, that all 49.4
#               scales, `6 6 6` is 167 GiB, and 260 G is that with 40 per cent
#               over. Tasks 0 and 2 do not need it and are short.
#SBATCH --job-name=defumat-piezo-close
#SBATCH --partition=batch-milan
#SBATCH --array=0-2
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=260G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-audit/out/close-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-audit/out/close-%A_%a.err

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
    0) CASE=nc; RUNG=response-k10; ARGS="--kmesh 10 --skip-difference" ;;
    1) CASE=us; RUNG=response-k6;  ARGS="--kmesh 6 --skip-difference" ;;
    2) CASE=us; RUNG=berry-11-4x4; ARGS="--skip-response --nppstr 11 --transverse 4 4 --shears 0.005" ;;
    *) echo "no such rung"; exit 1 ;;
esac

echo "=== node ==="; hostname; date
echo "rung:   ${SLURM_ARRAY_TASK_ID:-0}  $CASE $RUNG"
echo "commit: $(git -C "$REPO" rev-parse HEAD)  $(git -C "$REPO" log --oneline -1)"
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"
python3 -c "import os; print('affinity:', len(os.sched_getaffinity(0)), 'cores')"

# shellcheck disable=SC2086
python3 -u tools/cluster/piezo_measure.py "$CASE" $ARGS \
        --out "$WORK/out/close-${SLURM_ARRAY_JOB_ID}-${CASE}-${RUNG}.json"
echo "exit $?"

echo "=== done ==="; date
