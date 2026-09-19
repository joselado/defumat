#!/bin/bash
# The ultrasoft cell, laddered the way the calibration cell already was.
#
# `20338160` closed the calibration side and left one hole. The norm-conserving
# response reads -0.763786, -0.687475, -0.672897 and -0.669907 C/m^2 at `4 4 4`,
# `6 6 6`, `8 8 8` and `10 10 10`, with successive differences of 0.0763, 0.0146
# and 0.0030, so it is converging geometrically by about a factor of five a rung
# and the Berry-phase value it is walking toward sits at -0.660 to -0.662. The
# ultrasoft cell has never been laddered at all, because
# `tools/cluster/piezo_measure.py`'s `--kmesh` refuses a symmetry-reduced cell
# and `alas-piezo.in`, unlike `alas-raman.in`, keeps its point group. `--nosym`
# is the way through and it is what the calibration cell already does.
#
#   0, 1  the ultrasoft response on the whole `4 4 4` grid, 64 points against
#         the reduced 8 the input gives, once at the default batching and once
#         at `k_batch = 1`. **Two rungs for one number on purpose**: the peak
#         at 64 points decides whether `6 6 6` can be asked for at all, and
#         `piezo.py`'s docstring claims the tape "does not move with `k_batch`",
#         measured at 64 k on the small cell and never checked where the peak
#         is close to affine in `nk`. Running the pair measures the number and
#         the claim in the same wall clock.
#   2     the ultrasoft Berry value at 36 strings, 11 over 6x6, which is the
#         calibration cell's own mesh. It is the rung that died of mapping
#         exhaustion last time, and `run_polarization` now clears the compiled
#         code every 32 strings, which holds `/proc/self/maps` at 3705 lines
#         against 19031 on exactly this shape.
#   3     the ultrasoft Berry value at a longer string, 15 over 4x4, so that
#         this cell gets the string-length check the other one has. It already
#         reads +0.687757 at 7 over 4x4 and +0.690046 at 11 over 4x4, 0.3 per
#         cent apart.
#
# ---- the #SBATCH values ----
#
# partition     batch-milan, 128 cores and 510 G a node.
# array         0-3.
# time          04:00:00. The ultrasoft response took 13.5 minutes at 8 reduced
#               points and the Berry rungs 8 to 9 minutes at 16 strings, so the
#               margin is for the eightfold k-set and the 36-string rung.
# cpus-per-task 4, MKL and OMP pinned to 1.
# mem           200G. The ultrasoft response peaked at 49.4 GiB at **8** points
#               (`20336374_1`), and the peak is close to affine in `nk` on the
#               norm-conserving side, so 64 points is the measurement rather
#               than the extrapolation -- which is why task 1 exists. 200 G is
#               four times the only ultrasoft peak on record and still leaves a
#               node able to take two of these at once.
#SBATCH --job-name=defumat-piezo-us
#SBATCH --partition=batch-milan
#SBATCH --array=0-3
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-audit/out/us-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-audit/out/us-%A_%a.err

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
    0) RUNG=response-k4;       ARGS="--nosym --kmesh 4 --skip-difference" ;;
    1) RUNG=response-k4-kb1;   ARGS="--nosym --kmesh 4 --skip-difference --k-batch 1" ;;
    2) RUNG=berry-11-6x6;      ARGS="--skip-response --nppstr 11 --transverse 6 6 --shears 0.005" ;;
    3) RUNG=berry-15-4x4;      ARGS="--skip-response --nppstr 15 --transverse 4 4 --shears 0.005" ;;
    *) echo "no such rung"; exit 1 ;;
esac

echo "=== node ==="; hostname; date
echo "rung:   ${SLURM_ARRAY_TASK_ID:-0}  us $RUNG"
echo "commit: $(git -C "$REPO" rev-parse HEAD)  $(git -C "$REPO" log --oneline -1)"
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"
python3 -c "import os; print('affinity:', len(os.sched_getaffinity(0)), 'cores')"

# shellcheck disable=SC2086
python3 -u tools/cluster/piezo_measure.py us $ARGS \
        --out "$WORK/out/us-${SLURM_ARRAY_JOB_ID}-${RUNG}.json"
echo "exit $?"

echo "=== done ==="; date
