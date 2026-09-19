#!/bin/bash
# Does the multipliers' own response close the 1.8 per cent?
#
# `20338722_0` put the transcribed route at **+0.830702** C/m^2 on ultrasoft
# AlAs at 64 k-points where the differentiated route reads **+0.815802**, and
# the cause was that `zstar_eu.f90:90` hands an augmented dataset to
# `zstar_eu_us.f90` while the transcription stopped at the first file. What was
# missing is one contraction rather than three hundred lines, because only the
# *strain* leg of this derivative moves `S`: the constraint enters the
# stationary functional as `-Re Tr[Lambda (G - 1)]`, so the mixed derivative
# gains `-Re sum_mn dLambda^E_mn S'_nm` and the tensor gains
# `+(1/Omega) Re sum_mn dLambda^E_mn S'_nm`, with both factors already present
# as `strain.overlap_derivatives` and `born._multiplier_response`.
#
# **The norm-conserving half of the check has already passed on the
# workstation**: the two routes still agree to 6.2e-15 on `alas-raman.in`,
# unchanged to the digit, because both factors are identically zero when `S`
# does not deform. This is the other half, and it is the one that can fail.
#
#   0  the ultrasoft cell at the same 64 points. **The target is +0.815802.**
#      Agreement to the norm-conserving floor says the term is right and the
#      dataset refusal on the transcribed route can go with a number. A miss by
#      a clean factor of two, or by a sign, says which part of the derivation
#      to re-read rather than which constant to tune.
#   1  the same at `6 6 6`, which is only worth reading if 0 passes, and then
#      it is the first rung of an ultrasoft ladder that costs 2.6 GiB instead
#      of a node.
#
# ---- the #SBATCH values ----
#
# partition     batch-milan.
# array         0-1.
# time          02:00:00; the same two rungs took 6 and 14 minutes before the
#               extra contraction, which is one einsum over the occupied block.
# cpus-per-task 4, MKL and OMP pinned to 1.
# mem           32G. Measured, not guessed: this route peaked at 2.32 GiB at 64
#               points and 2.63 at 216, and the new term is an `nbnd x nbnd`
#               matrix per k-point on top.
#SBATCH --job-name=defumat-dlambda
#SBATCH --partition=batch-milan
#SBATCH --array=0-1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-audit/out/dlambda-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-audit/out/dlambda-%A_%a.err

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

MESH=$(( ${SLURM_ARRAY_TASK_ID:-0} == 0 ? 4 : 6 ))

echo "=== node ==="; hostname; date
echo "rung:   ${SLURM_ARRAY_TASK_ID:-0}  us zstar_eu + dLambda at ${MESH}^3"
echo "commit: $(git -C "$REPO" rev-parse HEAD)  $(git -C "$REPO" log --oneline -1)"
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"

python3 -u tools/cluster/piezo_measure.py us \
        --nosym --kmesh "$MESH" --skip-difference --method zstar_eu \
        --out "$WORK/out/dlambda-${SLURM_ARRAY_JOB_ID}-k${MESH}.json"
echo "exit $?"

echo "=== done ==="; date
