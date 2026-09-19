#!/bin/bash
# Ladder the ultrasoft cell with the route that carries no tape.
#
# `20338380` measured what the taped route costs on that cell: **139.6 GiB** at
# 64 k-points against 49.4 at 8, about 1.6 GiB a point, so `6 6 6` projects to
# 385 GiB and `k_batch = 1` is worth only eleven per cent of it. The transcribed
# contraction (`piezoelectric_zstar_eu_style`, now reachable as
# `method = 'zstar_eu'`) is the same number with no forward-over-reverse tape at
# all, and this array is whether that lets the cell be laddered.
#
#   0  the **validation rung**, and it comes first for a reason. The same 64
#      points the taped route already read at **+0.815802**, assembled the other
#      way. The two agree to 6.2e-15 on the *norm-conserving* cell and have never
#      been compared on an augmented one, where `_bare_strains` carries an extra
#      term -- it subtracts `eps dS|psi>` when the Hamiltonian has an overlap
#      (P39) and rebuilds the potential inside the traced function, so the
#      `Q_ij` strain derivative is differentiated rather than frozen. That the
#      term is *there* is reading the source; that it is *right* is this rung.
#      If it disagrees, the cheap route cannot ladder this cell and the
#      disagreement is itself the missing term `drift.3` has been looking for.
#   1, 2  the rungs that need it: `6 6 6` and `8 8 8`, 216 and 512 points. With
#      the calibration cell's curve for comparison (13.4, 3.7, 1.6, 1.2 per cent
#      at `4^3` to `10^3`) these say whether the ultrasoft deficit converges the
#      same way, which is the whole of what separates a dataset effect from a
#      difference of mesh errors in the 1.65-point gap.
#
# ---- the #SBATCH values ----
#
# partition     batch-milan.
# array         0-2.
# time          04:00:00. The taped route took 451 s at 64 points and the
#               transcribed one is cheaper in the assembly but pays the same
#               field response, so the cost is roughly linear in `nk` from
#               there: `8 8 8` is the one with the margin.
# cpus-per-task 4, MKL and OMP pinned to 1.
# mem           200G, which is deliberately more than this route should need and
#               is the same request that scheduled last time. The point of task 0
#               is partly to measure the peak, since nothing on record says what
#               the transcribed route costs on an augmented dataset; if it comes
#               in low the next array asks for much less.
#SBATCH --job-name=defumat-piezo-cheap
#SBATCH --partition=batch-milan
#SBATCH --array=0-2
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-audit/out/cheap-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-audit/out/cheap-%A_%a.err

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
    0) RUNG=zstar-k4;  MESH=4 ;;
    1) RUNG=zstar-k6;  MESH=6 ;;
    2) RUNG=zstar-k8;  MESH=8 ;;
    *) echo "no such rung"; exit 1 ;;
esac

echo "=== node ==="; hostname; date
echo "rung:   ${SLURM_ARRAY_TASK_ID:-0}  us $RUNG"
echo "commit: $(git -C "$REPO" rev-parse HEAD)  $(git -C "$REPO" log --oneline -1)"
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"
python3 -c "import os; print('affinity:', len(os.sched_getaffinity(0)), 'cores')"

python3 -u tools/cluster/piezo_measure.py us \
        --nosym --kmesh "$MESH" --skip-difference --method zstar_eu \
        --out "$WORK/out/cheap-${SLURM_ARRAY_JOB_ID}-${RUNG}.json"
echo "exit $?"

echo "=== done ==="; date
