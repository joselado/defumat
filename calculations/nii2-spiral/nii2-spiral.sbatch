#!/bin/bash
# NiI2 monolayer: E(q) and dE/dq over a line of spin-spiral wavevectors.
#
# One array task per q. The points are independent -- run_spiral_scan does not
# warm-start one q from the last, so a loop would only save the setup and the
# compile (seconds against an SCF that is minutes) and would serialise eleven
# SCFs behind one GPU. This is the cluster policy's "group short tasks into one
# array" read the right way round: one submission, twenty tasks.
#
# ---- the #SBATCH values, and why each one ----
#
# partition   gpu-h200-141g-short. 141 GB of HBM is the reason: 81 k-points x 64
#             spinor bands x 26315 plane waves x 2 components is 4.4 GB of
#             wavefunctions before the Davidson subspace, and the real-space FFT
#             boxes during h_psi are (45 x 45 x 270) x 16 B = 8.7 MB each, so
#             k_batch is what keeps the working set finite (see --k-batch below).
#             `sbatch --test-only` first -- verify the reading, do not inherit it.
# time        8 h, and this one *is* measured: the first attempt's twenty SCFs
#             took 40 min to 2 h 11 min each (sacct on 19988867). Eight hours
#             covers the longest with room for the gradient and backfills far
#             better in a short partition than the 24 h placeholder did.
# cpus-per-task 4, matching the existing defumat GPU jobs; XLA is pinned to it.
# mem         160G host, matching phase5-si10-gpu.sbatch. Device memory is
#             measured rather than requested.
# array       0-19, the twenty points of the circuit M -> K -> Gamma -> M,
#             spaced uniformly in *reciprocal* space rather than in fractional
#             coordinates (the three segments are 1 : 2 : 1.73 in length), so
#             the corners land exactly: M at index 0, K at 4, Gamma at 12 and M
#             again at 19. The last point repeats the first, which is
#             deliberate: E must come back to E(M) and the integrated curve to
#             zero, and running M in two independent jobs is also the cross-job
#             reproducibility check GPU.md Phase 0 left unrun. M is also the
#             commensurate antiferromagnet -- the one wavevector on the circuit
#             a supercell could check independently. %4 caps how many run at
#             once so one calculation does not take four GPUs' worth of the
#             partition.
#
#             Override it on the command line for the first point --
#             `sbatch --array=0 nii2-spiral.sbatch` -- and release the rest only
#             once that one has written a JSON with a non-null gradient. The
#             first attempt at this calculation put all twenty on the same rock
#             at once and lost every one of them.
#SBATCH --job-name=nii2-spiral
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --time=08:00:00
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --array=0-19%4
#SBATCH --output=/scratch/work/ladovj1/calculations/nii2-spiral/out/nii2-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/nii2-spiral/out/nii2-%A_%a.err

WORK=/scratch/work/ladovj1/calculations/nii2-spiral
REPO=/scratch/work/ladovj1/apps/defumat
mkdir -p "$WORK/out" "$WORK/results" "$WORK/cache-gpu"
cd "$REPO" || exit 1
source /scratch/work/ladovj1/calculations/defumat-gpu/venv/bin/activate

# Leave XLA_PYTHON_CLIENT_PREALLOCATE alone: a single-tenant job wants the
# preallocation, which is what avoids fragmentation over a long SCF (GPU.md).
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export DEFUMAT_CACHE_DIR="$WORK/cache-gpu"
export PYTHONPATH="$REPO"

echo "=== node ==="; hostname; date
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"
git -C "$REPO" rev-parse HEAD
# ...and whether the tree actually *is* that commit. The k-chunked dE/dq
# was rsynced onto this checkout rather than pulled, so the hash above is
# only half the provenance; anything listed here is a local modification.
git -C "$REPO" status --porcelain -- defumat tests

# k_batch is set by hand rather than inherited. The platform default on a GPU is
# the whole k axis, and 81 k-points at 64 spinor bands is far past anything
# GPU.md has measured -- a fully batched Davidson subspace plus its real-space
# boxes is tens of GB. 8 is the calibration value; raise it once one q has run.
# dE/dq is chunked over the same k axis (--grad-k-batch, which follows
# --k-batch when it is not given). Its backward pass carries vkb(k +- q/2) and
# the states for every k-point it has in flight, and unchunked at 81 k-points it
# asked for 133 GiB on a 141 GB card -- which is how the first attempt lost
# twenty converged SCFs. The energy is now written to the JSON before the
# gradient is attempted, so a repeat of that costs the gradient alone.
python3 "$REPO/calculations/nii2-spiral/run_spiral.py" \
        "$REPO/calculations/nii2-spiral/nii2-spiral.in" \
        --pseudo-dir "$REPO/tests/data/pseudo" \
        --path MKGM --npoints 20 --q-index "$SLURM_ARRAY_TASK_ID" \
        --k-batch 8 \
        --json "$WORK/results/MKGM-q${SLURM_ARRAY_TASK_ID}.json"
status=$?
echo "exit $status"
echo "=== done ==="; date
# Propagate it: `echo "exit $?"` alone made sacct report COMPLETED 0:0 for
# twenty tasks that had all died, which is why the first failure went unnoticed
# until the results directory was found empty.
exit $status
