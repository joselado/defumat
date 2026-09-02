#!/bin/bash
# GPU.md Phase 1, continued: *inside* a Davidson step, and what the guard costs.
#
# Phase 1 profiled the SCF by stage and named what it could not resolve there --
# "the small dense eigh and matmuls inside a lax.while_loop" -- and measured the
# consequence: tightening conv_thr from 1e-8 to 1e-10 costs the CPU nothing per
# iteration and costs the GPU 13x at sixteen atoms. That made the subspace
# algebra the first thing worth attacking, but "the subspace algebra" is four
# operations with four different fixes and a stage timing cannot say which.
#
# tools/gpu/davidson_profile.py resolves it at each case's real shapes, and this
# job runs it on a card. Three candidates, all of them things cegterg does that
# static shapes gave up, and the profile is here to rank them rather than to
# justify one:
#
#   (a) size the projected solve and the two Ritz rotations by the *live* basis
#       rather than by nvecx = 4 nbnd. On the CPU this is worth ~7% of a step at
#       sixteen atoms; the reason to measure it on a device is that si64's
#       committed stage timings are h_psi 0.012 s against Davidson 0.640 s, so
#       the subspace algebra is ~80% of the solve there and this is aimed at it.
#   (b) size the *expansion* by notcnv. Measured premise, on this workstation:
#       in the seeded regime the SCF actually runs, notcnv falls 20 -> 13 -> 10
#       -> 0 over four steps, and h_psi is exactly linear in the block's width.
#       Whether it is linear on a device with band_batch=all is the question --
#       if a narrow block is launch-bound there, (b) is worth nothing.
#   (c) is already fixed and this job measures the fix. See below.
#
# **What (c) was.** The 64-atom NaN fix (a351005) put a lax.cond in
# generalised_eigh between the Cholesky route and canonical orthogonalisation.
# Under vmap a cond with a batched predicate lowers to select_n over *both*
# branches -- and k_batch=None, the GPU default since e562427, is a vmap over
# the k axis. So every multi-k GPU subspace solve has been paying the fallback
# it does not use. Measured on this workstation at si10-nc's shapes: 42.5 ms
# against 14.9 for the Cholesky route alone, 2.85x. The guard now lives one
# level out, outside map_k, where its predicate is a scalar and lax.cond is a
# branch again; the returned values are bit-for-bit unchanged (verified: same
# total energy to 12 digits and the same eigenvalue SHA-256 on si10-nc).
#
# **So the second half of this job is a before/after with a committed before.**
# The physics sweep (PERFORMANCE.md, "The physics sweep on a GPU") ran phase0.py
# on these same cases at c5dc7d4 -- which is downstream of a351005 and therefore
# paid the doubled solve -- on an H200 at k=all, b=all. Re-running the identical
# invocation here gives the after column against a before that already exists,
# on the same card and the same dials. The rows to beat, in ms/iteration:
# si10-nc 30, si10-us 43, si10-paw 57, al10-metal 127, ni10-ldau 321.
#
# Its pair is davidson-cpu.sbatch: the same script on four pinned Milan cores,
# which is §2.3's metric (GPU defumat against CPU defumat, same code).
#
# ---- the #SBATCH values, and why each one ----
#
# partition   gpu-h200-141g-short, as sweep-gpu.sbatch and phase5-gpu.sbatch
#             use. si64-1k-ecut30 is the largest case here and it is the one the
#             (a) question is really about. **Verify with `sbatch --test-only`
#             rather than inheriting this reading** -- partitions and limits are
#             the cluster's to set, not this repository's.
# time        03:00:00. The profile itself is minutes per case; what dominates
#             is the SCF each one needs first, plus the trajectory's
#             recompilation per step. Each case carries its own `timeout`, so an
#             overrun is a reported case and not a lost job.
# cpus-per-task 4, matching davidson-cpu.sbatch exactly: defumat narrows its
#             own affinity mask to four on import, so both sides get the same
#             host-side threading.
# mem         120G host memory. Nothing here is a reverse-mode tape; this is the
#             ground state, and si64 peaked at 3.28 GB of device memory.
#SBATCH --job-name=defumat-gpu-davidson
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --time=03:00:00
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-gpu/out/gpu-davidson-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-gpu/out/gpu-davidson-%j.err

WORK=/scratch/work/ladovj1/calculations/defumat-gpu
REPO=/scratch/work/ladovj1/apps/defumat
mkdir -p "$WORK/out" "$WORK/cache-gpu"
cd "$REPO" || exit 1
source "$WORK/venv/bin/activate"

# Single-tenant job: XLA preallocating the bulk of the card is what we want.
unset XLA_PYTHON_CLIENT_PREALLOCATE
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export DEFUMAT_CACHE_DIR="$WORK/cache-gpu"
export PYTHONPATH="$REPO"

echo "=== node ==="; hostname; date
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"
git -C "$REPO" rev-parse HEAD

# ---------------------------------------------------------------- the profile
#
# The dials are left unset so that each case runs in the mode a bare run_scf
# picks on the platform -- which on a card is k=all, b=all, and is the mode (c)
# is about. davidson_profile.py records what it resolved them to.
profile () {  # profile <case> <timeout-seconds> [extra args]
  local case=$1 budget=$2; shift 2
  echo; echo "########## profile $case ##########"
  timeout "$budget" python3 tools/gpu/davidson_profile.py "$case" \
          --repeats 20 --conv-thr 1e-8 "$@" \
          --json "$WORK/out/gpu-davidson-${SLURM_JOB_ID}-$case.json"
  echo "exit $?"
}

# Multi-k first: these are the cases (c) changed, and the trajectory is cheap on
# them. --trajectory recompiles the solver once per step, so it is asked for
# only where the cell is small enough to afford it.
profile si10-nc         1800 --trajectory --trajectory-steps 10
profile al10-metal      1800 --trajectory --trajectory-steps 10

# Then single-k and large, which is where (a) and (b) are aimed. No trajectory:
# the recompilation per step is what would overrun, and the notcnv premise is
# already established on the cells above.
profile si16-1k-ecut30  1800
profile si32-1k-ecut30  2400
profile si64-1k-ecut30  3600

# --------------------------------------------------- the before/after on (c)
#
# The identical invocation the sweep used, so that its committed ms/iteration is
# the before column. --max-iterations 200 because 100 is a cap and not a
# convergence failure (ni10-ldau needs 151).
for case in si10-nc si10-us si10-paw al10-metal ni10-ldau; do
  echo; echo "########## scf $case ##########"
  timeout 1800 python3 tools/gpu/phase0.py $case --k-batch all --band-batch all \
          --repeats 2 --skip-precision-probe --max-iterations 200 \
          --json "$WORK/out/gpu-davidson-scf-${SLURM_JOB_ID}-$case.json"
  echo "exit $?"
done

echo; echo "=== done ==="; date
