#!/bin/bash
# GPU.md Phase 1, continued -- the CPU half: the baseline the GPU is quoted
# against, on a *stated* core count.
#
# The SAME script on the SAME cases as davidson-gpu.sbatch, for §2.3's reason:
# GPU pypresso against CPU pypresso, same input, same code, and a separate job
# because CPU-only work does not belong on a GPU partition.
#
# It carries a second deliverable that needs no card. The three candidates
# davidson_profile.py ranks are (a) sizing the projected solve by the live basis,
# (b) sizing the expansion by notcnv, and (c) the guard's second branch -- and
# **two of the three are constrained by the same JAX fact**: lax.switch, like
# lax.cond, executes every branch under vmap, so (a) and (b) can only ever be
# taken on the *unbatched* path. On a CPU that is the default path (k_batch=1),
# which makes this job the one that says what they are worth where they apply.
# The GPU half then says whether the same shapes behave the same way on a device.
#
# ---- the #SBATCH values ----
#
# partition   batch-milan (AMD EPYC Milan), as phase0-cpu.sbatch and
#             phase5-cpu.sbatch settled on. **Re-check with `sbatch --test-only`
#             rather than inheriting this reading.**
# time        06:00:00, twice the GPU half: the SCF each profile needs first has
#             no device to make it cheap, and si64-1k-ecut30 on four cores is
#             the case that sets this budget. Each case carries its own timeout.
# cpus-per-task 4, the stated core count, matching davidson-gpu.sbatch. pypresso
#             narrows its own affinity mask to four on import and only narrows.
# mem         120G, matching the GPU half. This is the ground state; no
#             reverse-mode tape is taken here.
#SBATCH --job-name=pypresso-cpu-davidson
#SBATCH --partition=batch-milan
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --output=/scratch/work/ladovj1/calculations/pypresso-gpu/out/cpu-davidson-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/pypresso-gpu/out/cpu-davidson-%j.err

WORK=/scratch/work/ladovj1/calculations/pypresso-gpu
REPO=/scratch/work/ladovj1/apps/pypresso
mkdir -p "$WORK/out" "$WORK/cache-cpu"
cd "$REPO" || exit 1
source "$WORK/venv/bin/activate"

export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYPRESSO_CACHE_DIR="$WORK/cache-cpu"
export PYTHONPATH="$REPO"
export JAX_PLATFORMS=cpu   # a CPU node has no device; say so rather than discover it

echo "=== node ==="; hostname; date
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core"
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"
python3 -c "import os; print('affinity:', sorted(os.sched_getaffinity(0)))"
git -C "$REPO" rev-parse HEAD

# The dials are left unset here too, so each side inherits its platform's
# default -- QE's loop on this one, which is what every committed CPU number in
# PERFORMANCE.md was measured under. **The (c) measurement needs the batched
# mode**, which a CPU does not default to, so the multi-k cases are run twice:
# once on the default and once at k=all, which is what a card does.
profile () {  # profile <case> <timeout-seconds> [extra args]
  local case=$1 budget=$2; shift 2
  echo; echo "########## profile $case ##########"
  timeout "$budget" python3 tools/gpu/davidson_profile.py "$case" \
          --repeats 20 --conv-thr 1e-8 "$@" \
          --json "$WORK/out/cpu-davidson-${SLURM_JOB_ID}-$case.json"
  echo "exit $?"
}

profile si10-nc         3600 --trajectory --trajectory-steps 10
profile al10-metal      3600 --trajectory --trajectory-steps 10
profile si16-1k-ecut30  3600
profile si32-1k-ecut30  5400
profile si64-1k-ecut30  7200

# The same two multi-k cells in the *batched* mode, which is the only way a CPU
# can see (c) at all: with k_batch=1 there is no vmap and the guard's cond is a
# real branch that costs nothing.
for case in si10-nc al10-metal; do
  echo; echo "########## profile $case, k-batched ##########"
  timeout 3600 python3 tools/gpu/davidson_profile.py $case \
          --repeats 20 --conv-thr 1e-8 --k-batch all --band-batch all \
          --json "$WORK/out/cpu-davidson-kall-${SLURM_JOB_ID}-$case.json"
  echo "exit $?"
done

echo; echo "=== done ==="; date
