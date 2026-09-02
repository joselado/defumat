#!/bin/bash
# GPU.md Phase 0, CPU half: the baseline the GPU number is quoted against.
#
# This is the SAME driver on the SAME cases at the SAME dial settings as
# phase0-gpu.sbatch. That is the entire point of §2.3's metric -- GPU defumat
# against CPU defumat, same input, same code -- and it is why the two halves
# are one script run twice rather than two scripts.
#
# It is a separate job because CPU-only work does not belong on a GPU
# partition, and because the CPU side must be pinned to a *stated* core count:
# one core, four and a whole node differ by an order of magnitude, and an
# unpinned baseline is not a baseline. `tools/compare_qe.py` pins both codes to
# one core for the QE comparison for exactly this reason.
#
# ---- the #SBATCH values ----
#
# partition   batch-milan (AMD EPYC Milan). batch-csl was the first choice --
#             Cascade Lake is the same class of host CPU the GPU nodes carry, so
#             the comparison would not also be a comparison of two CPU
#             generations -- but `sbatch --test-only` put its start ~1h50m out
#             on 2026-08-25 where batch-milan was immediate, and the metric asks
#             for a CPU side pinned to a *stated* core count on a *stated* part,
#             not for a particular part. So: four cores of an EPYC Milan, said
#             out loud, and re-check the queue rather than inheriting this.
# time        40 min: the batched dial settings are the slow ones on a CPU
#             (al10-metal at k_batch=all was measured at 2.2x the loop on the
#             development workstation) and this side has no device to make them
#             cheap. Still far inside the partition's limit.
# cpus-per-task 4, and this is the stated core count. defumat narrows its own
#             affinity mask to four CPUs on import and only ever narrows, so the
#             scheduler's allocation is what it gets; four is also the measured
#             optimum (PERFORMANCE.md: 1.7x over XLA's every-core default, and
#             8 and 14 cores are both worse than 1).
#SBATCH --job-name=defumat-cpu-phase0
#SBATCH --partition=batch-milan
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-gpu/out/cpu-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-gpu/out/cpu-%j.err

WORK=/scratch/work/ladovj1/calculations/defumat-gpu
REPO=/scratch/work/ladovj1/apps/defumat
mkdir -p "$WORK/out" "$WORK/cache-cpu"
cd "$REPO" || exit 1

# A self-contained venv, no module -- see phase0-gpu.sbatch for why.
source "$WORK/venv/bin/activate"

# Its own cache: a CPU kernel and a GPU kernel are different compilations, and
# giving the two jobs one directory would also be the concurrent-writer case
# GPU.md flags as unverified on this filesystem. One writer each, no question.
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export DEFUMAT_CACHE_DIR="$WORK/cache-cpu"
# The venv is isolated, so this is the only thing that makes the
# checkout importable; defumat is deliberately not pip-installed into
# it, so the code that runs is the checkout at the commit logged below.
export PYTHONPATH="$REPO"
export JAX_PLATFORMS=cpu   # a CPU node has no device; say so rather than discover it

echo "=== node ==="; hostname; date
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core"
echo "=== jax ==="
python3 -c "import jax; print(jax.__version__, jax.devices())"
python3 -c "import os; print('affinity:', sorted(os.sched_getaffinity(0)))"
git -C "$REPO" rev-parse HEAD

run () {  # run <case> <k_batch> <band_batch>
  local tag="$1-k$2-b$3"
  echo; echo "########## $tag ##########"
  python3 tools/gpu/phase0.py "$1" --k-batch "$2" --band-batch "$3" --repeats 2 \
          --json "$WORK/out/cpu-${SLURM_JOB_ID}-${tag}.json"
  echo "exit $?"
}

run si-1k      1   1
run al10-metal 1   1
run al10-metal all all
run al10-metal all 1
run si8-1k     1   1
run si8-1k     1   all

echo; echo "=== done ==="; date
