#!/bin/bash
# GPU.md Phase 5, CPU half: the baseline the GPU number is quoted against.
#
# The SAME driver on the SAME cases as phase5-gpu.sbatch, for §2.3's reason --
# GPU pypresso against CPU pypresso, same input, same code -- and a separate job
# because CPU-only work does not belong on a GPU partition and because the CPU
# side must be pinned to a *stated* core count.
#
# It also carries a second deliverable of its own, and this one needs no card at
# all: GPU.md §4 item 3, the tape measurement per property. `phase5.py` reports
# the peak host RSS over the SCF for each, which is the same measurement
# PERFORMANCE.md's "What a stress costs" section is built from -- so these rows
# extend that table rather than starting a new scale.
#
# ---- the #SBATCH values ----
#
# partition   batch-milan (AMD EPYC Milan), as phase0-cpu.sbatch settled on:
#             four cores of a stated part, with the queue re-checked by
#             `sbatch --test-only` rather than inherited from this comment.
# time        08:00:00. Twice the GPU half's budget on purpose -- the response
#             is the expensive half and this side has no device to make the
#             Sternheimer solves cheap. Each case carries its own `timeout`, so
#             an overrun is a reported case rather than a lost job.
# cpus-per-task 4, the stated core count, matching phase5-gpu.sbatch. pypresso
#             narrows its own affinity mask to four on import and only ever
#             narrows; four is the measured optimum (PERFORMANCE.md).
# mem         160G, matching the GPU half. si8-us's reverse tape alone is 11 GB
#             on this workstation and it is the row this job exists to bound.
#SBATCH --job-name=pypresso-cpu-phase5
#SBATCH --partition=batch-milan
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --output=/scratch/work/ladovj1/calculations/pypresso-gpu/out/cpu-phase5-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/pypresso-gpu/out/cpu-phase5-%j.err

WORK=/scratch/work/ladovj1/calculations/pypresso-gpu
REPO=/scratch/work/ladovj1/apps/pypresso
mkdir -p "$WORK/out" "$WORK/cache-cpu"
cd "$REPO" || exit 1

# A self-contained venv, no module -- see phase0-gpu.sbatch for why.
source "$WORK/venv/bin/activate"

# Its own cache: a CPU kernel and a GPU kernel are different compilations.
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYPRESSO_CACHE_DIR="$WORK/cache-cpu"
export PYTHONPATH="$REPO"
export JAX_PLATFORMS=cpu   # a CPU node has no device; say so rather than discover it

echo "=== node ==="; hostname; date
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core"
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"
python3 -c "import os; print('affinity:', sorted(os.sched_getaffinity(0)))"
git -C "$REPO" rev-parse HEAD

# The dials are left unset here too, so that both halves inherit their
# platform's default -- which on this side is QE's loop, and is what every
# committed CPU response number was measured under.
run () {  # run <case> <property> <timeout-seconds>
  echo; echo "########## $1 $2 ##########"
  timeout "$3" python3 tools/gpu/phase5.py "$1" --property "$2" --repeats 2 \
          --json "$WORK/out/cpu-phase5-${SLURM_JOB_ID}-$1.$2.json"
  echo "exit $?"
}

run si2-nc-stress  force       900
run si2-nc-stress  stress      1200
run si-epsilon     dielectric  3600
run si-epsilon     born        3600
run si-epsilon-us  dielectric  5400
run si-epsilon     phonon      5400
run alas-raman     raman       7200
run si8-us         stress      7200

echo; echo "=== done ==="; date
