#!/bin/bash
# GPU.md Phase 5, continued -- the CPU half: the baseline, and one diagnostic
# that PERFORMANCE.md records as unexplained and not run.
#
# The SAME script on the SAME cases as phase5-si10-gpu.sbatch, on a *stated*
# core count, which is §2.3's metric. The ten-atom response is what settles
# whether the Sternheimer path's 1.0x on an H200 is a fact about the two-atom
# cells it was measured on or a ceiling.
#
# **The second deliverable is the alas-raman failure**, which PERFORMANCE.md
# describes as follows and then says the diagnostic is not run:
#
#   alas-raman raman died with a MemoryError raised inside the JAX computation,
#   on a node where 160 G was granted and the same job's largest sampled RSS was
#   11.7 GB (si8-us, which ran after it and succeeded). This workstation runs
#   that exact case in 2.59 GB, and the GPU node -- same jax, same commit -- ran
#   it in 5.45 GB of host RSS. So it is not the case being too large, and the
#   two cluster nodes differ in what they did with it.
#
# It is one more short case here with JAX_TRACEBACK_FILTERING=off, so that the
# traceback names the allocation instead of being filtered to the call site.
#
# ---- the #SBATCH values ----
#
# partition   batch-milan, as phase5-cpu.sbatch settled on. **Re-check with
#             `sbatch --test-only`.**
# time        10:00:00. The ten-atom response is twentyfold the two-atom one --
#             245 s and 302 s per property on four cores of the development
#             workstation, against 12 s -- and the thirty-mode dynamical matrix
#             is the row that sets this budget. Each case carries its own
#             timeout.
# cpus-per-task 4, matching the GPU half; pypresso narrows to four on import.
# mem         160G, matching phase5-cpu.sbatch -- and deliberately the *same*
#             160G the alas-raman MemoryError happened under, since reproducing
#             it is half the point of that row.
#SBATCH --job-name=pypresso-cpu-phase5-si10
#SBATCH --partition=batch-milan
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --output=/scratch/work/ladovj1/calculations/pypresso-gpu/out/cpu-phase5-si10-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/pypresso-gpu/out/cpu-phase5-si10-%j.err

WORK=/scratch/work/ladovj1/calculations/pypresso-gpu
REPO=/scratch/work/ladovj1/apps/pypresso
mkdir -p "$WORK/out" "$WORK/cache-cpu"
cd "$REPO" || exit 1
source "$WORK/venv/bin/activate"

export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYPRESSO_CACHE_DIR="$WORK/cache-cpu"
export PYTHONPATH="$REPO"
export JAX_PLATFORMS=cpu

echo "=== node ==="; hostname; date
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core"
free -g | head -2
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"
python3 -c "import os; print('affinity:', sorted(os.sched_getaffinity(0)))"
git -C "$REPO" rev-parse HEAD

run () {  # run <case> <property> <timeout-seconds>
  echo; echo "########## $1 $2 ##########"
  timeout "$3" python3 tools/gpu/phase5.py "$1" --property "$2" --repeats 2 \
          --json "$WORK/out/cpu-phase5-si10-${SLURM_JOB_ID}-$1.$2.json"
  echo "exit $?"
}

run si10-epsilon  dielectric  5400
run si10-epsilon  born        5400
run si10-epsilon  phonon      10800
run si-epsilon    dielectric  1800
run si-epsilon    phonon      2400

# ---------------------------------------------- the unexplained MemoryError
#
# Last, and on its own, because it is expected to fail and the useful output is
# the traceback rather than a row. The unfiltered traceback names the allocation
# JAX raised on; /usr/bin/time -v gives the kernel's own view of the peak, which
# is the number to put against the 11.7 GB the job that failed had already
# sampled. If it *succeeds* here, that is also the answer -- the failure was the
# node, and the two runs' `free -g` and `ulimit -v` above say how they differed.
echo; echo "########## alas-raman raman -- the unexplained MemoryError ##########"
export JAX_TRACEBACK_FILTERING=off
ulimit -a
timeout 10800 /usr/bin/time -v python3 tools/gpu/phase5.py alas-raman --property raman \
        --repeats 1 --json "$WORK/out/cpu-phase5-si10-${SLURM_JOB_ID}-alas-raman.json"
echo "exit $?"

echo; echo "=== done ==="; date
