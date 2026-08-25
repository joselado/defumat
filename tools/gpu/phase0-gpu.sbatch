#!/bin/bash
# GPU.md Phase 0, GPU half: first contact with a real device.
#
# The deliverable of this job is not a speedup. It is a job that runs and a
# number that matches -- so it runs `tools/gpu/phase0.py`, which encodes
# GPU.md's five checks, and writes one JSON per (case, dial setting).
#
# Its pair is phase0-cpu.sbatch, which runs the SAME script on a CPU node.
# That is deliberate and is §2.3's metric: GPU pypresso against CPU pypresso,
# same input, same code. CPU-only work is not run on a GPU partition.
#
# ---- the #SBATCH values, and why each one ----
#
# partition   gpu-debug: 30-minute limit, which is the shape of first contact --
#             it must not wait behind a long queue. Verify the queue with
#             `sbatch --test-only` before submitting rather than inheriting a
#             stale reading; gpu-h200-141g-short is the fallback when gpu-debug
#             is full. No compute-capability floor is set: every GPU generation
#             on this cluster is a data-centre part with a usable fp64 rate, and
#             which one this job landed on is *recorded by the driver* (check 1)
#             rather than constrained in advance.
# time        25 min inside gpu-debug's 30. The six invocations below cost about
#             two CPU minutes of arithmetic; the rest of the budget is compilation,
#             which on a first GPU run is the unknown.
# cpus-per-task 4, matching phase0-cpu.sbatch exactly. pypresso narrows its own
#             affinity mask to four CPUs on import and only ever narrows
#             (PERFORMANCE.md: four is the measured optimum, 1.7x over the
#             default), so both sides get the same host-side threading and the
#             GPU number is not flattered or penalised by it.
# mem         32G host memory. Device memory is the constraint that matters and
#             is measured, not requested.
#SBATCH --job-name=pypresso-gpu-phase0
#SBATCH --partition=gpu-debug
#SBATCH --time=00:25:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/work/ladovj1/calculations/pypresso-gpu/out/gpu-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/pypresso-gpu/out/gpu-%j.err

WORK=/scratch/work/ladovj1/calculations/pypresso-gpu
REPO=/scratch/work/ladovj1/apps/pypresso
mkdir -p "$WORK/out" "$WORK/cache-gpu"
cd "$REPO" || exit 1

# A batch shell does not source .bashrc, so the environment is built here.
module load scicomp-python-env/2025.2
# No cuda module: JAX's CUDA libraries come from the nvidia-* wheels inside that
# environment, and loading a second CUDA on LD_LIBRARY_PATH is a way to break
# them. (Verified on the login node: the plugin loads and reaches cuInit, which
# then fails with 303 only because a login node has no device.)
source "$WORK/venv/bin/activate"

# Not what the sibling dmrgpy job sets. XLA preallocating the bulk of the device
# is what a single-tenant job WANTS -- it avoids fragmentation over a long SCF --
# and PREALLOCATE=false is for the case where another framework shares the card.
# Nothing shares this one. GPU.md §3, Phase 0.
unset XLA_PYTHON_CLIENT_PREALLOCATE

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYPRESSO_CACHE_DIR="$WORK/cache-gpu"
export PYTHONPATH="$REPO:$PYTHONPATH"

echo "=== node ==="; hostname; date
nvidia-smi
echo "=== jax ==="
python3 -c "import jax; print(jax.__version__, jax.devices())"
git -C "$REPO" rev-parse HEAD

# si-1k first because two atoms and one k-point is debuggable; al10-metal next
# and at both ends of both dials, because it is the only case here with more
# than one k-point and so the only one on which either dial means anything;
# si8-1k last because it is the smallest cell whose cost is physics rather than
# fixed overhead, and the two-atom cell shows none of that.
run () {  # run <case> <k_batch> <band_batch>
  local tag="$1-k$2-b$3"
  echo; echo "########## $tag ##########"
  python3 tools/gpu/phase0.py "$1" --k-batch "$2" --band-batch "$3" --repeats 2 \
          --json "$WORK/out/gpu-${SLURM_JOB_ID}-${tag}.json"
  echo "exit $?"
}

run si-1k      1   1
run al10-metal 1   1
run al10-metal all all
run al10-metal all 1
run si8-1k     1   1
run si8-1k     1   all

echo; echo "=== done ==="; date
