#!/bin/bash
# GPU.md Phase 5, GPU half: the response path on a device.
#
# Everything measured on a GPU so far is the ground state. This is the half
# CLAUDE.md's "Why JAX" lists as reason **one** -- the dielectric tensor, the
# Born charges, the phonons, the Raman tensors -- and the question it asks is
# feasibility before speed:
#
#   * a **reverse** derivative tapes the whole forward pass, and the stress on
#     eight-atom ultrasoft silicon is 11 GB of it on a CPU: the largest single
#     allocation anywhere in this code. On a card that decides whether the
#     calculation exists;
#   * a **forward** one (the Sternheimer solves behind epsilon and Z*) carries a
#     tangent and tapes nothing;
#   * a **forward-over-reverse** one -- the dynamical matrix and the Raman
#     tensors -- tapes the inner reverse pass, and how much that costs is what
#     this job measures rather than assumes.
#
# Its pair is phase5-cpu.sbatch, which runs the SAME script on a CPU node.
# §2.3's metric is GPU pypresso against CPU pypresso, same input, same code.
#
# ---- the #SBATCH values, and why each one ----
#
# partition   gpu-h200-141g-short, as sweep-gpu.sbatch uses: this is hours of
#             work, not gpu-debug's 30 minutes, and the H200's 141 GB is the
#             point on the si8-us row -- the tape is the measurement. Verify with
#             `sbatch --test-only` rather than inheriting this reading.
# time        04:00:00, matching the sweep. Each case below carries its own
#             `timeout` so one overrun is a reported case and not a lost job.
# cpus-per-task 4, matching phase5-cpu.sbatch exactly, for phase0-gpu.sbatch's
#             reason: pypresso narrows its own affinity mask to four CPUs on
#             import, so both sides get the same host-side threading.
# mem         160G host memory. Device memory is what is being measured, not
#             requested -- but a response through the radial transforms also
#             holds host-side setup, so this is not the place to be tight.
#SBATCH --job-name=pypresso-gpu-phase5
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --time=04:00:00
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --output=/scratch/work/ladovj1/calculations/pypresso-gpu/out/gpu-phase5-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/pypresso-gpu/out/gpu-phase5-%j.err

WORK=/scratch/work/ladovj1/calculations/pypresso-gpu
REPO=/scratch/work/ladovj1/apps/pypresso
mkdir -p "$WORK/out" "$WORK/cache-gpu"
cd "$REPO" || exit 1

# The self-contained venv, not an overlay -- phase0-gpu.sbatch records why, and
# it is the same environment here.
source "$WORK/venv/bin/activate"

# Single-tenant job: XLA preallocating the bulk of the card is what we want.
unset XLA_PYTHON_CLIENT_PREALLOCATE

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYPRESSO_CACHE_DIR="$WORK/cache-gpu"
export PYTHONPATH="$REPO"

echo "=== node ==="; hostname; date
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"
git -C "$REPO" rev-parse HEAD

# **The dials are not set here and that is deliberate.** Since 2026-08-26
# batching.py defaults them per platform, so a bare run on a card already picks
# `all`/`all`; leaving them unset is what makes this job a test of that default
# as well as of the response path. phase5.py records what it resolved them to.
run () {  # run <case> <property> <timeout-seconds>
  echo; echo "########## $1 $2 ##########"
  timeout "$3" python3 tools/gpu/phase5.py "$1" --property "$2" --repeats 2 \
          --json "$WORK/out/gpu-phase5-${SLURM_JOB_ID}-$1.$2.json"
  echo "exit $?"
}

# Cheapest and least taped first, so that a job which runs out of time still
# answers the questions in the order GPU.md asks them.
run si2-nc-stress  force       600     # reverse, small: the control
run si2-nc-stress  stress      900     # reverse through the radial transforms
run si-epsilon     dielectric  1800    # forward, norm-conserving (P24)
run si-epsilon     born        1800    # forward, a mixed second derivative (P24b)
run si-epsilon-us  dielectric  2400    # the same, with an augmentation charge
run si-epsilon     phonon      2400    # forward-over-reverse (P25)
run alas-raman     raman       3600    # a third derivative, 64 k-points (P35)
# The tape that decides the phase, and the reason this partition and not
# gpu-debug: 11 GB on a CPU, and the first time it is asked to live in HBM.
run si8-us         stress      3600

echo; echo "=== done ==="; date
