#!/bin/bash
# Do P58's force-theorem anisotropy and P59's local-TF mixer run on a GPU?
#
# GPU.md Phase 0's two questions asked of the work added since it was written.
# `tools/gpu/p59_check.py` is the same script the CPU reference was produced
# with, so the deliverable is a JSON pair to diff -- not a speedup.
#
# Three of its four checks need no reference at all and hold on any backend: a
# preconditioner may not move the SCF's fixed point, and with the spin-orbit
# coupling scaled to zero the anisotropy is exactly zero by spin-rotation
# invariance. Only the fourth, the anisotropy at full coupling, is compared
# against a number.
#
# ---- the #SBATCH values, and why each one ----
#
# partition   gpu-debug: 30-minute limit, checked up with 51 nodes. This is a
#             smoke test, so it must not wait behind a long queue. Verify with
#             `sbatch --test-only` before submitting rather than trusting this
#             comment; gpu-h100-80g-short is the fallback if gpu-debug is full.
# time        25 min inside gpu-debug's 30. The arithmetic is about three CPU
#             minutes on the workstation; the rest of the budget is compilation,
#             which is what a first run of *this* code on a card has to pay.
# cpus-per-task 4, matching what phase0 used, so the host-side threading is the
#             same one every earlier GPU number here was taken with.
# mem         32G host. Device memory is measured by the script, not requested.
#SBATCH --job-name=pypresso-p59
#SBATCH --partition=gpu-debug
#SBATCH --time=00:25:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/work/ladovj1/calculations/pypresso-gpu/out/p59-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/pypresso-gpu/out/p59-%j.err

WORK=/scratch/work/ladovj1/calculations/pypresso-gpu
# A SEPARATE checkout from /scratch/work/ladovj1/apps/pypresso, deliberately:
# that one is 13 commits behind and carries its own uncommitted spiral work in
# pypresso/scf/driver.py, which this code also touches. Nothing here writes to it.
REPO=/scratch/work/ladovj1/apps/pypresso-p59
mkdir -p "$WORK/out" "$WORK/cache-p59"
cd "$REPO" || exit 1

# A batch shell does not source .bashrc. The venv is the self-contained one
# phase0 built -- no module load, and no cuda module either: JAX's CUDA
# libraries come from its own nvidia-* wheels and a second CUDA on
# LD_LIBRARY_PATH breaks them.
source "$WORK/venv/bin/activate"

unset XLA_PYTHON_CLIENT_PREALLOCATE
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYPRESSO_CACHE_DIR="$WORK/cache-p59"
export PYTHONPATH="$REPO"
export P59_OUT="$WORK/out/p59-gpu-${SLURM_JOB_ID}.json"

echo "=== node ==="; hostname; date
nvidia-smi
echo "=== jax ==="
python3 -c "import jax; print(jax.__version__, jax.devices())"
echo "=== code under test ==="
md5sum pypresso/scf/mixing.py pypresso/workflows/anisotropy.py

echo "=== p59_check ==="
python3 -u tools/gpu/p59_check.py
echo "=== done ==="; date
