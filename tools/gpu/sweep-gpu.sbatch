#!/bin/bash
# A physics sweep on one H200, against Quantum ESPRESSO on one CPU core.
#
# Everything before this compared *one* kind of calculation -- unpolarised
# silicon -- at several sizes. A GPU claim that rests on that is a claim about
# `h_psi` on a norm-conserving insulator and nothing else. This sweep is the
# other axis: the pseudopotential kinds, a GGA, a metal with smearing,
# collinear and noncollinear magnetism, spin-orbit coupling and DFT+U, each of
# which reaches different code, and then three larger cells so that size and
# physics are not confounded.
#
# **Unlike everything above, these are multi-k** (7 and 10 irreducible points
# on the silicon and aluminium cells). That is deliberate: `benchmarks/` is
# single-k on purpose, which makes `k_batch` a no-op, and the k axis is exactly
# where a GPU has something to win. Both dials are set to the batched end.
#
# The baseline is one CPU core (sweep-qe.sbatch), which GPU.md §2.3 forbids
# quoting by default -- it measures hardware and code together. It is what was
# asked for; the table says so beside every number.
#SBATCH --job-name=pypresso-sweep-gpu
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --time=04:00:00
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --output=/scratch/work/ladovj1/calculations/pypresso-gpu/out/sweepgpu-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/pypresso-gpu/out/sweepgpu-%j.err

WORK=/scratch/work/ladovj1/calculations/pypresso-gpu
REPO=/scratch/work/ladovj1/apps/pypresso
mkdir -p "$WORK/out" "$WORK/cache-gpu"
cd "$REPO" || exit 1
source "$WORK/venv/bin/activate"
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYPRESSO_CACHE_DIR="$WORK/cache-gpu"
export PYTHONPATH="$REPO"

echo "=== node ==="; hostname; date
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"
git -C "$REPO" rev-parse HEAD

# Ten atoms first, across the physics -- those are the cases that decide whether
# the sweep means anything -- then the larger cells, which are the ones that
# could overrun.
CASES="si10-nc si10-us si10-paw si10-nc-pbe al10-metal h10-chain-lsda \
h10-chain-noncolin bi10-soc ni10-ldau h20-chain-lsda bi20-soc h40-chain-lsda"

for case in $CASES; do
  echo; echo "########## $case ##########"
  timeout 1800 python3 tools/gpu/phase0.py $case --k-batch all --band-batch all \
          --repeats 2 --skip-precision-probe \
          --json "$WORK/out/sweepgpu-${SLURM_JOB_ID}-$case.json"
  echo "exit $?"
done
echo; echo "=== done ==="; date
