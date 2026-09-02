#!/bin/bash
# 64 atoms: defumat on one GPU, against Quantum ESPRESSO on CPU.
#
# **This comparison is the one GPU.md §2.3 forbids by default, and it is run
# here deliberately and labelled rather than quietly.** The project's metric is
# single-core defumat against single-core QE, precisely so that a ratio
# measures *code* and not hardware; a GPU number against a serial CPU number
# measures both at once and cannot be decomposed afterwards. So the CPU twin
# (qe64-cpu.sbatch) runs pw.x at **1, 4, 16 and 32 cores** rather than at one,
# and every number is quoted with the core count that produced it. What this
# answers is "what does the accelerator buy me over a QE run I might actually
# launch", which is a fair question with an unfair-looking answer.
#
# Cell: si64-1k-ecut30 -- 64 atoms, 256 electrons, 128 bands, c/a = 8. The
# largest in benchmarks/, added for this. Verified by the folding identity
# (si64 at Gamma == si32 at {Gamma, (0,0,1/2)}) rather than by an
# energy-per-atom comparison, which single-k sampling makes invalid here.
#
# ---- #SBATCH ----
# partition  gpu-v100-32g, NOT gpu-debug: the same card family as the 8/16/32
#            ladder so the rung is comparable, but a 5-day limit instead of 30
#            minutes. The 32-atom job timed out in gpu-debug with 13 of 17
#            sections unrun, and this cell is four times that one.
# mem        96G host. Device memory is measured, not requested; 128 bands on a
#            36x36x288 box is ~0.76 GB per band-batched real-space array, so a
#            handful of them plus the Davidson subspace should sit well inside
#            a 32 GB card -- but that is an estimate and check 4 is what settles
#            it.
#SBATCH --job-name=defumat-qe64-gpu
#SBATCH --partition=gpu-v100-32g
#SBATCH --time=02:00:00
#SBATCH --gpus=v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-gpu/out/qe64gpu-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-gpu/out/qe64gpu-%j.err

WORK=/scratch/work/ladovj1/calculations/defumat-gpu
REPO=/scratch/work/ladovj1/apps/defumat
mkdir -p "$WORK/out" "$WORK/cache-gpu"
cd "$REPO" || exit 1
source "$WORK/venv/bin/activate"
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export DEFUMAT_CACHE_DIR="$WORK/cache-gpu"
export PYTHONPATH="$REPO"

echo "=== node ==="; hostname; date
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"
git -C "$REPO" rev-parse HEAD

for case in si64-1k-ecut30 si64-1k; do
  for bands in all 1; do
    echo; echo "########## $case band_batch=$bands ##########"
    python3 tools/gpu/phase0.py $case --k-batch 1 --band-batch $bands --repeats 2 \
            --skip-precision-probe \
            --json "$WORK/out/qe64gpu-${SLURM_JOB_ID}-$case-b$bands.json"
    echo "exit $?"
  done
  echo; echo "########## stages $case ##########"
  DEFUMAT_K_BATCH=1 DEFUMAT_BAND_BATCH=all python3 tools/benchmark.py "benchmarks/$case.in"
done
echo; echo "=== done ==="; date
