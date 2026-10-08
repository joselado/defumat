#!/bin/bash
# The third harmonic on CPU cores: the CPU arm of thg-gpu.sbatch, the same driver and cases.
#
# GPU.md §2.3 asks for the CPU side pinned to a stated core count. The default is four, which
# is the thread count defumat itself narrows to and the best measured on identical cores
# (PERFORMANCE.md, "Threads"); `-c 1` on the command line gives the single-core arm, and the
# job prints what it got. DEFUMAT_THREADS=off leaves Slurm's mask as it is, so XLA's pool is
# the allocation. One case per array task:
#
#   task 0   4^3, eta_t 6   18 points,  3600 steps
#   task 1   8^3, eta_t 6   100 points, 3600 steps
#
# each with the frequency-domain hierarchy at the same frequency. A batch node is
# shared unless asked otherwise, so the node's load is printed before and after; a time
# taken beside a loaded neighbour is slow by the shared L3 and memory bandwidth.
#
# Not submitted by any tool here (CLAUDE.local.md): by hand, after `sbatch --test-only`.
#
#   sbatch --array=0-1 --export=ALL,DEFUMAT_REPO=/scratch/work/ladovj1/apps/defumat-thg tools/gpu/thg-cpu.sbatch
#
# partition   batch-csl, one named partition so both tasks get the same CPU model; on
#             2026-10-08 at 10:41 `sbatch --test-only` said 10:51 there, 11:21 on batch-skl
#             and 18:19 on batch-milan.
# time        06:00:00: at D22's 18 to 25 ms a point and step on one core, the 8^3 case is
#             about two hours of propagation; the warm-up and the hierarchy are minutes.
# mem         16G; the resident set is the states of one k-point at a time, eight copies.
#SBATCH --job-name=defumat-thg-cpu
#SBATCH --partition=batch-csl
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-thg/out/cpu-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-thg/out/cpu-%A_%a.err

WORK=/scratch/work/ladovj1/calculations/defumat-thg
REPO=${DEFUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-thg}
PY=/scratch/work/ladovj1/calculations/pypresso-gpu/venv/bin/python3
mkdir -p "$WORK/out" "$WORK/cache-cpu"
cd "$REPO" || exit 1
unset DEFUMAT_K_BATCH DEFUMAT_BAND_BATCH DEFUMAT_MEMORY_MODE
export JAX_PLATFORMS=cpu PYTHONPATH="$REPO" DEFUMAT_THREADS=off
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export DEFUMAT_CACHE_DIR="$WORK/cache-cpu"
CASES=("4 6" "8 6")
set -- ${CASES[$SLURM_ARRAY_TASK_ID]}
OUT="$WORK/out/cpu-${SLURM_ARRAY_JOB_ID}-c${SLURM_CPUS_PER_TASK}-g$1-t$2.json"

echo "=== node ==="; hostname; date; grep -m1 "model name" /proc/cpuinfo
echo "cpus-per-task $SLURM_CPUS_PER_TASK, affinity $(taskset -cp $$)"
scontrol show node "$SLURMD_NODENAME" | grep -oE "CPUAlloc=[0-9]+|CPULoad=[0-9.]+"
"$PY" -c "import jax; print('jax', jax.__version__, jax.devices())"
git -C "$REPO" rev-parse HEAD

echo; echo "########## grid $1, eta_t $2 ##########"
"$PY" -u tools/realtime/thg_timing.py "$OUT" "$1" "$2" --hierarchy
echo "exit $?"
echo; date
scontrol show node "$SLURMD_NODENAME" | grep -oE "CPUAlloc=[0-9]+|CPULoad=[0-9.]+"
