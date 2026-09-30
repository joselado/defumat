#!/bin/bash
# Phase 0e of the parallel plan: the pools' collectives on Triton, on one node and on two.
#
# gloo over TCP is the transport (defumat/parallel.py); on the workstation an
# 83 MB all-reduce over three local processes took 352 ms. What a Milan node
# and the network between two of them give is unmeasured, and so is whether
# jax.distributed finds its peers from the venv under srun. The NiBr2 slab's
# four-component density is 83 MB and a PAW slab's is several times that, so the
# sizes are 1.5, 83 and 330 MB. Then the failure case: one rank of four, on two
# nodes, leaves before its fourth all-reduce, and the others' stderr says how
# long they took to find out. On the workstation they were aborted by the
# coordination service after its 100 s heartbeat timeout rather than hanging.
#
#   sbatch tools/cluster/parallel_comm.sbatch
#
# ---- the #SBATCH values ----
#
# nodes/tasks   2 x 16, one CPU each: the largest step is 32 processes, 16 a node.
# mem           64G a node: a 330 MB all-reduce stacks the size-long global array
#               on every process, so 16 processes hold a few GB each.
# time          01:00:00 for a few minutes of collectives and a 400 s kill test.
#SBATCH --job-name=defumat-par-comm
#SBATCH --partition=batch-milan
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/work/ladovj1/calculations/parallel-milan/out/comm-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/parallel-milan/out/comm-%j.err

set -u
REPO=${DEFUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-jobs}
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv
WORK=/scratch/work/ladovj1/calculations/parallel-milan/comm-$SLURM_JOB_ID
mkdir -p "$WORK" || exit 1
cd "$REPO" || exit 1
export PYTHONPATH=$REPO JAX_PLATFORMS=cpu DEFUMAT_THREADS=off \
       MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       DEFUMAT_CACHE_DIR=off
PY=$VENV/bin/python3
# Rank 0 (SLURM_PROCID 0) is on the first node of the *step*, so every task
# names that node as the coordinator; the port is chosen once, before srun.
step() {  # step <label> <nodes> <ntasks> [comm_bench options]
  local label=$1 nodes=$2 n=$3; shift 3
  export DEFUMAT_POOLS=$n COORD_PORT=$((20000 + RANDOM % 20000)) PY
  timeout -s KILL 600 srun --nodes="$nodes" --ntasks="$n" --cpus-per-task=1 \
      --ntasks-per-node=$(((n + nodes - 1) / nodes)) bash -c \
      'DEFUMAT_COORDINATOR=$(scontrol show hostnames "$SLURM_STEP_NODELIST" | head -1):$COORD_PORT \
         exec "$PY" tools/parallel/comm_bench.py "$@"' comm "$@" \
      >> "$WORK/comm.jsonl" 2> "$WORK/$label.err"
  echo "{\"step\": \"$label\", \"exit\": $?}" >> "$WORK/steps.jsonl"
}
step one-node-2 1 2
step one-node-8 1 8
step one-node-16 1 16
step two-nodes-2 2 2
step two-nodes-8 2 8
step two-nodes-32 2 32
start=$(date +%s)
step kill-two-nodes-4 2 4 --kill-after 3
echo "{\"step\": \"kill-two-nodes-4\", \"wall_s\": $(( $(date +%s) - start ))}" >> "$WORK/steps.jsonl"
