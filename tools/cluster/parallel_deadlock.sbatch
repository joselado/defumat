#!/bin/bash
# Phase 0g of the parallel plan: the thread-pool deadlock at the pools' widths, on Milan.
#
# OPEN.md Part IV item 1: a long-lived process that has compiled many cells can
# park with every XLA worker blocked and no CPU at all, at a rate that follows
# the affinity mask -- 8 of 8 at two cores, 6 of 14 at four, 0 of 8 at eight, on
# the workstation's hybrid chip, with three test files that hang together and
# pass separately. The pools want one- or two-core masks, so a hang inside a
# 100-core job is the cost to size before one is submitted. Two probes, one task
# each, all at --cpus-per-task=8 with the mask narrowed inside the task:
#
#   tasks  0-23   the three test files together, at T = 1, 2 and 8, eight each
#   tasks 24-31   tools/parallel/pool_soak.py, two pools in one task at T = 1
#                 and T = 2 (so two and four of the eight CPUs), four each: the
#                 pooled form of the same condition, thirteen cells compiled in
#                 one process
#
# A hang is `timeout` killing the run with little CPU spent; /usr/bin/time's
# user and system seconds are written beside the exit code, so a hang cannot be
# mistaken for a slow pass.
#
#   sbatch --array=0-31 tools/cluster/parallel_deadlock.sbatch
#
# ---- the #SBATCH values ----
#
# partition     batch-milan; no timing is read here, so no --exclusive.
# cpus-per-task 8, the widest mask probed; narrower ones are taskset inside.
# mem           16G: the three files peak under 6 GB on the workstation, and two
#               soak pools hold two small cells each.
# time          00:40:00: the files pass in 32 to 38 s, the soak in a few
#               minutes, and each probe is killed at 15 minutes.
#SBATCH --job-name=defumat-par-deadlock
#SBATCH --partition=batch-milan
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/work/ladovj1/calculations/parallel-milan/out/deadlock-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/parallel-milan/out/deadlock-%A_%a.err

set -u
REPO=${DEFUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-jobs}
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv
WORK=/scratch/work/ladovj1/calculations/parallel-milan/deadlock-$SLURM_ARRAY_JOB_ID
mkdir -p "$WORK" || exit 1
cd "$REPO" || exit 1
export PYTHONPATH=$REPO JAX_PLATFORMS=cpu DEFUMAT_THREADS=off \
       MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       DEFUMAT_CACHE_DIR=/tmp/defumat-jax-$SLURM_JOB_ID
PY=$VENV/bin/python3
ID=$SLURM_ARRAY_TASK_ID
mine=($("$PY" -c "import os; print(' '.join(map(str, sorted(os.sched_getaffinity(0)))))"))
cpus() { local n=$1 i s=""; for ((i = 0; i < n; i++)); do s+="${s:+,}${mine[$i]}"; done; echo "$s"; }
record() { echo "{\"task\": $ID, \"probe\": \"$1\", \"T\": $2, \"exit\": $3, \"time\": \"$4\"}" >> "$WORK/summary.jsonl"; }

if [ "$ID" -lt 24 ]; then
  widths=(1 2 8); T=${widths[$((ID / 8))]}
  log=$WORK/files-T$T-$ID
  /usr/bin/time -f "%e s wall %U s user %S s sys %M kB" -o "$log.time" \
    timeout -s KILL 900 taskset -c "$(cpus "$T")" "$PY" -m pytest -q -p no:cacheprovider \
      tests/unit/test_seed_symmetry.py tests/unit/test_textured_symmetry.py \
      tests/unit/test_textured_seeding.py > "$log.out" 2>&1
  status=$?
  record files "$T" $status "$(tail -1 "$log.time")"
else
  widths=(1 2); T=${widths[$(((ID - 24) / 4))]}
  log=$WORK/soak-T$T-$ID
  port=$((20000 + RANDOM % 20000))
  pids=()
  for r in 0 1; do
    list=$(cpus $((2 * T)) | cut -d, -f$((r * T + 1))-$(((r + 1) * T)))
    DEFUMAT_POOLS=2 DEFUMAT_POOL_RANK=$r DEFUMAT_COORDINATOR=localhost:$port \
      /usr/bin/time -f "%e s wall %U s user %S s sys %M kB" -o "$log.r$r.time" \
      timeout -s KILL 900 taskset -c "$list" "$PY" tools/parallel/pool_soak.py \
        > "$log.r$r.out" 2> "$log.r$r.err" &
    pids+=($!)
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
  record soak "$T" $status "$(tail -1 "$log.r0.time") | $(tail -1 "$log.r1.time")"
fi
