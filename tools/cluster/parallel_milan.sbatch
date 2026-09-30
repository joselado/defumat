#!/bin/bash
# Phase 0b, 0c and 0f of the parallel plan, and the Si40 target, on one Milan node.
#
# Everything the k-point pools were measured at so far is six desktop cores
# (PERFORMANCE.md, "k-point pools against pw.x -nk"); this is the server-node
# measurement those numbers were always a stand-in for. Four blocks, in the
# order their answers are needed, each appending JSON lines to its own file so
# that a job cut short still leaves the finished blocks behind:
#
#   topology   lscpu -e, lscpu -p, numactl -H: which cores share an L3, which
#              a socket, which a NUMA node. Every CPU list below is built from
#              it by tools/parallel/cpu_layout.py.
#   threads    (0b) one process, warm SCF per iteration, at T = 1 to 64 cores
#              on si16-1k-ecut30, si64-1k-ecut30 and bi20-soc (fixed three
#              iterations, it is 105 s an iteration in pw.x on one core), and
#              one band_batch crossing on the largest. This is where the pool
#              width comes from.
#   copies     (0c) P independent single-k copies at once, P x T = 128 at T of
#              1, 2 and 4 on si32-1k-ecut30, and 32 x 4 and 16 x 8 on si64,
#              each beside the same copy alone. The ideal pool throughput with no
#              communication at all, and the memory-bandwidth ceiling of two
#              sockets with every core streaming.
#   pools      (0f and the target) the Si40 cell without symmetry at 27
#              k-points, this code at 1, 27 x 1, 27 x 2, 25 x 2 and 9 x 6 cores
#              against pw.x at 1 rank, -nk 27 on 27 and on 54 ranks, -nk 25 on
#              50 and -nk 9 on 54; then the three D22 cells (si16-12k, si32-6k,
#              si16-spinor-6k) at six pools against -nk 6, so the desktop
#              percentages can be read against a server core.
#
#   sbatch tools/cluster/parallel_milan.sbatch
#   BLOCKS="pools" sbatch tools/cluster/parallel_milan.sbatch   # one block only
#
# ---- the #SBATCH values ----
#
# partition   batch-milan: 2 sockets of 64 cores, ThreadsPerCore=1, 510 GB.
# exclusive   a timing on a shared node shares its L3 and memory channels with
#             whatever else runs there (milan6 read CPULoad=128.78 on 2026-09-30);
#             Triton's reference lists --exclusive "for large parallel jobs".
# ntasks      128 x 1 CPU, so that mpirun sees 128 slots for pw.x (it dies in the
#             launcher under --ntasks=1, CLAUDE.local.md); this code's processes
#             are started by the scripts and pinned with taskset.
# mem         480G of the 510: 128 si32 copies at about 1 GB, bi20-soc at 35 GB.
# time        10:00:00 against an estimate of 5 to 6 hours from D22's per-core
#             speeds; the blocks write as they go.
#SBATCH --job-name=defumat-par-milan
#SBATCH --partition=batch-milan
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=480G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/work/ladovj1/calculations/parallel-milan/out/milan-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/parallel-milan/out/milan-%j.err

set -u
REPO=${DEFUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-jobs}
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv
WORK=/scratch/work/ladovj1/calculations/parallel-milan/$SLURM_JOB_ID
BLOCKS=${BLOCKS:-"topology threads copies pools"}
mkdir -p "$WORK" || exit 1
cd "$WORK" || exit 1

export PYTHON=$VENV/bin/python3
export DEFUMAT_CACHE_DIR=/tmp/defumat-jax-$SLURM_JOB_ID
export QE_BIN=/scratch/work/ladovj1/apps/qe-env/bin QE_RUNS=$WORK/qe-runs
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
P=$REPO/tools/parallel
IN=$REPO/tools/parallel/inputs
B=$REPO/benchmarks
echo "repo $(git -C "$REPO" rev-parse --short HEAD) on $(hostname), blocks: $BLOCKS"

has() { case " $BLOCKS " in *" $1 "*) return 0 ;; esac; return 1; }
first() { "$PYTHON" "$P/cpu_layout.py" 1 "$1"; }

if has topology; then
  { lscpu; lscpu -e; } > topology.txt 2>&1
  lscpu -p=CPU,CORE,SOCKET,NODE,CACHE > topology.csv 2>&1
  numactl -H >> topology.txt 2>&1 || echo "numactl not present" >> topology.txt
  "$PYTHON" "$P/cpu_layout.py" --describe > layout.txt
fi

if has threads; then
  export OUT=$WORK/threads.jsonl
  masks=""
  for T in 1 2 4 8 16 32 64; do masks+=" T$T:$(first $T)"; done
  MASKS="$masks" CELLS="$B/si16-1k-ecut30.in" REPEATS=5 "$P/thread_scan.sh"
  MASKS="$masks" CELLS="$B/si64-1k-ecut30.in" REPEATS=3 "$P/thread_scan.sh"
  MASKS="$masks T16-bb16:$(first 16):16" CELLS="$B/bi20-soc.in" REPEATS=2 \
    MAX_ITERATIONS=3 "$P/thread_scan.sh"
fi

if has copies; then
  export OUT=$WORK/copies.jsonl
  for T in 1 2 4; do
    "$P/copies.sh" "$B/si32-1k-ecut30.in" si32 1 $T
    "$P/copies.sh" "$B/si32-1k-ecut30.in" si32 $((128 / T)) $T
  done
  for T in 4 8; do
    REPEATS=2 "$P/copies.sh" "$B/si64-1k-ecut30.in" si64 1 $T
    REPEATS=2 "$P/copies.sh" "$B/si64-1k-ecut30.in" si64 $((128 / T)) $T
  done
fi

if has pools; then
  export OUT=$WORK/pools.jsonl
  si40=$IN/si40-27k.in
  REPEATS=1 "$P/run_pools.sh" "$si40" si40-1x1 1 1
  for layout in "27 1" "27 2" "25 2" "9 6"; do
    set -- $layout
    "$P/run_pools.sh" "$si40" "si40-$1x$2" "$1" "$2"
  done
  "$P/run_qe.sh" "$si40" qe-np1 1 1
  "$P/run_qe.sh" "$si40" qe-np27-nk27 27 27
  "$P/run_qe.sh" "$si40" qe-np54-nk27 54 27
  "$P/run_qe.sh" "$si40" qe-np50-nk25 50 25
  "$P/run_qe.sh" "$si40" qe-np54-nk9 54 9
  for cell in si16-12k si32-6k si16-spinor-6k; do
    "$P/run_pools.sh" "$IN/$cell.in" "$cell-1x1" 1 1
    "$P/run_pools.sh" "$IN/$cell.in" "$cell-6x1" 6 1
    "$P/run_qe.sh" "$IN/$cell.in" qe-np1 1 1
    "$P/run_qe.sh" "$IN/$cell.in" qe-np6-nk6 6 6
  done
  "$P/run_pools.sh" "$IN/si16-12k.in" si16-12k-12x1 12 1
  "$P/run_qe.sh" "$IN/si16-12k.in" qe-np12-nk12 12 12
fi
echo "done $(date)"
