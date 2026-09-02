#!/bin/bash
# 64 atoms: Quantum ESPRESSO on CPU, at four core counts, plus defumat on CPU.
#
# The CPU half of qe64-gpu.sbatch. **pw.x is run at 1, 4, 16 and 32 cores** and
# every number is reported with its core count, because a GPU quoted against an
# unstated CPU baseline is the flattering comparison GPU.md §2.3 exists to
# prevent. One core is the project's own canonical metric; four matches the
# defumat CPU baseline used all through this work; 16 and 32 are what someone
# would actually launch for a 64-atom cell.
#
# defumat is run on 4 cores too, so the table has all three corners: defumat
# GPU, defumat CPU, QE CPU. Without the middle one, a GPU-vs-QE ratio cannot be
# split into "what the accelerator bought" and "what the two codes differ by".
#
# QE 7.2 from the cluster module, not the 7.5 vendored in this repo (which is
# not on the cluster -- it is gitignored). The SCF core is materially the same;
# the version is recorded beside the number, per GPU.md §4a.
#
# ---- #SBATCH ----
# partition  batch-milan: 128 cores per node, so a 32-core task fits on one node
#            and the MPI is intra-node -- comparing across an interconnect would
#            add a variable this measurement is not about.
# disk       QE writes wavefunctions by default; disk_io='none' is set in the
#            run so that neither code is timed doing I/O the other does not do,
#            and so the shared filesystem's small-file policy is respected.
#SBATCH --job-name=defumat-qe64-cpu
#SBATCH --partition=batch-milan
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=192G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-gpu/out/qe64cpu-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-gpu/out/qe64cpu-%j.err

WORK=/scratch/work/ladovj1/calculations/defumat-gpu
REPO=/scratch/work/ladovj1/apps/defumat
RUN="$WORK/qe64-$SLURM_JOB_ID"
mkdir -p "$WORK/out" "$RUN"
cd "$RUN" || exit 1

echo "=== node ==="; hostname; date
lscpu | grep -E "Model name|^CPU\(s\)"

module load triton/2024.1-gcc openmpi/4.1.6 quantum-espresso/7.2
echo "=== pw.x ==="; which pw.x; pw.x -h </dev/null 2>&1 | grep -m1 PWSCF

export ESPRESSO_PSEUDO="$REPO/tests/data/pseudo"
export OMP_NUM_THREADS=1
export ESPRESSO_TMPDIR="$RUN/tmp"
mkdir -p "$ESPRESSO_TMPDIR"

# QE writes wavefunctions to disk by default. defumat does not, so timing one
# doing I/O the other skips would not be a comparison of the physics -- and the
# cluster's small-files policy says not to anyway.
for case in si64-1k-ecut30 si64-1k; do
  sed "s|^ &control|\&control\n    disk_io = 'none'\n    outdir = '$ESPRESSO_TMPDIR'|" \
      "$REPO/benchmarks/$case.in" > "$RUN/$case.qe.in"
  for n in 1 4 16 32; do
    echo; echo "########## QE $case on $n core(s) ##########"
    /usr/bin/time -f "wall %e s" srun --ntasks=$n --cpus-per-task=1 --mpi=pmix \
        pw.x -in "$RUN/$case.qe.in" > "$WORK/out/qe64cpu-${SLURM_JOB_ID}-$case-n$n.out" 2>&1
    echo "exit $?"
    grep -E "^!|PWSCF.*WALL|electrons  " "$WORK/out/qe64cpu-${SLURM_JOB_ID}-$case-n$n.out" | tail -4
    rm -rf "$ESPRESSO_TMPDIR"/*
  done
done

# The middle corner: defumat on CPU, four cores, exactly as everywhere else here.
source "$WORK/venv/bin/activate"
export DEFUMAT_CACHE_DIR="$WORK/cache-cpu" PYTHONPATH="$REPO" JAX_PLATFORMS=cpu
export MKL_NUM_THREADS=1
for case in si64-1k-ecut30 si64-1k; do
  echo; echo "########## defumat CPU $case ##########"
  srun --ntasks=1 --cpus-per-task=4 python3 "$REPO/tools/gpu/phase0.py" $case \
      --k-batch 1 --band-batch all --repeats 2 --skip-precision-probe \
      --json "$WORK/out/qe64cpu-${SLURM_JOB_ID}-$case-defumat.json"
done

rm -rf "$RUN"
echo; echo "=== done ==="; date
