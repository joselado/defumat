#!/bin/bash
# Quantum ESPRESSO on CPUs for the 64/128-atom ladder: 1 core first, then 4, 16, 32.
#
# **Serial only.** One core is the number asked for and is the project's own
# canonical baseline. It is split out from h200-ladder-cpu.sbatch, which asks
# for 32 tasks and therefore queues behind a whole node, because the serial run
# is the comparison and a one-task job starts immediately. The 4/16/32-core
# columns are that other job's business and are what stop this one being read as
# "what a GPU is worth against QE".
#SBATCH --job-name=pypresso-qe-serial
#SBATCH --partition=batch-milan
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=/scratch/work/ladovj1/calculations/pypresso-gpu/out/qeser-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/pypresso-gpu/out/qeser-%j.err

WORK=/scratch/work/ladovj1/calculations/pypresso-gpu
REPO=/scratch/work/ladovj1/apps/pypresso
RUN="$WORK/qeser-$SLURM_JOB_ID"
mkdir -p "$WORK/out" "$RUN"
cd "$RUN" || exit 1
echo "=== node ==="; hostname; date; lscpu | grep -E "Model name"

module load triton/2024.1-gcc openmpi/4.1.6 quantum-espresso/7.2
export ESPRESSO_PSEUDO="$REPO/tests/data/pseudo"
export OMP_NUM_THREADS=1
export ESPRESSO_TMPDIR="$RUN/tmp"; mkdir -p "$ESPRESSO_TMPDIR"
pw.x -h </dev/null 2>&1 | grep -m1 PWSCF

for case in si64-1k-ecut30 si128-1k-ecut30; do
  sed "s|^ &control|\&control\n    disk_io = 'none'\n    outdir = '$ESPRESSO_TMPDIR'|" \
      "$REPO/benchmarks/$case.in" > "$RUN/$case.qe.in"
done
for n in 1; do
  for case in si64-1k-ecut30 si128-1k-ecut30; do
    echo; echo "########## QE $case on $n core(s) ##########"
    srun --ntasks=$n --cpus-per-task=1 --mpi=pmix \
        pw.x -in "$RUN/$case.qe.in" > "$WORK/out/qeser-${SLURM_JOB_ID}-$case-n$n.out" 2>&1
    echo "exit $?"
    grep -E "^!|PWSCF.*WALL" "$WORK/out/qeser-${SLURM_JOB_ID}-$case-n$n.out" | tail -2
    rm -rf "$ESPRESSO_TMPDIR"/*
  done
done
rm -rf "$RUN"
echo; echo "=== done ==="; date
