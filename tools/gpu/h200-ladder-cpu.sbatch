#!/bin/bash
# Quantum ESPRESSO on CPUs for the 64/128-atom ladder: 1 core first, then 4, 16, 32.
#
# One core is the number asked for and is the project's own canonical baseline.
# The other three are here so it cannot be read as "what a GPU is worth against
# QE" without seeing what QE does when it is given a machine -- for a single
# k-point pw.x has only plane-wave parallelism and saturates early, which is
# most of why the single-core ratio looks the way it does.
#
# Ordered so the single-core runs finish first: they are the comparison, and a
# 128-atom serial run at 30 Ry is the one that could overrun.
#SBATCH --job-name=defumat-h200-ladder-cpu
#SBATCH --partition=batch-milan
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=192G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-gpu/out/h200ladcpu-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-gpu/out/h200ladcpu-%j.err

WORK=/scratch/work/ladovj1/calculations/defumat-gpu
REPO=/scratch/work/ladovj1/apps/defumat
RUN="$WORK/lad-$SLURM_JOB_ID"
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
for n in 1 4 16 32; do
  for case in si64-1k-ecut30 si128-1k-ecut30; do
    echo; echo "########## QE $case on $n core(s) ##########"
    srun --ntasks=$n --cpus-per-task=1 --mpi=pmix \
        pw.x -in "$RUN/$case.qe.in" > "$WORK/out/h200ladcpu-${SLURM_JOB_ID}-$case-n$n.out" 2>&1
    echo "exit $?"
    grep -E "^!|PWSCF.*WALL" "$WORK/out/h200ladcpu-${SLURM_JOB_ID}-$case-n$n.out" | tail -2
    rm -rf "$ESPRESSO_TMPDIR"/*
  done
done
rm -rf "$RUN"
echo; echo "=== done ==="; date
