#!/bin/bash
# The same sweep through pw.x on ONE core -- the baseline for sweep-gpu.sbatch.
#
# Serial by request. One task, so it starts without waiting for a node, and
# `disk_io = 'none'` so neither code is timed doing I/O the other skips.
# QE 7.2 from the cluster module (the 7.5 vendored in this repo is gitignored
# and not on the cluster); the version is recorded beside the number.
#SBATCH --job-name=pypresso-sweep-qe
#SBATCH --partition=batch-csl
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=/scratch/work/ladovj1/calculations/pypresso-gpu/out/sweepqe-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/pypresso-gpu/out/sweepqe-%j.err

WORK=/scratch/work/ladovj1/calculations/pypresso-gpu
REPO=/scratch/work/ladovj1/apps/pypresso
RUN="$WORK/sweep-$SLURM_JOB_ID"
mkdir -p "$WORK/out" "$RUN"; cd "$RUN" || exit 1
echo "=== node ==="; hostname; date; lscpu | grep -E "Model name"

module load triton/2024.1-gcc openmpi/4.1.6 quantum-espresso/7.2
export ESPRESSO_PSEUDO="$REPO/tests/data/pseudo"
export OMP_NUM_THREADS=1
export ESPRESSO_TMPDIR="$RUN/tmp"; mkdir -p "$ESPRESSO_TMPDIR"
pw.x -h </dev/null 2>&1 | grep -m1 PWSCF

CASES="si10-nc si10-us si10-paw si10-nc-pbe al10-metal h10-chain-lsda \
h10-chain-noncolin bi10-soc ni10-ldau h20-chain-lsda bi20-soc h40-chain-lsda"

for case in $CASES; do
  src="$REPO/benchmarks/$case.in"; [ -f "$src" ] || src="$REPO/tests/data/qe/$case.in"
  sed "s|^ *&control|\&control\n    disk_io = 'none'\n    outdir = '$ESPRESSO_TMPDIR'|" \
      "$src" > "$RUN/$case.qe.in"
  echo; echo "########## QE $case on 1 core ##########"
  timeout 3600 pw.x -in "$RUN/$case.qe.in" > "$WORK/out/sweepqe-${SLURM_JOB_ID}-$case.out" 2>&1
  echo "exit $?"
  grep -E "^!|PWSCF.*WALL" "$WORK/out/sweepqe-${SLURM_JOB_ID}-$case.out" | tail -2
  rm -rf "$ESPRESSO_TMPDIR"/*
done
rm -rf "$RUN"
echo; echo "=== done ==="; date
