#!/bin/bash
# Does Elk's own per-atom fixed spin direction hold two iron moments at 90 degrees?
#
# `MAGNETISM-NEXT.md` item B. This code's transcription of `bfieldfsm.f90`
# (`'atomic fsm direction'`, Elk's `fsmtype = -2`) runs away on the two-atom
# canted iron cell at every fixed gain from 0.0025 to 0.04, and the vector
# penalty holds it (0.39 degrees per site in 76 iterations). The question that
# decides whether that is the scheme or this code's coupling of it to the mixer
# is whether Elk, with its own mixer, holds the same cell with the same scheme.
#
# The cell is `tests/data/qe/fe2-canted-nosoc.in`'s: simple cubic, a = 5.42 bohr,
# two iron atoms at 0 and (1/2, 1/2, 1/2), no spin-orbit coupling, targets along x
# and y. Everything that is not the controller is Elk's default or Elk's own
# `examples/magnetism/Fe-FSM-MT-AFM` (the small seed field per atom along its
# target, `nempty 8`), so what is tested is the scheme and not a tuning of it.
#
#   task 0   fsmtype -2, taufsm 0.01 (Elk's default)
#   task 1   fsmtype -2, taufsm 0.0025
#   task 2   fsmtype  0, the control: what the canted seed does left alone
#
# partition     batch-milan, which scheduled every small job put to it within
#               a minute on 2026-09-22.
# time          06:00:00. Two atoms, 8 k-points, LAPW; tens of seconds a loop
#               at 4 threads and at most 300 loops.
# cpus-per-task 4, OpenMP; no timing pair is taken here.
# mem           8G.
#SBATCH --job-name=elk-fsm-b
#SBATCH --partition=batch-milan
#SBATCH --array=0-2
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-magnetism/out/elkfsm-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-magnetism/out/elkfsm-%A_%a.err

WORK=/scratch/work/ladovj1/calculations/defumat-magnetism
TASK=${SLURM_ARRAY_TASK_ID:-0}
FSMTYPES=(-2 -2 0)
TAUS=(0.01 0.0025 0.01)
FSMTYPE=${FSMTYPES[$TASK]}
TAU=${TAUS[$TASK]}
RUN="$WORK/elk-fsm-b/task$TASK"
mkdir -p "$RUN" "$WORK/out" || exit 1

module load triton/2024.1-gcc openmpi/4.1.6 elk/10.2.4 || exit 1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=1
SPECIES="$(dirname "$(dirname "$(which elk)")")/species/"

echo "=== node ==="; hostname; date
echo "task $TASK   fsmtype $FSMTYPE   taufsm $TAU"
cd "$RUN" || exit 1

cat > elk.in <<EOF
tasks
  0

fsmtype
  $FSMTYPE

taufsm
  $TAU

mommtfix
  1 1   1.0 0.0 0.0     : is, ia, mommtfix
  1 2   0.0 1.0 0.0     : is, ia, mommtfix

spinpol
  .true.

maxscl
  300

nempty
  8

stype
  0

swidth
  0.025

avec
  1.0  0.0  0.0
  0.0  1.0  0.0
  0.0  0.0  1.0

scale
  5.42

sppath
  '$SPECIES'

atoms
  1                                   : nspecies
  'Fe.in'                             : spfname
  2                                   : natoms
  0.0  0.0  0.0    0.001  0.0    0.0  : atposl, bfcmt
  0.5  0.5  0.5    0.0    0.001  0.0

ngridk
  2  2  2

vkloff
  0.5  0.5  0.5
EOF

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" elk > elk.log 2>&1
echo "elk exit $?"

echo "--- last loop ---"
grep -E "Loop number" INFO.OUT | tail -1
grep -E "reached self|Convergence targets" INFO.OUT | tail -2
echo "--- muffin-tin moments, last three loops ---"
grep -A3 "Moments :" INFO.OUT | tail -24
echo "--- total energy ---"
grep -E "total energy  " INFO.OUT | tail -3
echo "=== done ==="; date
