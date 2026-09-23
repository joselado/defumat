#!/bin/bash
# Elk's magnetoelectric tensor (task 390) on Cr2O3, the cell `PLAN.md` P57 moved to
# (`tests/data/qe/cr2o3-magnetoelectric.in`). The same crystal in pw.x's own
# ibrav = 5 vectors, so the Cartesian components of the two tensors are the same
# components, and the same + - + - order along the threefold axis, seeded by a
# small field per muffin tin as Elk's own NiO-AFM example seeds its order.
#
# **No base field**, because the antiferromagnetic order already breaks time
# reversal, so the linear tensor is what task 390's central difference measures.
# That also disarms the one defect of `magnetoelt.f90` P57 found: it never resets
# `bfieldc0(j)` after column j, so later columns run with earlier half-steps still
# applied, and on a linear response a small leftover field changes nothing.
#
# The step is this code's `delta = 0.01` Ry converted by `2 cb`:
# `deltabf = 0.01 / (2 * 3.6529e-3) = 1.36878`.
#
#   task 0   ngridk 2 2 2, spin-orbit on       -- the grid the first defumat number used
#   task 1   ngridk 4 4 4, spin-orbit on       -- converged sampling
#   task 2   ngridk 2 2 2, spin-orbit off      -- the null, which must be zero
#
# partition     batch-milan.
# time          1-00:00:00. Ten atoms, LAPW with spin-orbit coupling, six ground
#               states per task.
# cpus-per-task 16, OpenMP. No timing pair is taken here.
# mem           32G.
#SBATCH --job-name=elk-cr2o3-me
#SBATCH --partition=batch-milan
#SBATCH --array=0-2
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-magnetism/out/elkcr-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-magnetism/out/elkcr-%A_%a.err

WORK=/scratch/work/ladovj1/calculations/defumat-magnetism
TASK=${SLURM_ARRAY_TASK_ID:-0}
GRIDS=(2 4 2)
SOCS=(.true. .true. .false.)
NK=${GRIDS[$TASK]}
SOC=${SOCS[$TASK]}
RUN="$WORK/elk-cr2o3-me/task$TASK"
mkdir -p "$RUN" "$WORK/out" || exit 1

module load triton/2024.1-gcc openmpi/4.1.6 elk/10.2.4 || exit 1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=1
SPECIES="$(dirname "$(dirname "$(which elk)")")/species/"

echo "=== node ==="; hostname; date
echo "task $TASK   ngridk $NK   spinorb $SOC"
cd "$RUN" || exit 1

cat > elk.in <<EOF
tasks
  390

spinpol
  .true.

spinorb
  $SOC

xctype
  20

deltabf
  1.36878

nempty
  10

ngridk
  $NK  $NK  $NK

maxscl
  200

! pw.x's ibrav = 5 vectors for a_rh = 10.1271 bohr, cos(alpha) = 0.57237
avec
  4.682785  -2.703607   8.562721
  0.000000   5.407214   8.562721
 -4.682785  -2.703607   8.562721

sppath
  '$SPECIES'

! Cr in the + - + - order along the axis, seeded by a small field per atom
atoms
  2
  'Cr.in'
  4
  0.152500  0.152500  0.152500    0.0  0.0  0.01
  0.347500  0.347500  0.347500    0.0  0.0 -0.01
  0.652500  0.652500  0.652500    0.0  0.0  0.01
  0.847500  0.847500  0.847500    0.0  0.0 -0.01
  'O.in'
  6
  0.555600  0.944400  0.250000
  0.250000  0.555600  0.944400
  0.944400  0.250000  0.555600
  0.444400  0.055600  0.750000
  0.750000  0.444400  0.055600
  0.055600  0.750000  0.444400
EOF

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" elk > elk.log 2>&1
echo "elk exit $?"
echo "--- MAGNETOELT.OUT ---"
cat MAGNETOELT.OUT 2>/dev/null
echo "--- moments and gap, last ground state ---"
grep -A10 "Moments :" INFO.OUT | tail -12
grep "Estimated indirect band gap" INFO.OUT | tail -1
echo "=== done ==="; date
