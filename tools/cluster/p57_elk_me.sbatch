#!/bin/bash
# Elk's magnetoelectric tensor (task 390) on the AlAs cell of
# `tests/data/qe/alas-magnetoelectric.in`, as the external number `PLAN.md` P57
# has never had.
#
# **The field is converted, not copied.** Elk adds `cb * bfieldc` to the
# Kohn-Sham field with `cb = g_e/(4c) = 3.6529e-3` (`genbs.f90`), and matching the
# up-down splitting against this code's `-B . m` in Rydberg gives
# `B_defumat = 2 cb bfieldc` (`PLAN.md` P86). The defumat input carries
# `B_field(3) = 0.05` Ry with a step `delta = 0.02` Ry, so Elk gets
#
#     bfieldc = 0.05 / (2 cb) = 6.84370     deltabf = 0.02 / (2 cb) = 2.73748
#
# and its tensor, which is d(dipole per cell)/d(bfieldc) in Cartesian e.bohr, is
# this code's alpha (e/bohr^2 per Ry) times the cell volume times 2 cb.
#
#   task 0   ngridk 2 2 2, spin-orbit on       -- the matched grid
#   task 1   ngridk 4 4 4, spin-orbit on       -- is Elk's own tensor converged?
#   task 2   ngridk 2 2 2, spin-orbit off      -- the null, which must be zero
#
# The crystal, the functional (PBE, xctype 20) and the lattice constant are the
# defumat input's; the basis is Elk's default, which is what P57 compared bands
# at. Elk's task 390 turns spinpol on and reducebf off itself.
#
# partition     batch-milan.
# time          12:00:00. Six ground states per task with spin-orbit coupling;
#               the 4x4x4 one is the long pole.
# cpus-per-task 8, OpenMP. No timing pair is taken here.
# mem           16G.
#SBATCH --job-name=elk-me-p57
#SBATCH --partition=batch-milan
#SBATCH --array=0-2
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-magnetism/out/elkme-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-magnetism/out/elkme-%A_%a.err

WORK=/scratch/work/ladovj1/calculations/defumat-magnetism
TASK=${SLURM_ARRAY_TASK_ID:-0}
GRIDS=(2 4 2)
SOCS=(.true. .true. .false.)
NK=${GRIDS[$TASK]}
SOC=${SOCS[$TASK]}
RUN="$WORK/elk-me-p57/task$TASK"
mkdir -p "$RUN" "$WORK/out" || exit 1

module load triton/2024.1-gcc openmpi/4.1.6 elk/10.2.4 || exit 1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=1
SPECIES="$(dirname "$(dirname "$(which elk)")")/species/"

echo "=== node ==="; hostname; date
echo "task $TASK   ngridk $NK   spinorb $SOC"
cd "$RUN" || exit 1

cat > elk.in <<EOF
tasks
  390

spinorb
  $SOC

xctype
  20

bfieldc
  0.0  0.0  6.84370

deltabf
  2.73748

nempty
  8

ngridk
  $NK  $NK  $NK

! fcc, a = 10.6134 bohr, in pw.x's own ibrav = 2 vectors a/2 (-1,0,1), (0,1,1),
! (-1,1,0): the crystal coordinates (1/4, 1/4, 1/4) then put As at a/4 (-1, 1, 1)
! in both codes. The conventional (0,1,1), (1,0,1), (1,1,0) would put it at
! a/4 (1, 1, 1), the mirror image, and flip the signs of the tensor components.
avec
 -5.3067   0.0      5.3067
  0.0      5.3067   5.3067
 -5.3067   5.3067   0.0

sppath
  '$SPECIES'

atoms
  2
  'Al.in'
  1
  0.0   0.0   0.0
  'As.in'
  1
  0.25  0.25  0.25
EOF

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" elk > elk.log 2>&1
echo "elk exit $?"
echo "--- MAGNETOELT.OUT ---"
cat MAGNETOELT.OUT 2>/dev/null
echo "--- moments, last ground state ---"
grep -A6 "Moments :" INFO.OUT | tail -8
echo "=== done ==="; date
