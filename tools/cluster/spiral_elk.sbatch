#!/bin/bash
# Elk's own basis convergence on the spin-spiral hydrogen chain -- the last
# untested candidate for the factor of five between the two codes' `E(q) - E(0)`
# (`PLAN.md` P86, `MAGNETISM-NEXT.md` item E).
#
# **What is already excluded, and by measurement rather than argument.** The
# k-grid is worth nothing: the defumat scan at `1 1 4/8/16/32` converges by 8 and
# is flat to 0.03 meV out to 32, leaving the ratio at 6.19 and 5.08. The defumat
# basis is worth a tenth of it: `ecutwfc` from 25 to 80 takes the ratio from 6.18
# to 5.42 at `q = 1/4` and 5.08 to 4.67 at `q = 1/2`, converged by 60. And the
# held field is now excluded twice over -- P86 measured it at 6 per cent, and
# that measurement used a field **274 times too large**, because the conversion
# behind it read Elk's `bfieldc` as an energy in Hartree and applied only the
# factor of two to Rydberg. Elk couples it as `cb * bfieldc` with
# `cb = g_e/(4 c) = 3.6529e-3` (`genbs.f90:15`, `energy.f90:77`), so
# `bfieldc = 0.002` is **1.46e-5 Ry** in this code's units and not 0.004 Ry.
#
# **So what is left is Elk's side of the basis**, which nobody has moved. Elk's
# `rgkmax = 7` is its own default and is presumably under-converged in the same
# way `ecutwfc = 25` was on this side -- the coincidence that made the two look
# matched (rgkmax 7 over `rmt = 1.4` is `|G+k|max = 5.0` a.u., which is 25 Ry)
# was a coincidence between two under-converged bases, which is P86's own
# conclusion.
#
#   rgkmax   7, 8, 9, which is 25, 32.6 and 41.3 Ry of interstitial plane waves
#   q_3      0, 1/4, 1/2, because the quantity is `E(q) - E(0)` and the `q = 0`
#            leg has to be taken at every basis rather than once
#
# **The control is inside the sweep and is not optional.** P86's Elk numbers came
# from a 9.6.8 build that no longer exists on any machine here, and this runs
# 10.2.4 from Triton's module tree, so `rgkmax = 7` is a version A/B before it is
# a rung: it must reproduce -136.294 meV at `q = 1/4`, and a sweep whose first
# rung already disagrees is measuring the version rather than the basis.
#
# Two settings the fixture carries that must not be lost, both paid for in P86:
# `mixtype 1`, because Elk's default Broyden falls off the magnetic branch at
# loop 10 with its history full and then converges, reports success and is
# nonmagnetic; and `reducebf 1.0`, because a fading seed field leaves the same
# quietly nonmagnetic run. Always read the total moment out of `INFO.OUT` before
# using an energy from one of these.
#
# ---- the #SBATCH values ----
#
# partition     batch-milan.
# array         0-8, the three bases at the three wavevectors, one array.
# time          06:00:00. Elk on this cell was measured at about 90 s per SCF
#               loop pinned and 42 s at six threads on the workstation, with
#               tens of loops per wavevector, so `rgkmax = 7` is under an hour;
#               the ladder is the unknown, an LAPW diagonalisation going as the
#               cube of a basis that grows as `rgkmax^3`, so rung 9 could be an
#               order dearer. Six hours bounds that without the backfill penalty
#               an eight-hour request for an eight-minute job earned on this
#               account before.
# cpus-per-task 4, with `OMP_NUM_THREADS` set to the same 4 rather than to 1.
#               Elk is OpenMP-parallel and no timing pair is being taken here --
#               a `PERFORMANCE.md` number would need one core and an idle
#               machine -- so the threads are free speed.
# mem           16G. One hydrogen atom, four k-points and eight empty states;
#               this is headroom, not a size.
#SBATCH --job-name=elk-spiral
#SBATCH --partition=batch-milan
#SBATCH --array=0-8
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-magnetism/out/elk-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-magnetism/out/elk-%A_%a.err

WORK=/scratch/work/ladovj1/calculations/defumat-magnetism
REPO=${DEFUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-jobs}
FIXTURE="$REPO/tests/data/elk/h_chain_spiral"

TASK=${SLURM_ARRAY_TASK_ID:-0}
RGKMAXES=(7.0 8.0 9.0)
QVALUES=(0.0 0.25 0.5)
RGKMAX=${RGKMAXES[$((TASK / 3))]}
QZ=${QVALUES[$((TASK % 3))]}

RUN="$WORK/elk/rgkmax${RGKMAX}-q${QZ}"
mkdir -p "$RUN" "$WORK/out" || exit 1

module load triton/2024.1-gcc openmpi/4.1.6 elk/10.2.4 || exit 1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=1

echo "=== node ==="; hostname; date
echo "task $TASK   rgkmax $RGKMAX   q_3 $QZ"
echo "commit: $(git -C "$REPO" rev-parse HEAD)"
which elk && elk --version 2>&1 | head -2

cp "$FIXTURE/elk.in" "$FIXTURE/H.in" "$RUN/" || exit 1
cd "$RUN" || exit 1

# Overwrite the two blocks this rung varies. Both are "keyword on one line, value
# on the next" in Elk's input, so the substitution is on the line *after* the
# keyword and nowhere else -- `vqlss` also appears in a comment above its block.
python3 - "$RGKMAX" "$QZ" <<'PY'
import sys
rgkmax, qz = sys.argv[1], sys.argv[2]
lines = open("elk.in").read().split("\n")
out, index = [], 0
while index < len(lines):
    out.append(lines[index])
    if lines[index].strip() == "vqlss":
        out.append(f"  0.0  0.0  {qz}")
        index += 2
        continue
    if lines[index].strip() == "rgkmax":
        out.append(f"  {rgkmax}")
        index += 2
        continue
    index += 1
if not any(line.strip() == "rgkmax" for line in lines):
    out += ["", "rgkmax", f"  {rgkmax}"]
open("elk.in", "w").write("\n".join(out))
print("rgkmax", rgkmax, "vqlss z", qz)
PY

elk > elk.log 2>&1
echo "elk exit $?"

echo "--- total energy and moment ---"
grep -E "Total energy per unit cell|total moment  " INFO.OUT | tail -6
grep -E "Loop number" INFO.OUT | tail -1
grep -E "Convergence targets|reached self" INFO.OUT | tail -3

echo "=== done ==="; date
