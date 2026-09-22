#!/bin/bash
# The spiral against Elk with both codes reporting the same quantity.
#
# P105 found that the two codes' `E(q) - E(0)` were not the same object: Elk
# computes its entropy term only for Fermi-Dirac smearing (`energy.f90:242`,
# `IF (stype == 3)`), so under the Gaussian its total is an **internal** energy
# where this code's and QE's are **free** energies. At `degauss = 0.1 Ry` the
# entropy moves by 112 meV between `q = 0` and `q = 1/4`, five times the
# difference being compared, and subtracting it brought the two codes to 2.4 and
# 0.5 per cent.
#
# **This is the version that needs no subtraction**, and that is the whole reason
# to run it: a correction is a thing to remember and a fixture is not. With
# `fermi-dirac` on this side and `stype = 3` on Elk's, both totals carry `-TS`.
#
#   0, 1, 2   Elk at q_3 = 0, 1/4, 1/2
#   3, 4, 5   defumat at the same three
#
# **No number carries across from the Gaussian pair.** A different smearing
# function is a different free-energy surface, so the moment, the energy and
# `E(q) - E(0)` all move; what this buys is that the two codes are on the same
# surface as each other, which the Gaussian pair was not. The expectation is
# therefore agreement at the level the two bases explain -- about one per cent
# from Elk's `rgkmax` and rather more from `ecutwfc = 25`, both measured -- and
# **not** a reproduction of P105's -132.986 and -263.813 meV, which belong to the
# other surface.
#
# ---- the #SBATCH values ----
#
# partition     batch-milan.
# array         0-5.
# time          03:00:00. The Gaussian Elk runs on this cell took 42 to 95 loops
#               at under a minute each; Fermi-Dirac is the same cost per loop and
#               there is no reason to expect a different count, so this is
#               headroom rather than an estimate.
# cpus-per-task 4, with OMP pinned to the same 4 for Elk and to 1 beside XLA for
#               defumat. No timing pair is taken here.
# mem           16G. One hydrogen atom; headroom, not a size.
#SBATCH --job-name=spiral-fd
#SBATCH --partition=batch-milan
#SBATCH --array=0-5
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-magnetism/out/fd-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-magnetism/out/fd-%A_%a.err

WORK=/scratch/work/ladovj1/calculations/defumat-magnetism
REPO=${DEFUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-jobs}
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv
FIXTURE="$REPO/tests/data/elk/h_chain_spiral_fd"

TASK=${SLURM_ARRAY_TASK_ID:-0}
QVALUES=(0.0 0.25 0.5)
if [ "$TASK" -lt 3 ]; then CODE=elk; QZ=${QVALUES[$TASK]};
else CODE=defumat; QZ=${QVALUES[$((TASK - 3))]}; fi

mkdir -p "$WORK/out" || exit 1
echo "=== node ==="; hostname; date
echo "task $TASK   code $CODE   q_3 $QZ"
echo "commit: $(git -C "$REPO" rev-parse HEAD)"

if [ "$CODE" = "elk" ]; then
    RUN="$WORK/fd/elk-q$QZ"
    mkdir -p "$RUN" || exit 1
    module load triton/2024.1-gcc openmpi/4.1.6 elk/10.2.4 || exit 1
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
    export MKL_NUM_THREADS=1
    cp "$FIXTURE/elk.in" "$FIXTURE/H.in" "$RUN/" || exit 1
    cd "$RUN" || exit 1
    # The line after the `vqlss` keyword, and that keyword also appears in a
    # comment above its block, so the substitution is anchored on the bare line.
    python3 - "$QZ" <<'PY'
import sys
qz = sys.argv[1]
lines = open("elk.in").read().split("\n")
out, index = [], 0
while index < len(lines):
    out.append(lines[index])
    if lines[index].strip() == "vqlss":
        out.append(f"  0.0  0.0  {qz}")
        index += 2
        continue
    index += 1
open("elk.in", "w").write("\n".join(out))
print("vqlss z", qz)
PY
    # `srun` rather than a bare `elk`: the module's binary is MPI-linked and
    # dies in its launcher outside one.
    srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" elk > elk.log 2>&1
    echo "elk exit $?"
    grep "^ total energy" INFO.OUT | tail -1
    grep "total moment  " INFO.OUT | tail -1
    # Under `stype = 3` the printed parts must now sum to LESS than the total by
    # the entropy, which is the two-minute check that says the term is live --
    # the same check that, returning equality, said it was empty on the Gaussian
    # fixture.
    grep -E "electron kinetic|Coulomb  |exchange |correlation " INFO.OUT | tail -4
    grep -c "Loop number" INFO.OUT
else
    cd "$REPO" || exit 1
    source "$VENV/bin/activate" || exit 1
    export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
    export JAX_PLATFORMS=cpu PYTHONPATH="$REPO" DEFUMAT_CACHE_DIR=off
    python3 -u tools/cluster/spiral_fd.py --q "$QZ" \
        --out "$WORK/out/fd-${SLURM_ARRAY_JOB_ID}-q$QZ.json"
    echo "exit $?"
fi

echo "=== done ==="; date
