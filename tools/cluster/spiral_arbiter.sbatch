#!/bin/bash
# The collinear arbiter for the spin spiral's factor of five: the same magnetic
# energy in both codes, with no spiral machinery on either side.
#
# `q = b3/2` of the unit cell **is** the collinear antiferromagnet of the cell
# doubled along `z`, so `E(AFM) - E(FM)` in the doubled cell is twice
# `E(q = 1/2) - E(q = 0)`, and both sides can be had from ordinary collinear
# runs. This code's own spiral satisfies that identity to **1e-10 Ry** already
# (`tests/regression/test_spin_spirals.py`, against a reference whose `nspin` is
# 2 and which is not a spiral at all), so this is not defumat checking itself --
# it is the number Elk's *collinear* path can be asked for.
#
# Written down before the run, because a prediction made afterwards is not one:
# near **-113 meV** means the codes agree about this chain and Elk's spiral is
# the outlier, which its own manual leaves room for by calling `spinsprl` an
# "Experimental feature"; near **-520 meV** means both spirals are exonerated and
# the codes disagree about the magnetism itself, which points at the
# pseudopotential or the functional and is a different investigation.
#
#   0, 1   Elk, ferromagnetic and antiferromagnetic
#   2, 3   defumat, the same two
#
# ---- the #SBATCH values ----
#
# partition     batch-milan.
# array         0-3.
# time          03:00:00. The spiral fixture's Elk runs take tens of loops at
#               under a minute each on this cell; doubling the cell roughly
#               eights the diagonalisation, and there are two atoms rather than
#               one, so this is headroom on a run measured in tens of minutes.
# cpus-per-task 4, with OMP pinned to the same 4 for Elk and to 1 beside XLA for
#               defumat. No timing pair is taken here, so Elk's threads are free
#               speed.
# mem           16G. Two hydrogen atoms; this is headroom, not a size.
#SBATCH --job-name=spiral-arbiter
#SBATCH --partition=batch-milan
#SBATCH --array=0-3
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-magnetism/out/arb-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-magnetism/out/arb-%A_%a.err

WORK=/scratch/work/ladovj1/calculations/defumat-magnetism
REPO=${DEFUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-jobs}
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv
FIXTURE="$REPO/tests/data/elk/h_chain_doubled"

TASK=${SLURM_ARRAY_TASK_ID:-0}
case $TASK in
    0) CODE=elk;     ARR=fm ;;
    1) CODE=elk;     ARR=afm ;;
    2) CODE=defumat; ARR=fm ;;
    3) CODE=defumat; ARR=afm ;;
    *) echo "no such task"; exit 1 ;;
esac

mkdir -p "$WORK/out" || exit 1
echo "=== node ==="; hostname; date
echo "task $TASK   code $CODE   arrangement $ARR"
echo "commit: $(git -C "$REPO" rev-parse HEAD)"

if [ "$CODE" = "elk" ]; then
    RUN="$WORK/arbiter/elk-$ARR"
    mkdir -p "$RUN" || exit 1
    module load triton/2024.1-gcc openmpi/4.1.6 elk/10.2.4 || exit 1
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
    export MKL_NUM_THREADS=1
    cp "$FIXTURE/elk.in" "$FIXTURE/H.in" "$RUN/" || exit 1
    cd "$RUN" || exit 1
    if [ "$ARR" = "afm" ]; then
        # Flip the SECOND atom's bfcmt and nothing else, so the two runs carry
        # the same field magnitude and differ only in its arrangement. The line
        # is matched on its position label rather than on the field, since both
        # atoms' field columns are identical in the committed fixture.
        sed -i 's/^  0.0  0.0  0.5    0.0  0.0  0.002/  0.0  0.0  0.5    0.0  0.0 -0.002/' elk.in
    fi
    grep -A 4 '^atoms' elk.in
    # `srun` rather than a bare `elk`: the module's binary is MPI-linked and
    # dies in its launcher outside one.
    srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" elk > elk.log 2>&1
    echo "elk exit $?"
    grep "^ total energy" INFO.OUT | tail -2
    grep "total moment  " INFO.OUT | tail -2
    grep -c "Loop number" INFO.OUT
else
    cd "$REPO" || exit 1
    source "$VENV/bin/activate" || exit 1
    export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
    export JAX_PLATFORMS=cpu PYTHONPATH="$REPO" DEFUMAT_CACHE_DIR=off
    python3 -u tools/cluster/spiral_arbiter.py --arrangement "$ARR" \
        --out "$WORK/out/arbiter-${SLURM_ARRAY_JOB_ID}-$ARR.json"
    echo "exit $?"
fi

echo "=== done ==="; date
