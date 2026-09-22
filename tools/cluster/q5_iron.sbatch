#!/bin/bash
# The external number an augmented spinor with several non-parallel moments has
# never had -- `MAGNETISM-NEXT.md` Q5, ranked first in
# `MAGNETISM-RANKED-2026-09-22.md` because every production texture in this
# project runs on exactly that combination and nothing in it has ever been
# compared with another code.
#
# Four tasks: two cells, two codes.
#
#   0, 1   `pw.x` on the antiferromagnetic and the canted cell, which is where
#          the reference comes from. There is no committed benchmark for either
#          because neither cell existed until now, and QE's own test-suite has no
#          fully relativistic ultrasoft cell with two inequivalent moments.
#   2, 3   defumat on the same two inputs, through `Calculator`, reporting the
#          total energy, both site moments, the angle between them and the
#          forces.
#
# **The pair is the measurement rather than either cell.** The antiferromagnet is
# a state one global spin axis still describes, so its off-diagonal spin blocks
# carry nothing a sign cannot absorb; the canted cell at 90 degrees populates
# them. A disagreement of the same size in both is about the augmented spinor
# machinery generally, and one that appears only in the canted cell is about the
# terms a texture switches on, which are `add_becsum_so` on a textured `becsum`
# and `qq_so` inside an overlap between two atoms whose local frames differ.
#
# The second atom is displaced by 0.02 alat in both cells so that the force is a
# real number: at the undisplaced positions the two atoms are related by
# inversion and every force is zero by symmetry, which would make the force
# comparison an agreement about a residue.
#
# ---- the #SBATCH values ----
#
# partition     batch-milan: 32 nodes, 128 cores and 510 GB each, 5-day limit,
#               which is what this project's other CPU jobs use.
# array         0-3, the four tasks above, one array rather than four
#               submissions as the cluster policy requires.
# time          04:00:00. Neither side has been run on this cell, so this is
#               headroom rather than an estimate: the cell is two atoms on a
#               30^3 dense grid with 64 k-points, which is small, and the risk
#               is iterations rather than cost per iteration, a canted iron
#               state being where `pw.x` has limit-cycled before. Four hours is
#               about thirty times what the geometry suggests and still short
#               enough not to backfill badly.
# cpus-per-task 8. XLA sizes its pool from the affinity mask and Slurm's cgroup
#               is what bounds it; MKL and OMP are pinned to 1 beside it. The
#               `pw.x` tasks run **serially** on one of the eight and leave the
#               rest idle, which is deliberate: an `--ntasks=1` allocation gives
#               MPI a single slot whatever the CPU count is, so a first attempt
#               at `mpirun -np 8` died in the launcher rather than in the
#               physics, and a two-atom cell on a 30^3 grid does not need the
#               ranks. Serial is also the only form a `PERFORMANCE.md` pair may
#               be taken in, so nothing is lost.
# mem           32G. Measured rather than copied: `Calculator.estimate()` on
#               both cells gives a 232 MB augmentation table over a ~400 MB
#               working set, and the force is a reverse-mode derivative that
#               carries that table on the tape, so the ask is roughly fifty
#               times the forward path. `pw.x` needs a fraction of it.
#SBATCH --job-name=defumat-q5iron
#SBATCH --partition=batch-milan
#SBATCH --array=0-3
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-magnetism/out/q5-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-magnetism/out/q5-%A_%a.err

WORK=/scratch/work/ladovj1/calculations/defumat-magnetism
REPO=${DEFUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-jobs}
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv
QEBIN=/scratch/work/ladovj1/apps/qe-env/bin

mkdir -p "$WORK/out" "$WORK/scratch" || exit 1
cd "$REPO" || exit 1

export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1

echo "=== node ==="; hostname; date
echo "task:   ${SLURM_ARRAY_TASK_ID:-0}"
echo "commit: $(git -C "$REPO" rev-parse HEAD)  $(git -C "$REPO" log --oneline -1)"

case ${SLURM_ARRAY_TASK_ID:-0} in
    0) CODE=qe;      CASE=fe2-afm-soc ;;
    1) CODE=qe;      CASE=fe2-canted-soc ;;
    2) CODE=defumat; CASE=fe2-afm-soc ;;
    3) CODE=defumat; CASE=fe2-canted-soc ;;
    *) echo "no such task"; exit 1 ;;
esac
echo "code:   $CODE   case: $CASE"

if [ "$CODE" = "qe" ]; then
    # `pseudo_dir` and `outdir` are deliberately not in the committed input --
    # every reference input in `tests/data/qe` leaves them out so that the file
    # is the physics and the paths are the environment's, which is what QE's
    # own two variables are for.
    export ESPRESSO_PSEUDO="$REPO/tests/data/pseudo"
    export ESPRESSO_TMPDIR="$WORK/scratch/$CASE"
    mkdir -p "$ESPRESSO_TMPDIR"
    "$QEBIN/pw.x" -in "$REPO/tests/data/qe/$CASE.in" \
        > "$WORK/out/qe-${SLURM_ARRAY_JOB_ID}-$CASE.out" 2>&1
    STATUS=$?
    echo "pw.x exit $STATUS"
    tail -40 "$WORK/out/qe-${SLURM_ARRAY_JOB_ID}-$CASE.out"
    # The scratch tree holds the wavefunctions and is reproducible output; the
    # `.out` is the reference and is what gets committed.
    rm -rf "$ESPRESSO_TMPDIR"
else
    source "$VENV/bin/activate" || exit 1
    export JAX_PLATFORMS=cpu
    export PYTHONPATH="$REPO"
    export DEFUMAT_CACHE_DIR=off
    python3 -c "import jax; print('jax', jax.__version__, jax.devices())"
    python3 -c "import os; print('affinity:', len(os.sched_getaffinity(0)), 'cores')"
    python3 -u tools/cluster/q5_measure.py "$CASE" \
        --out "$WORK/out/q5-${SLURM_ARRAY_JOB_ID}-$CASE.json"
    echo "exit $?"
fi

echo "=== done ==="; date
