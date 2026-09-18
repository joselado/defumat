#!/bin/bash
# Which cluster failures are this session's and which are the machine's, plus
# whether the prescribed cache fixture cures the mapping exhaustion at all.
#
# A whole `slow` run on a node (job 20336106) failed in thirteen files. Eight are
# XLA mapping exhaustion -- 33 of 37 individual failures in the six worst files
# are `JaxRuntimeError`, and the two that abort do so inside
# `backend_compile_and_load` -- and the rest are assertions, listed in `OPEN.md`
# Part XIII item 4. None of those is attributed, because reading the diff cannot
# settle it: `test_relaxed_anisotropy`'s identity calls the functional entry
# point directly rather than the facade this session changed, and the node runs
# JAX 0.11.1 / NumPy 2.5.2 where the workstation runs 0.11.0 / 2.4.6.
#
# So three tasks, and each answers one question:
#
#   0  the seven assertion failures at `b247662`, the commit the session started
#      from. A failure here predates the session.
#   1  the same seven at HEAD, in the same environment, so the pair is an A/B in
#      the code and nothing else.
#   2  `test_electrostriction.py` and `test_spectra.py`, the two files that now
#      carry the autouse `jax.clear_caches()` fixture, at HEAD. They have never
#      run on a node *with* it, and commit `40d8fe2`'s own message says clearing
#      caches does not unmap -- so this says whether the cure prescribed for
#      eleven more files is a cure. If it is not, the answer is a process
#      boundary per test and those eleven edits are wasted.
#
# Two of the seven are given as **node IDs** rather than files: their assert sits
# behind 26 to 70 mapping errors in the same file, and the point is to reach the
# number in a fresh process at both commits rather than to reproduce the
# exhaustion twice.
#
# ---- the #SBATCH values ----
#
# partition     batch-milan, as the rest of this directory.
# array         0-2, the three questions above.
# time          04:00:00. Task 0 and 1 are ~70 minutes of tests each, of which
#               `test_ten_site.py` alone was 40 on the last run.
# cpus-per-task 4, MKL and OMP pinned to 1.
# mem           32G, the regression array's size; the largest of these files
#               resided in 17 GB there (`test_ten_site`), and the cap inside
#               `run_regression.sh` is set below it so a file is named rather
#               than the task killed.
#SBATCH --job-name=defumat-attribute
#SBATCH --partition=batch-milan
#SBATCH --array=0-2
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-audit/out/attr-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-audit/out/attr-%A_%a.err

WORK=/scratch/work/ladovj1/calculations/defumat-audit
OLD=${DEFUMAT_OLD_REPO:-/scratch/work/ladovj1/apps/defumat-audit}
NEW=${DEFUMAT_NEW_REPO:-/scratch/work/ladovj1/apps/defumat-jobs}
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv

# The seven assertion failures. `OPEN.md` Part XIII item 4 carries what each one
# says; two are node IDs for the reason in the header.
SEVEN="tests/regression/test_relaxed_anisotropy.py \
tests/regression/test_magnetic_constraints.py \
tests/regression/test_ten_site.py \
tests/regression/test_stm.py \
tests/regression/test_stress.py \
tests/regression/test_lsda_response.py::test_the_polarized_dielectric_constant_reduces_to_the_unpolarized_one \
tests/regression/test_spinor_dielectric.py::test_a_spinor_with_no_magnetization_gives_the_scalar_dielectric_tensor"
FIXTURE="tests/regression/test_electrostriction.py tests/regression/test_spectra.py"

case ${SLURM_ARRAY_TASK_ID:-0} in
    0) REPO=$OLD; FILES=$SEVEN;   LABEL=attr-old ;;
    1) REPO=$NEW; FILES=$SEVEN;   LABEL=attr-new ;;
    2) REPO=$NEW; FILES=$FIXTURE; LABEL=attr-fixture ;;
    *) echo "no such task"; exit 1 ;;
esac

mkdir -p "$WORK/out" || exit 1
cd "$REPO" || exit 1
source "$VENV/bin/activate" || exit 1

export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export PYTHONPATH="$REPO"
# Off on every task, because a comparison of two commits must not also be a
# comparison of two cache states -- `CLAUDE.md` records one 603 MB cache entry
# worth 6.3 GB resident, in the direction that flatters whichever side was the
# miss.
export DEFUMAT_CACHE_DIR=off
export DEFUMAT_QE_ROOT=/scratch/work/ladovj1/apps/qe-7.5-reference
export DEFUMAT_TEST_MEM_MAX=28G

echo "=== node ==="; hostname; date
echo "task:   ${SLURM_ARRAY_TASK_ID:-0}  ($LABEL)"
echo "repo:   $REPO"
echo "commit: $(git -C "$REPO" rev-parse HEAD)  $(git -C "$REPO" log --oneline -1)"
python3 -c "import jax, numpy; print('jax', jax.__version__, 'numpy', numpy.__version__)"
echo "files:"; for one in $FILES; do echo "  $one"; done

tools/run_regression.sh "$WORK/out/$LABEL" slow "$FILES"

echo "=== summary ==="; cat "$WORK/out/$LABEL/summary.txt"
echo "=== the assertions, if any survived ==="
grep -h "^E " "$WORK/out/$LABEL"/*.log 2>/dev/null | sort | uniq -c | head -20
echo "=== mapping errors per file ==="
for one in "$WORK/out/$LABEL"/*.log; do
    echo "  $(basename "$one"): $(grep -c 'Failed to materialize\|Cannot allocate memory' "$one")"
done
echo "=== done ==="; date
