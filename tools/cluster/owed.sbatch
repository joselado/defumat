#!/bin/bash
# The slow files owed by P109, P110 and P111, one per task (`OPEN.md` Part XVI item 1).
#
# Three sessions changed code whose slow files had not run: the magnetism and
# ultracell review (P109), and the two quick-items sessions (P110, P111). The
# record names 35 files still owed. Seven of them peaked at 11 to 20.5 GB when
# `bigtests.sbatch` ran them cold on 2026-09-22, which the workstation's 12 G
# per-file cap kills, so the whole list runs here instead.
#
# One file per task, for `bigtests.sbatch`'s reason: a shared process
# accumulates XLA executables, so a failure or a peak is attributable to a file
# only when the file has a process of its own.
#
# **Submitted twice with different memory**, since an array has one `--mem`:
#
#   sbatch --array=0-6     --mem=32G tools/cluster/owed.sbatch   # the seven hogs
#   sbatch --array=7-34%10 --mem=16G tools/cluster/owed.sbatch   # the rest
#
# The hogs come first in FILES for that reason. 32 G is 1.5 times the largest
# cold peak on record (`test_ldau_flavours`, 20548M); 16 G is four times the
# 3.9 GB `test_ultracell.py` read on the workstation, the largest of the rest
# on record. `%10` keeps the second array to 160 GB and 80 cores at once.
#
# ---- the #SBATCH values ----
#
# partition     batch-milan: 32 nodes, 128 cores and 510 GB each, 5-day limit.
# time          06:00:00. `test_ten_site.py`, the longest on record, ran 31 m
#               cold; compilation is paid per file with the cache off.
# cpus-per-task 8, because `tests/conftest.py` sets `DEFUMAT_THREADS=8` for the
#               thread-pool deadlock in `OPEN.md` Part IV item 1.
#SBATCH --job-name=defumat-owed
#SBATCH --partition=batch-milan
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-owed/out/owed-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-owed/out/owed-%A_%a.err

set -u
FILES=(test_ldau_flavours test_noncollinear_hubbard_resume test_ten_site \
       test_piezoelectric_paw test_lsda test_piezoelectric_augmented test_ldau \
       test_ultracell_augmented test_ultracell_stm test_ultracell_sts \
       test_anisotropy test_relaxed_anisotropy test_magnetic_constraints \
       test_noncollinear_magnetism test_magnetoelectric \
       test_spinor_hubbard_symmetry test_noncollinear_gga test_paw_noncollinear \
       test_elk_seed test_stress test_uspp test_spin_spirals \
       test_spin_spirals_augmented test_conductivity test_tddft test_shg \
       test_photocurrent test_magnons test_piezoelectric test_piezoelectric_wedge \
       test_spinor_projection_symmetry test_pdos test_noncollinear_pdos \
       test_spinor_pdos test_continuation)
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

# Overridable so that an A/B at another commit (DEFUMAT_REPO pointing at a
# worktree) writes beside this run rather than into it: the runner skips a file
# already named in its summary, so a shared directory would run nothing.
WORK=${DEFUMAT_OWED_WORK:-/scratch/work/ladovj1/calculations/defumat-owed}
REPO=${DEFUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-jobs}
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv

mkdir -p "$WORK/out" || exit 1
cd "$REPO" || exit 1
source "$VENV/bin/activate" || exit 1

export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export DEFUMAT_THREADS=8
export JAX_PLATFORMS=cpu
export PYTHONPATH="$REPO"

# The whole vendored tree on the group share, not the test-suite-only copy, so
# that a test reading a Fortran source finds it as it does on the workstation.
export DEFUMAT_QE_ROOT=/scratch/phys/t30428-staff/apps/qe-7.5-ReleasePack/qe-7.5

# Off for `bigtests.sbatch`'s reasons: an unchecked concurrent writer on this
# filesystem, and peaks that then read as a lower bound on the warm ones.
export DEFUMAT_CACHE_DIR=off

# Seven eighths of the allocation, so the watchdog names the test before Slurm
# kills the task without naming anything. SLURM_MEM_PER_NODE is in MB.
export DEFUMAT_TEST_MEM_MAX="$(( SLURM_MEM_PER_NODE * 7 / 8 ))M"

echo "=== node ==="; hostname; date
echo "file:   $FILE"
echo "cap:    $DEFUMAT_TEST_MEM_MAX of ${SLURM_MEM_PER_NODE}M"
echo "commit: $(git rev-parse --short HEAD)  $(git log --oneline -1)"
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"

tools/run_regression.sh "$WORK/out/reg-$FILE" "slow" "tests/regression/$FILE.py"
status=$?

echo "=== summary ==="
SUMMARY="$WORK/out/reg-$FILE/summary.txt"
cat "$SUMMARY" 2>/dev/null

# A task that ran nothing must not report success.
if ! grep -qE "[0-9]+ (passed|failed)|killed \(SIGKILL" "$SUMMARY" 2>/dev/null; then
    echo "NOTHING RAN: no passed/failed/killed line in $SUMMARY" >&2
    status=1
fi
echo "exit $status"
date
echo "=== done ==="
