#!/bin/bash
# The two magnetism items that are code rather than a cell: P83's kernel and
# F2's Option 0.
#
#   0    **P83.** The longitudinal and transverse spin susceptibility of the
#        iodine atom, two ways -- the same screened fixed point the dielectric
#        constant is built on, and a central difference of two SCF runs under a
#        uniform Zeeman field. They share the ground state and nothing above it,
#        so agreement clears the kernel and its self-consistency together and
#        leaves the electric-field source term or `ph.x` holding the 5.3 per
#        cent, while disagreement locates the kernel with no reference to
#        another code at all.
#   1    **F2 Option 0, the deconfounder.** `fe-noncolin-pbe-stress.in` against
#        its nonmagnetic twin. The 43 iterations against `pw.x`'s 19 are
#        attributable to nothing until this runs, exactly as `fe-mag-1k`'s 25
#        against 12 was until its own twin came back at 21.
#   2    **F2 Option 0, the split.** One noncollinear run's residual, five ways
#        per iteration: charge, the length of `m`, the rigid rotation on the
#        three generators, the rest of the transverse channel, and the magnetic
#        part of `becsum`.
#   4    **P104's own outstanding item.** The iodine cell's transverse
#        susceptibility came out at -658.9 mu_B/Ry, negative, which says its
#        moment direction is a stationary point that is not a minimum -- and that
#        cell is the one P83's entire refusal rests on. Seeding the moment off
#        `z` and reading where it goes is the measurement, swept over four angles
#        because one angle cannot tell a slow return from a slow departure.
#   3    **F2 Option 0, the trip test.** A rotated *converged* state fed back,
#        whose residual must be zero to the level the run converged at, because
#        `F(R rho) = R F(rho)` for a spin-rotation-invariant functional. A
#        number here that is not at the convergence level says something in the
#        run is not invariant, which is worth more than the bin it was built to
#        test.
#
# ---- the #SBATCH values ----
#
# partition     batch-milan.
# array         0-3.
# time          08:00:00, set by task 0 and headroom for the rest. P83's own
#               probe script took 637 s for about thirty `chi_0` evaluations on
#               this cell, so a sixty-iteration screened solve is roughly twenty
#               minutes a direction, and the four finite-difference legs are
#               four more SCFs of a cell that converges in minutes. The iron
#               tasks are tens of minutes each. Array tasks schedule
#               independently, so the short ones do not wait on the long one.
# cpus-per-task 8, MKL and OMP pinned to 1 beside XLA.
# mem           64G. The iodine cell is one atom on a 54^3 dense grid and the
#               iron one is smaller still; what this is sized for is the
#               screened solve holding several whole `(nspin_mag, ...)` density
#               arrays at once beside the Sternheimer solver's own state.
#SBATCH --job-name=defumat-magnext
#SBATCH --partition=batch-milan
#SBATCH --array=0-4
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-magnetism/out/next-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-magnetism/out/next-%A_%a.err

WORK=/scratch/work/ladovj1/calculations/defumat-magnetism
REPO=${DEFUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-jobs}
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv

mkdir -p "$WORK/out" || exit 1
cd "$REPO" || exit 1
source "$VENV/bin/activate" || exit 1

export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export PYTHONPATH="$REPO"
export DEFUMAT_CACHE_DIR=off

echo "=== node ==="; hostname; date
echo "task:   ${SLURM_ARRAY_TASK_ID:-0}"
echo "commit: $(git -C "$REPO" rev-parse HEAD)  $(git -C "$REPO" log --oneline -1)"
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"
python3 -c "import os; print('affinity:', len(os.sched_getaffinity(0)), 'cores')"

JOB=${SLURM_ARRAY_JOB_ID:-local}
case ${SLURM_ARRAY_TASK_ID:-0} in
    0) python3 -u tools/cluster/p83_kernel.py --case i-atom-soc \
           --out "$WORK/out/p83-kernel-$JOB.json" ;;
    1) python3 -u tools/cluster/f2_option0.py --stage deconfounder \
           --out "$WORK/out/f2-deconfounder-$JOB.json" ;;
    2) python3 -u tools/cluster/f2_option0.py --stage split \
           --out "$WORK/out/f2-split-$JOB.json" ;;
    3) python3 -u tools/cluster/f2_option0.py --stage trip \
           --out "$WORK/out/f2-trip-$JOB.json" ;;
    4) python3 -u tools/cluster/p104_tilt.py --angles 0,2,10,30 \
           --out "$WORK/out/p104-tilt-$JOB.json" ;;
    *) echo "no such task"; exit 1 ;;
esac
echo "exit $?"

echo "=== done ==="; date
