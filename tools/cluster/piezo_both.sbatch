#!/bin/bash
# Both routes on the ultrasoft cell, now that both carry the constraint terms.
#
# The 1.8 per cent was never one route being wrong: `clamped_ion_piezoelectric`
# carries only the states tangent -- no multipliers, no `add_for_charges`
# sandwich -- so on an augmented dataset it is short of exactly what
# `born_effective_charges` supplies in the position coordinate, and
# `born_charges_from_stress_route`'s docstring has said so all along. Both
# routes now have both terms:
#
#   `_multiplier_strain_term` / the multipliers' tangent
#         `-Re sum_mn dLambda^E_mn S'_nm`, derived from the functional, whose
#         constraint is `-Re Tr[Lambda (G - 1)]`. Worth -0.00325 C/m^2 at 64
#         k-points.
#   `constraint_strain_term`
#         `sum_n w_n <psi_n| dS/d(eps) | P_c r_k psi_n>`, which is
#         `add_for_charges.f90` and which `_multiplier_response` cannot reach
#         because `<psi_m|r|psi_n>` is the Berry connection. **Worth 0.55 on
#         ultrasoft silicon in the position coordinate**, the difference
#         between +0.47 and -0.079, so it is the term that should dominate.
#
# **Both are identically zero for a norm-conserving dataset**, where `S = 1`
# does not deform, and that half of the check has passed on the workstation:
# `alas-raman.in` reads -0.763785227646695 either way, one ulp from what it read
# before any of this, and the two routes still agree to 6.2e-15.
#
#   0  the taped route on the ultrasoft cell at 64 points. It read **+0.815802**
#      without the terms.
#   1  the contracted route on the same 64 points. It read **+0.830702** bare
#      and **+0.827448** with the multiplier term alone.
#
# **The test is that they now agree with each other**, and they are different
# assemblies sharing only the field response, so agreement is a statement.
# Whatever they agree on is then the number to put against the Berry-phase
# +0.692986 -- at this mesh it still carries the k-error the calibration cell
# measures at 13.4 per cent, so the *value* is not the answer yet and the
# *agreement* is.
#
# ---- the #SBATCH values ----
#
# partition     batch-milan.
# array         0-1.
# time          04:00:00.
# cpus-per-task 4, MKL and OMP pinned to 1.
# mem           200G. The taped route's tape alone is 31.5 GiB at 64 points
#               (`memory_analysis`, compiled and never run) and its whole job
#               peaked at 139.6; the multipliers add an `nbnd x nbnd` Gram per
#               k-point on top, which on four occupied bands is nothing.
#SBATCH --job-name=defumat-piezo-both
#SBATCH --partition=batch-milan
#SBATCH --array=0-1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-audit/out/both-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-audit/out/both-%A_%a.err

WORK=/scratch/work/ladovj1/calculations/defumat-audit
REPO=${DEFUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-jobs}
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv

mkdir -p "$WORK/out" || exit 1
cd "$REPO" || exit 1
source "$VENV/bin/activate" || exit 1

export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export PYTHONPATH="$REPO"
export DEFUMAT_CACHE_DIR=off
export DEFUMAT_QE_ROOT=/scratch/work/ladovj1/apps/qe-7.5-reference

if [ "${SLURM_ARRAY_TASK_ID:-0}" = "0" ]; then METHOD=autodiff; else METHOD=zstar_eu; fi

echo "=== node ==="; hostname; date
echo "rung:   ${SLURM_ARRAY_TASK_ID:-0}  us $METHOD at 4^3, both constraint terms"
echo "commit: $(git -C "$REPO" rev-parse HEAD)  $(git -C "$REPO" log --oneline -1)"
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"

python3 -u tools/cluster/piezo_measure.py us \
        --nosym --kmesh 4 --skip-difference --method "$METHOD" \
        --out "$WORK/out/both-${SLURM_ARRAY_JOB_ID}-${METHOD}.json"
echo "exit $?"

echo "=== done ==="; date
