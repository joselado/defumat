#!/bin/bash
# OPEN.md Part VII item 1, step 1: where the 45-atom NiBr2 slab's device peak is set, stage by
# stage, on one H100, two arms on the same node: projectors rebuilt, then stored
# (plans/open-d-vii1-sizing.md as amended by plans/open-d-vii1-review.md).
#
# What it measures. tools/gpu/stage_peaks.py prints Calculator.estimate()'s report, inventories
# the device arrays after setup by shape, then runs the SCF with every stage bracketed: each
# bracket blocks on its result and records the device's whole memory_stats() before and after.
# The brackets nest, diagonalize > eigensolver > _every_k, with depth, parent and open/close
# sequence numbers on every entry, so a rise in diagonalize that its eigensolver does not hold
# is what ran between them (the psi0[spin] slice and the jnp.stack restack, A16's candidate).
# The diagonalize bracket logs ethr, the Davidson steps and the unsettled bands, the SCF runs
# at verbose=True, and a robust retry (a second _every_k with robust=True) is in the record
# with its warning, which PYTHONWARNINGS=always keeps from being deduplicated. Sixteen
# iterations an arm, past the point where ethr reaches 2.3e-6 and the Davidson average goes
# from 2 steps to 73 (iteration 10 on 2026-09-14), at conv_thr 1e-14 so nothing stops early.
#
# Prediction, written before it runs.
#   * The peak is set inside diagonalize at tight ethr, not in setup and not in the
#     torque or the density.
#   * If A16's candidate is the gap, diagonalize's peak_rise exceeds its eigensolver's by
#     one to two wavefunction sets (11.27 GiB each: the slice, then the restack); if it is
#     not, the two rises agree to a few per cent and the gap is inside the eigensolver.
#   * Inside the eigensolver, _every_k's rise matches the compile-only temp of
#     vii1-compile-h100.sbatch (about 29.6 GiB predicted there) plus its arguments.
#   * The stored arm's in_use after setup exceeds the rebuilt arm's by vkb, 13.00 GiB
#     (the September report's line), and its peak by about the same, 13.96 GB, as every
#     row of A13 did.
#   * September's H100 rebuild arm read 76.51 GB (A13), with the 4 GB solver change since;
#     the measured peak today is compared with today's estimate, not with that figure.
#
# The slab and its start, as jobs 20257133, 20258175 and 20258183 ran them:
# NiBr2_defumat/nibr2_k161_anderson.scf.in, pseudopotentials in NiBr2_defumat/pseudo, the
# helix seed of NiBr2_defumat/nibr2_persite.scf.in built by that project's own
# helix_starting_density (run_scf.py), passed as starting_density. Dials: k_batch 1, band
# batch 16, david 2, wavefunction store on the device, memory mode pinned to memory, the
# projectors the only dial that differs between arms. Every DEFUMAT_* variable is unset,
# preallocation is XLA's default at a pool of 0.97, autotuning at level 0 as nibr2.sbatch ran
# it in September.
#
# Each arm is its own process under `timeout`, so the stored arm gets its share of the four
# hours even if the rebuilt one is slow; a killed arm keeps every bracket written before the
# kill in its JSON, and <json>.host names the last stage and the brackets open.
#
# Submitted by hand only, after `sbatch --test-only` (CLAUDE.local.md):
#   sbatch /scratch/work/ladovj1/calculations/defumat-vii1/vii1-stages-h100.sbatch
#
# partition   gpu-h100-80g, one named H100 partition, so both arms and the compile twin are
#             on the same kind of card.
# time        04:00:00: the review's estimate; at September's rates an arm is about nine
#             loose iterations at 21 s and five to seven tight ones at 390 s, plus setup and
#             compilation, so 1 h 50 min an arm is the cap given to each.
# mem         240G of host memory, as every September job on this cell asked.
#SBATCH --job-name=vii1-stages-h100
#SBATCH --partition=gpu-h100-80g
#SBATCH --time=04:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240G
#SBATCH --output=/scratch/work/ladovj1/calculations/defumat-vii1/out/%x-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/defumat-vii1/out/%x-%j.err

WORK=/scratch/work/ladovj1/calculations/defumat-vii1
SLAB=/scratch/work/ladovj1/calculations/NiBr2_defumat
REPO=/scratch/work/ladovj1/apps/defumat-vii1
PY=/scratch/work/ladovj1/calculations/pypresso-gpu/venv/bin/python3
INPUT=$SLAB/nibr2_k161_anderson.scf.in
PSEUDO=$SLAB/pseudo
SEED=$SLAB/nibr2_persite.scf.in
SEED_SCRIPT=$SLAB/run_scf.py
ITERATIONS=16
ARM_SECONDS=6600
mkdir -p "$WORK/out" "$WORK/cache-gpu"
cd "$WORK" || exit 1

# Every DEFUMAT_* dial off, so the script's flags are the only dials.
for name in $(compgen -e | grep '^DEFUMAT_'); do unset "$name"; done
unset JAX_PLATFORMS XLA_PYTHON_CLIENT_PREALLOCATE
export DEFUMAT_CACHE_DIR="$WORK/cache-gpu"
export PYTHONPATH="$REPO" PYTHONWARNINGS=always OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.97
export XLA_FLAGS="--xla_gpu_autotune_level=0"

echo "=== node ==="; hostname; date
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$PY" -c "import jax, sys; print('jax', jax.__version__, jax.devices()); sys.exit(0 if jax.devices()[0].platform == 'gpu' else 4)" || exit 4
git -C "$REPO" rev-parse HEAD
git -C "$REPO" status --porcelain --untracked-files=no
env | grep -E '^(DEFUMAT_|XLA_|JAX_|PYTHONWARNINGS)' | sort
df -h "$WORK" | tail -1

for projectors in rebuild store; do
  tag="$SLURM_JOB_NAME-$SLURM_JOB_ID-$projectors"
  echo; echo "########## projectors $projectors ##########"; date
  timeout --signal=TERM --kill-after=60 "$ARM_SECONDS" \
    "$PY" -u "$REPO/tools/gpu/stage_peaks.py" "$INPUT" --pseudo-dir "$PSEUDO" \
      --iterations "$ITERATIONS" --k-batch 1 --band-batch 16 --david 2 \
      --wfc-store device --memory-mode memory --projectors "$projectors" \
      --seed-script "$SEED_SCRIPT" --helix-seed "$SEED" \
      --out "$WORK/out/$tag.json" > "$WORK/out/$tag.log" 2>&1
  echo "exit $?"; date
  grep -E '^\[mem\] (end|setup|resolved)|STAGE_PEAKS_DONE' "$WORK/out/$tag.log"
  cat "$WORK/out/$tag.host" 2>/dev/null; echo
done
echo "=== done ==="; date
