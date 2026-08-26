#!/bin/bash
# GPU.md Phase 5, continued: the response path on a cell that is not two atoms.
#
# The first Phase 5 run answered feasibility and found that **the mode decides
# the speedup**: a Sternheimer solve got 1.0x from an H200 while a reverse-mode
# gradient got 339x. It also carried its own caveat, in PERFORMANCE.md's own
# words:
#
#   the response cases here are two-atom cells, and CLAUDE.md's rule is that a
#   two-atom cell shows none of this. The SCF sweep needed ten atoms to move
#   from 3x to 115x. So the 1.0-1.4x on the Sternheimer rows is a fact about
#   *these* cells, not a ceiling for the response path -- the next measurement
#   is si10-epsilon or larger, where the per-k work is real.
#
# This job is that measurement. si10-epsilon.in is ten silicon atoms, sixteen
# k-points, nosym, conv_thr = 1e-12, and it is already validated against the
# vendored ph.x to 7.0e-6 on every component of epsilon, Z* and the whole 30-mode
# Gamma dynamical matrix (PLAN.md, tests/regression/test_ten_site.py) -- so the
# value each property returns is a correctness check as well as a timing.
#
# For scale, on the development workstation at four pinned cores:
#
#   dielectric  245 s warm, +1.30 GB of host RSS over its own SCF
#   born        302 s warm, +1.58 GB
#
# against 12.4 s and 10.8 s for the two-atom si-epsilon. That is a twentyfold
# larger response and it is the regime the caveat asks about.
#
# Its pair is phase5-si10-cpu.sbatch: the same script on four pinned Milan
# cores, which is §2.3's metric.
#
# ---- the #SBATCH values, and why each one ----
#
# partition   gpu-h200-141g-short, as phase5-gpu.sbatch uses. **Verify with
#             `sbatch --test-only` rather than inheriting this reading.**
# time        04:00:00, matching phase5-gpu.sbatch. Each property carries its
#             own `timeout`, so an overrun is a reported row and not a lost job.
# cpus-per-task 4, matching the CPU half exactly.
# mem         160G host memory, matching phase5-gpu.sbatch. Device memory is
#             what is measured rather than requested -- every two-atom property
#             sat under 0.25 GB of HBM, and what this job establishes is how
#             that scales with nk and ngm.
#SBATCH --job-name=pypresso-gpu-phase5-si10
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --time=04:00:00
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --output=/scratch/work/ladovj1/calculations/pypresso-gpu/out/gpu-phase5-si10-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/pypresso-gpu/out/gpu-phase5-si10-%j.err

WORK=/scratch/work/ladovj1/calculations/pypresso-gpu
REPO=/scratch/work/ladovj1/apps/pypresso
mkdir -p "$WORK/out" "$WORK/cache-gpu"
cd "$REPO" || exit 1
source "$WORK/venv/bin/activate"

unset XLA_PYTHON_CLIENT_PREALLOCATE
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYPRESSO_CACHE_DIR="$WORK/cache-gpu"
export PYTHONPATH="$REPO"

echo "=== node ==="; hostname; date
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"
git -C "$REPO" rev-parse HEAD

# The dials stay unset: since e562427 they follow the platform, and phase5.py
# records what it resolved them to. --repeats 2 separates compile from warm and
# gives every row its own determinism check -- which on a device is a *tolerance*
# and not a byte comparison, because three of the eight rows in the first run
# were not bit-reproducible (§4a).
run () {  # run <case> <property> <timeout-seconds>
  echo; echo "########## $1 $2 ##########"
  timeout "$3" python3 tools/gpu/phase5.py "$1" --property "$2" --repeats 2 \
          --json "$WORK/out/gpu-phase5-si10-${SLURM_JOB_ID}-$1.$2.json"
  echo "exit $?"
}

# Forward first, because that is the mode the 1.0x was measured in and the whole
# question here is whether ten atoms move it.
run si10-epsilon  dielectric  3600
run si10-epsilon  born        3600
# Forward-over-reverse at thirty modes, which is the other unmeasured axis: the
# two-atom dynamical matrix got 1.0x and taped 2.1 GB of host RSS.
run si10-epsilon  phonon      5400
# The two-atom rows again in the same job, so that this card and this driver
# carry their own copy of the numbers the ten-atom rows are read against. Cheap.
run si-epsilon    dielectric  1800
run si-epsilon    phonon      2400

echo; echo "=== done ==="; date
