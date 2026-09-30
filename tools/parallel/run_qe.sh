#!/bin/bash
# pw.x on <np> MPI ranks with -nk <nk>, every rank pinned to its own core.
#
#   OUT=results.jsonl QE_BIN=/path/to/bin tools/parallel/run_qe.sh <input> <label> <np> <nk>
#
# Each rank is pinned by taskset to one entry of cpu_layout.py's list (one CPU
# per physical core, cache order), because mpirun's own binding has left ranks
# unbound on a desktop and its policy differs between Open MPI versions. QE's
# own timer is read: the electrons WALL divided by the iteration count. Two
# repeats, both reported. Runs in a directory of its own under $QE_RUNS.
#
# Environment: QE_BIN (the directory holding pw.x and mpirun), QE_RUNS (default
# ./qe-runs), REPEATS (default 2), OUT (required), PYTHON (default python3).
set -u
input=$(readlink -f "$1"); label=$2; np=$3; nk=$4
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-python3}
: "${OUT:?set OUT to the results file}"; : "${QE_BIN:?set QE_BIN to the directory holding pw.x}"
OUT=$(readlink -f "$OUT")
dir=${QE_RUNS:-$PWD/qe-runs}/$label-$(basename "$input" .in); mkdir -p "$dir"; cd "$dir" || exit 1
ulimit -s unlimited
export ESPRESSO_PSEUDO=$REPO/tests/data/pseudo ESPRESSO_TMPDIR=$dir/tmp \
       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export QE_CPUS=$("$PYTHON" "$REPO/tools/parallel/cpu_layout.py" "$np" 1)
cat > pin.sh <<'SH'
#!/bin/bash
IFS=';' read -ra lists <<< "$QE_CPUS"
rank=${OMPI_COMM_WORLD_LOCAL_RANK:-${PMIX_RANK:-${SLURM_LOCALID:-0}}}
exec taskset -c "${lists[$rank]}" "$@"
SH
chmod +x pin.sh
outs=()
for ((rep = 1; rep <= ${REPEATS:-2}; rep++)); do
  if [ "$np" -gt 1 ]; then
    "$QE_BIN/mpirun" -np "$np" --bind-to none -x QE_CPUS -x ESPRESSO_PSEUDO -x ESPRESSO_TMPDIR \
        -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS \
        ./pin.sh "$QE_BIN/pw.x" -nk "$nk" -in "$input" > out.$rep 2> err.$rep
  else
    ./pin.sh "$QE_BIN/pw.x" -in "$input" > out.$rep 2> err.$rep
  fi
  outs+=(out.$rep)
done
"$PYTHON" - "$label" "$input" "$np" "$nk" "$QE_CPUS" "${outs[@]}" >> "$OUT" <<'PY'
import json, re, sys
label, inp, np_, nk, cpus, *outs = sys.argv[1:]
per, iters, energy = [], None, None
for name in outs:
    text = open(name).read()
    m = re.search(r"convergence has been achieved in\s+(\d+) iterations", text)
    w = re.search(r"electrons\s*:.*?([\d.]+)s WALL", text)
    e = re.search(r"!\s+total energy\s+=\s+([-\d.]+)", text)
    if m and w:
        iters = int(m.group(1)); per.append(round(1e3 * float(w.group(1)) / iters, 1))
    if e:
        energy = float(e.group(1))
print(json.dumps({"label": label, "input": inp.split("/")[-1], "code": "pw.x", "np": int(np_),
                  "nk": int(nk), "cpus": cpus, "iterations": iters, "ms_per_iter_all": per,
                  "ms_per_iter_min": min(per) if per else None, "energy_ry": energy,
                  "failed": not per}))
PY
