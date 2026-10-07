#!/bin/bash
# Elk's Si-dielectric propagation against defumat's, one core each, from a converged
# ground state on both sides (CLAUDE.md "Performance"; the reference-timing skill).
#
#   tools/realtime/time_against_elk.sh <elk binary> <elk species dir> <work dir> [core]
#
# Elk: task 0 (the ground state, not timed), then tasks 450 and 460 timed together
# (450 writes the vector potential, 460 propagates), with Elk's example input at
# scissor 0. defumat: tools/realtime/time_realtime.py, which times the fixed-density
# solve on the field's wedge and the propagation apart. The machine must be quiet.
set -euo pipefail
elk=$1; species=$2; work=$3; core=${4:-0}
here=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$work/elk"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_STACKSIZE=64M
ulimit -s unlimited 2>/dev/null || true
sed -e "s|'../../../species/'|'$species/'|" -e "s|^  0.0331|  0.0|" \
    "$here/tests/data/elk/si_rt/si-dielectric.elk.in" > "$work/elk/elk.in.full"
# the ground state alone
awk 'BEGIN{t=0} /^tasks/{print; print "  0"; t=1; next} t==1 && /^ *[0-9]+ *$/{next} {t=0; print}' \
    "$work/elk/elk.in.full" > "$work/elk/elk.in"
(cd "$work/elk" && taskset -c "$core" "$elk" > gs.stdout 2>&1)
# the propagation, timed
awk 'BEGIN{t=0} /^tasks/{print; print "  450"; print "  460"; t=1; next} t==1 && /^ *[0-9]+ *$/{next} {t=0; print}' \
    "$work/elk/elk.in.full" > "$work/elk/elk.in"
start=$(date +%s.%N)
(cd "$work/elk" && taskset -c "$core" "$elk" > td.stdout 2>&1)
end=$(date +%s.%N)
echo "{\"elk_450_460_s\": $(echo "$end - $start" | bc)}"
cd "$here"
DEFUMAT_THREADS=off taskset -c "$core" python3 -u tools/realtime/time_realtime.py "$work/defumat.npz"
