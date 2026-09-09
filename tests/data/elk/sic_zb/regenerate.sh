#!/bin/bash
# Regenerate this fixture.  Run it from inside tests/fixtures/sic_zb/.
#
#   ./regenerate.sh [path/to/elk]
#
# Defaults to the out-of-tree build (../../../build/elk/src/elk), so run
# ./build_elk.sh at the repo root first.  The binary must carry the elkpy
# patch series -- task 9006 is patches/0024-initial-state.patch, not
# upstream Elk.  Seconds on one core: 0.5 s for the initial state, 6 s for
# the ground state and the density plot.
#
# TWO runs, because Elk writes both states to the same filename.  Task 9006
# runs first in a scratch subdirectory and its STATE.OUT is moved out as
# STATE_INIT.OUT; then tasks 0 and 33 run here and write STATE.OUT.  Doing
# it the other way round would have task 9006 overwrite the converged file.
#
# Everything Elk writes that is not part of the fixture is deleted at the
# end, so the committed set stays the files listed in KEEP below.
set -euo pipefail

cd "$(dirname "$0")"
ELK="${1:-$(cd ../../.. && pwd)/build/elk/src/elk}"

if [ ! -x "$ELK" ]; then
  echo "no elk binary at $ELK -- run ./build_elk.sh at the repo root" >&2
  exit 1
fi

# elk.in, elk_init.in and C.in are inputs, not outputs.  The two elk.in
# files differ only in the tasks block; tests/test_state_fixture.py asserts
# that, so they cannot drift apart silently.
KEEP=(elk.in elk_init.in C.in Si.in regenerate.sh README.md
      GEOMETRY.OUT INFO.OUT STATE.OUT STATE_INIT.OUT RHO3D.OUT)

rm -rf init
mkdir init
cp elk_init.in init/elk.in
cp C.in Si.in init/
( cd init && "$ELK" )
mv init/STATE.OUT ./STATE_INIT.OUT
rm -rf init

"$ELK"

args=()
for f in "${KEEP[@]}"; do args+=(-not -name "$f"); done
find . -maxdepth 1 -type f "${args[@]}" -delete

echo "fixture regenerated:"
ls -la "${KEEP[@]}"
