#!/bin/bash
# Regenerate this fixture.  Run it from inside tests/fixtures/h_sc/.
#
#   ./regenerate.sh [path/to/elk]
#
# Defaults to the out-of-tree build (../../../build/elk/src/elk), so run
# ./build_elk.sh at the repo root first.  Takes seconds on one core: one
# hydrogen atom, a 4x4x4 k-mesh, tasks 0 (ground state) and 33 (3D density).
#
# Everything Elk writes that is not part of the fixture is deleted at the
# end, so the committed set stays the six files listed in KEEP below.
set -euo pipefail

cd "$(dirname "$0")"
ELK="${1:-$(cd ../../.. && pwd)/build/elk/src/elk}"

if [ ! -x "$ELK" ]; then
  echo "no elk binary at $ELK -- run ./build_elk.sh at the repo root" >&2
  exit 1
fi

# elk.in and H.in are inputs, not outputs: elk.in pins the run (including
# tshift = .false.) and H.in is copied in beside it so sppath is './' and
# the fixture does not depend on the vendored species directory.
KEEP=(elk.in H.in regenerate.sh README.md GEOMETRY.OUT INFO.OUT STATE.OUT RHO3D.OUT)

"$ELK"

args=()
for f in "${KEEP[@]}"; do args+=(-not -name "$f"); done
find . -maxdepth 1 -type f "${args[@]}" -delete

echo "fixture regenerated:"
ls -la "${KEEP[@]}"
