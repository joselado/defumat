#!/usr/bin/env bash
# The pre-push gate: everything that is not marked `slow`.
#
# 2609 tests in about ten minutes at a 5.5 GB peak, against the 1279 slow ones
# that take over two hours. Run this before pushing; run the slow set when you
# want it, with `tools/run_regression.sh` (resumable) or `pytest -m slow`.
#
# The gate is kept at roughly that size deliberately: it is paid on every push
# by someone who is not doing physics at the time. When it drifts, the lever is
# the `slow` marker and the rule is cost -- a test above about five seconds
# that is not a direct number against `pw.x`, `projwfc.x` or Elk belongs in the
# slow set, and a reference comparison stays.
#
# The split is a marker, not a directory, so it cuts across `unit` and
# `regression` both: a cheap regression case against a two-atom QE reference is
# in the gate, and a slow *unit* test is not.
#
#   tools/test-fast.sh [extra pytest arguments]
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 -m pytest -m "not slow" "$@"
