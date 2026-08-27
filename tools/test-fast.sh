#!/usr/bin/env bash
# The pre-push gate: everything that is not marked `slow`.
#
# 1634 tests in about four and a half minutes, against the 588 slow ones that
# take over two hours. Run this before pushing; run the slow set when you want
# it, with `tools/run_regression.sh` (resumable) or `pytest -m slow`.
#
# The split is a marker, not a directory, so it cuts across `unit` and
# `regression` both: a cheap regression case against a two-atom QE reference is
# in the gate, and a slow *unit* test is not.
#
#   tools/test-fast.sh [extra pytest arguments]
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 -m pytest -m "not slow" "$@"
