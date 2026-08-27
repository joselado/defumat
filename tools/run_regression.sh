#!/usr/bin/env bash
# Run the regression suite one file at a time, appending a durable summary line
# per file. A kill loses at most the file in flight, not the whole run -- rerun
# and it resumes, because a file already named in the summary is skipped.
#
# It exists because the suite is 1477 tests and hours long, so it was not being
# run: three separate phases' claims had drifted unnoticed by the time P38
# looked (P29's refusal list, P36's wedge number, and a `relax_spiral_q` that
# raised before taking a step). `python3 -m pytest -m regression` in one go is
# the same tests and is the right thing when you can wait for it.
#
# It runs the **slow** set by default, because that is the one that needs this
# treatment: the fast group is `tools/test-fast.sh` and finishes in one go.
# A second argument overrides the marker expression.
#
#   tools/run_regression.sh [output-directory] [marker-expression]
cd "$(dirname "$0")/.."
OUT=${1:-regression-results}
MARK=${2:-slow}
mkdir -p "$OUT"
SUMMARY=$OUT/summary.txt
for f in tests/regression/test_*.py; do
    name=$(basename "$f" .py)
    grep -q "^$name " "$SUMMARY" 2>/dev/null && continue      # already done
    python3 -m pytest "$f" -q -m "$MARK" --tb=line > "$OUT/$name.log" 2>&1
    status=$?
    # pytest exits 5 when a file holds nothing matching the marker, which is a
    # normal outcome here and not a failure.
    [ "$status" -eq 5 ] && status=0
    line=$(tail -3 "$OUT/$name.log" | grep -E "passed|failed|error|no tests" | tail -1)
    echo "$name exit=$status | $line" >> "$SUMMARY"
done
echo "ALL FILES DONE" >> "$SUMMARY"
