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
#   tools/run_regression.sh [output-directory]
cd "$(dirname "$0")/.."
OUT=${1:-regression-results}
mkdir -p "$OUT"
SUMMARY=$OUT/summary.txt
for f in tests/regression/test_*.py; do
    name=$(basename "$f" .py)
    grep -q "^$name " "$SUMMARY" 2>/dev/null && continue      # already done
    python3 -m pytest "$f" -q -m regression --tb=line > "$OUT/$name.log" 2>&1
    status=$?
    line=$(tail -3 "$OUT/$name.log" | grep -E "passed|failed|error|no tests" | tail -1)
    echo "$name exit=$status | $line" >> "$SUMMARY"
done
echo "ALL FILES DONE" >> "$SUMMARY"
