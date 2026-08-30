#!/usr/bin/env bash
# Re-execute every tutorial notebook and refresh its markdown export.
#
# Notebooks are committed with their outputs, and each one also has a .md export
# beside it so the tutorial is readable and diffable without a notebook viewer --
# raw .ipynb JSON is not something anyone can review in a plain editor.
#
# **It times each one and fails over the ceiling.** Ten minutes per notebook is a
# hard limit and five is the target (CLAUDE.md): a notebook is re-executed every
# time the code under it changes, so its runtime is paid over and over by people
# who are not doing physics at the time. The times print as the run goes and are
# summarised at the end, which is how the set gets measured at all -- most of it
# has never been timed, and the one notebook that was found over the ceiling took
# 25 minutes for cells that were doing the test suite's job in public.
#
# Run from the repository root after changing code the notebooks depend on.
# A single notebook: tools/export_notebooks.sh notebooks/02_*.ipynb
set -uo pipefail

cd "$(dirname "$0")/.."

CEILING=${CEILING:-600}          # seconds; the hard limit CLAUDE.md sets
notebooks=("$@")
if [ ${#notebooks[@]} -eq 0 ]; then
    notebooks=(notebooks/*.ipynb)
fi

summary=()
over=0
failed=0

for notebook in "${notebooks[@]}"; do
    echo "== $notebook"
    start=$SECONDS
    if jupyter nbconvert --to notebook --execute --inplace \
           --ExecutePreprocessor.timeout=900 "$notebook"; then
        elapsed=$((SECONDS - start))
        jupyter nbconvert --to markdown --output-dir notebooks "$notebook"
        mark=""
        if [ "$elapsed" -gt "$CEILING" ]; then
            mark="  <- OVER THE ${CEILING}s CEILING"
            over=$((over + 1))
        fi
        printf '   %4ds%s\n' "$elapsed" "$mark"
        summary+=("$(printf '%6ds  %s%s' "$elapsed" "$notebook" "$mark")")
    else
        elapsed=$((SECONDS - start))
        failed=$((failed + 1))
        summary+=("$(printf '%6ds  %s  <- FAILED' "$elapsed" "$notebook")")
    fi
done

echo
echo "-- wall time, slowest last"
printf '%s\n' "${summary[@]}" | sort -n

if [ "$failed" -gt 0 ]; then
    echo
    echo "$failed notebook(s) failed to execute"
    exit 1
fi
if [ "$over" -gt 0 ]; then
    echo
    echo "$over notebook(s) over the ${CEILING}s ceiling. The cell to cut is the"
    echo "sweep, not the physics: measure an expensive series once offline and"
    echo "quote its numbers (CLAUDE.md, 'Tutorial notebooks')."
    exit 1
fi
echo
echo "done -- review the .md diffs before committing"
