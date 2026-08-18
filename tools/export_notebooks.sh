#!/usr/bin/env bash
# Re-execute every tutorial notebook and refresh its markdown export.
#
# Notebooks are committed with their outputs, and each one also has a .md export
# beside it so the tutorial is readable and diffable without a notebook viewer --
# raw .ipynb JSON is not something anyone can review in a plain editor.
#
# Run from the repository root after changing code the notebooks depend on.
set -euo pipefail

cd "$(dirname "$0")/.."

for notebook in notebooks/*.ipynb; do
    echo "== $notebook"
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=900 "$notebook"
    jupyter nbconvert --to markdown --output-dir notebooks "$notebook"
done

echo "done -- review the .md diffs before committing"
