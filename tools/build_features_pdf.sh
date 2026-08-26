#!/bin/bash
# Typeset docs/features.md as a PDF.
#
# The markdown is the GitHub-facing copy and the PDF is the readable one: the
# maths renders properly here whatever the reader's viewer does, which is why
# this exists. Both come from the same source, so they cannot drift.
#
#   tools/build_features_pdf.sh
#
# Needs pandoc and xelatex -- xelatex rather than pdflatex because the guide
# carries Unicode (≤, Å, Γ, superscripts) that pdflatex refuses outright. ```math fences are pandoc code blocks, not maths,
# so they are turned back into $$ display maths on the way in.
set -e
cd "$(dirname "$0")/.."
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

python3 - "$tmp/features.md" <<'PY'
import re, sys
from pathlib import Path
text = Path("docs/features.md").read_text()
# ```math ... ``` -> $$ ... $$, which pandoc reads as display maths -- and each
# block is collapsed onto ONE line first. A continuation line beginning "+" or
# "-" is a markdown *list item* to pandoc's reader, which runs before the maths
# does, so a wrapped equation silently becomes a bullet with LaTeX in it.
# LaTeX treats newlines inside maths as whitespace, so joining changes nothing.
def _one_line(m):
    body = " ".join(line.strip() for line in m.group(1).split("\n"))
    return "$$\n" + body + "\n$$"
text = re.sub(r"```math\n(.*?)\n```", _one_line, text, flags=re.S)
# The contents list is pandoc's job, not the source's.
text = re.sub(r"## Contents\n\n(?:\d+\..*\n)+\n---\n", "", text)
Path(sys.argv[1]).write_text(text)
PY

pandoc "$tmp/features.md" \
  --from=gfm+tex_math_dollars+pipe_tables \
  --to=latex --standalone --toc --toc-depth=2 \
  --pdf-engine=xelatex \
  --highlight-style=tango \
  -V documentclass=article -V papersize=a4 -V fontsize=11pt \
  -V geometry:margin=2.3cm -V colorlinks=true -V linkcolor=NavyBlue \
  -V urlcolor=NavyBlue -V toccolor=black \
  -V mainfont="DejaVu Serif" -V monofont="DejaVu Sans Mono" \
  -V title="pypresso: what it computes, and how to ask for it" \
  -V subtitle="Plane-wave DFT in Python and JAX, validated against Quantum ESPRESSO" \
  -V date="$(date +%Y-%m-%d)" \
  -o docs/features.pdf

pandoc "$tmp/features.md" --from=gfm+tex_math_dollars+pipe_tables --to=latex \
  --standalone --toc --toc-depth=2 --highlight-style=tango \
  -V documentclass=article -V papersize=a4 -V fontsize=11pt \
  -V geometry:margin=2.3cm -V colorlinks=true \
  -V title="pypresso: what it computes, and how to ask for it" \
  -V subtitle="Plane-wave DFT in Python and JAX, validated against Quantum ESPRESSO" \
  -o docs/features.tex

echo "wrote docs/features.pdf and docs/features.tex"
