---
paths:
  - "notebooks/**"
  - "tools/export_notebooks.sh"
  - "tests/unit/test_notebook_conventions.py"
---

*Moved out of `CLAUDE.md` verbatim on 2026-10-08; it loads only when files matching the paths above are read. The text is binding exactly as it was.*

## Tutorial notebooks

`notebooks/` holds worked examples on concrete systems — the readable counterpart to the
test suite. **Every new feature adds a notebook or extends an existing one.** Demonstrate
on the two-atom silicon cell from `test-suite/pw_scf/scf.in` wherever possible, compare
against the committed QE benchmark whenever the reference contains the quantity, and commit
the notebook executed so it reads without being run. `notebooks/README.md` carries the
cell-by-cell shape, the per-notebook timings and the index — which is **by the property a
reader wants to compute**, not by the order the code gained it. Conventions are enforced by
`tests/unit/test_notebook_conventions.py`, whose `REWRITTEN` set **only ever grows**, so a
new notebook joins it in the commit that adds it. The three that are *not* in it — `01`,
`03` and `17` — are the under-the-hood tier by design, whose internals are their subject;
they are exempt rather than unfinished.

**A notebook is about the physics, and nothing else.** What the quantity is, the equation
that defines it, what the number means, how it compares with experiment or with Quantum
ESPRESSO. **The implementation is not the subject and must not appear**: no `PLAN.md` phase
numbers, no QE Fortran file names, no transcribed-versus-differentiated tables, no `jvp`,
tangents, frozen spheres, padding or compilation, no catalogue of traps, and no account of
how something was developed or debugged. Two things survive from that side because they are
claims about capability rather than about code: one sentence saying a derivative is taken of
the energy itself rather than derived by hand, and one sentence where a reference is unusual
and the reader would otherwise not trust the comparison. **No em dashes** anywhere in a
notebook.

**That rule binds the code cells, not only the prose.** Bounding only the prose is why the
notebooks drifted into validation reports: the project's validation instinct moved into the
code, where the rule did not reach. An identity check looped over four pseudopotentials, a
derivative checked against a closed form on a random matrix, a hand-built linear solve with
a probe potential — each is the test suite's job being done in public. They belong in
`tests/`, and the notebook's footer names the file they went to. **Where a `get_*` method
exists, the notebook uses it** rather than building the same quantity from internals.

**A notebook is five minutes long, and it has a figure.** Header saying what this computes
and the headline number against QE; the shortest code that runs it; **one plot that shows
the physics**, a band structure wherever the feature shows in bands; one comparison table;
at most one "how it works" cell for the single best idea, and it is a *physical* idea. About
eight code cells. Each notebook also has a `.md` export committed beside it — raw `.ipynb`
is unreadable in a plain editor or a diff — regenerated together with the notebook by
`tools/export_notebooks.sh`.

**Ten minutes is the hard ceiling on executing one**, and it is a ceiling rather than a
target. A notebook is re-executed every time the code under it changes, so its runtime is
paid over and over by people who are not doing physics at the time. Time it before
committing it:

```bash
time jupyter nbconvert --to notebook --execute --inplace notebooks/<n>.ipynb
```

`tools/export_notebooks.sh` times each one as it re-executes it and exits non-zero over the
ceiling, so the set stays measured without anyone remembering to measure it. **If one does
not fit, the cell to cut is the *sweep*, not the physics**: measure the expensive series
once offline and quote its numbers in prose. A figure that needs ten SCF runs to draw is a
figure whose points belong in a test — which is what the one notebook that was ever over
the ceiling turned out to be doing (`notebooks/README.md` has the case study).
