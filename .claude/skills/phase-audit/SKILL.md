---
name: phase-audit
description: Check that a defumat phase actually left behind all five deliverables - a measured number, a README feature-table row, a docs/features.tex entry with an executed snippet and an amber box, a notebook, and a timing against the reference code. Runs the set-difference audit that catches documentation drift. Use when declaring a phase done, before a release, after a subagent reports a feature complete, and when unsure whether an implemented feature was ever written up. Triggers - "is this phase done", "audit the docs", "did we document X", "what is missing before I commit this".
---

# Phase audit

A phase is not done when the code runs. `CLAUDE.md` names five deliverables, and the
reason this skill exists is that **each of the five has gone stale silently at least
once**. So each is checked, never assumed.

Take the phase number or the feature name from the user. If they give neither, ask which
phase, or infer it from the most recent `PLAN.md` section and say which you picked.

## The audit is a set difference, not a read-through

This is the single most important instruction here. Reading the documentation and asking
"does this look complete" found nothing. Listing what the package exports and subtracting
what the documentation mentions found **ten missing entry points at once**, including a
whole implemented feature with no mention anywhere.

So build the two lists and subtract them. Do not skim.

```bash
# what the package actually exposes
grep -rn "^__all__" defumat/workflows/ defumat/response/ defumat/forces/ defumat/stress/ \
  defumat/topology/ defumat/transport/ defumat/stm/ | head -60
# and what the guide mentions
grep -c "" docs/features.tex
```

## 1. A number

`PLAN.md` section 3 must carry a **concrete figure** for this phase: against `pw.x`,
against Elk, or against an identity that shares no machinery with the assembly.

Check three things:

- The number is against something **independent**. A quantity checked only against
  another route through the same code is not validated. Where the check is an identity,
  say what machinery it does *not* share.
- **Where a term is missing, its absence is measured and refused by name.** A refusal
  with no number behind it is an untested claim; a term dropped silently is worse.
- The refusal list is current. A stale refusal list has been found more than once - a
  phase refusing something it had since implemented, and a phase inheriting a refusal
  that belonged to different machinery entirely.

An adjective where a number should be is a finding. So is "agrees well".

## 2. A row in the README feature table

- **One row, and it names a quantity, not a routine or a knob.** A new feature adds one
  row; its variants, schemes and internal terms go in `PLAN.md`. Smearing belongs inside
  the row about metals, not in a row of its own. A variant of an existing quantity
  updates that row rather than adding one.
- **The entry-point column is a claim about this code.** `grep` the variable in
  `defumat/io/pwin.py` and `defumat/system/builder.py`, and check the function is
  actually exported. A row naming something nothing parses is worse than no row.
- **The QE and Elk ticks are claims about someone else's source.** Before ticking either,
  the routine must have been located. Before leaving both blank - which is the mark of a
  quantity neither established code has, and the thing that tells a reader this is an
  extension rather than a reimplementation - both trees must have been grepped. Delegate
  this to the `reference-source` agent rather than doing it inline; it is a large search
  with a one-line answer.

## 3. An entry in `docs/features.tex`

There is no markdown copy and none should be added - two copies drift. An entry is four
things, and **the last two are the ones that get skipped**:

- **What it computes**, as an equation where there is one. This is a physics document.
- **The entry point**, checked by `grep` rather than remembered.
- **A snippet that has been run.** Checking a name exists in `dir()` is not enough - an
  audit that only did that passed six broken snippets. **Execute it.**
- **The amber box.** What the feature refuses. The refusals are the promise that a run
  which starts is a run whose physics is there, and that promise is only usable if its
  edges are written down.

Do not re-document standard `pw.x` variables - `ecutwfc`, `ibrav`, `nbnd` mean what they
mean in QE. Document this code's own knobs and the ones that gate a feature.

Build it to confirm it still compiles: `xelatex docs/features.tex` twice, for the table
of contents.

## 4. A notebook

`notebooks/`, and `tests/unit/test_notebook_conventions.py` enforces most of the shape -
run it rather than re-deriving what it checks, and check the new notebook joined the
`REWRITTEN` set in the commit that added it. The three exempt by design are `01`, `03`
and `17`, whose internals are their subject.

What the test cannot check, and you must:

- **The notebook is about the physics and nothing else.** No phase numbers, no QE Fortran
  file names, no transcribed-versus-differentiated tables, no `jvp`, tangents, frozen
  spheres, padding or compilation, no catalogue of traps, no account of how something was
  debugged. Two sentences survive from that side: one saying a derivative is taken of the
  energy itself rather than derived by hand, and one where a reference is unusual and the
  reader would otherwise not trust the comparison.
- **That rule binds the code cells too.** This is where notebooks drift, because the
  project's validation instinct moves into the code where a prose rule does not reach. An
  identity check looping over four pseudopotentials, a derivative against a closed form on
  a random matrix, a hand-built linear solve with a probe potential - each is the test
  suite's job being done in public. They belong in `tests/`, and the footer names the file
  they went to.
- **Where a `get_*` method exists, the notebook uses it** rather than building the
  quantity from internals.
- No em dashes anywhere.
- A figure that shows the physics, about eight code cells, and a `.md` export committed
  beside the `.ipynb`.
- **Ten minutes is a hard ceiling** on executing it, and `tools/export_notebooks.sh`
  enforces it. If one does not fit, the cell to cut is the sweep, not the physics.

## 5. A timing against the code it was taken from

`PERFORMANCE.md` must carry the reference implementation's wall clock **beside** ours -
not a ratio to a previous version of this code, not an absolute number alone. Single core
each. The reason is that the absolute number is the one worth having and the one nobody
measures: an internal timing says a feature costs 94 s without saying whether that is what
the physics costs or what this implementation costs.

Two things that must be *stated* in the entry, not left for a reader to notice:

- **What is not comparable.** LAPW's basis is not a plane-wave sphere; that belongs beside
  the number.
- **Which steps were added up.** Codes split post-processing differently, so the two sides
  must start from the same place, usually a converged ground state, and the entry says
  which steps were included.

Where the quantity genuinely has no counterpart in either code, the entry says **that**,
explicitly. An absent timing with no explanation is indistinguishable from a forgotten one.

## Report

One line per deliverable: present, missing, or stale - with the evidence. Then the set
difference in full: every exported entry point for this phase that `docs/features.tex`
does not mention. Then a short list of what to do, ordered by what a reader loses without it.
