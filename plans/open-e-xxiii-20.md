# Plan: the Davidson step at the width of `notcnv` (OPEN.md Part XXIII item 20)

Written 2026-10-09 on branch `open-e`, for a fable review before any code. Nothing here is
implemented.

## What the entry asks, and what the code does today

`cegterg` forms the correction vectors, the residual norms and the new rows of `hc` and `sc`
for the `notcnv` unconverged roots only (`cegterg.f90:353-395`, `:478-499`), and rotates to
all `nbnd` Ritz vectors only at a refresh or at the exit. This code, read at `aa08c82`:

- `ritz` (`solvers/davidson.py`) forms `evc` and `hevc` for all `nbnd` roots at the top of
  every step;
- `expansion` builds the residual, the preconditioned correction and its norm for all
  `nbnd` rows, sorts the unsettled ones to the front and zeroes the rest;
- `_extend_projection` is called with `block = nbnd`, so its rows past `notcnv` are exact
  zeros, and only the `h_psi` of that block is already narrowed (the band ladder,
  `_band_ladder`, backlog item 3, `07a4332`).

So what is left is the width of three things: the rotation that forms `evc`/`hevc`, the
residual chain in `expansion`, and the projection rows.

## The bound, and why this is phase-sized rather than small

The entry's own bound is 50 to 100 ms of a 5.5 s SCF on `si16-1k-ecut30`, 1 to 2 per cent:
a Ritz rotation is 3.2 to 11.7 ms at widths 32 to 128 and `_extend_projection` 11.3 ms
(`PERFORMANCE.md`, "Inside a Davidson step"), at a live fraction of 0.72 and about 11
in-loop steps. It grows as `nbnd^2 npwx` against `h_psi`'s `nbnd npwx log N`, so the only
place it pays is a large cell, which is also where the rematerialisation pass is the binding
constraint and each new `lax.switch` branch is a copy of HLO (the reason `DEFUMAT_BAND_RUNGS`
exists). That trade, speed on a large cell against compile size and peak on the same cell,
is the reason this is not done as a small change.

## Design

1. **Permute first, then narrow.** `expansion` already computes the stable sort that puts
   the unsettled roots first. Apply that permutation to the *coefficient columns* (an
   `(nvecx, nbnd)` gather, cheap) before forming anything, so the first `notcnv` columns
   are the roots to correct.
2. **One rung, three consumers.** Inside the existing band-ladder `lax.switch` (the rung
   that covers `notcnv`, `r`), form `evc_r = coefficients[:, :r]^T psi[:live]` and
   `hevc_r` the same way, the residual and the correction for those `r` rows only, and call
   `_extend_projection(..., block=r)`. The rung is already chosen for `h_psi`; reusing it
   adds no new branch count, only wider branches.
3. **Full `evc` only where it is read.** The refresh (`refresh`) and the exit
   (`final` in `davidson_eigensolver`) rotate to all `nbnd`, as `cegterg` does. The
   `residual_threshold` path, off by default, keeps today's full form.
4. **Routes where `narrow` is false** (an unnamed `vmap`, `lax.map` chunks): keep today's
   form. A `cond` or `switch` under an unnamed `vmap` evaluates every branch, which is the
   existing rule in `_every_k`.

## What must not move, and how it is checked

- Per-k Davidson step counts and `notcnv` identical on `si8-1k-ecut30`, `si16-1k-ecut30`,
  `si8-us-1k`, `pt-so-1k` (spinor) and one gamma-only cell, one process per arm.
- Eigenvalues to 1e-13 Ry and the converged total to 1e-12 Ry; the rows are the same sums
  over the same terms in a different order of evaluation, so the expectation is round-off,
  not bit-identity. Rule D4 for any degenerate multiplet.
- `memory_analysis().temp_size_in_bytes` of the compiled `_every_k` on `si64` at
  `band_batch = 16` (837.1 MB on record) and the compile time of `_every_k`, both arms,
  since a wider switch branch is where a regression would show.

## Measurement

`tools/compare_qe.py --repeats 5` on `si16-1k-ecut30.in`, `si8-1k-ecut30.in` and
`benchmarks/si64-1k-ecut30.in`, single core of D22 (`taskset -c 0`), the Davidson steps
equal between arms. Accept if si64 gains at least what the bound predicts (about 2 per cent)
with the temp size not above the old one; otherwise record the null in the entry and stop.

## Risks to look at in review

- The permutation of the coefficients has to be the same one the correction block is
  written in, or the projection rows are attached to the wrong vectors; a test that compares
  `hc`/`sc` after one step between the two forms catches it.
- At `notcnv = 0` the step is skipped by the loop condition; at the top rung `r = nbnd` the
  narrowed path must reduce to today's arithmetic exactly.
