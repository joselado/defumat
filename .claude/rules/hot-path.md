---
paths:
  - "defumat/hamiltonian/**"
  - "defumat/solvers/**"
  - "defumat/basis/**"
  - "defumat/scf/**"
  - "defumat/batching.py"
  - "benchmarks/**"
  - "tools/compare_qe.py"
  - "tools/benchmark.py"
---

*Moved out of `CLAUDE.md` verbatim on 2026-10-08; it loads only when files matching the paths above are read. The text is binding exactly as it was.*

## Mirror QE in the performance-critical path

**Where performance matters, reproduce QE's implementation rather than inventing
one.** Not just its formulas — its data layout, its loop structure, and the order it
does things in. Thirty years of plane-wave practice is encoded in choices that look
arbitrary until they are measured, and the measurement usually agrees with the Fortran.

This is a standing rule because guessing has now been wrong more than once, always in
the same direction — an idiomatic-JAX version that looked equivalent and was slower:

- **The FFT layout.** QE transforms the wavefunction `z` axis only over the *sticks*
  the sphere occupies, then does a 2D `xy` pass — and its arrays are Fortran-ordered,
  so the `xy` plane is contiguous. Transcribing the decomposition into a C-ordered box
  puts the 2D pass on the two strided axes, where it costs more on its own than a fused
  3D transform of the whole box; done in QE's layout it is a win. Same algorithm,
  opposite result, and the difference is entirely the layout (`basis/sticks.py`).
- **The Davidson loop.** `cegterg` extends its projected matrices a block at a time and
  tests convergence *after* expanding. Recomputing the projections each step costs a
  factor of `nvecx/nbnd`; testing before expanding wastes one `h_psi` per call. Both
  were invisible on a two-atom cell and obvious on eight.
- **The diagonalisation threshold.** `electrons.f90` schedules `ethr` against the error
  in the density. A fixed tight threshold does three times the eigensolver work.

The corollary for measurement: **a two-atom cell will not show you any of this.**
Benchmark on `benchmarks/si8-1k*.in` or `si16-1k*.in`, where the cost is the physics
rather than fixed overheads, and check that a change helps *there* before believing it.

Two things this rule does not mean. It does not license transcribing QE's Fortran
control flow into Python — the JAX rules above still bind, and `cegterg`'s dynamic
reshaping becomes masks and static shapes. And it does not override differentiability:
where QE's fast path is a table lookup, the differentiable equivalent wins (`PLAN.md`
D1/D2), and that trade is recorded rather than silently taken.
