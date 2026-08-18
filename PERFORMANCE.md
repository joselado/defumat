# Performance log

Measurements taken as features land, so that P10 (optimisation) starts from data
rather than from guesses. **Nothing here has been optimised yet** — the code is
written for clarity and for the differentiability rules in `PLAN.md` §6, and the
numbers below are the honest cost of that.

Reproduce with:

```bash
python3 tools/benchmark.py quantum_espresso/.../pw_scf/scf.in
```

## Against Quantum ESPRESSO

QE's own timing report is printed at the bottom of every reference output, so
the comparison costs nothing — but it is **indicative only**: those runs were
made on a different machine in a different year (the `pw_scf` reference is
stamped July 2017), with a compiled Fortran binary against tuned BLAS/FFTW.

| | QE (reported) | pypresso, first run | pypresso, warm |
|---|---|---|---|
| `pw_scf/scf.in` — Si, 2 atoms, 2 k-points, 186 PWs | 0.15 s total (`electrons` 0.02 s) | ~23 s | 1.0 s SCF (0.16 s/iteration) |
| `pw_metal/metal.in` — Al, 1 atom, 10 k-points, 107 PWs | 0.23 s total (`electrons` 0.08 s) | ~30 s | 2.3 s SCF (0.45 s/iteration) |

So: **roughly 100–150× slower on a first run, and 30–50× slower once warm**, on
the smallest systems in the test suite. Both figures are dominated by things
that are fixable and that do not scale the way the physics does — see below.

## Where the time goes (Si, `pw_scf/scf.in`, warm unless stated)

| Stage | Time | Note |
|---|---|---|
| Setup, first call | 11.3 s | almost entirely XLA compilation |
| ├ `build_projectors` | 4.7 s | |
| ├ `local_potential` | 2.8 s | |
| ├ `ewald_energy` | 1.3 s | includes a Python double loop over atoms |
| ├ per-k kinetic + FFT index | 1.0 s | |
| ├ `starting_charge` | 0.9 s | |
| ├ `build_basis` | 0.8 s | NumPy G-vector enumeration, genuine work |
| └ `find_symmetries` | 0.04 s | |
| First SCF run | 11.4 s | compilation again |
| Warm SCF, 6 iterations | 1.0 s | 0.16 s/iteration |
| ├ dense diagonalisation | 0.080 s × nk | `O(npw^3)`, see below |
| ├ `v_of_rho` | 0.027 s | |
| ├ `symmetrize` | 0.006 s | was 0.203 s before batching, see history |
| └ `h_psi`, all bands at one k | 0.004 s | the only part QE spends its time in |

## Diagnosis

**1. Compilation dominates at this size.** ~22 of the ~23 seconds of a first run
are XLA compiling kernels for arrays with a few thousand elements. QE has no
equivalent cost. This is a fixed overhead, not a scaling problem: it will be the
same few seconds on a system a hundred times larger, where it stops mattering.
It is also mostly avoidable — most of it is *eager* dispatch of many small
un-jitted operations, each compiled separately.

**2. The eigensolver is deliberately the wrong algorithm.** `solvers/dense.py`
builds the full Hamiltonian by applying it to every basis vector and calls
`eigh`: `O(npw^3)` time, `O(npw^2)` memory, against QE's Davidson which touches
only the few lowest states. At `npw = 186` this is free; at a realistic
`npw = 20000` it is impossible. This is the single largest algorithmic gap and
it is what P4's Davidson closes.

**3. Almost nothing is jitted yet.** The SCF iteration body dispatches its
operations one at a time. The `symmetrize` measurement is the evidence: it fell
from 0.203 s to 0.006 s purely by replacing a 48-iteration Python loop with one
batched gather — a 34× gain with no change to the arithmetic. The same
transformation applies to the whole iteration.

**4. Form factors are recomputed per G rather than tabulated.** This is a
deliberate trade for differentiability (`PLAN.md` D1/D2): QE interpolates a
`dq = 0.01` table, which is faster but not differentiable in `q`. The cost is
visible in `build_projectors` and `local_potential`. If it ever matters, the
answer is a *differentiable* interpolation, not a lookup table.

## Optimisation backlog

Ordered by expected gain per unit of effort. None of these may change a
validated number; P10 re-runs the regression suite to prove it.

1. **`jit` the SCF iteration body** — potential, `h_psi`, density, mixing as one
   compiled unit. Removes most per-op dispatch. (Compile once, not per call.)
2. **Davidson eigensolver** — the only change that alters the asymptotic cost.
3. **`vmap` over k-points and bands** instead of the Python loops in
   `Calculation.diagonalize` and `sum_band`.
4. **Batch the radial transforms** across species and projectors — currently one
   dispatch per projector.
5. **Vectorise the Ewald real-space sum** — it is a double Python loop over atom
   pairs.
6. **Shell-based radial evaluation** for quantities that depend only on `|G|`
   (~100 shells vs 1459 G-vectors for Si). Note this is *not* strain-safe: shells
   split under strain, so it must stay off the stress path.
7. **Avoid the per-iteration host round trip** in mixing (`np.asarray` on the
   density, then back).

## History

| Date | Change | Effect |
|---|---|---|
| 2026-08-18 | Batched symmetrisation: one gather over `(nsym, ngm)` instead of a Python loop | `symmetrize` 0.203 s → 0.006 s |
