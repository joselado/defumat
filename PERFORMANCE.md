# Performance log

The measurement this project is judged by is **the same input through Quantum
ESPRESSO and through pypresso, on this machine, with both restricted to one
core**. Everything else — component breakdowns, compilation counts, guesses
about where time goes — is diagnosis in service of that one number.

```bash
python3 tools/compare_qe.py benchmarks/si-1k.in --repeats 5
```

Single core on both sides is what makes the comparison mean anything. QE is
built serial (`./configure --disable-parallel --disable-openmp && make -j pw`)
and run with `OMP_NUM_THREADS=1`; pypresso runs with XLA's intra-op thread pool
pinned to one thread, since otherwise JAX quietly uses all 14 cores and the
comparison flatters it by the core count. The tool pins both itself.

Two benchmark inputs, both a single k-point (`benchmarks/`):

| Input | | Why |
|---|---|---|
| `si-1k.in` | Si, 2 atoms, `ecutwfc = 12`, 180 plane waves | the test suite's silicon, one k-point |
| `si-1k-ecut40.in` | the same cell at `ecutwfc = 40`, 1131 plane waves | a production cutoff, where scaling starts to show |

One k-point on purpose: both codes parallelise over k, so a multi-k comparison
measures batching rather than the cost of the physics.

## Where it stands

Single core, this machine, 2026-08-18. `conv_thr = 1e-10` where the input allows
it, so both codes converge to the same place and the energy agreement doubles as
a correctness check on every optimisation.

| | QE 7.5 | pypresso | ratio |
|---|---|---|---|
| **`si-1k.in`** — 180 PWs, 1 k-point | | | |
| per SCF iteration | 0.003 s | 0.007 s | **3.0x** |
| total energy | −15.25444871 Ry | −15.25444945 Ry | Δ 7e-7 Ry |
| **`si-1k-ecut40.in`** — 1131 PWs, 1 k-point | | | |
| per SCF iteration | 0.011 s | 0.038 s | **3.3x** |
| total energy | −15.30461021 Ry | −15.30461021 Ry | Δ 3e-9 Ry |
| **`pw_scf/scf-kauto.in`** — 2 k-points, reduced from 8 | | | |
| per SCF iteration | 0.003 s | 0.012 s | **4.7x** |
| total energy | −15.79449452 Ry | −15.79449594 Ry | Δ 1e-6 Ry |
| **`pw_metal/metal.in`** — Al, 10 k-points | | | |
| per SCF iteration | 0.013 s | 0.061 s | **4.6x** |
| total energy | −4.18546970 Ry | −4.18546964 Ry | Δ 6e-8 Ry |

Best of five runs on each side, and worth taking as ±20%: QE prints its timings
to 0.01 s, so on the small cases its per-iteration figure is one significant
digit and the ratios inherit that. The multi-k cases are the worst because ten
k-points multiply the per-dispatch overhead that the single-k cases mostly hide.

**Both codes are pinned to one CPU by affinity**, which is the only mechanism
that works — see "Threads" below. An earlier version of the harness passed
`intra_op_parallelism_threads=1` inside `XLA_FLAGS`; XLA ignored it silently
(the token does not begin with `--`, so it was read as a filename) and pypresso
ran on 1.8 cores while the table claimed one. The ratios were not flattered by
it — extra threads make this workload *slower*, so the honest single-core
numbers came out slightly better — but the claim was wrong and is now enforced
by `os.sched_setaffinity`.

**Setup and process wall time.** With the compilation cache warm, setup is
1.0–1.3 s across all four cases and a complete silicon SCF takes **4.3 s of
process wall time**, against 9.7 s before the cache and 0.09 s for QE. What
remains is Python and JAX import, plus the compilations the cache cannot serve.
The honest comparison of the *physics* is still the per-iteration number; this
row is what a user waits for.

## What moved, and by how much

Baseline is the state before this optimisation pass, measured the same way on
`si-1k.in`.

| | before | after | |
|---|---|---|---|
| setup, cache warm | 9.96 s | 1.1 s | 9x |
| process wall, whole run | — | 9.7 s → 4.3 s | 2.2x |
| SCF, warm | 0.920 s | 0.058 s | **16x** |
| per SCF iteration | 0.131 s | 0.008 s | **16x** |
| against QE, per iteration | 53x | 3.3x | |
| Davidson steps for a run | 75 | 21 | 3.6x |

Every validated number is unchanged: the full test suite passes, and the two
eigensolvers agree on silicon's total energy to 2e-13 Ry.

### 1. The eigensolver (the big one)

Two changes, in the order they were made, because the profile said the solver
was 40 ms of a 52 ms iteration.

**The dense matrix now uses its matrix elements.** It was built by applying the
operator to every basis vector — one FFT per plane wave. The local part is
`V(G_i - G_j)`, a gather from a single transform of the potential; the kinetic
part is the diagonal; the nonlocal part is `vkb D vkb†`. 180 FFTs became 1, and
the build went 49 ms → 6 ms. The old build is kept as
`Hamiltonian.matrix_by_application` and the test suite asserts the two agree —
it uses no formula at all, which is what makes it the reference.

**Block Davidson** (`solvers/davidson.py`), transcribed from QE's `cegterg.f90`,
completing P4. This is the change that alters the asymptotic cost: `nbnd`
applications of `H` per step and a `4·nbnd` subspace solve, instead of an
`O(npw³)` diagonalisation.

| | dense | Davidson | |
|---|---|---|---|
| `si-1k.in`, 180 PWs | 17 ms/iteration | 12 ms/iteration | 1.5x |
| `si-1k-ecut40.in`, 1131 PWs | 1239 ms/iteration | 70 ms/iteration | **18x** |

Same total energy to 1e-12 Ry in both cases. The ratio grows with `npw`, which
is the whole point — at the test suite's 180 plane waves an iterative solver is
nearly pointless, and at any real size it is the only option.

The driver now carries the wavefunctions from one SCF iteration to the next and
seeds the solver with them, as QE does; that alone roughly halves the cost of
every iteration after the first.

Two transcription traps, both found by a number being wrong rather than by
reading the Fortran, both recorded in the module docstring:

* a converged root must stop being expanded — normalising its ~1e-14 residual
  turns round-off into a basis vector and the overlap matrix goes singular;
* the subspace must grow by the number of *unconverged* roots, not by the block
  size, or the periodic collapse discards a stubborn root's search direction
  before it has converged.

Both showed the same symptom: silicon's highest band a few meV above the
reference, everything else exact.

### 2. Compilation, not arithmetic

Every JAX operation dispatched outside a `jit` is compiled by XLA separately and
costs ~50 ms the first time — measured, on this machine, for an operation on
1459 elements whose arithmetic is microseconds. Setup was **81 separate
compilations**. So the work was not to make the arithmetic faster but to reduce
the number of compiled units:

* whole functions wrapped in `jit` rather than left as eager op-chains — the
  radial transforms, the projector assembly, the structure factors, both Ewald
  terms, the derived quantities of `Cell`, the G-vector maps;
* host-side constant arithmetic moved to NumPy where nothing differentiates
  through it (Simpson weights of a tabulated mesh);
* Python loops over k-points, atoms, species, symmetry operations and projector
  channels replaced by batched operations or gathers.

Setup 9.96 s → 3.97 s, of which 0.04 s is arithmetic; the rest is still the
per-compilation floor.

### 3. The SCF loop

* `vmap` over k-points in the eigensolver and in `sum_band`, replacing Python
  loops (rule R6 exists precisely so this is available).
* The iteration body is three compiled units — potential, solve, density — with
  the occupation weights decided on the host between them, because the Fermi
  level is a bisection whose bracket is data.
* The Fermi bisection moved into a `lax.fori_loop`. It was 200 host round trips
  **per SCF iteration**, by far the most expensive thing about a metal.
* `eband`, `deband` and the density residual computed together, so the loop
  synchronises with the device once per iteration instead of six times.
* The Ewald real-space sum is one broadcast over `(nat, nat, ntranslations)`
  instead of a Python double loop over atom pairs — which also makes it
  differentiable with respect to the atomic positions, which forces will need.
* `symmetry_maps` looks rotated G-vectors up through the FFT box instead of a
  Python dict: 48 × 1459 dictionary probes became one array index.

### 4. Scheduling the diagonalisation threshold

QE never converges the eigenvalues further than the density warrants
(`electrons.f90`): `ethr` starts at 1e-2, is reset to 1e-2 at the second
iteration, and thereafter follows `min(ethr, 0.1 dr2 / nelec)` with a floor at
1e-13. pypresso was asking for 1e-12 from the first iteration — twelve digits of
eigenvalue against a density still wrong in the second.

Implementing the schedule needed QE's `dr2` (`rho_ddot` in `scf_mod.f90`), which
is the Hartree energy of the density residual — the same expression as
`v_of_rho`'s Hartree energy, applied to `rho_out - rho_in`. That is now
`scf_accuracy`, and since it is also what QE compares against `conv_thr`, the
driver's convergence test uses it too: **`conv_thr` now means the same thing here
as in a `pw.x` input**, where before it was a change in total energy.

The schedules, side by side, and the Davidson steps each one buys:

| SCF iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | total steps |
|---|---|---|---|---|---|---|---|---|---|
| QE `ethr` | 1e-2 | 1.6e-3 | 7.0e-5 | 1.5e-6 | 3.3e-8 | 3.4e-10 | 1.7e-11 | 2.0e-12 | |
| QE steps | 2 | 1 | 2 | 5 | 6 | 4 | 2 | 3 | **25** |
| pypresso `ethr` | 1e-2 | 1.8e-3 | 4.5e-5 | 9.9e-7 | 1.5e-8 | 3.5e-9 | 2.9e-12 | — | |
| pypresso steps | 8 | 4 | 4 | 4 | 4 | 3 | 6 | — | **33** |
| *(before)* | 26 | 16 | 12 | 9 | 6 | 4 | 2 | — | **75** |

75 → 33 steps, against QE's 25. QE's own `ethr` sequence is reproduced closely
enough that the two are clearly running the same schedule on slightly different
densities.

Also implemented is QE's guard against the schedule being *too* clever: after the
first iteration, if the density turns out to be more accurate than the
eigenvalues were (`dr2 < ethr·nelec`), the loose starting threshold was a false
economy and the iteration is diagonalised again at `0.1 dr2 / nelec`.

**What it does not cost.** Silicon's total energy still matches QE to
**1.09e-8 Ry** and the metal to **2.53e-8 Ry**, term by term, and eigenvalues to
5.4e-4 eV — the same numbers as before the schedule. The eigenvalues from the two
solvers now differ by up to 1e-6 Ry rather than 1e-12, because Davidson stops
where it is asked to stop; that is QE's behaviour, it is four orders inside the
comparison tolerance, and the total energy is unmoved because it is variational.

### 5. Doing less work: the irreducible wedge

The largest factor of all, and not an optimisation of code. Two k-points related
by a symmetry of the crystal give the same eigenvalues and contribute the same
thing to the density, so only one of each orbit needs diagonalising — the rest is
recovered by symmetrising the density, which this code already did. On
`K_POINTS automatic` the whole grid was being run.

`kpoint_grid.f90`, transcribed into `system/kpoints.py` and applied in
`build_system` where QE applies it (`setup.f90`, after the symmetry analysis).
Reduction uses the *crystal's* symmetries rather than the lattice point group, so
a structure with fewer symmetries than its lattice is never over-reduced; QE
reaches the same orbits by a longer road (lattice group, then a remap that can
carry representatives off the grid entirely).

Measured on `pw_scf/scf-kauto.in`, 8 k-points against 2:

| | k-points | warm SCF | per iteration |
|---|---|---|---|
| full grid | 8 | 346 ms | 49.4 ms |
| irreducible wedge | 2 | 122 ms | 17.4 ms |

**2.8x**, with a total energy identical to nine decimals — which is the real
check: the reduction is exact only because the density is symmetrised, and the
energy agreeing to 1e-9 is that identity being verified rather than assumed. And
it grows with the grid, which a production calculation makes denser than the test
suite's: 6.4x on 4x4x4, 8.5x on 8x8x8.

Across the 22 automatic-grid cases in the test suite — every Bravais lattice
including the triclinic ones — the reduced count matches QE's exactly, 22 out of
22. The test that used to *skip* the eigenvalue comparison for `scf-kauto.in`
("comparison needs P6's IBZ") now runs.

### 6. Starting from atomic orbitals

The first SCF iteration cost 8 Davidson steps where QE's cost 2, and the reason
was the starting guess: QE begins from the pseudo-atomic orbitals in the
pseudopotential's `PP_PSWFC` section, this code from a random vector damped by
`1/(1+|k+G|^2)`. The atoms already know roughly where their electrons are.

`pseudo/atomic.py` builds them — the projectors' expression with `chi` in place
of `beta`, sharing the same radial transform, the same angular part and the same
assembly — and `solvers/subspace.py` diagonalises the Hamiltonian inside their
span (QE's `rotate_wfc`) before handing the result to Davidson. **The phase is
`i^l`, not the `(-i)^l` of the projectors**; the Fortran comments say why, and
getting it wrong produces a merely worse guess rather than a failure.

| Davidson steps per SCF iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | total |
|---|---|---|---|---|---|---|---|---|
| random start | 8 | 4 | 4 | 4 | 4 | 3 | 6 | **33** |
| atomic start | 3 | 2 | 3 | 3 | 3 | 4 | 3 | **21** |
| QE | 2 | 1 | 2 | 5 | 6 | 4 | 2 | **25** |

Where a species has fewer orbitals than the calculation has bands — aluminium
has four and a smeared run asks for six — the rest are random, as QE tops up.

### 7. The compilation cache

Nothing to do with the loop, and the largest effect on what a user actually
waits for. XLA compilations are now written to `~/.cache/pypresso/jax`
(`PYPRESSO_CACHE_DIR` overrides it; `off` disables it), so the second and every
later run skips them:

| | first run | later runs |
|---|---|---|
| process wall, complete silicon SCF | 9.7 s | **4.3 s** |
| setup | 3.7 s | 1.1 s |

Two defaults have to be overridden or it silently caches nothing: JAX only
persists a kernel that took more than a second to compile, and ours take about
fifty milliseconds each — there are simply a great many of them. Failure to write
the cache is a warning, never an exception.

The test suite is a side beneficiary: 134 s to 57 s.

## What the remaining 3x is made of

Everything below the top item has now been either done or measured and rejected,
so what is left is short. Per Davidson step at 1131 plane waves:

| | |
|---|---|
| `h_psi`, 4 bands | 5–7 ms — **~85% of a step** |
| ...of which the two 3D FFTs | ~75% |
| subspace solve, 16x16 (Cholesky + `eigh`) | 0.85 ms |

So the SCF is **FFT-bound**, which is where a plane-wave code is supposed to be,
and it is the same regime QE runs in. Roughly half the loop is 3D transforms.
That leaves three things, none of them large:

1. **Per-dispatch overhead**, still visible on small arrays: it is why `metal.in`
   with its ten k-points sits at 5.2x while the single-k cases sit at 3.2x.
2. **FFT throughput.** XLA's transform is already 2x faster than pocketfft
   single-threaded, and QE's sticks decomposition was measured at a 1.13x
   ceiling here, so this is close to a floor rather than an opportunity.
3. **Form factors computed rather than interpolated** — a deliberate trade for
   differentiability (`PLAN.md` D1/D2), paid in setup and now largely hidden by
   the compilation cache.

## Threads: the default was the worst choice

XLA sizes its CPU thread pool from the process's **affinity mask**, and its
default — every core on the machine — was costing a factor of nearly two. A
plane-wave SCF is a long chain of FFTs and small matrix products with little
parallelism inside any single operation, so past a handful of threads the pool
spends more time synchronising than computing.

Per SCF iteration, same code, only the number of visible cores changing:

| cores | `si-1k` (180 PWs) | `si-1k-ecut40` (1131 PWs) | `metal.in` | `ecutwfc=80` (3215 PWs) |
|---|---|---|---|---|
| 1 | 7.4 ms | 37.6 ms | 67.4 ms | 123 ms |
| 4 | **6.1 ms** | **31.6 ms** | **66.8 ms** | **114 ms** |
| 8 | 9.9 ms | 54.4 ms | 119.5 ms | — |
| 14 (default) | 10.0 ms | 54.6 ms | 120.3 ms | 152 ms |

Four is best everywhere measured, including a 3215-plane-wave cell, so this is
not an artefact of small cases. `pypresso` now narrows the affinity mask to four
CPUs on import — **1.7x faster out of the box** — with `PYPRESSO_THREADS` to
change or disable it, and it only ever *narrows*, so an outer `taskset` or a
scheduler's allocation is respected.

Nothing else moves it: `OMP_NUM_THREADS` changes the time by a few percent
(64 → 56 → 63 ms for 1, 4, 14) because it is not what sizes the pool. That is
also why the affinity mask, rather than any environment variable, is what the
benchmark harness sets.

**The conclusion this points at is the more important one.** XLA's intra-op
threading gives this workload essentially nothing — 1 core to 4 is a 15% gain on
a 14-core machine. The parallelism that is actually available here is over
k-points, which is exactly why the k index leads every wavefunction-shaped array
(`PLAN.md` §5). `metal.in` has ten independent k-points and runs them through one
thread pool; sharding them across CPU devices is a factor the thread pool cannot
give.

## Optimisation backlog

Ordered by expected gain per unit of effort, and by measurement rather than
instinct. None of these may change a validated number.

1. **`jax.sharding` over the k-axis**, and GPU. Now measured to be the *only*
   parallelism worth having on CPU: the thread pool gives 15% between one core
   and four and loses badly beyond that, while `metal.in`'s ten k-points are
   independent and are currently run through a single pool. This is where the
   remaining structural factor is, and the k-axis already leads every
   wavefunction-shaped array so that it can be taken.
2. **Fold `dr2` into the iteration's other reductions.** It costs a transform and
   a dispatch of its own (~3% of an iteration) for a quantity the loop already
   computes a residual for. Mixing in G space would save another transform.
3. **Shell-based radial evaluation** for quantities depending only on `|G|` (~100
   shells vs 1459 G-vectors for Si). Note this is *not* strain-safe: shells split
   under strain, so it must stay off the stress path.

### Measured and rejected

* **A faster FFT library.** XLA's CPU FFT is already **2x faster than SciPy's
  pocketfft** single-threaded (0.92 ms against 2.36 ms for a `(4, 30^3)`
  transform). There is no library-level win available; we are using a good one.

* **Sticks/pencil FFTs**, QE's decomposition, where the 1D transforms are done
  only along columns the wavefunction sphere actually occupies. The sphere does
  touch only 16% of the z-columns and 50% of the x-planes, which looks like a
  large saving — but against the *fused* 3D transform the ceiling is **1.13x**,
  because the final pass runs on dense data and is 58% of the cost on its own.
  Splitting the transform by hand also gives up XLA's fusion: three separate
  passes cost 4.8 ms where the fused `fftn` costs 3.3 ms.

* **Applying `H` only to unconverged bands**, as `cegterg` does. This was second
  on the backlog until it was measured. Counting the unconverged roots at every
  Davidson step of a whole run and pricing them with the measured cost of
  `h_psi` as a function of block size gives a ceiling of **6% (180 PWs) to 8%
  (1131 PWs)** of `h_psi` time — about 3% of an SCF iteration.

  The reason it is so small is that the two changes above it took its value
  away. Davidson calls now last two to four steps rather than eight to
  twenty-six, and the first step of every call has *all* roots unconverged by
  construction, since the convergence test compares against the previous step.
  Compaction pays when a call runs long; scheduling `ethr` and starting from
  atomic orbitals are precisely what stopped calls running long.

  It is also more expensive to implement here than in Fortran: shapes inside
  `jit` are static, so masking converged bands costs exactly what computing them
  costs, and realising the saving needs `lax.switch` over precompiled block
  sizes. That cannot sit under the `vmap` over k-points either — a batched
  switch index makes every branch execute — so it would need the per-k solver
  rewritten as one joint loop over all k, where the block size is the *maximum*
  over k-points and the saving shrinks again. A substantial rewrite of a
  freshly validated solver, for 3%.

* **Folding the `1/N` FFT normalisations.** `g_to_r` multiplies by `N` and
  `r_to_g` divides by it, and inside `h_psi` the two cancel exactly. Removing
  them helps the 180-plane-wave case by 1.3x and the 1131-plane-wave case not at
  all: XLA already fuses the scaling into the neighbouring elementwise pass at
  any size where it would matter.

## History

| Date | Change | Effect |
|---|---|---|
| 2026-08-18 | Batched symmetrisation: one gather over `(nsym, ngm)` instead of a Python loop | `symmetrize` 0.203 s → 0.006 s |
| 2026-08-18 | Single-core QE comparison established as the primary metric (`tools/compare_qe.py`, `benchmarks/`) | baseline: 53x QE per iteration |
| 2026-08-18 | `vmap` over k-points; iteration body compiled in three units; Fermi bisection moved on-device; one host sync per iteration | 0.131 → 0.036 s/iteration |
| 2026-08-18 | Setup compilation count cut: jitted radial transforms, projector assembly, structure factors, Ewald, `Cell` quantities; vectorised `symmetry_maps` and the Ewald real-space sum | setup 9.96 → 4.2 s, 81 → 69 compilations |
| 2026-08-18 | Dense Hamiltonian built from its matrix elements, `V(G-G')`, instead of one FFT per plane wave | matrix build 49 → 6 ms |
| 2026-08-18 | Block Davidson eigensolver (P4), solver registry, wavefunctions carried between SCF iterations | 0.036 → 0.012 s/iteration; 13x at 1131 PWs |
| 2026-08-18 | QE's adaptive `ethr` schedule, `scf_accuracy` (`dr2`), and `conv_thr` on the same quantity QE uses | 75 → 33 Davidson steps; 0.067 → 0.044 s/iteration at 1131 PWs |
| 2026-08-18 | k-point reduction to the irreducible wedge (`kpoint_grid.f90`) | 2.8x on `scf-kauto.in`; matches QE's count on 22/22 lattices |
| 2026-08-18 | Pseudo-atomic starting wavefunctions and Rayleigh-Ritz (`wfcinit`) | 33 → 21 Davidson steps; first iteration 8 → 3 |
| 2026-08-18 | Persistent XLA compilation cache, on by default | process wall 9.7 → 4.3 s; test suite 134 → 57 s |
| 2026-08-19 | Cap XLA's CPU thread pool at four cores by affinity; fix the harness, which had claimed one core while using 1.8 | 1.7x out of the box; 14 cores was the worst setting measured |
