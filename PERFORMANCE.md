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

The benchmark inputs are all a single k-point (`benchmarks/`):

| Input | | Why |
|---|---|---|
| `si-1k.in` | Si, 2 atoms, `ecutwfc = 12`, 180 plane waves | the test suite's silicon, one k-point |
| `si-1k-ecut40.in` | the same cell at `ecutwfc = 40`, 1131 plane waves | a production cutoff, where scaling starts to show |
| `si8-1k*.in`, `si16-1k*.in` | the same crystal in 8- and 16-atom cells | where the cost is physics rather than fixed overhead |
| `si2-us-1k.in`, `si8-us-1k.in` | ultrasoft, `ecutwfc = 20`, dual 8 | a different *shape* of calculation, not just a bigger one |
| `si2-paw-1k.in`, `si8-paw-1k.in` | PAW, same cutoffs | ultrasoft plus the one-centre radial work |
| `si8-pbe-1k.in`, `si8-paw-pbe-1k.in` | the same cells under PBE | what a gradient correction costs, on the grid and on a PAW sphere |

One k-point on purpose: both codes parallelise over k, so a multi-k comparison
measures batching rather than the cost of the physics.

## Where it stands

Single core, this machine, re-measured 2026-08-19. `conv_thr = 1e-10` where the
input allows it, so both codes converge to the same place and the energy
agreement doubles as a correctness check on every optimisation.

Re-measured 2026-08-20 after the band axis became a dial ("The band a
transform holds", below). The `before` column is the 2026-08-19 measurement it
replaces, on the same machine and the same inputs.

| | QE 7.5 | pypresso | ratio | before |
|---|---|---|---|---|
| **`si-1k`** — 2 atoms, 180 PWs, 1 k | 0.003 s | 0.008 s | **3.1x** | 3.0x |
| **`si-1k-ecut40`** — 2 atoms, 1131 PWs | 0.013 s | 0.032 s | **2.6x** | 3.2x |
| **`si8-1k`** — 8 atoms, 738 PWs | 0.020 s | 0.044 s | **2.2x** | 2.1x |
| **`si8-1k-ecut30`** — 8 atoms, 2950 PWs | 0.071 s | 0.158 s | **2.2x** | 3.9x |
| **`si16-1k`** — 16 atoms, 1476 PWs | 0.081 s | 0.185 s | **2.3x** | 3.0x |
| **`si16-1k-ecut30`** — 16 atoms, 5900 PWs | 0.283 s | 0.722 s | **2.5x** | 4.2x |
| **`pw_scf/scf-kauto`** — 2 k, reduced from 8 | 0.007 s | 0.020 s | **2.7x** | 4.6x |
| **`pw_metal/metal`** — Al, 10 k | 0.027 s | 0.133 s | **5.0x** | 4.0x |

Read the first and last rows together, because they say the same thing from
opposite ends. **The gain is in the box, not in the code paths**: the two cells
whose real-space grid is small — 180 plane waves, and aluminium's six bands on a
15 Ry grid — did not move at all, and the two whose grid is large nearly halved.
Anything else would have been evidence that the change was not what its
explanation says it is.

A pair added 2026-08-21 with DFT+U (P20), listed apart because they differ only by
the `HUBBARD` card and the ratio *between them* is the measurement — see "What
DFT+U costs" below:

| | QE 7.5 | pypresso | ratio |
|---|---|---|---|
| **`ni-noldau-1k`** — Ni, ultrasoft, LSDA, 1 k | 0.070 s | 0.134 s | **1.9x** |
| **`ni-ldau-1k`** — the same with `U = 3 eV` | 0.069 s | 0.155 s | **2.3x** |

`metal` reads worse than it did and it is not this change: at the commit before
it the same input measures 0.128–0.134 s against 0.133–0.135 s after, so the
25% is older than this entry and belongs to the spin-orbit or the forces work.
Ten k-points of six bands each is the regime where nothing is large enough to
amortise a dispatch, and it is the case backlog item 1 exists for.

**A caution about this table that cost an hour to relearn.** `si8-paw-pbe-1k`
was measured at 3.5x here and at 1.7x twenty minutes later with neither code
changed — QE's own time for the same binary swung between 0.51 s and 1.08 s.
Every ratio in this file is a quotient of two measurements each carrying ±20%,
so a *pair* of runs taken together is the only comparison that means anything,
and a ratio that moves without a code change is the machine, not the code.

and, for ultrasoft and PAW (2026-08-19, `ecutwfc = 20`, `ecutrho = 160`):

| | QE 7.5 | pypresso | ratio | before |
|---|---|---|---|---|
| **`si2-us-1k`** — 2 atoms, 395 PWs, 9185 G | 0.021 s | 0.049 s | **2.3x** | 2.2x |
| **`si2-paw-1k`** — the same, PAW | 0.034 s | 0.083 s | **2.4x** | 1.8x |
| **`si8-us-1k`** — 8 atoms, 1607 PWs, 36257 G | 0.121 s | 0.327 s | **2.7x** | 2.9x |
| **`si8-paw-1k`** — the same, PAW | 0.173 s | 0.484 s | **2.8x** | 2.8x |

None of these four moved by more than the spread. That is what it looks like
when the box is small: at `dual = 8` the *dense* grid is large but the
wavefunctions live on the smooth one, and it is the smooth box a band is
transformed in. The augmentation charge, `newd` and the one-centre terms are
dense-grid work that no band loop touches.

and, for the gradient-corrected functionals (2026-08-19, PBE). All four rows were
measured in one session, the local ones re-run alongside the PBE ones, so each
pair differs only in the functional — which is the only way the "what does a
gradient correction cost" question has an answer:

| | QE 7.5 | pypresso | ratio |
|---|---|---|---|
| **`si8-1k`** — 8 atoms, LDA | 0.030 s | 0.074 s | **2.5x** |
| **`si8-pbe-1k`** — the same cell, PBE | 0.025 s | 0.077 s | **3.1x** |
| **`si8-paw-1k`** — 8 atoms, PAW, LDA | 0.170 s | 0.479 s | **2.8x** |
| **`si8-paw-pbe-1k`** — the same cell, PBE | 0.529 s | 0.841 s | **1.6x** |

Re-measured 2026-08-20: `si8-pbe-1k` is 0.025 s against 0.049 s, **2.0x**, and
`si8-paw-pbe-1k` is **1.7x** — from an interleaved pair, because this is the one
case in the file whose QE side is not reproducible run to run (see the caution
above the first table). The conclusions below are unchanged; the gradient
correction is still nearly free on the grid and still cheaper here than in the
Fortran on a PAW sphere.

(`si8-1k` reads 0.064 s in the table above and 0.074 s here, on the same machine
and the same input. That ~15% between sessions is the spread this measurement
has, and it is exactly why a pair of runs has to be taken together before any
difference between them is believed.)

**On the plane-wave grid a gradient correction is nearly free here, and on a PAW
sphere it is nearly free *relative to QE*.** The two rows say different things and
both are worth reading.

The first pair: PBE costs pypresso 4% per iteration (0.074 → 0.077 s) for four
extra FFTs on the dense grid and a longer pointwise expression. Four transforms
of a 24³ box is not nothing, but it is a fixed handful of large, dense
operations — exactly the shape XLA is good at — where the iteration's real cost
is still the eigensolver's many small dispatches. QE's side of that pair is
within its own run-to-run spread, so the ratio moving from 2.5x to 3.1x is
mostly QE getting slightly faster, not this code getting slower.

The second pair is the interesting one. PAW's one-centre terms are radial work:
per atom, per iteration, a spherical quadrature that a GGA grows from 28
directions to 45, a radial derivative on a 1141-point mesh for each direction,
and a spherical divergence. QE pays 3.1x for that (0.170 → 0.529 s per
iteration); this code pays 1.8x (0.479 → 0.841 s), and the ratio against QE
*improves* from 2.8x to 1.6x. The reason is the one this file keeps recording
from the other direction: the radial work is loops in Fortran and batched array
operations here, so where QE's cost scales with how many radial points and
directions there are, this code's scales with how many *compiled units* — and
the one-centre gradient adds work to existing kernels rather than adding
kernels. It is the first place in this project where the JAX formulation wins on
its own terms rather than catching up.

Total energies against QE: **9.7e-10 Ry** on the sixteen-atom cell at 30 Ry,
5.8e-9 at 12 Ry, 3.5e-9 and 1.5e-9 on the eight-atom cell, 3.8e-9 for two atoms
at 40 Ry, 3.0e-9 for two atoms at 12 Ry, 5.6e-9 for the metal.

`scf-kauto` is the one case where the two codes differ by more than that
(1.0e-6 Ry), and it is not a disagreement about the physics: that input asks for
`conv_thr = 1e-6`, so QE stops there while this comparison runs pypresso to
1e-10. The regression suite compares it against a reference regenerated at 1e-10,
where the two agree to 2e-9.

The supercells are independent checks as well as bigger ones. All of them are the
same crystal in different cells — the primitive fcc cell has two atoms, the
conventional cubic cell eight, and doubling that along z gives sixteen — so the
energy per atom is known in advance. At 30 Ry the sixteen-atom cell comes out at
−126.72076070 Ry against exactly twice the eight-atom cell's −63.36038036, which
is 2e-8 Ry apart on a 126 Ry total.

Ultrasoft and PAW totals against QE: **2.5e-9 Ry** and **3.8e-10 Ry** on the
two-atom cell, **5.0e-10** and **2.3e-9** on the eight-atom one — the same
accuracy the norm-conserving path reaches, which is the point of quoting them
next to the timings rather than only in `PLAN.md`.

Two things in that table are worth reading carefully rather than skimming.

**The ultrasoft ratios are no worse than the norm-conserving ones, and on eight
atoms slightly better.** That is not because the extra work is free — the
augmentation charge is built on a 45³ grid every iteration, the nonlocal
coefficients are rebuilt from the potential, and the wavefunctions and the
density now live on *different* grids with an interpolation between them. It is
because all of that is dense array work on the larger of the two grids, which is
exactly what XLA does well, while the part that was already the bottleneck — the
eigensolver's many small dispatches — grows only with the number of plane waves,
and a dual of 8 means there are fewer of those per G-vector than before. QE pays
the same new costs in Fortran.

**PAW's one-centre terms are radial work, and batching them was worth 35%.**
Per atom, per iteration, they are nine radial Poisson solves on a 1141-point
mesh for each of the all-electron and pseudo densities, plus an
exchange-correlation evaluation on a 28-direction angular grid. On the Fortran
side that is a handful of tridiagonal solves and some loops, close to free;
here every one of them is a dispatch, and the arrays are far too small for the
arithmetic to matter — the same lesson as the rest of this file, that on these
sizes the cost is the number of compiled units.

Two axes are batched, and the order they were found in is the useful part.
Atoms were batched from the start (`vmap` over `becsum`). The **multipoles**
were not: the first version looped over all nine `lm` in Python because the
solver takes `l` as a static argument. Grouping them by `l` instead -- the
`2l+1` components of one multipole solve the *same* equation -- turns nine
dispatches into three and took the eight-atom PAW cell from **3.0x to 2.8x**
against QE, with the underlying per-iteration time dropping from 0.507 s to
0.477 s. What is still not batched is the all-electron and pseudo pass, which
differ only in which tensor and core charge they use and could be one `vmap` of
width two.

Best of three or five runs on each side, and worth taking as ±20%: QE prints its timings
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

**Superseded 2026-08-20; kept because point 2 was wrong in an instructive way.**
This section concluded that FFT *throughput* was near a floor, and it was right
about that and wrong about what followed from it. The transform was not slow per
plane; there were simply far more bytes moving through it than there needed to
be, and no per-plane measurement could see that (see "The band a transform holds
is the working set"). The lesson is that "we are at the library's floor" is a
statement about one operation, never about the program — the question left
unasked was how much memory the operation was being handed. The ratios below are
also from before that change; the current ones are in the first table.

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

## What the eight-atom cell showed

Two-atom cells are small enough that fixed overheads dominate. Going to eight
atoms -- four times the bands, plane waves and volume -- made the ratio *worse*
(3.3x to 4.5x), and the two reasons were both real inefficiencies invisible at
the smaller size.

**The projected matrices were being rebuilt every step.** Davidson's subspace
grows by a block per step, so all but the newest rows of `<psi|H|psi>` and
`<psi|psi>` are unchanged -- `cegterg` computes only the new ones and keeps the
rest. Recomputing costs `O(nvecx^2 npw)` against `O(nvecx nbnd npw)`, a factor of
four at the default subspace size. On the eight-atom cell that was 7.7 ms per
step against 2.2; on a two-atom cell it is half a millisecond either way.

**Every Davidson call ended by applying H to a block of zeros.** Convergence was
tested *after* the expansion, so the last step of every call preconditioned a set
of residuals that had all just converged -- and therefore were all zero -- and
applied `H` to them anyway. Reordering the loop so it expands, then
diagonalises, then tests (which is `cegterg`'s order) removed one full `h_psi`
per call: 26 expansions became 18 for the two-atom run, 23 became 16 for the
eight-atom one.

Together: 4.5x to **3.8x** on the eight-atom cell at 30 Ry, and every other
benchmark improved too. The eight-atom cell remains the worst ratio of the six,
which is the honest place to look next.

## At sixteen atoms

The largest case here — 5900 plane waves, 32 bands, a 36x36x72 grid — is where
the cost is genuinely the physics rather than fixed overheads. Per SCF iteration
(1055 ms):

| | | |
|---|---|---|
| diagonalise | 886 ms | **84%** |
| density | 133 ms | 13% |
| everything else | 36 ms | 3% |

and inside one Davidson step (236 ms, so about 3.75 steps per iteration):

| | | |
|---|---|---|
| `h_psi`, 32 bands | 167 ms | **71%** |
| projection block update | 32 ms | 14% |
| two rotations, 128 -> 32 vectors | 32 ms | 14% |
| generalised eigh, 128x128 | 4.5 ms | 2% |

The subspace linear algebra, invisible at two atoms and 5% at eight, is 27% of a
step here. Both terms are `O(nvecx nbnd npw)` and both are what `cegterg`'s ZGEMMs
do too, so there is no obvious waste left in them — but they are the reason the
ratio drifts up with size rather than down.

## QE's FFT layout

A wavefunction's sphere touches under a fifth of the box's `z` columns, so QE
transforms only those — `cft_1z` over the sticks, then `cft_2xy` over the planes.
Reproducing it took three measurements, the first two of which said it was not
worth doing, and both were wrong:

| | | |
|---|---|---|
| three separate 1D passes | 1.13x *ceiling* | wrong: gives up XLA's fused `fftn` |
| 1D on sticks + 2D over the C-ordered box | 0.48x | wrong: 2D over two *strided* axes |
| 1D on sticks + 2D over an `xy`-contiguous box | **1.13x / 1.02x** | QE's actual layout |

The whole difference is the layout. QE's arrays are Fortran-ordered with `x`
fastest, so an `xy` plane is contiguous; a C-ordered `(n1, n2, n3)` box has `z`
fastest, and the same 2D transform then runs over the two strided axes, where on
a 36x36x72 box it costs more alone (107 ms) than the entire fused 3D transform
(68 ms). Holding the field as `(n3, n1, n2)` instead — and storing the local
potential to match — recovers it.

`basis/sticks.py` builds the layout, `basis/fft.py` has the pair of transforms,
and `h_psi` uses them; the dense-grid quantities still transform the whole box,
which for them is the right thing. Measured end to end on the local potential
term, 1.13x at eight atoms and 1.02x at sixteen, and about 4% on a whole SCF
iteration.

It is a small win for a fair amount of machinery. It is in because the layout is
also the precondition for anything further here — and because the standing rule
is now to mirror QE where performance matters, which this is the reason for.

## What the spin axis cost the unpolarized path (P9)

LSDA gave the density, the potential, `becsum`, `D_ij`, the eigenvalues and the
wavefunctions a leading channel axis, and an unpolarized run now goes through
that axis with length one: `sum_band` and the symmetrisation are `vmap`ped over
it, `v_of_rho` transforms each channel, and the Hamiltonian is built in a Python
loop over `range(nspin)`. The obvious worry is that a length-one batch axis is
not free.

Measured, 2026-08-19, single core, against the same QE build:

| | QE 7.5 | pypresso | ratio | pypresso before P9 |
|---|---|---|---|---|
| **`si-1k`** — 2 atoms, 180 PWs | 0.003 s | 0.010 s | **4.1x** | 0.011 s |
| **`si8-1k`** — 8 atoms, 738 PWs | 0.027 s | 0.074 s | **2.8x** | 0.064 / 0.074 s |

**Nothing to pay.** `si-1k` came out a millisecond *faster* and `si8-1k` landed
on the upper of the two numbers the same input has produced in earlier sessions
(0.064 s and 0.074 s, on the same machine, which is the ~15% spread this
measurement has). The ratios moved more than the times did, and on `si-1k` that
is arithmetic on QE's side: 0.004 s became 0.003 s, one significant figure at
three milliseconds, and 0.011/0.004 = 3.0 while 0.010/0.003 = 4.1 without
anything about either code having changed by more than a millisecond.

Why a length-one `vmap` costs nothing here: XLA sees the batch dimension at trace
time, and a batch of one collapses in the same pass that would otherwise fuse the
elementwise work. The Python loop over channels is host-side and runs once per
iteration.

**That paragraph is true of the spin axis and false in general, and the
difference cost 37% of a Davidson solve for a day.** What collapses for free is
a width-one batch over *elementwise* work, which is all the spin axis carries
here — `sum_band`, `v_of_rho`, the symmetrisation. A width-one batch over a
matrix product, a `eigh` or a `dynamic_update_slice` does not collapse: XLA
lowers it to the *batched* kernel, and the batched kernel at width one is much
slower than the plain one. The Davidson body is nothing but those. See "A batch
of one is not a batch" below. What *would* have cost something is a `lax.scan` or a dynamic branch
over the spin index, and neither is there -- `nspin` is static, which is the
whole reason it is an `eqx.field(static=True)` on `System`.

The polarized path costs what it should: two Hamiltonians and two
diagonalisations, so roughly twice an unpolarized iteration on the same cell,
plus one extra reduction for `dr2`'s magnetization term.

## The k-axis: memory, and what batching it is actually worth (P14)

QE holds **one k-point at a time**. `c_bands.f90`'s `k_loop` calls `diag_bands`
on a single `ik`, `sum_band.f90` accumulates that k-point's bands into
`rho%of_r` inside the same loop, and the rest of `evc` sits in a buffer that is
RAM or disk according to `io_level`. Its parallelism over k is MPI pools. This
code did the opposite -- one `vmap` over the whole k axis, everywhere -- which
is what a GPU wants and what made the first Python-loop version slow.

That deviation was never measured until a converged bismuthene run (19
irreducible k-points, two-component spinors, 35 Ry) died at **12.7 GB**. So the
k axis is now a dial, `pypresso/batching.py`, default 1 -- QE's loop -- with
`k_batch=None` for the old behaviour and anything between as a chunk. It is a
`lax.map`/`lax.scan`, not a Python loop, so the body is compiled once whatever
the chunk count; the answer is identical to round-off (1.8e-15 Ry) because only
the order the per-k contributions are summed in changes.

Measured, 2026-08-20, single core except where noted:

| case | k-points | `k_batch=1` (QE's loop) | one `vmap` over all k |
|---|---|---|---|
| `pw_scf/scf.in` — 2 atoms, 180 PWs | 2 | 19.6 ms/it, 0.41 GB | 19.3 ms/it, 0.41 GB |
| `pw_metal/metal.in` — Al, smeared | 10 | 78.4 ms/it, 0.51 GB | 72.1 ms/it, 0.47 GB |
| `bismuthene-soc-small` — spinors, ultrasoft, PBE | 7 | **14.58 s/it, 4.6 GB** | 17.94 s/it, 5.0 GB |
| `bismuthene-nosoc` — converged, 35 Ry | 19 | **22.5 s/it, 3.16 GB** | 44.9 s/it, 4.91 GB |

**The loop is not a tax on anything that matters -- it is a win.** It costs 9%
on aluminium, whose per-k work is a handful of plane waves and six bands, the
regime where the batch is hiding dispatch overhead, and nothing at all on two
k-points. On the two cases whose per-k work is a real calculation it is *faster*:
23% on the small bismuthene cell, and **exactly 2x** on the converged one, at
two thirds the memory and with the total energy identical to every digit
(-310.10972823072876 Ry both ways). Each k-point's Davidson subspace and its
band-by-band real-space fields already saturate XLA's threads on their own;
stacking 19 of them buys no parallelism and costs cache.

That is the same lesson as the FFT layout above, from the other direction: the
Fortran's structure was not a limitation of Fortran. What batching still buys is
the GPU, where the launch overhead the aluminium case shows is an order of
magnitude worse and thousands of cores want feeding -- which is exactly why this
is a dial with both ends kept working rather than a rewrite in either direction.

## What a force costs (P15)

Forces are ``jax.grad`` of one energy expression evaluated at the converged
state (`pypresso/forces/energy.py`), and QE's six hand-derived terms are
implemented beside them as a cross-check (`analytic.py`). Both are timed against
the SCF that has to happen first, since that is what decides whether a
relaxation is affordable, and against QE's own `forces` clock.

Both codes pinned to one core by the affinity mask, as `tools/compare_qe.py`
does it. 2026-08-20:

| case | | QE 7.5 | autodiff | analytic |
|---|---|---|---|---|
| `si8-us` — 8 atoms, 1 k, ultrasoft | force | 80 ms | 308 ms | **102 ms** |
| | per SCF iteration | 128 ms | 596 ms | — |
| | force, in its own iterations | 0.63 | 0.52 | **0.17** |
| `si2-us-force` — 2 atoms, 18 k, ultrasoft | force | 20 ms | 67 ms | **18 ms** |
| | per SCF iteration | 101 ms | 511 ms | — |
| | force, in its own iterations | 0.20 | 0.13 | **0.04** |

**A force costs a fraction of an SCF iteration on either path**, so it is free
next to the ten or twenty iterations that produced the state, and a relaxation
costs what its ionic steps cost. Against QE: the transcribed force is at parity
(1.3x and 0.9x — faster than the Fortran on the smaller case), and the
differentiated one is 3.4-3.8x slower, which is the same factor this code pays
on the SCF itself (4.7x and 5.1x per iteration here, at the high end of the 2-4x
recorded above because both cases are ultrasoft at `dual = 8`). Forces add no
penalty of their own on either path.

Peak RSS for `si8-us` is 1.40 GB including the SCF. The gradient's own working
set is one reverse-mode tape over the energy — a constant times the
`(nk, nbnd, npwx)` wavefunctions and the real-space fields of one k-point at a
time — so the batching dial applies to it unchanged, since it runs through the
same `map_k`.

### What made the transcription 33x faster

It started at 3392 ms on `si8-us`, eleven times slower than the differentiated
force and forty times slower than QE. Three changes, in the order they were
worth:

1. **`force_corr` was integrating the atomic charge once per atom** — 2902 ms of
   the 3392, 91% of the whole force in one term. The radial transform
   `rho_atomic(|G|)` depends on the *species*, not on the atom, and this cell
   has eight atoms of one species, so seven eighths of that was the same
   integral again. It is now a per-species table built once at setup next to
   `vloc` and `rho_core` (`species_atomic_charge`), which is what QE does with
   `init_tab_rhoat`. Cost: `(ntyp, ngm)` floats — 0.3 MB here. **2902 ms -> 5 ms.**
2. **The six terms are assembled inside one compiled function** rather than
   evaluated one at a time. Each is a handful of contractions over the G-vector
   sphere; run eagerly they cost more in dispatch and in intermediates than in
   arithmetic. The compiled function is cached on the calculation and does not
   depend on the geometry, so a relaxation compiles it once.
3. **The density and the total potential come from the converged state** instead
   of being rebuilt (105 ms and 16 ms of transforms). That is not a shortcut but
   the faithful reading: `forces.f90` consumes `rho%of_r` and `v%of_r` as the
   SCF left them. The differentiated force cannot do this and does not — it
   needs the density as a *function* of the positions, which is the entire point
   of it, and that difference is most of why it stays the more expensive of the
   two.

The general lesson is the one this file keeps recording from other directions: a
factor of thirty was not in the algorithm, the arithmetic or the language. It
was one radial integration repeated per atom instead of per species, in a term
whose *value* is ~1e-7 Ry/bohr and which nobody thought to time.

## The band a transform holds is the working set

**QE's own clock report is what found this, and it is the reason to read it
rather than to profile only this side.** `pw.x` prints the call count next to
every timer, so `h_psi`, `vloc_psi` and `fftw` can be divided out and the two
codes compared per unit of work instead of per SCF iteration — which is the
comparison that matters, since the two do not take the same number of iterations
or the same number of Davidson steps.

| | QE | pypresso, before | |
|---|---|---|---|
| `h_psi`, `si8-1k-ecut30`, 16 bands | 14.5 ms | 24.1 ms | 1.7x |
| `h_psi`, `si16-1k-ecut30`, 32 bands | 51.3 ms | 169.6 ms | **3.3x** |
| one wave FFT, 8 atoms | 0.43 ms | 0.71 ms | 1.7x |
| one wave FFT, 16 atoms | 0.72 ms | 2.65 ms | **3.7x** |

**A gap that grows with the box is not an arithmetic gap.** Per plane, XLA's
36x36 transform is within 1.4x of FFTW and there is no factor of three anywhere
in it; what changed between the two rows is only how much memory the operation
touches at once. `vloc_psi_k` walks its bands one at a time — `DO ibnd = 1, m`
around a single `invfft` — and this code transformed the whole block in one
call. One band's real-space box on the sixteen-atom cell is 1.5 MB and
thirty-two of them are 48 MB, so the batched form streamed the array from
memory on every pass where the looped one stays in cache.

Walking them instead, measured on `h_psi`'s local term:

| case | all bands | one at a time | |
|---|---|---|---|
| `si16-1k-ecut30` | 153.9 ms | 62.0 ms | **2.48x** |
| `si16-1k` | 23.5 ms | 13.9 ms | 1.69x |
| `si8-1k-ecut30` | 23.1 ms | 13.9 ms | 1.66x |
| `si8-1k` | 4.2 ms | 3.7 ms | 1.14x |
| `si-1k` | 0.31 ms | 0.34 ms | 0.91x |

The one case that loses is the 180-plane-wave cell, where the whole box is
70 kB and the only thing a loop can add is dispatch. That monotone ordering by
box size is the evidence that the explanation is the right one — no arithmetic
change produces it. `sum_band` gets the same treatment for the same reason
(`calc.density` 119 → 47 ms on the sixteen-atom cell), and the band axis is now
a dial beside the k axis (`pypresso/batching.py`, `PYPRESSO_BAND_BATCH`),
defaulting to QE's loop, with the batched end kept working for the GPU — which
wants the batch that a cache does not.

Intermediate profile that made the search finite, `si8-1k-ecut30` per SCF
iteration before the change: diagonalise 69%, density 9%, potential 8%, the
first wavefunctions 8%; and inside a Davidson step, `h_psi` 65% and the subspace
algebra 35%.

### A batch of one is not a batch

Found on the way, and independent of the above. `k_batch = 1` is the default —
QE's `k_loop` — and it was implemented as a batch axis of width one:
`jax.vmap` when `nk = 1`, and `lax.map(..., batch_size=1)` otherwise, which
JAX *defines* as a scan over `vmap(fn)` of one-element chunks. Four ways of
running one Davidson solve on `si8-1k-ecut30`:

| | |
|---|---|
| direct call, no map | **110 ms** |
| `jax.vmap` over `nk = 1` | 150 ms |
| `lax.map`, plain scan | **108 ms** |
| `lax.map(batch_size=1)` | 151 ms |

37%, and it was being paid on every k-point of every run. Not on the FFTs —
`h_psi` measures 22.1 ms batched and 22.2 ms not — but on the subspace algebra,
which is a third of a Davidson step and every part of it a small dense
operation that XLA lowers to a batched kernel the moment there is an axis to
batch. `batch = 1` now means *no batch axis*: a direct call at one k-point, a
plain `lax.map` beyond that.

### One evaluation of the functional, not two

`v_xc` is the derivative of `rho e_xc` and `e_xc` is the same expression's
forward value, and they were being computed by two separate passes over the
grid — two cube roots, two logs and two square roots per point. `value_and_grad`
with the energy density as an auxiliary output returns both for the cost of the
derivative. It is exact rather than an approximation because the local
functional is a function of `|rho|` alone, so the absolute value the potential
needs is also where the energy density is wanted. `exchange_correlation`
7.3 → 3.5 ms, `v_of_rho` 8.0 → 5.5 ms.

### What was tried and rejected

* **`diago_david_ndim` 4 → 3.** Worth 12% of a whole SCF on the eight-atom cell
  and 7% on the sixteen-atom one, and not landed, for two independent reasons.
  It changed a validated number: a band-structure run has no SCF around it to
  re-seed a root left short, and on the bismuthene spin-orbit path the Kramers
  splitting — degenerate by symmetry, so a pure measure of solver error — went
  from below 1e-6 eV to 5.9e-6. And by the time that was understood the speed
  was gone too, because the 12% had been measured before the band loop existed
  and the two are the same saving: it was in the cache, not in the flop count.
  The reasoning is kept in `solvers/davidson.py`.
* **Reshaping `(lead, n3, n1, n2)` to `(lead·n3, n1, n2)` around the 2D pass.**
  Looked like 1.1–1.4x in two measurements and is 0.93–1.12x across nine grid
  sizes, i.e. nothing. Both early readings were closures over constants that
  XLA folded.
* **Shrinking the `h_psi` block to `notcnv`**, as `cegterg` does. The blocks are
  genuinely full: instrumenting the run shows 16 nonzero rows of 16 on all but
  two of nineteen calls, because the SCF's `ethr` schedule means roots converge
  together rather than one at a time.

## What magnetism costs when it becomes a vector (P17-P19)

Three features landed on the noncollinear path: a density with four components that
actually carries a magnetization (P17), fields and constrained moments (P18), and spin
spirals (P19). Each one was measured against the thing it extends.

**A magnetic noncollinear run against QE, single core.** bcc iron, one k-point, ultrasoft,
LDA, `nosym` — `benchmarks/fe-mag-1k.in`, which is single-k and symmetry-free for the same
reason every other input in that directory is: both codes parallelise over k, and a
symmetry-reduced list is not the same list in both codes unless the *magnetic* group is
reproduced exactly, which is a correctness question and not a timing one.

| | QE 7.5 | pypresso | ratio |
|---|---|---|---|
| setup / `init_run` | 0.550 s | 7.078 s | 12.9x |
| SCF, cold | 1.420 s | 21.801 s | 15.4x |
| SCF, warm | 1.420 s | 6.647 s | 4.7x |
| **per SCF iteration** | 0.118 s | 0.237 s | **2.0x** |
| iterations | 12 | 28 | |

**2.0x per iteration**, which is the good end of the 2-4x band P10 established, and the
total energies agree to 3.4e-9 Ry on the same run. The iteration *count* is the honest
difference: 28 against 12, because this cell's magnetic state is soft and the two codes'
mixers take different routes to it. That is why the "warm" row is 4.7x where the per-iteration
row is 2.0x, and why the per-iteration number is the one to read.

**The four-component density is not what costs.** The extra work in `nspin_mag = 4` against
a nonmagnetic spinor run is four grid arrays instead of one in `v_of_rho`, the mixer and the
symmetrisation — elementwise work on the dense grid, which is not where the time is. The
Hamiltonian is unchanged in *shape*: it was already a spinor operator on `2 npwx` for P14.

**Symmetrising a magnetization costs a rotation and a gather, not a second pass.** The
charge and the three magnetization components share one set of symmetry maps
(`symmetry_maps`), and the magnetization's extra work is a `(nsym, 3, 3)` einsum against
arrays that were being gathered anyway. Measured on `pw_noncolin/noncolin.in` — iron's
16-operation magnetic group on a 24^3 dense grid, 2026-08-21:

| | per call |
|---|---|
| four components (`_symmetrize_noncollinear`) | 7.75 ms |
| one component, same group and grid | 1.04 ms |

Four components cost 7.5x one, which is what four transforms plus the rotation should cost
and is **0.8% of that run's ~0.99 s iteration** (22 k-points, 18 iterations in 17.9 s). The
symmetrisation is not where a magnetic run spends its time; the eigensolver is, as always.

**A spin spiral costs 1.3x an ordinary noncollinear iteration.** The hydrogen chain of
notebook 12, same cell, same k-grid, with and without `spiral_q`:

| | per SCF iteration |
|---|---|
| noncollinear, one sphere | 335 ms |
| spiral, two spheres | 440 ms |

**1.31x**, and the number is worth explaining because a naive guess is 2x. The transform
count does not change — a spinor has two components either way, so `vloc_psi_nc` does two
FFTs per state in both cases. What doubles is the *bookkeeping* around them: two `fft_index`
gathers instead of one shared, two stick layouts, and two `vkb` blocks in the projector
contraction. The kinetic term and the local potential are unchanged, and the eigensolver
never learns that anything happened.

**What a spiral really costs is symmetry.** It breaks the space group down to the operations
with `S^T q = q`, and even those are not usable until the spin space group is written
(`PLAN.md` P19), so a spiral run needs `nosym` and the *full* k-grid. On a cubic crystal that
is up to 48x the k-points of the same cell without a spiral — far more than the 1.31x above.
An `E(q)` scan is therefore priced as `nq` runs of a symmetry-free calculation, and
`Calculation.at_spiral_q` is what keeps the `nq` factor from also multiplying the setup: it
rebuilds the two spheres, `|k+G|^2`, the stick layout and `vkb`, and shares the cell, both
G-vector sets, the local potential, the core charge and the Ewald sum — the same sharing
`at_kpoints` does, measured at 29.8x on a large cell in P16.

**Memory.** The spiral doubles every array carrying both a `k` and a `G` index: `vkb` at
`2 nk npwx nkb` complex, `kinetic` and the stick tables at `2 nk npwx`. On the notebook's
chain that is 8 rows of 1532 plane waves instead of 4 — a few megabytes. On a real magnetic
crystal the number to watch is still `nk`, which the loss of symmetry has already multiplied.

## What DFT+U costs (P20)

**The Hubbard term is a small separable operator, and it prices like one.** The comparison
is the one this file always makes -- single-core pypresso against single-core QE 7.5 on the
same input -- with a *pair* of inputs that differ only by the `HUBBARD` card, so the ratio
between them is the term's own cost. fcc nickel, one k-point, `nosym`, ultrasoft,
`ortho-atomic` projectors, `U = 3 eV` (`benchmarks/ni-noldau-1k.in` and
`benchmarks/ni-ldau-1k.in`, `--repeats 3`, 2026-08-21):

| | QE 7.5 | pypresso | ratio |
|---|---|---|---|
| per SCF iteration, no U | 0.070 s | 0.134 s | 1.9x |
| per SCF iteration, `U = 3 eV` | 0.069 s | 0.155 s | 2.3x |
| **the Hubbard term's own cost** | **1.0x** | **1.16x** | |

Total energies on the same runs: 1.0e-6 Ry apart without the U (both codes at
`conv_thr = 1e-8`) and **5.1e-9 Ry** with it.

**On QE's side the term is free and on this one it is 16%.** QE's own breakdown for the FeO
benchmark says so directly: `vhpsi` 0.07 s and `new_ns` 0.02 s out of an 18.54 s
`electrons`, i.e. **0.4%**. The difference is not the algorithm -- it is the same
contraction -- it is that this cell is small enough for the fixed cost of one more operator
inside `h_psi` to show. Measured separately on it:

| | per call |
|---|---|
| `occupation_matrix` (`new_ns`), whole k axis, jitted | 0.041 ms |
| `HubbardTerm.apply` on a 9-band block | 0.293 ms |
| `build_hubbard_projectors` (`orthoUwfc`), once per geometry | 9 ms |

The occupation matrix runs *once* per SCF iteration and is noise. What the 21 ms is made of
is the second row, applied as many times per iteration as Davidson calls `h_psi` -- and on
this cell a state is **138 plane waves** while the projector block is 5 wide, so a rank-5
correction is a visible fraction of the whole application. On a cell with a real `npwx` the
projector count per atom is still five or ten and `npwx` is thousands, so the fraction
falls; the FeO benchmark below is where that shows.

**Four atoms, ultrasoft, `nspin = 2`, symmetry-reduced 2x2x2** -- QE's `pw_lda+U/lda+U.in`
at `conv_thr = 1e-10` on both sides:

| | QE 7.5 | pypresso |
|---|---|---|
| SCF wall | 18.54 s | 81.5 s |
| iterations | 24 | 53 |
| **per SCF iteration** | 0.77 s | 1.54 s (**2.0x**) |

**2.0x**, the good end of the 2-4x band P10 established, and pypresso's figure is an upper
bound: it includes the compilation the first iteration pays for. The iteration *count* is
the honest difference -- 53 against 24 -- and it is the usual one, two different mixers
taking different routes to the same fixed point (the total energies agree to 6.7e-9 Ry).
`ns` being part of the mixed state is what makes those routes comparable at all; mixing the
density and leaving `ns` at its output value does not converge on this cell.

**Memory.** `wfcU` is `(nk, npwx, nwfcU)` complex, and `nwfcU` is `2l+1` per correlated atom
-- five for a `d` shell. On the nickel benchmark that is **10.8 kB**; on a 16-atom oxide with
`npwx = 20000` and eight correlated atoms it is `nk x 20000 x 40 x 16 B` = 12.8 MB per
k-point, against `nk x npwx x nbnd x 16 B` = 32 MB per k-point for the wavefunctions
themselves. It is a fixed cost per geometry, rebuilt by `at_positions` and `at_kpoints` and
by nothing inside the loop. QE keeps the same array in the `iunhub` buffer one k-point at a
time, because it stores it beside `evc` on disk; here the whole k axis is resident already
(rule R6) and this follows it. The occupation matrix itself is
`(nspin, nslot, ldmx, ldmx)` -- a few hundred numbers, whatever the system.

## What relaxing a spin spiral costs (P21)

**`dE/dq` is a tenth of an SCF iteration**, because only two terms of the energy
depend on `q` and neither of them is an FFT. At frozen coefficients the
rotated-frame density is a lattice-periodic function on a grid that does not
move, so the Hartree, exchange-correlation, local and Ewald terms have no `q` in
them at all and reverse-mode differentiation never enters the transforms; what is
left is `|k ± q/2 + G|²` and `vkb(k ± q/2)`, and the second is the whole cost.

The hydrogen chain of P19, one core, `conv_thr = 1e-10`, 2026-08-21:

| case | | |
|---|---|---|
| `ecutwfc = 25`, 4 k, `npwx = 1540` | per SCF iteration | 576 ms |
| | `dE/dq`, warm | **54 ms** (0.09 iterations) |
| | `dE/dq`, cold | 253 ms |
| `ecutwfc = 60`, 8 k, `npwx = 5684` | per SCF iteration | 3573 ms |
| | `dE/dq`, warm | **440 ms** (0.12 iterations) |
| | `dE/dq`, cold | 2388 ms |

Same shape as the force (P15, 0.13-0.52 iterations): a gradient is free next to
the ten or twenty iterations that produced the state, so a relaxation costs what
its steps cost and nothing else. There is no QE column because `pw.x` has no
spin spiral to time against.

**The compilation is paid once per step, not once per run, and that is
deliberate.** A new `q` is a new plane-wave sphere and therefore a new `npwx`, so
the traced gradient would be retraced anyway; the cache is dropped explicitly in
`at_spiral_q` rather than left to be silently evaluated at the previous step's
cutoff. The cold-warm difference above — 200 ms and 1.9 s — is what a step pays
for it, against the 3-5 SCF iterations the same step needs. That is the one place
this differs from an ionic relaxation, where the geometry enters through an
argument and one compiled kernel serves every step.

**What makes a step cheap is the warm start.** The wavefunctions cannot travel
between steps — they are coefficients on a sphere that no longer exists — but the
density can, and handing it over is worth most of the SCF: on the chain at
`ecutwfc = 40` the first wavevector takes **9** iterations from the atomic
superposition and every one after it takes **4 to 6**. Six steps to walk from
`q3 = 0.30` to the antiferromagnet at `0.50003`, so the relaxation costs about
three times what a single SCF costs — against thirteen SCF runs for the `E(q)`
scan that would locate the same minimum in one dimension, and a cube of that in
three.

**Memory.** Nothing new is allocated per step. The gradient's working set is one
reverse-mode tape over the two terms that carry `q`: `vkb` at
`(2 nk, npwx, nkb)` complex and its cotangent, which is the same array the SCF
already holds. Peak RSS for the `ecutwfc = 60` case above is 1.02 GB including the
SCF, and the frozen-basis `at_spiral_q` shares both G sets, the local potential,
the core charge and the Ewald sum with the calculation it came from — it rebuilds
only `kinetic` and the projectors.

## What a stress costs (P11)

**A stress is not a force.** A force differentiates the *positions*, which enter through
one complex exponential per atom over a cached radial table; a strain moves `|G|` itself,
so every radial transform in the setup — `V_loc(|G|)`, `rho_core(|G|)`, `f_l(|k+G|)`,
`Q^L_nm(|G|)` — is *inside* the gradient and is recomputed forward and differentiated
backward. That is the whole of the difference between 0.13-0.52 SCF iterations for a force
and what is below.

`conv_thr = 1e-10`, 2026-08-21. These are **internal ratios** — a stress against an SCF
iteration of the same run on the same machine — so they are not pinned to one core the way
`tools/compare_qe.py` pins the QE comparison; there is nothing to compare against, since
`pw.x` does not report a separate stress timing.

| case | | time | vs one SCF iteration |
|---|---|---|---|
| `si2-nc-stress` — 2 atoms NC, 18 k, `npwx = 200`, `ngm = 1459` | SCF iteration | 740 ms | |
| | stress, warm | **450 ms** | **0.6** |
| | stress, cold | 1.20 s | |
| | the term breakdown (`jacfwd`, 9 tangents) | 2.76 s | 3.7 |
| `si2-paw-stress` — 2 atoms PAW, 18 k, `npwx = 415`, `ngm = 9185` | SCF iteration | 2.59 s | |
| | stress, warm | **6.49 s** | **2.5** |
| | the term breakdown | 28.1 s | 10.8 |
| `si8-us` — 8 atoms ultrasoft, 1 k, `npwx = 1607`, `ngm = 36257` | SCF iteration | 1.90 s | |
| | stress, warm | **30.3 s** | **16** |
| | the term breakdown | 173 s | 91 |

The trend is the point and it is not the k-points: from 0.6 iterations on a small
norm-conserving cell to 16 on a large ultrasoft one, tracking `ngm` and the number of
augmentation channels rather than `npwx` or `nk`. A stress is still cheap against the ten
or twenty iterations that produced the state — one evaluation per run, as `run_pwscf` calls
`stress()` once after `electrons()` — but it is not free the way a force is, and on a big
cell it is worth about one extra SCF iteration per atom.

**The breakdown is forward mode on purpose.** The total is one scalar of nine inputs, so
reverse mode gives the whole tensor in one pass. The *decomposition* is eleven scalars of
the same nine, where reverse mode would cost eleven passes and forward mode costs nine and
delivers every term at once — hence `jacfwd` in `autodiff_stress_terms`. It is off by
default (`compute_stress(..., terms=True)`) because nothing but a diagnostic wants it.

### Memory: reverse mode through the setup is the working set

**This is the trade this phase makes and it is a large one.** Peak RSS, same runs:

| case | after the SCF | after a stress | after the breakdown |
|---|---|---|---|
| `si2-nc-stress` | 726 MB | 758 MB | 758 MB |
| `si2-paw-stress` | 4282 MB | 4415 MB | 4415 MB |
| `si8-us` | 899 MB | **11105 MB** | 11049 MB |

The third column is not a typo and the second one is the story: **the breakdown costs
nothing over the plain gradient**, so the 11 GB is the *reverse* pass and not the
forward-mode Jacobian. On the two-atom cells the gradient costs a few tens of MB over the
SCF, which is nothing; on the eight-atom ultrasoft cell it is the dominant allocation of
the whole calculation,
and the reason is the augmentation charge's radial transform: `_qrad_kernel`'s intermediate
is `(ngm, kkbeta)` — 36257 by ~1100, so 300 MB — with a handful of temporaries inside
`spherical_bessel` on top and one of them per `L`. Evaluated forward they are transient;
**taped for a backward pass they are all live at once.** The same is true, more mildly, of
`_vloc_kernel` and `_beta_kernel`, which are chunked at 4096 `q` values but whose chunks
are all taped.

In the terms of `CLAUDE.md`'s memory rule, the peak of a stress evaluation is

    O( ngm x kkbeta x nl )  for the augmentation transform, plus the SCF's own set

which is independent of `nbnd` and of `nk` — a stress does not touch the wavefunctions
beyond one contraction — and grows with the *density* cutoff rather than the wavefunction
one. `jax.checkpoint` on `_qrad_kernel` was tried and **measured to be worth nothing** on
this case (11.0 GB against 10.7), so the intermediates are spread across the radial kernels
rather than concentrated in one; rematerialising all of them, or evaluating the radial
transforms on `|G|` *shells* under a custom JVP, is the fix and it is in the backlog rather
than guessed at here.

The practical consequence, stated so that nobody meets it as a surprise: **a stress on a
cell with `ngm` in the tens of thousands wants ~10 GB**, an order of magnitude more than the
SCF that produced the state, and `terms=True` on such a cell wants the same again for
ninety times the time.

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
4. **The stress's reverse-mode tape through the radial transforms** (P11). 11 GB on
   eight-atom ultrasoft silicon against the SCF's 0.9, and the largest single
   allocation anywhere in the code. `jax.checkpoint` on the augmentation kernel
   alone was measured and is worth nothing, so the next thing to try is a
   `custom_jvp` on each radial transform: `dF/d|G|` has a closed form (one more
   Bessel integral, which `stress/analytic.py` already writes for `dvloc_of_g` and
   `drhoc`), so the transform could carry its own derivative and tape a vector of
   length `ngm` instead of an `(ngm, mesh)` intermediate. That is a ~100x
   reduction in the dominant term and it makes the gradient *cheaper* as well.

### Measured and rejected

*(The stick decomposition was in this list twice, on two different wrong
measurements, before being implemented. See "QE's FFT layout" above.)

* **A faster FFT library.** XLA's CPU FFT is already **2x faster than SciPy's
  pocketfft** single-threaded (0.92 ms against 2.36 ms for a `(4, 30^3)`
  transform). There is no library-level win available; we are using a good one.

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
| 2026-08-19 | Davidson: extend the projected matrices a block at a time instead of rebuilding them, and test convergence after expanding rather than before | 4.5x → 3.8x on eight atoms; one wasted `h_psi` per call removed |
| 2026-08-19 | Sixteen-atom benchmark added | 3.9x / 4.2x |
| 2026-08-19 | QE's stick FFT layout implemented, with the field held xy-contiguous | 1.13x on the local term at eight atoms; ~4% per iteration |
| 2026-08-19 | LSDA: a leading spin axis on the density, potential, `becsum`, `D_ij` and the wavefunctions | no change to the unpolarized path (`si-1k` 0.011 → 0.010 s, `si8-1k` unchanged) |
| 2026-08-20 | Spin-orbit coupling (P14): spinor wavefunctions, `j`-resolved projectors | one Hamiltonian on a doubled space; a nonmagnetic run keeps `nspin_mag = 1`, so the density, potential and XC paths are untouched |
| 2026-08-20 | The k axis chunked as QE's `k_loop`, default one k-point (`pypresso/batching.py`) | converged bismuthene 44.9 → 22.5 s/iteration and 4.91 → 3.16 GB; 9% slower on `metal.in`'s ten cheap k-points |
| 2026-08-20 | The band axis chunked as `vloc_psi_k`'s `DO ibnd`, default one band (`map_bands`/`sum_bands`) | `h_psi` local term 2.48x at 16 atoms, 1.66x at 8, 0.91x at 180 PWs; `si16-1k-ecut30` 4.2 → 2.5x against QE |
| 2026-08-20 | `batch = 1` stops meaning a width-one batch axis: direct call at `nk = 1`, plain `lax.map` beyond | 37% of a Davidson solve, on every k-point of every run |
| 2026-08-20 | `v_xc` and `e_xc` from one `value_and_grad` pass instead of two evaluations | `exchange_correlation` 7.3 → 3.5 ms |
