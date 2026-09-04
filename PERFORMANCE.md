# Performance log

The measurement this project is judged by is **the same input through Quantum
ESPRESSO and through defumat, on this machine, with both restricted to one
core**. Everything else — component breakdowns, compilation counts, guesses
about where time goes — is diagnosis in service of that one number.

```bash
python3 tools/compare_qe.py benchmarks/si-1k.in --repeats 5
```

Single core on both sides is what makes the comparison mean anything. QE is
built serial (`./configure --disable-parallel --disable-openmp && make -j pw`)
and run with `OMP_NUM_THREADS=1`; defumat runs with XLA's intra-op thread pool
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

| | QE 7.5 | defumat | ratio | before |
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

| | QE 7.5 | defumat | ratio |
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

| | QE 7.5 | defumat | ratio | before |
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

| | QE 7.5 | defumat | ratio |
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

The first pair: PBE costs defumat 4% per iteration (0.074 → 0.077 s) for four
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
`conv_thr = 1e-6`, so QE stops there while this comparison runs defumat to
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
(the token does not begin with `--`, so it was read as a filename) and defumat
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
1e-13. defumat was asking for 1e-12 from the first iteration — twelve digits of
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
| defumat `ethr` | 1e-2 | 1.8e-3 | 4.5e-5 | 9.9e-7 | 1.5e-8 | 3.5e-9 | 2.9e-12 | — | |
| defumat steps | 8 | 4 | 4 | 4 | 4 | 3 | 6 | — | **33** |
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
waits for. XLA compilations are now written to `~/.cache/defumat/jax`
(`DEFUMAT_CACHE_DIR` overrides it; `off` disables it), so the second and every
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
not an artefact of small cases. `defumat` now narrows the affinity mask to four
CPUs on import — **1.7x faster out of the box** — with `DEFUMAT_THREADS` to
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

| | QE 7.5 | defumat | ratio | defumat before P9 |
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
k axis is now a dial, `defumat/batching.py`, default 1 -- QE's loop -- with
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
state (`defumat/forces/energy.py`), and QE's six hand-derived terms are
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

| | QE | defumat, before | |
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
a dial beside the k axis (`defumat/batching.py`, `DEFUMAT_BAND_BATCH`),
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

| | QE 7.5 | defumat | ratio |
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
is the one this file always makes -- single-core defumat against single-core QE 7.5 on the
same input -- with a *pair* of inputs that differ only by the `HUBBARD` card, so the ratio
between them is the term's own cost. fcc nickel, one k-point, `nosym`, ultrasoft,
`ortho-atomic` projectors, `U = 3 eV` (`benchmarks/ni-noldau-1k.in` and
`benchmarks/ni-ldau-1k.in`, `--repeats 3`, 2026-08-21):

| | QE 7.5 | defumat | ratio |
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

| | QE 7.5 | defumat |
|---|---|---|
| SCF wall | 18.54 s | 81.5 s |
| iterations | 24 | 53 |
| **per SCF iteration** | 0.77 s | 1.54 s (**2.0x**) |

**2.0x**, the good end of the 2-4x band P10 established, and defumat's figure is an upper
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

## What a projected density of states costs (P8 x projwfc)

**The projection is a few percent of the SCF that produced the states**, which is
what it should be: it is one `natomwfc x npwx` times `npwx x nbnd` product per
k-point, on top of an orthogonalisation that is `natomwfc` cubed and therefore
nothing. Ultrasoft silicon, one core, `conv_thr = 1e-10`, 2026-08-21:

| case | | |
|---|---|---|
| `si2-us-dense`, 29 k, `nbnd = 8`, `npwx = 415`, `natomwfc = 8` | SCF (6 iterations) | 18.7 s |
| | projection, symmetrised | **1.42 s** (7.6% of the SCF) |
| | projection, `lsym = .false.` | 1.10 s |
| | plain DOS, gaussian, 452 energies | 12.6 ms |
| | projected DOS, same, 8 channels | **42.0 ms** |
| `al-tetrahedra`, 10 k, `nbnd = 6`, 384 tetrahedra, 584 energies | plain DOS, linear tetrahedra | 228 ms |
| | projected DOS, 4 channels | **739 ms** |

Three things in those numbers are worth naming.

**The integration costs 3x, not `nproj`x.** The smearing scheme evaluates one
`(nE, nk, nbnd)` array of deltas and then contracts it with the projections, so
the delta — the expensive part, an `erfc` and an exponential per entry — is
evaluated once however
many channels there are; what grows is one `einsum`. The tetrahedron version pays
more (3.2x here) for a structural reason: the corner weights have to be kept
separately instead of being summed into one occupied fraction, which is a factor
of four on the intermediate, and `PROJECTED_ENERGY_CHUNK = 8` divides the energy
chunk by the same four so the working set does not move.

**The symmetrisation's cost is setup, not arithmetic.** 0.32 s of it on 29
k-points — and 0.31 s on *two* — is `build_projection_symmetry`: a least-squares
fit per `l` for the harmonic rotation matrices, then a Python triple loop over
`natomwfc x nsym` to build the gather. Host-side, once per geometry, and flat in
`nk`. The per-k part is a gather of `(nsym, natomwfc, nbnd)` walked over `2l+1`
steps, which is milliseconds.

**The projector build is the projection's own cost, and it is a Python loop over
k.** `build_atomic_projectors` — shared with DFT+U (P20) — walks the k axis in
Python because the eigendecomposition inside it is tiny and batching it would
hold every k-point's overlap at once. At 29 k-points that loop dispatches ~29
un-jitted sequences, which is most of the 1.1 s above. Jitting the body is on the
backlog; it is not on the SCF path, so it has never been worth it.

**Memory.** Nothing here is a new order of magnitude, and every term is small
beside the wavefunctions the projection is taken *of*:

| array | size | `si2-us-dense` |
|---|---|---|
| the projector functions `phi` | `nk x npwx x natomwfc` complex | 1.5 MB |
| the projections | `nspin x nk x natomwfc x nbnd` real | 15 kB |
| the symmetrisation's per-k work | `nsym x natomwfc x nbnd` complex | 49 kB |
| the smearing scheme's deltas | `nE x nk x nbnd` real | 839 kB |
| the tetrahedron intermediate | `chunk x ntetra x nbnd x 4` real | 590 kB (`al-tetrahedra`) |

The one to watch is the first, because it scales like a fraction
`natomwfc / nbnd` of one spin channel's wavefunctions — five to ten percent for a
main-group crystal, and *more than one* for a transition metal with a small band
window, where nine atomic orbitals per atom meet a `nbnd` chosen for the valence
states alone. It is held for the whole projection and freed with it; QE keeps the
same array one k-point at a time in its `swfcatom` buffer, which is the trade
rule R6 already makes everywhere else.

The `run_pdos` entry point costs about **1 s more than the projection** on these
cases, and all of it is one avoidable thing: an `SCFResult` does not carry the
`Calculation` that produced it, so the workflow builds a second one — the basis,
the projectors and the augmentation tables again. Passing the calculation through
would remove it; the same applies to `run_dos` and it has never mattered next to
an SCF.

### Against `projwfc.x` on a spinor run (P69, 2026-09-04)

**The whole call is 2.7x `projwfc.x`, and on both sides it is mostly setup.**
`projwfc.x` is a separate executable, so its 0.62 s is dominated by rebuilding
from the pseudopotential files and reading `evc` off disk — `pw.x`'s own
`init_run` on this cell is **0.69 s**, which is more than the entire `projwfc.x`
run, so the projection arithmetic is not what either number measures. What is
comparable is therefore rebuild against rebuild: **defumat 1.65 s against 0.62
s**, of which **1.53 s is `Calculation.__init__`**. The paragraph above calls
that rebuild "about 1 s" and says it "has never mattered next to an SCF"; on a
PAW dataset it is 87 per cent of the call and about 2.5x QE's.

Fcc platinum, one atom, `Pt.rel-pbe-n-kjpaw_psl.0.1.UPF`, `noncolin` +
`lspinorb`, 2x2x2 unshifted with `nosym`, `nbnd = 18`, `npwx = 286`,
`natomwfc = 18`, `ecutwfc = 30`, `ecutrho = 250`. Both codes start from **their
own converged ground state** — the SCF is outside the pair — one core each
(`taskset -c 0`, `OMP_NUM_THREADS=1`, the affinity set before JAX is imported),
`DeltaE = 0.01` eV on both sides, so the energy grids are 3318 points against
3319. QE's times are `projwfc.x`'s own closing WALL line, not a stopwatch:

| step | defumat | `projwfc.x` |
|---|---|---|
| **the entry point** (`get_pdos`, and `projwfc.x` whole) | **1.65 s** | **0.62 s** |
| the same with QE writing k-resolved files (`kresolveddos`) | | 0.80 s |
| — of it, rebuilding from the pseudopotential files | 1.53 s | not separable |
| — of it, the projection of 18 columns onto 18 bands at 8 k | 0.111 s | not separable |
| — of it, Löwdin charges and the spilling parameter | < 1 ms | |
| — of it, the projected DOS, 3318 energies, 18 channels | 0.022 s | 0.02 s |
| the three above together, from a live `Calculation` | 0.137 s | |

**What is not comparable, and why the last row is not the headline.**
`projwfc.x` cannot skip its setup — a separate process has nothing to inherit —
so its 0.62 s has no 0.137 s inside it to compare against. That row is not a
win over QE; it is a *forecast*: it says where `get_pdos` lands once the
`Calculation` is threaded through, which is below 0.62 s on work QE has to
repeat. Quoting it as a speedup today would be comparing defumat without setup
against QE with it. The other incomparability is I/O and it cuts both ways:
QE writes six files where `get_pdos` returns an object, and its own k-resolved
output is the 0.80 s row against the 0.62 s one — 0.20 s for 8x the rows.

**Where the entry point's 1.65 s goes** (`cProfile`, warm):

| | |
|---|---|
| `Calculation.__init__` | **1.53 s** |
| — of which `build_paw` (`_build_species`, `_truncated_weights`) | 0.93 s |
| — of which `build_augmentation` (`radial_augmentation_transforms`) | 0.42 s |
| `project_states` (`atomic_projections` 0.131 s inside it) | 0.204 s |

So 87 per cent of the call is setup that the SCF has already done once, and it
is the *soft* dataset's setup — a norm-conserving case would not show it, which
is why the scalar cases above put the rebuild at "about 1 s" and left it there.
The fix is unchanged and is now worth something concrete: thread the
`Calculation` through the `SCFResult`, and a call that costs 2.7x `projwfc.x`
costs a fraction of it, because the setup QE cannot skip is the setup this would
stop repeating.

**Memory.** A spinor doubles the first row of the table above and nothing else:
`phi` is `nk x 2npwx x natomwfc` complex, 1.3 MB here, because a spin-angle
function has both components where a scalar orbital has one. `natomwfc` is
`sum (2j + 1)` over the dataset's radial functions, which is **not** in general
twice the scalar `sum (2l + 1)`: it is 18 against 9 for this PAW file only
because every `l` it carries has both of its `j` partners present. A dataset
holding one of the pair — or one whose `j` shells are separate entries with
different occupations — lands somewhere between, so the array is sized from the
file rather than from a factor.

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


## How many iterations, and what a Jacobian costs (P22)

Every other section here measures the cost of *one* SCF iteration. This one measures
**how many of them there are**, which is the other half of the wall clock and the half a
per-iteration timing cannot see.

**The case.** `benchmarks/al-slab.in`, a five-layer Al(100) slab, is here for this and
nothing else: half the cell is a metal, where the dielectric function diverges as `q^-2`,
and half is vacuum, where there is no screening. Plain Anderson reaches **+105 Ry on its
second iteration** before recovering. The vacuum is varied to make it harder.

**Evaluations of `F` to `conv_thr = 1e-8`** -- diagonalisations, which is the only currency
in which a mixer's iteration and a Krylov iteration are comparable, since a diagonalisation
is 84% of a step:

| vacuum (bohr) | dense grid | Anderson | Anderson + Kerker | Newton-Krylov |
|---|---|---|---|---|
| 16 | 12x12x72  | 24 | **14** | 19 |
| 32 | 12x12x100 | 34 | **20** | 74 |
| 48 | 12x12x145 | 32 | **28** | 139 |
| 64 | 12x12x180 | **35** | 36 | 123 |

All twelve runs land on the same energy to <= 7e-8 Ry.

**Kerker is the win, and it is nearly free.** One FFT per iteration for 24 -> 14 and
34 -> 20. It is `mix_rho.f90`'s `approx_screening`, and the one thing to get right is that
**`q_TF` is derived from the cell, not chosen.** A first version here hardcoded 1.5 1/bohr
where QE's `rs = (3 Omega/4 pi nelec)^(1/3)`, `q_TF^2 = (12/pi)^(2/3)/rs` gives 1.008 for
this slab -- over-screening by 2.2x in `q^2`, which cost **48 iterations against 28** at
48 bohr of vacuum, i.e. it turned a win into a loss. Over-screening is worse than no
preconditioning.

Two further details that are only visible on a spin-polarized run: **only the charge channel
is screened.** QE applies `approx_screening` to `drho%of_g(:,1)`, the *charge*, and the
magnetization has no `q^-2` divergence to divide out; densities are carried here as
`(up, down)`, so they are rotated into `(charge, magnetization)` around the screening and
back. And the `G = 0` factor is identically zero, so a preconditioned step cannot change
the electron count.

**Newton-Krylov does not win on cost anywhere, and the reason is structural.** One outer
iteration is one residual evaluation, plus one per GMRES iteration, plus one or two per
line-search backtrack -- and every one of those is a diagonalisation, where Anderson spends
exactly one per iteration and a negligible least-squares solve. The predicted flat outer
count is *real* -- 1, 4, 5, 5 across the four rows, against 3, 29, 61, 53 GMRES iterations -- and it does not help, because the
flatness is bought with GMRES iterations: the 19 -> 74 -> 139 column is the inner count
growing to replace the outer one. Anderson's own count, meanwhile, is nearly flat too
(24, 34, 32, 35): an eight-deep residual history spans the few badly-conditioned
long-wavelength directions this cell has.

**Where the Jacobian action comes from, and what it costs.** Two backends, measured on the
16-bohr slab, one `J v` at the converged density:

| backend | agreement with the other | wall |
|---|---|---|
| `jax.jvp` through the step, **cold** start | 109% apart | 5.9 s |
| `jax.jvp` through the step, **warm** start | 0.8% apart | 2.9 s |
| central difference of `r` (two extra `F`) | -- | 0.4 s |

The finite difference is both more accurate and **4-7x faster**, so it is the default.
Forward mode through Davidson's `lax.while_loop` costs more than two extra primal solves do
when both start from a converged guess, and warm-started Davidson exits in one or two steps,
which makes its tangent a one-step approximation to the eigenvector response rather than the
response. Reverse mode is not an option at all and the number is worth recording: the tape is
`n_davidson_steps * nvecx * nbnd * npwx` complex, gigabytes on the eight-atom cell.

**Memory.** The packed state is `nspin * nr` floats -- 83 kB for the 16-bohr slab, 21 MB for
a 128^3 dense grid with two spin channels -- and a Krylov subspace holds a few tens of them,
which is negligible beside the wavefunctions. What is not negligible is that a JVP holds a
tangent `psi` of the same shape as `psi`, doubling the wavefunction working set for the
duration of the call; that is why it goes through the same `k_batch` dial the primal does.
The dense Jacobian is never formed and the number says why: `(nspin nr)^2` is
10368 x 10368 on the *smallest* case here.

**DFT+U costs the residual solver nothing extra to support and a great deal to use.** ``ns``
joins the packed state as ``mix_rho.f90`` puts it in ``mix_type``, and ``v_hubbard`` is already
``jax.grad`` of the Hubbard energy (P20), so no new Jacobian machinery was needed. On fcc
nickel with ``U = 3 eV`` (``benchmarks/ni-u-unstable.in``, one k-point) the solver reaches the
mixer's fixed point on 58 evaluations of ``F`` against 9 -- the same ratio as everywhere else.
What it buys is again the unstable solution: from a state perturbed 2% off the non-magnetic
saddle, Anderson runs away to the ferromagnet in 9 evaluations and Newton-Krylov returns to the
saddle in 31.

**What would change the cost verdict** is P22c -- the Sternheimer `custom_jvp`, a projected
CG solve per occupied band instead of a differentiated Davidson. That is the only way an
inner Krylov iteration stops costing a diagonalisation, and it is the same routine DFPT
needs, which is the argument for writing it.

## How many iterations a continuation saves (P23)

The same currency as P22 -- **iterations, not seconds** -- because a continuation changes
nothing inside an SCF step and everything about how many steps there are. All pairs at
`conv_thr = 1e-8` (`1e-9` for platinum), and every pair lands on the same energy, which is
the check that makes the count meaningful at all:

| case | from the atoms | continued | same answer to |
|---|---|---|---|
| Si, `nspin` 1 -> 2 (seeded, decays to zero) | 5 | 4 | 2e-9 Ry |
| Si, `nspin` 2 -> 4 (nonmagnetic) | 5 | **1** | 1e-9 Ry |
| bcc Fe, 2 -> 4, moment rotated onto `x` | 25 | **1** | 2e-8 Ry |
| bcc Fe, 4 -> 2, moment found and laid on `z` | 30 | **1** | 4e-8 Ry |
| bcc Fe, 1 -> 2, magnetization seeded | 30 | 27 | 5e-9 Ry |
| Pt, scalar PAW -> relativistic PAW + `lspinorb` | 13 | **7** | 2e-10 Ry |

**The rule the table shows is one rule.** What carries over is the *charge*. Where the
charge is the whole answer -- a nonmagnetic run rewritten as a spinor one, a converged moment
merely rotated onto another axis -- the continued run converges on the state it was handed
and the iteration is the one that builds the result. Where the run has to *find* a
magnetization the source does not have (Fe 1 -> 2), the saving is three iterations out of
thirty, because the magnetization is the slow variable and it is exactly what was not
carried. Platinum is in between: the dataset changes with the spin-orbit coupling, so
`becsum` is re-seeded and only the density crosses -- and that is still half the iterations.

**What it costs is one promotion**, which is arithmetic on the dense grid: a decomposition
into `(n, m)`, at most a 3x3 eigendecomposition to find an axis, and a recomposition. It is
not measurable beside a diagonalisation. The wavefunctions cross as a span for the first
Rayleigh-Ritz, which costs what `wfcinit` already costs and nothing more.

**Memory is unchanged.** The promoted state is the same three arrays the mixer already holds
-- `(rho, becsum, ns)` -- and the span is at most `2 nbnd` vectors per k-point, which is what
`_as_spinors` already builds from the atomic orbitals. Both runs of a pair are separate
processes' worth of state only if the caller keeps the first `SCFResult` alive for its
wavefunctions; dropping them (`wavefunctions=False`) leaves a working set no larger than one
SCF's.

## What a linear response costs, and what the exact Jacobian bought (P24)

Three quantities, all measured on the two-atom silicon of `test-suite/ph_base/si.scf.in`
(`ecutwfc = 18`, ten k-points, four occupied bands) on one core.

**The velocity operator is one `jvp`, and it costs about one `h_psi` per direction.**
`dH/dk_a` and `dS/dk_a` applied to every band at every k-point: 2.9 s for all three
cartesian directions on a single k-point at `ecutwfc = 12`, most of which is compiling the
projector rebuild. The local potential contributes nothing — its tangent is symbolically
zero, so the `jvp` never issues its FFTs — and nothing dense is formed, so the peak working
set is one extra copy of `vkb`, `(nk, npwx, nkb)` complex.

**The Sternheimer solve against the two routes P22 had.** One `chi_0 K v`, on the same cell
and from the same converged state:

| route | time | what it is |
|---|---|---|
| Sternheimer (`response/sternheimer.py`) | **0.5 s** | exact; 23-28 CG iterations per band |
| `jax.jvp` through Davidson (`ScfResidual.jvp`) | 3.5 s | a one-step approximation to the eigenvector response |
| central difference of `F` (`jvp_finite_difference`) | 0.1 s | two evaluations of `F` from a warm start |

The finite difference is the cheapest here and it is *also* the least accurate, which the
sweep below is the evidence for. That the exact solve should additionally **scale** better —
a projected CG over `nocc` bands against a Davidson subspace of `nvecx = 4 nbnd` — has not
been measured: si2 at `ecutwfc = 12` is far too small to separate them, and the case that
would is a cell where the Sternheimer solver is not yet the bottleneck of anything.

**And the exact Jacobian's first result was about P22 rather than about itself.** Sweeping
the finite difference's step against `chi_0 K` gives a textbook U between truncation and
noise:

| step | relative difference from the exact Jacobian |
|---|---|
| 0.3 | 8.3e-2 |
| 0.1 | 9.7e-3 |
| 0.03 | 8.0e-4 |
| **0.01** | **4.0e-4** |
| 0.003 | 1.0e-3 |
| 1e-6 (`jvp_finite_difference`'s default) | **1.1e-1** |

P22's default step is chosen for a gradient and sits five orders below the minimum, deep in
the eigensolver's noise. Its reported 0.8% agreement between `jax.jvp` and the difference
was two noisy numbers agreeing, and there was no way to know that without a third.

**The dielectric constant.** 18 linear-mixing iterations at `alpha_mix = 0.7` to
`|ddv_scf|^2 < 1e-14`, averaging 28 CG iterations per band, **66 s** on ten k-points — three
Sternheimer solves per iteration plus three for the bare `P_c r|psi>`. `ph.x` takes 5
iterations for the same fixed point with Broyden mixing, so the iteration count here is a
straightforward 3.6x that could be bought back with the mixer the SCF already has. The Born
charges add six more `jvp`s through `at_positions`, and on this cell their cost is inside
the run-to-run spread of the loop they follow (66 s with them, 77 s without, on two runs
that differ in compilation state as much as in work).

**The memory is the CG's four band-blocks** — the gradient, its preconditioning, the search
direction and the previous one — at `(nk, nocc, npwx)` complex each with the whole k axis in
flight, which is the Davidson subspace's working set at `nvecx = 4 nbnd`, through the same
`k_batch` dial. The electric field holds **two sets of three more**, one per direction, for
the whole self-consistent loop: `bare` and `dpsi` are each `(3, nspin, nk, nocc, npwx)`
complex. On this cell (`npwx = 350`, `nk = 10`, `nocc = 4`) one band-block is 0.22 MB, the
CG's four are 0.9 MB and the field's six are 1.3 MB -- nothing. On a cell with 200 occupied
bands and 20000 plane waves at 64 k-points the same block is **4.1 GB** and the field's six
are **25 GB**, which is the number that decides whether such a run happens at all: the CG
already goes through the `k_batch` dial, and those six do not.

### What ultrasoft and PAW add to it (P24a)

Same cell, same k-grid, same wedge -- only the dataset changes -- so the columns are the
cost of the augmentation charge and of PAW's one-centre terms and of nothing else. One
core, `conv_thr = 1e-12`, the self-consistent response run to `|ddv_scf|^2 < 1e-14`:

| case | dataset | iterations | response |
|---|---|---|---|
| `si-epsilon` | norm-conserving, `ecutrho = 4 ecutwfc` | 18 | **44 s** |
| `si-epsilon-us` | ultrasoft, `ecutrho = 8 ecutwfc` | 19 | **82 s** |
| `si-epsilon-paw` | PAW, `ecutrho = 8 ecutwfc` | 19 | **95 s** |
| `c-epsilon` | ultrasoft carbon, `ecutrho = 11 ecutwfc` | 17 | **49 s** |

Ultrasoft costs about **1.9x** the norm-conserving run and PAW **2.2x**, and most of that
is not the response at all: the dual is 8 rather than 4, so every dense-grid quantity is
twice the size before anything new is computed. The iteration count is unchanged, which is
the thing worth knowing -- the augmentation charge does not make the response harder to
converge, only more expensive per step.

**None of the new terms is an extra pass over anything.** `dbecsum` and the augmentation
charge's response ride inside the `jvp` of the density builder that was already being
taken; `int3` is a `jvp` of `newd`, which the SCF already evaluates once per iteration; and
`PAW_dpotential` is a `jvp` of `onecenter`, likewise. What each adds is a tangent alongside
a primal that was going to be computed anyway -- which is the cost model of forward-mode
differentiation and the reason this phase is not 2x slower again.

## What a phonon costs (P25)

Same cell as P24 — the two-atom silicon of `test-suite/ph_base/si.scf.in`, `ecutwfc = 18`,
ten k-points, four occupied bands — on one core, and this time there is a **direct QE
number to compare against**, because the `ph.x` run that produced the reference prints its
own timings and its own iteration counts. That run does `epsil = .true.` as well, so its
phonon part has to be separated out: its cumulative clock reads 1.9 s when the electric
field's loop ends and 4.1 s when the second representation does, so the six `Gamma` modes
cost it about **2.2 s** of its 4.15 s wall.

| | defumat | `ph.x` |
|---|---|---|
| the six `Gamma` modes, after the SCF | **57 s** | ~2.2 s |
| self-consistency of the response | 17 iterations | 5 per representation |
| perturbations solved | 6 modes | 6 modes, in 2 irreducible representations of 3 |
| linear solves | 6 x 17 = **102** | 3 x 5 x 2 = **30** |
| CG steps per band per solve (`av.it.`) | **27.7** | 9.0 - 9.7 |

**A metal costs what the same cell would cost without one** (P28). Two-atom
aluminium's six modes take **78 s** here against a `ph.x` whose whole `PHONON`
clock is 4.73 s wall (11.10 s CPU -- that reference was not run pinned to one
core, so it is not the single-core comparison this document otherwise insists
on, and the ratio is quoted only as an order). The structural comparison is the
one that transfers, and it is P25's unchanged: **9** self-consistent iterations
against `ph.x`'s 7 per representation, at `av.it. = 23.0` against 3.3-6.3. Both
gaps are the two backlog items below and neither is metallic -- the smearing
branch adds a weight to a projector and a `def ldos` to a density, and nothing
in either scales with anything. The **split assembly P28 introduced is free**:
it is one more `jvp` of `grad_u L` per mode, against a stage that is already
96% linear solves.

**About 26x, and none of it is the second derivative.** The two ratios in the table
multiply to about 10, which leaves a factor of 2.6 per CG step — the same place the SCF
sits against `pw.x` (P10), so the arithmetic is not the problem. The two counts are, and
both have a named cause read out of the Fortran rather than guessed:

- **QE schedules the linear solve's threshold and this does not.** `dfpt_kernels.f90`:
  `thresh = 1e-2` on the first iteration and `min(0.1 sqrt(dr2), 1e-2)` thereafter, so the
  early solves — whose right-hand side is about to change anyway — are cheap, and only the
  last ones are tight. Here `threshold` is fixed at 1e-12 from the start, which is why
  `av.it.` is 27.7 against 9.3. **This is `electrons.f90`'s `ethr` schedule in a second
  place**, and it is worth what it was worth there: CLAUDE.md already records "a fixed
  tight threshold does three times the eigensolver work", and three times is what this is.
  `response/sternheimer.py`'s docstring quotes the rule; the phonon loop does not use it.
- **QE mixes the induced potential with Broyden.** `LR_Modules/mix_pot.f90` is a modified
  Broyden over `nmix_ph = 4` previous iterations — it prints `alpha_mix = 0.700` while
  doing it, which is easy to misread as plain linear mixing — and reaches 3e-16 in 5
  iterations where linear mixing here takes 17 to reach 6e-15.

What the **irreducible representations** buy is not fewer solves: `ph.x` perturbs along all
six modes, exactly as this does. What they buy is that each representation carries its own
self-consistent loop, so it converges, is stored and is *released* independently — which is
the memory item below rather than a time one.

**The second derivative is free next to the response.** Timed stage by stage (which costs a
few extra compilations, so the stages sum to 70 s where the single call takes 57):

| stage | time |
|---|---|
| six `jvp`s through `at_positions` — the bare perturbations (`dvqpsi_us`) | 1.4 s |
| the self-consistent loop, 17 iterations x 6 solves | 67.5 s |
| six `jvp`s of `jax.grad(frozen_energy)` — the matrix itself | **1.4 s** |

So the derivative that replaces `dynmat0` + `d2ionq` + `drhodv` is **2%** of the run, and
both autodiff stages together cost less than one iteration of the loop. That is the pattern
P15 and P11 both measured: an autodiff route pays for the *forward* function, and a second
derivative of a cheap forward function stays cheap. All of the cost, and all of the
backlog, is in the linear solves.

**Memory: `3 nat` wavefunction-shaped arrays, twice.** The bare perturbations and the
first-order wavefunctions are both held for every mode, `(nspin, nk, nocc, npwx)` complex
each:

| cell | working set |
|---|---|
| si2, 10 k-points, 4 bands, ~300 PW | **2 MB** |
| 16 atoms, 100 k-points, 32 bands, 3000 PW | **7 GB** |

That is the trade named in the module docstring and it is deliberate: the bare terms are
re-used at every iteration, so recomputing them would cost 17 x 6 `jvp`s through
`at_positions`. Solving one representation at a time is what bounds it — 3 modes in flight
instead of `3 nat` — which is what QE's representation loop is doing to its memory while
this holds all of them.

## What a third derivative costs (P26)

`tests/data/qe/si-electrostriction.in` — two-atom silicon, `ecutwfc = 12`, the **whole
unshifted 2x2x2 grid** with `nosym`, 8 k-points and 190 plane waves. The grid is the
smallest one closed under the point group, which is what P26 requires (`PLAN.md`), so it
is also the cheapest case the phase has. One core.

| stage | 8 k-points | solves | what it is |
|---|---|---|---|
| SCF | 1.9 s | — | |
| `refined_states` | 0.03 s | — | one Davidson pass from an excellent guess |
| field response (P24) | 38.5 s | 60 | 3 perturbations x 19 iterations, plus 3 for `P_c r|psi>` |
| **strain response** | **80.1 s** | **108** | 6 perturbations x 18 iterations |
| third derivative | 36.2 s | 18 | `db/dx`, plus 6 `jvp`s of the functional |
| elastic constants | 9.7 s | 0 | 6 `jvp`s of the stress gradient |
| total | **166 s** | 186 | |

One run, one machine, so the rows are comparable with each other; the absolute numbers
carry whatever else the machine was doing.

**The strain response is the expensive half and the reason is counting, not physics.** A
strain has six independent components where a field has three, and each is driven to
self-consistency by the same `solve_linter` loop at the same 22 CG steps per band per
solve. It is P24's cost times two and P25's (six modes on this cell) times one; nothing
about it is new, which is the point.

**The third derivative itself is cheap.** 36 s for six columns, and half of that is
the `db/dx` solves — the position operator's own strain derivative, one Sternheimer
solve per (direction, strain). The six `jvp`s of the functional are ~3 s each: each is
forward-over-forward through `at_strain`, three Hamiltonian applications over the whole
k axis, three density `jvp`s and three kernel `jvp`s. **A `jvp` of a second derivative is
not more expensive than the second derivative** — which is the whole economic argument for
the 2n+1 route over the published one. The alternative is a sweep: five re-converged
SCF-plus-DFPT calculations per independent strain, which on this cell is 5 x 6 x 40 s
against 36 s, a factor of **33**.

**The elastic constants are nearly free** — 9.7 s, six percent of the run — because they
reuse the strain response the third derivative already paid for. Standing alone they would
cost the 80 s as well.

**Memory, and it scales better than the phonon's.** Six bare strain perturbations and six
first-order wavefunctions at `(nspin, nk, nocc, npwx)` complex each, plus P24's three of
each and the three `db` columns of the strain being assembled: **21 blocks**, about 3 MB on
this silicon. The number is **six**, not `3 nat` — a strain has six components whatever the
cell contains — so on P25's 16-atom yardstick (100 k-points, 32 bands, 3000 plane waves)
this is *below* the phonon's 7 GB rather than above it, and the gap widens with every atom
added. The way down, if one is wanted, is P25's: the six strains are independent except
through the symmetrisation, so they can be run one at a time at the cost of re-entering the
loop — and on a `nosym` grid, which this phase requires anyway, nothing couples them at
all.

**Backlog, and it is P25's verbatim.** The linear solve holds a fixed 1e-12 threshold where
`dfpt_kernels.f90` schedules it against the self-consistency of the response, and the
induced potential is mixed linearly where `mix_pot.f90` uses a modified Broyden over four
iterations. 18 iterations at 22 CG steps is what that costs; QE reaches its answers in 5
iterations at 9. Both fixes serve P24, P25 and P26 together and neither has been made.

## P24b and P24c — what the Born charges and the metal branch cost

**The Born charges are a derivative of the force, and the expensive part of that is three
calls.** P24b replaced the transcribed `zstar_eu` contraction with one `jvp` of
`jax.grad(frozen_energy)` per *field* direction — three in all, whatever the cell contains,
because the position tangent is zero and a single call returns the whole `3 nat` column.
Two smaller terms beside them do scale with the cell; the paragraph below counts them.
**Measured** on the ultrasoft silicon of `si-epsilon-us.in`, which is the case the
correctness claim rests on: the field response alone is **25.0 s** and the response plus the
charges is **31.9 s**, so they cost **6.9 s**, 28% on top. That is more than "free" and less
than the second self-consistent loop the alternative would have been.

**Where those 6.9 s go is a count of tangent evaluations, and only one of the three terms
scales with the cell.** Three `jvp`s of `jax.grad(frozen_energy)` — one per field direction,
independent of `nat`, and individually the expensive ones, since each carries the density
builder and the potential. One `jacfwd` of the frozen polarization over the `3 nat`
positions. And `3 nat` `jvp`s *per field direction* inside `constraint_position_term`, which
is `9 nat` in all — each cheap (a projector sandwich, no FFT) and each a separately
dispatched compilation, which on these array sizes is the cost that matters (the standing
observation of this file). **Backlog:** that last one is a `jacfwd` over the positions
written as a loop, for no reason but that it was written before the shape was clear;
collapsing it removes `9 nat - 3` dispatches. It has not been made, because 6.9 s on a
two-atom cell is not where the time is.

**A metal's solve pays the empty bands' share of the CG.** `orthogonalize`'s smearing branch
sums over every band (`nbnd_eff = nbnd`), and where QE truncates the *solve* at
`setup_nbnd_occ`'s per-k count, this keeps the block whole — a per-k count is a dynamic
shape and rule R2 does not allow one inside a compiled loop. The bands past the cutoff carry
an occupation of zero, so the answer is exact and the cost is `nbnd/nbnd_occ`. Measured on
`al-metal.in`: 8 bands against a per-k `nbnd_occ` of 1 to 3, so this cell pays between two
and eight times the block it needs — the worst ratio a cell can have, since `nbnd` is chosen
a fixed *margin* above the occupied count and one aluminium atom occupies barely more than a
band. It shrinks with every atom added. **Backlog**, not a fix made: the way down is `cegterg`'s — compact with a
mask rather than change a shape — and it is the same item the response loop's fixed 1e-12
threshold already sits in.

## What a variable-cell relaxation costs (P29)

**The comparison, single core on both sides, best of three.** `pw.x` is the vendored
serial build and the number is its own `PWSCF ... WALL`; defumat is pinned to one CPU by
the affinity mask, as `tools/compare_qe.py` pins it. On QE's `pw_vc-relax/vc-relax4.in`
(rhombohedral arsenic at 500 kbar, 2 atoms, 10 k-points, 10 ionic steps on both sides):

| | best of 3 | samples |
|---|---|---|
| `pw.x` | **4.39 s** | 4.4, 4.5, 4.4 |
| defumat | **32.36 s** | 39.0, 33.1, 32.4 |
| ratio | **7.4x** | |

**Best of three and not a single sample, and this is the finding to read first.** This
machine is shared, and a single timing of the same run an hour earlier gave **222 s** --
a factor of **6.9** out. Both codes are pinned, but they are pinned to the *lowest-numbered
available core*, which four of someone else's processes were also using, so the pin does
not isolate. The first version of this section reported the resulting **28x** as the
headline and built an analysis on it that did not survive re-measurement: the stress's
retrace, which looked like 10.8 s per ionic step and "half the run", is **0.6 s**. Nothing
about the code changed between the two measurements. The rule this file already states
about benchmarking on a two-atom cell has a companion: **a single wall-clock sample on a
shared machine is not a measurement**, and the repeat-and-take-the-minimum that
`compare_qe.py --repeats` exists for is not optional.

**Where the 7.4x goes.** The per-piece costs, measured the same way:

| | seconds |
|---|---|
| setup (`Calculation`) | 1.53 |
| first SCF, 6 iterations | 2.89 (0.48 per iteration) |
| force | 0.38 first call, 0.02 after |
| stress | 1.93 first call, **0.57** after |
| `at_cell` | 0.27 |
| whole relaxation: 10 ionic steps, 57 SCF iterations | 32.4 |

57 SCF iterations at 0.48 s is **27 s**, which is 85% of the run: **a variable-cell
relaxation costs what its SCFs cost**, and the ratio against `pw.x` is P10's ordinary
per-iteration ratio rather than anything the cell introduced. The force, the stress and
`at_cell` together are about 1.2 s a step.

**The one thing the cell *does* add is a retrace, and it is a fifth of the run.**
`at_strain` drops `_energy_gradient` on every call -- it has to, since the compiled
gradient closes over the cell it was traced at -- so every ionic step compiles the strain
derivative again:

| | first call | second call | retrace |
|---|---|---|---|
| base cell | 1.93 s | 0.57 s | 1.36 |
| moved cell (x0.99) | 1.18 | 0.55 | 0.63 |
| moved cell (x0.98) | 1.17 | 0.57 | 0.60 |
| moved cell (x0.97) | 1.27 | 0.57 | 0.71 |

0.6 s of compilation for 0.57 s of arithmetic, ten times over: **6 s of 32**. A fixed-cell
`relax` does not pay it, because `at_positions` keeps its compiled force and one
compilation serves the whole trajectory. Making the cell an *argument* of the traced
function rather than a constant folded into it would do the same here. **Backlog item 4**,
and it is worth about 20%, not the 50% the contaminated measurement claimed.

**What the ionic step count says, and it is the good news.** Both codes take **10** steps
on `vc-relax4`, `vc-relax5` and `vc-relax6` and **11** on `vc-relax3`: the transcribed
BFGS, its trust radius and its Wolfe line search reproduce QE's trajectory step for step
even with the cell in the coordinate vector. None of the 7.4x is the optimizer taking a
worse path.

**What a `treinit_gvecs` run costs, and when it is not affordable at all.** Rebuilding the
grids every step is a full setup per step -- 1.53 s on this two-atom cell, and it grows
with the cell rather than with the step count. `vc-relax6` is the affordable end of it. The
unaffordable end is a case that does not converge: five-layer graphite with
`cell_dofree = 'z'` and `treinit_gvecs` ran **50 ionic steps in 18 minutes of `pw.x`** and
stopped with "The maximum number of steps has been reached", because rebuilding the grids
makes the energy surface *discontinuous* -- the FFT dimension along `c` changes as `c`
does, and `etxc` is evaluated pointwise on it -- so a line search with Wolfe conditions
has nothing to converge to. That is the trade `treinit_gvecs` makes: it removes the Pulay
error and buys a surface the optimizer cannot walk. `PLAN.md`'s P29 entry has the case.

## What a Raman tensor costs (P35)

The third derivative of P26 with the atoms as its geometry variable, on AlAs
(`ecutwfc = 10`, 2 atoms, the unshifted 4x4x4 grid run whole under `nosym`, so
64 k-points and ~110 plane waves) and on the silicon of `si-epsilon-unshifted-nosym`
(`ecutwfc = 18`, 64 k-points):

| stage | AlAs | silicon |
|---|---|---|
| SCF to `conv_thr = 1e-12` | 5.2 s | 2.8 s |
| field response, 3 perturbations, 9 iterations | 41 s | 63 s |
| displacement response, `3 nat` perturbations, 9 iterations | 71 s | 104 s |
| `3 nat` `jvp`s plus their `db` solves | 33 s | 35 s |
| **total** `raman_tensors` | **145 s** | **202 s** |

**No ratio against QE here, and the reason is the phase's own finding**: the
`ph.x` branch that would be the other side of it does not reproduce QE's own
committed example and fails its internal consistency check (`PLAN.md` P35), so
its timings measure a calculation that is not the same calculation. Every other
stage in this table has a QE counterpart timed elsewhere -- the SCF in the tables
above, the two response loops in P24's and P25's entries.

**The two self-consistent responses are 78% of it and the third derivative is 23%**,
which is the same shape P26 measured and for the same reason: a `jvp` of an
assembly costs about what the assembly costs, and the assembly is cheap beside a
projected CG solve. The consequence is that a Raman tensor costs roughly *twice*
a dynamical matrix rather than more -- the displacement response is shared with
it, and the field response is P24's.

Backlog item 5 (scheduling the response solver's threshold against the
self-consistency of the response, `dfpt_kernels.f90`'s
`thresh = min(0.1 sqrt(dr2), 1e-2)`) applies here at full strength: 78% of this
phase is in the two loops it would speed up.

The peak working set is the responses': `(3 + 3 nat)` arrays of
`(nspin, nk, nocc, npwx)` complex, plus the same again in the mixer's Anderson
history. On these cases that is under 1 GB; on the 16-atom cell P25 already
records for the dynamical matrix it is the same 7 GB, since the Raman tensors add
three field responses and nothing per atom.

## What the wedge saved a third derivative (P36)

The rank-3 average lifted the closed-grid refusal P26 and P35 shipped with, so
the same k-sample can now be run reduced. Both pairs are the *identical*
Monkhorst-Pack sample, unshifted, one run whole under `nosym` and one reduced to
its irreducible wedge:

| | k-points | closed grid | wedge | ratio |
|---|---|---|---|---|
| `raman_tensors`, AlAs `ecutwfc = 10` | 64 -> 8 | 112 s | **51 s** | 2.2x |
| `electrostriction`, silicon `ecutwfc = 18` | 64 -> 8 | 205 s | **73 s** | 2.8x |

**Not 8x, and the shortfall is not overhead**: the cost is per *perturbation* as
well as per k-point, and the `3 nat + 3` perturbation set does not shrink with
the k-set. What does shrink is every projected CG solve inside it. The wedge run
also does more work per iteration -- `symmetrize_directional` and
`symmetrize_atom_displacement` run on the induced densities, and the value/derivative
split P36 added puts one more `symmetrize_directional` inside the functional -- and
that is inside the 2.2x.

The agreement is 8.7e-14 and 7.9e-14 respectively (`PLAN.md` P36), so this is a
free factor of two on every case either phase runs, and more on a crystal whose
group is larger relative to its k-grid.

## What sharing the displacement response saved (P36)

A Raman tensor and a dynamical matrix need the same `solve_linter` output, and
before P36 a spectrum solved it twice. `raman_tensors(keep_internals=True)` hands
it over and `dynamical_matrix(response=...)` takes it:

| stage | solved again | reused |
|---|---|---|
| `dynamical_matrix` after `raman_tensors`, silicon | 104 s | **1.4 s** |

What is left is the force constants themselves -- the two `jvp`s per mode -- and
the diagonalisation. The equality against a matrix built from its own solve is a
committed test (1e-12), because an optimisation that changes a number is not one.

The **memory** trade is stated rather than taken silently: `keep_internals` holds
`3 nat` arrays of `(nspin, nk, nocc, npwx)` complex alive after `raman_tensors`
returns, which is the same working set the response loop already had at its peak
and so does not raise it -- but it does *hold* it, where the default drops it. It
is off by default for that reason.

## What a spectrum costs (P36)

Nothing worth a table: `mode_activities` is a contraction of `(nat, 3, 3, 3)`
with `(3 nat, nat, 3)` and runs in under a millisecond. The whole of
`vibrational_spectrum` is the two responses above.

## First contact with a GPU (P10 / GPU.md Phase 0)

**A different metric, and it is never mixed with the one above.** Everything
else in this file is single-core defumat against single-core Quantum ESPRESSO.
This section is **GPU defumat against CPU defumat, same input, same code, per
SCF iteration**, with compile time as its own column — `GPU.md` §2.3, and the
reason the two must not share a table. The QE comparison stays a CPU claim.

Run 2026-08-25 on Aalto's Triton: **Tesla V100-SXM2-16GB** (driver 580.173.02),
jax 0.11.1 with `jax-cuda12-plugin` 0.11.1, against **four pinned cores of an
AMD EPYC Milan**, both sides `tools/gpu/phase0.py` at commit `ed14741`.
`al10-metal` is ten atoms and **ten k-points** with a committed QE reference;
the `si` cells are one k-point, where `k_batch` is a no-op by construction.

| case | dials | GPU ms/it | CPU ms/it | ratio | peak device |
|---|---|---|---|---|---|
| `al10-metal` | `k=1, b=1` (the defaults) | 801 | 793 | 0.99x | 0.10 GB |
| `al10-metal` | `k=all, b=1` | 2075 | 931 | 0.45x | 0.15 GB |
| `al10-metal` | **`k=all, b=all`** | **177** | 963 | **5.44x** | 0.40 GB |
| `si8-1k` | `k=1, b=1` | 75 | 27 | 0.35x | 0.09 GB |
| `si8-1k` | `k=1, b=all` | 19 | 22 | 1.13x | 0.05 GB |
| `si-1k` | `k=1, b=1` | 16 | 5 | 0.33x | 0.14 GB |

**Nothing was ported to produce this.** The same source, unmodified, on a CUDA
device — which `GPU.md` predicted and which had never been evidence.

**The dials invert, and that is the measurement.** On the GPU, `al10-metal` goes
**801 → 177 ms** between QE's loop and the fully batched mode, a factor of
**4.5**; on the CPU the same change goes 793 → 963, i.e. 1.2x the *wrong* way.
The defaults are cache-shaped and a GPU has no such cache, exactly as
`batching.py`'s own docstring says — so **a GPU run left on the defaults gives
up 4.5x**, and both dials are part of a GPU job's configuration rather than a
tuning afterthought.

**`k=all, b=1` is worse than either end, and that is the finding worth
carrying.** 2075 ms, against 801 for the loop and 177 for the batch. Batching k
while looping bands does not interpolate between the two: it multiplies the
per-band kernel launches by `nk`, so it buys the batched mode's memory and the
looped mode's launch count. **A dial is not a slider here** — the two axes have
to move together, and a plausible half-measure is the worst setting available.

**Small cells lose on a GPU and that is not a defect.** `si-1k` at 0.33x and
`si8-1k` at 0.35x are launch-overhead dominated; `si8-1k` recovers to 1.13x on
the band dial alone. This is the standing rule ("a two-atom cell will not show
you any of this") in its GPU form, and the reason `benchmarks/` — single
k-point *on purpose* — cannot be the whole first-contact set.

**fp64 costs 1.78–1.98x on a matmul and 0.85–1.44x on a batched 3D FFT**, over
the six records. Measured on the two kernels this code is made of rather than on
a generic FLOPs probe. The V100 is a 1:2 fp64 part and the matmul reproduces
that ratio; **the FFT's cost is grid-dependent and must be quoted as a range**,
1.44x on `al10-metal`'s 27x15x72 down to 0.85x on `si-1k`'s 16³ — where double
came out *faster*, which is a latency-bound transform whose timing is noise
rather than an inversion of the hardware. The first draft of this table quoted a
single "1.01x" because the comparison tool collapsed six measurements into
whichever record sorted last; `phase0_compare.py` reports the spread now, and
the flattering end of a spread presented as the number is exactly the failure
mode this file exists to prevent.

**This is the datum `GPU.md` Phase 3's rank waits on and it settles it** — the
range does not change the conclusion: single precision's ceiling is ≤2x on the
dense algebra and at most ~1.4x on the transforms, so float32 belongs *after*
Phase 4's sharding, not before it. `GPU.md` declined to guess this and flagged
the ordering as provisional; it is now measured.

**Memory is not the binding constraint at this scale** — 0.40 GB peak against
the allocator's 11.8 GB limit, read from `jax.Device.memory_stats` and never
from `nvidia-smi`, which reports the preallocated pool. The dial's cost *is*
visible and is the §2.4 trade in miniature: 0.10 → 0.40 GB for the 4.5x. It is
`GPU.md`'s bismuthene case at 12.7 GB, not these, that will find the edge of a
16 GB card.

**Correctness, which is what Phase 0 was actually for:**

* the total energy agrees with the CPU run to **1.6e-13 Ry** worst case over all
  six configurations — round-off, and three orders inside the `conv_thr` all
  three cases were run at (1e-10);
* `al10-metal` on the GPU reproduces the **committed QE reference to 1.88e-09 Ry**,
  which is the identical figure the development workstation gets. Not "matches
  the other run" — matches the external reference to the same digit;
* **determinism holds bit for bit**, on every case, run twice in one process.
  `basis/fft.py`'s accumulating scatter over deliberately duplicated indices was
  the named hazard — XLA lowering it through atomics with an irreproducible
  summation order — and on this hardware it did not materialise. That is a
  measurement on a V100 with this plugin, not a guarantee; §4a's rerun triggers
  are what keep it true;
* compile time is **3.7–14.3 s**, reported and never amortised.

The cross-job form of the determinism check — the identical *job* twice, which
catches a compilation that is not itself reproducible where one process cannot —
is not yet run. `phase0_compare.py`'s `across` column is what reads it, and it
is asserted only between records from the same platform (cuFFT and pocketfft sum
in different orders, so a GPU/CPU pair is expected to differ and does).

## Does the diagonalisation win on a GPU? (P10 / GPU.md Phase 1, 16 atoms)

Phase 0's 5.44x was on `al10-metal`, which has **ten k-points**, so it could not
say how much was k-parallelism and how much was the per-k path. These three
cells are **single k-point by construction**, so no k-parallelism exists to find
and every ratio below belongs to the diagonalisation path itself.

Run 2026-08-25, **Tesla V100-SXM2-32GB** (`dgx3`, pinned `--gpus=v100:1` so the
card generation is held fixed against Phase 0's) versus four pinned EPYC Milan
cores. `tools/gpu/phase1-si16.sbatch`. Per-stage numbers from
`tools/benchmark.py`, which already breaks `h_psi` and Davidson out separately.

**Per stage, `si16-1k-ecut30` — 16 atoms, 5900 plane waves, 32 bands, FFT
36x36x72 — at `band_batch = all`:**

| stage | GPU | CPU (4 cores) | ratio |
|---|---|---|---|
| `h_psi` (one k, all bands) | 0.015 s | 0.102 s | **6.8x** |
| `diagonalise` (one k, Davidson) | 0.386 s | 1.490 s | **3.9x** |
| `v_of_rho` | 0.039 s | 0.029 s | **0.74x** — the GPU *loses* |
| `symmetrize density` | 0.000 s | 0.002 s | — |

**So yes, the diagonalisation wins — but only with the bands batched, and the
band dial is worth more than the device.** On the same case and card, Davidson
costs **2.632 s at `band_batch = 1` and 0.386 s at `all`**, a factor of 6.8;
on the CPU the same change goes 1.224 → 1.490 s, i.e. *the wrong way*, which is
the cache argument `batching.py` was written around. A GPU run on the default
band dial does not merely fail to win, it loses outright — whole-SCF 0.20x on
this cell.

**Whole SCF, both dials, all three cells:**

| case | `b=1` | `b=all` |
|---|---|---|
| `si16-1k-ecut30` (5900 PW) | 0.20x | **1.80x** |
| `si16-1k` (1476 PW) | 0.70x | **2.67x** |
| `si8-1k-ecut30` (2950 PW) | 0.40x | **2.35x** |

**And the whole-SCF ratio is much worse than the stage ratios, for a reason
worth having.** `benchmark.py` converges to `conv_thr = 1e-8` and reports
**0.036 s/iteration** on the GPU; `phase0.py` runs the input's own **1e-10** and
reports **0.130**. The same case, the same dials, the same process warmth — only
the threshold differs. Splitting it:

| | first 7 iterations | the last 2, to 1e-10 |
|---|---|---|
| **GPU** | 36 ms each | **459 ms each — 13x** |
| **CPU** | 233 ms each | 237 ms each — **1.0x, flat** |

**Tightening `ethr` is nearly free on a CPU and costs 13x per iteration on a
GPU.** That is the accelerator pathology `GPU.md` Phase 1 named in advance and
the one this section was meant to find: QE's adaptive `ethr` schedule tightens
as `dr2` falls, a tighter threshold means more Davidson steps, and each step is
**small dense linear algebra — an `nvecx x nvecx` `eigh` and its matmuls —
inside a `lax.while_loop`**. `batching.py` already measures that algebra at
about a third of a Davidson step on CPU. On a GPU it is what the endgame of
every SCF is made of, and it does not vectorise: the FFTs got 6.8x and the
subspace solve got nothing.

The consequence for the headline number is direct: **at a production
`conv_thr` the 16-atom cell keeps 1.80x, not the 6.5x its loose-threshold
iterations suggest.** Quoting a GPU speedup without saying which `conv_thr`
produced it would be off by 3.6x on this cell.

### Thirty-two atoms, and the advantage grows

Run 2026-08-25 on the same V100-SXM2-32GB and the same four Milan cores.
`si32-1k-ecut30` is **32 atoms, 11781 plane waves, 64 bands, FFT 36x36x144** —
the largest case in `benchmarks/`, added for this measurement and verified by
the folding identity in its own header (si32 at Gamma reproduces si16 sampled at
{Gamma, (0,0,1/2)} to 8.9e-16 Ry/atom).

| stage, `band_batch = all` | GPU | CPU | ratio | at 16 atoms |
|---|---|---|---|---|
| `h_psi` | 0.019 s | 0.340 s | **17.9x** | 6.8x |
| `diagonalise` (Davidson) | 0.554 s | 7.457 s | **13.5x** | 3.9x |
| `v_of_rho` | 0.037 s | 0.047 s | **1.3x** | 0.74x |
| whole SCF, `conv_thr = 1e-10` | 82 ms/it | 1109 ms/it | **13.5x** | 1.80x |

**The diagonalisation's advantage grows sharply with the cell** — Davidson goes
3.9x to 13.5x for one doubling — and `v_of_rho`, which *lost* at 16 atoms, wins
at 32. Both point the same way: what a GPU needs is enough work per kernel, and
16 atoms was not yet enough.

**The `ethr` endgame penalty largely disappears here, and that revises the
paragraph above.** At 16 atoms the ratio collapsed from 6.5x at `conv_thr = 1e-8`
to 1.80x at 1e-10 — a 3.6x tax. At 32 atoms it is 16.5x to 13.5x, a 1.2x tax.
The small dense subspace algebra has not got any faster; it is simply a much
smaller share of a much larger cell's work. So "a GPU speedup has to say which
`conv_thr` produced it" is true at 16 atoms and nearly irrelevant at 32.

**Which means the whole-SCF ladder is not monotonic: 2.35x (8 atoms), 1.80x
(16), 13.5x (32).** That is not measurement noise. The number of expensive
tight-`ethr` iterations varies case by case and `si16-1k-ecut30` happened to hit
it hard — its last two iterations cost 459 ms each against 36 for the first
seven. **The per-stage ratios are the trustworthy measurement and they scale
cleanly; the whole-SCF ratio inherits the SCF's own path.** Quote the stages.

**And the band dial's cost at this size is no longer a ratio but a wall.** The
GPU run at `band_batch = 1` on `si32-1k-ecut30` **did not complete two SCF runs
in the ~15 minutes it had**, where the CPU does one in 11 s (1380 ms/iteration,
8 iterations) — the job timed out with that section's header printed and nothing
under it. So the 0.20x measured at 16 atoms understates what the default costs
at 32; the precise figure is unmeasured and wants a longer job. Everything else
in this ladder says the same thing more mildly: **the band dial is not a tuning
knob on a GPU, it is a correctness-of-configuration matter.**

**The 2.6e-09 anomaly did not grow, which was the thing to check.** On twice the
cell the same configuration — `si32-1k-ecut30`, `band_batch = all` — agrees with
the CPU to **-7.4e-13 Ry**, back at the round-off floor every other case sits on.
So the 16-atom outlier was particular to that case rather than a scaling defect
in the batched dial, and the dial's "moves the answer by round-off" claim
survives the largest cell here.

**One correctness flag, and it is not round-off in the sense the other rows
are.** On `si16-1k-ecut30` at `b=all` the GPU converged to **-126.720760703578
Ry** where CPU `b=1`, CPU `b=all` and GPU `b=1` all give **-126.720760700971**
— agreeing with each other to every printed digit, across platforms. The odd
one out differs by **2.6e-09 Ry** (2e-11 relative), and its `dr2` at
convergence differs too (2.525e-12 against 2.161e-12). Every other case in both
jobs agrees to 1e-13 or better, so this is the one configuration where the band
dial's "moves the answer by round-off" claim is visibly strained. It is far
below any physical tolerance and far below this project's own QE agreements
(~1e-9 to 1e-8 Ry), so it changes no conclusion — but it is the largest
dial-induced difference measured anywhere here, it is GPU-only, and it is on the
largest cell, which is the direction that matters. It should be re-measured on a
bigger cell before the batched dial is called answer-preserving on a GPU.

## Sixty-four atoms: defumat on a GPU against Quantum ESPRESSO on CPUs

**Read the baseline column before the ratio.** `GPU.md` §2.3 rules this
comparison out by default and the reason is not pedantry: the project's metric
is single-core against single-core so that a ratio measures *code*, and a GPU
number against a CPU number measures code *and* hardware at once and cannot be
decomposed afterwards. It is run here because it was asked for, with `pw.x` at
**four core counts** and defumat-on-CPU included as the middle corner, so that
the reader can take whichever comparison they actually mean.

Run 2026-08-25/26. **defumat on one NVIDIA H200**, jax 0.11.1; **Quantum
ESPRESSO 7.2** (the cluster module — the 7.5 vendored in this repo is gitignored
and not on the cluster) on **AMD EPYC Milan** cores, one node, `disk_io='none'`
so neither code is timed doing I/O the other skips. `benchmarks/si64-1k*.in`:
64 atoms, 256 electrons, 128 bands.

### The clean case: `si64-1k`, everything converged to `conv_thr = 1e-10`

Both codes agree on the answer first — **QE -505.71932000 Ry, defumat
-505.71932002 Ry**, 2e-8 apart — which is what makes the timings comparable at
all.

| code | hardware | iterations | ms/iteration | whole SCF |
|---|---|---|---|---|
| **defumat** | **H200, `band_batch=all`** | 8 | **39** | **0.31 s** |
| defumat | H200, `band_batch=1` | 8 | 342 | 2.74 s |
| defumat | 4 Milan cores | 8 | 1975 | 15.8 s |
| QE 7.2 | 1 core | 16 | 1352 | 21.6 s |
| QE 7.2 | 4 cores | 16 | 531 | 8.50 s |
| QE 7.2 | 16 cores | 17 | 223 | 3.79 s |
| QE 7.2 | 32 cores | 17 | 214 | 3.64 s |

**So the number depends entirely on which QE you mean**: **34.7x** per iteration
against one core, **5.5x** against thirty-two. The honest one-line answer is the
second: one H200 is worth about **5.5x a 32-core Milan node** on this cell,
per iteration.

**Whole-SCF is a different ratio again — 70x and 11.7x — and the gap is not the
GPU.** defumat converges in **8** iterations where QE takes **16-17**, which is
a difference of starting guess and mixing, not of hardware, and it would show
identically on a CPU. Time-to-answer is what a user feels, so it is quoted; it
just must not be filed as a GPU speedup.

**QE's own scaling caps out**: 1 → 32 cores buys 6.3x from 32 cores, and 16 → 32
buys 4%. A single k-point leaves QE only plane-wave and FFT parallelism, and it
saturates. That is the real reason the GPU comparison looks the way it does.

**And defumat is still the slower code per core** — 1975 ms/iteration on four
cores against QE's 531, i.e. **3.7x slower**, consistent with the ~3.3x this
file records elsewhere. The accelerator is doing the work, not the
implementation.

### The case that did not converge, and the one line that was wrong

**`si64-1k-ecut30` reached `conv_thr = 1e-8` on the GPU and gave the right
answer; asked for `1e-10` it ran 100 iterations and returned `NaN`.** Four CPU
cores converged the identical input in nine. **Fixed** — the table above is the
run before the fix and is kept because the diagnosis is the useful part.

| | iterations | result |
|---|---|---|
| GPU, `conv_thr = 1e-8` | 7 | -507.16606166 Ry — matches QE |
| GPU, `conv_thr = 1e-10`, before | 100 (max) | **NaN** |
| GPU, `conv_thr = 1e-10`, after | 9 | **-507.16606180 Ry**, 1.6e-8 from QE |
| CPU 4 cores, `conv_thr = 1e-10` | 9 | -507.16606166 Ry |

**The cause was not the GPU.** As Davidson's subspace fills it expands with
normalised residuals of roots that have *already converged* — amplified
round-off — which go linearly dependent, so the overlap's smallest eigenvalue
lands on the round-off floor and **its sign is arbitrary**. Measured on the
device at the ninetieth `generalised_eigh` call: `min eig(S) = -4.3e-16` against
`max|S| = 1.0`. `jnp.linalg.cholesky` of a matrix with a tiny negative
eigenvalue takes the square root of a negative pivot and **returns `NaN` rather
than raising**, and it travelled into the wavefunctions, the density, the mixer
and the total energy with nothing reporting a problem — `S` arrived non-finite
452 times afterwards. QE's `cdiaghg` hits the same wall and *stops* with
"problems computing cholesky"; returning `NaN` is strictly worse than stopping.
The CPU landed on the positive side of the same coin flip.

Cholesky remains the fast path and is taken **bit-for-bit** whenever it works;
canonical orthogonalisation runs only when the factor comes back non-finite, and
parks the round-off-floor directions above the spectrum rather than inverting
them. The GPU now matches the CPU's iteration count exactly, which is the sign
the SCF path is the same on both platforms rather than diverging at the endgame.

**Where that guard lives moved on 2026-08-26 and the mechanism above is no
longer where to read it** — see "The guard was inside the k batch, where a
`cond` is not a branch". It was a `lax.cond` inside the per-k solve, which under
`k_batch=None` is inside a `vmap`, where a batched predicate runs **both**
branches on every step. The test is the same test and the values are bit-for-bit
the same; the choice is now made once over the whole k-set, outside `map_k`.

**Two wrong turns, recorded because both were nearly committed.** The Anderson
mixer was the first suspect — its Gram matrix spans the square of the residual
magnitudes and reaches cond 1.1e11 by the eighth iteration. That is a real defect
and is fixed too (writing `c_i = e_i/|r_i|` gives *identical* coefficients at
cond 2.7e4), but it is **not** the `NaN`: the mixer *receives* a non-finite
density, and a synthetic history at cond 1e53 still solves to `max|c| = 1` under
partial pivoting. And the first regression test written for the fix passed on the
*unfixed* code, because it built a rank-deficient Gram matrix — only
semidefinite, smallest eigenvalue `+7e-16` — where Cholesky survives and leaves
its `NaN` in the unused triangle. **A test that passes before the fix is not a
regression test**; it builds a genuinely indefinite overlap now and carries the
old routine verbatim to assert that it fails.

**The stage timings for that cell** (from the converging run): `h_psi` 0.012 s,
Davidson 0.640 s, `v_of_rho` 0.019 s.

## The case that did not converge, which matters more than any of the above

**`si64-1k-ecut30` on the GPU reaches `conv_thr = 1e-8` and gives the right
answer; asked for `1e-10` it runs 100 iterations and returns `NaN`.**

| | iterations | result |
|---|---|---|
| GPU, `conv_thr = 1e-8` | 7 | **-507.16606166 Ry** — matches QE's -507.16606164 |
| GPU, `conv_thr = 1e-10` | 100 (max) | **NaN, not converged** |
| CPU 4 cores, `conv_thr = 1e-10` | 9 | -507.16606166 Ry, `dr2` = 5.2e-12 |

Same cell, same card, same code, same dials — **only the threshold differs**, and
the CPU reaches 1e-10 on the identical input in nine iterations. So this is not
a hard case, it is a **GPU-specific numerical robustness failure at the largest
cell measured here**, and it is new at 64 atoms: `si32-1k-ecut30` converged to
1e-10 on a GPU in eight iterations. It is deterministic — the twice-run check
reproduced the same `NaN` bit for bit — and it is not memory (3.28 GB peak of
141 GB).

**This is the first thing on the GPU backlog and it outranks every speedup in
this section.** A code that is fast and silently stops converging at a
production threshold is worse than a slow one. The diagnosis is not done; what
is known is that it appears between 32 and 64 atoms, at the higher cutoff, and
only on the device.

**The stage timings for that cell are still good** (they come from the
converging 1e-8 run): `h_psi` **0.012 s**, Davidson **0.640 s**, `v_of_rho`
**0.019 s**, whole SCF **83 ms/iteration** — against QE's 378 ms/iteration on 32
cores at the tighter threshold, which is *not* a matched comparison and is
recorded here only so the next session does not have to re-run it.

## The physics sweep on a GPU, against serial QE (P10 / GPU.md)

**The baseline is one CPU core and the table says so in every row.** `GPU.md`
§2.3 rules this comparison out by default — the project's metric is single-core
against single-core so that a ratio measures *code*, and a GPU number against a
serial CPU number measures code *and* hardware at once. It is here because it
was asked for. Two bounds travel with it: **one core is the softest baseline**
(`pw.x` on a single-k cell has only plane-wave parallelism and saturates by ~16
cores, where the same silicon comparison falls to about 1x), and **the iteration
counts differ in both directions** — fewer than QE on silicon, five times more
on `ni10-ldau` — which is mixing and starting guess, not hardware.

Everything before this measured *one* kind of calculation, unpolarised silicon,
at several sizes. That is a claim about `h_psi` on a norm-conserving insulator
and nothing else. This is the other axis. Run 2026-08-26 at commit `c5dc7d4`:
**one NVIDIA H200** (jax 0.11.1, `k_batch=all`, `band_batch=all`) against
**Quantum ESPRESSO 7.2 on one core**, every case to `conv_thr = 1e-10`.
Typeset with the figure in `performance/gpu-sweep.tex`; raw numbers in
`performance/gpu-sweep.json`.

| case | physics | at. | k | QE it | QE ms/it | GPU it | GPU ms/it | per-it | ΔE (Ry) | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| `si10-nc` | norm-conserving | 10 | 7 | 14 | 98 | 8 | 30 | **3.3x** | -3.1e-09 | 0.18 |
| `si10-nc-pbe` | GGA (PBE) | 10 | 7 | 16 | 221 | 8 | 44 | **5.0x** | -1.7e-09 | 0.39 |
| `si10-paw` | PAW | 10 | 7 | 14 | 644 | 8 | 57 | **11.3x** | -3.5e-09 | 0.66 |
| `si10-us` | ultrasoft | 10 | 7 | 15 | 595 | 8 | 43 | **13.8x** | -1.1e-09 | 0.65 |
| `al10-metal` | metal, smearing | 10 | 10 | 47 | 266 | 10 | 127 | **2.1x** | -1.9e-09 | 0.40 |
| `h10-chain-lsda` | collinear magnetic | 10 | 2 | 12 | 1219 | 17 | 130 | **9.4x** | -4.3e-09 | 0.55 |
| `h10-chain-noncolin` | noncollinear | 10 | 2 | 13 | 2942 | 17 | 161 | **18.2x** | -4.4e-09 | 1.37 |
| `ni10-ldau` | DFT+U, magnetic | 10 | 10 | 29 | 3191 | 151 | 321 | **9.9x** | +2.4e-09 | 1.60 |
| `h20-chain-lsda` | collinear magnetic | 20 | 1 | 15 | 2991 | 33 | 140 | **21.4x** | +3.7e-09 | 1.10 |
| `h40-chain-lsda` | collinear magnetic | 40 | 1 | 22 | 12669 | 104 | 414 | **30.6x** | +4.6e-09 | 4.06 |
| `bi10-soc` | spin-orbit | 10 | 1 | 14 | 21686 | 17 | 260 | **83.3x** | -1.9e-04 † | 16.92 |
| `bi20-soc` | spin-orbit | 20 | 1 | 20 | 105350 | 20 | 912 | **115.5x** | -3.7e-04 † | 34.70 |

† **The bismuth rows are a pre-existing defumat/QE difference, not a GPU one.**
`PLAN.md` records exactly 1.9e-4 Ry for `bi10-soc` as "the one that does not
close", and QE 7.5 and 7.2 agree with each other to 1e-8 on that case, so it is
not a version effect either. The GPU reproduces it.

**Every non-spin-orbit case agrees to 4.6e-09 Ry or better, and those figures
reproduce the *CPU* agreements already in `PLAN.md`** — 1.9e-9 on `al10-metal`,
4.3e-9 on `h10-chain-lsda`, 4.4e-9 on `h10-chain-noncolin`, 1.8e-8 on
`ni10-ldau`. That is the finding: **the GPU reproduces defumat's behaviour
including its known imperfections, rather than having a numerical character of
its own.** A further check that costs nothing: `h10-chain-lsda` and
`h10-chain-noncolin` return the same total to all eight digits (-9.56782281), so
the noncollinear path reproduces the collinear one on the device.

**The speedup spans two decades and tracks work per k-point**, which is what a
GPU needs to fill: 2.1x on `al10-metal`, where a single core is already
efficient; 3-5x on cheap norm-conserving silicon; 9-21x once there is an
augmentation charge, a GGA or a magnetization; and **83-115x on spin-orbit**,
where QE serial spends 22-105 **seconds per iteration** on two-component spinors
with a fully-relativistic ultrasoft dataset and the GPU spends 0.26-0.91.

**Memory is the constraint that bites, and spin-orbit is where it bites.**
`bi20-soc` peaked at **34.7 GB**. It fits a 141 GB H200 with room to spare and
**would not fit a 32 GB V100** — the sort of fact that decides whether a
calculation exists rather than how fast it is. Everything else in the sweep sits
under 2 GB.

**An iteration cap is not a convergence failure**, and this sweep's first run
reported one as the other. `ni10-ldau` and `h40-chain-lsda` came back "not
converged in 100" — which is `run_scf`'s default, where
`tests/regression/test_ten_site.py` has always used **200** for exactly those
cases. `ni10-ldau` reproduces the same non-convergence on a CPU at 100, which is
what settled it; both converge at 200, in 151 and 104 iterations.
`tools/gpu/phase0.py` takes `--max-iterations` now so the driver cannot report
the one as the other again.

## What each response property's working set is (P10 / GPU.md Phase 5)

**`GPU.md` §4 item 3: "the number that says whether the response path fits on a
card at all, and it needs no card to obtain."** Everything measured on a GPU so
far is the ground state, and `CLAUDE.md`'s "Why JAX" lists autodiff response as
reason *one*. This is the CPU half of that phase, run 2026-08-26 on the
development workstation with `tools/gpu/phase5.py` — one converged SCF and
**one** response property per process, because peak RSS is a high-water mark
that cannot be reset and two properties in one process report the larger of the
two twice.

Peak RSS, the same measurement "What a stress costs (P11)" is built from, so
these rows extend that table rather than starting a new scale:

| property | mode | case | cell | property time | peak RSS | over the SCF | value |
|---|---|---|---|---|---|---|---|
| force | reverse | `si2-nc-stress` | 2 at, 18 k, `npwx` 200, `ngm` 1459 | 0.01 s warm | 0.71 GB | **+0.00** | max F 0.0604 Ry/bohr |
| stress | reverse | `si2-nc-stress` | the same | 0.14 s warm | 0.79 GB | **+0.03** | P = -23.58 kbar |
| dielectric | forward | `si-epsilon` | 2 at, 10 k, `npwx` 350, `ngm` 2733 | 13.3 s warm | 1.34 GB | **+0.79** | eps 13.806646 |
| Born charges | forward, `jvp` of the force | `si-epsilon` | the same | 16.3 s warm | 1.42 GB | **+0.96** | Z* -0.0757150 |
| dynamical matrix | forward-over-reverse | `si-epsilon` | the same | 42.1 s cold | 1.52 GB | **+1.06** | 510.1023 cm-1 |
| dielectric | forward | `si-epsilon-us` | 2 at, 10 k, `npwx` 407, `ngm` **9185** | 39.8 s cold | 1.50 GB | **+0.66** | eps 14.3253 |
| Raman tensors | forward-over-reverse | `alas-raman` | 2 at, **64 k**, `npwx` 169, `ngm` 1243 | 151.1 s cold | 2.59 GB | **+2.01** | eps 12.9674 |
| stress | reverse | `si8-us` | 8 at, 1 k, `npwx` 1607, `ngm` **36257** | 16.2 s cold | **10.49 GB** | **+9.53** | P = 3.87 kbar |

**Every value reproduces the number already committed for it** — 13.806646
against P24's 13.806646, -0.0757150 against P24b's every printed digit, 510.1023
against P25's 510.102 — which is what says the driver is measuring the same
calculation the tests validate and not a cheaper one.

**The mode decides the tape, not the property, and that is the finding.**
`GPU.md` predicted that "the forward-mode paths that P25, P26 and P35 are built
from are much better behaved" but explicitly declined to assert it. Measured: a
forward response costs **0.7-1.0 GB** over its own SCF, and a
**forward-over-reverse** one — a `jvp` of a gradient, which tapes the inner
reverse pass and could have behaved like the other end — costs **1.0-2.1 GB**,
the same order rather than the other one. The dynamical matrix and the Raman
tensors are not memory problems.

**And the spread inside that range is the property, not the k-count**, which is
worth stating because the two cases in the table above invite the opposite
reading: `alas-raman` has 64 k-points where `si-epsilon` has 10, so 1.06 against
2.01 GB looks like it scales with `nk`. It does not. The same third derivative on
**eighteen** k-points (`si2-nc-stress`, a control run for this) costs **+2.13 GB**
— more than the sixty-four-point cell — so what separates a dynamical matrix from
a Raman tensor here is how many first-order responses each solves, and the
k-count is not the term that matters at these sizes.
What is a memory problem is the *reverse* stress through the radial transforms:
**10.49 GB on eight-atom ultrasoft silicon, +9.53 over its own SCF** — an
independent reproduction, by a different driver, of the 11.1 GB the P11 section
records — which is **ten times the largest response tape here** and is what a
card has to hold. It has a fix in the backlog (item 8, a `custom_jvp` on each
radial transform, ~100x on the dominant term) rather than a guess.

**The batching dial does not bound the response's k-dependent state**, and that
is structural rather than measured: `SternheimerResult.dpsi` is
`(nspin, nk, nbnd_occ, ndim)` and is stacked over the whole axis
(`response/sternheimer.py:477`), because it is what the next iteration's `drho`
is built from — so `map_k` chunks the *work* and not that state. Every row above
was measured at the CPU default of one k-point at a time, and **the measurement
did not show that term dominating**: the two Raman cases put 18 and 64 k-points
within 6% of each other. On a card at `k_batch = None` the SCF's own set grows as
well, which is §2.4's memory-versus-parallelism trade arriving in the response
path exactly as it does in the ground state — and it is unmeasured here.

**The ultrasoft row is the one worth reading twice.** `si-epsilon-us` has
**3.4x the `ngm`** of the norm-conserving cell and its response costs *less*
memory over the SCF (+0.66 against +0.79), because the augmentation charge is
paid for in the SCF's own working set rather than in the tape: a forward
response does not tape the setup at all. That is the same asymmetry from the
other side.

**Two caveats on the timings, which are not this phase's headline.** The
`vs SCF iteration` ratio `phase5.py` prints divides by an SCF that carries its
own compilation, so it is a lower bound; and a response solve is 35-220 SCF
iterations here, which is P24's and P25's own measurement (the Sternheimer stage
is 96% of a phonon) and is what backlog item 5 — scheduling the response
threshold, `av.it. 27.7` against `ph.x`'s 9.3 — is for. That item is
platform-independent and pays before any hardware does.

**The GPU half is written and unrun**: `tools/gpu/phase5-gpu.sbatch` and its CPU
pair run the same driver over the same eight rows, with `si8-us stress` last
because 11 GB of tape in HBM is the row the phase exists to bound. Per
`CLAUDE.local.md`, an `sbatch` is proposed rather than submitted.

## The response path on a GPU (P10 / GPU.md Phase 5, the GPU half)

Run 2026-08-26 at commit `e562427`. **One NVIDIA H200** (`gpu63`, 143771 MiB,
driver 580.173.02, jax 0.11.1) against **four EPYC Milan cores** (`milan1`, same
jax, same commit) — §2.3's metric, GPU defumat against CPU defumat on the same
input and the same code, with the CPU side pinned to a stated core count. Both
jobs ran `tools/gpu/phase5.py` twice per property, so every row carries its own
determinism check. Warm times, the cold ones beside them:

| property | mode | GPU cold/warm | CPU cold/warm | warm ratio | device peak | GPU vs CPU value |
|---|---|---|---|---|---|---|
| force `si2-nc-stress` | reverse | 1.67 / **0.004** s | 1.15 / 0.01 s | 1.8x | 0.11 GB | 4.6e-12 |
| stress `si2-nc-stress` | reverse | 0.36 / **0.004** s | 0.61 / 0.09 s | ~24x † | 0.08 GB | 2.1e-11 |
| dielectric `si-epsilon` | forward | 24.97 / 12.40 s | 19.73 / 12.48 s | **1.0x** | 0.13 GB | 5.4e-12 |
| Born `si-epsilon` | forward | 20.53 / 10.81 s | 18.28 / 13.04 s | **1.2x** | 0.12 GB | 2.0e-10 |
| dielectric `si-epsilon-us` | forward | 28.72 / 17.14 s | 31.50 / 23.67 s | **1.4x** | 0.21 GB | 4.6e-12 |
| dynamical matrix `si-epsilon` | fwd-over-rev | 34.56 / 28.01 s | 36.81 / 28.88 s | **1.0x** | 0.15 GB | 2.4e-12 |
| Raman `alas-raman` | fwd-over-rev | 123.50 / 100.01 s | **failed** | — | 0.25 GB | — |
| stress `si8-us` | reverse | 26.20 / **0.009** s | 10.71 / 3.14 s | **339x** | **2.73 GB** | 9.0e-12 |

† **The small stress's ratio divides 4 ms by 90 ms and is near timer resolution
on the GPU side** — it is quoted as an order of magnitude and nothing finer. The
339x below has a real denominator (3.14 s) and does not depend on it.

**The mode decides the speedup exactly as it decided the tape, and this is the
finding.** A **Sternheimer solve gets nothing from an H200** — 1.0x on the
dielectric constant, 1.0x on the dynamical matrix, 1.2-1.4x on the other two —
while a **reverse-mode gradient flies**: 24x on a small stress and **339x** on
eight-atom ultrasoft silicon. The two are different shapes and the ratios say so.
A projected CG over occupied bands is small dense algebra inside a device loop,
which is Phase 1's named pathology (`ethr`'s 13x tax) arriving in the response;
a gradient through the radial transforms is one enormous dense graph — `ngm` x
`kkbeta` is 36257 x ~1100 — which is what a card eats and a cache-bound CPU
chokes on.

**And 339x is the warm number, which nothing here actually pays.** A run calls
`stress()` **once**, as `run_pwscf` does, and the cold column is what that costs:
**26.20 s on the GPU against 10.71 on four CPU cores.** The device spends 26 s
compiling a graph it then runs in 9 ms. So for the calculation people actually do,
the GPU *loses* on this row by 2.4x, and it would take four stress evaluations of
the same cell to break even. The same reading applies to the response rows, where
compile is 6-12 s on either side. **Quote the cold column for a one-shot property
and the warm one only where something iterates** — a relaxation, a `q`-loop, a
finite-difference check.

**Memory: the card holds a quarter of what the host figure suggested.** The
reverse tape that costs **10.56 GB of host RSS** on the CPU node needs **2.73 GB
of HBM** on the device, and every response property sits **under 0.25 GB**. Against
a 141 GB card that is not a constraint at any point, which answers Phase 5's
feasibility question in the affirmative and by two orders of magnitude. Host RSS
on the GPU side is a flat 4.1-5.5 GB across every row — the CUDA context and the
setup, not the physics.

**The platform default works in production, and this job is its first end-to-end
test.** Neither job passes a dial. The GPU records resolve
`k_batch=None, band_batch=None` and the CPU records `1, 1` — same commit, same
driver, same script — which is what §1's change was for.

**Three rows are not bit-reproducible on the device, and none is on the CPU.**
`GPU.md` Phase 0 check 5 named this hazard — `basis/fft.py` scatters plane waves
into the box with an accumulating scatter over deliberately duplicated indices,
and XLA may lower that through atomics whose summation order is not reproducible
— and Phase 0 did not see it in the SCF. **It is here, in the response path**:

| row | run-to-run spread | relative |
|---|---|---|
| stress `si2-nc-stress` | 2.8e-14 kbar on -23.576594906741 | 1.2e-15 |
| dynamical matrix `si-epsilon` | 1.1e-13 cm-1 on 510.10233994501 | 2.2e-16 |
| stress `si8-us` | 1.2e-12 kbar on 3.870512510214 | 3.1e-13 |

That is round-off, not a defect: the phonon wobbles in its **fourteenth**
significant figure where the number is quoted against `ph.x` to 0.05 cm-1. But it
is a **property of the platform, not of the code** — the identical rows are
bit-identical on the CPU node — so "bit-identical run to run", which Phase 0 could
assert of an SCF, **cannot be asserted of a response on a device**. A GPU
regression set (§4a) has to compare a response to a tolerance and say which
tolerance, rather than diffing bytes.

**One row failed on the CPU node and it is unexplained.** `alas-raman raman` died
with a `MemoryError` raised inside the JAX computation, on a node where 160 G was
granted and the same job's largest sampled RSS was 11.7 GB (`si8-us`, which ran
after it and succeeded). **This workstation runs that exact case in 2.59 GB**, and
the GPU node — same jax 0.11.1, same commit — ran it in 5.45 GB of host RSS. So it
is not the case being too large, and the two cluster nodes differ in what they did
with it. Unexplained is the accurate word; the diagnostic is one more short job
with `JAX_TRACEBACK_FILTERING=off` and a memory profile, and it is **not** run.

**The standing caveat applies and is worth stating plainly**: the response cases
here are **two-atom cells**, and `CLAUDE.md`'s rule is that a two-atom cell shows
none of this. The SCF sweep needed ten atoms to move from 3x to 115x. So the
1.0-1.4x on the Sternheimer rows is a fact about *these* cells, not a ceiling for
the response path — the next measurement is `si10-epsilon` or larger, where the
per-k work is real.

## The dials default to the platform now (P10 / GPU.md §1)

**Every GPU number above was produced by a dial set by hand in an sbatch
script.** That is the finding this entry closes: `batching.py` defaulted both
dials to `1` on every platform, so a bare `run_scf` on a card inherited the
cache-shaped end of both — measured at **4.5x** on `al10-metal`, at **0.20x**
on `si16-1k-ecut30` (a loss against four CPU cores), and at 32 atoms at a wall,
where `band_batch = 1` did not finish two SCF runs in the fifteen minutes four
cores need eleven seconds for. Nothing in the code said so; the knowledge lived
in `GPU.md` and in the job scripts.

`_platform_default()` now answers for both dials at once — `1` on a CPU, `None`
on anything else — under `GPU.md` §5's rule that a platform-dependent choice is
"a dial with a per-platform default and both settings tested". What did *not*
change is the reason to trust it:

* **the CPU default is bit-identical to what it was**, so no validated number
  moves. The whole suite is the check;
* **the order of precedence is unchanged**: an explicit `k_batch`/`band_batch`
  beats `DEFUMAT_K_BATCH`/`DEFUMAT_BAND_BATCH`, which beat the platform;
* **the two axes move together**, because `k=all, b=1` was measured at 2075 ms
  against 177 and 801 — worse than either end, since batching k while looping
  bands multiplies the per-band launches by `nk`;
* **a single k-point is untouched** either way: `map_k`/`sum_k` short-circuit
  `nk == 1` before they look at the chunk size, so the benchmark cells — single
  k on purpose — do not acquire a width-one batch axis, which the module
  docstring measures at 37% of a Davidson solve.

Two implementation notes worth having: the platform is asked **lazily and
cached**, because asking initialises the backend and on a GPU node that
allocates device memory — importing `defumat.batching` must not do that — and
the two constants are module attributes (PEP 562) rather than values frozen at
import, which also fixes a smaller bug of its own: `DEFUMAT_K_BATCH` set after
the import used to be ignored. `tests/unit/test_batching.py` substitutes the
backend, so the accelerator branch is tested on this CPU-only workstation.

**The response path inherits it for free**, which is the point of R6: everything
that walks the k axis goes through `map_k`/`sum_k`, so a Sternheimer solve on a
card gets the batched mode without a dial being passed down through
`dielectric_tensor` or `dynamical_matrix`.

## The guard was inside the k batch, where a `cond` is not a branch (P10 / GPU.md Phase 1)

**A `lax.cond` under `vmap` has no branch to take.** JAX's batching rule for a
conditional whose predicate is batched lowers it to `select_n` over the results
of *both* branches — there is no per-element branch on a device — and
`k_batch=None`, the default on an accelerator since "The dials default to the
platform now", is exactly a `vmap` over the k axis. So the guard the 64-atom
`NaN` fix put inside `generalised_eigh` has been computing canonical
orthogonalisation on **every step of every k-point of every multi-k run**, in
addition to the Cholesky route it actually uses.

Measured on the development workstation, four pinned cores, at `si10-nc`'s own
subspace shapes (80 x 80, seven k-points), by `tools/gpu/davidson_profile.py`:

| | unbatched | batched over 7 k |
|---|---|---|
| `generalised_eigh` (guarded) | 3.683 ms | **42.546 ms** |
| `_cholesky_route` alone | 3.424 ms | **14.940 ms** |
| ratio | 1.08x — the guard is free | **2.85x** |

Unbatched the two are within the spread of each other, which is what a real
branch looks like. Batched they are not, and the factor is **2.85x** — a second
measurement in a separate process put it at 3.26x, so quote it as "about three".
**It is a lowering fact rather than a hardware one**, which is why a CPU can
measure it at all: the same `vmap` costs the same extra work on either platform,
in a different ratio.

**The fix is where the predicate is a scalar, not what the predicate is.**
`davidson_eigensolver_all` now takes the Cholesky route unconditionally inside
`map_k` — no conditional in the batched graph at all — and asks once, over the
whole k-set and outside the batch, whether the eigenvalues *and* the
eigenvectors came back finite; if they did not, every k-point is re-solved with
canonical orthogonalisation. Three things make that safe rather than merely
faster:

* **the values are bit-for-bit unchanged**, because the guard passing *was* the
  Cholesky route: `select_n` chose between two computed arrays and took that
  one. Verified end to end — `si10-nc` at `k=all, b=all` gives
  `-78.997489763108 Ry` in 8 iterations with eigenvalue SHA-256 `98dc1c570b6c…`
  before and after, identical;
* **the per-k loop exits the moment the eigenvalues stop being finite**, so a
  solve that has failed costs a few steps rather than the whole 100-step budget
  before the retry outside can see it. Without that the retry would be worse
  than the `cond` in exactly the case that was already broken;
* **the guard watches the wavefunctions too, not only the energies.** The first
  regression test written for the 64-atom `NaN` passed on the *unfixed* code
  because Cholesky left its `NaN` in a triangle nothing read — a guard that
  watches only the eigenvalues can pass a wavefunction with a `NaN` in it
  straight into the density.

**What it is worth, on a CPU, which is not where it matters.** Same machine,
four pinned cores, `si10-nc` at `k=all, b=all`, the Davidson stage alone:

| | before | after | |
|---|---|---|---|
| cold-start solve, `ethr = 1e-13` | 6820.5 ms | 5852.4 ms | **1.17x** |
| seeded solve, `ethr = 1e-13` | 869.6 ms | 761.5 ms | **1.14x** |
| seeded solve, `ethr = 1e-10` | 453.8 ms | 413.3 ms | **1.10x** |
| whole SCF | 1138 ms/it | 1037 ms/it | **1.10x** |
| compile, cold-start solve | 3.69 s | 4.79 s | **+1.10 s** |

The 2.85x on the operation becomes 1.1-1.17x on the stage because on a CPU
`h_psi` is most of a Davidson step — 146 ms for a 32-band block against 19 ms
for the whole 128-wide subspace solve, on `si16-1k-ecut30`. **On a device that
ratio inverts**, and by a lot: Phase 1 measured `h_psi` at 6.8-17.9x faster on a
GPU while the subspace algebra took nothing, and the committed 64-atom stage
timings are `h_psi` 0.012 s against Davidson 0.640 s. That is a *prediction*
this change is expected to cash in, not a measurement — `tools/gpu/davidson-gpu.sbatch`
is the job that tests it, and it re-runs the physics sweep's own invocation so
that the sweep's committed ms/iteration is the before column on the same card.

**And the compile cost is real and is not amortised away here**: the retry
branch is a second copy of the solver in the graph, +1.10 s on the cold-start
variant and +0.58 s on the seeded one. That is paid once per shape; the 2.85x
was paid per Davidson step.

## Inside a Davidson step: what each part costs (P10 / GPU.md Phase 1)

Phase 1 profiled the SCF by *stage* and named what it could not resolve there —
"the small dense `eigh` and matmuls inside a `lax.while_loop`". This is that
level, from `tools/gpu/davidson_profile.py`, on the development workstation at
four pinned cores. Two cells, chosen for what each can show: `si16-1k-ecut30`
is single-k and large-basis, `si10-nc` is seven k-points and small.

**`si16-1k-ecut30`** — 16 atoms, 1 k, `nbnd` 32, `ndim` 5900, `nvecx` 128:

| operation | width | ms |
|---|---|---|
| `h_psi` | 32 bands | **146.3** |
| `h_psi` | 1 band | 6.8 |
| the subspace solve | 128 | 18.8 |
| the subspace solve | 96 / 64 / 32 | 5.5 / 2.1 / 0.6 |
| a Ritz rotation | 128 | 11.7 |
| a Ritz rotation | 96 / 64 / 32 | 9.0 / 5.9 / 3.2 |
| `_extend_projection` | 32 | 11.3 |

**`si10-nc`** — 10 atoms, 7 k, `nbnd` 20, `ndim` 952, `nvecx` 80:

| operation | width | ms |
|---|---|---|
| `h_psi` | 20 bands | **18.6** |
| the subspace solve | 80 / 60 / 40 / 20 | 3.4 / 2.8 / 1.2 / 0.3 |
| a Ritz rotation | 80 / 60 / 40 / 20 | 1.1 / 1.1 / 0.8 / 0.6 |

Three things follow, and they are the ranking the next change is chosen from.

**`h_psi` is exactly linear in the block's width** — 1.08 / 4.98 / 9.63 / 18.58
ms at 1 / 5 / 10 / 20 bands — which is what a CPU walking its bands one at a
time looks like. Whether it is linear on a device at `band_batch = all`, where a
narrow block may be launch-bound, is the question `davidson-gpu.sbatch` asks.

**The solve at the live basis is a third of the solve at `nvecx`.** `cegterg`
sizes its `cdiaghg` and its ZGEMMs by `nbase`; static shapes made this code size
them by `nvecx = 4 nbnd` always, and the four widths above say what that costs:
over one fill of the subspace, 4 x 18.8 ms of solve against 0.6 + 2.1 + 5.5 +
18.8, and 4 x 11.7 of rotation against 3.2 + 5.9 + 9.0 + 11.7. About 7% of a
Davidson step here — and aimed at ~80% of one on the 64-atom cell, where the
committed device timings put `h_psi` at 0.012 s and Davidson at 0.640.
**`lax.switch` is `lax.cond`'s twin under `vmap`** and runs every branch, so
this can only ever be taken on the unbatched path — which is where the large
single-k cells live, so that is not the obstacle it sounds like.

**Roots settle, but only in the regime the SCF actually runs.** The number of
unconverged roots is reconstructed without instrumenting the `while_loop`, by
capping it at `m` steps for successive `m` and differencing the eigenvalues
outside — which is the solver's own test. At `ethr = 1e-13` on `si10-nc`:

| | step 2 | 3 | 4 | 5 |
|---|---|---|---|---|
| cold start | 20/20 | 20/20 | 20/20 | 20/20 |
| seeded from the converged states | 20/20 | 13/20 | 10/20 | **0/20** |

So expanding by `notcnv` rather than by `nbnd` — which is again `cegterg`'s
behaviour and again lost to static shapes — is worth about half the `h_psi` of
the endgame steps *in the SCF*, and nothing at all on a cold band-structure
solve. The cold column is why the premise had to be measured rather than
assumed: a profile of the wrong regime would have said the opposite.

**And the `ethr` tax is here in its CPU form**: the whole seeded solve costs
58.4 / 111.9 / 161.4 ms at `ethr` 1e-6 / 1e-10 / 1e-13, i.e. **2.77x**, where
the same tightening on a device cost 13x at sixteen atoms. Same shape, different
size, and the difference is the subspace algebra's share.

## The response at ten atoms, not two (P10 / GPU.md Phase 5)

`PERFORMANCE.md`'s own caveat on the first Phase 5 run was that every case in it
was a two-atom cell, "and `CLAUDE.md`'s rule is that a two-atom cell shows none
of this ... the next measurement is `si10-epsilon` or larger". This is the CPU
half of that measurement — development workstation, four pinned cores, jax
0.11.0, `k_batch=1, band_batch=1`, `tools/gpu/phase5.py`:

| property | mode | cold / warm | tape over SCF | value | against `ph.x` |
|---|---|---|---|---|---|
| dielectric `si10-epsilon` | forward | 256.3 / **245.3** s | +1.30 GB | `eps_xx` 19.111965069 | 19.111968489 → **3.4e-6** |
| Born `si10-epsilon` | forward | 298.3 / **302.0** s | +1.58 GB | `Z*_xx` -0.723689217 | -0.72368 → **9.2e-6** |

The cell is 10 atoms, 16 k-points, `nbnd` 20, `npwx` 952, `ngm` 7373; its SCF is
9 iterations at 1472 ms each. Three things worth carrying:

* **it is twentyfold the two-atom response** — 245 s against the committed 12.48
  s for `si-epsilon`'s dielectric on four Milan cores — so it is the first
  response case where the per-k work is real. Whether the H200's 1.0x on a
  Sternheimer solve survives that is what `tools/gpu/phase5-si10-gpu.sbatch`
  asks;
* **the tape does not scale with the response's size**: +1.30 and +1.58 GB
  against +0.79 and +0.85 on the two-atom cell, for twenty times the work. A
  forward response carries a tangent and tapes nothing, and this is that
  statement holding at a size that could have broken it;
* **both are bit-identical run to run on the CPU**, which is the comparison the
  device could not make on three of its eight rows.

The `born_xx` figure is the one component quoted; `tests/regression/test_ten_site.py`
already checks the whole tensor against this reference, and `PLAN.md` records
7.0e-6 across every component.

## The Davidson step count against `pw.x` on a 157-atom slab (2026-09-04)

**Open, not fixed, and the headline is the step count rather than any one
cause**: on the identical input at the identical `diago_david_ndim = 2`,
defumat's Davidson takes roughly **12x** more steps than `pw.x`. Three
candidates are below, ranked; the per-band threshold is the one that was
measured and it accounts for **1.47x** of the 12x, so the rest is elsewhere
and the leading suspect is the missing hard restart. Diagnosed by running FePc on SnTe(001) through `pw.x` and through
defumat on the *identical* input and comparing per-iteration timings. Recorded
here so that whoever takes it does not re-derive any of it.

**The case.** 157 atoms, `nelec = 1700`, `nbnd = 1020`, `K_POINTS gamma`,
`nspin = 2`, `ecutwfc 60` / `ecutrho 240`, `nosym`, `diago_david_ndim = 2`,
`mixing_mode local-TF`, `smearing mv`, `degauss 0.01`; `npwx` about 381k on the
gamma half-sphere, FFT 225x216x256. defumat on one H200 at `f2be49f`
(`DEFUMAT_BAND_BATCH 64`, pool 132.81 GiB); `pw.x` 7.2 on 128 Milan cores,
`-npool 1 -ndiag 64`.

**The symptom.** defumat's iteration 1 takes **62 s** and its iteration 2 takes
**2187 s** — 35x, at an `ethr` that only tightened 4.5x. Memory is not involved:
14.5 GB in use and 38.1 GB peak against a 132.81 GiB pool.

**Defect 1: there is no `btype` and no `empty_ethr`.** `cegterg.f90:559-561` (and
`regterg.f90:1108-1112`) test each band against **two** thresholds --
`ABS(ew - e) < ethr` for an occupied band and `< empty_ethr` for an empty one,
with `empty_ethr = MAX(ethr * 5, 1.D-5)` (`cegterg.f90:129`). `btype` is set in
`sum_band.f90:122-127` from the *previous* iteration's occupations, `0` wherever
`wg/wk < 0.01`, unless `diago_full_acc`. Neither name occurs in this package:
`davidson.py:372` compares every band against one scalar `ethr` and `:386` is
`jnp.all(settled)`, so a handful of stubborn bands keeps the whole block
expanding. About 170 of this cell's 1020 bands are empty, and they are the
slowest-converging part of an atomic start.

**Why iteration 2 and not iteration 1**, which is the part that identifies it:
`btype` comes from the previous iteration's weights, so it is all-ones during
iteration 1 in both codes and only bites from iteration 2. That is exactly where
the anomaly starts, and it is why `pw.x`'s iteration 2 gets **cheaper** than its
iteration 1 while defumat's gets 35x dearer.

**Defect 2: `H` is applied at full `nbnd` width.** `davidson.py:434-437` sorts the
unconverged roots to the front and *zeroes* the settled rows, and `:508` then
applies `hamiltonian.apply` to the whole `(nbnd, npwx)` block. The subspace grows
by `notcnv` (`:514`), which is `cegterg`'s behaviour and is right; the H
application never narrows, so zero rows are transformed at full price. **This is
the static-shape rule** (`davidson.py:20-24`) -- a `lax.while_loop` body has one
shape -- and therefore an intrinsic cost of keeping the solver on device rather
than an oversight. It is what the reference trace below shows `pw.x` not paying:
its cost does not track its step count, because its late steps touch a handful of
vectors.

**The reference trace, which is the acceptance test.** `pw.x` on the same input at
the same `diago_david_ndim = 2` (its "avg # of iterations" is `avg_iter/nkstot`
with `nkstot = 2` here, so per spin channel):

| iter | `ethr` | avg steps | seconds | accuracy (Ry) |
|---|---|---|---|---|
| 1 | 1.00e-02 | 15.5 | 340.4 | 24.79 |
| 2 | 1.46e-03 | 8.5 | 113.4 | 20.04 |
| 3 | 1.18e-03 | 2.0 | 64.9 | 13.55 |
| 4 | 7.97e-04 | 12.0 | 59.8 | 26.59 |
| 5 | 7.97e-04 | 2.5 | 47.6 | 22.75 |
| 6 | 7.97e-04 | 8.0 | 59.4 | 16.12 |
| 7 | 7.97e-04 | 2.0 | 46.0 | 2.85 |
| 8 | 1.68e-04 | 5.0 | 48.7 | 2.72 |
| 9 | 1.60e-04 | 3.0 | 48.9 | 7.79 |

After a fix, defumat's step count should land in that range rather than at
`MAX_ITERATIONS = 100` (`davidson.py:156`). **No new `pw.x` run is needed** -- the
table is the reference.

**The step counts are the comparable quantity and the energies are not.** That
trace is `pw.x` **7.2** (spack, gcc 12.3, openblas 0.3.24, fftw 3.3.10), where
everything else in this repository is validated against the vendored **7.5**.
Nothing in `cegterg`'s convergence structure moved between those releases, so the
iteration counts carry across; the total energies and `scf accuracy` values in the
table are 7.2's and are **not** a reference for a 7.5 build. This file exists in a
project where a 6.0 benchmark drifting in the sixth decimal is why
`tools/generate_reference.py` was written, so the caveat is stated rather than
left to be discovered.

**Instrument before fixing.** The loop counter already exists: `final[14]`, beside
`final[8]` and `final[10]` at `davidson.py:524`. Emit it per SCF iteration
(averaged over channels, as `pw.x` does) together with the number of unsettled
bands at exit. The prediction to falsify is ~100 for iteration 2 with the
unsettled bands being the empty ones.

**The fix for defect 1** is a transcription and is contained: thread a per-band
threshold *vector* into the solver in place of the scalar `ethr`, built from the
previous iteration's weights as `where(wg/wk < 0.01, max(5*ethr, 1e-5), ethr)` and
all-ones on iteration 1; `settled` becomes an elementwise comparison against it.
No shape changes and nothing leaves the `jit`. Expose `diago_full_acc` as the
off-switch, which QE has for the same reason.

**Defect 1 is confirmed and is worth about 1.5x, not 12x** -- measured, and the
number matters more than the confirmation. Running `pw.x` on the same input with
`diago_full_acc = .true.`, which is exactly the switch that forces `btype` to
all-ones, against the same run without it:

| iteration | `ethr` | with `empty_ethr` | without it |
|---|---|---|---|
| 1 | 1.00e-02 | 15.5 steps, accuracy 24.79143461 | 15.5 steps, accuracy 24.79143461 |
| 2 | 1.46e-03 | 8.5 steps | 12.5 steps |

Iteration 1 is bit-identical to eight decimals, which is the control saying the
two runs differ only in that switch. Iteration 2 goes 8.5 -> 12.5, **a factor of
1.47** -- the right sign, in the right place, and an order of magnitude short of
the ~12x by which defumat's inferred ~100 steps exceed `pw.x`'s 8.5.

**So the diagnosis is right and incomplete, and whoever implements it should
expect ~1.5x.** Measuring 1.5x is confirmation, not refutation; do not conclude
from it that the per-band threshold was the wrong lead.

**Where the rest most likely lives** -- for the backlog, not as a claim, and both
testable against the trace above with no new `pw.x` run:

* **There is no hard restart.** `c_bands.f90` re-enters `cegterg` up to **5
  times** at `maxter = 20`, each re-entry restarting Davidson from the current
  `evc` with a fresh subspace -- which is why `CLAUDE.md` says QE's real budget is
  100 steps. defumat's `MAX_ITERATIONS = 100` is one *continuous* loop, and the
  collapse-to-Ritz at `nvecx` is **not** the same thing: it keeps the iteration
  history and the preconditioner state where a re-entry discards them. A periodic
  hard restart is the standard rescue for a stalled Davidson.
* **The preconditioner**, compared directly against `g_psi.f90`'s form. A weak
  diagonal preconditioner shows up as exactly this signature -- many cheap steps
  rather than few expensive ones.

**A backlog idea for defect 2, speculative and unmeasured** -- do not bundle it
with the above: apply `H` on a fixed-size *leading* slice, `nbnd`, `nbnd/2` or
`nbnd/4` chosen by a `lax.switch` on `notcnv`, which keeps a small number of
static shapes instead of one. The unconverged roots are already sorted to the
front, so the slice is contiguous.

**Ruled out, and listed so they are not re-chased.** Recompilation on iteration 2
(one `lax.while_loop` in one `jit`; every shape is fixed by the basis, and neither
`ethr` nor the trip count is a shape). Allocator thrash (a preallocated BFC pool
aborts when over budget, it does not degrade to slowness). `diago_david_ndim = 2`
being pessimal -- it collapses the subspace every step, which is real, but `pw.x`
at the *same* setting does iteration 2 in 8.5 steps at a *tighter* `ethr`, so
raising it would mask the defect rather than explain it. And the H200 being slow
per unit of work: defumat is at `>= 10.9` s/step against `pw.x`'s ~6.7, about
1.6x, but defumat's step does more work (defect 2), so once that is normalised the
GPU is probably not behind -- **do not go hunting a slow FFT or GEMM until
work-per-step is normalised**. `5779eae` and `5b9eb0f` are memory only (84.3 ->
73.3 GiB at `david = 2`, `band_batch = 64`), not time.

## Optimisation backlog

Ordered by expected gain per unit of effort, and by measurement rather than
instinct. None of these may change a validated number.

1. **`jax.sharding` over the k-axis**, and GPU. Now measured to be the *only*
   parallelism worth having on CPU: the thread pool gives 15% between one core
   and four and loses badly beyond that, while `metal.in`'s ten k-points are
   independent and are currently run through a single pool. This is where the
   remaining structural factor is, and the k-axis already leads every
   wavefunction-shaped array so that it can be taken.
2. **Size the Davidson subspace solve by the live basis**, and the two Ritz
   rotations with it (`cegterg`'s `nbase`, lost here to static shapes). Measured
   at about 7% of a Davidson step on `si16-1k-ecut30` and aimed at ~80% of one
   on the 64-atom cell, where the device stage timings are `h_psi` 0.012 s
   against Davidson 0.640 s. **Unbatched path only** — `lax.switch` runs every
   branch under `vmap`, exactly as `lax.cond` does — which is where the large
   single-k cells are anyway. See "Inside a Davidson step".
3. **Expand by `notcnv` rather than by `nbnd`.** In the seeded regime the SCF
   runs, the live roots fall 20 -> 13 -> 10 -> 0 over four steps at
   `ethr = 1e-13`, and `h_psi` is exactly linear in the block's width on a CPU —
   so this is about half the expansion cost of the endgame steps. It is worth
   **nothing** on a cold band-structure solve, where no root settles at all, and
   whether it is worth anything on a device depends on whether a narrow block is
   launch-bound there. Same `vmap` restriction as the item above.
4. **Fold `dr2` into the iteration's other reductions.** It costs a transform and
   a dispatch of its own (~3% of an iteration) for a quantity the loop already
   computes a residual for. Mixing in G space would save another transform.
5. **Shell-based radial evaluation** for quantities depending only on `|G|` (~100
   shells vs 1459 G-vectors for Si). Note this is *not* strain-safe: shells split
   under strain, so it must stay off the stress path.
6. **Stop closing over the cell in the stress gradient** (P29). `at_strain`
   drops `_energy_gradient` on every call, because the compiled gradient closes
   over the cell it was traced at, so a variable-cell relaxation compiles the
   strain derivative again at every ionic step: **0.6 s of retracing for 0.57 s
   of arithmetic**, about a fifth of the run. Making the cell an argument of the
   traced function rather than a constant folded into it is the fix, and it
   costs a signature change in `stress/energy.py`. A fixed-cell `relax` does not
   pay it -- `at_positions` already keeps its compiled force -- which is why this
   surfaced only here.
7. **Schedule the response solver's threshold** (P25). `dfpt_kernels.f90` uses
   `thresh = min(0.1 sqrt(dr2), 1e-2)` where `response/phonon.py` holds a fixed
   1e-12, and the cost is `av.it. = 27.7` against `ph.x`'s 9.3 — a factor of
   three, on the stage that is 96% of the run. It is `electrons.f90`'s `ethr`
   schedule in a second place, the rule is already quoted in
   `response/sternheimer.py`'s docstring, and the same fix applies to
   `response/efield.py`. Cheapest item on this list by a wide margin.
8. *(done, 2026-08-22)* **A mixer in the response loop.** Was: 17 linear-mixing
   iterations against `ph.x`'s 5, whose mixer is `LR_Modules/mix_pot.f90`. It
   turned out not to be a speed item at all -- linear mixing of a map whose
   Jacobian has an eigenvalue below -1 **diverges**, which two systems then did
   (see "What a mixer in the response loop was worth"). `defumat/response/mixing.py`
   now wraps `scf/mixing.py`'s Anderson history for all three loops.
9. **One irreducible representation at a time** (P25), for the *memory* rather
   than the time: it bounds the working set at 3 modes in flight instead of
   `3 nat`, which is 7 GB on a 16-atom cell. It does not reduce the number of
   solves — `ph.x` perturbs along all `3 nat` modes too.
10. **The stress's reverse-mode tape through the radial transforms** (P11). 11 GB on
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

## What a van der Waals correction costs (P27)

**Nothing per SCF iteration, which is the whole shape of it.** Grimme's D2 is a pair sum over
the nuclei and never enters `v_of_rho`, so it is evaluated **once** when the `Calculation` is
built, once more per geometry in a relaxation, and once per gradient in a force or a stress.
The SCF loop does not see it at all.

What it does cost is set by `london_rcut`, and QE's default of 200 bohr is large on purpose:
the number of pairs in a shell grows as `r^2`, so a `1/r^6` sum truncates with an error
falling only as `1/rcut^3`. The kernel broadcasts to `(nat, nat, ntrans, 3)` and `ntrans`
grows as `(4 pi/3)(rcut + fold_radius)^3 / Omega`:

| case | `rcut` | `ntrans` | separations | energy | `jax.grad` |
|---|---|---|---|---|---|
| graphite, 4 atoms, `Omega` = 227 bohr³ | 60 | 5 675 | 2.2 MB | 5.1 ms | 4.4 ms |
| graphite, 4 atoms | 200 | 163 685 | 62.9 MB | 51 ms | 153 ms |
| bilayer graphene + 14 bohr vacuum, 4 atoms | 200 | 104 799 | 40.2 MB | 33 ms | 105 ms |
| silicon, 2 atoms, `Omega` = 265 bohr³ | 200 | 143 897 | 13.8 MB | 12 ms | 42 ms |

Against a 2-second SCF iteration on the bilayer, 33 ms once is not a cost worth optimising,
and the 105 ms gradient is paid once per ionic step. **The expensive case is a small dense
cell, not a slab**: a vacuum makes `Omega` large, and `ntrans` scales as `1/Omega`.

The peak working set is `nat^2 ntrans` doubles for each temporary, so it grows quadratically
in the atom count where the rest of the code grows linearly or better. A 32-atom cell of
graphite's density at 200 bohr would be 4 GB of separations, and the fix when it is needed is
to `lax.scan` the translation axis in chunks — the sum is a scalar reduction, so a chunked
form is exact and differentiable. It is not written, because nothing has needed it: 60 bohr
costs 27 times less and is within 3e-5 Ry of the converged value, and the convergence table is
in `PLAN.md`'s P27 entry.

## What the response loop costs on a slab (P27 x P26)

The electrostriction path run on bilayer graphene rather than on silicon, in one
process, at `ecutwfc = 30` on the 2 x 2 x 1 closed grid (4 k-points, 8 occupied
bands, ~1100 plane waves, FFT grid 18 x 18 x 72):

| stage | time |
|---|---|
| SCF to `conv_thr = 1e-12` | 5 s |
| strain response, 68 iterations at `alpha_mix = 0.3` | 399 s |
| the third derivative on top of it (field response + `db/dx` + six `jvp`s) | 241 s |
| a five-point second difference of the energy, per component | 5 s |
| a central difference of `epsilon` over re-converged cells | 410 s |

**The strain response is 62% of it and the iteration count is the reason, not
the cost per iteration.** Silicon's converges in about ten passes at QE's
`alpha_mix = 0.7`; this cell needs 68 at 0.3, because 0.7 *diverges* — 14 bohr of
vacuum puts the smallest nonzero `G_z` where `4 pi e^2/G^2` is two orders larger
than a compact cell reaches, and simple linear mixing of a map with a Jacobian
eigenvalue that large is unstable (`PLAN.md` P27). So the honest statement of
this cost is that it is a **mixing** problem: the ground-state SCF answers the
same stiffness with Kerker preconditioning and pays about 1.7x, where the
response loop has no preconditioner and pays ~7x in iterations. Preconditioning
`_self_consistent_response`, or replacing its linear mixing with the Anderson
mixer `defumat/scf/mixing.py` already has, is the obvious next move and is in
the backlog.

## What a mixer in the response loop was worth (P24 x P25 x P26)

The three response loops -- electric field, displacement, strain -- advanced with
one line of linear mixing, `dvscf += alpha_mix (induced - dvscf)`, where QE uses
a modified Broyden over four previous iterations (`LR_Modules/mix_pot.f90`).
Replacing it with `scf/mixing.py`'s Anderson history, which was already written
for the SCF density and which neither response loop used:

| case | linear, `alpha_mix = 0.7` | Anderson, same `alpha_mix` |
|---|---|---|
| silicon, dielectric response | 19 iterations | **9** |
| silicon, strain response | 18 | **11** |
| bilayer graphene, strain response | **diverges** (1.34x per iteration) | -- |
| bilayer graphene at `alpha_mix = 0.3` | 68 | -- |
| rhombohedral BN, strain response | **diverges after 61 iterations** | **18** |

**This was filed as a speed item and it is a correctness one.** The induced
Hartree potential is `4 pi e^2/G^2` against the induced charge, so a cell whose
smallest nonzero `G` is small has a Jacobian eigenvalue large and negative, and
linear mixing above `alpha_mix ~ 2/(1 + |lambda|)` walks away from the fixed
point. Bilayer graphene says so on the first iteration. **Rhombohedral BN says so
on the sixty-second** -- it converges at 0.625 per pass down to `3.9e-7`, then
turns around and grows at 1.30, because a subdominant mode with an amplification
above one is invisible until the dominant one has died. There is no `alpha_mix`
a caller can be told to use, because whether their system needs a smaller one
cannot be seen until the run is most of the way through.

Answers are unchanged: silicon's `epsilon_11` is 56.292875149 both ways, and the
P24, P25 and P26 regression suites (47 tests) pass with the new default.

**What is still on the table is better than this.** The response fixed point is
**linear** -- `(1 - K chi_0) dV = K chi_0 dV_bare` -- so the right solver is not
a mixer at all but GMRES on that operator, which is optimal over the Krylov space
where Anderson with a truncated window approximates it, and which *cannot*
diverge. One operator application per Krylov vector is exactly one mixing
iteration, so the currency is the same. P22's `newton_krylov` is not directly
reusable: it wraps a *nonlinear* residual and its finite-difference Jacobian
would cost two applications per vector.

## What ten sites cost (P28b)

Ten atoms per cell is the first size at which the working set of a *force* is the number
that decides whether a case runs at all. Same displaced silicon cell, same 4x4x4 grid (24
k-points in the wedge), forces and stress in one process, single machine, `/usr/bin/time -v`:

| case | dataset | peak RSS | wall |
|---|---|---|---|
| `si10-nc-force` | norm-conserving | **1.5 GB** | 0:37 |
| `si10-us-force` (forces only) | ultrasoft | **16.2 GB** | 2:19 |
| `si10-us-force` (stress only) | ultrasoft | **16.4 GB** | 2:29 |
| `si10-paw-force` | PAW | **16.6 GB** | 2:11 |
| `bi10-soc` (SCF only) | relativistic ultrasoft | **18.4 GB** | 46:22 |

**An order of magnitude between the first row and the rest, and it is the augmentation
charge.** The norm-conserving cell has no `Q_ij(G)` at all; the other two build it on a
dense grid of ~68000 G-vectors for `nh (nh+1) / 2` projector pairs per atom, and the
reverse pass keeps it live. It is the same trade P11 measured on `si8-us` (11 GB for a
stress against 899 MB for the SCF), one cell size up and with the k-axis in it as well.
The bismuthene row is the SCF alone -- no derivative at all -- and its 18.4 GB is
`Q_ij(G)` on 302569 G-vectors for a `dn` dataset with a `j`-resolved projector set.

Three practical consequences, stated so nobody meets them as a surprise:

* **`tests/regression/test_ten_site.py` runs in 42:55 and peaks around 11.4 GB.** It ran
  in **1:28:39 at 22.8 GB** until the compiled code was released between cases, and the
  bound on converged states was only half the story. The file caches at most two of those
  (`_converged`'s `lru_cache(maxsize=2)`) and makes all of a case's assertions in one
  function, so states are released as it goes; with an unbounded cache it was **killed**
  before finishing. But **the executables were unbounded**, and nothing here shares them:
  every case has its own `npwx`, `nbnd`, FFT grid and spin rank, so each compiles the whole
  SCF stack afresh and XLA keeps it for the life of the process. Measured over the first
  ten cases, resident memory went 0.55, 0.81, 1.27, 1.47, 1.60, 1.77, 1.82, 2.14, 2.47,
  **3.67 GB** -- monotonic, and none of it the states. With `jax.clear_caches()` between
  tests the same ten are 0.52, 0.65, 0.94, 1.09, 1.22, 1.17, 1.13, 1.28, 1.45, **1.45 GB**:
  flat. **It got faster as well as smaller**, which is the tell that it was paging rather
  than recompiling -- the recompilation the clear costs is worth less than the pressure it
  removes. The remaining peak is the largest single case, and that case is large.
  (11.4 GB is a sampled maximum from `ps` during the spin-orbit case, not `/usr/bin/time`'s
  true high-water mark; the post-test resident maximum is 7.7 GB.)
* A machine with 16 GB runs the norm-conserving ten-site set and not the ultrasoft one.
* Attributing the 16 GB to a *route* -- `addusforce`'s transcription against the `jax.grad`
  one -- was measured as a total and not decomposed; both routes ran in the same process.
  Decomposing it is in the backlog, next to P11's shell-wise radial transform, which is the
  fix both of them want.

What the symmetry fix was worth in *time*, on the same machine: `si10-nc`'s SCF went from
10.7 s to 5.6 s when the lattice point group search stopped missing four of the six
operations -- the density is symmetrised over six operations instead of two, and the
k-point set is unchanged. A bug that costs accuracy usually costs time as well.

## What the Tran-Blaha potential costs (P30)

Silicon, `ecutwfc = 30`, a 6x6x6 grid reduced to 16 k-points, a 32^3 dense grid,
one core. Timed per call, warm:

| | LDA (PZ) | TB09 |
|---|---|---|
| `sum_band` (the density) | 148 ms | 200 ms |
| `tau` from the states | — | **1125 ms** |
| `v_of_rho` | 9.3 ms | 86 ms |
| SCF iterations to `conv_thr = 1e-9` | 6 | 10 |

**`tau` is the cost, and it is 7.6x the density it sits beside.** It should be
3x: it transforms `i(k+G) c_G` for three cartesian directions where the density
transforms `c_G` once, and `sum_band.f90`'s meta branch has exactly that
structure. The extra factor is that each direction is scattered into the FFT box
separately, so the gather of `fft_index` and the zero-fill of the box are paid
three times per band instead of once for a `(3, ...)` batch. **Backlog**: build
the three components as one `(3, npwx)` array and transform them together, the
way `basis/gradients.py:gradient` already does for a scalar field — the same
change, one level up.

`v_of_rho` going 9.3 -> 86 ms is the 80-step bisection of the Becke-Roussel
inversion over every grid point, plus four transforms (a gradient and a
Laplacian per channel). The bisection is 80 exponentials per point per channel
and is the obvious thing to shorten: the bracket is wide only where `Q` is
large, so a Newton polish after ~30 halvings would do, and libxc's Brent takes
50-60 iterations for the same tolerance. Left at 80 because it is branch-free
and fixed-length, which is what `lax.fori_loop` wants, and because it is a
quarter of what `tau` costs.

**Nothing else changes.** The Hamiltonian, the eigensolver and the mixer are
untouched: mBJ is a multiplicative potential, so unlike an energy-carrying
meta-GGA (TPSS, SCAN) it needs no `dE/dtau` term acting on the wavefunction and
no `h_psi_meta` counterpart. What the functional costs is the two builds above
and the 1.8x in iterations.

## What spin-orbit and PAW add to the Tran-Blaha potential (P31-P33)

Neither is a new cost model, which is the useful part:

* **Spinors (P31).** `tau` becomes four components instead of one, but they come
  from the *same* three transforms per band -- the spin algebra happens on
  `grad psi` after it is on the grid, not before -- so a noncollinear `tau`
  costs what a collinear one costs per band, and the band count doubles for the
  usual reason rather than for a meta-GGA reason.
* **PAW (P32).** `becsum -> tau_lm` is the same einsum as `becsum -> rho_lm`
  against a tensor of the same shape, so the one-centre `tau` is free relative to
  the one-centre density. What is not free is the extra `_meta_exchange_onecenter`
  pass over the sphere: it evaluates the Becke-Roussel inversion at every
  (direction, radius) of the quadrature, `nx * mesh` points against the grid's
  `n1 n2 n3`, and on silicon that is 120 x 1200 = 1.4e5 against 32^3 = 3.3e4 --
  **four times the grid's own bisection work, per atom**. It is the one place
  where the 80-step fixed-length bisection is worth revisiting first (see the
  backlog note under P30).
* **The noncollinear PAW gradient correction (P33)** adds one multipole
  round-trip per channel -- `PAW_rad2lm` on the rotated densities -- over what
  the collinear branch does. Two einsums against the harmonic table; not
  measurable beside the quadrature it feeds.

## What an optical spectrum costs (P37)

**There is no QE ratio for this one and there cannot be**, which is worth saying
before the numbers: `pw.x` has no counterpart, and `TDDFPT/` reaches a spectrum
by Liouville-Lanczos rather than by a Dyson equation in G space, so the two do
not compute the same intermediate. The reference is Elk, which is all-electron
LAPW. What follows is therefore a cost *model* rather than a comparison.

Silicon, `Si.pz-vbc`, `ecutwfc = 18`, the unshifted 4x4x4 grid (64 k-points),
`nbnd = 60`, a 20^3 grid, one core, warm:

| stage | | |
|---|---|---|
| fixed-density run, 60 bands x 64 k | **58.0 s** | the largest single cost |
| `chi_0`, `ecut_response = 4` (`nm = 29`), 1 frequency | 6.5 s | |
| `chi_0`, `ecut_response = 4`, 121 frequencies | 7.4 s | |
| `chi_0`, `ecut_response = 8` (`nm = 115`), 1 frequency | 6.1 s | |
| `chi_0`, `ecut_response = 8`, 121 frequencies | **23.0 s** | |
| Dyson, RPA, 121 frequencies, `nm = 115` | 0.45 s | one solve per frequency |
| Dyson, bootstrap, the same | 2.29 s | 9 passes of the fixed point |
| `chi_0`, `ecut_response = 16` (`nm = 285`), 121 frequencies | 107.1 s | |
| Dyson, RPA / bootstrap at `nm = 285` | 5.68 s / 37.6 s | |

**The two halves scale differently and the table is arranged to show it.** The
pair transforms are a *fixed* ~6 s — one FFT per occupied-empty pair per
k-point, `64 x 224` of them here — and they do not care about the response
cutoff or the frequency count at all. The assembly is `nw nm^2` and is
everything else: `(115/29)^2 = 15.7` against a measured 17.0/1.0 in the
frequency-dependent part. So the response cutoff is quadratic and the frequency
grid is linear, and neither touches the transforms.

Which means the **`nbnd` that the physics needs is what a spectrum costs**, not
the matrix it is a spectrum of. Going from 30 to 60 bands doubles the pair count
and therefore the 6 s floor; going from `ecut_response = 4` to 8 quadruples
`nm^2` and costs 16 s once. The convergence they buy is not symmetric either:
60 bands takes the static residual from 0.67 to **0.013**, and
`ecut_response = 8` takes the local-field effect from half-missing to converged
at 2e-3. Both are needed and neither substitutes for the other.

**Peak working set: ~1.0 GB** for the case above, and it is the assembly rather
than the answer. The stored matrix is `nw nm^2` complex = **24.4 MB**; the
`einsum` that builds it holds `nw (2 npairs) nm` in flight per k-chunk, which is
124 MB, and the rest is the fixed-density run's own wavefunction buffer
(`64 x 60 x 360` complex). The trade is stated in `tddft/chi0.py`'s docstring
rather than taken silently: one matrix product per frequency instead of one
batched `einsum` would cut the 124 MB to 0.4 MB and halve the flop rate on CPU.
**Backlog**: make the frequency axis a chunking dial the way `batching.py` makes
the k axis one — the same shape of change, and it only becomes worth doing when
`nw nm^2` stops fitting.

**The bootstrap's self-consistency is free at this size** and will not stay that
way: 9 passes cost 2.29 s against RPA's single 0.45 s, because each pass is one
`nm x nm` inversion per frequency. That is `O(nw nm^3)`, and the `nm = 285` row
is there to check the exponent rather than to be quoted: **37.6 s against 2.29,
a factor of 16.4 where `(285/115)^3 = 15.2`**. The `chi_0` row checks the other
one — 107.1 s against a 6 s floor plus `17.0 x (285/115)^2 = 104`, so 110
predicted against 107 measured. Both scalings are as written, which is the only
reason to trust the model at a size nobody has run.

Nothing here needs `ecut_response = 16`: it moves `eps_M(0)` by **1.9e-3** from
the value at 8, where 8 against 2 is worth half the local-field effect. A system
that does need it should expect the *kernel* rather than `chi_0` to be the bill,
and its peak working set to be **2.9 GB** rather than 1.0.

## What an ultrasoft or PAW phonon costs (P39)

Measured on `si-epsilon-us.in` and `si-epsilon-paw.in` against `si-epsilon.in` --
the same two-atom silicon, the same 4x4x4 shifted grid, one core.

| | norm-conserving | ultrasoft | PAW |
|---|---|---|---|
| SCF | 1.8 s | 3.2 s | 3.8 s |
| dynamical matrix | **37.7 s** | **77.5 s** | **90.8 s** |
| response iterations to `tr2 = 1e-14` | 10 | 11 | 11 |

**2.1x for ultrasoft and 2.4x for PAW, and it is not the linear solves.** The
Sternheimer iteration count moves by one, and the cost of each solve is
unchanged to within the augmentation's own `s_psi`. What the factor buys is the
one-time work P39 adds, all of it outside the loop: `3 nat` extra `jvp` through
`at_positions` for the overlap derivatives, `3 nat` more for the mixed state's
own change (`drhous`), the same again for the core-charge term (`addcore`), and
a `map_k` per mode for `dLambda`. Each of those rebuilds the projectors and the
augmentation phases, which is what `at_positions` costs, so it is the *count* of
those calls that matters rather than the arithmetic inside any one of them.

PAW's further 17% is the one-centre response inside the loop: `3 nat`
`PAW_dpotential` per iteration where the electric field pays for three, and each
is a radial quadrature over every sphere.

The SCF column is there for scale: the response is 20x the ground state it
starts from, on every dataset.

**The way down, when it is wanted, is the one QE already takes** -- solve one
irreducible representation at a time rather than all `3 nat` modes, which cuts
the one-time work by the same factor as the storage.

### And what P43's third derivative adds on top of it: nothing measurable

The two tangents that made the Raman tensor work on a moving overlap are both
`jvp`s and neither is a solve. The tail on `db`
(`efield.ultrasoft_position`) is one extra `jvp` per mode per cartesian
direction through `at_positions` -- `9 nat` of them over the whole tensor,
against the `3 nat` linear solves the same loop is already paying for -- and the
occupied block is an array that `orthogonality_states` has already built for the
dynamical matrix.

Counted rather than timed, and said that way on purpose: over a whole tensor the
tail is `9 nat` `jvp`s through `at_positions` against the `3 nat` **linear
solves** the same loop already pays for, and the occupied block is free. **The
memory is unchanged**, because nothing new is stored per k-point -- the tail is
evaluated and contracted inside the `jvp`, and the peak is still the `3 nat`
first-order states the dynamical matrix already holds. A separate before/after
timing of `raman_tensors` has *not* been taken; the ratio above for the
dynamical matrix is the one to reason from, since the added work is of the same
kind and a third of the count.

## What an effective mass and a set of site angular momenta cost (P48)

Both are riders on a run that already happened, and this is the first entry here
with an **all-electron code on the other side** rather than `pw.x`.

Two-atom silicon, PAW PBE at `ecutwfc = 30` / `ecutrho = 180`, `a = 10.26` bohr,
one k-point stencil at `Gamma`. One core on both sides, the affinity mask set
before JAX is imported (the mechanism `tools/compare_qe.py` documents), Elk with
`OMP_NUM_THREADS=1`.

| | | |
|---|---|---|
| defumat SCF (4x4x4, `conv_thr = 1e-12`) | **4.3 s** | the thing both quantities ride on |
| `effective_mass`, velocity route | **4.3 s** warm, 4.8 s cold | 13 NSCF k-points |
| `effective_mass`, eigenvalue route | **6.0 s** warm, 6.4 s cold | 49 NSCF k-points |
| Elk ground state (LAPW, same cell, PBE) | 1.36 s | |
| Elk task 25 (`effmass`) | **1.08 s** | 27 k-points, all states |
| `angular_momenta` on a converged run | **1.2-1.4 s** | measured on the nickel case below: 64 k-points, `natomwfc = 18` |

**About 4x on the mass step, and the comparison is not like-for-like** — 733
plane waves against LAPW's much smaller basis, and 13 stencil points against 27.
The number worth keeping is the absolute one: **both codes do this in seconds**,
which is what makes the effective mass the cheapest derived quantity in the
package. The eigenvalue route costs 40% more than the velocity route for 3.8x the
k-points, because a stencil point's cost is one Davidson solve either way and the
velocity route adds three `jvp`s per point on top.

**Compilation is 0.4-0.5 s of the cold figure**, which is small here because the
stencil re-uses the SCF's own shapes: an NSCF at 13 k-points where the ground
state ran 64 compiles nothing new but the k axis.

**Memory** is one NSCF's, `(nspin, nk, nbnd, npwx)` with `nk` the stencil size —
13 or 49 k-points against the ground state's own count, so the peak is *below*
the SCF's on any cell whose k-grid is larger than the stencil. `angular_momenta`
adds `(nk, npwx, natomwfc)`, the same array a projected DOS or a Hubbard `U`
already holds, and **that array is where its 1.2-1.4 s goes** — building the
Löwdin-orthogonalised orbitals at 64 k-points, not the contraction, which is a
few `einsum`s on `(18, 2, 18, 2)`.

**The notebook is 63 s**, most of it the one spinor SCF on nickel (60 Ry, 64
k-points, `noncolin` + `lspinorb`), well inside the ten-minute ceiling.

## What a piezoelectric tensor costs, and why it is not a Born charge (P50)

`e_(k)ij` and `Z*` are the same construction with one coordinate changed -- one
`jvp` of a coordinate gradient of the frozen energy along the electric field's
response -- so they should cost the same and they do not. Measured on the same
cell, in the same process, off the same field response: AlAs (2 atoms, NC LDA,
`ecutwfc = 10`, 64 k-points `nosym`, `npwx = 137`), one core,
`OMP_NUM_THREADS=1`, 2026-08-31.

| | time | peak RSS |
|---|---|---|
| SCF, `conv_thr = 1e-12` | 2.1 s | 0.48 GB |
| the electric-field response (3 directions, self-consistent) | 15-20 s | 0.98 GB |
| **Born charges** on top of it (P24b) | **0.8 s** | 1.13 GB |
| **the piezoelectric tensor** on top of it | **6.4 s** warm, 8.7 s cold | **4.2 GB** |
| the transcribed contraction beside it (`zstar_eu` with a strain label) | 4.0 s | no change |

**Eight times the time and four times the peak, for a derivative with *fewer*
terms in it.** The strain leg drops the orthonormality multipliers the
displacement leg needs, and it is still the expensive one, for the reason P11's
entry above already gives about a stress against a force: `at_positions` changes
one complex exponential per atom over cached radial tables, where `at_strain`
moves `|G|` itself, so every radial transform in the setup is *inside* the
differentiated function and is rebuilt forward and taped backward. A
second derivative pays that twice. The ratio here (8x) is larger than the
first-derivative ratio P11 measured (a stress at 0.6 SCF iterations against a
force at 0.13-0.52) because forward-over-reverse holds the whole tape.

**The peak does not move with `k_batch`**, which is worth knowing before anyone
reaches for that dial: 4.2 GB at the platform default, 3.9-4.9 GB at
`k_batch = 8` and 16, and the SCF four times *slower* at both. What the tape
holds is not the k axis but the setup's radial and reciprocal-space
intermediates, which every chunk rebuilds in full.

**So the transcribed route is the cheap one here**, which is the reverse of the
usual arrangement in this project: 4.0 s and no extra memory at all, against 6.4
s and 3.2 GB, because it contracts a bare perturbation rather than taping a
gradient. It is kept as the cross-check rather than as the implementation
because the differentiated route is what extends -- but on a large cell it is the
one to reach for.

**What it does not need.** The other cross-check, the same derivative contracted
with the strain response on the screened side, costs **25 s** for six more
Sternheimer solves. It is in the slow test set and nowhere near the default path.

**The whole test file is 8 tests in 93 s** and the notebook is 33 s, both of
which include their own SCF and field response.

## What an optical conductivity costs (P51)

Single core, `taskset -c 0`, `OMP_NUM_THREADS=1`, 2026-08-31.

**The frequency axis is free and the band count is not.** The whole expense is
one fixed-density diagonalisation with empty states plus three `jvp` calls of
`H(k)` over the k axis, which produce `(3, nk, nbnd, nbnd)` matrix elements;
after that a frequency grid costs one matrix product per k-point, `(nw, nbnd^2)`
against `(nbnd^2, 3, 3)`. Two hundred and fifty frequencies and two thousand
cost the same to within noise. What does cost is `nbnd`, which enters the pair
sum **squared** and is also the truncation the f-sum rule measures.

| case | k-points | bands | time |
|---|---|---|---|
| fcc Al, `ecutwfc = 15`, one atom | 512 | 12 | **12 s** |
| fcc Ni + spin-orbit, `ecutwfc = 60`, spinor | 64 | 36 | 188 s |
| the same, denser | 216 | 36 | 561 s |

The nickel rows are a run that took **both** static routes, so each route is
about half, and they were measured with another job on the machine — read them
as an upper bound and as a ratio rather than as a benchmark. Tripling the
k-points costs three times as much, which says the expense is the NSCF and the
three `jvp` calls rather than the frequency sum.

**The whole regression file is 6 tests in 559 s**, of which the nickel pair is
about 250: one spinor SCF at `ecutwfc = 60` and four conductivities off it.

**Where the k-grid has to be spent, and it is not where a total energy spends
it.** Three quantities here converge at three different rates on the same cell.
Aluminium's plasma frequency moves 13.78 → 12.98 eV between 4x4x4 and 8x8x8, 6
per cent, because it integrates a Fermi surface. Nickel's anomalous Hall
conductivity is far worse, because its integrand lives on near-degeneracies at
that surface: it **changes sign** between 4x4x4 and 6x6x6 (`PLAN.md` P51 has
the table), so one mesh's value on its own is not a number. And silicon's f-sum ratio
moves 1.185 → 0.985 → 0.934 over the same three, converging onto a diamagnetic
weight of 0.9432 that itself moves by 1e-3 -- which is the sharpest k-convergence
diagnostic in the package, and the only quantity here that is an integral of a
*second* derivative.

### Against Elk, which is where this one was taken from

Two-atom silicon, `a = 10.2` bohr, the **same** 4x4x4 non-reduced grid (64
k-points), **20 states** (Elk's four occupied plus `nempty = 16`, which is also
this code's clean band cut), 200 frequencies from 0 to 0.3 Ha. One core on both
sides, `OMP_NUM_THREADS=1` and `taskset -c 0`, the affinity mask set before JAX
is imported. Elk's tasks were run **one per invocation** against a state already
on disk, with `tshift = .false.` so the atomic basis is identical across them.

| | Elk | defumat |
|---|---|---|
| ground state | 6.08 s | **1.50 s** |
| form the operator | 0.92 s (task 120, `pmat` to disk) | — |
| re-diagonalise | — | 2.55 s (NSCF, 21 bands) |
| contract, one tensor component | 0.48 s (task 121) | — |
| contract, **all nine** | 1.19 s | — |
| operator **and** all nine components | 2.11 s | **0.85 s** warm, 1.21 cold |
| **from a converged state, whole tensor** | **2.11 s** | **3.40 s** |

**Two numbers point opposite ways and both are worth having.** The step that
does the same work -- form `dH/dk` (or `pmat`) in all three directions and
contract it into the whole 3x3 tensor at every frequency -- is **2.5x faster
here**, and not for a clever reason: Elk writes the momentum matrix elements to
disk and re-reads them once per component, where the `jvp` and the contraction
stay in memory and the frequency axis is hoisted into one matrix product per
k-point instead of an `axpy` inside the band-pair loop. Its marginal cost per
component is small (0.48 s for one, 1.19 s for nine, so the k-loop and the file
dominate), which is why the ratio is 2.5 and not nine.

**Where defumat loses is the re-diagonalisation, and it is the whole of the
gap.** Elk's ground state leaves `EVECSV.OUT` on disk, so its post-processing
never diagonalises again; this code runs an NSCF, and that 2.55 s is the entire
1.6x on the bottom row. It is exactly the memory-for-time trade this file
describes elsewhere, taken the other way round, and it is a **choice rather than
an oversight** -- keeping every k-point's wavefunctions is what the k-batching
dial exists to avoid.

**What is not comparable, stated rather than discovered.** Elk is all-electron
LAPW: its basis is muffin-tin orbitals plus an interstitial expansion and it
carries all fourteen of silicon's electrons, against 200 plane waves per k-point
and eight valence electrons here. The ground-state row is therefore a comparison
of two methods and not of two implementations, and the 4x should be read that
way. The rows below it are the ones that compare like with like, because both
sides start from a converged state and produce the same object.

**Nickel was attempted and abandoned**, which is worth recording so the next
attempt does not repeat it: an Elk run of `ni-soc-nosym.in`'s cell with
`spinpol`/`spinorb` and a `bfieldc` seed converged to a moment of **0.008**
`mu_B` against this code's 0.617, so its `sigma_xy` is the wrong physics and
timing MOKE on it would mean nothing. The magneto-optical half of P51 therefore
has **no** Elk timing beside it yet.

**Peak.** `(nspin, nk, nbnd, ndim)` for the states, `(3, nk, nbnd, nbnd)`
complex for the matrix elements, and `nw x nbnd^2` complex per k-chunk for the
frequency sum. Sixty-four k-points, forty bands and five hundred frequencies is
13 MB of frequency workspace against 5 MB of matrix elements; the spinor nickel
at 512 k-points and 36 bands holds 1.0 GB of wavefunctions, which is what sets
the ceiling and what `k_batch` moves.

## What a nesting function costs (P52)

`N(q)` is two things bolted together and they cost wildly different amounts:
an NSCF that resolves the Fermi surface, and a correlation over the k-grid that
is essentially free. The whole point of the phase's one non-transcription is
the second, so both are timed.

### Against Elk, which is where this one was taken from

fcc aluminium, `a = 7.5` bohr, the **same** 12x12x12 grid, the same Gaussian
smearing of 0.025 Ha = 0.05 Ry. One core on both sides, `OMP_NUM_THREADS=1`
and the affinity mask set before JAX is imported (the mechanism
`tools/compare_qe.py` documents). Elk's tasks were run one per invocation
against a state already on disk.

| | Elk | defumat |
|---|---|---|
| ground state | 1.63 s (12x12x12, LAPW) | 1.17 s (10-point wedge, PW) |
| eigenvalues on the nesting grid | — (its SCF ran there) | 0.32 s (NSCF, 72 wedge points) |
| **the correlation itself** | **0.37 s** (task 105) | **0.001 s** |
| everything after the ground state | 0.37 s | 0.41 s warm, 0.93 cold |
| **from atomic densities to `N(q)`** | **2.00 s** | **1.58 s** |

The fourth row is `run_nesting` end to end and is 0.09 s more than the two rows
above it add up to: building the denser `KPoints`, walking the symmetry orbits
for the unfold map, and the dispatch around them. The bottom row is the first
and fourth added, warm.

**The correlation is 370x faster and that is the phase's one idea.**
`nesting.f90` writes `N(q)` as an `O(N_q N_k)` double loop -- 1728 x 1728
products with a `mod` fold inside -- and the fold is exactly what makes the sum
a cyclic cross-correlation, so `ifftn(|fftn(g)|^2)` gives every `q` at once.
The ratio is a complexity ratio and it grows with the grid: at 24x24x24 the
loop is eight times more work and the transform is nine.

**The rest is not a like-for-like comparison and the table says which rows
are.** Only the middle two are. The ground-state row is two methods rather
than two implementations -- all-electron LAPW with a muffin-tin basis and all
thirteen of aluminium's electrons, against 107 plane waves and three valence
electrons. And the two codes reach the eigenvalues differently on purpose: Elk
runs its *SCF* on the nesting grid and reads `EVALSV.OUT` back, where this code
converges the density on the input's ten-point wedge and pays 0.32 s for an
NSCF on 72 points. Both unfold a symmetry-reduced set onto the complete grid
rather than diagonalising all of it -- Elk with `ivkik`, this code with
`grid_equivalence` -- which is the same saving arrived at independently, and it
is worth 24x here.

**What the numbers agree on physically** is in `PLAN.md` P52: `N(0) = 396.42`
here against Elk's 378.759 in its own units, 4.7 per cent, of which the whole
is the Fermi surface underneath (`D(E_F)` 4.9719 states/Ry against 4.8228, and
`N` is quadratic in it). The dimensionless `N(0)/D(E_F)^2` agrees to 1.5 per
cent.

**Peak.** One `(nspin, nk_irr, nbnd)` eigenvalue array and one real
`n1 n2 n3` box. A 24x24x24 grid is 110 kB, so the working set is the NSCF's and
nothing else -- which is what makes the k-grid the axis to spend on. The
correlation never forms an `N_k x N_q` anything.


## What a second-harmonic tensor costs (P54)

AlAs, `a = 10.575` bohr, `ecutwfc = 30` (`npwx = 869`), the whole unshifted
6x6x6 grid (216 points), 22 bands, 200 frequencies. One core,
`OMP_NUM_THREADS=1`, the affinity mask set before JAX is imported.

All of these were taken on an idle machine and repeated: two runs agree to 8
per cent and a third, taken while a second test suite was running, is 20 per
cent slower across the board. The numbers below are the quiet ones.

| stage | s |
|---|---|
| SCF on the input's own 4x4x4 grid | 5.6 |
| NSCF, 22 bands on 216 points | 51.3 |
| `second_harmonic` | 22.6 |
| &nbsp;&nbsp;of which `dH/dk` matrix elements | 20.3 |
| &nbsp;&nbsp;of which the two k-sums (the tensor, and `truncation`'s re-run) | 2.3 |
| **SCF to `chi^(2)(omega)`** | **79.5** |

**The velocity matrix elements are the second-largest line and were mistaken
for the tensor.** The first version of this table put 24.5 s against "the
tensor" and another 24.5 against the `truncation` diagnostic; both were wrong,
and the way they were wrong is worth recording because it is the ordinary
failure of timing a call rather than a computation. `second_harmonic` costs
22.6 s, of which `VelocityOperator.matrix_elements` is **20.3** and both k-sums
together are **2.3** -- so `truncation`, which re-runs only the sum, costs about
a second rather than the price of the whole quantity again. **Timing an entry
point measures whatever it happens to call**, and here that was 90 per cent
something else.

**Where the time actually goes is the NSCF and the velocity operator**, which
is the same shape P51 and P53 both have: the second-order sum on top of a set
of states is the cheapest stage of the four. So the lever is the eigensolver
and the `k_batch` dial, not the assembly.

### What the assembly costs, and what it used to

Per k-point, jitted, one core, 200 frequencies:

| bands | as first written | now | |
|---|---|---|---|
| 22 | 23.3 ms | **8.9** | 2.6x |
| 30 | 54.5 | **17.0** | 3.2x |
| 40 | 156.3 | **31.6** | 4.9x |

The answer is unchanged to **3e-15**, which is the re-association's own
round-off, and the literal transcription of `nonlinopt.f90`'s triple loop in
`tests/unit/test_shg_machinery.py` is what both forms are checked against.

**Only `chi_II` needs a sum over the intermediate state.** Its denominator
`2E_l - E_n - E_m` couples all three band indices and does not factorise. Every
`l`-dependent factor in `eta_II` and `sigma_II` is either linear in `E_l` or
independent of it, so each of their sums is a pair of matrix products --
`sum_l r^i[.,l] r^j[l,.]` and the same with `E_l` inserted. Eighteen of those
cover all eighteen cartesian triples, so three `nb^3` arrays per triple become
eighteen `nb^3` matrix multiplications per k-point: BLAS work rather than
strided elementwise work, and no `nb^3` temporary at all. `chi_II`'s own
summand is then never materialised either -- `cc2` needs only its sum over `l`,
which carries no current direction, and the two `cc1` accumulations are
contractions over `n` and over `m`. What is left is one `nb^3` object per
*field pair* rather than per triple: six per k-point instead of eighteen.

**Two things that looked like optimizations and were not, both measured before
being believed.** Sharing the three velocity tangents through `jax.linearize`
instead of three `jvp` calls gives a bit-for-bit identical answer in **6.33 s
against 6.35** -- XLA had already deduplicated the common primal. And hoisting
the shared denominators and field-pair products into named variables, on its
own, was worth 10 per cent at 22 bands and **minus 12 at 40**, because those
too were already common subexpressions and the stacking cost more than it
saved. The rule `CLAUDE.md` states -- that an idiomatic-JAX rewrite which looks
equivalent has been slower more than once -- held both times. What did work is
the one change XLA cannot make on its own, because it is an algebraic
re-association across a sum rather than a common subexpression.

**And one that works and is refused.** Wrapping `matrix_elements` in
`eqx.filter_jit` is 2x -- 12.3 s to 6.4 warm, bit-for-bit identical, the two
measured back to back in one process and so a ratio rather than a figure
comparable with the 20.3 above -- and it takes the peak resident set from
**1.33 GB to 8.96**. That is the trade this
file exists to prevent: the unjitted path streams over k-points through
`over_kpoints`, and jitting the whole 216-point call materialises what the
streaming was avoiding. It is left unjitted, and `k_batch` is the dial.

### Against Elk, which is where this one was taken from

The same crystal, the same 6x6x6 grid, the same 0.005 Ha broadening, one core
each. Elk ran tasks 0, 120 and 125 in one invocation from nothing.

| | Elk | defumat |
|---|---|---|
| ground state | — | 5.6 s |
| the states the sum runs over | — | 51.3 s (NSCF, 22 bands) |
| `dH/dk` and the tensor | — | 22.6 s |
| **from atomic densities to `chi^(2)(omega)`** | **10.5 s** | **79.5 s** |

**Elk is about seven and a half times faster here and it is carrying more
bands**, which is the entry's uncomfortable half and the reason the rule says
to measure it: 37 second-variational states against 22. Both sides are the best
of two pinned runs on an idle machine; Elk's spread was 10.5 to 11.6 s. Its three stages could not be separated
without re-running each against a state on disk, so only the bottom row is a
comparison; the rows above it are this code's own breakdown.

**What is not comparable**, stated rather than discovered. The bases are two
methods and not two implementations -- an all-electron LAPW muffin-tin basis
carrying every electron of aluminium and arsenic, against 869 plane waves and
eleven valence electrons. And the two reach the matrix elements differently on
purpose: Elk's task 120 writes momentum matrix elements to `PMAT.OUT` and task
125 reads them back, where this builds `dH/dk` in memory and never forms a
momentum matrix element at all -- which is a correctness choice rather than a
performance one, `[H, r] = p` failing for a nonlocal pseudopotential, and it is
the one line of `nonlinopt.f90` this code deliberately does not transcribe.

**Peak.** One `nb^3` complex array per field pair in flight -- 40^3 x 16 B =
1 MB at 40 bands, times `k_batch` -- against the NSCF's wavefunctions at
`nk x nbnd x npwx` = 216 x 22 x 869 x 16 B = 66 MB. So `nbnd` costs time here
and not memory, and the working set is the states' and nothing the tensor adds.

## What the LO-TO splitting and the static permittivity cost (P55)

**Nothing measurable, and saying so is the point of the entry.** Both quantities
are contractions of tensors that are already in hand: the non-analytic term is a
rank-one addition to a `3 nat x 3 nat` matrix and one more diagonalisation of it,
and the ionic permittivity is a sum of `3 nat` outer products. The whole
assembly -- `nonanal`, the diagonalisation, `RamanIR` and the permittivity
together -- is **0.072 ms** on AlAs, averaged over 200 repeats after warm-up.

### Against `dynmat.x`, which is where this one was taken from

Same machine, one core each. `dynmat.x` was run on the `fildyn` this code writes
(`asr = 'no'`, `q = (1, 0, 0)`, `lplasma = .true.`), which is the same input and
the same arithmetic.

| | `dynmat.x` | defumat |
|---|---|---|
| its own timer | 0.00 s | -- |
| whole invocation, 50 runs averaged | 4.2 ms | -- |
| the assembly alone, 200 runs averaged | -- | 0.072 ms |

**The two columns are not the same work and the table says which**: 4.2 ms is a
whole process -- fork, `environment_start`, parsing the dynamical-matrix file,
the ASR, the diagonalisation, `RamanIR`, `polar_mode_permittivity` and the
printing -- where 0.072 ms is the arithmetic with the tensors already in memory.
The comparable statement is the one `dynmat.x`'s own timer makes: **0.00 s**, its
resolution. Neither is worth optimising, and the reason to write the pair down is
that it puts a bound on where a Raman-and-infrared run's time actually goes: the
field and displacement responses that produce `Z*`, `eps` and the force constants
are **40 s** on this cell, so the whole of P55 is 2 parts per million of the
calculation it completes.

## What a Berry-phase polarization costs (P56)

`run_polarization` is an NSCF on the string mesh plus a product of determinants,
and the second is free at any size a plane-wave code would run -- the strings
are the whole cost, and there is one diagonalisation per point on them.

### Against `pw.x`, which is where this one was taken from

AlAs at `ecutwfc = 10`, `gdir = 3`, `nppstr = 6`, a 2x2 transverse grid --
`tests/data/qe/alas-berry.in` and the ground state of `alas-raman.in`. One core
on both sides, `OMP_NUM_THREADS=1` and the affinity mask set before JAX is
imported (the mechanism `tools/compare_qe.py` documents), both starting from a
converged density already on disk.

| | `pw.x` (`lberry` nscf) | defumat (`run_polarization`) |
|---|---|---|
| k-points diagonalised | 24 (4 strings x 6) | 20 (4 strings x 5) |
| wall clock | **0.11 s** | **0.34 s** warm, 1.73 s cold |
| per k-point | 4.6 ms | 17 ms |

**3.1x on the wall clock and 3.7x per k-point**, which is the band `PERFORMANCE.md`
already reports for the SCF itself -- no surprise, because the string walk *is*
`c_bands` and nothing else. The two rows are directly comparable: same cell,
same cutoff, same threshold, same ground state, and both are dominated by the
eigensolver.

**The k-point row is the one asymmetry and it is in our favour by design.**
`kp_strings` lays out `nppstr` points spanning the whole reciprocal vector, so
its last point is the first one's periodic image and is diagonalised a second
time; the mesh here drops the repeat and closes the string with the index shift
instead. That is a free 1/`nppstr` -- 17 per cent here, 9 per cent at
`nppstr = 12` -- and it is why the per-k ratio is worse than the wall-clock one.

**The determinants are not the cost and were measured to say so.** Four strings
of five links each is twenty `4x4` `slogdet` calls: below the timer's resolution,
and under a millisecond even on the 6x6 x 12 mesh (396 points) the convergence
study used. The phase needs no equivalent of P52's FFT trick, because there is
nothing quadratic in it -- a polarization is a sum over strings, not a
correlation between them.

### The working set

One string at a time, which is the memory decision
`wannier_centers` already makes: the resident set is
`npoints x nbnd x npwx` complex and the *number of strings never enters it*.
For AlAs that is 5 x 4 x 331 x 16 B = 106 kB; for a 200-electron cell on a
12-point string at `npwx = 40000` it is 12 x 100 x 40000 x 16 B = 768 MB, which
is the number to check against a machine before asking for a dense transverse
grid. Raising the transverse grid costs time and no memory at all.

## What a magnetoelectric tensor costs (P57)

Six ground states and eighteen polarizations for the whole tensor; one column is
two and six. Both codes run the *same algorithm* -- a full self-consistent run
per field step and a Berry phase of each -- which makes this a rarer thing here
than most of the rows above: a like-for-like timing rather than two methods
compared by their output.

### Against Elk, which is where the route was taken from

AlAs, `ngridk 2 2 2`, field along z, step 0.02, spin-orbit coupling on. Elk built
from the vendored source with `gfortran` and the system BLAS/LAPACK/FFTW.

| | Elk (task 390) | defumat |
|---|---|---|
| threads | 2 | 4 |
| whole tensor, three columns | **70 s** | ~230 s (3 x the column below) |
| one column | -- (390 does all three) | **76 s** |
| one self-consistent run | -- | 20-32 s, 9 iterations |

**Elk is about 3x faster on the wall clock and about 6x per core, and that is
the honest way round.** An earlier draft of this section had it inverted, from
an estimate rather than a measurement -- the number is why `CLAUDE.md` asks for
the reference implementation's wall clock beside ours rather than an internal
timing. The reason is not mysterious: this is a two-atom cell, which is where a
LAPW basis is at its most efficient, and Elk's ground state converges quickly on
it. `ph.x` has no counterpart to time against.

**What the comparison does establish** is that the route is affordable on a
workstation, which is the question that decided whether the phase could be
finished here at all -- and that was not obvious, because Elk's own example cell
(GaAs) does not converge here at *zero* field, which is a physics problem rather
than a cost one.

**One cost is a choice rather than physics.** `chain` (Elk's `trdstate`) would
seed each run from the previous converged state and make all but the first
cheaper. It is **off by default here** because it biases the difference
(`PLAN.md` P57), and it costs nothing on this cell anyway -- 45 s either way,
since each run converges in nine iterations from the atomic density, which is
exactly the regime where Elk's reason for chaining does not apply.

### The working set

One string at a time, inherited from the polarization: `npoints x nbnd x npwx`
complex, with the number of strings never entering it -- under a megabyte for
this AlAs. The ground states are sequential and share nothing, so the peak is one
self-consistent run's.

## What a structure factor costs (P61)

Almost nothing, which was the entry's whole claim in `ELK-FEATURES.md` and is
the one part of it that needed no defending: `F(H)` is one transform of a
density that already exists and a gather at `nh` frequencies. What is worth
timing is therefore the *pair* -- the quantity beside the ground state that has
to precede it -- because the ratio between them is the point.

### Against Elk, which is where this one was taken from

Two cells, and the same cell and k-grid on both sides. Silicon in the diamond
structure at `a = 10.20` bohr, LDA, 8x8x8, `hmaxvr = 5`; and ferromagnetic bcc
iron on Elk's own `examples/magnetism/Fe` cell at `a = 5.416` bohr, LSDA, 8x8x8,
`hmaxvr = 6`, with Elk's Gaussian `swidth = 0.025` Ha matched to a Rydberg
`degauss = 0.05`. One core on both sides, `OMP_NUM_THREADS=1` and the affinity
mask set before JAX is imported (`tools/compare_qe.py`'s mechanism); Elk timed
by wall clock rather than by its own CPU-seconds line, which is what a threaded
run makes meaningless.

| | Elk | defumat |
|---|---|---|
| silicon, ground state + task 195 | **2.68 s** | **1.98 s** |
| iron, ground state + tasks 195/196 | **6.58 s** | **5.00 s** |
| silicon, the structure factors alone | 0.61 s (task 195 on a stored state) | **0.006 s** |
| iron, the structure factors alone | — (not separable from the pair above) | **0.005 s** |

**Only the last two rows are a like-for-like comparison, and even they are not
quite one.** Elk's 0.61 s is `sfacinit` plus `zftrf`: it reads `STATE.OUT` back
from disk and re-runs `init0`/`init1`/`linengy`/`genapwfr`/`genlofr` before it
transforms anything, where the step here consumes a converged state that is
already in memory. And `zftrf` genuinely has more to do -- an LAPW density is an
interstitial plane-wave part *plus* a muffin-tin expansion, so it carries a
radial Bessel integration per reflection per atom that a plane-wave code has no
counterpart for at all. The honest headline is the first two rows, where both
codes start from atomic densities and end with a table of reflections, and where
the difference is the ground state rather than the quantity.

**The ground-state rows are two methods, not two implementations**, the same
caveat every Elk comparison here carries: all-electron LAPW with muffin-tins and
28 electrons for silicon's cell against a norm-conserving plane-wave sphere with
8, and 26 against 8 for iron's. That the plane-wave side is *faster* on both is
not a statement about either code's quality.

**A soft dataset costs the same and pays it earlier.** The step is one transform
of whatever density the run produced, so ultrasoft and PAW add nothing to it:
0.005--0.006 s on all three kinds. What they do change is the box it runs on --
the same silicon cell is 16^3 norm-conserving and 30^3 at `ecutrho = 240` -- and
the SCF that produced it, which is where a dual of 8 is paid for. The
augmentation charge is already in `result.density`, so there is no soft branch
here at all.

**Peak.** One complex `n1 n2 n3` box for the transform -- 16x16x16 for silicon's
dense grid, 2 MB at 45x45x45 -- plus `(nh,)` complex output. Nothing scales with
the band count or the k-points, so the working set is the density's and the
quantity adds a copy of it. The `method="direct"` path is the exception and is
a test fixture rather than a route: it forms an `(nh, n1, n2, n3)` phase array,
which is `nh` times the density, and it exists because it shares no index
arithmetic with the transform.

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
| 2026-08-20 | The k axis chunked as QE's `k_loop`, default one k-point (`defumat/batching.py`) | converged bismuthene 44.9 → 22.5 s/iteration and 4.91 → 3.16 GB; 9% slower on `metal.in`'s ten cheap k-points |
| 2026-08-20 | The band axis chunked as `vloc_psi_k`'s `DO ibnd`, default one band (`map_bands`/`sum_bands`) | `h_psi` local term 2.48x at 16 atoms, 1.66x at 8, 0.91x at 180 PWs; `si16-1k-ecut30` 4.2 → 2.5x against QE |
| 2026-08-20 | `batch = 1` stops meaning a width-one batch axis: direct call at `nk = 1`, plain `lax.map` beyond | 37% of a Davidson solve, on every k-point of every run |
| 2026-08-20 | `v_xc` and `e_xc` from one `value_and_grad` pass instead of two evaluations | `exchange_correlation` 7.3 → 3.5 ms |
| 2026-08-21 | Kerker/Thomas-Fermi preconditioning in the mixer (`mixing_mode = 'TF'`), with QE's own `q_TF` | Al slab 24 → 14 iterations; 34 → 20 with more vacuum; one FFT per iteration |
| 2026-08-21 | Newton-Krylov on the SCF residual (`scf_solver`) | **not a speedup** — 19 to 139 `F` evaluations against 14 to 36; it buys unstable SCF solutions, not time |
| 2026-08-21 | `ns` joins the residual's packed state; `run_scf(starting_ns=...)` | DFT+U reaches the mixer's fixed point (58 `F` against 9) and a saddle the mixer cannot hold (31 against a runaway) |
| 2026-08-21 | Linear response (P24): the velocity operator from one `jvp`, the Sternheimer solve, and `epsilon_infinity` | exact `chi_0 K` at 0.5 s against the differentiated eigensolver's 3.5 s; silicon's dielectric constant in 66 s |
| 2026-08-22 | Phonons at `Gamma` (P25): the dynamical matrix as one `jvp` of the gradient that already gives the force | silicon's six modes in 57 s against `ph.x`'s ~2.2 s, of which the second derivative itself is 1.4 s; the 26x is 3.4x the linear solves and 3x the CG steps each, both with named causes |
| 2026-08-21 | Ultrasoft and PAW linear response (P24a): `dbecsum`, the augmentation charge's response, `int3` and `PAW_dpotential`, all as `jvp`s of code that existed | `epsilon_infinity` on four cases at 44-95 s; 1.9x (US) and 2.2x (PAW) the norm-conserving run, mostly the doubled dual |
| 2026-08-22 | Electrostriction (P26): the strain perturbation, the elastic constants and `d(chi)/d(strain)` as a mixed third derivative | 166 s on 8 k-points, of which the strain response is 80 s and the third derivative 36 s; **33x** cheaper than the published sweep of re-converged calculations |
| 2026-08-22 | Grimme's D2 dispersion (P27): a pair sum over the nuclei with its neighbour list fixed once, the force and the stress `jax.grad` of it | **zero per SCF iteration** -- it never enters `v_of_rho`; 33 ms per geometry and 105 ms per gradient on bilayer graphene at QE's default 200-bohr cutoff |
| 2026-08-22 | The third derivative on a slab (P27 x P26): bilayer graphene through `electrostriction`, and a guard on a diverged first-order solution | 645 s end to end, of which the strain response is 399 s and 68 iterations; QE's `alpha_mix = 0.7` diverges here at 1.34 per iteration, 0.3 converges at 0.5 |
| 2026-08-23 | The dynamical matrix of a metal (P28): the `jvp` split so the electronic half takes `wk` where the frozen Hessian keeps `wg` | two-atom aluminium's six modes in **78 s**, 9 response iterations at `av.it. = 23.0` against `ph.x`'s 7 and 3.3-6.3; the extra `jvp` per mode is not measurable against the linear solves, and the iteration gap is P25's two backlog items unchanged |
| 2026-08-22 | A mixer in the three response loops (`response/mixing.py`), Anderson over the packed state | silicon 19 -> 9 and 18 -> 11 iterations with identical answers; **bilayer graphene and rhombohedral BN converge where linear mixing diverged** -- a correctness fix filed as a speed one |
| 2026-08-24 | The Tran-Blaha potential (P30): `tau` from the states, the Laplacian as `-G^2 rho(G)`, the Becke-Roussel inversion as a fixed-length bisection with an implicit `custom_jvp` | silicon's gap 0.49 -> 1.13 eV for **1.8x the SCF iterations**; `tau` costs 1125 ms an iteration against the density's 200, which is 7.6x where the algorithm says 3x -- three separate box scatters, and the backlog item |
| 2026-08-25 | **First contact with a GPU** (GPU.md Phase 0): the same source, unmodified, on a V100 | `al10-metal` **5.44x** the four-core CPU baseline at `k=all, b=all` — and **0.99x on the defaults**, because the cache-shaped dials invert on a GPU (801 → 177 ms). Energy agrees to 1.6e-13 Ry, determinism bit-identical, fp64 costs 1.78-1.98x on a matmul and 0.85-1.44x on an FFT (grid-dependent) |
| 2026-08-24 | Spin-orbit (P31) and PAW (P32, P33) for the Tran-Blaha potential | `tau` as a 2x2 spin matrix costs the same three transforms per band; PAW's one-centre `tau` is free (same einsum as the density) but its Becke-Roussel inversion runs on `nx * mesh` = 1.4e5 points per atom against the grid's 3.3e4 |
| 2026-08-31 | The Fermi-surface nesting function (P52): Elk's `O(N_q N_k)` double loop written as one FFT, and a symmetry wedge unfolded rather than re-diagonalised | the correlation **0.001 s against Elk's 0.37 s** on a 12x12x12 aluminium grid; 72 diagonalisations instead of 1728; whole quantity from atomic densities 1.58 s against 2.00 s |
| 2026-08-31 | The shift current (P53): `w^ab = <n|d^2H/dk_a dk_b|m>` as a `jvp` of the velocity operator's `jvp`, and the sum rule's `sum_{p != n,m}` written as a commutator | **no reference to time against** — `pw.x` computes no photocurrent and Elk's `nonlinopt.f90` is a different response, so this is the absolute cost. AlAs: 18 s on 216 k-points at `nbnd = 16`, 62 s on 512 at 24, of which the six `w^ab` pairs are ~0.03 s per k-point after compilation. The pair sum is `O(nbnd^2)` and the intermediate one `O(nbnd^3)`, so the band count — which is also the accuracy parameter — is what the cost scales in. A converged AlAs peak (`ecutwfc = 22`, 30 bands, 64 k-points) is 55 s |
| 2026-08-31 | The LO-TO splitting and the static dielectric constant (P55): a rank-one term on the `Gamma` dynamical matrix and a sum of mode dipoles over `omega^2` | **0.072 ms** against `dynmat.x`'s whole 4.2 ms invocation, whose own timer reads 0.00 s. Both free beside the 40 s of response solves that produce their inputs |
| 2026-09-01 | Magnetocrystalline anisotropy by the force theorem (P58): the converged collinear density rotated onto each direction and diagonalised once, with the coupling on | **The one-shot leg is the like-for-like unit: 50.0 s against `pw.x`'s 23.2 s, both pinned to one core** on QE's 3-layer Co(0001) film (`PP/examples/ForceTheorem_example`, 16 k-points, ultrasoft, fully relativistic) -- **2.2x**, the band this package sits in, and both timings are a whole leg from scratch including setup. The pinning is the measurement: XLA sizes its thread pool from the **CPU affinity mask** and ignores `OMP_NUM_THREADS` (`tools/compare_qe.py:45`), so an unpinned run here quietly takes all 14 cores -- and the unpinned numbers first recorded for this phase (42-66 s) were *also* contended by concurrent jobs and were slower than the pinned one, which is the second reason not to quote them. **The SCF leg is not comparable at any core count**, and the reason is a missing mixer rather than a slower one: QE converges that film in 24 iterations with `mixing_mode = 'local-TF'` (`approx_screening2`), which at the time was refused here, so the run was 250 Kerker iterations -- and plain Anderson at QE's own `beta = 0.7` *diverges* on this slab. `local-TF` is implemented as of the row below and the SCF takes 48 iterations; this row's timings predate it. The 5m38s and 1m54s in QE's committed reference outputs are another machine's; every number above was measured here. |
| 2026-09-01 | QE's `local-TF` mixer (`approx_screening2`): Thomas-Fermi screening with a **space-dependent** length, solved by the Krylov least-squares method of the Fortran in its own Coulomb metric | **The Co(0001) slab runs at QE's own `beta = 0.7`, where plain Anderson diverges (to +335 Ry).** It reaches -223.13842 Ry at iteration 48 against `pw.x`'s -223.13876 (3.4e-4) and then oscillates about -223.142; at a 60-iteration cap (279 s, 4.65 s an iteration) the estimated accuracy is 3.2e-9 and the energy 3.5e-3 Ry out, so **it does not reach `conv_thr = 1e-10`**. Against Kerker at `beta = 0.3`, the only other mixer that survives this cell -- 2.0e-2 Ry out after 250 iterations -- that is six times closer in a quarter of the iterations. QE takes 24 to 1e-10, and closing that is the backlog item. **The preconditioner is not the cost**: 0.23 s per call against Kerker's 0.002 s and a ~5 s SCF iteration, because its Krylov solve is a handful of FFT pairs on the dense grid. What made it affordable is following the Fortran's *order*: `mix_rho.f90` preconditions the **combined search direction once** ('preconditioning the new search direction'), after the Broyden combination -- where preconditioning each history entry separately runs the solve up to eight times an iteration. For a linear preconditioner the two orders are identical, so Kerker's results are unchanged to the last digit. |
| 2026-09-01 | P58's force-theorem anisotropy and P59's `local-TF` mixer on a GPU (`tools/gpu/p59_check.py`, job 20026794) | **Both run unmodified on a Tesla V100-SXM2-16GB and reproduce the CPU numbers**: silicon's total energy to **7.1e-15 Ry**, the cobalt SCF to **3.2e-12 Ry**, and the anisotropy 2.397460 meV against 2.397461, **5.4e-7 meV** apart. **The two backend-independent identities hold on the card as they do on the host**: `local-TF` reaches the plain mixer's fixed point to 1.70e-11 Ry on both, and `soc_scale = 0` gives 2.2e-8 meV of anisotropy against the CPU's 1.2e-8 -- both zero. Iteration counts are identical (7 and 9), x64 is on, peak device memory **0.472 GB**. The whole check is 2m15s of wall clock on the node, of which the silicon leg is 15.1 s on the GPU against 1.2 s on the CPU -- **compilation, not arithmetic**, on cells this small, which is what GPU.md Phase 0 already recorded and is why no speedup is claimed here. This is a correctness result, not a performance one. |
| 2026-09-01 | The magnetic torque (P60): the anisotropy as one `jax.grad` of the band energy in the moment's angle, instead of a difference of two diagonalisations | **One angle instead of two, and the same K1 to 2.4e-5 meV.** On the one-atom tetragonal cobalt cell a torque is 16 s against the two-direction difference's ~15 s, so it is not cheaper -- both need one diagonalisation per angle and the torque needs one gradient on top. **What it buys is accuracy, not time**: the difference cancels seven digits and the derivative cancels none, and the torque is *smearing-robust* where the difference is not -- at `degauss = 0.02` Ry the band-energy difference reads 1.235 meV against a converged 0.531, while the torque reads 0.552 at the same width. The gradient itself is a small fraction of the cost: one extra pass over `sum w <psi|H|psi>` on frozen states, where the diagonalisation that produced them is the expensive half. |
| 2026-09-02 | The full (Liechtenstein) DFT+U functional (P62a): a four-index `vee` contraction in place of a scalar `U_eff`, built once per calculation from `harmonic_products` | **The functional is free and the SCF is the same one.** On antiferromagnetic FeO at `U = 4.3`, `J = 1.0` eV, one core each: **42.8 s of SCF against `pw.x`'s 7.64 s** (52.1 s and 8.46 s end to end, of which 9.3 s is JAX compilation), in 42 iterations against 21 — so the per-iteration ratio is **2.8x**, the band this package sits in, and half the wall-clock gap is the iteration count rather than the cost of an iteration. Against the *simplified* functional on the same cell the interaction is not measurable: `vee` is `(nslot, 5, 5, 5, 5)` built once on the host, and the per-iteration einsum is 625 multiplies against a matrix the Hamiltonian applies to every band. **The comparison is like-for-like** — the same input, the same pseudopotentials, the same `conv_thr` — which is rare for a row here, and it is only available because `kind = 1` is one of the two axes `pw.x` has. |
| 2026-09-02 | Around-mean-field double counting (P62c) and Slater integrals from the orbital (P62d) | **Neither is a per-iteration cost and that is the whole measurement.** AMF is the same `vee` contraction on a shifted matrix — the shift is one trace per slot per iteration, unmeasurable — and it converges FeO in 39 iterations against FLL's 42. The Yukawa route is **setup only**: `manifold_radial` 0.2 ms, `screening_length` 9.5 ms (about thirty evaluations of a 1195-point double integral inside a bracketed root find), `slater_set` 1.9 ms — **12 ms once**, against a 12 s SCF, so the phase's cost is three parts in ten thousand of the run it configures. **The Elk pair this project's rule asks for is not taken, and the reason is that it would not be informative rather than that it was skipped**: Elk reaches these through `dftu`/`inpdftu` inside its own ground state, so the only timeable unit is a whole all-electron LAPW SCF against a whole pseudopotential one — a comparison of two basis sets, not of this feature. What would be comparable is `genfdu` + `fyukawa` alone, which Elk does not time separately either. The absolute number above is the honest one and it is 12 ms. |
| 2026-09-02 | The spinor occupation matrix (P62b) and tensor moments (P62e): four spin blocks as four quadrants of one operator, and the moment basis built from Wigner 3-j symbols | **The spinor DFT+U term costs what the collinear one costs**, because it is the same contraction on a projector set twice as long: relativistic BN with spin-orbit coupling converges in 10 iterations and 250 s where the same cell at `U = 0` takes 250 s too -- the correction is not measurable against the spinor SCF it rides on, whose `2 npwx` states are the whole cost. **The tensor-moment basis is setup and is cached**: `moment_matrices(2)` builds all 100 matrices of a `d` shell once, in under 40 ms, and the constraint is then one `einsum` over `(nfix, 4, ldmx, ldmx)` per iteration -- unmeasurable beside a diagonalisation. **The Elk pair is not taken and the reason is the one P62c and P62d already give**: Elk reaches both through its own ground state, so the only timeable unit is a whole all-electron LAPW SCF against a pseudopotential one, which compares two basis sets rather than this feature. What is comparable is that Elk needs a *ground state per constraint* and so does this, and the constraint itself is free on both sides. |
| 2026-09-03 | Magnons (P63): the transverse spin susceptibility as a matrix over reciprocal lattice vectors, its Dyson equation, and the pole located as a root of `lambda_max = 1` | **The comparison is like-for-like and the ratio is 1.8x**, on Elk's own `Ni-magnetic-response` example (fcc Ni, a = 6.66 bohr, 10x10x10 shifted, one core each): `elk` does its ground state plus task 330 in **20m35s and 486 MB**, this package does the ground state (508 s), the fixed-density run at 30 bands (440 s) and one wavevector's response (1242 s) in **36m30s** -- 2190 s against 1235 s. **And the physics agrees**: Elk's pole is at 117.92 meV, of which 1.99 meV is the Zeeman gap of its own applied field (`bfieldc = 0.01` with the default `reducebf = 1.0`, scaled by `cb = gfacte/(4 solsc)` in `eveqnsv.f90` -- reading that rather than estimating it is what turned a "not comparable" into a comparison), so its field-free magnon is 115.93 meV against **115.04** here uncorrected and 123.08 Goldstone-corrected: **-0.8 and +6.2 per cent**, with the moment agreeing on the same run (0.6178 mu_B against 0.6152) and the Goldstone residual 2.04 per cent on that grid against 1.99 on a 4x4x4 one -- fifteen times the k-points moving it by 0.05 per cent, which says the residual is a truncation and not a sampling error. **Say what is not comparable**: LAPW against a 60 Ry plane-wave sphere; Elk's `gmaxrf = 5.0` is a **25 Ry** response sphere against the 60 Ry this cell needs for a 2 per cent residual, and the cost is quadratic in it (`nm` 561 against ~120); `emaxrf = 1.5` Ha caps the states it sums over where `nbnd` here does not; and the smearing schemes differ. Rebuilding the example with Elk 11.0.2 reproduces its committed `CHI_T.OUT` pole to the meV and its values to 0.49 absolute on numbers of order 8 to 28, which is that reference's own resolution. **What the response costs is `nw npairs nm^2`** and that shape is the design decision: fcc nickel on a 4x4x4 grid at `nbnd = 30`, `ecut_response = 60` is 12m42s and 6.3 GB for an SCF plus three wavevectors, and **79 s** at `nbnd = 24`, `ecut_response = 40` for one, which is what the notebook runs. Two things were measured and fixed on the way: the majority-band loop accumulates in a `scan` carry rather than stacking `nbnd` matrices of `(nw, nm, nm)` before summing them (1.2 GB of intermediate for a 40 MB result on nickel), and **the pole is found as a root of `lambda_max(omega) = 1` on a handful of frequencies rather than as a peak of a broadened spectrum** -- a hundred frequencies at `nm = 561` is twenty minutes a wavevector where eight is ninety seconds, and the root needs no broadening at all. |
| 2026-09-03 | The orbital magnetization (P64): the covariant k-derivative as dual states of the neighbouring manifolds, and `H` contracted between two of them | **The comparison is like-for-like and the ratio is 2.07x**, which is the band this package sits in. An iodine atom in a twelve-bohr box (`i-atom-soc.in`, 8829 plane waves, seven spinor bands, a 3x3x3 grid), one core each with the affinity mask set before JAX is imported: `pw.x`'s non-self-consistent `lorbm` run is **52.6 s** (its own timer agrees) and the same thing here -- the fixed-density diagonalisation on the 27-point mesh plus the assembly -- is **108.6 s**, at a 1.33 GB peak. The self-consistent legs are 3.9 s against 9.8 s (2.48x) on the same input, so the response leg is *not* the expensive half relative to QE. **Both sides start from the same place**, which is what makes the pair meaningful: each is one whole run from scratch, reading a converged density and producing the two Kubo terms. What the assembly itself costs is six overlap matrices, six `nbnd x nbnd` solves and four Hamiltonian applications per k-point -- the applications dominate, and they are the same `h_psi` a diagonalisation calls, so the ratio here is the ratio of `h_psi` and nothing new. The gather plans are `(nk, npol npwx)` integers per neighbour direction, 11 MB on this cell against the states' 53 MB. |
| 2026-09-03 | STM images (P65): the tunnelling density as `Calculation.density` called with delta-weighted occupations, and the plane read out of the grid by its own Fourier series | **The comparison is like-for-like and this end is 5.4x faster, which is the direction this file does not usually record.** Elk's task 162 on fcc aluminium (the cell and the 4x4x4 shifted grid of QE's `pw_metal/metal.in`, a 40x40 plane, one core each, both starting from a converged ground state) is **0.49 s** against **0.09 s** here warm and 0.29 s including compilation. Each side rebuilds the density from the new occupations and then evaluates it on 1600 points; Elk's leg also re-reads `STATE.OUT` and regenerates its radial functions, which is what a post-processing task does there and has no counterpart in a cached `Calculator`. **What is not comparable is the evaluation**: `rfpts` sums spherical harmonics inside each muffin tin and does a plane-wave sum only in the interstitial, where a pseudopotential code has one Fourier sum everywhere -- so the ratio is a statement about LAPW's geometry rather than about either implementation. **The ground states go the usual way**, 0.69 s by Elk's own timer for its thirteen iterations against 1.57 s here, so the quantity is cheap on both sides and the phase's own cost is a rounding error on the SCF that feeds it. Against `pp.x` the timing is not measured and would not be informative: `plot_num = 5` re-reads the collected wavefunctions off disk and writes a text dump of the whole grid, which is I/O rather than the sum. **Peak** is `chunk x ngm` complex for the phase table -- the points are chunked to ~32 MB, because a 40x40 plane against a 30000-vector sphere is 768 MB in one block -- plus the tunnelling density itself at one dense grid, which is what a density already costs. |
| 2026-09-03 | Vertical tunnelling transport (P66): the substrate's exit plane as a closed-form Gram matrix per k-point, the tip amplitudes as one plane-wave sum, and the whole thing as one quadratic form | **There is no like-for-like pair and the reason is the geometry rather than the physics.** QE's `pwcond.x` computes a Landauer transmission, but between two semi-infinite crystalline leads with the current along one axis -- one conductance per energy and no point contact, so no map; timing it against a tip-resolved image would compare two different calculations. Elk has nothing. So this is the absolute cost, one core, affinity mask set before JAX is imported. Monolayer graphene, the **whole** 6x6x1 grid (36 k-points, a wedge being refused), 20 bands, a 40x40 tip plane: **36.7 s** warm and 39.4 s cold, of which the fixed-density diagonalisation is **29.9 s** and the transport assembly itself is **6.8 s** -- so four fifths of the run is the NSCF every k-resolved quantity here pays, and the phase's own cost is the smaller part. **The nearest relative is P65's STM image at 8.6 s on the same call, and the gap is not the assembly**: `run_stm` may reduce its grid to a wedge and this may not, so it diagonalises 7 k-points where this does 36. That is a real cost of the quantity and it is the refusal, not the implementation. **The energy axis is free and this is the measurement that says so**: 1, 21 and 81 energies cost 35.7, 36.6 and 38.6 s, so eighty-one tip energies are **8 per cent** more than one -- the tip sampling and the Gram matrices do not depend on `E`, only the per-state weight does, which makes a transport dI/dV at every pixel a by-product rather than a sweep. **Cost shape**: sampling is `nk nbnd npts npwx` and dominates, the Gram matrices are `nk nbnd n_hpar`, and the contraction is `nk nbnd^2 npts` per energy. **Peak 0.85 GB** on that cell, of which the amplitudes are `npol nk nbnd npts` complex (18 MB here) and the phase table is chunked to ~32 MB as P65's sampler already chunks it. |
| 2026-09-04 | **Two `(nbnd, npwx)` blocks out of the Davidson working set**: the expansion rebuilt at the step that consumes it instead of carried across the loop, and the robustness retry moved from a `lax.cond` to a host branch | **The eigensolver's XLA temp buffer falls by 11.6 GiB on a 157-atom slab, and nothing gets slower.** That buffer is one contiguous allocation and it is what decides whether a large run starts: on FePc/SnTe at `ecutwfc = 60`, `nbnd = 1020`, gamma storage, an H200 was asked for **109.95 GiB** against a 104.85 GiB limit and died before the first iteration, where `defumat/sizing.py` had budgeted 69 GB. **That limit is a setting rather than the card**: it is JAX's default preallocation, 0.75 of the memory free when the client initialises (143158 MiB on these nodes, against an `nvidia-smi` total of 143771 -- the CUDA context holds the rest), where jobs after 2026-09-04 14:30 set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` and get a 132.81 GiB pool (`bytes_limit` 142606336000). **A buffer has to be read against the fraction its own job ran under**, and the pool taken from that job's own `bytes_limit` rather than from a remembered constant -- the same number read against the wrong one of those two pools says a configuration fits when it does not. Fitting `memory_analysis().temp_size_in_bytes` of the compiled solve gives `2.18 nvecx npwx zc + 4.20 nbnd npwx zc + 2.00 band_batch N_smooth zc` -- the `psi`/`hpsi` pair, the band blocks, and `h_psi`'s two FFT boxes per band in flight -- and the middle coefficient was **6.20** before this change: a refit measures the drop as **2.00 blocks exactly**, with the other two coefficients unmoved (2.15 -> 2.18, 1.98 -> 2.00). **The buffer assignment is what found them**, not the fit: XLA's own `memory-usage-report` gives `while.4{9}` and `while.5{9}` a `(nbnd, npwx)` slot each, which is the retry `cond` holding two copies of the whole loop -- it shares the big `(nvecx, npwx)` carries between them and not the small one. **Measuring it needs the right cell**: what separates the three terms is `nbnd npwx / (band_batch N_smooth)`, which is 0.5 on the slab and **0.06** at si16 with `band_batch = 32`, where the same change reads as 1 per cent. At a production-like ratio it is **-10.0%** (si16, `band_batch = 4`: 59.6 -> 53.6 MB) and **-10.2%** (si64, `band_batch = 16`: 932.6 -> 837.1 MB). **Time went the same way**, best of three at one core: si16 2.62 -> **2.13 s**, si32 11.93 -> **11.69 s**, iteration counts identical -- the retry `cond` was half the HLO and it is gone. **Seven of eight reference cells are bit-for-bit unchanged** (si8, si16, si8-us, si2-paw, the two-k-point test-suite silicon, Pt with spin-orbit, magnetic Fe) and the eighth moves by **1.1e-15 Ry**, from rewriting `force_real_g0` as a select on the same pass -- a scatter in the middle of the correction chain is a fusion barrier, and that path is gamma-only, which is what production runs. **Not touched, and the larger term at `band_batch = 64`**: the two FFT boxes per band belong to `vloc_psi`, not to this solver. |
