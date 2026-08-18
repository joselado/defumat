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

Single core, this machine, 2026-08-18. `conv_thr = 1e-10` on both sides so the
two are converging to the same place, which also makes the energy agreement a
correctness check on every optimisation.

| | QE 7.5 | pypresso | ratio |
|---|---|---|---|
| **`si-1k.in`** — 180 PWs | | | |
| setup / `init_run` | 0.050 s | 3.7 s | 74x |
| SCF, warm | 0.020 s | 0.064 s | 3.2x |
| per SCF iteration | 0.003 s | 0.009 s | **3.6x** |
| total energy | −15.25444871 Ry | −15.25444945 Ry | Δ 7e-7 Ry |
| **`si-1k-ecut40.in`** — 1131 PWs | | | |
| setup / `init_run` | 0.250 s | 4.4 s | 18x |
| SCF, warm | 0.100 s | 0.309 s | 3.1x |
| per SCF iteration | 0.013 s | 0.044 s | **3.5x** |
| total energy | −15.30461021 Ry | −15.30461021 Ry | Δ 3e-9 Ry |

Best of five runs on each side, and worth taking as ±20%: the small case in
particular swings between 0.064 s and 0.091 s from run to run, and QE prints its
timings to 0.01 s.

And the two cases the earlier log tracked, now measured the same way:

| | QE, per iteration | pypresso, per iteration | ratio |
|---|---|---|---|
| `pw_scf/scf.in` — Si, 2 k-points | 0.003 s | 0.018 s | 7.4x |
| `pw_metal/metal.in` — Al, 10 k-points | 0.017 s | 0.133 s | 8.0x |

QE prints its timings to 0.01 s, so its per-iteration figures on the small cases
are two significant figures at best and the ratios there carry that uncertainty.
The multi-k cases are worse than the single-k ones because pypresso pays a fixed
per-dispatch overhead that k-point batching does not yet hide; that is the same
overhead the setup row shows, seen from a different angle.

**Cold versus warm.** The tool reports both. A first SCF costs ~3.5 s more than
a warm one, all of it XLA compiling kernels, and setup is ~3.9 s of which the
arithmetic is 0.04 s. QE has no equivalent cost and never will. This is a fixed
overhead, not a scaling problem — it is the same few seconds on a system a
hundred times larger — but it is real for short runs, and the honest comparison
of the *physics* is the warm number.

## What moved, and by how much

Baseline is the state before this optimisation pass, measured the same way on
`si-1k.in`.

| | before | after | |
|---|---|---|---|
| setup | 9.96 s | 3.7 s | 2.7x |
| SCF, warm | 0.920 s | 0.064 s | **14x** |
| per SCF iteration | 0.131 s | 0.009 s | **14x** |
| against QE, per iteration | 53x | 3.6x | |

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

## What the remaining 3.5x is made of

Measured in situ — a clock around each step of the real SCF loop, not around a
micro-benchmark. (Timing the solver on its own with the *converged* wavefunctions
as its seed understates it by more than half.)

| Stage | `si-1k.in` (180 PWs) | `si-1k-ecut40.in` (1131 PWs) |
|---|---|---|
| **diagonalise** | 6.7 ms — **39%** | 39.6 ms — **60%** |
| density + symmetrise | 2.3 ms | 9.4 ms |
| `v_of_rho` | 1.6 ms | 7.4 ms |
| occupations | 2.3 ms | 1.8 ms |
| scf accuracy (`dr2`) | 0.9 ms | 2.3 ms |
| mixing (host round trip) | 0.8 ms | 1.4 ms |
| energies (one host sync) | 0.7 ms | 0.9 ms |
| build the Hamiltonian object | 0.4 ms | 0.5 ms |
| loop overhead | 1.2 ms | 3.2 ms |
| **per iteration** | 17.0 ms | 66.2 ms |

The eigensolver is still the largest single item but no longer overwhelming: 60%
at the production cutoff, down from 81%. Inside a Davidson step, at 1131 plane
waves:

| | |
|---|---|
| `h_psi`, 4 bands | 7.33 ms — **85% of a step** |
| subspace solve, 16x16 (Cholesky + `eigh`) | 0.85 ms |
| rotations, 16 -> 4 vectors | 0.19 ms x2 |
| **one step** | **8.6 ms** |

What is left, in order:

1. **Each step applies `H` to every band.** `cegterg` applies it only to the
   unconverged ones (34 `h_psi` calls in the QE run above, on one to four vectors
   each); the static-shape design here applies it to all four and masks the rest.
   Cheap early, when nothing has converged, and up to 4x wasteful at the end. The
   machinery already exists — the expansion compacts unconverged roots with a
   stable `argsort`, and the same permutation would let `h_psi` run on a masked
   block.
2. **Each FFT is ~1.5x FFTW.** `h_psi` on 4 bands is 8 transforms of the 30^3 box
   in 7.33 ms, or 0.92 ms each, against roughly 0.6 ms for QE's `fftw` calls.
   This is close to a floor: XLA's CPU FFT against FFTW, single threaded.
3. **Per-dispatch overhead**, which dominated before this pass, is now under 5%
   at the production cutoff. It is why the small case still shows a worse ratio
   than the big one, and why the multi-k cases are worse than either.

## Optimisation backlog

Ordered by expected gain per unit of effort. None of these may change a
validated number.

1. **A persistent compilation cache** (`jax_compilation_cache_dir`). Does nothing
   for a first run on a new machine, but it removes the ~7 s of compilation from
   every run after that, which is what a user actually experiences. The cheapest
   remaining win by a wide margin.
2. **Apply `H` only to unconverged bands.** Worth up to 4x in the late SCF
   iterations and nothing in the early ones — the largest remaining item in the
   eigensolver. The machinery already exists: the subspace expansion compacts
   unconverged roots with a stable `argsort`, and the same permutation would let
   `h_psi` run on a masked block.

3. **Fold `dr2` into the iteration's other reductions.** It costs a transform and
   a dispatch of its own (3% of an iteration) for a quantity the loop already
   computes a residual for.
4. **Fuse the iteration body further.** Three compiled units plus host glue could
   be two, at the cost of putting the occupation weights on device for the fixed
   -occupation case.
5. **Shell-based radial evaluation** for quantities depending only on `|G|` (~100
   shells vs 1459 G-vectors for Si). Note this is *not* strain-safe: shells split
   under strain, so it must stay off the stress path.
6. **k-point reduction to the irreducible wedge** — not an optimisation of code
   but of how much of it runs, and worth more than anything above on a real
   system.
7. **`jax.sharding` over the k-axis**, once there is more than one device worth
   using. The k-axis is already leading on every wavefunction-shaped array.

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
