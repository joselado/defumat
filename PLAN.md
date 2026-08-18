# pypresso — architecture and implementation plan

Target for the first milestone: **SCF, band structure, DOS**, in pure Python with JAX
(Numba where a host-side loop is the better tool). Everything else in Quantum ESPRESSO is
out of scope for now but the code is expected to grow toward it, so the structure below is
chosen to make additions local rather than invasive.

JAX is chosen for two reasons that constrain the design throughout: **autodifferentiation**,
so that response properties (forces, stress, polarization, second harmonic generation) come
from differentiating the code rather than from hand-derived expressions (§6), and **GPU
execution** of the same code (§5). Good performance is a requirement; it may be optimized
later, but not designed out.

Reference Fortran source and its per-subsystem map: see `CLAUDE.md`.

---

## 1. Design rules

These exist because QE's own structure is what makes it hard to extend, and because the
JAX/GPU target constrains the code shape. They apply to every module.

**R1 — No mutable global state; objects with methods are welcome.** QE keeps nearly
everything in shared `MODULE` variables (`scf_mod`, `wvfct`, `uspp`, ...) — that is the
thing not to copy. State is passed explicitly instead. But this is *not* a literal
transcription of the Fortran: write it as idiomatic Python, with classes and bound methods
wherever they make the code clearer (`ham.apply(psi, k)`, `pseudo.projectors(k)`,
`density.symmetrize(sym)`, `Wavefunctions.from_random(...)`). The one hard constraint is
that any object crossing a `jit`/`grad` boundary must be a **frozen, pytree-registered
class**: methods and properties are free, mutation is not. Arrays are pytree children;
shapes, flags, and functions are static metadata, because changing them should trigger a
retrace.

*Decided:* use **equinox** (`eqx.Module`) as the base class. It gives exactly this — frozen
dataclass, automatic pytree registration, `eqx.field(static=True)` for metadata, methods and
inheritance that behave normally — without the boilerplate of hand-rolling
`register_dataclass` on every state object. It is a small, stable, JAX-native dependency.
`eqx.filter_jit`/`filter_grad` also remove the usual friction of pytrees that hold both
arrays and static config.

**R2 — Setup and compute are separate worlds.**
- *Setup* (host): parsing, lattice/symmetry analysis, k-point generation, G-vector
  enumeration and sorting, radial→G pseudopotential tables. NumPy (or Numba where loops
  dominate), dynamic shapes, runs once, allowed to be slow and imperative.
- *Compute*: everything inside the SCF/diagonalization loop. JAX, static shapes, jittable,
  no Python branching on traced values, no host sync except the one convergence check per
  iteration.
Setup produces immutable, fully-shaped arrays that compute consumes. Nothing flows back.

**R3 — One-directional layering.** `io → system → basis → pseudo → xc → hamiltonian →
solvers → scf → workflows`. A module never imports from a later layer. If it needs to, the
abstraction is in the wrong place.

**R4 — Pluggable pieces go behind a named registry.** XC functionals, density mixers,
eigensolvers, pseudopotential families, smearing types, DOS integration schemes. Input-file
strings (`'pbe'`, `'david'`, `'mp'`) map to implementations through a registry so adding a
variant is a new file plus one registration line, never an edit to a growing `if/elif` in
the driver.

**R5 — Write the general interface even when the first implementation is trivial.**
Concretely: `apply_s(psi) -> psi` exists from day one (identity for norm-conserving,
real for ultrasoft/PAW later) and the eigensolver calls the generalized form; the
`Density` object carries a spin axis of length 1 for `nspin=1`; `Hamiltonian` is built from
a list of potential terms so DFT+U or external fields are an added term. Retrofitting any
of these later means touching every call site.

**R6 — Rydberg atomic units internally**, matching QE (energies in Ry, lengths in bohr,
masses in Ry a.u.). Conversion happens only in `io/`. Every public function's docstring
states the units of its arguments.

**R7 — Padding and masking, not ragged arrays.** The number of plane waves varies per
k-point. Allocate to `npwx = max_k(npw_k)` and carry a boolean mask, as QE does. Never
branch on `npw_k` inside compute code — it would retrace per k-point and defeat batching.

**R8 — Everything is validated against QE numerically, not by inspection.** See §4.

**R9 — Every feature gets a tutorial notebook.** `notebooks/` demonstrates each capability
on a concrete system (silicon by default), executed with outputs committed, ending in a
comparison against the QE reference where one exists. A phase is finished when its
notebook exists, not when its tests pass. See `notebooks/README.md`.

---

## 2. Package layout

```
pypresso/
  __init__.py           # enables x64 before anything else; top-level API
  config.py             # precision policy, device selection, runtime options
  units.py              # constants + conversions (ref: Modules/constants.f90)
  io/
    pwin.py             # pw.x input: namelists + cards -> InputConfig
    upf.py              # UPF v2 XML -> PseudoPotential
    output.py           # human-readable log; JSON results; bands/DOS files
    qeref.py            # parse QE reference outputs (test/validation side)
  system/
    cell.py             # ibrav/celldm -> lattice, reciprocal lattice, volume
    structure.py        # species + positions + cell (the geometry pytree)
    symmetry.py         # symmetry op detection, IBZ reduction, symmetrization
    kpoints.py          # Monkhorst-Pack grids, band paths, weights
  basis/
    gvectors.py         # G enumeration/sorting, cutoff spheres, per-k index maps
    fft.py              # sphere<->FFT-box gather/scatter, jnp.fft wrappers
    wavefunctions.py    # padded (nspin, nk, nbnd, npwx) container + masks
  pseudo/
    radial.py           # radial grids, Simpson integration, spherical Bessel
    harmonics.py        # real spherical harmonics Y_lm(G) and derivatives
    local.py            # vloc(G), rho_atomic(G), core charge
    projectors.py       # beta projectors vkb(k), D_ij coefficients
    tables.py           # PseudoPotential dataclass + per-species G tables
  xc/
    registry.py         # name -> functional
    lda.py, gga.py      # PZ/PW, PBE (libxc bridge optional, behind the registry)
  hamiltonian/
    terms.py            # kinetic, local, nonlocal as composable term objects
    operator.py         # Hamiltonian pytree; apply_h / apply_s
  solvers/
    registry.py
    davidson.py         # block Davidson (QE default)
    cg.py               # later
  scf/
    density.py          # sum_band: occupied states -> rho(r), rho(G)
    potential.py        # v_of_rho: Hartree + XC + local -> V(r)
    occupations.py      # Fermi level, smearing (gaussian/MP/FD), fixed occ
    mixing.py           # simple / Anderson / Broyden, behind the registry
    energy.py           # total-energy decomposition matching QE's printout
    ewald.py            # Ewald sum for the ion-ion term
    driver.py           # the SCF loop
  workflows/
    scf.py, nscf.py, bands.py, dos.py
  cli.py                # pypresso scf|bands|dos <input>
tests/
  unit/                 # analytic and self-consistency checks
  regression/           # against QE reference outputs
  data/
```

---

## 3. Phases

Each phase ends with a concrete, checkable number — no phase is "done" on the basis that
the code runs.

**P0 — Scaffolding. ✅ DONE.** Package skeleton, `pyproject.toml`, x64 enabled at import,
`config.Precision` dtype policy, `units.py`, pytest with tolerance module and markers,
`io/qeref.py`, `cli.py inspect`. *Check met:* 17 tests pass; the parser reads `pw_scf/scf.in`
(SCF + stress), `scf-1.in` (bands), `scf-2.in` (nscf), `pw_metal/metal.in` (Fermi level,
smearing) and `pw_lsda/lsda.in` (two spin channels), and the parsed energy terms sum to the
parsed total energy. `units.py` is checked against `Modules/constants.f90` by parsing the
Fortran, not by restating numbers.

**P1 — Input and geometry. ✅ DONE.** `io/pwin.py` (namelists + cards), `system/cell.py`
(`latgen` for every ibrav, `Cell`), `system/structure.py`, `system/kpoints.py` (MP grids,
explicit lists, band paths), `system/builder.py`. *Check met:* 288 tests pass; a sweep over
every input with a committed benchmark in `pw_lattice-ibrav`, `pw_scf`, `pw_metal`,
`pw_atom`, `pw_lsda` (87 cases, 174 test items) reproduces QE's printed `alat`, volume, crystal axes and
reciprocal axes, and — for explicit k-point lists and band paths — the k-points and weights.

Two conventions found the hard way, both now covered by tests:

- **Fortran `NINT` rounds half away from zero; NumPy's `rint` rounds half to even.** The
  Monkhorst-Pack fold `x - nint(x)` hits exact halves for every unshifted even grid, so
  `rint` would leave points at `+0.5` where QE puts them at `-0.5`.
- **QE's printed k-list for `K_POINTS automatic` is not a subset of the MP grid.**
  `kpoint_grid.f90` reduces using the point group of the *Bravais lattice* and keeps grid
  points, but `irrek.f90` then rotates those representatives into the wedge of the
  *crystal's* point group, and a rotation carries a shifted grid off itself. (Concretely,
  `lattice-ibrav2-kauto` prints a k-point at crystal `(0.25, 0.5, 0.5)` from a shifted
  2×2×2 grid.) Point-by-point comparison of automatic grids therefore has to wait for P6;
  until then the sweep checks grid size, uniform weights, first-BZ membership and
  uniqueness.

**P2 — Plane-wave basis. ✅ DONE.** `basis/fftgrid.py` (FFT dimensions), `basis/gvectors.py`
(`GVectors`, including gamma-only half-sphere), `basis/planewaves.py` (per-k selection,
padded to `npwx` with a mask), `basis/fft.py` (sphere↔box, QE's scaling), `basis/builder.py`
(dense + smooth grids). *Check met:* 631 tests pass; across the 86-case sweep, `ngm` on the
dense grid, the dense FFT dimensions, `ngm` and the FFT dimensions on the smooth grid (the
14 ultrasoft cases with `dual > 4`), and `npw` at every k-point (52 cases with a
non-symmetry-reduced k-set) all match QE **exactly** — these are integers, so there is no
tolerance involved.

Three things this phase settled:

- **Gamma-only halves the G-vector set, and QE reports the halved count.** At k = 0 a real
  wavefunction gives `c(-G) = conj(c(G))`, so `ggen` keeps the half-space `x > 0` plus the
  half-plane `x = 0, y > 0` plus the half-line `x = y = 0, z >= 0` — G = 0 kept once, hence
  `(ngm_full + 1)/2`. The selection is implemented; the *use* of it (real wavefunctions in
  `h_psi`) comes at P4.
- **One benchmark's FFT grid differs for a machine-dependent reason, not a physical one.**
  `pw_scf/scf.in` and `pw_scf/scf-cg.in` are the same system at the same cutoffs and both
  report 1459 G-vectors, but their committed grids are 15³ and 16³. 15 = 3·5 is a valid FFT
  size everywhere except IBM ESSL, whose `allowed` additionally demands a factor of 2. The
  references predate the current release (the suite records `REFERENCE_VERSION 6.0`) and
  were evidently not all produced on one build. The Miller range is identical either way
  ((15−1)/2 = (16−1)/2 = 7), so the G-vector set is the same; the test checks that instead
  for this one case.
- **Miller indices are stored, cartesian G is derived.** Storing cartesian components would
  freeze the cell and make stress-by-differentiation impossible; a test confirms
  `grad(|G|²)` w.r.t. the lattice is non-zero (rule D2).

Carried forward as a known limitation: `build_plane_wave_basis` selects from and indexes
into the **dense** G set. That is exact while `dense is smooth` — i.e. for every
norm-conserving run, hence all of the first milestone — but QE keeps wavefunctions on the
**smooth** grid. When ultrasoft arrives, the wavefunction FFTs must use the smooth grid's
dimensions and index map, and `PlaneWaveBasis.indices` must be rebased. P4's `h_psi` picks
an FFT grid for `vloc_psi` and should inherit this warning rather than rediscover it.

**P3 — Pseudopotentials. ✅ DONE.** `pseudo/upf.py` (UPF v2), `pseudo/radial.py`
(QE's Simpson, the 10-bohr mesh truncation, spherical Bessel with series branches),
`pseudo/harmonics.py` (`Y_lm` in QE's ordering), `pseudo/formfactors.py` (`vloc(G)`,
atomic and core charge, projector form factors — integrated directly at each `|G|`
rather than interpolated, so they stay differentiable in `q`), `pseudo/projectors.py`
(`vkb(k)`), `pseudo/potentials.py` (structure factors and the crystal quantities).
*Check met:* all 11 shipped UPF files parse; `Y_lm` satisfies the addition theorem and
is orthonormal to 2e-14; Bessel functions match SciPy to 1e-10; the atomic charge
integrates to the valence; and the Ewald energy — which exercises the cell, the
G-vectors, the structure factors and the positions together — matches QE on **76
benchmarks** to better than 1e-6 Ry.

**The trap:** `vloc`'s `G = 0` term integrates `r(r V + Z e^2)`, **not** the `q -> 0`
limit of the `erf`-split integrand used for `q > 0`. QE's source says so in a comment.
Getting it wrong shifts every eigenvalue by a constant (2.5 eV for silicon) while the
calculation still converges beautifully.

**P4 — Hamiltonian and diagonalization. ✅ DONE (dense solver).**
`hamiltonian/operator.py` (kinetic + local via FFT + nonlocal, plus `apply_s`),
`solvers/dense.py`. *Check met:* eigenvalues match QE to <1e-3 eV wherever they are
printed. **Davidson is still outstanding** — the dense solver is `O(npw^3)` and is
correct-by-construction ground truth, not a production algorithm. See `PERFORMANCE.md`.

**P5 — Full SCF. ✅ DONE.** `scf/` — `v_of_rho` (Hartree + LDA), `sum_band`,
occupations for every QE smearing plus `from_input`, Anderson mixing, Ewald, and the
driver with QE's energy decomposition. `xc/lda.py` writes only the energy density and
gets `v_xc` from `jax.grad`. *Check met:* **silicon's total energy matches QE to
1.1e-8 Ry** and the metals to ~2.5e-8 Ry, term by term, across 8 regression cases.

**The traps:** (1) XClib returns **Hartree**, not Rydberg — `v_of_rho` multiplies by
`e2`, and missing that halves the XC energy. (2) **Density symmetrisation is not
optional**: a symmetry-reduced k-point set gives an unsymmetric density, which
converges happily and splits degenerate levels by tens of meV. That forced P6 forward.

**P6 — Symmetry. ◐ PARTIAL.** `system/symmetry.py` finds the space group (48
operations for diamond silicon, non-symmorphic, matching QE) and symmetrises the
density. *Still to do:* reducing a k-point grid to the irreducible wedge — currently
the full grid is used, which is correct but costs time, and is why an automatic-grid
run cannot yet be compared to QE point by point. Also `crystal_sg` (Wyckoff) input.

**The trap:** with `M` defined by `S a_i = sum_j M_ij a_j`, crystal coordinates
transform as `c' = c M`. Transposing keeps only the operations that happen to be
symmetric — 12 of diamond's 48.

**P6 — Symmetry.** Point/space group detection, IBZ k-point reduction, density
symmetrization, and `ATOMIC_POSITIONS crystal_sg` (Wyckoff) expansion. *Check:* the
symmetry operation count and the reduced k-list match QE — including the `irrek` rotation
into the crystal's wedge described under P1, which is what makes the automatic-grid k-lists
comparable point by point — and the P5 energies are reproduced with symmetry on at lower
cost.

**P7 — Band structure. ✅ DONE.** `workflows/bands.py`: NSCF from a fixed converged
density, on an explicit k-path. *Check met:* silicon's bands along the 21-point path of
`pw_scf/scf-1.in` match QE to **0.0002 eV**, with the threefold degeneracies at Gamma
exact; the `nscf` run of `scf-2.in` matches on its own grid.

**P8 — DOS.** Smearing DOS and the tetrahedron method, on top of an NSCF grid run.
*Check:* against `dos.x` output on the same grid; the integrated DOS returns the electron
count.

**P9 — Spin.** LSDA (`nspin=2`), collinear magnetization. Non-collinear/SOC stays out.
*Check:* `pw_lsda` benchmarks.

**P10 — Performance and parallelism.** Profile, widen `jit` regions, `vmap` over k-points
and bands, buffer donation, k-axis sharding across CPU-cores-as-devices and across GPUs,
Numba `prange` on the setup hot spots. *Check:* a documented timing and scaling table; no
numerical drift from P5–P8 results.

**P11 — Higher-order autodiff quantities (after the first milestone).** Forces are already
validated at P5; here: stress by differentiation w.r.t. strain, implicit differentiation of
the SCF fixed point (D3), then polarization/dielectric response and second harmonic
generation. *Check:* stress matches QE to 1e-4 Ry/bohr³, and every response quantity has a
finite-difference test.

Ordering note: P6 (symmetry) can slip after P7/P8 if band structures come first, since
`nosym` runs are fully testable — but it must land before any timing claims, as it changes
the k-point count.

---

## 3a. Environment decisions (settled)

- Dependencies are installed into the **base anaconda env** (`pip install equinox`);
  equinox 0.13.8 is verified working with JAX 0.11.0 under x64.
- The repo **is** a git repository now. `quantum_espresso/` is gitignored — 285 MB of
  vendored reference does not belong in history — so any test touching it must skip
  cleanly when it is absent (`tests/conftest.py` does this).
- Pseudopotentials for the target tests are downloaded and **committed** under
  `tests/data/pseudo/` (10 UPF files covering `pw_scf`, `pw_atom`, `pw_metal`, `pw_lsda`).

---

## 4. Validation strategy

The primary test is **the same input run through QE and through pypresso**.

- QE's `test-suite/` ships inputs *with committed reference outputs* — use those first
  (`pw_scf/scf-*.in` plus `benchmark.out.git.inp=scf-*.in`), so no Fortran build is needed
  for the common cases. `io/qeref.py` parses them.
- Build QE only if a genuinely new comparison is needed (e.g. a `bands` or `dos` run not
  covered by a committed benchmark). Record how it was built and store the produced
  reference output next to the test so it never has to be regenerated.
- **The test-suite pseudopotentials are not shipped with QE.** Inputs reference files like
  `Si.pz-vbc.UPF` that `test-suite/check_pseudo.sh` downloads from
  `pseudopotentials.quantum-espresso.org`. They must be fetched once into a local
  `tests/data/pseudo/` and committed (they are small text files), or nothing is runnable.
- The canonical first target is `test-suite/pw_scf/scf.in`: Si in diamond structure,
  `ibrav=2`, `celldm(1)=10.20`, `ecutwfc=12`, LDA (`Si.pz-vbc.UPF`, norm-conserving),
  2 explicit k-points, 8 electrons, 15³ FFT grid, ~1459 G-vectors. Small enough to debug by
  hand, and its benchmark carries energy terms, eigenvalues, and stress.
- **The inputs in a test directory are a sequence sharing one `outdir`, not independent
  runs** — `test-suite/jobconfig` gives the order. For `pw_scf/` it starts
  `scf.in` → `scf-1.in` → `scf-2.in`, which are exactly SCF → `calculation='bands'` →
  `calculation='nscf'` on the same Si system, each restarting from the previous density.
  That single trio covers the whole first milestone with committed references, so it is the
  spine of the test suite: P5 targets the first, P7 the second, P8 builds on the third.
  A future session that treats `scf-1.in` as a standalone run will be confused by its
  missing `&electrons` convergence and its dependence on an existing charge density.
- Test tiers: `unit` (fast, analytic — FFT round-trip, Ewald for a known lattice, `Y_lm`
  orthonormality, Simpson accuracy, parser round-trips), `regression` (full runs vs QE
  references, marked slow), and per-quantity checks at the phase boundaries above.
- Tolerances are declared centrally, per quantity, not per test: total energy 1e-6 Ry,
  energy terms 1e-6 Ry, eigenvalues 1e-6 Ry, forces 1e-4 Ry/bohr, DOS 1e-3 states/eV.
  A test that needs a looser tolerance says why in a comment.
- Every phase's check is a committed test, not a one-off script.

---

## 5. Performance, parallelism, and GPU

Performance is a real requirement, not an afterthought. The rule is: never make a design
choice that forecloses good performance, but do not micro-optimize before P5 is numerically
correct. Optimization is P10, and it must not change any validated number.

**Precision — single precision must stay viable (decided).** The GPU target is hardware
where float64 is expensive, so float32 has to remain a usable mode. Two consequences:

- `jax.config.update("jax_enable_x64", True)` is still set before any array exists, and all
  *validation* against QE happens in float64. Single precision cannot reproduce QE to 1e-6
  Ry, so it is a performance mode, never the mode a correctness claim is made in.
- **Never hardcode a dtype.** No literal `jnp.complex128`, `dtype=float`, or `1.0j` in
  compute code. Real and complex dtypes come from a single policy object in `config.py`
  (`dtypes.real`, `dtypes.complex`) that is threaded through construction, and every array
  is created with an explicit dtype from it. Constants are written dtype-agnostically.
  Retrofitting this is a whole-codebase edit, which is why it is a rule from P0.

From P5 on, every regression test runs in float64; a small set also runs in float32 and
asserts only the looser tolerance, so the single-precision path cannot silently rot. Where
a reduction is accuracy-critical in float32 (density accumulation over bands and k-points,
subspace overlap matrices in the eigensolver), accumulate in float64 regardless of the
policy — this is a per-site decision to record in a comment, not a global switch.

**Static shapes.** Pad to `npwx`, fixed `nbnd`, fixed FFT grid. Anything shape-dependent is
decided during setup and passed as a static argument. Never branch on `npw_k` in compute
code — it retraces per k-point and kills both batching and sharding.

**Where the parallelism comes from.** JAX has no OpenMP pragmas; the equivalents are:

- *Intra-op threading (free).* On CPU, XLA runs each op on an internal thread pool, so
  large FFTs and GEMMs already use all cores. Control it with the standard thread-count
  env vars rather than in code. This covers most of what an OpenMP loop inside `h_psi`
  would have bought.
- *Batching — `vmap` over k-points and bands.* This is the primary mechanism and the reason
  R7 (padding, not ragged) exists: with a uniform `npwx` the k-index becomes a leading array
  axis, one big batched op instead of a Python loop. Do this first; it is also what makes
  the GPU path fast.
- *Sharding — the k-axis across devices.* `jax.sharding` with a mesh over the k-axis gives
  genuine parallelism, and the same code maps to (a) CPU cores exposed as devices via
  `XLA_FLAGS=--xla_force_host_platform_device_count=N`, and (b) multiple GPUs later,
  unchanged. Design for it now by keeping k as the leading, independent axis of every
  wavefunction-shaped array; actually enabling it is P10.
- *Numba `prange` for host-side setup.* G-vector enumeration/sorting, symmetry search, and
  radial table construction are irreducible loops that run once. `@njit(parallel=True)`
  there is real OpenMP and is the right tool. Every such function keeps a plain-NumPy
  reference implementation beside it for testing.

k-point parallelism is the natural top-level axis because k-points are independent within
an SCF iteration and only couple in `sum_band` — the same decomposition QE uses for its
pool parallelization. Anything else that is embarrassingly parallel (per-species radial
tables, per-band residuals, DOS tetrahedra) follows the same pattern: make it an array axis
first, and it becomes both `vmap`-able and shardable for free.

**Loop structure.** The SCF convergence test is data-dependent, so the outer loop stays in
Python; the iteration body (`h_psi` → diagonalize → `sum_band` → `v_of_rho` → mix) is
jitted as one unit. Inside the eigensolver, the inner iteration uses
`lax.while_loop`/`fori_loop` with a fixed subspace size so the solver stays on device.

**Avoid host syncs in the inner loop:** no `.item()`, no `float()`, no Python `if` on device
values, except the single once-per-iteration convergence check. Use `donate_argnums` for
the large wavefunction and density buffers.

---

## 6. Differentiability and response properties

Autodiff is a primary reason for choosing JAX, not a side benefit. The goal is that
quantities like forces, stress, polarization, dielectric response, and second harmonic
generation come from differentiating the code rather than from separately derived and
separately debugged expressions. That imposes requirements from the very first phase,
because differentiability is nearly impossible to retrofit.

**D1 — The whole compute path must be differentiable.** Every function from inputs to
observables is a pure JAX function. No NumPy, no Numba, no `libxc` C calls anywhere a
gradient has to flow. This is why the XC functionals are written in JAX rather than bound
from `libxc`: XC needs to be differentiated once for the potential, twice for the kernel,
and more for higher-order response — `grad` of the energy density is strictly better than
hand-coded `v_xc`/`f_xc` routines, and it is the same code on GPU. This does not conflict
with using Numba for setup: setup produces *constants* (radial knot values, G-vector index
maps), and gradients w.r.t. `k`, positions, or fields flow through the JAX interpolation
and structure factors built on top of them, never through the Numba code itself.

**D2 — `k` stays a traced argument of the Hamiltonian.** The velocity/position operator is
what response theory needs, and for a nonlocal pseudopotential `[H, r] ≠ p` — QE hand-codes
this in `commutator_Hx_psi.f90`. If `H(k)` is built by differentiable JAX code from `k`
(rather than looked up from precomputed host tables), then `jacfwd` of `H(k)` w.r.t. `k`
gives the velocity operator exactly, nonlocal contributions included, for free. This is a
concrete constraint on P2–P4: `|k+G|²` and the projectors `vkb(k)` are *computed* in JAX
from `k`, with the radial form factors interpolated differentiably (JAX splines), not
gathered from a per-k table built during setup.

**D3 — Differentiate the SCF fixed point implicitly, never by unrolling.** Backpropagating
through the SCF iterations costs memory proportional to the iteration count and gives
inaccurate gradients. Use the implicit function theorem: define the converged density
through a `custom_vjp` whose backward pass solves the linear response equation
(the Dyson/Sternheimer equation) with an iterative solver. `scf/driver.py` must be
structured so the converged result is the output of one function with a well-defined
residual `R(rho, params) = 0` — this shapes the driver's signature, so it has to be decided
at P5, not later.

**D4 — Beware `eigh` gradients.** The derivative of an eigendecomposition is singular at
degeneracies, which crystals have everywhere by symmetry. Do not differentiate through the
diagonalization. Formulate response quantities with degeneracy-safe constructions instead:
projectors onto occupied subspaces (invariant under rotations within a degenerate manifold),
and Sternheimer-style linear solves for the first-order wavefunctions. Any observable
written for response must be checkable against a finite-difference reference in a test.

**D5 — Differentiability is tested, not assumed.** From P4 onward, each phase adds a test
that a representative gradient matches central finite differences. Cheapest useful ones:
`d(total energy)/d(atomic position)` against the QE force (this validates autodiff forces
against QE's Hellmann-Feynman + Pulay forces — a strong independent check on both), and
`d(eigenvalue)/dk` against a finite-difference band velocity.

**Deferred but planned.** SHG and other higher-order optical responses come after bands and
DOS. They need: the velocity operator (D2), degeneracy-safe formulations (D4), and either
sum-over-states or a Sternheimer solver. Nothing in P0–P8 should make them harder — which
in practice means D1, D2, and D3 are respected as the earlier phases are written.
