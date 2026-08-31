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

**R10 — A derived `Calculation` owns what it inherits.** `at_positions`, `at_strain`,
`at_kpoints`, `at_kcart` and `at_spiral_q` are `copy.copy` plus a rebuild of the parts that
moved, so *everything not explicitly rebuilt or dropped is silently shared* — including
attributes cached long after the method was written. Every such bug found so far has the
same signature: no exception, no shape error, a plausible number computed for the wrong
geometry. The audit that found them is mechanical and worth repeating whenever one of these
methods gains a line — list every `self._x = ...` assignment on `Calculation` and, for each,
say which derivations must drop it:

* a **compiled kernel that closed over the calculation** must be dropped by every derivation,
  because `jit` baked the captured arrays in as constants. `_analytic_terms` and the stress's
  `_strain_gradient`/`_strain_term_gradients` were not — the latter because `at_strain`
  popped `_energy_gradient`, which is the *force's* cache under a similar name, so the
  docstring's claimed invariant had never been true. They are keyed on the calculation they
  captured now, which is invalidation by identity rather than by remembering to add a pop.
* a kernel that takes the moving quantity as an **argument** is the exception and must not be
  keyed that way, or a relaxation recompiles at every step: `forces/autodiff.py` takes the
  positions, so it crosses `at_positions` legitimately — but it still closes over the k-set,
  which is why it is dropped by `at_kpoints`/`at_kcart`.
* a **host-side table** cannot be rebuilt under tracing at all. `_tetrahedra` is the k-grid's
  own object and was dropped by nothing; the magnetic field's `LocalRegions` are the atoms'
  and were built once in `__init__`. The rule for those is the spiral sphere's (P21): rebuild
  on a concrete move, freeze while differentiating, which is exact because the assignment is
  piecewise constant in the geometry.

The same reasoning applies to **refusals**: a guard in `io/` only protects inputs that came
through `io/`. `spiral_q` is installed by `dataclasses.replace` in `at_spiral_q` and in
`workflows/spiral`, and a `System` can simply be constructed, so the spiral-needs-`nosym`
refusal had to move to `Calculation.__init__` where the symmetry decision is actually made.

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
    spiral.py           # P19: the spin-spiral q, the two shifted k-lists, its symmetry
  basis/
    gvectors.py         # G enumeration/sorting, cutoff spheres, per-k index maps
    fft.py              # sphere<->FFT-box gather/scatter, jnp.fft wrappers
    gradients.py        # grad and div in G space (what a GGA potential needs)
    wavefunctions.py    # padded (nspin, nk, nbnd, npwx) container + masks
  pseudo/
    radial.py           # radial grids, Simpson integration, spherical Bessel
    harmonics.py        # real spherical harmonics Y_lm(G) and derivatives
    local.py            # vloc(G), rho_atomic(G), core charge
    projectors.py       # beta projectors vkb(k), D_ij coefficients
    tables.py           # PseudoPotential dataclass + per-species G tables
  xc/
    functional.py       # QE's four slots, composed; name -> functional registry
    lda.py, gga.py      # Slater/PZ/PW, and the PBE family's gradient corrections
  projwfc/
    channels.py         # P8: what the natomwfc projection columns are (fill_nlmchi)
    projections.py      # P8: <phi|S|psi>, and sym_proj_k's group average of it
  hubbard/
    manifold.py         # P20: which orbitals carry U, with what parameters (setup)
    projectors.py       # P20: wfcU = S phi, or O^{-1/2} S phi (orthoUwfc.f90)
    occupations.py      # P20: ns from the states, its symmetrisation, init_ns/ns_adj
    energy.py           # P20: E_U written down; v_ns is its grad (v_hubbard is the check)
    operator.py         # P20: the separable term vhpsi.f90 adds to H
  hamiltonian/
    terms.py            # kinetic, local, nonlocal as composable term objects
    operator.py         # Hamiltonian pytree; apply_h / apply_s
  forces/
    energy.py           # the stationary functional the force is the gradient of
    autodiff.py         # -grad of it (the default)
    analytic.py         # QE's six hand-derived terms, as a cross-check
    spiral.py           # P21: the same, with the spiral q as the coordinate
    registry.py
  relax/
    bfgs.py             # bfgs_module.f90: trust radius + Wolfe line search
    registry.py         # ion_dynamics
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
    locals.py           # P17: pointlists, local moments, report_mag
    fields.py           # P18: the external/local field energy (its potential is grad)
    constraints.py      # P18: constrained magnetization / fixed spin moment, registry
    driver.py           # the SCF loop
  workflows/
    scf.py, nscf.py, bands.py, dos.py
    pdos.py             # P8: projwfc.x -- the projected DOS and the Lowdin charges
    spiral.py           # P19: an E(q) scan; P21: relaxing q down its gradient
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
  2×2×2 grid.) A point-by-point comparison of automatic grids is therefore not the right
  test even now that P6 reduces them: what must agree is the number of orbits and their
  weights, which it does on all 22 cases.

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
- **The FFT grid is not set by the cutoff alone, and the first explanation for that was
  wrong.** `pw_scf/scf.in` and `pw_scf/scf-cg.in` are the same system at the same cutoffs
  and both report 1459 G-vectors, but their committed grids are 15³ and 16³. The guess
  recorded here originally was that 15 is disallowed on IBM ESSL and the two references
  came from different builds. It is not: **QE requires the FFT dimensions to be a multiple
  of the denominators of the crystal's fractional translations** (`fft_fact` in
  `PW/src/symm_base.f90`), because a grid maps onto itself under a translation of `1/n`
  only if it has a multiple of `n` points along that axis. Diamond silicon's are `1/4`, so
  15 is rounded up to 16. The rule postdates the committed references
  (`REFERENCE_VERSION 6.0`); running the vendored `pw.x` on `pw_scf/scf.in` prints 16³
  today, which is how it was settled.

  It is not cosmetic. `etxc` is evaluated pointwise on the grid, so a different grid is a
  different exchange-correlation energy in the sixth decimal — ~1e-6 Ry, a hundred times
  the agreement this project claims. Stacked on it is a second rule: if the identity plus
  a non-lattice translation already maps the structure onto itself, the cell is a
  *supercell* and QE disables fractional translations entirely, which for the eight-atom
  cubic silicon cell means 24 operations rather than 48 and no divisibility constraint.
  Both are implemented (`Symmetries.fft_factors`, `is_supercell`), and where a committed
  benchmark predates them the reference is regenerated with the vendored `pw.x`
  (`tools/generate_reference.py`).
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

**P4 — Hamiltonian and diagonalization. ✅ DONE.**
`hamiltonian/operator.py` (kinetic + local via FFT + nonlocal, plus `apply_s`),
`solvers/` — a name registry holding `davidson.py`, transcribed from `cegterg.f90`.
*Check met:* eigenvalues match QE to <1e-3 eV wherever they are printed, and Davidson
matches an exact diagonalisation of the same Hamiltonian to 1e-12 Ry. That exact solve is
a **test fixture** (`tests/exact_reference.py`) and not a solver the package offers:
`O(npw^2)` memory and `O(npw^3)` time is precisely what an iterative solver exists to
avoid, correctness here comes from Quantum ESPRESSO input for input, and a name in the
registry is an invitation to select it. `Hamiltonian` does keep two matrix
builds, one from matrix elements and one from applying the operator, and the suite asserts
they agree on the single-k and the padded multi-k silicon cases. The fast build is used
only where the density grid resolves every `G - G'` (`ecutrho >= 4 ecutwfc`) and the full
sphere is stored; gamma-only falls back to applying the operator, since a difference of
two stored half-sphere G's need not be stored at all.

**The traps:** (1) a **converged root must stop being expanded** — its residual is ~1e-14,
and normalising that to unit length turns round-off into a basis vector, making the
overlap matrix singular so the *other* roots stop converging. (2) The subspace must grow
by the number of **unconverged** roots, not by the block size, or the periodic collapse
discards a stubborn root's search direction first. `cegterg` avoids both by compacting
unconverged roots to the front; under `jit` the same compaction is a stable `argsort`.
Both traps showed the identical symptom — silicon's highest band a few meV off, nothing
else wrong. (3) `c_bands.f90` re-enters `cegterg` up to five times, so QE's real iteration
budget is `maxter × ntry = 100`, not the 20 the solver itself declares.

**P5 — Full SCF. ✅ DONE.** `scf/` — `v_of_rho` (Hartree + LDA), `sum_band`,
occupations for every QE smearing plus `from_input`, Anderson mixing, Ewald, and the
driver with QE's energy decomposition. `xc/lda.py` writes only the energy density and
gets `v_xc` from `jax.grad`. *Check met:* **silicon's total energy matches QE to
1.1e-8 Ry** and the metals to ~2.5e-8 Ry, term by term, across 8 regression cases.
*(Since P12 corrected the FFT grid — see the P2 note on `fft_fact` — and the stale QE 6.0
references for these cases were regenerated with the vendored `pw.x`, silicon agrees to
**under 1e-9 Ry** and the metals to 5.6e-9. The number above is what P5 measured against
the references it had.)*

**The traps:** (1) XClib returns **Hartree**, not Rydberg — `v_of_rho` multiplies by
`e2`, and missing that halves the XC energy. (2) **Density symmetrisation is not
optional**: a symmetry-reduced k-point set gives an unsymmetric density, which
converges happily and splits degenerate levels by tens of meV. That forced P6 forward.

**P6 — Symmetry. ✅ DONE** (bar Wyckoff input). `system/symmetry.py` finds the space
group (48 operations for diamond silicon, non-symmorphic, matching QE) and symmetrises
the density; `kpoints.irreducible_wedge` reduces an automatic grid, transcribed from
`kpoint_grid.f90` and applied in `build_system` where `setup.f90` applies it. *Check met:*
the reduced count matches QE on all 22 automatic-grid cases in the test suite — every
Bravais lattice, including the triclinic ones — and the eigenvalue comparison for
`scf-kauto.in`, which used to be skipped for want of this, now runs. Worth 2.8x on that
input and 8.5x on an 8×8×8 grid. *Still to do:* `crystal_sg` (Wyckoff) input.

**Which operations survive, and what a shifted grid does to them** (measured at P15, on
the displaced silicon of `tests/data/qe/si2-nc-force.in`). Two facts that look like bugs
and are not, and one consequence that is worth knowing before trusting the fifth decimal
of a shifted-grid run:

* **This code keeps symmetry operations QE discards.** QE drops an operation whose
  fractional translation is not commensurate with the FFT grid, because it symmetrises the
  density in *real space* and cannot represent such a translation there. This code
  symmetrises in **G space**, where a translation is a phase and is exact for any value, so
  it keeps them. On the displaced cell that is 8 operations against QE's printed
  "4 Sym. Ops.", and 20 irreducible k-points against QE's 40. Neither count is wrong; they
  are answers to slightly different questions.
* **A shifted Monkhorst-Pack grid is not invariant under every operation the crystal has.**
  On the displaced cell, 4 of the 8 map the shifted 4x4x4 grid onto itself and 4 do not.
  The k-point reduction handles this correctly on both sides -- an operation that carries a
  point off the grid simply merges nothing -- but the **density symmetrisation** does not
  ask the question at all, in either code: it applies the crystal's operations to a density
  built from a sample that does not have all of them.
* **The consequence is 1e-4 Ry, and it is a choice rather than an error.** Symmetrised, the
  two codes give -15.80141873 (here, 8 operations) and -15.80131502 (QE, 4). With `nosym`
  on the full 64-point grid they give **-15.80140078 and -15.80140078** -- identical to
  every digit either prints, which is what `tests/regression/test_forces.py`'s shifted case
  pins. So the SCF, the basis and the k-point generation agree exactly and the spread is
  entirely in which operations each code symmetrises with. Making the symmetrisation use
  only the subgroup the k-sample respects would make a symmetric run agree with its own
  `nosym` run exactly; it would also stop matching QE. That trade has not been made, and it
  should not be made without measuring what it does to the 22 automatic-grid cases.

**The trap:** QE reduces with the point group of the *Bravais lattice* and then remaps the
representatives into the wedge of the crystal's group, which can carry them off the grid
entirely. Reducing directly with the crystal's symmetries — the same operations the
density symmetrisation uses — lands on the same orbits without the detour, and is what
makes the counts agree. It is also the only version that is *safe*: the lattice group
over-reduces a crystal whose symmetry is lower than its lattice's.

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

**P8 — DOS. ✅ DONE.** Smearing and tetrahedron Brillouin-zone integration behind a name
registry, on top of an NSCF grid run lifted out of `workflows/bands.py` into
`workflows/nscf.py`, plus `dos.x`'s `.dos` file and a `pypresso dos` subcommand. The
tetrahedron method is also an *occupation* scheme inside the SCF, which is where the hard
numbers are. *Check met:* all three of QE's fcc-aluminium benchmarks in `pw_metal`, which
between them cover every variant — `metal-tetrahedra.in` (`tetrahedra-opt`, an SCF) to
**2.5e-8 Ry** in the total energy and 0.0002 eV in `E_F`; `metal-tetrahedra-1.in`
(`tetrahedra`) and `metal-tetrahedra-2.in` (`tetrahedra-lin`), NSCF on 6×6×6 from that
density, reproducing 8.3056 eV and 8.2622 eV **exactly to the four decimals QE prints**.
The three differ by 40 meV, so matching all three is a real three-way check. No `dos.x`
reference is committed anywhere in the test suite, so the DOS itself is held to its sum
rules: `∫D = N` by construction, `N` = 7.99999995 at silicon's valence-band maximum,
`N(E_F)` = 3.000001 for aluminium, zero states inside silicon's gap, and a √E aluminium
DOS to 5%.

Design decisions, each one a JAX consequence rather than a transcription choice:

* **Only `N(E)` is written down.** `sumkt`, `tetra_dos_t`'s `dosint` and
  `opt_tetra_dos_t`'s are one piecewise cubic; `D(E)` is `jax.grad` of it and QE's matched
  `dost` is never transcribed. Likewise `w0gauss` is `jax.jvp` of `wgauss` — QE's own
  docstring says it is that derivative, and taking it means the delta cannot drift out of
  step with the occupation function the SCF uses. Same pattern as `xc/functional.py`.
* **The Fermi search stays on device**: a fixed-step `fori_loop` mirroring
  `occupations.py`'s `bisect_fermi`, which could not be reused because its count function
  is `wgauss` and the count is the entire difference.
* **The construction is host-side numpy**, producing a static `(ntetra, nntetra)` table
  the compiled path only gathers through.
* **The spin degeneracy is `sum(wk)`** — 2 unpolarised, 1 per channel — so nothing in
  `scf/tetrahedra.py` needs an `nspin` axis. `tetra.f90` otherwise never looks at the
  k-point weights: the tetrahedra carry the Brillouin-zone measure themselves.

Transcription traps, in the order they cost time:

1. **The benchmark names lie about which algorithm they test.** `set_occupations.f90` is
   the authority: `metal-tetrahedra.in` is `tetrahedra-opt`, not the plain method, and the
   two numbered files are NSCF continuations of it (QE's `jobconfig` runs them in one
   `outdir`), not independent SCF runs.
2. **Bloechl and Kawamura do not cut a microcell into the same six tetrahedra.**
   `tetra_init` hardwires one decomposition along the n1–n8 diagonal; `opt_tetra_init`
   picks the *shortest* of the four body diagonals as its shaft, encoded in a fourth,
   otherwise unused component of `ivvec0`/`divvec`. Using one family's decomposition with
   the other's weights gives a plausible answer that is wrong in the third decimal of
   `E_F`.
3. **The sort permutation indexes `wlsm`'s first axis**, not its second, in
   `opt_tetra_weights_only`'s scatter — also a plausible wrong answer.
4. **The NaN-in-`grad` trap.** QE's `IF/ELSEIF` chain becomes evaluate-all-branches plus
   `jnp.where`, and degenerate corner energies (routine: any high-symmetry point, any flat
   band) make the *dead* branches divide by zero. The forward value survives the `where`;
   the gradient does not, because `where` hands the dead branch a zero cotangent and
   `0 * inf` is NaN. Denominators are therefore clamped *before* the division, which is
   exact — a branch spanning `e_i ≤ E < e_j` is empty precisely when `e_i == e_j`.
5. `tetrahedra-opt`'s stencil has **negative weights**, so its corner energies can fall
   outside the range of the eigenvalues themselves. Its integrated DOS therefore reaches
   the electron count only to ~1e-3 on the grid `dos.x` sizes from the eigenvalues, and a
   band edge leaks a little weight across a gap. Both are properties of the method, not
   defects, and QE behaves the same way.
6. `opt_tetra_weights_only` averages the weights of degenerate bands afterwards. QE's
   version is a sequential scan comparing each band to the *first* of the group it is
   building; the symmetric pairwise mean used here agrees whenever the relation is
   transitive, which at 1e-6 Ry it is, and both preserve the total.

**P8 x P9 — what integrating them needed.** Both schemes stay per channel; the workflow
loops over them and the `.dos` file grows a `dosup`/`dosdw` pair with one summed
`Int dos`, as `dos.f90` writes it. Two things were not mechanical:

1. **The tetrahedron Fermi level is found from both channels together, and `sumkt` is
   where that is written down.** With `is = 0` and `nspin = 2` it loops over both channels
   accumulating `1/ntetra` from each and applies its factor of two *only* when
   `nspin == 1`. So the count whose root is `E_F` is the sum over channels weighted by one
   each — not two independent searches for half the electrons, which is a different
   physical problem and would fix the magnetization at zero. The degeneracy bookkeeping
   then needs no special case, because the per-channel core already reads it off
   `sum(weights)`: 2 unpolarized, 1 per polarized channel, exactly what `sumkt` applies and
   withholds. Constraining the magnetization *is* two independent searches, and
   `weights.f90` supports it for tetrahedra as it does for smearing (`is = 1`/`is = 2` with
   `nelup`/`neldw`), so it is implemented rather than refused.
2. **A second k-set is where the spin weight convention goes wrong.** Every `KPoints`
   constructor applies the spin degeneracy unconditionally and `build_system` divides it
   out again for `nspin = 2`; a grid built *later* — which is precisely what a denser DOS
   grid is — never passed through that step and counted every electron twice. Nothing
   raised: the density of states still integrated to ten electrons, at a Fermi level 2.3 eV
   too low. `system.kpoints.for_spin` is now the one place that knows the rule and both
   callers go through it. `pw_lsda/lsda-2.in` pins it: nickel's Fermi level on the 8x8x8
   NSCF grid comes out **15.3379 eV against QE's 15.3379**.

**P8 x projwfc — the projected density of states. ✅ DONE.** The remainder of P8:
`pypresso/projwfc/` (what the projection columns are, and the projection itself),
`workflows/pdos.py` (the integration, the Löwdin charges and the workflow), the
`filpdos` writer in `io/output.py` and a `pypresso pdos` subcommand. What is
computed is `PP/src/projwfc.f90`'s

    proj[i, n, k] = |<phi_i| S |psi_nk>|^2,   phi = O^{-1/2} S chi,  O_ij = <chi_i|S|chi_j>

resolved by atom, by `l` and by `m`, then integrated by whichever scheme the run's
own occupations name (`partialdos.f90`), and integrated against the occupations
instead of a delta to give `print_lowdin`'s charges and Sanchez-Portal's spilling
parameter.

*Check met:* **seven cases against the vendored `projwfc.x`**, which had to be
built (`make pp`) and whose references are committed — QE's test-suite has no
`projwfc` case at all, no input and no `filpdos` file anywhere in the tree, so
this is the first phase with nothing to borrow. Three quantities per case, at
increasing distance from the projection:

* **the projections themselves**, band by band and k-point by k-point against
  `print_proj`'s listing: **≤ 6.9e-4**, which is the resolution of its own
  three-decimal printout and not a bound on the agreement;
* **the Löwdin charges** per atom, per `l` and per `m`, and the spilling:
  **≤ 4.7e-5** and **≤ 4.8e-5** against `f8.4`;
* **the density of states itself** on `partialdos`'s grid, every `filpdos` column
  of every file: **≤ 0.8% of the peak**, which is what `e11.3` — three
  significant digits — supports, and which the eigenvalues put the same floor
  under anyway (see trap 6).

The cases cover every path through the projection: norm-conserving silicon with
fixed occupations (`pw_scf/scf.in`), the same cell ultrasoft and PAW (`si2-us`,
`si2-paw`, where `S` is not the identity), ultrasoft silicon on an 8x8x8 wedge
(`si2-us-dense`, which is where the symmetrisation has something to do),
spin-polarized nickel with a Marzari-Vanderbilt smearing (`pw_lsda/lsda.in`), and
aluminium with **both** tetrahedron families (`pw_metal/metal-tetrahedra.in` and a
new `al-tetrahedra.in`, which exists for trap 1).

Design decisions, in the pattern the rest of the code follows:

* **A projected density of states is the plain one with a weight in front of the
  delta**, so it goes through the *same* registry rather than a second
  implementation: every entry of `DOS_SCHEMES` takes an optional `projections`
  argument, `(nk, nbnd, nproj)`, and grows a trailing channel axis on both
  returned arrays. Both families implement it, which is what makes
  `sum_p D_p = D` exact for a unit weight — asserted to 1e-12 (smearing) and
  1e-10 (tetrahedra) in `tests/unit/test_projwfc.py` — rather than something to
  hope for. QE's own `partialdos` is two disjoint code paths for the same reason
  it has `dosint` and `dost`.
* **Only the integral is written down here too.** The tetrahedron version
  resolves the occupation weight *per corner* (`_linear_weights` + `_scatter`,
  which is `opt_tetra_weights_only`'s machinery) into
  `N_p(E) = sum_kb w_kb(E) proj[k, b, p]` and takes `jax.jvp` of that in `E` —
  a vector of `nproj`, so `jvp` and not `value_and_grad`. QE's
  `opt_tetra_partialdos`, a fourth copy of the same four-branch chain, is never
  transcribed.
* **The projectors are DFT+U's.** `orthoUwfc` and `projwave` build the same
  object, so `build_hubbard_projectors` was generalised into
  `build_atomic_projectors(..., kind, columns)` and the Hubbard version is now a
  column selection out of it. `projwfc.x` has only `ortho-atomic`; `atomic` and
  `norm-atomic` are reachable here because `pw.x` spells them for the Hubbard
  manifold and they cost nothing extra.
* **The primary path projects the SCF's own states.** `projwfc.x` reads the
  wavefunctions `pw.x` left in its `outdir` and diagonalises nothing, and
  `run_pdos` follows it. That is also the only route open to a PAW dataset,
  since a fixed-density re-diagonalisation needs a `becsum` that is not carried
  across (P12) — and it costs nothing, because re-solving the same Hamiltonian
  at the same k-points gives back the same states. `grid=` still runs an NSCF on
  a denser grid, for the same reason a DOS does.
* **`pypresso/projwfc/` sits below the workflows and does not import them**
  (rule R3). The projection is a `pseudo`/`scf`-level object; the density of
  states built from it needs `DOS_SCHEMES`, so it lives in `workflows/pdos.py`.
  Putting both in one package is an import cycle, which is how the layering rule
  announces itself.

Transcription traps, in the order they cost time:

1. **`do_projwfc` silently promotes `tetra_type = 0` to 1.** A projected density
   of states asked for with `occupations = 'tetrahedra'` is computed with the
   **linear** method, never Bloechl's — and the two are not the same weights on
   the same tetrahedra, they are *different tetrahedra* (P8, trap 2: Bloechl
   fixes one body diagonal, the linear family picks the shortest of the four).
   So the substitution has to happen where the tetrahedra are built, and
   `tetrahedron_projected_dos` refuses a Bloechl decomposition rather than
   differentiating a curvature correction that is not the derivative of an
   occupation. `al-tetrahedra.in` exists to pin it: the Löwdin charges there come
   from Bloechl's occupations and the integrated PDOS from the linear method, and
   they differ by **0.10 electrons of 3** — QE is inconsistent in exactly the
   same way and by the same amount.
2. **`partialdos` writes one energy point more than `dos.f90`.** Both compute
   `ne = nint((Emax - Emin)/DeltaE + 0.500001)`; `dos.f90` then writes `1..ndos`
   and `partialdos` writes `0..ne`. A comparison against `filpdos` that reuses
   the DOS grid is off by a row at one end, and looks like a half-step shift.
3. **`S` is applied even for `atomic` projectors** — P20's first trap, unchanged
   and equally silent here, because `projwave` calls `s_psi` on `wfcatom` before
   anything else. Norm-conserving silicon cannot see it, which is why
   `si2-us`/`si2-paw` are in the case list.
4. **The Löwdin charge is weighted by `wg`, which already carries `w_k`.**
   `print_proj` multiplies by `wg(ibnd, ik)` and by nothing else. Using the
   occupation `f` without the k-point weight gives a number that is wrong by the
   size of the irreducible wedge and still looks like a charge.
5. **`sym_proj_k` is on by default and is not cosmetic.** It averages
   `|sum_m' D^l_S[m', m] proj0[S(a), m']|^2` over the group — the same
   `d_matrix` machinery PAW's `becsum` symmetrisation uses
   (`paw/symmetry.harmonic_rotations`), with the atom index following `irt`.
   Because `D` is orthogonal the sum over `m` is invariant, so a missing
   symmetrisation leaves every per-`l` charge and the total *right* and every
   per-`m` charge wrong: on the 8x8x8 wedge silicon's three `p` channels come out
   0.9567/0.9201/0.9283 where symmetry says three times 0.9350, and their sum is
   2.8051 either way. The per-`m` columns of `filpdos` are what catches it.
6. **Each code sizes the energy grid from its own band extremes.** `Emin` is the
   lowest eigenvalue less `3 degauss`, and the two codes' lowest eigenvalues
   differ by 2e-4 eV — so the two curves are sampled at points 5e-4 eV apart, and
   a 0.05 eV Gaussian whose peak is 17 states/eV has a slope of 200 states/eV^2.
   That is a 0.13 states/eV difference that is not a disagreement about anything,
   and it made the first comparison look 1% worse than it is. The regression test
   pins `emin`/`emax` to the reference file's and says why.
7. **`opt_tetra_partialdos` has no degenerate-band average**, where
   `opt_tetra_weights_only` does. Reusing the occupation weights would move
   weight between two bands that cross inside a tetrahedron and carry *different*
   projections, which changes the answer rather than symmetrising it. Worth 4e-5
   electrons on aluminium — small, and in the wrong direction for the wrong
   reason.
8. **`fill_nlmchi`'s `n` counts the orbitals in the file, including the skipped
   ones**, and it is what a file name carries (`pdos_atm#1(Si)_wfc#2(p)`). It is
   *not* the index among the kept orbitals, which is what indexes the radial
   tables. A dataset with a negative-occupation channel makes the two differ.
9. **The `m` order inside a shell is `ylmr2`'s**, so `l = 1` is `(z, x, y)` and
   `l = 2` is `(z2, xz, yz, x2-y2, xy)`. `print_lowdin`'s `lm_label_global_frame`
   is the authority and is transcribed; a table headed `px` that holds `pz` is
   wrong in a way no total ever shows.

*Refused, by name rather than ignored:* noncollinear and spin-orbit projections
(`atomic_wfc_nc_proj`, `sym_proj_so`, `partialdos_nc` — the projector set is
spinor and `natomwfc` doubles, so it is a different set rather than a wider one).
`atomic_projections` raises on `noncolin` before it builds anything, and
`projection_channels` — which cannot see the flag, taking only the
pseudopotentials and the structure — names the branch it does not implement in
its docstring rather than silently labelling half the columns. Not implemented
and unreachable by any input: `pawproj` (`projwave_paw`, projecting on the PAW
all-electron partial waves), `tdosinboxes` (`projwave_boxes`, the local DOS in a
real-space box), `diag_basis` (`rotate_basis`), `kresolveddos`, `lforcet` (the
force theorem), and the `atomic_proj.xml`/`lwrite_ovp` outputs.

*Notebook 16.*

**P9 — Spin. ✅ DONE.** LSDA (`nspin=2`), collinear magnetization. Non-collinear
wavefunctions and spin-orbit coupling came later, in P14, and reused this axis rather than
widening it. The density, the potential, `becsum`, `D_ij`, the eigenvalues and the
wavefunctions all grew a leading channel axis, with `k` still the leading *independent*
axis inside each channel. `xc/` gained the polarized correlation parameterisations (PZ,
PW92 and PBE's) and nothing for exchange, which the spin-scaling relation supplies
exactly; `scf/occupations.py` gained `set_nelup_neldw`, two Fermi levels, and — see the
traps — `efermig`'s root selection; `paw/` grew a spin index through the one-centre terms.
*Check met:* **eight spin-polarized benchmarks**, all generated with the vendored `pw.x`
at `conv_thr = 1e-10`:

| case | what it isolates | total energy |
|---|---|---|
| `pw_atom/atom-lsda` | the whole plumbing, no Fermi search | **4.8e-9 Ry** |
| `pw_lsda/lsda` | fcc Ni, one shared Fermi level | **1.2e-9 Ry** |
| `pw_lsda/lsda-tot_magnetization` | two Fermi levels | **1.2e-9 Ry** |
| `pw_lsda/lsda-nelup+neldw` | the same constraint, spelled `2.0` | as above, to 1e-10 |
| `pw_atom/atom-sigmapbe` | spin-polarized PBE | **1.8e-9 Ry** |
| `o-paw-spin` | PAW one-centre terms per channel | **2.0e-7 Ry** |
| `o-paw-spin-pbe` | the same with a gradient correction on the sphere | **2.0e-7 Ry** |
| `pw_pawatom/paw-atom_spin_lda` | the LDA case QE ships | 4.1e-6 Ry |

Magnetizations match the two decimals QE prints them with throughout: oxygen's
2.0000000001 against 2.00, nickel's **0.7280 against 0.73** with an absolute moment of
0.7842 against 0.78. Nickel's Fermi energy matches to the 0.0001 eV it is printed with,
its two constrained Fermi levels likewise, and its `-TS` term to 7.5e-9 Ry. The
unpolarized path did not move at all: silicon norm-conserving/ultrasoft/PAW and LDA/PBE
reproduce their references to the identical last digit they did before the spin axis
existed, and `PERFORMANCE.md` records that a length-one spin axis costs nothing.

**The last two rows need their own paragraph, because the looseness is the cases' and not
this code's.** `o-paw-spin` and `o-paw-spin-pbe` are new inputs committed under
`tests/data/qe/`; the two QE ships put the minority channel's electron into *one* of the
three 2p orbitals — which is why they set `nosym` — and that leaves the state nearly
degenerate under which orbital it is. QE needs 71 iterations on the LDA one, fails to
converge at all at `mixing_beta = 0.3`, and moves its own answer by 5.8e-7 Ry between
`mixing_ndim = 8` and `4`; the PBE one is slow enough in both codes that it is excluded
from the suite, with `o-paw-spin-pbe` covering the identical code path instead. The two
new inputs change nothing but the occupations and land at 2e-7.

**And there is a control that settles what the residual is not.** Run the *unpolarized*
PAW oxygen (`pw_pawatom/paw-atom_lda`, which this code reproduces to 2.2e-9 Ry) with
`nspin = 2` and the two channels occupied equally. Every spin path executes — two
Hamiltonians, `becsum` and `D_ij` per channel, the polarized functional evaluated at
`zeta = 0`, the one-centre terms per channel, the magnetization term in `dr2` — and the
answer moves by **2.0e-12 Ry**. Whatever the remaining 2e-7 is on a genuinely polarized
isolated atom, it is not arithmetic in the spin path. The check is committed as a test.

**The traps:**

1. **`efermig` does not bisect with the smearing it is given, and the reason is a
   non-monotonic occupation.** Cold and Methfessel-Paxton occupations *overshoot* — a
   cold-smeared level reaches 1.07 before settling at 1 — so `N(E_F) = nelec` has several
   roots and a plain bisection lands on whichever its bracket selects. QE bisects with a
   **Gaussian** first, which is monotonic, and then refines with Newton's method on the
   real occupation function. The unpolarized metals never showed it because their count
   function is never flat; nickel with `tot_magnetization = 2` has a majority channel that
   is nearly full, flat to 1e-5 electrons over a whole eV, and the wrong root sits 0.74 eV
   away — same density, same occupations to 1e-5, `-TS` and the total energy out by
   3e-4 Ry. (Newton's two derivatives of the count come from `jax.grad` of the count,
   where QE hand-writes `sumkg1` and `sumkg2`.)
2. **Three quantities are the *total* density's and three are per channel, and QE's array
   layout hides which is which.** `rho%of_r(:,1)` is the total and `(:,2)` the
   magnetization, but `sum_band` accumulates in `(up, down)` and converts afterwards
   (`rhoz_or_updw`), and `gradcorr` converts back. What is genuinely a functional of the
   total alone: the Hartree potential (copied to both channels), the PAW one-centre
   Hartree (`PAW_h_potential` sums over spin before solving), and the gradient
   correction's *correlation* half — which is why `gcc_spin` takes one `grho` where
   `gcx_spin` takes two.
3. **`rho_ddot` grows a second term with no `1/G^2` in it.** The magnetization enters
   `dr2` with a G-independent weight `e2 4pi/(2pi)^2` and with its `G = 0` component
   *included*, where the Hartree half excludes it. An error in the total charge is
   expensive in proportion to its wavelength; an error in the magnetization is expensive
   at every wavelength equally, and a uniform shift of the magnetization is a real error
   where a uniform shift of the charge is forbidden by neutrality. This feeds `conv_thr`
   and the `ethr` schedule, so at `conv_thr = 1e-10` it is not cosmetic.
4. **`nosym` is a correctness switch, not an optimisation.** An input whose occupations
   break the crystal's symmetry — an atom with one of its three p channels filled — needs
   it, and symmetrising the density or `becsum` anyway averages the three and converges
   somewhere else. Both of the PAW spin benchmarks QE ships set it.
5. **`set_nelup_neldw` truncates the electron count before splitting it.** With an integer
   charge *and* an integer magnetization it is `(INT(nelec) ± m)/2`, and the default when
   `tot_magnetization` is absent is `INT(nelec + 1)/2` rather than `nelec/2` — the odd
   electron goes up. The LSDA `nbnd` default is then `MAX(nelec/degspin, nelup, neldw)`
   with `degspin = 1`, so it is the *fuller* channel that decides it.
6. **`pbec_spin`'s `gamma` is not `pbec`'s.** The unpolarized routine carries
   `0.0310906908696548950` and the polarized one the rounded `0.031091` — a relative
   difference of 2e-6 that QE has never reconciled, and that is visible in the sixth
   decimal of a PBE correlation energy. Reproduced rather than unified.
7. **`from_input` occupations are halved only when `nspin = 1`.** `wg = f_inp(:, isk) *
   wk`, and the `/2` is not a normalisation choice: one channel takes occupations in
   [0, 2] against a weight that already carries the spin degeneracy, two channels take one
   row each in [0, 1] against a weight that does not (`DEGSPIN` drops to 1).

*What autodiff bought again.* The polarized `v_up`/`v_dw` — QE's `slater_spin` returning
two potentials and `pz_spin`/`pw_spin` three terms each including an explicit `df/dzeta` —
are `jax.grad` of one energy expression, and agree with QE's algebra transcribed
independently to 1.3e-15. The GGA's cross term `v2c_ud`, which exists only because
correlation depends on the *total* density's gradient and which QE assembles by hand into
`h`, is not a separate quantity here at all: differentiating with respect to the gradient
*field* rather than to `|grad rho|^2` produces it, correctly paired with the two channels,
for free.

*What P9 also had to fix first* (committed separately, since none of it is spin): the
gamma-point half-sphere storage was generated but never consumed — `h_psi`, the
eigensolver, `addusdens`/`newd` and the symmetry maps all assume the full sphere, so a
`K_POINTS gamma` run is now substituted by an explicit k = 0 with the full sphere and says
so; the Vanderbilt `q_with_l = F` augmentation format was refused outright, though every
`rrkjus` file in the test set uses it and `set_upf_q` is fifteen lines; and the local
functional was switched off where the plane-wave density goes slightly negative, where
`xc_lda` evaluates it at `|rho|` — worth 1.3e-7 Ry on an isolated atom and invisible in a
crystal. `PP_AUGMENTATION shape='BESSEL'` was refused too, on the assumption that the
shape needed reconstructing; grepping QE settles it — `upf%paw%augshape` is read,
broadcast, printed and written back, and never used in a calculation.

*Not covered:* `occupations='fixed'` with `nspin = 2` (no committed benchmark exercises
either of QE's two fillings for it, so it is refused rather than guessed), and
non-collinear magnetism, which stays out of scope. `pw_lsda/lsda-2.in` — an `nscf` run on
an 8x8x8 grid restarting from `lsda.in`'s density — is **done**, at the merge with P8: it
matches QE's Fermi energy of 15.3379 eV exactly, and it is what found the second trap in
the P8 x P9 note above. `pw_pawatom/paw-atom_spin.in` is validated but kept out of the test
suite: QE converges it in 32 iterations and this code's Anderson mixer does not do so
quickly on a landscape that flat, which is a mixer-robustness item for `PERFORMANCE.md`'s
backlog rather than a correctness gap — `o-paw-spin-pbe` pins the identical code path.

**P10 — Performance and parallelism. 🔶 FIRST PASS DONE.** The metric is single-core
pypresso against single-core QE on the same machine (`tools/compare_qe.py`, inputs in
`benchmarks/`), and it now stands at **~3.3x per SCF iteration**, from 53x. Done: `vmap`
over k-points, the iteration body compiled in three units, the Fermi bisection moved
on-device, one host sync per iteration, the Ewald real-space sum vectorised, setup's
compilation count cut by wrapping whole functions in `jit`, and QE's adaptive `ethr`
schedule (which needed `dr2`/`rho_ddot`, so `conv_thr` now means what it means in a `pw.x`
input). Also done: a persistent compilation cache (process wall 9.7 s → 4.3 s), and — the largest
factor of all, though it belongs to P6 rather than P10 — reducing automatic k-grids to the
irreducible wedge. Applying `H` only to unconverged bands was measured and dropped: its
ceiling is 3% of an iteration, because scheduling `ethr` and starting from atomic orbitals
already stopped Davidson calls running long enough for compaction to pay. Still to do:
buffer donation, k-axis sharding across CPU-cores-as-devices and across GPUs, Numba
`prange` on the setup hot spots.

**Phase 0 of the GPU half is done (2026-08-25) and it ran on a Tesla V100.** The same
source, unmodified — nothing was ported, which is what `GPU.md` predicted and what had
never been evidence. `al10-metal` reproduces the committed QE reference to **1.88e-09 Ry**
on the device, agrees with the CPU run to **1.6e-13 Ry**, and is **bit-identical run to
run**. The measurement is that **the two batching dials invert**: 801 ms/iteration on the
cache-shaped defaults against **177 at `k=all, b=all`**, a factor of 4.5, where the same
change costs 1.2x the *wrong* way on a CPU — and `k=all, b=1` is worse than either end at
2075, because batching k while looping bands buys the batched mode's memory with the looped
mode's launch count. fp64 costs 1.78-1.98x on a matmul and 0.85-1.44x on an FFT, which
ranks `GPU.md`'s float32 phase **after** its sharding phase rather than before. Harness in
`tools/gpu/`, numbers in `PERFORMANCE.md`.

**Phase 1 has had a first pass** on a ladder of single-k cells — 8, 16 and 32
atoms — chosen so that no k-parallelism exists and every ratio belongs to the
per-k path. The diagonalisation *does* win and its advantage **grows with the
cell**: Davidson 3.9x at 16 atoms and **13.5x at 32**, `h_psi` 6.8x and
**17.9x**, with `v_of_rho` crossing from a loss to a win. Two things came with
it. QE's adaptive `ethr` schedule taxes a GPU where it costs a CPU nothing —
the last iterations of a 16-atom SCF cost 459 ms against 36 for the first
seven, because a tighter threshold is more Davidson steps and a Davidson step is
small dense algebra in a `lax.while_loop` — though that tax falls from 3.6x to
1.2x by 32 atoms. And the band dial at 32 atoms is a wall rather than a
slowdown: the GPU did not finish two SCF runs at `band_batch = 1` in fifteen
minutes where four CPU cores do one in eleven seconds. `benchmarks/si32-1k*.in`
were added for this and are verified by a folding identity rather than by an
energy-per-atom comparison, which single-k sampling makes invalid across these
cells. Phases 2-4 are unrun; Phase 5's CPU half is measured (below).

**Phase 1's second pass went one level in, and found a trap of the same family
as the `abs` one — a JAX *lowering* that is invisible in the source.**
`tools/gpu/davidson_profile.py` profiles the parts of a Davidson step at a
case's own shapes, and the first thing it measured was a regression nobody had
looked for: **`lax.cond` under `vmap` executes both branches.** A conditional
whose predicate is batched has no branch to take — JAX lowers it to `select_n`
over the results of both sides — so the guard the 64-atom `NaN` fix put inside
`generalised_eigh` was computing canonical orthogonalisation on every step of
every k-point of every multi-k run, in exactly the mode `k_batch=None` puts an
accelerator into by default. **2.85x of the subspace solve**, measured at
`si10-nc`'s shapes on a CPU, because it is a lowering fact and not a hardware
one. The fix is where the predicate is evaluated rather than what it tests: the
batched solve takes the Cholesky route unconditionally and
`davidson_eigensolver_all` asks once, outside `map_k`, whether the eigenvalues
*and* the eigenvectors came back finite. Bit-for-bit unchanged (`si10-nc`:
same energy to twelve digits, same eigenvalue SHA-256).

**The trap generalises and it constrains the next two changes.** `lax.switch` is
`lax.cond`'s twin and runs every branch under `vmap` too — which is precisely
the mechanism the two remaining `cegterg` behaviours would need (sizing the
projected solve by the live `nbase`, and the expansion by `notcnv`). Both are
therefore *unbatched-path* changes, and that is less of a restriction than it
sounds: the large single-k cells, where the subspace algebra is ~80% of a
Davidson solve, never go through a `vmap` at all. Ranked and measured in
`PERFORMANCE.md`'s backlog, items 2 and 3; the premise for the second one had to
be measured rather than assumed, because the live-root count falls 20 → 13 → 10
→ 0 in the seeded regime the SCF runs and does not fall at all from a cold
start.

**Two things landed 2026-08-26, and the first is a configuration bug rather than an
optimisation.** `batching.py` defaulted both dials to `1` on *every* platform, so a bare
`run_scf` on a card inherited the cache-shaped end of both — the mode measured at 4.5x on
`al10-metal`, at an outright loss (0.20x) on sixteen atoms, and at a wall on thirty-two.
Every GPU number above was produced by a dial set by hand in an sbatch script, and nothing
in the code said so. The default now follows the platform (`_platform_default`): QE's loop
on a CPU, the whole of both axes on anything else, with an explicit argument beating
`PYPRESSO_*_BATCH` beating the platform, both dials moving together because `k=all, b=1` is
worse than either end, and the CPU default bit-identical to what it was. The accelerator
branch is tested here by substituting the backend, so it needs no card. `GPU.md` §5's rule
is what shapes it: a platform-dependent choice is a dial with a per-platform default, never
a rewrite.

**And `GPU.md` Phase 5's tape is measured, which is the number that says whether the
response path fits on a card at all.** `tools/gpu/phase5.py` runs one converged SCF and one
response property per process — peak RSS is a high-water mark, so two properties in one
process report the larger twice — and reports the working set over the SCF's beside the
parameters it should scale with. The finding is that **the mode, not the property, decides
the tape**: a *forward* response (the Sternheimer solves behind `epsilon` and `Z*`) costs
0.7-1.0 GB over its SCF, and so does a *forward-over-reverse* one (the dynamical matrix,
the Raman tensors), which is what `GPU.md` guessed but did not know — a `jvp` of a gradient
tapes the inner reverse pass and could have behaved like the other end. The other end is
the *reverse* stress through the radial transforms, which is where the 11 GB lives. So the
response path fits a card comfortably except on the one axis P11 already flagged, and that
axis has a fix in the backlog (a `custom_jvp` on each radial transform) rather than a
guess. Numbers in `PERFORMANCE.md`.

**And the GPU half ran the same day** (one H200 against four EPYC Milan cores,
commit `e562427`), which returned three things. **The mode decides the speedup
exactly as it decided the tape**: a Sternheimer solve gets **1.0x** from the card
— 1.0 on `epsilon`, 1.0 on the dynamical matrix, 1.2-1.4 on the rest — where a
reverse-mode gradient gets 24x and **339x**, because a projected CG over occupied
bands is P10's small-dense-algebra pathology and a gradient through the radial
transforms is one enormous dense graph. **That 339x is warm and nothing pays it**:
`stress()` is called once per run, and cold the GPU costs **26.20 s against
10.71** — it compiles for 26 s what it then runs in 9 ms. **The card holds a
quarter of what the host figure suggested** — the 10.56 GB tape is 2.73 GB of HBM,
every response under 0.25 GB — so feasibility is answered by two orders of
magnitude. And **three rows are not bit-reproducible on the device where none is
on the CPU**, at 1e-13 to 1e-16 relative: `GPU.md` check 5's atomics hazard, absent
from the SCF, present in the response, which means a GPU regression set compares a
response to a stated tolerance rather than diffing bytes. Two open items: the
`alas-raman` row died with a `MemoryError` on the CPU node that this workstation
does not reproduce at 2.59 GB, and every response case here is a two-atom cell,
which the standing rule says shows none of this.

**`GPU.md` is the roadmap for the GPU half of this phase** — what is already GPU-ready by
design and needs no work, what is blocked on first contact with real hardware, and what
can be done here without any. Read it before starting that work. Three of its points
change how this entry should be read: **nothing here is a port** (JAX emits GPU code from
this source already; what is missing is evidence, not a backend), **the single-core metric
above does not transfer** and a GPU number must never be quoted against it, and there is no
GPU on the development machine, so every measurement is made on a cluster and the cluster's
own rules govern how.
*Check met:*
`PERFORMANCE.md` carries the timing and scaling table; no numerical drift — the full suite
passes and a converged SCF sits on the exact eigenvalues of its own converged Hamiltonian.

**The k axis is a dial now, and its default is QE's.** Batching every k-point into one
`vmap` — which is what R6 exists to allow — was never measured against the alternative
until a converged bismuthene run needed 12.7 GB for 19 irreducible k-points of
two-component spinors. `pypresso/batching.py` chunks the axis instead, defaulting to one
k-point at a time exactly as `c_bands.f90`'s `k_loop` and `sum_band.f90` do, with
`k_batch=None` for the old behaviour; it is a `lax.map`/`lax.scan`, so the body is still
compiled once and still differentiable, and the answer moves by round-off alone (1.8e-15
Ry). The measurement is in `PERFORMANCE.md`, and it is not a trade at all on any case whose
per-k work is real: the loop costs 9% on aluminium's ten cheap k-points, nothing on
silicon's two, and is **twice as fast at two thirds the memory** on converged bismuthene's
nineteen (44.9 → 22.5 s per iteration, 4.91 → 3.16 GB), whose per-k work saturates XLA's
threads on its own. What batching still
buys is the GPU, which is why both ends are kept working.

**The trap:** on these array sizes the cost is **compilation, not arithmetic**. Every JAX
operation dispatched outside a `jit` is compiled separately at ~50 ms, so setup spent 10 s
compiling 81 kernels to do 0.2 s of work. Optimising here means reducing the number of
compiled units, not the number of flops — which is the opposite of the instinct the
Fortran encourages.

**P12 — Ultrasoft and PAW. ✅ DONE for LDA.** `basis/interpolate.py` (the smooth/dense
grid split), NLCC in `v_of_rho`, `pseudo/coupling.py` (real-harmonic Gaunt coefficients),
`pseudo/augmentation.py` (`qvan2`, `addusdens`, `newd`), the overlap operator and a
generalised Davidson, and `paw/` (radial Poisson, spherical quadrature, one-centre terms,
`becsum` symmetrisation). *Check met:* silicon matches QE to **≤3e-9 Ry** on six generated
references — norm-conserving at dual 8, ultrasoft and PAW at 2 and 8 atoms, and PAW on an
unreduced k-grid — with `ngm`, both FFT grids, `npw` and the symmetry count matching
exactly and eigenvalues within 0.05 meV. Timing is **1.8–2.9x serial QE per SCF
iteration**, the same band the norm-conserving path sits in (`PERFORMANCE.md`).

**The traps, in the order they were found:**

1. **`msh` is off by two if you take the last point inside 10 bohr.** QE's loop stops at
   the *first point beyond* `rcut` and rounds *that* index up to odd, so the integration
   range ends one or two points past 10 bohr. On `Si.pz-vbc` the two answers agree to
   1e-11 and nothing shows; on the `psl`/`rrkj` sets they differ in the eighth decimal of
   `V_loc(G=0)`, which shifts every eigenvalue and costs ~1e-6 Ry **at any cutoff**. Its
   cutoff-independence is what identified it: everything basis-related was excluded first.
2. **`pseudo_type` is free text.** `atomic` writes `USPP` where the format documentation
   says `US`; matching only `US` reads an ultrasoft file as norm-conserving and silently
   omits the augmentation charge.
3. **The FFT grid rules above** (P2), worth 1e-5 Ry on the eight-atom cell through the
   symmetry count as well as through `etxc`.
4. **`becsum` must be symmetrised explicitly.** The density is symmetrised in G space,
   which covers the augmentation charge; `becsum` goes to PAW's radial machinery directly,
   where there is no grid to act on. On a symmetry-reduced k-set silicon's three `p`
   channels come out 1.003/1.268/1.268 instead of equal, and the one-centre energy is
   wrong in the fifth decimal — 3e-5 Ry, the last error to be found and the largest.
5. **`Q` is integrated over `kkbeta`, not the 10-bohr mesh**, and for PAW `kkbeta` is
   widened to the augmentation radius, past which the tabulated `beta` is still of order
   1e-3 because it is truncated rather than tapered.
6. **The radial Poisson solver is transcribed, not replaced.** The closed-form integral is
   correct and shorter, but the one-centre energy must agree with QE's to ~1e-8 relative,
   and two discretisations of one integral do not agree to 1e-8 while one discretisation
   trivially does.

*Not covered:* the pre-2.0 `q_with_l = F` augmentation format, and gamma-only storage with
an augmentation charge. Both are refused with a clear error rather than approximated. (GGA
was the third such gap and is now closed — P13.)

*A later trap, and it hid behind every benchmark this project has.* `PAW_symmetrize`
contracts the harmonic rotation on the **source** index of `becsum(irt(isym,ia))` —
`becsym[ia] = sum_S D_S^T becsum[irt(S,ia)] D_S` — while `paw/symmetry.py` contracts it on
the target one. The two are the same sum with `S` relabelled `S^-1`, so the atom index has
to be relabelled with it: the gather needs `irt(S^-1, a)`, the atom sent **onto** `a`, not
the one `a` is sent to. Pairing the rotation of `S` with the forward permutation is not a
subtler average, it is not an average at all — the maps stop composing, so the result is
not invariant and the operator is not idempotent. `hubbard/occupations.py` had the identical
pairing against `new_ns.f90`. **Neither shows on any validated cell**, because a two-atom
silicon, an eight-atom silicon, fcc nickel and antiferromagnetic FeO all have atom orbits on
which every operation's permutation is its own inverse, and there the two orderings coincide
exactly. Three atoms on the faces of a cube separate them — 16 of the 48 operations then
cycle the orbit — and the error is O(1), not a tolerance. The test is that the group average
is a **projector** (`tests/unit/test_symmetrisation_projector.py`): `P(P x) = P x` holds iff
`{M_S}` is a representation, it needs no opinion about whose index convention is whose, and
it fails by 1.9 on a result of size 1.4 when the pairing is wrong.

**P13 — Gradient-corrected functionals. ✅ DONE.** `xc/` restructured into QE's four
independently chosen slots — local exchange, local correlation, and a gradient correction
to each — behind a name registry (`xc/functional.py`), with `xc/gga.py` holding the PBE
family and `xc/lda.py` gaining Perdew-Wang correlation. `basis/gradients.py` takes the
gradient and divergence in G space; `scf/potential.py` assembles `v = v1 - div(v2 grad
rho)` as `gradcorr.f90` does; `paw/gradient.py` does the same on each PAW sphere, where
the gradient is radial-plus-angular and the divergence is the spherical one. The
functional is resolved once per calculation from the pseudopotentials' headers, with
`input_dft` overriding them as in QE. *Check met:* PBE silicon matches QE to **≤6e-9 Ry**
on seven generated references — norm-conserving, ultrasoft and PAW, at 2 and 8 atoms, with
revPBE and PBEsol on top — and a PBE band structure to **5e-5 eV**. Still only what a
gradient needs: meta-GGA and hybrids remain out of scope.

**The traps, all of them in how QE composes and gates the functional rather than in the
formulas:**

1. **PBE's local half is not the LDA.** `pbex`/`pbec` return only the *gradient
   correction*; the local part underneath is Slater exchange plus **Perdew-Wang**
   correlation, not the Perdew-Zunger correlation an LDA run uses. Pairing PBE's gradient
   terms with PZ is a functional QE never prints, and it converges perfectly well.
2. **The GGA thresholds are four orders of magnitude coarser than the LDA one.** XClib
   zeroes the energy *and both potentials* wherever `rho <= 1e-6` or `|grad rho|^2 <=
   1e-10`, against `1e-10` for the local part. A plane-wave density has low-density
   regions over much of the cell, so evaluating the gradient terms there instead of
   zeroing them is worth ~1e-6 Ry.
3. **Old UPF files spell the gradient terms `PBE`, not `PBX`/`PBC`.** `igcx = 14` and
   `igcc = 9` are legacy aliases that `set_dft_from_name` maps back to 3 and 4 under a "TO
   BE REMOVED" comment. Every `*.pbe-*.UPF` in the test set uses them, so a parser that
   only knows the canonical names reads a PBE dataset as having no gradient correction at
   all — the same silent-LDA failure as trap 2 of P12, from the other direction.
4. **A PAW sphere needs a bigger angular grid, and more tabulated on it.**
   `paw_init.f90` adds `xlm = 2` to the quadrature's exactness when the functional is a
   GGA, and the vector field whose divergence gives the potential is expanded two
   multipoles past the density (`ladd`), because taking a divergence costs two. Reusing
   the local functional's 28-direction grid gives a one-centre energy wrong in the fifth
   decimal.
5. **The `theta` component is divided by `sin(theta)` before it is projected**, with the
   factor restored inside the divergence. QE's comment explains it: the `lm` expansion of
   `dY/dtheta` converges far too slowly to truncate at `ladd = 2`, while the same
   derivative divided by `sin(theta)` does not.
6. **`PAW_gradient` fills its components in the order `(r, phi, theta)`** — not the order
   its own comment claims. Only the pairing with `dylmp`/`dylmt` in `PAW_divergence`
   settles it, and swapping the two is invisible in `|grad rho|^2`.

*What autodiff bought here.* QE hand-derives `v1x`, `v2x`, `v1c`, `v2c` — four routines
whose correctness depends on agreeing with an energy expression written elsewhere, and
whose `v1c` contains `d(rho ec)/d rho` for the *local* correlation, so it also depends on
the two halves agreeing about which parameterisation is in use. Here only the energy
density is written down and all four come from `jax.grad`; a unit test checks them against
QE's algebra transcribed independently, and they agree to machine precision. The same
`grad` gives the PAW one-centre `ddd` with no extra code, since `rho_lm` stays linear in
`becsum` whether or not the functional has a gradient term.

**P14 — Spin-orbit coupling. ✅ DONE (nonmagnetic).** `noncolin` makes a wavefunction a
two-component spinor stored as one vector of length `2 npwx`, exactly as QE stores it, so
the eigensolvers see a larger vector space and not a new kind of problem; `lspinorb` puts
the `j`-resolved projectors of a fully-relativistic dataset into `D_ij`, which becomes a
complex 2x2 matrix in spin space. New modules: `pseudo/spinorbit.py` (`rot_ylm`, `spinor`,
`sph_ind`, `fcoef`, `dvan_so`, `qq_so`, and the `becsum`/`newd` transforms) and
`hamiltonian/noncollinear.py` (`SpinorHamiltonian`). The collinear `Hamiltonian` was not
touched — it is the hot path of everything else — and the two present the same surface to
the solvers (`ndim`, `state_mask`, `diagonal`, `s_projections`) instead.

*Check met:* the three platinum cases QE's test suite ships, one per pseudopotential kind,
all regenerated with the vendored `pw.x` at `conv_thr = 1e-10`:

| case | what it isolates | total energy |
|---|---|---|
| `pw_spinorbit/spinorbit` | ultrasoft + LDA: `fcoef`, `dvan_so`, `qq_so`, `newd_so` | **1.3e-8 Ry** |
| `pw_spinorbit/spinorbit-pbe` | the same with a gradient correction | **3.8e-9 Ry** |
| `pw_spinorbit/spinorbit-paw` | PAW's one-centre terms under the spin transform | **8.4e-9 Ry** |
| `bismuthene-soc` / `-nosoc` | the target system: 2D, ultrasoft + PBE, 30 electrons | 3.5e-5 Ry, see below |
| `bismuthene-soc-small-lda` | the control that says which part owns that 3.5e-5 | **7.1e-9 Ry** |

...and, ahead of any of them, an identity that needs no reference at all: **a noncollinear
run with no spin-orbit coupling and no magnetization must reproduce the collinear answer
term by term and double every eigenvalue.** It does, to 1e-14 Ry, on norm-conserving,
ultrasoft and PAW silicon. That check is what gates the spinor infrastructure — it fails
on any error in `vloc_psi_nc`, the doubled eigensolver, the spinor `sum_band`, the
projector occupations or the k-point weights — and it was written and passed *before*
`fcoef` existed, so the spin-orbit work started from a Hamiltonian already known to be
right.

*Bismuthene, which is the system the feature exists for.* A planar honeycomb layer of
bismuth at the SiC(0001) lattice constant, run twice with everything but the
pseudopotential held fixed: once with the fully-relativistic dataset and `lspinorb`, once
with the *scalar*-relativistic one and no spin at all. (Not the relativistic dataset with
the coupling switched off — that asks for `average_pp`, which is a third pseudopotential
and is refused here.) Without the coupling the frontier bands come within **0.14 eV** of
each other at a pair of points flanking K along M–K–Γ, and the fundamental gap is the same
0.14 eV; with it the smallest direct gap on the path is **0.63 eV** and the fundamental gap
**0.49 eV**. Note *where*: the bands do not meet at K, where they are already split, which
is why the quantity quoted is the smallest gap on the path rather than the gap at a
symmetry point. Two sizes are committed with their own QE references — 20 Ry on 6x6x1,
which is what the tests run and what the numbers above are, and 35 Ry on 12x12x1,
where the path resolves the near-degeneracy properly: there the gap without the
coupling is **0.040 eV** and with it **0.508 eV**, both matching QE to the four
decimals it prints, the bands agree to **0.52 meV** over the whole path, and Kramers
degeneracy holds to 4e-8 eV. That pair costs 11 minutes of SCF and 10 of bands per
run at a peak of **9.4 GB** — which it manages at all only because of the k-loop
(P10): batched over all 19 k-points it was killed at 12.7 GB and still climbing.

**Its total energy agrees to 3.5e-5 Ry, not the 1e-8 the platinum cases reach, and the
cause is measured rather than argued.** The control is `bismuthene-soc-small-lda`: the same
cell, the same fully-relativistic dataset, the same grids, the same k-points and the same
spinor path, with `input_dft = 'PZ'` switching the gradient correction off — and it agrees
to **7.1e-9 Ry**, four orders better. What is left is therefore the gradient correction
evaluated over the two thirds of this cell that are vacuum, where XClib's thresholds
(P13 trap 2) decide whether a point contributes at all: a property of P13 and of the
geometry, not of P14. The collinear `nosoc` run carries the identical offset with the
identical sign pattern term by term, and the *difference* between the two runs — which is
the physical claim — matches QE to **1.6e-6 Ry** on the test-sized pair and
**1.4e-7 Ry** on the converged one (0.741399926 Ry against 0.741400070). Every PBE case validated before this one
was a dense bulk crystal, which is why nothing earlier reached it.

`fcoef` itself has an exact characterisation that the unit tests use instead of a
reference: for each radial projector it is the **orthogonal projector onto that `(l, j)`
shell** in the basis of real harmonics times spin, so it is Hermitian, idempotent, of
trace `2j+1`, and the two `j` shells of an `l` sum to the identity. All four hold to
machine precision, which pins down `rot_ylm`, `spinor` and `sph_ind` together.

*Traps, in the order they cost time:*

1. **`fcoef` is used before it is zeroed.** `init_us_1` builds it for every pair with
   matching `(l, j)`, multiplies `dion` by it to get `dvan_so`, and *then* sets the entries
   whose two radial projectors differ to zero. `transform_qq_so`, `newd_so` and
   `add_becsum_so` all consume the **zeroed** array and rely on it to kill the cross-radial
   terms, having no same-radial guard of their own. One array used everywhere gives a
   correct `dvan_so` and a silently wrong `qq_so`, `deeq_nc` and `becsum`.
2. **`PP_AEWFC_REL` is not part of `PP_AEWFC`.** A fully-relativistic PAW file carries the
   small component of the Dirac partial waves in its own tag series beside the large one.
   Selecting the partial waves by tag *prefix* returns both series interleaved — twice as
   many as there are projectors, each attached to the wrong channel — and nothing fails:
   the one-centre energy simply comes out **tens of Ry** wrong. This was a latent bug in
   the parser that only a relativistic PAW dataset could expose.
3. **The small component carries charge at every `nspin_mag`.** `read_upf_new` adds
   `|phi^rel_i phi^rel_j|` into `pfunc` inside the augmentation sphere for *any* run with
   `has_so`; only its use in the magnetization (`pfunc_rel`, `with_small_so`) is gated on
   `nspin_mag == 4`. Leaving it out is worth 1e-3 Ry on platinum — small enough to look
   like a convergence difference, large enough to be wrong.
4. **`domag` is a property of the input, not of the answer.** `setup.f90` sets it from
   `starting_magnetization` being nonzero *somewhere*, and it decides `nspin_mag`. A
   spin-orbit run on a nonmagnetic crystal therefore has a **scalar** density and
   potential, so `v_of_rho`, the mixer, `rho_ddot` and `sym_rho` all run exactly as they do
   unpolarized. Allocating four components anyway is not merely wasteful — it symmetrises,
   mixes and exchange-correlates a magnetization that is identically zero, and QE's own
   comment ("to make a spin-orbit calculation with zero magnetization") says this is the
   intended case rather than an edge one.
5. **`vltot` and `rho_core` go into the first component only when there are four.** In the
   `(up, down)` representation an unpolarized quantity is shared *equally*; in
   `(n, m_x, m_y, m_z)` it is all charge and no magnetization. `set_vrs` writes the
   distinction out explicitly (`IF (is > 1 .AND. nspin == 4)`), and a single `/ nspin`
   silently loses three quarters of the core charge.
6. **`add_paw_to_deeq` runs before `newd_so`, not after.** PAW's one-centre coefficients
   are an addition to the *scalar* integral `int V Q`, and the `fcoef` sandwich is applied
   to the sum. Adding them to the already-transformed `deeq_nc` puts them in the wrong spin
   structure and converges perfectly well to the wrong answer.
7. **The degeneracy factor moves from the weights to the band count.** `setup.f90` does not
   multiply the k-point weights by `degspin` for `nspin = 4` — a spinor band holds one
   electron — and it *does* double `nbnd`. Everything downstream that is weight-driven
   (the Fermi search, the tetrahedra, the DOS) then needs no change at all, and the only
   places the factor reappears are where a *count* of filled bands is taken.
8. **`usnldiag_nc` preconditions with the diagonal spin blocks only.** Blocks 1 and 4 of
   `deeq_nc` and `qq_so`, and the charge component of the local potential. The
   preconditioner does not have to be the operator.

*Deferred, and refused rather than approximated:* symmetrising a noncollinear
magnetization (`sym_rho` rotates it as an axial vector, with a sign from the determinant
and another from time reversal on a magnetic operation) — so `nspin_mag = 4` needs
`nosym`; `gradcorr` in the local spin frame — so `nspin_mag = 4` needs an LDA functional;
`average_pp`, QE's `j`-averaging of a relativistic dataset asked for with
`lspinorb = .false.`; and PAW's `with_small_so` magnetization term, which only
`nspin_mag = 4` reaches.

**The `average_pp` refusal was written one condition too narrow, and that is a
trap worth recording because the refusal existing is what made it invisible.**
It read `noncolin and not lspinorb` — the case a reader thinks of, a spin-orbit
run with the coupling switched off — where `setup.f90` calls `average_pp` in the
*else* branch of `IF (lspinorb)` and so reaches **every** run without spin-orbit
coupling. The one it therefore missed is the common one: an ordinary `nspin = 1`
calculation with a `rel-` dataset picked off a pseudopotential table, which used
the two `j = l +- 1/2` channels as though they were one, converged, and reported
a total energy wrong by **20 Ry** — measured on rhombohedral BN with ONCVPSP's
`B-PBE`/`N-PBE` against `pw.x`, -100.42 against -80.16, with the Ewald and
dispersion terms — the only two that touch no projector — still agreeing to
4e-9. The condition is now `not lspinorb`, and
`tests/unit/test_spinorbit_coefficients.py` covers both halves of it. The lesson
generalises past this case: a guard written for the *interesting* instance of a
condition and not for the *general* one reads, in review, exactly like a guard. The atomic starting guess is the scalar orbitals tensored with
`{up, down}` rather than QE's `atomic_wfc_so` spin-angle functions: the same span, so the
same answer after `rotate_wfc`, at the cost of a Davidson step or two.

**P11 — Higher-order autodiff quantities (after the first milestone).** Forces are done
(P15); here: stress by differentiation w.r.t. strain — **done, below** — implicit
differentiation of the SCF fixed point (D3), then polarization/dielectric response and
second harmonic generation. P16 has taken the first bite of the response half of this: the
velocity operator from `jacfwd` of a model `H(k)` is written and validated against the
lattice Chern number, and what is left for P11 is the **plane-wave** velocity operator —
`d(vkb)/dk` and the k-dependence of the sphere — which P16 refuses rather than
approximates.

### P11a — The stress tensor. ✅ DONE.

`pypresso/stress/` (the strained energy, its gradient, QE's seven hand-derived terms behind
a name registry, and the `Stress` object with its two residues), `Calculation.at_strain`,
`symmetrize_matrix` (`symme.f90`'s `symmatrix`), `tstress` on `System` and on `run_scf`,
`SCFResult.stress`, and a `pypresso stress` subcommand.

**It is P15's construction with the cell in place of the atoms.** The total energy is
written down once — `forces/energy.py`'s `energy_at`, which the force and the stress now
share — and `jax.grad` of it with respect to a strain is the answer:

    σ_ab = −(1/Ω) dE/dε_ab,   h → (1 + ε) h,  τ → (1 + ε) τ,  G → (1 + ε)^−T G

with the atoms carried in *crystal* coordinates and the reciprocal lattice following from
the Miller indices being what is stored (`basis/gvectors.py` was written that way at P1 for
exactly this). Nothing is derived for the kinetic term's `(k+G)_a (k+G)_b`, the Hartree
kernel's `G_a G_b/(G²)²`, `dV_loc/d|G|`, the exchange-correlation diagonal, the gradient
correction, the core charge, the augmentation charge or the Ewald sum: **one gradient
produces all of them**, and it produces PAW's one-centre terms and DFT+U's projectors as
well, neither of which has a stress routine of its own here.

Two things make the partial derivative the total one. The stationarity argument P15 makes,
and one step it does not need: **the frozen quantity is the coefficient vector, not the
wavefunction.** A strain moves the plane waves with the cell, so holding the coefficients
fixed holds the state fixed in crystal coordinates — the variational parameter the SCF
minimised over. And unlike the force, the orthonormality constraint carries no strain at
all (`<ψ|ψ>` is a sum over integers, and `qq_ij = ∫Q dr` has no cell in it), so a stress
has no Pulay term of *that* kind. It has a different one, below.

*Check met* — the target is 1e-4 Ry/bohr³ and the worst of thirteen cases is **6e-6**,
which is QE's stopping point rather than this code's: every case with a reference
regenerated at `conv_thr = 1e-10` agrees to **≤4.0e-7**, and the two that do not are
compared against committed QE 6.x benchmarks that stopped at 1e-6.

| case | what it adds | max \|Δσ\| (Ry/bohr³) |
|---|---|---|
| `pw_scf/scf.in` | the canonical insulator, 2 explicit k | **1.2e-9** |
| `pw_scf/scf-occ.in` | occupations from input | 1.4e-9 |
| `pw_scf/scf-kcrys.in` | crystal-coordinate k-points | 8.8e-9 |
| `pw_scf/scf-k0.in` | Γ only | 4.0e-9 |
| `si2-nc-stress` | a *displaced* atom: anisotropy, and a real shear | 4.3e-8 |
| `si2-nc-sheared` | a **tilted cell**: 2 operations, every entry of σ free | 1.3e-8 |
| `si2-us-stress` | ultrasoft — `addusstress`, and the first `stres_cc` | 2.2e-7 |
| `si2-paw-stress` | PAW's one-centre terms | 2.2e-7 |
| `si2-us-pbe-stress` | PBE — `stres_gradcorr` | 2.7e-7 |
| `ni-ldau-stress` | **DFT+U** — `stres_hub`, 2291 lines not transcribed | **4.0e-7** |
| `pw_lsda/lsda.in` | `nspin = 2`, ultrasoft nickel | 2.4e-7 |
| `pw_metal/metal.in` | smearing (committed 6.x benchmark) | 5.9e-7 |
| `pw_uspp/uspp2.in` | ultrasoft molecule (committed 6.x benchmark) | 6.0e-6 |

...and three checks that are sharper than any of those, because they compare things that
share no machinery:

| check | what it isolates | agreement |
|---|---|---|
| **every term** against QE's `verbosity = 'high'` table — kinetic, local, **nonlocal**, Hartree, exc-cor, corecor, ewald | a total right by cancellation | **≤3.8e-8**, which is 0.006 kbar: the precision `pw.x` prints them in |
| autodiff vs the transcribed `stres_*` expressions, term by term | two implementations | **≤5e-16** |
| the antisymmetric part of `dE/dε` | rotational invariance, needing no crystal symmetry | **≤1.3e-15** (PAW 2.6e-7, DFT+U 6.2e-6 — trap 5) |
| `strained_energy(0)` vs the SCF total | that the functional *is* the energy | ≤1e-8 Ry |
| central difference of `strained_energy` vs `jax.grad` | the differentiation alone | 1e-5 relative |
| `tr σ/3` vs `−dE/dV` from a **re-converged** SCF | **stationarity**, and the basis | 1.1 kbar at `ecutwfc = 40`, **0.11 at 60** — the Pulay stress, and it converges |

**The Pulay stress is the one real approximation, and it is the cutoff's fault rather than
the method's.** The plane-wave sphere is held fixed while differentiating — membership is a
host-side decision that cannot be traced, and it is piecewise constant in ε, so on each
piece the frozen-sphere derivative is exact. What it misses is the jump where a plane wave
crosses the cutoff, and unlike a spiral's `q` (P21), where one sphere shifts, a strain moves
`|k+G|` for *every* plane wave at once. Measured on silicon against a finite difference of
the SCF energy with the sphere reselected at each volume:

| `ecutwfc` | `tr σ/3` (kbar) | `−dE/dV` (kbar) | Pulay (kbar) |
|---|---|---|---|
| 12 | −27.4 | 16.1 | **−43.6** |
| 20 | 0.7 | 6.0 | −5.3 |
| 30 | 7.0 | 8.7 | −1.7 |
| 40 | 7.4 | 8.4 | −1.1 |
| 60 | 8.2 | 8.3 | **−0.11** |

It falls by a factor of 400 between 12 and 60 Ry, monotonically, which is what says it is a
basis-set artefact rather than a bug — and the test suite's own `ecutwfc = 12`, chosen for
speed, is nowhere near enough for a stress to mean anything.

QE makes the same approximation, which is why it agrees with this code to 1e-9 at
`ecutwfc = 12` while both disagree with the true `−dE/dV` by tens of kbar. Anyone who wants
a stress to mean something has to converge the cutoff, and that is a property of plane
waves rather than of either implementation.

**What the analytic route covers, and what it refuses.** `stres_knl` (the kinetic half),
`stres_har`, `stres_loc` with `dvloc_of_g`, the `−(etxc − vtxc)/Ω` diagonal, `stres_cc` with
`drhoc`, `stres_gradcorr` and `stres_ewa` are transcribed. **`stres_us` and `addusstress`
are not**: they need `gen_us_dj` (the radial form factor differentiated with respect to
`|k+G|`) and `gen_us_dy` (`dylmr2`), which together are a transcription the size of
everything else in the module, and 632 + 234 lines of Fortran on top. So the analytic route
offers **no total** — a sum of what is written would be missing the whole nonlocal
pseudopotential, which on silicon is a third of the pressure — and `compute_stress` refuses
`method = 'analytic'` by name, pointing at `analytic_terms` for the cross-check.

*The traps:*

1. **A coordinate singularity that only a *cell* derivative reaches.** `ylmr2` writes
   `Y_lm` as `sin^m θ · P(cos θ) · {cos, sin}(mφ)` with `φ` from an `atan2` — a
   parameterisation with a singularity on the `z` axis, where `sin θ` has an infinite
   derivative and `φ` is undefined. The *function* is smooth (`r^l Y_lm` is a polynomial in
   `x, y, z`); only the spherical coordinates are not. No phase before this one noticed,
   because a force differentiates the positions and `Y_lm(G)` is a constant there. **fcc
   silicon at `ecutrho = 48` has ten dense G-vectors exactly on the `z` axis**, and every
   one of them turns the whole stress tensor into NaN with every value on the way to it
   correct. The fix is algebraic and exact: `sin^m θ · cos(mφ) = Re[(x + iy)^m]/r^m` is a
   polynomial over a power of `r`, so the two factors are combined *before* they are
   evaluated and the recursion runs on `Q(l,m)/sin^m θ`. Values unchanged to 1e-15.
2. **`sqrt` guarded after the root is a correct value and a NaN gradient.** P15's Ewald
   trap, in three more places: `|G|` at `G = 0` (the first entry of every G set), `|k+G|` at
   Γ, and `|G|` in the augmentation charge. Each was `where(cond, ..., 0)` applied to the
   *result*; `sqrt(0)` has an infinite derivative and `0 × inf` is NaN. `gvectors.modulus`
   is now the one place the mask goes on `|v|²` before the root.
3. **The k-points do not follow the reciprocal cell on their own, and nothing says so.**
   `KPoints.coords` are cartesian in units of `2π/alat` and `alat` is a *static* field, so
   `kpoints.cartesian(strained_cell)` returns the **unstrained** k-points. What is invariant
   under a strain is `k` in *crystal* coordinates, exactly as for `G`. This costs nothing at
   Γ (`scf-k0.in` passes either way) and a k-dependent amount everywhere else. It bit twice:
   once in `at_strain` itself, where threading `kcart` through `kinetic`,
   `build_projector_core` and the **Hubbard** projectors fixed it; and once in the
   finite-difference check, where scaling `cell.at` while leaving `alat` alone produced a
   "Pulay stress" of 47 kbar **that did not fall with the cutoff** — which is how it was
   caught. Scale `celldm`, not the lattice vectors.
4. **`tprnstress` does not exist.** It is a plausible name — `tprnfor` is real — and there
   is no such variable anywhere in QE 7.5; `INPUT_PW.txt` lists one stress switch and it is
   `tstress`. Accepting an alias would have been inventing input syntax. `input.f90`'s own
   rule is worth recording instead: `tstress_ = lmovecell .OR. (tstress .AND. lscf)`, so a
   variable-cell run turns it on whatever the input says.
5. **The rotational residue is 1e-15 everywhere except PAW (2.6e-7) and DFT+U (6.2e-6).**
   The antisymmetric part of `dE/dε` is zero by rotational invariance, which needs no
   crystal symmetry and no converged density — so it is a check on the gradient that works
   on any structure, and it is worth reading before the number. Two subsystems do not meet
   it exactly and both build something from the cell that the plain path does not: PAW's
   one-centre exchange-correlation is a Gauss-Legendre × φ quadrature exact only to a finite
   `l`, and DFT+U's `ns` is averaged over the group with orbital rotation matrices rebuilt
   from the cell. In both cases the residue is the same size as the agreement with QE on
   that case (0.04 and 0.9 kbar) — i.e. it *is* the accuracy floor for that dataset, and
   knowing which of the two it is would need a run with the machinery in question switched
   off, which has not been done.
6. **Reverse mode through a setup routine is a memory problem before it is a time one.**
   A force differentiates one complex exponential per atom over a *cached* radial table; a
   strain moves `|G|` itself, so every radial transform in the setup is inside the gradient.
   The augmentation charge's is the worst: its intermediate is `(ngm, kkbeta)` — 36257 by
   ~1100, so 300 MB, one per `L` — transient forward and all live at once on a backward
   pass. Peak RSS for a stress on eight-atom ultrasoft silicon is **11 GB against the SCF's
   0.9**, and it is the reverse pass, not the forward-mode term breakdown, which costs
   nothing extra. `jax.checkpoint` on that kernel was tried and **measured to be worth
   nothing**, so the intermediates are spread across the radial kernels rather than
   concentrated in one; the fix that has not been written is a `custom_jvp` carrying
   `dF/d|G|` in closed form — which `stress/analytic.py` already writes twice, for
   `dvloc_of_g` and `drhoc`. `PERFORMANCE.md` has the numbers and the backlog entry.

7. **An improvement in the last bits of a setup quantity is not a no-op for an iteration
   count.** Trap 1's rewrite changes `Y_lm` by **2.6e-15** on bcc iron's dense G set and
   moves accuracy in the right direction. It also pushed
   `test_fixed_spin_moment_holds_the_moment` from 746 iterations to **1380**, through a
   ceiling of 1200, and the test failed with nothing wrong with it: the fixed-spin-moment
   scheme is a *proportional controller*, the eigensolver's starting guess is built from
   the same `Y_lm`, and where a controller starts ringing from decides how long it rings.
   The moment it converges to (1.999459 against a target of 2.0) and the field it finds are
   unchanged. **The rule is the general one, because the instance will not repeat:** a
   last-bit change anywhere upstream of a feedback loop moves the seed, the seed moves the
   ringing, and any budget tuned against the old seed is a test of the seed rather than of
   the physics. That test's docstring had already recorded the same thing happening at P20
   for QE's `upf_check_atwfc_norm` renormalisation (~350 to 746); this is the third
   instance, and three of them make the argument that the count is not a claim about
   anything. Diagnose it by *bisecting the change*, not by reading the failure -- "does not
   converge in N" and "is wrong" look identical from the outside, and the thing that
   settles it is whether the converged answer moved.

*Deferred, by name:* **noncollinear and spin-orbit** (`nspin = 4`), refused where the
functional is written — the constraint term needs `qq_so` and the nonlocal one `dvan_so`,
and `pw_noncolin/noncolin.in` is therefore the rung this phase did not reach; **spin
spirals**, because `spiral_q` is in lattice coordinates so a strain turns the spiral and the
generalized Bloch theorem's own term would be missing; **magnetic fields and constrained
moments**, for P21's reason verbatim (the field's energy is outside the reported total, so
the state is stationary for a different functional); **meta-GGA** (`stres_mgga`), which has
no functional here to differentiate; and the `stres_us`/`addusstress` transcriptions above.

**What `vc-relax` would need**, since the stress is what was blocking it and the cell
gradient now exists. It is deliberately *not* started, because it is a design decision about
the run rather than about the stress. **P29 did it, and this forecast was right about
everything but one thing** — the coordinate is the cell matrix `h` itself, nine numbers,
not the six of a symmetric strain, because `bfgs_module.f90` appends `h(i,j)` and lets
`iforceh` mask the entries; a strain would have needed the rotational gauge fixed by hand.
Everything else below is what happened, including the last bullet, which named the bug P29
had to fix:

* **The FFT grid and the symmetry group would have to move, or be pinned.** Both are chosen
  once from the cell (`setup.f90`, and P15 trap 4): the FFT dimensions must divide the
  fractional translations' denominators, and `etxc` is evaluated pointwise on that grid. A
  cell that changes by 10% wants a different grid, and changing it mid-run moves the energy
  by ~1e-6 Ry for a reason that is not physics. QE's answer is a *fixed* basis chosen at the
  starting cell and then one more run at the relaxed one with the G-vectors rebuilt
  (`run_pwscf.f90`'s `reset_gvectors`), and it says the consequence out loud: **"Final scf
  calculation at the relaxed structure. The G-vectors are recalculated for the final unit
  cell. Results may differ from those at the preceding step."** That last sentence is the
  Pulay stress, admitted in the output.
* **The Pulay stress becomes the error bar of the answer**, not a diagnostic. A cell relaxed
  at `ecutwfc = 12` would land tens of kbar away from the true minimum; the sweep above is
  what such a run would have to quote.
* **The optimizer is a bigger change than it looks.** `bfgs_module.f90` would take a
  combined `(3 nat + 6)`-dimensional coordinate — positions in crystal coordinates plus the
  strain — with a metric that mixes them, and `cell_dofree` masks the strain the way
  `if_pos` masks a position. `relax/bfgs.py` is already written against a general lattice
  metric (P21 reuses it with the *reciprocal* cell), so the optimizer itself is reachable;
  what is not written is the coordinate.
* **The Ewald neighbour list, the plane-wave sphere and `alat`** all follow the cell.
  `at_strain` freezes the first two on purpose, which is right for a gradient and wrong for
  a step — the same distinction `at_spiral_q(rebuild_basis=...)` draws for a spiral.
  *(P29: `Calculation.at_cell` is that distinction, and the neighbour list is the half of it
  that had to be rebuilt. The sphere is **not** — QE freezes it too, for the whole
  relaxation, and rebuilds it only in the final SCF.)*

**P15 — Forces and structural relaxation. ✅ DONE.** `pypresso/forces/` (the stationary
energy functional, its gradient, and QE's six hand-derived terms behind a name registry),
`pypresso/relax/bfgs.py` (`bfgs_module.f90`), `workflows/relax.py`, `Calculation.at_positions`,
`if_pos`, `symvector`, `checkallsym`, and a `pypresso relax` subcommand. *Check met:* forces
match QE on **five references** — displaced silicon norm-conserving, ultrasoft, PAW and PBE,
and a spin-polarized O₂ molecule — to **≤2e-5 Ry/bohr** by both methods and to 6e-7 on the
crystals; the two methods agree with each other to the size of `force_corr`; every *term*
matches QE's own `verbosity = 'high'` breakdown; a central finite difference of the SCF
energy reproduces the force to 1e-6; and the relaxations reproduce QE's final geometry to
**1e-6 bohr** and its final energy to **3e-10 Ry**, on displaced silicon and on QE's own CO
molecule with a frozen atom.

The force is `-grad` of a functional evaluated at frozen wavefunctions, occupations and
eigenvalues — *not* of the SCF driver (that would unroll the iteration, D3) and never of an
eigendecomposition (D4). What makes the partial derivative equal the total one is
stationarity, and what makes that true for ultrasoft is carrying the orthonormality
constraint explicitly: `- Σ w f ε (⟨ψ|S(τ)|ψ⟩ - 1)`, zero at the solution with a nonzero
derivative, which is QE's `ε ⟨ψ|dS|ψ⟩`. `D_ij` is not an input to the functional — its
self-consistent part is `∫V_eff Q`, already present through the augmentation charge in `ρ` —
so the nonlocal term takes the **bare** `dion` and nothing is double counted.

**The traps:**

1. **A NaN that exists only in the gradient.** The Ewald real-space sum masks out the self
   term *after* computing `r = sqrt(Σ(...)²)`. That is enough for the energy and not for its
   derivative: `sqrt(0)` has an infinite derivative and `0 × inf` is NaN. The mask has to be
   applied to `r²` before the square root. Every energy test passed; the first force ever
   computed came back as six NaNs.
2. **`gradcorr` is called from inside `v_xc`, not beside it.** `force_cc` uses "the
   exchange-correlation potential", and it reads as though the gradient correction is added
   separately by `v_of_rho`. It is not — `PW/src/v_of_rho.f90` line 607 is *within* the
   `v_xc` that starts at line 440, and it is the only call to `gradcorr` in `PW/src`.
   Building the core-charge force from the local part alone is wrong by 9e-4 Ry/bohr, a
   thousand times the agreement every other term reaches.
3. **The density is in `(ρ, m)` outside `sum_band`.** `sum_band.f90` converts on the way
   out, so the `rho%of_r(:,1)` handed to `force_lc` is the **total charge**, not the up
   channel — while `force_corr` *averages* its two channels, because a potential is always
   stored as `(v_up, v_dw)`. This code stores `(up, down)` throughout, so the local force
   sums and the correction averages. Getting it wrong halves the local force in an LSDA run
   and changes nothing in any unpolarized test.
4. **The FFT grid and the symmetry group are chosen once, before the ions move.**
   `setup.f90` runs before the ionic loop and `move_ions` only ever *checks* the symmetry
   afterwards (`checkallsym`). Re-deriving either mid-relaxation changes the objective
   function: the FFT dimensions must divide the fractional translations' denominators, and
   `etxc` is evaluated pointwise on that grid, so a step that breaks a symmetry would move
   the energy by ~1e-6 Ry for a reason that is not physics. `Calculation.at_positions`
   rebuilds exactly what the structure factor multiplies and shares everything else.
5. **The two force methods must not agree exactly.** What separates them is `force_corr`,
   the correction for a density that stopped short of the fixed point — which the
   differentiated force cannot have, since it assumes the fixed point. Their difference
   tracks that term across four orders of magnitude in `conv_thr`, which is a sharper
   statement than either agreeing with QE, and it makes `force_corr` a direct measure of
   convergence in the units of the answer.
6. **A finite-difference check has to run `nosym`.** Displacing one atom along one axis
   breaks the starting structure's symmetry, and the group is deliberately held fixed
   (trap 4), so a symmetrised run would compare against a density symmetrised with
   operations the displaced structure no longer has. The first FD check "failed" on a
   component that symmetry forbids, and the force was right.

*Found while doing this, and not caused by it:* a **shifted** Monkhorst-Pack grid does not
have every symmetry the crystal has, and neither code's density symmetrisation asks whether
it does. It is worth 1e-4 Ry on a symmetric run and nothing at all with `nosym`, where the
two codes agree to the last digit. The measurement and what it does and does not mean are
under P6 above; relaxations meet it routinely, because a shifted grid is an ordinary input.

*Deferred:* noncollinear forces (`qq_so`/`dvan_so` in
the constraint and nonlocal terms — refused rather than approximated), and the ion dynamics
other than BFGS (`damp`, `fire`, molecular dynamics), which are a file and a registration
each. `vc-relax` was deferred here and is **P29**.

**P16 — Berry curvature, Chern numbers and Z2 invariants. ✅ DONE.** `pypresso/topology/`
(mesh and the reciprocal-lattice wrap, the state sets and their overlap, the augmentation
charge at an arbitrary wavevector, link variables, Berry curvature behind a name registry,
Wilson loops, Fu-Kane parities, and the two Z2 methods behind a second registry) and
`workflows/topology.py`. *Check met:* the Chern number of the Haldane model is an **exact
integer on a 6x6 mesh** (2e-16); the doubled Qi-Wu-Zhang model reproduces `elkpy`'s four Z2
values by three independent routes (Wilson, parity, and the parity of a spin Chern number);
Kane-Mele's transition lands on the analytic `3 sqrt(3) lambda`; the three-dimensional
lattice Dirac model gives all four `(nu0; nu1 nu2 nu3)` phases in closed-form agreement by
both routes; and on real Kohn-Sham states silicon's eight parity products give `nu0 = 0`
with the parity matrix Hermitian to 1e-16 and squaring to the identity to 5e-11.

**Everything is built from `<u_mk|S|u_nk'>` and nothing differentiates an eigendecomposition.**
Two reasons, and the second is the decisive one. Gauge: a determinant of overlaps is blind
to any unitary mixing inside a degenerate manifold, which is the freedom the eigensolver
actually has (D4). Quantisation: the Fukui-Hatsugai-Suzuki lattice field strength sums to
an *exact* integer on any mesh, where a Riemann sum of a pointwise curvature converges to
one and never equals it. Measured on Haldane: the lattice construction is exact (2e-16) at
6x6 and at every finer mesh; the Kubo route through `jacfwd` of `H(k)` — D2's stated
intent, and registered as `kubo` — is 8.6e-3 off at 6x6, 1.7e-5 at 12x12 and 4.8e-11 at
24x24, converging spectrally because the integrand is smooth and periodic, and collapsing
to algebraic convergence wherever the curvature concentrates. So **invariants from overlaps, curvature plots
from either**, and the choice is recorded in `topology/states.py` rather than assumed.

**The traps:**

1. **The neighbour of the last mesh point is the first one plus a reciprocal lattice
   vector**, and the periodic gauge `u_{k+b}(G) = u_k(G+b)` makes that a *shift of the
   Miller index*. Measured on silicon: `|det M|` between k = 0.4 and 0.5 is 0.9904 computed
   directly and 0.9904 computed from k = -0.5 through `b1`; with the shift omitted it is
   **0.0096**. On a mesh that becomes a Chern number that is smooth, plausible and not an
   integer — the failure that reads as a convergence problem.
2. **Two neighbouring k-points do not share a plane-wave sphere.** The coefficients are
   aligned by Miller index (a packed integer key and a `searchsorted`), and a plane wave
   inside one sphere and outside the other contributes nothing rather than being gathered
   from the wrong slot.
3. **Ultrasoft `S` between two k-points is not `qq`.** It is `q_ij(b) = ∫ Q_ij(r) e^{-i b r} dr`
   — Vanderbilt's ultrasoft Berry phase, QE's `bp_c_phase.f90` — and `b` is a *fraction* of
   a reciprocal lattice vector, so it is not on the dense G set and the radial transforms
   are evaluated afresh at `|b|`. A uniform mesh has exactly two distinct `b` however many
   k-points it has, wrap included, so they are computed twice per plane.
4. **A relativistic dataset needs `q_ij(b)` through `transform_qq_so`.** The `fcoef`
   sandwich is linear and k-independent, so it applies to the b-dependent integrals
   unchanged — but using the scalar ones in a spinor run leaves the overlap plausible and
   the invariant meaningless, which is the silent-wrong class the P14 row warns about. The
   check is `b -> 0`: 4e-16 against `qq` on ultrasoft silicon, 1e-16 against `qq_so` on
   bismuthene, and `<u|S|u> = 1` to 5e-15 over thirty spinor bands.
5. **A cache keyed on the `Calculation` must be weak.** A calculation owns `Q_ij(G)` on the
   dense grid — a gigabyte on bismuthene — and streaming a Wilson loop one row at a time
   exists precisely so the previous row's is dropped. An ordinary `lru_cache` on the
   b-dependent integrals pinned every one of them; the fix is a `WeakKeyDictionary` and the
   symptom was memory, never a number.
6. **The parity route is exact and the Wilson route is not.** `elkpy` records two cases
   where a well-gapped band structure returned a confident *wrong* integer from an
   unresolved loop mesh. So the manifold's isolation is checked from the eigenvalues the
   eigensolver has already produced (`gap_tol`), the parity tolerance is 1e-6 rather than
   the 5e-2 the reference needs for Elk's truncation floor, and the two routes are run
   against each other wherever there is an inversion centre.

*Memory.* A state set is `nk * nocc * npol * npwx * 16` bytes: 2.6 MB per k-point on
bismuthene, 63 MB for a 24-point loop, 810 MB for a 24x13 mesh. So the Wilson workflow
streams one loop at a time by default — `npump` costs time, not space. The plaquette
layer costs nothing next to that: the links are `(n1, n2)` unit-modulus phases, and
`StateSet.overlaps` walks the pair axis through `map_k` with the wavefunctions closed
over, so nothing of size `(npair, nbnd, npol * npwx)` is ever built.

**Streaming was the expensive option until `Calculation.at_kpoints` existed**, and the
episode is the general lesson rather than a detail of this phase. A `DFTSource` built a
whole `Calculation` per call, so each row of a mesh rebuilt the dense G set and `Q_ij(G)`
— **~1 GB and ~4.7 s per row on `si8-us`** — which on any mesh worth taking is more than
the states of the *entire* mesh cost to hold. That left a choice between a streaming loop
that was ruinous in time and a resident mesh that was ruinous in memory, and the honest
thing at the time was to state the dial. It was the wrong dial: a k-list changes only what
carries a `k` index — the plane-wave spheres, `|k+G|^2`, the stick layout, `vkb(k)` — and
everything else is a property of the cell and the atoms. Sharing it makes a row **0.16 s,
29.8x cheaper**, with the dense G set, the augmentation charge and the local potential the
same objects across calls, so streaming is now simply the cheap option and there is no
trade to state. `tests/unit/test_at_kpoints.py` pins both halves: identical to a
calculation built from scratch at those k-points, and *the same objects* for the rest —
`is`, not `==`, because equality would pass on the full rebuild this exists to avoid.

*Validated against the literature, after a false start worth recording.* Two real
materials now agree with published invariants, and getting there took correcting the
*geometry*, not the code.

| system | geometry | gap | expected | parity | Wilson |
|---|---|---|---|---|---|
| Bi(111) bilayer | a = 4.34 A, buckling 1.74 A | 0.586 eV | `nu = 1` | **1** | **1** (12x7, 16x9) |
| flat bismuthene | a = 5.35 A, both atoms at z = 0 | 0.505 eV | — | **0** | 0 (12x7, 16x9) |
| germanene | a = 4.06 A, buckling 0.68 A | **0.0244 eV** | `nu = 1` | **1** | 0 (12x7, 16x9) |

Buckled Bi(111) is the freestanding QSH insulator of Murakami (PRL 97, 236805) and Liu
(PRB 83, 235401), and it comes out `nu = 1` by both routes with the band inversion at
Gamma — `N_- = 14` there against 16 at all three M. Germanene's gap is the sharper check
of P14 than of P16: **24.4 meV against the ~24 meV of Liu, Feng and Yao** (PRL 107,
076802), and that gap *is* the spin-orbit coupling, since germanene without it is a
gapless Dirac semimetal. Inputs: `tests/data/qe/bi111-bilayer-soc.in`,
`germanene-soc.in`.

**The first attempt used flat bismuthene and concluded the two routes disagreed.** They
do not. Flat Bi at the SiC lattice constant is a *different material* from the buckled
bilayer — trivial where the buckled one is topological, and neither is the
substrate-supported system of the QSH literature, where the substrate saturates the p_z
orbitals. Re-run on the flat cell, parity and Wilson **both** give 0 at two mesh sizes;
the earlier Wilson `= 1` does not reproduce. Two lessons, and the second is the one that
generalises: a plausible disagreement between two methods is worth suspecting the
*inputs* before either method, and this file previously blamed a hard case for the
largest-gap tie-break when the real fault was a geometry nobody had questioned.

**Wilson is nonetheless the weaker route, and germanene is the proof.** Parity is exact
on all three systems. Wilson is right on both Bi cells and *wrong on germanene* — 0 where
the invariant is 1 — and refining 12x7 to 16x9 does not fix it, so this is not simply a
coarse mesh. A 24 meV gap makes the Wannier centres wind sharply near K, which is where a
largest-gap crossing count is most fragile, and inversion plus time reversal makes the
charge centres symmetric about zero so that the largest gap is degenerate by symmetry and
`argmax`'s tie-break arbitrary. **`WannierFlow.gap_step` does not currently discriminate**:
it reads 0.30-0.43 on the runs that are right and on the one that is wrong alike, so it is
a number to eyeball rather than a convergence criterion. Fixing that — a gap-tracking rule
that follows the *same* branch between pumping steps instead of re-choosing the widest one
— is the open work here, and until it is done **parity is the answer of record wherever
there is an inversion centre.** A 24x13 refinement of germanene was started and killed
before finishing, so whether Wilson converges to 1 on a fine enough mesh is **unknown**.

`elkpy`'s own reference systems remain unvalidated at material level: they need
pseudopotentials this repository does not commit (graphene with Elk's `soc_scale`, the
h-BN slab, caesium dimerized diamond, bulk Bi2Se3), so what carries over from it is the
*conventions* — the sign pins, the TRIM ordering, the largest-gap and orientation
arithmetic, the six-plane assembly — and the doubled-QWZ integers.

*Memory, and the vacuum that was not needed.* Notebook 10 peaks at **6.0 GB**, over the
five it was aimed at, because an SCF and a topology run each build their own
gigabyte-scale `Calculation`. The 2D cells above are also a reminder that **vacuum is not
free**: the first buckled Bi cell used 18 A of it and cost 5.25 GB and 168 s, and cutting
it to 13.0 A -- still 11.3 A between images -- moved the gap by 4 meV while costing 4.06 GB
and 118 s. A 2D cell needs enough vacuum that the images do not see each other and no more;
every extra angstrom is plane waves and FFT grid along z.

*Deferred:* the plane-wave velocity operator (P11, above), a nonzero Chern number from a
DFT run — which needs a magnetization *and* spin-orbit coupling, the one spin regime P14
refuses — the mirror and spin Chern numbers of a symmetry sector, and the quantum
geometric tensor's metric half.

**P17 — Noncollinear magnetism, completed. ✅ DONE.** The ground P18 and P19 stand on, and
it turned out to be hollow: `nspin_mag = 4` was built in P14 and never *run* against a
reference, and the first magnetic noncollinear calculation attempted here came out
**-184.57 Ry against QE's -55.70** on bcc iron. `system/symmetry.py` gained the magnetic
filter and `t_rev`; `scf/driver.py` the axial-vector symmetrisation of the magnetization;
`paw/symmetry.py` the same rotation on `becsum`; `paw/onecenter.py` the local spin frame on
the radial sphere; `scf/potential.py` `gradcorr`'s noncollinear branch; `scf/locals.py` is
new (`make_pointlists`, `get_locals`); and `system/kpoints.py` gained `expand_to_subgroup`,
which is `irreducible_BZ` and runs for *every* input k-list.

*Check met:* bcc iron, ultrasoft, LDA, moment along `x` — QE's `pw_noncolin/noncolin.in` —
to **2.8e-9 Ry**, with the moment `(3.1763, 0, 0)` against the `3.18` QE prints and the
absolute moment `3.1768` against `3.18`. The same cell with PBE (`noncolin-pbe.in`) to
**2.8e-9 Ry**, which is the local-frame gradient correction's own check. The magnetic group
comes out at **16 operations of the lattice's 48, eight of them with time reversal**,
which is the "16 Sym. Ops., with inversion, found" in QE's header; the input's 11 k-points
expand to the **22** QE runs, with QE's weights. And ahead of any of them, two identities
that need no reference: a noncollinear run with the moment along `z` reproduces the
**collinear LSDA** run (P9's validated path) to 1e-10 Ry, and the total energy does not
depend on which way the moment points — `x`, `y`, `z` and `(1,1,1)` agree to 1e-10 with
norm-conserving hydrogen and to 1e-9 with a PAW oxygen atom, one-centre terms included.

*Traps, in the order they cost time:*

1. **The Hartree potential was broadcast over all four components.** `v_h` is added to
   `v(:,1)` alone when `nspin == 4` (`v_of_rho.f90`'s own `IF`), and to *every* channel of
   an `(up, down)` potential — the same distinction `as_potential_components` already made
   for `vltot`, missed one line further down. Broadcasting it puts `v_H` into all three
   magnetization components: an enormous spurious magnetic field, which converges
   beautifully. It cost **129 Ry** on iron and was invisible for the whole of P14, because
   a nonmagnetic spin-orbit run has `nspin_mag = 1` and never reaches the branch.
2. **PAW's one-centre terms had the same bug twice.** `PAW_h_potential` is called on
   `rho_lm(:,:,1:nspin_lsda)` — the charge alone, `nspin_lsda` being 1 for `nspin_mag = 4`
   — and the core charge goes entirely into the charge component rather than `1/nspin` of
   it into each. Both are the *density* and *potential* rules of P14 trap 5, on the radial
   mesh instead of the grid.
3. **The magnetization is an axial vector and a force is a polar one.** `sym_rho` rotates
   the three components with the cartesian rotation, times `det(R)`, times a further `-1`
   for a time-reversed operation. Reusing `symvector` — which P15 already had, and which is
   right for forces — is the plausible-and-wrong version.
4. **The symmetry group is smaller, and it contains operations that reverse the moment.**
   `sgam_at_mag` keeps an operation if it maps every moment onto its image *or* onto minus
   it, the second with `t_rev = 1`. Symmetrising with the full nonmagnetic group averages
   the magnetization to zero and converges to the nonmagnetic solution — and reducing the
   k-set with it, or with `-k = k`, is wrong in the same direction (`setup.f90`:
   `time_reversal = .NOT. noinv .AND. .NOT. magnetic_sym`).
5. **An explicit k-point list is not taken as given.** `irreducible_BZ` treats it as the
   wedge of the *lattice's* point group and completes it for the crystal's. It does nothing
   whenever the two groups agree, which is why it went unnoticed until a moment pointed
   along `x` in a cubic crystal made them differ — and then QE runs 22 k-points where the
   input lists 11, and a comparison against it is off by the weight of half the wedge.
6. **The `nspin = 4` exchange-correlation energy integrates against `|rho|`, not `rho`.**
   `v_xc`'s own branch (`arho = ABS(rho%of_r(ir,1))`), where the `nspin = 1` and `2`
   branches use the signed density. In a dense crystal the two agree; in a cell with vacuum
   they do not, and the identity "noncollinear along `z` equals collinear" then fails by
   **2.6e-4 Ry** on a PAW oxygen atom whose truncated density is negative on a fifth of the
   grid. That is QE's convention reproduced, not an error here — but it decides which
   systems the identity can be *tested* on, which is why the tests use hydrogen and iron.

*Deferred, and refused rather than approximated:* `PAW_gcxc_potential` with a noncollinear
magnetization — the plane-wave `gradcorr` is written, PAW's radial counterpart needs
`compute_rho_spin_lm`'s own local-frame rotation and `segni_rad`, so PAW + GGA + magnetic
noncollinear is refused while ultrasoft and norm-conserving are not; and `with_small_so`,
PAW's small-component magnetization term, which only a relativistic PAW dataset with
`nspin_mag = 4` reaches.

*Notebook 11* (with P18).

**P18 — External and local magnetic fields, and constrained moments. ✅ DONE.** Elk's
`bfieldc` (a field over the whole cell) and `bfcmt` (a field inside one atom's sphere) with
`reducebf`, and QE's `constrained_magnetization` — the same machinery driven by a target
rather than by an input field. New: `scf/fields.py` (the energies and, from `jax.grad`,
their potentials), `scf/locals.py` (shared with P17), a `LOCAL_MAGNETIC_FIELDS` card, and
the input variables `B_field`, `constrained_magnetization`, `lambda`, `fixed_magnetization`,
`reducebf`, `r_m`, `local_weights`.

**The energy is the primitive and the potential is its gradient**, which is the rule
`PLAN.md` §6 states and which paid immediately: QE writes each constraint's potential out
by hand — five expressions, one of them three lines of quotient rule — and every one is
exactly the derivative of the penalty. Writing the penalty once gives all five, and the
Fortran expressions become a *test* rather than a second implementation. All nine of those
comparisons pass to 1e-10 (`tests/unit/test_magnetic_fields.py`), including a finite
difference that would catch a missing quadrature weight.

*Check met:* QE's three constrained runs on bcc iron, one per scheme —
`constrain_atomic` (`i_cons = 1`) to **1.9e-7 Ry**, `constrain_angle` (`i_cons = 2`) to
**2.7e-9 Ry**, `constrain_total` (`i_cons = 3`) to **1.6e-9 Ry** with the moment
`(0.3052, 0.4070, 0.5087)` against QE's `(0.31, 0.41, 0.51)`. The constraint energy itself
agrees at **eight decimals at the starting density** (0.04011011 against 0.04011011), where
it is decided by the penalty expression and the integration spheres alone, and to 2e-7 at
convergence, where it also depends on where each SCF stopped. `make_pointlists`' derived
radius comes out at **1.8637 bohr, QE's printed value exactly**, and the charge and moment
inside that sphere match `report_mag`'s to the six decimals it prints.

*A constrained total energy is a softer number than an unconstrained one*, which is why
those tolerances are 1e-7 rather than 1e-9: the penalty holds the moment off its minimum,
so the energy is first-order sensitive to exactly where it ends up, and QE's own last two
iterations still move the constraint energy by 1.3e-6 Ry.

*Traps:*

1. **The field's energy is not in the total energy.** `add_bfield` is called from *inside*
   `v_of_rho`, so the field is felt by every eigenvalue and removed again by `deband`;
   `etcon` is local to `add_bfield`, which prints it and never returns it — the string does
   not occur in `electrons.f90` at all. Elk states the same convention from the other side
   (its manual: the muffin-tin field energy "is always removed from the total", the
   physical field's "is also not included"). Adding it would be defensible physics and an
   immediate 1e-2 Ry disagreement with every benchmark, so both numbers are carried and the
   reported total is QE's.
2. **`mcons` for `constrained_magnetization = 'atomic'` is compared against a moment in
   Bohr magnetons although it is built from `starting_magnetization`**, which is a fraction
   of the valence charge. QE's benchmark shows the consequence plainly: iron's moment falls
   from 3.06 to 1.61 under a target of 0.5. Transcribed as it is.
3. **The committed benchmark does not belong to the committed input.**
   `noncolin-constrain_atomic.in` carries a commented-out `lambda = 1` above the
   `lambda = 0.005` it sets, and the 2017 output prints a constraint energy of 8.022 Ry at
   the starting density — the *unscaled* sum of squares, which is what `lambda = 1` gives.
   All five noncollinear cases are regenerated with the vendored `pw.x` instead.
4. **`mixing_beta` is not a detail for a total-moment constraint.** It is a uniform field
   proportional to the moment's error, a stiff global feedback: QE's input asks for 0.3 and
   still oscillates by several Bohr magnetons per iteration, and at the 0.7 default neither
   code converges in 150 iterations. The tests read the input's own value.
5. **A field breaks symmetry, so the symmetry group has to know about it.** A per-atom
   field alternating along `±z` is *how* an antiferromagnet is set up, and symmetrising with
   the nonmagnetic group erases exactly what the field was added to create. QE refuses a
   nonzero `B_field` together with a constraint (`input.f90:1614`) and so does this.

*The taper is a recorded choice, not an accident.* `factlist`/`pointlist` are a host-side
integer map that depends on the atomic positions, so QE's local field contributes a force QE
never computes. Two schemes are registered: `qe`, the integer nearest-atom map with the
linear taper, reproducing `make_pointlists` point for point and the default because it is
the one that can be compared; and `smooth`, a differentiable partition of unity in which the
positions stay in the autodiff graph and the constraint's force falls out. Elk's
fixed-spin-moment scheme is registered as `constrained_magnetization = 'fsm'` and validated
on its own: bcc iron held at **2.0 mu_B** where it wants 3.18, by a field the run finds for
itself, with nothing added to the energy. Two things about it are worth recording. **The
sign is not Elk's**, because the field is not: Elk's Hamiltonian term is `+(g_e/4c) sigma.B`
where QE's potential takes `-B`, so `B <- B + tau (m - m_fix)` becomes a minus here — and
getting it backwards does not oscillate, it drives the moment to saturation and converges
looking untroubled. **And the moment has to be part of the convergence test**: the field is
outside the density, so `dr2` falls below `conv_thr` while the moment is still far from its
target, and a run that stopped there would report an *unconstrained* answer under a
constrained heading. 

**The scheme was slow, and the reason was not the gain.** Elk updates the field after
*every* SCF iteration, so the controller reads a moment that has not finished responding to
the last nudge: instrumented on `fe-fsm.in`, the susceptibility it appears to see swings
between `+2591` and `-1252` mu_B/Ry between consecutive iterations, and the ringing takes
**1380 iterations** to damp below the 1e-3 the moment is held to. The gain is innocent —
Elk's `tau = 0.02` against a measured `1/chi` of `0.022` is already a Newton step. What is
wrong is *when* it is taken.

At converged density `m(B)` is smooth: 2.499, 2.274, 2.036, 1.837 mu_B at `B = 0`, -0.005,
-0.010 and -0.020 Ry on that case. So `fsm_update = 'secant'` (the default; `'elk'` keeps
the transcription) holds the field until the inner SCF has converged, then steps by the
susceptibility measured from the last two *converged* pairs, falling back to Elk's `tau`
step for the first one and whenever `chi` is not positive and finite, and never moving more
than `FSM_TRUST` times the previous step. Same field, same moment, same energy, **74
iterations instead of 1380**. The mixer keeps its history across a step, which is most of
why the later solves are cheap: the density at the previous field is a far better start
than the atomic guess, and the steps shrink.

*Notebook 11* (with P17).

**P19 — Spin spirals by the generalized Bloch theorem. ✅ DONE.** A flat spiral is not a
supercell calculation: with no spin-orbit coupling, a magnetization that turns by `q . R`
from cell to cell leaves a Hamiltonian invariant under a *combined* lattice translation and
spin rotation, so the states are still labelled by `k`. Elk's ansatz (manual §5.146) is the
one followed here,

    Psi^q_k(r) = ( U_up(r) e^{i(k + q/2).r},  U_dn(r) e^{i(k - q/2).r} )
    m^q(r)     = ( m_x(r) cos(q.r),  m_y(r) sin(q.r),  m_z(r) )

with `U` and `m_x, m_y, m_z` lattice periodic. New: `system/spiral.py`, `workflows/spiral.py`
(an `E(q)` scan and a Heisenberg fit), `Calculation.at_spiral_q`, and a `spiral` flag on
`SpinorHamiltonian`. **There is no QE counterpart** — `pw.x` has no spin spiral and a grep
of the vendored tree finds nothing — so this is P16's situation, and the references are the
literature (Sandratskii, *J. Phys. Condens. Matter* **3**, 8565 (1991); Kurz et al., *PRB*
**69**, 024415 (2004); Marsman and Hafner, *PRB* **66**, 224409 (2002)) plus Elk's source.

**How little it changes is the result.** The up component is built at `k + q/2` and the
down at `k - q/2`, each with its own `G+k` set (Elk's `gengkqvec.f90`), and one call to
`build_plane_wave_basis` on the concatenated `2 nk` list gives both a common `npwx` — which
is what keeps rule R7's padding, the `vmap` over k and the stick layout working unchanged.
The local term is untouched: in the rotated frame the potential is lattice periodic, so each
component transforms with its own index map and the 2x2 mixing stays pointwise. What gains a
component axis is `kinetic`, `fft_index`, `mask`, `vkb` and the stick rows, and nothing else
in the SCF knows a spiral is happening.

*Check met:* three identities against calculations that are **not** spirals, on a hydrogen
chain (one atom per cell, 5 bohr apart along `z`, 12 bohr from its images):

| identity | reference | agreement |
|---|---|---|
| `q = 0` | the ordinary noncollinear run | **4e-15 Ry** |
| `q = b3/2` | the **collinear** antiferromagnet of the doubled cell (P9's path) | **7e-13 Ry** |
| `q = b3/4` | a four-cell noncollinear supercell with moments at 0°, 90°, 180°, 270° | **3e-12 Ry** |

...plus `E(-q) = E(q)` to 1e-12, and `E(q + G) = E(q)` to 1e-12. (The tests assert 1e-9 on
all of them: the numbers above are what the committed inputs give at `conv_thr = 1e-11`, and
holding a test to them would be pinning the mixer's path rather than the physics.) The last two rows are the
sharp ones: they pin the two shifted spheres, the `q/2` split, the cross term between
components on *different* spheres and the rotated-frame potential together, and nothing else
does. What is compared is the energy **without** the Ewald term, because two cells of
different size do not agree on it to better than QE's own `upperbound` tolerance of 1e-7 Ry;
the Ewald sums are checked separately at the tolerance they deserve.

*Traps:*

1. **The supercell k-set that corresponds to a spiral's is shifted by `q/2`.** A spiral's
   plane waves sit at `k ± q/2`, not at `k`, so for `q = b3/4` and a primitive 1x1x4 grid
   the up component's wavevectors are odd multiples of `b3/8` — *not* supercell reciprocal
   vectors, but the supercell zone-boundary point. Sampling the supercell at Gamma instead
   compares two different calculations and disagrees in the third decimal, which reads
   exactly like a bug in the spiral.
2. **`E(q + G) = E(q)` needs a k-grid invariant under a shift by `G/2`.** Adding `G` to `q`
   moves one sphere by `+G/2` and the other by `-G/2`, which is the same calculation with
   every `k` shifted by `G/2` — an identity for the *sum* over the zone only if the k-set
   survives that shift. Measured: 2e-9 on a 1x1x4 grid, **2e-3 on a 1x1x3 grid**.
3. **The rotated-frame moment is gauge dependent and the energy is not.** `q` and `q + G`
   differ by relabelling `G` in each component, which multiplies the transverse
   magnetization by a lattice-periodic phase: its modulus is unchanged pointwise, so the LDA
   energy is, but its integral over the cell is not — 0.540 against 0.205 for the identical
   energy. A spiral's "moment" is an amplitude in a gauge, not an observable on its own.
4. **A soft magnetic surface will hide all of this.** At `degauss = 0.02` the spiral and its
   supercell converge to *different* minima and disagree in the fourth decimal; at 0.1 they
   agree to 1e-10. The same effect makes `E(q + 2G) = E(q)` — which is exact — fail in
   practice at `q + 2G`, where the magnetic solution lives in a gauge whose transverse
   magnetization winds twice across the cell and the uniform starting guess falls into the
   nonmagnetic minimum instead.

*Refused, and each for its own reason.* **Spin-orbit coupling, permanently**: it ties spin
to the lattice, so the combined translation-and-rotation the theorem rests on is not a
symmetry, and Elk refuses the same combination (`init0.f90`). **Symmetry, until the spin
space group is written**: only the operations with `S^T q = q` survive at all
(`findsymlat.f90`, transcribed as `invariant_operations`), those act on the rotated-frame
magnetization with a spin rotation of their own, and time reversal sends `q` to `-q` — so a
spiral run needs `nosym` and the full grid, which is the phase's real cost. **Ultrasoft and
PAW, until `q_ij(q)` is threaded through**: the cross-spin block of `becsum` pairs projectors
at two different k-points, so what it needs is the arbitrary-wavevector augmentation charge
`topology/augmentation.py` already builds for P16, and PAW additionally needs Elk's per-atom
phase `e^{-i q.tau/2}` (`zqss`). Using the plain `qq` instead leaves the overlap plausible
and the answer wrong, which is the failure this repository has now met twice.

*Memory and cost.* The spiral doubles what carries both a `k` and a `G` index — `vkb` at
`2 nk npwx nkb` complex, `kinetic`, the stick tables — but that is not the dominant cost.
Losing symmetry is: a spiral on a cubic crystal can multiply `nk` by up to the order of the
point group, so an `E(q)` scan is priced as `nq` runs of a `nosym` calculation.
`Calculation.at_spiral_q` is what keeps the scan affordable — everything except the
k-dependent tables is independent of `q`, and P16 measured what rebuilding the rest costs.

*Notebook 12.*

**P20 — DFT+U. ✅ DONE (simplified rotationally-invariant).** `pypresso/hubbard/`, four
modules and a term in the Hamiltonian. The functional is Dudarev's (PRB **57**, 1505
(1998)) with the `J0`/`beta` extension of Himmetoglu et al. (PRB **84**, 115108 (2011)) —
QE's `lda_plus_u_kind = 0`:

    E_U = sum_{I,s} [ (alpha + Ueff/2) Tr n^{Is} - (Ueff/2) Tr (n^{Is} n^{Is}) ]
        + sum_{I,s} [ sgn(s) beta Tr n^{Is} + (J0/2) Tr (n^{Is} n^{I,-s}) ],   Ueff = U - J0

with `n^{Is}_{m1 m2} = sum_{kv} f_{kvs} <phi_{Im1}|psi_kvs><psi_kvs|phi_{Im2}>` measured on
a chosen set of localised orbitals. **Only the energy is written down**; `v_ns` is
`jax.grad` of it and QE's `v_hubbard` is transcribed as the cross-check, which is the
arrangement P18 established for the magnetic field. The correction reaches the Hamiltonian
as *another separable term* — structurally `_nonlocal` with `wfcU` in place of `vkb` and a
block-diagonal `v_ns` in place of `D_ij` (`vhpsi.f90`) — so it costs two small matrix
products per band and no transform.

`ns` joins the **mixed state** beside `becsum`, because the Hubbard potential is built from
it before the Hamiltonian exists; `ns_ddot` joins `rho_ddot` so that `dr2` and the `ethr`
schedule are QE's; and `deband` gains `- sum ns v_ns` for the same reason it has
`- int rho v_scf`.

*Check met:* seven cases against the vendored `pw.x` at `conv_thr = 1e-10`, total energies
agreeing to **≤ 6.7e-9 Ry** and the Hubbard term itself to **≤ 4.6e-7 Ry**. Antiferromagnetic
FeO (QE's `pw_lda+U/`) with `U = 4.3 eV`, the same cell at `U = 1e-8` as the null test, the
same with `starting_ns_eigenvalue` (which converges to a *different* self-consistent
solution, -174.5374 against -174.4716 Ry), and the displaced-geometry force case; fcc nickel
at `nspin = 1`, with `ortho-atomic` projectors, and with `J0 = 1 eV`. `Tr[ns]` matches
`write_ns` to the last of the five decimals it prints (1e-5) throughout, and the eigenvalues
of `ns` to the three decimals it prints for those. **Forces come free**: the projectors are atomic orbitals
centred on the atoms, so `force_hub` — 2552 lines of Fortran, and for ortho-atomic
projectors it carries `d(O^{-1/2})/d tau` — is what `jax.grad` produces by differentiating
through `Calculation.at_positions`. Against QE on `lda+U_force.in`: **4.8e-6 Ry/bohr** on a
force of 0.14, of which the Hubbard term itself is a few percent. The analytic force path has no `force_hub` and refuses rather than returning
a force short of one term.

*Traps, all silent:*

1. **`Hubbard_projectors = 'atomic'` still applies `S`.** `orthoUwfc` runs `s_psi` on the
   atomic orbitals unconditionally, so `wfcU = S phi` and the projection is `<phi|S|psi>`.
   With a norm-conserving dataset `S = 1` and the distinction disappears, which is why
   testing on silicon would never find it.
2. **The atomic orbitals are renormalised at read time**, in the *generalised* metric:
   `upf_check_atwfc_norm` (`Modules/read_pseudo.f90`) rescales `chi` by
   `sqrt(<chi|chi> + sum q_ij <beta_i|chi><beta_j|chi>)` and QE prints the labels it
   touched. `Fe.pz-nd-rrkjus` and `Ni.pz-nd-rrkjus` both have an unnormalised `4s`. It does
   not matter for the starting wavefunctions (they are rotated anyway) and it does not
   matter for `atomic` projectors on a `3d` manifold — but `ortho-atomic` orthogonalises
   over **all** the atomic orbitals, so the `4s` enters the transform that produces the `3d`
   projectors. Cost of getting it wrong on nickel: 4e-3 in `Tr[ns]` and **7e-4 Ry** in the
   total, which reads exactly like a convergence difference. It has one side effect worth
   recording, because it is the only thing outside DFT+U that this phase moved: `chi` is
   also what the *starting wavefunctions* are built from, so every run now seeds its
   eigensolver slightly differently. Nothing converges anywhere else — but the
   fixed-spin-moment case (`fe-fsm.in`, P18) is a proportional controller that rings before
   it settles, and where it starts ringing from is exactly what changed: ~350 iterations
   before, **746** after, to the same moment and the same field.
3. **The `nspin = 1` factor of two is on the energy, not the potential.** `new_ns` halves
   `ns`, so it always means one channel's occupation; `eth` and the `deband` term are
   doubled at the end and `v_hub` is not. Differentiating the doubled energy gives a
   potential twice too large — and an SCF that converges perfectly well to the wrong number.
4. **`ns_adj` runs after the *first* iteration, not on the starting matrix.** QE calls
   `init_ns` in `potinit` and `ns_adj` in `electrons.f90` after `sum_band`, on the ns
   *measured* from the first diagonalisation, replacing both the input and the output copy.
   Applying `starting_ns_eigenvalue` to the Hund's-rule matrix instead steers the first
   Hamiltonian rather than the second, and lands on the other self-consistent solution.
5. **The Hubbard term has to reach `Hamiltonian.matrix` as well as `apply`.** The dense
   reference fixture uses the first and the SCF the second; a term in only one of them makes
   two solvers disagree about which operator they are diagonalising, and no pre-existing
   consistency test notices, because none of them switches a U on.
6. **`ns` is not a function of the density**, so a fixed-density run (`run_nscf`,
   `run_bands`) needs it passed in — the same situation PAW's `becsum` is in there, and it
   is a `ValueError` rather than a term quietly left out.

*Refused, by name rather than ignored:* `lda_plus_u_kind = 1` (Liechtenstein's full
formulation with `J`, `B`, `E2`, `E3`), `lda_plus_u_kind = 2` (the intersite `V`), the
background channels (`Hubbard_U2`), the orbital-resolved variant, the `wf` and `pseudo`
projector sets, noncollinear `ns_nc`, DFT+U on a spin spiral, and a symmetry group carrying
time-reversed operations (`new_ns`'s `colin_mag == 2` spin flip, which no case here
exercises: FeO's two sublattices are distinct *species*, so no operation maps one to the
other). Also not done: `hub_pot_fix`, QE's protocol of freezing `v_hub` when
`Hubbard_alpha` is nonzero for a linear-response U — the `alpha` energy term is implemented
and the protocol around it is not.

*Notebook 13.*

**P21 — Relaxing the spin spiral: `dE/dq`. ✅ DONE.** P19 made `q` a coordinate of the
calculation; this makes it one that can be *optimised*. The ground state of a spiral magnet
is the minimum of `E(q)`, and for an incommensurate pitch that minimum is a wavevector no
supercell can represent — so it has to be found by following a gradient, not by scanning a
grid in three dimensions. New: `forces/spiral.py`, `relax_spiral_q` in `workflows/spiral.py`,
`Calculation.at_spiral_q(q, rebuild_basis=False)`, `Calculation.state_kinetic`, a `kcart`
override on `PlaneWaveBasis.kinetic` and `build_projector_core`, and a `hessian_scale` on
`BFGSSettings`. **No QE counterpart again** — `pw.x` has no spiral, so there is no
`force_q` to transcribe either.

**It is P15's construction with one coordinate swapped.** The total energy is written down
as a function of `q` at *frozen* wavefunctions, occupations and eigenvalues, and `jax.grad`
of it is the gradient. What makes that the total derivative rather than a partial one is the
same stationarity argument, plus one step P15 does not need: **the frozen quantity is the
periodic part of the spinor, and that is the right one.** The stored coefficients *are*
`U_up` and `U_dn` — the sphere carries the `e^{i(k ± q/2).r}` — so freezing them freezes `U`
and lets the spiral turn, which is exactly what the SCF minimised over. `S` is the identity
(ultrasoft and PAW spirals are refused), so there is no Pulay term to carry either.

**Two terms out of seven depend on `q`,** and this is worth knowing because it is what makes
the gradient cheap: at frozen coefficients the rotated-frame density is a lattice-periodic
function on an FFT box that does not move, so `∫ vltot ρ`, Hartree, exchange-correlation,
Ewald and the orthonormality constraint are all independent of `q` and `grad` finds zero in
them without differentiating through a single FFT. Only `|k ± q/2 + G|²` and `vkb(k ± q/2)`
survive. The energy is written out in full anyway: evaluated at the converged state it must
reproduce the SCF total energy, and that identity is the only check on the five terms the
gradient never sees.

*Check met*, on the hydrogen chain of P19 — four identities, since there is no reference:

| identity | what it isolates | agreement |
|---|---|---|
| `spiral_energy` at the converged state = `etot` | the functional is the total energy | **2e-16 Ry** |
| central difference of `spiral_energy` vs `jax.grad` | the differentiation alone | **2e-9**, and `δ`-limited |
| central difference of a **re-converged SCF** vs the gradient | **stationarity** | 5.2e-5 at `δ = 0.02`, falling by 4 per halving of `δ` |
| `dE/dq` at `q = 0` and `q = b3/2` | `E(-q) = E(q)` makes both stationary | **1e-9 / 1e-12** |
| the same three, with the atom off the origin | the **structure factor**'s half of `vkb` | 2e-12 Ry, 1e-7 in `dE/dq` |

...and the relaxation itself, started at `q3 = 0.30`, reaches the antiferromagnet at
`q3 = 0.50003` in **6 SCF runs** with `max |dE/dq|` falling from 5.1e-3 to 1e-6. The last
row of the table is the sharpest thing in the phase: nothing about it is a tolerance
judgement, and it catches the two transverse components that a one-dimensional finite
difference never exercises.

*Traps:*

1. **The plane-wave sphere has to be frozen while differentiating, and rebuilt to move.**
   Which plane waves satisfy `|k ± q/2 + G|² ≤ ecutwfc` is a host-side decision that cannot
   be traced — and is *piecewise constant* in `q`, so freezing it is exact on each piece
   rather than an approximation. What it does not see is the jump where a plane wave crosses
   the cutoff, which is the Pulay error of a finite basis. **Measured:** against a
   sphere-rebuilding finite difference the gradient disagrees by 8.3e-4 at `ecutwfc = 25`
   and 8.3e-6 at 60 — and by 5.8e-4 at 40, which is *not* between them, and that
   non-monotonicity is the point. The error is not a truncation error and does not converge
   smoothly: it is the sum of the jumps that fall inside the particular window of `q` being
   differenced, and how many plane waves cross in that window depends on where the shells
   sit relative to the cutoff. Each jump is the size of a coefficient *at* the cutoff, so
   they all shrink as the basis approaches completeness, and by 60 Ry they have. A *step*
   of the relaxation rebuilds the sphere; only the derivative freezes it.
2. **The compiled gradient cannot follow a moved calculation.** It closes over the sphere,
   the k-list, the local potential, the Ewald sum and the projector positions it was built
   with, and all three of `at_spiral_q`, `at_positions` and `at_kpoints` go through
   `copy.copy`, so a cached one would be carried onto the moved object and *evaluated* — at
   the old cutoff or the old geometry, giving a plausible wrong number rather than an error.
   All three drop it, and on the `q` axis that costs nothing: a new `q` is a new `npwx` and
   would force a recompilation anyway. This is the opposite of the force's cache, which is
   valid at any geometry because the geometry is an *argument* to it.
3. **Every obvious test system has its atom at the origin, where the structure factor is
   one for every `q`.** `vkb(k ± q/2)` has two `q`-dependent factors and the second is
   `e^{-i(k ± q/2 + G).tau}`; with `tau = 0` it is identically one, so a hydrogen chain
   validates only half of `vkb`. Its contribution turns out to **cancel** — the derivative
   brings down `∓ i tau/2` on the two components and `dvan_so` is spin-diagonal without
   spin-orbit coupling, so the two halves of each diagonal term subtract — but that is a
   fact to *test*, not to assume, and it stops being true the moment `D` gains an
   off-diagonal block. Translating the crystal is a symmetry of a spiral (a lattice
   translation with the spin rotation the theorem pairs it with), so `E(q)` and `dE/dq` are
   both unchanged by it, and that is the check.
4. **BFGS's initial inverse Hessian is wrong by two orders of magnitude for `q`.** QE's
   guess is the inverse metric — a curvature of 1 Ry/bohr² — which is the size of a chemical
   bond only because Rydberg atomic units were chosen to make it so. A magnetic energy
   surface is milli-Rydberg over a coordinate of order one, so the first Newton step comes
   out a hundredth of the trust radius and the relaxation crawls (measured: `q3` moving by
   0.002 a step, and 15+ steps to travel 0.2). `BFGSSettings.hessian_scale` is set so the
   first step is exactly `trust_radius_ini` long — "with no curvature information, take a
   steepest-descent step of the length the trust radius allows" — and every step after it
   uses a measured curvature. Six steps instead of fifteen.
5. **The trust radius has no natural unit either**, so the three radii are fractions of the
   Brillouin zone's linear size (the cube root of its volume), not of the shortest reciprocal
   vector — which in an anisotropic cell is the one along a vacuum direction the physics
   never uses.
6. **The energy convergence threshold has to be loose, and the gradient one carries the
   physics.** Near the minimum the energy differences being resolved are the size of the
   basis-set jumps of trap 1 (measured at `ecutwfc = 40`: 3e-6 Ry between two wavevectors
   0.006 apart, where the physics is 8e-7), so `etot_conv_thr` defaults to 1e-5 Ry and the
   way to tighten anything is to raise `ecutwfc` first.
7. **A magnetic field is refused rather than corrected.** The field's energy is deliberately
   outside the reported total (P18), so the converged state is stationary for a *different*
   functional than the one being differentiated and the missing term would be invisible.

*Warm start.* The wavefunctions cannot travel between steps — a new `q` is a new plane-wave
sphere, so the coefficients are on a basis that no longer exists — but the density can,
because in the rotated frame it is lattice periodic on a grid that does not move. That is
`update_pot.f90`'s `pot_extrapolation = 'file'` rather than its atomic extrapolation:
nothing has moved for an atomic superposition to follow. It is worth most of the SCF: on the
chain the first wavevector takes 9 iterations and every one after it takes 2 to 6.

*Notebook 14.*

**P22 — The SCF as a root-find: the residual, its Jacobian, and Kerker. 🔶 FIRST PASS DONE.**
The SCF is a fixed point `rho = F(rho)`, and P5 solves it the way `mix_rho.f90` does — by
iterating `F` and damping the iteration. This phase writes the same fixed point as a
*residual* `r(rho) = F(rho) - rho` and solves *that*, with a Jacobian rather than a fitted
one. New: `scf/residual.py` (the pure step and its Jacobian action), `scf/solvers.py`
(the `scf_solver` registry and Newton-Krylov), Kerker preconditioning in `scf/mixing.py`,
a `custom_jvp` on `bisect_fermi`, and `benchmarks/al-slab.in`, which is the first entry in
that directory that is a *convergence* case rather than a timing one.

**Say the calibration first, because this phase is easy to oversell: Anderson mixing already
is a quasi-Newton method on this residual.** `scf/mixing.py` fits a secant Jacobian to the
residual history and takes the Newton step inside its span; QE's Broyden is the same idea.
Nothing here replaces iteration with a closed form. What changes is that the Jacobian is
computed rather than fitted — and the measurements below say that buys less than it costs.

**What pyqula had already established, and what does not need repeating.** The
`scftk/densitydensity_jax.py` and `scftk/vjinteraction_jax.py` experiments carry two
negative results, both transferable and both load-bearing:

1. **Minimising the physical energy off the self-consistency surface fails.** Written down
   properly — the grand potential of the trial Hamiltonian, minus `Re Tr[x rho]`, minus the
   double-counting — the gradient is *exactly* right (`jax.grad` ~ 1e-16 at converged fixed
   points). The Hessian there is nevertheless **indefinite** (6 negative, 13 zero, 13
   positive of 32 on a dimer): the SCF solution is a **saddle** of the off-shell extension.
   L-BFGS with a correct gradient walks away from the answer. This is why the residual, and
   not the energy, is what gets driven to zero here.
2. **A scalar-loss optimiser on `||r||^2` stalls**, because it sees only `J^T r` and discards
   the residual *vector*. L-BFGS-B did not reach 1e-6 in 3000 iterations on a 60-orbital
   chain. What worked there was matrix-free Levenberg-Marquardt and Newton-Krylov, which use
   the vector — so this is a root-finder, not a minimiser.

**The dense Jacobian is not an option and the number is worth writing down.** It is
`(nspin * nr)^2`: on the smallest case here, a 12x12x72 grid, that is a 10368 x 10368 matrix
per spin channel — and `nr` is the *dense* grid, so an ordinary production cell puts it past
any machine. Everything is matrix-free: GMRES asks only for `J v`.

**The headline technical finding: autodiff through the eigensolver is not the way, and this
is now measured rather than argued.** `ScfResidual.jvp` differentiates one whole SCF step
with `jax.jvp` — Davidson's `lax.while_loop` included, which forward mode supports and
reverse mode does not. `jvp_finite_difference` central-differences the same residual and
shares no machinery with it. On the aluminium slab:

| starting wavefunctions | `jax.jvp` vs central difference | cost |
|---|---|---|
| cold (pseudo-atomic orbitals) | **109% different** | 5.9 s |
| warm (converged `psi`) | 0.8% different | 2.9 s vs 0.4 s |

The cold-start disagreement is not a tolerance to be tuned: differentiating Davidson's
trajectory from the atomic orbitals is the derivative of a *different map*, one that merely
lands in the same place. Warm-started, Davidson exits in one or two steps, so its tangent is
a one-step approximation to the eigenvector response — good enough to hand a Krylov solver a
direction, and not the response operator. And it is **4-7x slower** than the finite
difference, because forward mode through the `while_loop` costs more than two extra primal
solves do when both start from a converged guess. So `finite-difference` is the default
backend, `autodiff` is kept and tested, and the whole table is the empirical case for
writing the response down (P22c below) rather than taking it from the eigensolver.

**One thing did have to be written down already: `dE_F`.** `bisect_fermi` halves a bracket
and then refines with Newton, and **differentiating a bisection is silently useless** — every
number in it is a midpoint chosen by a comparison, so the tangent is zero or the derivative
of the bracket. Worse, it would have been silently *inconsistent*: the Newton refinement
gives the right answer by accident (differentiating a converged contraction is the implicit
derivative), so Methfessel-Paxton and cold smearing would have been right while Gaussian and
Fermi-Dirac, which return the raw bisection, were wrong. It now carries a `custom_jvp` with
the implicit derivative of `N(E_F) = nelec`, in all four tangent slots, matching central
differences to 1e-11 on all four smearings. `d(wgauss)/dx` in it comes from `jax.grad` and
`w0gauss` is the test of it, which is what pins the sign convention. **This term is the
Fermi-level shift of metallic linear response and outlives the solver that motivated it.**

**Kerker, and the trap in it.** `mixing_mode = 'kerker'`/`'TF'` replaces the scalar `beta`
with `beta |G|^2/(|G|^2 + q_TF^2)` on the residual in G-space — `mix_rho.f90`'s
`approx_screening`, and an approximate *inverse Jacobian*: in a metal the dielectric function
diverges as `q^-2` at long wavelength, and dividing that out is one FFT per iteration.
**`q_TF` is derived from the cell and must not be picked by hand.** A first version here
hardcoded 1.5 1/bohr; QE's `rs = (3 Omega/4 pi nelec)^(1/3)`, `q_TF^2 = (12/pi)^(2/3)/rs`
gives 1.008 for the slab, so the hand-picked value over-screened by 2.2x in `q^2` — and
over-screening is *worse than not preconditioning*: it cost 15 iterations against 14 at one
vacuum and 48 against 28 at another. QE's `local-TF` (`approx_screening2`), a
space-dependent screening length, is refused by name rather than substituted.

**The measurement, on the case built for it.** `benchmarks/al-slab.in` is a five-layer
Al(100) slab: half the cell is a metal, where screening diverges as `q^-2`, and half is
vacuum, where there is none. Plain Anderson reaches **+105 Ry on its second iteration**
before recovering. Evaluations of `F` — i.e. diagonalisations, which is the only currency in
which a mixer's iteration and a Krylov iteration are comparable — to `conv_thr = 1e-8`:

| vacuum (bohr) | Anderson | Anderson + Kerker | Newton-Krylov |
|---|---|---|---|
| 16 | 24 | **14** | 19 |
| 32 | 34 | **20** | 74 |
| 48 | 32 | **28** | 139 |
| 64 | **35** | 36 | 123 |

All twelve runs converge to the same energy, to <= 7e-8 Ry — which is the real result of the
table, because a solver that reached a *different* fixed point would be a bug in one of them.

**Newton-Krylov does not win on cost, anywhere, and the reason is structural rather than a
tuning failure.** One outer iteration is one residual evaluation plus one per GMRES iteration
plus one or two per line-search backtrack, and **every one of those is a diagonalisation** —
84% of an SCF step (`PERFORMANCE.md`). Anderson spends exactly one diagonalisation per
iteration and a negligible least-squares solve. So Newton-Krylov must cut the outer count by
more than the inner Krylov work adds, and it cannot: as the conditioning worsens the inner
count grows to replace the outer one, which is what the 19 -> 74 -> 139 column is. The
predicted "outer iteration count is flat in the conditioning" is *true* — it is 1, 4, 5, 5
across the four rows — and it does not help, because the flatness is bought with GMRES iterations.
Anderson's own count, meanwhile, is nearly flat too (24, 34, 32, 35): an 8-deep residual
history is enough to span the few badly-conditioned long-wavelength directions this cell has.

**Where it does win is not speed, and this was the point.** Newton is **stability-blind**: it
converges to a root whether or not that root is a minimum, while damped mixing is a discrete
relaxation dynamics that falls into *stable* fixed points. That makes the residual solver the
only way here to reach an **unstable SCF solution** — the non-magnetic state of a magnetic
metal, the symmetric structure at the top of a Peierls or Jahn-Teller instability, the
reference state a stabilisation energy is quoted against. `test_scf_solvers.py` demonstrates
it on bcc iron. From `starting_magnetization = 0.05`, on the same cell and the same k-grid:

| | E (Ry) | moment (mu_B) | `F` evaluations |
|---|---|---|---|
| Anderson, `nspin = 2` | -55.44642602 | **+3.4053** | 44 |
| Newton-Krylov, `nspin = 2` | **-55.38228995** | -0.0003 | 274 |
| Anderson, `nspin = 1` (independent reference) | **-55.38228995** | -- | 28 |

Anderson amplifies the small moment into the ferromagnetic ground state, which is the
*stable* fixed point and the physically right answer for a ground-state calculation.
Newton-Krylov, given the same start, collapses the moment and converges on the **non-magnetic
solution** — a fixed point of the SCF that no mixer *reaches* from a physical start.

> **Corrected at P24a.** This paragraph originally called that solution a saddle "one no
> mixer can be made to hold". It is not a saddle in the linear sense: it is **metastable,
> with a finite basin**, and mixing holds it perfectly well once it is in it — a kick of
> 0.05 in the atomic magnetization's shape decays back to it, and one of 0.20 runs away.
> The measurement is in P24a, and it was forced by a **3.5 eps** change in how `|psi|^2`
> is evaluated flipping the test that made this claim. The saddle language *is* right for
> the DFT+U nickel case below, whose 2% kick runs away. It is also why the table's Newton
> column is a number this phase should not have asserted from a distant start: P22's own
> warning, three paragraphs down, says which root that finds is not systematic.

**The validation is free and independent:**
that `nspin = 2` energy has to equal a plain `nspin = 1` run's on the same cell, and it
agrees to 4.3e-10 Ry. The 64 mRy between the two rows is iron's magnetic stabilisation
energy, which is the quantity such a reference state exists to give — on a 2x2x2 k-grid
chosen to run in seconds, so it is a demonstration of the *capability* and not a converged
number.

It costs 274 diagonalisations against 44 — but the 44 do not produce this number at all.

**And there is a trap in it, which is silent.** With the Kerker preconditioner *off*, the
same solver from the same guess converges — reporting an accuracy below `conv_thr`, exactly
as before — on the **ferromagnetic** solution instead (`-55.44642602`, `m = +3.4052`). A
Newton method is stability-blind only to the extent that its inner solve actually delivers
the Newton direction; with a badly conditioned Krylov system the inexact step degrades
towards a damped-mixing step, and a damped-mixing step flows to the *stable* fixed point. So
on this kind of problem the preconditioner is not a tuning knob for speed — **it decides
which physics comes out**, and both answers look equally converged. Both are pinned by
`test_an_inexact_newton_is_only_as_stability_blind_as_its_inner_solve`.

**DFT+U joins the packed state rather than being refused.** `ns` is not a function of the
density — the Hubbard potential is built from it before the Hamiltonian exists — which is why
`mix_rho.f90` carries it inside `mix_type` and the mixing loop mixes it beside `rho` and
`becsum`. A root-finder solves for it on the same footing, and nothing about the Jacobian
action had to be added: `v_hubbard` is already `jax.grad` of the Hubbard energy (P20) and
`wfcU` is fixed while the atoms are. `ns_ddot` joins `accuracy_of` exactly as it joins the
loop's `dr2`, so `conv_thr` still means one thing. On fcc nickel with `U = 3 eV` the solver
reaches Anderson's fixed point to every digit printed — `-86.41841670` Ry, and both `ns`
traces (4.9961, 2.9994) identical — which is the check that the packing is right.

**And DFT+U has a saddle of its own, but reaching it needs something the code did not have.**
`init_ns` fills the occupation matrix diagonally by **Hund's rule**, read off
`starting_magnetization` — so on a magnetic species the starting `ns` is 1.0 in every majority
`d` orbital against 0.8 in every minority one whether `starting_magnetization` is 0.7 or 0.05.
Turning that knob down does *not* start a run near the unpolarised solution: the density is
barely polarised and the occupation matrix is already deep in the ferromagnetic basin, and the
Hubbard potential it generates undoes a kick applied to the density alone. So `run_scf` gains
**`starting_ns`**, the third member of the mixed state beside `starting_density` and
`starting_becsum`, with `hubbard.uniform_ns` and `hubbard.spin_averaged_ns` to build one and
`hubbard.ns_shape` to say what shape it must be. `ns_adj` is skipped when it is given —
`starting_ns_eigenvalue` exists to steer a *fresh* run towards one of several solutions, and
overriding an explicitly requested matrix with it would defeat the only mechanism that targets
a solution reliably.

With that, the saddle is reachable and is demonstrated the way pyqula demonstrated its own:
**perturb it and see which solver comes back.** On fcc nickel with `U = 3 eV`, from a
spin-symmetric density and `uniform_ns`, the SCF converges to `-86.20620046` Ry at
`m = 0` — the non-magnetic solution, confirmed by an independent `nspin = 1` run giving the
same number. Kick it by 2% along the magnetization direction, in *both* `rho` and `ns`, and
hand the identical perturbed state to both solvers:

| | E (Ry) | moment (mu_B) | `F` evaluations |
|---|---|---|---|
| Anderson | -86.41841670 | **+2.000000** | 9 |
| Newton-Krylov | **-86.20620046** | +0.000005 | 31 |

Mixing amplifies the kick into the ferromagnet — which is *what makes the state a saddle* —
and Newton puts it back, to the saddle's own energy to eight decimals. Both forcing terms
tried (0.1 and 0.5) return to it.

**One finding here is a warning rather than a feature.** Started *far* from the saddle —
from Hund's rule, the default — which root Newton-Krylov lands on depends on the inner-solve
accuracy in no systematic way: `forcing = 0.5` gave the ferromagnet, `0.05` gave a third
solution at `m = -0.343`, and `0.01` gave the ferromagnet again. All three converged and all
three reported an accuracy below `conv_thr`. That is the same mechanism as the aluminium
preconditioner trap, seen from the other side, and the conclusion is: **on a problem with
several solutions the starting state is what targets one, not the solver's tuning.** From a
perturbed saddle — a genuinely small perturbation — the result is robust; from far away it is
not, and no setting makes it so.

**Refused by name rather than approximated:** external and constrained magnetic fields
(the field is driven by a secant *outside* the density, so `F` is not a function of the state
alone — P18's own convention makes this unavoidable), tetrahedron and `from_input`
occupations (built on the host, and the second is not a function of the eigenvalues at all),
and spin spirals.

**Three driver constraints the solver imposes, all of them costs.** `ethr` is fixed and tight
for the whole solve — QE's schedule is most of why a mixing run is cheap, and a root-finder
handed a moving target does not converge, so every `F` here costs what a *converged* mixing
iteration costs rather than what an early one does. The mixing loop still runs afterwards,
handed the solution as a starting density: it converges in one iteration and is what builds
the `SCFResult`, so no energy term has a second implementation. And a warm-up of ordinary
mixing is allowed (`warmup`), because Newton converges from a good enough guess and wanders
from a bad one, and the atomic superposition is a bad one for exactly the systems worth using
this on.

**What is not done, and it is the part that would make the cost case.** `P22c` — replacing
the differentiated eigensolver with a **Sternheimer `custom_jvp`**: `chi_0 = drho/dV` from
`(H - eps_n S)|dpsi_n> = -P_c dV|psi_n>`, one projected linear solve per occupied band, with
`K = dV/drho` already free from `v_of_rho` by D1. That is the exact response instead of a
0.8% approximation, it is degeneracy-safe in D4's sense (it never divides by
`eps_n - eps_m`), and — the point — **a projected CG solve is far cheaper than a Davidson
solve**, which is the only way the inner Krylov iterations stop costing a diagonalisation
each. It is also D3's implicit-differentiation backward pass and the core of DFPT, so the
argument for writing it was never really the SCF solver. **Until it exists, `mixing` is the
default and should stay the default**, and the honest summary of this phase is: Kerker is a
genuine and nearly free win, `dE_F` is a term the code needed anyway, and the exact-Jacobian
solver is a capability (unstable solutions) rather than a speedup.

*Notebook 17.*

**P23 — Continuing a calculation across a change of spin regime. ✅ DONE.** An unpolarized
run, a collinear one and a noncollinear one are three descriptions of the same electrons,
and the expensive part of all three — the charge density — is very nearly the same object.
This phase maps the converged state of one onto the starting state of another. New:
`scf/continuation.py`, `System.with_spin`, `SCFResult.becsum` and `SCFResult.system`,
`run_scf(starting_from=..., starting_wavefunctions=...)`, and a `span` argument on
`Calculation.starting_wavefunctions`.

**One representation, and every direction is the same code.** The three regimes differ only
in how they write the same pair `(n(r), m(r))`: `[n]`, `[n_up, n_dw]`, `[n, m_x, m_y, m_z]`.
So a promotion is *decompose, decide what `m` should be, recompose* — `spin_components` and
`from_spin_components` — and a demotion is the same function read the other way. There is no
1→2, 2→4 and 4→2 path to keep consistent. The collinear magnetization is placed on `z`,
which is what makes 2 and 4 one representation and the promotion between them a rotation.

**The decision is taken once and applied to the whole mixed state.** `run_scf` mixes
`(rho, becsum, ns)` together and its own docstring says that giving one without the others
starts the run from two states at once; the same applies here, so `_SpinTransfer` is built
from the *density* — the only part big enough to say reliably whether the source is magnetic
— and then reused for `becsum`. That is the requirement `_becsum_split` already states for
the atomic start: the two guesses have to agree about how polarized the atom is, or the
first iteration contradicts itself.

**The trap, and it is the whole reason the phase is not three lines.** Nothing in the SCF
breaks spin symmetry on its own. Promote a converged unpolarized density to two identical
channels and the run converges straight *back* to the unpolarized solution, having found a
stationary point rather than the magnetic one — and it reports convergence, because it did
converge. The magnetization has to be put in by hand, exactly as `starting_magnetization`
puts it into a fresh run, which is what `magnetization="auto"` does: carry the source's when
it has one (`int |m| > 1e-4 mu_B` — the *absolute* magnetization, since an antiferromagnet's
signed one is zero), and otherwise seed the target's atomic magnetization on top of the
converged charge. `"carry"` raises rather than starting on the symmetric solution, `"seed"`
forces the seed — which is how a *different* magnetic state is reached from the same charge
— and `"none"` starts unpolarized deliberately.

**What `pw.x` has is two pieces of this and neither is the whole.** `startingpot = 'file'`
reads a density whose `nspin` need not match, and `read_rhog` handles the mismatch by
`infomsg('read_rhog', 'some spin components not found')` and zero-filling — so QE's own
continuation from an `nspin = 1` file into an LSDA run starts *unpolarized* and converges
back to the non-magnetic solution, silently. And `nc_magnetization_from_lsda`
(`PW/src/potinit.f90`) does rotate a collinear `m` onto a noncollinear one — but only on the
`lforcet` path (the force-theorem magnetocrystalline-anisotropy calculation), and it uses
`angle1(1)`/`angle2(1)`, **species one's angles, for the whole cell**. That restriction is
real and is kept here rather than papered over: a collinear source carries one scalar field
and cannot point two ways at once, so a target whose species point along different axes is
*refused*, with the message naming `magnetization="seed"` as the way to keep the converged
charge and take the magnetization from the atomic superposition, which does honour
per-species angles.

**The demotion has to *find* the axis, and the obvious way to find it is wrong.** Going back
down from `nspin_mag = 4` to 2 means writing a vector field as one scalar on a fixed axis,
which is only possible if the state is collinear — and QE's `pw_noncolin` benchmark points
its moment along `x`, so reading `m_z` would give zero. The axis is the dominant eigenvector
of `M_ab = int m_a(r) m_b(r) dr`, **not of `int m(r) dr`**: an antiferromagnet's signed
integral vanishes and leaves the axis undefined, while the second moment is blind to the sign
and finds it. The other two eigenvalues are then the refusal test — a genuinely noncollinear
state has no collinear form and is refused rather than projected — and the projection is
`m . n` rather than `|m|`, so the sign structure that makes it an antiferromagnet survives.
The eigenvector's own sign is arbitrary and would flip which channel is "up" (a global spin
flip: harmless, and confusing to read), so it is fixed by the signed integral and, where that
vanishes, by the point carrying the most magnetization.

**The wavefunctions are transferred as a *span*, not as wavefunctions.** What is handed over
is a set of vectors for the first Rayleigh-Ritz, exactly as `wfcinit` hands over the
pseudo-atomic orbitals — so the set need not be orthonormal in the target's overlap operator
(with spin-orbit coupling `S` mixes the components and it is not), need not number `nbnd`,
and need not be sorted. 1 → 2 seeds both channels with the same states, as the atomic start
does; 2 → 4 makes the two channels the two components of `2 nbnd` spinors, which is
`_as_spinors` applied to states that are already self-consistent and is exactly the `nbnd` a
noncollinear run asks for; 4 → 4 (spin-orbit on or off) carries them untouched. **A spinor is
not split back into two channels** — its components are not separately normalised and with
spin-orbit coupling not separately eigenstates — and a *magnetic* noncollinear target usually
gets no wavefunctions at all, because its smaller symmetry group means `irreducible_BZ` hands
it k-points the collinear run never had. Dropping them costs a few Davidson steps and is a
warning, never an error; carrying states belonging to different k-points would be wrong.

**`with_spin` rebuilds the k-points rather than relabelling them, and both ways of not doing
so are silent.** `dataclasses.replace(system, nspin=2)` leaves the weights with `degspin` in
them, so every electron is counted twice and the Fermi level comes out somewhere else; and it
leaves the k-*set* reduced with the wrong group, where a magnetic noncollinear run needs the
points `irreducible_BZ` adds (22 where the input lists 11, in QE's own `pw_noncolin`
benchmark). An automatic grid is reduced again with the target's group and an explicit list
goes through `expand_to_subgroup`, both exactly as `build_system` does them. The demotion
direction does not re-reduce an explicit list — it is then merely wasteful, not wrong.
`nbnd` is doubled crossing into `nspin = 4` and halved coming back, since a spinor band holds
one electron where a collinear band holds two.

**The measurements, all six validated as identities** (`tests/regression/test_continuation.py`)
— the continued run must reach the *same* self-consistent solution as a fresh one, since a
starting guess is a guess and nothing else:

| case | fresh | continued | agreement |
|---|---|---|---|
| Si, `nspin` 1 → 2 (seeded, decays to zero) | 5 | 4 | 2e-9 Ry |
| Si, `nspin` 2 → 4 (nonmagnetic, `nspin_mag = 1`) | 5 | **1** | 1e-9 Ry |
| bcc Fe, 2 → 4 with `angle1 = 90` | 25 | **1** | 2e-8 Ry |
| bcc Fe, 4 → 2 back again (the axis found, not read) | 30 | **1** | 4e-8 Ry |
| bcc Fe, 1 → 2, magnetization seeded | 30 | 27 | 5e-9 Ry |
| Pt, scalar PAW → fully-relativistic PAW + `lspinorb` | 13 | **7** | 2e-10 Ry |

**Where the saving is large and where it is not, and the reason is the same in both.** What
carries over is the *charge*; when the charge is the whole answer — a nonmagnetic run
rewritten as a spinor one, a converged moment merely rotated onto another axis — the
continued run converges on the state it was handed, in one iteration. When the run has to
*find* a magnetization the source does not have (bcc Fe, 1 → 2), the magnetization is the
slow variable and the saving is a few iterations out of thirty. What that pair does give for
free is the **magnetic stabilisation energy** — 21.6 mRy for bcc iron here, two runs of the
same cell — which is P22's non-magnetic reference state reached by the other route: by
constraining the regime rather than by out-running the instability with a root-finder.

**Switching spin-orbit coupling off with the same dataset is refused, and the refusal is
QE's.** For an ultrasoft or PAW pseudopotential `lspinorb = .false.` with a fully-relativistic
file needs `average_pp`, which QE itself refuses for `tvanp` (and for `lda_plus_u`) in
`PW/src/average_pp.f90`. So the reversible spin-orbit toggle is a *dataset* swap — scalar
PAW against fully-relativistic PAW — which the continuation handles by carrying the density
and re-seeding `becsum`: the projector counts differ (18 against 34 for platinum), and a
source `becsum` of the wrong shape is dropped with a warning rather than reshaped, because
that is a different pseudopotential and not a different spin regime.

**Also refused by name:** a Hubbard `U` crossing into `nspin = 4` (`ns_nc` is refused by name
in P20, so there is nothing to promote into), a grid or an electron count that does not
match, and a spiral target for the wavefunctions, whose two components live on different
plane-wave spheres.

*Notebook 18.*

**P24 — Linear response by autodiff: the velocity operator, the Sternheimer equation,
and the dielectric constant. ✅ DONE.** `pypresso/response/` — `velocity.py`,
`sternheimer.py`, `efield.py` — plus `Calculation.at_kcart`,
`Calculation.symmetrize_directional`, and `symmetrize_vector_density` /
`symmetrize_atom_tensor` in `system/symmetry.py`. This is P11's remaining half, P22c, and
D3's backward pass, and it ends on QE's own `ph_base/si.phG.in` benchmark.

**The velocity operator is what never tabulating a form factor was for.** Rule D2 says the
whole of `H(k)` is built by differentiable JAX code from `k` so that the velocity operator
falls out of differentiating it; `pseudo/formfactors.py` integrates every radial transform
directly rather than interpolating QE's `dq = 0.01` table for exactly this reason, and
`pseudo/harmonics.py` avoids `ylmr2`'s `atan2` for the same. Cashing it in is one
`jax.jvp` of `H(k)|psi>` at a **frozen sphere** — `at_kcart` is `at_spiral_q`'s
`rebuild_basis = False` on the `k` axis — and nothing is derived for `[V_NL, r]`, which
`commutator_Hx_psi.f90` hand-codes term by term from `gen_us_dj` and `gen_us_dy`.

Three things fall out that are not obvious until the gradient is written:

- **The local potential costs nothing.** It is a field on a box that does not move with
  `k`, so its tangent is symbolically zero and the `jvp` never issues its FFTs.
- **The overlap carries a velocity too.** `S(k) = 1 + sum |beta(k)> q <beta(k)|` has the
  same `k` in it, so the band velocity is the *generalised* Hellmann-Feynman derivative
  `<psi|dH/dk - eps dS/dk|psi>` — `commutator_Hx_psi`'s ultrasoft correction, from the
  same `jvp`. **Ultrasoft and PAW both come along rather than being refused**, and PAW
  adds nothing to the differentiation and one thing to the *setup*: `ddd_paw` is built
  from `becsum` rather than from the density, and since it multiplies `vkb(k)` it belongs
  to `dH/dk` as much as to `H`. Handing it in is not optional and is not defaulted —
  the first version of `band_velocities` passed `None` and was **2% wrong** on PAW
  silicon (1.7e-2 Ry/bohr against 8.7e-7) while printing a velocity that looked
  ordinary, so the constructor now refuses it, exactly as it refuses a Hubbard `U`
  without its `ns`.
- **Nothing dense is ever formed.** `dH/dk` as a matrix is `npw^2`, the same reason a dense
  diagonalisation is a test fixture here. One `jvp` per cartesian direction gives
  `v_a|psi>` for every band at every k-point; the three are separate calls because a
  `jacfwd` would hold three tangents of `vkb` at once.

*Check met:* against a central difference of the band structure at a generic k-point,
**1.2e-6 Ry bohr** norm-conserving, **8.6e-7** ultrasoft and **8.7e-7** PAW, all at a step
where that is the difference's own truncation error. `dS/dk` is identically zero for the
norm-conserving dataset and 1.5e-2 for the ultrasoft one, which is what makes the second
case a test of anything; the third is what the `ddd_paw` guard above rests on. A fourth
check shares nothing with any of them: at `Gamma` an inversion-symmetric crystal has states
of definite parity, so every band velocity is exactly zero, and what comes out is 1e-4 —
the eigensolver's tolerance. (The finite difference has one failure mode of its own, and it is the
reference's: eigenvalues come back **sorted**, so a step straddling a band crossing
compares two different bands and disagrees by the band width — 0.21 Ry bohr at `h = 1e-3`
against 1.2e-6 at `1e-4`.)

**The Sternheimer equation replaces the sum over states, and it is P22c.**
`(H - eps_n S + alpha Q) |dpsi_n> = -P_c^+ dV |psi_n>`, one projected conjugate gradient
per occupied band — `cgsolve_all.f90`, `ch_psi_all.f90`, `orthogonalize.f90`,
`h_prec.f90`, `setup_alpha_pv.f90` and `incdrhoscf.f90` transcribed, with the Fortran's
repacking of the unconverged bands becoming a **mask at a static shape**, as `cegterg`'s
did on the way into `solvers/davidson.py`. No empty states, and no division by
`eps_n - eps_m`, which is rule D4's requirement rather than a convenience: a crystal is
degenerate everywhere by symmetry.

*Check met, twice, against things sharing no machinery with it.* `chi_0 dV` for a
symmetry-breaking `cos(G.r)` probe against a central difference of the density under the
same perturbation: **8e-7 relative**. And `chi_0 K` — the SCF Jacobian, with
`K = dV_scf/drho` free from one `jvp` of `v_of_rho` — against P22's
`jvp_finite_difference`: **4.0e-4 relative**.

**That second number came with a finding about P22 rather than about this phase.** The
agreement is 4.0e-4 only at the finite difference's *own* optimal step. Sweeping it gives
a textbook U — 0.3 → 8.3e-2, 0.1 → 9.7e-3, 0.03 → 8.0e-4, **0.01 → 4.0e-4**, 0.003 →
1.0e-3 — with truncation above and noise below. P22's default step is chosen for a
gradient and sits two orders below the minimum, where the same two numbers disagree by
**11%**. So P22's Jacobian was noise-limited on this cell and could not have said so on
its own; its 0.8% agreement between `jax.jvp` and the difference was two noisy numbers
agreeing. On silicon at `ecutwfc = 12` the Sternheimer solve costs **0.5 s** against the
`jvp`-through-Davidson route's **3.5 s**, and is exact rather than accurate to `ethr`.

**The electric field is the one perturbation a periodic code cannot write down**, and the
commutator that makes it computable is the velocity operator again. `V = E.r` is neither
bounded nor lattice periodic, so what is solved for is `P_c r|psi>` through
`(H - eps_v S) P_c r|psi_v> = P_c^+ [H - eps_v S, r]|psi_v>` (`dvpsi_e.f90`) — and in the
periodic gauge `dH/dk_a = i [H, r_a]`, so the right-hand side is stage 1's output with a
factor of `-i`. The self-consistent loop is `solve_e.f90`'s, and **its kernel needed no
transcription at all**: `dv_of_drho.f90` is the Hartree kernel with its `G = 0` component
dropped plus `f_xc`, and `scf/potential.py`'s `hartree` already drops that component while
`v_of_rho` is already a differentiable function of the density (D1), so `K` is one
`jax.jvp`. The exchange-correlation kernel QE tabulates in `setup_dmuxc` is the second
derivative of the energy this code writes down once.

*Check met.* On QE's own `ph_base/si.scf.in` — the ten-point wedge, reproducing its total
energy to 2.6e-9 Ry — the loop converges `|ddv_scf|^2` to 3.8e-15 in 18 linear-mixing
iterations at `alpha_mix = 0.7` (QE takes 5 with Broyden; the fixed point is the claim,
not the trajectory) and gives

    epsilon_infinity = 13.806646105   against ph.x's   13.806689470

a difference of **4.3e-5**. What is left is the same thing that puts a floor under the
eigenvalue comparison (`tests/tolerances.py`): QE interpolates every radial form factor
from a `dq = 0.01` table where this code integrates it directly. The tensor comes out
**diagonal to 3.6e-15** with nothing imposing the crystal class. (The reference is the
*vendored* `ph.x`, regenerated: `ph_base`'s committed 13.806375297 is a release-6.0 number
and has drifted — see P24a.)

**The Born effective charges come from the same two solutions**, and the bare phonon
perturbation is not transcribed either: `dV_bare/du |psi>`, which `dvqpsi_us.f90` builds
term by term, is one `jvp` through `Calculation.at_positions` at frozen `v_scf` — the same
method the force differentiates, which already moves `vltot` and `vkb` traceably.
`zstar_eu.f90` pairs it with the self-consistent `dpsi/dE`. Silicon's `Z*` is zero by
symmetry in a converged calculation, so the benchmark's **-0.07568** is a residue — 4
against an electronic part near 4.076 — which makes it a sharper check than the dielectric
constant. *Check met:* **-0.075715** on both atoms against the vendored `ph.x`'s
**-0.07571** — every digit it prints — with the off-diagonal entries at 1e-17.

**The trap of this phase is P6's, in a second place, and it is silent.** A response is
direction-dependent, so a symmetry-reduced k-set needs its average put back
(`symdvscf.f90`) — and what is averaged is not three scalar densities but a **polar vector
field**, `drho_a <- (1/N) sum_S R_ab drho_b({S|f}^-1 r)`. That is
`symmetrize_magnetization`'s construction *without* the axial sign, which is why
`symmetrize_vector_density` now serves both and the distinction is written down in one
place. The obvious escape — run the whole k-grid, where a reduction has nothing to put
back — **only works if that grid is closed under the point group, and a shifted
Monkhorst-Pack grid is not.** Measured on this cell: **2304 of the 3072 rotation images of
a shifted 4x4x4 grid land off it** (an unshifted grid: zero of 3072). The consequences,
all of which look like a working calculation:

| shifted 4x4x4, `nosym`, nothing symmetrised | |
|---|---|
| total energy against the reduced run | **+3.1e-5 Ry** |
| density asymmetry `max|rho - sym(rho)| / max|rho|` | **2%** |
| dielectric tensor diagonal | 13.848 against 13.806 |
| off-diagonal entries, which cubic symmetry forbids | **3.77** |

The combination is now refused by name. It is the same fact `si2-nc-shifted-nosym.in`
records for P6 — a shifted grid does not have the crystal's symmetry — reached from the
other side.

**And the escape *does* work on an unshifted grid, which is the only check
`symmetrize_directional` has**, since QE computes the wedge route alone. The same
unshifted 4x4x4 sample run two ways — reduced to 8 points with the response symmetrised,
and whole at 64 points with the symmetrisation switched off — agrees to every digit
printed: `E = -15.830647095` Ry and `epsilon = 23.608844285` from both, with anisotropy
1e-14. (That `epsilon` is nowhere near 13.8 because an unshifted grid includes Gamma,
where silicon's gap is smallest and the response largest, and 4^3 points do not average it
away. It is a property of the k-sample, not of the method.)

### P24a — Ultrasoft and PAW linear response. ✅ DONE.

**Almost none of what they add is transcribed**, because the density and the Hamiltonian
are already written down once as differentiable functions of the things that move:

| QE | here |
|---|---|
| `incdrhoscf` + `addusdbec` + `lr_addusddens` | one `jvp` of the density builder w.r.t. the *states* |
| `newdq`'s `int3`, `adddvscf` | one `jvp` of `newd` w.r.t. the *potential* |
| `PAW_dpotential` | one `jvp` of `onecenter` w.r.t. `becsum` |

*Check met:* `epsilon_infinity` against the **vendored** `ph.x` — 13.806646 against
13.806689 (norm-conserving Si, 4.3e-5), 14.325321 against 14.325270 (ultrasoft Si,
5.2e-5), 14.320211 against 14.320177 (PAW Si, 3.4e-5) and 5.756059 against 5.756182
(ultrasoft C, 1.2e-4). The carbon case is the independent one — different element, cutoffs
and lattice constant — so agreeing on it is not agreeing twice on the same arithmetic. And
`chi_0 dV` against a central difference of the density is 1e-6 relative on all three
datasets.

**The committed `ph_base` benchmarks are stale and were not used.** They date from release
6.0: 13.806375297 against the vendored `ph.x`'s 13.806689470 on silicon, 5.756035041
against 5.756181864 on carbon. This is the staleness `tests/conftest.py` already documents
for `pw.x`, met for the first time in `PHonon`, and the regenerated outputs are committed
as `tests/data/qe/reference.out.ph-*`. **The first version of this phase compared against
the committed numbers and reported 2.7e-4** — six times worse than the truth, and in the
direction that hides a real disagreement rather than inventing one.

**One change here reached outside the phase, and what it found was a correction to
P22.** Writing `|psi|^2` as `Re(conj(psi) psi)` moves every density in this code by **3.5
eps** — and that was enough to flip two knife-edge regression tests. Both were asserting
path-dependent outcomes of dynamics P22 itself calls chaotic, and the fix was in the tests;
but rebuilding one of them measured something P22 had stated slightly wrong.

- **`test_newton_krylov_reaches_an_unstable_solution`** started both solvers from the
  atomic superposition, which P22's own text says is the regime where *which* root Newton
  lands on "depends on the inner-solve accuracy in no systematic way". Restructured to the
  perturbed-root protocol the DFT+U nickel case already used, the iron result is:
  **the symmetric solution is not a saddle in the linear sense — it is metastable, with a
  finite basin.** A kick of 0.05 in the atomic magnetization's shape decays back to it, and
  one of 0.20 runs away. So the demonstration lives in a window whose edges are both
  physical: below ~0.08 mixing comes back too, above ~0.12 the perturbed state is nearer
  the ferromagnet and Newton converges on *that*. Three points between behave identically.
  "A root no mixer can hold" stays exactly true of the Hubbard saddle, whose 2% kick runs
  away; for iron the honest statement is a root no mixer *reaches*.
- **`test_the_two_fixed_spin_moment_rules_find_the_same_field`** asserted
  `secant.iterations * 5 < elk.iterations`. Elk's damping time is itself chaotic at that
  resolution — 1380 iterations once, 288 another time, separated by nothing but the 3.5 eps
  — so the ratio was asserting a number that does not exist. Every *physics* assertion in it
  (same field, same energy, same moment) was unaffected and is kept.

The general lesson, which is P22's own and is now measured at eps scale: **a test that
asserts which root a stability-blind solver finds, or how long a marginally damped
controller rings for, is asserting a property of the arithmetic and not of the physics.**

**Three things did have to be written down, and each was a trap.**

- **`|psi|^2` is `Re(conj(psi) psi)`, not `abs(psi)**2`.** `abs`'s derivative is
  `Re(conj z t)/|z|`, which is `0/0` wherever the field vanishes — and a wavefunction has
  nodes *on grid points* by symmetry. It is `modulus`'s trap (P11a) in a second place, and
  it is reachable only once the density is differentiated with respect to the **states**,
  which nothing before this phase did.
- **The projector derivative in `adddvepsi_us` is the one about the atom's own centre.**
  `vkb` carries `e^{-i(k+G).tau}`, so the true `d(vkb)/dk` contains `-i tau vkb`;
  `gen_us_dj` and `gen_us_dy` differentiate the radial and angular parts and leave the
  structure factor alone. Everywhere else the distinction is invisible, because a projector
  appears as `|beta_i> D_ij <beta_j|` and the two `tau` terms cancel between ket and bra —
  which is why the velocity operator itself never had to care. Here a *single* projector
  derivative meets a state, nothing cancels, and the term is worth **2%**: 14.620 against
  14.325.
- **`dbecsum` on a wedge is a polar vector.** `Calculation.becsum` ends with
  `PAW_symmetrize`, and a *response* must not go through it: the three directions are
  averaged together (`PAW_dusymmetrize`, here the magnetization branch of
  `BecsumSymmetry.apply` with the **unsigned** rotation, exactly as
  `symmetrize_vector_density` was to `symmetrize_magnetization`). Measured on PAW silicon:
  **14.3045** scalar-symmetrised, **14.3177** not symmetrised at all, **14.3202**
  vector-symmetrised, against `ph.x`'s **14.3202**.

**`dpqq` *is* transcribed** (`compute_qdipol.f90`), and the reason is the same coordinate
singularity: it is `i d/dq [int Q(r) e^{-i q.r} dr]` at `q = 0`, and that form factor is a
radial function of `|q|` times a harmonic of `q/|q|`, which at the origin are `0` and
`infinity`. Only `L = 1` survives the angular integral, so the closed form is one radial
moment times the harmonic product the augmentation charge already uses. Checked against a
direct real-space integral of `r Q(r)` on the dense grid: **1e-3**, which is that grid's own
resolution of a sharply peaked charge.

**Born effective charges stay norm-conserving, and are refused by name for the other two.**
`zstar_eu.f90` is the whole story for a norm-conserving dataset; `zstar_eu_us.f90` adds
five stages needing the ionic displacement's own density response (`iudrhous`), the pre-`S`
position operator (`iucom`), `dvkb3`, `psidspsi` and `int1`/`int2`. Without them the
norm-conserving expression is wrong **in sign as well as size** — `+0.1625` against `ph.x`'s
`-0.07945` on ultrasoft silicon — while the dielectric constant from the *same* field
response is right to 5e-5. Two quantities out of one solve, one complete and one not, is
what a refusal is for.

**Refused rather than approximated**, each by name and each because the missing piece is
invisible in the answer: **metals** (`orthogonalize`'s smearing branch replaces
the sharp projector with occupation-difference weights, and the Fermi level itself shifts —
`ef_shift`; the level's own derivative already exists, since P22 wrote `bisect_fermi`'s
`custom_jvp`, so this is the projector's gap); **noncollinear magnetism and spin-orbit
coupling** (`incdrhoscf_nc`, `set_int3_nc`); **DFT+U** (`adddvhubscf` — the induced
potential carries a `dns` that is not a function of `drho`); and spin spirals.

*Notebook 19.*

### P24b — Born effective charges for ultrasoft pseudopotentials. ✅ DONE.

`pypresso/response/born.py`, plus two generalisations of
`forces/energy.py`: the mixed state may be handed in as a **builder** rather than
an array (`_mixed_state_part`), and the orthonormality constraint's multipliers
may be a **matrix** (`_constraint_energy`). P24a left `Z*` norm-conserving and
refused the other two by name, because `zstar_eu_us.f90` is five further stages
and without them the norm-conserving expression is wrong in sign as well as size
(+0.1625 against `ph.x`'s −0.07945). Four of those five stages are terms of one
derivative.

**`Z*` is a mixed second derivative, so it is computed as one.**
`Z*_(a)ij = dF_(a)j/dE_i = −d²E/du_(a)j dE_i`, and P25 already differentiates the
force along a tangent: one `jvp` of `jax.grad(frozen_energy)` per **field**
direction, along the electric field's response, returns a whole `3 nat` column.
Three `jvp` calls and the tensor is complete — the position tangent is zero, so
only the electronic one is switched on.

    d²E/du_j dE_i = d_E d_j L + (d_psi d_j L).dpsi_i + (d_Lambda d_j L).dLambda_i

| QE | here |
|---|---|
| `zstar_eu`'s main term | the `dpsi` half of the `jvp` |
| `iudrhous` × `dv_of_drho` (stage 1) | the `dLambda` half, screening part |
| `psidspsi` (stage 2) | the `dLambda` half, bare part |
| `add_dkmds` (stage 3b) | `jax.grad` of `frozen_polarization` |
| `add_for_charges` (stage 3a) | **transcribed** — `constraint_position_term` |

**Why this is affordable and an ultrasoft `Gamma` phonon is still not.** P25's
identity leaves over `−<psi|dS/du_j|psi>·dLambda_i`, which vanishes when `S` does
not move with the atoms. For a *phonon* both legs move `S` and `dLambda` is a
response that has to be solved for; for a Born charge only the `u` leg does, and
the `E` leg's `dLambda_mn = w_n <psi_m|dV_E|psi_n>` is a matrix element of the
same perturbation the Sternheimer solve was already driven by. Nothing new is
computed for it — the perturbation is contracted differently.

**Four things had to be supplied to the tangent, and three of them were traps.**

- **The mixed state has to stay a *function* of where the atoms are.** P25 could
  hand `density` in as a constant array, because a norm-conserving density does
  not move at frozen states. An ultrasoft one does — `Q_ij(r − tau)` is part of
  it — and freezing it deletes exactly the dependence the second derivative is
  made of. So `energy_at` now takes a *builder* for both the density and
  `becsum`; the builder rebuilds the raw, unsymmetrised quantity at the moved
  calculation and adds a constant offset, so the value is the converged
  symmetrised one and the tangent is the wedge's own. That is `zstar_eu`'s
  convention and `symtensor` completes it at the end. Leaving the SCF's *scalar*
  symmetrisation in the chain rule gives **−3.96** where the answer is −0.0757,
  so it is not subtle — but it is invisible in a force, which is why nothing
  before this needed the hook.
- **The multipliers are a matrix and the index order is not a convention.**
  Stationarity gives `Lambda_mn = w_n <psi_m|H|psi_n>` — diagonal at the ground
  state, and *not* diagonal to first order, so a diagonal-only tangent (which
  `FrozenState.eigenvalues` already admits) drops the off-diagonal block, which
  is `psidspsi`. `Lambda_mn` pairs with `<psi_n|S|psi_m>`, so the weight belongs
  to the **column**; transposing it costs 0.28 on ultrasoft silicon and *nothing*
  on a norm-conserving one, where the term is zero either way. That is the class
  of error the norm-conserving gate below cannot catch, and it was made.
- **The frozen polarization's operator is `adddvepsi_us`', not the moment of the
  augmentation charge.** At frozen coefficients the smooth charge does not move
  at all, so the only electronic term that survives is the augmentation's, and
  writing it as the obvious `tau_a q_ij + dpqq^a_ij` is **wrong by 0.38**. The
  right operator is the one the position operator already carries,
  `i q_ij <d(beta_j)/dk_a| + dpqq^a_ij <beta_j|`: the own-centre derivative
  deliberately excludes the structure factor's `−i tau` (P24a's 2% trap), so
  `i q <dbeta/dk|` is not `tau q <beta|` and the difference — the projector's
  internal `k` dependence — is a real part of the position operator of a periodic
  crystal. `jax.grad` of `−sum w <psi|A_a|psi>` is the whole of `add_dkmds`,
  three hundred lines of Fortran, and the projectors' motion is written as a
  phase rather than rebuilt, because `vkb` and `d(vkb)/dk` about the atom's own
  centre carry the *same* structure factor.
- **One term is transcribed, and the reason is a coordinate singularity** — the
  same exception `dpqq` already is. `dLambda` wants the occupied-occupied block
  of the position operator, and `<psi_m|r|psi_n>` is the Berry connection: not a
  matrix element at all in a periodic cell. It is finite only in the combination
  it enters, contracted with `<psi_n|dS/du|psi_m>`, because `dS/du` is localised
  on one atom. `add_for_charges.f90` is that combination and
  `constraint_position_term` is it, worth **0.55** on ultrasoft silicon — the
  difference between +0.47 and −0.079.

*Check met.* Against the **vendored** `ph.x`, regenerated:

| case | here | `ph.x` | difference |
|---|---|---|---|
| norm-conserving Si | −0.0757150 | −0.07571 | every digit |
| ultrasoft Si | −0.0794417 | −0.07945 | **8.3e-6** |
| ultrasoft C | +0.0415594 | +0.04179 | 2.3e-4 |

Carbon is the independent case — different element, cutoffs and lattice
constant, and the **opposite sign** — and its 2.3e-4 is where its dielectric
constant already is (1.2e-4 against silicon's 4.3e-5, the radial form factors'
interpolation floor), arriving amplified because `Z*` is the residue of 4 against
3.958. And the norm-conserving number agrees with the transcribed `zstar_eu.f90`
beside it to **1.3e-14**, which is the phase's regression gate: every term added
here has to switch itself off when `S = 1`, and that equality is what says it
does.

**PAW is refused by name**, and the gap is one term rather than a method.
Everything above reaches **1.3e-3** on it — −0.078293 against `ph.x`'s −0.07961 —
and what is left is QE's fifth stage, `int3_paw` against `becsumort`: the
one-centre twin of `add_for_charges`, pairing the field's response of the
one-centre coefficients (which `paw_response` already produces) with the
displacement's orthogonality `becsum`. It has no counterpart in the plane-wave
part because `<psi|S|psi> = 1` carries the whole of `becsum`'s share of the
energy for an ultrasoft dataset and not for a PAW one, whose one-centre energy is
a second, independent function of it. Its factor was not settled from the
Fortran: `compute_drhous` builds its `dbecsum` without the one-half the
orthogonality correction carries, and `addusdbec` accumulates one of the two
cross terms rather than both, so the coefficient is a product of two conventions.
Fitting it to −0.07961 would make the number a measurement of `ph.x`. 1.3e-3 is
sixteen times the last digit it prints, so it is refused; the dielectric constant
from the same run is right to 3.4e-5 and is not.

*Notebook 19.*

### P24c — Metals in linear response. ✅ DONE (the solve and `ef_shift`).

`pypresso/response/sternheimer.py` — `Smearing`, `SternheimerSolver._smeared_projection`,
`local_density_of_states`, `fermi_level_shift`, `fermi_level_shift_states` — plus two new
test inputs, `tests/data/qe/al-metal.in` (QE's own `pw_metal/metal.in`, copied so the tests
run without the vendored tree) and `al2-metal.in` with its regenerated `ph.x` reference.
P24 refused a metal by name: the
insulator projector applied to one is silently wrong, and the Fermi level moves.
Both are here now.

**What a metal changes is the projector, not the solve.** For an insulator
`P_c^+` is a projector — a band is occupied or it is not. For a metal there is
no such partition, and `orthogonalize.f90`'s smearing branch replaces the sharp
step with a pair of *weights* (Rev. Mod. Phys. 73, 515, Eq. 75):

    dvpsi_i  <-  wg1_i dvpsi_i
    ps_(j,i) <-  wwg_(j,i) <psi_j|dvpsi_i>
    wwg_(j,i) = wg1_i (1 - t) + wg1_j t + alpha_pv t (wg1_j - wg1_i)/(eps_j - eps_i)

with `t` a *second* smeared step, this one of the energy difference. Three
consequences run through the module and each is a place the answer could be
silently wrong:

- **Every band stays in the block**, because `nbnd_eff = nbnd` there. QE
  truncates the solve at `setup_nbnd_occ`'s count; here the block is whole at a
  static shape (rule R2) and the bands past that point carry an occupation of
  zero. `nbnd_occ` survives as a **mask** — it is still needed, because
  `orthogonalize`'s `alpha_pv` correction and `ch_psi_all`'s level shift both
  admit only `j` inside it.
- **`alpha_pv` is measured to where the smearing dies**, `ef + xmax degauss`,
  since the occupied manifold has no top (`setup_alpha_pv`).
- **The density response is accumulated with `wk`, not `wg`.** The occupation is
  applied to `dpsi` itself, so weighting the density by it again counts it twice.
  The two coincide for an insulator — a filled band has `wg = wk` — which is
  exactly why nothing before this had to tell them apart, and why the mistake
  would have passed every case in the suite.

The `0/0` at a degeneracy is taken to its limit `-alpha_pv t w0gauss_i` rather
than guarded, which is rule D4's requirement and not a nicety: a crystal is
degenerate everywhere by symmetry. It is written with `jnp.where` on a *safe*
denominator, so the unused branch never makes a NaN the gradient would carry.

**`localdos` is the density builder with a different weight.** `ldos(r) =
sum_kn w_k delta(ef - eps_kn) |psi_kn|^2` is `density_at` called once more with
the smeared delta in place of the smeared step, so `localdos.f90` — a hundred
lines, `becsum1` included — is not transcribed. And `ef_shift` stands on it: a
perturbation at `q = 0` is not orthogonal to the identity, so it changes the
number of states below the Fermi level, the level moves by
`def = -(integral of drho)/N(ef)`, and the response fills the Fermi surface,
`drho <- drho + def ldos`.

*Check met, twice, against things that share no machinery with the solve.*

| what | reference | agreement |
|---|---|---|
| `chi_0 dV` on fcc aluminium | a central difference of the density, **re-occupied at the same `ef`** | **2.5e-7** relative at the difference's own optimal step; 1.4e-6 at `1e-3` and 1.3e-5 at `3e-3`, so it is `h^2` and the floor is the reference's |
| `ldos` | its own integral against `dos_ef` | 1e-10 |
| `drho` after `ef_shift` | **zero**, on two probes with shifts of opposite sign (-0.036 and +0.009 Ry) | 1e-15, where the uncorrected response moves 0.21 and -0.053 electrons |
| `ef_shift_wfc` | `ef_shift` — the corrected *states* rebuilt into a density must give back `def ldos` | 1e-10 relative, which is what pins its factor of one half |

The finite difference re-occupies at the **same** Fermi level rather than
re-converging it, and that is not a shortcut — it is what the quantity is. The
Sternheimer response of a metal is the response at fixed `ef`; the level's own
motion is `ef_shift` and is a separate correction, so testing the two together
would test neither.

**The dynamical matrix of a metal is refused, and the reason is a weight
convention rather than a missing routine.** In the metal branch the occupation
lives *inside* `dpsi`; the second derivative here is a `jvp` of
`jax.grad(frozen_energy)`, and that functional weights the states by `wg = wk f`,
so the same tangent enters the energy carrying `f` twice. Dividing it back out
band by band is not the fix — it diverges at the Fermi surface, and the metal
response is not a plain wavefunction response in the first place: the
`(f_i - f_j)` structure is inside it, which is why QE's `drhodv` has its own
`wgg`-weighted contraction instead of reusing the insulator assembly. The
occupations' own first-order change has to enter as `df_n` against `d(eps_n)/du`
and the entropy's derivative, not folded into `dpsi` — `ef_shift_wfc` reproduces
it correctly in the *density* (tested: `response_density` of the corrected states
equals `def ldos` to 1e-10 relative) and cannot in the energy.

**The Hartree and exchange-correlation half is already right**, which is why the
answer is wrong by half the spectrum rather than nonsense: `drho` is handed to
the assembly as a separate tangent, built with the right weights. The double
count sits only in the direct contractions against the states — kinetic,
nonlocal, constraint. The way in is a split assembly: the frozen Hessian at `wg`,
the electronic response with the metal's own weights, which is de Gironcoli's
Eq. (B19) structure.

*Measured with the guard lifted*, on the two-atom aluminium of `al2-metal.in`
against the vendored `ph.x` on the *same* input — whose ground state this
reproduces to 1e-9 Ry (−8.332103799 against −8.33210381):

| mode | here | `ph.x` |
|---|---|---|
| acoustic | 155.74, 155.74, 155.74 | 1.1, 1.8, **1.9** |
| optical (the doubling's folded pair, and the zone-centre mode) | 197.96, 197.96, 309.26 | 146.7, 146.7, 311.0 |

in cm⁻¹, from a run that converges to `|ddv_scf|^2 = 8.7e-17` and returns a
matrix symmetric to 4.3e-8. Nothing in the numbers says it is wrong; the
reference and the sum rule do. `reference.out.ph-al2-metal` is committed, so the
phase that lifts this refusal has its target already. **P28 lifted it**, and the
split assembly guessed at here is what it turned out to be — except that the
`df_n` term this paragraph predicts is not needed, being already inside `dpsi`. (`ph.x` will not run this
cell with the symmetry on — "FFT grid incompatible with symmetry" out of
`phq_setup` — so the input is `nosym` on an **unshifted** grid, which is the
combination a response can be computed on without symmetrising it anyway.)

**Refused rather than approximated.** The **tetrahedron** occupations, whose
`orthogonalize` branch reads `dfpt_tetra_beta` — a response weight per band
*pair*, which the tetrahedron machinery here does not build; the smearing family
is what is implemented, and the refusal says so by name. And
`occupations='from_input'`, whose occupations are not a differentiable function
of a Fermi level at all. **`epsilon_infinity` and the Born charges stay refused
for a metal**, not because the solve cannot do it but because the quantities do
not exist there — `pw.x` refuses `epsil` for a metal for the same reason. That
distinction is now one flag on `require_a_sternheimer_regime` rather than three
separate refusals.

**No README row of its own, and that was the honest answer rather than an
oversight.** The table's rows are quantities someone would want to compute, and
this phase produced none that was new: `epsilon_infinity` and `Z*` do not exist
for a metal, and the dynamical matrix that would was refused above. What P24c is
is the *layer under* a row — the metallic response every one of those quantities
stands on. **P28 gave it that row**: the second derivative works now, and the
phonon row covers metals.

### P25 — Phonons at `Gamma`: the dynamical matrix. ✅ DONE.

`pypresso/response/phonon.py`, plus `Calculation.symmetrize_atom_displacement`,
`symmetrize_atom_displacement_density` and `symmetrize_atom_pair_tensor` in
`system/symmetry.py`, and one new argument on `forces/energy.py`'s functional. It is
what P24 was built toward — the electric field's response gave `epsilon_infinity` and
the Born charges, and the *ionic* perturbation gives the force constants — and it ends on
the frequencies of the same regenerated `ph.x` run.

**The second derivative is one `jvp` of the gradient the force already is.** P15 wrote
the total energy as a function `L(u, psi)` of the coordinate and the state, carrying the
orthonormality constraint with its multipliers so that `L` is *stationary* in `psi` at
the solution — which is why the force is `grad_u L` at frozen wavefunctions rather than a
total derivative. Differentiate once more and

    d^2E/du_i du_j = d_i d_j L + (d_psi d_j L) . dpsi_i

with no second-order wavefunction, no `<dpsi|H - eps S|dpsi>` term and no factor to get
right. Both pieces are components of *one* tangent vector, so **one `jvp` per mode
returns a whole column of the matrix**. `dynmat0`, `d2ionq`, the local potential's and
the projectors' second derivatives are the `u` half; `drhodv` is the `psi` half; and
neither is written down. What is transcribed is `solve_linter`'s loop, `symdvscf`,
`symdynph_gq` and `dyndia`. The bare perturbation `dvqpsi_us` was already there: it is the
`jvp` through `at_positions` that P24 built for `Z*`.

*Check met.* Silicon's optical mode on the ten-point wedge of `si-epsilon.in`:

    510.102374 cm^-1   against   ph.x's   510.151844

a difference of **0.049 cm^-1**, 9.7e-5 relative — the same floor as the dielectric
constant's 4.3e-5 and the same cause, QE's `dq = 0.01` form-factor table against direct
integration here. The mode is triply degenerate to 1e-6 with nothing imposing the crystal
class, and the on-site block is isotropic to 1e-9.

**Three further checks, each sharing no machinery with the assembly.**

- **A rigid translation of the crystal is a translation of its density.** Sum the response
  densities of all the atoms along one direction and the answer must be `-d(rho)/dx`,
  which is got by differentiating the converged density in G-space. Measured: **6.5e-5
  relative**, on all three axes. It holds only for the *screened* response — the bare one
  is **52%** off — so it tests the linear solve, the kernel and the symmetrisation at once.
- **Finite-differenced forces.** Displace an atom by `+-h`, re-converge, difference the
  symmetrised forces, and compare whole columns. This is the only check that reaches the
  *response* half of the derivative: `jacfwd` of the force with respect to the positions
  alone is the frozen Hessian and is checking itself. Measured on the unshifted `nosym`
  grid: **2.55e-5** Ry/bohr^2 at `h = 1e-2` and **2.14e-5** at `3e-3`, against force
  constants of 0.2865 — improving with the step, so what is left is the floor rather than
  truncation.
- **The wedge against the whole closed grid.** The same unshifted 4x4x4 sample, once
  reduced to 8 points with the response symmetrised and once whole at 64 with the
  symmetrisation idle: **2.7e-14** on the matrix and 1.5e-9 cm^-1 on the frequencies. It
  is the only check the two new symmetrisations have, exactly as it was the only one
  `symmetrize_directional` had in P24, and diamond is the right cell for it because half
  the operations exchange the two sublattices.

**The acoustic modes are the diagnostic and not a target.** Translating the crystal costs
nothing, so three frequencies are zero exactly and what comes out instead is the finite
basis's own error — the energy depends slightly on where the atoms sit relative to a grid
that does not follow them. `ph.x` prints **2.045258** and this code prints **4.088**;
both are 1e-4 of the force constants (`D_00 + D_01 = 3.6e-5` against `D_00 = 0.2766`) and
neither is physics. QE does **not** impose the sum rule in `ph.x`, so `acoustic_sum_rule`
defaults to `False` here and the residue stays visible; switching it on gives 3e-6 cm^-1
and moves the optical mode by 0.016.

**The trap of the phase was the symmetrisation, and it cost the whole answer.**
`frozen_energy` builds its density with the SCF's own **scalar** symmetrisation, which is
right for the ground state — it is how a wedge sum is completed to the whole Brillouin
zone, and the functional has to be the one the SCF minimised. It is wrong for a
*response*: displacing one atom breaks the crystal's symmetry, and averaging that
perturbation over the full group of the *undisplaced* crystal projects most of it away. A
second derivative differentiates the functional with respect to the **states**, so the
chain rule pushes the state tangent straight through that average. What that produced was
not an obviously broken number:

| | |
|---|---|
| optical mode, chain rule through the scalar average | **667.0 cm^-1** against 510.2 |
| acoustic sum rule `D_00 + D_01` | **-0.716 Ry/bohr^2**, i.e. 580 cm^-1 where the answer is 2 |
| the matrix's symmetry and cubic form | perfect, both before and after |

The fix is that the density becomes an **independent argument** of `energy_at`, so the
caller supplies both it and its tangent: the ground state's symmetrised density, and the
response density already averaged the way `symdvscf` averages one. It is
`SternheimerSolver.density_at`'s rule ("`Calculation.density` **without** the
symmetrisation, because the caller symmetrises it as a vector") one level up, met for the
first time in a *second* derivative. The confirming control was free and is what located
it: the `nosym` run, which symmetrises nothing at all, satisfied the sum rule to 4e-5
throughout.

**Two smaller traps.** `frozen_energy` still wrote `jnp.abs(psi)**2` in its kinetic and
constraint terms, which was harmless while only the positions were differentiated and is
`0/0` the moment the states are — P24a's trap in a **third** place, and this time the
symptom is a NaN in every force constant rather than a wrong number. And the asymmetry
`max|D - D^T|` is worth exposing but only *after* the group average: before it, a column
is a wedge sum and the raw asymmetry is **5.1e-2** against force constants of 0.28, which
says nothing; after it, it is 2e-16 on every case here and reports the linear solves.

**Refused by name: ultrasoft and PAW, and the gap is in the formula rather than in a
missing routine.** The identity holds because `L` is stationary in `psi` at *fixed*
multipliers, and those multipliers sit on the constraint `<psi|S(u)|psi> - 1`.
Differentiating twice leaves a term `-<psi|dS/du_j|psi> deps_i` which vanishes identically
when `S` does not move with the atoms — a norm-conserving dataset — and does not otherwise;
beside it, the augmentation charge `Q_ij(r - tau)` moves at frozen `becsum`, which
`addusdynmat` and `drhodvus` account for. With the guard lifted the measurement is
`zstar_eu_us`'s shape, **wrong in sign as well as size**:

| | ours | `ph.x` |
|---|---|---|
| ultrasoft Si, optical | **-504.32** | +513.28 |
| PAW Si, optical | **-503.63** | +513.40 |
| ultrasoft Si, acoustic residue | 618.4 | 6.13 |

— an imaginary frequency where the crystal is stable, from a run that converges in 17
iterations and gives a matrix that is cubic and symmetric to 1e-16. The **dielectric
constant** from the same solver is right for both datasets to 5e-5; two quantities out of
one machinery, one complete and one not, is what a refusal is for. Everything
`require_a_sternheimer_regime` refuses is refused here too — metals, noncollinear
magnetism, DFT+U, spirals — and so is a `nosym` run on a *shifted* grid, through the same
`require_a_symmetrisable_response` the electric field uses.

**One more refusal, and it is a gap in P24 rather than in this phase.** `nspin = 2` is
refused, because the occupied-band count in `response/` is a *single* number
(`nelec / 2`) applied to both spin channels. That is right for an unpolarized insulator
and wrong for a magnetic one, whose channels are filled to different depths — the response
would be solved for the wrong bands in one of them, with no shape error and no failure to
converge to show for it. The same arithmetic is in `dielectric_tensor`, and since 2d7d9d5
the refusal lives in `require_a_sternheimer_regime` so that every entry point — the field,
the displacement, the strain, the third derivative — inherits it rather than restating it.
Making `nocc` per-channel is one change in `SternheimerSolver` and would serve all of them;
what it needs is a magnetic insulator to validate against.

**Memory.** `3 nat` bare perturbations and `3 nat` first-order wavefunctions are held at
once, each `(nspin, nk, nocc, npwx)` complex: **2 MB** on this silicon, and **7 GB** on a
16-atom cell with 100 k-points and 3000 plane waves. The bare terms are stored rather than
recomputed because the loop re-uses them every iteration, which is
`response/efield.py`'s trade for its three. The way down is QE's and is the backlog item:
solve one irreducible representation at a time, which cuts the count as well as the
storage.

**One finding that is about the cost rather than the answer**, and it is a rule this
project already wrote down once. `ph.x` does the same six modes in about 2.2 s where this
takes 57, and the reference output says why: `dfpt_kernels.f90` schedules the linear
solve's threshold against the self-consistency of the response
(`thresh = min(0.1 sqrt(dr2), 1e-2)`) while `response/phonon.py` holds a fixed 1e-12, which
costs `av.it. = 27.7` against 9.3; and `LR_Modules/mix_pot.f90` is a modified Broyden over
four iterations, which is why 5 iterations do what 17 of linear mixing do here. It is
`electrons.f90`'s `ethr` schedule met a second time, on the stage that is 96% of the run.
What the irreducible representations buy is *not* fewer solves — `ph.x` perturbs along all
six modes too — but a bounded working set, since each representation is converged and
released on its own. All three are in `PERFORMANCE.md`'s backlog.

**What is left for phonons proper** is `q != 0`: the up and down components of a
displacement pattern live at `k` and `k + q`, so it needs two plane-wave spheres per
k-point, which is machinery P19 already built for the spin spirals and which nothing here
reuses yet. With that and `q2r`/`matdyn`'s Fourier interpolation there is a dispersion;
without it there is one point of it.

*Notebook 20.*

### P26 — Electrostriction: `d(chi)/d(strain)` as a mixed third derivative. ✅ DONE.

`pypresso/response/strain.py`, `elastic.py` and `electrostriction.py`, plus
`Calculation.symmetrize_strain_response` and `symmetrize_tensor_density` in
`system/symmetry.py`, a `kcart` argument on `VelocityOperator`, and a `keep_internals`
flag on `dielectric_tensor`. It is the first quantity in this project that is a **third**
derivative of the energy, and the first perturbation whose label is a rank-2 tensor.

**Why a derivative of `chi` and not a field.** Electrostriction is the quadratic
electromechanical coupling every dielectric has, and Tanner, Bousquet and Janolin
([arXiv:2012.03841](https://arxiv.org/abs/2012.03841), Eqs. 2) is the current method: by a
thermodynamic identity the four tensors are derivatives of the dielectric susceptibility
with respect to a mechanical variable,

    eps0 d(chi_ij)/d(X_kl) =  2 M_ijkl        (1/eps0) d(eta_ij)/d(X_kl) = -2 Q_ijkl
    eps0 d(chi_ij)/d(x_kl) = -2 m_ijkl        (1/eps0) d(eta_ij)/d(x_kl) =  2 q_ijkl

with `eta = chi^-1`. The alternative — optimising under a finite `E` or `D` field and
fitting a parabola — puts a band-gap-dependent ceiling on the k-point density, entangles
electrostriction with non-linear piezoelectricity in a non-centrosymmetric crystal, and
needs a constrained relaxation. **The literature was consulted before the design and not
during the debugging**, which is the standing rule this phase was asked to follow.

**The whole of it is the 2n+1 theorem in one sentence: the second-order energy is
stationary in the first-order wavefunctions, so it may be differentiated with them held
fixed.** That is P15's envelope argument at the next order and P25's sentence about the
force one order up. Write

    F_ij[x; psi, rho, b, u] = sum_kn w [ <u_i|H(x)|u_j> - Lambda_mn <u_i,n|u_j,m>
                                        + 2 Re <u_i|P_c b_j> ] + (1/2) int drho_i K(x) drho_j

whose stationary point in `u` is the self-consistent Sternheimer solution P24 already
computes and whose stationary *value* is `sum w Re <b_i|u_j>` — the expression
`dielec.f90` assembles. Then `d(eps)/dx` is **one `jvp`** along
`(e_x, dpsi/dx, drho/dx, db/dx)` with no tangent for `u` at all.

**Three tangents, and none of them is transcribed.** `dpsi/dx` and `drho/dx` are a strain
perturbation, which in Abinit is a phase of its own — the metric-tensor formulation of
Hamann, Wu and Vanderbilt ([cond-mat/0409269](https://arxiv.org/abs/cond-mat/0409269),
[cond-mat/0501548](https://arxiv.org/abs/cond-mat/0501548)), whose difficulty is that a
strain moves the plane-wave basis so the nonlocal projectors' strain derivative has to be
derived by hand in reduced coordinates. `Calculation.at_strain` is already written in
exactly those coordinates — what is stored is a set of **Miller indices** and the sphere is
frozen while differentiating (P11) — so the bare perturbation is one `jvp` through it, the
way `dvqpsi_us` became one `jvp` through `at_positions`. `db/dx` is one further Sternheimer
solve, because `b = P_c r|psi>` is the one argument with no closed form: it is *defined* by
a linear equation, so its equation is what gets differentiated.

**The trap of the phase, and it is worth 2%.** `u` is frozen; the space it is constrained
to live in is not. The Sternheimer solution is orthogonal to the occupied manifold and that
manifold moves with the strain, so the variable of the functional is `P_c(psi) u` and never
the stored array. Writing `u` changes **no value** — `P_c u = u` where everything is
evaluated — and destroys the stationarity the construction rests on: an unrestricted
variation gives `A u + P_c b + (K drho) psi = 0` where the loop solves the same equation
with the screening term projected too, and the difference is the occupied component of
`(K drho) psi`. It survived the value identity against `dielec.f90`, the cubic form of the
rank-4 tensor, *and* a finite-difference check of each of the four tangents separately.
What found it was splitting the disagreement in two — the `jvp` against a difference of
`F` at frozen `u`, and that against the true `epsilon` — where the first pair agreed to
1e-4 and the second did not.

**Two more traps, both of them index orders that are right in value and wrong in
derivative.**

- *The multiplier is a matrix.* `Lambda_mn = <psi_m|H|psi_n>` in place of `eps_n`, and
  contracted as `sum_mn Lambda_mn <u_i,n|u_j,m>` — which is `Tr(Lambda Ov)`, invariant
  under the unitary mixing a degenerate multiplet is defined only up to. The transposed
  form `Lambda_mn <u_i,m|u_j,n>` is `Tr(Lambda Ov^T)`, gives the identical number whenever
  `Lambda` is diagonal, and leaves **11%** of the scale in components cubic symmetry
  forbids. The same matrix form is what makes `d(eps_n)/dx` not be an input at all, and
  what makes the `G = 0` deformation-potential ambiguity cancel: a constant added to `dH`
  enters `<u|dH|u>` and `dLambda <u|u>` with opposite signs.
- *The SCF's wavefunctions are eigenvectors of the previous iteration's Hamiltonian.*
  `<psi_m|H[rho_out]|psi_n>` is diagonal only to 1.6e-7 Ry on silicon at
  `conv_thr = 1e-12`, and the quantity that multiplies that error is `<u|u> ~ 10^3`, so the
  variational identity fails by 7e-7 *relative* — systematically, and without shrinking
  when the response's own thresholds are tightened. `refined_states` re-diagonalises at the
  converged density and takes it to 3.5e-15.

**A wedge is refused.** The object being differentiated carries a field label *and* a
strain label, so completing a symmetry-reduced sum needs a rank-3 average
(`R_ai R_bk R_cl`) that is not written; P24 wrote the rank-1 case and P25 the
rank-1-plus-atom case. An **unshifted** Monkhorst-Pack grid is closed under the point group
and needs no average at all, which is the route P24 already uses as its independent check,
so that is what P26 requires. The rank-2 symmetriser *is* written and is what
`strain_response` uses on its own — `symmetrize_tensor_density`, checked by running the
same k-sample reduced and whole.

*Checks met.* Six, each against something sharing no machinery with it:

| what | reference | agreement |
|---|---|---|
| `drho/dx`, `deps/dx` | central difference of a re-converged SCF | 1.6e-5 relative at the step's optimum, falling as `h^2` above it |
| `F_ij` at the stationary point | `dielec.f90`'s own assembly | 9e-10 relative |
| `d(chi)/dx` | cubic symmetry, imposed nowhere | 3e-14 of the scale in forbidden components |
| `d(eps)/dx`, all three independent components | central difference of `epsilon` over re-converged strained cells | **2e-4 relative** |
| `C_ijkl` | a five-point second difference of the SCF energy at the same frozen sphere | **209.38 against 209.38 GPa** |
| the unit chain `m, q -> M, Q` | Tanner et al.'s MgO table | reproduces their `M_11`, `M_12`, `Q_11`, `Q_12` |

**The elasto-optic tensor is the same object, and it is the one a laboratory has
measured.** `d(eps)/dx` inverted twice is the photoelastic tensor,
`p = -eps^-1 (d eps/dx) eps^-1`, which costs nothing and gives the phase a reference
outside the DFPT literature entirely: silicon's `p_11 = -0.094`, `p_12 = +0.017`,
`p_44 = -0.051` (Biegelsen, [PRL 32, 1196 (1974)](https://doi.org/10.1103/PhysRevLett.32.1196)).
The symmetry story decides which of the three is a fair comparison and it is the *same*
story the elastic constants have: in the diamond structure no internal displacement is
compatible with a tetragonal strain, so `p_11` and `p_12` are complete at clamped ion where
`p_44` is not.

**Clamped-ion, and the elastic constants came with it.** What is computed is the
electronic susceptibility's strain derivative, so `m` and `q` are clamped-ion. `M` and `Q`
— the coefficients giving a *strain*, which is what experiment quotes — need the elastic
compliance, and with `dpsi/dx` in hand that is one more `jvp` of the stress along the same
tangent (`response/elastic.py`): P25's construction with the cell in place of the atoms,
and a capability of its own that `pw.x` has no counterpart for.

They are *clamped-ion*, so `C_44` is expected above the measured 79.6 GPa while `C_11` and
`C_12` are complete by symmetry. `M` and `Q` inherit that through the compliance; `m`, `q`
and the elasto-optic tensor do not. On two-atom silicon at `ecutwfc = 12` this gives
**`C_11` = 209.38 GPa** against a five-point second difference of the energy's **209.38** —
five significant figures, at the same frozen sphere — with a measured 166.

**And getting there cost two bugs, both of them silent and one of them not in this phase.**

- *The density cannot be an argument here.* `_force_constants` hands the density in as an
  independent argument, because the functional symmetrises its own as a *scalar* — right for
  a ground state, wrong for a response. Doing the same for a strain makes `jax.grad` of the
  functional a partial derivative at fixed `rho` rather than the **stress**, and the two
  differ by `(dE/drho).(drho/dx)|_psi`, which vanishes for a displacement (moving an atom
  does not change `sum w |psi|^2` at frozen coefficients) and does not for a strain (the
  density carries a `1/Omega`). Worth a factor of **three**: 671 GPa against 209. The fix is
  to let the functional build its own density, which puts the term back through the chain
  rule — and the price is that a symmetry-reduced k-set is then refused here too.
- *`run_scf` was reading the wrong cell.* The SCF loop's energy assembly, its convergence
  measure and its Kerker preconditioner all read the `system` **argument**'s cell rather
  than `calculation.system`'s. They are the same object for every ordinary call and not for
  one that supplies its own `calculation` — which is exactly what a run on a deformed cell
  does. The reported total energy then acquires a slope in the strain of **3.9 Ry per unit
  strain** against a true `dE/d(eps)` of 0.09, while the density, the potential and every
  response stay correct: only the number printed at the end is wrong, which is why nothing
  had caught it. It is a `vc-relax` waiting to happen, and it is fixed.

*Convergence, and it is two different stories.* Two-atom silicon on the whole unshifted
2x2x2 grid, `nosym`, at four cutoffs:

| `ecutwfc` | npw | `eps` | `dchi_11` | `dchi_12` | `dchi_44` | `C_11` | `C_12` | `C_44` | `B` |
|---|---|---|---|---|---|---|---|---|---|
| 12 | 190 | 56.29 | 108.11 | 8.10 | 197.02 | 209.4 | 68.0 | 134.0 | 115.1 |
| 18 | 344 | 58.70 | 119.94 | 30.43 | 202.45 | 205.2 | 69.1 | 132.2 | 114.5 |
| 30 | 754 | 58.66 | 117.63 | 31.65 | 197.82 | 198.6 | 68.7 | 129.3 | 112.0 |
| 45 | 1363 | 58.89 | 118.28 | 33.11 | 198.72 | 198.5 | 68.9 | 129.2 | 112.1 |
| measured | | 13.8 | | | | 165.7 | 63.9 | 79.6 (relaxed) | 97.9 |

The **elastic constants are converged in the basis by `ecutwfc = 30`** and are within 5% of
their converged values already at 12 — 30 and 45 agree to 0.1%. So the ~20% that separates
`C_11` from the measured value is not the basis: it is LDA at the *experimental* lattice
constant (where LDA silicon is slightly compressed, hence stiff), the pseudopotential, and
for `C_44` the missing internal relaxation. `C_12` is 68-69 against 63.9 throughout.

The **susceptibility derivative is not converged at 12** and is by 18-30: `dchi_11` moves
11% from 12 to 18 and 2% after, and `dchi_12` — a small number and a difference of larger
ones — moves by a factor of four and then settles near 32. `eps` itself is 58.7 rather than
silicon's 13.8, and that is the **k-sample**, not the cutoff: this grid is the smallest one
closed under the point group, and the closed-grid requirement above is what forces it.
**The elasto-optic `p_12` comes out negative here where experiment has +0.017**, and that
does not move with the cutoff either; a converged unshifted grid is what it would take to
settle, and it is not claimed. `p_11` has the right sign and is within a factor of three.

The **machinery numbers in the checks table are independent of all of this**, because both
sides of every comparison use the same basis and the same k-set.

**What is not here.** The *relaxed-ion* coefficients add the lattice's contribution to
`chi` — `chi_ion ~ (1/Omega) sum_m (Z* e_m)^2/omega_m^2` — and its strain derivative needs
`dZ*/dx` and `d(omega^2)/dx`, two more third derivatives of the same family, each again a
`jvp` of an assembly that exists (P24's `Z*`, P25's force constants) along the strain
tangent this phase already builds. The internal-strain tensor `Lambda_ij,ak` that turns the
clamped-ion elastic constants into relaxed-ion ones is the same kind of object. Ultrasoft
and PAW are refused: `Q_ij(r)` is a function of the cell, so `dbecsum` acquires a strain
term of its own beside the one the `jvp` gives. Everything
`require_a_sternheimer_regime` refuses — metals, noncollinear magnetism, DFT+U, spirals —
is refused here too, and so is `nspin = 2`, for P25's reason.

*Notebook 21.*

### P27 — Van der Waals dispersion: Grimme's D2. ✅ DONE.

`pypresso/vdw/` (`grimme.py`, `registry.py`, `analytic.py`), plus
`pypresso/system/elements.py`, `Calculation.dispersion_sum`/`dispersion`, and one
row each in `forces/analytic.py`, `stress/analytic.py` and `io/qeref.py`.

**Why it is needed and why it is cheap.** A semilocal functional has no London
attraction at all: its correlation energy is a functional of the density *where
the orbitals are*, and the dispersion force comes from the correlated fluctuations
of two densities that do not overlap. Grimme's D2
([J. Comp. Chem. 27, 1787 (2006)](https://doi.org/10.1002/jcc.20495), as QE
implements it after [Barone et al.](https://doi.org/10.1002/jcc.21112)) adds it
back as a pair potential over the nuclei,

    E_disp = -(s6/2) sum_{a b R} C6_ab / d^6 f_damp(d)
    f_damp(d) = 1/(1 + exp(-beta (d/(R_a + R_b) - 1)))     beta = 20

with `C6_ab = sqrt(C6_a C6_b)` from a table indexed by **Z**, up to 86. It is a
function of the nuclei and of nothing else, so it never enters `v_of_rho`: QE adds
`elondon` to `etot` after the SCF loop, and `force_london`/`stres_london` at the
end of `forces`/`stress`.

**This is the Ewald sum's twin and is written as one.** `EwaldSum` and `GrimmeD2`
are the same object: a pair sum whose neighbour list is fixed on the host once, so
that the energy is a pure JAX function of the positions and the force is
`jax.grad` of it; and whose list is a set of *lattice translations*, so a strain
deforms it and the stress is `jax.grad` in the other coordinate. Nothing about D2
needed a new idea — which is the point, and is why the two hand-derived Fortran
routines are here as a **test** (`vdw/analytic.py`) rather than as the
implementation. They agree with the autodiff route to round-off — 4e-17
Ry/bohr on the force and 3e-19 Ry/bohr³ on the stress, on a geometry with the
symmetry broken so that neither is a comparison of two zeros — and with `pw.x`'s
own printed blocks to 3.9e-9 Ry/bohr and 1.2e-8 Ry/bohr³, which is `pw.x`'s print
resolution rather than the agreement.

**Four traps, and the first is the one the phase is really about.**

- *The correction must not reach the density, and "agrees to `conv_thr`" is not
  the test for that.* A pair potential that had leaked into the potential would
  still give a plausible total energy. So the check is an **equality**: the same
  cell run with and without the correction must produce a density, an
  eigenvalue array and four energy terms that are *bit for bit* identical, and two
  totals differing by exactly the printed `Dispersion Correction`. Both hold. The
  same statement one derivative up is that `d(chi)/d(strain)` is unchanged to
  0.0 — the Sternheimer perturbation is built from the Hamiltonian and a pair
  potential is not in it — which is what will have to change the day a
  density-dependent correction (Tkatchenko-Scheffler, XDM) is added.
- *`rgen`'s fold is not cosmetic.* QE reduces each pair's separation into the cell
  at the origin before building images around it. Doing the same here — one
  translation list for the whole cell, and the separation folded before the list
  is added to it — is what lets the list be built to `rcut + fold_radius(at)`, a
  bound that depends on the **cell alone**. Without the fold the list has to reach
  `rcut` plus the largest separation the *current* geometry has, which is
  `EwaldSum`'s choice and is why that one consults the positions it was built
  with; a `GrimmeD2` built that way loses images the moment an atom is written
  outside the cell — measured, by moving one carbon of QE's graphite cell by
  `a_1 - 2 a_3` at `rcut = 45`: **3.1e-6 Ry**, for a shift that changes no
  distance at all. Silent, and exactly what a relaxation walks into. `jnp.round` has zero
  derivative, so the fold is invisible to `grad`: the integer is frozen and the
  lattice vector it multiplies deforms with the cell like the translations do.
- *Which way the separation points.* `rgen` returns `r = R - (tau_a - tau_b)`; the
  kernel here broadcasts `s = tau_a - tau_b + R`, its negative. The stress is
  quadratic in it and does not notice. The force is linear in it, and getting it
  wrong is a relaxation that walks uphill.
- *The masking rule now has to survive a **second** derivative.* Sanitise the
  *squared* distance before the square root, never the result after it: `sqrt(0)`
  has an infinite derivative and `0 * inf` is NaN. `_real_kernel` already carried
  that comment for the force; here the elastic constants take one more derivative
  of the same expression, so it is tested directly.

**The cutoff looks extravagant and is not.** `london_rcut = 200` bohr for a
`1/r^6` potential invites cutting it, and the shell count is why not: the number
of pairs at distance `r` grows as `r^2`, so the truncation error falls only as
`1/rcut^3`. On QE's graphite cell the sum is -0.039133 Ry at 30 bohr, -0.039945 at
60, -0.039975 at 200 and -0.039975 at 300; reaching the 1e-8 `pw.x` prints takes
~150 bohr. The default here is QE's so that an input reproduces `pw.x` term for
term. The **memory** that buys is `(nat, nat, ntrans, 3)` — 1.6e5 translations and
63 MB of separations for graphite's four atoms, 51 ms for the energy and 153 ms
for its gradient (`PERFORMANCE.md`). A cell with a vacuum has
a large `Omega` and therefore *fewer* translations at the same radius, so the
expensive case is a small dense cell, not a slab.

*Checks met.* On bilayer graphene (PBE, norm-conserving, 12x12x1, `conv_thr =
1e-10`) against the vendored `pw.x`, and on silicon for the two D2 *identities*
at third order:

| what | reference | agreement |
|---|---|---|
| `energy_london` | QE's **committed** `pw_vdw/vdw-d2.in` benchmark, no SCF at all | 1e-8 Ry, its print resolution |
| the total energy | `pw.x` | 3.1e-9 Ry |
| the `Dispersion Correction` term | `pw.x` | 4.9e-9 Ry |
| the density, the eigenvalues, four energy terms | the *same run without the correction* | **exactly zero** |
| `force_london` | `pw.x`'s own `Dispersion contribution to forces` block | 3.9e-9 Ry/bohr |
| `stres_london` | `pw.x`'s own `DFT-D stress (kbar)` row | 1.2e-8 Ry/bohr³ |
| the total force, analytic | `pw.x` | 3.7e-7 Ry/bohr |
| the total stress | `pw.x` | 4.1e-8 Ry/bohr³ |
| `d(chi)/d(strain)` with and against without D2 | itself | **exactly zero** |
| `C_ijkl(D2) - C_ijkl(none)` | `(1/Omega) d^2 E_disp/dx^2`, computed on its own | 2.2e-18 Ry/bohr³ |

**The relaxation is the interesting one, and the geometry is the wrong thing to
compare.** Bilayer graphene relaxed from 7.2 bohr: PBE alone has no minimum in
the interlayer separation, and PBE+D2 settles at **6.10 bohr (3.23 Å)** against a
measured 3.35 and the ~3.2 that D2's known overbinding gives. `pw.x`'s own BFGS
on the same input stops at **6.58 bohr**, 0.48 bohr away and 8.3e-4 Ry higher.
Neither is a bug. The interlayer force constant here is ~2e-4 Ry/bohr² — three
orders below a chemical bond — so the `forc_conv_thr = 1e-5` both codes stop at
pins the separation only to a few tenths of a bohr, and what `max |F|` is actually
measuring is the *stiff* mode, the A/B sublattice buckling inside each layer.
What settles it is asking `pw.x` about pypresso's geometry
(`graphene-bilayer-d2-relaxed.in`, committed): it gives the same total energy to
**1e-8 Ry** and a force of **3e-6 Ry/bohr** there. The two codes walk the same
surface and pypresso walked further down it. So the comparison the test makes is
through the **energies and the forces at each other's answers**, never through
the coordinates.

**What is not here.** The other four corrections `set_vdw_corr` offers, each
refused by name with what it would take: **D3**, whose `C6` depends on each atom's
coordination number, so it is a function of the geometry with a derivative of its
own rather than a table lookup; **Tkatchenko-Scheffler** and **MBD**, whose
coefficients are rescaled by Hirshfeld volumes — functionals of the
self-consistent density, so they enter `v_of_rho` and the null test above is
exactly what would break; and **XDM**, density-dependent for the same reason.
`set_vdw_corr` warns and runs on with *no* correction for a name it does not know,
which for an input asking for D3 is 30 meV on a layered crystal and nothing in the
output that greps as an error; here it stops.

**The third derivative runs on graphene itself**
(`tests/data/qe/graphene-bilayer-electrostriction.in`), and getting there took
two things, one of which is a trap worth more than the case.

*The k-grid has to miss K.* Graphene is a **semimetal**, and the Sternheimer
response here is the insulator one — refused for a metal rather than silently
applied. A `Gamma`-centred `n x n x 1` grid contains `K = (1/3, 1/3, 0)` exactly
when `3` divides `n`, so **2 x 2 x 1 misses it**: the sample is `Gamma` and the
three `M` points, each with a 3.7 eV gap, and `occupations = 'fixed'` is then the
truth about this k-set rather than an approximation to it. What it costs is the
physics — a 2 x 2 x 1 bilayer is not graphene's band structure, and a slab in
vacuum has no bulk `epsilon` at all — so the numbers to read are the agreements.

*And a diverged first-order solution was being consumed in silence.* At QE's
default `alpha_mix = 0.7` the strain response of this cell **diverges**:
`|ddv_scf|^2` grows by 1.34 per iteration, 1.7e7 to 8.9e9 in twenty-five. The
loop then ran out of iterations, returned what it had, and every stage above it
took it — producing an elastic tensor that was not symmetric under the
`C_ijkl = C_klij` that a second derivative of a scalar *is*: **49817 GPa against
-243233** for the same index pair, on a crystal whose stiffest constant is 859.
Nothing in the numbers said "unconverged"; only the identity did.
`require_converged_responses` (and the same refusal inside `elastic_constants`)
now stops it, with `allow_unconverged` for a diagnostic run.

Why a slab and not silicon: the induced Hartree potential is `4 pi e^2/G^2`
against the induced charge, and 14 bohr of vacuum puts the smallest nonzero
`G_z` at `2 pi/c`, where that kernel is two orders larger than a compact cell
reaches. Linear mixing of a map with a Jacobian eigenvalue that large is
unstable above roughly `alpha_mix = 2/(1 + |lambda|)`. Measured: **0.7 diverges
at 1.34 per iteration, 0.3 converges at 0.5 per iteration** and reaches
`tr2 = 1e-14` in **68**. It is the same stiffness the ground-state SCF meets on a
slab and answers with Kerker preconditioning; the response loop has no
preconditioner, so the mixing parameter is the whole of the remedy.

| what | reference | agreement |
|---|---|---|
| `C_ijkl = C_klij` | itself — six independently assembled `jvp` columns | below **1e-9**, the tolerance asserted (and a factor of 5 *with the wrong sign* when the response diverged) |
| `C_1111`, `C_1212`, `C_3333` | a five-point second difference of the SCF energy, dispersion included | 5.8e-5, 4.1e-5, **2.4e-3** |
| `d(eps_ij)/dx_11` | a central difference of `epsilon` over re-converged strained cells | **2.2e-4** — P26's own figure on silicon |
| `C_ijkl` and `d(chi)/dx` under the crystal's twelve operations | `D_3d`, which is **trigonal**: `C_14 = -1.18` GPa is allowed, not a bug | 5.3e-12 and 4.7e-11 |
| `C_66 = (C_11 - C_12)/2` | the three-fold axis, relating three separately assembled columns | 416.264 against 416.264, below 1e-9; `C_11` against `C_22` is 2.7e-12 |

Clamped-ion, in GPa: `C_11 = C_22 = 859.03`, `C_12 = 26.50`, `C_33 = 56.63`,
`C_44 = 27.72`, `C_66 = 416.26`, `C_14 = -1.18`. **These are properties of the
supercell, not of graphite** — the stress is a force per unit volume and half
this cell is vacuum — so they are quoted for the identities they satisfy and not
against a measurement. The compliance is healthy (condition number 32, every
eigenvalue positive from 27.7 to 886.9), so `M` and `Q` are ordinary here rather
than the slab artefact a near-singular shear would have made them.

**One thing found on the way that is not this phase's.** pypresso's symmetry
finder is stricter than `symm_base.f90`'s `accep = 1e-5`: an `ATOMIC_POSITIONS`
card written with QE's customary six digits (`0.333333`) gives bilayer graphene
**4** operations here where `pw.x` finds 12, and 43 k-points where `pw.x` uses 19.
The answers agree; the cost does not. Recorded rather than fixed — changing a
symmetry tolerance is its own decision — and the committed inputs write twelve
digits.

*Notebook 22.*

### P28 — The dynamical matrix of a metal. ✅ DONE.

`pypresso/response/phonon.py` — `_state_weights` and the split in `_force_constants` — and
the deletion of `require_a_metallic_assembly`, which P24c wrote and this phase closes. It
is the smallest phase here in lines changed and it is the one with the sharpest before and
after, because the refusal it lifts had a committed reference sitting beside it
(`reference.out.ph-al2-metal`) and a diagnostic that said "wrong" without saying "how".

**The bug was a weight, and QE's own layout is the proof.** P25 assembles the second
derivative as *one* `jvp` of `grad_u L` along the tangent that carries the coordinate, the
states and the density together — which is right, and which silently assumes that the
weight `L` puts on its states is the weight the *state tangent* wants. For an insulator it
is. For a metal it is not: `orthogonalize`'s smearing branch scales the right-hand side by
`wg1 = f` (P24c), so the occupation is already inside `dpsi`, and `L` weights its states by
`wg = wk f`. The same `f` twice. QE never meets this because its two halves are two
routines and they read different arrays — `dynmat_us.f90:172` takes `wg(ibnd, ikk)` for the
frozen Hessian, `drhodvnl.f90:181` takes `2 wk(ikk)` for the electronic term, and
`drhodvloc` contracts the response *density* against the bare `dvloc`. So the fix is to
split the one `jvp` into the two those routines are:

    D[:, i] = jvp_(u, rho)( grad_u L[wg] )(e_i, drho_i)
            + jvp_psi(     grad_u L[wk] )(dpsi_i)

which is de Gironcoli's Eq. (B19) structure — the frozen Hessian at `wg`, the electronic
response with the metal's own weights. The density tangent can go in either half, because
the terms it reaches (`int vltot(tau) rho`, `E_xc[rho + rho_core(tau)]`) carry no state
weight at all.

**The `df` term the refusal predicted is not needed, and that is the finding of the
phase.** `require_a_metallic_assembly` said the occupations' own first-order change "has to
enter the energy as `df_n` against `d(eps_n)/du` and the entropy's derivative". It does
enter — but it is already inside `dpsi` and needs no term of its own. The
`(f_i - f_j)/(eps_i - eps_j)` structure of `orthogonalize`'s `wwg` *is* the valence-valence
block that generates `df`, and it vanishes identically for an insulator where every
occupied `f` is 1; `ef_shift_wfc` supplies the Fermi level's own motion. Two things
confirm it rather than one: the identity this code already tested (`wk 2 Re[psi* dpsi]`
equals the corrected response density to 1e-10, so for a *local* perturbation contracting
the tangent **is** `int drho dV_bare`, which is `drhodvloc`); and `dfpt_kernels.f90`, where
after the loop QE does exactly two metal-specific things — subtract `def` from `dvscf`,
which the matrix assembly never reads, and run `ef_shift_wfc` on `dpsi`, which this code
already did. Nothing else reaches `dyn`. The prediction was a reasonable reading of
`drhodv` having "its own `wgg`-weighted contraction"; what that contraction actually is, is
the weight swap.

*Check met.* Two-atom fcc aluminium (`al2-metal.in`, `marzari-vanderbilt`, `degauss = 0.05`,
`nosym` on an unshifted 4x4x2 grid) against the vendored `ph.x` on the same input, whose
ground state this reproduces to 1e-9 Ry (−8.332103799 against −8.33210381):

| mode | before (P25's assembly) | after | `ph.x` |
|---|---|---|---|
| acoustic | 155.74, 155.74, 155.74 | 0.700, 1.607, 1.725 | 1.109, 1.827, 1.925 |
| folded pair | 197.96, 197.96 | **146.7093**, **146.7132** | 146.710511, 146.714378 |
| zone centre | 309.26 | **311.0335** | 311.035401 |

in cm⁻¹. The worst of the three real modes is **0.0019** from `ph.x` — an order tighter
than silicon's 0.049, and the same floor either way (QE's `dq = 0.01` form-factor table
against direct integration here). From a run that converges in 9 iterations to
`|ddv_scf|^2 = 8.7e-17` at `av.it. = 23.0`, and whose matrix is symmetric to **1.26e-9**
with nothing imposing it — this input is `nosym`, so nothing symmetrises the assembled
matrix and that figure is the free report on the linear solves it was always meant to be.

**The folded pair is *nearly* degenerate and not exactly so, and the near-miss is the
sharper statement.** It splits by **0.0039**, and `ph.x` splits it by 0.0039 too. This
table read `146.711240` twice when P28 landed, with the pair flat to 2.8e-14 and the
agreement 0.0031 — because `symdynph_gq` was being applied to a `nosym` run (the third
bug in P28a below) and a group average flattens exactly this. The artifact read as a
*stronger* result than the truth, which is the way that kind of error usually presents.

**The acoustic sum rule is the diagnostic and it is the reason this was ever a refusal.**
`sum_b D_(a i)(b j) = 0` exactly, and it now holds to **1.06e-5 Ry/bohr²** against on-site
force constants of 0.0476 — 2.2e-4 relative, which is the finite basis and nothing else.
Before, it was violated by half the optical spectrum. What makes the identity worth
asserting instead of the acoustic *frequencies* is that a factor of `f` on the electronic
half leaves the optical modes looking plausible — 198 against 147 is a 35% error that reads
like a converged answer — while the acoustic modes absorb the rest. `ph.x` prints 1.1/1.8/1.9
and this prints 1.088/1.559/1.559; neither is physics, exactly as on silicon, where this
code prints 4.09 against `ph.x`'s 2.05.

**Three checks, and the insulator is one of them.** The split is unconditional — there is no
metal branch in `_force_constants` — so on an insulator, where `wk = wg` on every occupied
band and `dpsi` is zero on the rest, the two `jvp` must sum to the one they replace.
Silicon's optical mode is unchanged and P25's eleven regressions pass untouched, which is
the whole of the evidence that the refactor is an identity. Beside it, the rigid-translation
identity (`sum_a drho_(a i) = -d(rho)/dr_i`) runs on aluminium too and shares no machinery
with the assembly — it is also the only check on `ef_shift`'s place in the loop, since a
displacement at `q = 0` moves charge in and out of the cell.

**What stays refused, and none of it is this phase's.** The **tetrahedron** occupations
(`dfpt_tetra_beta`, a response weight per band *pair*) and `occupations='from_input'`, both
through `require_a_sternheimer_regime`, which fires whatever the `metals` flag says.
`epsilon_infinity` and the Born charges stay refused for a metal because the quantities do
not exist there, which is why `pw.x` refuses `epsil` for one. Ultrasoft and PAW stay refused
by `require_norm_conserving` for P25's reason, which is orthogonal to this one.

**P24c gets its README row now.** The phase deliberately took none — "it gets its row when
the second derivative does" — because `epsilon_infinity` and `Z*` do not exist for a metal
and the dynamical matrix that would was refused. It does now, so the phonon row covers
metals and P24c is the layer under it.

*Notebook 20 extended.*

### P28a — A supercell is a regime, and it found two bugs. ✅ DONE.

`pypresso/scf/ewald.py`, `pypresso/system/symmetry.py`, and
`tests/data/qe/al4-metal.in` with its `ph.x` reference. This phase has no new
feature in it: it is P28 run on a **bigger cell** — the four-atom conventional
cubic cell of fcc aluminium instead of the two-atom one — and the whole of its
content is what that turned up.

**Why a supercell is a regime and not a bigger case.** Its atoms sit at exact
fractions of the cell, and two things that are otherwise approximate become
*exact*. The structure factor vanishes identically on a whole set of G-vectors —
the extinction rule — rather than cancelling to round-off: **92 of al4's 3287
G-vectors have `|rho|` exactly `0.0`**, where `al2-metal.in` and `si-epsilon.in`
bottom out at 4e-16 and never reach it. And the point group's atom permutations
acquire cycles longer than a transposition: al4's 48 operations contain
**3-cycles** on the three face-centring atoms, where every other cell committed
here permutes atoms only in involutions — one atom in the cell gives the
identity, and diamond silicon and two-atom aluminium only ever swap in pairs.
The first breaks `abs`; the second breaks anything pairing an atom label with a
spatial one. Neither had a cell that could see it.

**Bug 1: `symdvscf` averaged over the atom each operation moves, not the one it
moves onto.** `symmetrize_atom_displacement_density` gathered at `irt[s, a]`
where the derivation gives `irt^-1[s, a]` — an operation carries a displacement
of atom `a` along `i` into one of atom `irt[s,a]` along `R i`, so labelling the
average by the atom it lands *on* puts `S^-1` under the sum. It is 2d7d9d5's
mistake in a second function. **`symmetrize_atom_pair_tensor` is right as it
stands** and does not share the direction, because it carries two atom labels
and *no spatial argument* — that is the thing that fixes it, and both are now
asserted. Measured on al4: **0.33** on a field invariant by construction, and
the wedge at 184.85/345.76/578.36 cm⁻¹ against the whole grid's 199.04/305.41.

**Bug 2: `abs(rho)**2` in the reciprocal Ewald sum.** `abs`'s derivative is
`Re(conj(z) dz)/|z|`, which is `0/0` at `z = 0`, and the extinction rule puts it
there exactly. **The energy and the forces are right either way and the second
derivative is not.** This is `band_density`'s trap, and `modulus`'s, and
`forces/energy.py`'s, in a **fourth** place — and what is new is the way in: the
other three are a node of a wavefunction, a measure-zero accident, and this one
is *forced by the crystal's symmetry* on every cell that is a supercell.

**How it was found, because the route is the lesson.** The assembled matrix
disagreed with a finite difference of forces by 5.5e-4 Ry/bohr² where `ph.x`
agreed with it to 2.3e-5. Eliminated in turn, each by measurement: the
symmetrisation (the unreduced route showed the same error), the ground state
(total energy and Fermi level match `pw.x` exactly), `ef_shift` (`ph.x` prints
1e-25 here — this cell's modes move no charge), the band count (`nbnd` 10 and 16
agree to 1.6e-7), the near-degenerate projector branch (al2 already exercises it
15 times), and the per-mode response density (9.1e-5 against its own finite
difference, its floor). What was left was `jvp(grad)` disagreeing with a finite
difference *of that same gradient* — no SCF, no solve, no symmetrisation on
either side — which a per-term split put entirely on `ewald` (3.037e-4, every
other term ≤ 7e-7). The mechanism was then pinned by the step scaling: **constant
in `h` from 1e-2 to 1e-5 at the symmetric geometry, and 3.2e-8 as soon as the
atoms were displaced 0.05 bohr off it.**

**The identities P25 rests on are blind to this, and that is the finding to
carry forward.** The Ewald error was a *transfer* between the on-site block and
one neighbour — `[-5.47e-4, +5.24e-4, +2.2e-5, +2.2e-5]`, summing to 2.1e-5. The
**acoustic sum rule and the rigid-translation identity are both sums over
atoms**, so a transfer cancels in each and both stayed healthy while the matrix
was wrong by half a percent. The first check here that is not an atom-sum is the
per-mode response density against a finite difference, and it is what cleared
the response and left the energy functional as the only suspect.

*Check met.* `al4-metal.in` against the vendored `ph.x` on the same input, both
codes reducing the 4x4x4 grid to the same 10 k-points — **the first metal phonon
computed on a symmetry-reduced wedge**, since `ph.x` accepts this cell's
symmetry where it refuses al2's:

| multiplet | before (both bugs) | after | `ph.x` |
|---|---|---|---|
| acoustic | 3.18 | 3.1841 | 4.013795 |
| optical `T_1u` | 184.85 | **200.3534** | 200.373700 |
| optical `T_2u` | 345.76 | **200.3538** | 200.387839 |
| optical `T_1u` | 578.36 | **305.4065** | 305.432658 |

**0.020, 0.034 and 0.026 cm⁻¹** on the three optical multiplets — silicon's
floor. Each is a triplet to below 1e-3 with nothing imposing the crystal class,
and the two middle ones are *different irreducible representations* split by
0.014 in `ph.x`, which is the sharpest thing this cell asserts.

**Tests, and they need no reference.** `tests/unit/test_supercell_derivatives.py`
— twelve of them, in 11 s, no SCF anywhere. Both bugs are pinned by identities
the code must satisfy against itself: a displacement field invariant by
construction must survive the average unchanged; a group average must be a
**projector**, `P(P(x)) = P(x)`; and `jvp(grad E_ewald)` must converge to its own
finite difference as `h^2`. Each is parametrised over al4 *and* the two
primitive cells, and the primitive ones pass with the bugs reintroduced — which
is asserted directly (`test_the_supercell_is_the_only_case_with_a_non_involutive_permutation`)
so that the suite states *why* the old cells were blind rather than leaving it
to be rediscovered.

**A third bug, found while writing the two above up, and fixed with them.**
`Calculation.symmetries` returns the full group *even under `nosym`*, which is
deliberate — the group is a property of the crystal and `basis/builder.py` sizes
the FFT box from its fractional translations whatever the input says — and the
switch beside it is `use_symmetry`. Three places in `response/` read the group
instead of the switch and so **symmetrised a `nosym` run**: `symdynph_gq` on the
force constants, and `symtensor` on the Born charges in both `efield.py` and
`born.py`. The stress and the forces write the same two-clause guard by hand and
get it right, which is exactly why the asymmetry survived — there was no single
place for it to be wrong in. `al2-metal.in` is the case it mattered on: it
carries `nosym = .true.`, its header says its response is not symmetrised, and
its twelve operations were being applied to a random matrix to the tune of 2.0.

The fix is structural rather than three guards: `use_symmetry` is public, and
`Calculation.symmetrize_atom_tensor` and `symmetrize_atom_pair_tensor` join
`symmetrize_atom_displacement` as the methods that read it once. **The answers
do not move** — al2's grid is closed under its group, so symmetrising it was a
no-op numerically — which is why this was invisible in every number and is a
correctness fix rather than a change of result.

**Left open, and named rather than fixed.** `abs(...)**2` survives in five other
places: the Hartree energy
and `stres_har` (`rho_g` has *no* exact zeros — 4.9e-17 is its floor here — so
they are latent rather than firing), and `stress/analytic.py`'s and
`forces/spiral.py`'s `abs(psi)**2`, which do meet exact zeros from the `npwx`
padding but are only ever differentiated with respect to a **strain** or `q` at
frozen states, never with respect to the states, which is the direction that
makes a node bite.

*No notebook: this phase adds no feature. Notebook 20 keeps P28's metal section.*

### P28b — Ten sites: the whole feature set at ten atoms per cell. ✅ DONE.

`pypresso/system/symmetry.py`, `pypresso/response/efield.py`,
`tools/generate_reference.py`, twenty `pw.x` inputs and one `ph.x` one under
`tests/data/qe`, with their `pw.x`, `projwfc.x` and `ph.x` references, and
`tests/regression/test_ten_site.py`. Like P28a this phase adds **no feature**:
it is every comparable feature run at ten atoms per cell, and its content is
what that turned up.

Everything committed before ran on one, two, four or eight atoms, and eight was
the largest. Ten cannot be the primitive cell of anything here, so **every
ten-atom cell is a supercell** -- which P28a established is a regime and not a
size. Four things diverged from `pw.x`, three of them bugs on this side, and
none of the four was reachable from a smaller cell.

**Bug 1: the lattice point group was searched over a fixed window.**
`lattice_point_group` looks for the images of each basis vector among lattice
vectors of the same length, and it enumerated candidates over `range(-3, 4)` in
each direction. Five primitive cells stacked along `a3` need a coefficient of
**five**: `pw.x -v high` prints the three-fold of `si10-nc.in` as
`s(2) = ((0,-1,0), (1,-1,0), (0,-5,1))`, and no fixed window smaller than the
supercell multiplicity can hold it. The search found **2 operations where QE
found 6**, symmetrised the density over the wrong group and put the total energy
**3.2e-6 Ry** away from `pw.x`'s -- with both codes converged to 1e-10 and both
reporting success. The window is now the exact bound the geometry gives: an
image `v` of `a_i` has `|v| = |a_i|` and integer coordinates `n_j = v . b_j`, so
`|n_j| <= max_i |a_i| |b_j|`. The same cell then agrees to **3e-9 Ry**, and the
SCF is *twice as fast* because it symmetrises over six operations instead of
two. Nothing in the committed test set could see this: every cell there is
either primitive or a doubling, where a window of three is enough.

**Bug 2: `dielectric_tensor` symmetrised a `nosym` run.** The guard
`Calculation.symmetrize_atom_tensor` exists to hold in one place -- its
docstring names `response/efield.py` as one of the two call sites that reached
for `self.symmetries` directly -- and the dielectric tensor's own `symmatrix`
call at the end of `dielec.f90` was still missing it. It is invisible wherever
the k-grid is closed under the point group, because there the raw tensor is
already symmetric and the symmetrisation is a no-op applied to something that
does not need it. On `si10-epsilon.in`, whose 4x4x1 grid the three-fold does not
preserve, it is worth **0.97** in the off-diagonal entries against `ph.x` --
while the isotropic average, the part symmetrisation cannot move, agrees to
**5e-6**.

**Bug 3: a fractional translation was accepted whatever its denominator.**
`sgam_at` accepts `ft` only when every component is `0` or `1/n` with
`n in {2, 3, 4, 6}` -- the orders a screw axis or a glide plane can have in
three dimensions -- and `find_symmetries` had no such test. Five-layer graphite
has a mirror plane at `z = 2/5`, which with the origin at a layer is written
`ft = (0, 0, 4/5)`: a real symmetry that QE **drops**, and keeping it is not a
better calculation but a different one. `fft_fact` then wants the FFT dimensions
to be multiples of five, and `c10-graphite-d2` came out on a 20x20x**135** grid
where `pw.x` chooses 20x20x**128**. The exchange-correlation energy is evaluated
pointwise on that grid, so the totals differed by **1.7e-4 Ry** with neither
code wrong. QE's filter is transcribed now; the grid, the operation count and
the energy all follow.

**Not a bug, and the one worth knowing about: an unequal k-grid is not closed
under the lattice point group, and the two codes then build genuinely different
reduced sets.** `kpoint_grid` reduces with the *lattice*'s group, keeping only
rotations that map the grid onto itself, and `irreducible_BZ` (`irrek.f90`) then
completes that list for the crystal's smaller group by coset decomposition --
generating `S k` for coset representatives `S` of the **lattice** group. On a
`4 x 4 x 1` grid those images leave the grid: displacing one atom of
`si10-nc.in` drops the group from six operations to two, and seven of the
fourteen points `pw.x` ends with have a third crystal coordinate of `1/4`, which
a grid with one division along that axis has no points at. pypresso reduces the
*requested* grid with the crystal's own operations, so its seven points are grid
points and their weights are the orbit sizes. The totals differ by **6.9e-5 Ry**
and `pw.x`'s own `nosym` run over the same grid says which is which: it gives
-78.97348341, which is pypresso's reduced answer to nine digits and not QE's own
reduced -78.97341444. On `4 x 4 x 4` -- closed under every integer rotation --
`pw.x` reproduces its `nosym` total exactly and the two codes agree to 2e-9. The
committed pair `si10-nc-anisotropic.in` and `si10-nc-anisotropic-nosym.in` is
that experiment; the force cases run on `4 x 4 x 4` because of it.

**Two things `ph.x` will not do on such a cell, and both are its own limits.**
`phq_setup` requires every symmetry operation to map the FFT grid onto itself
and stops with "FFT grid incompatible with symmetry" on the 15 x 15 x 80 grid of
the five-cell stack, so `si10-epsilon.in` runs `nosym` -- which is the
configuration `pypresso.response` accepts on an unshifted grid, and which makes
the comparison exact rather than a comparison of two wedges. And with
`occupations = 'tetrahedra'` (Bloechl's) `pw.x` never converges the elongated
metallic supercell at all -- 100 iterations with the total energy stable in the
eighth decimal and the scf accuracy stalled at 1.3e-4 -- where the optimised
method converges in 55, which is why `al10-metal-tetra.in` says `tetrahedra_opt`.

**What ten sites cost the comparison itself, and it is a real limit rather than
a defect:** a five-cell supercell folds the primitive bands onto each other, so
nearly every level is degenerate, and `|<phi|S|psi_n>|^2` for a single band is
not a well-defined number inside a degenerate subspace (rule D4). The two
codes' projections differ by **0.138** band by band on `si10-nc` and by 0.0017
once each degenerate group is summed -- which is what `print_proj`'s three
decimals are worth over a group of five. The invariant quantities are compared
instead and agree: the Löwdin charges to **4.8e-5**, the spilling parameter to
the four decimals it is printed with, and every one of the twenty `filpdos`
columns to 0.3% of its peak.

**A trap in the tooling, not in the physics.** `pw.x` reading its input from
standard input copies it to a scratch file named `input_tmp.in` **in the working
directory** (`Modules/open_close_input_file.f90`), so two runs sharing a
directory overwrite each other's input and one of them silently computes the
other's system. Generating five of these references concurrently produced a
hydrogen chain whose stdout was silicon's, and it only surfaced at all because
the crossed input had a different `ATOMIC_SPECIES` count. `_invoke` now runs
each `pw.x` in its own directory, as `run_projwfc` already did.

**What the seventeen cases measure.** Every row is one `pw.x` input run through both codes
at the same threshold; `dE` is the total energy in Ry, and the symmetry count, the k-set,
the FFT dimensions and the G-vector count agree exactly on every one of them.

| case | what it adds | dE (Ry) | bands (eV) | other |
|---|---|---|---|---|
| `si10-nc` | LDA norm-conserving | 3.1e-9 | 4.9e-5 | HOMO 1.6e-5 eV |
| `si10-nc-pbe` | the gradient correction | 1.7e-9 | 5.5e-5 | |
| `si10-us` | ultrasoft: two grids, `Q_ij`, `D_ij` | 1.1e-9 | 5.5e-5 | |
| `si10-paw` | the one-centre terms | 3.5e-9 | 5.1e-5 | |
| `si10-paw-pbe` | PAW and PBE together | 2.1e-9 | 4.8e-5 | |
| `al10-metal` | a metal, `marzari-vanderbilt` | 1.9e-9 | 5.0e-5 | `E_F` 2.3e-5 eV, stress 3.0e-9 |
| `al10-metal-tetra` | optimised tetrahedra | 4.7e-9 | 5.0e-5 | `E_F` 5.4e-6 eV |
| `h10-chain-lsda` | `nspin = 2`, antiferromagnetic | 4.3e-9 | 1.0e-4 | `\|m\|` 7.1043 against 7.10 |
| `h10-chain-noncolin` | the same state as spinors | 4.4e-9 | 2.9e-4 | equals the collinear total both sides |
| `c10-graphite-d2` | Grimme D2, five layers | 3.2e-10 | 5e-5 (bar the empty top band) | dispersion term 6e-9, force 4.1e-7 |
| `ni10-ldau` | DFT+U, ten `ns` matrices | 1.8e-8 | 5.4e-5 | `m` 5.9036 against 5.90 |
| `bi10-soc` | spin-orbit, 150 spinor bands | **1.9e-4** | 1.6e-2 | the one that does not close; see below |
| `si10-nc-force` | one atom displaced | 2.2e-9 | 5.4e-5 | force 2.4e-7, stress 4.8e-9 |
| `si10-us-force` | the same, `addusforce` | 6.3e-9 | 5.7e-5 | force 7.9e-7, stress 9.4e-8 |
| `si10-paw-force` | the same, PAW | 9.8e-9 | 5.3e-5 | force 4.8e-7, stress 9.6e-8 |
| `si10-nc-bands` | an explicit band path | -- | 5.8e-5 | 19 k-points x 26 bands |
| `si10-nc-relax` | BFGS with ten moving atoms | 5.2e-9 | -- | geometry 6.8e-6 bohr, 11 SCF cycles each |

On top of those: `si10-epsilon` against `ph.x` gives **7.0e-6** on every component of
`epsilon_infinity` (a tensor of order 19) and **1.1e-5** on the mean `Z*` of all ten atoms,
which is the resolution `ph.x` prints them at; and `projwfc.x` gives forty channels in QE's
order, Löwdin charges to 4.8e-5 electrons, a spilling parameter of 0.0083 to its four
printed decimals, and all twenty `filpdos` columns to 0.3% of their peak.

**`bi10-soc` is the one case that does not reach 1e-6 Ry, and it is recorded rather than
explained.** Ten bismuth atoms with a fully-relativistic `dn` dataset is 150 valence
electrons, 150 occupied spinor bands and 30 empty ones on a 216 x 45 x 81 grid. Both codes
choose the same 8 operations, the same single k-point, the same grid and the same 302569
G-vectors, and both converge; the totals sit **1.9e-4 Ry** apart on -1477.737, which is
1.3e-7 relative. The signature is two slightly different converged densities -- the
one-electron and Hartree terms are 3.4e-3 and 3.6e-3 apart and cancel into the total, while
the Ewald term, which depends on no density at all, agrees to 4.6e-9. What it is not: a
geometry, a unit, a k-set or a grid. The test asserts the bound that was measured (5e-4 Ry)
rather than one that was hoped for, and this is the open item the phase leaves behind.

**Two more things ten sites cost, both stated rather than fixed.** The *memory*: a force and
a stress on the displaced cell at 24 k-points peak at 1.5 GB norm-conserving and **16 GB**
ultrasoft or PAW, and the spin-orbit SCF alone at 18.4 GB -- the augmentation charge on the
dense grid, kept live by the reverse pass, which is P11's `si8-us` measurement one cell size
up (`PERFORMANCE.md`). And the *degeneracy*: a five-cell supercell folds the primitive bands
onto each other, so `|<phi|S|psi_n>|^2` for a single band is not a well-defined number
(rule D4) -- the two codes differ by 0.138 band by band on `si10-nc` and by 0.0017 once each
degenerate group is summed, so the projection is compared through its invariants instead.

**A third thing, and it is a gap rather than a cost:** `run_relax` takes `etot_conv_thr`,
`forc_conv_thr` and `nstep` as arguments and **does not read them from the input** --
`System` carries no `&ions` field. A `pw.x` input asking for a tighter `forc_conv_thr` is
therefore honoured by `pw.x` and ignored here, and the two stop at different points on the
same curve: with `forc_conv_thr = 1e-4` in the file, `pw.x` took 26 BFGS steps to a residual
force under 1e-4 while this code stopped after 11 at 9.5e-4 and the geometries were 0.057
bohr apart. At QE's defaults on both sides they take 11 SCF cycles each and agree to
**6.8e-6 bohr**. `si10-nc-relax.in` says so in its header; threading the `&ions` namelist
through `System` is the fix and is not done here.

**Status of the sweep itself:** `tests/regression/test_ten_site.py` is **27 passed in
42:55** with a sampled peak of 11.4 GB -- it was 1:28:39 and 22.8 GB until the *compiled
executables* were released between cases as well as the converged states, since nothing in
this file shares a shape with anything else in it (`PERFORMANCE.md`) -- and
`tests/regression/test_shapes_against_qe.py` is 90 passed in twelve seconds. Every number
in the table above is enforced by one of those rather than measured once and written down.

*No notebook and no README row: this phase adds no feature. What it adds to the
test suite is `tests/regression/test_ten_site.py` -- the ten-site sweep itself --
and `tests/regression/test_shapes_against_qe.py`, which compares the two numbers
`pw.x` prints before it starts iterating (`N Sym. Ops.` and the FFT dimensions)
for **every** committed reference, in twelve seconds. All three bugs above were
visible in one of those two numbers long before any energy was.*


### P29 — Variable-cell relaxation: the cell as nine more coordinates. ✅ DONE.

`pypresso/relax/cell.py`, `pypresso/relax/settings.py`, `pypresso/workflows/vc_relax.py`,
`Calculation.at_cell`, `System.with_cell`, `check_lattice_symmetry`, and the variable-cell
half of `pypresso/relax/bfgs.py`. `calculation = 'vc-relax'`.

**The objection this phase was refused on is QE's own and QE answers it.** The
entry above read "a moving cell would also invalidate the rule that the FFT grid
and the symmetry group are fixed once for the whole run". It does not.
`scale_h.f90` re-expresses the **same** G-vectors — the same Miller indices, the
same sphere membership, the same FFT dimensions, the same k-points in crystal
coordinates — against the new reciprocal cell, and `igk_k` is not regenerated
either: QE prints the rising "New effective cutoffs" rather than changing the
basis. That is exactly `Calculation.at_strain`, which P11 already wrote. So the
relaxation is **one run** under the fixed-setup rule from beginning to end. When
it converges, `reset_gvectors` throws the setup away and runs **one more SCF
from scratch** — "Final scf calculation at the relaxed structure. The G-vectors
are recalculated for the final unit cell. Results may differ from those at the
preceding step." Two runs, each obeying the rule, and the difference between
their energies is the Pulay error of having relaxed in a basis chosen for a
different cell (`VCRelaxResult.pulay_error`), reported rather than left to be
noticed. `treinit_gvecs` is QE's escape hatch and is here too: rebuild
everything on every accepted step, pay a full setup per step, and the error is
zero — and, as QE does, skip the final SCF, because every step already was one.

**The cell gradient is the stress rearranged, and the rearrangement is exact.**
With `h` the matrix whose *columns* are the lattice vectors, `h -> (1+eps) h`
gives `eps = dh h^-1`, so `dE/dh = (dE/d eps) h^-T`; with
`sigma = -(1/Omega) dE/d eps` and `d Omega/dh = Omega h^-T`, the gradient of the
**enthalpy** `H = E + P Omega` is

    dH/dh = Omega (P I - sigma) h^-T,

which is `cell_base.f90`'s `cell_force` line for line. Two things fall out and
both are used: the stationary point is **`sigma = P I`** — a relaxed crystal
carries the applied pressure, it does not have zero stress — and
`(dH/dh) h^T / Omega = P I - sigma` recovers the stress from the gradient, which
is how the cell's convergence is measured and why QE prints it in kbar.

**Three transcription details, each of which is silent if dropped.** The metric
is rebuilt **every step** and the Hessian is not: QE allocates `metric`,
`inv_metric` and `hinv_block` inside `bfgs()` from the `h` it was passed while
`inv_hess` is read back untouched, and computing the metric once in the
constructor is exact at fixed cell and wrong at variable cell. `iforceh` is
re-applied after **every** product with the inverse Hessian, not once to the
gradient, because `inv_hess` stops being block diagonal after the first update
and mixes a free component back into a frozen one. And the cell block's metric
is `0.04 omega g^-1` where an atom's is `g` — a factor with no derivation in the
Fortran, whose job is to make one `trust_radius_max` govern both, carried over
verbatim because a different value is a different optimizer.

**The bug this phase found, and it is P28a's shape exactly: the energy was right
and its derivative was not.** `at_strain` rebuilt the k-points as
`system.kpoints.crystal(system.cell)`. `KPoints.coords` are cartesian in units
of `2 pi/alat`, so they describe a k-set only together with the cell they were
built for — and `at_strain` updates the cell without updating them. That is the
same number only while the two are consistent, which is true of every caller
P11, P24 and P26 ever had, and false the moment a cell has actually *moved*. A
**second** `at_strain` — which is what a stress on a stepped cell is —
differentiated at k-points **0.031 away in crystal coordinates** from the ones
the SCF had just run at. Measured on `vc-relax4` at QE's own relaxed cell, with
both codes holding the same 4159 G-vectors on the same 24³ grid:

| | frozen basis | fresh basis |
|---|---|---|
| `pw.x` | 500.04 kbar | 502.03 |
| pypresso, before | **564.05** | 502.03 |
| pypresso, after | **500.04** | 502.03 |
| a central difference of the energy | **500.12** | 510.30 |

The finite difference is what settled it: 64 kbar is not a Pulay term that
autodiff sees and QE's analytic expressions miss — a defensible-sounding story —
because the frozen-basis energy's own derivative is 500.12. `Calculation` now
carries `_kcrystal`, decided once, and every frozen-sphere mover rebuilds from
it. The relaxed volume of `vc-relax4` moved from 194.52 to QE's 190.79 bohr³,
**2%**.

**The second finding is a tolerance, and it is dimensional.** The lattice point
group was searched with an absolute `1e-6` applied to lengths in **bohr** and to
metric entries in **bohr²**, where `symm_base.f90`'s `eps1 = 1e-6` is applied to
`at`, which is in units of `alat`. The same crystal therefore loses operations
as its lattice constant grows — a factor of `alat²` ≈ 49 on this cell. QE's own
`vc-relax4.in`, whose cell is written to eight decimals, has its metric
off-diagonals spread by 1.7e-7 alat² (inside QE's threshold) and **8.5e-6 bohr²**
(outside a bare 1e-6): eight of twelve operations dropped, and a k-set of 20
where `pw.x` builds 10. A variable-cell run makes this worse than a fixed
setting, since the cell it applies to changes size during the run. The
comparison is scale-free now, and `test_shapes_against_qe.py`'s 91 cases are
unchanged by it.

**Two things `at_strain` leaves stale that a moving cell makes live**, and both
are `at_cell`'s job rather than `at_strain`'s — the pair being `at_spiral_q`'s
"frozen while differentiating, rebuilt to move". The **Ewald and dispersion
neighbour lists**: freezing them is right for a derivative at one geometry, where
no image is gained or lost and the `rmax`/`rcut` boundary sits where the terms are
1e-8 and 1e-12 Ry, and wrong for a *step*, which can shrink the cell by per cent
so that an image outside the enumeration radius is inside `rmax` and simply
missing — an `erfc` tail that converges and reports success. `rgen` and `ewald`
run afresh on every ionic step in QE for this reason. And **`rho_atomic_species`**,
which only the analytic `force_corr` reads and which a relaxation asking for the
analytic force as a cross-check does read.

**`check_symmetry` needed a second half.** It works in crystal coordinates, so a
deformation of the cell leaves every one of its numbers untouched: a cubic
crystal stretched tetragonal passes it unchanged with four rotations that are no
longer symmetries of anything. `check_lattice_symmetry` checks the metric,
`R g R^T = g`, which is where `symm_base.f90` finds the group in the first place.

**P28b's stated gap is closed on the way past.** `RelaxSettings` reads
`&control`'s `etot_conv_thr`/`forc_conv_thr`/`nstep`, `&ions`'s
`ion_dynamics`/`upscale` and `&cell`'s
`cell_dynamics`/`press`/`press_conv_thr`/`cell_dofree`/`treinit_gvecs` onto
`System`, so a file asking for a tighter threshold is honoured rather than parsed
and ignored — which is what left `pw.x` taking 26 BFGS steps to this code's 11 on
`si10-nc-relax.in`, the geometries 0.057 bohr apart and both codes reporting
success. `press` and `press_conv_thr` are in **kbar** in the input and are
converted once, at the point of use.

**What the cases measure.** `pw_vc-relax/` has six inputs; `vc-relax1` and
`vc-relax2` are `cell_dynamics = 'damp-w'` — Wentzcovitch damped dynamics with a
fictitious cell mass, a different optimizer rather than a different setting — and
are refused by name, as are the four `cell_dofree` values that impose a
constraint beyond their mask (`shape`, `2Dshape`, `volume`, `ibrav`).

**`vc-relax3` runs with a different symmetry group on each side, and it is the
sharpest case here because of it.** `symm_base.f90` tests a fixed catalogue of
rotation matrices written in a canonical cartesian frame, so QE finds a symmetry
only when the crystal is presented in one of those frames; `lattice_point_group`
here searches for lattice vectors of matching lengths and angles, which is
orientation-free and is what its module docstring has always claimed.
`vc-relax3` and `vc-relax4` are the *same* rhombohedral arsenic in two settings,
and QE finds **2** operations for the first and **12** for the second where this
code finds 12 for both. So on `vc-relax3` the two codes reduce the same 4x4x4
grid to **32** points and to **10**, symmetrise over groups of 2 and of 12 — and
agree on the relaxed volume to **3e-5 bohr³** and on the final energy to
**1e-8 Ry**. It was expected to be the case that could not be compared, on P28b's
unequal-grid reasoning; it is instead the statement that this grid *is* closed
under the larger group, which P28b established is not automatic. The case where
it fails is a grid with unequal divisions, and this is not one.

**The bigger cells, and both of them refused the case they were written for.**

**Eight-atom cubic silicon** is the supercell regime (P28a) applied to a cell
gradient: eight atoms and nine cell coordinates in one Hessian, and a structure
factor that vanishes *exactly* wherever the conventional cell's fractions kill
it. It was written at **500 kbar and is not a test there**, and the failure looks
like a bug in the SCF, which is what made it take a while to see: the two codes'
relaxed geometries agree to **5e-5 bohr** and their final energies are **7.3e-3
Ry** apart, on identical setups — 12557 G-vectors, a 30³ grid, 24 operations, 10
k-points — and pypresso reproduces its own 7.3e-3 discrepancy at *QE's* cell,
which rules the geometry out. The reference says what it is: `convergence NOT
achieved after 100 iterations`. 500 kbar compresses this cell by 25% and closes
silicon's gap, so a run at the default fixed occupations is ill-posed; `pw.x`
relaxes happily in the frozen basis and then fails its final SCF, while pypresso
converges the same ill-posed problem to a fixed-occupation solution QE never
reaches. Two codes disagreeing about a system neither should be describing is not
a comparison. **At 100 kbar** the cell moves 8.4% — as much as the arsenic cases
— stays an insulator on both sides, and the two agree on the relaxed volume to
**1.1e-4 bohr³** and on the final energy to **5e-8 Ry**.

That left a trap in the *parser* worth more than the case: `pw.x` prints
`!  total energy` **only** when the SCF converged, so `final_total_energy` — "the
last one" — is really "the last *converged* one", and on a run whose final SCF
failed it is silently the last ionic step's, in the relaxation's own basis rather
than the rebuilt one. `QEReference.scf_converged` now says so and the tests
assert it before comparing anything to it.

**Ten-atom five-layer graphite is not a variable-cell relaxation at this cutoff,
and finding that out is what it contributed.** It was chosen for the property no
cubic cell has -- `a` is a covalent bond and `c` is held only by the D2
correction, so the two respond to different physics and a cell block that got
them wrong together could not hide. It failed three ways, each of which is worth
having written down:

* **Whole cell free, basis frozen: it collapses by 26% in volume.** Not physics.
  `pw.x` prints a `Final enthalpy` of -113.0741 Ry and its own final SCF at that
  cell gives **-112.6281** — *above* the -112.7747 the starting geometry had.
* **Constrained with `cell_dofree = 'z'` so that only `c` can move: it still
  collapses**, 31.65 → 25.35 bohr, `Final enthalpy` -113.0268 against a final SCF
  of **-112.7102**. The prediction was that constraining it would keep the cell
  inside what a fixed basis can be trusted for; the constraint leaves free
  exactly the soft axis, which is the one that runs away. A 20% contraction along
  `c` makes the frozen sphere over-complete in precisely that direction, and
  `C.pbe-hgh` at `ecutwfc = 40` is far enough from converged that the basis gain
  outweighs the physics being minimised. **Silicon compressed 25% and paid
  8.7e-5 Ry for it; this pays 0.32.**
* **`treinit_gvecs` removes the Pulay error and the run then does not converge at
  all.** 50 ionic steps, 18 minutes of `pw.x`, and "The maximum number of steps
  has been reached" with the volume still oscillating around 555-560 bohr³.
  Rebuilding the grids every step makes the energy surface *discontinuous* — the
  FFT dimension along `c` changes as `c` does and `etxc` is evaluated pointwise on
  it — so a line search with Wolfe conditions has nothing to converge to. That is
  the trade `treinit_gvecs` makes, and it is the reason it is not simply the
  better setting.

Three failures in a row on one crystal is a statement about `vc-relax` rather
than about this code, and both codes make it identically: **a layered crystal at
a modest cutoff has no trustworthy relaxed cell by either route**, and
`VCRelaxResult.pulay_error` is the number that says which route you are on. The
case is not committed as an agreement test, because there is nothing converged to
agree about.

**The ten-atom case that works is silicon**, and it is P28b's own geometry: five
primitive cells stacked along `a3` **with the second atom displaced**, which is
`si10-nc-force.in`'s crystal. The displacement is the point rather than an
accident of where the file came from — it drops the group from six operations to
**two**, so nothing imposes the answer, and the ten atoms have to find their way
back to the ideal stack while the cell finds its way to the applied pressure,
coupled through one Hessian over `3 nat + 9 = 39` coordinates. Every other case
in the set is two atoms or cubic. The cell is as far from cubic as graphite is —
`a3` is five primitive cells where `a1` and `a2` are one — without a soft axis
held together by a dispersion correction. It runs on `4 4 4` rather than the
`4 4 1` its shape suggests, for P28b's reason: on a grid with unequal divisions
the two codes build genuinely different irreducible sets and the totals differ by
6.9e-5 Ry with neither wrong. And at `ecutwfc = 20` rather than the 12 the other
si10 cases use, because what a vc-relax needs small is not the basis error but
its *derivative* with respect to the cell.

**And it relaxes anisotropically, which is the property the case exists for.**
`pw.x` takes the volume to 91.6% — the same 8.4% compression si8 gets at the same
pressure, as it should, being the same material — but the three lattice vectors
do *not* scale together: **0.975595, 0.975595, 0.966743**. The long axis
contracts by a third more than the short ones. Nothing imposes either the
anisotropy or its absence (`cell_dofree` is `'all'` and the crystal has two
symmetry operations), so this is the first case in the set where the cell's
*shape* degrees of freedom are doing something a volume scaling could not, and
the comparison against `pw.x` is over all nine entries rather than effectively
over one.

**What the two big cells agree to, and why the numbers differ by four orders of
magnitude.** Eight-atom cubic silicon matches `pw.x`'s relaxed cell to
**2.65e-7 bohr**; the ten-atom stack to **2.28e-3**. Neither is a defect and the
gap is not accuracy: what `press_conv_thr = 0.5` kbar permits is a *linear
strain*, `0.5/(3 B)` with silicon's bulk modulus about 980 kbar, so 1.7e-4 — and
that is 1.7e-3 bohr on a 9.9-bohr cubic axis and **5.9e-3** on the 34.9-bohr long
axis of a five-cell stack. si8 stops 6300x tighter than its own allowance
because it is cubic with 24 operations, so the cell has effectively one free
parameter and both codes converge it hard; si10 stops at 40% of its allowance
because two operations and 39 coupled coordinates let the two BFGS trajectories
separate before both satisfy the same thresholds. Tightening si10's bound means
tightening `press_conv_thr` on both sides rather than fixing anything, so the
test asserts the volume and the energy beside it — neither of which inherits the
long axis's amplification.

**The 500 kbar cases are not harder versions of the 0 kbar one.** At zero
pressure the enthalpy is the energy and `P Omega` is identically absent; at 500
kbar arsenic compresses by 10% *and* its two atoms move from 0.2722 to 0.2500 —
the rhombohedral-to-simple-cubic transition — so the cell and the atoms are both
doing something and doing it at once.

*Notebook 23.*


### P30 — The Tran-Blaha potential: a functional that is not a derivative. ✅ DONE.

`pypresso/xc/mgga.py`, the meta slot of `pypresso/xc/functional.py`,
`meta_exchange` in `pypresso/scf/potential.py`, `kinetic_energy_density` in
`pypresso/scf/density.py`, `laplacian` in `pypresso/basis/gradients.py`,
`PlaneWaveBasis.kplusg`, and `tau` threaded through `Calculation.potential`,
`run_scf`, `run_bands`/`run_nscf` and `ScfResidual`. `input_dft = 'tb09'`,
`'bj06'`, and `mbj_c`.

**This phase runs the project's own rule backwards, and that is the whole of
what makes it different.** Every functional before it is written as an *energy*
and its potential comes from `jax.grad` (rule D1). Tran and Blaha's modified
Becke-Johnson potential is a **potential**: there is no `E_x[rho]` whose
functional derivative it is, and the 2009 Letter says so. What that costs is
that the SCF has no variational total energy — the number `run_scf` reports is
the band term plus the electrostatics plus *correlation only* — and therefore
forces, the stress, the dynamical matrix and the whole of linear response are
refused by name (`forces/energy.py:reject_potential_only`, reached by all of
them through `energy_at`). What it buys is a band gap: silicon goes from LDA's
0.49 eV to 1.13 eV against an experimental 1.17, at the cost of a
gradient-corrected functional and not of a hybrid or of `GW`.

**There is no Fortran to transcribe, and finding that out is half the phase.**
QE reaches TB09 only through libxc — `dft_setting_routines.f90` maps `imeta = 3`
to libxc's 208 under `#if defined(__LIBXC)`, and `qe_drivers_mgga.f90`'s native
`SELECT CASE` has entries for TPSS and M06L and nothing else. So the reference
followed here is libxc's own definition (`maple/mgga_vxc/mgga_x_tb09.mpl`,
`maple/mgga_exc/mgga_x_br89.mpl`, and the bracketing in `src/mgga_x_br89.c`),
and two things about QE's route mean a `pw.x` number is **not a reference for
this functional**:

- **QE passes a zero Laplacian.** `XClib/xc_wrapper_mgga.f90` declares
  `lapl_rho` with the comment `! not used in QE` and sets it to zero before
  every `xc_f03_mgga_vxc` call. `XC_MGGA_X_TB09` is flagged
  `XC_FLAGS_NEEDS_LAPLACIAN`, and the Laplacian is *the* ingredient of the
  Becke-Roussel fit — it is what `Q` is built from. In a plane-wave basis it is
  the cheapest derivative there is, `-G^2 rho(G)`, one transform per channel,
  which is why it is here.
- **QE never sets `c`.** `set_ext_params` is called with libxc's default
  parameter list, and libxc's own description of the parameter says why it has
  to be: *"This parameter involves an average over the unit cell and must be
  calculated by the calling program."* The default is `c = 1`. So
  `input_dft = 'tb09'` in `pw.x` is **Becke-Johnson, not Tran-Blaha**, and
  without a Laplacian at that. Both are offered here (`'bj06'` is the `c = 1`
  row) precisely so the difference is a measurement rather than a footnote: on
  converged silicon it is 1.018 eV against 1.134 eV, a fifth of the gap the
  functional opens.

**The validation is analytic, and it is stronger than a cross-code float
comparison would have been.** Two limits pin the whole chain — the sign of `Q`,
which branch of the nonlinear solve is taken, the `rho^(1/3)` prefactor, the
Hartree `tau` convention and the spin scaling:

- **The hydrogen atom.** For a one-orbital density `D = 2 tau - |grad rho|^2/(4
  rho)` vanishes identically, so `Q = lap rho / 6` whatever `gamma` is, and
  Becke-Roussel reduces to the exact Slater potential of the 1s orbital,
  `-(1/r)[1 - (1+r) e^(-2r)]`. It does, **to 6e-13 pointwise** — machine
  precision for arithmetic that runs through an exponential, a cube root and a
  bisection — everywhere the density is clear of the functional's own threshold,
  and `E_x = (1/2) int rho v_x^BR` comes out **-0.3125 Ha = -5/16**, the exact
  value, to 1e-5. Becke and Roussel's model *is* exact for a one-orbital
  density, so this is a transcription check with no tolerance to argue about.
- **The uniform electron gas.** Becke-Johnson's `sqrt(5/12)/pi` is fixed by
  requiring `v_x^BJ = v_x^LDA` there — the second term evaluates to exactly
  `+(1/2)(6 rho_s/pi)^(1/3)`, since `sqrt(5/12) sqrt(3/5) = 1/2` identically,
  and the Slater potential to `-(3/2)(6 rho_s/pi)^(1/3)`. The identity therefore
  holds *if and only if* Becke-Roussel reproduces the uniform gas's Slater
  potential, and it does not quite: the result here is **6.0e-4 relative, at
  every density**, scale-free. That residue is the model's, not the code's, and
  measuring it identified what `gamma = 0.8` actually is: the ratio
  `v_x^BR / v_x^Slater` in the uniform limit is 1.0281 at `gamma = 0.6`,
  **0.99960 at 0.8** and 0.9745 at 1.0, so Becke and Roussel's constant is the
  uniform-gas fit to four digits, and no member of the family does better than
  about 4e-4.

**`tau` is the second thing the SCF carries, and it is not a function of the
density.** It comes from the states — `sum_band.f90`'s meta branch, three extra
transforms per band, `i(k+G) c_G` one cartesian direction at a time — so a run
under this functional carries `(rho, tau)` where every other run carries `rho`.
The identity that pins it is exact and cheap: `int tau dr = sum_i w_i sum_G
|k+G|^2 |c_G|^2`, the band kinetic energy, which agrees to **1.8e-15 Ry**.
QE mixes `rho` and does *not* mix `kin_r` (`mix_rho.f90` never mentions it) —
`tau` is simply recomputed from the output states and used in the next
iteration's potential, and that is what the mixing loop here does too. The first
iteration has no states, so it starts from `potinit.f90`'s Thomas-Fermi guess
`(3/5)(3 pi^2)^(2/3) rho^(5/3)`.

**Traps, in the order they cost time:**

- **An `equinox` field silently shadows a method of the same name.** Adding
  `meta_c: float | None` to `Functional` — which already had a `meta_c(rho,
  grad_rho)` method — made the *method* the field's default value, so the
  dataclass carried a function as a pytree leaf and `jit` tried to trace it as
  an array. The error surfaced two frames away, inside `v_of_rho`, as "the
  problematic value is of type `<class 'function'>`". The field is `imposed_c`
  now, and the general rule is that a frozen-dataclass field name and a method
  name occupy the same namespace.
- **`nspin = 2` is stored as `(up, down)` here and as `(total, magnetization)`
  in QE**, and the meta branch was written for QE's. The result was an
  unpolarized silicon whose two channels came out **7.4 eV apart**, with a
  perfectly converged SCF reporting success. Nothing an unpolarized run does can
  see it: the `nspin = 1` path takes half the total density and is right either
  way. The test that catches it is the cheapest one available — `nspin = 2` with
  zero starting magnetization must reproduce `nspin = 1`, and it now does to
  **1e-12 in `c`, 3.6e-15 Ry in the energy and 6.3e-14 eV in the eigenvalues**,
  with `tau_up + tau_down = tau` to 1.9e-16. QE's own comment in `potinit.f90`
  ("for LSDA rho is (tot,magn), rho_kin is (up,down)") is about *its* two
  conventions and is exactly the thing not to transcribe.
- **`tau` must be symmetrised and it is easy to believe QE does not.**
  `sum_band.f90` calls `sym_rho` on `rho%of_g`, then twenty lines later calls it
  again on `rho%kin_g` inside the meta branch — the second call is easy to miss,
  and `mix_rho.f90`'s silence about `kin_r` makes the wrong conclusion look
  confirmed. It matters: on an irreducible wedge the unsymmetrised `tau` is
  **11% asymmetric**, and running with it moves the eigenvalues by **0.47 eV**
  and the total by 1.3e-2 Ry. `Calculation.kinetic_energy_density(symmetrize =
  False)` exists only so that number can be measured.
- **The highest band of an `nbnd` window does not converge under this
  functional where it does under LDA.** On QE's own silicon at `nbnd = 8`, band
  8 comes out **4.9e-3 Ry** from a dense diagonalisation of the *same*
  Hamiltonian while every band below it is within 5e-7; at `nbnd = 10` the whole
  window is within 1e-6. Davidson resolves the top of its window last, and the
  mBJ potential — which carries the structure of `|grad rho|/rho` and
  `sqrt(tau/rho)` — mixes that band with the ones just outside it far more than
  a local potential does. It is invisible in the SCF, because the density is
  built from the occupied bands: `c` and the total energy agree to every digit
  either way, and only a gap read off the top of the window is wrong. **How it
  surfaced is the part worth keeping**: a band structure rebuilt from the
  converged density disagreed with the SCF's own eigenvalues by exactly that
  4.9e-3, and it was the *band structure* that was right — it diagonalises from
  scratch at a tight `ethr`, where the SCF carries a subspace that was never
  asked to resolve its top. The first reading of that disagreement was that
  `tau` had gone stale, and it had not.
- **A band path cannot rebuild `tau`.** It is a property of the occupied states
  over the whole zone and a band path has no occupations at all, so
  `run_bands`/`run_nscf` take it as an argument and refuse without it — the same
  argument, and the same refusal, as PAW's `becsum`.
- **The nonlinear solve's derivative is not the bisection's.** `x` comes from a
  fixed-length bisection (branch-free, static shape, `lax.fori_loop`), whose
  tangent is zero; the implicit derivative `dx/dQ = -(2/3) pi^(2/3) / (Q^2
  f'(x))` is attached as a `custom_jvp`, which is what libxc's Maple
  `diff/br89_x` is. Without it the Newton-Krylov solver's Jacobian is missing
  the entire `d v / d tau` block and the SCF residual's derivative is wrong in a
  way that still converges.
- **Everything here is a ratio to a power of the density**, so the vacuum is not
  merely inaccurate but `0/0`, and one NaN poisons the SCF. The gate is the
  *GGA* threshold (1e-6) and not the LDA one (1e-10), with the masked points
  evaluated on substitute values so that `grad` sees no singular arithmetic
  through the `where`.
- **QE's own meta branch is inconsistent about the core charge.** `v_xc_meta`
  builds `grho` from `rhog_core + rho%of_g` and then passes `rho%of_r` — valence
  only — as the density. `setup.f90` prints "BEWARE: nonlinear core correction is
  not consistent with meta-GGA" and leaves it there. Here the core is folded into
  both, as the GGA path folds it, and the asymmetry is not reproduced.

**What is refused as this phase lands.** `PW/src/setup.f90` raises
`'Meta-GGA not implemented with USPP/PAW'` and `'Non-collinear Meta-GGA not
implemented'`, and both are refused here too, for the reasons stated in
`Calculation._require_meta_supported`: `tau` from the pseudo-states is not the
all-electron kinetic energy density and there is no `addustau` to write against;
and with spinors `tau` is a 2x2 matrix in spin space rather than two scalars,
whose local spin frame `gradcorr` supplies for a gradient and nothing supplies
for a Laplacian. Spin spirals are refused because the two spinor components live
on different plane-wave spheres, so their gradients do not add to a
lattice-periodic `tau`. A Hubbard `U` is refused as unvalidated, and because
`_solve_residual`'s convergence measure reads `ns` off the *end* of the packed
state, which `tau` now occupies.

**Two of those four did not survive contact with a real target.** P31 lifts the
noncollinear and spin-orbit refusal and P32 the PAW one, so what remains refused
here is plain ultrasoft, spin spirals and DFT+U. The wording above is kept as it
stood because the reasons it gives are the work those phases had to do.

**The numbers.** Silicon, `Si.pz-vbc.UPF`, `ecutwfc = 30`, a 6x6x6 grid reduced
to its wedge, gaps from a band path `L-Γ-X-W-K-Γ` at 30 points a segment:

| run | `c` | indirect gap | direct gap | SCF iterations |
|---|---|---|---|---|
| LDA (PZ) | — | 0.493 eV | 2.567 eV | 6 |
| BJ06 — *what `pw.x` runs when asked for `tb09`* | 1.000 | 1.018 eV | 3.075 eV | 11 |
| **TB09** | **1.0331** | **1.134 eV** | **3.168 eV** | 10 |
| TB09, `mbj_c = 1.12` | 1.120 | 1.455 eV | 3.429 eV | 21 |
| TB09, `mbj_c = 1.20` | 1.200 | 1.776 eV | 3.696 eV | 23 |
| TB09, `mbj_c = 1.30` | 1.300 | 2.215 eV | 4.067 eV | 24 |
| experiment | | 1.17 eV | 3.40 eV | |
| published mBJ (WIEN2k, all-electron, `c` = 1.12) | | 1.17 eV | | |

and diamond, `C.pbe-hgh.UPF`, `ecutwfc = 60`, the same grid and path:

| run | `c` | indirect gap | direct gap | SCF iterations |
|---|---|---|---|---|
| LDA (PZ) | — | 3.890 eV | 5.556 eV | 5 |
| PBE | — | 4.112 eV | 5.625 eV | 6 |
| **TB09** | **1.1777** | **4.428 eV** | **6.453 eV** | 9 |
| TB09, `mbj_c = 1.20` | 1.200 | 4.497 eV | 6.517 eV | 18 |
| experiment | | 5.48 eV | ~7.3 eV | |
| published mBJ (WIEN2k) | | 4.93 eV | | |

Diamond's cutoff was checked rather than assumed: at `ecutwfc = 90` instead of
60 the LDA gap moves 3.890 -> 3.923 eV, TB09's 4.428 -> 4.417 and `c` 1.1777 ->
1.1748, so the basis is converged to about 0.03 eV and the 0.5 eV shortfall
against the all-electron mBJ is not it.

**Silicon lands on the published mBJ number and diamond falls 0.5 eV short,
and `c` explains neither on its own.** Tran and Blaha's `c` averages
`|grad rho|/rho` over the cell, and that ratio is largest *in the core*, which
is exactly what a pseudopotential removes: norm-conserving silicon measures
`c = 1.033` where the all-electron calculation measures 1.12. The gap
nonetheless comes out right, because the density is not the all-electron one
either and the two departures are not independent — which is why imposing the
all-electron `c` on a pseudopotential density is **not** a correction: at
`mbj_c = 1.12` the same cell overshoots to 1.455 eV, and the sensitivity is
steep (2.215 eV at `c = 1.30`). Diamond's `c` is measured at 1.178 and its gap
is still 0.5 eV under the all-electron mBJ at any `c` near it (4.497 eV at
1.20), and the basis is not the cause either, so what is missing there is the
core the pseudopotential removed from `tau` and from the Laplacian as well as
from `c`. **The reproducible statement is the shift**: +0.64 eV on silicon and
+0.54 eV on diamond, against published all-electron shifts of +0.67 and +0.82.
An all-electron `c` is available as `mbj_c` for anyone who wants to test that
reading, and the phase does not claim to have settled it.

**What symmetrising `tau` is worth**, measured by running the same cell with
`symmetrize = False`: the unsymmetrised `tau` is 11% asymmetric relative to its
own maximum, `c` moves by 1.1e-3, the total energy by 1.3e-2 Ry and the
eigenvalues by up to **0.47 eV** — at `ecutwfc = 12` and at 30 alike, and with
the iteration count unchanged, so nothing about the convergence would have said
anything was wrong.

**Does a gradient-based route converge it more easily? No, and the reason is
worth stating.** The comparison, on silicon at 30 Ry, in *evaluations of `F`* —
one diagonalisation each, which is the only currency in which a mixer's
iteration count and a Krylov solver's step count are the same thing:

| | LDA | TB09 |
|---|---|---|
| Anderson mixing, `beta = 0.7` | 6 | 11 |
| Anderson mixing, `beta = 0.3` | 7 | 19 |
| Newton-Krylov on the residual (P22) | 40 | 75 |
| Newton-Krylov after 3 mixing steps | 17 | 59 |

Mixing wins by a factor of six, and that is not a defect of the residual solver.
**Anderson mixing already is a quasi-Newton method on this residual** — it fits
the Jacobian from the iteration history for nothing — and TB09's fixed point is
not ill-enough conditioned for an *exact* Jacobian to be worth what a step
costs — 5 Newton steps here spent 52 evaluations of `F` and 23 Jacobian-vector
products, so about fifteen apiece. What P22 established remains the place that route
earns its cost: a problem with more than one solution, where the mixer flows to
the stable one. TB09 is not such a problem; it is merely a slower one, by a
factor of about **1.8 in iterations** over LDA, growing with `c` (23 iterations
at `c = 1.20`, against 10 at the self-consistent 1.033) because a larger `c`
strengthens the coupling between `tau` and the potential.

Making the comparison possible at all needed one change, and it is the phase's
one deliberate deviation from QE: **`tau` joins the packed state**. The mixing
loop may lag it — a loop is allowed to depend on whatever it likes — but a
root-finder needs `F` to be a function of its argument, so the fixed point is
sought in `(rho, tau)` jointly. That is what puts the `d v / d tau` block into
the Jacobian, and that block runs through the implicit derivative of the
Becke-Roussel inversion, which is why the `custom_jvp` is not optional.

*Notebook 24.*


### P31 — The Tran-Blaha potential with spin-orbit coupling. ✅ DONE.

`spinor_band_kinetic_density` and `spinor_kinetic_energy_density` in
`pypresso/scf/density.py`, `_noncollinear_meta_exchange` in
`pypresso/scf/potential.py`, and the noncollinear branch of
`Calculation.kinetic_energy_density`. `noncolin = .true.` and `lspinorb = .true.`
with `input_dft = 'tb09'`.

**`pw.x` stops here and this does not.** `PW/src/setup.f90` raises
`'Non-collinear Meta-GGA not implemented'` and returns; the refusal P30 wrote for
the same combination gave the reason, and the reason turned out to be a
description of the work rather than an obstacle to it.

**The kinetic energy density of a spinor is a 2x2 matrix.** Not a number and not
two numbers:

    tau_ab(r) = sum_i w_i grad psi_ia^* . grad psi_ib,

which decomposes on the Pauli basis exactly as the density does — a trace and an
axial three-vector, with the same `nspin_mag` deciding whether the vector part
exists at all. So `spinor_band_kinetic_density` *is* `spinor_band_density` with
`grad psi` in place of `psi`, and the one thing that is not a substitution is
that every product becomes a dot product over the three cartesian directions
**before** the spin algebra. Taking the spin structure of each direction and
summing afterwards gives the same trace and a different vector part.

**Everything after that is the local spin frame, which was already written
twice.** `_noncollinear_meta_exchange` is `_noncollinear_gradient_correction`
with a second field along for the ride: rotate `(n, m)` onto `m-hat` with
`fixed_quantization_axis`'s sign, project `tau` onto the *same* axis, run the
collinear functional, attach the splitting back to `m-hat`. Two things are worth
saying out loud:

- **The axis is the density's, not `tau`'s.** They are not parallel in general —
  `tau_vec` is the Pauli expectation of a *gradient*, and nothing makes it
  collinear with the magnetization — so `tau_vec . m-hat` is a real projection
  and its transverse part is discarded. That is not an approximation invented
  here: it is what "evaluate the collinear functional in the local frame" means,
  and the LSDA and GGA branches discard the same transverse information. It is
  stated because for `tau` it is easier to miss than for `m`.
- **The rotated channels' gradient *and Laplacian* need their own transform.**
  The rotation runs through `|m|` and is not linear in the components. That trap
  is `gradcorr`'s, and the Laplacian inherits it unchanged.

**Validated as algebra, not as agreement.** Three identities, each of which a
plausible sign error breaks:

| check | result |
|---|---|
| two spinor bands `(psi, 0)`, `(0, psi)` at half weight give the scalar `tau` | **3e-17** |
| a magnetization along `z` reproduces the collinear branch's `v_0` and `v_z` | **1.8e-15**, transverse identically 0 |
| turning `m` to an arbitrary axis turns `v` with it | **2e-14**, transverse 2e-16 |

and end to end, a spin-orbit silicon reproduces the Kramers-doubled scalar
eigenvalues to **5e-5 eV** rather than to machine precision. That gap is
measured, not excused: `tau` weights the wavefunction by `|k+G|^2`, so it
amplifies whatever the two eigensolver paths leave differing, and mBJ is
nonlinear in `tau` on top. The same comparison under PZ agrees to 3.6e-12, and
the algebraic test above is what says the builders agree exactly while the two
*SCFs* do not quite.

*Notebook 24.*

### P32 — The Tran-Blaha potential on PAW spheres. ✅ DONE.

`_kinetic_tensor`, `_radial_laplacian` and `_meta_exchange_onecenter` in
`pypresso/paw/onecenter.py`, `PawSpecies.kinetic_ae`/`kinetic_ps`, and `becsum`
threaded through `fixed_density_states`/`run_bands`/`run_nscf`/`run_dos`.

**The obstacle was supposed to be the coefficients and it was not.** A
potential-only functional has no `dE/dbecsum`, so what the Hamiltonian must
receive from each sphere is the *matrix element*
`<phi_i|v|phi_j> - <phi~_i|v~|phi~_j>` — and that is the contraction
`onecenter_species` already performs. For a local functional `ddd = dE/dbecsum`
happens to equal that matrix element, because `rho_lm` is linear in `becsum`;
here only the second reading survives, and the same line of code implements it.
Nothing had to be added, which is the whole reason PAW is reachable for this
functional at all.

**What did have to be written is `tau` inside the sphere.** Its two halves do
*not* share an angular structure:

    grad phi_i . grad phi_j = R'_i R'_j Y_i Y_j
                            + (R_i R_j / r^2) (grad_Omega Y_i . grad_Omega Y_j),

with `R_i = u_i / r`. The first term expands on exactly the multipoles the
*density* does — it is `Y_i Y_j` again — so it reuses the same Clebsch-Gordan
table. The second does not, and its expansion is computed once per species by
quadrature. Both fold into one `(nh, nh, nlm, mesh)` tensor holding
`r^2 tau_lm`, so `becsum -> tau_lm` is the same einsum `becsum -> rho_lm` is, and
`_radial_laplacian` is the third ingredient: diagonal in `lm`, two radial
derivatives and a `-l(l+1)/r^2`, with no QE counterpart because nothing QE
evaluates on a sphere asks for one.

**Which angular table carries the `1/sin(theta)` is not inferable from the
variable names**, and the module docstring of `pypresso/paw/gradient.py` says
one thing while the code does another. Settled by the exact identity
`int |grad_Omega Y_lm|^2 dOmega = l(l+1)`: `dylmt^2 + dylmp^2` reproduces it to
every digit on two grid sizes and `dylmt^2 + dylmp^2/sin^2` does not (4.58
against 2 for `Y_1,+-1`). The lone `divide(sin_theta)` in the gradient
correction belongs to the *divergence*'s input convention, not to the modulus.

**The finding: a UPF has no core kinetic energy density, and pretending
otherwise inverts the functional.** Every other one-centre term here sees
`rho_valence + rho_core`. This one cannot: the format carries a core *charge*
(`PP_AE_NLCC`) and nothing for `tau`, QE has no `tau_core` anywhere, and inside a
sphere the all-electron core dominates `rho` and `lap rho` while contributing
nothing to `tau` — so `2 tau - |grad rho|^2/4 rho` and `sqrt(2 tau/rho)` are both
evaluated on mismatched halves. The symptom is not noise, it is the wrong sign of
`d(gap)/dc`:

| `mbj_c` | core in the sphere term | core left out |
|---|---|---|
| 1.00 | 1.249 eV | 0.917 eV |
| 1.10 | 0.934 eV | 1.260 eV |
| 1.28 | 0.348 eV | 1.991 eV |

Every norm-conserving cell has the gap *rise* with `c` (P30). So the one-centre
term sees the valence density alone on both sides, and the frozen core reaches
it only through the shape of the all-electron partial waves. (VASP's meta-GGA
PAW datasets tabulate the core kinetic energy density for exactly this reason;
no UPF has it to read.)

**What PAW buys, in one number.** Tran and Blaha's `c` averages `|grad rho|/rho`
over the cell, and that ratio is largest in the core — which a norm-conserving
pseudopotential removes and a PAW augmentation charge puts back:

| silicon, same cell and grid | `c` | LDA gap | TB09 gap |
|---|---|---|---|
| norm-conserving `Si.pz-vbc` | 1.000 | 0.645 eV | 1.163 eV |
| **PAW** `Si.pz-n-kjpaw` | **1.107** | 0.589 eV | **1.285 eV** |
| all-electron (published) | 1.12 | | 1.17 (experiment) |

P30 explained the norm-conserving shortfall by the pseudised core and could not
test it. This is the test: put the core back and `c` moves to within 0.013 of
the all-electron value.

**`c` is passed down, never recomputed.** It is an average over the *cell*, so a
sphere that computed its own would give each atom a different functional;
`Calculation.onecenter` takes it from `Potential.meta_c` and refuses to default
it, because using 1 on the spheres while the grid used 1.03 is a wrong gap with
no other symptom.

**A PAW band structure works now, and that was a two-line change.** `nscf`
refused PAW outright because `ddd_paw` cannot be rebuilt from a density;
`SCFResult.becsum` has carried it since P12, so passing it is the fix and the
refusal now only catches *not* passing it. Still refused: plain **ultrasoft**,
which has the augmentation charge but no partial waves to reconstruct `tau`
from.

### P33 — PAW's one-centre gradient correction, noncollinear. ✅ DONE.

`_noncollinear_gradient` in `pypresso/paw/gradient.py`, and the quantization axis
threaded from `Calculation` down to `onecenter_species`.

P12 refused this and named the reason precisely: `PAW_gcxc_potential` needs the
local-frame rotation done on the radial sphere, `compute_rho_spin_lm`, "and that
is a second implementation rather than a call into the first". It is not — it is
the same three steps `_noncollinear_gradient_correction` takes on the plane-wave
grid, with `PAW_rad2lm` where the grid version takes an FFT:

1. `rho_up/dw = (n +- s|m|)/2` at every (direction, radius), `s = sign(m . ux)`
   and `+1` where `|m|` vanishes — QE's `segni_rad`. The frozen core is
   unpolarized, so it goes wholly into the charge before the split and half
   lands in each channel.
2. **Project the rotated channels back to multipoles afresh.** This is what the
   refusal was really about: the rotation runs through `|m|`, so no combination
   of the stored `rho_lm` is the expansion of the result, and the angular part of
   the gradient and the divergence both read those multipoles.
3. Rotate back on the radial grid, where `m-hat` lives — `compute_pot_nonc`.

**Validated against the collinear branch as algebra.** Fed `(n, 0, 0, m)` and
`(up, down)` — the same physical state — it returns a **bit-identical energy**, a
potential agreeing to **5.5e-11 relative** (the residue is the multipole
round-trip the noncollinear branch does and the collinear one does not), and
transverse components that are exactly zero. End to end, a magnetic oxygen atom
polarises to 2 mu_B either way and the two totals agree to 2.8e-6 Ry — which is
*not* this branch's error, since the same comparison under LDA, which never
enters it, differs by 3.1e-6.

**Not reproduced: `add_small_mag`.** A fully-relativistic dataset's small
component carries magnetization of its own and QE folds it in here. The *local*
part of this package's one-centre XC does not fold it in either, so leaving it
out of both keeps them consistent; putting it in one and not the other would be
worse than in neither.

### P34 — Running on a cluster: a submit/fetch harness for sweeps. 📋 PLANNED.

`tools/cluster/`. Not started. This entry is the design, written down before the
work and reviewed before being written down, so that the session which picks it
up does not re-derive it — and does not repeat the two mistakes the review
caught.

**The cluster's rules are not this project's to set, and they are recorded outside this
repository.** A shared HPC system's access policy, account paths and site-specific limits
belong in the private notes beside this checkout rather than in a public tracker; read
those before implementing any of what follows, because they fix several things this entry
leaves open — which login host, where output may be written, how often the queue may be
polled, and which operations are proposed to the user rather than run.

Two of their rules sharpen items below rather than adding to them. A small-files policy
applies to "every case writes JSON plus `.npz`": one pair per *case* is what is meant, and
one pair per SCF iteration is what must not happen. And the thread pinning below is the
same rule such systems state for BLAS — pin XLA to the CPUs the job asked for, and set
`MKL_NUM_THREADS=1 OMP_NUM_THREADS=1` for anything that does not genuinely use more.

**Design, kept thin on purpose.**

- `tools/cluster/env/` — builds the environment on a login node (JAX 0.11, NumPy
  2.4, equinox, Numba, x64). This is the real cost of the phase and is done
  **interactively, not through the queue**: iterating on a broken environment one
  queue round-trip at a time is the slowest possible way to do it.
- A **case** is a directory: a `pw.x`-style input plus a small JSON of run
  options (functional, cutoffs, k-grid, `nbnd`, whether to follow with a band
  path).
- `submit.py` — cases to one array job. Carries `--dry-run`, which prints the
  sbatch script and submits nothing, and **enforces the sweep-size cap itself**
  rather than leaving it to convention.
- `fetch.py` — sentinel-gated rsync back, **including stdout and stderr on
  failure**, which is the difference between debugging locally and debugging
  over SSH.
- `scancel` tooling, because a wrong sweep will be launched eventually.
- **Every case writes JSON plus `.npz`** — scalars (energies, gaps,
  coefficients, iteration counts, timings) and arrays (eigenvalues, k-point
  coordinates) — with the provenance triple above. Nothing should ever have to be
  re-run to re-derive a number, and that rule has already paid for itself: the
  first NiI2 script computed its gap at the wrong band index, and because the
  eigenvalues were on disk the fix cost a re-read rather than three hours.
- **Analysis and plotting stay local.** The cluster runs pypresso and nothing
  else.

**Build the thin path first.** Polling and fetch machinery written blind against
a cluster nobody has touched will mostly be wrong. One case, submitted by hand
through the harness and fetched back, settles the environment, the sentinel and
the thread pinning; the array job and the sweep cap come after first contact.

**Two things are the user's and not the agent's.** Interactive authentication —
`kinit`, 2FA — cannot come from a tool call; key-only SSH can. And a cluster
allocation is a shared, accounted resource: key access is not standing permission
to spend it, so the first submission and any large sweep are confirmed rather
than assumed. (The agent's shell is also sandboxed by default, and SSH needs that
relaxed explicitly, as the pseudopotential downloads of P30 did.)


### P35 — Raman tensors: two fields and a displacement. ✅ DONE.

`pypresso/response/nonlinear.py`, `raman_tensors`. The third derivative of the
energy with respect to **two electric fields and one atomic displacement** —
`d(eps)/d(tau)`, which is what a non-resonant Raman intensity is computed from.
It is P26's construction with one substitution: the *same* variational
second-order energy `F_ij`, differentiated once more at frozen first-order
wavefunctions (the 2n+1 theorem), along the atomic positions where P26 went
along a strain.

    d(eps_ij)/d(tau_c) = jvp(F_ij)( tau, psi, rho, b ; e_c, dpsi_c, drho_c, db_c )

**Every tangent already existed**, which is the point of the phase: `dpsi_c` and
`drho_c` are the displacement response P25 solves for the dynamical matrix, `b`
and `u` are the field response P24 hands back through `keep_internals`, and
`db_c` is the further Sternheimer solve P26 wrote — reached here through the
same function, with `at_positions` as its geometry variable instead of
`at_strain`. The only edit outside the new module is that
`_position_response` and `_require_a_closed_grid` take what moves as an
argument. **The whole phase is an assembly**, and P26's machinery went from one
consumer to two, which is what makes the second one worth having: machinery with
one consumer is unproven.

*Check met*, on AlAs (zincblende, LDA norm-conserving, `ecutwfc = 10`, the
unshifted 4x4x4 grid run whole under `nosym`):

| quantity | pypresso | reference | |
|---|---|---|---|
| `d(eps_yz)/d(tau_(Al,x))` | **-3.118279** | -3.118310 | central difference of `eps` over re-converged displaced cells, **1.0e-5** relative |
| `d(eps_yz)/d(tau_(As,x))` | **+3.119166** | +3.119194 | the same, 9e-6 |
| translational sum rule | **8.9e-4** | 0 | 2.8e-4 of the tensor; `ph.x` gives 1.11 (43%) |
| zincblende form | all nine forbidden entries **< 1e-13** | 0 | nothing imposes it — `nosym`, no average |
| `eps` | 12.9674206 | 12.9673215 (`ph.x`) | P24's own agreement, unchanged |

**The reference for this phase is broken, and establishing that had to come
first.** QE reaches the same tensor through `ph.x` with `lraman = .true.`, and
the vendored 7.5 build **does not reproduce its own committed example**. On
`PHonon/examples/example05`'s own input it gives a Raman tensor of **-1.8681**
where the reference (generated with v6.0 in 2016) says **-0.78497**, and an
electro-optic tensor of **157.87** against **40.4578**. Its *own* internal
consistency check fails too: `dhdrhopsi` obtains the k-derivative of the
wavefunctions by finite differences and prints the dielectric constant they
imply beside the analytic one, and where the v6.0 reference has 8.8116 against
8.8147, the vendored build gives **-0.288** against 8.8143. Tightening
`eth_rps` and `eth_ns` by four orders moves it by 1e-2, so it is not a
threshold. Reading the two literature values the same way — the v6.0 number is
consistent with Veithen, Gonze and Ghosez's ABINIT table for AlAs and the 7.5
one is four times it — says which of the two is the regression.

The obvious objection is answered rather than left: **the pseudopotentials are not
the same files.** The MD5s the example's own output prints (`614279c8…`,
`451cd336…`) are not the committed ones here (`f06ceae8…`, `2c53d869…`) — they
are the same two pseudopotentials from a different distribution. That cannot be
the explanation, for two reasons. `eps` agrees to **4e-4** between the two runs
(8.8147 against 8.8143), so the datasets are equivalent everywhere the ground
state and the linear response can see. And the decisive evidence is *internal to
a single run*: `dhdrhopsi`'s own finite-difference dielectric constant against
the analytic one built from the same wavefunctions, which no choice of
pseudopotential enters.

So the validation is **a finite difference of the dielectric tensor over
re-converged displaced geometries**, which shares nothing with the third
derivative but the linear response underneath both — the route P26 already used
for `d(chi)/d(strain)`, and available here because `eps` is a quantity this code
computes from scratch at any geometry.

**What `pw.x` refuses and this does not.** `phq_readin.f90` and `phq_setup.f90`
refuse, by name: PAW, ultrasoft, noncollinear magnetism, Hubbard `U`, `lsda`,
metals, `q != 0` — and

    IF (xclib_dft_is('gradient').and.(lraman.or.elop)) call errore('phq_setup', &
       'third order derivatives not implemented with GGA', 1)

because `PHonon/PH/d2mxc.f90` is the third derivative of `E_xc` hand-coded as a
Perdew-Zunger parameterisation and nothing else. Here that object is never
written: `dv_of_drho` is one `jvp` of `v_of_rho` (P24), the screening term of
`F` contracts two density responses against it, and differentiating `F` a third
time differentiates *that*. `delta^3 E_Hxc/delta n^3` is whatever the loaded
functional's is, and no third derivative is transcribed. **This is the phase
where D1 pays the most**: the ratio of what had to be written to what came out
is the smallest anywhere in this code.

**`chi^(2)` and the electro-optic tensor are refused by name, and the missing
term is identified rather than fitted.** They are the same functional
differentiated along a *third field*, every tangent for it exists, and the
result is wrong — because **the field enters this code only through the source
term** `b = P_c r|psi>` and through the density. `H` is built from `rho` and
carries no field at all, so the term of the 2n+1 expression in which the
perturbing operator sits between two first-order wavefunctions (`<u_i|r_k|u_j>`,
which QE builds by going to *second*-order response in `dvpsi_e2`/`solve_e2`)
has nothing here to build it from: the position operator exists only as
`P_c r|psi>`, through a commutator solve that uses `psi`'s own eigenvalue and
does not apply to a general first-order state.

**How large that term is, is a measurement rather than an estimate**, because
its displacement counterpart *is* computed here. Zeroing the geometry tangent in
the Raman derivative — which puts it in exactly the position the field
derivative is in — moves `d(eps_yz)/d(tau)` from **-3.118279** to
**-1.809983**: the explicit `dH/d(parameter)` term is **42% of the answer**.

**And no symmetry check catches its absence**, which is the finding worth
carrying forward. Without that term the field tensor still

* vanishes identically in a centrosymmetric crystal — silicon gives **1.2e-13**;
* comes out in the exact zincblende form on AlAs, eight zeros per block at
  1e-13, with nothing imposing it;
* is symmetric under **every permutation of its three labels to 2.5e-13**,

because the omitted term has all three properties itself. Kleinman's condition
is the check this phase expected to be decisive and it is not: `F_ij` is
symmetric in its own two labels by construction, so what the permutation test
measures is real but blind to anything symmetric. The tensor is roughly an order of
magnitude below the published `chi^(2)` of AlAs and moves *away* from it as the
k-grid is refined, and every symmetry statement about it is exact —
which is the same shape of trap as P28a's, one order up. It is kept in the code
(`susceptibility_field_derivative`) precisely so that the tests can measure it,
and refused at the entry point.

**Two backlog items, and the first is the phase after this one.** The
`<u_i|r_k|u_j>` term needs the second-order response `solve_e2` is; with it,
`chi^(2)`, the electro-optic tensor and — with the Raman tensors and P25's modes
and P24's `Z*`, all of which exist — the full Pockels tensor are an assembly.
**Grüneisen parameters** are the cheapest thing adjacent to this and were not
attempted: one more `jvp` of the dynamical matrix along the strain tangent P26 already
builds, checked by finite-differencing phonon frequencies over strained cells. And a
**rank-3 symmetriser** (`symme.f90`'s `symtensor3` and `symmatrix3`) would lift the
closed-grid refusal that P26 introduced and this phase inherits: two
field labels and an atom make a wedge sum incomplete, and an unshifted grid run
whole is the escape both phases take.

**`NONLINEAR.md`** is the roadmap for everything above third order that this phase
makes reachable — what each item costs, what it is blocked on, and the three
constraints (the field's absence from `H`, the broken reference, and the k-convergence
of a third derivative) that shape the order they should be taken in.

*Notebook 25.*

---

### P36 — Raman and infrared spectra, and the rank-3 symmetriser. ✅ DONE.

Two things, and the second one is why the first is cheap.

**The symmetriser first.** `pypresso/system/symmetry.py`,
`symmetrize_cartesian_tensor` and `symmetrize_atom_cartesian_tensor` —
`symme.f90`'s `symmatrix3` and `symtensor3`, written **at any rank** rather than
at rank 3, because Fortran wrote the rank-2 case out in four nested loops and
the rank-3 case again in six, and P26's object has **four** cartesian labels and
so has no QE counterpart at all. `symmetrize_matrix` and
`symmetrize_atom_tensor` delegate to them and keep their own names, which are
the ones `symme.f90` uses.

It lifts the closed-grid refusal P26 introduced and P35 inherited. A wedge sum
is exact for a scalar and for nothing else, so an object with three or four free
cartesian indices needs the group average afterwards; until this it was refused
by name and both phases escaped by running an unshifted grid whole.

*Check met*, and both are the same construction one rank apart:

| | wedge | closed | agreement | cost |
|---|---|---|---|---|
| P35 `d(eps)/d(tau)`, AlAs, rank 3 + atom | 8 k-points | 64 | **3.3e-9** relative (8.7e-14 at the time; see below) | 51 s against 112 |
| P26 `d(eps)/d(strain)`, silicon, rank 4 | 8 k-points | 64 | **7.9e-14** relative | 73 s against 205 |

**The average alone is not enough, and the reason is the finding of this phase.**
Applied to P35's assembled tensor it left AlAs at **-3.195188** against the
closed grid's -3.118279 — 2.5% — with the translational sum rule at 1.0e-2
where the closed grid gives 2.8e-4. The average completes a wedge sum only when
the tensor is a **linear** k-sum of a covariant per-k quantity, since then
`T_true = (1/N) sum_S R⊗R⊗R T_wedge` follows from `t(Sk) = R⊗R⊗R t(k)` term by
term. Every piece of `F_ij` is such a sum except **the screening term**, which
is `int drho_i K drho_j` — *quadratic* in one. A product of two incomplete sums
is not the incomplete version of the product.

What repairs it is a **split between the value and the derivative**: the value
of each density-response factor must be the full-zone object and its derivative
must stay the raw wedge sum. Then for a covariant `X` and a raw `Y`,

    (1/N) sum_S R R R [ int X_i K Y_jc ] = int X_i K Y^true_jc,

by changing variables in the integral and using `X`'s own covariance — so the
assembled tensor's average completes this term with the rest. In code that is
three lines around `jax.lax.stop_gradient` in `_second_order_energy_at`, and on
a closed grid it is a bit-for-bit no-op (the closed-grid numbers are unchanged:
-3.118279, +3.119166, 8.878e-4).

**Symmetrising the derivative too is the plausible wrong answer** and it is
worse than doing nothing: it puts an *extra*, independent group average on the
displacement label, giving `d(eps_yz)/d(tau_As)` = **3.009778** against 3.119166
and a sum-rule residue **115x** the closed grid's, where applying no average at all costs
37x. Measured, relative to the tensors' own scale: **2.8e-4** right, **1.0e-2** with no
average, **3.3e-2** with the wrong one. The **sum rule is what caught
both**, which is P35's lesson in a second place: the tensor stayed exactly
zincblende, exactly permutation-symmetric and exactly cubic through all three
variants.

**One refusal is *not* lifted and it is a different one.** The clamped-ion
elastic constants still need the whole grid, because `elastic_constants` has to
let the energy functional build its own density — so that its gradient is the
stress and not a partial derivative at fixed `rho` — and that functional
symmetrises the density it builds as a **scalar**. No average of the assembled
tensor undoes one applied inside the chain rule. `electrostriction(elastic=False)`
is the wedge route and the refusal says so.

**Then the spectra**, `pypresso/response/spectra.py`. P35's Raman tensors are
per *atom* and what an experiment resolves is a *mode*, so this is the
contraction with the phonon eigendisplacement — `R^(nu) = sum_(a,c) dchi/dtau z`,
`p^(nu) = sum_(a,c) Z* z` — followed by Placzek's two rotational invariants
and the standard `45 alpha^2 + 7 beta^2` and `3 beta^2/(45 alpha^2 + 4 beta^2)`.
Every ingredient existed: P35's tensors, P25's modes, P24b's Born charges.

**This phase has a working reference, which P35 did not.** QE reaches the same
table through `dynmat.x`, whose `RamanIR` (`LR_Modules/dynmat_sub.f90`) is *pure
post-processing* — it reads `dchi_dtau`, `zstar` and `eps0` off a file and
contracts them — and shares nothing with the `lraman` branch P35 established has
regressed. `pypresso/io/dynmat.py` writes the file `ph.x` would have written and
the test runs the vendored binary on it.

*Check met*, against the vendored `dynmat.x` on tensors this code computed:

| | pypresso | `dynmat.x` |
|---|---|---|
| AlAs optical triplet | 353.25 cm⁻¹, IR 5.9262, Raman **446.8854**, depol 0.7500 | every digit |
| silicon `T_2g` | 519.20 cm⁻¹, IR **0.0000**, Raman **9815.5635**, depol 0.7500 | every digit |
| AlAs polarizability | 41.722971 Å³, `cmfac` 0.200435 | every digit |

Silicon's **519.2 cm⁻¹** is against an experimental 520, and the structure of
its table is the check that needs no reference at all: one Raman-active triplet,
and **no infrared activity whatever**, because an operation carries one silicon
onto the other and so gives them the same `Z*` — the optical mode moves them
against each other and has no dipole. That is why silicon is transparent in the
infrared, and it holds at any cutoff.

**A degenerate multiplet is comparable only as a sum, and running the test is
what demonstrated it.** `alpha`, `beta^2` and the depolarisation ratio of a
single mode are not invariant under the orthogonal mixing the eigensolver is
free to apply inside a multiplet (rule D4); the multiplet's *sum* is, both
invariants being quadratic in `R`. On silicon's acoustic triplet the two
eigensolvers land in different bases and print
**0.3544/0.7163/0.4065** here against **0.5873/0.2446/0.7264** from `dynmat.x`
— on modes whose Raman activity *both* codes print as 0.0000, so the ratio is
`0/0` amplified out of a sum-rule residue. The frequencies and the summed
activities agree exactly. `VibrationalSpectrum.by_manifold` is the comparable
form; AlAs's triplets happen to be per-mode invariant too and that is an
accident of zincblende, not a licence.

**The displacement response is solved once.** A Raman tensor and a dynamical
matrix need the *same* `solve_linter` output and it is the dominant cost of
both, so `raman_tensors(keep_internals=True)` hands back a `DisplacementResponse`
and `dynamical_matrix(response=...)` takes it: the phonons then cost **1-2 s**
against 50. It is an optimisation, so the test is an equality against a matrix
built from a solve of its own (1e-12).

**A new polar `Z*`.** Every Born-charge case here was silicon or carbon, where
the answer is a residue near 0.05 and agreeing is a statement about precision.
AlAs's are **1.924598** and **-3.181161** against the vendored `ph.x`'s 1.92461
and -3.18098 on the same input — and they do **not** sum to zero. At
`ecutwfc = 10` charge neutrality is violated by -1.256; `ph.x` reports the same
violation and prints an ASR-corrected ±2.5528 beside it. Reproducing the
uncorrected pair *including* its violation is the check.

**Not implemented, and named rather than approximated:** the **non-analytic**
LO-TO term (`rigid.f90`'s `nonanal`), so an optical triplet comes out unsplit
and the `dynmat.x` comparison is run with its `q` at zero for the same reason;
and the mode-resolved ionic permittivity (`polar_mode_permittivity`). Both are
arithmetic of the same kind as this module's and both have their ingredients
here already.

*Notebook 26.*

### P37 — The bootstrap kernel: excitons from TDDFT. ✅ DONE.

`pypresso/tddft/`, `pypresso/workflows/tddft.py`, `tests/regression/test_tddft.py`,
`tests/unit/test_tddft_machinery.py`. The design below was written and reviewed
twice **before** the work; what the work then found is at the end of the entry,
and three of its four findings share one shape — they produce a spectrum that is
smooth, positive, has the right peaks, and is wrong.

**What it computes.** The macroscopic dielectric function `eps_M(w)` of an
insulator in the optical limit, from the Dyson equation of TDDFT with an
exchange-correlation kernel that is *not* zero — so that `Im eps_M(w)` carries
the bound electron-hole pair that RPA and ALDA cannot produce. The kernel is
Sharma, Dewhurst, Sanna and Gross's **bootstrap** (PRL **107**, 186401 (2011);
arXiv:1107.0199), which is Elk's `fxctype = 210`, and its single-iteration form
`211`. There is no `pw.x` counterpart and no QE counterpart at all: `TDDFPT/` is
a Liouville-Lanczos solver with ALDA/RPA and has no bootstrap kernel and no
Dyson-in-G-space route (`grep -ril bootstrap` over the vendored tree hits only
`Modules/mp_pools.f90`, unrelated). **The reference is Elk**, whose source and a
built binary are outside this repository (see the private notes).

**The kernel is one algebraic line and the phase is a transcription. Say so.**

    f_xc^BS(q, w) = - eps^-1(q, 0) v(q) / (eps_0(q, 0) - 1),   eps_0 = 1 - v chi_0

with `eps^-1` determined *self-consistently* against the Dyson equation itself:
set `f_xc = 0`, solve for `eps^-1`, rebuild `f_xc`, repeat. `genvfxc.f90` is that
in symmetrised form — with `X = v^1/2 chi_0 v^1/2` and `F~ = v^-1/2 f_xc v^-1/2`,
it is `F~ = eps^-1(w=0) / X_00`, and `211` is the same loop stopped after one
pass. There is nothing here to differentiate, and inventing an autodiff story for
it would be dishonest. **Two genuine ones exist and are the phase's real content**
— see the head/wings and the kernel registry below.

**The expensive part is the object this code does not have.** `chi_0(G, G', w)`
as a *matrix* over a response G-set and a frequency axis. Everything in
`pypresso/response/` is Sternheimer, static, and gives `chi_0` only as an
operator `drho = chi_0 dV`. So P37's weight is `chi_0`, not the kernel.

**Adler-Wiser, and the hybrid is a trap.** `chi_0` is built by sum over states
(`genvchi0.f90`), from an NSCF with empty bands — not by a column-per-`G'`
Sternheimer solve. The temptation is real, because the *kernel* needs only
`w = 0` where Sternheimer lives: resist it. `tddftlr.f90` feeds `genvfxc` the
`w = 0` **slice of the same array** the Dyson equation then inverts at every `w`,
so the kernel and the spectrum carry one set of convergence errors — the same
band truncation, the same `swidth`, the same response cutoff. A Sternheimer
kernel would be band-complete and `eta = 0` against a truncated, broadened
spectrum, the `210` self-consistency would couple the two, and **no diagnostic
here can see that inconsistency**. One builder; the Sternheimer stack is the
referee instead (validation, below), which is a better use of it than a second
implementation of one object.

The frequency axis is nearly free in that layout and it decides the assembly.
Per `(v, c, k)` pair the body is a fixed Hermitian rank-1 matrix
`(sqrt(v) rho_G)(sqrt(v) rho_G')^*` times a scalar `wt / (e_ij + w + i eta)`, so
the pair densities — one FFT of `conj(u_v) u_c` per pair, gathered onto the
response sphere — accumulate as a `(pairs, ngrf)` array per k-chunk and the
assembly is **one GEMM per frequency**. Chunked through `map_k`/`sum_k` per the
batching rule, which is what keeps it GPU-shaped.

**q = 0 only, refused by name otherwise.** Optical excitons are the deliverable
and every matrix element then lives on one k, one sphere, one grid. Finite `q`
needs the `k+q` two-sphere machinery P19 built for the spiral and P28's successor
will need for phonons away from `Gamma`; `q` stays in the signature so that it
slots in rather than being retrofitted.

**Transcribe the `t3hw` layout from the start**: at `q = 0` the head is a 3x3
block and the wings are `3 x ngrf`, so `nm = ngrf + 2`, and the bootstrap's `z1`
takes the trace over three. Two details of `genvchi0.f90` that look like noise
and are not — the head is *re-accumulated* with an extra `1/w` factor
(`cw/wrf`), which is a numerical-accuracy device rather than redundancy; and
`init3.f90` forces the kernel's frequency to `0 + i*swidth`, so the "static"
point carries the broadening. Keep the `i eta`. **Deviation, stated:** append a
dedicated `w = 0` point rather than clobbering the caller's first grid point as
Elk does.

**The one line of Elk that must *not* be transcribed faithfully is the head.**
`genvchi0` reads bare momentum matrix elements `pmat`, which is legitimate in an
all-electron code and silently wrong in a pseudopotential one, where
`[H, r] != p` because of the nonlocal projectors. What is needed is the velocity
operator — and that is `pypresso/response/velocity.py`, one `jvp` of `H(k)` at a
frozen sphere, rule D2 cashed in for the third time. **This is the phase's honest
autodiff claim**: not that the kernel is differentiated, but that the ingredient
Elk reads off a file is here a derivative of the Hamiltonian, and has to be.

**The scissors shift is part of the method, not a nicety.** PRL Eq. (3) replaces
`chi_0` by a gap-corrected model response, and every published bootstrap spectrum
is computed that way; without it nothing validates. pypresso has no scissors knob
at all today (`grep -rn scissor pypresso/` is empty), so P37 adds one — and with
it Elk's renormalisation of the matrix elements, `getpmat.f90:61`:
`p -> p * e_ij/(e_ij -+ Delta)` for the valence-conduction pairs, the Del
Sole-Girlanda factor, since the eigenvalues have already been shifted.

**Modules.** `pypresso/tddft/`, shaped like `topology/` — a new subsystem rather
than a fourth kind of thing inside `response/`, because nothing here is a
Sternheimer solve.

- `tddft/chi0.py` — pair densities, head and wings from the velocity operator,
  the symmetrised `v^1/2 chi_0 v^1/2` in the `nm = ngrf + 2` layout, the GEMM per
  frequency. States from an NSCF with an **energy window**, not a band count.
- `tddft/kernels.py` — the `fxc` **name registry**: `rpa`, `alda`, `lrc`,
  `bootstrap`, `bootstrap-1`. `alda` is the second autodiff story and is nearly
  free — `f_xc` is one `jvp` of `v_of_rho`, which `efield.py` already takes as
  `dv_of_drho` — and it is the **control experiment** the notebook's claim needs,
  since "RPA and ALDA cannot bind an exciton" has to be shown rather than
  asserted. `lrc` is Reining/Botti's `-alpha/q^2` (Elk's `fxclrc`, `fxctype 200`),
  two lines, and the foil the bootstrap self-consistently approximates.
- `tddft/dyson.py` — `eps_0 = 1 - X`, the solve and inversion per frequency, the
  3x3 macroscopic head, and the `210` fixed point with its convergence test.
- `workflows/tddft.py::run_absorption` — an `eqx.Module` result carrying the
  frequency grid, `eps_M(w)`, the LRC `alpha` the kernel is equivalent to, the
  iteration count, and `static_residual`. **What Elk prints under that name is
  not `F~_00`**: by the time `tddftlr.f90` reads `vfxc(1,1,1)`, `genvfxc` has
  already right-multiplied by `vchi0`, so the array holds `F~ X` and the printed
  number is `-4 pi (F~ X)_00` — and `(F~ X)_00 = (eps^-1 X)_00 / X_00` is not
  `F~_00 = eps^-1_00 / X_00`. Elk's own comment on that line ("head of matrix
  v^-1/2 f_xc v^-1/2") is stale. The convergence test reads the same array;
  transcribe both faithfully, and compare like with like.

**Refused by name.** Ultrasoft and PAW — the matrix element
`<v k| e^{-i(q+G) r} |c k>` needs the augmentation charge, which is
`topology/augmentation.py`'s `augmentation_at_q` rather than `qq` and is
reachable later, not now. Metals — the `f_i - f_j` weight kills the intraband
term at `q = 0`, so the Drude response is missing entirely and Elk's `tddftlr`
does not add it either. `nspin != 1`, noncollinear and spin-orbit (Elk's spin
path is a separate routine, `tddftsplr.f90`). Finite `q`. **A reduced k-set** —
`genvchi0` sums the *full* non-reduced grid, and symmetrising `chi_0(G, G')` on a
wedge needs a double G-space rotation that P36's rank-N symmetriser is not (it is
Cartesian). The compensation is that the shifted-grid refusal of P24 does not
apply, because nothing is being symmetrised; state the trade rather than
inheriting the refusal. And **non-convergence of the bootstrap loop is an error**,
not a warning, on `require_converged_responses`' precedent — Elk stops at
`maxit = 500`. Whether TB09 eigenstates may feed `chi_0` is decided explicitly
(they may: the sum over states needs no functional derivative, and mBJ +
bootstrap is a published combination) rather than left to whatever happens.

**The traps, ranked by how quietly they fail.**

1. **Band truncation has no refusal.** An underconverged spectrum is perfectly
   plausible. So the result carries `static_residual` — the sum-over-states
   `eps_M(w -> 0)` minus the Sternheimer value — computed always and reported, on
   P29's `pulay_error` precedent. It turns the phase's one unrefusable error into
   a number on the result object. **Kernel-matched, for the reason validation
   check 1 is**: ALDA-Dyson against the existing Sternheimer, RPA-Dyson against
   the xc-off flag. A `bootstrap` spectrum differenced against a Sternheimer
   number that contains `f_xc` measures the kernel, not the truncation, and stops
   being the diagnostic it is there to be.
2. **Both pair orderings.** Elk's loop runs all `(i, j)` with the `(f_i - f_j)`
   sign — resonant *and* antiresonant. Restricting to `v < c` and doubling gets
   `Re` roughly right and `Im` wrong.
3. **The `degspin` factor** in the weight: the classic silent factor of two,
   which the static identity and the f-sum both catch.
4. **`nbnd` cut through a degenerate conduction multiplet.** Individual matrix
   elements inside a multiplet are gauge-dependent (rule D4) and only the sum
   over the whole multiplet is not — which is why the window is an energy window
   (Elk's `emaxrf`), not a band count.
5. **The response G-set is its own cutoff** (Elk's `gmaxrf`, default 3.0), not
   the density's: inheriting `ecutrho` makes `nm` explode, and cutting it to the
   head removes the local-field body that is part of the physics. Own knob, own
   convergence figure in the notebook.
6. **Do not copy Elk's single precision.** `vchi0` is `complex(4)` there; here the
   dtype comes from the policy object and every correctness claim is float64.

**Validation, ranked — and the obvious identity is mis-paired.** The tempting
check is "RPA sum-over-states at `w -> 0` against the existing Sternheimer
`eps_infinity`". It is wrong as stated: `efield.py`'s screening kernel is one
`jvp` of `Calculation.potential`, so it is Hartree **plus f_xc**, and the exact
identity is therefore **ALDA**-TDDFT against it. The RPA identity needs the xc
term switched off in the existing solve — a small, honest flag on code we own.
Both are worth having.

1. **ALDA(`w -> 0`) == Sternheimer `eps_infinity`**, exact, across two disjoint
   code paths (sum over states plus a Dyson inversion, against a projected CG),
   and the single check that certifies the matrix elements, the weights, the
   Coulomb symmetrisation, the head and the inversion at once. This is the
   phase's `test_qeref`-grade test. Plus the same identity in RPA with the flag.
2. **Elk on silicon**, same grid, `swidth` and window. Soft — LAPW against a
   pseudopotential — so compare *structure*: the equivalent `alpha` against
   `1/eps_M`, the size of the local-field effect (head-only against the full
   matrix), and the `210`-against-`211` difference.
3. **The f-sum rule** as a *diagnostic*, not a tolerance: it undershoots
   systematically with band truncation, so it is plotted against `nbnd` rather
   than asserted.
4. **The physics**, in the notebook: silicon's E1/E2 redistribution, and for LiF
   the falsifiable statement is "a peak below the Kohn-Sham gap that vanishes
   when the kernel is set to zero" — *not* the binding energy, which the
   bootstrap is known to get wrong (below).
5. Kramers-Kronig is near-tautological for a sum of Lorentzians. Decoration; drop.

**Read before committing**, per the arXiv rule. The original PRL above; Reining,
Olevano, Rubio and Onida, PRL **88**, 066404 (2002) and Botti *et al.*, PRB **69**,
155112 (2004) for the LRC kernel the `lrc` registry entry is and the bootstrap
approximates; Rigamonti *et al.*, PRL **114**, 146402 (2015) with its Comment and
Reply, which is the sharpest criticism of the self-consistency and the source of
the "RBO" variant; and Byun and Ullrich (arXiv:1703.01663), whose assessment is
the reason claim 4 above is worded as a peak rather than a binding energy —
**the bootstrap family gives good spectra and poor exciton binding energies**,
and is sensitive to head-only against full-matrix and to which scissors shift is
used. Verify each reference rather than carrying it from here.

**Memory, per the standing rule.** Peak working set is `n_w * nm^2` complex for
the stored `chi_0` plus `pairs_chunk * ngrf` per chunk — so the response cutoff
and the frequency count multiply, and both are knobs. Put the number in
`PERFORMANCE.md` beside the timing.

**Deliverables.** The notebook (27), a `docs/features.tex` entry with its amber
refusal box, a README row whose provenance column says `new` and says *why*
precisely — QE has TDDFT and does not have this — and a `PERFORMANCE.md` line.
**`CLAUDE.md`'s scope paragraph listed `TDDFPT/` as out of scope and has been
edited**: this phase deliberately enters that territory, from Elk's side rather
than QE's.

---

**What the work found.** The design above survived; these are the four things it
did not contain, and the first three are of one kind — each leaves a spectrum
that is smooth, positive, correctly peaked and wrong, and none of them is caught
by any symmetry, sum rule or convergence test the phase has.

**1. `eps_M` is the inverse of the 3x3 head of `eps^-1`, not the head of the
inverse.** `tddftlr.f90` computes both from the same array thirty lines apart —
`zminv(nm, eps0)` gives `EPSILON_TDDFT_ij.OUT` and `zminv(3, epsm)` gives
`EPSM_TDDFT_ij.OUT` — and only the second is the macroscopic dielectric function
an experiment measures. Taking the first is not an approximation, it is a
different quantity: in RPA the head of the whole matrix's inverse is
*identically* `1 - X_head`, the **no-local-field** result, so the Dyson equation
runs, converges, and has no local-field effect at all. Measured on silicon: 24.53
against 22.33, **9%**, with the local-field-free number being the smooth,
plausible one. The symptom that gave it away was not the size but an *equality* —
`eps_M` came out bit-identical to `1 - X_head`, which cannot happen if local
fields are being included. That equality is now a test.

**2. The certifying identity holds only when the two kernels match**, and the
obvious pairing is the wrong one. The plan already said the sum-over-states route
should be checked against the Sternheimer `epsilon_infinity`, and the natural
reading is RPA-against-`eps_infinity`. It is not an identity: `efield.py`'s
screening kernel is one `jvp` of `Calculation.potential`, so it is Hartree **plus
`f_xc`** — an ALDA response, not an RPA one. The residue is 1.26 on a constant of
22, six percent, and it looks exactly like band truncation. So `dielectric_tensor`
gained `screening = 'hartree'`, which drops the exchange-correlation term and
nothing else, and **both** pairings are asserted: ALDA against the default and
RPA against the switch. With them matched the two routes agree to **1.3e-2 on
22.35**, across a sum over states plus a Dyson inversion on one side and a
projected conjugate-gradient solve that never forms a matrix on the other.

**3. A scissors shift breaks the truncation diagnostic in the same way.**
`static_residual` exists because band truncation has no refusal; it is
`eps_M(0)` here minus the Sternheimer value, and the Sternheimer solve knows
nothing about a scissors shift. On silicon at `scissor = 0.05 Ry` the reported
residual went from **+0.013 to -3.46** — the shift's own effect on `eps_M`,
wearing the name of a convergence error. A shifted run now builds one more
`chi_0`, at a single frequency and no shift, rather than reporting a number that
means something else. The general statement is the one the two findings share:
*a diagnostic that differences two routes measures every way they differ*, and
listing those ways is part of writing it.

**4. The head was the only place autodiff was load-bearing, and it was worth
checking that it was.** `genvchi0.f90` reads momentum matrix elements off a file;
in a pseudopotential code `[H, r] != p` and what is wanted is `dH/dk`, which
P24's `jvp` already gives. Measured against the local-only expectation
`2(k+G)`, the nonlocal correction is 8% on the dominant matrix elements of
silicon — small enough to look like noise in a spectrum and far larger than the
1.3e-2 the identity is asserted at. The rest of the phase is a transcription and
the module docstring says so; inventing an autodiff story for the Dyson algebra
would have been dishonest.

**What was measured.** Silicon, `Si.pz-vbc`, `ecutwfc = 18`, the unshifted closed
4x4x4 grid with `nosym` (the k-set `si-epsilon-unshifted-nosym.in` was committed
for, and the only kind this phase accepts):

| quantity | sum over states + Dyson | Sternheimer | difference |
|---|---|---|---|
| `eps_M(0)`, RPA (`screening = 'hartree'`) | 22.3322 | 22.3451 | **1.3e-2** |
| `eps_M(0)`, ALDA (`screening = 'full'`) | 23.6214 | 23.6088 | **1.3e-2** |
| a column of `chi_0`'s body, 16 / 30 / 60 bands | — | — | 19% / 2.3% / **0.19%** |

and the cost model is in `PERFORMANCE.md`: the pair transforms are a fixed ~6 s
that no cutoff touches, the assembly is `nw nm^2` and the bootstrap's fixed point
`nw nm^3`, with both exponents checked at `nm = 285` rather than extrapolated.

The body check is the sharper of the two and is blind to the head: `chi_0`
applied to `cos(G'.r)` is a column of the same matrix, and the Sternheimer route
is band-*complete*, so the sequence in `nbnd` is the measurement of the
truncation rather than a tolerance. The bootstrap fixed point converges
geometrically — a factor of about 25 per pass — in **9 iterations**, and the
local-field effect is 9% at `ecut_response = 8` Ry, converged to 2e-3 there and
half-missing at 2 Ry.

**And the physics, at the size this cell can show.** Silicon has no bound
exciton, so what an attractive kernel does here is redistribute: on a 0.005 Ry
grid with a common cut at RPA's absorption maximum, the weight below it goes
from **0.606** (RPA) to **0.643** (bootstrap), and the first moment of the
spectrum falls from 3.647 eV to 3.586. (The cut is an index, so the *fractions*
move with the frequency spacing — notebook 27 runs 0.004 Ry and reads 0.571
against 0.609 — where the first moment does not. Both orderings are what is
being claimed, and the test asserts both.) A scissors shift strengthens it as it should, because the kernel is
built from a `chi_0` whose gap has moved: at 1.5 eV the equivalent
`alpha = -4 pi F_00` goes from **0.023 to 0.046**, `eps_M(0)` from 23.1 to 16.8,
and the absorption maximum moves down by a full **1.0 eV** relative to RPA's.
**ALDA redistributes too**, and saying otherwise would be wrong — what it cannot
do is *bind*, because its head and wings are identically zero (`f_xc` is finite
where `v` diverges). That is a structural statement, it is asserted as one, and
it is the difference between the two kernels rather than any difference in size.

**And then LiF, which is what the kernel exists for.** Silicon has no bound
exciton, so the first test of the phase that could *fail interestingly* was Elk's
own canonical example — `examples/TDDFT-optics/LiF-bootstrap`, run unmodified
against the vendored binary and committed as
`tests/data/qe/reference.out.elk-lif-bootstrap`. Rocksalt LiF, LDA, the
head-only kernel (`gmaxrf = 0.0`), 8x8x8:

| | pypresso | Elk |
|---|---|---|
| RPA peak (no local fields) | 3.47 at **24.37 eV** | 2.67 at **24.49 eV** |
| bootstrap peak | 16.78 at **14.05 eV** | 18.53 at **13.67 eV** |
| on Elk's own shifted grid | 17.72 at 14.05 eV | — |
| `eps_M(0)`, bootstrap | 2.471 | 2.302 |

That is the bound exciton: a peak five times RPA's height, ten eV below it, in a
place RPA has nothing at all. **Repeating the run on Elk's exact `vkloff` moves
the height and not the position**, which separates the two candidate causes — the
0.37 eV is the pseudopotential (no Li semicore, and an LDA gap 0.5 eV below the
all-electron one, hence a larger scissors shift), not the k-sample. It is an
all-electron LAPW spectrum against a pseudopotential plane-wave one, so 0.1-0.4
eV on a peak position is about what the comparison can resolve.

**Elk's example asks for `gmaxrf = 0.0`, and that was refused here.** An empty
body is the **head-only** kernel of the long-range-correction literature — Byun
and Ullrich use it throughout — not a degenerate case, so it is supported now,
with its consequence written down: no body means no local-field effect, so
`eps_M` is `1 - X_head`, and with the `alda` kernel, whose head and wings are
identically zero, head-only is *exactly* RPA. It also pays for itself: with no
body there is no plane-wave matrix element to form, so **head-only does no
transforms at all**, where the first version transformed every pair and gathered
nothing from the result.

The regression test runs it on 4x4x4, and which claims it asserts is decided by
that: **the exciton survives a coarse grid and the RPA continuum does not** —
13.81 eV against 14.05 at 8x8x8, where the RPA maximum is at 15.65 against
24.37. A bound state is one transition and is sampled long before a continuum is.

**Left for a successor**, in the order they would come: ultrasoft and PAW, whose
matrix elements need `Q_ij(G)` and where `topology/augmentation.py`'s
`augmentation_at_q` is already the right object; finite `q`, which needs the same
`k + q` two-sphere machinery as a phonon away from `Gamma`; metals and their
intraband term, which Elk's `tddftlr` does not have either; and a **bound**
exciton, which needs a wide-gap material and a k-grid several times denser than
anything committed here — LiF is the canonical case and its pseudopotentials are
not in `tests/data/pseudo/`.


### P38 — The calculator: one object, bound methods. ✅ DONE.

**What.** `pypresso/calculator.py`. A `Calculator` is a `System` together with its
pseudopotentials, and every workflow, force, stress, response and invariant in the
package is a method on it. `Calculator.from_file("scf.in")` reads the input and loads
the pseudopotentials the `ATOMIC_SPECIES` card names; `get_scf()` runs and caches the
ground state; every other method consumes that cache. `from pypresso import Calculator`
is the one import a script needs, resolved lazily through a module `__getattr__` so that
`import pypresso` does not pull the whole package into a process that wanted `units`.

**Why it is a facade and not methods on `System`.** Two reasons, the second decisive.
`System` is an `eqx.Module` crossing `jit`/`grad` — `at_positions`, `at_strain` and
`at_spiral_q` all differentiate through it — so a `pseudos` field would change the
pytree every compiled path sees, as a static field it would become a jit cache key
hashing radial tables, and a cached `SCFResult` cannot live on a frozen module at all.
And **`System` does not have the pseudopotentials**: it carries the file *names*, so a
`system.get_bands()` would still need them as an argument, which is the API this exists
to shorten. The unit that can compute is `system + pseudos`, which is exactly what
`Calculation.__init__` already takes. `System.calculator()` is a constructor and nothing
more.

**The three rules, which are the design.**

- **Nothing mutates.** `with_positions`, `with_cell`, `with_spin` return a *new*
  calculator with an empty cache. pyqula's `h.add_swave(...)` is safe there because
  nothing expensive is cached on `h`; here a mutated cell under a cached `SCFResult`
  would answer for the geometry it was converged at — which is exactly the defect
  `test_geometry_invalidation` records one layer down. The converged state crosses as a
  **starting guess** (`starting_from`, P23) rather than as a cache, which is what makes
  a promotion cost one SCF iteration instead of twenty-five.
- **The implicit SCF announces itself**, on stderr, and `get_scf()` is the same code, so
  there is one behaviour rather than two. `scf_result` reads the slot without triggering
  anything — a property that ran the SCF in order to say whether one had been run would
  be useless in an `if`. An unconverged state is refused rather than differentiated.
- **The cache is one slot**, holding the result and the options that produced it. Not a
  dictionary keyed by option sets: what it holds is the wavefunctions, the largest arrays
  in the process, and the memory rule applies to the front end too. Options are compared
  defensively — an array-valued `starting_density` has no scalar `==` — and a comparison
  that cannot be made counts as a miss, so the failure mode is recomputing rather than
  serving the wrong state.

**Option routing.** Shared options (`SHARED_OPTIONS`: `nbnd`, `conv_thr`, `k_batch`,
`diagonalization`, the mixing knobs, `verbose`) are given once to the constructor and
forwarded to each entry point **by named parameter**, never by a `**kwargs` catch-all —
`electrostriction` has one and forwards it to the Sternheimer solvers, which have no
`nbnd` and would raise. An unknown constructor keyword is a `TypeError` naming the
allowed set, because the ergonomic risk of `**defaults` is a typo becoming a silently
ignored setting. The one special case is `run_spiral_scan`, whose SCF options *do* go
through a `**kwargs` to `run_scf`, so its filter is `run_scf`'s signature.

**What the threading actually costs, stated exactly.** The state carried beside the
density is `becsum` (PAW), `ns` (Hubbard) and `tau` (meta-GGA), none of them recoverable
from the density. The package was already careful here: `fixed_density_bands`,
`run_nscf` and `dielectric_tensor` all **refuse** without them rather than computing
something else. So the facade does not close a silent-wrong-number hole — that hole was
already closed — it removes a stopped run and a puzzle. Worth writing down, because the
first draft of this entry claimed the stronger thing and the refusals disprove it.

**It found one real gap.** `run_dos` did not forward `becsum` or `ns` to the `run_nscf`
it calls, so a **PAW or DFT+U density of states on a denser grid stopped on that
refusal** — the feature was unreachable rather than wrong, which is why no test saw it.
Both are parameters of `run_dos` now. The facade is what exposed it: it passes what each
entry point *names*, and this one named too little.

**Sugar that shipped with it, because that is where the remaining verbosity was.**
`.plot()` on `BandStructure`, `DensityOfStates`, `ProjectedDOS` and `OpticalSpectrum`
(matplotlib imported inside the method, so it stays out of a calculation's dependencies),
returning the axes; the band and DOS plots take their zero from the SCF's own Fermi level
and **name it after what it is** — an insulator has no Fermi level and gets `E_HOMO`.
`SCFResult.__repr__`, because the generated dataclass one prints the wavefunctions, the
density and the potential in full. `get_stress()` returns `SCFResult.stress` when
`tstress` already produced it rather than differentiating twice.

**Validation.** `tests/unit/test_calculator.py`, 19 tests. The load-bearing ones: the
bound method reproduces the functional entry point to 1e-10 on the same input; the cache
is one slot and options key it; reading the cache builds no basis; a derived calculator
does not serve its parent's ground state and *does* carry it as a seed; a PAW band
structure through the facade equals the hand-threaded one **and** the call that omits
`becsum` raises; and `get_dos(grid=...)` now works on PAW at all. One test is
structural — every `get_*` method is short enough to be a delegation — because a method
here that grew a computation of its own would be a second implementation of something
already validated against QE.

**It also found a broken feature that is nobody's front end.** Sweeping every method once
— because a signature checked by `grep` is not a call that works — reached
`relax_spiral_q` (P21) and it **raised before taking a step**, for everyone, through the
plain functional API. P29 gave `BFGS.__post_init__` a `_rebuild_metric(0)`, so the metric
is `(0, 3, 3)` until the first `step` fills it in; `_first_step_scale` measures a trial
step *before* that first step, and `_norm` then reduces over an empty array.
`tests/regression/test_spiral_relaxation.py::test_relaxation_finds_the_antiferromagnet`
had been failing since P29 landed. One line in `_first_step_scale` fixes it. The lesson
is the one this file keeps recording in other forms: **a cross-phase regression is
invisible to the phase that causes it**, and the only thing that finds one is running
everything, which is what a front end makes cheap enough to do.

**Running the whole regression suite found fourteen failures and none of them was
P38's.** They are recorded here because each had a different cause and two were not what
they looked like.

**Eleven were `test_input_sweep`, and P29 left them.** The eleven `vc-relax` inputs stayed
in `EXPECTED_REFUSALS` after P29 implemented the feature, so the sweep failed with "now
builds; drop its EXPECTED_REFUSALS entry" — exactly what that test exists to say. Same
shape as the `relax_spiral_q` regression above: **P29 landed its own feature and left a
sibling's expectations stale**, twice.

**One was `chi_0`'s finite difference, and it was not a tolerance.** `si2-us` missed by
6.0e-5 against a 1e-5 bound — but only in a *window* of steps around `h = 1e-4`, and it was
fine at 5e-5 and at 2e-4 and above, which is why it read as noise. The reference
diagonalises the perturbed Hamiltonian asking for **exactly** the bands the density needs,
and Davidson converges its *topmost* root worst — nothing above it to push it down — so the
badly converged highest occupied state went straight into the reference density. Four
buffer bands, discarded after, make it 9e-7 at every step from 5e-5 to 4e-4 and stop
`|ref|max` depending on `h` at all; `si2-nc-force` improves from 4.3e-6 to 4.6e-7 at its
worst step. **Two plausible fixes are wrong and were measured**: widening the bound hides
it, and seeding Davidson with the SCF wavefunctions makes it *ten times worse* (1e-3),
because the solver stops early against a guess it believes.

**Two were the Raman wedge against the closed grid, and the cause is a good commit.**
Bisection put the first bad commit at `a351005` (the GPU Cholesky NaN), and `git show` then
cleared the half everyone would suspect: `_cholesky_route`'s **body is unchanged** across
it, so the fast path really is bit-for-bit as that commit claimed. What moved is the other
half — the Anderson mixer's Gram normalisation, cond 1.1e11 → 2.7e4. A third derivative
multiplies the difference between two converged densities by `<u|u>`, of order 10^3, so the
wedge and the closed grid agree only to what their two SCFs agree to; the old mixer drove
both k-sets to the same fixed point bit for bit and the new one does not. Convergence-
limited and measured: **3.3e-9 at `conv_thr = 1e-12`, 6.5e-10 at 1e-14**. The bounds are the
measured floor now (1e-8 and `rel=1e-4`), with the measurement written into the test, and
they still catch what the assertion exists for — the wrong wedge repair missed by 3.3e-2,
four orders above the new bound. `CLAUDE.md`, `NONLINEAR.md` and the table above carried
8.7e-14 and now carry the truth. **The silicon rank-4 7.9e-14 beside it is from a
grid-sharing pair no test exercises and has not been re-measured.**

The lesson for the suite rather than for any phase: 1477 regression tests take hours, so
they were not being run, and three separate phases' claims drifted without anyone seeing
it. `tools/` has no runner for them; the one written for this pass
(one file per invocation, a durable summary line each, resumable) is the shape that
survives being interrupted.

**Not done.** The functional API is untouched and every existing call site stands; the
CLI still has its five copies of the `read_upf` loop. Migrating it is the obvious next
step and is not in this phase. Notebooks 01 and 03-27 still open with the functional
form; `notebooks/README.md` permits both and asks new ones to use the calculator.

---

Ordering note: P6 (symmetry) can slip after P7/P8 if band structures come first, since
`nosym` runs are fully testable — but it must land before any timing claims, as it changes
the k-point count.

---

### P39 — The dynamical matrix when `S` moves with the atoms. ✅ DONE.

**What.** P25's norm-conserving restriction is lifted: `dynamical_matrix` runs on
ultrasoft and PAW datasets. Two-atom silicon's optical mode comes out at **513.2947**
(ultrasoft) and **513.3776** (PAW) cm⁻¹ against the vendored `ph.x`'s 513.275287 and
513.404419 — **0.019** and **0.027** cm⁻¹, tighter than the norm-conserving case's 0.05,
which is not a claim about the physics: both are the same `dq = 0.01` radial-table floor
landing on different sides of it. The acoustic residue is 6.1 and 6.2 cm⁻¹ against the
norm-conserving 4.1, and the raw force-constant sum is below 2e-4 Ry/bohr².

**Four terms, and every one of them switches itself off when `S` is the identity** —
which is the regression that guards the whole phase, since the norm-conserving silicon
and aluminium cases have to come out unchanged to round-off:

- **The source term is `(dH/du - eps dS/du)|psi>`.** `dvqpsi_us_only` builds it from
  `deff = deeq - et qq` (`compute_deff`) and not from `deeq`. Worth little on its own —
  0.9% on the response density — and wrong to leave out.
- **The first-order state has an occupied block the solve does not produce.**
  `orthogonalize`'s projector makes `dpsi` orthogonal to the occupied manifold in the
  `S` metric, but the physical first-order state is not: the constraint it satisfies is
  `<psi + dpsi|S(u + du)|psi + dpsi> = 1` and `S` has moved. `orthogonality_states`
  is `-1/2 sum_m psi_m <psi_m|dS/du|psi_n>` (`compute_drhous`), and the check on it is
  an *identity*: the first-order constraint residual is **1.6e-16** against a `dS/du` of
  6.7e-2.
- **The mixed state changes at frozen states** (`drho.f90`: "the change of the charge
  density due to the displacement, at fixed wavefunctions; the orthogonality part is
  included"). It is one `jvp` of the raw mixed-state builders along `(e_i, dpsi^ort)`,
  and it is consumed in **two** places — it screens inside the self-consistent loop, and
  it enters the assembly, where the builder generates its own half so only the rest is
  handed in.
- **The multipliers move, and as a matrix.** `d_Lambda d_j L = -<psi|dS/du_j|psi>`
  vanishes identically when `S` is the identity, and that — not a missing routine — is
  the whole of why P25 was norm-conserving. Written with a *diagonal* multiplier the
  functional stops being invariant under a unitary mixing of the occupied states, while
  the state tangent is only defined up to exactly such a mixing; the sum rule stops at
  1.7e-2 Ry/bohr² whatever else is switched on. `dLambda_mn = w_n[<psi_m|dH|psi_n> -
  1/2 (eps_m + eps_n) S'_mn]` — the `1/2 (eps_m + eps_n)` and not `eps_n`, which is
  `born.py`'s expression and is right *there* because a field leaves `dpsi` orthogonal
  to the occupied manifold. **Two independent formulations were built and agree to the
  last digit** on the residual they left: the hermitian gauge with a matrix `dLambda`,
  and the perturbation-theory gauge with a diagonal one, which is what said the gauge
  machinery was finished and the remaining error was elsewhere.

**Two bugs found on the way, and neither is an ultrasoft term.**

- **`addcore` was missing, for every dataset.** `_bare_displacements` builds the source
  at a *frozen* `v_scf`, which is right for the local potential and the projectors and
  wrong for `v_xc`: `rho_core(r - tau)` travels with its atom, so the exchange-correlation
  potential changes at a frozen valence density. QE keeps it as `drhoc` and hands it to
  `dv_of_drho`; here it is one `jvp` of the potential through `at_positions` at a fixed
  density. **Every committed phonon case before this had no core charge** — `Si.pz-vbc`
  and `Al.pz-vbc` are `core_correction="false"` — so a norm-conserving dataset *with*
  one had the same bug and nothing could see it. Leaving it out put the response density
  **45%** away from a finite difference of re-converged densities and the optical mode at
  785 cm⁻¹.
- **`addusforce` was missing from the differentiated gradient.** P25 hands the density to
  `frozen_energy` as a frozen array, which is right for a norm-conserving run — `rho` has
  no explicit position dependence there — and for an ultrasoft one deletes the
  augmentation charge's own force. The gradient being differentiated was then not the
  force: measured on displaced ultrasoft silicon, `-grad` gives **0.0198** against
  `compute_forces`' **0.0752**.

**And the term the sum rule could not see.** With all of the above the acoustic sum rule
held and a whole column of the force constants was still at **0.26** of a
finite-differenced one — P28a's lesson exactly, that an atom-sum is blind to a transfer
*between* atoms. The missing piece is the density's **cross derivative**
`d^2 rho / du dpsi . dpsi`: the augmentation charge's position dependence applied to the
state response, which is `addusdynmat`/`drhodvus`. It exists only when both tangents are
in the *same* `jvp`, and P28's `wg`/`wk` weight split had deliberately put them in two.
So for an ultrasoft dataset the assembly is **one** `jvp` again — which is exactly why an
ultrasoft **metal** is refused: there the two weights differ and the split cannot be
undone. Feeding the assembly an *exact* rigid translation, where every tangent is known
in closed form, is what isolated it: the norm-conserving functional returned 0 and the
ultrasoft one returned 0.494106, term by term in `local`, `hartree` and `xc`.

**Validation, in the order it was used.** The constraint identity on `ort` (1.6e-16); the
source term against a finite difference (1.7e-8); the response density against a central
difference of re-converged SCF densities on a small `nosym` cell (**2.3e-5**, with a
norm-conserving control at 6.6e-6); the acoustic sum rule, which caught three bugs at 758,
99 and 7 cm⁻¹ and then went quiet; a finite difference of the *forces*, which caught the
two it could not (**2.6e-4** on a column of 0.371); and finally `ph.x`.

**The strain response does not have the same hole, and checking rather than assuming
is the point.** `strain.py`'s screening `jvp` is the phonon loop's, so the suspicion was
natural; but its *bare* perturbation is not. `_bare_strains` rebuilds the potential from
the moved cell inside the traced function -- `moved.potential(density).v_scf`, where
`at_strain` has already rebuilt `rho_core` -- so the core charge's deformation is
differentiated there, in the one place the phonon path had frozen it
(`_bare_displacements` takes `v_scf` as an argument and holds it). Same omission, two
coordinates, and only one of them had it.

**Refused:** an ultrasoft or PAW **metal**, by name. The strain response, the elastic
constants, electrostriction and the Raman tensor still refuse ultrasoft and PAW through
`require_norm_conserving`, which is unchanged — each adds a `dbecsum` term of its own in
the strain coordinate on top of what this phase writes.


### P39a — PAW Born charges: two candidates, both measured, both rejected. 📋 OPEN.

**Where it stands.** Everything in `born.py` works for a PAW dataset up to **1.3e-3**
(-0.078293 against the vendored `ph.x`'s -0.07961), where the ultrasoft case of the same
assembly reaches 8e-6 and the norm-conserving one is exact to every printed digit. The
refusal stays; what is new is that the two obvious explanations are now *excluded* rather
than untried, and the machinery to try them exists as tested functions.

**Candidate 1 — QE's fifth stage, `int3_paw` against `becsumort`.** Both objects exist
after P39: `paw_response` along the field's `dbecsum` is `int3_paw` (`efield`'s
`internals["onecentre"]`), and `non_variational_response` gives the displacement's
`becsumort`, constraint-verified to 1.6e-16. Contracted in this code's full-matrix
convention the term is **0.004882** where the gap is **0.001317** — **3.7 times too
large**, in either sign, so no sign choice lands on `ph.x`. The reading that fits is that
the Lagrangian *already contains* it: the u-leg's orthogonality correction reaches the
one-centre energy through the multiplier tangent `d_Lambda d_j L`, which
`_multiplier_response` builds from a perturbation that already carries `dddd_paw`.
Adding QE's term on top would count it twice, and scaling it to fit would make the
number a measurement of `ph.x`.

**Candidate 2 — the wedge sum inside a nonlinear functional.** P36's finding says the
*value* fed to a nonlinear functional must be the full-zone object while its *derivative*
stays the raw wedge sum, and PAW's one-centre energy is exactly such a functional of
`becsum`. The raw and symmetrised field responses of `becsum` differ by **19 to 46 per
cent** on PAW silicon, so the effect is not small. Implemented (the symmetrised response
handed over from `efield`'s loop, with the chain rule's raw tangent corrected to it) it
moves PAW **the wrong way — 1.3e-3 to 2.8e-3** — while leaving norm-conserving silicon
exact to every digit and ultrasoft at 1.0e-5. Reverted. The norm-conserving invariance is
itself the useful half of the result: it confirms that for everything *linear* in the
response, `symtensor` on the assembled tensor really does complete the wedge, which is
the convention `born.py` was written on.

**Worth knowing before the next attempt.** Silicon's `Z*` is **zero by symmetry**; the
-0.0757 both codes print is a basis-set residue. So this comparison is between two codes'
*errors*, and that they agree to 8e-6 for ultrasoft and to every digit for
norm-conserving is what makes the PAW disagreement meaningful rather than noise. A polar
PAW crystal — AlAs with a PAW dataset, against `ph.x` — would say whether 1.3e-3 on a
residue is 1.3e-3 on a real charge, and that measurement does not exist yet.


### P40 — Ultrasoft and PAW in the sum-over-states `chi_0`. 📋 OPEN, two findings banked.

**Attempted and reverted**, because the identity it is validated by did not close and a
half-right spectrum is worse than a refusal: with the augmentation charge in every matrix
element and the head's dipole generalised, ultrasoft silicon's `eps_M(0)` still sits
**2.1%** from the Sternheimer solve where the norm-conserving control on the same
machinery sits at **0.06%** (-1.20 on 55.5 against -0.0129 on 22.3, both at 60 bands and
`ecut_response = 8`, both in RPA against `screening="hartree"`). The refusal is unchanged.
Two things were established and are worth more than the code that was thrown away.

**Finding 1 — the `1/Omega`, and it is measurable.** `AugmentationCharge.qgm` is tabulated
for the *density*, where `addusdens` pairs it with `becsum` and the result is a charge per
unit volume. A matrix element `<u_i|e^{-iG.r}|u_j>` is dimensionless and carries none, so
the table has to be multiplied by the cell volume before it can be paired with
`<beta|psi>` — the same restoration `topology/augmentation.py` makes for the overlap
between two k-points, and the check is `qq = Omega Q_ij(G = 0)`, which is what `s_psi`
contracts with. Leaving it out is a factor of **265** on this silicon, and the symptom is
an augmentation term that changes the answer by 0.008 in 55: it reads exactly like a
dataset whose augmentation charge happens to be small. With the factor in, the residual
halves (-2.45 to -1.20), which is how it was found.

**Finding 2 — the body is not where the augmentation lives.** With the body's `Q_ij(G)`
correct, adding the head's `q`-linear part (`adddvepsi_us`'s `dpqq` and
`i q_kl <d(beta_l)/dk|psi>`, Eq. 10 of Dal Corso and Mauri) moves `eps_M(0)` by **0.0015**
— nothing. So neither the body's augmentation nor the head's dipole accounts for the 1.2,
and the next attempt should not start with either. What is *not* excluded: the pair
density's normalisation against `sum_band`'s (the plane-wave half and the augmentation
half were matched by the `qq` identity at `G = 0`, but never by an independent check at
`G != 0`), and the `f_i - f_j` weight, which for an ultrasoft dataset multiplies a
generalised density whose norm is `<psi|S|psi>` rather than `<psi|psi>`.

**Kept for the next attempt** (all reverted, all reconstructible from this entry):
`SphereAugmentation`, a gather of `Q_ij(G)` from the dense table onto the response sphere
by Miller index; `VelocityOperator.dipole_elements`, which is
`<m|dH/dk - eps_n dS/dk|n>` with the **column** eigenvalue -- fixed by differentiating
`H|n> = eps_n S|n>` and projecting on `<m|`, and checkable on the diagonal against
`band_velocities`' generalised Hellmann--Feynman term.

**The validation to use is the one that worked here**: `run_absorption` in RPA against
`dielectric_tensor(screening="hartree")` on the same states, on a small `nosym` cell
(`si-us-nosym.in`, committed for P39) with a norm-conserving control beside it. It is
sharp — 0.06% on the control — and it needs no reference beyond this code.


### P41 — The strain response when `S` deforms with the cell. ✅ DONE.

**What.** `strain_response` runs on ultrasoft and PAW datasets. The density response
`drho/d(eps)` matches a central difference of *re-converged* strained SCF runs to
**4.6e-4** (ultrasoft) and **4.7e-4** (PAW) on the (0,0) strain, against a
norm-conserving control of 1.9e-4 on the same cell, and to 5.8e-5 / 5.7e-5 / 3.2e-5 on
the shear. The reference needs no `ph.x` -- `ph.x` has no strain perturbation at all --
and it is the same one P26 already used.

**It is P39 in the other coordinate, plus one term of its own.** The augmentation charge
`Q_ij(r)` is a function of the *cell*, so a homogeneous strain deforms the reciprocal-space
table it is tabulated on where a displacement only translates it; `at_strain` already
rebuilds `build_augmentation`, so that term is a `jvp` and not an expression. On top of it
come P39's, transferred: the source term is `(dH/d(eps) - eps dS/d(eps))|psi>`
(`compute_deff`); the first-order state has an occupied block the solve does not produce;
the mixed state changes at frozen states, which for a strain was *already* carried
(`_frozen_density_response`) and now carries its `becsum` too; and PAW's one-centre
response is mixed beside `dvscf`, which needed a rank-2 `PAW_dusymmetrize`
(`BecsumSymmetry.apply_strain` -- **not** two applications of the vector case, which would
average over the group twice).

**A suspicion checked and dismissed, which is why P39's `addcore` note here was wrong.**
The phonon loop froze `v_scf` in its bare perturbation and so never saw the core charge
move; `_bare_strains` rebuilds the potential from the *moved* cell inside the traced
function, where `at_strain` has already rebuilt `rho_core`, so the strain coordinate had
the term all along. Same omission, two coordinates, one of them clean.

**Still refused:** the strain *derivatives* -- deformation potentials' consumers, elastic
constants, electrostriction and the Raman tensor -- on ultrasoft and PAW. They stand on
P26's second-order functional `F`, which is norm-conserving in three places that a lift
has to write rather than inherit: `_project_conduction` is `1 - sum |psi><psi|` where the
`S` metric wants `1 - sum |psi><psi| S`; the multiplier term contracts `<u_i|u_j>` where
it wants `<u_i|S|u_j>`; and its `raw_density` passes an **empty** `becsum` literally
(`electrostriction.py:341`), which is what a Raman run on an ultrasoft dataset raises on
today. PAW adds a fourth: the one-centre energy is a second nonlinear functional of
`becsum`, so `F` gains a one-centre screening term beside the grid one.

### P42 — An ultrasoft spin spiral. 📋 OPEN, attempted and reverted, four findings banked.

**Reverted rather than shipped**, because the identity that discriminates did not close:
`E(q + G) = E(q)` came out **5.9e-3 Ry** against a norm-conserving control of
**9.6e-13** on the same script and the same cell. What was learned is worth more than the
code.

**Finding 1 -- `spinor_becsum` used one sphere's projectors for both components.** It
takes `vkb` as `(nk, npwx, nkb)` and contracts `"gc,bag->bac"`, applying the *same*
projector to the up and down halves of the spinor. For a spiral they live at `k + q/2`
and `k - q/2`, so the down component was projected on the wrong sphere. It is the same
class of bug `_project` avoids in the Hamiltonian, and the fix is one einsum
(`"agc,bag->bac"` on a paired `(nk, 2, npwx, nkb)`).

**Finding 2 -- the transverse table, and where its phase goes.** The cross-spin block of
`becsum` pairs the two spheres, so its augmentation charge lives at `G + q` rather than
`G`: a second `build_augmentation` with a cartesian `shift`. Whether the structure factor
moves with the shift or stays at `G` is **not** decidable on a cell whose atom is at the
origin -- both give the identical energy -- which is why the O chain used here could not
settle it and a two-atom cell is needed.

**Finding 3 -- the assembly, and its sign.** `becsum` is hermitian under exchanging its
two `(channel, spin)` labels, so the down-up block is the conjugate of the up-down one
and the pair is one complex field: `A(G) = sum Q_ij(G + q) (becsum_x - i becsum_y)_ij`,
giving `m_x <- Re A(r)` and `m_y <- -Im A(r)`, with `newd`'s partner the same way round.
The sign is settled rather than guessed: flipping it degrades `E(-q) = E(q)` from
**1.4e-14** to **4.6e-7** and periodicity from 5.9e-3 to 2.5e-2.

**Finding 4, and it is the one to carry -- `q = 0` is blind to all of this.** A spiral at
`q = 0` reproduces the ordinary noncollinear ultrasoft total energy to **0.0e+00 Ry**,
which reads as a complete validation of the two-sphere machinery and is not one: at
`q = 0` the moment lies along `z`, so `m_x = m_y = 0` and the transverse assembly is
never evaluated. It is a gate worth having -- it does check `becsum`'s pairing, `s_psi`,
`newd` and `addusdens` -- but it cannot see the term the phase actually lives in. The
identity that can is periodicity, and it is cheap.

**For the next attempt:** the sign and the layout above are settled; what is missing is a
term that periodicity sees and `E(-q) = E(q)` does not. Use a **two-atom** spiral cell so
that the structure factor's placement is observable, and put the periodicity identity
first rather than last. PAW is a further step again: its transverse one-centre term needs
Elk's `zqss = e^{-i q.tau/2}` inside the radial quadrature.


### P43 — The second-order energy with a moving overlap, and its third derivative. ✅ DONE.

**What landed.** P26's variational second-order energy `F` -- the object the elastic
constants, electrostriction, the elasto-optic tensor and P35's Raman tensor all stand on
-- is now exact on all three pseudopotential kinds. The check is the identity that
already pinned it for norm-conserving silicon: `F` at its stationary point is
`dielec.f90`'s single overlap, so `_epsilon_at` must reproduce `dielectric_tensor`'s
`epsilon`.

| | `F` route | `dielec.f90` | relative |
|---|---|---|---|
| norm-conserving | 56.29287520 | 56.29287515 | 8.4e-10 |
| **ultrasoft** | 61.52645643 | 61.52645641 | **3.4e-10** |
| **PAW** | 61.45827346 | 61.45827346 | **6.9e-11** |

**Four terms, found on a staircase — 21% → 2.2e-3 → 1.6e-4 → 3.4e-10** — and each is
identically zero when `S` is the identity:

- **`becsum` inside the functional's own density.** `raw_density` passed an empty tuple
  *literally* (`moved.augmented(..., ())`), which is what a Raman run on an ultrasoft
  dataset raised on: the augmentation charge is part of `rho` and was simply absent.
  `ddd_paw` in its Hamiltonian is the same omission one level over.
- **The `S` metric in the projector and the multiplier.** `<u_i|u_j>` becomes
  `<u_i|S|u_j>`, and "orthogonal to the occupied manifold" is a statement about the
  metric. Worth the first and biggest step, 21% → 2.2e-3.
- **A state and a right-hand side take *different* projectors**, and collapsing them is
  worth 2.2e-3 → 1.6e-4. `orthogonalize` builds `P_c^+ = 1 - sum S|psi><psi|` for the
  *source* of a Sternheimer equation; what makes a *state* orthogonal to the occupied
  manifold is `P_c = 1 - sum |psi><psi| S`. They coincide at `S = 1`, which is why one
  expression served both. The consequence reached the callers: `raman_tensors` and
  `electrostriction` pre-projected `b` and `u` with the state form before handing them
  over, which for an ultrasoft dataset *undoes* the right one -- they hand both over
  unprojected now and `F` projects each correctly itself.
- **PAW's one-centre screening**, 1.6e-4 → 6.9e-11 on PAW. Its one-centre energy is a
  second, independent nonlinear functional of `becsum`, so the second-order energy has a
  one-centre `1/2 dx K dx` beside the grid one -- `PAW_dpotential` contracted against the
  `becsum` response rather than added to a potential.

**And the third derivative closed too, with two tangents that are only right
together.** The Raman tensor -- one `jvp` of that same `F` -- was **3.0e-2** (ultrasoft)
and **3.2e-2** (PAW) from a central difference of `epsilon` over re-converged *displaced*
cells, where the norm-conserving control on the same script is **6.8e-4**. It is
**1.2e-4** (ultrasoft) and **1.2e-4** (PAW) now, tighter than the control, which does not
move by a single digit -- both terms are identically zero when `S` is the identity, and
that bit-identity is the check that the plumbing is right.

**What found them is a decomposition, not a guess.** `d(eps)/d(tau)` is a total derivative
of `F(geometry, psi, rho, b, u)`, so it is the sum of five partials and **each one can be
measured on its own** -- analytically as the `jvp` with one tangent non-zero, and by
finite difference as `[F(A(+h), rest frozen) - F(A(-h), rest frozen)]/2h` with `A` taken
from a re-converged run (`psi(±h)` aligned onto `psi(0)` by the unitary polar factor of
their overlap, with `b` and `u` rotated by the same matrix, which is legitimate exactly
because `F` is invariant under that rotation -- the `Tr(Lambda Ov)` order above). Three
partials agreed to 7e-4 and two did not:

| | analytic | finite difference | |
|---|---|---|---|
| geometry | -29.7019 | -29.7012 | ✅ |
| `psi` | -0.5308 | **+0.6274** | ❌ |
| `rho` | +7.9261 | +7.9257 | ✅ |
| `b` | -44.7790 | **-47.9855** | ❌ |
| `u` | 0 (frozen) | +0.0984 | envelope residue |

- **The state tangent is `P_c dpsi + ort`.** With `S` moving the orthonormality
  constraint fixes a piece of the first-order state that the Sternheimer solve does not
  produce, and it is P39's occupied block. Alone it gives +0.5856 against the measured
  +0.6274 -- right -- while making the *total* worse, 3.0e-2 to 8.0e-2, which is what the
  previous pass measured and read as an exclusion.
- **`b` is not the solution of its own linear equation.** `dvpsi_e` solves for
  `P_c r|psi>` and then `adddvepsi_us` applies `S` to it and adds the augmentation
  dipole (Dal Corso and Mauri Eq. 10), whose `beta`, `qq`, `dpqq` and `d(beta)/dk` all
  travel with their atom. So `db` is the tangent of a **composition**, and the frozen
  solution the differentiated residual is written about is `commutators`, not `b`. With
  the tail and the corrected state tangent the `b` partial is -48.0123 against the
  measured -47.9855.

**A third thing had to change for either to be reachable**: `VelocityOperator.projectors`
read the atoms with `np.asarray(positions)`, so `d(beta)/dk` about the atom's own centre
was **not differentiable in the geometry at all** -- the term simply vanished from any
derivative taken through it, silently.

**Checked past the one column that found it.** Atom 1 agrees with its own finite
difference to 1.2e-4 as well and the translational sum rule holds at 9.7e-15 of the
scale, which is the check that is *not* an atom-sum's blind spot; and on the symmetric
`si-epsilon-us.in` the tensor comes out exactly zincblende (forbidden components 3.9e-16)
with a sum rule at 5.8e-16, so the symmetry-reduced path works too.

**The `u` partial is left frozen and is the envelope residue**, 0.098 of 69. It is inside
the O(h^2) noise of the decomposition itself (the five finite-difference partials sum to
-69.035 against the total's -69.194), and the end-to-end 1.2e-4 is the authority.

**Still refused**, by name: the *strain*-coordinate third derivatives -- the elastic
constants, electrostriction and the elasto-optic tensor -- on ultrasoft and PAW. They
share `_position_response` with the Raman tensor and would inherit its tail, but neither
the occupied block's analogue under a strain nor the tail's behaviour there has been
measured, and P43's own lesson is that one of these terms alone is worse than neither.
The *strain response* (P41), the *dynamical matrix* (P39) and now the *Raman tensor* are
implemented on all three kinds. **P44 measured how far off it is** and wired in the two
tangents that do transfer, without lifting the refusal.


### P44 — The strain coordinate's third derivative: measured, still refused.

**What this phase produced is a measurement and an exclusion, not a lifted refusal.**
P43 closed `d(eps)/d(tau)` on ultrasoft and PAW with two tangents that are only right
together. This phase carried the same decomposition into the *strain* coordinate, where
the same functional `F` gives the elasto-optic tensor and the electrostriction
coefficients, and found the same two tangents transfer -- and are not enough.

**The five-partial decomposition, on `si-us-nosym`, `d(eps_00)/dx_00`.** `F` is a
function of `(strain, psi, rho, b, u)`, so the derivative is a sum of five partials and
each is measurable against its own finite difference over a *re-converged strained* cell
(`psi(±h)` aligned by the unitary polar factor, `b` and `u` rotated with it -- legitimate
because `F` is invariant under that rotation). Three agreed and two did not, exactly as
in the displacement coordinate:

| | analytic | finite difference | |
|---|---|---|---|
| geometry | +47.6371 | +47.6357 | ✅ |
| `psi` | +3.4501 | **−2.4855** | ❌ |
| `rho` | −28.8586 | −28.8575 | ✅ |
| `b` | +100.2647 | **+112.1054** | ❌ |
| `u` | 0 (frozen) | −0.0191 | envelope residue |

**Two of P43's ingredients transfer, and they are wired in behind the refusal**
(`susceptibility_strain_derivative`), because they are established and whatever closes
this will need them:

- **the state tangent is `dpsi + ort`** -- `StrainResponse.ort`, which P41 already built
  and the assembly was not using. It takes the `psi` partial's error from **+5.94 to
  +0.032**. It also removes an inconsistency between two arguments of one `jvp`:
  `_frozen_density_response` already puts that block's density contribution *into*
  `response.drho`, so the committed code handed `F` a density tangent containing a piece
  its state tangent left out.
- **`b` is not the solution of its own linear equation**, so `_position_response` is
  handed `internals["commutators"]` as `stored`.

| | ultrasoft | PAW | norm-conserving control |
|---|---|---|---|
| neither | 4.58e-2 | 5.53e-2 | 2.3e-4 |
| both | **1.30e-2** | **1.30e-2** | 2.3e-4, unmoved |

-- where the reference is the **sum of the five partial** central differences
above rather than the end-to-end difference of `epsilon` the committed test
uses. The two agree to O(h^2) because `F` is invariant under the alignment
rotation, and the norm-conserving control in that column is computed the same
way, so the comparison is like for like; `scratchpad/strain_b.py` prints it.

A thirty-fold improvement, fifty times the control, and past the phase's own
`THIRD_DERIVATIVE_TOLERANCE` of 5e-3 -- so **the refusal stays**. Either tangent alone is
worse than neither, one more time: on the `b` partial against its measured +112.105, the
occupied block alone gives −27.27 and the tail alone +14.25 where both give −1.72.

**What is left is entirely `b`, and it is −1.72 of 112 -- the *same* number on ultrasoft
and on PAW**, which is what says it is structural rather than a dataset's physics.

**And one candidate for it is excluded by measurement, which is this phase's finding.**
`_position_response` writes its *operator* with the multiplier matrix (`H b_n - sum_m b_m
Lambda_mn`) and has since P26, for a gauge reason its docstring gives -- but its
*source*, `c_a = -i (dH/dk_a - eps_n dS/dk_a)|psi_n>`, holds `eps_n` as a **frozen
scalar**, so `d(eps_n)` is absent from the differentiated equation on one side and
present on the other. Removing that asymmetry changes no value and:

| | strain (US / PAW) | Raman (US / PAW) |
|---|---|---|
| committed (operator matrix, source frozen) | 1.30e-2 / 1.30e-2 | **1.2e-4** / 1.2e-4 |
| source as the matrix | **1.72e-4** / **1.71e-4** | 1.14e-3 / — |
| source as a traced diagonal | 9.60e-4 / — | 1.25e-3 / 5.53e-4 |
| operator *and* source as traced diagonals | — | fails (US) |

**It closes the strain coordinate and breaks the displacement one**, in every pairing
tried. So one of the two coordinates carries a further term that compensates it, and
finding *that* is what closes this phase; adopting the term on the strength of the strain
column alone would be a fit, and would regress a validated result. The Raman tensor is
therefore untouched and the strain third derivative stays refused.

**A plumbing gap found on the way in and fixed**: `electrostriction` called
`dielectric_tensor` and `strain_response` without `result.becsum`, so an ultrasoft or PAW
run reached them correctly only when the caller supplied a `strain=` computed elsewhere.

**Where to start next time.** The `b` partial is the whole of the residue and it is
measurable on its own -- `scratchpad/strain_b.py` prints it against its own finite
difference in about half an hour, and `scratchpad/strain_tail.py` carries
`_position_response` with the source form as a flag. The question to answer is which
coordinate has the compensating term, and the cheapest discriminator is that the
displacement's Raman check is a five-minute pytest run
(`test_nonlinear.py -k moving_overlap`) where the strain's is an hour.

### P45 — The Sternheimer response with two spin channels. ✅ DONE, three narrower cases refused.

*(Number to be checked at paste time: P44 was the last one in `PLAN.md` when
this was written, and two other tracks are landing in the same pass.)*

**`GAPS.md`'s "single widest guard" is narrowed, and what it was guarding was a
count.** One refusal in `require_a_sternheimer_regime` blocked *every* response
quantity for *every* spin-polarized system — no dielectric constant, no phonons,
no Raman, no strain response for nickel, iron, NiO or FeO, all of which have
validated LSDA ground states here (P9's eight benchmarks). The stated reason was
that `SternheimerSolver` takes **one** occupied-band count and slices
`psi[:, :, :nocc]` across the spin axis with it, and that all three callers —
`efield.py`, `phonon.py`, `strain.py` — derived that count as `nelec / 2`
themselves, which is right for an unpolarized insulator and wrong for a magnetic
one whose channels are filled to different depths.

**QE never meets the problem, and that is why there is nothing to transcribe.**
LSDA doubles `nks` there, so the spin channels are separate k-points and
`setup_nbnd_occ.f90` writes one `nbnd_occ(ik)` per k-point that already carries
the channel; `setup_alpha_pv` then maximises over an `et` array that spans both.
Here the channel is an *axis*, so what QE gets from its k index is got from a
per-channel count plus a mask — rule R2, the shape stays static and the deficient
channel's extra bands are masked rather than removed.

**The reason did not bind a metal and never had**, which is half of what this
phase found before writing anything: the smearing branch keeps every band in the
block (`nbnd_eff = nbnd`) and the occupation rides inside `dpsi`, so the slice is
never taken there. A spin-polarized metal was refused for a reason that applies
only to an insulator.

What changed:

* `occupied_counts(calculation)` — one count per channel, `(NINT(nelup),
  NINT(neldw))` for `nspin = 2` and `nelec / degeneracy` otherwise. The three
  callers consume it instead of deriving `nelec / 2`.
* `SternheimerSolver.__init__` keeps `max(counts)` bands and builds
  `projector_mask` from `band < counts[spin]`; `_alpha_pv`'s insulator branch
  maximises `eps^occ` over the channels, which is what QE's doubled-`nks`
  `setup_alpha_pv` computes.
* `project()`'s sharp branch masks in **two** places and they do different jobs.
  The *column* mask (over `m`) keeps `P_c^+` projecting out this channel's own
  occupied manifold — without it the projector also removes the bands the *other*
  channel fills, which is exactly the subspace the deficient channel's response
  lives in. The *row* mask zeroes the right-hand side of a band this channel does
  not occupy, so the CG returns `dpsi = 0` at iteration zero instead of solving
  `H - eps_n S` for an empty `n`, where the operator is not positive definite and
  the level shift does not make it so. The metal branch is untouched and
  bit-for-bit what P24c validated.
* The guard is **parameterised, not deleted** — `spin_polarized = True`, exactly
  as `metals = True` already works. Deleting the branch would have silently
  opened `electrostriction.py` and `nonlinear.py`, the two third derivatives,
  neither of which has ever been run with a spin axis. They still refuse, with a
  message that says what is now true: the *solve* is spin-polarized, the
  *assembly* is not.
* `_nint` — Fortran's `NINT`, half rounding *away* from zero, because
  `_fixed_occupations_spin` fills `int(floor(count + 1/2))` bands and Python's
  `round` is banker's rounding. An **odd** electron count with an **even**
  `tot_magnetization` gives a half-integer `nelup`, and `round` would then build
  the mask one band shallower than the weights the density was made with — in
  exactly that case and no other.
* `_require_a_gap_at_the_cut` — refuses a `nspin = 2` filling whose boundary
  lands inside a degenerate multiplet, checked on the converged spectrum
  because it is a property of the spectrum and not of the input.
* `_require_a_finite_kernel` (`efield.py`) — refuses a screened response whose
  induced potential came back non-finite, naming the LSDA kernel's second
  derivative and counting the grid points where `|m| >= |n|`.
* `reject_potential_only` at the top of the guard, so a potential-only meta-GGA
  is refused by name instead of surfacing as `v_of_rho` asking for a `tau`
  nobody passed (`GAPS.md` §3, "Linear response + meta-GGA", part (a)). Reused,
  not restated: it is the refusal `forces/energy.py` already exports for the
  stress, the dynamical matrix and the elastic constants.

**The validation is QE-free and it is a finite difference of the density**,
because comparing two totals would not isolate the term. `chi_0 dV` against
`(rho[V + h dV] - rho[V - h dV]) / 2h`, with a **spin-dependent probe** — twice
the amplitude in one channel and the opposite sign in the other, since `chi_0` is
block-diagonal in spin and a probe equal in both channels leaves the blocks
indistinguishable. The AFM hydrogen chain (`h-chain-afm`, a smeared metal) and
the triplet oxygen molecule (`o2-fixed-lsda`, the sliced branch, ultrasoft) are
the two cases. The third check is the identity: silicon run as `nspin = 1` and as
`nspin = 2` with no magnetization must give the *same* `epsilon_infinity`, and it
is run end to end rather than on `chi_0` because the spin sum and the screening
kernel are where a factor of two would hide and `chi_0` is blind to both.

**The obvious insulator is the wrong one, it fails silently, and that is the
first of the phase's two findings.** `o-atom-fixed-lsda` — the oxygen atom
`GAPS.md` points at — cannot be used: at `tot_magnetization = 2` its minority
channel holds two electrons, so its filling cuts the triply degenerate 2p shell,
whose two levels are **1.4e-14 Ry** apart. What that does was *measured* rather
than argued: with the check bypassed **the CG converges normally** — 42
iterations to a residual of 5e-12 against a 1e-11 threshold — and the `chi_0` it
returns is **100 per cent** away from a central difference of the density (1.24,
1.02, 1.01 and 1.01 relative, for probes at Miller (1,0,0), (0,0,1), (1,1,0) and
(1,1,1)). The difference re-selects which member of the shell falls below the
cut, because the perturbation splits it at first order; the solve keeps the
member the eigensolver handed it, which is an arbitrary one. It is the **same
multivaluedness the residual solver is diagnosed for** (§3's closed
`occupations = 'fixed'` + `nspin = 2` entry), one layer up. A stalling solve
would have announced itself; this one does not, which is why it is a refusal
(`DEGENERATE_CUT_RY`) and not a warning. The oxygen *molecule* is the case that
works: twelve electrons and `tot_magnetization = 2` give seven up and five down,
both closed shells — measured gaps 0.438 Ry up and 0.517 Ry down.

**The second finding is what stopped the screened response on the one
magnetic-insulator cell there is here, and it is not in `pypresso/response/` at
all.** (One cell: an AFM bulk insulator whose magnetization stays below its
charge everywhere was not run, so this is a diagnosis of triplet O2 rather than
of the whole regime.)
`dv_of_drho` is one `jvp` of `v_of_rho`, so for `nspin = 2` it is the **second**
derivative of the LSDA exchange-correlation energy in the two channel densities
— and that diverges wherever a channel density reaches zero, which a plane-wave
magnetization does in vacuum. The two counts agree exactly on triplet O2 in a
10-bohr box: **1504 of 91125 grid points have `|m| >= |n|`, and `dv_of_drho` has
1504 NaN**. The *value* of `v_xc` is fine — `xc_lsda` clips `zeta` to [-1, 1]
and QE does the same — but the clip's own tangent is zero and `rho^(4/3)`'s
second derivative is infinite at the channel it zeroes, so the product is
`inf * 0`. It is the **`abs` trap of P28a in a fifth place**, one derivative
further out. Two obvious repairs were tried and **neither works**: pulling the
clip inside to `1 - eps` for `eps` of 1e-12, 1e-10 and 1e-8 leaves all 1504
points NaN and turns the largest entry into `inf`, so what diverges is the
kernel itself and not only the clip. Making `pypresso/xc`'s spin branch twice
differentiable at a fully polarized point is the missing piece; until then a
magnetic system with vacuum is **refused by name** in the response loop
(`_require_a_finite_kernel`), because a NaN `|ddv_scf|^2` never satisfies
`change < tr2` and the loop would otherwise spend its budget and report
`converged = False` on a tensor that is not a number. `chi_0` is unaffected and
is validated. **The reference for the day it is fixed is committed**:
`reference.out.ph-o2-fixed-lsda`, generated with the vendored `ph.x`, which
computes this quantity happily — `phq_readin.f90:546` refuses an electric field
only for *noncollinear* magnetism and `:957` only for a smeared or tetrahedron
metal, so LSDA is allowed. It gives diag(1.110916, 1.110916, 1.198005).

**Descoped and refused by name**: `tot_magnetization` with a *smearing*, which is
QE's `two_fermi_energies`. `Smearing` carries a single scalar `ef` and
`smearing_of` reads `result.fermi_energy`, which in that case is the **mean** of
the two levels — a number QE prints only so the field is not NaN. Every weight in
`orthogonalize`'s smearing branch would be evaluated at a level neither channel
has. Giving `Smearing.ef` a spin axis is the missing piece and the message names
it. Also refused by name: **Born effective charges** for `nspin = 2` (the
dielectric constant is a spin sum and is validated; `dF/dE` goes through the force
functional's `becsum`, whose spin axis `response/born.py` has never been run
with), the **dynamical matrix** and the **strain response** (their `nocc` is fixed
but their second-derivative assembly — `non_variational_response`,
`_multiplier_response`, P28's `wg` / `2 wk` split — has not been checked per
channel), and the two third derivatives.

**One more hole closed for free**: `reject_potential_only` is now called at the
top of the Sternheimer guard, so a Tran-Blaha or Becke-Johnson run asking for any
response is refused by name instead of surfacing as `v_of_rho` asking for a `tau`
nobody passed. It is the refusal the stress, the dynamical matrix and the elastic
constants already make, reused rather than restated.

Measurements, all from `tests/regression/test_lsda_response.py`:

| what | reference | agreement |
|---|---|---|
| `chi_0`, AFM H chain, `nspin = 2` metal | central difference of `rho`, `h = 3e-4` | **1.8e-6** relative |
| `chi_0`, triplet O2, `nspin = 2` insulator (ultrasoft) | central difference of `rho`, `h = 1e-4` | **1.1e-6** relative |
| `epsilon_infinity`, silicon, `nspin = 2` at `m = 0` | the same cell at `nspin = 1` | **6.2e-14** (13.806646105 both) |
| O2 ground state, `nspin = 2` fixed occupations | the vendored `pw.x` | **1.1e-9 Ry** (-63.36308378108187 against -63.36308378) |
| the whole `nspin = 1` response stack | `tests/regression/test_response.py` | **31 passed**, unchanged |
| `epsilon_infinity`, triplet O2, `nspin = 2` (ultrasoft) | the vendored `ph.x`, `epsil = .true.` → diag(1.110916, 1.110916, 1.198005) | **refused**: 1504 NaN in `dv_of_drho` at the 1504 points where `\|m\| >= \|n\|` |

---


### P46 — The force and the stress of a spinor. ✅ DONE.

**The functional was already written; what was missing was the layout.**
`GAPS.md` §3 sized this at "two substitutions in one functional — the nonlocal
quadratic form with `dvan_so`, and the constraint's `<psi|S|psi>` with
`qq_so`". The physics of that estimate is right and the size of it was not, for
a reason `CLAUDE.md` states in its own conventions: `nspin`, `npol` and
`nspin_mag` are three different numbers. `FrozenState.wavefunctions` is
documented as `(nspin, nk, nbnd, npwx)`, and a spinor does not fit that shape —
it is `(1, nk, nbnd, 2 npwx)`, one channel of two-component states, so the
*kinetic* term has to read `state_kinetic` (`|k+G|^2` in the coefficient
vector's own layout) as well. Three lines, in three places, and only one of them
is a matrix.

The result: `noncolin = .true.`, with or without `lspinorb`, has forces, a
stress and therefore a relaxation, on norm-conserving, ultrasoft and PAW
datasets. Bcc iron's noncollinear ground state can be relaxed.

**The validation is in the order that isolates terms.**

*The energy identity first*, because `energy_at` has to **be** the total energy
before anything differentiates it — and because a `deeq_nc` used where the bare
`dvan_so` was meant double-counts `int V_eff Q_ij` and is invisible in a force
whose reference is a finite difference of the same wrong functional:

| case | regime | `energy_at` − SCF total |
|---|---|---|
| `h-chain-90deg` | norm-conserving, `nspin_mag = 4`, `nosym` | 1.3e-15 Ry |
| `h4-noncolin-force` | norm-conserving, `nspin_mag = 4` | 4.4e-16 Ry |
| `pw_spinorbit/spinorbit` | ultrasoft + SOC | 1.4e-14 Ry |
| `pt2-soc-force` | ultrasoft + SOC | 2.8e-14 Ry |
| `pw_spinorbit/spinorbit-paw` | PAW + SOC | 7.5e-10 Ry |
| `pt2-soc-paw-force` | PAW + SOC | 9.9e-10 Ry |

PAW is the loosest of the six and that is the radial one-centre quadrature
rather than the spinor branch: a *collinear* PAW run has the same gap.

**What that identity is blind to, and it is the finding to carry.** The
constraint term `sum w eps (<psi|S|psi> - 1)` is *identically zero* at the
converged state, so the 1.4e-14 on an ultrasoft spinor says nothing whatever
about `qq_so` — only its **derivative** is the Pulay force, and the identity
cannot see a derivative. The finite difference below cannot see it either, in a
different way: it checks the gradient of the functional **as written**, so a
constraint that is wrong but self-consistent passes it exactly. The `qq_so` half
is therefore validated by exactly one thing — the `pw.x` force on
`pt2-soc-force` and `pt2-soc-paw-force` — and that is worth naming, because it
is the same shape as P35's "no symmetry check catches its absence". It *is*
sufficient (7.5e-6 and 7.3e-7 on terms of 1.7 Ry/bohr), but it is one check and
not three.

*Then a finite difference of that same frozen energy*, which is P39's anchor and
needs no Fortran at all: a central difference of `energy_at` under a
displacement against `jax.grad` of it, on the four-atom `nosym` chain, agrees to
**6.2e-9 Ry/bohr** on forces of 2.2e-2 — the difference's own floor at
`h = 1e-3`. On the *undisplaced* chain the same comparison agrees to 7.2e-13,
with both sides at 1e-7, which is the statement that an equally-spaced chain has
no force.

*Then the sum rule, which is run and is not rested on.* `sum_a F_a` is 1.3e-7
Ry/bohr on the chain and 3–4e-6 on platinum, against forces of 2.2e-2 and 5.2e-2.
An atom sum is blind to a transfer *between* atoms — which is exactly how P28a's
`abs` trap survived two identities — so it is a diagnostic and the finite
difference above is the anchor.

*Then `pw.x`*, which computes both quantities for a spinor run (`tprnfor`,
`tstress`):

| case | what it adds | ΔE (Ry) | max ΔF (Ry/bohr) | max Δσ (Ry/bohr³) |
|---|---|---|---|---|
| `h4-noncolin-force` | norm-conserving, `nspin_mag = 4` | 3.6e-9 | **8.9e-7** | 1.8e-9 |
| `pt2-soc-force` | ultrasoft, `lspinorb`, `qq_so` | 1.6e-8 | **7.5e-6** | 1.2e-6 |
| `pt2-soc-paw-force` | PAW, `lspinorb` | 1.3e-9 | **7.3e-7** | 3.5e-7 |
| `pw_spinorbit/spinorbit` | ultrasoft LDA, one atom | 1.3e-8 | — | 9.8e-8 |
| `pw_spinorbit/spinorbit-pbe` | ultrasoft PBE | 3.8e-9 | — | 4.4e-7 |
| `pw_spinorbit/spinorbit-paw` | PAW PBE | 8.4e-9 | — | 2.5e-7 |

against `FORCE_RY_BOHR = 1e-4` and `STRESS_RY_BOHR3 = 1e-4`. Two things about
that table are worth reading twice. The three `pw_spinorbit` rows needed **no
new reference**: QE's own spin-orbit test-suite inputs already carry
`tstress = .true.`, so a PAW spin-orbit stress was available all along and the
refusal was the only thing in the way. And the ultrasoft force's 7.5e-6 is not
loose — that case's individual force terms are **1.7 Ry/bohr** and cancel to
0.052, so 7.5e-6 is four parts per million of what is being subtracted.

**The k-set is a wedge and the force is symmetrised as a polar vector with the
magnetic group.** `Calculation.symmetries` is already `sgam_at_mag`'s group when
`nspin_mag = 4` (P17), and `compute_forces` already ends in `symvector`, so the
reuse is the whole of it; a time-reversal operation acts on a *polar* vector
through its spatial part alone, which is what makes the reuse right rather than
convenient. Measured on `pt2-soc-force`, whose unshifted 2x4x4 grid reduces 32
points to 14: the wedge sum agrees with the same run at `nosym` on the whole
closed grid to **2.3e-7 Ry/bohr** on the force and **6.3e-9 Ry/bohr³** on the
stress, with the two SCF totals themselves 2.4e-12 Ry apart. The crystal's own
`F_x = -F_z` holds to 1e-12 on both — on the closed grid *without* the average,
which is what "an unshifted Monkhorst-Pack grid is closed under the point group"
means, and on the wedge only because `symvector` put it there.

**A spinor stress on a slab does not fit in memory here, and the number is
worth having.** The first ultrasoft spin-orbit force case tried was bismuthene
displaced in-plane (`Bi.rel-pbe-dn-rrkjus`, 15 valence electrons, `ecutrho =
160` on a 20-bohr vacuum cell, so a dense grid of about 45x45x81). The SCF
itself is what already runs in the test suite; the *gradient* took free memory
from 24 GB to 0.65 GB and was killed, three times, on this 30 GB machine. The
cost is the augmentation table `Q_ij(G)` — `nh^2 x ngm` per atom, with `nh` in
the twenties for a fully-relativistic `dn` dataset — carried through the
backward pass. The doubled fcc platinum cell replaced it: same physics
(ultrasoft, `lspinorb`, an augmentation charge, a displaced atom), a cell of 204
bohr^3 instead of 1770, and it runs in 33 s. A user's first encounter with this
gap will be a *slab*, so it belongs in the record rather than in an anecdote.

**Relaxation comes with the force and needed nothing.** The BFGS of P15 runs
on `pt2-soc-force` unchanged and puts the displaced platinum back: **8 ionic
steps**, `max |F|` from 0.0524 to **8.96e-5 Ry/bohr**, and the two atoms
**0.499945** of a cell apart against the half the crystal's symmetry requires.
What is checked is the *separation* and not either position, and the reason is
worth recording: subtracting the mean force leaves a rigid translation of the
whole crystal free and BFGS uses it — this run ends with both atoms moved 0.015
of a cell along `a1`. Asserting on an absolute coordinate would be asserting on
which zero mode the optimizer happened to walk along.

**What stays refused, and why each one is a different missing term.**

* **The analytic force and stress expressions.** `force_us`, `stres_knl` and the
  rest are a transcription that shares no machinery with the functional and has
  no spinor form — `force_us` alone would need `deeq_nc`, `becsum_nc` and
  `qq_so` threaded through a second time. Both already refused `noncolin` by
  name in their own words and still do; the refusal that was lifted was the
  functional's, and a refusal lifted in one place and not its sibling is the
  defect this whole pass is about.
* **The Sternheimer response**, and so phonons, `epsilon`, Born charges and
  everything above them. `incdrhoscf_nc`/`set_int3_nc` are a second
  implementation rather than a spin axis on this one, and
  `require_a_sternheimer_regime` is not this phase's guard to lift.
* **Elastic constants and electrostriction**, which reach `energy_at`
  *directly* (`response/elastic.py:181`) and never see the Sternheimer guard at
  all. This is why the spinor path is **opt-in** rather than simply allowed:
  `energy_at`, `frozen_energy` and `strained_energy` all default to refusing,
  and the force and the stress pass `spinors=True`. Deleting the guard would
  have opened a third derivative for a regime whose *first*-order wavefunctions
  do not exist.
* **The force on an atom of a spin spiral**, by name and for a new reason. A
  spiral is `noncolin` with `spiral_q` on top, so it used to be caught by the
  same refusal as everything else; after the narrowing it would have walked into
  the spinor projector contraction with a `(2 nk, npwx, nkb)` `vkb` against `nk`
  rows of coefficients and died on an einsum shape rather than on a refusal.
  `reject_spinor_spiral` is what closes that, and the missing term is real: the
  two components live on `k ± q/2` and the nonlocal term needs the *pair* of
  projectors — which `spiral_energy` does write, in the other coordinate.
  `dE/dq` is what a spiral has instead of a force.
* **The matrix orthonormality multipliers** (`_constraint_energy`), which
  contract the scalar `qq` and would need `qq_so` with a spin pair on `Lambda`
  as well. Only the ultrasoft *second* derivative asks for them
  (`response/born.py`), and that path is refused for `noncolin` already.
* **A magnetic field or a constrained moment**, unchanged: the field's energy is
  deliberately outside the reported total, so the converged state is stationary
  for a different functional than the one being differentiated (P18, P21).

**The trap, and it is a small one worth writing down.** `dvan_so` is the *bare*
`D` — the spinor twin of `dij`, taken for the same reason the collinear branch
takes `dion` and not `deeq`: the self-consistent `int V_eff Q_ij` part is
already inside the augmented density. The spin transform is **not** a detour
around that accounting, because `newd_nc` sandwiches the scalar integrals
between `fcoef` and *adds* them to `dvan_so`, so the split survives one spin
index up. Taking `deeq_nc` instead double-counts, and the only check that sees
it is the energy identity above — a finite difference would agree with it
perfectly.

---


### P47 — The Kubo Berry curvature of a real crystal. ✅ DONE, ultrasoft/PAW refused by name.

`pypresso/topology/berry.py` has carried a `kubo` curvature method since P16 and
it has only ever been reachable from a tight-binding model. Its refusal for a
plane-wave calculation said the velocity operator "needs `d(vkb)/dk` and the
k-dependence of the plane-wave sphere" and pointed at P11. **P24 wrote both**,
and the refusal had been stale for two phases: `VelocityOperator` is one
`jax.jvp` of `H(k)` at a **frozen** sphere, which is exact on each piece of a
membership that is piecewise constant in `k`, and `VelocityOperator.projectors`
is `gen_us_dj`/`gen_us_dy`'s derivative about the atom's own centre. So the
smooth `Omega(k)` map of a real crystal — the anomalous-Hall picture, what
anyone wants beside the integer — was refused for a reason the repo had already
satisfied. It is in now: `pypresso/topology/kubo.py`.

**The route GAPS.md offered first is the illegal one.** A
`PlaneWaveStates.hamiltonian`-shaped adaptor would let the existing
`_kubo_point` run unchanged, and `_kubo_point` forms the dense `H(k)` and calls
`eigh` on it — `npw^2`, which `CLAUDE.md` rules out outright. So the sum is
written as **band matrix elements between the states an NSCF already
produced**: one `jvp` per crystal direction gives `v_a|psi_n>` for every band at
every k-point, and the expression is a contraction of that against the same
states. Nothing dense is ever formed.

**The expression is the generalised one and its band index is not free.**
Differentiating `H|m> = e_m S|m>` and projecting on `<n|` gives
`<n|S|d_a m> = <n|(dH/dk_a - e_m dS/dk_a)|m> / (e_m - e_n)`; carrying that
through the curvature leaves

    Omega_n^{12} = -2 Im sum_{m != n} A^1_nm A^2_mn / (e_n - e_m)^2,
    A^a_nm = <psi_n| dH/dk_a - e_n dS/dk_a |psi_m>

with **`e_n` — the band whose curvature is being computed — in both factors**,
not `e_n` in one and `e_m` in the other. The `dS/dk` piece is identically zero
for a norm-conserving dataset, which is exactly why **ultrasoft and PAW are
refused by name**: no norm-conserving validation can see whether the convention
is right, and an off-diagonal element with a moving `S` is the kind of thing
that comes out plausible when it is wrong. The term is written (one `jvp` of
`s_psi`, `VelocityOperator.apply_s`) and unvalidated, and there is a second one
beside it — the augmentation charge's own `k`-derivative, which an overlap
between two *different* k-points needs as `q_ij(b)` rather than `qq`.

**Three things validate it and they are independent of each other.**

*The operator.* `<psi_m|dH/dk_a|psi_n>` against a **central finite difference**
of the same matrix element at frozen sphere and frozen states —
**1.8e-9** at `eps = 1e-4` (1.8e-7 at 1e-3, the exact factor of 100 a
second-order difference owes) on a matrix whose largest element is 1.05. This
is QE-free and it isolates the `jvp` *and* the crystal-direction tangent, which
is the reciprocal lattice vector `bg[d]` and would otherwise be a wrong-units
curvature no downstream check distinguishes from a wrong assembly. `dH/dk` is
Hermitian to **3.8e-15**; `dS/dk` is exactly 0.0.

*The assembly.* Against the Fukui-Hatsugai-Suzuki lattice flux, which shares no
machinery with it — determinants of overlaps between separate diagonalisations,
differentiating nothing. **AlAs** is the crystal, because zincblende has time
reversal and no inversion centre, so `Omega(k)` is nonzero pointwise (order 1 in
crystal-k-area units) while the Chern number is zero. Two forms:

- a **centred plaquette shrunk around one k-point**, converging onto the
  pointwise value at `k = (0.1875, 0.3125, 0)`, against `Omega_kubo = 0.964705`
  at `nbnd = 30`: `h = 0.08` → 0.93502 (3.1e-2), `0.04` → 0.96056 (4.3e-3),
  `0.02` → 0.96784 (**3.2e-3**), `0.01` → 0.97956 (1.5e-2). Second order down to
  `h = 0.02` and then a **noise floor** — the flux is 1e-4 rad at that size and
  dividing an eigensolver's round-off by `h^2` amplifies it — so 3e-3 relative
  is what the check establishes and the mesh cannot be refined past it.
- the **whole mesh plaquette by plaquette**, one 24x24 Kubo mesh at the
  plaquette centres integrated by the midpoint rule over the plaquettes of
  coarser FHS meshes (`nbnd = 30`): `max|integral - flux|` of 9.49e-3 (4x4),
  3.59e-3 (6x6), 1.63e-3 (8x8), 6.56e-4 (12x12), **1.45e-4** (24x24) — 65x
  better over a sixfold refinement, faster than the `h^2` both sides' errors
  are — against a largest flux that itself shrinks from 3.97e-2 to 2.29e-3, so
  the *relative* figure improves fourfold (0.239 → 0.063) and then flattens onto
  the same round-off floor from the other side.

*The scope of the claim.* Everything measured is **scalar norm-conserving**.
The spinor path (`npol = 2`) runs — the coefficient layout flows through the
same contraction and no new term appears, unlike ultrasoft's `dS/dk` — and it is
**unmeasured**; it is not refused, and nothing here claims it.

*The symmetry.* Silicon has time reversal **and** inversion, so `Omega(k)` must
vanish **pointwise** and not merely integrate to zero: **3.5e-5** on a 4x4 mesh
at `nbnd = 12`, against a per-band curvature of 5.7 on the same mesh. A
curvature that is zero for the wrong reason is the trap, and the per-band scale
is what says this one is a cancellation rather than an empty calculation.

**The sum over empty states is truncated and the truncation is reported**, the
way P37 reports `static_residual`: `BerryCurvature.truncation` is the largest
shift in `Omega(k)` when the highest empty band is dropped, over `max|Omega|`,
with `truncation_abs` beside it. It is a number to read and never a knob. At
`k = (0.1875, 0.3125, 0)`, from one diagonalisation at `nbnd = 45` truncated
after the fact: 0.978886 (12), 0.965663 (20), 0.964705 (30), 0.961639 (45) —
1.8% between 12 and 45 and 0.3% between 30 and 45. **On the mesh comparison the
truncation is worth far more than that and it crosses over** (measured on a
16x16 Kubo mesh integrated over a 4x4 FHS mesh): at `nbnd = 8` the sum is 1.70
relative against the flux, at 12 it is 0.20, at 16 0.22, at 20 0.29, and from 25
up it settles at 0.265 — the accidental agreement at 12 is a
cancellation between the truncation error and FHS's own coarse-plaquette error,
and reading it as convergence is the trap the two-axis separation exists to
avoid. **Where symmetry forces `Omega` to vanish the ratio is noise over noise
and means nothing** — silicon reports `truncation = 0.98` beside a
`truncation_abs` of 3.4e-5 — so `truncation_abs` is the one to read there.

**`curvature_by_band` is gauge invariant only for a non-degenerate band**, and
that is P36's degenerate-multiplet finding one quantity over. Inside a
degenerate multiplet the eigensolver's arbitrary rotation moves the members'
values and only their sum is defined; the *manifold* total never has the problem
because it is built from occupied/empty pairs alone, so an intra-manifold
degeneracy never enters it. Measured directly rather than argued: at
`k = (0, 0.375, 0)` the second and third valence bands of AlAs are degenerate to
**9.4e-16 Ry**, and a unitary mixing of the pair takes their curvatures from
`(+0.203989, -0.218475)` to `(+0.273508, -0.287993)` — **0.0695** — while the
manifold total stays at `-0.0144856` to **1.1e-15** and the multiplet's own sum
to 1.0e-15 (the Kubo weight `1/(e_n - e_m)^2` is constant across the block,
which makes the sum a trace over it).

**`fhs` stays the default and stays the only method a Chern number is read
from.** The `1/(e_n - e_m)^2` denominator is what design rule D4 forbids, and
the Brillouin-zone sum is an ordinary Riemann sum that converges to an integer
without ever being one. What `kubo` is for is the map.

Three smaller things the phase left behind:

- **The state set had to grow two fields and one of them must be dropped on
  `select`.** `PlaneWaveStates` now optionally carries `all_coefficients` (the
  whole diagonalised band set, where everything else here is a property of the
  occupied manifold alone) and `velocity`. `select` sets **both to `None`**: the
  velocity operator is built on a `Calculation` at the *whole* k-list, its
  `vkb(k)` and `|k+G|^2` indexed by the original k-axis, so a sliced state set
  no longer lines up with it and a Kubo call on the selection would differentiate
  at the wrong k-points. Dropping them makes the next call refuse by name; slicing
  them would have been silent.
- **The state-source protocol gained an optional third argument.**
  `states(points, keep_projectors=False, keep_velocity=False)`, and
  `chern_number` passes `keep_velocity` **only when it is true**, so a source
  written to the two-argument signature still satisfies the protocol.
- **`DFTSource.states` already had everything the operator needs** — the frozen
  potential, `ddd_paw` and `ns` are the same three things it hands
  `calculation.hamiltonian` — so the operator is built from the arguments that
  were already there rather than rebuilt.

**Refused by name**: ultrasoft and PAW (the `e_n dS/dk` off-diagonal, plus
`q_ij(b)`), and everything `DFTSource` already refuses inherits unchanged —
`nspin = 2`, meta-GGA, a PAW dataset without `becsum`, a DFT+U one without `ns`,
and a manifold that is not gapped (`_check_gap`, which is also what stops a
*metal*: the Kubo expression here has no occupation factors and there is nothing
in it for a partially filled band).

---


### P48 — Two things Elk has and `pw.x` does not: the effective mass, and site-resolved `<L>`, `<S>`, `<J>`.

Chosen from a survey of Elk's task list against QE 7.5 (`ELK-FEATURES.md`, which
keeps the four that were not taken and the validation route each would need).
The filter was: no counterpart in `pw.x` or its post-processing tools **verified
by grep over the vendored Fortran**, NSCF cost or less, and an assembly of
machinery that already exists rather than a second implementation of something
validated.

---

#### P48a — The effective mass tensor. ✅ DONE.

`pypresso/response/effmass.py`. Elk's task 25 (`effmass.f90`); QE has nothing —
`grep -ri "effective mass"` over `PW/src` and `PP/src` is empty.

    (1/m*)_ab = (1/2) d^2 eps_n(k) / dk_a dk_b,

in units of `1/m_e`, the half being Rydberg atomic units: a free electron has
`eps = |k|^2` because `hbar^2/2m_e` is exactly 1 Ry bohr^2, so the free-electron
tensor is the identity and nothing normalises it.

**The autodiff route stops one derivative short, and saying where is the point.**
The *first* derivative is exact and analytic — `d(eps_n)/dk` is the generalised
Hellmann-Feynman expression built from one `jvp` of `H(k)` at a frozen sphere
(P24, rule D2). The *second* is not available the same way, for three reasons
that each had to be checked rather than assumed:

- differentiating that expression again **at frozen states** gives only
  `<n|d^2H/dk_a dk_b|n>` and drops the whole `k.p` sum, which for silicon's
  `Gamma_2'` band is most of the answer rather than a correction;
- the first-order state `|dn/dk>` is what the Sternheimer solver produces, and
  it **cannot supply this one**: `P_c` removes the whole occupied manifold,
  which is right for a density response — the occupied-occupied pairs cancel
  there — and wrong for one eigenvalue's second derivative, where they do not.
  And the band whose mass is wanted is usually **empty**, where `H - eps_n S` is
  indefinite and the projected CG has nothing to converge to;
- differentiating through the eigensolver is rule D4 and P22's 109%.

So the construction is **the first derivative by `jvp` and the second by one
central difference of it**, which is strictly better than Elk's difference of
*eigenvalues*: six stencil points against twenty-seven, `O(h^2)` on a quantity
exact at each point, and no Vandermonde fit. Elk's route is implemented beside
it (`method="eigenvalue"`) because it shares **nothing** with the velocity
operator, which makes it the independent check on the operator itself.

**The finding is a stencil that must not contain its own centre**, and it is a
plane-wave trap rather than a numerical-analysis one. The obvious three-point
second difference `[eps(+h) - 2 eps(0) + eps(-h)]/h^2` is wrong at a
high-symmetry k-point because the **sphere is rebuilt at every k** and that is
exactly where a shell of `G` sits on the cutoff. Measured on two-atom silicon at
`ecutwfc = 30`: `Gamma` holds **725** plane waves and *every* displaced point
holds **733**, whatever the displacement. The centre eigenvalue is therefore
variationally high by a fixed basis-set offset `delta ~ 1.2e-6 Ry`, and the
curvature inherits `-delta/h^2` — an error that **grows** as the stencil
shrinks. Measured growing by exactly four per halving, 2.1e-4 at `h = 0.05` to
**3.0e-2** at `h = 0.00625`, while the velocity route converged over the same
range. It is **not a PAW effect**, which is how it first presented: a
norm-conserving LDA silicon at the same `ecutwfc = 30` has the identical 725/733
split, and the same cell at `ecutwfc = 12` has none, which is why every earlier
test was clean.

The cure is a stencil in which every point is displaced — the diagonal entries
from the four axial points at `+-h` and `+-2h`,
`[eps(2h) + eps(-2h) - eps(h) - eps(-h)]/3h^2`, and the off-diagonal ones from
the four-point mixed difference, which was already centre-free. **`delta` is
then the stencil's outermost displacement rather than its step**, so that both
routes sample the same range of `k` and their agreement is about the operator
rather than about how far each one reached: leaving that out is worth **0.16**
on silicon's `Gamma_1`, where `+-0.05` is far enough to leave the parabolic
region, and the truncation estimate is what said so. With that the
two routes converge together at `O(h^2)`: their difference on `Gamma_1v` falls
7.8e-4 → 1.9e-4 → 3.8e-5 → **8.6e-6** over the same four halvings. **The
velocity route never had the problem** — its `jvp` freezes the sphere and its
difference is between two *displaced* points holding the same 733 — which is one
more reason it is the default.

**The truncation is removed and then reported.** Both routes are `O(h^2)`, and at
Elk's own `deltaem = 0.025` that error is not small: silicon's `Gamma_2'`
conduction band comes out at 5.131 against a converged 5.303, **3 per cent**.
So the stencil is run at `h` and `h/2` and combined as `(4 M(h/2) - M(h))/3`,
and what the two levels differ by is kept as `EffectiveMass.truncation` — P47's
rule, and P37's before it.

**Degeneracies are refused by name and the multiplet sum is offered instead.**
A per-band second derivative is not a property of the band inside a degenerate
multiplet; the eigensolver's arbitrary rotation rotates it (rule D4). `sum_n
d^2 eps_n/dk_a dk_b` over the manifold is the trace and is invariant, so that is
what is reported. Elk prints the per-band numbers regardless, and its own output
is the evidence: silicon's threefold `Gamma_25'` gets **-5.8938, -3.7690,
-3.7690** from Elk, three different tensors for three states whose energies
agree to 5e-8 Ha.

**Against Elk** — the vendored binary, all-electron LAPW, PBE, same cell
(`a = 10.26` bohr), against pypresso on the **PAW** `Si.pbe-n-kjpaw_psl.0.1`
dataset at `ecutwfc = 30`, both at Elk's default stencil. The quantity compared
is Elk's `d` matrix, which is in Hartree atomic units and is therefore
`inverse_mass` exactly:

| at `Gamma` | Elk (LAPW) | velocity | eigenvalue | agrees |
|---|---|---|---|---|
| `Gamma_1v`, band 1 | 0.8603044 | 0.8600902 | 0.8600781 | **0.02%** |
| `Gamma_25'v`, bands 2-4 (sum) | -13.4317801 | -13.6015326 | -13.6040409 | 1.26% |
| `Gamma_15c`, bands 5-7 (sum) | 7.7424153 | 7.5886921 | 7.5892084 | 1.99% |
| `Gamma_2'c`, band 8 | 5.8460134 | 5.8671956 | 5.8682740 | 0.36% |

as masses, `Gamma_1v` is **1.16267** `m_e` against Elk's 1.16238 and
`Gamma_2'c` **0.170439** against 0.171057. The two pypresso routes agree with
each other to **1.2e-5** on `Gamma_1v` and 1.1e-3 on `Gamma_2'c` — the same
distance from Elk on every row, which is what says the residual is Elk's drift
and the pseudopotential rather than either route's arithmetic — and the
tensors are isotropic to **2.9e-8** with nothing imposing it — the cubic
symmetry check, which neither route is told about.

**Elk's own number is not converged, and that is the finding the comparison
produced.** Scanning `deltaem` at `Gamma`, Elk's `Gamma_1v` goes 0.8583470,
0.8594683, **0.8603044**, 0.8642146, 0.8696507 for `h` = 0.1, 0.05, 0.025,
0.0125, 0.00625 — rising toward a minimum-error point at its **default** and
then **diverging**, which is the `delta/h^2` signature above. `Gamma_2'c` is
worse: 4.5966, 5.5284, 5.8460, 6.0222, 5.9158, non-monotone with no limit.
`effmass.f90`'s stencil includes its centre and Elk's APW basis is `|G+k| <
gkmax`, so it has the same trap; fitting the offset gives `delta ~ 1e-6` Ha,
the same order measured here. pypresso's velocity route over the same range is
0.8596006, 0.8599815, 0.8600630, 0.8600562 — monotone, converged to five
digits. **So the 0.02% agreement at Elk's default is partly luck about where the
drift has got to**, and the honest statement is that the two codes agree to
0.02-2% while only one of them converges.

The residual 1-2% on the multiplet sums is not pypresso's basis: `ecutwfc` 30 →
40 → 50 moves `Gamma_1v` by 4.6e-5 and the `Gamma_25'` sum by 3.6e-3, so it is
Elk's own drift plus the genuine pseudopotential-against-all-electron
difference.

**Timing**, one core each, same machine. Elk: 1.36 s for the ground state and
**1.08 s** for task 25 (27 k-points, all states). pypresso PAW: 4.3 s for the
SCF, **4.3 s** for the velocity route warm (4.9 s cold, compilation included)
and 6.0 s warm for the eigenvalue route. About 4x, and the comparison is not
like-for-like — a 733-plane-wave basis against LAPW's much smaller one, and 13
k-points against 27. The statement worth keeping is the absolute one: **both
codes compute this in seconds**, which is what makes it the cheapest derived
quantity in the package.

---

#### P48b — Site-resolved `<L>`, `<S>` and `<J>`. ✅ DONE, two regimes refused.

`pypresso/projwfc/angular_momentum.py`. Elk's tasks 15/16 (`writelsj.f90`,
`dmatls.f90`, `gendmat.f90`). **QE has `lorbm`** — `PW/src/orbm_kubo.f90`, the
*cell's* orbital magnetization by the modern theory — and nothing resolved by
atom, so the README column is `partly` and the new part is the site
decomposition: which atom carries the orbital moment, which `lorbm` cannot say.

One site density matrix per atom and per shell,

    rho^a_{(m s),(m' s')} = sum_{nk} wg_{nk} c^a_{ms,nk} conj(c^a_{m's',nk}),

with `c = <phi|S|psi>` the projection `projwfc.x` already builds, and then the
two traces `dmatls` takes of it: `<L_i> = sum_s Tr_m (L_i rho_ss)` and
`<S_i> = (1/2) sum_m Tr_s (sigma_i rho_mm)`. Everything before the traces
existed — the orbitals are `build_atomic_projectors`'s Löwdin-orthogonalised
set, the same one DFT+U measures `ns` with, and `S` is the calculation's own.

**`L` had to be written in the basis the code actually uses.** `L_z` is diagonal
on `Y_lm` and is *not* diagonal on the real harmonics `ylmr2` builds, so the
matrices are the complex ones conjugated with `rot_ylm` — the unitary
`pypresso/pseudo/spinorbit.py` already builds for `fcoef` —
`L^real = A^T L^complex conj(A)`. The result is **purely imaginary and
antisymmetric**, which is a consequence rather than a convention (`L` Hermitian,
the harmonics real) and is therefore the cheapest test that the transform is
right; `[L_x, L_y] = i L_z` and `L^2 = l(l+1)` are the others, and all four hold
to **1e-13** for `l = 0..3`.

**The validation is identities and rotations, not another code's floating
point**, and deliberately: Elk integrates over a muffin tin of a stated radius
and this projects onto an orbital set, so the two differ *by definition* the way
Löwdin and muffin-tin charges do. What is checked instead:

- **`<L>` is quenched to zero without spin-orbit coupling** — nothing in a
  scalar-relativistic Hamiltonian locks the orbital moment to the lattice.
  Measured at **1.7e-16** on silicon, which is the "vanishes pointwise" check of
  P47's curvature and is what would catch an `L` built in the wrong basis: a
  wrong unitary gives a *small* non-zero answer, not an obviously wrong one.
- **Nickel's orbital moment appears when the coupling does**: `<L_z>` =
  **0.0364767** hbar on a fully-relativistic norm-conserving dataset, against a
  measured 0.05 mu_B — the underestimate GGA is known for. The ratio is the
  better number: `|L|/|S| = 0.11665` against an experimental `m_L/m_S` of about
  0.1, and `<L>` is parallel to `<S>`, which is Hund's third rule for a
  more-than-half-filled shell.
- **`<L>` rotates with the magnetization and its magnitude does not.** Driving
  the moment along `z`, `x` and `y` gives `|<L>|` = **0.0364767 in all three**,
  a spread of **7.3e-11**, with `L.S/|L||S| = 1.00000000`. Nothing in the code
  imposes that a magnitude is a scalar — and the threshold is part of the claim:
  at `conv_thr = 1e-8` that spread is 1e-5, which is the SCF's own scatter.
- `<S_z>` against the SCF's own magnetization: 0.312699 against 0.61701/2, the
  gap being what the projector set does not capture (its charge is 17.90 of 18).

**Refused by name.** A **symmetry-reduced k-set**: `<L>` and `<S>` are vectors
and a wedge sum is a wedge sum — the axial-vector symmetrisation P24 records for
a response, one index up and with `det(R)` and the time-reversal sign on top, is
not written. The escape is the whole unshifted grid, which is closed under the
point group, and it is the same escape `dielectric_tensor` documents. And a
**fully-relativistic ultrasoft or PAW** dataset, because the spinor overlap's
off-diagonal spin blocks are `qq_so` (`transform_qq_so`) where the projection
here applies the *scalar* `S` to each component — which is `projwfc.x`'s
validated path in every other regime and is missing exactly that term in this
one. A fully-relativistic **norm-conserving** dataset has `S = 1` and is exact,
which is what nickel above is.

**Cost** is a rider on a run that already happened: the orbitals are the
`(nk, npwx, natomwfc)` array a projected DOS or a Hubbard `U` already holds, and
the site matrices are `(natom, nshell, 2l+1, 2, 2l+1, 2)`, which is nothing.

### P49 — The notebooks, rewritten for someone computing a property. ✅ DONE.

`notebooks/`, `notebooks/README.md`, `pypresso/calculator.py`, and one new test.
Not started. This entry is the design, written before the work and reviewed
before being written down, so the session that picks it up does not re-derive the
diagnosis. The complaint that started it is that the code in the notebooks is
hard to read and does not go to the physics, for a reader who wants to *use* the
code to compute something.

**The measurement first, because the obvious diagnosis is the wrong one.** Across
the 29 notebooks there are 2800 non-comment code lines in 159 code cells: 97 lines
per notebook against a convention that says "about eight code cells, not twenty",
and 17.6 lines per cell, so the shape is a handful of very long cells rather than
many short ones (the longest are 48, 42, 40, 40 and 39 lines). Of those lines
**14% are hand-rolled matplotlib and 15% are `print` formatting**. Every one of
the 29 imports from `pypresso` past `Calculator`, median four such lines, worst
nine. The instinct is to call this boilerplate and shorten it. That is a symptom.

**The cause is that the notebooks are doing the test suite's job in public.** The
conventions in `CLAUDE.md` and `notebooks/README.md` bind the *prose* — no `jvp`,
no Fortran file names, no phase numbers, no trap catalogues — and the notebooks
obey that faithfully. Nothing has ever bound the *code*, so this project's
validation instinct moved into the code cells, where the rule does not reach:

- `09_forces_and_relaxation`'s **first** code cell is a frozen-functional identity
  check looped over four pseudopotential cases, printing a four-row table of SCF
  total against frozen functional. A reader who wants a force meets `frozen_energy`
  and `state_from_result` before `get_forces()` appears at all.
- `13_dft_plus_u` checks `hubbard_potential` against `qe_hubbard_potential` **on a
  random symmetric matrix** (`np.random.default_rng(0)`). That is a unit test,
  transcribed into a tutorial.
- `19_linear_response` hand-builds a Sternheimer solve with a cosine probe
  potential, twice, once spin-polarized. It is 459 export lines, the longest in the
  set, and the worst offender on every axis measured here.

This cause drags the other two symptoms behind it, which is why fixing volume
directly would not work. The identity checks are *what force* the internal
imports; the multi-case comparisons are *what force* the printf tables. Evict them
to the tests and the 97-line mean falls to the 60s without anyone shortening
anything.

**Second cause: the facade is opened and then abandoned.** "Drive it with a
`Calculator`" is honoured as *open* with `from_file` and then drop to internals.
The damning subset is where the `get_*` method exists and the notebook builds the
quantity by hand anyway. `19` constructs band velocities from `VelocityOperator`
plus `fixed_density_states` over ten lines and only afterwards remarks that
`get_band_velocities` does the same in one call — which is exactly backwards.
`25_raman_tensors` imports `raman_tensors`, `refined_states`, `Calculation`,
`run_scf`, `build_system` and `read_upf` at the top and then calls
`get_raman_tensors()` regardless.

**Third: `.plot()` exists on four result objects and one notebook uses it.**
`BandStructure`, `DensityOfStates`, `ProjectedDOS` and the absorption result all
carry a `plot` method; only `28_the_calculator` calls any of them, and the other
28 hand-roll the axes. The `print` half of the presentation cost is the larger and
the uglier one, and it has no such method to reach for at all.

**Fourth, and it is a library gap rather than a notebook one.**
`Calculator.from_file` silently drops the `&electrons` namelist: nothing anywhere
in `system/builder.py` or `calculator.py` reads `conv_thr`, `mixing_beta`,
`mixing_mode`, `electron_maxstep` or `mixing_fixed_ns`, though the same builder
does read `&control`'s `etot_conv_thr`, `forc_conv_thr`, `nstep` and `tstress`.
So it is an asymmetry, not a decision, and it is generating boilerplate in
**eleven** notebooks, which define a local `load()` helper between them. They split
into two kinds and the fix reaches them differently. Four — `07`, `11`, `12`, `13` —
re-parse the input with `read_pw_input`/`parse_pw_input`, pull `mixing_beta`,
`conv_thr` or `mixing_fixed_ns` out of it, and hand them straight back to a
`Calculator` built from the same text; those helpers are deleted outright. The other
seven — `03`, `04`, `05`, `06`, `08`, `09`, `17` — are path-and-defaults shorthand
(`Calculator.from_file(DIR / name, pseudo_dir=PSEUDO, announce=False,
conv_thr=1e-10)`) and only shrink. **28 of the 29 notebooks pass a `conv_thr` by
hand and 25 pass `announce=False`**; the first number is what the `&electrons` fix
is worth, and the second says `announce` wants a quieter default or a context
manager of its own rather than a keyword on every construction.

**Fifth: the ordering is a development history.** `notebooks/README.md` says so in
its first line — "in the order the code gained them". The sharp version of the
problem is that `28_the_calculator`, the notebook that teaches how to use the
code, is second to last, and is **not linked from the root `README.md` at all**
(its table runs 01 to 27 and then 29). There is no lookup anywhere from a property
someone wants to the notebook that computes it.

**Sixth, and it is the gap worth the most: nothing shows a reader running their
own crystal.** All 29 open on QE's test suite or on `tests/data/qe/`. In 2800
lines there is no cell that writes a fresh `pw.x` input for a material of the
reader's choosing, points at pseudopotentials they fetched, and runs it. For the
audience this phase is for, that is the first need, and it is unserved.

**What is not wrong and must not be swept up.** Three notebooks are legitimately
about machinery and their internals *are* their subject: `01` (the basis), `03`
(the eigensolver and the QE comparison) and `17` (mixing and self-consistency).
They should be labelled as a separate tier in the index, not rewritten to a
skeleton they do not fit. And several hand-drawn figures show physics no result
object holds — `02`'s bonding-charge plane, `19`'s induced charge — and are the
best cells in the set. `.plot()` covers less of the 14% than it looks like it
should, and that is fine.

**Phases 1 and 2 have landed.** What they cost and what they found:

**Phase 1, the index.** `notebooks/README.md` is now keyed by the property —
six tables (ground state, structure, magnetism, response and spectra, topology,
choosing the physics of the run) each giving the `get_*` method beside the
notebook, with `01`, `03` and `17` moved into an explicit "under the hood" tier
and the file-order table kept below for anyone who wants it. `28_the_calculator`
is `00_the_calculator`, its one image and its two inbound links moved with it,
and it is in the root README, which had been listing 01 to 27 and then 29. Every
link and every `get_*` name in the new index is checked by grep against the file
system and against `calculator.py` rather than remembered, and so are the four
input variables the "choosing the physics" table names.

**The rule that was missing is now in `CLAUDE.md`**: the physics-only rule binds
the *code cells* and not only the prose. That is the whole diagnosis in one
sentence, and it is what the phase-4 test will enforce mechanically.

**Phase 2, the library.** Three things landed and one was measured and rejected.

- **`&electrons` is adopted** by `from_file` and `from_text`
  (`calculator.electrons_defaults`): `conv_thr`, `mixing_beta`, `mixing_mode`,
  `mixing_fixed_ns`, and `electron_maxstep` renamed to `max_iterations` on the
  way in, because that word means three different loops here and the input
  file's number is unambiguously the SCF's. An absent variable stays absent
  rather than being given a default, so `run_scf` keeps deciding; an explicit
  keyword argument still wins over the file. Verified on QE's own inputs:
  `pw_lda+U/lda+U.in` yields all four of its settings and `pw_noncolin` two,
  which are exactly the numbers four notebooks were re-parsing the input to
  recover. **`diagonalization` is read by `pw.x` and deliberately not adopted**,
  and the reasoning is in `_ELECTRONS_NOT_ADOPTED`: it is a `SETUP_OPTIONS`
  member rather than a run default, and this package offers one eigensolver, so
  adopting it would turn `diagonalization = 'cg'` — valid `pw.x` input — from a
  run that works into a `ValueError`, while mapping it onto Davidson would be
  the silent substitution the package refuses everywhere else.
  **The precedent that settles this was already in the repository and is better
  evidence than the `&control` asymmetry**: `tools/compare_qe.py` has always
  read `conv_thr` out of `&electrons` with a private regex (`_CONV_THR`),
  commented "so both codes stop at the same accuracy". The project had already
  concluded that the namelist must be honoured; it had just concluded it in a
  tool rather than in the library. That regex is now duplicated logic and could
  call `electrons_defaults`, which is a tidy-up deliberately **not** taken here,
  because the performance comparison is the one thing in the repository whose
  numbers must not move in a pass that is not about performance.
  **The blast radius is wide and is the intended behaviour**: 128 of the 136
  committed `pw.x` inputs under `tests/data/qe/` and `benchmarks/` carry an
  `&electrons` setting, nearly all of them a `conv_thr` of 1e-10 or 1e-12 where
  the old default was `run_scf`'s 1e-6. Runs get tighter and slower and agree
  with what `pw.x` does with the same file. Neither performance tool goes through
  `Calculator`, so `PERFORMANCE.md` is untouched; **notebooks `03`, `04`, `05`
  and `06` construct calculators without an explicit `conv_thr` on inputs that
  carry one**, so their committed outputs are stale as of this phase and are
  re-executed when phase 3 or 5 reaches them.
- **Four more `.plot()` methods**, chosen by the criterion that a notebook was
  already drawing them by hand: `VibrationalSpectrum` (a Lorentzian-broadened
  Raman or infrared curve, `26`), `SpiralScan` (`E(q)` with the turning moment
  on a twin axis, `12` and `14`), `RelaxResult` (energy and max force against
  the ionic step, `09` and `14`) and `BerryCurvature` (the `Omega(k)` map,
  `10`). The audit that picked them is the set difference the phase called for:
  69 classes carry array data and four had a `plot`, but most of the 69 are
  machinery nobody plots, so the operative filter is what the notebooks
  hand-roll, not what has an array in it.
- **`io.comparison_table`** — two columns of numbers beside each other, aligned,
  with the difference. It formats and decides nothing: no tolerance, no verdict.
  A missing reference prints `--` rather than becoming a zero.
- **`tests/unit/test_result_plots.py`**, 18 tests in 0.8 s, in the fast gate.
  The finding worth keeping is that **there was no test anywhere that called any
  `.plot()`** — the four that existed had never been executed by the suite,
  which is most of why 28 of the 29 notebooks hand-rolled their axes rather than
  reaching for one. The file covers all eight, and its parametrised contract
  test asserts the thing the notebooks depend on: `plot(ax=...)` composes into a
  figure the caller laid out.

**`announce` was looked at and left alone**, which is the one item of phase 2
that ends in "no". 25 of the 29 notebooks pass `announce=False`, so it looked
like the same kind of boilerplate as `conv_thr`. It is not: the announcement is
that an *implicit* SCF is starting, it is a documented rule in `CLAUDE.md` with
a test asserting it, and it is useful exactly where a notebook is not — at an
interactive prompt. Under the skeleton a notebook constructs one calculator in
cell 2, so the cost is one keyword on one line, which is the right price for
keeping the behaviour.

**Phase 3, the three exemplars, has landed**, and the headline is that the
skeleton cuts the code by more than half without losing a number:

| | code cells | code lines | longest cell | beyond-facade imports |
|---|---|---|---|---|
| `02` | 5 → **4** | 66 → **53** | 24 → **24** | 5 → **2** |
| `09` | 4 → **2** | 98 → **26** | 27 → **14** | 4 → **0** |
| `19` | 11 → **5** | 143 → **47** | 27 → **19** | 10 → **1** |

307 code lines become 126, every cell is inside the 25-line budget, and every
number in the three headline tables is reproduced by the rewritten notebook.

**The eviction was almost entirely deletion, which is the finding.** All three
of notebook 09's validation cells were **already in the test suite, verbatim**:
its opening four-case frozen-functional table is
`test_force_machinery.py::test_the_frozen_energy_reproduces_the_scf_total`, its
central difference is
`test_forces.py::test_the_force_is_the_derivative_of_the_energy`, and its
analytic-versus-autodiff term table is `test_the_two_methods_agree`,
`test_the_two_methods_differ_by_the_scf_correction` and
`test_each_term_matches_quantum_espresso`. The notebook was not carrying checks
that had no home; it was **restating the tests in public**. Only one gap was
real — the frozen-functional identity was asserted on one dataset where the
notebook showed four — and it is closed by
`test_forces.py::test_the_frozen_functional_reproduces_the_scf_total`,
parametrised over all five cases and passing in 14 s.

**What was dropped, and why each was safe.** `09`'s analytic-versus-autodiff
table is a *transcribed-versus-differentiated* comparison, which the notebook
conventions ban outright; its finite difference is quoted with its numbers and
named to the test that runs it. `19` lost its hand-built Sternheimer probe solve
(twice, once spin-polarized), its O2 level diagram and its second probe solve;
what those demonstrated survives as three sentences and the validated numbers
they produced. `19`'s velocity operator went from ten lines of
`VelocityOperator` + `fixed_density_states` to one `get_band_velocities()` call,
which is what the notebook itself used to point out immediately afterwards.

**Two prose claims were false against the numbers the rewrite printed**, and
both were inherited rather than introduced. The velocity cell's "the near-zero
entries are the band extrema" is wrong on the SCF's own shifted Monkhorst-Pack
grid, whose smallest speed is 0.189 Ry bohr — the grid is shifted *off* the
high-symmetry points, which is where `d(eps)/dk` vanishes. And `19`'s induced
charge figure has one dashed atom line, not the two its caption claimed: the
second atom sits at `x = 0`, on the plot's edge. Re-executing a notebook after
rewriting it is what surfaced both, and neither is the kind of thing a test
would catch.

**`get_born_charges` had to be *used* to keep the index honest.** The task index
routes Born charges to `19`, and the rewritten notebook was reading them off
`DielectricTensor.born_charges` instead, so the row named a method the notebook
never called. It calls it now; it costs nothing, because `get_born_charges`
delegates to `get_dielectric_tensor` and hits the same cache.

**One thing was noticed and deliberately not fixed**: `tests/regression/test_forces.py`
caches its converged states with `lru_cache(maxsize=None)` over five cells,
which is the exact leak this file's memory rule names. Changing it alters what
the slow suite holds and belongs to a pass that is about memory.

**Phase 4, the enforcement, has landed**: `tests/unit/test_notebook_conventions.py`,
**107 tests in 0.07 s**, in the fast gate, parsing the `.ipynb` JSON and executing
nothing.

It is a **ratchet rather than a wall**, which is what stops it being deleted the
first time it is inconvenient. Every notebook is checked for the things that are
cheap to keep true everywhere — the `.md` export having one fenced block per code
cell, and no implementation vocabulary (`jvp`, a `.f90` name, a `PLAN.md`
reference, a phase number, an em dash). Only the four in `REWRITTEN` are held to
the whole skeleton: the first code cell building a `Calculator` in twelve lines,
the 80-line and 25-line budgets, the facade import allowlist with its cap of two
*justified* exceptions, no `load()`/`scf()`/`namelist()` plumbing helper, and no
use of an entry point whose `get_*` wrapper exists. `REWRITTEN` only grows and
`JVP_DEBT` only shrinks, and the file says so.

**It caught a real defect on its first run**, which is the argument for it: the
rewrite of `02` had put `# no facade route to rho(G)` on the `build_basis` line
and nothing on the `r_to_g` line beside it, so one of the two deep imports was
unjustified and no human reading had noticed. **And it is not vacuous** —
adding `13_dft_plus_u` to `REWRITTEN` fires four of the five skeleton checks
immediately.

Two of its choices are worth keeping written down. The phase-number pattern
`\bP\d{1,2}[a-c]?\b` **would** false-positive on a space group written
`P4/mmm`, though not on `P6_3/mmc`, whose underscore is a word character; it
fires on none of the 29 today, and the fix when it does is to name that notebook
in an allowlist rather than to weaken the pattern. And the deep-import budget is
**two** rather than zero, because the two figures that survived the rewrites of
`02` and `19` both need `r_to_g` and there is genuinely no facade route to
`rho(G)`: a budget of zero would delete the best figures in the set.

The baseline it recorded: **4 of 29 notebooks pass the full skeleton**, and the
other 25 fail on code volume (up to 149 lines against 80), cell length (up to 48
against 25), or reaching past the facade (up to 7 imports).

**Phase 5 is the sweep and it is partly done. Pick it up here.**

Rewritten and passing the enforcement test: `00` (which already complied), `02`,
`09`, `19`, `11`, `13`, `08`, `10`, `18`, `22`, `15`, `24`, `29`, `26`, `25`, `04`,
`05`, `06`, `07`, `12`, `14`, `16`, `20`, `21`, `23`, `27`.

**`21` and `23` are the two that were edited in place** rather than rebuilt from
a scratchpad script like every other conversion, because what they needed was a
first cell and nothing else. There is no `build21.py`/`build23.py` to look for,
and the `.ipynb` is their source of truth — which is the safe half of checklist
item 2, since the danger there is a *stale* script overwriting a patch, not the
absence of one.

**Phase 5 is finished. 26 of the 29 are in `REWRITTEN`, `JVP_DEBT` is empty,
and the three that are not are the under-the-hood tier by design** — `01`, `03`
and `17`, whose internals are their subject and which `notebooks/README.md`
labels as a separate section. What they owed was not a skeleton: it was
re-execution, since phase 2's `&electrons` adoption made their committed
outputs stale, and their path helpers now carry the path and nothing else.
`03`'s energies came back identical to every digit and only its *timings*
moved, which is the machine rather than the code; `17`'s Newton root tightened
from 1.6e-9 to 3.1e-10 Ry, because its helper had been capping
`electron_maxstep` below what the input asks for.

**And the whole set is timed now**, which was the open tooling item:
`tools/export_notebooks.sh` measures each notebook as it re-executes it, marks
anything over the ten-minute ceiling and exits non-zero. Nothing is over it;
the slowest is `27` at 178 s and the median is under 30. The table is in
`notebooks/README.md` and is a by-product of keeping the outputs true rather
than something anyone has to remember to measure.

**The Raman merge is done**, which is the first of the two structural jobs.
`25_raman_tensors` is deleted and `26_raman_and_infrared_spectra` carries both:
the spectrum first, because "give me the Raman spectrum" is the question, and
the tensor as its "how it works". 167 code lines across thirteen cells become
43 across four. Both index rows point at `26` now, and `25` is the free number
the "your own crystal" notebook takes, so the set stays at 29 with no gap that
was not already there. What went: the `dynmat.x` subprocess cell (it is
`test_the_mode_table_matches_dynmat_x`), the five-run finite difference over
displaced cells (`test_the_raman_tensor_matches_a_finite_difference`), the
`ph.x`-regression table, and the hand-rolled Lorentzian broadening, which is
`VibrationalSpectrum.plot` since phase 2. **The depolarisation ratios of
silicon's acoustic triplet moved between the old committed output and this
one** — 0.3544/0.7163/0.4065 became 0.5899/0.3964/0.5132 — which is the
multiplet rule demonstrating itself: the basis inside a degenerate manifold is
arbitrary, and the activity beside them reads 0.0000 in every version.

| | code lines | code cells | runtime |
|---|---|---|---|
| `02` | 66 → 53 | 5 → 4 | 7 s |
| `09` | 98 → 26 | 4 → 2 | 6 s |
| `22` | 111 → 50 | 5 → 4 | 12 s |
| `18` | 114 → 56 | 9 → 4 | 3 min → **29 s** |
| `15` | 109 → 60 | 6 → 5 | 31 s |
| `24` | 103 → 48 | 5 → 3 | 31 s |
| `29` | 102 → 46 | 6 → 4 | 59 s |
| `25`+`26` | 167 → 43 | 13 → 4 | 107 s |
| `25` (new) | — → 66 | — → 5 | 28 s |
| `04` | 74 → 40 | 4 → 4 | 12 s |
| `05` | 98 → 59 | 5 → 4 | 10 s |
| `06` | 93 → 80 | 5 → 5 | 23 s |
| `07` | 82 → 63 | 4 → 4 | 22 s |
| `12` | 84 → 29 | 3 → 3 | 27 s |
| `14` | 91 → 46 | 5 → 3 | 89 s |
| `16` | 92 → 58 | 5 → 5 | 9 s |
| `20` | 73 → 54 | 6 → 4 | 57 s |
| `21` | 81 → 67 | 6 → 6 | 30 s |
| `23` | 72 → 70 | 6 → 6 | 35 s |
| `27` | 92 → 40 | 5 → 3 | 178 s |
| `19` | 143 → 47 | 11 → 5 | 46 s |
| `10` | 134 → 46 | 5 → 5 | 50 s |
| `13` | 123 → 43 | 5 → 3 | 66 s |
| `11` | 147 → 54 | 6 → 5 | 79 s |
| `08` | 149 → 56 | 6 → 4 | **25 min → 171 s** |

**Both structural jobs are done.** `25_your_own_crystal` is written and is the
only notebook in the set that does not open on a file QE shipped: diamond from a
lattice constant and a fetched pseudopotential, the input printed with its own
comments as the teaching material, the two convergence sweeps, then bands and a
density of states. It is linked second in `notebooks/README.md`, after `00`.

**Writing it found the trap it now teaches.** The first draft claimed "the k-grid
converges much faster than the cutoff" and "35 Ry for 1e-4 Ry", and the executed
figure said neither: at 35 Ry the total energy is **7.7e-4 Ry** from the 50 Ry
run, and the $6^3$ grid is **6.1e-4 Ry** from $8^3$, which is the same distance
40 Ry is from 50 and costs far more to close. Worse, the "gap" plotted against
the k-grid never settles — and that is not a convergence failure: it is the
smallest gap *among the k-points of the run*, and diamond's conduction minimum is
at 0.85 of $\Gamma$X, on none of those grids. A gap is read off a band structure,
and the two differ by 0.024 eV here (4.1305 against 4.1541). The notebook says so
now, which is the best content in it and was not in the plan.

**Nothing is left.** The sweep is complete: 26 notebooks to the skeleton, the
Raman merge, the "your own crystal" notebook, and the three under-the-hood ones
re-executed and trimmed. `01`, `03` and `17` are
the "under the hood" tier and are **not** held to the skeleton — trim them toward
the everywhere-rules and leave them out of `REWRITTEN`.

**Two of them needed a new input file, and that is the shape of the remaining
work.** `&electrons` is adopted, but `ecutwfc`, `K_POINTS` and `input_dft` are
not `Calculator` options and never should be — they describe the *system*. A
notebook that needs a cell at a different cutoff or a different functional was
therefore reaching for `read_pw_input` + `build_system` + `read_upf`, which is
three deep imports and a `def` helper. Committing the input instead costs one
small text file, deletes all of that, and makes the notebook reproducible from
the command line: `si2-nc-eos.in` for `15`, and `si2-lda-gap.in` /
`si2-tb09.in` for `24`, the last two differing by `input_dft` and nothing else.
Neither has a `pw.x` reference and neither needs one, which their comments say.
`29` needed two more — `si2-nosym.in` and `ni-soc-nosym.in`, both carrying the
`nosym` and the whole grid that the *axial* character of `<L>` and `<S>` demands
— which took two inline `from_text` heredocs of 20 lines each out of its code.

**And putting the band count in the input file was wrong**, which is worth
knowing because it is invisible: `29`'s effective mass wants a window past the
occupied bands, so `nbnd = 10` went into `si2-nosym.in`, and carrying those extra
bands through the *self-consistent run* moved silicon's quenched orbital moment
from **2.6e-16 to 2.0e-9** — seven orders, on the same cell at the same
`conv_thr`. `<L> = 0` without spin-orbit coupling is an identity, not a
tolerance, so that is the one number in the notebook that says so. `nbnd` belongs
on the `get_effective_mass` call, where it does not reach the ground state.

**What the four in this pass cost and found.** `08` was one cell split, and the
loop now reuses the first run's result rather than repeating it, so it cost no
execution time. `10` lost four validation cells that were already in the tests
verbatim — the zone-edge wrap, the Haldane mesh sweep, the Wilson-fermion closed
forms and silicon's TRIM parity table — keeping the AlAs curvature map and one
model cell, because no crystal in this repository carries a nonzero invariant
that can be run in a notebook. `18` ran fourteen self-consistent calculations
where two pairs make the point; the silicon and platinum rows of its table are
quoted and named to the tests that run them. `22` lost its
transcribed-against-differentiated force and stress table, which the conventions
ban outright, and its elastic-constant cell, which was the last of the three
`jvp` debts but one. `15` needed one new input, `tests/data/qe/si2-nc-eos.in` —
ideal silicon at `ecutwfc = 40` — because its equation of state was being built
by editing a parsed input's `celldm` in the notebook, which is five imports past
the facade; `with_cell` does it in one call and reproduces the old route's
printed residue **exactly** (2.58 kbar over -83 to 161), which is the check that
the k-point regeneration `with_cell` exists for is doing the same thing the hand
route did.

**And the fabricated header happened twice**, in the same session, after the rule
was written down: `15`'s five pressures were also written from memory and are
-23.58, -31.13, +10.53, +10.95 and +47.08 kbar. A headline number goes in after
the execution, from the output, every time.

**Three corrections the re-execution caught**, which is checklist item 3 doing its
job. `10`'s "the hot spots sit where the gap is smallest" was a claim the figure
does not support; what the figure *does* show is that `Omega(k)` is odd under
`k -> -k`, which is time reversal without an inversion centre, and that is the
sentence now. `18`'s figure was titled "the moment is the slow variable" and its
own right-hand panel says otherwise — both runs have most of the moment after one
iteration. And `22`'s header table carried two **fabricated** energies, written
from memory rather than from the run; they are -45.10439956 and -0.02305929, and
the rule that catches this is to fill a headline number in *after* executing, not
before.

**The per-notebook checklist, learned the hard way.** Each of these was missed at
least once and each cost a correction:

1. **Grep the tests before evicting anything.** Every eviction so far has been
   pure deletion — the check was already in `tests/`, better written. The one
   real gap found (`09`'s identity on four datasets against the suite's one) took
   one parametrisation. Do not assume: `08`'s finite difference is of the
   **frozen** energy in the suite and was of the **converged** energy in the
   notebook, which are different claims.
2. **The build script is the only source of truth.** Patching a `.ipynb`'s JSON
   directly to fix prose and then rebuilding from the script silently reverts it.
   That happened to `13` and was committed.
3. **Re-execute, and look at the output.** It is what catches prose that is false
   against its own numbers (`19`'s "near-zero entries are band extrema" on a grid
   shifted off the high-symmetry points; `19`'s caption claiming two dashed atoms
   where one is at the plot edge), and it is the only thing that renders mathtext
   (`\mathbf` without braces is a fatal error that no assertion sees).
4. **Edit the index rows in the same commit.** Both `README.md` tables and the
   runtime paragraph in `notebooks/README.md`. Missed three times.
5. **Sweep the orphaned figures.** Cell indices move, so `nbconvert` leaves the
   old `_files/*.png` behind and they get committed.
6. **Drop `conv_thr` where the input file already states it** — that is the
   boilerplate phase 2 exists to delete — and keep it, with the reason on the
   line, only where the notebook genuinely tightens the input.

**A stale-warning sweep over all 29 committed outputs found exactly one** and it
is fixed (`11` claimed spinor forces and stress were unimplemented, which P46
implemented). Worth re-running after any refusal changes: grep the `.md` exports
for warning text and check each string still exists in `pypresso/`.

**The skeleton, which every property notebook is held to.** Nine cells, 60 to 70
non-comment code lines, no cell over 25.

1. Markdown: the property as the title, its defining equation in display maths
   (for a derivative, *of what, holding what fixed*), the headline number, and the
   comparison against QE or Elk or experiment **as a markdown table of quoted
   numbers**. `19`'s opening table is the model already.
2. Code, **10 lines at most, and this cell is what the whole exercise is for**:
   the imports, `Calculator.from_file(...)`, the one `get_X()` call, and the
   number printed plainly. No `def load()`, no `read_pw_input`, no internals. Where
   the run needs particular mixing, it belongs in the input file, which the
   `&electrons` fix below is what makes possible.
3. Markdown: what the number means. Sign, magnitude, what experiment says.
4. Code: the figure. `result.plot(ax=...)` where one exists; hand-drawn only where
   it shows physics the result object does not hold.
5. One live comparison against QE. One case, one table, ten lines.
6. Optional: one cell for the single best *physical* idea. `13`'s figure of the
   penalty pushing occupations to 0 and 1 is the model; `09`'s frozen-functional
   identity is the anti-model.
7. Markdown: what the feature refuses.
8. Markdown footer naming the tests, extended to say where the identity checks and
   per-case tables this rewrite evicted now live — so they are moved rather than
   deleted.

Cell 2 is the anchor the enforcement test below is written around.

**Phases, in the order they have to happen.**

- **1. The index.** `notebooks/README.md` rewritten task-first: a table keyed by
  the property, giving the `get_*` method and the notebook, with a separate "under
  the hood" section for `01`, `03` and `17`. Rename `28_the_calculator` to
  `00_the_calculator` — one file, two inbound links, and it is the signpost the
  reader needs first — and add it to the root README, which is missing it. No code
  risk and visible in one sitting.
- **2. The library, and it must come before any rewrite** or the rewrites get
  rewritten. `from_file` adopts `&electrons` into `defaults`, with explicit keyword
  arguments still winning; `electron_maxstep` maps to `max_iterations` and needs
  care, because `SCF_ONLY_OPTIONS` already documents that three different loops
  share that one word. This is a behaviour change for every existing caller and
  wants a slow-suite pass behind it. `announce` gets its own answer in the same
  pass, since 25 notebooks pass `announce=False` on construction. Then an audit
  **by set difference** — result
  objects carrying array data against result objects carrying `plot` — and the
  missing ones written: `RelaxResult` (energy and force against step, hand-rolled
  in `09`), the vibrational spectrum's stick plot (`26`), band velocities (`19`).
  Plus one small table helper, so the `print("%-14s %16s ...")` art goes. **All of
  this lives on result objects**, which is where this project already put
  presentation: the facade rule constrains `get_*`, not results, and there are four
  precedents. A `plot` draws what the result already holds and computes nothing.
- **3. Three exemplars: `02`, `09`, `19`** — one already close, one
  validation-shaped, one a machinery tour. The evicted identity checks land in
  named test files in the same commit. These calibrate the budgets before the
  sweep, and they are what the user signs off on.
- **4. The enforcement test.** `tests/unit/test_notebook_conventions.py`, parsing
  the `.ipynb` JSON, executing nothing, in the fast gate. Per notebook: an import
  allowlist (`pypresso`, `pypresso.units`, `pypresso.system.kpoints`,
  `pypresso.io`), anything else needing an inline justification comment and capped
  at two; the first code cell containing `Calculator.from_` and at most twelve
  lines; 80 code lines per notebook and 25 per cell; banned tokens; the `.md`
  export's cell count matching the `.ipynb`, which is a staleness canary that costs
  no execution; and a facade-bypass check against a hand-maintained map from entry
  point to `get_*`, which is what would have caught `25` and `26`. The exemptions
  live in an explicit dict that **shrinks as notebooks land**, so the thing
  ratchets instead of rotting. One trap in the token list: a `P\d\d` phase-reference
  pattern false-positives on space groups and point groups and has to be anchored.
- **5. The sweep**, in index order: the remaining notebooks to the skeleton;
  `25` and `26` merged into one Raman notebook, the spectrum first and the tensor
  as its "how it works", since the physicist's question is "give me the Raman
  spectrum" and the answer currently spans two; `08` trimmed to a quoted
  measurement, which `CLAUDE.md` already flags as owed and unowned; and the new
  **"your own crystal"** notebook, which is the sixth finding above and the highest
  value single artifact in the phase.

**The cost is wall clock, not thinking.** Every rewritten notebook is re-executed
and re-exported through `tools/export_notebooks.sh`, and **that script times each
one and fails over the ten-minute ceiling now**, printing the whole set sorted by
wall time at the end. It also takes notebook paths as arguments, so a single
rewrite does not re-execute the other twenty-eight, and `CEILING=` moves the
limit for a deliberate experiment. What it is for is that the unmeasured set gets
measured as a side effect of the first full re-export rather than by anyone
remembering to time it.

**Why the count stays at 29.** Renumbering is roughly ninety file renames plus a
hunt through prose cross-references ("notebook 17 could only difference...") and
through each notebook's `_files/` directory, for a gain that a task-keyed index
delivers on its own. Twenty-nine was never the problem; the shape and the index
were. The set ends at 29 either way: one merge, one addition.


### P50 — The piezoelectric tensor: the third thing Elk has and `pw.x` does not. ✅ DONE, clamped-ion.

`pypresso/response/piezo.py`. Elk's task 380 (`piezoelt.f90`); the third entry
taken from `ELK-FEATURES.md`, and the first that fails that file's own
cheapness filter — it is a Sternheimer-scale computation rather than an
NSCF-scale one. It is here anyway because the *implementation* cost is one
`jvp` of code that already exists, and because it is the one entry where
autodiff does the work rather than an assembly.

**What it is.** `e_(k)ij = dP_k/d(eps_ij) = d(sigma_ij)/dE_k`: the polarization
a strain induces, which is the same number as the stress a field induces
because a mixed second derivative does not care which leg is taken first,

    e_(k)ij = -(1/Omega) d^2 E / d(eps_ij) dE_k.

The equivalence and the choice between the two are Baroni, de Gironcoli, Dal
Corso and Giannozzi's review (`cond-mat/0012092` §II.C.2), which also records
that the stress-under-a-field form is what de Gironcoli, Baroni and Resta took
for the III-V compounds (PRL **62**, 2853 (1989)).

**It is P24b with one coordinate changed, and that is the whole phase.** A Born
charge is `Z* = dF/dE` and `pypresso/response/born.py` computes it as one `jvp`
of the *force* along the field's response. The force is `jax.grad` of the
frozen-state energy in the positions; the stress is `jax.grad` of the *same*
functional in a strain. So the piezoelectric tensor is one `jvp` of the stress
along the same field response — three of them, one per field direction, on top
of a dielectric constant that was going to be solved anyway.

**The strain leg is cheaper than the displacement leg it copies, and the reason
is the orthonormality constraint.** `<psi|S|psi>` is a sum over the plane-wave
sphere of `|c_G|^2` and the sphere is a set of *integers*; ultrasoft's `qq_ij`
is an atom-centred integral over all space with no cell in it. The constraint is
therefore strain-independent **as a function of the states as well**, so its
mixed derivative vanishes identically and the multiplier response `dLambda` —
which is `psidspsi`, `add_dkmds` and `add_for_charges`, three of the four things
P24b had to supply — has nothing to contribute. What is left is one term.

**Three routes, and they are how the phase is validated, because there is no
reference.** `pw.x` computes no piezoelectric tensor at all: the word occurs
once in the vendored tree, in a citation of Vanderbilt's paper in a comment in
`PW/src/bp_c_phase.f90`. Elk's `piezoelt.f90` reaches it by a **finite
difference of the Berry-phase polarization over one full ground state per strain
tensor**, with a `2 pi` branch fix-up between the two — `nstrain` self-consistent
calculations where this is one, and it is clamped-ion in the same sense (the
atoms are carried by the lattice, `tshift = .false.`, and nothing relaxes).

| route | what it is | AlAs `e_14`, C/m² |
|---|---|---|
| the implementation | `jvp` of the stress along `dpsi^E` | **-0.7637852276** |
| `zstar_eu.f90`'s contraction, strain label | `-2/Omega sum w Re<dpsi^E \| dH/d(eps) psi>` | -0.7637852276 (**6.2e-15**) |
| the other ordering | `-2/Omega sum w Re<b^E \| dpsi^(eps)>` | -0.7637853253 (**1.3e-7**) |

The second needs no strain response at all — it is three `jvp` calls of `H|psi>`
and a contraction, 1.5 s against the field response's 11 — and it is the
transcribed expression put beside the differentiated one, the arrangement
`force_us` and `stres_knl` are already in. The third costs six more Sternheimer
solves and is the only one that puts the **strain** response on the screened
side, so its 1.3e-7 is that response's own convergence rather than the
assembly's.

**The trap is a factor of two and it is Rydberg's `e^2`.** `dielec.f90`
contracts the *same* field response with a **4** in front
(`response/efield.py:_assemble`) and `zstar_eu.f90` contracts it with a **2**.
Both are right: a susceptibility is a Coulomb-normalised quantity and a
Rydberg-unit code puts `e^2 = 2` there, where a bare mixed second derivative
does not — and a piezoelectric constant is a mixed second derivative, in units
of `e/bohr^2`, exactly as the Born charge it copies is in units of `e`. Taking
the 4 gives a tensor that is exactly zincblende, exactly symmetric in its two
strain labels, vanishes on silicon, agrees between the wedge and the closed
grid, and is **twice too large**. No symmetry check sees it; what said so is
that the two routes disagreed by exactly 2.0000005.

**What anchors the sign, the field's normalisation and the volume.** Nothing
external validates a piezoelectric tensor here, so what stands in for one is
that **the same assembly run in the position coordinate is the Born charge**:
`born_charges_from_stress_route` is `clamped_ion_piezoelectric` with
`at_positions` where it has `at_strain`, and for a norm-conserving dataset
`Z_a delta_ij` minus it *is* `Z*` — 1.92460 and -3.18116 against the vendored
`ph.x`'s 1.92461 and -3.18098 on this cell. A wrong sign, a wrong field
normalisation or a missing volume would show there.

**Proper against improper, and why it is a refusal rather than a correction.**
What the `jvp` gives is the *improper* tensor, the bare mixed second derivative.
A measurement sees Vanderbilt's proper piezoelectric response (J. Phys. Chem.
Solids **61**, 147 (2000)), and the two differ by
`delta_ki P_j - delta_ij P_k` — terms that arise because a strain carries the
charge distribution with the cell (`d(Omega P_k)/d(eps_ij)` at frozen states is
`delta_ki Omega P_j`) and changes the volume that divides it. **Both vanish
whenever the two Cartesian labels they pair are different**, so `e_14` of a
zincblende crystal — its only independent component — carries no ambiguity
whatever the polarization is; and both vanish for *every* component of a crystal
whose class admits no invariant vector. So a **polar crystal is refused by
name**: `polar_direction` averages the crystal's point group, which is the
projector onto the directions a polarization may point along, and a nonzero one
stops the run. It is searched from the *structure* rather than read off the run,
because a response is usually `nosym` and that list would call every crystal
polar.

**Validated by four statements, each of which fails differently.** Silicon is
centrosymmetric and its whole tensor vanishes (measured 2.4e-5 C/m², the
response solver's floor, against AlAs's 0.764 from the same code); AlAs is
`-43m` and the only components that survive are `e_14 = e_25 = e_36`, on a
`nosym` run where nothing imposes the crystal class (forbidden components
1.7e-14); the eight-point wedge reproduces the sixty-four-point closed grid to
**4.5e-9** — P36's rank-3 symmetriser again, and here it is the whole of the
completion because this assembly is *linear* in the response, where P35's
screening term is quadratic; and the three routes above.

**Where it lives.** `pypresso/response/piezo.py`, reached by
`Calculator.get_piezoelectric_tensor()`; `tests/regression/test_piezoelectric.py`
(8 tests, 103 s: the three routes, the two symmetry statements, the wedge, and
the `Z*` anchor) and `tests/unit/test_piezo_machinery.py` (the Voigt convention,
the polar guard, and one refusal per unrun regime, no SCF);
`notebooks/28_piezoelectricity.ipynb`.

**What is left out, and it is not an approximation.** This is the **clamped-ion**
constant. The measured one adds the internal-strain term
`sum Z* (C^-1) Lambda` of the review's Eq. (111), and the review is where the
warning belongs: the two contributions "are often of opposite sign and close in
absolute value, so that a well converged calculation is needed in order to
extract a reliable value for their sum". Bernardini, Fiorentini and Vanderbilt's
table of linear-response III-V values gives AlAs a **total** `|e_14|` of about
0.01 C/m² against this 0.76, which is that cancellation. Every ingredient of the
missing term is already here — `Z*` (P24b), the force constants at `Gamma`
(P25), and `Lambda = -d^2E/du d(eps)`, which is this module's `jvp` with the
strain response as its tangent instead of the field's — and the one thing it
needs that does not exist yet is a **two-coordinate** frozen functional
`E(eps, u)`, since unlike the field leg both of its legs are coordinates of the
energy and the explicit `d_u d_eps E|frozen` term does not vanish. That is the
next step and it is what makes the number comparable with experiment. Two things
to know before starting it. The functional is
`calculation.at_strain(eps).at_positions((1 + eps) tau_0 + u)` — the strain
carries the atoms and `u` is the *further* displacement, which is what makes
`Lambda` the force a homogeneous strain leaves behind. And `C` in Eq. (111) is
**singular**: the force constants at `Gamma` have three exact zero modes by
translational invariance (the acoustic sum rule P25 and P28a check), so the
inverse there is a pseudo-inverse on the complement of the translations, and
taking a plain `inv` will produce a large number rather than an error.

The composition and the explicit term are **checked rather than asserted**: at
AlAs's own geometry `energy(0, 0)` reproduces the SCF total to every digit,
`dE/du` is 3e-16 (the force vanishes, as it must at the symmetric positions),
and the frozen half of `Lambda` along a `yz` shear is **-1.309 Ry/bohr per unit
strain** on the aluminium atom along `x` with 1.7e-17 on the other two
components — which is the zincblende structure of the internal-strain tensor,
and it is what the response's own half then has to be added to.

**And lifting the ultrasoft refusal is a dataset rather than a term.** What it
takes is one **non-centrosymmetric, non-polar** ultrasoft or PAW crystal
committed under `tests/data/pseudo/` — a zincblende III-V is the obvious one,
since `alas-raman.in` can be copied with the soft datasets substituted — and
then the tests that already exist say whether the three routes still agree.
Every soft dataset here today is centrosymmetric, so all three agree on zero and
say nothing.

**What it costs, and it is not what the term count suggests.** The strain leg
drops the multipliers the displacement leg needs and is still eight times slower
and four times heavier: 6.4 s and a **4.2 GB** peak on two-atom AlAs where the
Born charge off the same field response is 0.8 s and 1.13 GB. It is P11's
stress-against-force ratio one derivative up — `at_positions` moves one complex
exponential per atom over cached radial tables and `at_strain` moves `|G|`
itself, so every radial transform in the setup is inside the taped function —
and the peak does **not** respond to `k_batch`, because what the tape holds is
the setup rather than the k axis. The consequence is that here the *transcribed*
route is the cheap one, 4.0 s and no extra memory, which is the reverse of the
usual arrangement; it stays the cross-check because the differentiated one is
what extends, and `PERFORMANCE.md` says which to reach for on a large cell.

**Refused rather than approximated**, beyond the polar crystals: everything
`require_a_sternheimer_regime` refuses, everything
`require_a_differentiable_cell` refuses (a spin spiral, a magnetic field), and a
shifted grid run with `nosym`. **Ultrasoft and PAW are refused by name**
(`require_a_measured_dataset`)**, and it is a gap rather than a missing term.** Nothing in the assembly is
norm-conserving — the density and `becsum` are handed to the functional as
builders that carry the strain, which is what P41 needed for the strain
response; `qq_ij` has no cell in it, so the constraint stays strain-independent;
and the *displacement* leg of this same assembly is the Born charge, validated
on all three kinds. What is missing is **a case to measure it on**: every
ultrasoft and PAW dataset committed here belongs to a centrosymmetric crystal,
whose piezoelectric tensor vanishes identically, so all three routes agree on
zero whatever is wrong. P44 is exactly why a plausible argument about the strain
coordinate is not enough — there, two of P43's ingredients transferred, the
residue did not, and it was localised only because it could be measured against
a finite difference. Lifting this is one non-centrosymmetric, non-polar
ultrasoft crystal plus the tests that already exist.

### P51 — The optical conductivity tensor, the Kerr angle and the anomalous Hall conductivity. ✅ DONE.

`pypresso/response/conductivity.py` and `pypresso/workflows/conductivity.py`.
Elk's tasks 121 (`dielectric.f90`) and 122 (`moke.f90`); the **fourth** entry
taken from `ELK-FEATURES.md`, and the first one there that the file itself
records as only *partly* absent from QE.

**What it computes.** The whole complex tensor,

    sigma_ij(w) = (i/Omega) sum_k sum_nm W_n (1 - f_m)/e_mn
                    [ z_nm/(w - e_mn + i eta) + conj(z_nm)/(w + e_mn + i eta) ]
                  + wp_ij^2 / (4 pi (gamma - i w)),
    z_nm = <n|v_i|m><m|v_j|n>,

with `W_n` the k-weighted occupation (QE's `wg`) and `f_m = W_m/w_k` the
fractional filling — `dielectric.f90`'s `t1` exactly, whose asymmetry is
deliberate. From it: the dielectric tensor `eps = 1 + 4 pi i sigma/w`, the
plasma frequency, the complex **Kerr angle** (`moke.f90`, and it is the only
line of those 87 that computes anything), and the **anomalous Hall
conductivity** as the static antisymmetric part.

**What is new is the antisymmetric part.** `sigma_xx` is an absorption spectrum
and P37 already produces one, so on its own this would be a second
implementation of a validated quantity. `sigma_xy` is the response that rotates
light reflected off a magnet and whose static limit is the intrinsic anomalous
Hall conductivity, and nothing here computed it.

**The vanishing rule, stated carefully, because the usual version is wrong.**
Time reversal kills it, so a **nonmagnetic** crystal has none whatever its
spin-orbit coupling. A magnetic crystal without spin-orbit coupling has none
either — but only when its moments are **collinear or coplanar**, where a spin
rotation composed with complex conjugation survives as an antiunitary symmetry.
A **noncoplanar** magnet has an anomalous Hall effect with no spin-orbit
coupling at all (the topological Hall effect, driven by the scalar spin
chirality). So "magnetism and spin-orbit coupling together" is the usual route
and not a theorem, and what the surviving symmetry actually forces is
`Omega(k) = -Omega(-k)` rather than `Omega(k) = 0` — so the cancellation is
exact only on a k-set closed under `k -> -k`, which a shifted grid is not.

**The one line not transcribed, and it is the same one P37 refused.**
`epsilon.x`'s `dipole_calc` accumulates `<psi_1|G|psi_2>` — a bare momentum
matrix element — and `[H, r] != p` when the pseudopotential is nonlocal. Here
the velocity is `VelocityOperator.matrix_elements`, one `jvp` of `H(k)` at a
frozen sphere (rule D2), and it works on **spinors** unchanged: the contraction
runs over the whole `2 npwx` coefficient vector, so the spin trace is implicit.
That was the phase's scope gate and it was checked before anything was written
(hermitian to 1.1e-16 on a four-atom noncollinear cell, real diagonal to
2.4e-17). Everything else in the module is a transcription of `dielectric.f90`
and says so.

**The units cancel, and that is worth writing down rather than assuming.** Elk
is a Hartree-unit code. Going to Rydberg doubles the energies, doubles `dH/dk`
with them, and makes a Hartree momentum matrix element `<n|dH/dk|m>/2`. The
expression carries `1/e_mn` and `1/(w - e_mn)` — two factors of two — against
the squared matrix element's `1/4`, and the product is exactly one. So the
formula evaluated with **Rydberg** energies and **Ry bohr** matrix elements
returns `sigma` in **Hartree atomic units**, `e^2/(hbar a_0)` = 4.59988e6 S/m,
with no conversion factor anywhere in the sum. It is a coincidence of this
expression's homogeneity and not a rule; P50's factor of two is the warning
about assuming it elsewhere, and the two checks below are what establish it.

**Validation, and there is no reference for any of it.** `pw.x` computes no
conductivity and no Kerr angle, and refuses ultrasoft outright; Elk's number is
an all-electron one. So the phase stands on two internal statements and one
analytic identity.

*The symmetric part is P37's, exactly.* `eps = 1 + 4 pi i sigma/w` reproduces
`run_absorption`'s `epsilon_no_local_fields` on silicon to **3.6e-14** in the
diagonal, at every frequency. That chain is a different assembly — a Dyson
solve over a response sphere rather than a resolvent sum over band pairs — and
it reaches `ph.x` through `dielectric_tensor`. It is the check the factor of
two P50 found would have failed. The two tensors' **off-diagonals** are zero by
cubic symmetry in both, and what is left of them is each one's own truncation:
1e-9 in both at a clean band cut, and 3e-5 (P37) against 9e-4 (here) at a cut
inside a multiplet.

*The f-sum rule is the absolute anchor, and it is not 1.* The familiar
`int Re sigma_aa dw = pi n_e/2` is the **local** Hamiltonian's special case.
The exact statement for a velocity `dH/dk` is

    int_0^inf Re sigma_aa dw = (pi/2 Omega) sum_k sum_n W_n
                               [<n|d2H/dk_a^2|n> - d2eps_n/dk_a^2],

which follows from the `k.p` identity once the occupied-occupied pairs cancel.
Two things separate it from `pi n_e/2` and **both were measured**. Silicon's
diamagnetic weight `<n|d2H/dk^2|n>`, taken by a central difference of the
velocity operator at frozen states, is **0.9432** in Hartree units rather than
1 — that is the pseudopotential's nonlocality, and it barely moves with the
k-grid (0.9423 at 4x4x4, 0.9432 at 8x8x8). And `sum_k w_k d2eps_n/dk^2` is
zero over the zone by periodicity **and not on a coarse grid**: the spectral
weight measured in closed form (no frequency axis, no broadening — `Re sigma`
is a sum of Lorentzians whose total area is exact) comes out at **1.185** of
`pi n_e/2` on 4x4x4, **0.985** on 6x6x6 and **0.934** on 8x8x8, converging onto
the diamagnetic weight to **1 per cent**. That convergence is the check: the
volume, the electron count, the spin degeneracy and the Rydberg-to-Hartree
cancellation all enter it and none of them is fitted. **It is also a
k-convergence error nothing else in this package sees**, because every other
quantity here is an integral rather than an integral of a second derivative.

*The Drude leg has its own limit.* Aluminium's plasma frequency is **12.98 eV**
against a free-electron `sqrt(4 pi n)` of 16.27 — the 20 per cent is the
zone-boundary gaps removing Fermi surface, which is what makes aluminium
"nearly" free-electron rather than free-electron — and the tensor is isotropic
to **1.7e-4 eV** on a cubic crystal run with `nosym`, with nothing imposing
that. It needs 512 k-points to get there, because what is being integrated is a
Fermi surface and not a total energy: on 4x4x4 the same cell gives **13.78 eV**
(`al-conductivity.in`).

**Four things the phase found, and none of them is a refusal.** Two are
diagnostics reported to the caller, one was a bug in a shared boundary, and one
is a rule applied where the reference code does not apply it.

*Where the band sum stops decides a quantity symmetry says is zero.* Silicon's
antisymmetric `sigma` must vanish, and it comes out at **4.0e-13** when the
truncation falls at a real gap (`nbnd = 20`, 0.028 Ry) and at **1.0e-5**,
2.3e-6 and 8.3e-7 at `nbnd = 12`, 24 and 32, whose gaps are 1e-13. Truncating
inside a degenerate multiplet keeps some of its members and drops others, and
the cancellation they were making between them does not happen. The diagnostic
is `band_cut_gap`, and getting it right cost a design decision: the gap that
matters is between the last band **kept** and the first **dropped**, so it
cannot be computed from the sum's own band set at all — `run_conductivity`
diagonalises one extra band to measure it. It is **necessary and not
sufficient**: `nbnd = 36` cuts a degeneracy and escapes anyway, because its two
sides happen to contribute equally.

*A k-set handed in is a boundary `for_spin` has to cross, and the symptom is
not an error.* An optical spectrum wants a denser mesh than the density needed,
so `run_conductivity` takes one — and every `KPoints` constructor applies the
unpolarized spin degeneracy unconditionally, because it cannot know what regime
it will be used in. A **spinor** band holds one electron rather than two, so a
mesh built with `KPoints.automatic` and handed to a noncollinear run carries
weights summing to 2 where the run needs 1. Nothing about that looks wrong: the
electron count is still met and the Fermi level simply lands somewhere else.
Measured on fcc nickel's *own* 64-point grid, rebuilt rather than reused, it put
the plasma frequency at **13.11 eV instead of 0.60** and **flipped the sign** of
the anomalous Hall conductivity. `for_spin`'s own docstring names this class of
mistake and it appeared in a new place; the fix is one idempotent call at the
boundary and the test is that the two paths are one path. **It cost most of a
session, because the wrong numbers were the plausible ones** — 13 eV is a
believable plasma frequency for a transition metal and 0.6 eV is not, so the
sweep that used them looked like the converging one.

*The Drude weight must be the multiplet block and not the diagonal.*
`dielectric.f90` writes it as `sum_n v_a^nn v_b^nn`, and the diagonal of an
operator is not invariant under the rotation a degenerate eigensolver is free in
— design rule D4, and a Fermi surface is precisely where a metal keeps its
degeneracies. What is written here sums `v_a^nm v_b^mn` over the pairs the
*interband* term already excludes as degenerate, which is invariant, reduces to
the diagonal wherever nothing is degenerate, and makes the two halves of the
tensor complementary rather than overlapping. **It is a rule rather than a
repair**: measured on nickel the two forms give **0.5971 eV** each, because an
exact degeneracy has to sit within 1e-8 Ry of the Fermi level to be caught at
all and a 4x4x4 mesh puts none there. It is written
the invariant way because the diagonal one has no reason to keep agreeing on a
denser mesh or a more symmetric metal, and the failure would be silent.

*The two static routes are one limit taken in two orders, and for a metal the
orders do not commute.* Collapsing the two resolvents analytically takes `eta -> 0`
before `w -> 0`; evaluating the frequency sum at `w = 0` takes them the other
way round. Where every gap is large compared with `eta` the limits commute. At
a **Fermi surface** they do not — a metal has occupied-empty pairs with
arbitrarily small `e_mn`, the curvature route weights them `1/e_mn^2` and the
frequency route regularises them at `eta` — which is the reason an intrinsic
anomalous Hall conductivity is famously mesh-hungry, its integrand living on
near-degeneracies. `method = "curvature"` is the quantity's definition and is
what `pypresso.topology.kubo` computes by an independently written assembly
anchored to the Fukui-Hatsugai-Suzuki flux (P47); `method = "frequency"` at
`w = 0` is what a spectrum extrapolates to at finite scattering.

**Neither is quoted for fcc nickel, and the k-convergence is why.** Measured
with 36 bands and `eta = 0.01` Ry:

| grid | curvature | frequency at `w = 0` | `hbar wp_xx` |
|---|---|---|---|
| 4x4x4, 64 k | **-77.4** S/cm | **-1101.7** S/cm | 0.597 eV |
| 6x6x6, 216 k | **+2689.2** S/cm | **+1713.6** S/cm | 4.649 eV |

Everything on the Fermi surface moves, and the Hall conductivity **changes
sign**. That is not a defect of the assembly -- the same code gives silicon's
dielectric function to 3.6e-14 on this grid density, and aluminium's plasma
frequency to 2.5 per cent between 4x4x4 and 8x8x8 -- it is what a Fermi-surface integral of a
quantity concentrated on near-degeneracies does on 64 or 216 points. A
published intrinsic anomalous Hall conductivity for nickel is about -2200 S/cm
and is reached with meshes two orders of magnitude denser. **8x8x8 was started
and cut**: two points already say the sequence is not converging, and a third
would have cost an hour to say it again. What the phase delivers here is the
machinery and the diagnostic, not the number.

**Where it lives.** `pypresso/response/conductivity.py` (the assembly),
`pypresso/workflows/conductivity.py` (the fixed-density run and the extra
band), `Calculator.get_optical_conductivity`;
`tests/regression/test_conductivity.py` (6 tests) and
`tests/unit/test_conductivity_machinery.py` (8);
`tests/data/qe/al-conductivity.in` and `notebooks/30_magneto_optics.ipynb` are
new, and `pypresso/workflows/nscf.py` gained the `for_spin` call the finding
above is about.

**Refused by name**, each for its own missing term: **ultrasoft and PAW**, for
`pypresso.topology.kubo`'s reason — the current operator of a generalised
eigenproblem carries `<n|dS/dk|m>` off the diagonal, the term is identically
zero for a norm-conserving dataset, and nothing validated here can see whether
its convention is right; a **spin spiral**, whose two spinor components live on
different spheres; a **symmetry-reduced k-set**, because the antisymmetric part
is an axial vector and the escape is the whole unshifted grid; and **`nspin =
2`**, whose two channels are two band structures whose conductivities add — a
loop this assembly does not have, and not the regime a magneto-optical spectrum
wants anyway, since `sigma_xy` needs spin-orbit coupling and collinear spin has
none. And the **intraband term of a tetrahedron run**, which is refused rather
than dropped: its weight is a Fermi-surface delta function and the tetrahedron
method has none to supply (it integrates the step function itself, which is why
it has no `-TS` term either), so falling through would return an insulator's
conductivity for a metal with a plasma frequency of exactly zero and nothing
saying why.

**Timed against Elk, which is what `CLAUDE.md`'s performance rule now requires
of anything taken from another code.** Two-atom silicon, the same 64-point grid,
the same 20 states, one core each. The step that does the same work — form the
operator in three directions and contract the whole 3x3 tensor at 200
frequencies — is **0.85 s here against Elk's 2.11 s**, because `dielectric.f90`
writes `pmat` to disk and re-reads it per component while the `jvp` and the
contraction stay in memory and the frequency axis becomes one matrix product per
k-point. **From a converged state the whole call is 3.40 s against 2.11 s**, and
the difference is entirely the **NSCF**: Elk's ground state leaves its
eigenvectors on disk and never diagonalises again, where this code re-runs one.
That is the memory-for-time trade taken the other way and it is a choice.
`PERFORMANCE.md` has the table and the caveats — an all-electron LAPW basis
against 200 plane waves is not a like-for-like ground state. **The
magneto-optical half has no Elk timing**: an Elk nickel run with `spinorb`
converged to a moment of 0.008 `mu_B` against this code's 0.617, so its
`sigma_xy` is the wrong physics.

**Cost and peak.** One NSCF with empty states, then three `jvp` calls over the
k axis for `(3, nk, nbnd, nbnd)` matrix elements — which is the whole expense —
and a frequency sum whose working set is `nw x nbnd^2` complex per k-point,
accumulated through `sum_k`. Sixty-four k-points, forty bands and five hundred
frequencies is 13 MB per chunk against 5 MB of matrix elements: **the frequency
axis is free and the band count is what has to be watched**, since it enters
squared and is also the truncation the f-sum measures.

### P52 — The Fermi-surface nesting function. ✅ DONE.

`pypresso/response/nesting.py` and `pypresso/workflows/nesting.py`. Elk's task
105 (`nesting.f90`); the **fifth** entry taken from `ELK-FEATURES.md`, and the
first one there that `pw.x` and its post-processing tools lack *entirely* —
`nesting` occurs nowhere in `PW/src` or `PP/src`, which is a grep and not a
recollection. EPW has it, and EPW is out of scope.

**What it computes.**

    N(q) = (1/N_k) sum_k g(k) g(k + q),   g(k) = degspin sum_{s,n} delta(eps_snk - E_F)

`g(k)` is the density of states *at one k-point* — a picture of the Fermi
surface on the grid — and `N(q)` counts how much of that surface maps onto
itself when translated by `q`. It is the `omega -> 0` imaginary part of the
Lindhard function stripped of its matrix elements: the **geometric** half of an
instability, and the half that is cheap. Where it is large a perturbation of
wavevector `q` connects many occupied states to many empty ones at no energy
cost, which is what softens a phonon, opens a charge-density-wave gap, or picks
a spin spiral's pitch.

**The one place this is not a transcription, and it is worth an order of
magnitude.** `nesting.f90` writes an `O(N_q N_k)` double loop with `k + q`
folded back onto the grid by `mod(ivk + ivq, ngridk)`. That fold is exactly
what makes the sum a **cyclic cross-correlation** of `g` with itself, so

    N = ifftn(|fftn(g)|^2) / N_k

gives the whole `q` dependence in one transform. Measured on a 12x12x12
aluminium grid: **0.001 s here against Elk's 0.37 s** for the same arithmetic.
The double loop is implemented beside it (`method = "direct"`) and is not
decoration — the two share no arithmetic and agree to 2.6e-16, which is the
check that the index fold and the transform's conventions describe the same
object. `ELK-FEATURES.md` predicted this one correctly: "looks like `N_k^2` and
is a convolution".

**A wedge is unfolded, not refused, and that is the whole symmetry saving.**
`N(q)` needs a value at every one of the `n1 n2 n3` grid points, but
`eps_n(Rk) = eps_n(k)` exactly, so the irreducible wedge carries all of them.
`grid_equivalence` — `tetra.f90`'s `equiv`, Elk's own `ivkik` — is the map, and
using it turns a 12x12x12 nesting function on fcc aluminium from 1728
diagonalisations into **72**.

**The trap is the P28a family and it is silent.** The group a grid is *reduced*
with and the group it is *unfolded* with are two different questions on a
`nosym`, `noinv` or magnetic run — `denser_grid`'s four lines. A mismatch maps
every point of the complete grid onto the wrong representative and raises
nothing: what comes back is a smooth, positive, plausible `N(q)` built from
somebody else's bands. Both sides now go through one function,
`workflows/nscf.py:grid_symmetry`, and the certifying test is `reduce = False`,
which diagonalises the complete grid: the two agree to **1.8e-14**.

**Validation, and it closes inside the package — which is what
`ELK-FEATURES.md` says to look for.**

*The analytic limit.* For `eps = |k|^2` in Rydberg the nesting function is a
closed form: both deltas put `k` on the Fermi sphere, and the angular integral
of the second over the first gives

    N(q) = Omega / (4 pi^2 q)   for 0 < q <= 2 k_F,   and 0 beyond.

Fed a free-electron band on a 48^3 grid, the code reproduces the `1/q` law to
**1e-4** averaged in shells of `|q|` up to `0.6 * 2k_F` (it degrades past that,
which is the delta's own width — the two Fermi shells become tangent as
`q -> 2k_F`), and gives **zero to 1e-16** beyond `2 k_F`, which is a sharp
analytic feature reproduced exactly rather than approximately. `D(E_F)` comes
out at `Omega k_F / 2 pi^2` to 2e-3, which is the linear half of the same
normalisation — and it matters because `N` is *quadratic* in `g`, so a factor
of two in the spin degeneracy is a factor of four here and would be hard to
attribute.

*Two identities, both used as tests.* The weights are this package's DOS
convention, so `(1/N_k) sum_k g(k)` is `D(E_F)` in states/Ry/cell and therefore
the **mean of `N` over the whole q-grid is exactly `D(E_F)^2`** — measured at
3.3e-16 relative, and it ties the unfold, the delta and the transform together
in one number that `compute_dos` reaches by summing over the *wedge* with its
multiplicities instead (the two agree to **6.3e-13** on aluminium). And
`N(0) >= N(q)` for every `q` by Cauchy-Schwarz, so **the nesting peak is always
a peak away from the origin**; `NestingFunction.peak` excludes `q = 0` for that
reason rather than as a plotting convenience, since including it would report
the same uninformative wavevector for every material.

*The pairing with P21, which is the check neither route can make alone.* A
non-magnetic hydrogen chain at 5 bohr has one electron per cell in a
spin-degenerate band — half filling, so `k_F = pi/2c` and the only wavevector
connecting the two Fermi points is `2 k_F = pi/c`, which is **`q = 0.5`** in
crystal coordinates. The code peaks there, and at **99.8 per cent of the
Cauchy-Schwarz bound `N(0)`**, which is what perfect one-dimensional nesting
means: the entire Fermi surface maps onto itself under one translation. P21's
`relax_spiral_q` writes the total energy of a *magnet* as a function of the
spiral wavevector, takes `jax.grad` of it at frozen wavefunctions, and walks
downhill from `q = 0.30` to **0.500014** (P21's own number is 0.50003 at the
input's cutoff; the gap is the basis-set jump in `E(q)` that phase measures, and the
notebook's raised cutoff is why the two differ). The two share nothing — one is Fermi-
surface geometry on the paramagnet, the other a gradient of a magnetic total
energy — and `ELK-FEATURES.md` singles this pairing out as the entry's reason
for existing. Aluminium is the control that gives it meaning: a nearly-free-
electron sphere reaches a few tenths of its own bound and its largest `N(q)` is
one grid step from the origin, which is the `1/q` tail every metal has and not
a feature of the surface.

*The spin regimes, which nothing refuses and which therefore had to be
measured.* A cell with **no magnetization** must give the same `N(q)` run as
`nspin = 1`, as `nspin = 2` and as a spinor, and it does: **2.4e-13** collinear
and **1.8e-8** noncollinear (the spinor's Hamiltonian acts on a doubled space
and its eigenvalues converge to their own threshold). That is the check that
catches a factor of two in `degspin` — which is a factor of **four** in `N`,
since it is quadratic in `g`, and would be *invisible* in every other test here:
the peak still lands at `q = 0.5`, the sum rule still closes against its own
wrong `D(E_F)`, and the shape is unchanged. It exercises the `for_spin`
weights, the two-channel `g` (one Fermi level shared between the channels, so
the surface is their union) and the `t_rev` branch of the unfold at the same
time. **The band count has to be held equal across the comparison**: a
different `nbnd` is a different SCF trajectory, and on a one-dimensional metal
at `conv_thr = 1e-10` that moves the Fermi level by 2e-5 Ry, which is 2.6e-7 in
`N` and swamps what is being looked for.

*Elk's own number, and what it can and cannot say.* Task 105 on the same cell,
grid and Gaussian width gives `N(0) = 378.759` in its units against **396.42**
here — 4.7 per cent, and the whole of it is the Fermi surface underneath:
Elk's all-electron `D(E_F)` is 4.8228 states/Ry against this code's 4.9719, and
6.3 per cent of the discrepancy is that ratio *squared*. The dimensionless
`N(0)/D(E_F)^2` agrees to **1.5 per cent**, which is the comparison that is
actually like-for-like. The unit conversion itself is pinned by a test rather
than asserted: `nesting.f90` reports `occmax * Omega_BZ * wkptnr * sum g~ g~`
with `g~` carrying no degeneracy and energies in Hartree, so its number is
`(Omega_BZ/occmax) * 4 * N` — and the **4 is a squared factor of two**, which
is exactly the shape of the trap P50 found in `dielec.f90`.

**Refused by name, in the order they are checked.** A constrained
`tot_magnetization`, which has one Fermi level per channel (`input.f90`'s
two-level branch), so `g(k)` is two different surfaces and `N(q)` would have to
be resolved by channel pair — P45's refusal on the same object. A
**fixed-occupation** run, which fills a set number of bands and never searches
for a level, so `delta(eps - E_F)` has no argument. And a **spin spiral**, for
a reason that is a statement about the physics rather than about the machinery:
the quantity predicts the wavevector at which a spiral will win, so it is about
the state the spiral grows *out of*, and computing it on the spiral itself
would compare a prediction with its own answer.

**The delta defaults to a Gaussian even when the run used Methfessel-Paxton or
cold smearing**, and that is deliberate rather than an oversight: both go
negative on the wings, so `g(k)` can be negative at a k-point whose bands sit
just off the level, and `N(q)` — a product of two such — then has no sign at
all. The run's own name is accepted if it is asked for.

**Cost and peak.** One NSCF on the wedge of a dense grid, then a transform.
The working set is one `(nspin, nk_irr, nbnd)` eigenvalue array plus one real
`n1 n2 n3` box — a 24x24x24 grid is 110 kB — so **the NSCF is the whole
expense** and the correlation is free. That is what makes the grid, rather than
the band count, the convergence parameter to spend on: `g(k)` is a delta
function on a surface and nothing else in the calculation resolves it.

### P21a — `E(q)` from `dE/dq`: the spiral scan integrates its own gradient. ✅ DONE.

`pypresso/workflows/spiral.py`. Not a new quantity and not a new derivative: P21's
`compute_spiral_gradient` and the scan of P19 were both already here, and what was
missing was the line joining them. `run_spiral_scan(..., gradients=True)` takes
`dE/dq` at each point's own converged state, and `SpiralScan.integrated`
accumulates it along the path,

    E(q_n) - E(q_0) = sum_m int dq . dE/dq

by the trapezoid rule on the segments between successive wavevectors. Both factors
are in **lattice** coordinates — the units `spiral_q` and `SpiralGradient.gradient`
are written in — so the contraction needs no metric and the path may bend, which a
scan along a zone boundary does.

**The reason to want it is the finite basis, and it is the only reason.** A scan
rebuilds the plane-wave spheres at every point, so `E(q)` steps by the Pulay error
wherever a plane wave crosses `|k +- q/2 + G|^2 = ecutwfc`; the gradient is taken at
a *frozen* sphere and does not see those steps. On the hydrogen chain of
`tests/data/qe/h-chain-spiral.in` at `ecutwfc = 25`, over eleven points, the direct
energies go **uphill on 2 of 10 steps** of a curve that falls throughout, where the
integrated curve descends on all 10.

**Two error sources live in the gap between the curves and refining tells them
apart** — which is the finding, because staring at one number does not. The
trapezoid rule contributes its own `h^2` and the basis noise does not. Measured at
`ecutwfc = 25`, refining 7 -> 13 -> 25 points takes the quadrature error from
**0.051 to 0.016 to 0.003** mRy (against a spline quadrature of the same gradients)
while the gap against the energies holds at **0.139, 0.138, 0.137** — so that gap is
basis noise essentially entirely and no amount of refining `q` will touch it. Two
consequences followed:

* **A cutoff sweep alone reads as the claim failing.** The trapezoid gap goes
  0.139 (25 Ry), 0.069 (40), 0.072 (60), 0.073 (80) — a plateau, not a decay, and
  the plateau is the quadrature floor rather than a residual basis error. Removing
  it shows the real number: at `ecutwfc = 60` the two curves agree to **0.005 mRy**
  against 0.130 at 25, a factor of 26 for a factor of 2.4 in the cutoff. The two
  routes *do* converge onto each other, which is the check that neither carries a
  term the other lacks.
* **A spline quadrature is a measurement instrument and not an API.** It lives in
  the validation script; the shipped rule is the trapezoid, because the quadrature
  error is subdominant to the basis noise the feature exists for *and* is the one
  the user can remove by refining a path they already chose.

**Three things it does not buy, and the intuition that it might is the reason to
write them down.** It is **not cheaper** — every point still needs its own SCF,
since the gradient is evaluated on the converged state; it does **not converge on a
coarser k-mesh** — `dE/dq` at the frozen converged state is the exact derivative of
the *same* fixed-mesh `E(q)`, which is what
`test_the_gradient_is_the_slope_of_the_converged_energy` establishes, so it inherits
the identical Brillouin-zone error; and it wants a **tighter `conv_thr`, not a
looser one** — an energy's error is second order in the density's where a
derivative's is first, which is why `relax_spiral_q` escalates `conv_thr` by
`UPSCALE` as it closes in.

Which of the two curves is *closer* to the converged-cutoff limit at a low cutoff is
**not** settled by any of this. The integrated one is smoother, which is a different
claim, and the honest check is raising `ecutwfc` until they agree.

**One guard.** The refusals live on the gradient, not on the scan, so
`gradients=True` calls `_require_a_differentiable_spiral` on the first wavevector
*before* the loop — otherwise an unsupported spiral converges a state and then
throws it away along with the whole run.

**Tests.** `tests/unit/test_spiral_integration.py` (the accumulation, against a
cosine's own primitive and against a constant gradient on a **bent** path, where
the trapezoid rule is exact and any wrongly-inserted metric shows immediately) and
`test_integrating_the_gradient_reproduces_the_scan` in
`tests/regression/test_spiral_relaxation.py` (the gap is flat under refinement, and
the endpoints `q = 0` and `q = b3/2` report zero gradients to `SYMMETRY_ZERO`).
Notebook `12`; `docs/features.tex`. **No README row** — it is the same quantity as
the scan already there. **No `PERFORMANCE.md` pair**: neither `pw.x` nor Elk
computes a spiral `E(q)` this way, so there is nothing to time it against.

#### P21b — `dE/dq` is chunked over the k axis, because its backward pass is not free

`pypresso/forces/spiral.py`. Not a new quantity: the same gradient, regrouped so the
peak working set is bounded. The single `value_and_grad` over the whole k axis carries
`vkb(k ± q/2)` — both shifted spheres, every projector channel — and the states beside
it, for every k-point at once. On the two-atom cells this was written against that is
nothing; on a NiI2 monolayer at 81 k-points, 64 spinor bands and 26315 plane waves per
component XLA asked for **133 GiB** on a 141 GB H200, *after* a 1½-hour SCF had
converged. That is the shape of the cost: **`dE/dq` is minutes of arithmetic and the
largest allocation in the run**, and the rule "a design is not finished until its peak
working set is known" had not been applied to it.

`compute_spiral_gradient(..., k_batch=...)` sums the same terms in chunks, each a
separate `value_and_grad` whose tape is discarded before the next begins. Three things
make it exact rather than approximate, and each was a way to get it wrong:

* **only the `q`-dependent terms are chunked.** The density, Hartree,
  exchange-correlation, local, Ewald and orthonormality terms are constants of `q` at
  frozen coefficients, so their gradient is zero and the chunking exists to keep the
  backward pass off them entirely; they are evaluated once, forward only
  (`q_independent_energy`), which is what keeps `|E_frozen − E_scf|` the identity check
  it was.
* **the sub-basis is selected, never rebuilt.** Which plane waves are in each sphere and
  in what order is exactly what the frozen coefficients are written against, so a
  rebuilt sphere would be a different Miller ordering against the same numbers — right
  shape, silent nonsense. `PlaneWaveBasis` rows are indexed instead, and the spiral's
  two blocks are indexed together (`rows` runs over the `2 nk` axis).
* **a Python loop and not `lax.map`.** Reverse mode through a scan stacks every chunk's
  residuals for the backward pass, so the mapped form would hold the same peak while
  looking like it had fixed it.

Every chunk is padded to one shape with a repeat of its own first k-point at **zero
weight**, so the whole loop shares one compilation and the padding contributes exactly
nothing (both terms are linear in `weights`).

One latent bug fell out on the way: `build_projector_core`'s `nkb == 0` branch sized its
empty arrays from `kpoints.nk` rather than from the basis it was handed, which is the
same number for every caller that passes a matching pair and is not for one
differentiating a subset. A local-only pseudopotential — the hydrogen chain every other
spiral test uses — is the only thing that reaches it.

**Tests.** `test_chunking_the_k_axis_regroups_the_same_gradient` in
`tests/regression/test_spiral_relaxation.py`: five chunk sizes against the single pass,
to 1e-12 on the gradient and 1e-10 Ry on the energy, on **spinor silicon** rather than
the hydrogen chain — the chain's pseudopotential has no projectors at all, so nothing
else in that file evaluates `vkb(k ± q/2)`, which is half of what `dE/dq` is and all of
what it costs. `test_a_moved_calculation_does_not_reuse_a_stale_gradient` covers both
compiled closures now: a cache drop that reached only the one the platform default
happens to use would stay invisible until the other was asked for. **No README row and
no notebook** — it is a dial on a quantity both already carry.

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
