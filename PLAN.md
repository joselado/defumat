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
`prange` on the setup hot spots. *Check met:*
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
`nspin_mag = 4` reaches. The atomic starting guess is the scalar orbitals tensored with
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
the run rather than about the stress:

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

*Deferred:* `vc-relax` (needs the stress, P11), noncollinear forces (`qq_so`/`dvan_so` in
the constraint and nonlocal terms — refused rather than approximated), and the ion dynamics
other than BFGS (`damp`, `fire`, molecular dynamics), which are a file and a registration
each.

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
solution** — a fixed point of the SCF that is a saddle of the energy in the magnetization
direction, and one no mixer can be made to hold. **The validation is free and independent:**
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
  same `jvp`. Ultrasoft comes along rather than being refused.
- **Nothing dense is ever formed.** `dH/dk` as a matrix is `npw^2`, the same reason a dense
  diagonalisation is a test fixture here. One `jvp` per cartesian direction gives
  `v_a|psi>` for every band at every k-point; the three are separate calls because a
  `jacfwd` would hold three tangents of `vkb` at once.

*Check met:* against a central difference of the band structure at a generic k-point,
**1.2e-6 Ry bohr** norm-conserving and **8.6e-7** ultrasoft, both at a step where that is
the difference's own truncation error. `dS/dk` is identically zero for the norm-conserving
dataset and 1.5e-2 for the ultrasoft one, which is what makes the second case a test of
anything. (The finite difference has one failure mode of its own, and it is the
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

    epsilon_infinity = 13.806646105   against QE's   13.806375297

a difference of **2.7e-4**, two parts in 1e5. What is left is the same thing that puts a
floor under the eigenvalue comparison (`tests/tolerances.py`): QE interpolates every
radial form factor from a `dq = 0.01` table where this code integrates it directly. The
tensor comes out **diagonal to 3.6e-15** with nothing imposing the crystal class.

**The Born effective charges come from the same two solutions**, and the bare phonon
perturbation is not transcribed either: `dV_bare/du |psi>`, which `dvqpsi_us.f90` builds
term by term, is one `jvp` through `Calculation.at_positions` at frozen `v_scf` — the same
method the force differentiates, which already moves `vltot` and `vkb` traceably.
`zstar_eu.f90` pairs it with the self-consistent `dpsi/dE`. Silicon's `Z*` is zero by
symmetry in a converged calculation, so the benchmark's **-0.07568** is a residue — 4
against an electronic part near 4.076 — which makes it a sharper check than the dielectric
constant. *Check met:* **-0.075715** on both atoms, against the five decimals `ph.x`
prints, with the off-diagonal entries at 1e-17.

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

**Refused rather than approximated**, each by name and each because the missing piece is
invisible in the answer: **ultrasoft and PAW** (the response density needs `dbecsum` and
the augmentation charge's own response, `addusdbec`/`lr_addusddens`, and the perturbed
`D_ij` needs `int3` from `newdq`); **metals** (`orthogonalize`'s smearing branch replaces
the sharp projector with occupation-difference weights, and the Fermi level itself shifts —
`ef_shift`; the level's own derivative already exists, since P22 wrote `bisect_fermi`'s
`custom_jvp`, so this is the projector's gap); **noncollinear magnetism and spin-orbit
coupling** (`incdrhoscf_nc`, `set_int3_nc`); **DFT+U** (`adddvhubscf` — the induced
potential carries a `dns` that is not a function of `drho`); and spin spirals.

*Notebook 19.*

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
