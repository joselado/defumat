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
`solvers/` — a name registry over `dense.py` and `davidson.py`, the latter transcribed
from `cegterg.f90` and the default. *Check met:* eigenvalues match QE to <1e-3 eV wherever
they are printed, and Davidson matches the dense solver to 1e-12 Ry. The dense solver
stays as correct-by-construction ground truth; `Hamiltonian` likewise keeps two matrix
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

Still out of P8: the projected DOS (`projwfc.x`), which needs atomic-orbital projections
rather than eigenvalues alone.

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
passes and the two eigensolvers agree to 2e-13 Ry on the total energy.

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
