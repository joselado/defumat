# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A ground-up reimplementation of Quantum ESPRESSO in Python + JAX ("pypresso"). The
Fortran QE 7.5 release is vendored here **as reference material only** — it is read to
understand algorithms and to validate numerical results, never modified or compiled into
the deliverable.

**Status: the first milestone — SCF, band structure, DOS — is met**, with ultrasoft/PAW,
LDA/GGA and collinear spin. P0–P9 and P12–P13 are done bar Wyckoff input in P6; P10 has
had one pass. A silicon SCF reproduces QE's total energy to **~1e-9 Ry** term by term, its
band structure to **0.0002 eV**, and metals with every smearing to ~2.5e-8 Ry.
**Ultrasoft and PAW pseudopotentials are supported** and match QE to **≤3e-9 Ry** on 2-
and 8-atom silicon (P12). **PBE, revPBE and PBEsol** work on all three pseudopotential
kinds, matching QE to **≤6e-9 Ry** and 5e-5 eV in the bands (P13). **The density of states**
(P8) has both the smearing and the tetrahedron families, the latter also as an occupation
scheme inside the SCF, matching QE's three aluminium benchmarks to 2.5e-8 Ry. **Collinear
spin** (P9) matches eight LSDA benchmarks — nickel's total energy to **1.2e-9 Ry** and its
magnetic moment to the two decimals QE prints (0.7280 against 0.73).
**Spin-orbit coupling** (P14) is in as well: `noncolin`/`lspinorb` give two-component
spinor wavefunctions and the `j`-resolved projectors of a fully-relativistic dataset, on
norm-conserving, ultrasoft and PAW pseudopotentials, matching QE's three platinum
benchmarks to **≤1.3e-8 Ry**.
`PLAN.md` §3 tracks the phases and records the transcription traps each one uncovered —
read it before writing code. P4 is complete: a block Davidson eigensolver behind a name
registry, with the dense solver kept as its reference, seeded from the pseudo-atomic
orbitals as QE seeds it. P6 is complete too: automatic k-grids are reduced to the
irreducible wedge. P10's first pass puts pypresso within **2–4x of serial Quantum ESPRESSO
per SCF iteration** on the same machine, ultrasoft and PAW included — see
`PERFORMANCE.md`. **Outstanding:** the projected DOS (`projwfc.x`), Wyckoff input, and the
rest of P10 (k-axis sharding and GPU). Non-collinear *magnetism* — a spin-orbit run whose
magnetization is nonzero — is partly in: the density, potential and occupations carry it,
but symmetrising it and gradient-correcting it are not written, so those combinations are
refused rather than approximated.

## Layout

- `quantum_espresso/qe-7.5-ReleasePack/qe-7.5/` — QE 7.5 Fortran sources. **Read-only.**
- `quantum_espresso/Doc-QE-7.5/Doc-7.5/` — input-file documentation (`INPUT_PW.txt` is the
  authoritative spec for the `pw.x` input namelists/cards) and theory PDFs.
- `pypresso/` — the Python package. `tests/` alongside it; `tests/data/pseudo/` holds the
  committed UPF files (QE's test-suite downloads rather than ships them).
- Git repository, with `quantum_espresso/` gitignored — 285 MB of vendored reference does
  not belong in history. Tests that need it skip cleanly when it is absent.
- The two configured working directories are one directory: one path is a symlink to the
  other, so the same file can arrive under either prefix. Do not treat them as separate
  copies.

## Scope

First milestone, in this order: **SCF → band structure → DOS**, for `pw.x` with
norm-conserving pseudopotentials, LDA/PBE and k-point grids — **now met**, and extended
since with ultrasoft/PAW, the PBE family and collinear spin. This is a large project that
will keep growing, so structure matters more than speed of delivery — see `PLAN.md` for
the architecture, the phase breakdown, and the validation strategy. Read it before writing
code.

**Gamma-only is a gap, not a feature.** `K_POINTS gamma` selects the half-sphere storage
of the gamma-point trick, and that storage is generated but not consumed anywhere:
`h_psi` would need `vloc_psi_gamma`'s packing, the eigensolver `regterg`'s real overlaps,
and `addusdens`/`newd` their `fact = 2`. Such a run is silently substituted by an explicit
k = 0 with the full sphere, which is the same physics at twice the storage, and it says so.

**Ultrasoft and PAW are in scope and implemented** (P12): the two-grid split, the
augmentation charge, the overlap operator, self-consistent `D_ij`, and PAW's one-centre
terms. **Gradient-corrected functionals are too** (P13) — PBE, revPBE and PBEsol, on the
plane-wave grid and on the PAW spheres — so the PBE datasets that most published
ultrasoft/PAW work uses run here. The functional comes from the pseudopotentials' headers
unless `input_dft` overrides it, and an unimplemented one is refused rather than silently
replaced by LDA.

**Collinear spin is in scope and implemented** (P9): `nspin = 2` gives the density, the
potential, `becsum`, `D_ij`, the eigenvalues and the wavefunctions a leading channel axis,
and one SCF iteration diagonalises a different Hamiltonian per channel. Whichever
occupation scheme is in use decides how many Fermi levels there are — one shared between
the channels, or one each when `tot_magnetization` constrains the magnetisation — and both
the smearing and the tetrahedron families implement both.

**Spin-orbit coupling is in scope and implemented** (P14): `noncolin = .true.` makes a
wavefunction a two-component spinor of length `2 npwx`, so there is *one* Hamiltonian on a
space twice as large rather than two Hamiltonians, and `lspinorb = .true.` puts the
`j`-resolved projectors of a fully-relativistic dataset into it. **Keep QE's three spin
numbers apart**, because collapsing them is the mistake that makes a spin-orbit run
allocate a magnetization it does not have: `nspin` says which regime (1, 2 or 4), `npol`
how many components a *wavefunction* has, and `nspin_mag` how many a *density* has —
which is **one** for a nonmagnetic spin-orbit run, exactly as for an unpolarized one. That
is why such a run costs about what a doubled unpolarized one costs: the density, the
potential, the exchange-correlation functional and the symmetrisation are untouched, and
all the new physics is in the spinors and in `D_ij` becoming a complex 2x2 matrix in spin
space. Non-collinear *magnetism* (`nspin_mag = 4`) is built but only partly validated —
`sym_rho`'s vector rotation and `gradcorr`'s local-frame rotation are not written and are
refused, so such a run needs `nosym` and an LDA functional.

Out of scope until the above works: EXX, DFT+U, phonons
(`PHonon/`), Car-Parrinello (`CPV/`), and everything in `EPW/`, `TDDFPT/`, `HP/`, `GWW/`.
The code should nonetheless be shaped so these are additions, not rewrites.

## Why JAX (this drives the design)

Two reasons, both of which constrain how code is written:

1. **Autodifferentiation.** Response and higher-order properties — polarization, dielectric
   response, second harmonic generation, forces, stress — should come from differentiating
   the code rather than from separately hand-derived expressions. This is the main reason
   for JAX, not a bonus. Consequences are in `PLAN.md` §6 and they are binding: the compute
   path must be differentiable end to end, including the XC functional and the k-dependence
   of the Hamiltonian.
2. **GPU.** The same JAX code must run on GPU unchanged (development is CPU-only here — no
   GPU on this machine).

Performance matters. It does not have to be optimal in the first version, but no design
choice should make good performance unreachable without a rewrite.

## Tutorial notebooks

`notebooks/` holds worked examples on concrete systems — the readable counterpart to the
test suite. **Every new feature adds a notebook or extends an existing one; a phase is not
finished until its notebook exists.** Demonstrate on the two-atom silicon cell from
`test-suite/pw_scf/scf.in` wherever possible, compare against the committed QE benchmark
whenever the reference contains the quantity, and commit the notebook executed so it reads
without being run. Each notebook also has a `.md` export committed beside it — raw `.ipynb` is unreadable in a
plain editor or a diff — regenerated together with the notebook by `tools/export_notebooks.sh`.
`notebooks/README.md` holds the index and the full conventions.

## Performance

**The measurement is single-core pypresso against single-core Quantum ESPRESSO on the
same machine and the same input.** That comparison is the starting point of any
performance discussion, not a summary of one:

```bash
python3 tools/compare_qe.py benchmarks/si-1k.in --repeats 5
```

It needs `pw.x` built serially once (`./configure --disable-parallel --disable-openmp &&
make -j pw` inside the vendored tree; the binary is gitignored along with the rest of it).
The tool pins both codes to one core — JAX otherwise uses every core and the comparison
flatters it by the core count — and reads QE's own timing report, so the numbers on the
QE side are QE's, not a stopwatch around it.

The benchmark inputs live in `benchmarks/`, and are **single k-point** on purpose: both
codes parallelise over k, so a multi-k comparison measures batching rather than the cost
of the physics. `si-1k.in` is the test suite's silicon at `ecutwfc = 12`; `si-1k-ecut40.in`
is the same cell at a production cutoff, where scaling starts to show.

`PERFORMANCE.md` is the running log: the comparison, where the time goes, what each change
was worth, and the backlog. **Add a measurement to it whenever a feature lands or a hot
spot moves** — including the QE ratio, not only an internal timing. `tools/benchmark.py
<input>` gives the component breakdown when a ratio needs explaining.

## Non-negotiable conventions

- Pure Python. JAX for anything numerical that runs inside the SCF/diagonalization loop;
  Numba only for host-side setup loops (G-vector enumeration, symmetry search, radial
  tables), never inside a jitted path.
- JAX code must stay GPU-ready and differentiable: static shapes, no host syncs in the
  inner loop, no Python branching on traced values, no in-place tricks that break `grad`.
  Pad plane-wave arrays to `npwx` with a mask instead of using per-k shapes.
- **Object-oriented is encouraged, mutable global state is not.** This is deliberately not a
  literal transcription of the Fortran: use classes with bound methods where they make the
  code read better (`ham.apply(psi, k)`, `density.symmetrize()`, `pseudo.projectors(k)`).
  The constraint is that any class crossing a `jit`/`grad` boundary is frozen and
  pytree-registered — methods are fine, mutation and module-level globals are not. (QE's
  shared-module globals are exactly what not to copy.) **`equinox.Module` is the base
  class** for all such state objects; static config uses `eqx.field(static=True)`.
- **Never hardcode a dtype.** Single precision has to stay viable for GPU, so real and
  complex dtypes come from the policy object in `config.py`, never from literals like
  `jnp.complex128` or `1.0j`. x64 is still enabled and all QE validation is float64;
  float32 is a performance mode, never one a correctness claim is made in.
- Pluggable pieces — XC functionals, mixers, eigensolvers, smearing, DOS schemes — go
  behind a name registry, so adding one is a new file plus a registration, not an edit to a
  growing branch in the driver.
- Parallelism in JAX is not OpenMP: XLA already threads each op on CPU, and explicit
  parallelism comes from `vmap` over the k-point axis plus `jax.sharding` over that same
  axis (CPU cores as devices now, GPUs later). Keep k the leading independent axis of every
  wavefunction-shaped array so this stays available. Numba `prange` is the right tool for
  the host-side setup loops only.
- Rydberg atomic units internally (Ry, bohr), matching QE; convert only in `io/`.
- **`nspin`, `npol` and `nspin_mag` are three different numbers.** `nspin` says which
  regime is in force, `npol` is the number of spinor components of a *wavefunction*, and
  `nspin_mag` the number of components of a *density*. They coincide for 1 and 2 and come
  apart at 4, where `npol = 2` and `nspin_mag` is 4 only if the run actually carries a
  magnetization. All three are static; `System` exposes them as properties so no call site
  recomputes the rule.
- **The spin channel is the leading axis, and it is squeezed on the way out.** Densities,
  potentials and `becsum` are `(nspin, ...)` internally with no special case for one
  channel; the result objects (`SCFResult`, `NSCFResult`, `DensityOfStates`,
  `BandStructure`) drop that axis when `nspin = 1` and expose a `*_by_spin` property that
  always has it. `k` stays the leading *independent* axis inside each channel, which is
  what the batching and the eventual sharding rest on. `nspin` is static
  (`eqx.field(static=True)`) because it is an array rank, not a value.

## Where each subsystem lives in the reference source

Paths relative to `quantum_espresso/qe-7.5-ReleasePack/qe-7.5/`.

| Subsystem | Reference | Notes for the port |
|---|---|---|
| Top-level driver | `PW/src/run_pwscf.f90` → `init_run.f90` → `electrons.f90` | `electrons_scf` is the SCF loop; ignore the EXX/RISM/OSCDFT branches. Its `ethr` schedule and `dr2` convergence test are transcribed — `conv_thr` means the same thing here as in a `pw.x` input |
| SCF iteration body | `c_bands.f90`, `sum_band.f90`, `v_of_rho.f90`, `mix_rho.f90` | diagonalize → build density → build potential → Broyden mix |
| Hamiltonian application | `h_psi.f90`, `vloc_psi_*.f90`, `add_vuspsi.f90`, `g2_kin.f90`, `s_psi.f90` | the hot path; the natural unit of `jit`/`vmap`; `k` must stay a traced argument, see `PLAN.md` §6 |
| Iterative diagonalization | `KS_Solvers/Davidson/`, `KS_Solvers/CG/`, `KS_Solvers/PPCG_legacy/`, `KS_Solvers/RMM/` | Davidson is QE's default and is ported (`solvers/davidson.py`); note `c_bands.f90` re-enters `cegterg` up to 5 times, so QE's real budget is 100 steps |
| FFT / G-vector grids | `FFTXlib/`, `PW/src/data_structure.f90`, `Modules/recvec*.f90` | replace with `jax.numpy.fft`; the sphere-to-box G-vector mapping still has to be reproduced |
| Pseudopotentials | `upflib/` (`read_upf_new.f90`, `pseudo_types.f90`, `init_us_2.f90`, `sph_bes.f90`, `ylmr2.f90`) | UPF v2 XML parsing + radial→G-space transforms. **`msh` is one or two points *past* 10 bohr** — QE's loop takes the first index beyond the cutoff, not the last inside; getting that wrong is worth 1e-6 Ry on a `psl` dataset and nothing at all on `Si.pz-vbc` |
| Ultrasoft augmentation | `upflib/qvan2.f90`, `uspp.f90` (`aainit`), `qrad_mod.f90`, `PW/src/addusdens.f90`, `newd_acc.f90`, `s_psi.f90` | `Q_ij(G)`, `becsum`, the overlap operator, and `D_ij` rebuilt each iteration from the potential |
| PAW one-centre terms | `PW/src/paw_onecenter.f90`, `paw_init.f90`, `paw_symmetry.f90`, `upflib/radial_grids.f90` (`hartree`) | radial Poisson (a Numerov tridiagonal solve — transcribe it, do not substitute the closed form), a Gauss-Legendre×φ spherical quadrature for XC, and `becsum` symmetrisation, which is **not optional** on a reduced k-set. A GGA adds `PAW_gcxc_potential`: the quadrature grows (`xlm`), the vector field is expanded two multipoles past the density, and its θ component is divided by `sin θ` before projection |
| XC functionals | `XClib/`, `PW/src/gradcorr.f90` | must be reimplemented in pure JAX — a `libxc` binding is neither differentiable nor GPU-capable (see `PLAN.md` §6). Only the **energy** is written down; `v_xc`, and a GGA's `v1`/`v2`, come from `jax.grad`. QE composes a functional from four independently chosen slots and UPF headers name all four, so `xc/functional.py` does the same |
| Spin-orbit coupling | `upflib/init_us_1.f90` (`fcoef`, `dvan_so`), `upflib/spinor.f90`, `upflib/sph_ind.f90`, `upflib/upf_spinorb.f90` (`transform_qq_so`), `PW/src/newd_acc.f90` (`newd_so`), `PW/src/compute_becsum.f90` (`add_becsum_so`), `PW/src/vloc_psi_acc.f90` (`vloc_psi_nc`), `PW/src/add_vuspsi_acc.f90`, `PW/src/usnldiag.f90` | `init_us_1` builds `fcoef` for every matching `(l, j)` pair, uses it for `dvan_so`, and **then** zeroes the cross-radial entries — everything downstream consumes the *zeroed* array and has no check of its own, so one array used for both is a correct `dvan_so` and a silently wrong `qq_so`/`deeq_nc`/`becsum` |
| Structure / symmetry / k-points | `PW/src/symm_base.f90`, `symme.f90`, `kpoint_grid.f90`, `setup.f90`, `Modules/cell_base.f90` | `ibrav` lattice conventions live in `Modules/latgen.f90`. `kpoint_grid` is called with the *lattice* point group and fixed up afterwards; reducing directly with the crystal's symmetries reaches the same orbits. Two rules in `symm_base.f90` change the **FFT grid**: dimensions must be a multiple of the fractional translations' denominators (`fft_fact`), and a cell that is a supercell has fractional translations disabled altogether |
| Starting wavefunctions | `PW/src/wfcinit.f90`, `Modules/atomic_wfc_mod.f90`, `upflib/atwfc_mod.f90` | the projectors' expression with `chi` for `beta` — but the phase is `i^l`, not `(-i)^l` |
| Ewald / local potential / forces / stress | `PW/src/ewald.f90`, `setlocal.f90`, `forces.f90`, `stress.f90` | forces/stress come after energies are correct, and should come from autodiff rather than the hand-derived Fortran expressions |
| Velocity / position operator | `PW/src/commutator_Hx_psi.f90`, `PP/src/` Berry-phase code | QE hand-codes `[H,r]`; here it should fall out of `jacfwd` of `H(k)` w.r.t. `k` |
| Input parsing | `Modules/read_input.f90`, `PW/src/input.f90`, `Modules/input_parameters.f90` | defaults for every input variable are declared in `input_parameters.f90` |
| Occupations / smearing | `PW/src/gweights.f90`, `Modules/wgauss.f90`, `Modules/w0gauss.f90`, `PW/src/set_occupations.f90` | |
| NSCF / band structure | `PW/src/non_scf.f90`, `PP/src/bands.f90`, `PP/src/plotband.f90` | fixed density, diagonalize once per k on an explicit path |
| DOS | `PW/src/tetra.f90`, `PP/src/dos.f90`, `PP/src/projwfc.f90` | `tetra.f90` has both the linear and the Bloechl-corrected tetrahedron method; `projwfc.f90` is PDOS, later |

Fortran conventions that carry over: arrays are column-major and 1-indexed, so index order
must be reversed when transcribing loops; internal units are Rydberg atomic units (energy
in Ry, length in bohr) throughout `PW/`.

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

## Reading beyond the source

The vendored Fortran is the primary reference and transcription from it is the method.
Where an algorithm's *reasoning* is not in the source — why a preconditioner has the form
it does, what a method's convergence properties are, what the alternatives are — **arXiv
is a legitimate thing to consult during implementation.** Cite what was used in the module
docstring, the same way the Fortran file it came from is cited.

## Validation against reference QE

`quantum_espresso/qe-7.5-ReleasePack/qe-7.5/test-suite/` holds ~100 test cases with
committed reference outputs — use these as the ground truth rather than re-running QE.
For the SCF core, `test-suite/pw_scf/` is the relevant set: `scf-*.in` are the inputs and
`benchmark.out.git.inp=scf-*.in` the expected outputs (total energy, eigenvalues, forces,
stress are all parseable from those files). `test-suite/pw_atom/`, `pw_lsda/`, `pw_metal/`,
`pw_relax/` extend coverage. Test pseudopotentials are in `pseudo/` (e.g. `C.UPF`,
`Si_r.upf`, `N-PBE.upf`).

The test-suite's pseudopotential files are **not** shipped — inputs name files like
`Si.pz-vbc.UPF` that `test-suite/check_pseudo.sh` downloads from
`pseudopotentials.quantum-espresso.org`. Fetch them once into `tests/data/pseudo/` and
commit them. The canonical first target is `test-suite/pw_scf/scf.in` (Si diamond, LDA,
`ecutwfc=12`, 2 k-points, 15³ FFT grid).

The testing method is running the *same input* through real QE and through pypresso and
comparing numbers. Building the Fortran QE is only needed when a comparison is not already
covered by a committed benchmark (likely for `bands`/`dos` runs); when that happens, store
the generated reference output alongside the test so it never has to be regenerated.
Tolerances per quantity are listed in `PLAN.md`.

## Environment

Dependencies live in the **base anaconda env** — there is no project virtualenv, so
`python3` is already the right interpreter. JAX 0.11.0, NumPy 2.4.6, SciPy 1.18, Numba
0.65, equinox 0.13.8 (verified working with this JAX under x64). Development is CPU-only
here; the JAX paths must run unchanged on GPU, so correctness is established in float64 on
CPU and performance work is a later, separate phase.

Compiled kernels are cached in `~/.cache/pypresso/jax` so that only the first run of a
process pays for them; `PYPRESSO_CACHE_DIR` moves it and `PYPRESSO_CACHE_DIR=off` disables
it.

```
python3 -m pytest                      # whole suite
python3 -m pytest -m unit              # fast checks only (markers: unit, regression, slow)
python3 -m pytest tests/unit/test_qeref.py::test_scf_silicon   # a single test
python3 -m pypresso.cli inspect <qe-output>   # summarise what the parser reads
tools/export_notebooks.sh                     # re-execute notebooks + refresh .md exports
```

## JAX rules

- **`jax.config.update("jax_enable_x64", True)` must be set before any array is created.**
  JAX defaults to float32; SCF will not converge and no comparison against QE benchmarks
  will be meaningful in single precision. Set it once in the package `__init__`, before any
  other import that touches JAX. Enabling x64 only *permits* 64-bit — the actual dtype of
  every array still comes from `config.dtypes` (see conventions above).
- The SCF loop's convergence test is data-dependent, so keep the loop in Python; `jit` the
  iteration body (`h_psi` → diagonalize → density → potential → mix). Inside the
  eigensolver, use `lax.while_loop`/`fori_loop` with a fixed subspace size so the solver
  stays on device.
- Batch over k-points with `vmap` where padded shapes allow, with a Python-loop-over-k
  fallback (jitted body) when memory forbids — selectable at runtime, not a rewrite.
- Use `donate_argnums` for the large wavefunction and density buffers.

`PLAN.md` §1 and §5 hold the full reasoning and the rest of the GPU notes.
