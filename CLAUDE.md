# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A ground-up reimplementation of Quantum ESPRESSO in Python + JAX ("pypresso"). The
Fortran QE 7.5 release is vendored here **as reference material only** — it is read to
understand algorithms and to validate numerical results, never modified or compiled into
the deliverable.

**Status: P0–P2 are done** — scaffolding and precision policy, units, the QE reference
parser, the `pw.x` input parser, geometry (`latgen` for every ibrav, Monkhorst-Pack grids,
band paths), and the plane-wave basis (G-vectors with gamma-only support, dense and smooth
FFT grids, per-k plane waves padded to `npwx`, sphere↔box transforms). `PLAN.md` §3 tracks
the phases and records the conventions each one uncovered — read it before writing code.
**P3 (pseudopotentials: UPF parsing, radial→G transforms, projectors) is next.**

## Layout

- `quantum_espresso/qe-7.5-ReleasePack/qe-7.5/` — QE 7.5 Fortran sources. **Read-only.**
- `quantum_espresso/Doc-QE-7.5/Doc-7.5/` — input-file documentation (`INPUT_PW.txt` is the
  authoritative spec for the `pw.x` input namelists/cards) and theory PDFs.
- `pypresso/` — the Python package. `tests/` alongside it; `tests/data/pseudo/` holds the
  committed UPF files (QE's test-suite downloads rather than ships them).
- Git repository, with `quantum_espresso/` gitignored — 285 MB of vendored reference does
  not belong in history. Tests that need it skip cleanly when it is absent.
- `/u/40/ladovj1/unix/Documents/...` is a symlink to `/u/40/ladovj1/data/Documents/...`;
  both working directories are the same files.

## Scope

First milestone, in this order: **SCF → band structure → DOS**, for `pw.x` with
norm-conserving pseudopotentials, LDA/PBE, k-point grids and gamma-only. This is a large
project that will keep growing, so structure matters more than speed of delivery — see
`PLAN.md` for the architecture, the phase breakdown, and the validation strategy. Read it
before writing code.

Out of scope until the above works: ultrasoft/PAW, EXX, DFT+U, non-collinear/SOC, phonons
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

## Where each subsystem lives in the reference source

Paths relative to `quantum_espresso/qe-7.5-ReleasePack/qe-7.5/`.

| Subsystem | Reference | Notes for the port |
|---|---|---|
| Top-level driver | `PW/src/run_pwscf.f90` → `init_run.f90` → `electrons.f90` | `electrons_scf` is the SCF loop; ignore the EXX/RISM/OSCDFT branches |
| SCF iteration body | `c_bands.f90`, `sum_band.f90`, `v_of_rho.f90`, `mix_rho.f90` | diagonalize → build density → build potential → Broyden mix |
| Hamiltonian application | `h_psi.f90`, `vloc_psi_*.f90`, `add_vuspsi.f90`, `g2_kin.f90`, `s_psi.f90` | the hot path; the natural unit of `jit`/`vmap`; `k` must stay a traced argument, see `PLAN.md` §6 |
| Iterative diagonalization | `KS_Solvers/Davidson/`, `KS_Solvers/CG/`, `KS_Solvers/PPCG_legacy/`, `KS_Solvers/RMM/` | Davidson is QE's default |
| FFT / G-vector grids | `FFTXlib/`, `PW/src/data_structure.f90`, `Modules/recvec*.f90` | replace with `jax.numpy.fft`; the sphere-to-box G-vector mapping still has to be reproduced |
| Pseudopotentials | `upflib/` (`read_upf_new.f90`, `pseudo_types.f90`, `init_us_2.f90`, `sph_bes.f90`, `ylmr2.f90`) | UPF v2 XML parsing + radial→G-space transforms |
| XC functionals | `XClib/` | must be reimplemented in pure JAX — a `libxc` binding is neither differentiable nor GPU-capable (see `PLAN.md` §6) |
| Structure / symmetry / k-points | `PW/src/symm_base.f90`, `symme.f90`, `kpoint_grid.f90`, `setup.f90`, `Modules/cell_base.f90` | `ibrav` lattice conventions live in `Modules/latgen.f90` |
| Ewald / local potential / forces / stress | `PW/src/ewald.f90`, `setlocal.f90`, `forces.f90`, `stress.f90` | forces/stress come after energies are correct, and should come from autodiff rather than the hand-derived Fortran expressions |
| Velocity / position operator | `PW/src/commutator_Hx_psi.f90`, `PP/src/` Berry-phase code | QE hand-codes `[H,r]`; here it should fall out of `jacfwd` of `H(k)` w.r.t. `k` |
| Input parsing | `Modules/read_input.f90`, `PW/src/input.f90`, `Modules/input_parameters.f90` | defaults for every input variable are declared in `input_parameters.f90` |
| Occupations / smearing | `PW/src/gweights.f90`, `Modules/wgauss.f90`, `Modules/w0gauss.f90`, `PW/src/set_occupations.f90` | |
| NSCF / band structure | `PW/src/non_scf.f90`, `PP/src/bands.f90`, `PP/src/plotband.f90` | fixed density, diagonalize once per k on an explicit path |
| DOS | `PW/src/tetra.f90`, `PP/src/dos.f90`, `PP/src/projwfc.f90` | `tetra.f90` has both the linear and the Bloechl-corrected tetrahedron method; `projwfc.f90` is PDOS, later |

Fortran conventions that carry over: arrays are column-major and 1-indexed, so index order
must be reversed when transcribing loops; internal units are Rydberg atomic units (energy
in Ry, length in bohr) throughout `PW/`.

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

Dependencies live in the **base anaconda env** (`/u/40/ladovj1/unix/apps/anaconda3`):
JAX 0.11.0, NumPy 2.4.6, SciPy 1.18, Numba 0.65, equinox 0.13.8 (verified working with this
JAX under x64). Development is CPU-only here; the JAX paths must run unchanged on GPU, so
correctness is established in float64 on CPU and performance work is a later, separate phase.

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
