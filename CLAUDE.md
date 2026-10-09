# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**This file is the rules. `PLAN.md` is the record.** Everything about *how* to work here —
conventions, refusals, where the reference source lives, what a finished phase looks like —
is below. Everything about *what was found* — per-phase validation numbers, the trap each
phase uncovered, what a refusal was measured at — is in `PLAN.md` §3, one section per phase.
Do not restate a phase's findings here; add them there.

## How to talk to the user

**The user is a physicist, so write to them as one, in plain English.** Explain what a
thing does physically before what it does in code, and keep the sentences short. When a
choice genuinely needs their input, put it in terms a physicist who writes some Python
will recognise — what the option changes in the calculation, what it costs, what it gets
wrong — not in software-engineering vocabulary (pytrees, tracing, cache invalidation)
that assumes the answer is about the code rather than the physics.

**This is about talking, not about writing.** It binds replies, questions and summaries in
the session. It does **not** loosen `PLAN.md`, `PERFORMANCE.md`, `docs/features.tex`,
docstrings or comments: the record stays exact, technical and numeric, because the whole
rule of this project is that a claim is a number rather than an adjective.

## What this project is

A ground-up reimplementation of Quantum ESPRESSO in Python + JAX ("defumat"). The
Fortran QE 7.5 release is vendored here **as reference material only** — it is read to
understand algorithms and to validate numerical results, never modified or compiled into
the deliverable.

**Status: well past the first milestone.** SCF, band structure and DOS are met, and on top
of them: ultrasoft and PAW, the PBE family, collinear spin, spin-orbit coupling and
noncollinear magnetism, spin spirals, DFT+U, forces, stress and both relaxations, the
topological invariants, the whole linear-response stack (dielectric constants, Born
charges, phonons at `Gamma`), third derivatives (Raman, electrostriction, the
elasto-optic tensor), and a long tail of quantities taken from Elk that `pw.x` does not
have. Phases run **P0 through P90**; the ones still open, and the exact term each is
missing, are indexed at the head of `PLAN.md` §3.

**Where to look for what:**

| question | file |
|---|---|
| what a phase found, and what it was measured at | `PLAN.md` §3 (one section per phase) |
| what is implemented, and whether QE or Elk has it | `README.md`'s feature table |
| how a user runs a feature, and what it refuses | `docs/features.tex` |
| what something costs, in time and in memory | `PERFORMANCE.md` |
| what is not here and what term is missing | `PLAN.md` §3, "What is outstanding" |
| a survey of Elk's tasks against QE 7.5 | `ELK-FEATURES.md` |
| what is known to be broken, and what to do about it | `OPEN.md` |
| what a read-only audit suspects but has not measured | `AUDIT-2026-09-18.md` |
| where the peak working set goes, sized and ranked | `MEMORY-AUDIT.md` |
| what is left to do about magnetism, sized, and what each piece needs first | `MAGNETISM-NEXT.md` |
| what is left to do about continuing one run from another, sized | `CONTINUATION-NEXT.md` |
| what still refuses an ultrasoft or PAW dataset, and what each piece needs first | `AUGMENTATION-NEXT.md` |
| what is left to do about GPU memory after `memory_mode`, sized and ranked | `GPU-MEMORY-NEXT.md` |
| what is left to do about GPU speed and the Davidson stall, sized, and the replay tools | `GPU-SPEED-NEXT.md` |
| which mixer a cell with vacuum wants, what the literature has, and what is left | `VACUUM-MIXING-NEXT.md` |
| real-time propagation, high harmonics and the third harmonic: the plan, and what is left of it (the frequency-domain hierarchy) | `HARMONICS-NEXT.md` |
| which QE or Elk routine each subsystem was ported from | `.claude/rules/reference-source-map.md` |
| the full account of each recurring trap | `.claude/skills/trap-check/traps.md` |

**The claims in this project are numbers, not adjectives.** A phase is done when it has a
concrete figure against `pw.x`, against Elk, or against an identity that shares no
machinery with it — and when the *absence* of a term is measured and refused by name
rather than approximated. That habit is the reason `PLAN.md` is as long as it is, and it
is the one thing not to compress away.

## Layout

- `quantum_espresso/qe-7.5-ReleasePack/qe-7.5/` — QE 7.5 Fortran sources. **Read-only.**
- `quantum_espresso/Doc-QE-7.5/Doc-7.5/` — input-file documentation (`INPUT_PW.txt` is the
  authoritative spec for the `pw.x` input namelists/cards) and theory PDFs.
- `defumat/` — the Python package. `tests/` alongside it; `tests/data/pseudo/` holds the
  committed UPF files (QE's test-suite downloads rather than ships them).
- Git repository, with `quantum_espresso/` gitignored — 285 MB of vendored reference does
  not belong in history. Tests that need it skip cleanly when it is absent.
- The two configured working directories are one directory: one path is a symlink to the
  other, so the same file can arrive under either prefix. Do not treat them as separate
  copies.

## Scope

The first milestone was **SCF → band structure → DOS** for `pw.x` with norm-conserving
pseudopotentials, LDA/PBE and k-point grids, and it is met. This is a large project that
will keep growing, so structure matters more than speed of delivery — read `PLAN.md` for
the architecture, the phase breakdown and the validation strategy before writing code.

**What is in scope and implemented**, with the rule that binds each piece, is
`.claude/rules/scope.md`; it loads whenever a file under `defumat/`, `tests/` or
`notebooks/`, or `docs/features.tex`, is read, and
`README.md`'s feature table is the short list.

**Out of scope until the above works:** EXX, the Liouville-Lanczos route to a spectrum
(`TDDFPT/`), Car-Parrinello (`CPV/`), and everything
in `EPW/`, `HP/`, `GWW/`. The code should nonetheless be shaped so these are additions,
not rewrites.

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

## The front door is `Calculator`

`defumat/calculator.py` (P38): a `System` with its pseudopotentials, every workflow a
`get_*` method on it, and `from defumat import Calculator` the one import a script needs.
It is a facade and nothing else, nothing on it mutates, and a new feature adds its `get_*`
in the same pass as its entry point. The binding rules (no state on `System`, the one-slot
cache, `SHARED_OPTIONS` forwarded by name) are in `.claude/rules/calculator.md`, which loads
when the calculator, a workflow or a response module is read.

## What a finished phase leaves behind

Five things, each checked rather than assumed, since each has gone stale silently at least
once: **a number** against `pw.x`, Elk or an independent identity, in `PLAN.md` §3; **a row**
in the README's feature table, with its `QE` and `Elk` ticks checked in their source; **an
entry** in `docs/features.tex` with an executed snippet and its amber box; **a notebook**;
and **a timing** against the code it was taken from. The full rules are
`.claude/skills/phase-audit/deliverables.md`, and the `phase-audit` skill runs the audit.

## Tutorial notebooks

Every new feature adds a notebook or extends one, executed and committed with its `.md`
export, about the physics only, with one figure, and under ten minutes to execute. The full
rules are `.claude/rules/notebooks.md`, which loads when anything under `notebooks/` is read.

## Performance

**The measurement is single-core defumat against single-core Quantum ESPRESSO (or Elk) on
the same machine and the same input**, and every feature taken from either is timed against
it, with the pair in `PERFORMANCE.md`:

```bash
python3 tools/compare_qe.py benchmarks/si-1k.in --repeats 5
```

Three rules about this code's own clock, each of which has produced a confident wrong
number: never time a first call (it times the kernel cache, not the work); a memory figure
says what the cache held, and a cache miss is the *cheaper* one in memory; and a check that
a quantity did not move takes a median and says how many samples. The full section is
`.claude/skills/reference-timing/performance-rules.md`; the `reference-timing` skill is how
to take a pair.

## The traps that recur

Each has been hit in more than one phase, and most give a plausible wrong answer rather
than an error. The account of each, with its numbers, is
`.claude/skills/trap-check/traps.md`; the `trap-check` skill audits a diff against them.

- `abs` is not differentiable at zero, and symmetry often forces the zero.
- A clamp's tangent at the boundary is halved by JAX; write it as a `jnp.where`.
- Rule D4: a band diagonal is not invariant in a degenerate multiplet; take the block average.
- A caller-built k-set is a `for_spin` boundary.
- A response on a reduced k-set is a polar or axial vector field, symmetrised as one.
- A wedge is detected by `KPoints.reduced`, never by a spread in the weights.
- A wedge sum completes only for a quantity linear in a covariant per-k object.
- `np.asarray` on anything differentiated kills the gradient silently.
- The energy can be right while its derivative is wrong.
- A stencil must not contain its own centre.
- Neighbouring k-points do not share a G-sphere; align by Miller index.
- Index order in a transposed pair reads as a sign.
- An `nspin = 2` screening kernel is not finite where a channel density vanishes.
- Inherit a refusal only after checking which machine it belongs to.
- A check whose null result cannot be told from a pass; test that the guard fires.
- A quantity symmetry forces to zero is a residue; ask which components the cell allows.
- A sum of N terms matching one measured total is one equation in N unknowns; an A/B is evidence.
- A method that could not have returned a surprise proves nothing by finding none.
- A confirming number counts only if the alternative hypothesis would have read differently.
- A test asserting a tolerance its own `conv_thr` does not deliver; rerun at an older commit.
- The forecast sentence under a verified table is the one nobody checks.
- A time on a card is not a measurement until the Davidson step counts are equal.
- A loop over a closure, called outside any `jit`, compiles again at every call.

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
- **The spin channel is the leading axis, and what is squeezed on the way out is what a
  reader indexes per band or per energy, not a real-space field.** Densities, potentials
  and `becsum` are `(nspin, ...)` internally with no special case for one channel, **and
  they stay that way on the result objects**: measured on the canonical silicon cell at
  `nspin = 1`, `SCFResult.density` and `.potential` are `(1, 16, 16, 16)` with no
  `*_by_spin` property, while `SCFResult.eigenvalues` and `.occupations` are `(nk, nbnd)`
  and `DensityOfStates.dos` is `(nE)`, each with a `*_by_spin` beside it. This rule used
  to say the result objects drop the axis for everything, which is false for the two
  fields and is the statement `scf/spin_torque.py` was written against -- a consumer that
  reinstates the axis itself then gets a second one. A new consumer of
  `SCFResult.density` takes it as `(nspin_mag, ...)` always. `k` stays the leading *independent* axis inside each channel, which is
  what the batching and the eventual sharding rest on. `nspin` is static
  (`eqx.field(static=True)`) because it is an array rank, not a value.

## Where each subsystem lives in the reference source

The table mapping every subsystem to its QE 7.5 (or Elk) routines, with the porting note for
each, is `.claude/rules/reference-source-map.md`. It loads when a file under
`quantum_espresso/` is read; read it by hand before transcribing anything else. Fortran is
column-major and 1-indexed, and `PW/` works in Rydberg atomic units throughout.

## Mirror QE in the performance-critical path

Where performance matters, reproduce QE's implementation (its layout, loop structure and
order), not only its formulas, and benchmark on `benchmarks/si8-1k*.in` or `si16-1k*.in`,
never on a two-atom cell. The cases and the limits of the rule are
`.claude/rules/hot-path.md`, which loads with the Hamiltonian, solver, basis and SCF code.

## Memory is part of the design

A design is not finished until its peak working set is known in terms of `nk`, `nbnd`,
`npwx`, `npol` and the FFT grid, and a deviation from QE's memory-saving choices is stated,
measured and made selectable when large. A test file sweeping more than about three cells
takes `jax.clear_caches()` in an autouse fixture and `lru_cache(maxsize=2)`. The full section
is `.claude/rules/memory-design.md`, which loads with any file under `defumat/` or `tests/`.

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

The testing method is running the *same input* through real QE and through defumat and
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

Compiled kernels are cached in `~/.cache/defumat/jax` so that only the first run of a
process pays for them; `DEFUMAT_CACHE_DIR` moves it and `DEFUMAT_CACHE_DIR=off` disables
it.

```
tools/test-fast.sh                     # THE GATE: everything not marked slow, ~10 min
python3 -m pytest -m slow              # the other 1279, over two hours
tools/run_regression.sh                # the same slow set, one capped process per file
python3 -m pytest tests/unit/test_qeref.py::test_scf_silicon   # a single test
python3 -m defumat.cli inspect <qe-output>   # summarise what the parser reads
tools/export_notebooks.sh                     # re-execute notebooks + refresh .md exports
```

**How to run the suite, what each group costs, and how not to get killed is the
`test-runs` skill.** The short form, and it is the part that binds without reading
anything: the suite is two groups and `slow` is the line; `tools/test-fast.sh` is the
gate and runs before a push; the slow set is over two hours and is **not** a push gate;
and **anything long goes through `tools/run_regression.sh`, after a commit**, because
that is what caps each file's memory and makes a kill cost one file instead of the
session. The skill carries the measured sizes, the drift history, the rule for deciding
whether a new test belongs in the gate, the cgroup cap and the RSS watchdog.

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
- **How many k-points are in flight is `defumat/batching.py`'s dial, and its default
  follows `memory_mode`** — QE's loop, one k-point at a time as `c_bands.f90` and
  `sum_band.f90` do it, on a CPU; in `'memory'` (the accelerator default since
  2026-09-28, the user's decision) the largest chunk the card holds, `'fit'`, since
  2026-10-02, also the user's (184.8 against 31.4 ms an iteration on an H100's 27-point
  mesh); and the whole axis at once in `'speed'`. The band dial
  follows the platform — one band on a CPU, the whole block on an accelerator — in both
  modes, **except** that a `'memory'`-mode `Calculation` on an accelerator whose whole
  block would not fit chooses the largest batch that does, from the card
  (`resolve_band_batch_for`), and carries it on its Hamiltonians and density kernels. `k=all, b=1` is in neither preset, being measured worse than either end, and
  `k=1, b=all` is `'memory'`'s, measured at 1.2x `all/all` on a card.
  `k_batch`
  reaches every entry point (`run_scf`, `run_bands`, `run_nscf`, `run_dos`, `Calculation`,
  `DEFUMAT_K_BATCH`), `None` asks for one `vmap` over the whole axis, and the chunked form
  is a `lax.map`/`lax.scan` so it stays compiled once and differentiable. Anything new that
  walks the k axis goes through `map_k`/`sum_k` rather than calling `vmap` itself; the
  chunk size must never be visible in a result beyond round-off.
- Use `donate_argnums` for the large wavefunction and density buffers.

`PLAN.md` §1 and §5 hold the full reasoning and the rest of the GPU notes.
