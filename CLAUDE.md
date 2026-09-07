# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**This file is the rules. `PLAN.md` is the record.** Everything about *how* to work here —
conventions, refusals, where the reference source lives, what a finished phase looks like —
is below. Everything about *what was found* — per-phase validation numbers, the trap each
phase uncovered, what a refusal was measured at — is in `PLAN.md` §3, one section per phase.
Do not restate a phase's findings here; add them there.

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
have. Phases run **P0 through P70**; the ones still open, and the exact term each is
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

**What is in scope and implemented**, one line each. The binding rule is stated; the
validation numbers and the traps are in the named phase, and the user-facing refusals are
in `docs/features.tex`'s amber boxes.

- **Ultrasoft and PAW** (P12): the two-grid split, the augmentation charge, the overlap
  operator, self-consistent `D_ij`, PAW's one-centre terms.
- **Gradient-corrected functionals** (P13): PBE, revPBE, PBEsol, on the plane-wave grid
  and on the PAW spheres. The functional comes from the pseudopotentials' headers unless
  `input_dft` overrides it, and **an unimplemented one is refused rather than silently
  replaced by LDA**.
- **Collinear spin** (P9): `nspin = 2` gives the density, potential, `becsum`, `D_ij`,
  eigenvalues and wavefunctions a leading channel axis. The occupation scheme decides how
  many Fermi levels there are — one shared, or one per channel when `tot_magnetization`
  constrains it. `occupations = 'fixed'` implements only the second, which is `pw.x`'s own
  rule (`input.f90:784-800`) and is enforced at input here in QE's order. A fixed
  occupation **cutting a degenerate multiplet** is diagnosed by name for the residual
  solver: which member the eigensolver returns is arbitrary, so `F` is not a function of
  the density.
- **Spin-orbit coupling** (P14): `noncolin` makes a wavefunction a two-component spinor of
  length `2 npwx`, so there is *one* Hamiltonian on a space twice as large rather than two
  Hamiltonians; `lspinorb` puts a fully-relativistic dataset's `j`-resolved projectors into
  it. Forces, stress and relaxation for this regime are P46, and they take `dvan_so` (the
  **bare** `D`, for the same reason the collinear branch takes `dion`) and `qq_so`.
- **Noncollinear magnetism, magnetic fields and spin spirals** (P17-P19;
  `defumat/scf/fields.py`, `scf/locals.py`). `sym_rho`
  rotates the magnetization as an **axial** vector. Fields: the *energy* is written down
  and the potential is `jax.grad` of it, with QE's five `add_bfield.f90` expressions as a
  test; **the field's energy is not in the reported total**, by QE's and Elk's shared
  convention, and is carried separately. Spirals (`spiral_q`, Elk's `vqlss`) put the up
  component at `k + q/2` and the down at `k - q/2`, each on its own sphere — the whole of
  the generalized Bloch theorem. Refused for a spiral: spin-orbit coupling permanently,
  symmetry (until the spin space group is written, so `nosym` and the full grid), and
  ultrasoft/PAW.
- **Relaxing the spiral wavevector** (P21): `q` is a coordinate like an atomic position, so
  `dE/dq` is `jax.grad` at frozen wavefunctions and a frozen sphere (`forces/spiral.py`,
  `workflows/spiral.relax_spiral_q`), with the same BFGS
  handed the **reciprocal** cell as its metric. Only `|k ± q/2 + G|^2` and `vkb(k ± q/2)`
  carry `q`. Two traps: the compiled gradient closes over its sphere and must be dropped on
  every `at_spiral_q`, and BFGS's initial inverse Hessian is out by two orders on a
  milli-Rydberg magnetic surface (`BFGSSettings.hessian_scale`). A magnetic field is
  refused — its energy is outside the reported total, so the state is stationary for a
  different functional.
- **Forces, stress and both relaxations** (P15, P11, P29, P46). The default is `jax.grad`
  of the total energy at *frozen* wavefunctions, with the orthonormality constraint carried
  explicitly so ultrasoft's Pulay term falls out of the same gradient; QE's hand-derived
  expressions are transcribed **beside** them as the cross-check, never instead. A
  vc-relax is **two runs** — the relaxation in a frozen basis, then one more SCF from
  scratch — and the gap between their energies is the Pulay error, reported
  (`VCRelaxResult.pulay_error`) rather than left to be noticed.
- **Berry curvature, Chern numbers, Z2, and the Berry-phase polarization** (P16, P47, P56).
  Everything is built from one primitive, `<u_mk|S|u_nk'>`, because a determinant of
  overlaps is blind to the mixing a degenerate eigensolver leaves (rule D4) *and* is an
  exact integer on any mesh. Z2 has two independent routes and running both where both
  apply is the check; where they disagree the parity one is the answer.
- **DFT+U** (P20, P62): `lda_plus_u_kind = 0` and Elk's other flavours, from the `HUBBARD`
  card, whose parameters are **in eV** and are converted at the input boundary. The energy
  is written down and `v_ns` is `jax.grad` of it; `v_hubbard` is transcribed as a test, and
  `force_hub` is *not* transcribed at all — it is `jax.grad` through moving projectors.
- **Continuing one run from another** (P23, `defumat/scf/continuation.py` and
  `System.with_spin`): `run_scf(starting_from=result)` across a
  change of spin regime. A promotion is *decompose, decide what `m` should be, recompose*;
  the magnetization is **seeded** when the source has none, because nothing in the SCF
  breaks spin symmetry on its own. `with_spin` rebuilds the k-points rather than
  relabelling them.
- **Linear response** (P24 and its letters, P45): the velocity operator from one `jvp` of
  `H(k)`, the Sternheimer solve in place of a sum over states, and the dielectric constant,
  Born charges, phonons at `Gamma`, and the strain response on top. **The perturbations are
  gradients of code that already exists** rather than expressions derived a second time —
  `dv_of_drho` is one `jvp` of `v_of_rho`, `dvqpsi_us` one `jvp` through `at_positions`,
  `int3` one `jvp` of `newd`, `PAW_dpotential` one `jvp` of `onecenter`. Still refused:
  noncollinear magnetism, DFT+U, spin spirals, a potential-only meta-GGA, and — for
  `nspin = 2` — a **GGA** kernel (P70 covered the LDA; `dgcxc_spin` has its own thresholds
  and gates in a different routine) and the *assemblies* above the solve, which is the
  dynamical matrix, the strain response and the two third derivatives.
- **Third derivatives** (P26, P35, P36, P43): the Raman tensor, electrostriction, the
  elastic and elasto-optic constants, from one `jvp` of the second-order energy at frozen
  first-order wavefunctions — the 2n+1 theorem, which is P15's and P25's envelope argument
  one order up. `symtensor3`/`symmatrix3` are implemented at **any** rank.
- **Meta-GGA, potential-only** (P30-P32): `tb09` and `bj06` are potentials with no energy
  functional, so they invert the rule above — nothing is differentiated, the expression
  *is* `v_x`. The consequences are enforced rather than documented: `run_scf` warns that
  its total is not the value of anything it minimised, and every consumer of
  `forces/energy.py:energy_at` refuses. `tau` comes from the states and is **not mixed**,
  exactly as `mix_rho.f90` leaves `kin_r` alone. Energy-carrying meta-GGAs (TPSS, SCAN,
  M06L) are **not** in — their potential has a `dE/dtau` piece acting on the wavefunction.
- **Van der Waals** (P27): Grimme's **D2** only, as a pair sum over the nuclei outside
  `v_of_rho`. The other four are **refused by name**, where QE's `set_vdw_corr` warns and
  silently runs with no correction at all — D3's `C6` has a coordination derivative of its
  own, and TS/MBD/XDM are functionals of the self-consistent density.
- **Optical spectra with excitons** (P37): the one place a **sum over states** earns its
  keep, because a spectrum needs `chi_0` as a matrix over `G` at every frequency where the
  Sternheimer stack gives a static operator. The Dyson equation is solved with a kernel
  from a registry; the bootstrap one is parameter-free.
- **The magnetism tail** (P48, P51-P66): site-resolved `<L>`/`<S>`/`<J>`, effective masses,
  the piezoelectric tensor, optical conductivity and the Kerr angle, Fermi-surface nesting,
  the shift current, second-harmonic generation, LO-TO splitting, magnetocrystalline
  anisotropy and the magnetic torque, structure factors, magnons, the orbital
  magnetization, STM images and vertical tunnelling transport.
- **Running a calculation too large for one job** (P67): `defumat/sizing.py` budgets a
  run's peak before it starts, the SCF checkpoints, and a dynamical matrix can be built
  one atom's column at a time across jobs.
- **The `j`-resolved projected density of states** (P69): `projwfc.x` for a spinor band,
  resolved by `j` and `m_j` instead of by `m` — the regime every heavy-element run here
  advertises and had no orbital decomposition for.
- **The screened response and Born charges of a magnetic insulator** (P70): `nspin = 2`
  above `chi_0`. The LSDA kernel is **defined** to be zero at `|zeta| >= 1` (QE's
  `dmxc_lsda` does it on both branches), so what was refused as an analysis was a
  convention — and masking the *argument the derivative is taken at* is what works, where
  clipping the density leaves the primal singular and the tangent `0 * inf`.

**Gamma-only storage** (P68) is a **memory** feature rather than a speed one:
`K_POINTS gamma` stores one plane wave of each `(G, -G)` pair, which halves `npwx` and
every array a band lives in — 96 GB against 189 on a 157-atom slab. **Only the
wavefunction sphere halves and the dense G set stays whole**, which is a deliberate
departure from `pw.x`: the memory is entirely in the plane-wave-sized arrays, and halving
the dense set would put a conjugate fill inside every consumer of a real field. The
consequence to know is that `ngm` here is **not** the `ngm` `pw.x` prints —
`ngm_full = 2 ngm_QE - 1`, asserted in `tests/regression/test_basis.py`. Three things
carry the trick and nothing else does: `g_to_r_gamma` rebuilds the field from both halves;
every plane-wave sum becomes `2 Re(sum) - (the G = 0 term)`, since `G = 0` is its own
conjugate partner (`gamma_inner`, `calbec_gamma`); and `Im c(0)` must stay zero
(`force_real_g0`, where `regterg.f90:174` and `:375` impose it). **Substituted rather than
refused**, with a warning, for ultrasoft/PAW, for a run that uses symmetry, and for a
spinor or spiral run — the same physics at twice the storage, and it says so.

**Out of scope until the above works:** EXX, real-time propagation and the
Liouville-Lanczos route to a spectrum (`TDDFPT/`), Car-Parrinello (`CPV/`), and everything
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

`defumat/calculator.py` (P38). A `Calculator` is a `System` together with its
pseudopotentials, and every workflow, force, stress, response and invariant is a method
on it — `Calculator.from_file("scf.in")`, then `get_scf()`, `get_bands()`,
`get_dielectric_tensor()`. It is what the README, the user guide and new notebooks use,
and `from defumat import Calculator` is the one import a script needs.

**It is a facade and nothing else.** No physics lives there: every method is a one-line
delegation to the functional entry point, which is unchanged and still the way anything
managing its own state is driven. A `get_*` that grew a computation of its own would be
a second implementation of something already validated against QE, and there is a test
asserting that none has.

Three things about it bind anything added to it:

- **State cannot move onto `System`.** `System` is an `eqx.Module` crossing `jit`/`grad`,
  so a `pseudos` field would change the pytree every compiled path sees and a cached
  result cannot live on a frozen module at all — and `System` does not *have* the
  pseudopotentials, only the file names. `System.calculator()` is a constructor, not a
  place to hang calculations.
- **Nothing mutates.** `with_positions`/`with_cell`/`with_spin` return a *new* calculator
  with an empty cache; the converged state crosses as a `starting_from` seed
  (`starting_state`), never as an answer. A cached result under a moved atom is the
  `test_geometry_invalidation` defect one layer up.
- **The refusals pass through untouched**, and the implicit SCF announces itself. The
  cache is one slot keyed by the options that filled it, because it holds the
  wavefunctions.

A new feature adds a `get_*` method in the same pass that adds its entry point. Shared
options go in `SHARED_OPTIONS` and are forwarded **by named parameter only** — a
`**kwargs` in a signature is not permission to pass everything, since several response
entry points forward theirs to solvers that would raise on `nbnd`.

## What a finished phase leaves behind

A phase is not done when the code runs. It is done when all five of these exist, and each
of the five has gone stale silently at least once, so each is checked rather than assumed.

**1. A number.** A concrete figure against `pw.x`, against Elk, or against an identity
that shares no machinery with the assembly — and, where a term is missing, a *measurement*
of what it is worth and a refusal by name. `PLAN.md` §3 is where it goes.

**2. A row in the README's feature table.** `README.md` carries every implemented feature,
with the input variable or entry point that reaches it and **two tick columns, `QE` and
`Elk`, saying whether either established code computes that quantity at all**. `(✓)` with
a numbered note means partly; **blank in both is the mark of a quantity neither code has**,
which is what tells a reader whether this is a reimplementation or an extension.

- **The table names quantities, not routines**, and **the rows are physics, not knobs.** A
  row is a thing someone would want to compute; smearing belongs inside the row about
  metals, not in a row of its own. A new feature adds *one* row; its variants, schemes and
  internal terms go in `PLAN.md`.
- **Each tick is a claim about someone else's source.** Before ticking `QE`, find the
  routine in the vendored tree; before ticking `Elk`, find it in the task list of
  `docs/elk_manual.txt` (§5.127) or in `vendor/elk/src/` in the user's `elkpy` checkout;
  before leaving both blank, grep both. Elk is the easier of the two to get wrong — its
  `z2*.f90` files are complex-matrix helpers with nothing to do with the Z2 invariant.
- **The entry-point column is a claim about *this* code.** `grep` the variable in
  `io/pwin.py` and `system/builder.py` and check the function is exported. A row naming
  something nothing parses is worse than no row.

**3. An entry in the user guide.** `docs/features.tex`, built with `xelatex docs/features.tex`
(twice, for the table of contents); there is no markdown copy and none should be added,
because two copies drift. An entry is four things and the last two get skipped:

- **what it computes**, as an equation where there is one — this is a physics document,
  not an API listing;
- **the entry point**, checked by `grep` rather than remembered;
- **a snippet that has been run.** Checking a name exists is not enough: an audit that only
  checked `dir()` passed six broken snippets. Execute it;
- **what it refuses**, in the amber box. The refusals are the promise that a run which
  starts is a run whose physics is there, and that promise is only usable if its edges are
  written down.

**Do not re-document standard `pw.x` variables** — `ecutwfc`, `ibrav`, `nbnd` mean what
they mean in QE and the guide says so once. Document this code's own knobs (`mbj_c`,
`spiral_q`, `LOCAL_MAGNETIC_FIELDS`) and the ones that gate a feature. **The audit that
catches drift is a set difference, not a read-through**: list the workflow, response, force
and stress entry points the package exports and check each appears in the `.tex`. It found
ten missing at once, including a whole implemented feature with no mention.

**4. A notebook.** See the section below.

**5. A timing against the code it was taken from.** See "Performance".

## Tutorial notebooks

`notebooks/` holds worked examples on concrete systems — the readable counterpart to the
test suite. **Every new feature adds a notebook or extends an existing one.** Demonstrate
on the two-atom silicon cell from `test-suite/pw_scf/scf.in` wherever possible, compare
against the committed QE benchmark whenever the reference contains the quantity, and commit
the notebook executed so it reads without being run. `notebooks/README.md` carries the
cell-by-cell shape, the per-notebook timings and the index — which is **by the property a
reader wants to compute**, not by the order the code gained it. Conventions are enforced by
`tests/unit/test_notebook_conventions.py`, whose `REWRITTEN` set **only ever grows**, so a
new notebook joins it in the commit that adds it. The three that are *not* in it — `01`,
`03` and `17` — are the under-the-hood tier by design, whose internals are their subject;
they are exempt rather than unfinished.

**A notebook is about the physics, and nothing else.** What the quantity is, the equation
that defines it, what the number means, how it compares with experiment or with Quantum
ESPRESSO. **The implementation is not the subject and must not appear**: no `PLAN.md` phase
numbers, no QE Fortran file names, no transcribed-versus-differentiated tables, no `jvp`,
tangents, frozen spheres, padding or compilation, no catalogue of traps, and no account of
how something was developed or debugged. Two things survive from that side because they are
claims about capability rather than about code: one sentence saying a derivative is taken of
the energy itself rather than derived by hand, and one sentence where a reference is unusual
and the reader would otherwise not trust the comparison. **No em dashes** anywhere in a
notebook.

**That rule binds the code cells, not only the prose.** Bounding only the prose is why the
notebooks drifted into validation reports: the project's validation instinct moved into the
code, where the rule did not reach. An identity check looped over four pseudopotentials, a
derivative checked against a closed form on a random matrix, a hand-built linear solve with
a probe potential — each is the test suite's job being done in public. They belong in
`tests/`, and the notebook's footer names the file they went to. **Where a `get_*` method
exists, the notebook uses it** rather than building the same quantity from internals.

**A notebook is five minutes long, and it has a figure.** Header saying what this computes
and the headline number against QE; the shortest code that runs it; **one plot that shows
the physics**, a band structure wherever the feature shows in bands; one comparison table;
at most one "how it works" cell for the single best idea, and it is a *physical* idea. About
eight code cells. Each notebook also has a `.md` export committed beside it — raw `.ipynb`
is unreadable in a plain editor or a diff — regenerated together with the notebook by
`tools/export_notebooks.sh`.

**Ten minutes is the hard ceiling on executing one**, and it is a ceiling rather than a
target. A notebook is re-executed every time the code under it changes, so its runtime is
paid over and over by people who are not doing physics at the time. Time it before
committing it:

```bash
time jupyter nbconvert --to notebook --execute --inplace notebooks/<n>.ipynb
```

`tools/export_notebooks.sh` times each one as it re-executes it and exits non-zero over the
ceiling, so the set stays measured without anyone remembering to measure it. **If one does
not fit, the cell to cut is the *sweep*, not the physics**: measure the expensive series
once offline and quote its numbers in prose. A figure that needs ten SCF runs to draw is a
figure whose points belong in a test — which is what the one notebook that was ever over
the ceiling turned out to be doing (`notebooks/README.md` has the case study).

## Performance

**The measurement is single-core defumat against single-core Quantum ESPRESSO on the
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
`performance/run_performance.py` runs the comparison over a whole set of inputs and
typesets it (`performance/README.md`). `PERFORMANCE.md` is the running log: the comparison,
where the time goes, what each change was worth, and the backlog. **Add a measurement to it
whenever a feature lands or a hot spot moves** — including the QE ratio, not only an
internal timing. `tools/benchmark.py <input>` gives the component breakdown.

**Every feature taken from Quantum ESPRESSO or from Elk is timed against the code it
was taken from, and the pair goes in `PERFORMANCE.md`.** Not the ratio to a previous
version of this code, not an absolute number on its own: the reference implementation's
wall clock beside ours, on the same machine and the same physics, one core each
(`OMP_NUM_THREADS=1` for both; the affinity mask set before JAX is imported, the mechanism
`tools/compare_qe.py` documents). The reason is that **the absolute number is the one worth
having and it is the one nobody measures** — an internal timing says a feature costs 94 s
without saying whether that is what the physics costs or what this implementation costs,
and only the other code answers that.

Two things to state rather than discover, because a comparison against an all-electron
code is never like-for-like and a misleading ratio is worse than no ratio:

- **Say what is not comparable.** LAPW's basis is not a plane-wave sphere; write that
  beside the number.
- **Time the same work, not the same task number.** Codes split a calculation into
  post-processing steps differently — Elk's `dielectric` (task 121) reads momentum matrix
  elements off a file that `writepmat` (task 120) produced, so timing 121 alone against a
  `defumat` call that *builds* `dH/dk` compares a contraction with a contraction plus its
  operator. Add the steps up until both sides start from the same place, usually a
  converged ground state, and say which steps were added.

## The traps that recur

Every one of these has been hit in more than one phase, and most of them produce a
plausible wrong answer rather than an error. `PLAN.md` has the phase that found each.

- **`abs` is not differentiable at zero, and the zero is often forced.** `|psi|^2` must be
  `Re(conj(psi) psi)`; `abs(rho)**2` in the reciprocal Ewald sum is `0/0` wherever a
  structure factor vanishes *exactly*, which symmetry arranges on a supercell; `|m|` in the
  gradient correction differentiates through its own nodes. Five sites so far (P24, P28a,
  P45, P58).
- **Rule D4: a diagonal is not invariant under the rotation a degenerate eigensolver is
  free in.** Anything built from `<psi_n|A|psi_n>` band by band — a Drude weight, a band
  velocity difference, an incoherent channel sum — takes the **multiplet block average**
  instead. Worth four orders of magnitude on silicon's `chi^(2)` and 69x against 26x on a
  bilayer's transport contrast, and **no symmetry check sees it** (P51, P54, P66).
- **A caller-built k-set is a `for_spin` boundary.** Every `KPoints` constructor applies
  the unpolarized `degspin` unconditionally, and a spinor band holds one electron. The
  wrong numbers are the plausible ones — a plasma frequency of 13.11 eV instead of 0.60
  (P51), a factor of four in a nesting function (P52).
- **A response on a reduced k-set is a polar (or axial) vector field and must be
  symmetrised as one**, and the obvious escape does not work: a **shifted** Monkhorst-Pack
  grid is *not* closed under the point group, so running the whole grid instead of the
  wedge is unsound there and is refused by name (P24).
- **A wedge sum completes only for a quantity *linear* in a covariant per-k object.**
  Where a functional is quadratic in one, the *value* inside it must be the full-zone
  object while its *derivative* stays the raw wedge sum. Getting it wrong is worth 2.5%,
  is worse than doing nothing, and only the sum rule catches it (P36).
- **`np.asarray` on anything differentiated kills the gradient silently** — a term that
  should be there simply vanishes (P43).
- **The energy can be right while its derivative is wrong**, which is the whole supercell
  family (P28a) and is also how a dropped gamma-storage `G = 0` term shows: the total was
  right to 3e-12 Ry and the force wrong by 0.4 Ry/bohr on a force of 0.06, because being
  stationary hides an error in the gradient (P68). The identities that are sums over
  *atoms* — the acoustic sum rule, the rigid-translation test — are blind to a transfer
  between atoms; the first check that is not an atom-sum is a per-mode response density
  against a finite difference.
- **A stencil must not contain its own centre.** The plane-wave sphere is rebuilt at every
  `k`, and a high-symmetry point is exactly where a shell sits on the cutoff — `Gamma`
  holds fewer plane waves than every displaced point, so its eigenvalue is variationally
  **high** against theirs and a second difference inherits an error that *grows* as the
  stencil shrinks (P48).
- **Neighbouring k-points do not share a G-sphere.** Coefficients are aligned by Miller
  index, and the wrap at the zone edge is a *shift* of that index — without which a Chern
  number comes out smooth and non-integer (P16).
- **Index order in a transposed pair reads as a sign.** `f(n, m)` against `e(m, n)`, and
  `G(r, r')` conjugating `psi` in the *exit* variable rather than the source one: both give
  results that are real, non-negative, correctly symmetric and wrong (P54, P66).
- **An `nspin = 2` screening kernel is not finite where a channel density reaches zero**,
  which a cell with vacuum guarantees. That is the `abs` trap one derivative further out,
  it lives in `defumat/xc`, and clipping inside the response does not fix it (P45).
- **Inherit a refusal only after checking which machine it belongs to.** P35's refusal is a
  statement about the Sternheimer stack and never applied to a sum over states; taking it
  as read left a whole quantity marked impossible (P54).

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
| Starting wavefunctions | `PW/src/wfcinit.f90`, `Modules/atomic_wfc_mod.f90`, `upflib/atwfc_mod.f90` (`n_atom_wfc`) | the projectors' expression with `chi` for `beta` — but the phase is `i^l`, not `(-i)^l`. **The noncollinear `n_atom_wfc` is `sum (2j+1)` for a relativistic dataset and `sum 2(2l+1)` otherwise, and that count decides whether the random top-up happens at all** — building 22 where `pw.x` builds 12 filled `nbnd = 18` with atomic vectors and removed the six random ones, which is the only part of the span with generic angular character. A state no atomic orbital carries is then unreachable, exactly at a nonzero TRIM where the overlap is zero rather than small (`GAPS.md` §2c) |
| Meta-GGA (potential-only) | no QE counterpart to transcribe — `XClib/dft_setting_routines.f90` maps `tb09` to libxc 208; `PW/src/sum_band.f90` (the `kin_r` branch and its `sym_rho`), `PW/src/v_of_rho.f90` (`v_xc_meta`), `PW/src/potinit.f90` (the Thomas-Fermi `tau` guess), `PW/src/setup.f90` (what it refuses) | the functional itself follows libxc's own definition (`maple/mgga_vxc/mgga_x_tb09.mpl`, `maple/mgga_exc/mgga_x_br89.mpl`, `src/mgga_x_br89.c`), because QE has no native implementation. **QE passes a zero Laplacian and never sets `c`**, so its `tb09` is BJ06; both are here separately. `tau` is symmetrised — `sum_band` does it too, and skipping it is worth 0.47 eV in the eigenvalues |
| Van der Waals dispersion | `Modules/mm_dispersion.f90` (`energy_london`, `force_london`, `stres_london`), `Modules/set_vdw_corr.f90`, `Modules/rgen.f90`, `upflib/atomic_number.f90` | the energy is written down (`defumat/vdw/grimme.py`) and the force and stress are `jax.grad` of it; QE's two expressions are transcribed as the cross-check. `rgen`'s **fold** of the pair separation into the cell is kept, and it is what lets one neighbour list serve every geometry |
| Ewald / local potential | `PW/src/ewald.f90`, `setlocal.f90` | the ion-ion sum and `V_loc(G)`; the Ewald neighbour list is fixed for the *cell*, not the geometry, so it survives a relaxation |
| Forces | `PW/src/forces.f90`, `force_lc.f90`, `force_cc.f90`, `force_ew.f90`, `force_us.f90`, `addusforce.f90`, `force_corr.f90`, `symme.f90` (`symvector`) | the default is `jax.grad` of the energy at frozen wavefunctions (`forces/energy.py`); the Fortran expressions are transcribed as a cross-check. `gradcorr` is called from **inside** `v_xc`, so `force_cc` needs it |
| Structural relaxation | `Modules/bfgs_module.f90`, `PW/src/move_ions.f90`, `run_pwscf.f90`, `update_pot.f90`, `checkallsym.f90` | BFGS in crystal coordinates with the cell metric; the setup (FFT grid, symmetry, k-points) is done **once** and only checked afterwards |
| Stress | `PW/src/stress.f90`, `stres_knl.f90`, `stres_har.f90`, `stres_loc.f90`, `stres_cc.f90`, `stres_gradcorr.f90`, `stres_ewa.f90`, `symme.f90` (`symmatrix`) | the default is `jax.grad` of the energy with respect to a strain at frozen wavefunctions (`stress/energy.py`, through `Calculation.at_strain`); the Fortran expressions are transcribed as a cross-check. `stres_us`/`addusstress` are **not** transcribed, so the analytic route offers terms and no total. `ylmr2`'s `atan2` parameterisation is singular on the `z` axis and only a *cell* derivative reaches it |
| Magnetic symmetry | `PW/src/symm_base.f90` (`sgam_at_mag`), `symme.f90` (`sym_rho`'s `nspin = 4` branch), `PW/src/irrek.f90` | the magnetization is an **axial** vector, so its rotation carries `det(R)` and a further sign for an operation that is a symmetry only with time reversal; `irreducible_BZ` completes an explicit k-list from the lattice's wedge to the crystal's, and runs for every SCF |
| Fields and constraints | `PW/src/add_bfield.f90`, `make_pointlists.f90`, `get_locals.f90`, `report_mag.f90`, `PW/src/input.f90` (`i_cons`) | the penalty's *energy* is written here and its potential comes from `jax.grad`; QE's five expressions are transcribed as the cross-check. Elk's counterparts: manual §5.2/§5.12/§5.104, `src/bfieldfsm.f90` |
| Spin spirals | no QE counterpart — Elk's `src/gengkqvec.f90`, `init0.f90`, `findsymlat.f90`, manual §5.146 | up at `k + q/2`, down at `k - q/2`, each with its own `G+k` set; one basis call on the concatenated list gives both a common `npwx` |
| Spiral relaxation | no QE counterpart — `Modules/bfgs_module.f90` reused with the *reciprocal* cell as its lattice | `dE/dq` is `jax.grad` of the energy at frozen wavefunctions and a frozen sphere (`forces/spiral.py`); only the kinetic and nonlocal terms carry `q` |
| Piezoelectric tensor | no QE counterpart — Elk's `src/piezoelt.f90` and `genstrain.f90`, manual task 380 | Elk runs one ground state per strain and finite-differences the Berry-phase polarization; here it is one `jvp` of the stress along the field's response (`defumat/response/piezo.py`), so nothing is transcribed but the *check* — `zstar_eu.f90`'s contraction with a strain label where it has a displacement. `genstrain` symmetrises each candidate strain over the crystal's group, so on a cubic crystal the only strain it keeps is the isotropic one |
| Second-harmonic generation | no QE counterpart — Elk's `src/nonlinopt.f90` and `getpmat.f90`, manual task 125 | `chi^(2)(-2w; w, w)` by a sum over states, so the assembly *is* transcribed, with one substitution: Elk reads momentum matrix elements and this uses `response/velocity.py`'s `dH/dk`, for the reason the TDDFT row already gives. Two things Elk's loop does not need and a plane-wave code does: the multiplet **block average** of the velocity diagonal that `Delta^a` is built from (rule D4 — Elk's 42x42x42 shifted mesh misses the symmetry points where it bites), and the reminder that Elk's `swidth` is in **Hartree**. `el_opt.f90` is QE's nearest thing and is the *static* electro-optic tensor, on the branch P35 found broken |
| X-ray and magnetic structure factors | no QE counterpart — Elk's `src/sfacrho.f90`, `sfacmag.f90`, `genhvec.f90`, `zftrf.f90`, manual tasks 195/196 | `zftrf` is `(1/Omega) int f e^{-iH.r}` and `sfacrho` prints `Omega` times its conjugate, which is the crystallographic convention; here the positive-phase transform is taken directly (`defumat/diffraction/`). `genhvec` reduces the H-set with the **symmorphic, non-magnetic** operations only, and that restriction is about the *phase* of `F` rather than its modulus |
| Orbital magnetization | `PW/src/orbm_kubo.f90` (reached by `lorbm`), `PW/src/kpoint_grid.f90` (`kpoint_grid_efield`), `PW/src/setup.f90` (what it refuses) | the assembly is transcribed and the mesh with it -- the dual states are `zgefa`/`zgedi` on the neighbour overlap, which is the covariant derivative. Two conventions are QE's and are documented rather than inherited silently: `ef` is imported and never used, so what is printed is `M(mu = 0)`, and the two printed terms are not the papers' LC/IC split. The vector direction is `b_l` while the derivatives are along the other two crystal directions, which is exact for any lattice (`a_i x a_j = Omega b_l/(2 pi)`) |
| Fermi-surface nesting | no QE counterpart — Elk's `src/nesting.f90`, manual task 105 | Elk writes an `O(N_q N_k)` double loop with `mod(ivk + ivq, ngridk)`; that fold makes the sum a cyclic cross-correlation, so `ifftn(|fftn(g)|^2)` replaces it (`defumat/response/nesting.py`) and the loop is kept as `method = "direct"`. The wedge is unfolded with `grid_equivalence` — `tetra.f90`'s `equiv`, Elk's `ivkik` — and the group it is unfolded with must be the group `denser_grid` reduced it with (`workflows/nscf.py:grid_symmetry`) |
| Vertical tunnelling transport | no `pw.x` or Elk counterpart for the quantity — QE's `PWCOND/src/` (`transmit.f90`, `compbs.f90`) is a Landauer transmission of a *different geometry*: two semi-infinite crystalline leads, one conductance per energy, no point contact and so no map | nothing is transcribed. The exit plane's Gram matrix is a closed-form Miller-index orthogonality (`transport/substrate.py`), the tip amplitudes are P65's sampler made complex and per-k (`basis/sample.py`), and the contraction is one quadratic form. The index order is the one trap: `G(r,r')` conjugates `psi` in the **exit** variable |
| Berry phase / topology | `PW/src/bp_c_phase.f90` (the ultrasoft `q_ij(b)` and the k-string overlaps), `Modules/bfgs`-free | the invariants themselves have no QE counterpart to transcribe — `defumat/topology/` follows Fukui-Hatsugai-Suzuki, Yu-Qi-Bernevig-Fang-Dai and Fu-Kane, with `bp_c_phase.f90` as the reference for how the augmentation charge enters an overlap between two different k-points |
| Velocity / position operator | `PW/src/commutator_Hx_psi.f90`, `PP/src/` Berry-phase code | QE hand-codes `[H,r]` term by term; here it is one `jvp` of `H(k)` at a frozen sphere (`response/velocity.py`), since `dH/dk_a = i[H, r_a]` in the periodic gauge. The overlap carries a velocity too, so a band velocity is `<psi|dH/dk - eps dS/dk|psi>` |
| Linear response / DFPT | `LR_Modules/cgsolve_all.f90`, `ch_psi_all.f90`, `orthogonalize.f90`, `h_prec.f90`, `setup_alpha_pv.f90`, `incdrhoscf.f90`, `symdvscf.f90`; `PHonon/PH/solve_e.f90`, `dvpsi_e.f90`, `dvqpsi_us.f90`, `dielec.f90`, `zstar_eu.f90` | the linear solve, the projector and the assembly are transcribed; the *perturbations* are not. `dv_of_drho` is one `jvp` of `v_of_rho` (which already drops the `G = 0` Hartree term), the E-field's commutator is the velocity operator, and `dvqpsi_us` is one `jvp` through `at_positions`. **A response on a reduced k-set is a polar vector field and must be symmetrised as one** |
| TDDFT: `chi_0`, the Dyson equation, the bootstrap kernel | no QE counterpart — Elk's `src/tddftlr.f90` (the driver and the fixed point), `genvchi0.f90` (Adler-Wiser, the `t3hw` head/wing layout), `genvfxc.f90` (the kernels), `init3.f90` (`ngrf`, and `wrf(1) = 0 + i swidth`), `getpmat.f90` (the scissors renormalisation), manual `fxctype`/`gmaxrf`/`swidth` | the head is the one line **not** to transcribe: Elk reads momentum matrix elements, which is right in LAPW and wrong with a nonlocal pseudopotential, so `response/velocity.py`'s `dH/dk` takes their place. `eps_M` is the inverse of the **3x3 head** of `eps^-1`, not the head of the inverse — Elk writes both, thirty lines apart, and the wrong one is 9% too large and otherwise perfect |
| Non-linear response (Raman) | `PHonon/PH/raman.f90`, `raman_mat.f90`, `el_opt.f90`, `dhdrhopsi.f90`, `dvpsi_e2.f90`, `solve_e2.f90`, `d2mxc.f90`, `write_ramtns.f90`, `symme.f90` (`symtensor3`, `symmatrix3`) | none of it is transcribed: `d(eps)/d(tau)` is one `jvp` of the second-order energy P26 already differentiates, and `d2mxc`'s third derivative of `E_xc` is a `jvp` of the kernel rather than a parameterisation, so a GGA works where `phq_setup.f90` stops. **The vendored 7.5 build's `lraman`/`elop` branch does not reproduce QE's own v6.0 example and fails its own internal check** -- use it as evidence, not as a reference. `dynmat_sub.f90`'s `RamanIR` (reached by `dynmat.x`) is the exception and *is* a reference: it is post-processing, reads `dchi_dtau` off a file, and shares nothing with that branch. `symtensor3`/`symmatrix3` are implemented (P36), at any rank |
| Input parsing | `Modules/read_input.f90`, `PW/src/input.f90`, `Modules/input_parameters.f90` | defaults for every input variable are declared in `input_parameters.f90` |
| DFT+U | `PW/src/ldaU.f90`, `hubbard.f90`, `new_ns.f90`, `init_ns.f90`, `ns_adj.f90`, `orthoUwfc.f90`, `offset_atom_wfc.f90`, `vhpsi.f90`, `v_of_rho.f90` (`v_hubbard`), `scf_mod.f90` (`ns_ddot`), `force_hub.f90` | the projectors are `S phi` even for `Hubbard_projectors = 'atomic'`; `ortho-atomic` orthogonalises over **all** `natomwfc`, not the Hubbard manifold alone, so `Modules/read_pseudo.f90`'s `upf_check_atwfc_norm` renormalisation of `chi` reaches the answer through the `4s`. `force_hub.f90` is *not* transcribed: it is `jax.grad` through `Calculation.at_positions` |
| Occupations / smearing | `PW/src/gweights.f90`, `Modules/wgauss.f90`, `Modules/w0gauss.f90`, `PW/src/set_occupations.f90` | |
| NSCF / band structure | `PW/src/non_scf.f90`, `PP/src/bands.f90`, `PP/src/plotband.f90` | fixed density, diagonalize once per k on an explicit path |
| DOS | `PW/src/tetra.f90`, `PP/src/dos.f90` | `tetra.f90` has both the linear and the Bloechl-corrected tetrahedron method |
| Projected DOS | `PP/src/projwfc.f90` (`projwave`, `sym_proj_k`, `print_lowdin`), `PP/src/projections_mod.f90` (`fill_nlmchi`), `PP/src/partialdos.f90`, `PW/src/tetra.f90` (`opt_tetra_partialdos`) | the projectors are `orthoUwfc`'s, so `hubbard/projectors.py` builds them for both; the weighted integration goes through the *same* DOS registry, and `do_projwfc` silently runs the **linear** tetrahedron method whatever the SCF used |

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

## Memory is part of the design

**A design is not finished until its peak working set is known.** A plane-wave code is
memory-bound as often as it is compute-bound, and what decides whether a calculation runs
at all is usually a working set rather than a flop count. Before landing anything that
allocates per k-point, per band, or per G-vector, say what the peak costs in terms of
`nk`, `nbnd`, `npwx`, `npol` and the FFT grid, and put that number against the RAM of a
real machine — the same reflex the performance rule above asks for with time.

**Where QE spends effort to save memory, copy it unless something better is on offer.**
None of these is incidental to the algorithm:

- **One k-point at a time.** `c_bands.f90`'s `k_loop` diagonalises a single k-point and
  `sum_band.f90` accumulates the density inside the same loop, so QE's working set is one
  k-point's whatever `nks` is, and the parallelism over k comes from MPI pools. Batching
  the whole k axis with `vmap` is this code's deliberate deviation — it is what a GPU wants
  — so it is a **dial** (`defumat/batching.py`), defaulting to QE's end of it on a CPU and
  to the batch on an accelerator. Rule R6 (k leading) is what keeps both available.
- **The sphere, not the box.** Wavefunctions live on the G-vectors inside the cutoff and
  are expanded into the FFT box only for the transform, and only over the sticks the
  sphere occupies (`basis/sticks.py`).
- **Two grids.** The smooth grid carries the wavefunctions and the dense one only the
  augmentation charge that needs it, which is most of the point of `ecutrho`.

A deviation is allowed and is sometimes right — this code trades memory for batching the
way QE trades it for MPI ranks, and `becsum` is carried as a full symmetric matrix where
QE packs the upper triangle. The rule is that such a trade is **stated, measured, and made
selectable when it is large**, never arrived at by accident: name it in the module
docstring, and put the number in `PERFORMANCE.md` beside the timing.

**A test file that sweeps many cells is a memory liability, and splitting the file is not
the fix.** The mechanism is accumulation, not any one peak: cells that share no shape each
compile the whole SCF (and, for a derivative, the gradient) stack afresh and **XLA keeps
every executable for the life of the process**, while an unbounded `lru_cache` of converged
states holds their wavefunctions beside it. Both grow monotonically through the file, and
two files have been killed on this machine that way. Two bounds, and they belong on any
file that runs more than about three distinct cells:

- **`jax.clear_caches()` in an autouse fixture**, after the `yield`. The results stay
  cached; only the compiled code is dropped, which trades recompilation for a peak the
  machine can afford. It gets *faster* as well as smaller, which is the tell that the
  process was paging rather than recompiling (`PERFORMANCE.md`, P28b).
- **`lru_cache(maxsize=2)` on the converged-state helper**, never `maxsize=None` — 2 is
  what a comparison between two cells needs and is the largest that is not a leak.

**Splitting the file only pays off under one of the three runners**, which is why it is the
weaker lever: `tools/run_regression.sh` invokes pytest once per file, so a file boundary
there *is* a process boundary; `tools/test-fast.sh` and a plain `pytest -m slow` run
everything in **one** process, where splitting changes nothing at all.

**Do not run demanding suites simultaneously — not in one process, and not in two at
once.** This machine has 30 GB and both mistakes have killed a session here:

- **Several slow files in one `pytest` invocation** is *one* process, so every file's XLA
  executables accumulate for the whole run — three spinor suites reached 2.4 GB in ninety
  seconds and kept climbing. Run them one at a time, through `tools/run_regression.sh`
  rather than a hand-written loop — it caps each file's memory as well as separating them,
  which is the next subsection.
- **Two test runs in parallel, or a test run beside anything being measured.** A timing
  taken next to a test run is not a timing — a `projwfc.x` comparison measured beside a
  background suite read 70% slow and had to be discarded and repeated.

One habit makes this cheap: write a **durable summary line per file** so a kill costs the
file in flight rather than the whole run, which is what `run_regression.sh` already does.
And **narrow the list before running it**: a `grep` for the inputs that can actually reach
the changed code path is minutes of work and routinely removes most of the suites, where
guessing adds them.

### An out-of-memory kill must cost one file, and it still costs the session

**This is an open defect rather than a fact of the machine, and it is the one thing in this
section that is not yet fixed.** Everything above is a way of staying under the ceiling;
none of it puts a floor under what happens when something goes over it anyway, and going
over it here kills whatever else the terminal was holding. It has happened at least three
times: `test_ten_site.py` (P28b) and `test_spinor_forces.py` (P46), both measured in
`PERFORMANCE.md`, and again on **2026-09-07** — that one by the user's account rather than
from a log — where it took a session down with a documentation restructuring uncommitted.
**What was in flight the third time is not recorded anywhere, and that is itself the
point** — an unbounded process that dies takes its account of what it was doing with it.

**The mechanism is a cgroup, and it works on this machine** (cgroup v2, user-slice
delegation, probed 2026-09-07). Name the unit rather than quieting it, so
`journalctl --user -u <unit>` can afterwards say the kill was `memory.oom` and not
something else:

```bash
systemd-run --user --unit=reg-<file> -p MemoryMax=8G -p MemorySwapMax=0 --scope \
    python3 -m pytest <file> -q
```

Two measurements, and the second is the one that matters: a plain Python allocation past a
512 MB limit is `SIGKILL`ed at it, exit 137, with the shell untouched; and **the eight
`test_scf.py` energy comparisons run to completion under `MemoryMax=4G`**, peak RSS 1.0 GB,
which is the same class of work that a 16 GB `ulimit -v` failed. That contrast *is* the
argument — XLA's address-space reservations are not charged to a cgroup, so a resident cap
can be set near what the work actually uses.

Two caps that look like this one and are not:

- **`ulimit -v` is not the cap to reach for** — it bounds *virtual* address space, and XLA
  reserves arenas far larger than it ever resides in, so a 16 GB cap fails tests that need
  a couple of GB (`PERFORMANCE.md` has the episode).
- **`XLA_PYTHON_CLIENT_MEM_FRACTION` is a GPU knob.** It sizes the PJRT *device*
  allocator's pool, which is what the 2026-09-04 H200 entry in `PERFORMANCE.md` reads
  against; development here is CPU-only, where there is no such pool and nothing to bound.
  The absence of a CPU equivalent inside JAX is exactly why this item is open.

**`tools/run_regression.sh` is where this is wired, and it is the way to run anything
long.** Each file goes into its own scope, a kill lands there rather than on the loop, and
the loop writes `killed (SIGKILL, cap=…)` as that file's durable summary line and starts
the next one — so a kill costs one file's *result*, which is what the per-file runner was
always for and what the kill taking the runner defeated. Three things it does that are not
obvious and are each a bug that was hit while writing it: the cap is `DEFUMAT_TEST_MEM_MAX`
(`off` for none, and a machine without cgroup delegation says so and runs uncapped); a file
the cap killed is **retried** on the next run rather than skipped, since the reason to
resume after a kill is that something changed; and the unit name carries the run's PID,
because a killed scope stays *loaded* and reusing the name fails with "already loaded",
which reads as a test failure. It also writes an `in-flight.log` line before starting a
file — **what was running is the thing a kill destroys**, and no cap can be trusted to
cover every way that happens.

**What is still missing is the named failure.** A `psutil` RSS watchdog in the autouse
fixture, failing a single test as it approaches the cap, turns an anonymous `SIGKILL` into
a test name — the difference between a lost afternoon and a bug report — and it is also
the only form of this that works inside `tools/test-fast.sh`, which is one process by
design. Until then: run anything long through `run_regression.sh`, and **commit before
starting it**.

What neither bound touches is the peak *inside* one test, which is a real cost to be sized
in advance rather than discovered: the backward pass of an ultrasoft or PAW derivative
carries the augmentation table `Q_ij(G)` — `nh^2 x ngm` per atom, and `nh` is in the
twenties for a fully-relativistic dataset. On a **slab** that is tens of GB and is why a
bismuthene spinor force does not run here at all (P46), while the same physics on a small
bulk cell runs in 33 seconds.

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
tools/test-fast.sh                     # THE GATE: everything not marked slow, ~4.5 min
python3 -m pytest -m slow              # the other 588, over two hours
tools/run_regression.sh                # the same slow set, one capped process per file
python3 -m pytest tests/unit/test_qeref.py::test_scf_silicon   # a single test
python3 -m defumat.cli inspect <qe-output>   # summarise what the parser reads
tools/export_notebooks.sh                     # re-execute notebooks + refresh .md exports
```

**The suite is two groups and `slow` is the line.** `tools/test-fast.sh` is
`pytest -m "not slow"`: **1634 tests in 4.5 minutes**, and it is what runs before
a push. The slow set is 588 tests and **over two hours** — it runs when it is
asked for, not on every change. The split cuts across `unit` and `regression`
both, because it is about cost and not about kind: a cheap regression case
against a two-atom reference is in the gate, and an expensive unit test is not.

**The slow set is not optional, it is just not per-push.** Run it before a
release, after touching anything in the SCF, the eigensolver or the response
stack, and whenever a number in this file changes. It is two hours precisely
because it is the part that catches what the gate cannot, and the one time it
was run end to end it found **three phases' claims had drifted** — P29's stale
refusal list and its broken BFGS metric, P36's 8.7e-14 wedge agreement, and two
notebooks whose committed outputs no longer matched their code (`PLAN.md` P38).
`tools/run_regression.sh` exists for running it in pieces: one **memory-capped**
pytest invocation per file, a durable summary line each, and a file already in
the summary is skipped, so an interrupted run resumes instead of restarting.

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
  follows the platform** — QE's loop on a CPU, one k-point at a time as `c_bands.f90` and
  `sum_band.f90` do it, and the whole axis at once on an accelerator, where the cache
  argument behind that loop does not exist and inheriting it gives up 4.5x. The band dial
  moves with it, never separately: `k=all, b=1` is measured to be worse than either end.
  `k_batch`
  reaches every entry point (`run_scf`, `run_bands`, `run_nscf`, `run_dos`, `Calculation`,
  `DEFUMAT_K_BATCH`), `None` asks for one `vmap` over the whole axis, and the chunked form
  is a `lax.map`/`lax.scan` so it stays compiled once and differentiable. Anything new that
  walks the k axis goes through `map_k`/`sum_k` rather than calling `vmap` itself; the
  chunk size must never be visible in a result beyond round-off.
- Use `donate_argnums` for the large wavefunction and density buffers.

`PLAN.md` §1 and §5 hold the full reasoning and the rest of the GPU notes.
