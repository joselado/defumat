# Capability gaps

Combinations of features that do not work, from the sweep of 2026-08-29 — four
agents over the SCF/spin matrix, the response stack, the `Calculator`/workflow
plumbing, and a repo-wide hunt for silent failure. The nine that were bugs with a
*refusal* for a fix are closed (`tests/unit/test_sibling_refusals.py`), so are
the seven of §1 whose fix was a term or a forwarding
(`tests/unit/test_state_across_boundaries.py`), and so is the whole of §2 plus
**one** entry of §3 — `occupations = 'fixed'` with `nspin = 2`, the fourth
(`tests/unit/test_unreachable_combinations.py`). What is left is the rest of §3,
and the widest of them is still the first one listed there: the Sternheimer
response with `nspin = 2`, which is untouched.

This is not `PLAN.md`. That file is the phase record and says what was built and
what each phase's traps were; this one is a list of what a user can ask for and
not get, with the missing term named and the ingredients that already exist
beside it. `CLAUDE.md`'s **Outstanding** paragraph is the short version of §3.

**Verification.** Entries marked *verified* were read in the source directly.
The rest carry the reporting agent's file:line and have not been independently
opened — the evidence is specific enough to act on and specific enough to be
wrong. Check before building on one.

---

## 0. Found after the sweep -- **closed 2026-09-01**

One entry, and it is the sweep's own §1 pattern in a place the sweep did not
look. **`DFTSource` did not forward the converged magnetic field.** Every
fixed-density topological invariant -- `run_berry_curvature`, `run_z2`,
`run_z2_3d`, and the polarization added in P56 -- builds its potential from a
frozen density through `DFTSource.states`, which called
`calculation.potential(self.density)` with no field argument;
`Calculation.potential` then falls back to `self.magnetic_field`, the field the
**input** asked for, at full scale. Wherever `reducebf` or the fixed-spin-moment
scheme had changed the field during the SCF, every eigenvalue was shifted by a
field the run never converged under.

It has no symptom, which is why it is worth recording: the invariant is still an
integer. `workflows/nscf.py:fixed_density_states` has refused this since the
sweep and the topological source had no counterpart, so this is the *same* entry
one layer over -- the pattern the sweep's own §1 is full of. Fixed:
`field`/`field_scale` on `DFTSource` and on all four entry points, the refusal
copied word for word from `fixed_density_states`, and `_STATE_ARGUMENTS` already
mapped both names so `Calculator` injects them unchanged. Tests in
`tests/unit/test_state_across_boundaries.py`, one structural and one that builds
a noncollinear silicon carrying `b_field(3)` and asserts the refusal fires.

## 1. Still bugs — **closed 2026-08-29**

All seven are fixed; `tests/unit/test_state_across_boundaries.py` holds the
tests. Kept here rather than deleted, because what each one *was* is the useful
part.

| bug | what it did | fix |
|---|---|---|
| `denser_grid` ignored `nosym`, `noinv` and the magnetic group | reduced a DOS/PDOS grid with operations the SCF did not use — worst on a spin spiral, which is *required* to be `nosym` | mirrors `System._recelled_kpoints`' four lines (`workflows/nscf.py`) |
| the PDOS symmetrised a `nosym` run | `sym_proj_k`'s average over a group the states do not share; P28b's `dielectric_tensor` defect in a second place | `symmetrize and calculation.use_symmetry` (`projwfc/projections.py`) |
| DFT+U reached the topology workflows without its Hubbard term | `hamiltonian` called with two positional arguments of three, so the invariant came off eigenvalues wrong by the whole Hubbard shift — and `_check_gap` certified the manifold with them | `ns` on `DFTSource` and the three `run_*`, refused when absent; meta-GGA refused by name at the boundary (`workflows/topology.py`) |
| `with_kpoints` skipped `degspin` | counted every electron twice on exactly the comparison its own docstring recommends | `for_spin` made **idempotent** via `KPoints.spin_normalized`, then applied on the way in (`system/kpoints.py`, `calculator.py`) |
| `SHARED_OPTIONS` documented as universal | nine methods dropped them; the one that forwarded was capping a **Dyson** iteration with the SCF's `max_iterations` | `SCF_ONLY_OPTIONS` split out and `_defaults_for(exclude=)`; the nine forward the rest (`calculator.py`) |
| `qcutz`/`ecfixed`/`q2sigma` read by nothing | ran with the plain `G^2` and converged elsewhere | `_REFUSED_SWITCHES` — see below (`system/builder.py`) |
| the converged field did not cross into a fixed-density run | `reducebf` scales the field towards zero as the SCF runs, and a band structure afterwards re-applied the **input** field | `SCFResult.field_scale` added beside `magnetic_field`, `field`/`field_scale` on `fixed_density_states`/`run_nscf`/`run_bands`/`run_dos`, injected by `_call_options` (`scf/driver.py`, `workflows/`, `calculator.py`) |

Three of them are worth a note beyond the table.

**`for_spin` needed a flag, not a call.** The division by `degspin` is not
idempotent and a k-set reaches a polarized run by four routes — `build_system`,
`_recelled_kpoints`, `denser_grid`, `with_kpoints` — none of which can see where
the set came from. `KPoints.spin_normalized` is set **only** when the division
happens, so a set built for `nspin = 1` is still normalised when it later
reaches a polarized run.

**`qcutz` is refused rather than implemented, and that is the whole fix.** It is
one `erf` in `basis/planewaves.py:kinetic` and the stress would come free — the
stress here is `jax.grad` of the energy where QE needs a separately hand-derived
`kfac` in `stres_knl.f90`. But every vendored input that sets it is a `CPV/`
case, so there is no `pw.x` benchmark to validate against, and an unvalidated
`erf` term does not close a silent bug, it moves it. The refusal does close it.
Implementing it needs a generated reference first.

**`SHARED_OPTIONS` was wrong in both directions and forwarding everything would
have spread the worse half.** `max_iterations` is the SCF's loop in `run_scf`,
the self-consistent *response's* in `dielectric_tensor`, and the *Dyson fixed
point's* in `run_absorption` — three loops, one word. What stays shared is what
means the same thing wherever it is named: `nbnd`, `conv_thr`, `k_batch`,
`diagonalization`, `verbose`. An SCF-only option is still reachable per call,
where the name is unambiguous.

**One entry from §2 closed with them**, because it was on the same line: the
PAW/meta-GGA projected DOS on a denser grid. `run_pdos` passed nine positional
arguments into a longer signature; it now forwards the whole mixed state off the
`result` it already had.

## 2. Unreachable — **closed 2026-08-29**

All four are fixed; `tests/unit/test_unreachable_combinations.py` holds the
tests. Kept for the same reason §1 is: what each one *was* is the useful part.

| gap | what it did | fix |
|---|---|---|
| `Calculator.get_band_velocities` did not exist | a Fermi velocity needed `Calculation` and `SCFResult` threaded by hand, against a README that says every feature is a method | three lines of delegation (`calculator.py`) — **and** the mixed state forwarded inside `band_velocities`, which passed only `ns` into its NSCF branch and so refused every PAW run (`response/velocity.py`) |
| `tau` sized as `nspin` where it is `nspin_mag` | four channels asked of a quantity produced with one, for a *nonmagnetic* spin-orbit meta-GGA: the residual solver died on the reshape and a `starting_from` promotion dropped its converged `tau` in silence | the density's own shape, read off rather than rebuilt (`scf/residual.py`, `scf/driver.py` twice) |
| a field could not induce a moment in a nonmagnetic noncollinear run | `nspin_mag = 1`, so the potential build died on a shape mismatch rather than a message | refused by name, pointing at `starting_magnetization` (`scf/driver.py`) |
| `occupations = 'fixed'` + `nspin = 2` | refused for want of a benchmark | one generated (`o-atom-fixed-lsda`), then implemented — see §3 |

Three of them are worth a note beyond the table.

**The band-velocity gap was two gaps and the second was the worse one.**
`band_velocities` forwards into `fixed_density_states` for its `kpoints=` branch
and passed `ns` alone, with a comment saying PAW "raises here rather than below"
— which was true when it was written and stopped being true when the previous
pass gave that function `becsum`, `tau`, `field` and `field_scale`. Adding a
facade over it without the forwarding would have shipped P38's own defect
through a new front door.

**`pw.x` does not support a field on a nonmagnetic noncollinear run either, and
its failure is the worse one.** `setup.f90:219` decides `domag` from
`starting_magnetization` and from nothing else, but `scf_mod.f90:140` allocates
the density with `nspin = 4` whatever `domag` says — so `add_bfield` has
channels 2:4 to write the field into and writes it there, and then
`vloc_psi_nc` applies those channels only `IF (domag)` (`vloc_psi_acc.f90:331`).
The field never reaches a wavefunction. Such a run converges, reports success,
and is the field-free calculation. GAPS' original suggestion — "force `domag`
when a field is present" — would have been a deviation from QE dressed as a
transcription; the refusal names the one input variable that fixes it.

**`nspin_mag` is not a spelling of `nspin`, and the fix is to stop spelling it at
all.** All three sites rebuilt a channel count from a spin number where the
density beside them already had it. Reading it off the density is correct in all
three regimes at once and cannot come apart again.

## 2b. Unreachable — the original list

### ~~PAW or meta-GGA projected DOS on a denser grid~~ — **closed**

`run_pdos` now forwards `tau`, `becsum`, `field` and `field_scale` off the
`result` it already receives, beside the `ns` it always passed. See §1.

### ~~`Calculator.get_band_velocities` does not exist~~ — **closed**

`response/velocity.py:345` exports `band_velocities(calculation, result,
kpoints=None)` — already the facade's calling convention — and there is no
`get_*` for it. A user driving the package the way the README says to cannot get
a Fermi velocity or an effective mass without threading `Calculation` and
`SCFResult` by hand.

CLAUDE.md's own rule ("a new feature adds a `get_*` method in the same pass that
adds its entry point") was not met when P24 landed the velocity operator. Three
lines.

### ~~`nspin` where `nspin_mag` was meant, for a meta-GGA spinor run~~ — **closed**

`scf/residual.py:290`: `tau_shape = (calculation.nspin,) + ...`, which is 4 for a
nonmagnetic spin-orbit run — but `tau` is produced with `nspin_mag` channels,
which is **1**. This is the three-spin-numbers rule CLAUDE.md states, missed in
two expressions (`residual.py:290` and `_starting_tau`, `driver.py:506`), three
lines from a correct use of `nspin_mag` at `driver.py:2544`.

TB09 + spin-orbit is validated (P31) and the residual solver is validated (P22);
together they die on a reshape from inside the residual packing. On the
`starting_from` path the same mismatch is **silent**: `driver.py:3269` finds the
shapes unequal, drops the converged `tau` without a word and falls back to
Thomas-Fermi.

### ~~A field cannot induce a moment in a nonmagnetic noncollinear run~~ — **closed**

`driver.py:1395` guards only `nspin == 1`, so a `noncolin` run with a `B_field`
and every `starting_magnetization` at zero passes — and then `nspin_mag` is 1,
the density has no magnetization channels, and the potential build dies on a
shape mismatch rather than a message.

This is the standard way to break spin symmetry with a field, and it is the
escape hatch from the LSDA-needs-`starting_magnetization` rule now enforced at
input. **Fix:** force `domag` when a field or a constraint is present
(`setup.f90:215-226` is where QE decides it), or refuse by name and say to set
`starting_magnetization`. `System.domag` is one property.

---

## 3. Refused by name, and a real term is missing

Ranked by what they unlock.

### ~~Sternheimer response + `nspin = 2`~~ — **closed 2026-08-30 (P45)**, three narrower things are not

The solve is spin-polarized: one occupied-band count *per channel* (`occupied_counts`),
`_alpha_pv` maximised over channels, and all three callers that derived `nelec/2`
themselves (`efield`, `phonon`, `strain`) fixed. `chi_0` against a central difference of
the density: **1.8e-6** (antiferromagnetic H chain, smeared metal) and **1.1e-6** (triplet
O2, the sliced branch, ultrasoft). The `nspin = 1` / `nspin = 2` identity on silicon's
`epsilon_infinity` is **6.2e-14**. What is still refused, each by name: `tot_magnetization`
with a smearing (`Smearing.ef` has no spin axis); a filling that cuts a **degenerate
multiplet**, where the CG converges and the answer is 100% wrong (measured, not argued);
the **screened** response of a magnetic cell with vacuum, because `dv_of_drho` for
`nspin = 2` genuinely diverges where a channel density reaches zero (1504 of O2's 91125
grid points, and exactly 1504 NaN); and the *second-derivative assemblies* — `Z*`, the
dynamical matrix, the strain response.

The original entry follows.

### The refusal as it stood

`response/sternheimer.py:836`. This one refusal blocks **every** response
quantity for **every** spin-polarized system: no dielectric tensor, no phonons,
no Raman, no strain response for nickel, iron, NiO or FeO — all of which have
validated SCF ground states here (P9's eight LSDA benchmarks).

The stated reason is that the occupied-band count is one number for both
channels and a spin-polarized insulator's channels are occupied to different
depths. **That reason does not bind a metal**, whose branch never takes the
slice. QE never meets the problem because LSDA doubles `nks` there, so its spin
channels are separate k-points (`setup_nbnd_occ.f90`).

**Missing:** a per-channel occupied-band count in `SternheimerSolver.__init__`
and `_alpha_pv` — for a metal, just `emax` over the kept bands. The shared-Fermi
case is cheap. The `tot_magnetization` case is real work: two Fermi levels means
`Smearing.ef` needs a spin axis.

### ~~Forces, stress and relaxation + noncollinear magnetism or spin-orbit~~ — **closed 2026-08-30 (P46)**

The estimate below was right about the physics and wrong about the size: the *layout* was
the larger half, because a spinor state is `(1, nk, nbnd, 2 npwx)` and the kinetic term has
to read `state_kinetic` too. Against `pw.x`: **8.9e-7 Ry/bohr** (four-atom noncollinear
hydrogen chain), **7.5e-6** (ultrasoft Pt2 with spin-orbit) and **7.3e-7** (PAW), the stress
on six cases to **<=1.2e-6 Ry/bohr^3**, and a finite difference of the frozen energy — no
Fortran involved — to **6.2e-9 Ry/bohr**. The refusal was *narrowed*, not deleted: the
analytic transcriptions, anything through the Sternheimer solver, the elastic constants
(which reach the functional directly, so the spinor path is opt-in) and the force on an atom
of a spin spiral all still refuse, each naming its own missing term.

The original entry follows.

### The refusal as it stood

`forces/energy.py:156`, `reject_spinors`. P17's noncollinear magnetism and P14's
spin-orbit coupling are both validated against `pw.x` to ~1e-9 Ry and have no
forces, no stress, no relaxation, no vc-relax and no phonons. Bcc iron's
noncollinear ground state converges and cannot be relaxed.

**Missing:** two substitutions in one functional — the nonlocal quadratic form
with the complex 2x2-in-spin `dvan_so`, and the constraint term's `<psi|S|psi>`
with `qq_so`. Nothing else in `energy_at` cares about the spinor axis.

**The decisive ingredient is already written**: `forces/spiral.py` contains a
complete spinor frozen-energy functional — `spiral_energy` (`spiral.py:116-165`)
assembles kinetic + nonlocal + local + Hartree + XC + Ewald + dispersion +
constraint + entropy for a two-component run, and `_nonlocal_energy`
(`spiral.py:271`) is the spinor form. It was written for `dE/dq` and is the same
functional a spinor force needs.

### Strain response and elastic constants + a metal

`response/strain.py:234` calls `require_a_sternheimer_regime(calculation)`
without `metals=True`; the sibling perturbation passes it
(`response/phonon.py:293`). So the elastic constants of any metal are
unreachable — aluminium's `C_11`/`C_12`/`C_44` being the textbook case and the
very cell P28/P28a already run phonons on. The message a user gets says a metal
"has no `epsilon_infinity` and no Born effective charge", which is not what they
asked for.

**Missing:** the two pieces P28 already wrote for the *displacement* coordinate
— the Fermi-level shift (a strain changes the volume, so `ef` moves) and the
split between the frozen-Hessian weight `wg` and the electronic `2 wk`.
`localdos` and `ef_shift` are implemented (`sternheimer.py:565-585`) and the
metal path is validated (P24c, `chi_0` on fcc aluminium to 2.5e-7). No QE
counterpart to transcribe: `ph.x` has no strain perturbation.

### ~~`occupations = 'fixed'` + `nspin = 2` + `tot_magnetization`~~ — **closed**

Implemented, and the refusal it replaced was wrong about QE in a way that made
the job half the size. `fixed_occupations` takes `counts` now, `residual.py`
forwards it beside the smeared branch's, and the builder makes QE's two
input-time checks.

**The old refusal said QE "fills the two channels from `tot_magnetization` or
from a shared Fermi level".** It does not. `input.f90:784-800` refuses
`occupations = 'fixed'` with LSDA outright unless `tot_magnetization` is given
("fixed occupations and lsda need tot_magnetization"), and then requires it —
and the total charge — to be an **integer**. So there is no shared-Fermi fixed
branch to write, in either code, and both refusals are made here now, at input,
where QE makes them. They are checked in QE's order too, which matters because
both fire on a bare `nspin = 2`: the fixed rule is at line 784 and the
`starting_magnetization` one at 1507.

**`NINT` and not `floor`**, which is what the entry above guessed: an odd
electron count with an even magnetization gives a half-integer `nelup` and QE
rounds it up rather than refusing.

**The reference had to be generated** (`tests/data/qe/o-atom-fixed-lsda.in`),
because QE's test-suite has no fixed-occupation LSDA case anywhere — that is
what the old refusal's "no committed benchmark" meant and it was accurate. An
oxygen atom at `tot_magnetization = 2` fills four bands up and two down, which is
the Hund's-rule ground state and the configuration `pw_atom/atom-lsda.in`
reaches by writing the occupations out by hand. Against the vendored `pw.x`:
**4.9e-9 Ry** in the total energy, 2.0000 in the moment, and every occupied
eigenvalue to the four decimals QE prints.

**One thing it cannot do, and it is the physics rather than the implementation.**
A fixed occupation whose filled/empty boundary cuts a **degenerate multiplet** —
which is exactly what a Hund's-rule atom is — makes the density map multivalued:
which member of the multiplet the eigensolver returns is arbitrary, so `F` is not
a function of the density and the **residual solver** has nothing to converge on.
Measured rather than assumed: the same atom with a smearing converges in four
Newton steps, and a *gapped* fixed LSDA cell agrees between the mixer and the
Newton solve to **5.3e-12 Ry**. The mixer damps its way past it. It is diagnosed
by name in `solvers.py` now instead of surfacing as a scipy "RHS must contain
only finite numbers".

### Projected DOS + noncollinear / spin-orbit

`projwfc/projections.py:163`. Every heavy-element calculation the package
advertises — the platinum spin-orbit benchmarks, a topological insulator whose
Z2 it computes — cannot have its orbital character resolved, in exactly the
regime where a `j`-resolved projection is the interesting one.

**Missing:** a spinor atomic-orbital builder (`atomic_wfc_nc_proj`, which builds
the `j`-resolved orbital from `sph_ind`/`spinor` and doubles `natomwfc`), the
spinor symmetrisation `sym_proj_so`, and `partialdos_nc`'s binning by `j`/`m_j`.
Everything downstream — the `S` metric, Löwdin orthogonalisation, the spilling
parameter, the DOS registry as a per-band weight — is regime-agnostic and
written. `hubbard/projectors.py` builds the same projector set and would inherit
it.

### ~~Kubo Berry curvature on a plane-wave calculation~~ — **closed 2026-08-30 (P47)**

Written against `VelocityOperator` band matrix elements (`topology/kubo.py`), never as a
dense `H(k)` — the `O(npw^2)` route CLAUDE.md forbids. Against the FHS flux, which shares
no machinery with it: a shrinking centred plaquette converges onto the pointwise value to
**3.2e-3**, and a 24x24 mesh agrees plaquette by plaquette to **1.45e-4**. Silicon's
curvature vanishes *pointwise* to **3.5e-5**. The truncation of the sum over empty states
is reported (`BerryCurvature.truncation`). **Ultrasoft and PAW stay refused by name**: the
`eps_n dS/dk` term is identically zero for a norm-conserving dataset, so nothing validated
here can see whether its off-diagonal convention is right, and `q_ij(b)` is the second
missing piece.

The original entry follows.

### The refusal as it stood

`topology/berry.py:189` refuses because "the Kubo route needs `H(k)` as a
differentiable function, which a plane-wave calculation does not yet expose: the
velocity operator needs `d(vkb)/dk`". **P24 wrote that**: `VelocityOperator.projectors`
(`response/velocity.py:195`) is "the projectors' derivative about their own
atom". CLAUDE.md's topology paragraph ("belongs to P11") is stale with it. So a smooth `Omega(k)`
map of a real crystal — what anyone plotting anomalous-Hall physics wants beside
the integer — is refused for a reason the repo satisfied two phases ago.

**Missing:** a `PlaneWaveStates.hamiltonian`-shaped adaptor, or a `kubo`
implementation written against `VelocityOperator`. The registry entry already
exists and is reachable only from `ModelStates`. **The caveat that keeps FHS the
right default stands**: the Kubo expression has the `1/(eps_n - eps_m)^2`
denominator rule D4 forbids, so a near-degenerate pair still needs handling the
lattice sum does not.

### Meta-GGA + DFT+U — *verified*

`scf/driver.py:1228`. The refusal's stated reason — "`_solve_residual`'s
convergence measure reads the Hubbard block off the *end* of the packed state,
and with `tau` packed after it that slice is `tau`" — **is no longer true**:
`driver.py:3097-3101` says the opposite in as many words -- "sliced by
``unpack`` and not off the end of the vector: ``ns`` is the last block only
while nothing follows it, and ``tau`` does". The refusal itself concedes
"nothing in the physics forbids it — `vhpsi` is a separate term and does not
touch `tau`".

A correlated oxide is the archetypal case for both a Hubbard `U` and mBJ's gap.
What is missing is validation, not code, and there is no QE counterpart to
compare against (`pw.x` reaches TB09 only through libxc and refuses meta-GGA
with USPP/PAW). Whoever reads the current message goes looking for a packing bug
that is not there.

### Linear response + meta-GGA — *the hard one*

`require_a_sternheimer_regime` has no meta-GGA branch, so what stops such a run
is incidental: `efield.py:263` calls `calculation.potential(density)` and
`v_of_rho` raises asking for `tau`. That reads like a missing keyword argument
and is not.

**Two separable things.** (a) A named refusal — `reject_potential_only` inside
`require_a_sternheimer_regime` — so the message says the functional has no
energy. **Done 2026-08-30 with P45**, which was inside that function anyway. (b) The kinetic-energy-density response, if it is ever to
work: `tau` comes from the states, so `dtau` is a tangent of `sum_n w |grad
psi_n|^2`, the screening kernel needs `dv/dtau` beside `dv/drho`, and the
Hamiltonian needs the `grad . (dv/dtau) grad` operator (`h_psi_meta.f90`). None
of it is written, and CLAUDE.md is explicit that the potential-only branch was
chosen *because* it needs no such term. QE has no meta-GGA DFPT either.

---

## 4. The structural finding

Almost every entry above and every one of the nine already fixed is **the same
guard, or the same forwarding, missing from a sibling**: `run_dos` against
`run_pdos`, the stress against the force, autodiff against analytic,
`electrostriction` against `elastic_constants`, `_recelled_kpoints` against
`denser_grid`, `phonon.py` against `strain.py`. Guards are attached to entry
points one at a time and the second entry point to the same physics arrives
later.

`tests/unit/test_sibling_refusals.py` is the concrete form of the answer for the
nine that are fixed. The general form — enumerate the paths that reach a
refusal and assert it fires on each — would have caught most of this list at
once, and is worth writing before the next feature adds a third path to
something.
