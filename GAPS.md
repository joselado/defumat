# Capability gaps

Combinations of features that do not work, from the sweep of 2026-08-29 — four
agents over the SCF/spin matrix, the response stack, the `Calculator`/workflow
plumbing, and a repo-wide hunt for silent failure. The nine that were bugs with a
*refusal* for a fix are closed (`tests/unit/test_sibling_refusals.py`); what is
left is here.

This is not `PLAN.md`. That file is the phase record and says what was built and
what each phase's traps were; this one is a list of what a user can ask for and
not get, with the missing term named and the ingredients that already exist
beside it. `CLAUDE.md`'s **Outstanding** paragraph is the short version of §3.

**Verification.** Entries marked *verified* were read in the source directly.
The rest carry the reporting agent's file:line and have not been independently
opened — the evidence is specific enough to act on and specific enough to be
wrong. Check before building on one.

---

## 1. Still bugs

A run that starts and returns a plausible number. Unlike the nine that were
fixed, none of these has a refusal for a fix — each is a term or a forwarding
that should be there.

### The denser k-grid ignores `nosym` and the magnetic group — *verified*

`workflows/nscf.py:241`, `denser_grid`, calls `find_symmetries(cell, structure)`
directly and reduces the grid with the whole crystal group. `System.
_recelled_kpoints` (`system/builder.py:385-407`) is the correct four lines and
does consult `nosym`, `noinv` and the magnetic group, and `System.
symmetry_group()` already applies `magnetic_symmetries` for `nspin_mag == 4`.

So `get_dos(grid=...)` / `run_dos(grid=...)` / `run_pdos(grid=...)` reduce with
symmetries the run said not to use. It bites a spin spiral hardest, because a
spiral is *required* to be `nosym` (the spin space group is not written), and a
noncollinear magnet next, because the magnetic group is smaller than the
crystal's and there is no `-k = k`.

**Fix:** mirror `_recelled_kpoints`. `KPoints.automatic` already takes
`time_reversal` and `t_rev`. Cheap.

### The PDOS symmetrises a `nosym` run — *verified (site)*

`projwfc/projections.py:192` gates the projection symmetrisation on the
`symmetrize` keyword (default `True`) and never on `calculation.use_symmetry`,
which exists at `scf/driver.py:1131` for exactly this. QE gets it for free:
`sym_proj_k` reads `nsym`, which `setup.f90` has already collapsed to 1.

This is the shape of the `dielectric_tensor` bug P28b found — symmetrising a
`nosym` run — in a second place.

**Fix:** `if symmetrize and calculation.use_symmetry`, at
`projections.py:192` and `workflows/pdos.py:523`. One boolean.

### DFT+U reaches the topology workflows without its Hubbard term

`workflows/topology.py:178-180`: `DFTSource.states` builds
`calculation.hamiltonian(potential.v_scf, self._ddd_paw())` — two positional
arguments into a signature whose third is `hubbard=None` (`scf/driver.py:2560`).
So `run_berry_curvature` / `run_z2` / `run_z2_3d` diagonalise the **un-Hubbard**
Hamiltonian, and `DFTSource._check_gap` then certifies the occupied manifold as
isolated using those eigenvalues. A confident integer off the wrong bands.

**Fix:** an `ns` field on `DFTSource` threaded the way `becsum` already is, plus
`calculation.hubbard_terms(ns)`, plus a refusal at the boundary when it is
absent. `Calculator._call_options` would supply `ns` the moment `run_z2` names
it. The meta-GGA half of the same hole is at least loud — `v_of_rho` raises —
but the message lands deep in the potential builder rather than at the workflow
boundary.

### `with_kpoints` does not apply `degspin`

`calculator.py:656`: the k-set is substituted raw. `system/kpoints.py`'s
`for_spin` halves the weights for `nspin` in (2, 4) and every constructor applies
the spin degeneracy unconditionally, so an LSDA or noncollinear user handing in
a `KPoints.automatic` — the thing `with_kpoints`'s own docstring advertises,
"the unreduced grid beside its irreducible wedge … same energy, fewer
k-points" — counts every electron twice. It does not fail; it moves the Fermi
level and integrates to the right electron count at the wrong energy.

**Fix is not a blind wrap:** `for_spin` is **not idempotent**, so a k-set that
came out of `denser_grid` must not be halved again. It needs a convention —
normalise at the boundary and stop doing it in `denser_grid`, or carry a flag on
`KPoints`.

### `SHARED_OPTIONS` are documented as universal and are not forwarded

`calculator.py:129` promises "any of `SHARED_OPTIONS`, applied to every method
that names them", and nine response methods (`get_dielectric_tensor`,
`get_phonons`, `get_strain_response`, `get_raman_tensors`,
`get_electrostriction`, `get_elastic_constants`, `get_forces`, `get_stress`) do
not call `_defaults_for`. A `Calculator(system, verbose=True,
max_iterations=200)` gets a verbose 200-iteration relaxation and a silently
default, silently quiet dielectric solve.

**And the one method that *does* forward is the sharper problem.**
`get_absorption` forwards, and `run_absorption`'s `max_iterations`
(`workflows/tddft.py:159`) is the **bootstrap Dyson fixed point's** limit, not
the SCF's — so a calculator built with `max_iterations=12` to cap the SCF
silently caps the Dyson iteration at 12. `mixing_mode`/`mixing_beta` collide the
same way between the SCF mixer and `response/mixing.py`.

**Fix:** forward in the nine, *or* narrow `SHARED_OPTIONS` so the docstring is
true. Whichever is chosen, the name collision needs deciding first — it is the
part that is currently wrong rather than merely absent.

### `qcutz` / `ecfixed` / `q2sigma` are read by nothing

Zero hits in the package, and not in `_REFUSED_SWITCHES` either
(`system/builder.py:768-810`), whose stated criterion — "input switches that
change the physics of a run this code otherwise supports … refused rather than
ignored" — covers them exactly. An input using QE's smoothed constant-cutoff
kinetic functional runs with the plain `G^2` and converges somewhere else.

It matters where P29 already records the problem it exists to solve: the 0.45 Ry
frozen-basis Pulay error on five-layer graphite in a variable-cell relaxation.

**Fix:** one `erf` term in `basis/planewaves.py:116`
(`g2_kin.f90:43-53`), three `&system` variables, and the pass-through. **The
stress comes free** — QE needs a separately hand-derived `kfac`
(`stres_knl.f90:63-66`) and here the stress is `jax.grad` of the energy. Cheap,
and the alternative (a `_REFUSED_SWITCHES` entry) is cheaper still.

### The converged field does not cross into a fixed-density run

`workflows/nscf.py:111` builds a fresh `Calculation`, whose `__init__` rebuilds
the magnetic field from the *input's* `b_field` / `constrained_magnetization`.
`SCFResult.magnetic_field` carries the converged object and is dropped.

That is wrong in both directions and `reducebf` is why. Elk's `reducebf`
(manual §5.104) exists so a symmetry-breaking field can drive the SCF off the
unpolarized solution and then be scaled to nothing — after ~25 iterations at
0.9 the field is 7% of its input value, and the converged state is a field-free
one. A band structure or DOS afterwards re-applies the **full** input field. The
fixed-spin-moment case is the same: `i_cons = 3`'s converged field is state, not
input.

**Fix:** a `field`/`field_scale` parameter on `fixed_density_states` and the
three workflows above — `Calculation.potential` already takes both — and
`Calculator._call_options` injecting `result.magnetic_field` the way it already
injects `ns`, `tau` and `becsum`.

---

## 2. Unreachable — the physics works, the plumbing cannot express it

### PAW or meta-GGA projected DOS on a denser grid — *verified*

`workflows/pdos.py:595` passes nine positional arguments into
`fixed_density_states`' ten-argument signature: `ns` goes, `tau` and `becsum` do
not. They are sitting on the `result` argument the function already has.

This is P38's defect — "`run_dos` never forwarded `becsum` or `ns`, so a PAW or
DFT+U density of states on a denser grid was unreachable" — surviving in the
sibling workflow that P38 did not touch. `run_dos` shows the exact fix, two
parameters and a call.

### `Calculator.get_band_velocities` does not exist — *verified*

`response/velocity.py:345` exports `band_velocities(calculation, result,
kpoints=None)` — already the facade's calling convention — and there is no
`get_*` for it. A user driving the package the way the README says to cannot get
a Fermi velocity or an effective mass without threading `Calculation` and
`SCFResult` by hand.

CLAUDE.md's own rule ("a new feature adds a `get_*` method in the same pass that
adds its entry point") was not met when P24 landed the velocity operator. Three
lines.

### `nspin` where `nspin_mag` was meant, for a meta-GGA spinor run

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

### A field cannot induce a moment in a nonmagnetic noncollinear run

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

### Sternheimer response + `nspin = 2` — *the single widest guard*

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

### Forces, stress and relaxation + noncollinear magnetism or spin-orbit

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

### `occupations = 'fixed'` + `nspin = 2` + `tot_magnetization`

`scf/occupations.py:55`, and `residual.py:262` forwards the same call, so no
route reaches it. An LSDA insulator or a fixed-moment magnet at fixed
occupations — QE's own supported combination — stops outright.

**Missing:** `iweights_only` per channel with `degspin = 1`: fill
`floor(nelup)` bands in channel 0 and `floor(neldw)` in channel 1, report the
highest occupied level of each (`PW/src/iweights.f90:82-129`). No new physics.
`spin_electron_counts` already returns `(nelup, neldw)` and is computed
unconditionally; `fixed_occupations` already takes a `degeneracy` argument.

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

### Kubo Berry curvature on a plane-wave calculation — *verified*

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
energy. Do this one regardless; it is the same guard-on-one-sibling shape as the
nine already fixed. (b) The kinetic-energy-density response, if it is ever to
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
