# What Elk computes that Quantum ESPRESSO does not

A survey of Elk's task list (`docs/elk_manual.txt` §5.127 in the user's `elkpy`
checkout — see `CLAUDE.md`'s reading notes for where the source lives) against
QE 7.5, kept as a backlog of **cheap** additions that each buy a new physical
observable rather than another knob on an existing one.

The filter applied throughout: the feature must (a) have no counterpart in
`pw.x` or its post-processing tools, verified by `grep` over the vendored
Fortran rather than by recollection, (b) run at NSCF cost or less, and (c) reuse
machinery this package already has, so that the new code is an assembly and not
a second implementation of something validated.

**Two of the six were taken in P48** — the effective mass tensor and the
site-resolved angular momenta — **a third in P50**, the piezoelectric tensor,
which is the one entry that fails the cheapness filter above and was taken
anyway (the *implementation* is one `jvp` of code that already exists), **a
fourth in P51**, the optical conductivity tensor with the Kerr angle and the
anomalous Hall conductivity, **a fifth in P52**, the Fermi-surface nesting
function, and **a sixth in P54**, second-harmonic generation. The rest are
recorded here with their validation route, because that is the part that decides
whether a phase is worth starting.

**Re-surveyed 2026-09-01**, against the package as it stands now rather than as
it stood when this file was written, and that added the last two entries below.
Neither passes the cheapness filter above and both are here anyway, for the
reason P50 established: what predicts effort is **how much of the assembly
already exists**, not what the quantity costs in Elk. §7, the magnetoelectric
tensor, is the one selected to be taken next; it had been missed altogether
rather than rejected, which is its own lesson about walking a task list once.

Doing the two taught one thing that applies to the rest: **the value of the
comparison was in what it found, not in the agreement.** Running Elk's own
`effmass` beside this one showed that Elk's number does not converge in its
stencil width — a basis-set discontinuity at a high-symmetry k-point that
neither code's output flags — and that was only visible because there were two
independent methods. Any entry below is worth more if it can be built with a
second route beside it.

---

## Taken

| | Elk task | Phase |
|---|---|---|
| Effective mass tensor `m*_ij` | 25 (`effmass.f90`) | P48a |
| Site-resolved `<L>`, `<S>`, `<J>` | 15/16 (`writelsj.f90`) | P48b |
| Piezoelectric tensor `e_(k)ij`, clamped-ion | 380 (`piezoelt.f90`) | P50 |
| Optical conductivity `sigma_ab(omega)`, MOKE, anomalous Hall | 121/122 (`dielectric.f90`, `moke.f90`) | P51 |
| Fermi-surface nesting function `N(q)` | 105 (`nesting.f90`) | P52 |
| Second-harmonic generation `chi^(2)(-2w; w, w)` | 125 (`nonlinopt.f90`) | P54 |

**The second-harmonic row was in the *rejected* table until P54 and was wrong
there**, which is worth recording because the reasoning that put it there is
the kind that repeats. It was rejected by pointing at P35's `chi^(2)` refusal
-- "the `<u_i|r_k|u_j>` piece, worth 42%" -- and that refusal is a statement
about the **Sternheimer stack**, where a field enters only through a source
term. `nonlinopt.f90` is a **sum over states**, which needs no such object: its
only ingredient is the interband dipole `-i v_nm / w_nm`, exact for an
eigenstate of the full `H(k)`. P53 drew that distinction for the shift current
and flagged this row; P54 is it being acted on. **Check which machine a
refusal belongs to before inheriting it.**

---

## 1. The optical conductivity tensor, MOKE and the anomalous Hall conductivity

**Elk task 122** (`moke.f90`, 87 lines) on top of task 121 (`dielectric.f90`).

The complex Kerr angle is

```
theta_K + i eta_K = -sigma_xy / ( sigma_xx sqrt(1 + 4 pi i sigma_xx / omega) )
```

so `moke.f90` is pure post-processing; all the work is the **full tensor**
`sigma_ab(omega)`, interband plus an intraband Drude term.

**What QE has, precisely.** `PP/src/epsilon.f90` does have an `offdiag_calc`
that forms the whole tensor, so this row's README ticks would be **(✓) for QE
and ✓ for Elk**, not blank in both. But it refuses ultrasoft outright (`okvan`
→ `errore('grid_build', 'USPP are not implemented')`), computes no Kerr angle,
and builds the dipole from momentum matrix elements `<psi|p|psi>` — which is
the one line `CLAUDE.md` already records as *not* to be transcribed, because it
is wrong with a nonlocal pseudopotential. `dH/dk` replaces it, as it already
does in P37's head.

**What is reused.** `VelocityOperator.matrix_elements` already returns
`(3, nspin, nk, nbnd, nbnd)` — every Cartesian component including the
off-diagonal ones. `tddft/chi0.py`'s occupied-empty pair sum is the same loop
with a different contraction.

**Why it is a phase and not an assembly.** `sigma_xy` vanishes identically
without magnetism *and* spin-orbit coupling at the same time, and its targets
are ferromagnetic metals (bcc Fe, hcp Co). `tddft/chi0.py` currently refuses
`nspin != 1`, `noncolin` and metals by name (`chi0.py:284-315`). Lifting those
three is the work; the runtime stays one NSCF plus a band-pair sum.

**Validation, and it needs nothing external.** The `omega -> 0` limit of
`sigma_xy` **is** the Berry-curvature integral P47 already computes by the Kubo
route — two constructions sharing no assembly must land on the same number.
`sigma_xx` must reproduce P37's RPA-without-local-fields `eps_M`. The local Elk
binary gives an all-electron sanity check on top.

**Payoff:** magneto-optical Kerr and Faraday rotation, and an independent check
on P47.

---

## 2. The Fermi-surface nesting function — **taken, P52**

**Elk task 105** (`nesting.f90`, 91 lines).

```
N(q) = sum_k  [ sum_n delta(eps_nk - E_F) ] [ sum_m delta(eps_m,k+q - E_F) ]
```

**QE has nothing** in the base distribution — `grep -ri nesting PP/src PW/src`
is empty. EPW has it, and EPW is out of scope.

**The cheapness claim was right and it was the interesting half.** Elk writes it
as an `O(N_k N_q)` double loop; the `mod(ivk + ivq, ngridk)` fold that puts
`k + q` back on the grid makes the sum a cyclic cross-correlation, so one FFT
over the k-grid gives the whole `q` dependence. Measured, that is **0.001 s
against Elk's 0.37 s** on a 12x12x12 aluminium grid — the one place in the
phase that is not a transcription, and the only one that needed to be.

**The validation closed inside the package**, which is what this file's method
note says to look for, and all three routes it named were used. The
free-electron Lindhard function is analytic and the code reproduces its `1/q`
law to 1e-4 and its hard cut at `2 k_F` to 1e-16. The hydrogen chain nests at
`q = 0.5` where `relax_spiral_q` relaxes the spiral to 0.500014 — the pairing
with P21 that this entry was chosen for, and it worked: nesting predicts the
pitch from the paramagnet's Fermi-surface geometry, the relaxation finds it in
a magnetic total energy, and they share no machinery. The Elk binary was run
too and is the weakest of the three, exactly as this file predicts: its
all-electron `D(E_F)` differs from a pseudopotential one by 3 per cent and `N`
is quadratic in it, so what agrees is the dimensionless `N(0)/D(E_F)^2`, to 1.5
per cent.

**One finding is worth carrying to the remaining entries.** The saving is not
in the correlation at all — it is that a symmetry-reduced wedge can be
**unfolded** onto the complete grid, because `eps_n(Rk) = eps_n(k)`, which
turns 1728 diagonalisations into 72. Elk does the same thing with `ivkik` and
the machinery was already here as the tetrahedron method's `equiv`. The trap
that comes with it is the P28a family: the group a grid is reduced with and the
group it is unfolded with are different questions on a `nosym`, `noinv` or
magnetic run, and a mismatch produces a plausible answer built from the wrong
bands. Anything below that walks a k-grid should go through the one function
(`workflows/nscf.py:grid_symmetry`) rather than deciding for itself.

**README ticks:** blank for QE, ✓ for Elk (task 105).

---

## 3. X-ray and magnetic structure factors

**Elk tasks 195/196** (`sfacrho.f90`, `sfacmag.f90`, ~85 lines each).

`F(H) = integral rho(r) exp(i H.r) d3r` — which in a plane-wave code **is**
`rho(G)`, an array that already exists. The magnetic one is `m(G)`, the
neutron-diffraction observable, and the noncollinear `m(r)` has been there since
P17.

**QE has nothing** — the only `PP/src` hits for "structure factor" are
`pw2wannier90.f90` and `addusdens1d.f90`, neither of which outputs one.

**Cost.** Free: an indexing operation plus the symmetry multiplicity of each
`H`. Elk's `genhvec`/`vhmat` machinery for choosing and transforming the `(hkl)`
set is the only real transcription.

**The caveat that must be stated rather than discovered.** A pseudopotential
density is **valence-only**, so this is not the experimental structure factor
unless the core is added back — by PAW reconstruction, or at least the NLCC core
charge. Claim it as the *valence* structure factor and do **not** promise
digit-agreement with the all-electron Elk binary; the size of that mismatch *is*
the physics content of the comparison. `wsfac` (Elk's energy window on which
states contribute) is worth having for the same reason: it is what makes the
quantity a probe of bonding rather than of the whole density.

**README ticks:** blank for QE, ✓ for Elk (tasks 195/196). It is the only
output in the package that a diffraction experiment can be held against
directly.

---

## 4. The exchange-correlation spin torque

**Elk task 160** (`torque.f90`, 37 lines — short because both fields already
exist there too).

```
tau = integral m(r) x B_xc(r) d3r
```

Zero at a converged noncollinear ground state, so it is a **diagnostic** rather
than an observable: it measures how far a constrained or non-self-consistent
magnetic configuration is from stationarity, and it is the quantity that
underlies spin dynamics and the adiabatic magnon picture.

**Not a phase.** A rider on notebook 11 (noncollinear magnetism) plus a test
asserting it vanishes at self-consistency and does not under a constraint.

---

## 5. The piezoelectric tensor — **taken, P50**

**Elk task 380** (`piezoelt.f90`). `pw.x` has the Berry-phase polarization
(`bp_c_phase.f90`) and nothing that differentiates it with respect to a strain;
the only occurrence of "piezo" in the vendored tree is a citation of
Vanderbilt's paper in a comment in that file.

**Elk's route is the expensive one and it was measured rather than assumed:**
`piezoelt.f90` runs one full ground state per strain tensor, computes the
Berry-phase polarization of each, and finite-differences them with a `2 pi`
branch fix-up between the two. It relaxes nothing (`tshift = .false.`, the atoms
carried by the lattice), so what it produces is the **clamped-ion** constant —
the same quantity P50 computes, which is what makes the comparison meaningful in
principle.

**The cheap route here, and it is what was implemented.** The clamped-ion
constant is a **mixed second derivative**,

```
e_(k)ij = - d2 E / d(eps_ij) d(E_k)
```

and *both* perturbations already existed — the electric field from P24 and the
strain from P26/P41. It is the Born-charge pattern (`Z* = dF/dE`, P24b) with one
coordinate changed: one `jvp` of the **stress** along the field's response,
three of them for the whole tensor, on top of a dielectric constant that was
going to be solved anyway. `PLAN.md` P50 has the phase; two things it found are
worth carrying back here.

*The cheapness filter was the wrong test for this entry.* It is
Sternheimer-scale, as this file said, and it still cost one afternoon, because
the filter that actually predicts effort is **how much of the assembly already
exists** — and here all of it did.

*The validation closed inside the package, which is what this file says to look
for.* Three routes to the same mixed derivative (a `jvp` of the energy
gradient, `zstar_eu.f90`'s contraction with a strain label, and the same
contraction with the two perturbations interchanged) agree to 6e-15 and 1.3e-7,
and **the same assembly run in the position coordinate is the Born charge**,
which is validated against `ph.x`. Elk's own number was never needed — and
`genstrain.f90` is why it would have been awkward to get: it symmetrises each
candidate strain over the crystal's group, so on a cubic crystal the only
surviving strain is the isotropic one, whose piezoelectric response is
identically zero.

**Still outstanding, and both are in `PLAN.md` P50 with what they take.** The
**relaxed-ion** constant, which is what experiment quotes and what the near-total
cancellation between the two contributions makes interesting: its ingredients are
all here (`Z*`, the `Gamma` force constants, and the internal-strain tensor as
one more `jvp`) and what it needs is a two-coordinate frozen functional
`E(eps, u)`. And the **ultrasoft or PAW** case, which needs no new term at all --
only a non-centrosymmetric, non-polar soft dataset to measure the existing one
on, since every soft crystal committed here is centrosymmetric and agrees with
zero however wrong the strain leg is.

---

## 6. The transverse spin susceptibility, and magnons

**Elk tasks 330/331** (`tddftsplr.f90`, 314 lines, on `genspchi0.f90` and
`genspfxcg.f90`).

`genspchi0` builds the Kohn-Sham response as a **4x4 matrix in the spin-density
components**,

```
chi_{ab,a'b'}(G, G', q, w) = delta rho_{ab}(G, w) / delta v_{a'b'}(G', w)
```

`tddftsplr` then Dyson-solves it against a spin-dependent `f_xc`, with the
regularised Coulomb interaction added to the `(1,1)` element **only** -- the
charge channel is the one that carries the Hartree term and the spin channels do
not -- and for a collinear ground state projects out the transverse block
(`CHI_T.OUT`, through `tfm2213`/`tfm13t`). **The pole of that block at finite `q`
is the magnon**, so this one task gives the magnon dispersion, the Stoner
continuum it decays into, and the spin-wave stiffness.

**What QE has, precisely, and the tick is not blank.**
`TDDFPT/src/lr_magnons_main.f90` is turboMagnon, with
`lr_apply_liouvillian_magnons.f90`, `lr_calc_dens_magnons.f90` and
`lr_dvpsi_magnons.f90` beside it. It is a Liouville-Lanczos solver -- the branch
`CLAUDE.md` scopes out -- and it never forms a Dyson equation in G space. There
is nothing in `PW/src` or `PP/src`. So this row is **`(✓)` for QE with the note,
`✓` for Elk**, exactly as the conductivity row is, and blank would be the kind of
wrong row the method note at the bottom of this file warns about.

**What is reused.** P37 built the sum over states and the Dyson solver; what
changes is that `chi_0` grows a 4x4 spin structure and that `q` is finite.
P19's two-sphere `k ± q/2` machinery is what supplies the states at `k + q`, and
`tddft/chi0.py:246` already names finite `q` as the missing piece by that name.
The transverse ALDA kernel is `jax.grad` of the LSDA energy in the house style
rather than a transcription of `genspfxcg`.

**Validation, and it closes inside the package twice.** This is the strongest
route in the file, and neither statement needs a second code:

- **Goldstone.** A ferromagnet's `chi^{+-}` has a pole at exactly `omega = 0` as
  `q -> 0`, because a global spin rotation costs no energy. How far the computed
  pole sits from zero measures how consistently the kernel and the ground state
  were built, and nothing else here tests that at all.
- **The frozen-magnon stiffness.** P21's `run_spiral_scan` already gives `E(q)`,
  and in the adiabatic limit `omega(q)` is proportional to `[E(q) - E(0)] / M`
  (the constant is convention-dependent and is fixed once, on the small-`q`
  limit, not per case). A spiral scan is a sequence of SCF total energies and a
  susceptibility is a sum over states plus a matrix inversion: **they share no
  machinery**, which is the P52 pairing -- nesting against `relax_spiral_q` --
  one level up.

**The ordering of the cases matters and is not the obvious one.** The natural
magnon targets are ferromagnetic metals (bcc Fe, fcc Ni) and `chi0.py` refuses
metals by name. The refusal's *reason* does not transfer unchanged -- the
spin-flip occupation factor `f_i↑ - f_j↓` does not vanish for `i = j`, so the
"the intraband term is absent entirely" argument is about the charge channel --
but establishing that is the phase's work and not an assumption to start from.
**Start on the antiferromagnetic hydrogen chain**: insulating, norm-conserving,
`nosym` already for the spiral, and it has committed `E(q)` references from P21
to check the stiffness against.

**Cost, including the working set, which is the part that bites.** The Dyson
inversion is `(4 ngrf)^2` complex per frequency: at `ngrf = 100` and 200
frequencies that is 0.5 GB held at once, four times the charge-only case in each
G index and sixteen in the pair. The frequency axis is trivially chunked and
should be from the start.

**README ticks:** `(✓)` for QE (turboMagnon, `TDDFPT/`, Liouville-Lanczos), `✓`
for Elk (tasks 330/331).

---

## 7. The magnetoelectric tensor -- **selected as the next entry to take**

**Elk task 390** (`magnetoelt.f90`, 94 lines).

```
alpha_ij = dP_i / dB_j = dM_j / dE_i
```

**This entry was missing from this file entirely until 2026-09-01**, which is
worth recording beside it: the original survey walked the task list once, and a
row can be absent rather than rejected.

**QE has nothing**, and this one was checked wider than the usual two
directories: `grep -ril magnetoelec` over the *whole* vendored tree is empty.

**Elk's route, read rather than assumed.** `magnetoelt.f90` is the expensive
pattern `piezoelt.f90` uses. It forces `spinpol = .true.` and `reducebf = 1`,
and for each Cartesian component of the external field runs **two full ground
states** at `B ∓ deltabf/2`, takes the Berry-phase polarization of each
(`polar`), brings the two into coincidence modulo `2 pi`, and finite-differences.
Six ground states for nine numbers. The atoms are held (`tshift = .false.`), so
what it produces is the **clamped-ion** tensor -- the same restriction P50
carries, and for the same reason.

**Why it belongs here.** It is the defining observable of a **type-II
multiferroic**, where the inversion symmetry that would forbid it is broken by
the magnetic order itself -- and a spin spiral is the canonical way that happens.
This package already computes the spiral (P19), relaxes its wavevector (P21) and
carries the magnetic point group (P17), so the state the effect grows out of is
here and the coupling it produces is not.

**Two routes, and they are each other's check.**

*Route A -- Elk's, adapted, and it needs no new response machinery.* P18 already
has the uniform magnetic field (`B_field`, its energy written down and its
potential `jax.grad` of that), so `dP/dB` is six SCF runs and a finite difference
exactly as Elk does it. What it needed was the **Berry-phase polarization**, and
**that is in as of P56** -- `run_polarization`,
`Calculator.get_polarization` -- built out of the k-string overlaps
`topology/links.py` already had. So this route is now six SCF runs and a
difference, with nothing left to write but the loop. That one piece pays twice,
as predicted: it is also what P50's refusal of a **polar** crystal is waiting on,
the improper-to-proper correction `delta_ki P_j - delta_ij P_k` being built
entirely from `P`.

**"Nothing left to write but the loop" is too optimistic**, and it is left in
view above rather than edited away because taking the next step is what showed
it. Two things the sentence did not allow for. `DFTSource` -- the fixed-density
source every invariant and now the polarization runs on -- **did not forward the
converged magnetic field at all**, so it rebuilt the potential from the *input*
field at full scale; wherever `reducebf` or the fixed-spin-moment scheme had
changed it, every eigenvalue was shifted by a field the SCF never converged
under, and the ME loop would have differenced two Hamiltonians that were not the
ones it thought. Fixed and refused by name, and it was the same defect
`fixed_density_states` had carried a guard against since the 2026-08-29 sweep.
And **there is no committed insulating spinor case**: every noncollinear input
here uses a smearing, where a Berry phase needs a gapped manifold.

**The spinor path itself is done and has a `pw.x` reference.** `pw.x` accepts
`noncolin` with `lberry`, and spinor silicon agrees with it on the ionic phase
(1.00000), the electronic phase (0.00000) and `MOD_TOT` -- which is **1** for a
spinor against the scalar run's 2, a spinor band holding one electron. That also
pins the spin bookkeeping the tensor depends on: the electronic phase is doubled
for `nspin = 1` and not for a spinor.

**What is left is one crystal, and it is this entry's real cost.** A linear ME
response needs magnetism *and* spin-orbit coupling *and* a gap *and* a magnetic
point group that permits the tensor -- and the last is what bites, needing
inversion and time reversal both broken. A centrosymmetric magnet gives zero
however wrong the assembly is, which is precisely the P50 trap recorded above.
Naming the crystal is the decision that starts this phase, not a detail inside
it.

**One thing P56 changed about this entry.** It said the polarization's reference
would be Elk's `polar`. It is not: `pw.x` computes a Berry-phase polarization
(`bp_c_phase.f90`, reached by `lberry`) and has a committed `test-suite/pw_berry`
directory, so QE is the primary reference and the same-basis one -- Elk drops to
the weakest tier, as everywhere else in this file. What the `pw_berry` cases
could *not* supply is a runnable case: both are PbTiO3 with Vanderbilt ultrasoft
datasets in **UPF v1**, which this package's reader refuses by name. The
reference was generated instead. **Check whether `pw.x` has a quantity before
assuming an Elk-only entry's ingredients are Elk-only too**; this file's own
method note says exactly that, and the ingredient slipped past it.

*Route B -- the house style.* One `jvp` of the magnetization along the electric
field's Sternheimer response: P50's assembly with the stress replaced by `M`. It
is blocked on the **noncollinear response** (`incdrhoscf_nc`/`set_int3_nc`, the
largest of P24's remaining refusals) and cannot be reached before that lands.

**Which coordinate is differentiable decides the route, and here it is the
reverse of Elk's situation.** There is no ground-state electric field in this
package at all: `tefield` and `lelfield` are both in `_REFUSED_SWITCHES`
(`system/builder.py`). So `dM/dE` cannot be finite-differenced the way `dP/dB`
can, and Route A is the cheaper half -- which is why it is the one to start with
even though Route B is the one that looks like every other phase here.

**Spin-orbit coupling is required, and that is physics rather than plumbing.**
Without it spin space and real space decouple, a global spin rotation is free,
and `dM/dE` vanishes identically. A collinear `nspin = 2` run is therefore not a
weak test case, it is a **zero** -- the P50 trap in a second place, where every
committed soft dataset was centrosymmetric and agreed with zero however wrong the
strain leg was. The first case must be a crystal whose tensor is not forced to
vanish by symmetry, and that must be established from the **magnetic** point
group before any number is believed.

**Validation.** The two routes are independent and are the primary check, in the
pattern P50 used (three contractions of one mixed derivative agreeing to 6e-15).
Beyond that, and cheaply: the tensor must vanish identically when either
inversion or time reversal survives, which is two extra runs; and the pattern of
surviving components must match what the magnetic point group permits, which P17
can produce rather than being asserted. Elk's own number is available and is the
weakest of the three, for the reason this file gives everywhere else -- an
all-electron polarization against a pseudopotential one.

**README ticks:** blank for QE (verified over the whole tree), `✓` for Elk (task
390).

---

## Considered and rejected

| Elk task | Why not |
|---|---|
| Electron momentum density, Compton profiles (170/171) | Looks free in a plane-wave basis, and is precisely the case where pseudization destroys the observable: the high-momentum tails are what the pseudopotential removed. An all-electron method's quantity. |
| Electric field gradient at the nuclei (115) | QE genuinely lacks it, but it needs PAW reconstruction inside the sphere and is notoriously sensitive to how that is done — a large validation burden for one number. |
| Mössbauer contact density and hyperfine field (110) | Same objection, harder: it is the density *at* the nucleus, which a pseudopotential does not have at all. |
| ELF (51/2/3) | QE has it — `PP/src/elf.f90`. |
| STM images (162) | QE has it — `PP/src/stm.f90`, Tersoff-Hamann. |
| Fermi surface plots (100/101/102) | QE has it — `PP/src/fermisurface.f90`, `fermi_velocity.f90`. |
| Wannier90 interface (550) | QE has it — `PP/src/pw2wannier90.f90`. |
| BSE (185/186/187), GW (600-640) | Out of scope per `CLAUDE.md`, and neither is cheap. |
| Electron-phonon, Eliashberg (240-285) | Needs phonons at `q != 0`, which is the outstanding two-sphere work. |
| Molecular dynamics (420/421) | A driver, not a new observable. |
| Tensor moments (400) | Cheap given `ns`, but it decomposes the DFT+U energy rather than producing a measurable. Niche. |

---

## Method note, for whoever picks one of these up

Every entry above was checked the same way and it is the only way that does not
produce a wrong claim:

- **"QE does not have it" is a `grep` over the vendored tree**, not a memory.
  Two of the original candidates died this way (ELF, STM) and one changed its
  README tick from blank to `(✓)` (the conductivity tensor, because
  `epsilon.x` has an `offdiag_calc` nobody remembered).
- **"It is cheap" is a claim about the algorithm, checked against Elk's own
  implementation.** The piezoelectric tensor looks cheap in Elk's task list and
  is a ground state per strain tensor; the nesting function looks like `N_k^2`
  and is a convolution.
- **The validation route is chosen before the phase starts**, because it is
  what decides whether the phase is worth starting. The best ones close inside
  the package (the `omega -> 0` conductivity against P47's Berry curvature);
  the next best are analytic limits; an all-electron code's floating point is
  the weakest, and for the structure factors it is not even comparable.
