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
site-resolved angular momenta. The rest are recorded here with their validation
route, because that is the part that decides whether a phase is worth starting.

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
that forms the whole tensor, so the README column for this row would be
**partly**, not `new`. But it refuses ultrasoft outright (`okvan` →
`errore('grid_build', 'USPP are not implemented')`), computes no Kerr angle,
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

## 2. The Fermi-surface nesting function

**Elk task 105** (`nesting.f90`, 91 lines).

```
N(q) = sum_k  [ sum_n delta(eps_nk - E_F) ] [ sum_m delta(eps_m,k+q - E_F) ]
```

**QE has nothing** in the base distribution — `grep -ri nesting PP/src PW/src`
is empty. EPW has it, and EPW is out of scope.

**What is reused.** Eigenvalues from one dense NSCF and the smearing registry's
delta functions, both already there. Elk writes it as an `O(N_k^2)` double loop
over q and k; on a uniform mesh it is a **convolution**, so one FFT over the
k-grid makes the whole q-dependence essentially free. That is the one place
this would not be a transcription.

**Cost.** One NSCF on a dense grid, then milliseconds.

**Physics.** It predicts where a charge- or spin-density-wave instability will
appear — the wavevector at which a phonon will soften or a spiral will win. It
pairs directly with P21: nesting *predicts* the pitch, `relax_spiral_q` *finds*
it, and agreement between them on the hydrogen chain is a check neither can
make alone.

**Validation.** The free-electron Lindhard function analytically; the Elk binary
on Cr or a one-dimensional chain; internal agreement with `relax_spiral_q`.

**README column:** `new`.

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

**README column:** `new`. It is the only output in the package that a
diffraction experiment can be held against directly.

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

## 5. The piezoelectric tensor

**Elk task 380** (`piezoelt.f90`). Genuinely new — `pw.x` has the Berry-phase
polarization (`bp_c_phase.f90`) but nothing that differentiates it with respect
to a strain.

**Elk's route is the expensive one:** a finite difference of the Berry-phase
polarization over re-converged strained cells, one full ground state per strain
tensor, with a `2 pi` branch-tracking fix-up between them.

**The cheap route here.** The clamped-ion piezoelectric constant is a **mixed
second derivative**,

```
e_ij,k = - d2 E / d(eps_ij) d(E_k)
```

and *both* perturbations already exist — the electric field from P24 and the
strain from P26/P41. It is the Born-charge pattern (`Z* = dF/dE`, P24b) with one
coordinate changed, so the assembly is known.

**Why it is listed as the bigger option.** It is a Sternheimer-scale
computation, not an NSCF-scale one, so it fails the cheapness filter this file
applies. It is here because the *implementation* cost is low relative to what it
produces, and because P44's finding bounds what to expect: the strain
coordinate's higher derivatives are validated on norm-conserving and refused on
ultrasoft/PAW, and a piezoelectric tensor would inherit that line.

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
| Second-harmonic generation (125) | Not cheap, and `chi^(2)` is already refused by name with the missing term identified (P35: the `<u_i|r_k|u_j>` piece, worth 42%). |
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
  README column from `new` to `partly` (the conductivity tensor, because
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
