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
anomalous Hall conductivity, and **a fifth in P52**, the Fermi-surface nesting
function. The rest are recorded here with their validation route, because that
is the part that decides whether a phase is worth starting.

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
