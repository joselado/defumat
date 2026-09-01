# Non-linear response: what is here, what is next, and what each one costs

A roadmap for the third-order-and-beyond half of `pypresso/response/`, written after
P35 so that the session which picks any of it up does not re-derive the constraints.
`PLAN.md` is the phase tracker and stays the authority on what is *done*; this file is
about what is *reachable*, in what order, and at what price.

The organising fact is stated once and applies to everything below:

> **The 2n+1 theorem with first-order wavefunctions reaches third derivatives and
> stops.** (P53 added the qualification this whole file needed: that is a statement
> about the **Sternheimer stack**. A *sum over states* has no field-source problem
> and reaches the second-order optical family — shift current, injection current,
> `chi^(2)` — with none of Tier B's missing term. §7 and §8 carry it.) Every quantity in Tier A is a third derivative and needs nothing new.
> Everything in Tier B is also a third derivative but has an *electric field* in a
> place this code cannot differentiate. Tier C and D need either a frequency or a
> second-order wavefunction, which are two different new machines.

---

## 1. Where the code stands

| built | what it gives | phase |
|---|---|---|
| velocity operator, one `jvp` of `H(k)` | `dH/dk`, hence `[H, r]` | P24 |
| Sternheimer solve, projected CG | first-order `dpsi` for any perturbation | P24 |
| field response `u`, `drho`, `b = P_c r|psi>` | `epsilon`, `Z*` | P24, P24b |
| displacement response | the dynamical matrix at `Gamma` | P25, P28 |
| strain response | elastic constants, `d(chi)/d(strain)` | P26 |
| the variational second-order energy `F_ij` | the object every third derivative differentiates | P26 |
| `d(eps)/d(tau)` | **Raman tensors** | P35 |
| the group average at any rank (`symmatrix3`/`symtensor3`) | a wedge sum with 3 or 4 free cartesian labels | P36 |
| the mode projection of `dchi/dtau` and `Z*` | **Raman and IR spectra** | P36 |
| `<u_mk|S|u_nk'>` between neighbouring k, with the zone-edge `G` shift | Berry phases, Chern numbers, Wilson loops | P16 |

Two of those are load-bearing in ways that are easy to miss. `F_ij` is a *functional*
whose stationary value is `epsilon`, so differentiating it once more along any tangent
is a third derivative for free — that is what P26 and P35 both are. And P16's
k-to-k overlap machinery is the discretised Berry-phase apparatus, which is exactly what
the "PEAD" formulation of §3.1 needs and which nothing in `response/` currently uses.

## 2. Three constraints that shape the whole roadmap

**2.1 The field enters only through the source term.** `H` is built from `rho`; there
is no `dH/dE` anywhere. The position operator exists only as `b = P_c r|psi>`, obtained
from a commutator solve that uses `psi`'s own eigenvalue and therefore does not apply to
a general first-order state. So the 2n+1 term with the perturbing operator between two
first-order wavefunctions, `<u_i|r_k|u_j>`, has nothing to build it from. This is why
P35 delivers the Raman tensor and refuses `chi^(2)`: a *displacement* tangent carries
its `dH/d(tau)` through `at_positions`, and a field tangent carries nothing. **It is
42% of the answer**, measured on the displacement counterpart, and no symmetry check
sees its absence (P35).

**2.2 The reference is broken above second order.** The vendored `ph.x` 7.5's
`lraman`/`elop` branch does not reproduce QE's own committed `PHonon/examples/example05`
(v6.0, 2016) and fails its own internal consistency check. Anything in Tiers B–D that
would be validated against `ph.x`'s third-order output must instead be validated the way
P35 was — see §6. Second-order `ph.x` (`epsil`, `trans`, `zeu`) is *fine* and remains the
reference for everything linear.

**2.3 Third-order quantities converge slowly in k, and the formulation decides how
slowly.** Veithen, Gonze and Ghosez ([arXiv:cond-mat/0409067](https://arxiv.org/abs/cond-mat/0409067),
Fig. 1) measure `d123` of AlAs on `n x n x n` grids in both formulations: discretising
the Brillouin-zone sum *after* the perturbation expansion (DAPE — the analytic `d/dk`
route, which is what the velocity operator and the Sternheimer solves here are) is still
climbing at `n = 20` where discretising *before* it (PEAD — the finite-difference Berry
phase) is flat by `n = 6`. This is not an accuracy claim about either; it is a statement
about which one a converged number is cheap in. It bears directly on §3.1.

---

## 3. Tier A — third derivatives that need nothing new

These are assemblies. Each is days rather than a phase, and each adds a README row.

### 3.1 Raman and infrared spectra — **DONE (P36)**

`pypresso/response/spectra.py`, `vibrational_spectrum`. Placzek's two invariants of the
per-mode Raman tensor and the mode projection of `Z*`, in the units `dynmat.x` prints them.

`RamanIR` turned out to be more than a transcription check: it is **the only working QE
reference above second order**, because it is post-processing and never enters the
`lraman` branch. `pypresso/io/dynmat.py` writes the `fildyn` `ph.x` would have written and
the test runs the vendored binary on our tensors — every digit either code prints, on AlAs
(353.25 cm⁻¹, Raman 446.8854) and on silicon (519.20 cm⁻¹, Raman 9815.5635, IR 0.0000).

Two things came out of it that were not in the plan:

* **A degenerate multiplet is comparable only as a sum.** Per-mode `alpha`, `beta²` and the
  depolarisation ratio are not invariant under the orthogonal mixing the eigensolver is free
  to apply inside one; the multiplet's sum is. Silicon's acoustic triplet prints
  0.3544/0.7163/0.4065 here against `dynmat.x`'s 0.5873/0.2446/0.7264 — both meaningless,
  on modes whose activity is 0.0000. This is rule D4 arriving in the *output* rather than in
  a solver.
* **The displacement response is the expensive half of a Raman tensor and of a dynamical
  matrix, and it is the same object.** `raman_tensors(keep_internals=True)` hands it over
  and `dynamical_matrix(response=...)` takes it, so the phonons cost 1-2 s instead of 50.

**Both of the two things this section named as not done are done** (P55): the
non-analytic LO-TO term (`rigid.f90`'s `nonanal`) and the mode-resolved ionic permittivity
(`polar_mode_permittivity`), which needed only `Z*` and `eps` and were an afternoon each.
They were taken first because of §6 rather than because of §8: `dynmat.x` applies both
itself, so the transcription check existed already.

**What they added to §5's list of checks is a case where the reference agrees with the
wrong answer.** Both codes read the same `Z*` off the same file, so `dynmat.x` reproduces
every digit of a splitting computed from Born charges that violate `sum_a Z*_a = 0` — and
at `ecutwfc = 10` AlAs's violate it by -1.257, which charges the crystal and lifts a
*longitudinal acoustic* mode from 1.8 to 33.8 cm^-1. The Lyddane-Sachs-Teller relation
separates them: violated by 1.6e-3 with the raw charges and by **5.0e-11** with the
neutralised ones. It is §5's second bullet in a new place — an identity that ties two
independently computed quantities, where a comparison against another code's arithmetic
cannot see a shared input being wrong.

### 3.2 Grüneisen parameters and quasi-harmonic thermal expansion

**What.** `gamma_m = -(V/omega_m) d(omega_m)/dV`: how a mode stiffens under compression,
and through it the thermal expansion coefficient.

**How.** One more `jvp` of the dynamical matrix along the **strain** tangent P26 already
builds — the same relationship P35 has to P26, with the roles of the two geometry
variables swapped. Mode Grüneisens come from the eigenvector projection of `dD/d(strain)`.

**Validation.** Finite-difference the phonon frequencies over re-converged strained cells;
silicon's negative `gamma` for the transverse acoustic modes is the physics check, and it
is one no assembly error is likely to reproduce by accident. At `Gamma` only until
`q != 0` lands, which limits it to optical modes — say so rather than quoting a thermal
expansion from three modes.

**Cost. Not nearly free, and the estimate above is the one this file got wrong** --
corrected by reading the code rather than by attempting it, before P55 was chosen instead.
"One more `jvp` of the dynamical matrix" is not available, because **the dynamical matrix
here is not assembled variationally**: `_force_constants` computes it as one `jvp` of the
*force's gradient* along a tangent carrying `(u, dpsi_u, drho_u)`, and differentiating that
along a strain differentiates the tangents too -- `d(dpsi_u)/d(eps)` is the mixed
second-order response, which does not exist and is a Sternheimer solve of its own.

The 2n+1 route is the right one and it is what P26 and P35 both are, but it needs an
object neither of them built: **a second-order energy functional in the *displacement*
coordinate**. `electrostriction._second_order_energy_at` is that functional for the
*field* -- and it is hardcoded to the field's three components (`for axis in range(3)`,
thirteen times) and repairs its wedge sum as a **polar vector**, where a displacement
perturbation has `3 nat` components and symmetrises as `symdvscf` does. So the phase is:

1. generalise the functional's perturbation axis and hand it its symmetriser, with P26's
   and P35's committed numbers as the guard that nothing moved;
2. hand it the displacement's bare perturbation `b`, which is
   `phonon._bare_displacements` and already exists -- and, unlike the field's `b`, needs
   **no extra Sternheimer solve**, since `dV_bare/du` is an explicit function of geometry;
3. put the **frozen** second derivative (`dynmat0`, `d2ionq`) inside the same functional,
   which the field version has no counterpart to because a field does not move ions. Its
   strain derivative is then a triple-nested autodiff of the frozen energy;
4. one `jvp` along the strain tangent `strain_response` already builds.

The check that step 1-3 are right before any of step 4 is worth naming, because it is free:
the functional's **stationary value must reproduce the electronic part of the force
constants** P25 already computes.

Two things that stay true from the original entry: the validation is a finite difference of
the frequencies over re-converged strained cells, which is decisive and needs no `ph.x`;
and at `Gamma` the negative-`gamma` physics check is unavailable, since it lives on the
transverse acoustic modes.

### 3.3 Third-order force constants at `Gamma`

**What.** `d^3 E/d(tau)^3` — the cubic anharmonicity: mode-mixing coefficients, the
frequency shift of an excited mode, and (once `q != 0` exists) phonon linewidths and
lattice thermal conductivity.

**How.** One more `jvp` of the dynamical matrix along a *displacement* tangent, which
already exists. The `3 nat` tangents make it `(3 nat)^2` `jvp`s, so it is the first item
here whose cost is quadratic in the cell — bound it by symmetry or by the irreducible set
before running anything larger than eight atoms.

**Validation.** Finite differences of the force constants over displaced geometries
(P25's route, one order up), and the permutation symmetry of the rank-3 object, which for
*displacements* is a real check rather than a blind one (§5).

**Note.** QE reaches this only through `d3q.x`/`thirdorder`-style external tools; there is
no vendored counterpart, so this is a `new` row.

---

## 4. Tier B — the third derivatives with a field in the wrong place

All of these are blocked on the same missing object, `<u_i|r_k|u_j>`, and unblocking it
unblocks all of them at once. **This is the single highest-value item in the file.**

### 4.1 The two routes to the missing term

| | route | what it needs | k-convergence |
|---|---|---|---|
| (a) | **second-order response** — QE's `dvpsi_e2` + `solve_e2` | a second self-consistent response loop, driven by a source quadratic in the first-order states | DAPE: slow (§2.3) |
| (b) | **PEAD / discretised Berry phase** — Nunes and Gonze | the position operator as a finite difference in k, i.e. **P16's overlap machinery** | fast (§2.3) |

Route (b) deserves the first look and this document exists partly to say so. It needs
`<u_mk|u_nk+b>` with the Miller-index alignment and the zone-edge `G` shift — which
`pypresso/topology/` already implements and validates against exactly-integer Chern
numbers. It converges faster in k, it is what ABINIT actually ships, and it avoids
writing a second response loop. Its cost is that the field perturbation stops being an
analytic derivative and becomes a finite difference over the k-mesh, which is a different
kind of error to characterise (and one this project has never had to). Route (a) is the
literal transcription and is the safer estimate if (b) turns out to fight the
Sternheimer solver's gauge.

### 4.2 What falls out once it exists

* **`chi^(2)` and the electro-optic tensor** — P35's refusal lifted. Immediately
  comparable to QE v6.0's committed `example05` numbers (40.4578 and -0.78497) and to
  Veithen's published table (AlAs `d123 = 35` pm/V, AlP 21).
* **The full Pockels tensor `r_ijk`** — electronic (`chi^(2)`) plus the ionic term, which
  is the Raman tensors divided by the mode frequencies and contracted with `Z*`. Every
  piece of the ionic half exists *today*; only the electronic half is missing.
* **Second-order Born charges**, `d(Z*)/dE`, and the rest of the field-field-displacement
  family.

### 4.3 The infrastructure item that goes with it — **DONE (P36)**

**The rank-3 symmetriser is in**, and at any rank:
`pypresso.system.symmetry.symmetrize_cartesian_tensor` and
`symmetrize_atom_cartesian_tensor`. P26 and P35 both run on a symmetry-reduced wedge now —
AlAs to 3.3e-9 of its closed grid on 8 k-points instead of 64 (8.7e-14 when the
phase landed; the residue is the two SCFs' convergence footprint amplified by a third
derivative, and it moved with the mixer normalisation in `a351005` -- see `CLAUDE.md`),
silicon's rank-4
elasto-optic tensor to 7.9e-14 — at roughly half the cost. Rank 4 has no QE counterpart,
because QE does not compute that tensor.

**It was not forty lines, and the extra part is the thing to carry forward.** The group
average completes a wedge sum only where the tensor is a *linear* Brillouin-zone sum of a
covariant per-k quantity, because then `T_true = (1/N) Σ_S R⊗R⊗R T_wedge` follows term by
term. The screening term of the second-order energy is **quadratic** in a k-sum
(`∫ drho_i K drho_j`), and a product of two incomplete sums is not the incomplete version
of the product. The repair is a split — the *value* of each density-response factor must be
the full-zone object and its *derivative* must stay the raw wedge sum, after which
`∫ X_i K Y_jc` averages correctly by a change of variables in the integral. Symmetrising the
derivative too, which is the obvious thing to write, is **worse than doing nothing**: it puts
an extra independent group average on the displacement label.

Sizes, in the order they were measured: no average, -3.195188; the wrong repair, 3.009778 on
the other atom against 3.119166; the right one, exact. **Every symmetry check passed all
three** — zincblende form, permutation symmetry, cubic — and the translational sum rule
(2.8e-4 / 1.0e-2 / 3.3e-2) is what separated them. §5's first two bullets, again.

**One refusal did not lift**: the clamped-ion elastic constants, because their functional has
to build its own density and symmetrises it as a *scalar*, inside the chain rule. No average
applied afterwards undoes that. `electrostriction(elastic=False)` is the wedge route.

## 5. What P35 learned about checking any of this

Carry these into every item above; they were paid for once.

* **Kleinman/permutation symmetry is a weak check.** The static third derivative of one
  scalar with respect to three components of the same field is symmetric under every
  permutation, and P35's *incomplete* tensor satisfies it to 2.5e-13 — because the
  omitted term is symmetric too. Same for vanishing in a centrosymmetric crystal and for
  coming out in the exact crystal class. **None of the three sees a missing symmetric
  term.** For a *displacement* rank-3 object the permutation check is stronger, because
  the three labels are not interchangeable a priori.
* **Sum rules are the checks that bite.** The translational sum rule on the Raman tensors
  (`sum_atoms = 0`) held to 2.8e-4 here and is violated by 43% by the broken `ph.x` — it
  is what found the regression. Every third derivative has one: the acoustic sum rule for
  the anharmonic constants, charge neutrality for `Z*`.
* **The envelope theorem's hypothesis is fragile and its failure is invisible.** P26's
  frozen `u` had to be projected onto the *moving* conduction manifold, worth 2%, and it
  survived every value check. A diverged first-order response consumed by a third
  derivative is wrong at **first** order — `require_converged_responses` applies to every
  new tangent.
* **Zero the tangent to size a term.** The cheapest diagnostic in P35 was recomputing the
  Raman tensor with the geometry tangent zeroed, which put a number (42%) on what the
  field derivative was missing. Any new assembly can be interrogated the same way.

---

## 6. Validation without a working `ph.x`

In rough order of how much they are worth:

1. **Finite-difference a second derivative.** `epsilon`, the force constants and the
   stress are all computed from scratch at any geometry, so their derivatives have a
   reference that shares only the linear response. This is what validated P26 and P35
   (1.0e-5), and it works for every item in Tier A.
2. **QE v6.0's committed example outputs.** The numbers in
   `PHonon/examples/example05/reference/` predate the regression and agree with the
   literature; they are usable as *numbers* even though the binary that produced them is
   not available. Cite them as v6.0.
3. **Published tables.** Veithen et al. for `chi^(2)`, Raman efficiencies and electro-optic
   coefficients of AlAs, AlP, LiNbO3, BaTiO3, PbTiO3.
4. **Model Hamiltonians with known answers** — P16's route, and the right one for the
   geometric quantities of §7 where no plane-wave reference exists.
5. **Sum rules and symmetry**, with §5's caveat about what they cannot see.

---

## 7. Tier C and D — the two genuinely new machines

Listed so the boundary is explicit, not because either is next.

**Frequency dependence (a new solver).** `alpha(omega)` and resonant Raman need the
Sternheimer operator at `H - eps +- omega`, which is **indefinite**, and non-Hermitian
once a broadening `eta` is added.

*(This paragraph named `chi^(2)(-2 omega; omega, omega)` as well until P54, and that
was the same conflation §8 records for the shift current: it is a statement about the
**Sternheimer** route to SHG. The **sum over states** reaches the whole frequency axis
with no solver at all — it has the eigenvalues, so the resonant denominators are
arithmetic — and that is what `pypresso/response/shg.py` is. What still needs the
indefinite solver is a *truncation-free* second-order response, which is the same
`|u_m^{;a}>` §7's shift-current entry identifies below.)* The projected
CG breaks; a shifted/complex solver (BiCGStab, or QE's own `solve_e_fpol.f90`) is new
machinery. QE's `fpol` (`polariz.f90`, `PHonon/examples/example09`) is the validation
target for the linear frequency-dependent case and **should be checked for the same
regression before being trusted**. The LDA gap error stops being a scale factor here and
starts moving resonance positions.

**`chi^(3)` and beyond (second-order wavefunctions).** A fourth derivative is out of
reach of 2n+1 with first-order wavefunctions; it needs `dpsi^(2)`, i.e. differentiating
through the linear solve. That is the same object route (a) of §4.1 builds, so if that
route is taken, `chi^(3)` becomes reachable rather than impossible.

**The geometric family — and it is the best fit for this codebase.** Shift current and
the bulk photovoltaic effect, the Berry-curvature dipole and the non-linear Hall effect
are second-order responses governed by quantum geometry rather than by a self-consistent
loop. **QE has none of it** (`grep` finds neither term in `PW/src`, `PP/src` or
`PHonon`; the only implementation in the tarball is the bundled `external/wannier90`),
the published route is Wannier interpolation, and the two ingredients here are the
velocity operator (one `jvp` of `H(k)`) and P16's Berry connection. It is the most cited
corner of non-linear optics right now
([2507.00864](https://arxiv.org/abs/2507.00864), [2412.16477](https://arxiv.org/abs/2412.16477)),
it is validated against model Hamiltonians rather than against another code, and it is
a `new` row rather than a reimplementation.

**The shift current half of it is now in** — `pypresso/response/photocurrent.py`,
`run_shift_current`, `Calculator.get_shift_current` — and the thing to carry forward is
*why it was reachable while §4 stays refused*. §2.1 is a statement about the **Sternheimer
stack**: the field enters only through a source term, so a 2n+1 expression with two field
labels has nothing to build `<u_i|r_k|u_j>` from. A **sum over states** has no such
problem. It needs the interband dipole, which is `-i v_nm / w_nm` — *exact* for an
eigenstate of the full `H(k)`, no truncation, and the reason a plane-wave code has no
counterpart to Wannier90's `AA_R` (that array exists only to undo the Wannier gauge). So
the whole second-order optical family is reachable by the route P37 already took for
absorption, and §4's refusal does not reach it. The same is true of `chi^(2)(-2w; w, w)`,
which `ELK-FEATURES.md` currently rejects by pointing at §4 — that row conflates the two
routes and is wrong.

**What it cost was one new operator and one piece of algebra.** The operator is
`w^ab_nm = <n|d^2H/dk_a dk_b|m>`, one `jvp` of the velocity operator's `jvp`
(`VelocityOperator.apply_second`) — forward over forward, matching a central difference
of `dH/dk` to **8.9e-9**, symmetric in `a <-> b` to 0.0 and Hermitian to 2e-15. The
algebra is that the sum rule's `sum_{p != n,m}` collapses to `[v^c, g^a]_nm - g^a_nm
D^c_nm` with `g^a = v^a/w_nm`, because `1/w_pm` already kills `p = m` and `1/w_np` kills
`p = n`. That is what makes the assembly two matrix products instead of a loop over
triples, and it has its own unit test on random matrices.

**Three findings, and two of them are older findings arriving again.**

* **The truncation is the real cost and it is severe.** The intermediate sum over `p` is
  an *identity* only over a complete basis; truncated it is a few per cent and no
  symmetry check sees it. Measured on AlAs on a frozen sphere of 158 plane waves, where
  the band set can be run all the way out: 6.0e-2 at 20 bands, 4.8e-2 at 80, 4.3e-2 at
  120 — and **1.8e-4 at 158**. IATS18 avoid this with `k.p` inside a closed Wannier
  subspace and there is no closed subspace here. **The route that removes it is
  identified**: the two sums are exactly `-<u_n|dH/dk_c|u_m^{;a}> - <u_n^{;a}|dH/dk_c|u_m>`
  minus the same `D^c` term, which is a Sternheimer solve in place of a spectral sum —
  but `|u_m^{;a}>` is a *per-band* derivative, so the operator to invert is `H - e_m` on
  the complement of one band, **indefinite** whenever anything lies below `e_m`. It is
  P48's objection to getting an individual band's `|dpsi/dk>` that way, and lifting it
  needs the indefinite solver of §7.
* **A stencil must not straddle a change of basis** — P48's finding, one derivative up
  and in the *arm* rather than the centre. The validating finite difference rebuilt its
  sphere per k-point and got 158 plane waves at `k` and `k - delta` but **157** at
  `k + delta` on a generic AlAs point; that fixed variational offset divided by
  `2 delta` made the disagreement *grow* as the step shrank. Freezing the sphere across
  the stencil is both the cure and the correct comparison, since the analytic route is a
  frozen-sphere derivative by construction.
* **The degeneracy threshold is the broadening, not 1e-8.** A linear conductivity carries
  `1/w_nm` once; this carries it three times. At the `W` points of a 4x4x4 fcc mesh AlAs
  has a band pair **9.18e-5 Ry** apart — far above `dielectric.f90`'s 1e-8, far below
  anything resolvable — and `|r^{c;a}|` there is **9.2e6 bohr^2** against a mesh median
  of 4.4e3, which put a spike in the spectrum that moved by two orders of magnitude
  between a 4x4x4, a 6x6x6 and an 8x8x8 grid. Elk's `nonlinopt.f90` guards it with the
  smearing width and that is the rule adopted. **The tensor was exactly `-43m`
  throughout**, to five figures: §5's first bullet, for the third time.

**The overall scale is pinned against the literature and the convention is
declared.** Every other check is blind to a constant, which is P50's trap here.
AlAs's first peak converges to **35.0 uA/V^2 at 4.17 eV** (about 1% in both the
cutoff and the band count), where published
calculations across the fourteen III-V and II-VI zincblende semiconductors span
14 (CdSe, smallest) to 83 (AlSb, largest) and find the aluminium compounds the
strongest of the family: a factor of two either way breaks that *ordering*
rather than moving the number. And the convention is IATS18's
`j = 2 sigma E E`, quoted from its text -- Cook, Fregoso, de Juan, Coh and Moore
(arXiv:1507.08677) write `J = sigma E E` and report numbers twice as large for
the same physics, so any comparison against a paper that does not state its
normalisation is worthless to better than a factor of two. This matters for
everything else in this corner: an injection current and a `chi^(2)` will have
the same ambiguity, and the same two checks answer it.

**The `chi^(2)(-2w; w, w)` half of it is now in too** — `pypresso/response/shg.py`,
`run_shg`, `Calculator.get_shg` (P54). It is the same sum over states with two resonant
denominators in place of a smeared delta, it needs *less* than the shift current did (the
triple sum over the intermediate state is the sum-rule expansion of the generalised
derivative, so `second_matrix_elements` never appears), and it is the one thing in this
file with a **real reference implementation**: Elk's task 125, agreeing to 0.5% on the
resonance position, 7% on the peak height and 11% on the static value with the basis shown
converged. The finding to carry is that **`Delta^a` needs the multiplet average** — it is
built from the diagonal of an operator, which rule D4 says is not defined inside a
degenerate multiplet, and on silicon that is worth four orders of magnitude with no
symmetry check seeing it. §5's first bullet, for the fourth time.

**Still open in this corner**, in the order they are cheap: the **injection current**
(CPGE), which needs *nothing* new — the same `v_nm` and the diagonal `Delta^a` — but is
identically zero in every non-gyrotropic class, `-43m` included, so it has no committed
cell to run on and its decisive check is the quantized Weyl-node trace on a
`ModelStates` model; and the **Berry-curvature dipole**, which is a
Fermi-surface integral of a k-derivative of an already-singular quantity and is the one
entry here a direct plane-wave code cannot honestly converge.

---

## 8. Suggested order

1. ~~**Raman and IR spectra** (§3.1)~~ — **done, P36.**
2. ~~**The rank-3 symmetriser** (§4.3)~~ — **done, P36**, and it was done first, because its
   check is decisive today (the wedge must reproduce P35's committed closed-grid numbers)
   and it halves the cost of every third-derivative run after it.
3. ~~**The LO-TO term and the ionic permittivity** (§3.1)~~ — **done, P55**, and taken
   out of order for the reason §6 gives: `dynmat.x` computes both, so the check was
   already there. See §3.1.
4. **Grüneisen parameters** (§3.2) — still owed, and **not** nearly free: §3.2 carries the
   corrected estimate and the four steps it actually needs. Note that at `Gamma` the
   negative-`gamma` physics check is unavailable: it lives on the *transverse acoustic*
   modes, which are zero at `Gamma`. Until `q != 0` the only check is a finite difference of
   the frequencies over re-converged strained cells.
5. **The missing `<u_i|r_k|u_j>` term** (§4) — the real phase, and the one that lifts a
   refusal rather than adding a quantity. Look at the PEAD route first, but note what §4.1
   understates: P16 supplies the *primitive* `<u_mk|S|u_nk+b>`, not the assembly. Individual
   matrix elements from a finite difference in k are gauge-dependent — the gauge cancels only
   in the full discretised Berry-phase expression — so what gets written is
   Veithen-Gonze-Ghosez's PEAD third-order formulas, not a substitution into the existing
   one. Route (a) is the fallback and it buys something PEAD does not: `dpsi^(2)` is what
   makes `chi^(3)` reachable at all (§7).
6. **Third-order force constants** (§3.3) — after `q != 0` phonons, which is where their
   payoff is.
7. ~~Then choose between **frequency dependence** and **the geometric family** (§7)~~ —
   the geometric family was chosen, and its **shift current** half is done (P53) and
   its **`chi^(2)(-2w; w, w)`** half with it (P54). What those phases established is
   that the boundary drawn in §2.1 is a boundary of the *Sternheimer stack* and not of
   the physics: a sum over states reaches the whole second-order optical family without
   `<u_i|r_k|u_j>`, so §4's refusal never applied to it. **P35's refusal is not lifted
   by either** — the 2n+1 route to a truncation-free static `chi^(2)` and to the
   electro-optic tensor still lacks the same term. Next in the same corner, cheapest
   first: the **injection current** (nothing new but a gyrotropic case), the
   **Berry-curvature dipole**, and then **frequency dependence** in the Sternheimer
   sense, which is still the other machine.
