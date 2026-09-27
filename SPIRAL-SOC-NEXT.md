# Spin spirals with spin-orbit coupling, as a perturbation on the generalized Bloch theorem

A plan recorded on 2026-09-26. **Steps 1 and 2 of the build order are done** (2026-09-27,
`PLAN.md` P123): the spiral runs on a fully-relativistic dataset at `soc_scale = 0`, and
`workflows/spiral_soc.py:spiral_spin_orbit_energy` gives the coupling's first-order energy.
The derivation the build order asked for first is the section "What the derivation found",
below the plan, and it changes step 3: the charge harmonic the plan names as step 3's number
is second order, not first. We will now see what the idea is, why spin-orbit coupling breaks
the construction it starts from and what that does to a perturbation expansion, what in this
code it builds on, how it differs from the ultracell (and where the two should agree), and
the order in which to build and measure it. The sections up to the build order are the plan
as it was written, and the statements in them that the derivation corrected are marked.

## The idea

A spin spiral without spin-orbit coupling is exact in the minimal unit cell. Translating by
a lattice vector `R` and turning every spin by `q . R` about the spiral axis `n` is then a
symmetry, and that is the generalized Bloch theorem (Sandratskii): the state at `k` is a
spinor whose component along `+n` lives at `k + q/2` and whose component along `-n` lives
at `k - q/2`. This code has it (`spiral_q`, `PLAN.md` P19, Elk's `vqlss`), with `q` relaxed
by `jax.grad` (P21) and ultrasoft and PAW datasets included (P95, P96). The cost is that of
the unit cell whatever the period, and `q` can be incommensurate.

The proposal is to add spin-orbit coupling **afterwards, as a perturbation** on those
states:

1. converge the spiral without the coupling, in the minimal cell;
2. take the coupling's first-order energy at the frozen spiral states, which is where the
   chirality of the spiral first appears, the Dzyaloshinskii-Moriya energy;
3. take the first-order change of the wavefunctions, which mixes each spiral state with
   states at other wavevectors, and from it the change of the density: the charge and
   magnetization harmonics of the texture in the large supercell, and for an
   incommensurate `q` in a cell that does not exist;
4. if needed, make that response self-consistent.

We believe step 2 is what FLEUR, the Jülich FLAPW code, does for the Dzyaloshinskii-Moriya
interaction: Heide, Bihlmayer and Blügel, Physica B 404, 2678 (2009), on top of the
generalized Bloch theorem as implemented there (Kurz, Förster, Nordström, Bihlmayer and
Blügel, PRB 69, 024415 (2004)). **Both references are from memory and are to be checked**,
together with whether FLEUR goes beyond first order in the energy and whether it
reconstructs the wavefunctions or the density at all, which is the part of the proposal
that is new if it does not.

## Why the coupling breaks the theorem, and what that does to the expansion

Spin-orbit coupling ties the spin to the lattice, so turning every spin by `q . R` is no
longer free, and the spiral with the coupling is not a single-`q` state. This code refuses
the combination for that reason, permanently (`system/builder.py:_spiral_q`), and Elk
refuses it too (`init0.f90` sets `spinorb = .false.` when `spinsprl`).

In the spinor layout the spiral uses, the break has a simple form, and it is worth deriving
properly first. Write the coupling's operator with the spiral axis `n` as the spin
quantization axis and split it into its `sigma_n` part and its `sigma_+-` parts.

- The `sigma_n` part keeps each spinor component where it is: the `+n` component at
  `k + q/2` stays there, and so does the `-n` one. It connects the spiral state at `k` to
  itself, so **it is the only part with a first-order energy**.
- The `sigma_+` part turns a `-n` component at `k - q/2` into a `+n` one at the same
  wavevector, which is the `+n` component of the spiral state at `k - q`. So the `sigma_+-`
  parts connect the state at `k` to the states at `k -+ q`. They have no first-order
  energy, and **they are what first-order perturbation theory mixes in**.

So the perturbed state is a ladder over `k + m q`, with `|m|` growing by one per order in
the coupling. The density picks up harmonics at multiples of `q`, and which harmonic
appears at which order is the first thing to derive. (Derived below: for a flat spiral the
charge has even harmonics only, at every order, so the charge at `2q` is second order and
the charge at `q` is zero.) The test that decides it exists
already: NiBr2's charge at `2q` against `q` is 15 in Elk's converged supercell, and the
three-cell ultracell on the Kramers-closed basis (P121) puts both near 1e-8 at
`conv_thr = 1e-8`. The expectation, to be checked, is that the perturbation is controlled
when the coupling is small against the exchange splitting and against the band separations
across `q`, and fails where a state at `k` and one at `k + q` are close in energy, which
is exactly where a Fermi surface nests at `q`.

## What in this code it builds on

- **The spiral itself** (P19, P21, P95, P96): the spinor on two spheres, the displaced
  augmentation table `Q_ij(G - q)` for the transverse block, `dE/dq` by `jax.grad`.
- **The first-order spin-orbit term at frozen states** (`workflows/anisotropy.py:
  frozen_expectation`, P58, P120). Step 2 is that function on a spiral instead of a
  collinear magnet. P120's lesson carries over unchanged: **the first-order operator is the
  coupled Hamiltonian minus the reduced one at the same frozen potential, entry by entry**
  (bare `dvan_so`, `qq_so`, and `newd_so`'s sandwich of the augmentation integrals). Writing
  it as a list of terms left one out from P58 to P120, and the structural test that holds
  it against the two Hamiltonians is the template for the spiral's version. `soc_scale`
  blends all of them linearly, so the first-order term is `dF/d(soc_scale)` at 0.
- **The `soc_scale` dial** (P58, P115, P117), whose reduced Hamiltonian at 0 is the
  variational coupling-free functional of a fully-relativistic dataset. Step 1 would run on
  it with the spiral, which the current refusal would have to allow at `soc_scale = 0`
  alone. On PAW the one-centre small component carries the dial too (P117), and a first-order
  operator on PAW needs it; `frozen_expectation` refuses PAW for that reason today.
- **The Sternheimer stack** (P24 onward) for step 3, which **refuses spin spirals today**
  (`CLAUDE.md`, the linear-response entry): the first-order state at `k` has components on
  the spheres of `k +- q`, so the projector and the solve need the spiral's two-sphere
  layout. That is the largest piece of new machinery in the plan.
- **The velocity operator and the magnon code** (P63), for how a `q`-shifted matrix element
  between two k-points is already built here.

## Is it different from the ultracell?

In part it is the same physics reached from a different starting point, and the two should
agree where both apply. That agreement is the check the plan needs.

| | ultracell (P88 on, Kramers-closed since P121) | spiral plus perturbation |
|---|---|---|
| expanded around | the uniform (collinear or noncollinear) unit cell's states at `k0 + Q` | the self-consistent spiral's states at `k + m q` |
| what is exact | nothing beyond the basis truncation; the coupling is in the frozen states in full | the spiral without the coupling; the coupling to a chosen order |
| `q` | commensurate, `q = b/N`, the basis `N nbnd` per `k0` | any `q`, incommensurate included; the basis is the ladder length times `nbnd` |
| self-consistency | full, inside the frozen basis | none at first order; linear response if iterated |
| coupling strength | any | small against the exchange splitting and the gaps across `q` (to quantify) |
| what it gives | the converged texture, its energy, STM and transport | the Dzyaloshinskii-Moriya energy `E1(q)`, then the first-order texture and density |

**The two are closer than the table makes them look.** Restrict the ultracell's `Q` list to
`{m q : |m| <= M}` and replace its frozen states by the spiral's own states at `k + m q`,
and the ultracell's matrix, with the coupling as the "difference potential", is the ladder
of the section above solved by diagonalisation rather than by perturbation. That is a third
route, a **spiral ultracell**. It is non-perturbative in the coupling, variational in the
ladder, and needs no commensurate cell, because the ladder is not a folded grid. The
Kramers closure of P121 has a counterpart there to think through: the spiral's states
already contain both spinor directions at every point of the texture, so the lean P121
removed may not arise at all.

So the genuinely new capabilities are an **incommensurate** `q` and the
**Dzyaloshinskii-Moriya energy as a first-order quantity**, antisymmetric in `q`, taken in
the unit cell. At a commensurate `q` with weak coupling, the ultracell on the Kramers-closed
basis is the reference the new route should reproduce.

## The order to build it in, and the number each step has to produce

1. **Derive, before any code**: the coupling's matrix elements in the spiral's two-sphere
   layout, which blocks move `k` by `+-q`, which density harmonic appears at which order,
   and the `sigma_n` projection of `dvan_so`, `qq_so` and the `newd_so` sandwich for an
   arbitrary axis `n`. Check FLEUR's documentation and the two papers above for their
   version of each.
2. **The first-order energy `E1(q)`** at frozen spiral states, norm-conserving first. What
   it must satisfy: `E1(q) - E1(-q)` is the chirality, and it vanishes for a cell with
   inversion symmetry about a site, which is the null. The number to reproduce is
   `E(q) - E(-q)` from a commensurate supercell or ultracell **with** full coupling at
   small `q`, where first order should hold; the ratio between the two as `q` grows is the
   measurement of where it stops holding. A heavy-element chain or bilayer small enough for
   a real supercell is needed, and choosing it is the first decision.
3. **The first-order wavefunctions and density**, by a sum over the ladder `k +- q` first
   (as P37 and P54 use one where a Sternheimer solve is not set up), then by a Sternheimer
   solve. The number: the charge at `2q` against `q` on the three-cell NiBr2 helix against
   the Kramers-closed ultracell at the same `N`, after that ultracell is rerun at
   `conv_thr = 1e-11` (`OPEN.md` Part XX), and against Elk's 15 on 15 cells.
4. **Self-consistency**: the spiral in the Sternheimer stack. Size it first; it is the step
   most likely to be a phase of its own.
5. **The spiral ultracell** of the previous section, if 2 and 3 show the perturbation
   running out, and compared with them where both converge.
6. **An incommensurate `q`**, which nothing above requires to be special once the ladder
   is in, and which no reference code here can check. The check is the approach to a
   commensurate `q` from both sides.

## What to decide at the start of that session

Decided on 2026-09-27: perturbation theory first (the user's request); the nickel chain
with an iodine beside each bond as the validation cell (the user's pick, the only magnetic
fully-relativistic norm-conserving dataset committed being `Ni.rel-pbe-nc-dojo`); and the
one-file route at `soc_scale = 0`, which was not a choice once written down, since the
first-order operator is the difference of one dataset's two Hamiltonians and the
scalar-relativistic partner file's `D` is not a perturbation of anything. `soc_scale`
between 0 and 1 is now admitted on norm-conserving datasets (the user's decision), so the
weak-coupling limit is an input knob. What is next,
sized: the first-order wavefunctions over `k -+ q` and the tilt of the plane they carry
(a sum over the ladder, P37's and P54's pattern, then a Sternheimer solve); a cutoff and
k-mesh sweep of the chain's first order, which is a Triton array of unit-cell runs; and an
ultrasoft dataset, which needs the `newd_so` blocks derived first.

- Perturbation theory (steps 2 to 4) first, or the spiral ultracell (step 5) first. The
  first gives the Dzyaloshinskii-Moriya energy soonest and is the published route; the
  second is closer to machinery that exists and is not limited to weak coupling.
- The validation cell for step 2: it needs a spiral, enough spin-orbit coupling to measure,
  broken inversion so that `E1(q) - E1(-q)` is not zero by symmetry, and a supercell small
  enough to run with full coupling on the workstation.
- Whether the refusal of `spiral_q` with `lspinorb` should allow `soc_scale = 0`, which is
  what step 1 would converge at on a fully-relativistic dataset, or whether step 1 uses the
  scalar-relativistic partner file as the force theorem does (P58's two-file route, which
  cannot carry PAW).

## What the derivation found (2026-09-27)

Let us write the operator the plan calls the coupling as it enters this code. At a frozen
potential, on a norm-conserving fully-relativistic dataset, `soc_scale` changes `dvan_so`
alone, linearly, so the first-order operator is `dD = dvan_so(1) - dvan_so(0)`, and its spin
trace is zero exactly, because `soc_scale` scales the spin-traceless half and nothing else.
Write it as `dD = sum_a dD^a sigma_a`, `a = x, y, z`, each `dD^a` a Hermitian matrix over the
projectors, and take the spiral's axis, the spin direction of its up component, along `n`.
What matters is that `dD` is the same matrix in every cell while the spins turn from one
cell to the next, so in the spiral's frame its `sigma_n` part keeps each spinor component on
its own sphere and its `sigma_+-` parts move a component from the state at `k` to the state
at `k -+ q`, exactly as the plan's section on the break said.

**The first-order energy is linear in the axis.** Only the `sigma_n` part has an expectation
value, the transverse part summing to `sum_R exp(-i q . R) = 0` over the cells, so

    E1(q, n) = sum_k w sum_n f [ <u_up| n . dD |u_up>_(k+q/2) - <u_dn| n . dD |u_dn>_(k-q/2) ]
             = n . V(q),

with each component projected on its own sphere's projectors. At zeroth order turning the
spiral rigidly in spin space costs nothing, so one vector `V(q)` gives the first-order energy
of every orientation of the spiral plane, and the spiral code's fixed rotation axis (`z`) is
no restriction. This also settles the plan's worry about "the `sigma_n` projection of
`dvan_so` for an arbitrary axis": it is a dot product with three Pauli components computed
once.

**It is odd in `q`, all of it.** The spiral at `(q, n)` is the same texture as the one at
`(-q, -n)`, so `V(-q) = -V(q)`: there is no first-order part even in `q`, meaning that the
whole first-order energy is the chirality, the Dzyaloshinskii-Moriya energy, and the
anisotropy of a spiral starts at second order as that of a ferromagnet does. Inversion
through a site sends `(q, n)` to `(-q, n)`, so a centrosymmetric crystal has `V = 0`
identically, which is the null. At small `q`, `V(q) = D q` and `D` is the micromagnetic
tensor of `E = D_ij n_i q_j`.

**The chirality is not first order alone, and on one of the validation cells the first
order is the smaller part.** It is tempting to argue that the odd-in-`q` energy is odd in the
`sigma_n` part of the coupling, the transverse part entering only as its square, so that
the next odd term is third order. That is wrong, and it was measured wrong before it was
seen to be. The transverse part enters second order as a quadratic form in its two
components, and the form has an antisymmetric piece, the `L_-` transition to `k - q`
against the `L_+` one to `k + q`, which a half-turn of the spins perpendicular to `n`
reverses and nothing else forbids. So the odd part of the energy is `E1 lambda + E2odd
lambda^2 + ...`: on the tight-binding chain `(O / lambda - E1) / lambda` is -0.0597, -0.0606
and -0.0615 at `lambda` = 0.01, 0.02 and 0.04, and on the nickel-iodine chain at 40 Ry and
`1 1 8` the supercell's odd part over `lambda` is 0.0727, 0.1143 and 0.2941 meV at 0.1, 0.2
and 1, against a first order of 0.0278. The quadratic through those three points reaches
0.0267 at zero, and its slope is 0.48 meV, seventeen times the first order. The same chain at
34 Ry (the test cell) gives a first order of -1.2179 meV, a limit of -1.21792 through
`lambda` = 0.05, 0.1 and 0.2, and a slope of +0.148 meV, a tenth of it and of the other sign.
The first order is always the right limit; what the numbers say is that how far it is from
the chirality energy at the physical coupling depends on how much the k-points cancel, which
is the whole of the first order at 40 Ry and a small part of it at 34, and that step 2's
"where first order stops holding" has to be measured cell by cell and converged, not
assumed. Neither cutoff is converged for nickel, and the first order is -2.35 meV at 30 Ry,
-1.22 at 34 and +0.028 at 40 on eight k-points.

**On a site that is itself an inversion centre, the null holds k-point by k-point.**
Inversion, time reversal and a half-turn of the spins about an axis perpendicular to `n`
together map the spiral at `k` onto itself, and they reverse `sigma_n` while leaving the
orbital part of `dD` alone, so every `k` contributes zero, not only the sum. Measured on
bulk fcc nickel at `q = b3/4`: the largest single k-point is 7e-8 meV. Such a cell cannot
check anything but the null, which is why the validation cell had to be polar.

**Which harmonic appears at which order.** The step the plan asked for first has a clean
answer for a flat spiral, the one whose rotating-frame `m_n` vanishes. Its coupling-free
state is invariant under `S`, time reversal followed by a half-turn of the spins about `n`
(time reversal reverses the moment, and the half-turn rotates the reversed in-plane moment
back, which is only a phase shift of the spiral). Under `S` the `sigma_n` part of `dD` is
even and the transverse part is odd. Give the two parts separate strengths, `lambda` and
`mu`; then `S` maps the problem at `mu` onto the one at `-mu`, every power of `mu` moves a
harmonic by one multiple of `q`, and every power of `lambda` moves nothing. So:

- the **charge** is even under `S`, hence even in `mu`, hence it has **even harmonics only,
  at every order**: the charge at `2q` appears at second order in the transverse coupling
  and the charge at `q` is zero for a flat spiral at every order, while `lambda` alone gives
  a lattice-periodic change of the charge at first order;
- the moment along the axis, `m_n`, is odd under `S`, so it has odd harmonics only, and its
  `q` harmonic appears at **first order**: the coupling tilts the spiral plane, which is the
  first-order torque on `n` that `E1 = n . V` implies;
- the in-plane moment is even under `S`: its corrections at `q` come at first order in
  `lambda`, and the first new harmonics, at `-q` and `3q` (the ellipticity of the spiral),
  at second order in `mu`.

This contradicts two things written above. Step 3's number, the charge at `2q` against `q`,
is not a first-order quantity: the first-order density has no charge harmonic at either, and
the `2q` one needs the second-order wavefunctions or the spiral ultracell. And Elk's 15 to 1
on NiBr2 cannot be the perturbation's `2q` against its `q`, since the `q` harmonic of a flat
spiral is forbidden; a nonzero charge at `q` has to come from a cone (a uniform `m_n` breaks
`S`) or from the supercell converging to something that is not a flat spiral. What step 3
can measure at first order is the tilt, the `q` harmonic of `m_n`, against the torque that
`V` implies.

