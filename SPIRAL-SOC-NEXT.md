# Spin spirals with spin-orbit coupling, as a perturbation on the generalized Bloch theorem

A plan recorded on 2026-09-26 for a later session, not started. We will now see what the
idea is, why spin-orbit coupling breaks the construction it starts from and what that does
to a perturbation expansion, what in this code it builds on, how it differs from the
ultracell (and where the two should agree), and the order in which to build and measure it.
Nothing below has been derived in detail or run; every physics statement is to be checked
before code is written, and the ones that are an expectation rather than a result are said
to be so.

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
appears at which order is the first thing to derive. The test that decides it exists
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
