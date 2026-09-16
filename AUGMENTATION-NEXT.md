# Ultrasoft and PAW: what still refuses them, and what each piece needs

## What this file is

The queue for the one axis that cuts across every other feature in this project. A
dataset that carries an augmentation charge changes three things at once, and each of
them reaches a different part of the code: the density is no longer `|psi|^2`, the
overlap operator is no longer the identity, and `D_ij` is a function of the potential
rather than a number in the file. The ground state, the forces, the stress, both
relaxations, the dynamical matrix at `Gamma`, the dielectric constant and the Raman
tensor all carry that already. What is below is everything that does not.

Where the other files fit (`CLAUDE.md` has the full table):

| question | file |
|---|---|
| what a phase found, and what it was measured at | `PLAN.md` §3 |
| what a feature refuses, and how a user reaches it | `docs/features.tex` |
| what is known to be broken, and what to do about it | `OPEN.md` |
| **what still refuses ultrasoft or PAW, and what each piece needs first** | this file |

**File and line references go stale.** The function or the guard is named wherever there
is one, so `grep` the name rather than trusting the number.

## How the list was made, and what that method cannot see

Two sweeps, because one of them cannot surprise you. The first was a scan for every
`raise` whose message contains "ultrasoft" or "PAW", which finds a refusal that says
which dataset it is about and misses one that names only the quantity. The second was a
scan for every `if` on `is_ultrasoft`, `is_paw` or `augmentation is (not) None` within
eight lines of a `raise` or a `warn`, plus the callers of
`response.phonon.require_norm_conserving`, which is how three of the third derivatives
refuse without the word appearing anywhere near the raise. The amber boxes of
`docs/features.tex` were taken as an independent third list and the set difference
checked. The one thing neither sweep reaches is a path that runs on an augmented dataset
and is quietly wrong, which is what `OPEN.md` is for.

Two definitions that matter for reading the table. `Calculation.is_ultrasoft` is
`augmentation is not None`, which is **true for a PAW dataset as well**, so a guard
written as `if calculation.is_ultrasoft: raise` refuses both. `Calculation.is_paw` is the
narrower one. Where an item below says "ultrasoft only" or "PAW only" it is because the
guard was read, not because the message was.

## What was excluded, and why

- `hubbard/manifold.py:632`, `hubbard_slater = 'yukawa'`. The refusal points the other
  way: the full interaction matrix is built from the manifold's all-electron partial
  wave, which only a PAW dataset carries, so this is a norm-conserving and ultrasoft gap
  rather than an augmented one.
- `pseudo/spinorbit.py`, `soc_scale`. It works on ultrasoft and PAW, where QE's
  `average_pp` refuses them outright.
- `workflows/stm.py`. The docstring's norm-conserving remark is about `stm.f90`, not
  about this code: the tunnelling weights go through `Calculation.density`, so the
  augmentation charge follows them and nothing here refuses.
- `pseudo/augmentation.py:797`, gamma-only storage. Substituted with a warning rather
  than refused, which is the decision `CLAUDE.md` records: the same physics at twice the
  storage, and the run says so.
- `workflows/transport.py:1214`, a tip plane inside an augmentation sphere. Geometric and
  correct: inside the sphere the pseudo-wavefunction is not the true one, and a tip
  belongs in vacuum.
- Meta-GGA with an ultrasoft dataset (`scf/driver.py:1863`). PAW works here and `pw.x`
  refuses both, so this is ahead of the reference rather than behind it. An ultrasoft
  dataset has no partial waves to reconstruct `tau` from and the refusal is permanent.

---

## 1. A term that has to be written

The largest class and the slowest. Each of these is missing an object, and the object is
named.

### 1a. A sum-over-states `chi_0` with an ultrasoft dataset

`tddft/chi0.py:301`, `require_a_sum_over_states_regime`. Ultrasoft and PAW, both.

**What is missing.** `<u_i|e^{-iG.r}|u_j>` gains the augmentation charge `Q_ij(G)` in
every matrix element, so without it the matrix is wrong by the whole augmentation and
still looks like a dielectric function.

**This entry said "what it needs first: nothing" and that was wrong.** P40 had already
built exactly this -- the body's `Q_ij(G)` with the volume factor restored, and the
head's `q`-linear dipole beside it -- and measured that it does not close: ultrasoft
silicon's `eps_M(0)` sat **2.1 per cent** from the Sternheimer solve where a
norm-conserving control on the same machinery sat at **0.06 per cent** (-1.20 on 55.5
against -0.0129 on 22.3, both at 60 bands and `ecut_response = 8`, both in RPA against
`screening = "hartree"`). The refusal was kept and the code reverted. The mistake in
writing this entry was reading the raise at `chi0.py:301` without reading the docstring
twelve lines above it, which is the "inherit a refusal only after checking which machine
it belongs to" trap run backwards.

**What P40 excluded, and what it did not.** Finding 1: the tabulated `Q_ij(G)` is a
charge per unit volume and has to be multiplied by `Omega` before it is paired with
`<beta|psi>`, which is a factor of 265 on that silicon and halved the residual when it
went in. Finding 2: with the body right, adding the head's `q`-linear part
(`adddvepsi_us`'s `dpqq`) moves `eps_M(0)` by **0.0015**, so neither half of the
augmentation accounts for the 1.2 and the next attempt should not start with either. Left
open by name: the pair density's normalisation at `G != 0`, never checked independently,
and the `f_i - f_j` weight, which for an ultrasoft dataset multiplies a generalised
density whose norm is `<psi|S|psi>` rather than `<psi|psi>`.

**What is worth doing, and it is the check P40 named.** The objects P40 reverted can be
had from a *different* code path: `tddft/spinchi0.augmentation_factors` builds
`q^a_ij(q + G)` from `topology.augmentation.augmentation_at_q` -- radial Bessel
transforms evaluated at `|G|`, the structure factor and the volume already in it, pinned
at `b -> 0` by reproducing `qq` -- where P40 gathered it from the dense table by Miller
index. Running P40's own validation with that route is exactly the independent check at
`G != 0` that P40 said had never been made. A second thing P40 did not do is the `nbnd`
trend: its numbers are at 60 bands on both datasets and there is a reason to expect the
ultrasoft sum to converge slower, since the augmentation term carries `<beta|psi_j>` and
a localised projector keeps weight at high `G` where a pseudo pair density does not.

**Both checks were run, and P94 has them.** The two routes disagree at `G != 0`: 57.200
against P40's 55.5 at the same `nbnd = 60`, with the residual at +0.540 against -1.20, so
the identity P40 could not check fails and the dense-table gather is the half with no
support. And the trend says what the single number could not: the norm-conserving control
falls to zero like a truncation (-0.0675, -0.0129, -0.0022 at 30, 60 and 90 bands, the
middle one reproducing P40's own digit for digit), while the ultrasoft residual is
positive at every count and does not tend to zero (+0.2497, +0.4778, +0.5403, +0.5941 at
30, 45, 60 and 80, where `npw = 169` caps the count). **No limit is claimed** -- the last
increment is still positive -- but a truncation goes to zero and this does not, so the
disagreement is neither a truncation nor a missing normalisation. At the largest count
that runs it is about 1 per cent, opposite in sign to P40's -2.1 and smaller. Two
qualifiers on that number: only the **body** was corrected, the head still being
`<m|dH/dk|n>` with no `-e_n dS/dk` and no `dpqq`, and the code is reverted as P40's was.
It is three lines.

**Size:** closing it is a phase and nobody knows yet how large. **PAW stays refused
whatever happens**, for `spinchi0`'s reason rather than this one: its exchange-correlation
kernel has a one-centre part on the spheres that the grid does not carry.

### 1b. Born effective charges with a PAW dataset

`response/born.py:157`, `require_born_charges`. PAW only; ultrasoft works.

**What is missing.** `zstar_eu_us.f90`'s one-centre stage, `int3_paw` contracted against
`becsumort`.

**The gap is measured rather than guessed.** Without it: **-0.078293** against `ph.x`'s
-0.07945, so 1.5e-3. The ultrasoft answer beside it is -0.079442, 8e-6 from the same
reference, and the norm-conserving one reproduces every digit of -0.07571. So PAW is
wrong in the third decimal rather than in sign, which is what makes the refusal worth
keeping: 1.5e-3 on a Born charge is small enough to be read as convergence.

**What it needs first.** `PAW_dpotential` is already one `jvp` of `onecenter`, so the
one-centre response exists; what is not there is its contraction against the *ordered*
`becsum` that `zstar_eu_us.f90` uses. **Size:** a phase.

### 1c. A phonon at `q != 0` with an ultrasoft or PAW dataset

`response/phononq.py:845`. Both.

**What is missing.** `S` moves with the atoms, so the orthonormality multipliers carry
`<psi|dS/du|psi>` between states at `k` and at `k + q`, and that has no two-sphere form
here. The `Gamma` case is written (P39) because there both states sit on the same sphere.

**What it needs first.** The same object item 1a uses, `q^a_ij(b)` at `b = q`, since the
two spheres are separated by exactly `q`. **Size:** a phase, and the work is in the
multipliers rather than in the augmentation.

### 1d. A noncollinear ultrasoft or PAW response

`response/sternheimer.py:1202` and `:903`. Both, and only in the spinor regime.

**What is missing.** One object: `int3` as a 2x2 matrix in spin space, which is QE's
`set_int3_nc`. A norm-conserving dataset has no `dD` at all, so nothing already
validated reaches it.

**What this blocks.** The dielectric constant and the Born charges of a spinor run on an
augmented dataset, which is every heavy element. The collinear ultrasoft and PAW
responses are unaffected and are validated. **Size:** a phase.

### 1e. A spin spiral with an ultrasoft or PAW dataset

`scf/driver.py:1545`. Both, and it takes `forces/spiral.py:387` with it.

**What is missing.** The augmentation charge between the two spinor components is
`q_ij(q)` rather than `qq`, since they sit on spheres centred at `k + q/2` and `k - q/2`.
PAW needs one thing beyond that: Elk's per-atom phase `e^{-i q.tau/2}` (`zqss`,
`init0.f90`) on the transverse one-centre term.

**What it needs first.** `topology.augmentation.augmentation_at_q` again, which is the
third item on this list to want it. `dE/dq` for a spiral (`forces/spiral.py:387`) is
unreachable until this lands and is a line of work of its own afterwards. **Size:** a
phase, and the one with the most surface: the spiral path also refuses symmetry, so
everything runs on the full grid.

### 1f. A source-free exchange-correlation field with a PAW dataset

`scf/driver.py:1719`. PAW only.

**What is missing.** The one-centre `B_xc` on the spheres is a second copy of the field
that the projection does not reach, so the projection would make the grid field
source-free and leave the sphere field alone. **Size:** part of a phase; the projection
itself is short and the question is what "source-free" means for a field living on two
representations.

### 1g. The orbital magnetization with an ultrasoft or PAW dataset

`workflows/orbital_magnetization.py:112`, and `topology/orbital_magnetization.py:79`.
Both.

**What is missing.** The overlaps between neighbouring k-points that the covariant
derivative is built from need `q^a_ij(b)`, and the dual states are built in the `S`
metric.

**`setup.f90` refuses the same combination** ("Orbital Magnetization not implemented with
USPP/PAW"), so this is a gap shared with the reference rather than a deficit against it.
**Size:** a phase. Worth noting that the FHS invariants already carry `q^a_ij(b)`
correctly on all three dataset kinds, so the missing half is the `zgefa`/`zgedi` dual in
a non-trivial metric.

### 1h. The piezoelectric tensor with an ultrasoft or PAW dataset

`response/piezo.py:273`. Both.

**What is missing.** Nothing in the piezoelectric assembly itself is norm-conserving: it
is one `jvp` of the stress along the field's response. What it stands on is the strain
response, which is item 3b below, and the Born charge, which is item 1b for PAW. So this
is a **consequence** rather than a term, and it lifts when those do. **Size:** free, once
3b lands.

### 1i. Site-resolved angular momenta on a fully-relativistic augmented dataset

`projwfc/angular_momentum.py:267`. Ultrasoft and PAW, and only when fully relativistic.

**What is missing.** The spinor overlap's off-diagonal spin blocks are `qq_so`
(`transform_qq_so`), and the projection here applies the scalar `S` to each component. A
fully-relativistic **norm-conserving** dataset has `S = 1` and is exact, which is the
regime the `j`-resolved PDOS of P69 runs in.

**What it needs first.** `SpinOrbitCoupling.qq_so` exists and
`topology/augmentation.py` already routes `q^a_ij(b)` through it, so the map is written;
what is missing is applying it inside the projection. **Size:** part of a phase, and the
most likely of the class to be an afternoon.

### 1j. The force theorem for magnetocrystalline anisotropy with PAW

`workflows/anisotropy.py:265`. PAW only; ultrasoft is allowed.

**What is missing.** The handoff from the collinear first leg carries the density and
nothing else, and a PAW Hamiltonian needs `ddd_paw`, which is built from `becsum` -- a
property of the states rather than of the density.

**What it needs first.** Widening the handoff to carry `becsum`, which is the same object
`run_nscf` and the topology workflows already demand by name. **Size:** part of a phase,
and it is plumbing rather than physics.

### 1k. The ultracell with an ultrasoft or PAW dataset

`ultracell/driver.py:356`. Both.

**What is missing.** The augmentation charge is a function of the density through `D_ij`,
so the frozen unit-cell states the ultracell is built from are not a fixed basis any
more. P88's stage 1 is norm-conserving by design. **Size:** a phase, and it belongs to
the ultracell's own roadmap rather than to this one.

---

## 2. A term that is half written, and the half that is missing is a term

Three refusals, one term. All three say the same sentence in `docs/features.tex` and all
three would lift or stay together.

**This section used to be called "a term that is written and unvalidated" and that was
wrong** (P94). Writing the derivation out splits it: the *convention* is right by
construction and needed a derivation rather than a measurement, and what is actually
missing is a second term that has nowhere to go in the present assembly. So this belongs
with §1 rather than beside it, and the sizing is a phase rather than the afternoon this
file first claimed.

The term is the off-diagonal `<psi_n| dS/dk_a |psi_m>`, which enters the velocity of a
generalised eigenproblem as `-e_n dS/dk_a` and is **identically zero for a
norm-conserving dataset**. It is written -- `VelocityOperator.apply_s`, one `jvp` of
`s_psi`, the second tangent of the same `jvp` that gives `dH/dk`. What has never been
checked is its convention, because no norm-conserving validation can see a term that
vanishes.

**The obvious test was run and it cannot discriminate** (P94). Kubo against FHS on
ultrasoft AlAs agrees to 8.5 per cent at 18x18, beside 7.0 on a norm-conserving AlAs run
(not a matched control -- the functional and the cutoff move with the dataset), which
reads as a pass. But zeroing `dS/dk` moves `Omega` by only **2.5 per cent**, five times
less than the gap between the two methods at 12x12 and half of it at 24x24, so the term
is below the test's own resolution at every mesh and deleting it entirely would leave the
comparison looking the same.

- **The Kubo Berry curvature**, `topology/kubo.py:236`. `method='fhs'` is the default,
  carries both this term and `q^a_ij(b)` correctly, and is exact on any mesh, so nothing
  is unreachable.
- **The optical conductivity**, `response/conductivity.py:388`.
- **The shift current**, `response/photocurrent.py:523`, which needs the same term one
  order further out: the dipole carries `dS/dk` and its derivative carries
  `d^2 S/dk_a dk_b`.

**The convention is settled, on paper.** Differentiating `H c = e S c` and projecting on
`c_n` for `n != m` gives `<n|S|d_a m> = <n|d_a H - e_m d_a S|m> / (e_m - e_n)`, the
**ket** band's energy; the curvature `-2 Im <d_1 n|S|d_2 n>` with the identity resolved as
`sum_m c_m c_m^dagger S = 1` then has `n` as the ket in both factors, so `e_n` multiplies
`dS/dk` in each. That is exactly what `kubo_from_matrices` builds.

**What is missing is the augmentation dipole, and it has nowhere to go.** With
`S = T^dagger T`, the true states are `T c` and
`<Psi_n|d Psi_m> = c_n^dagger S d c_m + c_n^dagger T^dagger (d T) c_m`. The first piece is
the formula above; the second is `adddvepsi_us`'s `dpqq`, the position operator acting on
the augmentation charge, and `kubo_from_matrices` sees only `dH` and `dS` so there is no
slot for it. FHS has it, because `q^a_ij(b)` **is** `T^dagger(k) T(k')` to first order in
`b`. `response/efield.py:336` builds
`-i (dH/dk_a - eps_v dS/dk_a)|psi_v>` from `VelocityOperator.both`, solves it into the
empty space with `P_c`, and adds the dipole separately as `_ultrasoft_position` -- and its
ultrasoft dielectric constant is 8e-6 from `ph.x`, so the object exists and is pinned. It
is the *Kubo* assembly that does not carry it.

**What decides it is not a finer mesh.** The model that shows the missing term without a
mesh at all: Haldane's `H_0(k)` with `H = A^dagger H_0 A` and `S = A^dagger A` for a
smooth invertible `A(k)`, where the generalised problem's physical curvature is `H_0`'s
exactly. `kubo_from_matrices` then misses it by `(e_m - e_n) c_n^dagger (dA)^dagger A c_m`
in each factor, which is the model's `T^dagger dT`, and subtracting that recovers it. The
correction depends on `A` rather than on `S = A^dagger A` alone -- `U(k) A` leaves `S`
unchanged and moves the physical states -- which is why the plane-wave version needs
`dpqq` rather than only `dS/dk`.

**Size:** a phase. The model check is an afternoon and pins the structure; what follows is
the dipole in matrix-element form inside `velocity_matrices`, from `efield.py`'s
machinery, and then an **assembly** check on plane waves, because a sum of separately
validated pieces is this repository's most convincing wrong answer. That check is the one
P94 used for `dS/dk`: zero the dipole on AlAs-US and read the shift, and only run the
AlAs comparison if the two terms together exceed its 4.4 to 6.3 per cent floor.

---

## 3. A compound refusal

Not a dataset and not a quantity, but the two together. These are the ones where the
augmented path works and something else about the run makes it not work.

### 3a. The dynamical matrix of an ultrasoft or PAW **metal**

`response/phonon.py:1618`, `_require_a_moving_overlap_regime`. The guard is
`is_ultrasoft and occupations != "fixed"`, so it catches PAW and it catches only metals.

**What is missing.** P28's `wg`/`wk` weight split was derived for a response whose
`becsum` dependence is entirely through the wavefunctions, and with smearing the
occupations respond to the perturbation as well. **Size:** a phase, and the derivation is
the work rather than the code.

### 3b. Third derivatives in the **strain** coordinate

`response/phonon.require_norm_conserving`, reached from `response/elastic.py:169` and
`response/electrostriction.py:860`. The elastic constants, electrostriction and the
elasto-optic tensor. Both datasets.

**This is the best-measured item on the list, and P44 is why.** Against a central
difference of the strain over re-converged cells, on the `(0,0)` strain of the `nosym`
cells:

| tangents | ultrasoft | PAW |
|---|---|---|
| neither | 4.58e-2 | 5.53e-2 |
| both | 1.30e-2 | 1.30e-2 |

against a norm-conserving control of 2.3e-4 that does not move at all. Two of the three
ingredients transfer and are already wired in behind the refusal: the state tangent is
`dpsi + ort`, and `_position_response` is handed `internals["commutators"]`. A thirtyfold
improvement and still fifty times the control, which is what says the third ingredient is
a term rather than a tolerance.

**What is missing.** The third ingredient, which is the strain derivative of the
augmentation charge itself: `Q_ij(G)` depends on the cell through `G`, and
`stres_us`/`addusstress` are the QE routines that are not transcribed here, so the
analytic route offers terms and no total to check against. **Size:** a phase, and the
hardest one on the list. Note that the **displacement** coordinate of the same third
derivative is *not* refused and is validated at 1.2e-4 on both datasets (the Raman
tensor, P43), so what is wrong is specific to strain.

**The piezoelectric tensor (item 1h) lifts with this.**

### 3c. Orthonormality multipliers for an ultrasoft or PAW **spinor** force

`forces/energy.py:471`. Both, and only in the noncollinear regime, and only on the matrix
form of the constraint.

**What is missing.** `_constraint_energy` contracts the scalar `qq`, where a spinor's
metric is `qq_so` and `Lambda` carries a spin index. The scalar spinor forces of P46 run
and are validated; this is the matrix-multiplier path beside them. **Size:** part of a
phase.

---

## The order to do them in

By what the first step costs, not by what the item is worth.

The first two entries of this list were run in P94 and neither lifted a refusal; what
they left is a sharper target and a warning about the instrument. What is left:

1. **Site angular momenta on a relativistic augmented dataset** (§1i) and **the force
   theorem's PAW handoff** (§1j). Both are applying an object that exists in a place that
   does not yet call it, which is the only shape of work here that is not new physics.
2. **Born charges on PAW** (§1b). One named term, a measured 1.5e-3 gap, and a reference
   routine to transcribe.
3. **The `chi_0` gap** (§1a), now a band-converged 1.0 per cent rather than an unknown.
4. **The moving overlap in a Kubo sum** (§2), whose first step is finding a system where
   the term exceeds the mesh floor.
5. Everything else, in whatever order the physics wants.
