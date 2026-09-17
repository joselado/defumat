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

**The sizings in this file have been wrong seven times, always in the same direction and
always for the same reason.** §1a was called the easiest lift here when `PLAN.md` P40 had
already measured that route as not closing; §1i was called "the most likely of the class
to be an afternoon" when its projector set is scalar and cannot take the spinor operator
at all; §1j was called plumbing when the object it wanted to move is indexed by a
projector set that does not survive the handoff, and then, once that was corrected, was
called ill posed when the question it was asking was the wrong one (P97). §1e and §1e'
each named one missing term and had two and three. §1c named one and the code is missing
four, three of them in the augmentation rather than in the multipliers. §3b named the
strain derivative of `Q_ij(G)`, which `at_strain` had been rebuilding all along. Each came
from reading the refusal's *message* and not the code around it. So **a size in this file is a hypothesis until the
guard's surroundings have been read**, and the corrected entries say what reading them
changed.

**The fourth was the worst, because it was the entry this file ranked first.** PAW Born
charges were called "the only item here whose target, method and reference are all already
written down". The target was `int3_paw` against `becsumort`, a term that turned out not
to be missing; the reference number quoted, -0.07945, was the *ultrasoft* case's rather
than PAW's -0.07961; and the 1.3e-3 the whole entry rested on was a **wedge sum that had
never been completed**, which PAW inherited from the ultrasoft path and which silicon
could not show. On a `nosym` grid PAW reached `ph.x` to 8e-6 with nothing added. The entry
is gone and `PLAN.md` P39a has what closing it found. **The reading that would have caught
it is the one this file already prescribes** -- P39a's own last paragraph named the
missing measurement, a polar crystal, and the entry here was written from the refusal's
message instead.

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

### 1c. A phonon at `q != 0` with an ultrasoft or PAW dataset

`response/phononq.py:845`. Both.

**This entry named one term and the code has four missing, which is the sixth time this
file has been written from a refusal's message rather than from the code around it**
(2026-09-16). The message said the blocker was the orthonormality multipliers' term
`<psi|dS/du|psi>` between states at `k` and at `k + q`. That term is indeed absent, but so
are three others, and three of the four are in the augmentation rather than in the
multipliers, so the sizing sentence pointed at the smallest of the four. Read off
`response/phononq.py` rather than off its raise:

- **The response density has no augmentation at all.** `response_density_at_q` builds the
  pseudo pair density and returns `to_dense(2 total / volume, ...)`, where
  `addusddens.f90` adds `sum_ij dbecsum_ij Q_ij(q + G)` on top of it, with the table
  evaluated at the **shifted** modulus: the routine calls `setqmod(ngm, xq, g, qmod, qpg)`
  and passes that `qmod` to `qvan2`.
- **The induced potential has no `int3`.** `induced_perturbation_at_q` applies `dV_scf` as
  a local operator through the FFT and nothing else, where an augmented dataset also
  carries `int3_ij = int dV_scf Q_ij e^{iqr}` on the projectors
  (`LR_Modules/adddvscf.f90`).
- **The bare term freezes `D_ij`.** `bare_displacements_at_q` takes
  `dij = tuple(h.coefficients for h in solver.hamiltonians)` and closes over it inside the
  `jvp`, so `d/du` of `int V_eff Q_ij` is missing -- `dvanqq.f90`'s `int1` and `int2`.
- **The multipliers do not exist at `q != 0`.** There is no `overlap_derivatives`, no
  `orthogonality_states` and no `multiplier_response` in `phononq.py` at all; P39's are
  written with both `becp` at the same k-point, which is the `Gamma` case by construction.

**What it needs first.** The object the first two want is `q^a_ij(q + G)`, and it is
**written**: `tddft/spinchi0.augmentation_factors` builds it over a response sphere from
`topology/augmentation.augmentation_at_q`, pinned at `b = 0` against `projectors.qq`
(`tests/regression/test_topology.py`) and against `augmentation_at_q` at `+q` to 1e-14
with the opposite sign differing by more than 1e-6 (`tests/regression/test_ultrasoft_magnon.py`).
**Size:** a phase, and the work is spread over the four terms rather than concentrated in
the multipliers. The sign is the trap, and §1k has now settled it in code rather than in
prose: the displacement is the **ket's** wavevector minus the bra's, which the spiral's
special case pins and which the ultracell's supercell comparison was run against the
opposite spelling to confirm. Pin it here the same way, on an identity rather than on a
sentence -- the dynamical matrix at a zone-boundary `q` against the validated `Gamma` matrix
of the doubled supercell.

### 1d. A noncollinear ultrasoft or PAW response

`response/sternheimer.py:1202` and `:903`. Both, and only in the spinor regime.

**What is missing.** One object: `int3` as a 2x2 matrix in spin space, which is QE's
`set_int3_nc`. A norm-conserving dataset has no `dD` at all, so nothing already
validated reaches it.

**What this blocks.** The dielectric constant and the Born charges of a spinor run on an
augmented dataset, which is every heavy element. The collinear ultrasoft and PAW
responses are unaffected and are validated. **Size:** a phase.

### 1e. A spin spiral with an ultrasoft or PAW dataset. ✅ DONE for the ground state.

**Closed 2026-09-16.** `PLAN.md` P95 has the numbers. What is left of this entry is
`dE/dq`, which is item 1e' below.

**What the entry said was missing, and what it actually was.** It said the augmentation
charge between the two components is `q_ij(q)` rather than `qq` -- which is the right
idea at the wrong rank. `augmentation_at_q` evaluates the charge at a *single*
wavevector, which is what an overlap between two k-points needs; a *density* needs it
over the whole dense G set, as `Q_ij(G - q) e^{-i (G - q).tau_a}`. So the object was one
parameter away from `build_augmentation` rather than a call into the topology module,
and the lift was `shift=` on the builder that already existed.

**And the PAW half of the entry was wrong.** It said PAW needs Elk's per-atom phase
`e^{-i q.tau/2}` (`zqss`) on the transverse one-centre term. It does not need one:
`becsum` between the two components already carries `e^{i q.tau}` through its two
structure factors, and the one-centre energy depends on `|m|`, which a position-dependent
spin rotation leaves alone pointwise. **This was measured rather than argued** -- a
one-atom cell is translation invariant, so moving the atom by a third of the cell (onto
an exact grid point, or the egg-box error swamps it at 4.4e-06 Ry) must not move the
energy, and it moves it by **2.9e-12 Ry** for ultrasoft and **1.4e-11** for PAW. That is
the fifth entry in this file's own tally of sizings that were wrong for the same reason:
written from the refusal's message instead of from the code and the physics around it.

### 1e'. `dE/dq` for an ultrasoft or PAW spiral. ✅ DONE.

**Closed 2026-09-16.** `PLAN.md` P96 has the numbers, and `relax_spiral_q`, the
integrated `E(q)` route and `Calculator.get_spiral_relaxation` all run on an augmented
dataset now. What is still refused is a *tabulated* augmentation table, which reads
`|q|` on the host to size its radial interpolation and therefore cannot take a tracer at
all.

**The entry named one missing term and there were three**, which is the same failure this
file keeps recording: the sizing was written from the refusal's message rather than from
the functional. The named one was right -- the displaced table `Q_ij(G - q)` is a
function of `q` exactly as `|k + G|^2` and `vkb` are, and is rebuilt now inside
`at_spiral_q(rebuild_basis = False)`. The two that were not named were the ones a test
could have missed:

* **the orthonormality constraint**. `<psi|psi> - 1` carries no `q`, but on an augmented
  dataset the constraint is `<psi|S|psi> - 1`, and `S` pairs each spinor component with
  the projectors of its own shifted sphere, so it moves with `q` and its derivative is
  the spiral's Pulay term;
* **PAW's one-centre energy**, which was not in the differentiated functional at all --
  so the identity against the SCF total fails by the whole of `epaw` on the first PAW
  run, which is the one of the three that announces itself.

**What it cost, which the sizing did not name either.** The gradient is one pass over the
whole k axis whatever `k_batch` asks for, because the density carries `q` on this dataset
and the Hartree energy is quadratic in it, so a sum of per-chunk gradients is not the
gradient. The peak is **11.4 GB** on the one-atom oxygen chain at `ecutrho = 200` with
four k-points, against a few hundred megabytes for the SCF it follows, and it scales with
the dense G set: the same gradient at `ecutrho = 400` does not fit in 20 GB. That is the
practical gate on this feature and it is `MEMORY-AUDIT.md`'s kind of number rather than
a refusal.

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
response, which is item 3b below; the Born charge it also stands on was the other half
and is no longer refused on any dataset (`PLAN.md` P39a). So this is a **consequence**
rather than a term, and one of its two halves has already gone. **Size:** free, once 3b
lands.

### 1i. Site-resolved angular momenta on a fully-relativistic augmented dataset

`projwfc/angular_momentum.py:267`. Ultrasoft and PAW, and only when fully relativistic.

**What is missing.** The spinor overlap's off-diagonal spin blocks are `qq_so`
(`transform_qq_so`), and the projection here applies the scalar `S` to each component. A
fully-relativistic **norm-conserving** dataset has `S = 1` and is exact, which is the
regime the `j`-resolved PDOS of P69 runs in.

**What it needs first, read off the code rather than off the message.**
`Calculation._spinor_overlap` is the operator with `qq_so` in it and it exists; what it
cannot be handed is this module's projector set, which is *scalar*
(`build_atomic_projectors` is called with neither `noncolin` nor `spinor_basis`, giving
`(nk, npwx, natomwfc)`), and `_spinor_overlap` wants `(..., 2 npwx)`. The default `kind`
is `ortho-atomic`, so the Löwdin matrix `<phi|S|phi>` is spin-blocked too and cannot be
left scalar either.

The route that works is the one `workflows/anisotropy.py:_project_band_energy` already
takes: build the set with `Calculation._as_spinors`, which is
`atomic_wfc_nc_updown` -- a real harmonic times a pure up or down spinor -- orthogonalise
it in the spinor space against `_spinor_overlap`, and contract. That basis is the right
one here for a reason beyond convenience: `_contract` labels its blocks by `(m, spin)`,
and `m` and the spin are good labels in `nc_updown` and are **not** good labels in the
`j`-resolved `atomic_wfc_nc_proj` that `projwfc/projections.py` uses. So the columns
double to `2 natomwfc` with a known order and the shell bookkeeping survives.

**Size:** an afternoon for the code and a phase for the number, which is the part with no
route yet: `<L>` and `<S>` per site on a fully-relativistic augmented dataset need a
reference, and neither `projwfc.x` nor Elk's `LSJ.OUT` has been located for that
combination.

### 1j. The force theorem for magnetocrystalline anisotropy with PAW. ✅ DONE.

**Closed 2026-09-16.** `PLAN.md` P97 has the numbers.

**Both of this entry's sizings asked the wrong question**, and the second one asked it
more carefully. The first said to widen the handoff and carry `becsum` across the way
`run_nscf` demands it. The second said that cannot be done, the two legs using different
pseudopotential files whose projector sets are indexed differently -- which is true, and
then concluded that the open question was whether a PAW force theorem is well posed at
all. It is well posed. What the theorem freezes is the potential, and on a PAW dataset the
potential has **two representations**: the one-centre coefficients are a functional of
`becsum` exactly as the grid potential is a functional of `rho`, so the frozen object is
the pair. Once that is said, the route follows without moving anything between files --
**one** fully-relativistic file run twice, `soc_scale = 0` for the self-consistent leg and
1 for the one-shot, so both legs share a projector set by construction. `soc_scale` leaves
`nh` and `fcoef` untouched and blends only `dvan_so` and `qq_so` toward their spin trace,
which is exactly what `average_pp` cannot do for a PAW dataset.

**The numbers.** The rotation identity on antiferromagnetic PAW oxygen, five directions
with the coupling off: **3.142e-10 meV**. The same run with `becsum` carried but not
rotated with the density: **468.0 meV**, on a cell whose answer is exactly zero. What is
not yet measured is a magnetocrystalline anisotropy on a fully-relativistic PAW dataset,
and `PLAN.md` P97 names the two cells that were run for it and how each failed -- the
committed tetragonal nickel diverged and a platinum dimer converged nonmagnetic. The next
attempt wants a third cell rather than a third run of those two.

### 1k. The ultracell with an ultrasoft or PAW dataset. ✅ DONE in all three spin regimes.

`PLAN.md` P88 stage 5, 2026-09-17. `ultracell/augmentation.py`. The full record, with the
numbers, is there; what belongs here is the shape of the answer and the one sentence this
entry had wrong.

The refusal's two stated reasons were both wrong and this file already said so: the frozen
states are a fixed basis by construction, and `S` does not enter because the augmentation part
of the cross-`Q` overlap is `N delta_{QQ'}`. What was missing was one term used three times --
the matrix element, the density's `becsum`, and PAW's one-centre terms per copy -- and it is
the resident table displaced by `Q' - Q`, the **ket's** wavevector minus the bra's.

**The sign in this entry's derivation sentence was wrong** and `PAW-MODES-NEXT.md` §8 had
already flagged the disagreement: the formula line said the displacement is `Q' - Q`, which is
right, and the sentence after it said `build_augmentation(shift = Q - Q')`, which is not. The
guard's comment on `master` had `Q' - Q` throughout and was the consistent one. `Q' - Q` is
what the spiral fixes (`at_spiral_q` passes `shift = -qcart` and the spiral's `Q' - Q` is
`-q`) and what was implemented and measured; the other spelling was run deliberately and
breaks the variational bound.

**What is still refused**, and all three are in `PLAN.md` P88 stage 5 with what each needs:
the **spinor** combination, where `D_ij` is the scalar integrals sandwiched between `fcoef`
rather than the integrals themselves; the **double grid**, which is the wall such a dataset
meets first and is why every number is at `ecutrho = 4 ecutwfc`; and the ultracell
**transmission**, whose exit-plane Gram matrix needs `S` for a state spanning `N` spheres,
where the image and the spectrum do run because a tip is in vacuum. The double-grid lift is about three
lines -- mask `dV` to the tiled smooth sphere for the smooth half of the matrix element, keep
the dense `dV` for `newd`'s integral, which is QE's own split -- and was deliberately kept out
of the same measurement rather than folded into it.

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

**The term is measured, on a model with no mesh floor at all** (P94,
`tests/unit/test_topology_curvature.py::test_a_moving_overlap_needs_more_than_dh_and_ds`).
Haldane's `H_0(k)` with `H = A^dagger H_0 A` and `S = A^dagger A` has the generalised
problem's physical curvature equal to `H_0`'s exactly, so the reference is free. Feeding
`dH` and `dS` to `kubo_from_matrices` is wrong by **18 per cent** of the curvature's scale
and moves the Chern number by 0.010, so the term does not integrate away either; putting
the two `A^dagger dA` blocks in recovers it to **4.5e-16**.

**What that pins is the shape.** The two factors need *different* blocks and neither is
`dS`: `(e_m - e_n) L` on the first and `-(e_m - e_n) K` on the second, with
`L = c^dagger (dA)^dagger A c`, `K = c^dagger A^dagger (dA) c` and `L + K = dS` to
1.8e-15. So no arrangement of `dS/dk` can supply it, which is the same statement as the
correction depending on `A` rather than on `S = A^dagger A` -- `U(k) A` leaves `S`
unchanged and moves the physical states.

**Size:** a phase, and the model check is done. What is left is the dipole in
matrix-element form inside `velocity_matrices`, from `efield.py`'s machinery, and then an
**assembly** check on plane waves, because a sum of separately validated pieces is this
repository's most convincing wrong answer. That check is the one P94 used for `dS/dk`:
zero the dipole on AlAs-US and read the shift, and only run the AlAs comparison if the two
terms together exceed its 4.4 per cent floor.

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

**What is missing is not the strain derivative of the augmentation charge, and this
entry said it was** (corrected 2026-09-16). `Calculation.at_strain` rebuilds
`build_augmentation` whole -- the table is sampled on a moving `G` set, so it has to be --
and every link of the strain response is a `jvp` through that call: the bare perturbation,
`dS/deps`, the frozen `drho` and `dbecsum`, and `_position_response`'s operators. The one
object held at the unstrained cell is `projectors.qq`, and that is correct rather than an
omission, `int Q_ij(r) dr` carrying no cell at all. The strain *response* itself runs on
both datasets and is pinned against a central difference of the converged density at
4.6e-4 (ultrasoft) and 4.7e-4 (PAW) against a norm-conserving 1.9e-4, which is P41.

**What is missing is what P44's own measurement says**, and `require_norm_conserving`'s
docstring is where it is written rather than here: the residue is **entirely the `b`
partial**, -1.72 on 112, the *same* number on ultrasoft and on PAW, which is what says it
is structural rather than a dataset's physics. One candidate for it is excluded by
measurement: writing `_position_response`'s commutator source with the multiplier matrix
rather than the frozen scalar eigenvalue takes the strain coordinate to 1.7e-4 on both
datasets **and breaks the displacement one**, in every pairing tried. So one of the two
coordinates carries a further term that compensates it, and finding that is what closes
this. **Size:** a phase, and the hardest one on the list -- but the hard part is a term in
the position response, not an augmentation table nobody wrote. Note that the
**displacement** coordinate of the same third derivative is *not* refused and is validated
at 1.2e-4 on both datasets (the Raman tensor, P43), so what is wrong is specific to
strain.

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

The first two entries of this list were run in P94 and neither lifted a refusal; the
third, PAW Born charges, was run on 2026-09-16 and **lifted its refusal by finding that
the term it named did not exist** (`PLAN.md` P39a). What is left:

1. **The moving overlap in a Kubo sum** (§2). The missing term's shape is pinned exactly
   on a model, so what is left is writing it in matrix-element form from `efield.py`'s
   machinery and checking the *assembly*.
2. **Site angular momenta on a relativistic augmented dataset** (§1i). The code route is
   clear; the open question is what to validate it against.
3. **The `chi_0` gap** (§1a), now not a truncation and about 1 per cent rather than an
   unknown.
4. Everything else, in whatever order the physics wants.

**And one thing to do to the whole list rather than to an item in it.** What closed P39a
was not a term, it was a *cell*: silicon is centrosymmetric, so every `Z*` this project
had ever compared against `ph.x` was a quantity symmetry forces to zero, and five digits
of agreement about a residue said nothing about the half symmetry had deleted. Several
entries below are sized from a gap measured on silicon alone -- §1c, §1d and §3a among
them -- and a gap measured on a centrosymmetric cell is a hypothesis in exactly the way
this file's sizings keep turning out to be. **Before writing a term for any of them, run
the refused quantity on `alas-epsilon-us.in` and see what the number is there.**
