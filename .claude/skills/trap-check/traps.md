# The traps that recur, in full

*Moved out of `CLAUDE.md` verbatim on 2026-10-08. `CLAUDE.md` keeps one line per trap; this is the account of each, with the numbers.*


Every one of these has been hit in more than one phase, and most of them produce a
plausible wrong answer rather than an error. `PLAN.md` has the phase that found each.

- **`abs` is not differentiable at zero, and the zero is often forced.** `|psi|^2` must be
  `Re(conj(psi) psi)`; `abs(rho)**2` in the reciprocal Ewald sum is `0/0` wherever a
  structure factor vanishes *exactly*, which symmetry arranges on a supercell; `|m|` in the
  gradient correction differentiates through its own nodes. Five sites so far (P24, P28a,
  P45, P58).
- **A clamp's *tangent* at the boundary, where JAX splits a tie evenly.**
  `jnp.clip(z, -1, 1)` is `minimum(maximum(...))`, and JAX's `minimum`/`maximum` give
  **half** the gradient to each argument where the two are equal — so at `z = 1` exactly
  the tangent is `0.5`, not 1 and not 0. `|zeta| = 1` is every point of a **saturated**
  magnet, so it is reached rather than approached, and the halved tangent halved the
  minority potential's `de_c/dzeta` term: **0.162 Ry** in `v_down`, 0.07 Ry in the empty
  minority eigenvalues, with the **energy unchanged**, so nothing else saw it. Which side
  of the boundary a run lands on is *rounding* — the same H atom as an LSDA run and as a
  noncollinear one gave `rho_down` of 4.7e-18 and 1.4e-17, and only the first makes `zeta`
  round to exactly 1. Write the clamp as a `jnp.where` so the interior's tangent survives
  at the boundary (`xc/lda.py:clamp_polarization`); a one-sided derivative is what a
  potential at a vanishing channel density *is*. Four sites plus `minimum(|m|, |n|)`, which
  is a saturated point by definition (`OPEN.md` G1).
- **Rule D4: a diagonal is not invariant under the rotation a degenerate eigensolver is
  free in.** Anything built from `<psi_n|A|psi_n>` band by band — a Drude weight, a band
  velocity difference, an incoherent channel sum — takes the **multiplet block average**
  instead. Worth four orders of magnitude on silicon's `chi^(2)` and 69x against 26x on a
  bilayer's transport contrast, and **no symmetry check sees it** (P51, P54, P66).
- **A caller-built k-set is a `for_spin` boundary.** Every `KPoints` constructor applies
  the unpolarized `degspin` unconditionally, and a spinor band holds one electron. The
  wrong numbers are the plausible ones — a plasma frequency of 13.11 eV instead of 0.60
  (P51), a factor of four in a nesting function (P52).
- **A response on a reduced k-set is a polar (or axial) vector field and must be
  symmetrised as one**, and the obvious escape does not work: a **shifted** Monkhorst-Pack
  grid is *not* closed under the point group, so running the whole grid instead of the
  wedge is unsound there and is refused by name (P24).
- **A guard that detects a wedge cannot be a spread in the k-weights**, and six copies
  of that test were here. A symmetry-reduced **shifted** Monkhorst-Pack grid has exactly
  uniform weights whenever the group acts freely on it, which is what a shift arranges:
  on `pw_scf/scf.in`'s fcc cell at the ordinary `4 4 4 1 1 1`, time reversal alone keeps
  **32 of 64** points at `ptp(weights) = 0` exactly and `{E, C2z}` keeps 16 of 64, also
  0, against 36 and 30 points at `ptp` of 0.031 and 0.094 on the same grid *unshifted*.
  The identity is always in the group and time reversal is on by default, so even a P1
  cell halves. The answer is to record the fact on the set (`KPoints.reduced`, set by
  `KPoints.automatic`) and read it through one predicate (`is_reduced`), keeping the
  spread underneath as a fallback for a hand-built set the flag cannot see (P37).
- **A wedge sum completes only for a quantity *linear* in a covariant per-k object.**
  Where a functional is quadratic in one, the *value* inside it must be the full-zone
  object while its *derivative* stays the raw wedge sum. Getting it wrong is worth 2.5%,
  is worse than doing nothing, and only the sum rule catches it (P36).
- **`np.asarray` on anything differentiated kills the gradient silently** — a term that
  should be there simply vanishes (P43).
- **The energy can be right while its derivative is wrong**, which is the whole supercell
  family (P28a) and is also how a dropped gamma-storage `G = 0` term shows: the total was
  right to 3e-12 Ry and the force wrong by 0.4 Ry/bohr on a force of 0.06, because being
  stationary hides an error in the gradient (P68). The identities that are sums over
  *atoms* — the acoustic sum rule, the rigid-translation test — are blind to a transfer
  between atoms; the first check that is not an atom-sum is a per-mode response density
  against a finite difference.
- **A stencil must not contain its own centre.** The plane-wave sphere is rebuilt at every
  `k`, and a high-symmetry point is exactly where a shell sits on the cutoff — `Gamma`
  holds fewer plane waves than every displaced point, so its eigenvalue is variationally
  **high** against theirs and a second difference inherits an error that *grows* as the
  stencil shrinks (P48).
- **Neighbouring k-points do not share a G-sphere.** Coefficients are aligned by Miller
  index, and the wrap at the zone edge is a *shift* of that index — without which a Chern
  number comes out smooth and non-integer (P16).
- **Index order in a transposed pair reads as a sign.** `f(n, m)` against `e(m, n)`, and
  `G(r, r')` conjugating `psi` in the *exit* variable rather than the source one: both give
  results that are real, non-negative, correctly symmetric and wrong (P54, P66).
- **An `nspin = 2` screening kernel is not finite where a channel density reaches zero**,
  which a cell with vacuum guarantees. That is the `abs` trap one derivative further out,
  it lives in `defumat/xc`, and clipping inside the response does not fix it (P45).
- **Inherit a refusal only after checking which machine it belongs to.** P35's refusal is a
  statement about the Sternheimer stack and never applied to a sum over states; taking it
  as read left a whole quantity marked impossible (P54).
- **A check whose null result cannot be told from a pass.** A diagnostic that returns the
  same constant across the whole family it is meant to discriminate reads as agreement
  rather than as silence, and nothing downstream can tell the difference. Five in one
  production run: a chirality projected on `q` is identically zero for a **cycloid**, whose
  moments lie in the plane `q` lies in, so fourteen bonds of signed zero reported as
  "signs agree on 14 of 14"; a ratio between two quantities the driver evaluates at the
  *same* density is 1 by construction at convergence (`driver.py:3988` rebinds the
  potential to `rho_out`, `:4121` sets `rho = rho_out`), so it is informative only where
  `dr2` already is; a monitor read off a `{:6.3f}` log line stops moving when the **format**
  runs out, at 1e-3, rather than when the physics does; and QE's `i_cons = 2`
  (`atomic direction`) constrains `m_z/|m|` alone, so it is exactly satisfied by every
  in-plane texture, helix and collinear alike. The answer is to **test that the guard
  fires** -- feed it a case that must trip it -- rather than to read a clean zero as a
  pass. It has two siblings, both from the same day: an explanation that *fits* a number and
  is accepted because it fits; and a correction that is **right about a level and is allowed
  to discharge an observation about a trend** -- `int|m|` was shown to be the wrong absolute
  quantity (a vector-norm integral loses weight at the nodes a collinear state has and a
  spiral does not), which retired "the moments are inflating" without touching the separate
  fact that the number was still *climbing* at iteration 77. A constant cannot explain a
  derivative, and the arithmetic said the node effect covered 0.47 of a 3.2 excess in any
  case. (The NiBr2 helix runs, 2026-09-10/11.)
- **A quantity symmetry forces to zero is a residue, and agreeing about a residue
  says nothing about the half symmetry deleted.** The Born charge against `ph.x` was
  one of this project's tightest reference numbers from the day it was written (P24b,
  2026-09-02) to the day it was checked on another crystal (2026-09-16) -- **8e-6** on
  ultrasoft silicon, every printed digit on norm-conserving -- and silicon is
  centrosymmetric, so `Z*` is
  zero there by symmetry and what both codes print is the *symmetric* part alone, the
  sum-rule residue an incomplete basis leaves. The **antisymmetric** part is what a Born
  charge physically is, it had never been compared against anything, and it was wrong by
  **1.4e-2 on a charge of 3.3** the whole time. One polar cell found it in an afternoon
  (`alas-epsilon-us.in`, zincblende AlAs), and the same measurement dissolved a refusal
  that had been sized as a phase: PAW Born charges were refused for a missing one-centre
  term that does not exist, the 1.3e-3 behind the refusal being this error measured on a
  wedge. The habit is to ask, of any reference agreement, **which components of the
  quantity the validation cell allows to be nonzero** -- and silicon, carbon and every
  diamond-structure cell in `tests/data/qe` allow fewer than a reader assumes. It is the
  "a check whose null result cannot be told from a pass" trap one level out: the check
  returned a number, the number was right, and it was right about the wrong half. **Its
  sibling was in the same table**: ultrasoft *carbon* sat at 2.3e-4 where silicon sat at
  8e-6, and that was explained by the radial interpolation floor, because the dielectric
  constant's own error is the same order. The explanation fitted, so nobody looked again;
  the correction moved carbon **2.8e-4 toward** `ph.x`. A residue can show a defect of
  this kind and one of them was showing it the whole time.
  (`PLAN.md` P39a, 2026-09-16.)
- **A sum of N terms that matches one measured total is one equation in N unknowns.** It is
  the most convincing wrong answer this project produces, because the agreement is real and
  the arithmetic is right -- what is unproved is that the terms were ever *simultaneously
  live*. A 45-atom NiBr2 slab's peak was declared explained as
  `resident + temp + guard = 72.16 GiB` against a measured 72.30: three plausible numbers,
  one equation, and the analysis that wrote it had **named the hole itself** ("the overlap
  cannot be proved from a log") one paragraph earlier and then let a later measurement of
  *one* term discharge the whole sum. The A/B refuted it outright -- deleting the guard,
  claimed at 82 per cent of a 28 GB gap, moved the peak by **0.03 GB**, byte-identical per
  iteration. Two sessions then independently built the *next* four-term sum, agreeing to
  0.05 per cent, and both declined to publish it. The rule: a term is on the bill only if it
  was measured, or if removing it is measured to move the total. **An identity that closes
  is not evidence; an A/B is.** (`OPEN.md` Part VII item 1, reopened 2026-09-14.)
- **A method that could not have returned a surprise: finding none is not evidence of
  none.** The sibling of the entry above, and it lives one level further back -- in the
  *design* of the check rather than in the reading of its result. Two sessions confirmed a
  set of `memory_stats()` key names by grepping a binary with a fixed alternation of names
  they had guessed, one asking about six and one about eight; each could only ever return
  names already on its own list, so the two agreeing was agreement about **candidates** and
  said nothing about the set. Reading adjacent strings out of the binary does not rescue it
  either -- the string table is interleaved across translation units. Before believing a
  negative, ask what result would have **falsified** the thing being checked and whether the
  instrument could have produced it: a search that cannot surprise you is not a search.
  (2026-09-14, in passing, while both sessions were being careful about everything else.)
- **And its positive form, which is easier to fall for: a measurement that returns a real
  number the hypothesis did not need.** A cluster run kept dying with
  `INTERNAL: Failed to materialize symbols`, address space rather than memory, and the
  hypothesis was that the Berry-phase string loop recompiles because `npwx` is rebuilt at
  every string. That was *checked*, host-side and with no SCF: `npwx` takes 8 to 11 distinct
  values over these meshes, which is true, is against this project's own rule to pad to a
  common `npwx`, and was written into three files as the cause. It is not the cause.
  Counting `/proc/self/maps` after **each** string instead -- five minutes, the same cell --
  the process gains about 480 mappings on *thirteen of sixteen* strings, including strings
  whose phase takes 0.04 s and which therefore compiled nothing. Ten recompilations cannot
  make 7700 mappings; the growth is per string. The first measurement could only ever report
  how many distinct shapes there were, so it could not tell a per-shape cost from a
  per-string one, and it was cited for a conclusion it had no power to reach. **Ask of a
  confirming number what the alternative hypothesis would have made it read**: here the two
  hypotheses predict the same distinct-`npwx` count and different per-string curves, so only
  the per-string curve is evidence. The same run then overturned two more sentences that had
  been believed on argument -- that `jax.clear_caches()` "does not unmap", when it released
  8924 mappings where `gc.collect()` released none, and that the autouse fixture had failed
  on `test_nonlinear.py`, whose progress line is `...FF........`, two failures with eight
  passes after them. (`OPEN.md` Part XIII item 2, 2026-09-19.)
- **A test asserting a tolerance its own `conv_thr` or iteration budget does not
  deliver.** Four of the six failures in the 2026-09-21 slow pass were this and **none was
  a defect**, which is what makes it a trap rather than a nuisance: each looked like a
  regression, each had a plausible culprit among the day's fixes, and each was reproduced
  **bit-identical at a commit predating all of them**. The quantity being asserted converges
  more slowly than the residual `conv_thr` measures, and by enough to matter: an STM value
  read off a plane is still moving in its **sixth digit** while the density residual is at
  7.7e-13; a constrained noncollinear total energy is **4.31e-07 Ry** from `pw.x` at 1e-11
  and **1.91e-09** at 1e-13, four iterations later; a no-spin-orbit directional degeneracy
  reads 8.1e-2 meV at 1e-10 and 1.16e-2 at 1e-12. The iteration-count form is the same bet
  on a different number: a marginally damped fixed-spin-moment controller took **288, 1380
  and 3380** iterations across changes as small as 3.5 eps, so any fixed budget is a claim
  about a chaotic quantity. **Two habits.** A test on a derived quantity states its own
  `conv_thr` rather than inheriting the input's, and says what it measured there. And
  before attributing such a failure to a recent change, **re-run it at a commit that
  predates the change** -- it costs one worktree and it was decisive all four times, twice
  returning numbers identical to the last digit. The sibling trap is the opposite reading:
  a residue that shrinks with `conv_thr` is convergence, one that **plateaus** is real, and
  only a sweep tells them apart -- the anisotropy case did both, falling by seven and then
  stopping on a floor that a scalar-relativistic dataset then localised to seven orders
  below. **When the old commit passes, it can still be the test**: a bound read off the one
  rounding pattern that happened to land well passes there and nowhere else, and the
  falsifier is the old commit under a dial documented as moving results only at round-off
  (`DEFUMAT_K_BATCH=2`, `DEFUMAT_RADIAL_CHUNK`). Three tests failed on a change that moved
  only the last bit of the radial transforms (2026-10-03, two in `test_piezoelectric.py`, one
  in `test_streamed_strain.py`); master passed all three, and failed all three under such a
  dial -- an AlAs density converged at `conv_thr = 1e-12` to about 1e-8 had been asserted
  cubic to 1e-10.

- **The forecast sentence under a verified table is the one nobody checks.** Its two
  siblings above are about how a number was obtained; this is about *where the scrutiny
  went*. Every figure in the NiBr2 projector-dial entry was checked by two sessions --
  each had units, a baseline, and an obvious way to be wrong, so both attracted it. The
  sentence saying what those figures meant for **a run not yet taken** had none of that,
  read as a summary of work already verified, and was the only sentence a spending
  decision rested on. It claimed the cell was "about 7 GB short" by subtracting the
  eigensolver's 24.89 GB buffer from the headroom -- a term **already inside** the
  measured peak, because the peak is read *at* `<- diagonalize`, after that solve
  allocated it. One session wrote it, retracted it a minute later, and the other had
  already copied it into the record: a verified table lends a forecast credibility the
  forecast never earned separately. Three habits, in order of how much each is worth:
  **check the forecast sentence deliberately**, naming which quantities it treats as
  additive or simultaneous; **prefer a structural argument to an arithmetic one** wherever
  both exist (*"the peak is measured after the solve allocated it"* needs no sum and
  cannot be got wrong by a term); and remember that **a caveat which sounds more damning
  is not therefore safer** -- "7 GB short" was both wrong and more pessimistic than the
  truth, and on a sentence someone spends a day of cluster time against, over-caution
  costs exactly what over-claiming does.

- **A time on a card is not a measurement until the Davidson step count beside it is equal across
  the arms.** One call at the 100-step budget can be most of an SCF, and which executable lands
  there is decided by round-off, so every dial that changes only rounding (the band batch, the
  projector route, the FFT layout, the memory mode) looks like a speed effect. Four recorded
  readings were this and were wrong: a 13x "`conv_thr` tax" on sixteen atoms, a 2.7x gap between
  the memory modes, a factor 2 for `band_batch = 8` at 64 atoms and a 12 per cent loss for the
  fused box layout. The 64-atom memory-mode SCF took 132 s and takes 14.5 s. **The cause was in the
  solver and not in the threshold**: the subspace solve parked its unused directions at 1000 times
  the diagonal, which gave the device `eigh` a matrix of norm 3e4 and an error of 3.4e-12 median in
  the lowest roots (the host's LAPACK: 2.9e-15) against an `ethr` of 1e-13; the parked rows now sit one above a bound of the reduced live
  block (`solvers.subspace.generalised_eigh`'s `parked`; a factor of 4 came first). A floor under
  `ethr` of 3e-12 was adopted first, cut the run to 14.6 s, doubled a force error and was withdrawn
  the same day when a replay of one call on both platforms (the inputs of each machine on the
  other, then the subspace solve moved to the host piece by piece) located it. **The lesson is the
  method**: when one machine takes 3 steps and another 73 on the same call, replay the same inputs
  on both and swap one component at a time; a workaround that fits the symptom, and a "mechanism"
  that fits the numbers (here, slow genuine convergence and false convergence on stagnation), were
  both wrong. `tools/parallel/time_scf.py` prints `davidson_steps`; compare the arms over the
  iterations where they are equal, and run nothing else on the card meanwhile. The CPU does not
  show it, which is why it went unseen. (`PERFORMANCE.md`, "The endgame on a card is a stall",
  2026-10-01.)

- **A loop over a closure, called outside any `jit`, compiles again at every call.** `map_k`,
  `sum_k`, `lax.map` and `lax.scan` called eagerly with a function defined inside the caller trace
  that function anew each time, so a driver that calls such a helper once per k-point and per
  iteration compiles the loop every time, and with the persistent cache on each recompile is a cache
  hit that still loads a new executable the process keeps for good. Two sites, the same night: the
  ultracell's matrix and densities (2x on three iterations of a small cell) and the Berry-phase
  string loop, whose growth in `/proc/self/maps`, +256 a polarization call, was the cluster's
  "Failed to materialize symbols" (`OPEN.md` Part XIII item 2). Jit the helper with its shapes
  static and its arrays as arguments. **The third site was the whole response stack**, whose
  closures carry a callable (a perturbation) and so have no static argument to key on: a second
  dielectric tensor on one-k-point silicon compiled 127 programs. Such a site goes through
  `defumat.eager.compiled`, which keys on the traced structure with the arrays hoisted, and a
  `jax.jvp` at the top level is wrapped whole so the kept program is the derivative itself.
  **The check is a second call that compiles nothing**:
  `jax_log_compiles` with a handler on the `jax` logger, which sees them where `jax._src.dispatch`
  does not, validated on a cold run first. (`PERFORMANCE.md`, 2026-10-02.)
