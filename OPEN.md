# Open items, carried forward

What is known to be wrong or missing and is **not this-session work**. Each entry
says what to do about it, what it costs, and how to know it worked. The evidence and
the numbers stay in `PLAN.md` §3; `GAPS.md` is the separate list of what a *user* can
ask for and not get, and these are mostly internal.

**Part I** is the three items carried out of the 2026-09-10 regression run and the
NiBr2 staging on Triton. Two are recorded with their numbers in `PLAN.md` §3, in the
P73 section, under "Three test failures were seen while validating this phase".

**Part II** is the sweep of **2026-09-11** -- eight read-only agents over the package,
28 findings, ranked by what a wrong answer costs rather than by what it costs to fix.

**Part IV** is the **2026-09-13** magnetism session (`PLAN.md` P80): two entries, neither a
defect in this code -- a **JAX thread-pool deadlock** that the fast gate now hits
reproducibly, with a stack trace and the measurements that rule out the obvious causes, and
P63's spiral scan no longer reproducing its own numbers. **Item 1 was reopened and then
closed later the same day**: it hung a sixth time, and the cause is neither of the two the
entry had recorded as fact but the **affinity mask this package sets itself** -- 8 hangs in
8 runs at two cores, 6 in 14 at four, none at eight or above.

**Part VII** is from a peer session running the **NiBr2 helix on Triton**, reported
**2026-09-14**: four findings, two of them defects that were fixed the same day (the
Davidson finiteness guard's allocation, and a checkpoint refusal that was wrong on both
sides of its boundary at once). Three entries are carried: a `sizing.py` report that looked
60 per cent low on a 45-atom spinor PAW slab -- **reopened 2026-09-14**, since the A/B
that identity predicted withdrew the closure and the 28 GB is unexplained again -- a
Davidson inner-step count that may degrade at the minimum subspace, and the checkpoint's
remaining Hubbard refusal, which was as wide as the field's had been and is **closed
2026-09-15**.

**Parts VIII and IX** are from the workstation, **2026-09-14** and **2026-09-15**. What is
left open in them is the two that are measurements rather than fixes: `local-TF`'s 730 s
an iteration, unprofiled, and the `pp.x` reference pair P90 owes. The two defects are
closed -- a resume that spent a whole Davidson budget on bands nothing reads (Part VIII
item 3, **closed 2026-09-15**), and a bias window integrated with no check that the axis
resolves the broadening (Part IX item 1).

**Part V** is from the **2026-09-13** memory session: one entry, and it is not that
session's work -- four of `test_magnons.py`'s eight tests fail, all four downstream of a
ground state that stops four orders short of its own `conv_thr`, and the same input gives
the identical energy and accuracy at the commit before that session started.

**Part III** is the sweep of **2026-09-12** -- four read-only agents over the package
looking for **speed and memory** rather than for wrong answers, 23 entries, ordered by
ease times impact. **Nothing in it was measured and nothing in it is a defect**: each
entry is a reading of the source with the input and the command that would turn it into a
number, and none of it enters `PERFORMANCE.md`'s backlog until one does. The vendored QE
tree was absent while it ran, so its QE claims are from this repo's docstrings.

**Status, 2026-09-11 (later the same day).** Sixteen entries are closed, each with a
test that was checked to fail against the old code: **A1, A3, A4, A5, A6, A7, A8, A9,
A10, B2, D2, E1, E2, F1, F2, F3**, together with Part I item 1 and its two siblings
C2/C3. What was left after that day -- D1, D3, E3 and Part I item 2 -- was the set whose
*test* is expensive rather than whose fix is, and **all of it except Part I item 2 closed
on 2026-09-12**, together with A2, B3, B1, C1 and G1, which was opened and closed the same
day. **So one entry of Part I and Part II together is open: item 2**, the 11,088 MB peak of
`test_spinorbit.py`, whose first step is a measurement rather than a fix -- and whose
number predates P74's band-batching fixes, which should have moved it (`PLAN.md` P74,
"What is outstanding"). None
went the way the sweep predicted: A2's two non-refusal sites looked like a null
and are not; B3's NaN claim is a null while its other two
hold -- the absolute threshold turned out to be wrong in *both* directions at
once; and **B1 is the sweep's worst entry and its most useful** -- the guard
needed changing and all three of the entry's specific claims are false, the
route, the observable and the test it proposed, each for a reason worth more
than the fix. Each entry says which. Each closed entry is marked
**[closed]** below with what the fix turned out to be, because two of them turned out
not to be what the sweep predicted.

**Two corrections to the sweep itself, both worth more than the fixes.** A8 is a
**measured null**: JAX 0.11.1's complex `abs` has a finite derivative at exactly zero, so
`abs(rho_g)**2` was never the NaN predicted -- the expression was changed anyway, being
the package's convention, and the test now records which it was. And the ranking in this
file is by *cost of a wrong answer*; the order the entries were actually worked in is by
*quickness and testability*, which puts A1 and A2 -- the top two here -- in the middle
and at the end respectively.

---

# Part I -- from the 2026-09-10 regression run

---

## 1. The Kramers bound asserts round-off where the solver promises `empty_ethr` **[closed 2026-09-11]**

`tests/regression/test_spinorbit.py::test_kramers_degeneracy_survives_spin_orbit`
fails on `spinorbit.in` (6.5e-5 eV) and `spinorbit-pbe.in` (7.7e-5 eV) against a
bound of 1e-6 eV over every band. **The code is right and the test is wrong.**
Every band carrying weight is degenerate to 5e-12 eV; the whole failure is in
three bands whose occupation is 1e-87, more than 10 eV above `E_F`, which
`cegterg.f90:129` converges only to `empty_ethr = MAX(5 ethr, 1e-5)` Ry. With
`diago_full_acc = .true.` the same case gives 2.6e-11 eV.

**What to write.** Not "assert over the occupied bands only" — that gives up the
guard exactly where a non-Hermitian `D` or a mispaired spin block is least likely
to be noticed, which is the whole reason the test exists. Two bounds, matching
what the solver actually promises:

- pairs with `wg/wk >= 0.01` (QE's own `btype` rule, `sum_band.f90:118-128`):
  degenerate to round-off, and 1e-6 eV is the right number — measured 5e-12;
- every other pair: degenerate to `empty_band_threshold(ethr)` converted to eV,
  which at the floor is 1e-5 Ry = 1.36e-4 eV — measured 6.5e-5.

Take the occupations from the `SCFResult` the test already has rather than
re-deriving which bands are empty.

**Why it is not a one-liner.** It changes a P14 claim, so the commit that makes
the change says so and P14's entry in `PLAN.md` gains the two numbers. Worth
checking in the same pass whether any *other* test asserts an eigenvalue
degeneracy or difference over all bands without splitting occupied from empty --
the P10 continuation of 2026-09-08 wrote down that empty eigenvalues can now move
by `max(5 ethr, 1e-5)` Ry, and this is the first test found to have been written
against the old behaviour. It is a **slow** test, so the push gate could not have
caught it, and neither could anything else written the same way.

**Cost.** Minutes to write; about 2 minutes to verify, since only the two
non-PAW cases are involved:

```bash
python3 -m pytest tests/regression/test_spinorbit.py \
    -k test_kramers_degeneracy_survives_spin_orbit -q
```

> **Done** (`0e1b20c`, and the test change before it). The single bound over every
> band is replaced by the two the solver actually promises: pairs with
> `wg/wk >= 0.01` to 1e-6 eV -- measured 5e-12 -- and every other pair to
> `empty_band_threshold(ethr)` in eV, 1.36e-4 at the floor, measured 6.5e-5
> (`spinorbit.in`) and 7.7e-5 (`spinorbit-pbe.in`). `diago_full_acc = .true.`
> gives 2.6e-11 eV over *every* band, which is the check that the split is the
> threshold rather than the physics. P14's entry in `PLAN.md` carries the three
> numbers; the sweep for other tests asserting a degeneracy over all bands is
> C2/C3 below.

---

## 2. `test_spinorbit.py` peaks at 11,088 MB against a 12 GB cap **[closed 2026-09-13 at 6,708 MB -- but the diagnosis took two wrong turns first]**

92% of the cap on a 30 GB machine, so it is the next out-of-memory kill whether or
not it has happened yet. The watchdog named
`test_spin_orbit_total_energy[spinorbit-paw.in]` at 10,818 MB; **its assertions
passed** -- the memory is the finding, not a failure. For scale, the next file
down in the same run is `test_stress.py` at 6,317 MB.

**The number predates P74 and should have moved.** It was measured with the whole band
block in the FFT box, before `vloc_psi_nc` learned to call `map_bands`, so the spinor local
term was transforming every band at once in exactly the file this peak belongs to.

**The one lead, and what it is not.** The stderr shows XLA constant-folding and
transposing an `f64[25,1277,34,34]` inside `jvp(jit(_paw_onecenter))`, twice,
each fold over 2 s. That shape is `PawSpecies.density_ae`/`density_ps`,
`(nh, nh, nlm, mesh)` at `nh = 34` for a fully-relativistic platinum dataset --
295 MB each. They are ordinary pytree fields, so inside `_paw_onecenter` they are
arguments; appearing as XLA *constants* means the enclosing `jit(<lambda>)` closes
over the object holding them, which would give every compiled variant its
own copy in a cache that never shrinks. **But 590 MB is under 6% of an 11 GB
peak** -- so the entry dismissed it, and **that dismissal was wrong**; see below.

---

### The measurement, 2026-09-13, and it answers a different question than it asked

**The peak depends on whether the compiled kernels come from the on-disk cache, and
nothing else moves it nearly as much.** Same commit, same test, alternating, main tree,
`DEFUMAT_THREADS=4`, `/usr/bin/time %M`:

| `DEFUMAT_CACHE_DIR` | peak | wall |
|---|---|---|
| `off` -- compiles every kernel | **10,111 M**, then 10,115 M | 57.0 s, 57.2 s |
| default -- loads them | **16,406 M**, then 16,381 M | 39.3 s, 36.6 s |

Reproduced in a second checkout (10,078 / 10,071 against 16,364 / 16,380) and **at eight
cores as well** (16,470 / 16,440), so the affinity mask is *not* the variable here even
though it is the variable for the deadlock in Part IV.

**The mechanism, and it reverses this entry's own dismissal.** Pointed at an empty cache
directory, this one test writes **375 entries totalling 606 MB -- of which a single
`jit_<lambda>` entry is 602.8 MB**, which is the enclosing lambda the paragraph above
identified. Then:

| | peak | wall |
|---|---|---|
| miss (compile, then write the cache) | 10,120 M | 62.6 s |
| hit (load it back) | 16,383 M | 37.4 s |

**6,263 MB of resident memory for a 603 MB blob: a factor of 10.4.** So it is not that the
serialized executable is held -- the gap is larger than the whole 4.8 GB cache directory --
it is that deserialising it expands by an order of magnitude. "The constant is the memory"
was the right instinct; what made it look like 6% was counting one copy of the constant
instead of what loading it costs.

The effect is not a fixed factor. The same off/on pair on a `si8-paw-1k` force is
**1,082 M against 1,212 M**, 12% and 130 MB, so it scales with what the executable
embeds -- which on this platinum cell is a fully-relativistic dataset at `nh = 34` and on
silicon is nothing of the kind.

**The entry's two candidates are both answered.** It is **one test, not accumulation**:
that single test run alone peaks at 16,392 M where the whole 27-test file peaks at
16,378 M. And `jax.clear_caches()` is not the lever, because the file's peak is one
backward pass.

**What this costs today.** With a warm cache the file is **SIGKILLed under
`run_regression.sh`'s 12 G cap at any thread count** -- it was, at 12,394 M, before the cap
was raised. Until the constants are fixed it needs `DEFUMAT_TEST_MEM_MAX=18G`, or
`DEFUMAT_CACHE_DIR=off` at 18 s a run. The 11,088 M on record is neither figure: it is
whatever the cache happened to hold that day.

**Two methodological findings, and the second cost an hour.**

- **A memory measurement here must state the cache state**, exactly as a timing must.
  `CLAUDE.md` already says never to time a first call because it measures the cache; the
  same sentence is true of `%M` and points the *other* way -- a cache miss is **cheaper**
  in memory and dearer in time.
- **A bisection across commits is a bisection across cache states.** Walking
  9a1cc3d -> ad4b599 -> 3c5d779 -> HEAD gave 10,178 / 10,137 / 16,382 / 16,357 M and
  looked exactly like a regression introduced by the A5 structure-factor remat. It is not:
  every *new* code state is a cache **miss** and therefore cheap, and re-running ad4b599
  warm gives 16,386 and 16,405. A5 is innocent, and so is the affinity mask, which the
  same artefact had made look like the difference between passing and being killed.

### The fix, and it was two things rather than one

The 602.8 MB executable is the **stress** gradient -- `run_scf` -> `compute_stress` ->
`autodiff_stress` -- not `_paw_onecenter`, whose own cache entry is 0.1 MB. Its constants
were **1062.4 MB in 94 entries**, and the force gradient's were 694.3 MB in 45. Three
arrays were almost all of both, and they needed different fixes:

- **`BecsumSymmetry.operators`, 489.4 MB, `(nsym, nh, nh, nh, nh)`** -- an outer product
  `einsum("sik,sjl->sijkl", single, single)` of a **444 kB** factor, kept because its
  docstring said *"`nh` is a few for every element that exists, so the tensor is small"*.
  `nh` is 34 here. It is no longer built at all; the six contractions that used it apply
  the factor twice, once per channel index of `becsum`. This also takes 489 MB off the
  **host** at setup, which no amount of argument-passing would have done.
- **`PawSpecies.density_ae`/`density_ps` (281.6 MB each) and `AugmentationCharge.qgm`
  (120.9 MB)** are now **arguments** rather than closure captures
  (`defumat/forces/energy.py`'s `HOISTED_FIELDS`). Both are `equinox` modules, so they
  cross as ordinary pytrees.

| | before | after |
|---|---|---|
| stress gradient constants | 1062.4 MB | **10.1 MB** |
| force gradient constants | 694.3 MB | **10.0 MB** |
| cache this one test writes | 606 MB, one 602.8 MB entry | **4.4 MB**, largest 2.1 MB |
| the test's peak, cache warm | 16,383 M | **6,232 M** |
| the test's peak, cache off | 10,111 M | 6,584 M |
| the **whole file** (27 tests) | 16,378 M, 5m15s -- killed at a 12 G cap | **6,708 M, 4m43s** |

**So the entry closes at 6,708 MB against the 11,088 MB it opened at**, 55% under the cap
rather than 92% of it, and the cache-state bimodality goes with it -- warm and cold now
differ by 6% where they differed by 62%. The stress agrees to **12 significant figures**;
the residual is the different contraction order. Gate: 1945 passed, 176 skipped, 0
failures, 9m54s.

**What is still open** is `MEMORY-AUDIT.md` A12, which asks for `density_ae`/`density_ps`
to be *factored* rather than merely passed: they are themselves a rank-1 outer product, 553
MB per Ni species, and passing an array as an argument does not make it smaller. And the
shape is worth looking for elsewhere -- both arrays removed here were an `einsum` writing a
product of two indices the consumer immediately contracts away.

---

## 3. A single SCF is not restartable, only a relaxation is **[closed 2026-09-11]**

`run_scf` takes no checkpoint argument at all -- confirmed against its signature,
not remembered -- so the only way a ground state reaches disk is
`result.save(path)` **after the driver returns**. That covers the case P67 was
written for (a converged state handed to a later job, or to a second process) and
the case where the run stops at `max_iterations`, which still returns a result.
It does **not** cover the one a cluster user actually meets: a job killed at its
wall clock mid-SCF loses every iteration it ran. `run_relax(checkpoint_dir=...)`
writes the state, the geometry and the BFGS history every ionic step, so an
ionic loop already survives what an electronic one does not.

**Found from the outside**, by the NiBr2 session staging a production run of the
45-atom helix on Triton: this end told it a wall-clock kill was cheap because
"the SCF checkpoints", and it checked the script instead of believing it. The
consequence it drew is the right one and is what a user has to do today -- ask
for four hours rather than a tight backfill-friendly limit, because there is no
cheap way to be wrong about the length of the run.

**What to write.** A `checkpoint_dir`/`checkpoint_every` on `run_scf`, writing
the same `save_state` payload from inside the iteration loop and reading it back
through `starting_from`, which is machinery that already exists and is already
field-covered by `unhandled_fields()`. The loop is Python (the convergence test
is data-dependent), so there is no `lax` boundary in the way. Two things to get
right rather than discover: the write is the wavefunctions and so is not free --
it is the same array that dominates the checkpoint file -- so the interval is a
knob and not every iteration; and a resumed run must re-enter with the *density*
mixing history it left, or it pays back the iterations it saved, which is exactly
the trap P67 already solved on the BFGS side and asserted with "2 + 4 steps, not
2 + 6".

**Cost.** Small, and the test is P67's own pattern one level down: stop an SCF at
iteration `n`, resume, and assert the total iteration count matches the
uninterrupted run rather than merely that it converges.

> **Done, and the assertion above is exactly right -- but only once `ethr` crosses
> the file.** `run_scf` takes `checkpoint_dir`/`checkpoint_every` and resumes from
> the same directory the way `run_relax` does, so a resubmitted sbatch continues
> rather than starting over. A restart is **three** things, not two: the state,
> the mixer's history (`save_mixer`/`load_mixer`, policed by
> `unhandled_mixer_fields`), and the *loop state* -- `iter`, `dr2` and `ethr`,
> which is precisely what `save_in_electrons.f90` writes. `next_ethr` is indexed
> on the iteration number, so a resume that re-enters at 1 with a fresh threshold
> converges on a different schedule: 5 + 8 = **13 iterations against 17
> uninterrupted** on silicon at `conv_thr = 1e-12`, which looks like a saving and
> is a different calculation. With all three carried the count is **exact** --
> 17/17, 13/13, 11/11 at three mixing parameters, energies agreeing to 1.8e-15 Ry.
> `max_seconds` is QE's `check_stop_now`: the loop stops itself and writes on the
> way out, so the wall-clock case needs no signal handler.
> `tests/unit/test_scf_restart.py`.

---

## Neither of the first two is P73

Both reproduce on a worktree at `e22aa7d`, the commit before the P73 augmentation
work started, and item 1 reproduces there to four significant figures. The stress
`DID NOT WARN` failure from the same run is likewise pre-existing and is settled
in `PLAN.md`; it is not repeated here because it needs no decision, only a fix to
P11's refusal.

---

# Part II -- the sweep of 2026-09-11

Eight read-only agents over disjoint lenses: stale refusals, dropped options, the
recurring trap list, the regime/consumer combination matrix, the 144 commits since the
last audit, tests that cannot fail, memory and inert batching dials, and numerical
guards. **Read-only by construction** -- no agent ran a test, a notebook or an SCF, so
every claim here comes from reading the source and none of it is a measurement. That is
the standing caveat on the whole part: each entry names the wrong answer it predicts,
and none of those numbers has been produced.

**Verification.** Entries marked **[opened here]** were read in the source in the main
loop, not only by the reporting agent. The rest carry the agent's `file:line` and are
specific enough to act on and specific enough to be wrong -- check before building on
one, the same rule `GAPS.md` states.

**Excluded.** The agents were handed `PLAN.md` §3's outstanding index, `GAPS.md` §3,
`HOLES.local.md` and Part I above, and told that a known item counts only if the
*recorded claim itself* is wrong. §F is what came back on that filter.

---

## A. Wrong answers with no error

The class this project weights highest, because a wrong number that looks right is
worse than a crash. Ranked.

### A1. `STARTING_MOMENTS` starts no moments **[closed 2026-09-11]**

> Both halves. `domag` is now `System.is_magnetic`, which reads `local_moments`
> (the card already folded in) and the `LOCAL_MAGNETIC_FIELDS` card -- the same
> rule the k-point reduction and the symmetry group were already using, which is
> A6. And `starting_charge` takes a per-atom weight, which costs one weighted
> structure factor and no new radial transform: `_weighted_structure_factors` is
> `structure_factors` with the atoms *weighted* instead of *counted*, checked
> against a brute-force `sum_a w_a exp(-iG.tau_a)` to 0.0. Three tests in
> `tests/unit/test_textured_symmetry.py`, all three verified to fail before.
>
> **One residual, and it is new**: a **PAW** dataset's atomic `becsum` is still
> split by the per-*species* `starting_magnetization`, so with the card the
> charge's starting texture and the one-centre starting occupations now disagree
> at iteration 1 -- which is exactly what `spin_weights`'s docstring says must
> not happen. The SCF repairs it and nothing is wrong at convergence; it costs
> iterations. Lifting it is a per-atom `_becsum_split`, with `starting_becsum`'s
> `broadcast_to` over `len(atoms)` replaced by a per-atom stack. The test cell
> is ultrasoft (`rrkjus`, `paw is None`, `becsum` starts at zero), so nothing
> here sees it.
>
> **Not verified on this machine**: the entry's own how-to-know -- a two-atom
> antiferromagnet converging to `|M| > 0` per site and zero total. What is
> asserted is the *starting* density (opposite lobes, zero cell total) and the
> two structural consequences, which is the same claim one step earlier.

`defumat/system/builder.py:259` and `scf/driver.py:3005`. `domag` is
`any(abs(m) > 1e-6 for m in self.starting_magnetization)` -- the per-*species* array.
The per-atom card that `io/pwin.py` parses, that `builder.py:1375` documents as
overriding all three of `starting_magnetization`/`angle1`/`angle2`, and that
`builder.py:1520` validates, never enters it.

Two consequences, and the second survives fixing the first:

- A noncollinear input whose texture is given **only** through `STARTING_MOMENTS` gets
  `domag = False`, hence `nspin_mag = 1`, hence a nonmagnetic run. It converges, it
  reports a total energy, and it never had a magnetization.
- With `starting_magnetization` *also* set so `domag` is True, the starting density is
  still built from the per-type `magnetization_directions` (`angle1`/`angle2`) and the
  per-type magnitudes. `starting_moments` reaches only the constraint field
  (`driver.py:1743`, `scf/fields.py:179`). A card documented as a starting magnetic
  texture seeds no texture -- it starts the per-species ferromagnet and then penalises
  it toward the texture.

**What to write.** `domag` takes the per-atom moments into account, and the starting
density is built per *atom* where the card is present. The per-atom path already exists
on the field side; what is missing is the same `per_atom` argument reaching
`starting_charge`'s three component sums.

**How to know it worked.** A two-atom antiferromagnetic cell given only
`STARTING_MOMENTS` with opposite moments must converge to `|M| > 0` on each site and
zero total -- today it converges to the nonmagnetic state. This is directly under the
helix work, which is the reason it is first.

### A2. The Sternheimer stack rebuilds its potential from the **input** magnetic field **[closed 2026-09-12 -- the refusal half, plus two real fixes]**

> **The refusal is one line in one place**, because
> `require_a_sternheimer_regime` is the funnel all seven entry points share
> (`efield`, `strain`, `phonon`, `electrostriction`, `nonlinear`, `piezo`,
> `make_sternheimer` -- and `phononq` through the last of those). It is placed
> *above* the `metals` and `spin_polarized` checks on purpose: five of the seven
> pass neither flag, so a smeared magnetic run would otherwise have been refused
> with a message about metals and the field never mentioned. Threading the
> converged pair is what a field put in by hand needs and it is plumbing -- the
> induced `2 lambda dm` term then appears on its own, since `_field_potential`
> is `jax.grad` of the penalty and the induced potential is one `jvp` of
> `potential`. `constrained_magnetization = 'fsm'` needs more: its field is a
> feedback update, so `dB/drho` is not a derivative of anything.
>
> **The two sites outside that stack were threaded rather than refused**, and
> they turned out *not* to be the null they looked like.
> `velocity.py:band_velocities` and `workflows/conductivity.py` now pass
> `SCFResult.magnetic_field` and `.field_scale`. The field enters `v_scf` as a
> **local** potential and `dH/dk` at a frozen sphere cannot see one -- that much
> is exact, measured at 0.0. But `hamiltonian` rebuilds `deeq` from `v_scf` on
> every call and `deeq` multiplies `vkb(k)`, so on an **ultrasoft or PAW**
> dataset the local potential reaches the velocity through the nonlocal term:
> **0.37 out of 398 Ry bohr** for an arbitrary bump on `si2-us`, against exactly
> zero on `si2-nc-force`. Every band velocity, optical conductivity and
> anomalous Hall number of a soft magnetic run was built from the input field.
> `tests/unit/test_velocity_locality.py` holds both halves.
>
> **The third group the entry asked about is clean**, and that is a stated
> negative rather than an unchecked one. `projwfc`/`pdos`, `stm` and
> `transport` build no potential of their own: each routes through
> `fixed_density_states`, which already refuses a field it was not handed
> (`nscf.py:193`), and each already forwards `result.magnetic_field`.
> `workflows/shg`, `photocurrent` and `tddft` take no field argument at all, so
> the same refusal stops them. The forces refuse a field outright
> (`forces/energy.py:reject_magnetic_field`).


`calculation.potential(<density>)` is called with **no field argument at ten sites
across the response stack** -- `efield.py:280` and `:524`, `phononq.py:462` and `:951`,
`velocity.py:450`, `sternheimer.py:1199`, `electrostriction.py:299`, `:331`, `:447` and
`:607` -- so each falls back to `self.magnetic_field`, the field the *input* asked for.
This is `GAPS.md` §0's `DFTSource` defect one layer over, in a place the 2026-09-01 fix
did not look, and it is wider there than it was here.

Converge an `nspin = 2` run with `B_field` along z under Elk's `reducebf` or the
fixed-spin-moment scheme, so `SCFResult.field_scale` is ~0.07 and
`SCFResult.magnetic_field` is not the input's. Then `get_dielectric_tensor()`,
`get_phonons()`, `get_strain_response()` or `get_born_charges()`:
`calculation.potential(density)` falls back to `self.magnetic_field` and rebuilds
`v_scf` with the **full-strength input** Zeeman term. Every eigenvalue entering the
solve, and the screened tensor on top of it, is evaluated under a field the ground
state was never converged under. No error, no warning, and the tensor is still
symmetric and positive -- the exact signature §0 describes.

**What to write.** The same fix as §0: the converged field crosses with the density
rather than being re-derived from the input. `grep -rn '\.potential(' defumat/` is what
finds every site -- run it over `projwfc`, `stm` and `transport` too, since §0 was the
topological consumers and this is the response ones, and nobody has checked the third
group.

### A3. Four post-SCF consumers do full-sphere sums on half-sphere storage **[closed 2026-09-11 -- the refusal half]**

> The honest first move, as this entry proposed. DFT+U is substituted away in
> `gamma_storage_is_consumable` (it is the one that is wrong *inside* the SCF);
> the other three refuse by name through `refuse_gamma_storage`. **One thing the
> entry did not say:** the guard must ask `gamma_storage_is_consumable`, not
> `kpoints.gamma_only` -- an ultrasoft gamma run has already been substituted to
> the full sphere, and reading the input's flag would refuse a run whose states
> are fine. The `2 Re(sum) - G0` rule at the four sites is still open.

`gamma_storage_is_consumable` (`scf/driver.py:584-616`) checks four things --
`kpoints.gamma_only`, ultrasoft, `nspin == 4` or a spiral, and `nosym`. It says nothing
about what happens to the states *after* the SCF, and `gamma_only` appears nowhere in
`hubbard/`, `projwfc/`, `tddft/` or `basis/sample.py`.

Under `K_POINTS gamma` every plane-wave sum must be `2 Re(sum)` minus the `G = 0` term,
and `CLAUDE.md` says exactly three things carry the trick. These four do not:

- **`defumat/hubbard/occupations.py:74`** -- the worst of them, because it is *inside*
  the SCF rather than after it. A norm-conserving `nosym` DFT+U run at gamma keeps the
  half-sphere storage (DFT+U is on none of the substitution branches), and `ns` is built
  from a plain half-sphere inner product. The Hubbard potential and energy follow it,
  and the SCF converges silently to the wrong ground state.
- **`defumat/tddft/chi0.py:479`** -- the sum-over-states `chi_0` transforms with the
  full-sphere `g_to_r`, so an optical spectrum of a molecule in a box, which is the
  archetypal gamma case, is built from wrong real-space states.
- **`defumat/projwfc/projections.py:270`** -- `<phi|S|psi>` as a plain sum, so a Löwdin
  charge is roughly a quarter of its true value.
- **`defumat/basis/sample.py:160`** -- `psi(r)` as a bare sum over the stored `k+G`
  list, so the vertical-transport tunnelling amplitudes lose the conjugate half and gain
  a spurious imaginary part.

**What to write.** Either the `2 Re(sum) - G0` rule at each site, or -- cheaper and in
keeping with how the gate already treats ultrasoft -- extend
`gamma_storage_is_consumable` to refuse DFT+U, and have the four post-SCF consumers
check `planewaves.gamma_only` and refuse by name. The refusal is the honest first move;
the rule is the feature.

**How to know it worked.** Every one of the four has a full-sphere sibling: run the same
cell as `K_POINTS gamma` and as an explicit single k-point at the origin, which
`_without_gamma_storage` already documents as an *exact* substitution, and the two must
agree to round-off.

### A4. Four symmetry input variables are read by nothing **[closed 2026-09-11]**

> Four entries in `_REFUSED_SWITCHES`, which already existed. **The caveat this
> note left open is now closed (2026-09-12), on a machine that has the vendored
> tree.** It was right: `tests/regression/test_input_sweep.py` sweeps QE's own
> `pw_*` inputs expecting each to run or to hit a *listed* refusal, two of the
> 252 set one of these four, and both failed the sweep for the whole day between
> the refusals landing and the entries being written -- `scf-allfrac.in`
> (`use_all_frac`) and `scf-nofrac.in` (`force_symmorphic`), now declared in
> `EXPECTED_REFUSALS`. The other two switches, `no_t_rev` and `nosym_evc`, are
> set by **none** of the 252, so they have no entry rather than an unused one.
> (`pw_gau-pbe/gau-pbe-si444.in` looks like a third and is not: its
> `force_symmorphic` is commented out.) 250 pass, 2 skip.

`no_t_rev`, `force_symmorphic`, `use_all_frac`, `nosym_evc`: **zero** occurrences
anywhere in `defumat/`, including `io/`. They parse into the namelist without complaint
-- `io/pwin.py` has no whitelist of known variables and raises only on a line it cannot
parse structurally -- and are then never looked at.

A user sets `no_t_rev = .true.` precisely when the operations carrying `t_rev = 1` are
*not* symmetries of their state. Here `t_rev_array()` keeps them, the k-mesh is reduced
with them, and `sym_rho` averages the magnetization over them. The result is a wedge sum
and a symmetrised density built from operations the run was told to exclude, with a
total energy that will not match a benchmark generated from the same file and nothing
saying why. `force_symmorphic` is the same silence over the fractional-translation half
of the group.

**What to write.** Consume them, or refuse them by name at the input boundary. Refusing
is minutes and is the right first move -- this file's whole rule is that a run which
starts is a run whose physics is there.

### A5. `sqrt(sum(m**2))` is unguarded in two differentiated paths **[closed 2026-09-11]**

> Confirmed by measurement: the old form gives `[nan nan nan]` at a bit-exact
> zero and the new one `[0. 0. 0.]`, with the value unchanged. One function,
> `xc/functional.safe_modulus`, shared by both sites -- they were the same three
> lines and the same defect written twice.

`defumat/xc/functional.py:804` and `defumat/paw/gradient.py:191`. Both read
`modulus = jnp.sqrt(jnp.sum(magnetization**2, axis=0))`, and in `functional.py` the
`safe = jnp.where(modulus > 0.0, modulus, 1.0)` on the **next line** guards the division
that follows and not the sqrt's own argument. `d|m|/dm = m/|m|` is `0/0` at an exact
zero, so `jax.grad` through `v_of_rho` -- which is every spinor force, every spinor
stress, and every `jvp` in the response stack -- returns NaN there. The primal survives,
which is why P46's small bulk cells never saw it.

This is trap 1 at its sixth and seventh site, and `CLAUDE.md` already names this exact
shape ("`|m|` in the gradient correction differentiates through its own nodes").

**The plausible half is whether a bit-exact zero occurs.** Two mechanisms that produce
one: `sym_rho`'s axial-vector average at a grid point whose magnetic little group admits
no invariant axial vector (`m + (-m)` with ±1 rotation entries is exact), and a vacuum
region where the density underflows. **How to know it worked** is therefore a *test*
that forces the zero rather than a run that happens not to hit it.

### A6. Three different rules for "is this run magnetic" **[closed 2026-09-11]**

> `System.is_magnetic`, used by `domag`, `_respin_kpoints`, `_recelled_kpoints`
> and `build_system`. Closed together with A1, which is the same defect.

`defumat/system/builder.py:586`, and four call sites. The consequence named is that the
SCF can symmetrise the density with a larger group than the one its k-set was reduced
with -- a wedge sum completed against the wrong group, which is trap 4 arriving through
a bookkeeping disagreement rather than through a response. Related to A1, which is one
of the three rules.

**What to write.** One property on `System`, the way `nspin`/`npol`/`nspin_mag` are
already exposed so no call site recomputes the rule.

### A7. The Broyden mixer packs `ns` complex and unpacks it on a hardcoded dtype test **[closed 2026-09-11]**

> *PLAUSIBLE* resolved to real: three of the four precision cases were wrong, and
> the parametrised round-trip test fails on the old code for both float32 ones.

`defumat/scf/driver.py:408`. An unconditional `.view(float)` on the way in, a
`!= np.complex128` test on the way out. In float32 mode a complex `ns` is packed as
reinterpreted bits and unpacked as real -- silently garbage, not an error. Also a
hardcoded-dtype violation of the standing convention, which is what makes it findable.
*PLAUSIBLE: the float32 path was not exercised.*

### A8. The Hartree energy uses the banned `jnp.abs(rho_g) ** 2` **[closed 2026-09-11 -- as a null]**

> *PLAUSIBLE* resolved to **no**. `jnp.abs` of a complex number has a finite
> derivative at exactly zero in JAX 0.11.1 -- measured as 0 in reverse mode, in
> forward mode and in the Hessian -- so this was never a NaN. Changed anyway, on
> the convention; the two agree to 1.9e-16 in the energy and 2.8e-16 in its
> gradient, and the test asserts the *old* form is finite so it fails if a
> future JAX changes that rule.

`defumat/scf/potential.py:111`. Every other differentiated site in the package uses
`Re(conj(rho) rho)`. This is the one term that every force, every stress and every
phonon differentiates, and `abs` at a forced zero is trap 1 -- the reciprocal Ewald sum
is the recorded instance and a structure factor vanishing exactly is what symmetry
arranges on a supercell. *PLAUSIBLE: whether `rho_g` reaches an exact zero was not
established.* Cheap to change regardless, and the change is a no-op where it is safe.

### A9. The transport band-count diagnostic is rule-D4 basis-dependent **[closed 2026-09-11]**

> The topmost *multiplet*, in the same channel basis the denominator is taken in
> -- which is also the better diagnostic, since a truncation at `nbnd` cuts the
> multiplet rather than one member of it. The test rotates a degenerate top pair
> by a random unitary and asserts the old form moves by more than 1e-3.

`defumat/workflows/transport.py:448`. `band_edge_weight` is built from the raw diagonal
of the single topmost band, so it is basis-dependent whenever that band sits in a
degenerate multiplet -- and it is divided by a denominator taken in a different
(channel) basis. The check that certifies the truncation is itself the thing rule D4
says cannot be trusted band by band. Take the multiplet block average.

### A10. `get_relax` drops every `&electrons` option it adopted **[closed 2026-09-11]**

> Six options named in all three relaxation drivers, `None`-defaulted so that
> "not given" stays distinguishable from "given the default" and `run_scf`'s own
> numbers are not repeated. A signature test keeps the set from drifting.

`defumat/calculator.py:671`, and the same for `get_relax(variable_cell=True)` and
`get_spiral_relaxation`. `electrons_defaults`/`_ELECTRONS_OPTIONS`
(`calculator.py:195-212`) adopt `electron_maxstep` (as `max_iterations`),
`diago_full_acc` and `mixing_fixed_ns` from the input file's own namelist; the three
relaxation entry points then discard them through `**scf_options`. Every SCF inside the
relaxation runs at the default 100 iterations with loose empty states and no fixed-`ns`
warm-up.

This is the `SHARED_OPTIONS` rule not being followed in the one place the facade
actually holds state worth forwarding. On a DFT+U relaxation the dropped
`mixing_fixed_ns` decides which minimum of the +U functional the run lands in, so the
relaxed geometry can differ with nothing saying the request was ignored.

---

## B. Numerical guards that are fine on silicon and not on a real cell

### B1. The interband conductivity's degeneracy guard is below the eigensolver's accuracy **[closed 2026-09-12 -- three of the sweep's claims were wrong]**

> **The guard was right to change and every specific thing the sweep said about
> it was wrong**, which is why this entry is the longest of the closed ones.
> Measured on **nonmagnetic** fcc nickel with spin-orbit coupling
> (`ni-soc-nosym.in` with `starting_magnetization = 0`, so time reversal is
> unbroken and every band is exactly Kramers-degenerate): 102 weight-carrying
> occupied/empty pairs sit at the Fermi level split by nothing but arithmetic.
> That is the case the mechanism needs, and the magnetic nickel the sweep named
> cannot show it -- a magnet has no exact degeneracy, and its smallest
> weight-carrying gap is 5.1e-4 Ry, five orders above the guard.
>
> **1. The premise is false on the route it was made about.** The frequency
> route's `1/e_mn` does not diverge at a degeneracy, and not by luck: the two
> orderings of a pair carry `t_nm = -t_mn` and `z_mn = conj(z_nm)`, so what
> survives is `w_k [f(e_n) - f(e_m)]`, which is itself linear in the gap
> whenever `f` is a smooth function of energy. The singularity cancels
> analytically. Measured flat to four significant figures over **eight decades**
> of splitting on a synthetic pair, and on the nickel run `sigma_xx`,
> `sigma_xy` and the plasma frequency are identical to every printed digit
> (1.290657e5 S/cm, 8.5030e-6 S/cm, 0.688568 eV) at six tolerances from 5.6e-13
> to 1e-5, while the number of pairs dropped goes from 42 to 102.
>
> **2. The singularity is real in two places the sweep did not name.**
> `method = "curvature"` -- the intrinsic anomalous Hall route, which *is* the
> quantity the entry was worried about -- has a `1/e_mn^2` weight, and the
> numerator difference kills only one power: measured 7.4e13, 7.4e11, 7.4e9,
> 7.4e7, 7.4e5 at splittings of 1e-12 down to 1e-4, exactly `1/g`. And **fixed**
> occupations cutting a degenerate multiplet, where `f` is not a function of
> energy at all: `1/g` on the frequency route and `1/g^2` on the curvature one.
> The second already had a name and a diagnostic here, `band_cut_gap`.
>
> **3. The test the entry proposed is blind on the route it would have been run
> on.** "An AHC that moves with `conv_thr`" reads `hall_conductivity`, which on
> the default `method = "frequency"` comes from a leak of the form
> `(2t/eta) Re(z)` -- real and **symmetric**, so time reversal protects
> `sigma_xy` there by construction. And on the magnetic nickel the entry named,
> the two runs at `conv_thr` 1e-10 and 1e-6 agree to 1e-7 relative because
> neither ever reaches the guard. The test *does* work on the curvature route
> of a nonmagnetic spin-orbit metal, which is neither the route nor the case it
> was written for.
>
> **What the guard is worth, measured where it bites.** The curvature route on
> that nickel, whose answer is **zero** by time reversal:
>
> | `degeneracy_tol` (Ry) | pairs dropped | max abs sigma (S/cm) | `sigma_xy` (S/cm) |
> |---|---|---|---|
> | 5.6e-13 (the `ethr`-derived guard) | 42 | **1.2576** | 3.23e-2 |
> | 1e-10 | 102 | 4.668e-4 | 5.09e-5 |
> | 1e-8 (the old constant) | 102 | 4.668e-4 | 5.09e-5 |
> | 1e-5 (the new one) | 102 | 4.669e-4 | 5.07e-5 |
>
> **2700 times** between the guard that was nearly shipped and any guard above
> the round-off floor, and nothing at all between the old constant and the new
> one. So the honest summary is B3's: on every committed case the number does
> not move, and what changed is that the guard now means something and says
> when it fired. The warning fires on all five curvature rows and on none of
> the six frequency ones, which is the discriminator working.
>
> **What the number is now.** `EMPTY_ETHR_FLOOR`, 1e-5 Ry -- what an
> `SCFResult`'s *empty* bands are converged to, which is the loosest thing the
> signature accepts, since `optical_conductivity` takes an array and cannot see
> where it came from. Deriving it from the fixed-density run's own `ethr` was
> tried and is **wrong**: the splitting left on a symmetry-degenerate pair sits
> on an arithmetic floor before it is a multiple of anything -- 5.535e-12 Ry at
> `ethr` = 5.6e-13, 5.471e-12 at 5.6e-11, 1.242e-10 at 5.6e-9 -- so an
> `ethr`-derived guard sits under that floor and lets 60 of the 102 pairs
> through. `max(k ethr, floor)` is the shape `empty_ethr` already has, which is
> the second reason to take the constant from it rather than invent one.
> The old 1e-8 was **inside the empty window rather than wrong**: the gaps are
> bimodal, 102 below 5.5e-12 and none at all from there to 1e-4, so any constant
> in those seven decades behaves identically. What it was not is a statement
> about anything.
>
> **And it now says so.** `optical_conductivity` counts the pairs it removed --
> off-diagonal and weight-carrying only, since `e_nn = 0` exactly and every
> metal would otherwise report a large constant -- carries the count on
> `OpticalConductivity.degenerate_pairs`, and warns **only where the
> cancellation does not reach** -- the curvature route, or an occupation that
> is not a function of energy (fixed, or a tetrahedron run's step). Not the
> `_drude` test, which was the first attempt: a *smeared* run with
> `intraband = False` still cancels, so warning there would be noise.
> `degeneracy_tol`
> is a caller override on both entry points. Five tests in
> `tests/unit/test_conductivity_machinery.py`, including the two-route scaling
> comparison, which is the mechanism itself and costs no SCF.

`defumat/response/conductivity.py:611`: 1.0e-8 Ry. Davidson does not promise that. A
pair degenerate by symmetry comes back split by 3e-7 Ry of residue, survives the guard,
and enters a `1/gap^2` weight: with `wg_n(1-f_m) ~ 0.25` from the smearing that is
`~3e12 * Im(v_i v_j)` in one term of `sigma_ij`. fcc Ni with spin-orbit coupling, or any
metal slab whose Fermi surface crosses a symmetry line, gets an optical conductivity and
an anomalous Hall number that is large, finite, smooth in frequency, and entirely an
artifact of `conv_thr`.

**The tell, and the test:** tightening `conv_thr` changes the answer. That is the check
to write -- an AHC that moves with the convergence threshold is the failure, and one
that does not is the pass.

### B2. The tetrahedron degenerate-weight average does not conserve weight **[closed 2026-09-11]**

> A `lax.scan` partition -- compare to the *first* of the group, on sorted
> eigenvalues -- so the average is block-diagonal and conserves weight by
> construction. Tested on a synthetic chain, with the old symmetric form beside
> it as the control. **Written from `opt_tetra_weights_only`'s description and
> not transcribed**, because the vendored tree is absent on this machine; the
> docstring says so and asks for the check where it is available. The metal
> cases that would exercise it against QE (`reference.out.al10-metal-tetra`,
> `al-tetrahedra`) need that tree too, so the partition has been checked on
> synthetic chains and exact multiplets and not yet on a real Fermi surface.

`defumat/scf/tetrahedra.py:576`. `_average_degenerate` builds a symmetric "within 1e-6
Ry" matrix `S` and returns `w'_i = sum_j S_ij w_j / sum_j S_ij`. Its own docstring says
the relation may fail to be transitive in a chain and that "the operation is
weight-preserving either way". **It is not.** Weight is preserved only if
`sum_i S_ij / d_i = 1` for every column `j`, which holds for a block-diagonal
equivalence relation (`sum_{i in block} 1/|block| = 1`) and fails for a chain: three
bands with `a~b`, `b~c`, `a!~c` give `d = (2, 3, 2)` and column `b` sums to
`1/2 + 1/3 + 1/2 = 4/3`.

So after the Fermi level has been bisected to give exactly `nelec`, the weights returned
sum to something else, and `sum_band` builds a **charged cell** with no error and no
message. It needs bands dense near `E_F` -- a metal slab or a large supercell. Silicon
cannot show it: an insulator with exact degeneracies, where the symmetric and the block
form coincide.

**What to write.** QE's own sequential scan, which builds genuine blocks by comparing
each band to the *first* of the group -- the docstring already identifies it as the
reference and as the thing that was replaced. **How to know it worked** is
`sum(wg) == nelec` asserted after `_average_degenerate`, on a case with a chain in it.

### B3. The Kubo Berry curvature swallows the point a Chern number is about **[closed 2026-09-12 -- one of the sweep's three claims was wrong]**

> **The old guard had two behaviours and the sweep named one of them.** Both
> measured on graphene on a 3x3 mesh, which lands on both Dirac points exactly
> (`t2 = 0`, a sublattice mass setting the gap):
>
> * an **exactly** gapless model -- gap 8e-16, which is rounding rather than
>   physics -- fell *below* the absolute 1e-12 guard and was silently zeroed.
>   That is the sweep's "swallows the point", and it is right: a symmetry-forced
>   touching that sits on a mesh point is the common case, and nothing said so.
> * **any** perturbation off exact -- a 1e-9 mass, so a 2e-9 gap -- sails over
>   the guard and returns `Omega = 1.7e19`, nineteen orders above every other
>   point on the mesh, again with nothing to say the sum it entered is not a
>   Chern number.
>
> So the absolute threshold was wrong in *both* directions at once, and which
> one a run got depended on whether the touching was exact to the last bit.
>
> **The NaN half is a measured null**, like A8. `jnp.where` multiplies the
> untaken branch's tangent by zero, and `1/gap^2` is a large *finite* number
> for any gap an eigensolver can produce -- an infinity would need
> `gap < 1e-154`. Gradients are finite at gaps of 2e-9, 2e-13, 2e-14 and
> 8e-16. The two thresholds were made one anyway: a guard that means two
> things is one library change from meaning something wrong.
>
> **The threshold is now a fraction (1e-8) of the band width over the whole
> mesh**, and the "whole mesh" is the part that is not obvious -- for a
> two-band model the spectrum at *one* k **is** the gap, so a tolerance
> relative to the local spread can never fire. `kubo_curvature` takes one
> `eigvalsh` pass over the mesh first, which is free beside the `jacfwd` it
> was already paying per point.
>
> **And it says so**, which is what closes the first branch above rather than
> moving it. `_kubo_point` returns how many occupied/empty pairs it dropped,
> `BerryCurvature.singular_points` carries the total and `kubo_curvature`
> warns, naming the count and pointing at `method='fhs'` -- whose determinant
> of overlaps is an exact integer on any mesh. Three tests in
> `tests/unit/test_topology_curvature.py`, including the complement (a gapped
> model must report **zero**), so the count is a discriminator rather than a
> constant.
>
> The plane-wave sibling `topology/kubo.py:139` already had the one-mask form
> and a documented `DEGENERACY_TOL`; the two expressions are now the same.

---

## C. Tests that cannot tell a pass from silence

The trap `CLAUDE.md` added most recently, and Part I item 1 is the same family.

### C1. The spinor force symmetry test asserts an identity the code has already imposed **[closed 2026-09-12 -- the entry's diagnosis is right and its proposed fix is not]**

> **The diagnosis holds and the fix does not, for a reason that is worth more
> than either.** C1 asked for the identity on the *unsymmetrised* gradient,
> "where it is a claim rather than a tautology". It is not a claim there
> either: a wedge sum is exact for a scalar and **not for a vector**, which is
> the reason `symvector` is not optional in the first place, so the raw
> gradient carries a symmetry-forbidden part by construction. Measured on the
> displaced platinum, where the surviving operation forces `F_x = -F_z`, the
> unsymmetrised gradient breaks it by **1.1185e-2 Ry/bohr on forces of
> 5.2432e-2** (ultrasoft, 21.3 per cent) and **3.8661e-3 on 5.9167e-2** (PAW,
> 6.5 per cent). A bound loose enough to pass that discriminates nothing.
>
> **What was written instead is the pair**, which is a statement where neither
> half is: the unsymmetrised gradient must break the identity by more than
> `WEDGE_ASYMMETRY_FLOOR` = 1e-3 -- so the projection had real work to do on
> this case and the next line is not vacuous -- and the symmetrised force must
> satisfy it, measured 6.9e-18 on both datasets. That is `CLAUDE.md`'s "test
> that the guard fires" rather than reading a clean zero as a pass. The
> gradient is carried as `Forces.unsymmetrized`, whose docstring says plainly
> that it is not a better force.
>
> **What the test cannot do, said out loud rather than left implied.** C1's
> real worry -- a P46 term wrong by a factor the crystal symmetry respects, a
> dropped `dvan_so` or `qq_so` piece, a mis-scaled augmentation force -- moves
> the force's *magnitude* inside the invariant subspace and no symmetry check
> can see it. `test_forces_match_quantum_espresso` and
> `test_the_force_is_a_finite_difference_of_the_frozen_energy` are what catch
> those, and the docstring now names them instead of claiming this test is
> "the one check here that would survive both codes being wrong in the same
> way", which was false.
>
> **A memory figure came out of it**, and it belongs with Part I item 2:
> `pt2-soc-paw-force` peaks at **12,204 MB**, *over* `run_regression.sh`'s 12G
> default, so that case wants `DEFUMAT_TEST_MEM_MAX=20G`; the ultrasoft case
> peaks at 7,691 MB. Two harness kills were spent finding that out.

> **A third file over the cap, 2026-09-15.**
> `tests/regression/test_noncollinear_hubbard_resume.py` peaks at **20.1 GB**
> and is therefore SIGKILLed by `run_regression.sh`'s 12G default *every time*,
> which means the nickel spinor DFT+U resume has been contributing `killed` and
> not a result to every capped run it appears in. Both its tests **pass** given
> room: `2 passed` in 313 s at `DEFUMAT_TEST_MEM_MAX=20G`, and the watchdog
> still fires at teardown because 85% of 20G is below the peak, so the file
> wants **22G** to come back clean. It is one test that does it,
> `test_a_spinor_hubbard_run_resumes_from_its_own_checkpoint`, which the
> watchdog names.
>
> **Measured as an A/B across a commit rather than read off one run**, because
> the entry above this one is about exactly how misleading a single memory
> figure is: 20,206 M at `0da2922` against 20,120 M at `09cf109`, 0.4 per cent
> apart, with runtimes of 313.42 s and 312.17 s. So the peak is the file's own
> and predates the continuation work that found it. What has *not* been done is
> to ask where 20 GB goes on a cell this size, which is a `MEMORY-AUDIT.md`
> question and is the reason this is an entry rather than a note.


`tests/regression/test_spinor_forces.py:260`.
`test_the_force_carries_the_crystal_symmetry` asserts `F_x = -F_z` on forces that
`compute_forces` has **already** projected onto the symmetric subspace with
`symmetrize_vector`. Break any P46 term whose error respects the crystal symmetry --
drop the `dvan_so` or `qq_so` contribution, get the ultrasoft Pulay term wrong,
mis-scale the augmentation force -- and every force changes magnitude while staying
inside the invariant subspace. The assertion still reads `< 1e-12`.

The test detects symmetrisation not running at all, and a wrong `atom_mapping`. It
detects nothing about the value of the force. **What to write:** assert it on the
*unsymmetrised* gradient, which is where the identity is a claim rather than a
tautology.

### C2/C3. The second and third instances of Part I item 1 **[closed 2026-09-11]**

Part I asks explicitly whether any other test asserts an eigenvalue degeneracy or
difference over all bands without splitting occupied from empty. Two:

- `tests/regression/test_spinorbit.py:407` --
  `test_kramers_degeneracy_on_the_bismuthene_path` asserts 1e-6 eV over every band of
  the bands run, **including the top two the same file documents as unconverged in both
  codes**.
- `tests/regression/test_spinorbit.py:135` --
  `test_spinors_reproduce_the_collinear_answer` compares two independent SCF runs band
  by band to 1e-10 Ry over every band, where the solver promises `max(5 ethr, 1e-5)` Ry
  on the empty ones. Five orders looser than the bound asserted. *PLAUSIBLE.*

Fix all three in one pass with Part I item 1's two-bound rule, and the commit says so
because it changes a P14 claim.

---

## D. Memory

### D1. The sum-over-states `chi_0` materialises the whole pair axis **[closed 2026-09-12]**

> The pair axis is chunked by `pair_batch`, defaulting to the **band** dial --
> one pair density in flight is exactly what one band in flight is -- and
> `batching.map_axis` is `map_k`'s body under a name that does not claim the
> axis is k. *Measured*, by `memory_analysis()` rather than by a run: the
> compiler's temporaries fall from 8.19 MB to 3.13 MB on the silicon case, and
> the part that scales with `npairs` from 5.06 MB to nothing. **There is a floor
> and it is `fields`**, the `nbnd` states in real space, which is the natural
> working set; the entry's implied "bound it and it is bounded" is therefore
> only true of the grid-sized half. The `(nw, 2 npairs, nm)` assembly above the
> transform is still linear in `npairs` and is the docstring's stated trade,
> two hundred times smaller. The matrix agrees to 1e-14 of its maximum between
> the two settings, and there is a test that the dial reaches the *entry point*
> as well as one that it saves anything -- which is the half P74 says goes
> wrong.

`defumat/tddft/chi0.py:484`. `products` is `(npairs, n1, n2, n3)` complex with
`npairs = nocc * (nbnd - nocc)`, all live at once, and no dial bounds it -- while the
module's own docstring claims the pair densities are bounded by `map_k`. The P74
template exactly: a documented dial that does not reach the hot path.

### D2. `sizing.py` can report a peak below the floor it just discarded **[closed 2026-09-11]**

> The buffer carries `k_live`, because the fit was taken one k-point at a time
> (`tools/gpu/davidson_memory.py` defaults `--k-batch` to 1) and
> `davidson_eigensolver_all` holds `k_batch` of those at once. Two assertions,
> neither needing an SCF: the buffer is never below the floor it supersedes, and
> the peak grows with the batch.

`defumat/sizing.py:619`. `eigensolver_buffer` carries no `k_live` factor, yet
`peak_bytes` uses it to **supersede** the two Davidson `arrays` lines that do. At
`nk = 8`, `k_batch = None` (the accelerator default), `nbnd = 64`, `ndim = 10000`,
`nvecx = 256`, smooth grid `45^3`: the superseded lines are 655 MB + 328 MB = 983 MB and
`eigensolver_buffer` is 319 MB, so `peak_bytes` reports **664 MB less than the floor it
removed**, and reports a smaller peak at `k_batch = None` than at `k_batch = 1`.

A green light for the end of the dial that actually holds `nk` subspaces. This is the
module's own stated error inverted -- `sizing.py:541-543` warns against exactly this
direction of mistake. It is also the tool this project uses to decide whether a
calculation fits before starting it, which is what makes a 664 MB under-report worse
than no estimate.

### D3. The vertical-tunnelling workflow allocates outside every dial **[closed 2026-09-12]**

> The k loop is chunked by `k_batch`, which every branch of the contraction
> permits because each ends in `kweights @ term`. 512 entries against 8192 on
> `h-sheet.in`; 1.6 GB against 16 MB on a 100x100 map over 100 k-points and 50
> spinor bands, and the `(nk, nbnd, npoints)` intermediate inside `transmission`
> falls with it. **The entry is right that this sits outside every dial and
> incomplete about why it matters**: the array is on the *host*, so the platform
> default -- `None` on an accelerator, because a *device* wants the whole axis
> -- does not bound it, and a GPU run with an image-sized `npoints` has to pass
> a number. On a CPU the default already is 1. The map is unchanged to 1e-13 of
> its maximum, and old and new agree digit for digit where they are printed.

`defumat/workflows/transport.py:368`. One host array of `(npol, nk, nbnd, npoints)`
complex128 up front, and each k-point's wavefunctions pulled to host inside a Python
loop.

---

## E. Incompatibilities worth building, with the missing term named

### E1. The piezoelectric tensor on ultrasoft -- the blocker is a case, and the case exists **[closed 2026-09-11]**

> Scoped as the entry's caveat implies: the *dataset* claim is demolished and the
> refusal now names the real term (`dbecsum`'s strain piece from a cell-dependent
> `Q_ij(r)`, which `response/strain.py` refuses ultrasoft for). The missing twenty
> lines are `tests/data/qe/alas-piezo.in`, and the test checks the cell really is
> ultrasoft, non-polar and non-centrosymmetric (Td, 24 operations, no inversion).
> Running the piezo tests on it is tier 3, and waits on that term.

`defumat/response/piezo.py:258`. The refusal says "every ultrasoft and PAW case
committed here is a centrosymmetric crystal", and `PLAN.md`'s outstanding index repeats
it. `Al.pbe-n-rrkjus_psl.1.0.0.UPF` and `As.pbe-n-rrkjus_psl.1.0.0.UPF` are committed in
`tests/data/pseudo/`, and `tests/data/qe/alas-magnetoelectric-nosoc.in` already builds
zincblende from them -- `ibrav = 2`, `(0,0,0)`/`(0.25,0.25,0.25)`, which
`require_a_nonpolar_crystal` (`piezo.py:211`) itself names as the handled class. The
reporting agent dates the input to 2026-09-01 and the refusal text to 2026-09-02: the
stated reason was already wrong the day it was written.

**Caveat to carry:** both committed zincblende inputs set `noncolin` and a `B_field`, so
they are stopped upstream by the noncollinear branch of
`require_a_sternheimer_regime`. What is demolished is the *dataset* claim, not a claim
that those two files run today. The missing ingredient is a nonmagnetic AlAs scf input
of about twenty lines, after which the existing piezo tests apply.

### E2. The elastic constants of a metal, refused with a reason about a different quantity **[closed 2026-09-11 -- the message half]**

> `require_a_sternheimer_regime` takes a `metals_missing` reason, defaulting to
> the epsilon_infinity one and overridden by the strain response with the
> Fermi-level-shift one. The `ef_shift` term itself is still open.

`defumat/response/strain.py:238` is a bare `require_a_sternheimer_regime(calculation)`
with no `metals=True`, where the sibling perturbation passes it -- **[opened here]**
`response/phonon.py:342` is `require_a_sternheimer_regime(calculation, metals=True,
gamma_ok=True)`. The guard then raises
`sternheimer.py:1119-1126`: *"a metal has no epsilon_infinity and no Born effective
charge, which is why pw.x refuses epsil for one too"*.

Elastic constants are perfectly well defined for a metal, so the message answers a
question the user did not ask. Ask for aluminium's `C_11`/`C_12`/`C_44` -- the textbook
case, on the same fcc Al cell P24c already validates `chi_0` on -- and there is nothing
in the output to say that what is actually missing is the Fermi-level shift and the
`wg`/`2wk` weight split that P28 already wrote for the displacement coordinate
(`localdos`/`ef_shift`, `sternheimer.py:565-585`).

**Two separate pieces of work**, and the first is minutes: a refusal that names the
right missing term, then the term itself.

### E3. Projected DOS for a noncollinear run *without* spin-orbit coupling **[closed 2026-09-12]**

> The columns are routed into an up and a down density of states
> (`workflows/pdos.split_spin_columns`, `partialdos_nc`'s `nspin0 = 2`), so the
> result has the shape an LSDA projection has: `nspin = 2`, the spin an *axis*
> rather than a label on a column, `charges.polarization` meaning what it says,
> and `charges_lm` filled -- which a `j`-resolved projection leaves `None`, and
> which is a **documented divergence** from `print_lowdin`, since QE allocates it
> only for `nspin /= 4` and the reason it gives (a spin-angle function has no
> `m`) is true of the spin-orbit branch alone.
>
> **The entry's second requirement was not met and was replaced by something
> better.** No `projwfc.x` reference was generated: the vendored QE tree is not
> on this machine. What stands in its place is an identity that shares no
> machinery with the thing it checks -- without spin-orbit coupling a moment
> along `+z` block-diagonalises the noncollinear Hamiltonian into the two
> collinear ones, so the projection must reproduce an LSDA run of the same cell
> channel for channel, and the LSDA route *is* validated against `projwfc.x`.
>
> *Measured* on a hydrogen atom in a 12 bohr box: the two runs agree to 3e-12 Ry
> in total energy, the majority channel's curve to **2e-5 of the peak**, the
> minority one to 0.39 per cent, and the Löwdin charges to 5e-3. The minority
> bound is looser for a stated arithmetic reason rather than a tolerance chosen
> to pass: the noncollinear branch reaches `rho_down` as `(n - |m|)/2`, a
> cancellation of two numbers of order 0.1, and that is worth 1.1e-4 Ry on the
> empty minority eigenvalue where every occupied one agrees to 1e-6.
>
> **It found G1 on the way**, which is the more valuable half: the same
> comparison was 85 per cent out before the exchange-correlation clamp's tangent
> was fixed, and the cell here is saturated on purpose so that it stays the case
> that guards it.



`defumat/projwfc/projections.py:222`. The spin-angle orbitals are built, the labels
carry their `s_z`, `_updown_matrix` exists. The missing term is `partialdos_nc`'s split
of the columns into up and down channels (`nspin0 = 2`, route each column by
`ind <= 2l+1`), plus a generated `projwfc.x` reference -- `ls tests/data/qe` has
`pt-soc`, `pt-soc-paw`, `pw_lsda-lsda`, `al-tetrahedra`, `si10-nc` and no noncollinear
non-SOC case.

The regime is a real one: any bcc-Fe noncollinear run with scalar-relativistic datasets,
and `alas-magnetoelectric-nosoc.in` is already in that regime. **On none of the gap
lists** -- it is the residue P69 left behind.

---

## G. Opened 2026-09-12, while closing E3

### G1. A saturated magnet's minority potential is discontinuous in the last bit of `zeta` **[closed 2026-09-12, the same day]**

> **The cause was not the clamp's value but `jnp.clip`'s *tangent*.** JAX's
> `minimum`/`maximum` split the gradient **evenly at a tie**, and `clip` is built
> out of them, so `d clip(z, -1, 1)/dz` is `0.5` at `z = 1` exactly -- not 1,
> and not 0, which would at least have been visible. The minority potential
> therefore received half of its `de_c/dzeta . dzeta/drho_down` term. Written as
> a `where` (`defumat/xc/lda.py`'s `clamp_polarization`) the value is still
> clamped, the interior's tangent survives at the boundary, and beyond it the
> tangent is genuinely zero. Four sites: `spin_interpolation`, `pw_spin_hartree`,
> PBE's `gcc_spin` and `_spin_channels`, plus the same tie in
> `local_spin_frame`'s `minimum(|m|, |n|)`, which is a saturated point by
> definition.
>
> *Measured, on the hydrogen atom that found it.* Pointwise, `v_down` at
> `rho_down = 0` is now **-0.38993** against the limit's -0.38993 -- it was
> -0.22756 -- and continuity is asserted as a **rate** rather than a tolerance:
> the gap closes as `rho_down^(1/3)`, the exchange's cube root, so 6.9e-5 at
> 1e-12 and 3.2e-6 at 1e-16, a ratio of `10^(4/3)` that says the value at zero
> is the limit rather than a number that happens to be near it. On the whole
> calculation, the same ground state as an LSDA run and as a noncollinear one
> now agrees to **6.5e-6 Ry** in the minority potential (was 0.162), to
> **1e-6 Ry** on every occupied eigenvalue, and to **1.1e-4 Ry** on the empty
> minority one (was 7e-2). The remaining 1.1e-4 is not this: it is the
> cancellation in `rho_down = (n - |m|)/2`, two numbers of order 0.1 leaving
> round-off where the collinear branch carries `rho_down` itself.
>
> **The energy never moved** -- asserted, and it is the reason nothing else saw
> this: `h-atom-lsda.in`'s total is the same to 2e-16 Ry. That is `CLAUDE.md`'s
> "the energy can be right while its derivative is wrong" with a mask boundary
> in place of a structure factor, and the trap list has gained the entry.
> `tests/unit/test_xc_spin_kernel.py` (pointwise, three functionals) and
> `tests/regression/test_noncollinear_pdos.py` (the whole calculation).
>
> **What is still not done is the comparison against `pz_spin` itself**, which
> is what would settle the value against QE rather than against continuity.
> It needs the vendored tree and that is not on this machine.



`defumat/xc/lda.py`'s `spin_interpolation` clips `zeta` to `[-1, 1]`, and the
clip's *tangent* is what the potential is built from -- `v_xc` is `jax.grad` of
the energy. At `zeta` exactly `1.0` the clip stops passing the gradient through,
so `dE_c/dzeta . dzeta/drho_down` is dropped from the minority potential, and
that term is **0.162 Ry**:

| `rho_down` | `zeta` | `v_down` |
|---|---|---|
| 0 | `1.0` | -0.22756352 |
| 4.7e-18 | `1.0` (rounds to it) | -0.22756352 |
| 1.0e-17 | `0.9999999999999998` | **-0.38992878** |
| 1e-12 | `0.999999999999981` | -0.38986115 |
| 1e-6 | `0.999980655` | -0.40713967 |

at `rho_up = 0.10338550`. The limit from inside is the physical one -- `v_down`
is the one-sided derivative `dE/drho_down` at `rho_down = 0+`, which is what an
added minority electron feels, and QE's `pz_spin` gives it analytically and
continuously -- so the value at exactly `|zeta| = 1` is **the wrong one of the
two**, by 0.16 Ry.

**Which side a run lands on is rounding**, which is how this was found. A
hydrogen atom in a 12 bohr box is saturated by construction, and the *same
ground state* reached two ways lands on opposite sides: `h-atom-lsda.in` gives
`rho_down = 4.7e-18` (so `zeta` rounds to exactly 1) and the noncollinear run of
the same cell gives `1.39e-17` (so it does not). The two agree on **the total
energy to 7e-13 Ry**, on every occupied eigenvalue, and on the Löwdin charges to
1e-8 -- and disagree on the *empty minority* eigenvalues by **0.07 Ry**, and on
`v_down` by 0.162 Ry pointwise. It is `CLAUDE.md`'s "the energy can be right
while its derivative is wrong", with a mask boundary that symmetry reaches
exactly in place of a supercell's structure factor.

**What it affects.** Nothing occupied, and nothing in a total energy. Every
*empty* minority state of a saturated system, so: an NSCF band structure or a
density of states above `E_F` for a half-metal or any fully polarized cell, and
anything built on those -- a response, a spectrum, a projected DOS above the
Fermi level. P70's finding is the neighbouring one and is **not** this: there,
the *kernel* `dmxc_lsda` is **defined** to be zero at `|zeta| >= 1` on both of
QE's branches, which is a convention; here it is the potential, and the two
sides are not two conventions but a value and a dropped term.

**What to write.** The clip is the wrong instrument for a quantity whose
one-sided derivative is wanted: clamp the *channel densities* to be
non-negative instead (`rho_down = max((n - |m|)/2, 0)`), which is where the
constraint actually lives, or write `zeta` so the tangent survives at the
boundary. **Do not take the table above as the acceptance test** -- the check is
against `pz_spin`'s own `vc_up`/`vc_dn` at `zeta = 1`, which needs the vendored
QE tree, and that is not on this machine.

**Cost.** The fix is a line; the validation is not, and it reaches every
spin-polarized run in the suite. Worth doing beside a `pw.x` comparison rather
than alone.

---

## F. Three records that are wrong and mislead the next session **[all closed 2026-09-11]**

Each is minutes, and each is the kind of error that costs a phase: a stale refusal reads
as a closed question.

1. **`PLAN.md` §3's outstanding index**, the piezoelectric entry -- "every soft dataset
   committed here is centrosymmetric" is false (E1). The blocker is an input file, not a
   pseudopotential.
2. **`GAPS.md:391-404`**, "Projected DOS + noncollinear / spin-orbit" -- says the whole
   spinor regime is refused at `projections.py:163` and names three missing pieces, two
   of which P69 landed and validated against `projwfc.x`. `projections.py:227` now ends
   with *"lspinorb = .true. is implemented and validated against projwfc.x"*, and
   `reference.projwfc.pt-soc` and `pt-soc-paw` -- the platinum cases the entry calls
   blocked -- are committed. What remains is only E3 and the symmetrised-noncollinear
   half at `:206`.
3. **`defumat/scf/tetrahedra.py:573`**, the docstring claim that the symmetric average is
   "weight-preserving either way" (B2). It is the reason the defect was not seen.

---

# Part III -- the optimisation sweep of 2026-09-12

Four read-only agents over disjoint subsystems -- the SCF hot path, the response and
post-processing stack, memory and the working set, and setup/retracing -- looking for
**speed and memory** rather than for wrong answers. 23 entries.

**Two caveats bind every entry below, and they are why this is a candidate list rather
than a backlog.**

1. **Nothing was executed.** No test, no benchmark, no timing: four agents timing anything
   at once on this machine produces numbers that have to be discarded, which is the rule
   `PERFORMANCE.md` already states. So **no "gain" line below is a measurement of the
   proposed change.** Each is arithmetic on a figure already in `PERFORMANCE.md`, and each
   entry carries the input and the command that would turn it into a number. Where no
   existing figure bounds a claim the entry says `unbounded` rather than inventing a
   percentage.
2. **The vendored QE tree was absent from the checkout** (`quantum_espresso/` does not
   exist), so every "QE does it this way" claim here is sourced from *this repository's*
   docstrings and not from the Fortran. `CLAUDE.md` makes each such tick a claim about
   someone else's source; these are not yet. Re-read the named routine before acting on
   one. All four agents disclosed this unprompted.

**All eight H-tier sites were read and confirmed to say what the entry says they say**
before this was written; the M, S and X tiers were not, beyond the four spot-checks named
in their entries. Part II's record is that a sweep's *specifics* are wrong more often than
its instinct is -- B1 was right to change and all three of its particular claims false --
so treat an unverified entry as a place to look rather than as a fact.

**The order is by ease times impact**, cheapest-first, which is *not* Part II's rule --
that one ranked by what a wrong answer costs. Nothing here gives a wrong answer. Four
entries could move a validated number and are marked **[moves a number]**; they need a
`pw.x` or reference comparison re-run beside them rather than a timing alone.

**Nothing from this sweep goes into `PERFORMANCE.md`'s ten-item backlog until it has a
number**, which is the whole rule of that file. Entries that are *siblings* of a backlog
item say so.

**Status, 2026-09-12 (later the same day).** **H1 is measured and half of it is a null** --
the entry's central argument, that XLA does not remove a duplicated energy evaluation, is
true of the polarized branch (1.10-1.14x, implemented) and false of the unpolarized one
(1.00x, reverted). That is the first entry priced and it went the way Part II's did: right
that there was something there, wrong about what. Nothing else here has been measured.

---

## H. Cheap, and the gain is bounded by a figure already on record

### H1. A GGA evaluates its energy expression twice per iteration **[measured 2026-09-12 -- half of it is a null]**

> **Implemented for `nspin = 2` and reverted for `nspin = 1`, because the entry's
> central argument is wrong on the branch it was made about.** It reasoned from
> the LDA slot's measured 7.3 -> 3.5 ms that XLA does not remove this duplicate.
> On the *unpolarized* GGA branch it removes it entirely: 0.99x / 1.03x / 1.00x
> at 24^3 / 45^3 / 64^3. On the polarized branch it does not, and one call in
> place of two is **1.14x / 1.11x / 1.10x** on the same three grids. So the pair
> is now deliberately asymmetric, with the measurement in the docstring.
> **The run-level effect is below this machine's noise**: `v_of_rho` on a
> 20-atom antiferromagnetic hydrogen chain under PBE is 968.0 -> 955.7 ms, min
> of 15, inside a 7-8% spread -- which is what the kernel table predicts, the
> polarized kernel being ~250 ms of that 960. Four runs of the *same* binary
> swung 39%, so no number here is read off a single pair. `v1` and `h` are
> **exactly** identical between the forms and `etxc` is identical to the bit on
> the real density; on a synthetic sweep across the `_sanitise` boundaries 22
> points of 4050 differ by 1.7e-18, so this is equivalent arithmetic and not
> identical arithmetic -- which is why the branch that gains nothing was put
> back exactly as it was. Three cells' total energies unchanged to the last
> printed digit, fast gate 1790 passed; every `pw.x` comparison in
> `test_lsda.py` **skips**, the vendored tree not being on this machine.
> `PERFORMANCE.md`, "What one evaluation of the gradient correction was worth".
> **Two sites the entry did not name**: `paw/gradient.py` has the same pair
> three times over, on the PAW spheres.


`defumat/scf/potential.py:452`. `gradient_potentials` is `jax.grad` of
`sum(where(active, _gradient_energy(r, s), 0))`, so its **forward pass already computes
the energy** -- and the next line calls `gradient_energy` on the identical `_sanitise`d
pair to get it again. The polarized branch does the same at `:456`/`:459`. That is the PBE
exchange-plus-correlation expression, powers and square roots and an exponential, over
every dense-grid point per channel, evaluated a second time.

**Fix.** `jax.value_and_grad(total, argnums=(0, 1))` in `gradient_potentials` and
`spin_gradient_terms`. The caller's `sum(gradient_energy(rho, sigma))` is
character-for-character the scalar being differentiated, so the value is bit-identical.
This is `Functional.potential_and_energy_density` (`xc/functional.py:375`) one derivative
further out, and that method's docstring already argues the case.

**Bounded by.** `PERFORMANCE.md`, "One evaluation of the functional, not two":
`exchange_correlation` **7.3 -> 3.5 ms**, `v_of_rho` **8.0 -> 5.5 ms** when the LDA slot
was fused. That measurement is also the evidence that **XLA does not remove the duplicate
on its own**.

**Gain (arithmetic).** One forward pass out of (forward + backward + forward), so ~1/3 of
the gradient correction's pointwise cost against the LDA's 1/2. On `si8-pbe-1k` the whole
correction is the 3 ms of a 74 -> 77 ms iteration, so ~1 per cent; it scales with the
dense grid, so `si8-paw-pbe-1k` at 0.841 s/iteration is where to take it.

**Measure.** `tools/benchmark.py benchmarks/si8-pbe-1k.in` and
`benchmarks/si8-paw-pbe-1k.in`, comparing the `v_of_rho` line -- the same instrument the
LDA number was recorded with -- on an `nspin = 1` and an `nspin = 2` input. `etxc` and the
total energy must be bit-identical.

### H2. The strain kernel issues nine calls where the loop above it already knows six suffice

`defumat/response/strain.py:507`. The kernel loop runs `for a in range(3), b in range(3)`
on a `symmetrised` array whose `[a,b]` and `[b,a]` entries are the same object, so **three
of the nine calls are literal duplicates**. The solve loop directly above it already
exploits the symmetry (`for b in range(a, 3)`); the kernel loop does not.

**Fix.** `range(a, 3)`, matching the loop above, filling `[b,a]` from `[a,b]`.

**The precondition, checked rather than assumed.** `stacked[a,b]` and `[b,a]` are
literally one object -- the solve loop writes `becsum_response[a, b] =
becsum_response[b, a] = parts` -- but the kernel consumes `symmetrised =
symmetrize_strain_response(stacked)`, one map later. So what has to hold is *element*
equality after that map, not object identity before it. Physically it must, since the
label is a symmetric strain tensor; assert it to round-off before relying on it, and if it
fails, the symmetriser is the finding and not this loop.

**Gain (arithmetic).** 1/3 of that file's kernel stage, on any cell, free once the
precondition holds. The absolute size rides on H4, since each call is one of the
discarded-primal kernels.

**Measure.** Any strain-response input through `tools/benchmark.py`; the elastic/strain
stage line. The tensor must be bit-identical.

### H3. Five sum-over-states workflows build two or three `Calculation`s and discard all but one

`defumat/workflows/conductivity.py:109` and `:175`. `run_conductivity` builds **three**
`Calculation` objects on one system: one so `_default_nbnd` can read `nelec` and
`noncolin`, one to hand to `require_a_conductivity_regime`, which reads only
`is_ultrasoft`, `is_paw`, `spiral` and the k-point weights, and then the one
`fixed_density_states` actually uses. Each discarded build runs the whole constructor --
both G sets, both FFT grids, its own symmetry search, the local potential, the core and
atomic charges, the projector core. The same shape in `run_absorption`, `run_shg` and
`run_shift_current`.

**Fix.** Build the `Calculation` once and pass it to the refusal helper and to
`_default_nbnd`; `nelec` and `noncolin` are properties of `System` plus the
pseudopotentials' `z_valence` and need no `Calculation` at all. Where a caller already
holds one, thread it through and move it with `Calculation.at_kpoints`, which exists for
exactly this.

**Bounded by.** `Calculation.__init__` = **1.53 s** on the one-atom Pt PAW cell (P69,
where it is 87 per cent of the call); setup on bcc iron = **7.078 s** (P17-P19).

**Gain (arithmetic).** Two constructors per call: **3.06 s** on the Pt cell, **14.2 s** on
iron. Neither cell is large, and the constructor grows with `ngm` and with the species
count.

**Measure.** Wrap `Calculation.__init__` with a counter and a `perf_counter` accumulator;
one `get_absorption` and one `get_optical_conductivity` in a fresh process, one core,
nothing else running. The count should be 3 before and 1 after; the number to compare is
the workflow's wall clock.

### H4. Four screening kernels re-linearise `v_of_rho` at a point that never moves

`defumat/response/phononq.py:464`, and the same shape at `phonon.py:621`,
`efield.py:523`, `strain.py:507`. `induced_potential_at_q` splits the complex response
density and takes **two** `jax.jvp(potential, (density,), ...)` calls. Each evaluates the
full ground-state `v_of_rho` primal alongside its tangent and throws the primal away. The
linearisation point is the converged ground-state density and **never changes for the life
of the run**, yet this is called once per mode per self-consistency iteration:
`3 nat x niter` calls, twice that many discarded primals.

**Fix.** `_, linear = jax.linearize(potential, density)` once, then apply `linear` to the
real and imaginary parts. The tangent jaxpr is the same partial evaluation `jvp` builds,
so assert bit-identity rather than assume it.

**Bounded by.** `v_of_rho` at 8.0 ms on the sixteen-atom cell; P71's q-phonons at 32.0 /
45.7 / 46.3 s over 8-11 iterations and 6 modes.

**Gain (arithmetic), and it is small on the measured cells -- said plainly rather than
inflated.** 6 modes x 11 iterations x 2 = 132 discarded `v_of_rho` evaluations on a 20^3
grid, order 1-2 ms each: **~0.2 s of a 46 s run**. It grows with the dense grid and with
`3 nat` while the solves grow with `nk nbnd npwx`, so a GGA on a large grid with few
k-points is where it would show.

**Measure.** P71's silicon q-phonon through `tools/benchmark.py`; the dynamical matrix
bit-identical.

### H5. Davidson computes the collapse's projections on every step, and uses them on one in three

`defumat/solvers/davidson.py:531`. `evc_becp, evc_becq = project(evc)` runs
unconditionally, immediately before the `lax.cond(full, ...)` at `:532` that is its only
consumer. The collapse fires when `nbase + nbnd > nvecx`, roughly one step in three at
`DAVID_NDIM = 4`; on the other two the projections are computed and discarded.

**Fix.** Move the call inside the `cond`'s true branch. It is a pure function of `evc`,
which the branch already closes over. `cegterg` computes its refresh quantities inside the
refresh.

**Bounded by.** "Inside a Davidson step", `si16-1k-ecut30`: a step is ~190 ms, of which
`_extend_projection` is 11.3 ms.

**Gain (arithmetic).** On sixteen-atom ultrasoft silicon (`nh = 8`, `nkb = 128`) one
calbec of that block is ~5.6 ms of a ~190 ms step and is wasted two steps in three: about
**2 per cent**. Zero on a norm-conserving run, where `s_projections` is zero-width.

**Measure.** `benchmarks/si8-us-1k.in` through `tools/gpu/davidson_profile.py`, the tool
the per-operation table came from. Step count from `diagonalize(..., return_steps=True)`
must not move; norm-conserving inputs are the control and must be bit-identical.

### H6. The symmetry search runs twice for every `Calculation`, and more for a budgeted run

`defumat/basis/builder.py:132`. `build_basis` calls `find_symmetries(...)` to size the FFT
box, and `Calculation.__init__` then calls `system.symmetry_group()` (`driver.py:1409`),
which calls `find_symmetries` on the identical cell and structure. Neither is cached:
`System` is a frozen `eqx.Module` with nowhere to hang a result, and `find_symmetries` is
a module-level function with no memo. `sizing.py:429` makes a third call, and
`workflows/nscf.py:358` and `workflows/dos.py:446` make more. `build_basis` skips its call
when `system.nosym`, so a spiral run pays once rather than twice.

**Fix.** Memoise on a content key -- the cell's bytes, the crystal positions, the type
array -- the way `pseudo/coupling.py:37` memoises `harmonic_products`. It is host-side
NumPy on arrays already outside any trace (`at_strain` and `at_positions` deliberately do
not re-search, following `setup.f90`), so a process-level memo is exact.

**Bounded by.** `none`. The loosest bound is the 12.9x setup ratio on bcc iron. The
entry's own gain field says `unbounded`, and the honest first step is the measurement, not
the fix.

**Measure.** In a fresh process, one core: time `find_symmetries` directly on
`benchmarks/si128-1k.in`, `al-slab.in` and `si8-1k.in`, and see how it scales with `nat`.
Then count the calls in one `Calculation` build (expect 2).

### H7. Six columns of the elastic tensor each re-evaluate the identical primal

`defumat/response/elastic.py:200`. `gradient = jax.grad(energy)` is built once, and the
`VOIGT` loop then calls `jax.jvp(gradient, (zero, psi), ...)` six times **at the same
primal point**. Forward-over-reverse: each call runs the whole reverse sweep through
`at_strain(0)` -- the cell rebuild, the form factors, the Ewald sum, the density over the
whole k axis -- produces the stress, discards it, and keeps one tangent column. Nothing in
the file is jitted.

**Fix.** One `jax.linearize` shared across the six columns.

**Gain (arithmetic).** `2N` units against `(1 + N)`: 7/12 at `N = 6`, about **40 per cent
of a 9.7 s stage, ~4 s**.

**The reason this is not already settled** is worth keeping: P54 measured exactly this
sharing and found it worth nothing (6.33 s against 6.35, "XLA had already deduplicated the
common primal") -- but on a **jitted** `matrix_elements`. This file jits nothing, so the
deduplication has nothing to happen inside. If the measurement comes back null, that is
the explanation to check first.

**Measure.** An elastic-constants run through `tools/benchmark.py`; the tensor
bit-identical.

### H8. The q-phonon's CG threshold is fixed at 1e-14, two orders tighter than its own family **[moves a number]**

`defumat/response/phononq.py:903`. `dynamical_matrix_at_q(..., threshold = 1.0e-14)` goes
straight into the CG convergence test, fixed from the first self-consistency iteration to
the last. The `Gamma` phonon (`phonon.py:267`), the field (`efield.py:205`) and the strain
response (`strain.py:215`) all use 1e-12, and `sternheimer.py`'s own default is 1e-11.
`ph.x` schedules this instead: `1e-2` on the first iteration and `min(0.1 sqrt(dr2), 1e-2)`
after. The same signature carries `tr2 = 1.0e-14` for the self-consistency test beside it,
so there are two fixed thresholds on this entry point, not one.

**Sibling of backlog item 7**, which measures the `Gamma` case at 27.7 CG steps against
`ph.x`'s 9.3 -- a factor of three on the stage that is 96 per cent of the run. At `q != 0`
the threshold is a further 100x tighter.

**Gain.** At least item 7's 3x, on runs measured at 12-14x per solve. No sharper figure is
claimed, because none exists.

**Measure, and re-run the reference beside it.** P71's three q-phonon cells. Loosening a
threshold moves the answer at round-off, so the acceptance test is the `ph.x` comparison
P71 was validated against, not the timing.

---

### H9. `fe-mag-1k` takes 32 SCF iterations where `pw.x` takes 12 **[measured 2026-09-12]**

Surfaced by the new `fast` benchmark set (`performance/sweep.py`), and it is the
one row in that set whose *total* ratio and *per-iteration* ratio disagree:
**7.6x warm-SCF against 2.8x per iteration**, entirely because of the iteration
count. Same `conv_thr = 1e-8` on both sides, same input, and the two energies
agree to **6.7e-9 Ry** -- so nothing is wrong with the answer and nothing is
slow about the arithmetic. The cell is `benchmarks/fe-mag-1k.in`: one iron atom,
`nspin_mag = 4`, Marzari-Vanderbilt smearing at `degauss = 0.05`,
`starting_magnetization = 0.5`, `mixing_beta = 0.3`.

**Why it is filed here rather than as a defect.** This is the same family as the
Co(0001) slab entries in `PERFORMANCE.md` (2026-09-01): where QE converges a
magnetic cell in a couple of dozen iterations, the mixer here takes several
times as many, and closing that gap has each time been a mixer question rather
than a kernel one. The bound is already on record -- at 2.8x per iteration, an
iteration count matching QE's would put this cell at the band's median and is
worth **2.7x on the run**, no more.

**What to do first, and what not to.** Check whether the magnetization is what
takes the extra iterations, by logging `dr2` and `|m|` separately: the plausible
story is that the charge converges in a dozen and the *vector* magnetization
takes the other twenty, which no scalar `dr2` distinguishes. Do **not** profile
`h_psi` -- the per-iteration figure already says the arithmetic sits in the
project's 2-4x band, and a total-time table with no per-iteration column beside
it is exactly what would have sent someone there.

**It is not the machine (2026-09-12, later).** The same 12 against 32 reproduces on a
different box and a different Quantum ESPRESSO -- i7-1255U, QE 7.4.1 in place of 7.5 --
with the energies agreeing to the same 6.7e-9 Ry and the other nine cases' iteration
counts identical as well (`PERFORMANCE.md`, "The same set on a second machine"). So the
count is a property of the mixer, which is where this entry already put it, and the
`dr2`-versus-`|m|` logging above is still the first step.

## M. Contained, but each needs the right input before it means anything

### M1. The noncollinear `newd` runs entirely outside `jit`, where the collinear one thirty lines above is inside it

`defumat/scf/driver.py:2581`. `_noncollinear_coefficients` is a Python list comprehension
over the `nspin_mag` potential components, each doing an eager dense-grid `r_to_g`, an
eager `augmentation.integrals`, and `block_matrix` -- itself a Python loop over atoms with
one eager scatter per atom, each allocating a fresh `(nkb, nkb)`. So one SCF iteration
issues `nspin_mag x (1 dense FFT + nat scatters + a stack)` as separate dispatches with
nothing fused across them. The collinear twin at `:2544` hands the identical assembly to
`_newd`, which is `@jax.jit` at `:456`. `Calculation.onecenter` at `:2484` has the same
eager tail. **Checked: the caller (`Calculation.hamiltonian`) is not itself jitted**, so
these really are separate dispatches every iteration.

**Fix.** Wrap both assemblies in jitted helpers the way `_newd` already is. `augmentation`
is already a pytree argument there, so the pattern transfers unchanged and **the
arithmetic does not move** -- only the number of compiled units, which is this file's own
recorded lesson from PAW's one-centre terms.

**Bounded by.** P73's `si8-us-1k`: `integrals` **0.033 s** per call, whole SCF 5.56 s over
6 iterations. And `bismuthene-soc-small` -- spinors, ultrasoft, PBE -- at **14.58 s per
iteration, 4.6 GB**.

**Gain (arithmetic).** `nspin_mag = 4` copies of a 0.033 s call is 0.13 s against a 0.93
s iteration: **14 per cent, and that is a floor** -- it counts none of the four eager
dense FFTs, the `4 nat` eager scatters, or the dispatch latency, which is the part jitting
collects. The cell that figure comes from is collinear; the number should be taken on
`bismuthene-soc-small`.

**Measure.** A `noncolin` + ultrasoft/PAW input. `tools/benchmark.py` for the stage
breakdown, `JAX_LOG_COMPILES=1` and a `jax.profiler` trace to count dispatches in one
iteration. `deeq` must come back bit-identical -- assert equality, not a tolerance.

### M2. Two modules still build their setup per species *label* rather than per dataset

`defumat/pseudo/projectors.py:180` and `defumat/paw/onecenter.py:540`. Both loop over
`pseudos`, which is one entry per species **label**, with no `_dataset_key` check of the
kind `augmentation.py:564` now applies. Two labels naming one UPF build two identical
radial form-factor blocks, and two identical `(nh, nh, nlm, mesh)` PAW tensor pairs.

**This bites exactly the runs this code advertises**: one species per magnetic site is the
standard way to write a noncollinear texture in a `pw.x` input, and site-resolved DFT+U is
the same pattern.

**Fix.** The fingerprint `build_augmentation` already uses (a blake2b of the radial
content, not the path), and share the object. P73's test asserts object *identity* rather
than equality, and that is the right assertion here too. Then give the `ProjectorCore`
arrays their own line in `sizing.py`, since they are resident alongside `vkb`.

**Bounded by.** P73: "fifteen nickel labels built fifteen identical 65.4 GB arrays, taking
the cell from 76.5 GB to 992.4 GB" -- **13x on that cell, from duplicate labels alone**,
closed for the augmentation and open in these two. And `build_paw` = **0.93 s** of the
1.53 s constructor on the one-species Pt PAW cell.

**Gain (arithmetic).** On P73's NiBr2 texture the projector columns fall ~20x. For PAW,
`N x 0.93 s` becomes `0.93 s` -- **13.0 s at `N = 15`, per `Calculation` built** -- and
the tensors `2 nh^2 nlm mesh x 8 x (N-1)` bytes, **2.18 GB** at `nh = 18`, `nlm = 25`,
`mesh = 1200`, `N = 15`. On a cell with one label per species it is worth nothing, which
is why it has never shown.

**Measure. No SCF needed.** Take `benchmarks/si8-paw-1k.in` and a copy whose
`ATOMIC_SPECIES` lists eight labels naming the same UPF; time `Calculation(system,
pseudos)` on each in a fresh process with `/usr/bin/time -v` for the peak. The eight-label
time should fall to the one-label time. Correctness is object identity plus an unchanged
total energy.

### M3. The Anderson mixer rebuilds the whole `n^2` Gram matrix each iteration, when one residual is new

`defumat/scf/mixing.py:163`. `gram = [[float(a @ b) for b in residuals] for a in
residuals]`, preceded by a fresh norm per history entry. At `history = 8` that is **72 host
BLAS dots per SCF iteration over vectors of `nspin x n_dense` doubles**, of which one row
and one norm are new -- and the matrix is symmetric, so half of even the new work is done
twice.

**Fix.** Cache the Gram matrix and the norms beside `_residuals`: append one row and
column per iteration, drop the leading one when the buffer rolls. `n+1` dots instead of
`n^2+n`, each the same dot of the same two vectors, so the numbers are unchanged rather
than reassociated. The intermediate, QE-faithful half is the upper triangle alone, a
factor of two on its own.

**Gain (arithmetic).** On the NiBr2 grid `PERFORMANCE.md` sizes P74 against (200x240x54,
`nspin_mag = 4`) one residual is **83 MB**, so the Gram alone streams **10.6 GB of host
memory per SCF iteration** where the cached form streams 1.5 GB -- a factor of 7 on a
stage that is pure bandwidth. On two-atom silicon the same arithmetic is 64 x 2 x 0.1 MB,
which is why nothing has ever seen it. **A large-cell item only.**

**Measure.** Direct and cheap: build an `AndersonMixer` with a synthetic 8-deep history of
the right shape and time `mix()` alone -- no SCF, so it runs in seconds. Then in place on
`benchmarks/al-slab.in`, whose history actually fills. The mixed density must be
bit-identical and the iteration count identical.

### M4. `calbec` conjugates the large operand where the same package's `project` conjugates the small one

`defumat/hamiltonian/operator.py:155`. `Hamiltonian._becp` computes
`einsum("gk,...g->...k", vkb.conj(), vectors)`: `vkb.conj()` is a separate materialised
`(npwx, nkb)` buffer -- XLA does not fuse an elementwise op into a dot operand on CPU,
since the dot is a library call -- and `_nonlocal` at `:285` then uses the *unconjugated*
`vkb`, so both copies are live in one expression. `_becp` runs three times per Davidson
step plus once per `apply_s`. `Projectors.project` (`projectors.py:83`) computes the same
quantity the other way round, conjugating the band block.

**Gain.** `unbounded` from existing numbers, and the entry says so. The buffer is 0.70 GB
at the 157-atom slab's shapes against a 94 GiB fit -- under one per cent, so **this is not
a memory item**; the time side is a conj pass over `npwx nkb` complex numbers per call
against an `h_psi` measured at 146.3 ms.

**Measure.** `si16-1k-ecut30` and `si8-us-1k` through `tools/benchmark.py`; the Davidson
line. Bit-identical.

### M5. The preconditioner contracts a block-diagonal `D` as a dense `(nkb, nkb)` **[moves a number]**

`defumat/hamiltonian/operator.py:306`. `diagonal(ik)` builds `h_diag` as
`einsum("gi,ij,gj->g", vkb.conj(), dij, vkb)`, where `dij` is the full matrix that
`projectors.py:65` documents as **block-diagonal over atoms**. opt_einsum contracts
`(g,i)@(i,j)` first, so the cost is `npwx nkb^2` complex MACs with an `(npwx, nkb)`
intermediate materialised before the reduction. Since `nkb = sum_a nh_a`, the dense form
costs `nat` times the block form. Three more sites: `overlap_diagonal` at `:223`, and the
spinor twins at `noncollinear.py:387` and `:401`.

**Fix.** Group the columns by atom -- `Projectors.atom_of_channel` is already a static
field carrying the grouping -- and contract block by block. Static shapes,
differentiability and `vmap` all survive.

**Gain (arithmetic).** On `si16-1k-ecut30` the dense form is 2.4e7 complex MACs against
1.5e6, a factor of `nat = 16`; low single-digit per cent of `h_psi` there, growing
linearly in `nat`. **The claim that scales is the buffer**: the `(npwx, nkb)` intermediate
is 6 MB at si16 and **703 MB** at the 157-atom slab shape.

**Measure.** `si16-1k-ecut30` and `si8-us-1k`. Time through `tools/benchmark.py`, memory
through `memory_analysis().temp_size_in_bytes` of the compiled `_every_k` -- the
instrument `davidson.py`'s working-set fit was made with. The converged total to 1e-12 Ry
and **the per-k Davidson step counts must not move**, which is what makes this a
number-moving change rather than a free one.

### M6. `return_steps` is a static `jit` argument, so a process that runs an SCF and then anything else compiles Davidson twice

`defumat/solvers/davidson.py:606`. `True` and `False` are two distinct compilations of the
entire solver -- `h_psi`, the subspace solve, both Ritz rotations. `run_scf`'s mixing loop
is the only caller that passes `True` (`driver.py:4184`); every other caller takes the
default, and `scf/residual.py:172` is one that runs **at the SCF's own shapes**.

**Gain.** `unbounded`. No figure exists for the compile time or the resident size of the
duplicate executable; the only anchor is the 146.3 ms of arithmetic it holds. The first
step is `JAX_LOG_COMPILES=1` on a script that does both, not a fix.

### M7. `sum_band` transforms the whole FFT box where `h_psi` uses sticks **[moves a number]**

`defumat/scf/density.py:62`. `band_density`'s `one_band` scatters into a full `(n1,n2,n3)`
box and does a fused 3D `ifftn`, while `Hamiltonian._local` takes the stick path. The
density is built from a *wavefunction* on the smooth grid, so it is not covered by the
"dense-grid quantities transform the whole box, which for them is the right thing"
exemption.

**Why it is tagged.** A stick transform and a whole-box transform differ in their
rounding, so the density moves in the last bits. The acceptance test is the converged
total to 1e-12 Ry and the density to round-off, not a bit-comparison.

**Gain, and the agent ranked it last itself.** `_local` does two transforms per band and
gained 1.13x / 1.02x; the density does one plus a square-and-accumulate, so at most half
that applies: ~1.06x at eight atoms, ~1.01x at sixteen, on a stage measured at 47 ms --
**under one per cent**. The ratio *falls* with cell size in the measured data, which is
the honest reading.

---

## S. Slab memory: large numbers, each needs a slab to confirm

All four are negligible on the small cells and none appears in `sizing.py`'s model. The
cells they are priced on are the 157-atom slab and P73's 45-atom NiBr2.

### S1. The symmetry maps are `(nsym, ngm)` int64 plus `(nsym, ngm)` complex, resident for the run

`defumat/system/symmetry.py:644`. `permutations` (int64) and `phases` (complex128), both
on the **dense** G set, held for the life of the `Calculation` (`driver.py:1458`).
`apply_symmetry_maps` then materialises a further `(nsym, ngm)` temporary per channel,
once per iteration per channel.

**Gain (arithmetic).** At `ngm = 3 536 849` and a 48-operation group: **4.07 GB resident**
plus **2.72 GB transient**, together 12.6 per cent of that run's 32.30 GB peak, from a
table nothing measures. int32 indices plus dropping an all-ones `phases` takes the
resident part to **0.68 GB**. Caveat stated by the entry rather than buried: that
particular run is a spin spiral and therefore `nosym`, so the figure is a projection onto
a symmetric run of that size.

### S2. The Anderson history is sixteen whole real-space densities on the host, reported as zero

`defumat/scf/mixing.py:120`. `_densities` and `_residuals` each hold `history = 8` entries,
and what is handed to them is the whole mixed state flattened -- real-space
`(nspin_mag, n1, n2, n3)` plus `becsum` and `ns`, through `np.asarray(...).ravel()`.

**Gain (arithmetic).** On the 157-atom slab (225x216x256, `nspin = 2`) one copy is 199 MB
and the sixteen are **3.18 GB**, against the 597 MB `sizing.py` reports for dense-grid
fields -- **the omission is 5.3x what is reported**. `history = 4` halves it. Tens of MB
on `si8-us-1k`. **Sibling of backlog item 4** in the sense that QE has a knob here
(`mixing_ndim`) that this input parser does not read.

### S3. The augmentation's `(nat, ngm)` structure factor is resident, and only a chunked scan reads it **[moves a number]**

`defumat/pseudo/augmentation.py:80`. On the 45-atom NiBr2 slab (`ngm = 3 536 849`) that is
**2.55 GB** held for the life of the `Calculation` -- the largest resident line the
tabulated branch has, on the very cell where P73's whole achievement was removing the 76.5
GB `Q_ij(G)`. `sizing.py:590` lists it beside a table measured in kilobytes. Its only
consumer is `_aug_chunk`, which already walks G in blocks.

**Gain (arithmetic).** 2.55 GB -> 0 if built per chunk, or -> 1.3 MB in the factorised
form (`ngm / (n1+n2+n3)` = 3878). 4.6 MB on `si8-us-1k`: **a slab finding, the same shape
P73 had.**

### S4. The augmentation's Bessel intermediate is the one radial transform with no chunk

`defumat/pseudo/augmentation.py:303`. `_qrad_kernel` forms `(ngm, kkbeta)` and hands it to
`spherical_bessel`, whose body builds five arrays at that shape before the einsum reduces
it. **Its own docstring records the size** -- 36257 by ~1100 on eight-atom ultrasoft
silicon, 300 MB, one per `L`. `pseudo/formfactors.py` chunks all four of *its* transforms
at `CHUNK = 4096` for exactly this reason.

**This is backlog item 10's site**, from the other end: item 10 proposes a `custom_jvp` to
shrink the reverse-mode tape, and this is the forward transient, which chunking bounds
without touching the derivative. Both are wanted; neither substitutes for the other.

**Gain (arithmetic).** 319 MB -> 36 MB on `si8-us-1k`, a factor of `ngm/4096 = 8.9`. At
the 2 GiB gate with `nh = 8`, ~18.4 GB -> 36 MB.

### S5. `matrix_elements` stacks a second whole copy of the wavefunctions before contracting it away

`defumat/response/velocity.py:301`. For each cartesian axis,
`einsum("skmg,skng->skmn", psi.conj(), self.apply(psi, axis))`. `apply` runs through
`map_k`, and **`map_k` stacks**: a full `(nspin, nk, nbnd, ndim)` array of `dH/dk|psi>` is
materialised in its entirety before one element of the `(nbnd, nbnd)` answer exists. The
`k_batch` dial chunks the work and not that output -- the mechanism OPEN.md D1 and D3 were
closed for in two other modules.

**Fix.** Move the contraction inside the mapped body so `map_k` stacks `(nbnd, nbnd)` per
k. The `jvp` is over `kcart` and the conjugated bra carries no k tangent, so it commutes
with the contraction. **It does not apply** to the `over_kpoints` callers in `phonon.py`,
`strain.py`, `efield.py` and `electrostriction.py`, where the full-width output is the
Sternheimer right-hand side and is wanted.

**Gain (arithmetic).** The intermediate has exactly the shape of the states, so it is a
second **1.0 GB** on P51's spinor nickel (512 k-points, 36 bands), a peak of 2.0 GB where
the section accounts for 1.0. Contracting inside the map leaves 2 MB at the CPU default.

**Do not confuse this with the claim P54 rejected.** P54 measured `jax.linearize` here as
a *speed* fix and found it worth nothing (6.33 s against 6.35), and measured
`eqx.filter_jit` at 2x for 1.33 -> 8.96 GB and refused it on the memory. This is the
memory direction, which is the one still open.

---

## Y. `conv_thr` bounds a moment much more weakly than it bounds a charge, and the PAW one-centre terms not at all

### Y1. `dr2 = 9e-11` on a magnetic cell left the total energy 1.15e-8 Ry out **[found by a test]**

**Closed as a test fix, kept here as the measurement**, because the mechanism is general
and the next person to pick a `conv_thr` for a magnetic quantity needs it.

`tests/unit/test_angular_momenta.py::test_the_orbital_moment_rotates_with_the_magnetization`
drives nickel's moment along z, x and y in turn and asserts `|<L>|` is the same in all
three -- the cubic axes are equivalent, so it is a pure symmetry check. At the
`conv_thr = 1e-10` it used, the three runs **stop at different states**: x at
-335.167135773432 Ry in 15 iterations, z and y at -335.167135784928 in 16, which is
**1.15e-8 Ry** apart, and the spread in `|<L>|` is **2.3e-7** against the test's own 1e-9.
At `conv_thr = 1e-12` all three take 20 iterations, agree to **3e-12 Ry**, and `|<L>|`
spreads by **7.4e-10**. The tight value, 0.03647659, is the one the docstring quoted, so
the *x* run was the accurate one at 1e-10 and the other two were stopping short.

**Why `dr2` did not see it.** `rho_ddot` weights the charge residual by `1/G^2` and the
magnetization residual by a constant -- a factor of 13.6 apart at `G_min` on
`fe-mag-1k` -- so on a magnetic cell an `accuracy` below `conv_thr` bounds the moment much
more weakly than it bounds the charge. `<L>` is a moment. This is `NONCOLLINEAR.md` item
18's prediction, found in the wild the same day the split that measures it landed: the two
halves are now on `scf.history` as `charge_accuracy` and `magnetic_accuracy`.

**It recurred twice more the same day**, which is what makes it a section rather than a test
fix. A noncollinear DFT+U promotion and its collinear source agree in **total energy to
6e-9 Ry** and in `Tr ns` to **6e-5 out of 4.34**, both having stopped on `dr2 < 1e-8`. And a
from-scratch spinor run of the same cell converged 6.5e-6 Ry *above* the promoted one, into
a neighbouring minimum with off-diagonal spin traces of 1e-5 -- a real cant of the size the
convergence criterion does not resolve. In all three cases the energy is converged two to
four orders tighter than the magnetic quantity read off the same state.

**A fourth instance, 2026-09-15, and it is the sharpest reading of `accuracy` so far.**
`tests/regression/test_continuation.py`'s iron round trip (Part VIII item 1) fails at
`conv_thr = 1e-8` and fails again at `1e-10`, the gap staying at about 1.5e-7 Ry across two
orders of the threshold. The reason it does not move is that at both values the continued
run still exits after **one** iteration, and a one-iteration continuation is where the
weighting above does its worst: the run reports `accuracy = 4.5e-11` while its total energy
is 1.0e-7 from the fixed point, a factor of two thousand, against a factor of about two for
every from-scratch run of the same cell. The moment is what is moving, 5.7e-4 mu_B between
the two loose runs against 5e-6 between the tight ones. `1e-12` fixes it not by being
tighter but by rejecting that first residual, so the run takes three iterations; if
`accuracy` is ever recalibrated, that test starts passing at `1e-8` again and nothing will
say why, which is why the number is here as well as in the docstring.

**What is not settled.** The three-orientation spread is now measured at one cutoff on one
cell. Whether `1e-12` is the right default for a magnetic run generally, and whether the
`ethr` schedule should be driven by the magnetization half rather than the sum on a
magnetic cell, are both open and neither has a number. `pw.x` uses the same summed
`rho_ddot` for its schedule, so this would be a deliberate departure rather than a
correction.

### Y2. `accuracy` cannot see the PAW one-centre half of `becsum` at all, and the docstring gave the wrong reason **[opened 2026-09-14; the run it was found on is *not* stalled on this, measured and retracted the same day]**

Y1 is about a half that is weighted a factor of 13.6 too lightly. This is about a half that
is not in the number at all, and the two compound on exactly the same cells.

**What the code does**, checked rather than remembered. `scf_accuracy_split`
(`defumat/scf/potential.py:158`) takes the **smooth** density residual and returns
`(dr2, charge, magnetic)`; the mixing loop builds `accuracy` from it at
`scf/driver.py:5019` and the residual solver's `accuracy_of` at `:4326` does the same,
adding `ns_ddot` where there is a Hubbard U. `becsum` enters neither. What reaches the
number is whatever `addusdens` already put on the grid, `Q_ij(r) becsum`, at the charge
half's `1/|G|^2` weight; what does not reach it is the PAW one-centre piece, the
all-electron minus pseudo Hartree and XC on the radial grids.

**This matches `pw.x` exactly and is not a deviation.** QE writes the term and comments it
out, `PW/src/scf_mod.f90:843-845`:

```fortran
  ! Beware: commented out because it yields too often negative values
  ! IF (okpaw)         rho_ddot = rho_ddot + paw_ddot(rho1%bec, rho2%bec)
```

**What was wrong here was the *reason*.** `accuracy_of`'s docstring said `becsum`'s share
is "the `paw` term, which the loop adds through `addusdens` rather than separately", which
reads as redundancy: nothing missing, just added elsewhere. QE's comment is an admission
that the term is missing and was given up because `paw_ddot` is not positive definite. On a
PAW magnetic cell the difference is the whole question, because the moment lives in the
d-shell `becsum`. Both docstrings now say so.

**Where it bites, reported by the NiBr2 helix run and not re-measured here.** 45 atoms, PAW,
noncollinear with `lspinorb`, `1 6 1`, `nosym`, a 15-site helix held by a
`LOCAL_MAGNETIC_FIELDS` card on Ni only, `ecutwfc 45 / ecutrho 240`, `nbnd 403`,
`conv_thr 1e-6`, `mixing_beta 0.3`, Anderson with `mixing_ndim 12`. It stalls, and the split
Y1 added is what makes it legible:

| iteration | 9 | 13 | 19 | 22 | 28 | 33 | 37 |
|---|---|---|---|---|---|---|---|
| `accuracy` | 8.28e-3 | 7.69e-4 | 2.07e-4 | 1.84e-4 | 1.62e-4 | 1.47e-4 | 1.75e-4 |
| charge | 7.68e-3 | 6.11e-4 | 1.04e-4 | 1.03e-4 | 7.52e-5 | 5.57e-5 | 8.66e-5 |
| magnetic | 6.04e-4 | 1.59e-4 | 1.02e-4 | 8.12e-5 | 8.66e-5 | 9.13e-5 | 8.84e-5 |

The magnetic half falls two and a half orders and then stops dead: sixteen iterations inside
six per cent of 8.7e-5, drifting slightly upwards. The charge half is the noisy one and the
only one that ever goes lower. Three things reported with it that rule out the ordinary
explanations: the cell is **gapped** (indirect 1.10 eV, direct 1.17 to 1.21 eV over the six
k-points, 360 of 403 bands occupied and **no** band fractionally occupied at 1e-5), so
`degauss` is inert and charge sloshing is out, since sloshing needs the `q^-2` divergence of
a metallic response; the **total energy swings 0.27 Ry** over iterations 29 to 37 against a
reported accuracy of 1.5e-4, a ratio of about 2000 where `dr2` is meant to bound the energy
error within an order of magnitude; and the Davidson has not settled, 50.7, 40.3, 37.2, 35.5,
then 7.7, 4.0, 3.5, then 30.5, 21.0, 21.7, 5.8, 13.8 average steps, where a gapped insulator
at `ethr` 4e-8 belongs at 2 to 4. A large energy motion with a small smooth-density residual
is what points at the one-centre terms. The checkpoint is kept at
`/scratch/work/ladovj1/calculations/NiBr2_defumat/k161-h200/scf_iteration.npz`
(`becsum_0`, `becsum_1`, eigenvalues and occupations are all in it).

**That last step was a hypothesis, it was measured, and it is wrong.** Retracted by the
session that raised it on the same day, from the live checkpoint of the running job. The
checkpoint's `__meta__` carries `energy_terms`, so the swing **decomposes** instead of
having to be inferred, and the first pair of iterations off the local-TF run reads:

| Ry | iteration 29 | iteration 30 | change |
|---|---|---|---|
| total | -8926.24777949 | -8926.19519665 | **+0.052583** |
| one-electron | -13810.642537 | -13810.603247 | +0.039290 |
| Hartree | 7092.376134 | 7092.391386 | +0.015252 |
| XC | -886.177080 | -886.179266 | -0.002186 |
| one-centre PAW | -6554.149937 | -6554.149710 | **+0.000227** |

The four changes sum to the total move to every digit, so nothing is hiding in a term that
was not listed. The one-electron term is 75 per cent of the move and the Hartree 29; the
PAW one-centre term is **0.4 per cent**, and `becsum` itself moved 4.45e-2 against a norm of
25.54, which is 0.17 per cent. **The swing is in the eigenvalues and the smooth density**,
which is where the Davidson counts were pointing all along, and "`becsum` is what is stuck"
does not survive the measurement.

**The method is worth more than the retraction.** Decomposing `energy_terms` off the live
checkpoint is cheaper and sharper than norming `becsum` and inferring, and it samples
`__meta__` and `becsum_*` alone rather than reading the 12 GB of wavefunctions in the same
`npz`. It is the right first move on any stalled run whose energy is moving more than its
`dr2` allows. Note what it took: **it took a decomposition to rule `becsum` out, and no
number in the log could have**, which is the argument for the diagnostic below rather than
against it.

**Two things that make the blind half less damped than the rest.** Both preconditioners pass
`becsum` through at the plain `beta` -- `kerker_preconditioner` gives the argument
(`scf/mixing.py:296-308`, the parts whose Jacobian block is already well conditioned want no
preconditioning) and `local_tf_preconditioner` repeats it at `:411` -- so `mixing_beta` is
`becsum`'s only damping. And `ultracell/driver.py:378` already records the ultracell version
of Y1, that the charge half's weight at the envelope's own wavevector is `N^2` larger and
"a spin density wave is exactly a state whose whole answer lives in the half that is not
amplified". A 15-site helix is that state.

**What stands after the retraction.** The code reading does, and it is the entry: `dr2`
cannot see the one-centre residual, `addusdens` carries the augmentation charge and not the
all-electron minus pseudo Hartree and XC, and QE's `paw_ddot` is disabled for indefiniteness
rather than for redundancy. What is retracted is only that this was the cause of *this*
stall.

**What to do, and what not to.** Do **not** put `paw_ddot` back into `dr2`: QE disabled it
for a real reason, and here it would be worse, because `scf_accuracy_split`'s own docstring
records that one ulp in `dr2` moves the `ethr` schedule and with it the last digits of every
eigenvalue, so an indefinite term would corrupt eigenvalues and a negative `dr2` would break
the schedule outright. What is wanted is a **`becsum` residual reported beside the two
halves and deliberately fed to nothing** -- not to `ethr`, not to the `conv_thr` test. A
plain norm is enough; it need not be `paw_ddot` and need not be an energy. That makes the
blind half visible for the cost of one norm and keeps the property the docstring protects,
that a diagnostic must not change the run it is diagnosing. Unstarted. The episode above is
the argument *for* it: ruling `becsum` out cost a live-checkpoint decomposition, where one
norm on the log line would have done it on iteration 20.

### Y3. `davidson_unconverged` is computed every iteration and was never printed **[fixed 2026-09-14]**

The log gave `avg # of iterations` and nothing about whether bands were being abandoned,
and the two are not the same question: a step count says how hard the solve worked, not
whether it gave up. On the NiBr2 helix's first local-TF iteration the line read
`avg # of iterations = 100.0`, which is `MAX_ITERATIONS` **exactly** -- meaning every
k-point was cut off mid-flight rather than that the last one took a hundred steps, and
those read identically. `davidson_unconverged`, the worst k-point's `notcnv`, was already
computed at `scf/driver.py:4999` and already on `scf.history` at `:5221`; it is now
appended to the same print line, and only when it is nonzero, so a healthy run's line stays
byte for byte `pw.x`'s. It fires: silicon at `david = 2`, `nbnd = 40`, `diago_full_acc`,
`conv_thr = 1e-12` prints
`ethr = 3.55E-05,  avg # of iterations = 19.8,  up to 1 of 40 bands unsettled`.

**What it would have been worth on the run that asked for it**: the Anderson control ran
50.7, 40.3, 37.2, 35.0, 35.5 steps, settled to 7.7, 4.0, 3.5, then went back to 30.5 and sat
near 20 for the rest of the run -- it settles and then destabilises, which is not a
cold-start artefact and which a gapped insulator at fixed `ethr` has no business doing. That
was diagnosed at iteration 41 by a decomposition (Y2) where the line would have said it at
iteration 20.

**What is still open behind it** is the destabilisation itself, which this only makes
visible. `OPEN.md` Part VI item 2 (`diago_david_ndim = 2` at the minimum subspace) and the
`davidson-empty-ethr-defect` item are the two neighbouring candidates, and neither has been
run against a gapped spinor PAW cell.

### Y4. PAW's collinear and noncollinear paths do not converge onto each other, and neither do `pw.x`'s **[opened 2026-09-16, closed the same day: the gap is the reference's, reproduced to four digits]**

**The same cell, the same physics, twice.** An oxygen chain in a cell doubled along `z`
with the two moments antiparallel, run once as `nspin = 2` and once as `noncolin` with
both moments turned into the plane. Nothing about the state differs -- without spin-orbit
coupling the energy cannot know which axis the moments lie on -- so the two totals are the
same number computed by this code's two SCF paths.

| `ecutrho` | 200 | 300 | 400 |
|---|---|---|---|
| ultrasoft (`O.pz-rrkjus`) | 7.03e-07 | 1.32e-07 | **6.11e-13** |
| PAW (`O.pz-kjpaw`) | 8.51e-05 | 4.94e-06 | **2.65e-06** |

**Ultrasoft converges onto itself and PAW does not.** At `ecutrho = 400` the ultrasoft
pair agree to 6e-13 Ry, which is round-off, while the PAW pair sit 2.65e-06 apart and
stop falling -- the last two cutoffs move it by less than a factor of two where the first
step moved it by seventeen.

**It is not `conv_thr`, which is the obvious reading and is the one this section is
otherwise about.** Tightening from 1e-11 to 1e-13 costs each side four more iterations
(13 to 17 collinear, 16 to 19 noncollinear) and leaves the gap at **2.6466e-06 Ry,
identical to every printed digit**. So this is not Y1's mechanism at one more remove: it
is a reproducible difference between two code paths on the same physics, and the fact
that only the PAW dataset shows it points at the one-centre terms, which are the only
machinery the two paths do not share bit for bit.

**How it was found, and what it was nearly blamed on.** P95 validates an augmented spin
spiral against supercells, and one of those references is collinear. The residue that
identity left looked exactly like a G-sphere truncation -- ultrasoft-only, absent for a
norm-conserving dataset, falling with `ecutrho` -- and was written up as one before the
control above was run. The control showed the spiral residue is **half** the number in
this table at every cutoff, the factor a doubled cell's energy is divided by, so the
spiral was contributing nothing and the whole ladder was this. The spiral's own error is
1.65e-09 Ry, measured against a supercell that shares the noncollinear path and so never
touches this.

**What it is not, measured rather than argued.** It is not a structure factor or a
grid-alignment effect: both atoms of the doubled cell sit on exact FFT grid points (`z = 0`
and `1/2` on grids of 48 dense and 36 smooth), and on the one-atom cell of the same family
moving the atom to another exact grid point costs **1.4e-11 Ry** for the same PAW dataset.
That leaves the one-centre XC or Hartree on the spheres, or a difference in how `becsum`
is symmetrised or spin-transformed between the two paths -- which is the only machinery
the collinear and noncollinear routes do not share bit for bit.

**That was the first thing to run, and running it settles the direction.** `pw.x` has
both paths too, so the same four cells went through it, single core, same inputs, with
the totals read out of QE's own XML rather than off its eight-decimal stdout:

| `ecutrho` | 200 | 400 |
|---|---|---|
| ultrasoft, this code | 7.03e-07 | 6.11e-13 |
| ultrasoft, `pw.x` | **7.0339e-07** | **1.64e-11** |
| PAW, this code | 8.510e-05 | 2.6466e-06 |
| PAW, `pw.x` | **8.5102e-05** | **2.6469e-06** |

**QE has the same gap, to four digits, on both datasets and at both cutoffs**, including
the flattening: its PAW pair stops falling at 2.6469e-06 where its ultrasoft pair reaches
1.6e-11. So this is a property of the PAW method both codes implement and not a defect
introduced here, which is what the entry above could not tell apart.

**And each path separately reproduces its own reference**, which is the part that
excludes the coincidence of two different errors leaving the same difference: on the
committed `ecutrho = 200` cells this code sits **1.0e-09, 2.2e-09, 5.2e-09 and 8.3e-09
Ry** from `pw.x` for collinear ultrasoft, noncollinear ultrasoft, collinear PAW and
noncollinear PAW -- at or below the 1e-08 the reference is printed to. A gap that agreed
by accident would need both members to be wrong and wrong by the same amount.

**What it does not say.** It does not say 2.65e-06 Ry is right. Without spin-orbit
coupling the energy cannot depend on which axis the moments lie on, so the two paths
*should* give one number, and at `ecutrho = 400` they do not -- in either code. What is
settled is where to look for it, and it is not in this repository's transcription: the
one-centre machinery here follows `paw_onecenter.f90`, and a faithful transcription is
exactly what reproduces a shared convention's residue to four digits. Chasing it further
is a question about the PAW noncollinear one-centre treatment itself, which is outside
what this project validates against.

**The pair is committed rather than left as a session's scratch**, so nobody re-runs
`pw.x` for it: `tests/data/qe/o-chain-afm-nc-us.in` and `o-chain-afm-nc-paw.in` beside
the collinear pair that was already there, with all four references, and
`tests/regression/test_spin_spirals_augmented.py::test_the_two_paths_disagree_by_what_pw_x_disagrees_by`
pinning each path to its own reference **and** the two gaps to each other. The test
asserts the disagreement rather than agreement: an assertion that the two paths agree
would fail, and one about this code alone could not have told a shared convention from a
transcription error.

**What it changes upstream.** P95's `CONSISTENCY_RY = 2.0e-06` was set by PAW's 3.26e-07
on the quarter turn, with a note saying it wants revisiting downwards if that number is
ever traced to a term. It has now been traced as far as it goes: the floor is the
method's, shared with `pw.x`, so the tolerance is a property of the physics at that
cutoff rather than a number waiting on a fix.

**It showed up again the same day, one derivative out, and the discriminator is worth
keeping** (P96). `dE/dq` on the same PAW chain agrees with a finite difference of the
re-converged energy down to 1.4e-06 Ry per unit `q` and then stops falling, where the
ultrasoft arm keeps converging as `delta^2` to 2.4e-07. The dial that moves it is this
entry's: `conv_thr` and the mixing leave it where it is, and doubling `ecutrho` takes it
from 1.964e-06 to 2.812e-07. The cheap way to measure a gradient's own error is a
**wavevector symmetry forces to zero** -- `E(q)` is even and periodic, so `dE/dq` vanishes
exactly at `q3 = 0` and `1/2`, and what comes back there is the error with no finite
difference and no truncation in it, at one SCF and one gradient per rung instead of three.

## X. Downgraded, and test-suite hygiene

### X1. The analytic force recompiles per ionic step -- real, and not the severity the entry claims

`defumat/forces/analytic.py:112`. `_compiled_terms` keys its cache on object identity and
`at_positions` returns a `copy.copy`, so the six-term kernel is traced and compiled again
on every step. The mechanism is real and the identity key is deliberate -- `_terms` reads
the positions off the captured object, so the geometry genuinely is folded into the trace.

**The entry's framing is wrong and the correction is worth more than the fix.** It reports
this as contradicting `PERFORMANCE.md` ("`at_positions` already keeps its compiled
force"). That sentence is about the **default** `jax.grad` route. The analytic force is
`method='analytic'`, an opt-in cross-check, and is not on the relaxation path at all -- so
"a relaxation recompiles at every ionic step" holds only for a relaxation that asked for
the analytic force. Sibling of backlog item 6, at a fraction of its weight.

### X2. The two bounds `CLAUDE.md` requires of a multi-cell test file exist in one file out of thirty-two

`tests/regression/test_response.py:65` and thirty-one others. `CLAUDE.md` names two bounds
for any file running more than ~three distinct cells: `jax.clear_caches()` in an autouse
fixture, and `lru_cache(maxsize=2)` on the converged-state helper. **Thirty-two files carry
`lru_cache(maxsize=None)`; `jax.clear_caches()` appears in exactly one**,
`tests/regression/test_phonons.py`. `tests/conftest.py`'s single autouse fixture is the
memory watchdog, not a cache clear.

**Bounded by** P28b, which is the measurement of exactly this: without the clear, resident
memory over ten cases went 0.55 -> **3.67 GB**, monotonic; with it, flat at **1.45 GB**,
and faster as well as smaller.

**The trade is not free** -- it buys recompilation for a lower peak -- so measure file by
file with `DEFUMAT_TEST_MEM_MAX=12G tools/run_regression.sh <file>` and read the `peak=` on
the summary line, one file at a time and nothing else on the machine. This is the entry
most likely to pay for itself, given that Part I item 2 is still open.

### X3. A refusal test in the fast gate builds four full `Calculation`s to check four static guards

`tests/unit/test_piezo_machinery.py:97`. Four parametrized cases each construct a real
`Calculation` -- G-vector enumeration, both FFT grids, the projector core and `vkb`, and
`build_augmentation` for the ultrasoft case -- so that
`require_a_piezoelectric_tensor` can read `nspin`, `is_ultrasoft`, `occupations` and the
symmetry.

**The fix is the better boundary, not the cache.** Have the guard chain take the `System`
and the pseudopotentials it actually reads, which makes the refusal fire at the right
place for a *caller* as well as for the test. **A refusal that has to allocate the
calculation it is refusing is a refusal at the wrong boundary.**

---

## What the four lenses read and judged clean

A negative is a result: this is where **not** to look.

- **`defumat/batching.py`** -- clean, and the best-argued file in the sweep. Every
  `jax.vmap` call site in the package was grepped: **none walks the k axis by hand.** The
  only vmaps in the hot path are over the spin axis, already measured free at width one.
- **The SCF driver's iteration body** -- `next_ethr` has all three of `electrons.f90`'s
  details (reset at iteration 2, monotone `min`, 1e-13 floor) and takes the absolute
  iteration number across a resume; the attempt loop reproduces `electrons.f90:890-908`
  including rebuilding the band thresholds inside it. The per-iteration host syncs are
  outside the jitted body and are what a Python SCF loop costs.
- **`solvers/davidson.py` apart from H5** -- it is `cegterg`'s loop, and the two things it
  does at full `nvecx` width are backlog items 2 and 3, not re-reported.
  `solvers/subspace.py`'s `robust=False` path has no `lax.cond` in it, which is the 2.85x
  measured fix.
- **`basis/fft.py` and `sticks.py`** -- the stick pair is QE's `(n3, n1, n2)` layout with
  the potential stored to match. Nothing here that is not already in "Measured and
  rejected".
- **`xc/lda.py` and the local slot** -- already the one-pass `value_and_grad` (which is
  what makes H1 an omission rather than a design choice), and `clamp_polarization` is
  already the `jnp.where` form the saturated-magnet trap requires.
- **`topology/`** -- described as the most carefully optimised corner of the package:
  `eqx.filter_jit` on the four primitives, host-side `searchsorted` on a presorted key,
  and a **weak** augmentation cache whose docstring explains that an `lru_cache` would pin
  `Q_ij(G)` and turn a 200 MB working set into 14 GB.
- **`tddft/chi0.py`, `workflows/transport.py`, `sizing.py`** -- OPEN.md D1/D2/D3, closed,
  with their tables in `PERFORMANCE.md`. Not re-reported.
- **`pseudo/formfactors.py`** -- all four radial transforms already chunked at 4096, which
  is the template S4 asks for.
- **`pseudo/projectors.py`'s `at_positions`** -- rebuilds `vkb` from a species-resolved
  core times one phase per atom, so a moved geometry costs one complex exponential per
  atom and the radial integrals stay outside the gradient. The right factorisation.
- **`SternheimerSolver._preconditioner` and `project`'s `S|psi_occ>`** -- both rebuilt per
  solve, both genuinely redundant, and both **well under 1 per cent** of a solve that runs
  22-28 `h_psi` steps on the same block. Recorded here so the next sweep does not spend
  the hour this one did.
- **`response/magnetoelectric.py`, `effmass.py`, `nesting.py`, `spectra.py`,
  `diffraction/structure_factor.py`** -- their loops are over quantities that genuinely
  need a separate ground state or a separate diagonalisation. No repeated work to remove.
- **`projwfc/`** -- both known costs are already written up with their fixes in P8 and
  P69.

**What the sweep did not cover.** The memory half of the response lens is thin: almost
every allocation there is already sized in `PERFORMANCE.md`, and the two that are not --
the `(3 nat, nspin, dense grid)` arrays the phonon loop holds six or seven of at once, and
`keep_internals`' `3 nat` state blocks -- both fall inside **backlog item 9**, which bounds
them at three modes in flight.

---

# Part IV -- from the 2026-09-13 magnetism session (P80)

---

## 1. A JAX thread-pool deadlock, parked inside `Calculation.symmetry_residual`

**Reproducible in the fast gate, with a stack trace, and not a slow compile.** Seen five
times on 2026-09-13, always at `tests/unit/test_seed_symmetry.py`:

| where | wall | CPU in that time | what it was |
|---|---|---|---|
| the file alone, first run of the session | 5.5 min | **3.4 s** | killed; reran and passed in 7 s |
| the whole gate, at 74% | >2 min | **0 s** over a 20 s sample | killed |
| the whole gate again, same file | 85 s+ | **0 s** | killed |
| `pytest tests/unit` alone, ~70% | >2 min | **0 s** | dumped (below) |
| a bare measurement script, first cell | several min | **3 s** | killed |

`faulthandler_timeout` gives the answer (`pytest -o faulthandler_timeout=120`, which is the
way to get a stack here -- `py-spy` and `gdb` are not installed):

```
File ".../jax/_src/array.py", line 642 in _value
File ".../jax/_src/array.py", line 300 in __float__
File "defumat/scf/driver.py", line 2948 in relative
File "defumat/scf/driver.py", line 2963 in symmetry_residual
File "tests/unit/test_seed_symmetry.py", line 127 in ...
```

So it is parked in a blocking device-to-host transfer, with **28-35 threads all in
`futex_wait_queue` and no CPU at all** -- the main thread waiting for a result and every XLA
worker idle. Not compilation: the same file with the persistent cache **off**, so every
kernel compiled from scratch, runs its four tests in **9.8 s**. Not a file lock either: no
lock or temporary files in `~/.cache/defumat/jax` (19,362 entries, 1.5 GB), and the blocked
process holds no open file in it. That leaves a deadlock in XLA's CPU thread pool, which
nothing in this project can fix.

**What fixed it, after one wrong attempt.** The first try reduced `symmetry_residual`'s
**four** host syncs to one -- `float()` on each norm as it went became a single jitted kernel
returning both ratios. Fewer syncs was the right instinct and the wrong fix: it **deadlocked
the same way** on the next gate run, at the same 74%, moving only from the second test to the
first. So the trigger is dispatching *any* jitted op at that point, not the number of
transfers.

The ratios are now plain **NumPy on host arrays** (:func:`_relative_residual`), which removes
the dispatch entirely, and the gate then ran **1901 passed, 176 skipped, 0 timeouts, in
6m58s** -- its normal time, against the 30-plus minutes the hung runs were taking before they
were killed. The symmetrisation itself stays compiled, because it is the same kernel the SCF
runs every iteration. **Keep it on the host**: it is a once-per-run diagnostic on two arrays
already in hand, so a kernel buys nothing and costs this.

**`faulthandler_timeout = 600` is now in `pyproject.toml`**, which is what turned four
anonymous kills into a stack trace in one run.

**What is still open** is why that call site of all of them -- a small norm kernel dispatched
immediately after a jitted symmetrisation -- parks every worker thread, and whether anything
else in the package dispatches in the same pattern. Nothing else is known to hang, and this
was found rather than looked for, so the honest answer is that the underlying JAX behaviour is
not understood.

**The recipe, which is the part to keep.** A stalled JAX process and a busy one look
identical from outside: both sit in `futex_wait_queue` on the main thread, because that is
where the main thread waits for XLA's workers. What separates them is **CPU time sampled
twice**:

```bash
ps -p <pid> -o time --no-headers      # ... wait 20 s ...
ps -p <pid> -o time --no-headers      # unchanged => stalled, not slow
```

A healthy gate here moves several CPU-minutes per 25 s of wall clock (a handful of busy
threads); a stalled one moves nothing. Reading `wchan` alone says nothing, and neither does
`%CPU`, which is an average over the process's whole life and stays high long after it stops.

### **Reopened and then closed, 2026-09-13 (later the same day): it is the affinity mask, and it is this project's own code**

**The entry above says "a deadlock in XLA's CPU thread pool, which nothing in this project
can fix." That is wrong, and the word doing the damage is *nothing*.** The trigger is
`defumat._limit_thread_pool`, which narrows the process's affinity mask to four cores at
import. XLA sizes its CPU thread pool from that mask, and the hang rate follows the mask
and nothing else.

**How it came back.** A gate run for an unrelated change hung at **73%**, on
`tests/unit/test_seed_symmetry.py`, in the same test as before. The recipe above identified
it correctly -- CPU time `00:28:19` at two reads 25 s apart, 45 threads all in
`futex_wait_queue`. `faulthandler_timeout = 600` then produced a stack, and it is not the
one on record:

```
File "defumat/scf/driver.py", line 638 in _relative_residual      <- np.asarray(a)
File "defumat/scf/driver.py", line 2998 in symmetry_residual
File "tests/unit/test_seed_symmetry.py", line 127 in ...
```

**Two theories died before the real one turned up, and both had been written down here as
fact.**

- *The dispatch theory.* The previous fix's claim -- "doing the arithmetic on the host
  removes the dispatch entirely" -- was never true of the code. `np.asarray` on a
  `jax.Array` is the same blocking device-to-host wait a `float()` is, so the transfer moved
  one line rather than going away; and the `pairs` above it were still built from **eager
  device ops** (`symmetrized[0] + symmetrized[1]`, `symmetrized[1:4]`). That is now really
  fixed -- each array is transferred once, before any arithmetic, so nothing at all sits
  between the compiled symmetrisation and NumPy -- **and it did not stop the hang.** Worth
  keeping because it makes the code mean what its docstring says; worthless as a cure.
- *The cache theory.* This entry's strongest datum was "cache off, 9.8 s clean; cache on,
  hang". `DEFUMAT_CACHE_DIR=off` now hangs too, at 9.9 s of CPU over a 400 s wall clock.
  The clean run on record was a lucky draw, which is what a 1-in-3 failure looks like when
  it is sampled once.

**The measurement that settled it.** Three files that hang together and pass separately
(`test_seed_symmetry.py`, `test_textured_symmetry.py`, `test_textured_seeding.py`), run
eight times at each mask, 150 s timeout, otherwise idle machine:

| `DEFUMAT_THREADS` | cores | hangs | wall when it passes |
|---|---|---|---|
| `2` | 2 | **8 / 8** | -- |
| `4` (the package default) | 4 | **6 / 14** | 32-35 s |
| `8` | 8 | **0 / 8** | 36-38 s |
| `off` | 12-14 | **0 / 14** | 37-39 s |

The two-core failures are stalls and not slowness, by this entry's own recipe: **4 seconds**
of CPU, unchanged over a 30 s sample, 23 threads all in `futex_wait_queue`. That shape --
every worker blocked, no CPU, rate rising sharply as the pool shrinks -- is thread-pool
**exhaustion**: something waits on the pool from inside it, and with fewer workers the wait
has nowhere to go.

**Why the default is not simply raised.** Four cores is not an arbitrary number; it is the
fastest setting for the physics, and widening the mask costs far more than the hang does.
Median of five, compiled, on `benchmarks/si8-1k-ecut30.in`, per SCF iteration:

| cores | 4 | 6 | 8 | 12 |
|---|---|---|---|---|
| ms/iteration | **238** | 368 | 411 | 385 |

So eight cores costs **73%** of the production speed to buy reliability. That is the wrong
trade for a calculation and the right one for a test suite, and the two are therefore
separated:

- **Production keeps `DEFAULT_THREADS = 4`**, with the deadlock named in
  `_limit_thread_pool`'s docstring and `DEFUMAT_THREADS` the escape.
- **The suite runs at eight**, set in `tests/conftest.py` before anything imports
  `defumat`, because the mask is read once at import. The gate is not a performance
  measurement -- nothing may be timed beside a test run in any case -- so it can afford what
  the mask costs, which is **11% of wall clock and 13% of peak RSS**: 27.4 s and 1752 M at
  four cores against 30.3 s and 1981 M at eight, median of three each on
  `tests/unit/test_textured_seeding.py`. Eight rather than `off` because this machine is
  shared, and a gate that takes every core takes them from another session. An explicit
  `DEFUMAT_THREADS` still wins, so the hang can still be reproduced on demand.
  `tests/unit/test_config.py` guards the line, since the mask is set once at import and a
  conftest that stopped setting it would fail nothing else.

**The gate's own figures are one sample each and are not that measurement.** It ran
**1945 passed, 176 skipped, 0 failures in 10m37s, peak 8186 M**, against 7m22s and 5903 M
for the last four-core run. The wall clock sits inside a spread this project already records
as unattributable -- the same suite has read 7, 11 and 7 minutes on the same day -- and the
peak is above the 4.5-6.0 GB previously seen, of which the mask explains 13 points and the
rest is not explained. 8186 M is still under the 12 G cap and its 0.85 watchdog threshold,
with less margin than before.

**What is still open, and it is now a much smaller question.** *Which* nested wait exhausts
the pool. Every hang seen has been in a pytest process that had already compiled many
different cells, and none in a single long calculation, so the suspect is a dispatch issued
while a pool worker is itself blocked -- but that has not been localised, and the stack is
unhelpful because the main thread is only ever the one waiting for the result. **A hang
under a mask of eight or more would reopen this**; six months of clean gates would close
the question rather than the symptom.

**And a gate that hangs needs a kill.** `faulthandler_timeout` dumps the stack at 600 s and
then the process **sits there** holding its memory -- 4.9 GB here. The dump is the
diagnosis, not the recovery, which is one more reason to run anything long through
`tools/run_regression.sh`.

## 2. P63's spin-spiral scan no longer reproduces **[closed 2026-09-13 -- the cause is the seed, and the scan is better than P63's]**

**It was never a regression. `starting_magnetization = 1.0` on hydrogen is a *fully
polarized* atom** -- one valence electron, so the minority channel starts at exactly zero
and the first potential is built at `|zeta| = 1`, the saturated point every clamp in
`defumat/xc` is about. From there the run leaves the ferromagnetic branch. At **0.9** the
branch survives, and the whole scan comes back:

| `q_3` | converged | iterations | E (Ry) | \|m\| | E - E(0) |
|---|---|---|---|---|---|
| 0 | yes | 7 | -0.9802841602 | 0.53125 | 0.0 meV |
| 1/8 | yes | 10 | -0.9849565354 | 0.40165 | -63.6 meV |
| 1/4 | **no** | 200 | -0.9913133105 | 0.20020 | **-150.1 meV** |
| 3/8 | yes | 13 | -0.9868070126 | 0.15256 | -88.7 meV |
| 1/2 | yes | 59 | -0.9846191874 | 0.04355 | -59.0 meV |

P63's own numbers were `0, -150, -59` meV at `0, 1/4, 1/2` and the re-run gives
**0, -150.1, -59.0**. They reproduce to the digit.

**And the scan now does what P63 wanted it to and recorded as impossible.** The minimum is
at `q = (0, 0, 1/4)`, which is exactly the wavevector the transverse susceptibility names
as the first instability of the ferromagnet -- two calculations sharing no machinery, one a
sum over states plus a matrix inversion on a frozen density, the other five independent
self-consistent fields. `h-fcc-spiral-scan.in`'s header used to say "it does not work"; it
does.

The same seed is the whole of `Part V.1` below, on the same cell, and the two were one
defect. Three cautions are in the input's header: the `q = 1/4` point does not converge
(1.25e-7 after 200 iterations, which is what the bottom of a flat magnetic surface looks
like), `q = 1/8` and `3/8` are not commensurate with the 4x4x4 grid so their k-sampling
differs from the other three, and `|m|` falls monotonically along the scan -- this is a
spiral whose moment shrinks as it turns, not a rigid rotation.

**The three candidates the entry below proposed were all wrong**, and the reason is worth
keeping: none of them was tested before being written down. `mixing_ndim` was "the candidate
to test first" and is not the cause; the cell's own seed was never suspected because the
input had always carried it.

### The original entry

Its numbers, not its conclusion. P63 records `E(q) - E(0)` of `0, -150, -59` meV at
`q_3 = 0, 1/4, 1/2` on `h-fcc-spiral-scan.in`; the `q_3 = 1/2` point now comes out at
**-0.41 meV**. The reason is at `q = 0`: that run converges to **0.0273 mu_B** on an atom
seeded at 1.0, so it is no longer on the metastable ferromagnetic branch P63 measured (which
that phase itself found to be 58 meV *above* the nonmagnetic solution), and both ends of the
difference are now on the nonmagnetic one. The conclusion -- a spiral leaves the magnetic
branch, so its energy gain is demagnetization rather than a spin wave -- is untouched and if
anything strengthened, since `q = 0` now does it too.

**What to do:** find which change moved that metastable minimum. It is a *metastable* state,
so anything touching the seed or the mixing path can select a different one, and P77-P79
touched both -- `starting_magnetization`'s reinterpretation (P77c; a null for hydrogen, whose
valence is 1, so probably not this), the magnetic symmetry filters (P77/P78; this input is
`nosym`, so not this either), and `mixing_ndim` going from parsed-and-ignored to wired
(P78), which changes the Anderson history length and so which fixed point a marginal cell
falls into. **That last one is the candidate to test first**, by running the scan at the
`mixing_ndim` the pre-P78 code effectively used.

This is what the slow suite exists to catch and it was found by re-running one input by
hand, which is the second time that has happened (`PLAN.md` P38 found three phases' claims
drifted the one time the slow set was run end to end).

---

## 3. `h2-texture-120.in` does not converge as committed, and its header says it does **[closed 2026-09-13]**

**Taken as the entry recommended: the card is `'atomic'` at `lambda = 10` now**, with the
`STARTING_MOMENTS` scaled to the 0.26 mu_B sphere moment the cell converges to, and the
header rewritten to describe the run the file performs. That is the row `PLAN.md` P79 calls
the answer, and re-measured on the committed file it is unchanged: **38 iterations, pair
angle 121.13 degrees, 0.576 degrees per site**, where the unconstrained cell collapses to
179.5 degrees in 14.

Two test files referenced the input and both are fixed in the same pass, which is the part
that would otherwise have gone stale silently:

* `tests/regression/test_holding_a_texture.py` string-replaced *from* `'atomic texture'`
  and `lambda = 0.5`. Both literals are gone, so every rewrite in it was about to become a
  no-op -- silently, since `str.replace` does not complain about a missing pattern. It now
  rewrites *from* `'atomic'` and `lambda = 10`, and the direction-only test rewrites *to*
  the scheme it is about.
* `tests/unit/test_magnetic_fields.py` asserted `field.constraint == "atomic texture"` off
  the committed file. It builds the text itself now: the regression it guards is a crash in
  the **first potential build** and does not care how the run ends, so a scheme that cannot
  converge is fine there and is not fine in a committed input.

`PLAN.md` P79, `docs/features.tex` and `NONCOLLINEAR.md` all cite the file for the 0.55
degrees; each now names the scheme the number belongs to.

### The original entry

**Found while writing the textures notebook, 2026-09-13, and confirmed by a second run.**
The file states `constrained_magnetization = 'atomic texture'` at `lambda = 0.5`, and its
header says *"`lambda` is left at a value the SCF tolerates rather than the largest that
would hold the angle best"*. It is not. Two independent runs:

| run | budget | `conv_thr` | reached | state |
|---|---|---|---|---|
| notebook thread | 100 iterations | 1e-8 | `6.4e-4` Ry | site residuals 49.0 and 51.7 degrees, final angle 102.7 |
| confirmation | 120 iterations | 1e-6 (the file's own) | `4.9e-2` Ry | -- |

The two accuracies are two orders apart at comparable budgets, which is itself the
diagnosis: it is **oscillating rather than converging slowly**, so a larger budget is not
the fix.

**The code already knows.** `run_scf` prints a `RuntimeWarning` on this exact scheme
saying that `'atomic texture'` constrains a direction and not a length, so its potential
carries a `1/|m|` that *grows* as a site's moment shrinks, and that **no `lambda`
converged in 400 iterations** on this very cell. That warning and this input file are
each other's contradiction and both are committed.

**Why nothing caught it.** No test runs the file as written.
`tests/regression/test_holding_a_texture.py` reads the text and **rewrites** it to
`'atomic'` at `lambda = 10` -- the scheme that does converge, and the one P79's 0.55
degrees per site was measured with -- and `tests/unit/test_magnetic_fields.py` only parses
it. So the input is referenced by two test files, `PLAN.md`, `NONCOLLINEAR.md`,
`MAGNETISM-NEXT.md` and `docs/features.tex`, and run as committed by none of them.

**What to do**, and it is a choice rather than a fix:

* change the card to `'atomic'` with the `STARTING_MOMENTS` scaled to the expected moment,
  which is what the warning recommends and what the regression test already does by hand
  -- then the test stops rewriting the file and the input means what its header says; or
* keep it as the *demonstration* that `'atomic texture'` does not converge, and rewrite
  the header to say so, since a cell that fails by design is a legitimate thing to commit
  and this one is cited as evidence in three documents.

The first is better: a file whose header describes a converged run and which does not
converge is the failure mode this list exists for. Either way the header changes, and
`PLAN.md` P79's and `docs/features.tex`'s references to it should name which scheme the
0.55 degrees belongs to, because the number is `'atomic'`'s and the file is not.

**Cost.** Minutes, plus one run of `tests/regression/test_holding_a_texture.py` to check
the rewrite is still doing what it was doing.

---

# Part V -- from the 2026-09-13 memory session

## 1. Four of `test_magnons.py`'s eight tests fail, and all four are one unconverged SCF **[closed 2026-09-13 -- 8 passed; and the entry's own first lever was the right one]**

**`pytest tests/regression/test_magnons.py` is 8 passed in 247.82 s**, peak RSS 982 MB.

**The cause is the seed and nothing else.** `starting_magnetization = 1.0` on hydrogen is a
*fully polarized* atom -- one valence electron -- so the minority channel starts at exactly
zero and the first potential is built at `|zeta| = 1`. The entry guessed at this ("trips the
`pw.x` warning about values at or above 1") and then reached past it for `mixing_beta`. It
should not have: the cell has two converged solutions and the seed picks which one.

| seed | converged | iterations | E (Ry) | m | vs. nonmagnetic |
|---|---|---|---|---|---|
| 1.0 | yes | **126** | -0.9845889092 | 0.02728 | -0.13 meV |
| 0.9 | yes | **7** | -0.9802841602 | **0.53125** | **+58.44 meV** |
| 0.0 | yes | 6 | -0.9845791913 | 0.00000 | -- |

The second row is the state the input's header describes and the module is for: m = 0.53
mu_B, 58 meV above the nonmagnetic solution. The first is a different, barely-magnetic
solution -- and with almost no moment there is almost nothing for a spin wave to be a
rotation *of*, which is the 0.3958 Goldstone residual against a tolerance of 0.02. It was
not a defect in the susceptibility. (It also converges at 126 iterations against the
driver's default 100, so "not converged" was itself only true at the budget used.)

**Both halves of the second recommendation are done, and the library half is the one that
matters.** `SCFResult.require_converged(quantity)` is now the single implementation of that
refusal and `Calculator._ground_state` is a wrapper around it, so anything holding a result
can make the same check in one line -- which is what a caller that writes `scf.density`
needs, the array carrying no flag. `test_magnons.py`'s own `_converged` helper calls it, so
the three tests would now fail *by name* rather than as a physics identity.

**The same seed is `Part IV.2` above**, on the same cell, and the two were one defect. Fixing
it also turned P63's spiral scan from a recorded negative result into a confirmation of the
magnon prediction.

### The original entry

**Found while checking that MEMORY-AUDIT A8 had not moved a number, and it had not:
these predate the session entirely.** `tests/regression/test_magnons.py` is `slow`, so
the gate never runs it.

```
4 failed, 4 passed in 2m20s
test_the_goldstone_identity_converges_in_the_band_count       0.3958 < 0.02
test_the_goldstone_identity_converges_in_the_response_sphere  0.3958 < 0.02
test_the_leading_eigenvalue_at_zero_wavevector_is_one         0.6152 == 1.0 +- 0.05
test_without_the_kernel_there_is_no_collective_mode           ValueError
```

**One cause, and the fourth failure is the one that names it.** `h-fcc-magnon.in`'s
ground state **does not converge**: 100 iterations, `accuracy = 9.120108e-06 Ry` against
its own `conv_thr = 1e-10`, `E = -0.984496631887 Ry`, `m = 0.143057857`. Three of the four
tests consume that state and get a Goldstone residual of 0.40 where they ask for 0.02;
the fourth goes through `Calculator.get_spin_susceptibility`, whose refusal fires
correctly and raises instead. **So three of these four are the refusal *not* firing on a
path that reaches the same state by another route** -- `_states` calls
`fixed_density_states` on `scf.density` directly, and nothing between there and the
assertion asks whether that density is converged.

**Not this session's work, checked rather than assumed.** The same input at `47105e8`,
the commit before this session, gives `-0.984496631887 Ry` at `9.120108e-06` after 100
iterations -- **identical to every digit**, energy, accuracy, moment and iteration count.
The A8 change is value-neutral on top of that: `X_0` is bit-identical across it
(`x[0,0,0] = -0.05636548374877085+6.770340835610553e-21j`), and the failing eigenvalue is
`0.6151549544839339` on both sides.

**What to do**, and the order matters because the second is worth more than the first:

1. **Make the ground state converge, or say why it cannot.** `starting_magnetization = 1.0`
   on the only species trips the `pw.x` warning about values at or above 1; a hydrogen fcc
   cell at one Bohr magneton per atom is a strongly magnetic guess, and the run stalls four
   orders short rather than diverging, which is a mixing problem and not a broken
   functional. `mixing_beta`, `electron_maxstep`, or a `starting_from` seed are the three
   levers; the file is the one every other test in the module also uses, so a change to it
   is a change to all eight.
2. **Close the hole the fourth failure exposes.** `require_a_converged_ground_state` guards
   the `Calculator` route and nothing guards `fixed_density_states`, so the identical
   physics reaches an assertion through one door with a refusal and through the other
   without. That is `CLAUDE.md`'s "a check whose null result cannot be told from a pass"
   in its other form: **a refusal that one caller has and its sibling does not**. The three
   failing tests would then fail *by name* -- "the ground state is not converged" -- rather
   than as a Goldstone identity that looks like a physics defect.

**Cost.** The second is minutes. The first is a convergence study on one small cell, and
until it is done the module's own claims -- `PERFORMANCE.md`'s 115.04 meV against Elk's
115.93, and the 1.99 per cent Goldstone residual -- are measured on **fcc nickel**, a
different input, which is why they are not in question here.

**How to know it worked.** `pytest tests/regression/test_magnons.py` at 8 passed, and the
`h-fcc-magnon.in` run reporting `converged=True`.

# Part VI -- from the 2026-09-13 ultracell session (P88)

## 1. `nvecx = david * nbnd` is not capped against the size of the space, and QE stops where this does not **[closed 2026-09-15 -- the cap closes `nbnd = 48` outright and is marginal at 80, and this entry's own acceptance sentence contradicted its own fix]**

**Found while converging the ultracell in `nbnd`**, which is the one place a small cell
is asked for eighty bands: silicon at `ecutwfc = 12` has `npw` between 169 and 190 at the
folded k-points, so `nbnd = 80` at the default `diago_david_ndim = 4` asks for a Davidson
subspace of **320 vectors in a 169-dimensional space**. Nothing refuses it.

**The symptom is not an error, which is why this is here.** The run continues and the
overlap of a subspace that cannot be independent goes singular:

```
UserWarning: 1 of 8 k-points came back non-finite from the Cholesky route ([0])
and are being re-solved with canonical orthogonalisation
UserWarning: the fixed-density solve did not converge at 5 of 8 k-points: up to
66 of 80 bands are unsettled and the worst k-point took 100 Davidson steps
```

The fallback catches it, the answer that comes back is usable, and the cost is the whole
iteration budget at most k-points -- `nbnd = 80` took **48.1 s** against `nbnd = 64`'s
16.3 s for the same call, which reads as a scaling curve and is a solver falling over.

**QE refuses this and names it.** `PW/src/c_bands.f90:286-287` is

```fortran
IF ( nbndx > ipw ) &
   CALL errore ( 'diag_bands', 'too many bands, or too few plane waves',1)
```

with `ipw = npwx` summed over the band group, and `PW/src/memory_report.f90:484` says the
same thing a second time before any work is done. QE's own `nbndx = david * nbnd`
(`setup.f90:468`) is **identical** to this code's `nvecx = david * nbnd`
(`solvers/davidson.py:360`), so the arithmetic is not the difference: the guard is.

**What to do.** Two lines, and the second is the one that matters.

1. **Cap `nvecx` at the smallest `npw` on the k-set -- not at `ndim`.** This is the part
   that is easy to get wrong and was, in the first draft of this entry.
   `hamiltonian.ndim` is **`npwx`**, the padded maximum over k (`hamiltonian/operator.py:110`,
   and `npol * npwx` for a spinor), and the k-points that go singular are precisely the ones
   *below* it: on the cell above `npwx = 190` while the seven k-points that returned `nan`
   have `npw = 169`, so `min(4 * 48, 190)` is still oversubscribed at every one of them. The
   real bound is `min_k npw`, the smallest row-sum of `state_mask`, and it has to be computed
   **outside** the `vmap` because `ik` is traced inside it. `cegterg`'s only other constraint
   is `nvec > nvecx/2` (`cegterg.f90:125`), so a cap has to leave room for one expansion;
   where `2 nbnd` does not fit in the space there is nothing to iterate in at all and the
   honest answer is a direct diagonalisation of it.
2. **Refuse `nbnd > npw` by name at the door**, as QE does, rather than letting it become
   a Cholesky that returns `nan`. The number is known before any solve: it is the smallest
   `npw` over the k-set.

**Where it is worked around meanwhile.** `fixed_density_states` forwards `david`
(`workflows/nscf.py`), so the caller can drop the subspace to 2 and stay inside the
space -- which is what P88's `nbnd = 48, 64, 80` rows were measured at, and why
`PLAN.md` P88 says the seconds column of that table mixes two solver settings.

**How to know it worked.** `nbnd = 80` on `tests/data/qe/si-ultracell.in`'s cell at the
default `david` raising by name instead of warning, and the same run at a capped `nvecx`
coming back with no Cholesky fallback and a time on the 16.3 s trend rather than at 48.1.

### What was done, and where the entry was wrong about itself

**Both halves are in, as the entry asked.** `Hamiltonian` and `SpinorHamiltonian` carry the
per-k sphere counts as a **static** field, `npw`, filled from `basis.planewaves.npw` at the
two places a Hamiltonian is built, and a `space` property reads `npol * min_k npw` off it.
It has to be static because `nvecx` is an array *shape*: a value read off `state_mask`
inside the solve is a tracer, and one read on the host would make a traced caller and an
untraced one disagree about the subspace, which is the one thing a dial here may never do.
`davidson_eigensolver` then takes `nvecx = min(david * nbnd, space)`, and
`davidson_eigensolver_all` -- QE's `diag_bands`, which is the door -- refuses `nbnd > space`
and `2 nbnd > space` by name. Where `space` is unset the bound falls back to `ndim`, which
is QE's own `ipw`; only hand-built test fixtures reach that.

**The measurement, on `si-ultracell.in`'s own 32 k-point set** (spheres from 169 to 192,
`npwx = 192`), `ethr = 1e-8`, `david = 4`, the same Hamiltonian solved both ways:

| `nbnd` | | non-finite k-points | steps, max / mean | unsettled |
|---|---|---|---|---|
| 48 | uncapped | **25 of 32** | 4 / 4.0 | 0 |
| 48 | capped | **none** | 18 / 14.4 | 0 |
| 80 | uncapped | 1 of 32 | 33 / 15.3 | 0 |
| 80 | capped | **1 of 32** | 36 / 18.6 | 0 |

**This is not a repeat of the run at the head of this entry.** That one was 8 k-points
through `fixed_density_states` on a folded grid and reported 5 of 8 unconverged with up to
66 of 80 bands unsettled; this is `si-ultracell.in`'s own 32 k-point set solved directly at
`ethr = 1e-8`, where no row leaves a band unsettled at all. The table measures **the cap**,
by solving one Hamiltonian both ways, and the entry's original 8 k-point run was not
repeated.

The step counts on the uncapped `nbnd = 48` row are 4 because the loop **exits** as soon as
the eigenvalues stop being finite, so a short count there is the failure and not a fast
solve. The `nbnd = 48` case is closed: 192 vectors in a 169-dimensional space became 169,
and nothing goes singular.

**`nbnd = 80` is not closed and the cap cannot close it.** There `space` is 169 and
`2 nbnd` is 160, so the subspace is allowed and is capped to 169 -- which is the *entire*
space at k-point 0. A Davidson expansion that has filled the space has nothing independent
left to add, so the overlap goes singular at that one k-point whatever the cap says, the
canonical-orthogonalisation fallback catches it and the solve converges with no unsettled
band. The distinction to carry forward is that **a cap removes the case where the subspace
is larger than the space, not the case where it equals it**, and the second is a cell asking
for more bands than a `12 Ry` cutoff has room to iterate in.

**This entry's "how to know it worked" contradicted its own "what to do".** It asked for
`nbnd = 80` at the default `david` to *raise by name*, which is QE's behaviour --
`nbndx = 320 > ipw = 192` errors outright -- while its own point 1 asked for the subspace to
be **capped**, under which the same run is allowed. The cap is the better of the two here
and that is a deliberate departure: `david` is a tuning knob and `nbnd` is a physical
request, so an oversized *subspace* is this code's business to bound while an oversized
*band count* is the caller's to fix. Refusing is kept for the two cases no cap can rescue.

**It has already cost a test, which is why it is not merely latent.**
`test_the_ultracell_converges_to_the_supercell`'s `nbnd = 48` rung at the default
`david = 4` asks for 192 vectors in a 169-dimensional space: 7 of 8 k-points came back
non-finite, the frozen states they produced are not eigenstates of anything, and the
ultracell loop above them then ran 200 iterations to `dr2 = 1.15e-1`. **Nothing in that
chain reports the cause** -- the failure surfaces as "the ultracell did not converge",
three layers from the subspace that was too large. The rung passed `david = 2` meanwhile,
which is what the `PLAN.md` P88 measurement was taken at; **that workaround is removed and
the rung now runs at the default**, which makes it the regression test for the cap. The
energy ladder in `test_the_total_energy_bounds_the_supercells_from_above` carried the same
`(48, 2)` rung and it is removed with it.

**Five other `david = 2` call sites in that file said nothing about why they were there,
and all five turn out not to need it.** They were on the hydrogen ultracells rather than on
silicon, and a bare keyword is not evidence of its own reason, so each was run alone at the
default:

| test | | peak |
|---|---|---|
| `a_uniform_field_is_the_unit_cell_under_the_same_field` | passed, 76.1 s | 2712 MB |
| `the_magnetic_ultracell_converges_to_the_supercell` | passed, 75.9 s | 2785 MB |
| `a_uniform_vector_field_is_the_unit_cell_under_the_same_field` | passed, 67.7 s | 1315 MB |
| `the_noncollinear_ultracell_converges_to_the_supercell` | passed, 83.2 s | 1658 MB |
| `fixed_occupations_fill_spinor_bands_one_electron_at_a_time` | passed, 52.1 s | 1723 MB |

The magnetic one was measured both ways, since it was the one that looked like it might be
a *memory* choice rather than the subspace workaround: 2785 MB at the default against
**2866 MB** at `david = 2`, one sample each, which is the same number to three per cent and
does not support a direction. So the keyword is removed at all five.

**Two of those runs were killed before that was measured, and neither kill was the test.**
Five in one process died the way `CLAUDE.md` says a multi-cell file does -- XLA holds every
executable it builds -- and a second attempt died at the end of a sequential loop with 21 GB
free. Each test alone peaks under 3 GB. The rule that worked is the one already written
down: one process per test, and read the peak off the watchdog rather than off the run that
contained it.

## 2. An ultracell's `dr2` is even more charge-dominated than a unit cell's, by `N^2` **[measured 2026-09-14: harmless on a *driven* wave at 1.7e-5, and the argument says where it would not be]**

**Opened and measured 2026-09-14, while writing P88 stage 3a.**

`OPEN.md` Y1 is the unit-cell version of this and it has a number: at `dr2 = 9e-11` on a
magnetic cell the total energy was still 1.15e-8 Ry out, because `rho_ddot` weights the
charge residual by `1/|G|^2` and the magnetization residual by a constant
(`e2 4 pi/(2 pi)^2`). An ultracell makes the gap **wider rather than the same**, and the
factor is exactly the one the method exists to introduce.

**The arithmetic.** At the envelope's own wavevector the charge half of `dr2` carries
`e2 4 pi/|G+Q|^2` and the magnetic half carries `e2 4 pi/(2 pi)^2`, so their ratio is
`(2 pi)^2/|G+Q|^2`. On the notebook's eight-cell silicon cell `|Q| = 0.1334` 1/bohr, so
that ratio is **2.2e3**: a magnetization residual and a charge residual of the same size
contribute to `dr2` in the ratio 1 to 2200, and a run that stops at `dr2 = 1e-9` can be
carrying a moment residual ~47 times larger than the charge residual it is bounding. The
`N^2` is explicit: `|Q|_min` is `N` times smaller in an ultracell than in the unit cell,
so this ratio grows as `N^2` while nothing about the convergence test changes.

**Why this is an entry and not a fix.** The convergence test here is the **sum**, which is
QE's `rho_ddot` and is what makes `conv_thr` mean the same thing in an ultracell run as in
every other run in this package. Changing it -- a separate threshold on the magnetic half,
or an `ethr` schedule driven by that half -- is a deliberate departure from `pw.x`, and
`CLAUDE.md`'s rule is that a departure needs a number rather than an argument. Y1 says the
same thing about the unit-cell case and has been open since 2026-09-12 for the same reason.

**What exists meanwhile.** `UltracellResult.charge_accuracy` and `.magnetic_accuracy`, the
pair per iteration in `.history`, and the non-convergence warning naming both -- so a run
that stopped with its moment still moving can be *told apart* from one that stopped with
both converged, which is the half of the problem that costs nothing. On the silicon
field run of P88 stage 3a the two end at 1.96e-11 and 4.9e-14, three orders apart, which
is the shape the arithmetic above predicts.

**The measurement was done, and on this cell the weighting is harmless.** The same
eight-cell field run stopped at `conv_thr = 1e-8` and at `1e-11`:

| `conv_thr` | iterations | `dr2` | charge | magnetic |
|---|---|---|---|---|
| 1e-8 | 7 | 7.33e-10 | 7.32e-10 | 1.40e-12 |
| 1e-11 | 9 | 1.04e-12 | 1.04e-12 | 2.85e-16 |

and the per-cell moments differ by **2.0e-6 absolute, 1.7e-5 relative** across a threshold
700 times looser. So the entry downgrades to a note -- **with the reason, which is what
says where it would not downgrade.** The two residuals are not equipartitioned: the
magnetic half is already at 1.4e-12 when the loop stops on a charge half of 7.3e-10,
three orders below it, because the magnetization here is a **driven linear response** to a
field that does not move, while the charge is the soft direction that sloshes. The
weighting is charge-dominated and so is the residual, so the test bounds the thing that is
actually still moving.

**Where that argument was expected to fail was the case this code could not yet run.** A
*spontaneous* wave has no field holding its magnetization, so the moment would be the soft
direction and the one that sloshes, bounded by the half of `dr2` that carries no
`1/|G+Q|^2` -- with 2.2e3 in the weights at this cell's `|Q|`.

**Measured, 2026-09-17, and for a collinear wave the prediction does not hold**
(`PLAN.md` P88 stage 8, which put the seed into the *density* rather than into a faded
field). The seeded staggered wave on two cells of hydrogen, at two thresholds:

| `conv_thr` | iterations | `dr2` | charge | magnetic |
|---|---|---|---|---|
| 1e-8 | 6 | 1.494e-10 | 1.312e-10 | 1.819e-11 |
| 1e-11 | 8 | 1.126e-12 | 1.098e-12 | 2.798e-14 |

so the magnetic half is **7 times below** the charge one at the loose threshold and 39
times below it at the tight one, and across that thousandfold change the wave's own Fourier
amplitude moves 2.17e-5 relative and the energy 2.7e-11 Ry. **The reason is that a
collinear wave has no soft direction at all**: with `nspin = 2` the moment's *direction* is
not a degree of freedom and only its magnitude is, which is stiff. So the paragraph above
identified the right mechanism and the wrong regime -- the soft direction is the
**noncollinear** one, a rigid rotation of the whole texture, and there it shows as
iterations rather than as a residual: the seeded helix converges in 14 iterations when the
seed is written about the reference's own axis (where the truncated basis closes the sector)
and 290 when it is not, with residual halves 5.44e-11/1.54e-11 and 4.14e-11/3.33e-11. A
residual that does not separate those two is the thing still worth a note, and it is the
iteration count rather than the threshold that tells them apart.

**The case arrived from somewhere else, and it is the ordinary SCF rather than an ultracell**
(reported 2026-09-15 by the NiBr2 session on Triton, job 20260032, a 45-atom slab under a
15-site `LOCAL_MAGNETIC_FIELDS` helix). Two measurements, and together they say the weighting
bites here in the way the paragraph above predicts. **Where the magnetization moves is the
grid scale**: of the power of the magnetization difference between consecutive iterations, 93
to 96 per cent lies above `|G| = 0.8` 1/bohr and 0.2 to 0.3 per cent below 0.4, which is
exactly the band the charge half's `1/|G|^2` suppresses and the magnetic half cannot. **And
what it is doing is a slow coherent drift no single iteration shows**: `m_z` went 0.0000,
0.0034, 0.0197, 0.0489, 0.1153, 0.1420 over iterations 4, 40, 80, 120, 160, 186, with the
absolute magnetization going 30.8 to 36.5 mu_B, while the per-iteration angle change had a
mean an order of magnitude *below* its spread across sites. So four consecutive iterations
read as grid-scale noise and a hundred of them are the whole story, which is the shape of
thing a residual reads as converged.

**The cause there is the constraint and not the mixer**, which is worth recording because it
is the same lesson as `CLAUDE.md`'s `i_cons = 2` trap one step out: the card carries only `x`
and `y` components, so **out-of-plane canting costs the constraint nothing** and the field is
stiff only in the plane it has components in. A field pins the directions it has a component
along and leaves every other one free; a hard constraint on the moment direction is the thing
that would not. The in-plane helix held its 24 degree step to better than a tenth of a degree
throughout.

**And in the plane the field is stiff rather than rigid, which is the same statement about a
weak constraint from the other end.** The card is an exact ladder, `-90 - 24i` degrees with
`|B| = 7.31e-03` Ry on every Ni, and the converged moments sit off their own field directions
by up to **2.43 degrees**, rms 1.35, thirty to fifty times the 0.05 degree per-iteration
noise, and **in a pattern** -- largest at sites 3, 10 and 12 -- rather than at random. So what
a field buys is the compromise between itself and the exchange, not the angles that were
asked for, and a run that needs the asked-for angles needs a hard constraint on the direction
instead. Whether that pattern is anisotropy bunching the helix or the Br sublattice is not
established.

## 3. A noncollinear ultracell's rigid spin rotation has no restoring force, and a small `mixing_beta` cannot cross it **[opened 2026-09-14, P88 stage 3b]**

**The mechanism, and it is physics rather than a bug.** Without spin-orbit coupling or
magnetic anisotropy, turning *every* moment in the cell together by the same angle costs
exactly zero energy -- it is a Goldstone mode of the broken spin-rotation symmetry. In an
ultracell that mode is precisely the `Q = 0` transverse component of the magnetization, and
the self-consistency residual has **no component along it at all**: the fixed point is a
whole two-parameter family rather than a point. An Anderson mixer extrapolating along a
flat direction is then unbounded.

**Measured**, on a four-cell hydrogen ultracell (simple cubic, `a = 5.5` bohr, one electron
per cell, moment 0.62 mu_B/2) under a field that turns by 90 degrees per cell, `nbnd = 16`,
every run given a 300-iteration budget:

> **Two things this table does not say, and a later session needed both** (2026-09-15). The
> **0.62 is a k-grid**: that is the unit cell's moment on the `(4, 2, 2)` grid the stage 3a and
> 3b tests fold to, with no field, and the same cell is **1.0000** on `(4, 1, 1)`, 0.7901 on
> `(4, 4, 4)` and 0.8020 on `(6, 6, 6)`. This table does not state which grid it ran on, so the
> number in its own header does not identify its own cell. And it does not state the **field's
> functional form**, only that it turns 90 degrees per cell, which is not enough to repeat it:
> taking that description literally, as `0.01 (cos 2 pi x/4, sin 2 pi x/4, 0)` Ry at
> `(4, 1, 1)` with `kerker = False`, `anderson` at `beta = 0.7` converges in **83** iterations
> where this table says 48. **A field and a k-grid are inputs and belong beside the amplitude.**
> **`nbnd = 16` cuts a degenerate multiplet here and it does not matter, which is worth one
> line so the next session does not spend an afternoon on it.** The gap is 5.0e-9 Ry at
> `(4, 2, 2)` and 3.4e-9 at `(4, 1, 1)`, under `DEGENERATE_CUT`, and the warning correctly
> stays silent: the gate is `nbnd <= 2 occupied` and this one-electron spinor cell has
> `occupied = 1`, so the cut lies fifteen bands above anything that holds weight, which is the
> case `driver.py:625-635` measured as harmless. What `nbnd` does move is ordinary accuracy --
> `anderson` at 0.7 gives 0.3912 in 79 iterations at 16 and 0.3674 in 57 at 32, six per cent --
> so 16 is not converged for quoting a moment, and every moment in this table is quoted from it.
>
> **Redone at `nbnd = 32` the flat manifold is not where this table put it.** At `(4, 1, 1)` there is **one** state and no mixer effect: both mixers and `beta`
> across a factor of seven give `|m|` 0.4305 to 0.4318 and -42.5 degrees per cell, every run
> converged. At `(4, 2, 2)` there are **three**, all converged three orders below `conv_thr`,
> and the two clusters that converged at `nbnd = 16` are the same two, so the structure is the
> cell rather than the basis (only one `anderson` run converged there, so 16 shows two of the
> three and not all three) --
> `anderson` at 0.1 and 0.7 agree on one to eight digits of `E_band`, `anderson` at **0.3**
> finds a second, and `adaptive` finds a third, the one that follows the field's +90 degree
> ladder. So
> the step length selects a solution **inside** one mixer, and `PERFORMANCE.md`'s control
> across three values of `anderson`'s `beta` does not separate mixer from step length after
> all. The full table is there.

| field | `mixing_beta` | iterations | converged | final `dr2` |
|---|---|---|---|---|
| 0.01 Ry | 0.7 | 48 | yes | 2.7e-10 |
| 0.01 Ry | 0.5 | **36** | yes | 4.6e-10 |
| 0.01 Ry | 0.3 | 42 | yes | 4.9e-10 |
| 0.01 Ry | 0.2 | 115 | yes | 5.6e-10 |
| 0.002 Ry | 0.7 | **263** | yes | 1.5e-10 |
| 0.002 Ry | 0.3 | 300 | **no** | 1.7e-3 |

**The mixer is not unstable, it is stuck**, and the `dr2` traces are what say so: at
`beta = 0.3` under the weak field the residual sits between 2e-5 and 9e-5 for all three
hundred iterations and never leaves, while `beta = 0.7` covers the same range, spikes to
1.1e-2 -- passing through per-cell moments of 2.2 mu_B/2 on an atom that holds one
electron, a state that cannot exist -- and then drops to 3.4e-8. The flat manifold has to
be **traversed**, and `mixing_beta` is how fast.

**So lowering `mixing_beta` is the wrong reflex here and three earlier versions of this
entry gave it as the advice.** They are kept because each failed differently: the first
compared two runs at different fields *and* different budgets; the second called a
40-iteration non-convergence a divergence; the third made "lower beta" standing advice in
the driver's warning, the user guide and `PERFORMANCE.md`. The mechanism was right
throughout, which is exactly why each wrong version was plausible.

The collinear branch of the same cell has no such direction -- flipping a moment costs
energy, and rotating one is not expressible -- so this is new at `nspin = 4` and not a
rediscovery of ordinary charge sloshing.

**What exists meanwhile.** `run_ultracell`'s non-convergence warning names the mechanism
when `nspin_mag = 4`, rather than only naming the knob: a user who is told "lower
`mixing_beta`" cannot check that advice, where one who is told "the rigid rotation has no
restoring force" can. `box_kerker` already screens only the charge component and applies a
plain `beta` to the three magnetic ones, which is `approx_screening`'s own rule and is
correct as far as it goes -- Kerker damps *long wavelengths*, and this mode is at `Q = 0`
where Kerker's factor is already zero for the charge. **The magnetization is deliberately
not Kerker-screened** (stage 3a's reason: it would damp the direction a magnetic run has to
move in), so nothing currently touches it.

**The fix that is not taken, and why.** Project the rigid rotation out of the magnetic
residual before mixing -- subtract from `m_{Q=0}` its component perpendicular to the mean
moment, which is the generator of the rotation. That is a **departure from `pw.x`**, which
mixes every component with one `beta` and has no such projection, and `CLAUDE.md`'s rule is
that a departure needs a number rather than an argument. The number it needs is iterations
to convergence with and without the projection, on at least two cells, one of which has
spin-orbit coupling -- where the mode is *gapped* and the projection would be actively
wrong, so the projection has to be switched off by `lspinorb` and that gate has to be shown
to fire. Until then the answer is a **large** `mixing_beta` and a long budget -- the
opposite of the usual reflex -- and the warning says so in those words.

**How to know it worked.** The weak-field run above -- 0.002 Ry at `mixing_beta = 0.7`,
which takes **263** iterations as it stands -- converging in well under that with the
projection on, to the same converged moments to 1e-8; and the same run with `lspinorb`
giving the *same* answer with the projection on and off, which is the gate firing.

**Redo that run before using it as the target** (2026-09-15), for the two reasons above: at
`nbnd = 16` its basis is arbitrary, and neither its k-grid nor its field's functional form is
written down here, so the 263 cannot be reproduced from this entry. **And state the number it
converges *to*, not only how many iterations it took** -- on a clean basis this cell has three
self-consistent solutions at `(4, 2, 2)`, so "converged in well under 263" is satisfied by a
run that found a different one, which is precisely the failure this entry's own mechanism
predicts and which an iteration count cannot see.

**The target pair this entry says it lacks now exists, and it is cleaner than the 263**
(2026-09-17, `PLAN.md` P88 stage 8). A *seeded* helix on four cells of the same hydrogen
lattice converges in **14** iterations when the seed turns about the reference's own
magnetization and **290** when it turns about an axis 54.7 degrees away, at
`nbnd = 16, mixing_beta = 0.3, kgrid = (1, 2, 2), conv_thr = 1e-10`, with no field at all.
**Both arms reach the same state**, which is what the 263 could never establish: after one
global rotation their cell moments agree to 1.23e-3 on moments of 0.272 and their energies
to 3.5e-9 Ry. So the 276 extra iterations are the flat manifold being traversed and nothing
else, which makes this the pair to measure a projection against -- same cell, same
threshold, same converged state, and the only difference the distance the rotation has to
travel. It also hands this entry the missing half of its own "state the number it converges
*to*": **compare two candidate solutions after fitting the one global rotation between
them**, because a component-by-component comparison on a flat manifold counts frames rather
than states.

**The measurement the ultracell could not make, it now can** (2026-09-15, `PLAN.md` P88
stage 4). This entry was written when an ultracell reported only an eigenvalue sum, so
nothing in it ranked its own solutions; it reports a **total energy** now, and the three
solutions at `(4, 2, 2)` can be put in order. That run has not been made. The one caveat
to carry into it is that stage 4's *ladder* does not apply here -- with three stationary
points nothing says the same one is found at each `nbnd`, so the energy ranks solutions at
one fixed `nbnd` and certifies nothing across several.

**What the exact calculation said, before there was any other way to ask.** A real four-atom supercell
under a `LOCAL_MAGNETIC_FIELDS` ladder of 0.01 Ry turning 90 degrees per site converges in
**26** iterations to a clean helix, 90.0 degrees from site to site and `|m| = 0.604` on every
site; seeded ferromagnetically instead it does **not** converge in 400 iterations and ends on
no texture. So the lattice has one solution under that field and it is the helix, and the
bunched ultracell states are the frozen basis or this entry's flat manifold rather than
physics. The field shapes differ, a sphere ladder against a continuous `B(r)`, so this ranks
textures on the lattice rather than the ultracell's states.

---

# Part VII -- from the NiBr2 helix on Triton, reported 2026-09-14

A peer session running a 45-atom NiBr2 slab on GPU (noncollinear + `lspinorb`, FR PAW,
`ecutwfc` 45 / `ecutrho` 240, `nbnd` 403, `npwx` 156346, 45-site
`LOCAL_MAGNETIC_FIELDS`, `nosym`/`noinv`, anderson 0.3, `diago_david_ndim = 2`,
`DEFUMAT_BAND_BATCH` 16, `k_batch` 1) reported four things. **Two were defects and are
fixed**; the two below are not this session's work because neither can be measured on
this machine.

**What was fixed, for the record.** The finiteness guard in
`davidson_eigensolver_all` reduced over the whole returned k-set, ran every iteration
rather than only on a retry, and killed two production SCFs -- 21.40 GiB asked of an
H100 against an 11.27 GiB wavefunction store, `RESOURCE_EXHAUSTED` at iteration 13 in a
pool with 25.7 GB free and a 14.9 GB largest hole. It now reduces inside the per-k
solve. And the checkpoint refusal on `magnetic_field` was on both sides of the boundary
at once: a converged run holding a plain applied field could not be written at all,
while a fixed-spin-moment run wrote one every cadence with its *driven* field silently
dropped -- plus `field_scale` was saved and then reset to 1.0 on resume, so a `reducebf`
run came back at full field.

**One half of the report did not reproduce, and it is the useful kind of wrong.** The
claim was that mid-SCF checkpointing is dead for any run carrying a field. Measured on
one hydrogen atom with a `LOCAL_MAGNETIC_FIELDS` card, both plain and with
`reducebf = 0.5`: the checkpoints were written every cadence, all along. The refusal
bit the *final* `save_state` of a converged result, which is where a driver script
calls it, and the mid-SCF path was passing for the opposite reason -- it carried no
field for the refusal to see. Reading a refusal's source is not the same as watching it
fire, which is `CLAUDE.md`'s own rule about guards, applied to a guard's absence.

## 1. `sizing.py` reads 60 per cent low on a 45-atom spinor PAW slab **[REOPENED 2026-09-14 -- the closure below was withdrawn by the A/B it predicted, and the ~28 GB is still unexplained]**

> **Read this before the entry.** Everything from "Closed by subtraction" to the end of the
> caveat is **retracted**, and the commit that carried it (`23fc3b0`, *"The size report was
> not 60 per cent low: it was missing the guard"*) is wrong in its title. The relaunch is a
> clean A/B of the guard fix on the same architecture -- job **20252129** (A100 gpu41,
> `ad89fd9`, `wfc_store device -> device` printed) against **20244588** pre-fix -- and the
> per-iteration peaks are **byte-identical**: 79.14 / 79.43 GB at iterations 1 and 2 on both,
> 79.43 and 79.46 at 3 and 4 post-fix. **Removing an allocation claimed to be 82 per cent of
> a 28 GB gap moved the peak by 0.03 GB.** So the 21.40 GiB was served out of space the
> solver had already freed, and `resident + temp + guard = 72.16` against a measured 72.30
> was three plausible numbers summing to the right answer with **no evidence they were ever
> simultaneously live**.
>
> **The failure mode is the one this file keeps recording, one level up.** The caveat below
> names the hole exactly -- *"the overlap cannot be proved from a log, so 4.78 GiB is a
> residual and not a measured solver excess"* -- and the closure was then written as if a
> later sweep had filled it. It had not: the sweep measured the **temp**, which was the one
> term that could be checked, and **agreement in a total was read as confirmation of its
> parts.** A sum of three terms has one equation and three unknowns.
>
> **What survives, and it is not nothing.** The eigensolver temp at this cell is **23.18
> GiB** at `david 2 / band_batch 16`, and it is now measured **twice independently**:
> `tools/gpu/davidson_memory.py` compiled it (job 20252132), and the failing run asked the
> allocator for exactly **24,889,513,216 B** from `jit__every_k`, which is the same number.
> That is what makes `sizing.py`'s 18.55 **20 per cent low at this corner**, recorded in that
> module's docstring. `david 3` costs **+6.12 GiB** over `david 2` rather than a doubling
> (item 2 below). The arena read of 20244646 -- **57.24 GB in use** at the instant of the
> request, from the occupancy bar -- stands, and is worth noting as **the one figure sent
> here that used no model at all, and the one that survived.** And `0e85a14` remains right on
> its own terms: the whole-set reduction was wrong on a dense mesh. It is simply **not a
> memory fix on this cell**.
>
> **Where the allocation moved, which is the more useful finding.** Post-fix the guard is
> computed inside `_every_k`, so forcing it to the host is what makes that executable run,
> and what the allocator is asked for is the eigensolver's **own** temp buffer --
> `davidson.py:799`, still `failed = ~np.asarray(per_k)`. The guard did not stop costing an
> allocation; it stopped costing a **separate** one. On a cell whose problem is
> **fragmentation** rather than total bytes, that is a smaller win than it read as.
>
> **The next instrument is not arithmetic.** Job 20252135 runs the same two iterations under
> `DEFUMAT_STAGE_PEAKS=1`, bracketing every stage with `peak_bytes_in_use`: whatever holds 79
> GB has to appear between one of those pairs.
>
> **Two things about that instrument, both from 2026-09-14 and both in `MEMORY-AUDIT.md`
> A17.** `DEFUMAT_STAGE_PEAKS` is in **no committed file** in this repository, so every
> bracketed number in the record rests on a patch that lives only on Triton and that nobody
> here can read. And the brackets were suspected of depressing the peak they measure, which
> would have made every one of those numbers a lower bound; an unbracketed run reproduced the
> bracketed one byte for byte, so they do not, and the suspicion is closed. The last two sizings of this cell were both
> arithmetic, both wrong, and in **opposite directions**.



`run_scf.py --size-only` at a `1 6 1` mesh reported floor 42.60 GiB, eigensolver XLA temp
buffer 18.55 GiB, **peak 46.12 GiB = 49.5 GB**. The measured working peak was **79.4 GB**,
with the guard above asking 21.4 GB more on top of that once `ethr` tightened. Anyone
reading the report picks an 80 GB card for a cell that needs 141 GB.

**Closed by subtraction, from that job's own log.** The run prints its `defumat size`
block at its head, so the estimate and the peak are in one file, and the peer session did
the arithmetic:

| | GiB | GB |
|---|---|---|
| measured peak (H100 gpu45) | 72.30 | 77.63 |
| what the model predicted | 46.12 | 49.52 |
| gap | 26.18 | 28.11 |
| **the guard, one allocation** | **21.40** | **22.98** |
| residual | 4.78 | 5.13 |

The guard is **82 per cent of the gap**, and what is left is **6.6 per cent of the peak** --
inside the eigensolver fit's own error bar. So the line the report was missing was the
guard itself, and `0e85a14` deleted the thing it would have described. There is no missing
30 GB term; the entry's own estimate of one was wrong, and it was wrong because it read a
gap as a *model* error when most of it was a single allocation the model was never asked
to cover.

**An independent check that does not use the model at all**, from the occupancy bar: 31 per
cent free on an 82.95 GB pool is 57.24 GB in use at the instant of the request, **16 per
cent** above the predicted 49.52 GB. Not 60.

**One caveat on the 4.78 GiB, and it is the peer's own.** The model's peak is
`resident + max(the Davidson lines, the XLA temp)`, and the guard clearly allocated while
the solver's working set was still resident -- 27.58 + 21.40 does not reach 72.30. The
overlap cannot be proved from a log, so 4.78 GiB is a **residual** and not a measured
solver excess.

**What is still worth running**, and it is now a confirmation rather than a diagnosis:
`tools/gpu/davidson_memory.py` at this cell's shapes over `david` in {2, 3} and
`band_batch` in {16, 32, 64}, `k_batch` 1, against a real working peak from the same job,
at `5be8b15` or later. The prediction to falsify is 54.8-58.5 GB on the `1 6 1` mesh, which
if it holds puts the 141 GB card requirement back in question.

**The transferable finding is about the two backends, not about the model.** Normalised
per wavefunction element -- which survives the change of expression between the two jobs,
where a ratio to `psi` does not:

| job | commit | form | nk | asked | B/element |
|---|---|---|---|---|---|
| 20212071 | `f922702` | whole-set scalar, `davidson.py:742` | 24 | 26.00 GiB | 9.23 |
| 20244646 | `73ccb71` | per-k, `davidson.py:759` | 6 | 21.40 GiB | **30.39** |
| (this CPU) | `73ccb71` | the same per-k expression | 6 | 0.70 GiB | **1.00** |

1.00 B/element is the bool array and nothing else, which is what full fusion looks like.
The H100 materialised **30 times** that for the same expression, and 30.4 bytes against a
complex128 input reads as nothing fusing at all. Two things follow. The per-k form was the
**more expensive lowering per element** -- 3.3x the whole-set scalar -- which is worth
knowing because it is the form that shipped and ran for weeks. And a CPU
`memory_analysis()` is a lower bound on a GPU allocation by a factor that can be 30, not a
few per cent: `PERFORMANCE.md` says the *form* transfers and the coefficient does not, and
this is how far "does not" can go.

**A caveat that belongs beside the report and is now in the module docstring.** A short
calibration run measures the regime the calculation *leaves*. On this cell iterations 1-12
held a flat 77.63 GB at 21 s each; at iteration 10 `ethr` reached 2.30e-6 and the Davidson
average went from **2.0 inner steps to 73.5**, and the iteration cost from 21 s to 390 s.
Two iterations, and even twelve, said nothing about the arena three iterations later.

## 2. `diago_david_ndim = 2` may degrade at the minimum subspace once `ethr` tightens

Observed on the same run: 70+ Davidson inner steps per k-point at `ethr` 2.3e-6, against
2.0 at the loose starting threshold. `ndim = 2` is what a memory-constrained cell is forced
into, so if the restart logic is what degrades there the fix would pay twice -- fewer steps
*and* less allocate-and-free churn, which is what exhausted the arena in item 1.

**The "does not fit the card" half of this is struck, 2026-09-14.** `ndim = 4` does double
the subspace arrays at this mesh -- that arithmetic was never wrong -- but the *consequence*
drawn from it was, and for the same structural reason the guard above turned out to be
invisible: **the arrays it doubles live inside a buffer that is mostly not them.** Measured
(job **20252132**, A100-80GB, `ad89fd9`), `david 3` costs **+6.12 GiB** over `david 2`, not
a doubling:

| temp buffer, GiB | band_batch 16 | 32 | 64 |
|---|---|---|---|
| `david 2` | 23.18 | 23.94 | 25.25 |
| `david 3` | 29.30 | 27.74 | 29.07 |

**The table is the claim, and nothing is added to it here.** A predicted whole-run peak at
each `david` is exactly the arithmetic item 1 records being wrong twice on this cell, in
opposite directions, so it is not written down -- what is measured is that the step from
2 to 3 costs 6.12 GiB and not 23, which is the whole of what the struck sentence got wrong.
Whether the experiment fits a given card is a question for a stage-bracketed run on that
card, not for a sum here.

The cheap version remains worth having on its own: inner steps per
k-point against `ethr`, at `ndim` 2, 3 and 4, on `si16-1k-ecut30` or `si8-nc-1k` -- which
says whether the step count at `ndim = 2` climbs faster than at 3 and 4 or whether 70 steps
is simply what a tight threshold costs at any subspace size. This is related to the
`davidson-empty-ethr-defect` item (candidates 1 and 3 still open) but is not the same
claim: that one is about how many *outer* steps a loose threshold buys, this is about the
inner count at a tight one.

**Part VI item 1** is the neighbouring entry -- `nvecx = david * nbnd` uncapped against the
size of the space -- and a session that opens `nvecx` for either reason should read both.

## 3. The checkpoint's Hubbard refusal is probably as wide as the field's was **[closed 2026-09-15 -- it was, and the inference held; the routine to look at was the one the entry named]**

`checkpoint._REFUSED` still refuses any result carrying a `hubbard_setup`, with the reason
"`ns` without it is an array of numbers about nothing". That is true of the *file* and was
also true of the field, which turned out not to be the question: a checkpoint is loaded
against a `system` the caller supplies, so what matters is whether the input rebuilds the
thing, not whether the file describes it. A Hubbard setup is the `HUBBARD` card's and
nothing in the SCF loop appears to change it -- `hubbard_terms(ns_state)` reads it, the
loop mixes `ns` and not the setup -- so the same argument that narrowed the field's refusal
narrows this one to nothing.

**It is left as an inference rather than taken**, which is the whole lesson of the entry
above it: the field's refusal was wrong in *both* directions at once and reading the source
did not show it. The mid-SCF path is the existing evidence and it points the same way --
`_InProgressState` has always carried `hubbard_setup = None`, so a DFT+U run's mid-SCF
checkpoints are written without it and resume correctly (`tests/regression/
test_noncollinear_hubbard_resume.py` is a `4 -> 4` resume that passes).

**What would close it.** Confirm nothing in the loop mutates the setup -- `ns_adj`, and
whatever adjusts a starting `ns`, are where to look -- then narrow `_refusal` and assert a
converged DFT+U result round-trips to the same `ns` and the same energy. If something
*does* mutate it, the refusal is right and should say which routine, which is more than it
says now.

### What was done, 2026-09-15

**Nothing mutates it, and the argument is structural rather than a survey.**
`Calculation.hubbard` is assigned in exactly one place, `__init__`, from `system.hubbard`
and the datasets; the only attribute ever written on a `HubbardSetup` anywhere in the
package is `constraints`, inside `build_hubbard_setup` itself before it returns; and the
loop mixes `ns` and reads the setup through `hubbard_terms`. `ns_adj` is the one that
could have bitten and does not: it is gated on `iteration == 1`, which is QE's
`IF (first .AND. starting_pot == 'atomic')`, and a resume re-enters at `resumed_at + 1`.

So `hubbard_setup` moves from `_REFUSED` to `_FROM_CALLER`, beside `system`:
`load_state` takes it off the `calculation` when one is given and leaves it `None` when
only a system is, because `build_hubbard_setup` needs the datasets and a bare `System`
carries file names. What a load without a calculation gives back is then exactly what
every mid-SCF DFT+U checkpoint has always given back.

**Measured** on the two-atom cell with a `U` of 2.0 eV on silicon's `3p` -- not physics
anybody wants, and a manifold, a projector set and an `ns` for a fraction of a
transition-metal oxide's cost: `ns` round-trips bit for bit, `hubbard_occupations` comes
back identical, and `run_scf(starting_from=loaded)` converges to the same total energy,
which at a `conv_thr` two orders tighter than the state was converged at settles a further
**1.3e-8 Ry**. `tests/unit/test_checkpoint.py` has both halves, the round trip and the
`None` a load without a calculation gives; the refusal test it replaces is gone.
`docs/features.tex` had the refusal in two places, an amber box and the response
chapter's, and both now say what is carried instead.

---

# Part VIII -- from the workstation, 2026-09-14

### 1. A spinor-to-collinear demotion lands 1.45e-7 Ry out, and the regression test fails on master **[closed 2026-09-15 -- neither of the two candidates, and the experiment the entry proposed gave a third answer]**

> **What it turned out to be.** The entry offered two readings, a demotion that drifted and
> a convergence test satisfied before the residual is, and said nothing distinguished them.
> The run it asked for distinguishes them and picks neither cleanly. Repeating the pair at
> three thresholds: the gap is 1.45e-7 Ry at `conv_thr = 1e-8`, **1.53e-7 at 1e-10**, and
> 2.3e-13 at 1e-12. It does not shrink between the first two because at both the continued
> run still exits after one iteration; at 1e-12 the first residual is itself rejected, the
> run takes three iterations, and both directions of the round trip land within 2.3e-13.
> So the state the demotion recomposes is right and `with_spin`'s axis-finding is not where
> to look. What the loose runs measure is where a single Davidson pass happens to stop, and
> on a magnetic cell that is bounded very weakly indeed: the reported `accuracy` is 4.5e-11
> where the energy error is 1.0e-7. The moments differ by 5.7e-4 mu_B at 1e-8 and 5e-6 at
> 1e-12, which is the whole gap. That is Y1 above, in its fourth place.
>
> **Fixed** by putting the iron rotation on `CONTINUATION_THR = 1e-12` while the silicon
> cases stay at 1e-8, with the three measurements in the constant's docstring so a
> recalibration of `accuracy` cannot quietly undo it. The test takes 77 s against 51. P23's
> table carried 2e-8 and 4e-8 for this pair and now carries 3e-14 and 2e-13; the drift
> between those numbers and the ones this entry reported was drift in a quantity that is not
> a property of the continuation, which answers the entry's "whether the demotion drifted"
> as well.


**Found in passing** while A/B-ing an unrelated change, so the A/B that matters was already
run: `tests/regression/test_continuation.py::test_iron_collinear_to_noncollinear_rotates_the_moment`
fails at `f5e190e` **and** at `f5e190e` with the resume-pin fix stashed, to every printed
digit, so it is pre-existing and nothing in `040cce0` reaches it.

**The number.** Line 152, the `4 -> 2` demotion:

```
Fe 2 -> 4: fresh -55.699684334 Ry in 30 iterations, continued -55.699684241 Ry in 1
Fe 4 -> 2: fresh -55.699684327 Ry in 23 iterations, continued -55.699684182 Ry in 1
```

`back.total_energy` against `converged.total_energy` is **1.45e-7 Ry** against
`SAME_SOLUTION_RY = 1e-7`, so it fails by 45 per cent of the tolerance. The **promotion**
direction is 9.3e-8 and passes, just inside the same tolerance, which is the part that says
this is not simply a loose constant: the two directions of the same round trip sit either
side of the line, and the failing one is the direction whose comment says the demotion has to
*find* the magnetization axis rather than read `m_z`.

**What is not known.** Whether the tolerance was always this tight against this pair, whether
the demotion drifted, or when either happened. A continuation that converges in **one**
iteration and stops 1.45e-7 from the fresh answer is consistent with a demotion that starts
slightly off-axis and with a convergence test satisfied before the residual is, and those are
different defects. Nothing here distinguishes them.

**Why nobody saw it, and the gate is not lying.** The file is
`pytestmark = [pytest.mark.regression, pytest.mark.slow]`, so it is in the two-hour set and
not in `tools/test-fast.sh`, which is the push gate. A red gate would have been a different
and worse finding.

**What would settle it.** Run the demotion with `conv_thr` two orders tighter and see whether
the gap closes: if it does, the one-iteration exit is the story and the tolerance is measuring the
stopping rule rather than the demotion; if it does not, the recomposed state is genuinely
off and `with_spin`'s axis-finding is where to look. That is one cheap run and it has not
been done.


### 2. Three of `test_transport.py`'s 28 fail on master, two by a factor and one by eight orders **[closed 2026-09-15 -- three different causes, and the entry's own ranking was right]**

> **The factor of 7.7 is a mesh mismatch, and it is structural rather than arithmetic.**
> `test_the_two_limits_are_the_stm_image_and_the_fermi_surface_on_a_real_cell` asks
> `run_momentum_transport` for `grid = (3, 3, 1)` and compares it against `run_stm`, which
> has no `grid` argument at all and integrates the k-points the SCF converged on, `4 4 1`
> in `h-sheet.in`. Both sides are Brillouin-zone integrals of the same integrand, so they
> agree mesh for mesh and not otherwise. Measured: 0.87 relative at `(3, 3, 1)`, 9.0e-5 at
> `(4, 4, 1)` re-solved, and **1.17e-13 with no grid at all**, which is the figure the
> README quotes. The entry's observation that the weight sits on half the mesh is the
> coarse grid crossing the Fermi contour somewhere else and not a defect in the limit.
> What makes this structural is the sibling one test above, which passes: it hands
> `grid = (3, 3, 1)` to **both** of its sides through a shared dict, and this test copied
> the argument into the half that accepts one.
>
> **Two things came out of fixing it.** The docstring promised the `bare` column against
> `fermi_surface_weights` and the body never asserted it, the import sitting unused at the
> top of the function; the assertion is now there and holds **per k-point** at 2e-16 at
> both `eta = 0.02` and `0.05`. Writing it from the docstring would have been wrong twice:
> there is no `1/eta`, and the degeneracy must be **1** rather than the function's default
> of 2, because the k-weights already carry `degspin` and sum to 2 on this cell. That is
> P51's `for_spin` trap, and this is the only column here with an independent route to
> catch it with.
>
> **The two near misses are not tolerance slop, they are two different unbounded
> quantities.** For the three spin regimes, the control settles it: `nspin = 2` at zero
> moment is the same solve twice and agrees at **1.7e-15**, while the spinor run is an
> independent SCF and lands 1.7e-5 away -- and the same regime run again at `nbnd = 16`
> rather than 8 lands **7.5e-5** away, four times further than the spinor does. So the
> residual is what two independent SCF runs of this cell differ by, not anything the spinor
> regime brings. It is not the empty bands either: `diago_full_acc = .true.` leaves the
> ratio unchanged to every digit, which was the first hypothesis and is refuted. For the
> substrate across the moment, the residual is the converged moment's own tilt out of the
> plane, and **that tilt is not bounded by `conv_thr`** -- without spin-orbit coupling the
> direction costs no energy, so `m_z/|m|` reads 7.215e-6 at `conv_thr = 1e-10` and
> 7.251e-6 at `1e-12`, no change across two orders. The control that says it is the tilt
> rather than the projector is the other in-plane axis: `m_y/|m|` is a fifth of `m_z` and
> the `y` residual is an eighth of the `z` one, so the residual follows the component being
> projected -- to within a factor of 1.7 in the constant, which is tracking rather than
> proportionality and is all two points establish. Both axes are now asserted and both
> bounds are 1e-4, which each guard clears by four orders: a spin factor of two would put
> the three-regime ratio at 1.0, and a factor of two in the projector would put the
> substrate residual at 0.5 or 0.25.
>
> **The `for_spin` trap was looked for beyond the test and has no live instance.** Only two
> things in the package call `fermi_surface_weights`, and neither is wrong: the nesting
> function forwards `degeneracy` from its own caller, which sets it from the spin regime
> (`workflows/nesting.py:167`, P52's fix), and it contracts on a complete grid where every
> point has the same weight, so there is no k-weight for the factor to double against. The
> new assertion is the only place the two conventions meet.
>
> **What the 9.0e-5 at a re-solved `(4, 4, 1)` means**, since it is the same mesh and ought
> to be round-off: `fixed_density_states` converges the bands to its *own* `ethr` rather
> than to the one the SCF finished on, so asking for a grid re-solves states that are not
> quite the SCF's even when the grid matches. It bounds how well any `grid=` comparison can
> agree with a quantity built from the SCF's own states, and it is the same shape as
> Part VIII item 3 one workflow over.


**Found in passing** while checking that P89's refactor of `_assemble` changed nothing: the
file came back `3 failed, 25 passed` at `c381be3`, and the same three fail at **`d085b54`**,
before any of this branch, to every printed digit. So the refactor is clean and these are
pre-existing. They are `slow`, so the push gate is not red.

**The two near misses**, both tolerance-shaped:

* `test_the_three_spin_regimes_agree_where_there_is_no_magnetization` compares two means of
  order 3.135e-8 and gets `1.70e-5` where it wants `1e-5`: a factor of 1.7 over.
* `test_a_substrate_across_the_moment_has_no_preference` gets `3.02e-17 / 1.06e-12 = 2.8e-5`
  against the same `1e-5`. The quantities are 1e-12 and the difference is 1e-17, so what the
  tolerance is measuring at that size is not obvious.

**The third is not a tolerance**, and it is the one to look at first.
`test_the_two_limits_are_the_stm_image_and_the_fermi_surface_on_a_real_cell` asserts that the
momentum-resolved weight's Tersoff-Hamann limit sums to the STM image's mean times the number
of k-points, and gets `0.00164` against `0.01256`, a factor of **7.7** where the tolerance is
1e-9. The nine per-k entries are one at 3.9e-30, four at 4.098e-4 and four at 4.17e-9, so the
weight is concentrated on half the mesh in a way the image is not.

**What is not known.** Nothing here says whether the defect is in the limit, in the image, or
in the test's own arithmetic, and the identity it checks is one the README quotes at 1e-12, so
it worked once. Neither the commit that broke it nor the date is known: the file's last three
commits (`8049534`, `284d123`, `cccd9ab`) are the obvious places to bisect, and a bisect here
is cheap because the three tests run in **57 s** on their own.


### 3. A resume spends a whole Davidson budget re-tightening bands it does not need **[closed 2026-09-15 -- the fix is the one named here, and the guard it needed was not]**

**The mechanism.** `band_thresholds` (`driver.py:233`) reads `wg = None` as "the first SCF
iteration, which has no occupations yet" and returns a flat `ethr` for every band, which is
`pw.x`'s own rule: `btype` is all ones out of `init_run.f90:149` and `sum_band` overwrites it
only after the first diagonalisation. A **resume** arrives with `wg = None` too, because the
driver sets it and neither `starting_from` nor the checkpoint carries it. But `ethr` *is*
restored from the checkpoint, tight. So the first iteration back holds every empty band to
the converged `ethr`, where a steady-state iteration holds them to `max(5 ethr, 1e-5)`, and
the empty bands do not get there.

**Measured**, two-atom silicon, `conv_thr = 1e-12`, checkpoint at iteration 5 and resume,
comparing the resumed run's first iteration against the same iteration of the uninterrupted
run:

| `nbnd` | uninterrupted | resumed, first iteration back |
|---|---|---|
| 4, all occupied | 2.0 | 2.0 |
| 40, four occupied | 1.0 | **5.5** |

so it is absent with no empty bands and 5.5x with thirty-six of them. On the 45-atom NiBr2
slab at `nbnd = 403` the resumed iteration reports `avg # of iterations = 100.0`, which is
`MAX_ITERATIONS` exactly, on every k-point -- the same effect reaching the budget.

**The docstring already declares this a deliberate deviation** from `pw.x`, which never
resets `btype` between SCFs, and calls the cost "one iteration of extra accuracy on the empty
bands of a restarted run, which is the conservative direction". That is true and it
understates the size: on a many-band cell the cost is the whole Davidson budget, not a
little extra accuracy.

**The fix, and the test it needs.** The checkpoint already carries `occupations`; feeding
them back as `wg` on resume gives the resumed iteration the thresholds the uninterrupted run
had. What it needs beside the code is
`test_an_interrupted_scf_costs_the_same_as_an_uninterrupted_one` extended from **SCF
iterations** to **Davidson steps** -- as it stands that test passes with this defect present,
which makes it another check whose null cannot be told from a pass. It also changes the
resumed run's eigenvalues in the last digits, so the equality it asserts has to be stated at
a tolerance rather than exactly.

**One reading to retire with it, and the retraction has since been completed from the other
side.** The first report of this took the `100.0` as a placeholder rather than a measurement,
on the grounds that the iteration was faster than neighbouring ones. The first answer to that
was that the cost cannot check the count at all, since four iterations at 17.8 to 23.7 steps
all took 850 s to 1.3 per cent -- a fitted 0.6 s per step on an 836 s intercept, which is a
slope of nothing. **That answer was right about those four points and wrong about the run**,
and the NiBr2 session established which by plotting the whole of every run rather than the
four: across a fifty-fold range of step counts, 1.5 to 100, the two `anderson` runs fall on
one line at **4.2 s per Davidson step on a 29 s intercept**. A hundred steps is then
29 + 420 = 449 s, and the resumed `local-TF` iteration that reported `100.0` sits at a
cumulative 652.9 s which also carries process start, the JAX compile and a 12 GB checkpoint
load. **So the budget really was spent and the cost corroborates the count**, in the same
direction as the silicon reproduction rather than merely failing to contradict it.

The lesson is the narrow sample rather than the conclusion: **four points over a 1.3-fold
range of the independent variable cannot resolve a slope**, so a fit through them reads as
"uncorrelated" whatever the truth is. That is `CLAUDE.md`'s search-that-cannot-surprise-you
one variable further in -- the instrument could not have produced the answer it was being
asked for, and the fifty-fold range is what made it able to.

### What was done, 2026-09-15

**The fix is the one the entry named**, in the block where the rest of the loop state comes
back (`driver.py`, beside `ethr`, `accuracy` and `field_scale`): the checkpoint's
`occupations` are fed back as `wg`. They are the *same array* the loop builds -- the
in-progress state is written with `occupations=wg` -- so nothing is converted.

**The reshape beside it is narrower than it first reads, and the difference was measured
rather than argued.** A *converged* result saved as a checkpoint by hand comes back with
its channel axis squeezed, which is `SCFResult`'s convention at one channel and at
`nspin = 4`, and `band_thresholds` takes its target from `np.shape(wg)`. On the main
branch that is harmless: a rank-2 `(nk, nbnd)` broadcasts against `weights[None, :, None]`
into `(1, nk, nbnd)`, the right shape by construction, measured at
`band_thresholds(1e-9, wg_(2,12), w_(2,)).shape == (1, 2, 12)`. The `diago_full_acc`
branch returns before any broadcast happens and gives `(2, 12)`, one axis short. So the
reshape is a fix for that one branch and is kept as one, rather than as the general guard
the first version of this paragraph claimed.

**Measured, and the test was run against the unfixed driver first.** Two-atom silicon,
`conv_thr = 1e-12`, `nbnd = 40`, checkpoint at iteration 5, comparing the resumed run's
first iteration back with iteration 6 of the uninterrupted run: **10.0 Davidson steps
against 2.0** before, **2.0 against 2.0** after. The entry's own numbers were 5.5 against
1.0 at a different pair of iterations, so the factor rather than the level is what
reproduces, which is what a threshold effect on the empty bands should do.
`tests/unit/test_scf_restart.py::test_a_resume_does_not_re_tighten_the_bands_it_does_not_need`
is the test, marked `slow` beside its sibling and asserting the pair to within one step
because restoring `wg` moves the resumed eigenvalues in their last digits.

**What the entry did not name, and it would have turned a fix into a regression.** A resume
is allowed to change `nbnd`, and nothing upstream stops it: the fingerprint compares the
loaded state against *itself*, so the shapes always agree there, and the grid check is on
the density. The checkpoint's occupations are then about a different set of bands and the
reshape raises, so a case that **worked** before -- resume at `nbnd = 12` from a state
written at 8, which converges -- would have died on the new line. The occupations are
dropped with a `RuntimeWarning` when the count does not match, which is exactly the
behaviour every resume had until today, and the guard has a test that trips it rather than
a clean pass that cannot be told from silence.

### 4. `local-TF` costs about 730 s an iteration on a 3.5-million-G-vector dense grid **[opened 2026-09-14, from the NiBr2 helix run; unprofiled]**

**The evidence is an A/B and a fit, which is the right order.** `nibr2_k161_localtf.scf.in`
differs from `nibr2_k161_anderson.scf.in` in exactly one line, `mixing_mode`: same geometry,
same `LOCAL_MAGNETIC_FIELDS` card, same `nbnd`, same `conv_thr`, same `mixing_beta`. Fitting
wall time against the printed Davidson step count within each run:

| job | mixer | occupations | `nbnd` | s per step | intercept |
|---|---|---|---|---|---|
| 20258495 | anderson | smearing | 403 | 4.70 | 30.3 s |
| 20260032 | anderson | fixed | 362 | 4.24 | 26.7 s |
| 20259567 | **local-TF** | smearing | 403 | -- | **850 s flat** at 18 to 24 steps |

The two `anderson` runs agree on both coefficients to within 10 per cent across a change of
band count *and* a change of occupations, which is what makes the law worth applying. Put the
`local-TF` run's step counts through it and it predicts about 123 s an iteration; it spends
850. **So `local-TF` costs roughly 730 s every iteration, five to eight times the whole rest
of the iteration**, and the intercept says the density, `addusdens`, `v_of_rho` and `newd`
together come to about 28 s.

**What is in the routine, structurally, and none of it is attributed.** `local_tf_preconditioner`
is the only preconditioner here that is not a diagonal multiply: `approx_screening2`'s operator
`4 pi e2 v + |G|^2 (alpha v)` has `alpha(r)` applied in **real** space, so it is not diagonal in
`G` and is inverted iteratively. Four properties of how that is written scale the wrong way on a
large dense grid, and **which of them dominates is unmeasured**:

- the Krylov loop is a **Python** loop, up to `LOCAL_TF_MMX = 12` directions with
  `LOCAL_TF_REFRESHES = 4` restarts, so about 60 steps, each dispatching two separate jitted
  FFT round-trips rather than one fused kernel;
- its inner product `dot()` returns `float(...)`, a **host sync**, called once per new entry of
  the `aa` matrix, so the syncs go as `m` per step and `m^2` per restart;
- `_alpha` runs on the host in numpy over the whole dense grid and is rebuilt **every
  iteration**, because the screening is a function of `rho(r)` -- which is the entire difference
  between this and `kerker` and is why only this one is expensive;
- none of it batches with anything else, so it is pure added wall clock.

**Do not read this as an argument against the algorithm.** `local-TF` is in that input because
an inhomogeneous slab is what it is for, and `kerker`'s single screening length is wrong there;
if the 730 s is an implementation cost then the input stays right. `kerker` is the intermediate
worth timing first, one transform against sixty.

**First step.** Profile one `local_tf_preconditioner` call on a dense grid of that size,
separating the host syncs from the FFTs, before changing anything. On a GPU the syncs are the
first suspect and on a grid that size the FFT count is, and that cell has both -- which is
exactly why it needs measuring rather than reasoning.

---

# Part IX -- from the tunnelling spectrum, 2026-09-15 (P90)

## 1. A bias window is integrated with no check that the axis resolves the broadening **[closed 2026-09-15 -- measured on the transmission, and the threshold is the delta's]**

`run_vertical_transport(bias=)` and `run_ultracell_transport(bias=)` both turn a bias into an
axis through `_energies(energies, levels, bias, nenergies)` and then `np.trapezoid` over it
(`workflows/transport.py:278` and `:495`), and the only thing `_energies` checks is
`nenergies >= 2`. Nothing compares the **step** of that axis against `broadening`.

**Why that is not a small error.** A trapezoid over a Lorentzian or a Gaussian it does not
resolve does not return a slightly wrong number, it returns one of the wrong order. Measured
on the smeared delta this package uses, one level of width `w` on an axis of step `h`
integrates to **1.000000 at `h = w/2`, 1.00004 at `h = w`, 1.14 at `2w`, 0.10 at `4w` and
exactly 0.000000 by `250w`** -- so the failure is not monotone either, and a coarse axis can
read *high* before it reads zero. The default `nenergies` is 1, which `_energies` then
refuses; every larger value is accepted whatever the window is.

**What exists meanwhile.** `STMSpectrum.current` (P90) has the guard: it refuses an axis whose
largest step exceeds the width, naming both numbers, and `tests/unit/test_stm_machinery.py`
feeds it the case that must trip it. What is missing is the same guard on the transmission's
own `bias=`, which is a validated path and was left alone rather than changed in a phase about
something else.

**What it took, and the argument held.** The measurement was taken first, on the
transmission's own weight rather than on the delta: `h-sheet.in`, one tip point,
`broadening = 0.02` Ry over a 0.24 Ry window, every row against a `w/8` axis.

| `h/w` | current | ratio |
|---|---|---|
| `w/2` | 1.2878e-08 | 0.99917 |
| `w` | 1.2847e-08 | 0.99673 |
| `2w` | 1.3015e-08 | **1.00977** |
| `4w` | 3.8444e-09 | 0.29828 |
| `12w` | 1.5611e-09 | 0.12112 |

**The reference is converged to about 2e-4, not to the five decimals the ratios are
printed at**: `w/4` differs from `w/8` by 1.7e-4, so the first two rows are meaningful to
three decimals. The conclusions sit well above that -- the `w` row is 3e-3 from one and the
`2w` row 1e-2 -- but the ratios are not a fifth-decimal statement.

So the threshold carries in units of `broadening` and the *numbers* do not: the bare delta
reads 1.14 at `2w` and 0.10 at `4w`, and a sum over several levels reads 1.010 and 0.298.
What is the same is the shape -- one point per width passes at 0.997, and the failure is
**not monotone**, so a coarse axis reads high before it collapses and nothing downstream
can tell which it did.

The guard is in `_energies` (`workflows/transport.py`), after the `nenergies >= 2` check
and before the `linspace`, at the same `broadening * (1 + 1e-8)` boundary the STM one uses,
and it names the `nenergies` that would do rather than leaving the reader the division.
**The check is at the door rather than in `_energies`**, which runs *after* the
fixed-density re-solve a `grid=` asks for: it needs only `bias`, `nenergies` and
`broadening`, so being refused after paying for an NSCF would have been the same mistake
Part VI item 1 records one entry point over. `_check_bias_axis` is called first in each
entry point and from `_energies` as well, so a caller that reaches it another way is still
guarded. **There were three call sites, not the two this entry named**:
`run_vertical_transport`, `run_momentum_transport` and `run_ultracell_transport`, the last
in `workflows/ultracell.py` rather than beside the others.

**What was not run end to end**: only `run_vertical_transport` was exercised with a real
`bias=` window. The other two take `broadening` two lines from where they take `bias`, so
that they forward it is read rather than measured, and the ordering test above is a
statement about two calls in a source file rather than about a run. `run_sts`'s axis goes through `_spectrum_energies` instead and is
integrated by `STMSpectrum.current`, which already had the guard.

Four tests in `tests/unit/test_transport_machinery.py`, all in the gate and none needing an
SCF: one that **fires** and whose suggested `nenergies` is then checked to pass, one at the
boundary, one showing an `energies=` list without `bias=` is untouched -- there is no
trapezoid there, so each energy is its own zero-bias conductance -- and one reading the
order of the two calls off each entry point's source.

## 2. The reference pair for a unit-cell tunnelling spectrum was not taken **[2026-09-15]**

`pp.x` with `plot_num = 5`, run once per bias, is the honest reference for `run_sts` with an
energy axis, and `pp.x` is built in the vendored tree. It was not measured because the machine
was carrying another job at a full core for the whole session, and a single-core wall clock
taken beside one is not a measurement (`PERFORMANCE.md` says so in the entry). What went in
instead is this code's own two routes run back to back under the same load, where the **ratio**
is robust and the absolute figures are for scale. Take the pair on a quiet machine: 41 biases
through `pp.x` against one `run_sts` call, both single core, same cell.

# Part X -- from the augmented ultracell, 2026-09-17 (P88 stage 5)

## 1. The ultracell's reciprocal cutoff set is not closed under negation, so a real field on it is not exactly real

**Where it is.** `Ultracell.reciprocal_mask` in `ultracell/grid.py`, and every consumer of
`keep` in `ultracell/potential.py` and `ultracell/augmentation.py`.

**What it is.** The ultracell keeps the unit cell's dense sphere at *every* `Q` -- Elk's
choice, `ngvec` G-vectors per `Q`, and what makes the `N = 1` limit reduce to the unit cell
exactly. The negative of `G + Q` is `-G - Q`, which is that sphere **displaced** rather than
that sphere, so once `Q` is non-zero the kept set has entries whose negative is not kept.
Measured on the silicon cell of `tests/data/qe/si-ultracell-paw.in`: **235 of 4554 at
`N = 2` and 470 of 6831 at `N = 3`**, against none at all of the 2277 at `N = 1`.

A real function represented on such a set is therefore not exactly real, and both places
that hold one take the real part and move on: `ultracell_potential`'s Hartree term masks a
conjugate-symmetric density with `keep` and takes `jnp.real` of the transform, and
`ultracell_augmentation_charge` does the same with the augmented charge.

**What is measured and what is not.** The augmentation side is measured, because the check
that found this was written for it: `UltracellResult.augmentation_residual` is 1.71e-4 on
ultrasoft silicon at `ecutrho = 64` Ry, falling to 7.21e-5 at 96 and 2.96e-5 at 144, which
is `ecutrho^-2.2`, and exactly zero at `N = 1`. **The Hartree side is not measured at all**
and is the older of the two -- it has been there since stage 1 and nothing reports it.

**What the convention is worth on the total energy is now measured, and it is nothing**
(2026-09-20, P88 stage 9, jobs `20350372`/`20350451`). Masking the ultracell's own
`|G + Q|` sphere in place of the unit cell's dense sphere at every `Q` -- a genuinely
different set, 12872 kept against 12846 at `ecutrho = 8 ecutwfc` and 23835 against 23870 at
12 -- moves no digit of any rung of any silicon case, at 1e-10 on the total. The arm that
says the probe is live is a quarter of that cut-off, which moves the total by 8.3e-4 Ry.
**That is the sphere and not `keep & flip`**, so the symmetric hull below is still
unmeasured; what the two arms bound is the size of the effect, since the sphere differs
from `keep` by more vectors than the hull does. The `ecutrho` dependence of the residual
came with it, across grids rather than along one: 5.4e-5, 1.13e-5 and 5.24e-6 at
`ecutrho = 4, 8, 12 ecutwfc` on PAW silicon, which is the `ecutrho^-2.2` this entry
measured on one grid.

**What it would take.** Symmetrising the kept set is one line, `keep & flip` with
`flip = np.roll(keep[::-1, ::-1, ::-1], 1, axis=(0, 1, 2))`, and it changes the method's
cutoff from Elk's set to its symmetric hull, which is a *different truncation* rather than a
fix -- it throws away components that are there. So the thing to measure first is what the
present convention is worth: the Hartree energy and the total at `N = 2` with `keep` and
with `keep & flip`, on the same cell. If that difference is at the level the energy ladder
resolves (1e-7 Ry per cell), it belongs in the convergence story; if it is below, this entry
closes as a convention with a number behind it.

**Why it is not urgent.** It is a truncation and it converges away with `ecutrho`, it is
identical on both sides of every comparison this project has made (the ultracell against
itself at different `nbnd`), and it is *not* identical to the supercell's own truncation --
which is the one place it could matter, and where the measured 3.7e-3 induced-density
agreement at `nbnd = 48` already bounds it from above.


## 2. A nickel comparison that looked like a violated variational bound, and was a void test cell **[closed 2026-09-17, same day, from P88 stage 6]**

Kept because the chain of three refuted explanations is the point, not the answer.

**What it looked like.** Fcc nickel, `Ni.pz-nd-rrkjus.UPF`, `N = 2` at `ecutwfc = 22`,
`ecutrho = 88`, under a `0.05 cos(pi x_1)` Ry applied potential on both sides: the ultracell
reached **-85.55680649** Ry per unit cell against the supercell's **-85.55677126**, so
**3.5e-5 below**, and further below with more bands. The ultracell's basis is a subspace of
the supercell's plane-wave space, so that is forbidden.

**Three explanations, each tested and each wrong.** *Several self-consistent states*: the
supercell reaches -85.55677126 from `starting_magnetization` of 0.7, 0.4, 0.9 and 0.2, to
eight decimals, in 30 to 73 iterations. *The entropy*, which is what a metal suggests
first: it differs by 7.9e-6 where the total differs by 3.5e-5, and the electron count is
10.0000000000 per cell on both sides with the Fermi levels 4.4e-5 Ry apart. *The
linearisation* -- the ultracell's matrix being the supercell's Hamiltonian only to first
order in the density change, with the unit cell's bands frozen, which would bite exactly
where `v_xc` is nonlinear in `m`: refuted by the amplitude scan below.

**What the amplitude scan showed instead.** The ultracell's moment modulation scales with
the applied potential as a response must -- 0.0137, 0.0055, 0.0014, 0.0003 at amplitudes
0.05, 0.02, 0.005, 0.001 -- and **the supercell's does not move at all**, sitting at 0.242,
0.237, 0.235, 0.234. The gap converges to -3.92e-5 Ry as the perturbation goes to zero, so
it is a difference between the two *unperturbed* states and has nothing to do with the
modulation.

**And with nothing applied the supercell breaks its own symmetry.** Moments
`[1.4478, 1.2137]`, a 17.6 per cent spread, at `E/N = -85.55561421` Ry -- against the unit
cell's `-85.55565341` on the equivalent grid, which is **3.92e-5 lower**, the whole of the
gap. So the supercell SCF was converging to a symmetry-broken state *above* the uniform one,
the ultracell was reporting the uniform state's energy correctly, and the "violation" was
the supercell failing to reach its own minimum. The breaking survives refining the k-grid
(`[0.6753, 0.5572]`, 19.2 per cent, at `2 4 4`), where the moment also halves from 1.33 to
0.62 -- the physical LDA value -- which says the cell was under-sampled by a factor of two
in the moment as well.

**What to take from it.** The ultracell was never wrong here, and the test cell was built in
an afternoon without checking that its supercell reproduced its unit cell with nothing
applied. **That check costs one run and is the first thing a supercell comparison on a metal
should do** -- on a gapped cell it is invisible because there is nothing to break. The
separate observation, that `run_scf` on a doubled nickel cell converges to a symmetry-broken
state with nothing applied and stays there, is real and is not the ultracell's; whether it is
a genuine instability of LDA nickel at this cutoff or an SCF that stalls in a broken state is
not settled here.

## 3. At a dual the reference supercell is on a different box, and reads 2.2e-6 Ry per cell high **[opened 2026-09-20, P88 stage 9]**

**What it is.** A supercell chooses its own dense FFT grid, and at
`ecutrho = 8 ecutwfc` on the two-atom silicon cell that is `(54, 25, 25)` where the
ultracell's box, which is `N` times the unit cell's, is `(50, 25, 25)`. The two sides then
do not discretise the same functional, and the difference is not small against the number
an `nbnd` ladder is trying to resolve: **with nothing applied the supercell sits 2.1978e-6
Ry per cell above its own unit cell**, at `ecutrho = 4 ecutwfc` 5.4e-13.

**How it showed.** As a violated variational bound, which is the same costume `OPEN.md`
Part X item 2 wore: the first ladder at a dual put the ultracell 1.47e-6 Ry *below* the
supercell at `nbnd = 48` and 1.90e-6 below at 96, falling further with every rung. Nothing
was wrong with the ultracell. Reading each side against its **own** unmodulated state gives
+1.07e-4, +4.71e-6, +7.23e-7 and +3.02e-7 Ry at `nbnd = 12, 24, 48, 96`, above at every rung
and falling, which is the ladder at `ecutrho = 4 ecutwfc` with a slightly better
augmentation charge.

**Measured across nine cases** (jobs `20350372`, `20350439`, `20350451`; the table is in
`PLAN.md` P88 stage 9). The offset **follows the boxes rather than the dual**: 2.2e-6 Ry per
cell at `ecutrho = 8 ecutwfc` on every silicon case alike, collinear and spinor, where the
supercell picks `(54, 25, 25)` against a tiled `(50, 25, 25)`; **2.4e-8 at 12**, where it
picks `(64, 32, 32)`, which *is* the tiled box; and 5e-13 at 4. On platinum at its dual it
is -9.4e-7, so the sign is not fixed either.

**What is open.** The protocol -- compare the energy of the modulation, each side against
its own unmodulated state -- cancels the offset to **first order and not exactly**: the
spinor case's `nbnd = 192` rung reads -2.0e-7 Ry, about a tenth of the offset removed, so a
gap below a few 1e-7 on this cell says nothing. Whether the remainder is the modulated
state's own grid error or something else is not settled, and the clean way to settle it is a
cell whose supercell box *is* the tiled one at a dual, which `ecutrho = 12 ecutwfc` on
silicon happens to be. Whether any of it reaches the **density** comparison is also open;
that one is read Fourier component by Fourier component and has a two-box floor of its own,
which `tests/regression/test_ultracell.py` puts at about 1e-4 relative.

**The thing not to do** is to fix it by pinning the supercell's grid: `nr1/nr2/nr3` are
refused at input here by name, the norm-conserving file reaches the same end by choosing an
`ecutwfc` where the two boxes coincide, and a dual has no such lever. The difference of
differences needs no grid to agree.

# Part XI -- from the seeded NiBr2 helix on a GPU, reported 2026-09-17 (P88 stage 8)

## 1. A batched FFT plan fails to build at an ultracell's band count, and it is not an out-of-memory

**Reported by the NiBr2 session and not reproduced here** -- this workstation has no GPU, so
what follows is their observation with their numbers, recorded because the batching dial it
implicates is this package's.

A seeded `N = 15` ultracell on a fully relativistic PAW NiBr2 cell ran at `nbnd = 40` and
died at `nbnd = 56`, in the **NSCF Davidson** rather than in the ultracell loop, with

```
RET_CHECK failure ... fft_plan != nullptr
Failed to create cuFFT batched plan with scratch allocator
```

at `DEFUMAT_K_BATCH = 16` and `DEFUMAT_BAND_BATCH = 64` on an 80 GB card. Dropping to 4 and
16 is what they reran with.

**Why it is worth an entry rather than a shrug.** It is a *plan creation* failure and not an
allocation failure, so it is a limit on the transform's batched shape rather than on the
memory available -- and the step it happens in is the one an ultracell makes large in a way
no ordinary run does: `fixed_density_states` diagonalises the whole folded k-set, which is
`N` times the unit cell's, so `k_batch x band_batch` transforms at `nbnd = 56` on a 3-atom
cell at `ecutwfc = 45` is a batch shape that only this method reaches. The defaults follow
the platform (`defumat/batching.py`), and on an accelerator they are the batched end of the
dial, so an ultracell is exactly where they are least tested.

**What is not known**: whether the limit is cuFFT's own plan size, the scratch allocator's
budget under XLA, or a shape this code builds needlessly wide. Nothing here reads the batch
size in a way a result depends on -- the chunk size must never be visible beyond round-off,
which is asserted -- so the workaround is sound and the question is only where the wall is.

**What to do before anything else**: on a machine with a card, walk `DEFUMAT_BAND_BATCH` at
fixed `k_batch` and find the largest batch that plans, then the same for `k_batch`, and see
whether the product or one factor is the bound. If it is the product, the fix is to cap the
batched shape in `batching.py` for the folded-set solve rather than to leave a user to find
it; if it is one factor, the cap belongs there. **Do not size it from the card's memory**,
which is the reading this failure already rules out.

# Part XII -- from the augmented spinor response, 2026-09-17 (P98)

## 1. Bismuthene's ground-state energy sits 3.5e-5 Ry from `pw.x` where AlAs sits at 2e-9 **[opened 2026-09-17]**

`bismuthene-epsilon-us-soc.in`, the cell P98 added so that a heavier element could resolve
the `fcoef` dressing of the augmentation terms: two bismuth atoms in a honeycomb with
vacuum, `Bi.rel-pbe-dn-rrkjus_psl.1.0.0.UPF`, `noncolin` and `lspinorb`, `ecutwfc = 20`,
`ecutrho = 160`, `occupations = 'fixed'`, a 4x4x1 grid.

**The two totals.** `pw.x` gives **-295.59282302 Ry** and this code **-295.592858395 Ry**,
which is **3.5e-5 Ry**. On the fully-relativistic AlAs run of the same phase the two codes
agree to **2e-9 Ry**, so this is four orders worse in absolute terms and about a thousand
times worse relative to the total.

**What is already excluded, by reading rather than by argument.** Both codes are converged:
`pw.x`'s last `estimated scf accuracy` is 2.3e-13 Ry and this run asks for `conv_thr =
1e-12`. Both chose the **same grids**: dense `(45, 45, 81)` with 60543 G-vectors and smooth
`(30, 30, 60)`, read off `pw.x`'s output and off `Calculation.basis`. So it is neither a
stopping point nor a box.

**No explanation is offered here**, deliberately: the candidates are the radial
interpolation floor amplified by ten beta functions and a nonlinear core correction on a
heavy atom, the `dn` semicore channels, and the vacuum, and each of them *fits* -- which is
the reason to write the number down and not a story around it. What would discriminate is
the eigenvalues rather than the total, since the three candidates put their error in
different places, and a second heavy relativistic ultrasoft cell without vacuum.

**It does not touch P98's claims.** The dielectric comparisons of that phase are on AlAs,
where the ground states agree to the printed digit; bismuthene enters it only as the cell
whose A/B says how much the spin dressing of the augmentation terms is worth.

# Part XIII -- from the 2026-09-18 audit fixes

## 1. Two DFT+U regression files sit at or over the 12 GB per-file cap, and have for a fortnight

`tools/run_regression.sh` on the DFT+U set, 2026-09-18, one capped process per file:
`test_ldau.py` reaches **10,937 MB** of a 12 GB cap, so the in-process watchdog names
`test_converged[pw_lda+U/lda+U_force.in]` and fails it at 85 per cent while all 73 of the
file's own assertions pass, and `test_ldau_flavours.py` is **killed outright** at 12,423 MB.

**Neither is new and neither is a physics failure.** A regression summary from
**2026-09-04** already reads `test_ldau_flavours exit=137`, two weeks before anything in
this session was written, and the named `test_ldau` case is a `HUBBARD {atomic}` run,
which never reaches the code the same session changed. It is `CLAUDE.md`'s own
"a test file that sweeps many cells is a memory liability": cells that share no shape each
compile the whole SCF stack afresh and XLA keeps every executable for the life of the
process, so the peak is accumulation over the file rather than any one test's working set.

**What to do about it**, in the order the file's own rule gives: `jax.clear_caches()` in an
autouse fixture after the `yield`, and `lru_cache(maxsize=2)` on the converged-state helper.
**Raising the cap is not the workaround it looks like**: at `DEFUMAT_TEST_MEM_MAX=16G` the
same file is killed again, at **15,316 MB**, in
`test_a_spinor_occupation_matrix_matches_pw_x` -- three quarters of the way through rather
than half, which is what accumulation does to a cap. Selecting a subset with `-k` bounds it
where a larger number does not, because what is being bounded is how many distinct cells
one process has compiled.

## 2. The suite exhausts *mappings* rather than memory on a cluster node

Found by running the whole `slow` set on Triton, 2026-09-19, at a commit where
every one of these tests passes on the workstation. Fifteen of
`test_electrostriction.py`'s twenty failed, all with the same
`jax.errors.JaxRuntimeError: INTERNAL: Failed to materialize symbols`, and
`test_spectra.py` aborted outright; the piezoelectric measurement job died the
same way with `LLVM compilation error: Cannot allocate memory` and
`Failed to satisfy suballocation request for **118**` bytes, on a two-atom cell
with **120 GB** allocated and a resident set of three.

**It is not memory.** A failed 118-byte request with 120 GB free is an
*address-space* failure: XLA's CPU backend gives every jitted function its own
ORC dylib and mmaps its sections, `vm.max_map_count` on these nodes is the
ordinary 65530, and a process that compiles thousands of distinct executables
runs out of mappings. Which processes do that is exactly the set
`CLAUDE.md`'s memory section already names: the ones that sweep many cells, each
of which compiles the whole SCF stack afresh while XLA keeps every executable
for the life of the process.

**The prescribed cure is `jax.clear_caches()` in an autouse fixture after the
`yield`, and it has now been measured on a node: it does not cure the
exhaustion, it delays it.** Job `20337788_2`, 2026-09-19, ran the two files that
gained the fixture (`test_electrostriction.py` and `test_spectra.py`) at
`4773c75` on `batch-milan`, and against the same two files in the whole-set run
at `abee9c21`, before the fixtures landed:

| file | without the fixture | with it |
|---|---|---|
| `test_electrostriction.py` | 15 failed, 5 passed, 285 s | **1 failed, 19 passed**, 1521 s |
| `test_spectra.py` | `exit=134`, aborted inside `backend_compile_and_load` | **8 passed, 6 skipped**, `exit=0`, 569 s |

**What the fixture does, measured directly rather than argued about, and it
settles three things this item had wrong.** The measurement is a Berry-phase
polarization on norm-conserving AlAs at `ecutwfc = 10`, 16 strings of 6 points,
counting the lines of `/proc/self/maps` after each string:

| | mappings | wall clock |
|---|---|---|
| after the SCF, before the string loop | 2172 | |
| after 16 strings, no intervention | **9873** | 13.0 s |
| then `jax.clear_caches()` alone | **949** | |
| then `gc.collect()` alone, after that | 949 | |
| 16 strings with `jax.clear_caches()` after each | **976** | 36.6 s |

**So `jax.clear_caches()` does unmap.** It released 8924 mappings in one call,
`gc.collect()` released none on top of it, and clearing inside the loop keeps
the count flat for 2.8 times the wall clock. That is the opposite of
`40d8fe2`'s "clearing caches does not unmap", and of the mechanism this item
carried for a day, that clearing the cache forces a recompilation whose dylibs
add mappings. Both are withdrawn. The growth is also entirely **anonymous**
mappings, not file-backed ones: classifying every line of `/proc/self/maps`
puts all 7700 of the increase in `[anon]`, which is where JIT-compiled code is
mapped.

**Then why did the fixture not save `test_nonlinear.py`, which had it?** Because
an autouse fixture with a `yield` fires **between tests**, and these processes
run out of mappings *inside* one. `test_nonlinear.py` has eleven tests and the
run counted 84 occurrences of the error string, so an individual test is
producing them rather than the file accumulating across tests. The same shape
explains `test_electrostriction.py`: nineteen tests pass and the failure is
`test_the_wedge_reproduces_the_closed_grid`, one test that exhausts by itself.
**The fixture bounds accumulation across tests and can do nothing about a
single test that compiles 65530 mappings' worth**, which is exactly what a
396-string polarization or a many-cell sweep is. That is also why the piezo
harness's per-geometry child did not help: the string loop is inside one
geometry.

**So the fixture is right and it is in the wrong place.** The measured
offenders that do not have it are `test_spinor_dielectric.py`,
`test_dispersion.py`, `test_response.py`, `test_lsda_response.py` and
`test_gamma_only.py`, and adding it there is cheap and bounds what it can
bound. What it will not fix is any single test that walks a long loop, and for
those the clear has to go **inside the loop**, which is what
`run_polarization` now does.

**One more thing in the same file, against the project's own rule.**
`test_nonlinear.py:92` is `@lru_cache(maxsize=None)` on `_converged`, where
`CLAUDE.md` says "`lru_cache(maxsize=2)` on the converged-state helper, never
`maxsize=None` -- 2 is what a comparison between two cells needs and is the
largest that is not a leak". That is a memory leak rather than a mapping one
and it is not what failed here, but it is in the file the audit ranked worst.

**A process boundary per test remains the blunt cure, and it needs no plugin.**
`pytest --forked` is not available -- the cluster venv has neither `pytest-forked` nor `xdist`, and no
`psutil` either, which is why every cluster log carries "the memory watchdog is
off", and no `matplotlib`, which is why `tests/unit/test_result_plots.py`
cannot be collected. **What is available is `tools/run_regression.sh` itself**,
which invokes pytest once per entry of its file-glob argument and already
accepts node IDs there: the attribution array passed seven of them. So a file
that exhausts mappings can be run as a list of its own test IDs, one process
each, today. It is the fallback rather than the answer, because clearing the
caches in the right place is cheaper than paying process startup per test.

**Two non-test witnesses of the same exhaustion, and this time the cause is
in our code rather than in the test suite's shape.** Rungs 4 and 6 of the piezo
ladder both died with `Failed to materialize symbols` inside the Davidson
eigensolver, in a script that already puts each strained geometry in its own
child: rung 4 on the norm-conserving cell at 64 strings, rung 6 on the
ultrasoft one at 36, where the same 36 strings on the norm-conserving cell
(rung 3) completed in ten minutes. So the string loop inside one geometry
exhausts the mappings by itself and the per-geometry boundary is not a general
answer.

**Why the string loop compiles more than once, measured host-side on
`alas-raman.in` with no SCF.** `run_polarization` takes one string at a time and
`_source.states()` diagonalises it, and each string gets its *own* `npwx`,
because the sphere is rebuilt at every k and the strings sit at different
transverse points. Counting the plane waves inside `ecutwfc` for every point of
every string:

| transverse x `nppstr` | strings | distinct `npwx` | range |
|---|---|---|---|
| 4x4, 7 | 16 | 8 | 750 to 765 |
| 6x6, 11 | 36 | 10 | 755 to 766 |
| 6x6, 15 | 36 | 9 | 755 to 763 |
| 8x8, 15 | 64 | 11 | 747 to 763 |

Every distinct `npwx` is a distinct static shape, so the whole Hamiltonian and
Davidson stack is compiled again for each, ten or eleven times per geometry.
**This is `CLAUDE.md`'s own JAX rule being broken** -- "pad plane-wave arrays
to `npwx` with a mask instead of using per-k shapes" -- and it is worth fixing
on its own, since padding a mesh's strings to the mesh-wide maximum makes one
executable serve all of them.

**It is not, however, what exhausts the mappings, and the measurement above
says so.** Timing each string of the 16-string mesh beside its map count, the
count grows by roughly 480 to 650 on *thirteen of sixteen* strings, including
strings whose phase takes **0.04 s** and which therefore compiled nothing. So
the growth is per string rather than per distinct shape, and eight
recompilations cannot account for 7700 mappings. Extrapolated, 480 per string
puts the ceiling near 115 six-point strings, and the growth scales with the
points per string, which is what separates the rungs that ran from the rungs
that died: 16 strings of 6 and 36 strings of 14 finished, 64 strings of 14 and
36 ultrasoft strings of 10 did not. The fix is therefore to clear the caches
inside the loop, measured above to hold the count at 976 across the whole mesh,
and the padding is a separate improvement to the compile count and the wall
clock rather than the cure.

**The whole `slow` set was then run on a node** -- eight array tasks, 22 files
each, job `20336106`, all eight `COMPLETED`. **The first reading of it ranked
the files by how many times the error string appeared, and that ranking is
wrong**, because a single test that loops over displaced geometries raises once
per geometry. `test_nonlinear.py` led that list with 84 occurrences and is in
fact **2 failed of 13**, the least affected file of the eight. Ranked instead by
what the runner reports, which is tests:

| file | failed / run | peak | error strings, the old proxy |
|---|---|---|---|
| `test_spectra.py` | **aborted**, `exit=134` | 3.0 G | none: it aborts rather than raising |
| `test_gamma_only.py` | **aborted**, `exit=134` | 2.9 G | none, same |
| `test_electrostriction.py` | **15 of 20** | 2.9 G | 55 |
| `test_spinor_dielectric.py` | **4 of 6** | 3.0 G | 70 |
| `test_lsda_response.py` | **3 of 7** | 3.5 G | 26 |
| `test_dispersion.py` | **8 of 22** | 3.2 G | 60 |
| `test_response.py` | **7 of 35** | 3.8 G | 47 |
| `test_nonlinear.py` | **2 of 13** | 3.0 G | 84 |

Both aborts are `Fatal Python error: Aborted` with the faulthandler stack inside
`backend_compile_and_load`, which is the same exhaustion reaching `abort()`
inside LLVM instead of returning an error.

**And the ordered pass/fail sequence is the evidence the fixture paragraph above
needed.** `test_nonlinear.py`'s progress line is `...FF........`: three pass,
two fail, and then **eight pass after them**. The two are consecutive
parametrisations of one test, `si-us` and `si-paw`, the two augmented datasets
and the heaviest cases in the file. So a test that exhausts the mappings does
not poison the ones after it, which is the autouse `jax.clear_caches()` doing
exactly what the measurement says it does, and it is why the fixture is worth
adding while being no use to the test that exhausts inside itself.

Of the individual failures in the six raising files, the great majority are
`jax.errors.JaxRuntimeError` and eight are assertions, which item 4 below
attributes one at a time.

`test_stress.py`, `test_input_sweep.py`, `test_lsda.py`, `test_spinorbit.py`,
`test_noncollinear_magnetism.py`, `test_scf.py`, `test_magnetic_constraints.py`,
`test_uspp.py`, `test_topology.py` and `test_ldau.py` are *candidates* that this
run did not convict; several passed outright, and `test_ten_site.py` is the
memory outlier rather than a mapping one at **17.2 G**.

**One file fails on the node for a reason that is not this and not the code**:
`tests/unit/test_result_plots.py` cannot be collected because the cluster venv
has no `matplotlib`. Worth knowing before reading a cluster summary as a verdict
on the repository.

**Two things not to conclude.** The workstation does not show this, so nothing
here says those files are wrong; and `test_ldau.py`'s and
`test_ldau_flavours.py`'s appetite (Part XIII item 1) is a genuine *memory*
problem rather than this one, measured in bytes on a machine with no cgroup
surprises. What the two share is the cause, which is accumulation, and therefore
the fix.

## 3. The piezoelectric tensor's response route is not converged in `k` at the mesh its own input asks for **[opened 2026-09-19, cause found the same day]**

Measured on Triton (jobs `20336374`, `20336476` and the ladder `20337789`,
`tools/cluster/piezo_measure.py`), and the full tables are in
`AUDIT-2026-09-18.md` `drift.3` and `PLAN.md` P50. **The entry opened as "two
routes disagree by 13 per cent" and the ladder says which one was wrong.** On
norm-conserving AlAs, `e_14` in C/m^2:

| route | `4 4 4`, the committed mesh | `6 6 6` | `8 8 8` |
|---|---|---|---|
| the implementation, one `jvp` of the stress | **-0.763786** | **-0.687475** | **-0.672897** |
| Berry-phase difference, Elk's route | -0.661386 | -0.661964 | not run |

The response route moves twelve per cent of itself between the committed mesh
and `8 8 8`, all of it toward the Berry value, and the Berry value moves 0.7 per
cent across four string meshes and two ground states. The pair that shares a
ground state settles it without any argument: the `6 6 6` response rung and the
`6 6 6` Berry rung ran the same SCF, to -16.89293132223534 Ry in every printed
digit, and going from `4 4 4` to `6 6 6` moved the Berry `e_14` by 0.0006 and
the response `e_14` by 0.076. The density is converged where the input says; the
k-integration inside the response solve is not.

**What this costs a user.** `Calculator.get_piezoelectric_tensor()` on a
committed-quality input returns a number that is thirteen per cent out and
announces nothing, because every internal check this quantity has (the three
routes, the symmetry statements, the wedge, the `Z*` anchor) is insensitive to
the k-mesh: the routes share the response, and the anchor is the same assembly
in the position coordinate. The entry point has no convergence guard and none of
the eight regression tests would catch a repeat.

**What is still open.** Three things, in order of what they are worth. First,
`e_14` is still 1.6 per cent from the Berry value at `8 8 8` and still moving,
and nothing says whether that reaches zero; one rung, `nc --kmesh 10
--skip-difference`, would say, and rung 1 of the ladder took four minutes.
Second, the **ultrasoft and PAW refusal stays**. Both cells have now been
measured at the *same* two meshes (`20338380`, `20338430`): at the committed
`4 4 4` response mesh and 11 Berry strings over 6x6, the deficits are **13.41
per cent** norm-conserving and **15.05 per cent** ultrasoft, a difference of
**1.65 points**. That is the number the entry wanted and it is not attributable
yet, because essentially all of the 13.41 is k-convergence and two datasets at
`ecutwfc` 10 and 25 need not converge at the same rate, so 1.65 is a dataset
effect plus a difference of mesh errors with nothing separating them. Closing it
needs the ultrasoft response at `6 6 6`, which is above the **139.6 GiB** the
same route takes at 64 points by an amount nothing on record gives -- the only
other ultrasoft peak, 49.4 GiB at 8 points, is from a job with a different task
list and different symmetry, so the two do not make a slope. `k_batch = 1` is
worth 11 per cent and nothing in the answer, and the reason is structural:
`forces/energy.py:energy_at` does not go through `map_k` or `sum_k`, so the dial
never reaches the function being differentiated and the eleven per cent is the
SCF and the field response underneath.

**The cheap route was tried and it is norm-conserving only.**
`piezoelectric_zstar_eu_style` reads +0.830702 on that cell at those 64 points
against +0.815802, 1.8 per cent, where the two agree to 6.2e-15 on the
calibration cell, because `zstar_eu.f90:90` hands an augmented dataset to
`zstar_eu_us.f90` and this transcription stops at the first file. It is refused
by name now. **What would make the cheap route correct, sized rather than guessed, because
the template is in this repository.** `born.py`'s own table names the four
things an ultrasoft Born charge needs beyond `zstar_eu`'s main term, and for a
*piezoelectric* constant the structure is the same as the Born charge rather
than the same as a phonon: **only one leg moves `S`**. The strain leg does, and
`_bare_strains` already carries it; the field leg does not, so its `dLambda` is
a matrix element of what the field response has already built rather than a
response to be solved for. What is left is therefore the single contraction
`-<psi_m|dS/d(eps_ab)|psi_n> . dLambda^E_mn`, and **both factors exist as
functions already**:
:func:`defumat.response.strain.overlap_derivatives` is
`<psi_m|dS/d(eps_ab)|psi_n>` for the six strains and returns ``None`` for a
norm-conserving dataset, and
:func:`defumat.response.born._multiplier_response` is
`dLambda_mn = w_n <psi_m|dV_E|psi_n>`, which its own docstring calls "the whole
of QE's first two ultrasoft stages at once". Both vanish identically for a
norm-conserving dataset, **which is exactly why the two routes agree to
6.2e-15 there and differ by 1.8 per cent here**. The `add_dkmds` term of that
table is `jax.grad` of `frozen_polarization` and vanishes for the non-polar
crystals this route is allowed on at all.

**It is written, and the half of the check that this machine can do has
passed** (`_multiplier_strain_term`, 2026-09-19): the two routes still agree to
**6.217e-15** on the calibration cell, unchanged to the digit, because both
factors are identically zero when `S` does not deform. That says the term
cannot break a norm-conserving answer and nothing about whether it is right --
the first validation run died on `internals["nocc"]` being the per-spin tuple
where `solver.nocc`, the number, was wanted, which is a line no norm-conserving
cell reaches. **Measured, and the term is right and not sufficient** (`20339308_0`,
2026-09-19). Adding it moves the transcribed route from **+0.830702** to
**+0.827448** on ultrasoft AlAs at 64 k-points, where the differentiated route
reads **+0.815802**: the right sign, **21.8 per cent** of the gap, and the
disagreement falls from 1.79 to **1.41 per cent**. So it is one piece of what
`zstar_eu_us.f90` adds and not all of it, and the refusal stays.

**Both terms are in both routes now, and the measurement separates two things
that had been one** (`20339831`, ultrasoft AlAs, the whole `4 4 4` grid, 64
k-points):

| route | without the constraint terms | with them | shift |
|---|---|---|---|
| taped, `clamped_ion_piezoelectric` | +0.815802 | **+0.821330452** | +0.005528452 |
| contracted, `zstar_eu` | +0.830702 | **+0.836230910** | +0.005528910 |

**The two shifts agree to 4.6e-07**, which is the result worth having: the
constraint terms are implemented twice, once by handing the multipliers to the
functional as a third tangent and once by contracting `dLambda` and the
`add_for_charges` sandwich by hand, and the two independent implementations of
the same physics move their routes by the same amount. That is the check this
project asks for, and it passed.

**And the 1.8 per cent is untouched**: the gap was 0.014900000 and is now
0.014900458, a change of 4.6e-07. So the disagreement between the two routes
was never the constraint terms, and the terms were a real omission in both --
two separate facts that the single number 1.8 per cent had been standing in
for.

**What the gap is instead, and the record already contains it.** The two routes
differ in one more place: the taped one hands the density and `becsum` to the
energy as **builders that carry the strain**
(:func:`~defumat.response.born._raw_mixed_state`), while the contracted one
rebuilds the potential from the **frozen density array** inside
`_bare_strains`. For an ultrasoft dataset that is not the same "bare", because
the density itself moves with a strain -- the augmentation charge deforms and
the stored field carries a `1/Omega`. `strain.py`'s `StrainResponse.moved_drho`
is exactly this object and its docstring names the trap: "a consumer that hands
the mixed state to an energy as a function of the strain generates this half
itself and has to subtract it; one that freezes the density as an array does
not." **That is P39, and it is the candidate now.** Which of the two
conventions is right is what the Berry-phase value decides, and it is
+0.692986.

**The gap is closed, and the cell it was closed on is half the finding**
(2026-09-19, this workstation, no cluster job at all). The candidate above was
right, and the thing that made it cheap to test is a sentence that had not been
written down: **two assemblies of the same mixed second derivative must agree at
any cutoff and on any mesh**, so their disagreement is an assembly defect and
not a convergence question, and it can be reproduced on a cell chosen for cost.
`tests/data/qe/alas-piezo-tiny.in` is that cell -- the same zincblende AlAs and
the same ultrasoft datasets at `ecutwfc = 10`, `ecutrho = 44`, `nosym` on an
unshifted `2 2 2` grid, 8 k-points -- and it reproduces the defect at **1.6 per
cent** where `alas-piezo.in` at 64 points reads 1.8. The 139.6 GiB job was never
needed to find this.

| | taped | contracted | gap |
|---|---|---|---|
| `alas-piezo-tiny.in`, `ecutwfc = 12`, before | +1.473304405 | +1.497082023 | **-0.023777619** |
| the candidate term, measured on its own | | | **-0.023777621** |
| after, same cell | +1.473304405 | +1.473304408 | **2.6e-09** |

**The term, and it is not about the constraint at all.** The transcribed route
contracts the field response against `strain._bare_strains`, which rebuilds the
potential from the converged density **array** at the deformed cell, so what it
differentiates is `dH/d(eps)` at frozen `rho`. The derivative the tensor is is
taken at frozen *states*, and the density is a function of the states **and** of
the cell, so the chain rule has one more link, `K . (drho/d(eps))|_psi`.
Contracted with the field response and using that `K` is symmetric, that is
`-(1/Omega) int dV_scf^(E_k) [drho/d(eps_ab)]_psi`, a mean over the grid with no
factor of two and no volume left in it. `clamped_ion_piezoelectric` needs none
of it because it hands the density to the energy as a *builder* that carries the
strain, so its `jvp` generates the link itself.
`defumat.response.piezo._screened_strain_term` is the term and
`StrainResponse.moved_drho` is the object it integrates, which is what the
entry above predicted.

**What the norm-conserving agreement was worth, which is less than it looked.**
At frozen plane-wave coefficients the smooth density in crystal coordinates does
not move under a strain at all -- the exponentials are indexed by integers -- so
`[drho/d(eps_ab)]_psi` is `-delta_ab rho` exactly and **vanishes for every
traceless strain**. A zincblende crystal's only independent component is the
shear `e_14`, so the calibration cell could not have seen this term any more
than it could see the two constraint terms, and all three were missing while the
two routes agreed to 6.2e-15. Measured after the fix on `alas-raman.in`: the
routes agree to **1.9e-14** and `e_14` is **-0.763786071** in every digit it had
before. The trace part is not zero on a cell that has one, and it contracts with
the field's induced potential to a *vector*, which a non-polar class has none of
-- so on this crystal every component of the term but the shear reads 1e-15,
which is a symmetry rather than a tolerance.

**What this changes about the refusals, and it is only one of them.**
`require_a_norm_conserving_transcription` now refuses **PAW alone**: PAW's
one-centre energy is a function of `becsum` directly, so its cross term with the
field's `dbecsum` is on no grid and this term does not reach it, where an
ultrasoft dataset's `becsum` reaches the energy only through the augmentation
charge that *is* on the dense grid. `require_a_measured_dataset`, the refusal
about the quantity itself, is untouched and stays: nothing here compares against
an independent reference, and the Berry-phase value is still the only thing that
would.

**And the practical consequence is the one worth acting on.** The cheap route is
now complete on an ultrasoft dataset, and it costs **2.6 MB a k-point** against
the taped route's 16 MB of tape alone -- 2.32 GiB against 139.6 at 64 k-points.
The ultrasoft k-ladder that was sized as "a whole node" is now a workstation
job, which is what the entry above needs to separate the 1.65-point difference
of deficits into a dataset effect and a difference of mesh errors. That ladder
is the next measurement and it no longer needs the cluster to take it.

**One sentence of the entry above is retracted.** "Which of the two conventions
is right is what the Berry-phase value decides, and it is +0.692986" is wrong,
and it is the forecast-sentence trap in `CLAUDE.md`: a Berry value 15 per cent
away on k-convergence grounds cannot adjudicate a 1.8 per cent gap between two
routes, and what settled it was an identity that needs no reference at all.

**And the record's own cell says it too, on this workstation, in 19 minutes.**
The tiny cell establishes an identity and not a number, so the last step is to
read the same identity on `alas-piezo.in` at the mesh every entry above is
about. The contracted route on the whole `4 4 4` grid, 64 k-points, `--nosym`,
now reads **+0.8213303507** against the taped route's **+0.821330452** from
Triton `20339831`: **1.0e-07 apart**, where before the two terms it was
0.0149. That is a cross-machine, cross-route agreement on the cell the refusal
is written about, and it cost **3.6 GiB and 19m04** against the taped route's
139.6 GiB on a node.

**The wedge completes, and it is the taped route that does not.** The screened
term is a product of two k-sums, which is the trap that makes a rank-3
symmetrisation of the *result* insufficient unless one factor is already the
full-zone object -- and it is: `dielectric_tensor` mixes its `dvscf` from the
*symmetrised* `drho` (`efield.py`, `symmetrize_directional` then `screen`), so
the value inside the term is full-zone while the `moved_drho` factor it
multiplies stays the raw wedge sum, which is exactly the arrangement P36 asks
for on a term *linear* in the derivative factor. Measured rather than argued,
two runs on this machine: the contracted route's symmetrised **8-point wedge
reproduces the whole 64-point grid to 1.6e-06** on a value of 0.82, 2 parts in
a million, for 3.3 GiB and 6m28. The taped route's own recorded pair is
**+0.815929 against +0.815802, 1.3e-04 apart**, 78 times larger, and that is
the `_full_zone_field_response` absence the entry above named as the third of
the three terms `born_effective_charges` carries and the piezoelectric tape does
not. **So the cheap route is the better-behaved one on a reduced k-set**, which
is the opposite of what it is usually reached for, and the taped route's wedge
is the one open question left in this pair.

**The taped route's wedge is closed too, and it was the third of the three
terms.** The pair above left one thing open -- the taped route's own wedge, at
1.3e-04 against its closed grid where the contracted route was at 1.6e-06 -- and
it is `_full_zone_field_response`, the term
`born_effective_charges` has carried since 2026-09-16 and this tape did not.
**The same trap one coordinate over**: on an augmented dataset the density moves
with the strain at frozen states, so the mixed derivative carries
`int (drho/d(eps)) K (drho/dE)`, a product of *two* per-k tangents, and a wedge
sum of a product is not the product of the full-zone objects -- so no average of
the finished rank-3 tensor repairs it and one factor has to be made whole before
it is contracted. P36's rule, and the same choice of factor: the field response,
because an induced charge density is a polar vector field and `symdvscf`'s
average is already written for it.

A second committed cell was what it took to see it, and the cell is three
k-points. `alas-piezo-tiny-wedge.in` is `alas-piezo-tiny.in` with the point
group kept, so the same unshifted `2 2 2` sample reduces from 8 points to 3:

| route | wedge, before | wedge, after | closed grid |
|---|---|---|---|
| taped | 1.475427270 | **1.474382133** | 1.474377366 |
| contracted | 1.474382167 | 1.474382167 | 1.474377366 |

The two were **1.05e-03** apart on the wedge and are now **3.4e-08**, and both
now sit 4.8e-06 from the closed grid, **and the sentence first written here
about that number was wrong and is retracted**. It said the 4.8e-06 was the
residue the rank-3 average leaves on the factor that stays a raw wedge sum,
which predicts zero: the average of a term *linear* in a covariant per-k factor
is exact, which is P36's rule and what the norm-conserving cell confirms at
4.5e-09. It is inherited from below instead. **The dielectric constant itself
splits between the same two cells by 1.573e-05 relative**, 6.634e-04 on 42.16,
five times the piezoelectric tensor's 3.26e-06, so the strain leg adds nothing
measurable and there is no residue here to explain. Where that 1.573e-05 comes
from is a question about the *field response* on an ultrasoft wedge, it is not
this phase's, and it is opened as its own item with the two discriminators
already run.. **The closed-grid numbers did
not move by a bit**, which is the check the construction asks for: the shift is
`symmetrize_directional(raw) - raw` and that is identically zero on a `nosym`
run, so `alas-piezo-tiny.in` reads 1.474377366 and a two-route gap of
1.7135834085024726e-07 before and after, digit for digit. The norm-conserving
cell is likewise unmoved at -0.7637860707 and 1.93043553006969e-14.

**Why the contracted route never needed it**, which is the part worth keeping:
its screening factor is the field's converged `dvscf`, and `dielectric_tensor`
mixes that from the *symmetrised* density response, so the value inside the term
is full-zone already while the `moved_drho` it multiplies stays the raw wedge
sum -- exactly the arrangement P36 asks for. That is not a virtue of the
transcription, it is an accident of which object the response loop happens to
hand back, and it is worth saying because the taped route looks like the safer
one and on a reduced k-set it was not.

**And one branch went unreachable in the fixing.** `_field_column`'s
`multipliers`/`ground` arguments were the augmented path until the augmented
path needed a third and a fourth argument; the loop moved into
`clamped_ion_piezoelectric` and the branch stayed, dead, with a docstring still
calling it the augmented case. It is removed, which is the same drift the eight
doc-drift entries above were, arrived at from the other direction: not a
sentence that went stale beside live code, but live prose beside code that had
stopped running.

**A correction, and it is the largest thing this entry found.** Earlier today
this said "+0.815802 is the number and +0.830702 is the artefact", on the
grounds that the differentiated assembly run in the *position* coordinate is
the Born charge and that matches `ph.x` on ultrasoft AlAs. **That is the wrong
function.** What matches `ph.x` is
`defumat.response.born.born_effective_charges`, which carries three terms the
piezoelectric assembly does not: the multipliers' own response, the
`add_for_charges` sandwich (`constraint_position_term`), and the full-zone
density shift. The piezoelectric route that *is* built from the same tape,
`born_charges_from_stress_route`, says so in its own docstring -- in the
position coordinate it is "``Z*`` minus its bare ionic term and minus the
constraint term an ultrasoft dataset adds". Checked at the source rather than
inferred: `multipliers`, `constraint_position_term`, `commutator` and
`_full_zone` appear nowhere in `_frozen_energy_of`, `_field_column` or
`clamped_ion_piezoelectric`.

**So on an augmented dataset neither route is complete, and the 1.8 per cent
was two incomplete assemblies disagreeing rather than one being wrong.** The
term added today is real, is worth -0.00325 of the 0.0149, and belongs in
*both* routes rather than only in the transcribed one -- which is why it closed
a fifth of a gap it was never going to close, since it was aimed at a target
that is itself missing the same physics.

**And it falsifies a sentence in `require_a_measured_dataset`'s own docstring,
by measurement.** That said "`qq_ij` has no cell in it, so the constraint stays
strain-independent for an ultrasoft dataset exactly as it is for a
norm-conserving one". `qq_ij` has no cell in it and the conclusion does not
follow: `S = 1 + sum |beta> q <beta|` also carries `vkb`, which is
`beta(|k+G|)` and moves with the cell like every other radial transform. A term
measured at -0.00325 C/m^2 is not a term that vanishes. The docstring is
corrected.

**What this changes about the refusal.** It stops being a formality waiting on
a number and becomes the right answer: the piezoelectric tensor of an ultrasoft
or PAW dataset needs the constraint terms in whichever route computes it, and
the only reference here that does not need them is the Berry-phase finite
difference, +0.692986 at 11 strings over 6x6. That is now the target, and
+0.815802 is not.

**The correction is close to mesh-independent, which is worth one line.** At
`6 6 6` it takes the transcribed route from +0.730408 to **+0.727596**, a shift
of -0.00281 against -0.00325 at `4 4 4`, for 2.31 and 3.40 GiB. So this piece
of the missing term is very nearly a constant offset rather than something that
grows with the k-sampling. That is evidence about *this* piece and not about the
other one, and the temptation it feeds -- that the whole missing term is a
constant, so the transcribed route's convergence *curve* can stand in for the
differentiated route's -- stays refused, because the remaining piece's mesh
dependence has not been measured at all.

**It was a derivation and not a copy.** The index order
is the trap and is flagged in `_multiplier_response`'s own docstring: `Lambda_mn`
pairs with `<psi_n|S|psi_m>`, the weight belongs to the *column*, and
transposing it costs 0.28 on ultrasoft silicon while costing nothing at all on
a norm-conserving cell -- the kind of error the regression gate cannot see.
What makes it worth doing anyway is that the validation is already set up and
sharp: the target is **+0.815802** on `alas-piezo.in` at 64 k-points, the floor
is **6.2e-15** on the calibration cell where the new term must stay identically
zero, and the job that checks it costs 2.3 GiB and six minutes.

**The memory side of the same problem, sized without running anything.** Compiling one field column of the taped
route at four k-counts and never executing it
(`jit(...).lower(...).compile().memory_analysis()`, P73's instrument) gives
**0.630, 3.679, 8.166 and 15.724 GiB** of temporaries at 64, 216, 512 and 1000
k-points on the norm-conserving cell: **16 MB a k-point, essentially no fixed
part**. The dial cannot touch that, because `forces/energy.py:energy_at` has no
`map_k` or `sum_k` in it at all.

**And the cheap lever was tried and is worth nothing**, which is worth writing
down so nobody spends a day on it: `jax.checkpoint` around the strained energy,
and again with `policy=nothing_saveable`, leaves the tape at 0.630 and 3.679 GiB
unchanged to the byte. `jvp(grad(f))` of a function with no internal loop has
nothing to trade. The lever that would work is a `lax.scan` over k **inside**
`energy_at` with a rematted body, which is P73's fix for the augmentation table
and is a change to the function every force and stress in the package goes
through -- so it is sized here rather than taken.

**The prize is large and measured.** The transcribed route costs **2.32 GiB at
64 k-points and 2.63 at 216** against the taped route's 139.6 at 64, a factor of
60, and it barely grows with `nk`. Either adding `zstar_eu_us.f90`'s missing
term to it, or the `lax.scan`, turns the ultrasoft ladder from a whole-node job
into a small one. Third, the refusal *text* in `response/piezo.py:273` still
names `response/strain.py` as refusing the same datasets, which P41 measured to
be untrue, and keeping the refusal is not a reason to keep the wrong cause in
its message.

**What has been ruled out, so nobody chases it.** The step size (doubling the
shear moves the Berry value by 0.3 per cent, and away from the response route);
the sign difference between the two committed AlAs cells, which is the
**enantiomorph**, one file writing the As position in `alat` and the other in
`crystal`, so that for `ibrav = 2` the triple `(1/4, 1/4, 1/4)` is
`a(-1/4, 1/4, 1/4)`; and the harness's contraction of both strained cells'
phases with one cell's lattice vectors, which is exact for `e_14` because
`(S a_g)_x = 0` for a pure `y`-`z` shear, the same statement that makes `e_14`
free of the proper-against-improper correction and of the polarization branch.

**The three regression files pass on a node, and until now they had never been
run in one process anywhere.** Job `20345288`, an array of three tasks on
`milan3` under `batch-milan` at `c9bbfb1`, four cores and 32 G a task with
`DEFUMAT_CACHE_DIR=off`, so every kernel was compiled once: **14 tests passed
and none failed**, 3 in `test_piezoelectric_augmented.py` at 335.97 s, 1 in
`test_piezoelectric_wedge.py` at 198.74 s and 10 in `test_piezoelectric.py` at
563.03 s, with peak resident sets of **10.5, 8.0 and 5.2 GiB**
(`/usr/bin/time`'s `ru_maxrss`, and `sacct` agrees at 10.6, 8.1 and 5.2). The
JAX underneath is 0.11.1 and NumPy 2.5.2 against this workstation's 0.11.0 and
2.4.6, which is the pair that moved eight other assertions by 1e-12 to 1e-7,
and it moves none of these past a bound. Every tolerance tighter than 1e-07 in
the three files compares two things computed in the **same** process, which an
environment shifts together: 1e-12 between the transcribed and the
differentiated tensor, a relative 1e-06 between the two routes on a wedge, and
1e-10 on the components symmetry forbids. What is quoted across machines is
quoted loosely, and there are four of them: the closed-grid `e_14` of
1.474377366 at an absolute 1e-04, the screened term's -0.0227272 at a relative
1e-03, `ph.x`'s `Z*` of (1.92461, -3.18098) at 5e-04 and `epsilon` of 12.9674 at
1e-03. So what this array establishes is that the assertions hold on the node;
it is **not** a digit-for-digit comparison of the two machines, which nothing
here has taken.

**The split into three files was not optional and the peaks say so.** The
augmented file alone resides 10.5 GiB against `tools/run_regression.sh`'s 12 GiB
cap, so a second augmented cell in the same process has 1.5 GiB to live in,
which is the killed run the wedge file's own docstring describes. Cold, the
three cost 5m36, 3m19 and 9m23 against the workstation's warm 5m24, 1m55 and
5m00, a pair of numbers and not a cache cost: the machine and the cache state
both change between them and neither is isolated, and the shape of the three
argues against compilation as the explanation, since the file with the most to
compile is the one that barely moves, 5m36 against 5m24, while the two lighter
ones read 1.7 and 1.9 times. Isolating it needs a warm rerun on `milan3`, which
is one resubmission and has not been taken. The in-process watchdog was off for the
whole array, because `psutil` is not importable in the cluster venv and
`conftest.py` says so at collection time rather than silently, so the peaks
above come from `/usr/bin/time` around pytest and not from the watchdog, and no
test is named as the one that held them.

**One defect in the job script was found by reading it rather than by running
it** (`c9bbfb1`). `piezo_regression.sbatch` took its `REPO` line from
`regression.sbatch`, whose default is `/scratch/work/ladovj1/apps/defumat-audit`,
the *old* side of `attribute.sbatch`'s A/B sitting at `b247662`. Two of the
three files were written after that commit, so the array would have failed on a
missing path in two tasks out of three and the third would have tested the code
the session started from. Its sibling jobs that day all used `defumat-jobs` and
that is the default now.

**The PAW half of the completion had never run, and it does now**
(2026-09-19, this workstation). ``_full_zone_becsum_response`` is the half of
the wedge completion that lives in the one-centre terms, where ``becsum``
reaches a PAW energy directly rather than through the augmentation charge on the
dense grid, and it had been live in the strain coordinate since the completion
went in with nothing to exercise it: the tensor refuses PAW at the door, every
committed PAW crystal was centrosymmetric, and a centrosymmetric tensor is zero
whatever the assembly does. So the case was built. ``Al.pbe-n-kjpaw_psl.1.0.0``
and ``As.pbe-n-kjpaw_psl.1.0.0`` are committed beside the ultrasoft pair, and
``alas-piezo-tiny-paw.in`` is the ultrasoft tiny cell with the dataset kind
changed and nothing else -- same geometry, same ``celldm``, same ``10/44``, same
unshifted ``2 2 2`` -- which converges to **-215.3977782953 Ry** with a
dielectric constant of **42.051** against the ultrasoft cell's 42.160, so the
cheap cutoff has not made it a different problem.

| ``e_14``, C/m^2 | wedge, 3 points | closed grid, 8 points | apart |
|---|---|---|---|
| with the completion | **1.465023765** | 1.465022183 | **1.58e-06** |
| without it | 1.465718152 | 1.465022183 | 6.96e-04 |

**A factor of 440**, and the switch that produced the second row is the point:
``clamped_ion_piezoelectric`` gained ``full_zone``, whose only purpose is to be
turned off in a test, because a completion that is *identically zero* on a
``nosym`` run cannot be told from a deleted one by any test that never runs
without it. On the closed grid the two rows are equal digit for digit, which is
the other half of that statement. The file is
``tests/regression/test_piezoelectric_paw.py`` and it is the **fourth cluster
file**: one cell and two tapes peak at 14.9 GB in one process against the
runner's 12 GiB cap, with no accumulation to remove, so it joins the three in
``tools/cluster/piezo_regression.sbatch`` rather than being run here.

**The ultrasoft ladder was taken, on this workstation, and it separates the
1.65 points the entry above could not attribute** (2026-09-19). The contracted
route costs 2.3 GiB where the taped one costs 139.6, and the rungs are the
irreducible **wedge** of each unshifted grid rather than the whole of it, which
is the same sample for this route at 1.6e-06 and is what turns 216 k-points into
16. Three rungs, 3.3 GiB and six to ten minutes each, against the
norm-conserving ladder already in the record:

| ``e_14``, C/m^2 | ``4 4 4`` | ``6 6 6`` | ``8 8 8`` | ``10 10 10`` |
|---|---|---|---|---|
| ultrasoft, contracted route, wedge | **0.821332** (8 k) | **0.722701** (16 k) | **0.701828** (29 k) | **0.696932** (47 k) |
| norm-conserving, taped route, whole grid | 0.763786 | 0.687475 | 0.672897 | 0.669907 |
| deficit against the Berry value, ultrasoft | 15.63 % | 4.11 % | 1.26 % | **0.57 %** |
| deficit against the Berry value, norm-conserving | 13.33 % | 3.71 % | 1.62 % | 1.19 % |

**The difference of deficits collapses from 2.30 points to -0.62 and changes
sign**, so the 1.65 points was a difference of *mesh errors* and not a dataset
effect: at ``10 10 10`` the ultrasoft response sits **0.57 per cent** from its
Berry value where the norm-conserving calibration, which is the validated route,
sits at **1.19 per cent** on the same mesh. An augmented dataset is no further
from an independent reference than the calibration is -- it is nearer -- which is
the thing the entry wanted and could not get from one mesh. The four rungs cost
370, 434, 599 and 643 s at 3.3, 3.3, 3.2 and 3.5 GiB, so the whole ladder is
half an hour on a workstation where one rung of it was sized as a node.

**That is the measurement ``require_a_measured_dataset`` names as the thing that
would lift it**: "the ultrasoft tensor at a converged mesh against a Berry-phase
value on the same cell, and the cell is committed". The 0.57 per cent is inside
the Berry value's own 0.7 per cent spread over four string meshes, so what is
established is agreement to the reference's own resolution rather than a tighter
number.

**Two things this does not say.** The Berry values themselves move 0.7 per cent
over four string meshes, and both deficits at ``8 8 8`` are inside twice that,
so what is established is that the two datasets converge alike rather than a
number for either. And the two ladders are different routes on different k-sets
-- the taped route on whole grids for the norm-conserving cell, the contracted
one on wedges for the ultrasoft -- which is licensed by two measured identities
(the routes agree to 1.9e-14 norm-conserving and 1.0e-07 ultrasoft, and the
ultrasoft wedge reproduces its closed grid to 1.6e-06) and is stated rather than
left to be noticed.

**The guard is in, and the instrument that replaces it with a number agrees
with the one that found the problem** (2026-09-19). ``piezoelectric_tensor``
carries ``nk``, ``grid`` and ``kmesh_drift`` on its result and warns that
nothing inside this quantity can see its own zone sum, quoting the AlAs curve;
``defumat/workflows/piezo_ladder.py`` measures the curve of whatever crystal is
in front of it, one ground state and one response per rung, wedges by default.
Run on the calibration cell at ``4 4 4`` and ``6 6 6`` it reads **-0.763786**
and **-0.687475** C/m^2, which is ``tools/cluster/piezo_measure.py``'s ladder
digit for digit and the same pair the guide's table quotes, so the new
instrument is the old one's answer by a different route -- 73 s and 191 s, 14.4
GB, the taped route's tape at 216 k-points being all of it.

**What silences the warning is a step and not an error**, and the two are
measured apart: the steps on this crystal are 10, 2.1 and 0.44 per cent while
the densest rung is still 1.2 per cent from the Berry value, so the remaining
distance runs about three times the last step and ``KMESH_STEP = 0.01`` is a
place to stop warning rather than a claim of convergence. Both docstrings say
so. The falsifier for the guard itself is in the fast set
(``test_piezo_machinery.py``): the warning must fire with no ladder, must quote
the measured drift when a ladder ran above the threshold, and must be silent
below it, which is three states rather than the one a "does it warn" test would
have checked.

**The refusal is lifted for ultrasoft above the mesh it was measured at, and
kept whole for PAW** (2026-09-20). ``require_a_measured_dataset`` was one
refusal about two dataset kinds and is now two conditions: PAW raises whatever
the mesh, and an ultrasoft dataset raises below :data:`ULTRASOFT_MESH`
divisions in each direction. **Eight is where the ultrasoft deficit first falls
below the norm-conserving calibration's on the same mesh** -- 1.26 against 1.62
at ``8 8 8``, 0.57 against 1.19 at ``10 10 10`` -- which is the statement that
the dataset has stopped being the largest error and the mesh has taken over,
and the mesh has a warning of its own. Below eight the two are not separable at
all, the same cell reading 15.6 per cent out at ``4 4 4``, so a coarse
ultrasoft run is refused where a coarse norm-conserving one is only warned
about. A **measured** ladder overrides the threshold: ``kmesh_drift`` below
``KMESH_STEP`` is evidence where a division count is a guess generalized from
one cubic crystal, and the ladder's own coarse rungs are exempt by
``allow_a_coarse_mesh`` because they are what does the measuring.

**Why PAW keeps all of it**, and it is about the kind of evidence rather than
the amount: everything measured for PAW here is *internal*, the wedge
completion against its own closed grid, and an identity between two k-sets of
one calculation says nothing about whether the calculation is right. No PAW
crystal has been compared with anything outside this code. The test file ran on
a node and passed there (``20347768_3``, ``batch-milan``, **2 passed in 4m32**
at a peak of 12.1 GB, which is 180 MB under the workstation runner's cap and is
why it lives in the cluster array).

## 5. An **ultrasoft** dielectric constant does not reproduce its own closed grid from a symmetry-reduced k-set, by 1.6e-05 relative **[opened 2026-09-19, found in passing; attributed to the dataset under control the same day, six candidates excluded, term not found]**

An unshifted Monkhorst-Pack grid is closed under the point group, so a
symmetrised wedge and the whole grid are the same k-sample by two routes and
must give the same answer to round-off. On a **norm-conserving** cell they do:
`response/efield.py`'s own docstring records a spinor silicon wedge reproducing
the closed 64-point grid's tensor to **7.4e-13**, and the scalar pair to 4.1e-13.
On an **ultrasoft** cell they do not.

Measured on the two cells committed on 2026-09-19 for the piezoelectric
identity, which are the same crystal, the same datasets, the same cutoffs and
the same unshifted `2 2 2` sample, differing only in whether the point group is
kept -- 8 points against 3:

| quantity | closed grid | wedge minus closed | relative |
|---|---|---|---|
| `epsilon` | 42.15974622025 | 6.634e-04 | **1.573e-05** |
| the converged `drho` | max 8.614e-01 | 8.970e-06 | 1.041e-05 |
| the converged `dvscf` | max 5.143e+00 | 3.763e-04 | 7.315e-05 |

**Two things are already ruled out and they are what makes this worth opening
rather than noting.** It is **not the solver's convergence**: rerun with `tr2`
at 1e-18 and the inner `threshold` at 1e-14, six orders tighter, and the
`epsilon` split is **6.634e-04 to five digits, unchanged**. And it is **not the
assembly above the loop**: the split is already in the converged induced density
`drho`, so whatever is incomplete is inside the self-consistent response
iteration rather than in the contraction that turns it into a tensor. The two
ground states agree to 3e-11 Ry, so it is not the SCF either.

**What it is not, as far as the obvious suspect goes.**
`_symmetrize_becsum_response` is a no-op for a non-PAW dataset, which looks like
the answer and is probably not: `SternheimerSolver.response_density` builds the
augmentation charge's own response into `drho` before the loop symmetrises it
(`sum_ij Q_ij(r) dbecsum_ij`, its docstring's third bullet), and
`becsum_response` is carried separately only for PAW's one-centre potential,
which an ultrasoft run does not have. That was read rather than tested, so it is
a suspect discharged on an argument and not a measurement, which this file
usually refuses -- it is written down so the next person does not spend the
afternoon on it first.

**What it costs and what it touches.** 1.6e-05 relative is far below anything
currently claimed against `ph.x` or Elk, so nothing in the record is wrong
because of it. What it does set is a **floor** under every ultrasoft quantity
taken on a reduced k-set: the piezoelectric tensor's two routes agree to 3.4e-08
on the same wedge and sit 4.8e-06 from their own closed grid, which is this and
not theirs. Anything aiming below 1e-05 on an augmented wedge should run
`nosym` until this is understood.

**It is the dataset and not the mesh, and the first version of this sentence
varied both at once.** It read "the same comparison on `alas-raman.in` gives
3.545e-10 against the ultrasoft cell's 1.573e-05, a factor of 4.4e04" -- true,
and taken from a norm-conserving cell on a `4 4 4` grid reducing 64 points to 8
against an ultrasoft cell on a `2 2 2` reducing 8 to 3. Two variables, one
conclusion. The controlled version is the same crystal at the **same** cutoff on
the **same** `2 2 2` grid with the **same** 8-to-3 reduction, changing only the
dataset, and it says the same thing more sharply: **3.625e-14** norm-conserving
against **1.251e-05** ultrasoft, in the response density at the first pass.

**It is also not the cutoff, which is the other thing the cheap cell invites.**
`ecutrho = 44` on an ultrasoft dataset is far below what those datasets want, so
the augmentation charge is under-resolved on purpose. Raising it does not make
this go away: the split reads **1.251e-05** at `ecutrho = 44`, **5.323e-06** at
80 and **7.825e-06** at 160, so it falls by a factor of two and then stops and
turns round. A plateau at 5 to 8e-06 is not a discretisation artefact.

**Six things it is not, each tested rather than argued.** The split is at the
**first pass** of the response loop, 1.251e-05 with `dvscf = 0`, so everything
below is about one Sternheimer solve and one symmetrised density sum with
identical input:

* not the **symmetriser**: applying the wedge cell's `symmetrize_directional` to
  the closed-grid field, which is already full-zone and therefore invariant,
  moves it by **1.772e-14**;
* not the **ground state**: the two SCFs agree to 3.197e-14 Ry and their
  densities to 8.6e-07, an order below the response split;
* not the **solver's tolerance**: `tr2 = 1e-18` and `threshold = 1e-14`, six
  orders tighter, leave `epsilon`'s split at 6.634e-04 to five digits;
* not **`adddvepsi_us`**, the ultrasoft tail of the bare perturbation
  (`_ultrasoft_position`): disabled, the split is 1.213e-05;
* not **`dS/dk`** inside the commutator, the other ultrasoft term in the bare
  perturbation: zeroed, 1.254e-05;
* not the **augmentation charge inside `drho`**: dropped from the response
  density altogether, 1.051e-05.

**What that leaves**, and it is written as a suspect rather than a finding: the
only ultrasoft-specific machinery still in the first pass once those are out is
`S` itself -- inside the Sternheimer operator, inside the projector `P_c`, and
inside `refined_states`' generalised diagonalisation. The symmetriser being
exact means the *raw per-k* `drho` is not covariant, `drho_a(Rk) != R_ab
drho_b(k)` as a real-space field, and the three terms above are not where the
non-covariance is. `_symmetrize_becsum_response` being a no-op for a non-PAW
dataset was the first suspect and is discharged twice over: `becsum_response` is
collected only when PAW's one-centre potential exists, and the split survives
with the augmentation removed from the density entirely.

**Nothing above is a fix and this entry does not claim one.** What it claims is
that the hypothesis space is much smaller than it was and that six of the
obvious answers have been paid for.

## 4. Eight cluster test failures are assertions, none of them is this session's work, and one is a stale test rather than the environment **[opened 2026-09-19, attributed the same day]**

The whole `slow` set on a node (job `20336106`) failed in thirteen files, and
item 2 above accounts for eight of them. These are the rest. **They have now
been run at two commits in one environment** (job `20337788`, tasks 0 and 1,
`batch-milan`, `DEFUMAT_CACHE_DIR=off`): task 0 at **`b247662`**, the commit the
session started from, and task 1 at **`4773c75`**, its head. Neither
`tools/run_regression.sh` nor `tests/conftest.py` nor any of the seven files
changed between the two, so the A/B isolates package code. **The failure counts
are identical, file by file, at both commits**, so none of these is the
session's work:

| file and test | what it says | old | new |
|---|---|---|---|
| `test_relaxed_anisotropy.py::test_without_spin_orbit_coupling_every_direction_has_the_same_energy` | two directions differ by **2.545e-02 meV** (1.9e-6 Ry) on an exact identity | 1 | 1 |
| `test_magnetic_constraints.py::test_constrained_total_energy[noncolin-constrain_atomic.in-atomic]` and `::test_constraint_energy_matches_qe` | `-55.690556438787105` against `-55.69055687` ± 3.0e-07, so 4.3e-7, a factor 1.4 over | 2 | 2 |
| `test_ten_site.py::test_dft_plus_u_at_ten_sites` | the SCF stalls: **not converged after 200 iterations**, accuracy 6.9e-06 Ry, `E = -856.58298639`, `M = 5.9035 mu_B` | 1 | 1 |
| `test_stm.py::test_an_antiferromagnet_is_flat_in_charge_and_alternates_in_spin` | `0.029525447994362755` against `0.02952503643369007` ± 3.0e-07, so 4.1e-7, a factor 1.4 over | 1 | 1 |
| `test_stress.py::test_an_input_asking_for_an_impossible_stress_warns_rather_than_raising` | `DID NOT WARN`, `Emitted warnings: []` -- **repaired in `2dc42e1`** | 1 | 1 |
| `test_lsda_response.py::test_the_polarized_dielectric_constant_reduces_to_the_unpolarized_one` | `2.708606672285896e-07 < 1e-08`, a factor 27 over | 1 | 1 |
| `test_spinor_dielectric.py::test_a_spinor_with_no_magnetization_gives_the_scalar_dielectric_tensor` | `13.80661565177345` against `13.806615651772065` ± 1.0e-12, so 1.4e-12 on 13.8 | 1 | 1 |

**One of them is not the environment at all, and the repository already knew
half of it.** `test_stress.py::test_an_input_asking_for_an_impossible_stress_...`
asserts a refusal that **P46 lifted**. The test takes noncollinear
`h-chain-90deg.in`, turns `tstress` on, and expects `run_scf` to warn and leave
`SCFResult.stress` as `None`; that warning is raised in exactly one place
(`scf/driver.py:6027`) and only when `compute_stress` raises
`NotImplementedError`. It no longer does: `stress/autodiff.py:90` passes
`spinors=True` unconditionally, so `reject_spinors` is never reached on the
autodiff path, and `forces/energy.py:229-238` says so in its own docstring, that
the force and the stress run for `noncolin = .true.` with or without
`lspinorb`. The node emitted **no warnings of any kind**, which is what a
successful `compute_stress` looks like from outside. `PLAN.md` P74 recorded the
same failure on a **workstation** worktree at `e22aa7d`, same empty
`Emitted warnings: []`, and concluded it was pre-existing and open; the part
that was missing is the cause, and this is it. **Repaired in `2dc42e1`**: the
test is about the switching-off rather than about the regime, so it now runs
`h-chain-spiral.in`, which `require_a_differentiable_cell` still refuses by
name, and the spinor stress stays covered by
`test_spinor_forces.py::test_stress_matches_quantum_espresso` against `pw.x`.
It also means the statement that "every one of these tests passes on the
workstation" was false for this one and was written without a measurement
behind it.

**What the remaining seven look like, without naming a cause.** Five are
tolerance misses and four of those are within a factor of 30 of their bound:
1.4x twice on `test_magnetic_constraints`, 1.4x on `test_stm`, 27x on
`test_lsda_response`, and `test_spinor_dielectric`'s 1.4e-12 on 13.8, which is a
**relative 1e-13** against an *absolute* 1e-12 assertion and is therefore a
tolerance written in the wrong units rather than a defect. The sixth,
`test_relaxed_anisotropy`, is an identity that should hold exactly and misses by
1.9e-6 Ry. The seventh, `test_ten_site`, is not a tolerance at all: a ten-atom
DFT+U nickel supercell that does not converge in 200 iterations, sitting at
6.9e-06 Ry, and it is the most expensive test in the set at 54 minutes and 17 GB.

**What is measured and what is not.** Measured: the same eight failures at
`b247662` and at `4773c75`, on `batch-milan` under JAX 0.11.1 and NumPy 2.5.2,
against the workstation's 0.11.0 and 2.4.6. **Not measured**: that any of the
other seven passes on the workstation at these commits. The previous version of
this item asserted that, and for `test_stress` it is now known to be wrong, so
the rest of the claim is withdrawn until a workstation run is in the record.
What the A/B does establish is the thing it was built for, which is that none of
these is attributable to the session, and the remaining question is environment
against pre-existing rather than environment against session.


# Part XIV -- measured and left open, 2026-09-20

## 1. `dH/dk` at a reciprocal-lattice point is wrong on every `l = 1` channel **[closed 2026-09-20, route A: the ratio goes 0.3695 to 1.0000 and no primal moves]**

`AUDIT-2026-09-20.md`'s `defumat/basis/gvectors.py:117`, reproduced and
measured rather than fixed, because the repair is a choice between two routes
that differ in kind (below).

**The mechanism.** A projector column is
`real_spherical_harmonics(kg, lmax)` times `projector_form_factors(p, modulus(kg))`
(`pseudo/projectors.py:_angular_part`), and **both factors guard the origin by
zeroing**: `modulus` returns `jnp.where(norm2 > _TINY, sqrt(...), 0.0)`, whose
tangent on the false branch is 0, and `real_spherical_harmonics` sets `cost`,
`u` and `v` through the same kind of `where`. Each guard is right on its own
factor -- `sqrt` has an infinite derivative at zero and `Y_lm` has no limit
there, and the primal is correct because `f_l(0) = 0` kills the finite harmonic.
**The product is what carries the derivative.** For `l = 1`, `f_1(q) -> c q` and
`Y_1m(qhat) = sqrt(3/4pi) q_alpha/q`, so the product is
`sqrt(3/4pi) c q_alpha`, a *linear function of the vector* `q` whose derivative
is `sqrt(3/4pi) c`. The code computes `Y df + dY f` with `df = f'(0) * 0` and
`dY = 0` and returns zero. `l = 0` is genuinely flat and `l >= 2` genuinely
vanishes, so `l = 1` is the only channel -- and it is in almost every dataset.

**Structurally**, on `Si.pz-vbc`: `f_1(q) = 0.22912917 q` at small `q`, and
`d/dk [f_1(|k|) Y_10(khat)]` at the origin is **0 in the code against
0.11195307** in the closed form.

**Physically**, `<psi_m|dH/dk|psi_n>` at Gamma on `si2-nosym.in`, the jvp
against a central difference of the same operator at a *frozen sphere*:

| | value |
|---|---|
| worst element, `nbnd = 10` | **0.13245** Ry bohr out of 1.0775 |
| the same at `h = 2e-3`, `1e-3`, `5e-4` | 0.13245 every time |
| `Gamma_1` x `Gamma_15` block (Frobenius over the axes) | **0.16957 against 0.45892**, ratio 0.3695 |
| second `Gamma_1` x `Gamma_15` | 1.97044 against 1.88369, ratio 1.0461 |
| `Gamma_25'` x `Gamma_15`, silicon's own optical transition | 2.33892 against 2.33892 |

The blocks rather than the entries because `Gamma_15` is a degenerate triplet
and the entries are the eigensolver's basis inside it (rule D4). **It is not an
underestimate**: the first `Gamma_1` state is 63 per cent low and the second is
4.6 per cent high, the sign following the relative phase of `psi(G = 0)` and
`<beta|psi>`. And it reaches **only** the pairs with one partner carrying weight
at `G = 0` and the other seen by an `l = 1` projector, which is why silicon's
own optical transition does not move.

**Three controls.** Off the reciprocal lattice the same comparison is pure
truncation, 1.89e-7 at `h = 2e-3` falling to 1.18e-8 at `5e-4` at
`k = (0.1, 0, 0)` and 9.42e-8 to 5.89e-9 at `(0.25, 0.25, 0.25)`, where Gamma's
0.13245 does not move with `h` at all. With a dataset carrying **no `l = 1`
projector** (N2 with `N.pbe-hgh`) Gamma itself is clean, 4.43e-9 on a matrix
whose largest element is 0.89015. And `_TINY` is on `|k+G|^2` at 1e-8, so what
is affected is the exact-zero row rather than a neighbourhood.

**The null that looked like a pass, and it is the reason this is worth
writing down.** At the default `nbnd = 4` the whole matrix is 2.65e-7 and so is
the discrepancy. In diamond the occupied manifold at Gamma is `Gamma_1` plus
`Gamma_25'`, and every matrix element of a *vector* operator among those four
vanishes by symmetry, so the first run of this measurement read a clean zero and
looked like agreement. The test carries that sentence so the next version of it
is not written at `nbnd = 4`.

**What it is worth.** Gamma's own contribution to a static independent-particle
dielectric constant, `sum_vc |v|^2 / dE^3` over the 10-band window, moves by
**1.435e-4** relative -- small because the affected transition sits at 14.6 eV
where `1/dE^3` has crushed it, and because the transition that dominates is the
one the defect does not touch. On a mesh that is diluted again by Gamma's own
weight, and **every k-point off the reciprocal lattice is exact**, so a shifted
Monkhorst-Pack grid never meets it at all.

**Two of the quantities the audit entry named are protected and were checked
rather than inherited.** The *diagonal* is untouched: no `m -> m` entry appears
anywhere in the error map above 1e-6, because the term needs two different
states, one with weight at `G = 0` and one an `l = 1` projector sees. So a band
velocity at Gamma is exact, and it is zero there by symmetry in any case. And
the **effective mass at Gamma never evaluates `dH/dk` at Gamma**: P48's "a
stencil must not contain its own centre" rule already keeps the centre out, so
the six stencil points sit at `Gamma +/- delta e_a` with `DEFAULT_DELTA = 0.025`
1/bohr, where `|k+G|^2 = 6.25e-4` against `_TINY = 1e-8`. It would take a
`delta` below 1e-4, four hundred times smaller than Elk's own `deltaem`, to
reach the guard.

What is left is the **off-diagonal** elements at a reciprocal-lattice point,
between a state with weight at `G = 0` and one an `l = 1` projector sees: the
deep interband transitions in `chi_0` and the optical conductivity, SHG's and
the shift current's three-band terms, and any f-sum rule, which has no `1/dE`
suppression to hide behind. Nothing here takes a second `k` derivative
analytically, so `l = 2`'s own version of this defect has no consumer today.

**Pinned** by `tests/unit/test_velocity_locality.py::
test_the_velocity_at_gamma_matches_a_frozen_sphere_difference`, an
`xfail(strict=True)` carrying the numbers, so it fails the day the defect is
fixed and forces this entry to move, with the off-lattice control passing beside
it.

**Closed by route A, on the user's choice.** `_origin_tangent` in
`pseudo/projectors.py` is a `custom_jvp` whose primal is exactly zero at every
row and every `q`, so it exists only to own a rule; the rule fires on the rows
`gvectors.ORIGIN_TOL` selects, which is the same test `modulus` uses, so a row
is corrected **if and only if** it was guarded -- and the threshold is public
now for that reason, two of them being enough to correct a row that was never
guarded. After it:

| | before | after |
|---|---|---|
| `Gamma_1` x `Gamma_15` block | 0.16957 | **0.45892**, against a true 0.45892 |
| ratio | 0.3695 | **1.0000** |
| Gamma's `sum_vc \|v\|^2/dE^3` | 1.435e-4 off | **1.489e-9** |
| Gamma against a difference, `h = 2e-3` / `5e-4` | 0.13245 / 0.13245 | 2.11e-7 / **1.32e-8** |

so Gamma now falls as `h^2` like every other k-point (the off-lattice control
is 1.89e-7 / 1.18e-8 on the same cell). **Nothing in the primal moves**: the
total energy, the eigenvalues, the forces and the stress are bit-identical on a
norm-conserving, a mixed PAW-and-norm-conserving and an ultrasoft cell. The
stress is the one that had to be measured rather than argued, since it
differentiates through `modulus` with respect to the *cell*; it is exact
because `k + G = 0` scales to zero under any strain, so the tangent the rule
fires on is itself zero there.

`f_1'(0)` is taken analytically by `projector_origin_slopes`, on the same
`kkbeta` range with the same Simpson weights as the transform it is the limit
of, so the two agree by construction: **0.2291291689** against the table's
0.2291291689, and to 1.2e-9 or better for `l` up to 2 across a norm-conserving,
an ultrasoft and a PAW dataset. It was written down wrong once, with `r^l`
where `_beta_kernel` carries `r^(l+1)`, which reads 0.2465 against 0.2291 --
the test checks all three datasets and all three `l` for that reason.

**Route B is not done and is the thing to reach for if a second `k` derivative
ever appears.** Route A fixes the first derivative; `l = 2`'s second derivative
at the origin is the same defect one order up, and nothing takes one today (the
effective mass differences a `jvp`, on a stencil that excludes its centre).

**The two routes, as they were put:**

* **A, a `custom_jvp` on the column assembly** -- taken. (`_species_columns` in
  `pseudo/projectors.py`). The primal stays bit-identical everywhere; the
  tangent is patched only on rows with `|k+G|^2 < _TINY` and only for `l = 1`
  columns, by the closed form `(-i) sqrt(3/4pi) f_1'(0) e_alpha`. The slope is
  available in closed form and was checked against the table:
  `f_1'(0) = (4 pi / sqrt(Omega)) int dr (r beta)(r) r / 3` reproduces
  0.2291291689 to **4.5e-11**. Contained, and it fixes the first derivative
  only.
* **B, the smooth factorisation.** Build the column as
  `(|q|^l Y_lm(qhat)) * (f_l(q) / q^l)` -- a solid harmonic, which is a
  polynomial in the cartesian components, times an even function with a finite
  limit at the origin. Both factors are then differentiable there and no guard
  is needed. It touches `harmonics.py`, which is validated element by element
  against `ylmr2`, and `formfactors.py`, so every column moves at the ulp level
  and every validated primal has to be re-checked. It fixes **all** derivative
  orders, including `l = 2`'s second derivative at the origin, which is the same
  defect one order up.

## 2. The saturated-point mask in `spin_energy_density` zeroed the *first* derivative, and it was worth 11.3 kbar and a sign on the stress of a fully polarized cell **[closed 2026-09-21, the stress now reproduces `pw.x` to every printed digit]**

`AUDIT-2026-09-20.md`'s `defumat/xc/functional.py:487`, reproduced and measured. It is a
wrong answer on a committed input rather than a nuisance, and the number that says so is
`pw.x`'s own stress on `tests/data/qe/h-atom-lsda.in`.

**The mechanism.** The line is
`jnp.where(saturated, raw(jax.lax.stop_gradient(rho)), raw(regular))`, which is the right
shape in `spin_potential`, whose returned quantity is already a first derivative, so the
`stop_gradient` kills the second, exactly as `dmxc_lsda` defines it. Here the returned
quantity is the *value*, so the same line kills the first derivative instead. Two places
said the opposite in so many words and both are now corrected: this method's own docstring,
which claimed that only the second derivative is masked, and
`spin_potential_and_energy_density`'s comment, which gave that as the reason the two calls
are not fused into one `value_and_grad`.

**Structurally**, at `rho = (1.0, 0.0)` on `pz`, against a central difference at `h = 1e-6`:

| | up channel |
|---|---|
| autodiff, as committed | **0.0** |
| central difference along the saturated branch | **-0.6288833068** |
| autodiff with the tangent restored | -0.6288833066 |

The up channel is the row to read, since both `(1 + h, 0)` and `(1 - h, 0)` stay saturated
so the difference is taken along the branch, and the restored tangent reproduces it to
2e-10. The down-channel difference steps across the boundary into a negative channel
density and is not a comparable number: the restored tangent there is a one-sided
derivative, `dzeta/d(down) = -2` at the boundary with `clamp_polarization` keeping the
interior tangent, so it reads 1.4236 against a two-sided difference of 0.3900.

**Physically, on the stress, which is where it is large.** Under a strain at frozen
plane-wave coefficients the *whole* density moves, since `rho` carries `1/Omega`, so the
XC stress reads `d(rho e_xc)/drho` at every point rather than only where a core or an
augmentation charge sits. On `h-atom-lsda.in`, one hydrogen in a 12 bohr box at
`nspin = 2`, the moment is one electron and the minority channel is at 1e-18, so the
polarization is at the boundary over most of the atom (863 of 64,000 points, carrying 0.597
of the one electron, and where they are is below):

| | diagonal, Ry/bohr^3 | kbar |
|---|---|---|
| autodiff, as committed | **+6.65467905e-05** | +9.79 |
| autodiff, first derivative restored | **-1.04885927e-05** | -1.54 |
| `pw.x` 7.5, same input, serial | **-0.00001049** | **-1.54** |

**The restored tangent reproduces `pw.x` to every digit it prints, and the committed code
is wrong by 7.70e-05 Ry/bohr^3, which is 11.3 kbar and the opposite sign.** The total
energy is the same on both legs and agrees with `pw.x` at -0.94606495 Ry, which is the
control saying only the tangent moved, and it is this project's own "the energy can be
right while its derivative is wrong" in a sixth place.

**Why nothing caught it, and it is not that the stress is unvalidated at `nspin = 2`.**
`test_stress.py`'s six generated cases are silicon four times plus `ni-ldau-stress`, which
is `nspin = 1`, but its borrowed set carries `pw_lsda/lsda.in`, which is nickel at
`nspin = 2` with a `pw.x` benchmark of -0.00010170 Ry/bohr^3. That cell is a **metal with
both channels populated everywhere**, so not one of its points is saturated and the
reference could not have shown this whatever it agreed to. It is the "which components the
validation cell allows to be nonzero" habit one step over: the question to ask of a
reference agreement is not only which components are nonzero but which *branch* of the code
the cell reaches. Worth knowing for the test that closes it: 7.70e-05 Ry/bohr^3 is *inside* `tests/tolerances.py`'s
`STRESS_RY_BOHR3 = 1e-4`, so the tensor assertion would have passed it and the pressure
assertion, which is `abs = 1.0` kbar against 11.3, is the one that catches it.

**On the force it is small, and that is consistent rather than contradictory.** At frozen
wavefunctions the density moves with the geometry only through a core charge, an
augmentation charge or PAW, and a core charge takes a point *out* of the saturated set
rather than into it: `with_core` and `paw/onecenter.py:444` add `rho_core/nspin` to *each*
channel, so the minority is lifted to `c/2` beside a majority of `up + c/2` and the
cancellation that makes the two sums equal no longer happens. What is left to reach is the
spin-resolved augmentation charge, which extends past the core radius. On `o2-lsda-force.in`, PAW and LSDA,
restoring the tangent moves the force from 0.2105652 to 0.2105626 Ry/bohr, **2.6e-6**, and
that is *toward* `pw.x`: the error against the reference 0.21055892 falls from 6.280e-6 to
3.677e-6. The analytic route is 0.21055953 on both legs, as it must be, since it reads
`v_xc` from `spin_potential`, which is masked at the right order, and that is what makes
the pair a QE-free discriminator.

**Which points are saturated, and this is why the stress moves by 11 kbar rather than by
noise.** `_fully_polarized` is `|up - dw| >= |up + dw|`, which is a statement about a
channel reaching zero and is reached here by **float64 cancellation instead**: with the
minority channel at 1e-18 to 1e-19 everywhere, the two sums are bit-identical wherever
`dw/up` falls below the machine epsilon, so the equality holds exactly rather than
approximately. Measured on the hydrogen cell, the set that fires is **863 of 64,000 points
carrying 0.597 of the one electron**, with `up` between 4.10e-3 and 1.584e-1, which is the
peak of the density. The equality is exact at all 863 and the strict inequality holds at
none, and no point of either channel is negative. So the mask does not fire in the fringe,
it fires **in the core of the atom**, wherever the majority density is large enough to
swamp the minority, and that is what puts 60 per cent of the charge on a branch whose
tangent is zero.

The same count on `o2-lsda-force.in` is 15,366 of 157,464 points, 9.76 per cent, but there
they carry 0.0058 per cent of the charge and the largest density among them is 5.73e-5
against a peak of 1.195: O2 has a real minority channel, so only the ripples saturate. That
contrast is the whole difference between the two numbers above, and it says where to look
for the next case, which is a cell whose minority channel is *empty* rather than small.

**The repair, and it needs no `custom_jvp`.** What is wanted on the saturated branch is a
value and a first tangent that are the expression's own and a second that is zero, and
`f(x0) + J(x0) . (x - x0)` with `x0 = stop_gradient(x)` is that statement written down:
the displacement is exactly zero so the value is unchanged, the derivative is `J` at the
point itself, and `J` carries no tangent of its own, so a second differentiation gives zero
rather than the infinity `rho_sigma^(4/3)` has there. It is two lines, `jax.jvp` of the
same `raw` at the anchor, and **the regular branch is not touched at all**, which is what
separates it from the `custom_jvp` on the whole method that moved a regular point's second
derivative from -3.5306 to -2.0485. Checked at three points, value, gradient and Hessian
all bit-identical to the committed expression at `(1, 0.3)`, `(0.05, 0.02)` and `(2, 1.9)`.

**What it is measured at.** The hydrogen stress goes to **-1.04885927e-05 Ry/bohr^3**,
which is `pw.x`'s -0.00001049 to every digit it prints, and the O2 force to 0.2105626, its
error against `ph.x` falling from 6.280e-6 to 3.677e-6. At a saturated point the gradient
is -0.6288833066 against a central difference of -0.6288833068 and the Hessian is exactly
zero, so the mask that makes triplet O2's `Z*` finite is still there at the order it
belongs at. `tests/regression/test_stress.py` and `tests/regression/test_forces.py`
together are **58 passed in 3 m 36 s** with the new case in them, and the fast gate is
**2868 passed, 64 skipped in 14 m 12 s**, the two new unit tests being the difference from
2866.

**What that does *not* say is that the O2 Born charge is validated.** The test the mask was
written for, `test_the_lsda_born_charges_match_ph_x`, **fails**, and it fails identically
with the repair and without it, at 0.0326 against a 5e-4 tolerance. So what is established
is that the repair does not move it, which is the right check; whether it is right at all
is the pre-existing question below.

**The test that pins it** is `tests/unit/test_xc_spin_kernel.py`, two of them, and the
first was checked to fail against the old code (gradient 0.0 where it now reads the
difference). The reference case is new and committed: `h-atom-lsda-stress.in` with
`reference.out.h-atom-lsda-stress`, generated by `tools/generate_reference.py` with the
vendored serial `pw.x`, and it is in `test_stress.py`'s `GENERATED` list.

**What is still owed** -- and it was owed for one day. Three tests of
`tests/regression/test_lsda_response.py` failed on master and they are **not** this:
`test_the_polarized_dielectric_constant_reduces_to_the_unpolarized_one`
at 2.709e-07 against 1e-08, `test_a_magnetic_insulators_dielectric_constant_matches_ph_x`
at 1.1164461 against 1.110915996, and `test_the_lsda_born_charges_match_ph_x` at 0.0326
against 5e-4. Each was run against the committed `functional.py` and against the repaired
one and the numbers are identical, so they predate today and belong to one of the
twenty-seven audit fixes that the slow set has not been run over. The oxygen datasets are
not the Simpson weights: `O.pz-rrkjus` has `cutoff_radius_index = 865` and `O.pz-kjpaw`
761 and 773, all odd, so that branch is never taken on this cell.

**All three are resolved, 2026-09-21, and two of them were one cause** -- see item 3
below. The two `ph.x` comparisons are the `l = 1` tangent at `k + G = 0` that `27eeaa2`
restored the day before, attributed by an A/B at its single switch and closed by a
`Calculation(origin_tangent=False)` that puts QE's own zero back on a `ph.x` comparison.
The third is **not this, and is not attributed**: it *passes* now, both on its own and
in file order, and silicon's explicit k-list is shifted so no `k + G = 0` exists in it
and the flag cannot reach it. Why it read 2.709e-07 against 1e-08 the day before is
unmeasured, and a pass here does not discharge a failure seen there -- it stays open as
item 4.

**What was not measured.** Whether the 2.6e-6 on the O2 force sits in the plane-wave `etxc`
or in PAW's one-centre term, which one more pair of legs would separate; and the noncollinear
regime, where `local_spin_frame` clamps `|m|` to `|n|` and reaches the same branch.

## 3. `ph.x` has no `l = 1` tangent at `k + G = 0`, and three reference comparisons moved when this code grew one **[closed 2026-09-21, by a switch whose default is the physics]**

The two `ph.x` failures of item 2 are **one cause**, and it is not an audit fix: it is
`27eeaa2`, the `l = 1` tangent at `k + G = 0` restored on 2026-09-20. Attributed by an A/B
at the single switch that owns it, `_origin_slopes`' `axes`, with everything else held
fixed.

**The measurement.** `o2-fixed-lsda.in`, ultrasoft `O.pz-rrkjus`, LDA, **Gamma only**,
against `reference.out.ph-o2-fixed-lsda`:

| | eps_xx | eps_zz | Z*_xx | Z*_zz |
|---|---|---|---|---|
| tangent carried (master) | 1.11644639 | 1.19737703 | 0.1011009 | 0.2272242 |
| tangent dropped (QE's zero) | 1.11091517 | 1.19800116 | 0.1337198 | 0.2004090 |
| `ph.x` 7.5 | 1.110915996 | 1.198004867 | 0.13367 | 0.20023 |
| error, dropped leg | **8.2e-07** | **3.7e-06** | **5.0e-05** | **1.8e-04** |

Tolerances are 2e-5 and 5e-4, so the dropped leg passes both and the carried leg fails
both. The total energy is **-63.3630837811 Ry on both legs**, every digit, so only the
derivative moved. A second cell confirms it independently: `si10-epsilon`, a `4 4 1 0 0 0`
mesh holding Gamma at a sixteenth, failed at 4.0e-4 against a 1e-4 tolerance on a tensor of
19 and **passes with the tangent dropped**.

**QE zeroes the row twice over**, `PW/src/commutator_Hx_psi.f90:113-118`
(`gk_vpol = 0` where `g2k < 1.0d-10`) and `upflib/dylmr2.f90:88-92` (`dg = 0` where
`gg <= eps`), and the limit it is zeroing is smooth --
`f_1(q) Y_1m(qhat) -> c sqrt(3/4pi) q_m` is linear in the vector, so the gradient is
`c sqrt(3/4pi) delta_m,alpha` and a `k` of 1e-4 off Gamma computes it. QE's `dH/dk` is
discontinuous at Gamma; this code's is not. `PLAN.md` P24 carries the derivation and the
five reference cells that hold Gamma at all.

**The resolution, put to the user as a choice and decided by them**: keep the term as the
default and add `Calculation(origin_tangent=False)`, so a `ph.x` comparison stays an exact
live check instead of a widened tolerance. The flag threads through all six
`build_projector_core` sites and `Calculator.SETUP_ONLY_OPTIONS`; the three affected
reference tests take it, and the QE-free checks on the same cell keep the default.

**The scope was then measured rather than left named, by the slow set the same night, and
it was wider than the two cells above.** Four more tests failed on it, all on AlAs meshes
that hold Gamma, and an A/B under `origin_tangent=False` passed every one of them where
the default failed all four:

| test | anchor | out by | tolerance |
|---|---|---|---|
| `test_piezoelectric.py::test_the_same_assembly_in_the_position_coordinate_is_the_born_charge` | `ph.x`'s `Z*` | 2.0e-3 | 5e-4 |
| `test_piezoelectric.py::test_the_driver_reports_the_dielectric_constant_it_already_solved` | `ph.x`'s `eps` 12.967 | 3.7e-2 | 1e-3 |
| `test_spectra.py::test_the_born_charges_of_alas_match_ph_x` | the same `ph.x` `Z*` | 2.0e-3 | 3e-4 |
| `test_piezoelectric_wedge.py::test_the_wedge_completes_and_the_taped_route_is_where_it_did_not` | `CLOSED_GRID_E14`, **this code's own** | 5.5e-4 | 1e-4 |

**The reassuring half is what did *not* fail.** Each of those tests asserts a **route
identity** before the reference, and every one passed on both legs -- 1e-12 between the two
Born-charge assemblies, 1e-8 between the driver and its own solve, 1e-6 between the taped
and contracted piezoelectric routes. So the term is applied *consistently*; what moved is
the comparison with QE, which is the same finding as O2 and not a second defect.
`PLAN.md` P24 carries the rule for which leg a test takes and the table of what Gamma's
weight is worth. Closed by putting the piezoelectric family and `test_spectra.py` on QE's
convention: **10, 14 and 1 passed** on `test_piezoelectric.py`, `test_spectra.py` and
`test_piezoelectric_wedge.py`.

**What is not closed by it.** Nothing takes a second `k` derivative analytically today, so
`l = 2`'s second derivative at the origin is still the open one order up (item 1).
`si-epsilon-unshifted` and `al2-metal` are still unmeasured -- the first has a `-nosym`
twin that is `tests/regression/test_tddft.py`'s `CASE`, and a TDDFT head **is** `dH/dk`,
and that file passed, which is evidence at 1/64 weight rather than proof. A **dynamical
matrix is not a candidate**, not going through `dH/dk` at all, so what to grep for is an
E-field or an optical head, not a phonon. `test_piezoelectric_paw.py` and
`test_piezoelectric_augmented.py` take the convention through the shared `_field` and
**neither was run**, for the reason item 5 gives.

**One notebook is stale for the same reason, and it predates this entry.**
`notebooks/27_excitons_and_tddft.ipynb` runs `si-epsilon-unshifted-nosym`, Gamma at a
sixty-fourth, and was last executed **2026-09-02**, where the tangent landed on
**2026-09-20** -- so its committed output is from before the term existed. Nothing here
moved it further, the default being unchanged, and the size to expect is roughly a
quarter of `si10-epsilon`'s 2.2e-5 relative. It is one entry in the larger fact that the
notebook set has not been re-executed over the twenty-eight fixes.

## 4. A spin-polarized dielectric identity that read 2.709e-07 one day and passes the next

`tests/regression/test_lsda_response.py::test_the_polarized_dielectric_constant_reduces_to_the_unpolarized_one`
compares `si-epsilon.in` run as `nspin = 1` and as `nspin = 2` with no magnetization, which
must agree exactly, at a tolerance of 1e-08. It was reported **failing at 2.709e-07** on
2026-09-20 and **passes** on 2026-09-21, on its own and in file order, with the same input
and a `functional.py` the reporting session had already excluded by running both legs of it.

It is **not** the `k + G = 0` row of item 3: silicon's explicit k-list is shifted and holds
no such plane wave, and the flag was measured to leave that cell bit-identical. What is left
is that 2.709e-07 is 2e-08 *relative* on an `eps` of about 13, reached through two
independent SCF runs and two iterative response solves, so the honest reading is that the
number sits near the tolerance rather than that it is fixed. **What would settle it** is
running the pair a few times and reading the spread, which is cheap and has not been done;
until then a pass here does not discharge the failure seen there, which is this project's
own "a check whose null result cannot be told from a pass" pointing the other way.

## 5. Seven files of the slow set are **unrun rather than failed**, and they all stopped at the same 12.4 GB

The 2026-09-21 pass of `tools/run_regression.sh` (180 files, **1270 passed**) lost seven to
`SIGKILL` at the 12 G cap: `test_ldau_flavours` (12425M), `test_lsda` (12427M),
`test_noncollinear_hubbard_resume` (12400M), `test_piezoelectric_augmented` (12425M),
`test_piezoelectric_paw` (12420M), `test_spinor_forces` (12427M) and `test_ten_site`
(12426M). An eighth, `test_ldau`, survived and the **watchdog named** its worst test:
`test_converged[pw_lda+U/lda+U_force.in]` at 11101M, 73 passed beside it.

**The seven peaks are 12.42 GB give or take 10 MB, which is the cap and not the tests.**
Nothing here says what any of them *needs*, so the number to quote is that they did not
run. Two of the seven are on record running clean: `test_ten_site` is P28b's own file,
measured complete under `jax.clear_caches()`, and it died here after thirteen passing
tests. Against the `test-runs` skill's 1.0 to 1.3 GB per slow file, seven files arriving at
one ceiling is either a memory regression in this code or a **cache state** -- `CLAUDE.md`
already records that loading one 603 MB cache entry is worth 6.3 GB resident.

**The discriminator is one run and it is the experiment the skill says has never been
done**: `DEFUMAT_CACHE_DIR=off` on `test_ten_site` alone, against the same file warm.
**Raising the cap is the wrong first move**, since it answers nothing and hides the
question. Until that is run, `test_piezoelectric_paw` and `test_piezoelectric_augmented`
carry item 3's convention change **unverified** -- the wedge, which imports the same
fixture, is the only one of the three that was run.

## 6. Three failures the slow set found that are **not** the `k + G = 0` row, and one of them is not a tolerance question

All three move a **primal** quantity, which the origin tangent cannot touch, so none is
item 3.

- **`test_magnetic_constraints.py`, three failures on one cell.** Two are one number:
  `noncolin-constrain_atomic.in`'s total energy at -55.69055643867099 against QE's
  -55.69055687, out by **4.3e-07** against a 3e-07 tolerance. It is **not** the even-mesh
  Simpson closure: `Fe.pz-nd-rrkjus.UPF` has full mesh 957, `msh` 839 and `kkbeta` 751, all
  odd, so that branch is never taken here and `59b2ebb`'s claim holds on this cell. The
  candidates that remain are `9f089fa` (the direction constraint's normalisation),
  `2b84b0c` (the symmetry tolerance, which would move the k-set) and `21fe25b`
  (`scf/fields.py`); a bisect on the one test settles it.
- **The third is a different thing and should not be folded into that drift.**
  `test_the_two_fixed_spin_moment_rules_find_the_same_field` has one leg **not converged
  after 2000 iterations**, at accuracy 3.556e-04 against a `conv_thr` of 1e-08, reporting
  `M = 2.0705` where the leg beside it converges in **66** iterations at `M = 2.0005`. A
  fixed-spin-moment run that no longer converges is a defect rather than a tolerance.
- **`test_stm.py::test_an_antiferromagnet_is_flat_in_charge_and_alternates_in_spin`**,
  against 0.029525036433705836 at 3e-07. Unattributed.

## 7. A cell with no spin-orbit coupling broke its own directional degeneracy by 0.108 meV

`test_relaxed_anisotropy.py::test_without_spin_orbit_coupling_every_direction_has_the_same_energy`
fails with the two directions **1.079e-01 meV** apart. Without spin-orbit coupling the
total energy cannot depend on the direction the moment points, so this is a symmetry
statement the code makes about itself rather than a comparison with anything, and it is the
single most alarming number in the 2026-09-21 pass. It is a **total energy**, so it is not
the origin tangent. Unattributed, and it wants a session of its own rather than the tail of
one.
