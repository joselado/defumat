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
60 per cent low on a 45-atom spinor PAW slab -- **closed 2026-09-14, and it was the
guard**, not a missing model term -- a Davidson inner-step count that may degrade at the
minimum subspace, and the checkpoint's remaining Hubbard refusal, which is probably as
wide as the field's was.

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

**Where that argument fails is the case this code cannot yet run.** A *spontaneous* wave --
Elk's `rndbfcu` seed faded by `reducebf`, listed as outstanding in `PLAN.md` P88 -- has no
field holding its magnetization, so the moment is the soft direction and the one that
sloshes, and it is bounded by the half of `dr2` that carries no `1/|G+Q|^2`. **Take this
measurement again when that lands**; the factor of 2.2e3 in the weights is what it would
be paid at.

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

## 3. The checkpoint's Hubbard refusal is probably as wide as the field's was

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


### 3. A resume spends a whole Davidson budget re-tightening bands it does not need **[opened 2026-09-14, from the NiBr2 helix run]**

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
