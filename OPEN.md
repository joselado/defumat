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

## 2. `test_spinorbit.py` peaks at 11,088 MB against a 12 GB cap

92% of the cap on a 30 GB machine, so it is the next out-of-memory kill whether or
not it has happened yet. The watchdog named
`test_spin_orbit_total_energy[spinorbit-paw.in]` at 10,818 MB; **its assertions
passed** -- the memory is the finding, not a failure. For scale, the next file
down in the same run is `test_stress.py` at 6,317 MB.

**The number predates P74 and should have moved.** It was measured with the whole band
block in the FFT box, before `vloc_psi_nc` learned to call `map_bands`, so the spinor local
term was transforming every band at once in exactly the file this peak belongs to. Whether
11,088 MB is still the peak is unread (`PLAN.md` P74, "What is outstanding"), which is one
more reason the first step is a measurement.

**The one lead, and what it is not.** The stderr shows XLA constant-folding and
transposing an `f64[25,1277,34,34]` inside `jvp(jit(_paw_onecenter))`, twice,
each fold over 2 s. That shape is `PawSpecies.density_ae`/`density_ps`,
`(nh, nh, nlm, mesh)` at `nh = 34` for a fully-relativistic platinum dataset --
295 MB each. They are ordinary pytree fields, so inside `_paw_onecenter` they are
arguments; appearing as XLA *constants* means the enclosing `jit(<lambda>)`
closes over the object holding them, which would give every compiled variant its
own copy in a cache that never shrinks. **But 590 MB is under 6% of an 11 GB
peak.** The stderr says where the compile time goes. It does not say where the
resident set goes, and starting from "the constant is the memory" is chasing 6%.

**So the first step is a measurement, not a fix.** Two that cost little:

- `jit(f).lower(*ShapeDtypeStructs).compile().memory_analysis()` on the PAW
  spinor gradient at that case's shapes -- it runs the compiler and allocates
  nothing, the same route P73 used for the slab it could not run;
- the same test alone with the watchdog's own sampling, and the scope's
  `memory.current` beside `psutil`'s RSS, since the cgroup charges page cache and
  child processes and this process's anonymous pages are not the same number.

**Then the two candidates worth separating**, because the fixes are different: a
single large allocation inside one backward pass (the augmentation table
`Q_ij(G)` at `nh^2 x ngm` per atom is the known one, and `nh` is 34 here), versus
accumulation of XLA executables across the file's 28 tests. `jax.clear_caches()`
in an autouse fixture distinguishes them in one run -- if the peak drops, it is
accumulation.

**Cost.** The measurement is one 8-minute file at 11 GB. Run it **alone** --
nothing else on the machine, nothing else being timed -- and through
`tools/run_regression.sh` so a kill costs the file rather than the session.

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

> Four entries in `_REFUSED_SWITCHES`, which already existed. **Unverified on
> this machine:** `tests/regression/test_input_sweep.py` sweeps QE's own `pw_*`
> inputs and expects each to run or to hit a *listed* refusal; if any of them
> sets one of these four, it needs an entry there. The vendored tree is absent
> here so that file skips.

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

---

## H. Cheap, and the gain is bounded by a figure already on record

### H1. A GGA evaluates its energy expression twice per iteration

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
