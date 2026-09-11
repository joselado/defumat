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

**Status, 2026-09-11 (later the same day).** Sixteen entries are closed, each with a
test that was checked to fail against the old code: **A1, A3, A4, A5, A6, A7, A8, A9,
A10, B2, D2, E1, E2, F1, F2, F3**, together with Part I item 1 and its two siblings
C2/C3. What is left is A2, B1, B3, C1, D1, D3, E3 and Part I item 2 -- the entries
whose *test* is expensive rather than whose fix is. Each closed entry is marked
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

## 1. The Kramers bound asserts round-off where the solver promises `empty_ethr`

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

---

## 2. `test_spinorbit.py` peaks at 11,088 MB against a 12 GB cap

92% of the cap on a 30 GB machine, so it is the next out-of-memory kill whether or
not it has happened yet. The watchdog named
`test_spin_orbit_total_energy[spinorbit-paw.in]` at 10,818 MB; **its assertions
passed** -- the memory is the finding, not a failure. For scale, the next file
down in the same run is `test_stress.py` at 6,317 MB.

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

## 3. A single SCF is not restartable, only a relaxation is

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

### A2. The Sternheimer stack rebuilds its potential from the **input** magnetic field **[opened here]**

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

### B1. The interband conductivity's degeneracy guard is below the eigensolver's accuracy

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
> docstring says so and asks for the check where it is available.

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

### B3. The Kubo Berry curvature swallows the point a Chern number is about

`defumat/topology/berry.py:296`. `_kubo_point` masks an occupied-empty degeneracy to
zero weight while its own comment says such a pair is "left to blow up", and its inner
and outer guards use *different* thresholds, so a gap between 0 and 1e-12 overflows in
the un-taken branch -- `jnp.where` does not protect a gradient -- and poisons any
derivative through the curvature. The mask is an absolute 1e-12 Ry rather than a
fraction of the band width.

Run `method='kubo'` on graphene or a Weyl mesh that includes the band-touching point:
a finite number comes back where the quantity diverges, the mesh sum gives a plausible
non-integer Chern number, and nothing in the output says a singular point was dropped.
The overlap-determinant route (`CLAUDE.md` rule D4's reason for it) is exact on any mesh
and is what this should be checked against.

---

## C. Tests that cannot tell a pass from silence

The trap `CLAUDE.md` added most recently, and Part I item 1 is the same family.

### C1. The spinor force symmetry test asserts an identity the code has already imposed

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

### C2/C3. The second and third instances of Part I item 1

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

### D1. The sum-over-states `chi_0` materialises the whole pair axis

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

### D3. The vertical-tunnelling workflow allocates outside every dial

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

### E3. Projected DOS for a noncollinear run *without* spin-orbit coupling

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
