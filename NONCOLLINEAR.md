# Noncollinear magnetism: what works, what is unmeasured, and what to fix first

## 1. What this file is

> **Status, 2026-09-12, second pass.** Fixed: **all of Tier 1** (items 1-7, the guard half
> of 3), and from Tier 2/3 items **8**, **14**, **16**, **18**, **19**, **20** and the
> refusal half of **21**, plus four of the smaller items -- the `reducebf` range, the two
> objects called `local_moments`, the per-site constraint residual, and the phantom
> `_refuse_untextured_symmetry`. Numbers are folded into each entry below and into
> `PLAN.md` P77/P77a-d and **P78**.
>
Also fixed: **9** (the DFT+U continuation, including the noncollinear checkpoint resume),
> and **10** is *measured and largely false* -- `'atomic'` holds a 120-degree state to 0.55
> degrees per site in 38 iterations where the unconstrained run collapses to collinear in
> ten. Read its entry: the scheme that works is not the one this file recommends.
>
> **Still open, and every one is a phase:** **11** (no noncollinear linear response),
> **12** (no noncollinear magnons), **13** (`d_spin_ldau`, which gates three consumers),
> **15** (no external number for a spin spiral), **17** (the mixer's metric -- but see
> item 16's measurement, which points away from it), **22** (the memory wall, unmeasured
> since P73/P74), and Elk's per-atom feedback field, which item 10 now wants as an
> improvement on a working route rather than as the only route.
> Plus the notebook, and the open questions O1-O14.
>
> Three defects found *while* fixing, none of which is in this file's own list: `at_cell`
> never remeasuring the integration spheres, `forces/torque.py` guarding a per-point
> modulus with a global norm, and `tests/unit/test_angular_momenta.py`'s cubic-symmetry
> check on `|<L>|` failing at 2.3e-7 against its own 1e-9 -- three orientations of the same
> nickel cell stopping at states 1.15e-8 Ry apart at `conv_thr = 1e-10`, which is item 18's
> weighting in the wild.
>
> **Everything not listed as fixed still stands.** Each fixed entry keeps its full
> reasoning, because the reasoning is why the fix has the shape it does; read the bold line
> at its head for the state.
>
> **`MAGNETISM-NEXT.md` is the forward-looking half of this file** and is where to start if
> the question is "what next" rather than "what was found". It carries what is left, in
> order, with the first concrete step of each; this file keeps the reasoning behind every
> entry, which is why the two are separate.

This is an **audit**, run on **2026-09-12** at commit `314d676`, of everything in this
package that a physicist would touch to set up, converge, trust and analyse a
noncollinear magnetic structure. The method was five read-only agents over the source,
one per dimension -- stating a structure, converging it, the magnetic symmetry group,
what is validated, and what can be computed on top -- each of whose findings was then
handed to a separate verifier told to refute it. Corrections from that second pass are
folded into the entries below rather than appended, and the sub-claims the verifier
falsified are listed in section 7 so that nobody rediscovers and re-believes them.

Three things about the provenance, because each will otherwise cost a future session an
hour. **Nothing here was run**: no SCF, no test, no benchmark. Every number is *quoted*
from `PLAN.md`, `OPEN.md`, `PERFORMANCE.md`, `docs/features.tex`, a docstring or a
committed test, and the audit's own contribution is the reading, not the measurement.
Every `file:line` is as of `314d676` and should be re-grepped rather than trusted. And
the Quantum ESPRESSO side was read against **7.4.1 in `~/apps`**, not the vendored 7.5
tree `CLAUDE.md` names, which is absent from this checkout -- so `input.f90:1448-1449`,
`symm_base.f90:715-788`, `add_bfield.f90:185-192`, `setup.f90:271-276` and
`compute_ux.f90:50-56` are 7.4.1 line numbers. Elk is `~/apps/elk-9.6.8`.

One thing the audit did **not** examine, so that its silence is not read as a pass:
**relaxation of a textured magnet**. Nothing here follows `run_relax` or `vc_relax` under a
texture beyond the one-line note in section 3 that `starting_moments` is a tuple in
`ATOMIC_POSITIONS` order which `with_positions` does not touch, and that nothing rechecks the
magnetic group while the atoms move.

---

## 2. The short answer

**Yes for spin-orbit coupling, and for any magnetic order you can express as different
species. Not yet for a texture on one species that you need to trust without checking
it by hand.**

**What works.** The spinor machinery is in good shape and its numbers are against
`pw.x`: one Hamiltonian on a space twice as large, `j`-resolved projectors from a
fully-relativistic dataset, forces, stress and both relaxations on norm-conserving,
ultrasoft and PAW. The magnetic symmetry group of bcc iron comes out 48 operations
reduced to 16 with 8 of them carrying time reversal; its site moment is 3.18 mu_B to
1e-3; platinum's spinor PBE stress agrees to 4.4e-7; the four-hydrogen noncollinear
force cell agrees to 3.6e-9 Ry and 8.9e-7 Ry/bohr; spinor DFT+U on relativistic BN
agrees to 1.2e-7 Ry. Stating a structure is also better than it looks: `STARTING_MOMENTS`
is a card `pw.x` has no counterpart for, one cartesian moment per atom, and since
2026-09-11 it reaches all three places that matter -- `domag`, the magnetic symmetry
filter, and the starting density -- so a 120-degree Neel state, a canted state, a
skyrmion in a supercell and a cycloid can all be written down and the seed is not
symmetrised away at iteration zero. On top of that sit four QE constraint schemes plus a
fifth QE lacks, Elk's `reducebf` and fixed-spin-moment, per-atom penalty fields, and an
exact SCF restart. The input-boundary refusals are the best-guarded part of the package.

**What does not.** Nothing in a converged run tells you the texture survived: the
sphere-integrated per-atom moment that `pw.x` prints every iteration is computed nowhere
in a run, and the console shows the cell total, which is zero by construction for every
compensated state (item 2). No texture stated with `STARTING_MOMENTS` has ever been
carried through an SCF and inspected -- not in a test, not in a notebook. The one
converged noncollinear texture in the repository is four hydrogen atoms at 90 degrees per
site under `nosym`, checked as an energy identity against a spiral. The one production
attempt, a 45-atom NiBr2 cycloid, unwound to collinear between iterations 3 and 6 and
then reported `accuracy = 7.55e-07` in 23 iterations. The commonest magnetic structure of
all, a two-sublattice collinear antiferromagnet written on one species label, is
symmetrised back to zero (item 1). Nothing holds a structure that is not the ground
state: on fcc hydrogen a 90-degree spiral and a 15-degree cone both collapse to the
nonmagnetic solution (item 10). And nothing above the ground state exists for a
noncollinear magnet -- no phonons, no Born charges, no dielectric constant, no Raman, no
magnons (items 11, 12).

**So, concretely.** Use it today for spin-orbit coupling, magnetocrystalline anisotropy,
the topological invariants, and the ground state of a magnet whose sublattices can be
given different species labels. For a texture on one species, run with `nosym = .true.`
and check the answer by hand: `Calculator.get_angular_momenta()` returns a per-atom
`<S>` vector that will say whether the order survived, subject to its own refusals.
Do not yet trust an `E(q)`, an `E(theta)` or a magnetic phase diagram out of this code
until item 10 has a working per-atom constraint and open question O1 has an answer.

---

## 3. Ranked work, by what a wrong answer costs

The ordering rule is `CLAUDE.md`'s: **a silent wrong answer outranks a missing feature,
which outranks an inconvenience.** Within the first tier the order is how likely a
physicist pointing this at a real magnet is to meet it. Sizes are *session* (hours,
one edit plus its test), *phase* (a `PLAN.md` phase with its five deliverables), or
*unknown*.

### Tier 1 -- a converged, plausible number for the wrong physics

#### 1. A collinear run gets no magnetic symmetry filter, so a one-species antiferromagnet or altermagnet is averaged back to zero

**FIXED 2026-09-12, `9f806b0`.** `collinear_symmetries` (`sgam_at_collin`) is wired
into all three sites. On `tests/data/qe/h2-mirror-afm.in` -- two H related by a mirror
with `+-0.6` from the card -- the group is cut 16 -> 8, no survivor swaps the sublattices,
and the run reaches `-1.93526881` Ry with sites at `+-0.291` in 8 iterations against
`-1.93478487` Ry, sites `0.000`, in 5 before. **4.84e-4 Ry = 6.6 meV and the whole
magnetic state.** It now agrees with the `nosym` spelling to 1e-8 Ry. Time-reversed
operations are *discarded* (QE's `colin_mag = 1`); the `t_rev` channel-swap enlargement is
still not done. The filter is a no-op on all 22 committed collinear inputs, swept by a
test. Read the rest of this entry for the reasoning, not for the state.

**What.** The magnetic group is the subgroup of the crystal's operations that preserves
the moments. It is built only for `nspin_mag = 4`. A collinear run therefore keeps the
operation that carries sublattice A onto sublattice B, and the density symmetriser then
averages `rho_up` over both sites and `rho_down` over both sites, which is exactly the
statement that the staggered moment is zero.

**Evidence.** `is_magnetic` returns False at `if nspin != 4: return False`
(`defumat/system/builder.py:624-626`), so `System.symmetry_group()` (`builder.py:597-600`),
`build_system` (`:874-877`) and `_respin_kpoints` (`:509-514`) all skip the filter.
Meanwhile a `STARTING_MOMENTS` card is *accepted* for a collinear run as long as x and y
vanish (`builder.py:1453-1463`), and it seeds the density per atom through
`_per_atom_magnetization(axis=2)` into `rho_up - rho_down`
(`defumat/scf/driver.py:3031`). `Calculation.symmetrize` then sends `nspin_mag = 2` to
`_symmetrize`, which `vmap`s each channel independently over the full group
(`driver.py:2633-2639`, `:276-290`), and `find_symmetries` matched the two sites in the
first place because `_maps_structure` tests only that they are the same *species*
(`defumat/system/symmetry.py:603-609`). The k-set was reduced with that same full group,
so switching the symmetrisation off would not rescue the run either: **`nosym = .true.`
is the only working spelling today.**

Two corrections to fold in. The input takes **two** lines rather than one:
`build_system:773-790` refuses `nspin = 2` with no `starting_magnetization` at all, and
its own message instructs the user to add one, after which the card is honoured and the
group is not. And **QE's default shares the hole**: `symmetry_with_labels` and
`use_spinflip` both default to `.FALSE.` (`Modules/input_parameters.f90:204,208`), so
`colin_mag` stays at -1 and `sgam_at_collin` (`PW/src/symm_base.f90:715-788`, dispatched
at `:404-405`) is never called. The exposure is larger here only because
`STARTING_MOMENTS` makes a one-label collinear antiferromagnet *expressible*, which a
`pw.x` input cannot say at all.

**Why it matters.** Rutile is the canonical case: the two Ru at `(0,0,0)` and
`(1/2,1/2,1/2)` are related by `{C4z | 1/2 1/2 1/2}`, whose denominator of 2 is
crystallographic (`symmetry.py:559`) and which `is_supercell` does not disable
(`symmetry.py:209-238`, since the oxygen at `u ~ 0.305` is not mapped by the pure
translation). An altermagnet is *by definition* compensated order whose sublattices are
related by a rotation, which is precisely when the swap is a point-group operation of the
chemical structure. The user writes RuO2 or MnTe with one label and `+-m`, the run
converges, prints a perfectly good total energy and an absolute magnetization near zero,
and the spin splitting the material is defined by is gone. Nothing is raised, and
`docs/features.tex:881-895` promises the card "decides the symmetry group" with no
collinear caveat.

**Check.** Static, no SCF. Build the cell with one label, `starting_magnetization(1) = 0.5`
and a z-only `STARTING_MOMENTS` card of `+0.5 / -0.5`, then assert that
`atom_mapping(cell, structure, system.symmetry_group())` contains a row sending atom 0 to
atom 1. With two labels it contains none. To price it, one SCF each: the one-label run's
`absolute_magnetization` goes to zero and the two-label run's does not.

**Size.** Session for a refusal at the input boundary. Phase for the real fix, which is
`sgam_at_collin` transcribed with `t_rev` implemented as a channel swap in `_symmetrize`
-- what QE would do if it had this input.

#### 2. Nothing a run reports says whether the texture survived

**FIXED 2026-09-12, `9f806b0`.** `SCFResult.site_charges`/`site_moments`, in `history`
every iteration, `report_mag`'s block printed at the end with `theta` and `phi`, and
`|m|_site = min..max` on the per-iteration line; the noncollinear format is widened to
`{:7.4f}`. `LocalRegions` is packed as `pointlist`/`factlist` for `scheme = "qe"`, so the
spheres cost `ngrid` rather than `nat x ngrid` -- ~31 MB instead of ~5 GB on a 157-atom
slab -- and the readout is 0.25-0.72% of an iteration. **Not** yet reported by `run_relax`,
`run_vc_relax` or the response stack.

**What.** For a compensated magnet the two numbers a run prints are identical for the
state you asked for and for the collinear state the symmetriser may have given you
instead. The vector total is zero for both. `int|m|` moves in the *wrong* direction:
`CLAUDE.md`'s own NiBr2 note records that a vector-norm integral loses weight at the nodes
a collinear state has and a spiral does not.

**Evidence.** `get_locals` (`defumat/scf/locals.py:220`) is QE's `report_mag` per atom and
is validated against `pw.x`'s printed block at `abs = 1e-4` on the charge and 1e-3 on the
moment (`tests/regression/test_noncollinear_magnetism.py:245-267`). It has **no caller
anywhere in `defumat/`**. `SCFResult` carries `magnetization_vector` (`driver.py:841`) and
`absolute_magnetization` (`:835`) and no per-atom field at all. The per-iteration console
prints the cell total as `m = (mx, my, mz) [|m|]` in `{:6.3f}` (`driver.py:4409-4413`)
where the *collinear* branch beside it uses `{:7.4f}` (`:4417`) -- the harder regime is
the coarser one -- and `FSM_TOLERANCE = 1e-3` (`fields.py:96`) is exactly that format's
last digit, which is `CLAUDE.md`'s "stops moving when the format runs out" verbatim. The
machinery that would produce a site moment, `build_local_regions`, is constructed only
when a per-atom field or an `atomic`/`atomic direction` constraint is set
(`driver.py:1768-1776`), so a plain textured run cannot reach it.

There **is** a partial escape, and it should be used until this is fixed:
`Calculator.get_angular_momenta()` (`calculator.py:742`) returns `SiteAngularMomentum`
whose `.spin` is a `(natom, 3)` array of per-site `<S>`, with a `table()` printer
(`projwfc/angular_momentum.py:162-251`). It is a projector expectation value on the
ortho-atomic set rather than a sphere integral, it is post-hoc only, and it refuses a
symmetry-reduced k-set, a spiral, and a fully-relativistic ultrasoft or PAW dataset
(`angular_momentum.py:254-283`).

**Why it matters.** This is what made P75's failure invisible from the console. The
45-atom NiBr2 cycloid unwound to collinear between iterations 3 and 6, was held there for
twenty more, and converged to `E = -8925.9786` Ry at `accuracy = 7.55e-07` in 23
iterations -- a clean convergence to the wrong state, diagnosed by hand from outside. The
per-site angles (`0.0 0.1 0.1 179.8 ...`) and the SVD signature `[3.873, 0.0056, 0]` in
`PLAN.md:12354-12358` exist nowhere in the tree; the one instrument that has ever caught a
symmetrised-away texture on this project exists only as whatever was typed at a prompt.

**Check.** No measurement is needed to establish the gap -- it is a set difference, and
`get_locals` has no caller. What is worth measuring is the cost, since that decides
whether it goes in the loop or only at the end: one `einsum` of the `(nat, ngrid)` region
weights against the `(nspin_mag, ngrid)` density, against a ~0.24 s iteration on
`fe-mag-1k`. Then put it on `SCFResult`, in the per-iteration `history`, and in the
console the way `report_mag.f90` prints it, and widen the noncollinear format to
`{:7.4f}`.

**Size.** Session.

#### 3. The noncollinear GGA potential is differentiated by every spinor force, and no committed test differentiates it

**HALF FIXED 2026-09-12, `1828c4e`.** The NaN is gone: `_noncollinear_gradient_correction`,
`_noncollinear_meta_exchange` and `forces/torque.py` all use `safe_modulus`, verified to
give `nan` on 243 components without it and finite with it, energy unchanged at
`-0.0612744443880662` Ry. The source-text test is now a **sweep with an allowlist** over
six modules rather than two named functions. **The larger half is still open**: there is
no measured derivative of this branch against `pw.x`, and generating
`h4-noncolin-force.in` with `input_dft = 'PBE'` plus its reference is a phase.

**What.** At `nspin_mag = 4` with a gradient-corrected functional the exchange-correlation
potential goes through `_noncollinear_gradient_correction` (`defumat/scf/potential.py:304-353`):
rotate into the local spin frame, call the collinear GGA, rotate back. Every spinor force,
every spinor stress and every response `jvp` differentiates that function.

**Evidence, and this is the larger half.** `potential.py:704-712` dispatches into it on
`nspin == 4`. The one regression case that looks as though it exercises the branch does
not: `spinorbit-pbe.in` sets `starting_magnetization = 0.0`, so `domag` is false,
`nspin_mag = 1`, and the *unpolarized* branch at `:719-730` is what the 4.4e-7 PBE stress
(`tests/regression/test_spinor_forces.py:250-253`) measures. A search for a regression case
that is simultaneously noncollinear, magnetic and gradient-corrected *and* takes a force,
a stress or a response turns up nothing. So the sign convention, the rotate-back at `:353`
and every term between them have **no measured derivative anywhere in the project**, and a
dropped term there would be invisible to every committed number.

**Second half, the NaN.** `potential.py:332` writes
`modulus = jnp.sqrt(jnp.sum(magnetization**2, axis=0))` bare. `d|m|/dm` is `0/0` at a
bit-exact zero, and the `jnp.where` at `:344-351` guards the *division* below it rather
than the sqrt's own argument, which is the mistake `OPEN.md` A5 (`:433-456`, closed
2026-09-11) diagnosed and fixed at the two sibling sites by masking the argument
(`safe_modulus`, `xc/functional.py:812`, used by `local_spin_frame` at `:860` and
`paw/gradient.py:197`). A5 measured the difference: `[nan nan nan]` before and
`[0. 0. 0.]` after, with the value unchanged. The guard test,
`tests/unit/test_gamma_basis.py:336`, asserts the absent literal for exactly two functions
and its name says "both", which is why the third and fourth were never seen -- and
`paw/gradient.py:195-196` claims "the same guard the plane-wave branch uses", which is
false for the GGA branch. A5's two mechanisms for a bit-exact zero are exactly the user's
cells: `sym_rho`'s axial average is *exact* at a grid point whose magnetic little group
admits no invariant axial vector, and a vacuum region underflows. A compensated magnet on
a symmetry-reduced grid, or any slab.

Two siblings are latent rather than live and should be fixed in the same pass without
being ranked with it: `potential.py:584` (`_noncollinear_meta_exchange`) sits behind
`reject_potential_only` (`forces/energy.py:243`), which refuses every derivative consumer
of a potential-only functional by name; and `forces/torque.py:78-79` guards with a
*global* scalar norm over the whole grid, which protects no individual point.
`driver.py:569` is a third bare sqrt and is correctly left alone -- it is jitted,
reporting-only, and never differentiated.

**Check.** Two, in this order. (a) `jax.grad` of `_noncollinear_gradient_correction` on the
`(3, 4)` magnetization `test_gamma_basis.py:317-322` already builds, whose column 1 is
bit-exactly zero in all three components; assert every entry finite. Repeat for
`_noncollinear_meta_exchange`, and extend the source-text assertion to
`defumat.scf.potential` and rename it, since it is four sites and not two. (b) The one that
matters more: a magnetic noncollinear PBE cell with a force against `pw.x`.
`h4-noncolin-force.in` is LDA; the input to generate is the same four hydrogen atoms with
`input_dft = 'PBE'`.

**Size.** Session for the guard and for (a). Phase for (b), because the `pw.x` reference
has to be generated and committed.

#### 4. QE divides `starting_magnetization` by the valence charge above 1, and this code does not

**FIXED 2026-09-12.** The per-species rule is transcribed, clamp included: on
`o-atom-lsda.in` (`Z_v = 6`) a written `2.0` now seeds **+2.00** mu_B where it seeded
**+12.00** and a `N_down` of **-3 electrons**. The trap was a *second copy* of the
padding in `spin_weights`, which the collinear seed and PAW's `becsum` both read, so
the first version of the fix moved the noncollinear seed only. **The card's unit was put to the
user and they chose Bohr magnetons**, which is what it was always documented as: a row of
`(0,0,1.0)` now seeds 1.0 mu_B on oxygen where it seeded 6.0, and the division happens on
the way into the *seed* only -- `System.local_moments`, which the filter and the constraint
read, stays in Bohr magnetons.

**What.** `pw.x` reads a starting magnetization at or above 1 as **Bohr magnetons** and
divides every species by its valence charge; below 1 it is a fraction of the valence
charge. Here it is always the fraction. So the same input file asks the two codes for
different physics.

**Evidence.** `~/apps/qe-7.4.1/PW/src/input.f90:1448-1449`:
`IF (ANY(ABS(starting_magnetization(1:nsp)) .ge. 1._DP)) starting_magnetization(1:nsp) = starting_magnetization(1:nsp) / zv(1:nsp)`
-- unconditional, outside both the `noncolin` block and the `constrained_magnetization`
`SELECT`; `:1476-1480` then clamps to `[-1, 1]` inside `CASE('none')`. Neither is
transcribed: `grep z_valence defumat/system/` is empty and `builder.py:88` documents "per
species, in `[-1, 1]`". So `starting_magnetization(1) = 2.0` is 2 mu_B in `pw.x` and
2 `Z_v` here. It is invisible to the committed benchmark suite because every test-suite
value is below 1.

**The card is the same item's second face.** `STARTING_MOMENTS` rows are documented as
Bohr magnetons in four places (`io/pwin.py:51`, `builder.py:94-95`, `builder.py:1593`,
`docs/features.tex:875-877`) and are consumed as the per-atom *weight* on that species'
tabulated atomic charge (`pseudo/potentials.py:59-74`, `:279-297`), so the seeded moment on
atom `a` is `w_a` times that atom's share of `Z_v` -- the same meaning
`starting_magnetization` has on the branch three lines below. Nothing range-checks the rows
(`builder.py:1605-1613`, `:1446-1465`). The *constraint* side is fine and should not be
"fixed": `get_locals` returns Bohr magnetons (`locals.py:220-241`) and `constraint_targets`
passes the rows through unchanged (`fields.py:202-205`), so
`constrained_magnetization = 'atomic'` aims at exactly the number the user wrote.

**Why it matters.** A physicist writing 2.0 for a 2 mu_B nickel site asks the code for
roughly 20. Usually the SCF forgets the seed magnitude and this costs iterations; for a
magnetic insulator with several minima the seed magnitude is precisely what decides which
basin the run lands in, which is the whole reason a seed exists. And on the per-species
path it is a straight disagreement with `pw.x` on the same input file.

**Check.** One-atom-per-site cell, a card row `(0, 0, 1.0)`, no constraint. Integrate the
magnetization channel of `Calculation.starting_density()` over the cell and compare
against 1.0: it will come out at `Z_v`. Then transcribe `input.f90:1448-1449` and
`:1476-1480`, decide one meaning for the card, and put it in all four documentation sites.

**Size.** Session.

#### 5. A per-atom texture never reaches the DFT+U occupation matrix or the one-centre `becsum`

**FIXED 2026-09-12.** `initial_ns_noncollinear` takes `per_atom` and its axis per
*slot*; `_becsum_split_per_atom` gives `starting_becsum` one row per atom. On two PAW
oxygens at 90 degrees the one-centre moments now follow the card -- `(0,0,1.5)` and
`(1.5,0,0)` where both were `(0,0,0.3)` -- and the sphere-integrated charge moment and
the `becsum` agree in direction to **7.5e-6** on both sites, where the per-species split
had them 90 degrees apart on one. The **collinear** regime needed the same fix and got it in the same pass -- item 1 made a
one-species antiferromagnet converge, which made a ferromagnetic `becsum` beside a
staggered charge newly reachable; on two collinear PAW oxygens at `+-1.5` the one-centre
moments are now `+1.5 / -1.5` where both were the same number.
The unmeasured claim "the SCF repairs it and nothing
is wrong at convergence" is **removed rather than disproved**; the run that would settle
it is in `PLAN.md` P77d's outstanding paragraph, and P77's readout makes it cheap.

**What.** A texture is stated once and consumed in three places -- the charge density, the
Hubbard occupation matrix, and a PAW or ultrasoft dataset's atomic `becsum`. Only the
charge sees it. The other two start every site of a species pointing the same way, so
iteration 1 contradicts itself.

**Evidence, DFT+U.** `Calculation.starting_ns` calls
`initial_ns_noncollinear(self.hubbard, self.starting_magnetization, self.system.angle1, self.system.angle2)`
-- three per-*species* arrays, never `system.local_moments` (`driver.py:1703-1708`).
`initial_ns_noncollinear` then loops `for slot, t in enumerate(setup.types)` and writes one
2x2 spin block into every slot of that species (`hubbard/occupations.py:308-339`). Its own
docstring states the consequence: "nothing in the SCF turns a moment, so a spinor DFT+U run
started with a moment along z on a species whose `angle1` points elsewhere converges with
the shell polarised along the wrong axis and reports success" (`:302-307`). Noncollinear
DFT+U is already forced to `nosym` (`driver.py:1552-1565`), so `angle1`/`angle2` are the
only steering there is and they cannot express a texture at all. There is an escape:
`run_scf` takes a `starting_ns` override (`driver.py:3764`, `:3987-4004`), reachable through
`Calculator.get_scf(**options)` -- but nothing builds a textured one and nothing warns that
`Calculation.starting_ns()` silently ignores the card.

**Evidence, `becsum`.** `_becsum_split` (`driver.py:3177-3192`) builds the noncollinear
split from `self.starting_magnetization[t] * self.magnetization_directions[t]`, indexed by
species, and `starting_becsum` broadcasts it over `len(atoms)` (`:3160-3175`). The docstring
at `:3081-3090` and the amber box at `docs/features.tex:937-942` both state the gap and both
assert, with no number attached, that "the SCF repairs it and nothing is wrong at
convergence; it costs iterations". This affects ultrasoft as well as PAW -- `_becsum_split`
runs for any species with projectors -- but the consequence is worse for PAW, whose
one-centre terms are a function of `becsum`.

**Why it matters.** Transition-metal magnets are the systems that need U and the systems
that ship as PAW or ultrasoft datasets, and they are the user's stated targets. For that
class the U term is the strongest thing in the first Hamiltonian and it is initialised
pointing the wrong way on every site but one. The unmeasured claim "nothing is wrong at
convergence" is also exactly the shape P75 disproved for the closely related case: a
magnetic SCF started inconsistently can settle somewhere else and report success.

**Check.** The gap needs no SCF: build a two-atom noncollinear cell with a `HUBBARD U` line
and a card of `(0,0,+m)`, `(0,0,-m)`, then assert
`local_moments[1] == -local_moments[0]` while `initial_ns_noncollinear(...)` returns
identical 2x2 blocks. The number that would settle the `becsum` half: the same two-site
PAW antiferromagnet run with the per-species split and with a hand-built per-atom seed,
comparing iteration count and converged per-site directions. Two outcomes are informative
and different -- the same state in fewer iterations confirms the docstring, a different
state refutes it.

**Size.** Session each. Both are the same fix: thread `System.local_moments` in as a
`per_atom` argument and build the axis per slot rather than per species.

#### 6. A seed field between 1e-12 and 1e-5 Ry makes a run magnetic and is invisible to the filter

**FIXED 2026-09-12.** `_axial_fields` now divides each field by its own largest
component, so the filter tests a *pattern* and has no scale in it, and drops a field only
below `_VANISHING_FIELD = 1e-12` -- `is_magnetic`'s own floor, so one rule decides both.
On the four-atom cycloid the group is now cut from 8 to 2 at every scale from 1e-3 down to
1e-12, where before 1e-6 left it at 8. The threshold was moved *out*, not moved down.

**What.** Two rules disagree about how small a magnetic thing has to be before it stops
counting, and the gap between them is seven orders of magnitude. A field inside it
switches the run to `nspin_mag = 4` and switches time reversal off, then contributes
nothing to the symmetry filter -- so the full crystal group survives and averages away
exactly the texture the field was applied to create.

**Evidence.** `is_magnetic` compares an applied field against 1e-12 and a moment against
1e-6, deliberately and with the reason stated ("a field is compared against 1e-12 because
it is an applied constraint rather than a guess", `builder.py:616-632`). The filter it
feeds uses `_MAGNETIC_TOLERANCE = 1.0e-5` (`symmetry.py:52-54`): `_axial_fields` drops a
whole vector field when every component is below it (`:307-308`), and even if the array
survived, the image comparison at `:369-375` tests `|rotated - images| < 1e-5`, which every
1e-6 field passes for both `same` and `opposite`, so no operation is cut either way. At
`|B| = 2e-5` Ry a 24-degree rotation moves the vector by 8.3e-6, inside the tolerance,
which makes the failure graded rather than sharp. The tolerance is documented as QE's
`eps2` for `m_loc`, a magnetization in `starting_magnetization` units, and is being applied
to a field in **Rydbergs**, a quantity whose scale the user chooses freely
(`builder.py:1567-1571`). Every P75 test probes at `scale = 1.0e-3`
(`tests/unit/test_textured_symmetry.py:79, 96, 112, 133`), two orders above the gap.

The window is narrow and worth stating: it is reachable only when the applied field is the
*sole* texture carrier -- every moment zero or all parallel -- and `1e-12 < |B| < 1e-5` Ry.
Post-P75 guidance routes a user to `STARTING_MOMENTS`, whose moments are of order 1 in Bohr
magnetons. The exposed user is one still seeding a helix the Elk way, with a field alone.

**Why it matters.** An infinitesimal symmetry-breaking field is the standard way to start a
texture, and it is what the NiBr2 run used. Choose it small enough not to bias the energy
and the group stops seeing it, which reproduces the exact failure the card was written to
fix.

**Check.** Rerun `tests/unit/test_textured_symmetry.py::test_a_textured_field_cuts_the_symmetry_group`
with `scale = 1.0e-3` replaced by `1.0e-6`; the assertion `textured.nsym < ferro.nsym`
must fail. That is `CLAUDE.md`'s "test that the guard fires" and it costs no SCF. The fix
is to compare each field against a tolerance relative to its own largest component --
`_distinct_directions` (`builder.py:1616-1632`) is already scale-free and has no caller in
the package -- or to pass `is_magnetic`'s two thresholds down so one rule decides both.

**Size.** Session.

#### 7. A run can converge with `reducebf`'s symmetry-breaking field still on, and nothing says so

**FIXED 2026-09-12.** A `RuntimeWarning` at the end of `run_scf` naming `field_scale`,
the residual and `field_energy`, above `FADED_FIELD = 1e-4` Ry and only for
`reducebf < 1`. The size is now measured rather than bounded: on the H atom at
`reducebf = 0.99` the run stops after **six** iterations with 9.5e-2 Ry of field still on
and a total **4.7e-5 Ry** out, while `reducebf = 0.5` stops at 6.1e-6 Ry and is right to
5e-14. The error is second order in the residual, which is why the guard is on the field
and not on `field_energy` -- that is first order and is 6e-6 Ry in the clean row.

**What.** `reducebf` applies a field, lets it decay geometrically and keeps whatever is
left. The decay is applied *after* the convergence test, and nothing requires the field to
be small before the run stops -- so the state that is reported is the ground state of a
functional that includes a residual Zeeman term whose energy is, by convention, not in the
reported total.

**Evidence.** `converged = accuracy < conv_thr` at `driver.py:4278`, the break at `:4429`,
and `field_scale *= field.reducebf` at `:4470`, after it. The only extra gate is
`field.satisfied(...)` (`:4281`, `fields.py:449-459`), which returns True for every scheme
except `fsm`. The code's own docstring sizes it: "after ~25 iterations at 0.9 it is 7% of
its input value" (`driver.py:851-859`). The field's energy is excluded from
`total_energy` by QE's and Elk's shared convention (`fields.py:27-35`). The rest of the
package already treats this state as dangerous -- `sternheimer.py:1130-1156`,
`workflows/nscf.py:204`, `workflows/topology.py:255` and `scf/checkpoint.py:142` all refuse
a ground state converged under a field -- and `run_scf` is the one place that does not.

Two corrections, both of which shrink the claim without removing it. The residual field is
**reported**, on `SCFResult.field_energy` and per iteration in `history`
(`driver.py:4388-4391`), so it is not hidden; and the error in `total_energy` relative to a
field-free run is second order in the residual field, because the state is stationary. The
only bound on record is the `> 1e-5` Ry that the held-field control measures at a *full*
0.1 Ry field (`tests/regression/test_magnetic_constraints.py:200`). Also, 0.9 is nobody's
default: Elk's is 1.0, restricted to `[0.5, 1]` (`~/apps/elk-9.6.8/src/readinput.f90:219`,
`:1264`), and this code's is 1.0 (`builder.py:170`) and is accepted at any value
(`builder.py:963`) where Elk refuses outside that range.

**Why it matters.** This is the user's route into an antiferromagnet or a canted state from
an unmagnetised start, and the one entry point that does not check it is the one a script
drives.

**Check.** Warn -- or refuse -- when a run stops with `field_scale * |B|` above about 1e-6,
naming `field_scale` and `field_energy`; the shape is the one `MagneticField.satisfied`
already has for `fsm`. Measure the size once by running the committed H-atom case at
`reducebf = 0.95` instead of 0.5 and comparing `total_energy` and `magnetization` against
the properly seeded magnetic run.

**Size.** Session.

### Tier 2 -- a workflow that does not run

#### 8. DO FIRST: `constrained_magnetization = 'atomic texture'` crashes unless the input also carries a per-atom field

**FIXED 2026-09-12 (P78).** `fields.ATOM_RESOLVED`, one set beside `CONSTRAINTS`, replaces
the hand-written tuple, and `sphere_moments` refuses by name when the spheres are absent.
`tests/data/qe/h2-texture-120.in` is the input that could not run -- two hydrogen atoms of
one species asked to sit 120 degrees apart, no field card -- and the test reaches
`constraint_energy`, since constructing the `Calculation` succeeded on the broken code too.

**What.** The one constraint in the package that can tell a cycloid from the collinear
state raises `AttributeError` on the first potential build, for any input that does not
also happen to set `LOCAL_MAGNETIC_FIELDS`.

**Evidence.** `driver.py:1768` reads
`needs_regions = atomic is not None or constraint in ("atomic", "atomic direction")`.
`"atomic texture"` is not in that tuple, so `regions = None` (`:1769-1776`) and the
`MagneticField` is constructed with `regions=None` (`:1789`). `fields.py:331-338` then
evaluates the texture penalty through `self.local_moments(rho_r, cell)`, whose body is
`jnp.einsum("anmk,cnmk->ac", self.regions.weights, ...)` (`fields.py:292-296`). Nothing
validates `regions` on construction. **Confirmed by a second read of both lines during
assembly**, which makes this one of two claims in this file with two independent reads behind
it. It escaped because the only tests build
`MagneticField` directly with regions supplied
(`tests/unit/test_textured_symmetry.py:207-215, 245-320`), and because P75's production run
set 45 per-atom fields, so `atomic is not None` built the regions for it.

**Why it matters.** `'atomic texture'` is documented and parsed (`builder.py:722-738`,
`fields.py:83`), it is what `docs/features.tex` points a user at, and it is the constraint
that gives 0.0 for a cycloid against 4.0 for the collinear state where QE's
`'atomic direction'` gives 0.0 against 0.0. It also gates three of the measurements in
section 6.

**Check.** Add `"atomic texture"` to the tuple. The test that must fail against the old
code is the construction of a `Calculation` with that constraint and no
`LOCAL_MAGNETIC_FIELDS` card.

**Size.** One line plus the test.

#### 9. `promote_ns` refuses every DFT+U continuation into `nspin = 4`, naming a blocker P62b removed -- and it breaks the noncollinear checkpoint resume

**FIXED 2026-09-12 (P79).** `1 -> 4` and `2 -> 4` put the collinear channels into the two
diagonal spin blocks with the off-diagonal ones zero, which is the shape
`initial_ns_noncollinear` builds for a moment along z; `4 -> 4` is a pass-through, and it is
the checkpoint resume. Coming back keeps the diagonal and **refuses by name** above a
transverse block of 1e-8, because the diagonal of a canted occupation matrix is a different
state and not a coarser one.

**The end-to-end test found a second defect the array-algebra one could not.**
`driver.py`'s `starting_ns` branch pushed the matrix through `precision.as_real`, which for
a spinor `ns` discards the imaginary part -- the off-diagonal spin blocks, where a canted
shell lives. Every `run_scf(starting_ns=)` and every noncollinear DFT+U resume came back
with the shell rotated onto the collinear axis, converged, and silent; NumPy's
`ComplexWarning` was the only trace. Measured, on nickel with `U = 4`: the resume now
reproduces an uninterrupted run, and a collinear state promoted into a spinor run converges
in **4 iterations against 78** from scratch, agreeing with its collinear source to 6e-9 Ry.

**What.** Converging a hard magnet is staged: get a collinear ferromagnet or
antiferromagnet, then promote it and let the moments cant. That route is closed for
anything with a `HUBBARD` card.

**Evidence.** `defumat/scf/continuation.py:449-453` raises "a Hubbard U in a noncollinear
calculation needs `ns_nc`, which is refused by name (PLAN.md P20); drop the HUBBARD card or
stay collinear", gated on the *target's* `nspin == 4` alone, so 2 -> 4 and 4 -> 4 both fire.
`ns_nc` is not refused any more: `PLAN.md:9614` heads P62b "DONE, both functionals",
measured at 1.2e-7 Ry in the total energy and the Hubbard term on relativistic BN with
`noncolin` and `lspinorb`, with `Tr[ns]` 2.14903/2.14903 against every digit `write_ns`
prints (`PLAN.md:9655-9662`). The raise fires only when the source result carries an `ns`
(`continuation.py:445-447`), so a non-Hubbard promotion is unaffected.

**The 4 -> 4 branch is the checkpoint resume.** `checkpoint.py:174-192` rebuilds an
`SCFResult` carrying `ns`, `driver.py:3905` assigns it to `starting_from`, and `:3925-3926`
sends it through `continued_state`. So a wall-clock-killed noncollinear DFT+U run cannot
resume from its own `checkpoint_dir` -- which is exactly the long run P76 was written for.

**Check.** The 2 -> 4 promotion rule is the density's own, "decompose, decide, recompose":
the two collinear channels become the two diagonal spin blocks with the off-diagonal blocks
zero. The 4 -> 4 case is a pass-through. The test is P62b's collinear-as-spinor identity
reached through `run_scf(starting_from=<the collinear result>)` instead of from scratch;
the total energy and `Tr ns` per channel must match the from-scratch spinor run to the SCF
threshold.

**Size.** Session.

#### 10. Nothing holds a texture that is not the ground state

**MEASURED AND LARGELY FALSE, 2026-09-12 (P79).** By this item's own criterion -- "if the
angle is off by more than a degree at the largest `lambda` the SCF tolerates, the penalty is
not holding it" -- the penalty **is** holding it: **0.55 degrees**. On two hydrogen atoms of
one species at 120 degrees, `constrained_magnetization = 'atomic'` converges at every
`lambda` up to 10 and at `lambda = 10` holds the angle to 121.13 degrees in 38 iterations,
where the unconstrained run collapses to the collinear antiferromagnet (180 degrees) in ten
and reports success.

The surprise is *which* scheme. `'atomic texture'` -- the direction-only one this file
recommends -- has a `1/|m|` in its gradient, so a shrinking moment is amplified rather than
damped: **no** `lambda` converged, and above 2 one site blew up to 2 mu_B while the other
went to zero. `'atomic'` constrains the vector, so its gradient is bounded. The full table is
in `PLAN.md` P79 and `docs/features.tex`, and `'atomic texture'` now warns and names it.

**What stands.** A penalty leaves a residual at convergence by construction, and 0.55 degrees
is that residual. Elk's per-atom feedback field (`bfieldfsm.f90:50-73`) would converge to a
genuine stationary point instead, and is still worth writing -- but it is now an improvement
on a working route rather than the only route. Nothing has been compared against Elk.

**What.** Elk fixes a moment per muffin tin: `mommtfix(:, ia, is)` with `fsmtype = 2` or
`3`, updating one field per atom, and a negative `fsmtype` fixes the *direction* alone by
projecting the field perpendicular to the target
(`~/apps/elk-9.6.8/src/bfieldfsm.f90:32-73`, with a `t1 >= 1000` per-atom skip sentinel).
Here there is nothing per atom: `constraint_targets` returns a single 3-vector for both
`'total'` and `'fsm'` (`fields.py:231-239`) and `MagneticField.feedback` drives one uniform
field from `total_moment` (`:380-410`). `grep mommtfix|bfsmcmt` over `defumat` returns
nothing. What exists instead is QE's *penalty* per atom -- `'atomic'`, `'atomic direction'`
and this code's `'atomic texture'` -- which leaves a residual force at convergence by
construction (`fields.py:385-388` says so), where a feedback field converges to a genuine
stationary point of the unconstrained functional under that field.

**Evidence that it bites.** Already measured and written down: "on fcc hydrogen the
90-degree spiral at `q3 = 1/2` converges to `|m| = 0.0001`, the nonmagnetic solution ... a
15-degree cone collapses the same way, because nothing holds it. Elk's own magnon-spiral
example runs `fsmtype = -1` with a large field for exactly that reason"
(`docs/features.tex:1878-1884`, repeated with more numbers at `PLAN.md:10140-10152`).

**Why it matters.** This is the difference between being able to *state* a 120-degree Neel
state, a cone or a canted configuration and being able to *converge* one. A user mapping
out `E(theta)` gets the energy of whatever the run relaxed to, not of the configuration
they asked for.

**Check.** After item 8: a two-sublattice cell under `'atomic texture'` with a 120-degree
card, converged, then the angle between the two converged local moments and the residual
penalty at convergence. If the angle is off by more than a degree at the largest `lambda`
the SCF tolerates, the penalty is not holding it and the per-atom feedback is the fix --
`bfieldfsm.f90:50-73` transcribed into the machinery `'atomic'` already has.

**Size.** Phase.

#### 11. No noncollinear linear response, so a magnet with spin-orbit coupling has no phonons and no spectra

**What.** `sternheimer.py:1122-1126` refuses the whole Sternheimer stack for `noncolin`,
naming `incdrhoscf_nc` and `set_int3_nc` as "a second implementation rather than a spin axis
on this one". Taken at face value that overstates it, and `CLAUDE.md` already records the
cost of taking such a refusal as read: P35's was a statement about the Sternheimer stack and
never applied to a sum over states, and inheriting it left P54's whole quantity marked
impossible.

**What exists.** The induced density's spinor form is `spinor_sum_band`
(`defumat/scf/density.py:407`), needing only the cross term per Pauli component;
`set_int3_nc`'s input is `spinor_becsum` (`:422`), and `int3` is by this project's own rule
one `jvp` of `newd`, whose noncollinear form is `Calculation._noncollinear_coefficients`
(`driver.py:2556`); the occupied-band count is already spinor-aware -- `sternheimer.py:981`
reads `degeneracy = 1 if calculation.noncolin else 2`, which is a striking thing to find
behind a blanket refusal. The XC kernel exists too: it is a `jvp` of `potential.py:227` and
`:304`, which have the `nspin = 4` branch. What it lacks is item 3's guard. These are
pieces to wire in rather than pieces already reachable: `density_at`
(`sternheimer.py:617-636`) hardcodes the collinear `sum_band`.

**What is missing and is on nobody's list.** The response's symmetrisation.
`Calculation.symmetrize_directional` (`driver.py:2641-2676`) applies a plain cartesian
rotation to the perturbation-direction axis and treats every `nspin_mag` channel as a
scalar. At `nspin_mag = 4` three of those channels are the magnetization and need `det(R)`
and the time-reversal sign, which `symmetrize_magnetization` (`symmetry.py:402-424`) has and
this method does not call. Its own docstring says an induced charge density is polar "where
a magnetization is an axial one, and applying the wrong one is a different symmetry rather
than a worse average". That is `CLAUDE.md`'s trap verbatim, and it is a landmine for whoever
acts on this item.

**Why it matters.** Everything above the solve is closed for a spinor: phonons, Born
charges, the dielectric constant, LO-TO splitting, Raman, the strain response, the elastic
constants, electrostriction, the piezoelectric tensor, and the cheap route to the
magnetoelectric tensor -- `Calculator.get_magnetoelectric_tensor` says so in its own
docstring, doing six SCF runs and a central difference because "the cheap one needs a
noncollinear Sternheimer solve".

**Check.** First milestone, no new physics beyond the density: `chi_0` under a potential
probe against a central difference of the density, on `tests/data/qe/h-chain-90deg.in` --
four noncollinear atoms, norm-conserving, so `set_int3_nc` does not arise and the test is
`spinor_sum_band` plus the existing CG. That is the check P45 used to close `nspin = 2`. It
exercises the kernel at a node, because a 90-degree texture has grid points where `m` passes
through zero, so it will fail on item 3 until that is fixed -- which is the right order.

**Size.** Phase.

#### 12. Magnons refuse a noncollinear ground state, so the states whose excitations are interesting have none

**Evidence.** `defumat/tddft/spinchi0.py:229-235` refuses `system.noncolin` with a correctly
named reason -- the 4x4 spin-density response no longer block-diagonalises, so the
transverse channel is not a matrix in `(G, G')` on its own, and Elk's `genspchi0` carries all
sixteen blocks. `:237-243` then requires `nspin == 2` and `:221-228` requires
norm-conserving. Both `Calculator.get_spin_susceptibility` (`calculator.py:1063`) and
`get_magnon_dispersion` (`:1086`) route there. `PLAN.md:9979` heads P63 "DONE, collinear and
norm-conserving". This is a genuine refusal, not a stale one.

**Why it matters.** A 120-degree Neel state on a triangular lattice, a helix, a cycloid, a
canted antiferromagnet: their magnons are the interesting physics, and the code can compute
the ground state and not the excitations of it. The norm-conserving requirement additionally
rules out a collinear NiBr2 magnon.

**Check.** The bounded first step is not the sixteen blocks. For a **spin spiral** the
generalized Bloch theorem already reduces the problem: the transverse channel at wavevector
`q` is the collinear machinery in the rotating frame, which is Elk's `spinsprl` route. So
`chi^{+-}` for a spiral ground state at its own `q`, validated against the collinear
antiferromagnet of the doubled cell at `q = (0, 0, 1/2)` -- the same identity P19 used for
the spiral energy. It is refused today at `spinchi0.py:246` for its own reason (two spheres),
and it is the cheaper half of the problem.

**Size.** Phase.

#### 13. One missing matrix gates noncollinear DFT+U with symmetry and the symmetrised spinor PDOS

**Evidence.** `driver.py:1552-1565` refuses DFT+U with `noncolin` under symmetry because
`new_ns_nc` averages the occupation matrix with the SU(2) representation of each operation
(`d_spin_ldau`) beside the rotation of the `m` indices, "and nothing here builds those
matrices"; `projwfc/projections.py:203-214` refuses a symmetrised spinor projection for the
same object (`sym_proj_so`) and points at the first. A grep for `d_spin`, `SU(2)` or
`spin_rotation` over `defumat` finds only the refusals. A second refusal covers any group
carrying `t_rev` even in the collinear case (`driver.py:1572-1580`), because `new_ns` flips
the spin index for such an operation and that branch was never exercised -- the benchmarks'
two magnetic sublattices are different species
(`hubbard/occupations.py:165-176`).

**A third refusal looks like the same object and is not.** `angular_momentum.py:263-272`
refuses `<L>`/`<S>` on a reduced k-set, and what it needs is the *axial* 3x3 group average
plus the atom permutation, not an SU(2) spin rotation. The axial half is written:
`magnetization_signs` (`symmetry.py:390-399`) is `det(R) (-1)^t_rev`,
`symmetrize_magnetization` (`:402-424`) applies it, and `symmetrize_atom_cartesian_tensor`
(`:906-939`) does the per-atom average at any rank -- though it reads
`symmetries.rotation_array()` internally, so the axial variant is a new parameter rather
than a call with signed rotations passed in.

**Why it matters.** It is a cost multiplier on runs that are already expensive. There is an
honest nuance: a genuine texture's magnetic group is often small anyway (P75 measured a
four-atom test cycloid dropping from `nsym = 4` to 1), so `nosym` costs little there. Where
it bites is the collinear-as-spinor case and the ferromagnet with spin-orbit coupling, which
keep a large group and are the two commonest noncollinear runs.

**Check.** Build `d_spin_ldau` once (QE's `PW/src/d_matrix.f90` and `ldaU.f90`) and check it
two ways before wiring it anywhere: the representation property
`D(R1) D(R2) = +- D(R1 R2)` over the whole group, and P62b's identity -- a spinor `ns` with
the moment along z, symmetrised, must equal the collinear `ns` symmetrised, to 1e-13. Then
the wiring check is the wedge-versus-closed-grid comparison
`tests/regression/test_spinor_forces.py:331` already makes for the force.

**Size.** Phase, and it closes three consumers at once.

#### 14. A texture can only be stated in an input file, and the obvious Python workaround is silently wrong

**FIXED 2026-09-12 (P78).** `System.with_moments` and `Calculator.with_moments`, built
exactly as `with_spin` is. The hazard is measured on `tests/data/qe/h4-chain-ferro.in` and
needs no SCF: the ferromagnet has `nsym = 16` and 9 k-points, `dataclasses.replace` leaves
those 9 while `symmetry_group()` recomputes to 4, and `with_moments` rebuilds to 12 at
nsym = 4. Dropping the card restores 9 at 16.

**Evidence.** `starting_moments` is assigned in exactly one place, `build_system`
(`builder.py:947`). `System.with_spin` takes `nspin`, `lspinorb`, `starting_magnetization`,
`angle1`, `angle2`, `nbnd` and nothing else (`builder.py:338-346`), and
`Calculator.with_spin` forwards `**options` straight into it (`calculator.py:1407-1417`), so
`starting_moments=` is a `TypeError`. `_respin_kpoints` already passes
`per_atom=self.starting_moments` and rebuilds the k-set (`builder.py:500-514`) -- the
machinery exists with no method reaching it.

This is clumsy rather than impossible: `Calculator.from_text(text, pseudo_dir)`
(`calculator.py:402-412`) means a Python sweep over magnetic configurations needs string
formatting, not a round trip through disk. **The hazard is the workaround.**
`dataclasses.replace(system, starting_moments=...)` leaves `system.kpoints` reduced with the
*old* group, while `System.symmetry_group()` is a property recomputed from `axial_fields`
(`builder.py:596-601`) and read fresh at `driver.py:1409` -- two lines below a comment
asserting the agreement it cannot check. Adding a texture that way makes the symmetrisation
group *smaller* than the group the k-set was reduced with, and on a system built nonmagnetic
it also flips `domag`, so the k-set was built with `time_reversal = True`, which a magnetic
run must not have.

**Check.** Add `with_moments(per_atom)` to `System`, built exactly like `with_spin`: replace
the field, rebuild the k-points through `_respin_kpoints`, and keep the collinear x/y
refusal `local_moments` already makes. The test showing today's hazard needs no SCF: take a
ferromagnetic noncollinear `System`, `dataclasses.replace` a cycloid card into it, and assert
`system.kpoints.nk` is unchanged while `system.symmetry_group().nsym` has dropped.

**Size.** Session.

#### 15. Spin spirals have no external number of any kind, and two measured failure modes are unguarded

**What is validated.** Five identities against calculations that are not spirals, all on a
one-atom hydrogen chain: `q = 0` against the ordinary noncollinear run (4e-15 Ry),
`q = b3/2` against the collinear antiferromagnet of the doubled cell (7e-13),
`q = b3/4` against a four-cell noncollinear supercell (3e-12), plus `E(-q) = E(q)` and
`E(q + G) = E(q)`; the committed tests assert 1e-9 on all of them (`PLAN.md:2019-2031`).
P21's `dE/dq` adds four more on the same chain. There is no `pw.x` counterpart and **no Elk
number was ever taken**, although Elk implements the same ansatz (`gengkqvec.f90`, `vqlss`)
and was built here from the vendored source (`PLAN.md:8818-8820`).
`PERFORMANCE.md:1143-1144` states "there is no QE column because `pw.x` has no spin spiral to
time against" and has no Elk column either, which is the gap `CLAUDE.md`'s rule leaves open.

**The two unguarded failures.** `E(q + G) = E(q)` is an exact identity and fails by
**2e-3 Ry on a 1x1x3 grid**, because the k-set is not invariant under a shift by `G/2`
(`PLAN.md:2042-2045`; 2e-9 on 1x1x4) -- and nothing checks a user's grid. At
`degauss = 0.02` a spiral and its equivalent supercell converge to **different minima** and
disagree in the fourth decimal, agreeing to 1e-10 only at `degauss = 0.1`
(`PLAN.md:2052-2054`). `defumat/system/spiral.py` contains no `raise` and no `warn`, and the
spiral amber box (`docs/features.tex:3583-3590`) refuses spin-orbit coupling, symmetry,
ultrasoft/PAW and a field, and says nothing about either. A user scanning `E(q)` on a soft
magnetic surface can get a Heisenberg fit that is entirely the sampling.

**One more thing nobody has run.** A spiral SCF does *not* refuse a field or a constraint --
the refusals in `Calculation.__init__` cover symmetry (`driver.py:1424-1440`), ultrasoft/PAW
(`:1233-1246`), DFT+U (`:1566-1570`), stress (`:2056-2060`) and meta-GGA (`:1513-1516`), and
only the spiral *relaxation* refuses a field. So a user can and should hold a spiral, which
is item 10's fix for the collapse. But the quantity a constraint acts on is the
**rotated-frame** magnetization (`system/spiral.py:17-24`), which happens to be exactly the
right thing to constrain for a helix and is documented nowhere near
`constrained_magnetization`. A grep of `tests/` and `notebooks/` for a spiral together with a
field or a constraint returns nothing.

**Check.** Three, independent. (a) Build Elk and run the same hydrogen chain at three
wavevectors as an Elk spin-spiral ground state, comparing total energies after subtracting
each code's own reference -- that converts the feature from a set of identities to an
external comparison and is the only way to catch an error the identities share. (b) Make the
`G/2` invariance of the k-grid a checked precondition of `at_spiral_q`, with the 1x1x3 case
as the input that must trip it. (c) Run the fcc hydrogen 90-degree spiral at `q3 = 1/2` under
`constrained_magnetization = 'fsm'` with `fixed_magnetization` at the collinear `|m|`, against
the `|m| = 0.0001` baseline.

**Size.** Phase.

### Tier 3 -- cost, comfort and measurement debt

#### 16. `mixing_ndim` is parsed and silently ignored

**FIXED 2026-09-12 (P78), and the measurement says the knob is not the cure this item
assumed.** It reaches the mixer through `run_scf`, the facade and all three relaxation
drivers. On `fe-mag-1k.in` at `conv_thr = 1e-8`: 27 iterations at `ndim = 4`, **25 at 8**,
33 at 12, 39 at 20 -- the default is the best value. The deconfounder
`fe-unstable-nonmagnetic.in` has the same shape (26, **21**, 30, 26), so the sensitivity is
not magnetic and neither is most of the gap to `pw.x`'s 12: the nonmagnetic twin takes 21
here. That is the first measurement against item 17 and it points away from it.

`AndersonMixer.history` is fixed at 8 (`mixing.py:103`); `driver.py:3950` is
`get_mixer(mixing_mode, beta=mixing_beta)`; `run_scf` has no depth parameter
(`driver.py:3748-3776`); and `mixing_ndim` occurs **nowhere** in `defumat/` -- not in
`io/pwin.py`, which has no whitelist, not in `calculator.py`'s `_ELECTRONS_OPTIONS`
(`:195-212`), and not in `_REFUSED_SWITCHES` (`builder.py:1049-1128`). Raising `mixing_ndim`
is the first thing a QE user does to a magnetic cell that will not converge. Two mitigations:
`get_mixer(name, **kwargs)` already forwards keywords (`mixing.py:543-547`), so this is one
argument plus a signature; and 8 is `pw.x`'s own default, so an input that does not set it
behaves identically. **Session.**

#### 17. The mixer's least-squares metric is Euclidean where `mix_rho.f90` uses `rho_ddot`, and the one magnetic benchmark is 2.7x off

`benchmarks/fe-mag-1k.in` takes **32 SCF iterations where `pw.x` takes 12** at the same
`conv_thr = 1e-8`, with the two energies agreeing to 6.7e-9 Ry, reproduced on a second
machine against QE 7.4.1 (`OPEN.md:1331-1364`). Nine of the ten fast benchmarks -- a metal,
an ultrasoft cell, a PAW cell, a spinor cell -- match or beat `pw.x`'s count; only the
magnetic one is off. Three structural differences, all on the extrapolation rather than the
step. **Metric:** `AndersonMixer.mix` builds its Gram matrix as a flat Euclidean form on the
packed real-space vector (`mixing.py:156-186`) where QE uses
`rho_ddot` (`mix_rho.f90:318`, `:351`), which on this cell weights the lowest-G charge
residual 8.66 against the magnetization's 0.637, a factor of 13.6, crossing at
`|G| = 2 pi bohr^-1`. The package already has `rho_ddot` -- `scf_accuracy`
(`potential.py:126-167`) -- and uses it only for the convergence test. **Support:** QE
extrapolates only over the smooth sphere (`mix_rho.f90:121`, `ngm0 = ngms`) and linearly
mixes above it; this code extrapolates the whole dense grid, a deliberate and documented
choice (`driver.py:4203-4208`, "the conservative direction"). **`becsum`:** `_mix`
(`driver.py:399-423`) packs it unconditionally where QE allocates `rho%bec` only
`IF (okpaw)` (`scf_mod.f90:255`), and this cell is ultrasoft.

Causation is a hypothesis and nothing here measures it. The confound is named and the
deconfounder is committed: `fe-mag-1k.in` is the only one of the ten that sets
`mixing_beta = 0.3`, so beta and magnetism vary together, and
`benchmarks/fe-unstable-nonmagnetic.in` is the same cell, same dataset, same beta,
`nspin = 1`. **Phase.**

#### 18. `dr2` folds the charge and the magnetization into one scalar

**FIXED 2026-09-12 (P78).** `scf_accuracy_terms` returns the two halves; `run_scf` puts
both in `history` and prints them per iteration. It found something the same day: on the
nickel orbital-moment test `dr2 = 9e-11` while the total energy was **1.15e-8 Ry** from
converged, a hundredfold, and three runs of the same cell at three orientations stopped at
states 1.15e-8 apart -- exactly the weighting this item predicted. One trap: computing
`accuracy` **as** the sum of the halves differs from the fused expression by one ulp
(2.2e-16 relative, measured), and `accuracy` drives the `ethr` schedule, so the split is
computed separately and the reported total is still the fused value.

`scf_accuracy` computes the charge Hartree term and the magnetization term separately and
returns their sum (`potential.py:150-167`); `residual` is a max over all channels at once
(`driver.py:612`). `|m|` is already logged per iteration
(`driver.py:4402-4406`); the *split* is not, and it is two lines inside a function that
already has both halves. `OPEN.md:1352-1357` names it as H9's first step and it has not been
taken. It is the cheapest instrument that would settle items 2 and 17 and half of item 1 at
once, and it has a consequence a user should see: `rho_ddot` weights the magnetization 13.6x
less than the charge at `G_min` on `fe-mag-1k`, so a `dr2` below `conv_thr` bounds the moment
much more weakly than it bounds the charge. **Session.**

#### 19. Two silences at the ends of a run

**FIXED 2026-09-12 (P78).** Both. `noncolin` with `angle1`/`angle2` and no
`starting_magnetization` is refused by name; a spinor run with nothing magnetic in it stays
accepted, which is the discrimination that matters. And `run_scf` warns when it returns
unconverged -- a warning rather than a raise, because a deliberate `max_iterations = 1` is
legitimate and eight committed tests do it.

**A noncollinear run with angles and no magnitude is nonmagnetic and says nothing.**
`build_system` refuses `nspin = 2` with no `starting_magnetization`, with a paragraph
explaining that nothing in the SCF breaks spin symmetry on its own, and the condition is
`nspin == 2` (`builder.py:773-790`, `:775`). So `noncolin = .true.` with `angle1(1) = 90` and
no magnitude gives zero moments (`builder.py:1466-1485`), `domag = False`, `nspin_mag = 1`:
an unpolarized run with spinor wavefunctions, converged, with no warning. `pw.x` behaves the
same way, and `GAPS.md:172-183` records the sibling case -- a field in a nonmagnetic
noncollinear run -- as closed by a *named refusal*, on the reasoning that `pw.x`'s silence is
worse. The precedent exists; the refusal does not. **Session.**

**`run_scf` returns an unconverged result with no warning.** The loop falls out at
`driver.py:4126` and constructs `SCFResult(converged=False, ...)` at `:4558`; `verbose`
defaults to False, so by default nothing is printed either. QE prints "convergence NOT
achieved after N iterations: stopping" and exits non-zero. `Calculator._ground_state` raises
with a good message naming the accuracy (`calculator.py:562-568`), so the facade is guarded
and the functional entry point is not -- and a hard magnetic cell driven from a script (a
spiral scan, a lambda ramp, a moment-versus-field sweep) is exactly what uses `run_scf`.
`docs/features.tex:584` promises "hitting the cap is reported as not converged rather than as
an answer", which is true of the attribute and of nothing the user sees. **Session.**

#### 20. `constrained_magnetization = 'total direction'` returns a NaN potential when the moment lies along z

**FIXED 2026-09-12 (P78).** The angle is now `atan2(|m_perp|, m_z)` with **both**
arguments masked at the value the derivative is taken at. Off the axis the gradient is QE's
`fact1` to 1e-12; on it, zero, plus QE's literal `1.D-14` escape along x written as the
energy term whose derivative it is. Two mechanisms were live, not one: the diverging
`arccos'`, and -- at a *round-off* transverse moment of 6.3e-16 -- `m_z/|m|` rounding to
bit-exactly 1.0 so the clamp halved the tangent. Which fires is rounding, which is why the
guard is QE's threshold and not a test for zero.

`fields.py:350` computes `angle = jnp.arccos(jnp.clip(_polar_cosine(moment[None])[0], -1, 1))`
inside `constraint_energy`, and `:373` takes `jax.grad` of that same function to build the
potential. `arccos'` diverges at `+-1` and `jnp.clip` hands each argument half the tangent, so
the chain gives `-inf` and the VJP of `m_z/|m|` then produces NaN in all three components of
the potential's vector part. `_polar_cosine` is `m_z/|m|`, bit-exactly 1 whenever
`m_x = m_y = 0` -- the state a run seeded from `starting_magnetization` with no angles starts
in. The one test (`tests/unit/test_magnetic_fields.py:180-203`) uses a generic random density
with `m_perp != 0` and never reaches the boundary. **QE guards the same place and it was not
carried over**: `add_bfield.f90:185-192` tests `mperp < 1.D-14`, zeroes the transverse factors
and adds a `1.D-14` kick along x "in order to allow the magnetization to rotate", with an
`errore` below `1.D-12`. Mask the argument `arccos` is taken at, the way `_safe_modulus` does.
**Session.**

#### 21. The fixed-spin-moment secant assumes a diagonal susceptibility, and nothing enforces its scope

**FIXED 2026-09-12 (P78), the refusal half.** `constrained_magnetization = 'fsm'` with
`lspinorb = .true.` is refused by name at input. The 3x3 secant is still a phase and is not
written. The print-width half was closed by P77.

`_secant_step` measures `chi = response / change` elementwise per cartesian component and
takes `secant = -error / chi` the same way (`fields.py:429-438`), modelling `dm_a/dB_b` as
diagonal. The docstring states the scope -- "the components of a uniform field do not mix in
the cases this scheme is for" (`:414-417`) -- and spin-orbit coupling or magnetocrystalline
anisotropy breaks it, which is every heavy-element magnet this package advertises. Nothing
enforces it: `fsm` with `lspinorb` is accepted silently (`builder.py:1557-1565` checks only
the scheme name), and the only committed `fsm` case is collinear bcc iron. Note also that
`FSM_TOLERANCE = 1e-3` gates convergence while the noncollinear console print is `{:6.3f}`, so
a noncollinear `fsm` run cannot resolve the digit that decides it (item 2). **Session for the
refusal, phase for a 3x3 secant** -- a one-step Broyden update on the `(B, m)` pairs, the same
construction already used one level up in the density mixer.

#### 22. The noncollinear derivative memory wall has not been re-measured since two memory phases moved it

`PLAN.md` P46 (`:6353-6362`) records a bismuthene spinor force taking free memory from 24 GB
to 0.65 GB and being killed three times on this 30 GB machine, because the augmentation table
`Q_ij(G)` -- `nh^2 x ngm` per atom, with `nh` in the twenties for a fully-relativistic dataset
-- is live through the backward pass. That predates P73's radial table and P74's band
batching, which took the 45-atom NiBr2 noncollinear PAW cell from 78.51 GB to 32.30 GB peak
and ran two SCF iterations in 5 m 33 s where nothing had ever completed one
(`PERFORMANCE.md:4502-4503`). The two data points that stand are three orders of magnitude
apart: doubled fcc platinum (204 bohr^3, ultrasoft, `lspinorb`) runs a spinor force in 33 s,
and bismuthene (1770 bohr^3) was killed at a commit before both phases.
`Calculator.estimate()` answers only for the SCF and says so
(`sizing.py:29-32`, `calculator.py:338-359`), which is the "state, measure, make selectable"
rule being followed rather than a defect; what is missing is any post-P73/P74 number. See
O10. **Session for the measurement, phase for a derivative term in `SizeEstimate`.**

### Smaller items, one line each

- `builder.py:1603` tells a reader that a textured run with the wrong group is caught by
  `_refuse_untextured_symmetry`, a function that **does not exist anywhere in the tree**, and
  `tests/unit/test_textured_symmetry.py:15` says "the refusal fires on that shape" when there
  is no refusal. `_distinct_directions` (`builder.py:1616-1632`) is the diagnostic such a
  refusal would need and has no caller in the package. Decide whether the refusal should
  exist, then fix all three prose sites either way.
- **Two different things are called `local_moments`** and are trivially confusable:
  `System.local_moments` (`builder.py:289-300`) is the *input* per-atom directions the
  symmetry group is decided from, and `MagneticField.local_moments` (`fields.py:292-296`) is
  the *converged* sphere integral, which exists only when a field or an atom-resolved
  constraint does. Rename one.
- `reducebf` is accepted at any value (`builder.py:963`) where Elk refuses outside `[0.5, 1]`
  (`~/apps/elk-9.6.8/src/readinput.f90:1264-1266`). One line, and the kind of edge the amber
  boxes exist for.
- `SCFResult.constraint_energy` (`driver.py:846`) is one scalar over all sites, so under
  `'atomic texture'` a single flipped site out of fifteen reads as a small number
  indistinguishable from partial convergence. Carry the per-site cosine.
- `STARTING_MOMENTS` has a README row (`README.md:176`) and a `features.tex` entry and **no
  notebook** (`grep -rl STARTING_MOMENTS notebooks/` is empty), which is deliverable 4 of the
  five `CLAUDE.md` requires, for the feature everything in this file rests on.
- **The fast gate contains no magnetic noncollinear SCF.** Every `pw.x` noncollinear
  comparison is module-level slow (`test_spinorbit.py:49`, `test_spinor_forces.py:86`,
  `test_spin_spirals.py:52`, `test_spiral_relaxation.py:63`, `test_magnetic_constraints.py:37`,
  `test_paw_noncollinear.py:32`, `test_noncollinear_pdos.py:48`, five of seven in
  `test_noncollinear_magnetism.py`), and the only spinor SCF in the gate is nonmagnetic
  platinum (`test_spinor_pdos.py:163,169`, four cases). Promote one cheap magnetic case -- the
  two-atom hydrogen antiferromagnet, not iron.
- **Both settled 2026-09-12 (P79), and both are clean.** *(a)* The P75 group and
  `is_magnetic` **agree** for a run carrying both cards. Checked on five shapes of a
  four-hydrogen chain, `is_magnetic`, `nspin_mag`, `nsym` and `nk` together: moments only,
  fields only at 1e-6 Ry, both (the NiBr2 combination), a sub-threshold pair (moments 1e-8
  *and* field 1e-10), and the P75 failure mode itself -- ferromagnetic moments with a
  cycloid field. All five give `is_magnetic = True`, `nspin_mag = 4`, **`nsym = 4`** and
  `nk = 12`, against 16 and 9 for the ferromagnet. The mechanism is that `is_magnetic` reads
  both cards and `axial_fields` filters by both, so the only way they can differ is a
  quantity above one threshold and below the other -- and there the *group* is the
  conservative one, since P77b made the field comparison scale-free while `is_magnetic`'s
  own floor stayed at 1e-12.

  *(b)* Nothing rechecks the magnetic group during a relaxation and **nothing needs to**.
  The magnetic condition is `R m_{irt(s,a)} = +- det(R) m_a`: the moments are a fixed input
  tuple in `ATOMIC_POSITIONS` order, so a moving atom carries its own row, and `checkallsym`
  (`check_symmetry`, `symmetry.py:1070`) verifies that every operation still maps the
  structure onto itself -- which is the permutation the magnetic condition is stated in.
  Unchanged moments plus an unchanged permutation is an unchanged magnetic group. The one
  hole is that `_maps_structure` tests a *setwise* map rather than pinning the permutation,
  so two atoms of the same species exchanging roles would slip through; a continuous
  relaxation does not do that, and nothing else here can.

---

## 4. Structure by structure

Every cell is filled. "unknown" is followed by the check that would settle it. Item numbers
refer to section 3; O-numbers to section 6.

| structure | can it be initialised | will it converge | what is validated | what to watch |
|---|---|---|---|---|
| **Antiferromagnet, collinear** (`nspin = 2`) | Yes, two ways. Two species labels is the safe one. One label plus a z-only `STARTING_MOMENTS` card is accepted, but needs a `starting_magnetization` as well or `build_system:773-790` refuses | Two labels: yes, the ordinary `nspin = 2` path. One label: it converges **to zero magnetization** unless `nosym = .true.`, because the sublattice-swap operation survives and reduces the k-set too | The two-species route is the P9 collinear path validated against `pw.x` throughout. The one-species route has nothing: `OPEN.md:266-269` records the "two-atom AFM converging to `\|M\| > 0` per site and zero total" as **not verified** | **Item 1.** The failure is completely silent: a good total energy and `absolute_magnetization` near zero |
| **Antiferromagnet, noncollinear** (`nspin = 4`) | Yes, `STARTING_MOMENTS` with opposite moments, one species. The magnetic filter runs and the seed survives symmetrisation | unknown -- no textured `STARTING_MOMENTS` run has ever been carried through an SCF (**O1**). The mechanism that would break it is fixed at the input end | Symmetry-group counts and the *starting* density's lobes (`test_textured_symmetry.py`); nothing converged | **Item 2** (no per-site readout), **item 5** (DFT+U and PAW `becsum` still per species) |
| **Ferrimagnet** | Yes, two species with different `starting_magnetization` magnitudes. No card needed | Same machinery as a ferromagnet; no per-atom constraint is involved | Nothing specific to a ferrimagnet | The total moment is nonzero, so the printed diagnostic is informative for once -- but per-site moments still are not printed (item 2). Watch item 4 if any magnitude is written at or above 1 |
| **Canted** | Yes, `STARTING_MOMENTS`, or two species with different `angle1`/`angle2` | **Nothing holds it.** A 15-degree cone collapses to the nonmagnetic solution on fcc hydrogen (`features.tex:1878-1884`). The constraint that would hold it per atom, `'atomic texture'`, crashes today | Nothing | **Items 8, 10.** `'atomic direction'` looks like the right constraint and is not: it fixes the polar angle alone. Collapse to collinear is silent |
| **120-degree Neel** | Yes, three atoms of one species with a card at 0, 120, 240 degrees | unknown -- **O1** and **O4**. The seed and the group are right; nothing has held one | Nothing | The total moment is zero by construction, so the console says nothing whatever happens (item 2). On a Hubbard cell every site starts on one axis (item 5) |
| **Helix, commensurate supercell** | Yes, one card row per site | **Yes, demonstrated.** `tests/data/qe/h-chain-90deg.in`, four hydrogen atoms at 0/90/180/270, `conv_thr = 1e-11` | The strongest number in this table: its electronic energy per cell matches the equivalent spiral to **3e-12 Ry** (`PLAN.md:2023`, `test_spin_spirals.py:133-150`) | It carries `nosym = .true.`, so it never exercises the symmetrisation P75 fixed, and the demonstration is an *energy identity* rather than an inspection of the final moment directions. It is a hydrogen toy chain |
| **Helix, spin spiral** (`spiral_q`) | Yes, the generalized Bloch theorem: up at `k + q/2`, down at `k - q/2` | It converges and **does not stay magnetic**: `\|m\| = 0.0001` at `q3 = 1/2` on fcc hydrogen unless something holds it. Holding it is available and untested | Five internal identities on a one-atom hydrogen chain (4e-15, 7e-13, 3e-12 Ry, tests assert 1e-9), plus P21's four. **No external number, from `pw.x` or Elk** | **Item 15.** Forced `nosym`; refuses ultrasoft/PAW and spin-orbit coupling; `E(q + G) = E(q)` fails by 2e-3 Ry on a 1x1x3 grid; `degauss = 0.02` lands in a different minimum from the supercell |
| **Cycloid** | Yes, `STARTING_MOMENTS`; the group drops correctly (P75 measured 4 to 1 on a four-atom test cycloid) | unknown, and the one production attempt failed: the 45-atom NiBr2 cycloid unwound between iterations 3 and 6 and reported `accuracy = 7.55e-07` in 23 iterations. P75's fix addresses the cause; nobody has rerun it (**O1, O2**) | The symmetry-group drop and the starting density. Nothing converged | **Items 2, 8.** `'atomic direction'` returns exactly 0.0 for a cycloid **and** for the collinear state it should exclude (`test_textured_symmetry.py:223-239`) -- a null that reads as a pass |
| **Conical spiral** | Yes: the spiral ansatz `m^q(r) = (m_x cos q.r, m_y sin q.r, m_z)` carries the cone angle by construction (`system/spiral.py:17-24`), and a supercell cone is a card | No. A 15-degree cone "collapses the same way, because nothing holds it" (`features.tex:1878-1884`) | Nothing | **Item 10.** Elk runs `fsmtype = -1` with a large field for exactly this; there is no per-atom fixed moment here |
| **Skyrmion in a supercell** | Yes, one card row per atom; this is what the card was written for | unknown -- **O1** at a size nobody has tried. The largest committed textured cell is four atoms | Nothing | Memory (**item 22**) and cost: a textured group is 1 anyway, so the whole k-grid is paid, and a spinor derivative on a large cell has no post-P73/P74 measurement |
| **Altermagnet** | The natural spelling -- one species, `+-m` -- is the one that fails (item 1). Two labels works and hides the sublattices from the symmetry search | Two labels: yes, in principle. One label: converges to the compensated nonmagnetic answer | Nothing. No altermagnet appears anywhere in `tests/` | **Item 1** first. Then the observable: the defining quantity is the momentum-resolved spin splitting. Established by grep: no `Calculator` `get_*` returns a band- and k-resolved spin. Separately unknown -- nobody opened what `NSCFResult`/`BandStructure` carry, so check whether they expose the spinor wavefunctions well enough to form `<S>_nk` |
| **With spin-orbit coupling** | Yes: `noncolin` plus `lspinorb` and a fully-relativistic dataset. `angle1`/`angle2` per species, or the card | Yes. This is the best-supported regime: forces, stress and both relaxations run on norm-conserving, ultrasoft and PAW | Against `pw.x`: bcc Fe's magnetic group 48 to 16 with 8 `t_rev`, its site moment 3.18 mu_B at 1e-3; platinum's spinor PBE stress 4.4e-7; the four-hydrogen force cell 3.6e-9 Ry and 8.9e-7 Ry/bohr; spinor DFT+U on relativistic BN 1.2e-7 Ry. The `fcoef`/`qq_so` zeroing order is guarded and pinned | **The gap is textures on real datasets**: "several non-parallel moments plus ultrasoft or PAW, against another code" is empty (**O13**). `<L>`/`<S>` refuse a reduced k-set and a `rel-` US/PAW dataset; DFT+U with SOC forces `nosym` (item 13); no response of any kind (item 11); `fsm` with `lspinorb` is accepted and unsound (item 21) |

---

## 5. The traps, as a checklist

`CLAUDE.md`'s recurring traps, each site marked **guarded** (a fix is in place and, where
stated, was measured), **unguarded** (it is not), or **unknown/unexercised** (nothing tells
you either way).

**`abs` and `|m|` at a forced zero.**

- `xc/functional.py:812` `safe_modulus`, used by `local_spin_frame` (`:860`) and
  `paw/gradient.py:197` -- **guarded**, and measured: `[nan nan nan]` before,
  `[0. 0. 0.]` after (OPEN A5).
- `scf/potential.py:332`, `_noncollinear_gradient_correction` -- **unguarded and live**. This
  is item 3.
- `scf/potential.py:584`, `_noncollinear_meta_exchange` -- **unguarded, latent**: every
  derivative consumer is refused by `reject_potential_only` (`forces/energy.py:243`).
- `forces/torque.py:78-79` -- **unguarded, latent**: the `jnp.where` guard is a *global*
  scalar norm over the whole grid and protects no individual grid point.
- `scf/driver.py:569`, `_noncollinear_magnetization` -- **unguarded and correctly so**: jitted,
  reporting-only, never differentiated.
- `scf/fields.py:474` `_safe_modulus` -- **guarded** (masks the argument).

**A clamp's tangent at the boundary.**

- `zeta` in `xc/lda.py:218,286` and `xc/gga.py:173` through `clamp_polarization` --
  **guarded**.
- `local_spin_frame`'s `|m| <= |n|` clamp is a `jnp.where`, not `jnp.minimum`
  (`functional.py:860,866`) -- **guarded**.
- `fields.py:350`, `jnp.clip` inside `arccos` for `'total direction'` -- **unguarded**, and QE
  guards the same place (`add_bfield.f90:185-192`). Item 20.

**Rule D4, a diagonal under a degenerate rotation.**

- Velocity, effective mass and optical conductivity carry the multiplet block average --
  **guarded**.
- Every noncollinear consumer traced (`projwfc/angular_momentum.py`, P48b, P69, the force
  theorem) sums over bands with the occupation weight, which is equal inside an exactly
  degenerate multiplet -- **not exposed**.

**`for_spin` on a caller-built k-set.**

- Applied at every spinor entry point traced: `calculator.py:1402`, `builder.py:889`,
  `nscf.py:328` (and through it `denser_grid` and `stm.py:161`), `conductivity.py:107`,
  `nesting.py:130`, `shg.py:83`, `photocurrent.py:80`, `transport.py:539`;
  `anisotropy.py:409` reuses `system.kpoints`, already normalised -- **guarded**.
- The spiral's doubled k-list is a `for_spin` boundary that turns out to be clean and should
  be struck from anyone's worry list: `system/spiral.py:74-82` doubles the list for the
  **basis** only, while the occupation weights come from `system.kpoints`, which took
  `for_spin` in the builder (`driver.py:1166-1212`, `:2391-2403`) -- **guarded, checked**.

**A response on a reduced k-set is polar or axial and must be symmetrised as one.**

- `Calculation.symmetrize_directional` (`driver.py:2641-2676`) treats every `nspin_mag`
  channel as a scalar, so three magnetization channels would be symmetrised as polar --
  **unguarded, latent** behind the noncollinear Sternheimer refusal. Item 11.
- `paw/symmetry.py:348-351` folds `magnetization_signs` into the `becsum` rotations for
  `nspin_mag == 4` -- **guarded**, but validated only on a one-atom cell.
- `sym_rho`'s axial average for the density -- **guarded** (`symmetry.py:402-424`).

**A magnetic input variable no filter sees.**

- `b_field` (`builder.py:163-164`) and `fixed_magnetization` are not in `System.axial_fields`
  (`builder.py:302-316`) -- **unguarded**, and shared with QE, which builds `m_loc` without
  `B_field` either (`setup.f90:271-276`). What is specific here is that a workflow steps the
  field programmatically (`response/magnetoelectric.py:196-210`); all three committed inputs
  set `nosym`.
- `is_magnetic` at 1e-12 / 1e-6 against `_MAGNETIC_TOLERANCE` at 1e-5 -- **unguarded**. Item 6.
- A `dataclasses.replace` of `starting_moments` desynchronises the k-set from the group --
  **unguarded**, and there is no `with_moments` to use instead. Item 14.
- A texture that arrives as a *density* rather than as an input variable reaches no filter --
  **unguarded and stated**, in `features.tex:944-951` and `PLAN.md:12410-12418`, as prose
  rather than a raise. Only `starting_from` across systems is a live entrance.

**A check whose null result cannot be told from a pass.**

- `'atomic direction'` returns exactly 0.0 for a cycloid and 0.0 for the collinear state it is
  meant to exclude -- **known and asserted as a committed test**
  (`test_textured_symmetry.py:223-239`), stated in prose (`features.tex:918-922`), and
  **neither refused nor warned**. The right answer is `'atomic texture'` (0.0 against 4.0),
  which needs item 8 first.
- `tests/regression/test_nesting.py:226` lists "the `t_rev` branch of the unfold map" among
  what it exercises, on a cell built with `starting_magnetization = 0.0` (`:244-245`), so
  `nspin_mag` is 1 and the branch is never taken -- **unexercised, reads as a pass**.
- A time-reversed operation whose **atom permutation is non-trivial** -- **unexercised**. Every
  test reaching the magnetic filter is one magnetic atom (`test_noncollinear_magnetism.py:92-113`
  bcc Fe, `:163-191` H, `:198-221` O), where `mapping[s]` is the identity, so `det(R)` and the
  `t_rev` sign are tested and the permutation is not. The k-set reduction's `t_rev` branch is a
  *different* exposure and is validated (11 points to 22 against QE's own count).

**Others from the list, for completeness.**

- `np.asarray` in a differentiated path: `forces/spiral.py:343-344` is *after* the gradient --
  **clean**.
- Hardcoded dtypes: the `1j *` literals in `hamiltonian/noncollinear.py:325-326,448` are Python
  complex scalars, which JAX weak-types; `pseudo/spinorbit.py:554` casts its Pauli table to
  `fcoef.dtype`; `projwfc/angular_momentum.py:367` is host `np.einsum` and never enters a trace
  -- **clean**.
- The `fcoef` / `qq_so` zeroing order (`init_us_1` zeroes the cross-radial entries *after*
  building `dvan_so`): `pseudo/spinorbit.py:318-361`, pinned by
  `tests/unit/test_spinorbit_coefficients.py:152` -- **guarded**.
- A stationary state hiding a gradient error, and a stencil containing its own centre: nothing
  in this path was found to touch either -- **no finding**.
- Forces refuse a ground state converged under a field (`forces/energy.py:278`) -- **guarded**;
  `run_scf` itself does not warn (item 7).

---

## 6. Open questions

Each needs something to be **run**. Nothing below was.

**O1. Does a texture stated with `STARTING_MOMENTS` survive an SCF, with symmetry on?**
**Partly answered, 2026-09-12 (`9f806b0`).** The *collinear* half is now done and is a
committed test: `tests/data/qe/h2-mirror-afm.in` is a `+-m` card on one species carried
through an SCF and inspected per site, and it survives with symmetry on -- it did not
before that commit, and the failure was a clean convergence to the nonmagnetic state 6.6
meV up. The **noncollinear** half is still exactly as written below: no *vector* texture
has been carried through an SCF and inspected. What the site-moment readout added is that
the inspection no longer has to be typed at a prompt -- `SCFResult.site_moments` is the
`(nat, 3)` matrix the singular-value test below wants, and `history` has it per iteration.
Nobody has run one. Take the four-atom 90-degree cycloid already built at
`tests/unit/test_textured_symmetry.py:39`, converge it twice at `conv_thr = 1e-8`, once with
`nosym = .true.` and once without, and compare the per-site directions from `get_locals`. The
discriminator is the singular values of the `(nat, 3)` matrix of directions: two nonzero means
the texture held, and `[3.873, 0.0056, 0]` is the collapse signature P75 recorded.

**O2. Does one survive on a production magnet?** This is P75's own outstanding item
(`PLAN.md:12403-12409`): the 45-atom NiBr2 cycloid "has not yet converged to a textured state
at any k-mesh". The run that supplies the figure is the open question, not the feature.

**O3. How much does the symmetriser remove?** Compute `||m - sym(m)|| / ||m||` on the first
density, before mixing. Both arrays are in hand inside `Calculation.symmetrize`
(`driver.py:2629-2639`). Feed it a case that must trip it -- the four-atom cycloid handed to
`run_scf` as a starting density with **no** card, where the residual must be order one -- and
the same run with the card, where it must be at round-off. Warn or refuse above ~1e-3. This is
the in-run diagnostic that would have caught P75 at iteration 6 instead of after 23 clean ones.

**O4. Does `'atomic texture'` hold a 120-degree state or a cone, and at what `lambda`?**
Blocked on item 8. Report the angle between the converged moments and the residual penalty at
convergence.

**O5. Does a constraint hold a spiral, and does it constrain what one thinks?** Run
`tests/data/qe/h-fcc-spiral-scan.in` at `q3 = 1/2` bare and again under
`constrained_magnetization = 'fsm'` with `fixed_magnetization` at the collinear `|m|`, and
report the converged `|m|` per site for both. The bare run is on record at 0.0001. If the
constrained run holds it, that number belongs in `features.tex` beside the collapse it fixes,
together with the sentence that a constraint on a spiral constrains the **rotated-frame**
moment.

**O6. Is the one-species collinear antiferromagnet really averaged to zero, and what does it
cost?** The static half needs no SCF (item 1). The physical half is one SCF each on the rutile
cell or the four-atom Ni chain, with and without `nosym`, comparing
`absolute_magnetization` and the per-site moments.

**O7. Does the noncollinear GGA derivative return NaN?** `jax.grad` of
`_noncollinear_gradient_correction` on the `(3, 4)` magnetization
`tests/unit/test_gamma_basis.py:317-322` already builds, whose column 1 is bit-exactly zero.

**O8. Is the noncollinear GGA branch correct at all?** Bigger than O7 and more important:
generate a `pw.x` reference for a magnetic noncollinear **PBE** cell with forces --
`h4-noncolin-force.in` with `input_dft = 'PBE'` -- and compare. Today `potential.py:304-353` has
no measured derivative anywhere in the project.

**O9. Is the mixer's metric what costs `fe-mag-1k` its twenty extra iterations?** First
deconfound beta from magnetism: `python3 tools/compare_qe.py benchmarks/fe-unstable-nonmagnetic.in --repeats 5`
against the `fe-mag-1k` pair. Then dump one run's residual history and recompute the Anderson
coefficients under both quadratic forms (`scf_accuracy` gives QE's for free) and report the
angle between the two coefficient vectors.

**O10. What does a spinor derivative cost after P73 and P74?** `tools/gpu/force_memory.py` on
`bismuthene-soc-small` twice, once at `DEFUMAT_AUG_MAX_BYTES=off` and once at a value small
enough to force the table, comparing the reported buffer -- that settles both the staleness and
the possible non-monotonic cliff at the 2 GB default (`augmentation.py:316`) in one pass. Be
aware of the caveat: P73 replaced the *stored* `Q_ij(G)`, while `augmentation.py:288-301` says
the reverse-mode cost lives in `_qrad_kernel`'s `(ngm, kkbeta)` intermediate, which the table
route evaluates rather than avoids -- so P73 may not have helped the backward pass at all.

**O11. What do the noncollinear slow tests say today?** `tools/run_regression.sh` restricted to
the nine noncollinear files with `DEFUMAT_TEST_MEM_MAX=20G`, keeping the durable per-file
summary line as the dated record -- `regression-results/` currently holds none, only an
`in-flight.log` of starts. P46 measured `pt2-soc-paw-force` at 12,204 MB alone and the pair at
16,961 MB, over `run_regression.sh`'s 12G default, so the cap matters.

**O12. Does an Elk spin spiral agree with this one?** Elk is at `~/apps/elk-9.6.8` and was built
here from the vendored source. Run the same hydrogen chain at three wavevectors as an Elk
spin-spiral ground state and compare total energies after subtracting each code's own
reference. That is the only check that catches an error the five identities share, and it also
supplies the missing `PERFORMANCE.md` pair.

**O13. Ultrasoft or PAW with several non-parallel moments has no external number in any
regime.** Take `fe-kind1-noncol.in` (`Fe.rel-pbe-spn-rrkjus`), build a two-atom
antiferromagnetic version (`angle2 = 0` and `180`) and a canted one at 90 degrees, and generate
a `pw.x` reference for each. Compare total energy, both site moments and the forces. One new
`pw.x` run per case, on a cell whose cost is already on record. Until then, everything that only
turns on in that combination -- `add_becsum_so` on a textured `becsum`, `qq_so` in a textured
overlap, PAW's one-centre local spin frame with several different frames in one cell -- has no
external number.

**O14. What is a reduced k-set worth on a spinor run?** Every refusal in item 13 routes the user
onto the whole grid and `PERFORMANCE.md` has no entry pairing a symmetrised spinor run against
its `nosym` twin. The measurement exists in embryo: `test_spinor_forces.py:330` already builds
both the 8-point wedge and the 32-point closed grid of `pt2-soc-force.in` and runs both to
`conv_thr = 1e-12`. Time that pair and record its peak RSS.

---

## 7. Claims that did not survive

**No whole finding was refuted**, so this list is the *sub-claims* the adversarial pass
falsified inside findings that otherwise stand. They are here because each is plausible enough
to be rediscovered and re-believed, and each would send a session down a wrong path.

1. **"A one-species collinear antiferromagnet written the natural way -- one species, `+m`,
   `-m`, nothing else -- runs and converges to zero."** It does not run:
   `build_system:773-790` refuses `nspin = 2` with no `starting_magnetization`. True: the
   refusal's own message tells the user to add one, and with that second line the silent path
   opens exactly as described.
2. **"No textured noncollinear state has ever been converged in this repository."** One has:
   `tests/data/qe/h-chain-90deg.in`, four hydrogen atoms at 0/90/180/270, `conv_thr = 1e-11`,
   matched to the equivalent spiral to 3e-12 Ry. True and narrower: it is a hydrogen toy chain
   under `nosym`, checked as an energy identity rather than by inspecting the final directions,
   and nothing textured has converged on a production magnet or with symmetry on.
3. **"There is no per-site moment readout at all."** `Calculator.get_angular_momenta()` returns
   a per-atom `<S>` 3-vector with a printer. True: the sphere integral QE prints *every
   iteration* (`get_locals`) is computed nowhere in a run, and there is nothing per iteration.
4. **"Elk's usual `reducebf` is 0.9."** Elk's default is 1.0 and it restricts the value to
   `[0.5, 1]`; this code's default is also 1.0. True: nothing here range-checks it.
5. **"A 7% residual `reducebf` field puts a ~7 mRy error in the reported total."** The field
   energy is excluded by convention but *reported*, and the error in the total is second order
   because the state is stationary. True: nothing warns that a run stopped with the field still
   on.
6. **"`constrained_magnetization = 'atomic'` aims at the wrong unit."** It aims at exactly the
   number written: `get_locals` returns Bohr magnetons and the card's rows pass through
   unchanged. True: the *seed* is `Z_valence` times too large.
7. **"The dual-unit problem is the `STARTING_MOMENTS` card's."** It predates the card: the
   per-species path disagrees with `pw.x` for any `starting_magnetization` at or above 1
   (item 4).
8. **"Two thirds of a spiral's k-point cost is recoverable with the filter that already
   exists."** Not at `q != 0`: `magnetic_symmetries` knows nothing about `q` or about the spin
   rotation an operation performs on the rotated frame, which is exactly what the refusal says
   is missing. `defumat/system/spiral.py:110-128` `invariant_operations` exists to *price* the
   spin space group and says so in its own docstring.
9. **"`System.with_spin` drops a texture, and a checkpoint resume, a promotion or an Elk seed
   are new ways for one to arrive unseen."** `with_spin` is a `dataclasses.replace` that carries
   `starting_moments` through and rebuilds the k-set from them; a same-input resume has the same
   group by construction; `get_elk_seed` cannot carry a texture at all
   (`io/elk_density.py:574-578` raises for `nspin_mag != 1`). True: `with_spin` cannot
   *introduce* a texture, and `starting_from` across systems is a real entrance -- which is P75's
   own recorded item.
10. **"The 4.4e-7 spinor PBE stress shows the noncollinear GGA branch runs."** It does not:
    `spinorbit-pbe.in` sets `starting_magnetization = 0.0`, so `nspin_mag = 1` and the
    *unpolarized* branch was measured. This makes item 3 larger, not smaller.
11. **"The QE workaround for a collinear antiferromagnet -- split the species -- is written down
    nowhere."** It is, at `builder.py:1596-1598` and `PLAN.md:2158`. True: it is absent from
    `docs/features.tex`, which is the user-facing document.
12. **"A spinor ultrasoft GaAs converges to a spurious moment of -0.322 mu_B."** It does *not*
    converge: `PLAN.md:8822-8825` records that it fails in 80 iterations while drifting there.
    Non-convergence is the flag, so this is a convergence defect and not a silent wrong answer.
13. **"The `B_field`-outside-the-filter gap is this code's own."** QE builds `m_loc` without
    `B_field` either (`setup.f90:271-276`). True: what is specific here is a workflow that steps
    the field programmatically, which `pw.x` has no counterpart for.
14. **"The magnetic filter's threshold gap is `OPEN.md` A6 reappearing."** A6 was about *which
    rule* decides "is this run magnetic" and is closed. The disagreement between that rule and
    the filter it feeds is genuinely unrecorded.
15. **"`in-flight.log` shows the noncollinear slow tests have not been run."** The log records
    *starts* only, spans several invocations on the same day and carries no pass or fail. True:
    `regression-results/` holds no durable per-file summary, which is the record that would say.
16. **"`get_angular_momenta` refuses the whole ultrasoft and PAW family."** Only
    fully-relativistic (`rel-`) ultrasoft and PAW: the guard is
    `any(pseudo.has_so) and qq is not None` (`angular_momentum.py:270-276`), so a
    scalar-relativistic Ni PAW dataset passes. Related: swapping in
    `Calculation._spinor_overlap` to lift that refusal is **not** a one-argument change -- the
    module works in a scalar orbital basis with an explicit spin axis, so the projectors and
    `_site_density_matrix` both have to be rebuilt.
