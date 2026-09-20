# PAW: the modes that still refuse it, and what each piece needs

## What this file is

The remainder of an audit of what a PAW dataset refuses where a norm-conserving one
runs. Four items of that audit are being taken separately and are not tracked here, so
nothing below is about them; the numbering starts at 5 because that is where the audit's
own list continues.

`AUGMENTATION-NEXT.md` is the wider file, covering ultrasoft and PAW together, and every
entry below points into it rather than restating it. The split between the two is that
this file is about the modes a PAW dataset cannot be run in, which is a different
question from which terms an augmentation charge is missing: four of the seven items
below refuse ultrasoft on the same line they refuse PAW, one binds PAW alone, and two are
not refusals at all but modes with a cost attached. The one place PAW is ahead of
ultrasoft rather than behind it is outside the numbering, at the end.

## How each entry was checked

Every entry was written after opening the guard and reading the code around it, and each
one says what was read. That is not a habit worth stating for its own sake: it is the one
correction `AUGMENTATION-NEXT.md` has had to make five times, every time from writing an
entry off the refusal's message and not off the code, and every time in the direction of
making the work look like something other than what it is. Where a claim below could not
be confirmed from the code, the entry says so instead of repeating what it was told.

Guards are named by their function and their file and never by a line number, because the
line numbers in `AUGMENTATION-NEXT.md` have already gone stale once.

Three things to know before reading the tests quoted below.

- `Calculation.is_ultrasoft` is `augmentation is not None`, which is **true for a PAW
  dataset as well**, so a guard written on it refuses both. `Calculation.is_paw` is the
  narrower one.
- `Pseudopotential.is_ultrasoft` (`pseudo/upf.py`) is true for a UPF header reading `US`,
  `USPP` **or** `PAW`, since the generators disagree about the spelling. A guard written
  over `pseudos` with that property therefore also catches PAW, and item 11 is one.
- Items 8 and 10 were read on `master`, which is where the ultracell's guard was
  rewritten and where P96's spiral gradient landed (`7c5cb50` and `341723e`). The branch
  this file was written on was cut from an older commit and has since been moved onto
  `master`, so the two entries and the code now agree.

## 5. The augmentation dipole in the Kubo assembly

**What was read.** `_refuse_augmented`, `velocity_matrices` and `kubo_from_matrices` in
`topology/kubo.py`; `require_a_conductivity_regime` in `response/conductivity.py`;
`require_a_shift_current_regime` in `response/photocurrent.py`; the module docstring and
`ultrasoft_position` in `response/efield.py`.

**What is missing.** With `S = T^dagger T` the states a Berry phase is about are
`T|psi>`, so the connection carries `<psi_n|T^dagger dT/dk|psi_m>` beside
`<psi_n|S d/dk psi_m>`. That first object is the dipole of the charge sitting inside the
augmentation spheres, `adddvepsi_us`'s `dpqq`, and `kubo_from_matrices` is handed only
`dh` and `ds` and has nowhere to put it. The correction depends on `A` rather than on
`S = A^dagger A`, so no arrangement of `dS/dk` can supply it: `U(k) A` leaves `S` alone
and moves the physical states. Section 2 of `AUGMENTATION-NEXT.md` has the derivation and
the model measurement, which is 18 per cent of the curvature's scale on a Haldane model
where the exact answer is free, and 0.010 of the Chern number.

**What reading the code added.** `_refuse_augmented` guards `plane_wave_kubo` only, and
it tests `calculation.augmentation is not None`, so PAW and ultrasoft refuse together and
`kubo_from_matrices` itself is unguarded, which is what lets
`tests/unit/test_topology_curvature.py` feed it the model directly. The default
`method = "fhs"` carries both terms correctly and is exact on any mesh, so the curvature
itself is not unreachable on any dataset; what is unreachable is the smooth `Omega(k)`
map from the sum over states, the optical conductivity and the shift current.

**One number in `AUGMENTATION-NEXT.md` section 2 is the wrong quantity.** It says the
ultrasoft dielectric constant of `response/efield.py` is 8e-6 from `ph.x`. The module
docstring gives **14.325321 against 14.325270** on ultrasoft silicon, which is 3.6e-6
relative, with 14.320211 against 14.320177 on PAW and 5.756059 against 5.756182 on
ultrasoft carbon; the 8e-6 is a **Born charge** figure, ultrasoft silicon's in
`CLAUDE.md`'s trap list and PAW silicon's in P39a. The point the
sentence was making survives either way, which is that the object exists in this
repository and is pinned against `ph.x` where it is used.

**What is left.** The dipole in matrix-element form inside `velocity_matrices`, taken
from `efield.py`'s machinery, and then an **assembly** check on plane waves rather than a
second check of the term.

**Size, as a hypothesis.** A phase. The model check is done and the shape of the term is
pinned, so the risk is in the assembly and not in the object.

## 6. The three spin-space objects

Three separate items that share one sentence: a spinor's augmentation metric is a complex
2x2 matrix in spin space where the collinear one is a number, and each of the three
places that needs it needs a different one of them.

**What was read.** `require_a_sternheimer_regime` and the inner `apply` of
`local_perturbation` in `response/sternheimer.py`; `energy_at` in `forces/energy.py`;
`_refuse_what_is_not_written` in `projwfc/angular_momentum.py`; and, for the route the
third one would take, `Calculation._as_spinors` and `Calculation._spinor_overlap` in
`scf/driver.py`.

**`int3` as a 2x2 matrix in spin space**, which is QE's `set_int3_nc`. ✅ **DONE
2026-09-17**, `PLAN.md` P98 and `AUGMENTATION-NEXT.md` §1d.

The hypothesis this entry flagged -- "that the recombination is the whole of it, which the
comment asserts and nothing measures" -- was right to flag and wrong in both directions.
The recombination is **not** work at all: `Calculation.coefficients` dispatches on
`noncolin` itself, so the `jvp` passes through `_newd_noncollinear` and its `fcoef`
sandwich is linear, and the tangent comes out dressed with nothing written. And it is not
the whole of it: the *position* operator was the term that had to be written, an ultrasoft
state's augmentation dipole coupling the two spinor components through `dpqq_so`
(`compute_qdipol_so`), which `ultrasoft_position` could not even broadcast for a spinor.
The dielectric constant of an augmented spinor reaches `ph.x` to **3.5e-5** on
fully-relativistic ultrasoft AlAs. What is left of the entry is written at the end of
§1d: PAW's one-centre tangent now rides the same `jvp` rather than being added after the
sandwich, and no committed cell distinguishes the two orders.

**`qq_so` in the matrix orthonormality multipliers of a spinor force.** ✅ **DONE
2026-09-17**, `AUGMENTATION-NEXT.md` §3c and `PLAN.md` P98. `Lambda` does *not* carry a
spin pair, which is where this entry (and the guard it was read off) was wrong: it
multiplies the band pair and both states are whole spinors, so only the augmentation half
changes and it takes `qq_so`. The Born charges of a fully-relativistic ultrasoft cell
reach `ph.x` to 3.2e-6 and 3.9e-5 on the two atoms. Three further collinear sites in the
Born assembly came with it, all of them broadcast failures rather than wrong numbers.

**The spinor projector set for site-resolved angular momenta.**
`_refuse_what_is_not_written` keys on `any(pseudo.has_so)` together with
`calculation.projectors.qq is not None`, which is worth knowing because it is neither
`is_ultrasoft` nor `is_paw`: what it refuses is a fully-relativistic dataset carrying an
augmentation charge of any kind, and a fully-relativistic norm-conserving one has `S = 1`
and is exact. `AUGMENTATION-NEXT.md` section 1i has the route, which is
`Calculation._as_spinors` orthogonalised against `_spinor_overlap` as
`workflows/anisotropy.py` already does, and both of those exist. The open question is not
the code, it is what to validate against: section 1i says neither `projwfc.x` nor Elk's
`LSJ.OUT` has been located for this combination, and that claim is not confirmed
independently here. **Size:** an afternoon for the code and a phase for the number.

## 7. The dynamical matrix of an augmented metal

**What was read.** `_require_a_moving_overlap_regime` and, at the time, the
`require_norm_conserving` beside it in `response/phonon.py`, and the call site in
`response/nonlinear.py`. **That second guard no longer exists** (`PLAN.md` P100): the
third derivative it covered runs on all three dataset kinds now and the elastic constants
carry `response/elastic.require_a_measured_elastic_regime` instead. Nothing below depends
on it, because this item is about a *metal* and that one was about a dataset.

**What the guard is.** `calculation.is_ultrasoft and calculation.system.occupations !=
"fixed"`, so it catches PAW and it catches only metals. Insulators are implemented on all
three pseudopotential kinds and a norm-conserving metal is too.

**What is missing.** P28's weight split is between a state tangent at `wk` and everything
else at `wg`, and it was derived for a response whose whole `becsum` dependence goes
through `dpsi`. With smearing the occupations respond to the perturbation as well, and
the docstring names three further tangents, `dpsi^ort`, `becsumort` and `dLambda`, whose
weight the insulating case cannot decide because there the two weights are equal. The
failure mode if it is guessed is an acoustic sum rule violated by a plausible amount.

**What reading the code added.** The same guard is called from `response/nonlinear.py`,
so it is not the dynamical matrix alone: the **Raman tensor** of an augmented metal
refuses here too, where the augmented insulator is in as of P43. One inconsistency this entry
recorded is now moot: it noted that `_require_a_moving_overlap_regime`'s docstring said
three further tangents where the guard beside it said four, and that second guard has
since gone.

**Size, as a hypothesis.** A phase, and the work is the derivation rather than the code.
The caution that applies to this entry more than to any other here is
`AUGMENTATION-NEXT.md`'s last one: a gap sized on silicon is a gap sized on a
centrosymmetric crystal, and this one has never been run on a polar cell.

## 8. The ultracell. ✅ DONE in all three spin regimes.

`PLAN.md` P88 stage 5, 2026-09-17; `defumat/ultracell/augmentation.py`. The refusal is
lifted for `nspin = 1` and `2`, and what this entry got right is worth keeping because two
sessions had to rediscover it: the frozen states stay a fixed basis, `S` does not enter, and
what was missing is one term used three times -- the matrix element, `becsum` per atom copy
for the density, and PAW's one-centre terms per copy.

**The sign this entry flagged was the live question and it is settled.** The displacement is
the **ket's** wavevector minus the bra's, `Q' - Q`, which is what the guard's comment on
`master` said and what `AUGMENTATION-NEXT.md` §1k's derivation sentence had backwards. It was
settled by running the other spelling rather than by reading: the induced density's error
stalls at 6.7e-2 where the right one falls to 3.7e-3, and the total energy goes 2.6e-5 Ry
**below** the four-atom supercell's, which the nested-basis bound forbids. §1k is corrected.

**Two things this entry did not have.** Its sizing said PAW's one-centre terms per copy were
not sized; they turned out to need no new physics at all, because
`PawCorrections.energy_and_coefficients` reads its atom count off `becsum.shape[1]` and knows
nothing about positions, so `N` copies are simply `N nat` atoms of the same species. And the
entry did not name the second mixed variable: PAW's one-centre `D` is a functional of
`becsum` rather than of the density, so `becsum` joins the box density in the mixer, packed
after it exactly as `_mix` packs it in the unit cell. An ultrasoft run needs neither.

**And lifting it made three things reachable that the refusal had been protecting.** The
ultracell image and the ultracell spectrum run on an augmented dataset, because a tip sits
in the vacuum where the smooth states are exact, and both now inherit the unit cell's
`_refuse_an_augmented_plane`. The ultracell **transmission** does not: its exit-plane Gram
matrix needs `S`, and `_ultracell_geometry` passed `apply_s = None` with a comment saying
the ultracell refuses these datasets -- which is how the stale claim was found, and it is
refused by name now.

**The practical gate this entry named is gone, and the mask it asked for was never needed**
(`PLAN.md` P88 stage 9, 2026-09-20). The sizing was right that nothing has to be interpolated
between two boxes and wrong that a mask was missing: the matrix element gathers onto the
wavefunction sphere, so it reads `dV` only at `(G'' - G) + Q_d` with both `G` inside the
`ecutwfc` sphere, and `|(G'' - G) + Q_d| <= 2 sqrt(ecutwfc)` is a sphere in the ultracell's
own reciprocal space -- the supercell's smooth sphere, which the dense box holds whole. The
dense half is unreachable rather than wrongly included. Measured: truncating `dV` before the
matrix element moves modulated PAW silicon at `ecutrho = 8 ecutwfc`, `N = 2`, by 1.3e-12 Ry,
where zeroing it there moves 6.5e-4. Every number above is still at `ecutrho = 4 ecutwfc`,
which is now a choice about what a comparison costs rather than the only pair that runs.

## 9. Every fixed-density mode needs `becsum` beside the density

**What was read.** `fixed_density_states` in `workflows/nscf.py`, `DFTSource.states` in
`workflows/topology.py`, `VelocityOperator.__init__` and `band_velocities` in
`response/velocity.py`, `dielectric_tensor` in `response/efield.py`, and
`Calculator._call_options` with `_STATE_ARGUMENTS` in `defumat/calculator.py`.

**Why it is a refusal at all.** A PAW Hamiltonian's nonlocal coefficients are
`D^(0) + int V Q + ddd_paw`, and only the first two can be rebuilt from the density:
`ddd_paw` comes from `becsum`, which is a property of the wavefunctions. A fixed-density
mode is handed a density and nothing else unless it asks, so every one of them asks.

**Where it is a refusal, and what each says it costs.** All four tests are on `is_paw`
and not on `is_ultrasoft`, since ultrasoft's augmentation charge is already inside the
density that crosses.

- `fixed_density_states`, on `is_paw and not becsum`: wrong by **tenths of an eV** in the
  eigenvalues, and it converges perfectly well while being so.
- `DFTSource.states`, the same test and the same sentence, which is the topology stack's
  own copy of it.
- `VelocityOperator.__init__`, on `is_paw and ddd_paw is None`: `ddd_paw` multiplies
  `vkb(k)`, so it is part of `dH/dk` and not only of `H`, and leaving it out is worth
  **2 per cent**, measured as 1.7e-2 Ry bohr against 8.7e-7 on two-atom PAW silicon.
- `dielectric_tensor`, on `is_paw and not becsum`: wrong by the whole PAW correction, and
  the guard is placed before the response so that the refusal does not cost a
  self-consistent solve first.

**The facade threads it, so this binds the functional entry points only.**
`Calculator._call_options` inspects the signature of whatever it is about to call and
fills any parameter named in `_STATE_ARGUMENTS`, which is `ns`, `tau`, `becsum`, `field`
and `field_scale`, from the cached `SCFResult`, with an empty `becsum` tuple treated as
absent. `response/velocity.py` is threaded one step differently, because
`VelocityOperator` takes coefficients rather than `becsum`: its `band_velocities` entry
point builds `ddd_paw` itself with `calculation.onecenter(result.becsum)`. So a `get_*`
never trips any of the four, and a hand-threaded call is where they are met.

**Size, as a hypothesis.** Nothing to write, which is why this is here as a mode rather
than as a gap. What it costs is per new fixed-density entry point, which has to take
`becsum` in its signature so that the facade can see it, and that is minutes.

## 10. `dE/dq` on a PAW spiral, with a floor and a memory gate

**What was read.** `spiral_gradient` and `_require_a_differentiable_spiral` in
`forces/spiral.py` on `master`, `Calculation.at_spiral_q` in `scf/driver.py` on `master`,
`_qrad_kernel` in `pseudo/augmentation.py` on both, and the commit messages of `7c5cb50`
and `341723e`. This branch still carries the blanket `if calculation.is_ultrasoft` raise
that P96 replaced.

**It runs, which is the point of the entry.** An augmented spiral's gradient is validated
identity for identity as the norm-conserving one is, and PAW's own terms are measured one
at a time rather than inferred from a sum that closes: freezing the displaced table at
the old `q` is worth 5.79 per cent, deleting the overlap constraint 4.07 per cent, and
deleting PAW's one-centre energy 0.13 per cent, the last of which is three times the
floor and therefore not resolved by the check.

**The floor is the dense grid.** The discriminator is `q3 = 1/2`, where the energy is
even and periodic so that `dE/dq` is exactly zero and what comes back is the gradient's
own error with no truncation in it. It falls from **1.964e-06 at `ecutrho = 200` to
2.812e-07 at 400**, against an ultrasoft control of **3.040e-08** that does not fall
because it is already the k-sum's round-off. The two are on different cells and the rows
should not be merged: the `ecutrho` ladder runs on a chain with 9 bohr of vacuum, because
at the committed 12 bohr cell's size that gradient does not fit in 20 GB at 400.

**The memory gate.** On an augmented dataset the density carries `q` and the Hartree
energy is quadratic in it, so a sum of per-chunk gradients is not the gradient;
`spiral_gradient` therefore overrides `k_batch` to a single pass over the whole k axis
whenever the requested chunk is smaller than `nk`, and warns that it has. The committed
chain's gradient peaks at **11.4 GB**, which is the whole per-file test cap, where
seeded silicon peaks at 3.0 GB. The peak is the dense G set rather than the k axis: the
displaced table is rebuilt inside every gradient evaluation and the radial transform's
intermediates are all live at once in reverse mode.

**The obvious fix has a prior against it, which is the correction this entry makes.**
`341723e`'s message calls a `jax.checkpoint` on the radial kernel the obvious memory fix
and says it is unmeasured. `_qrad_kernel`'s own docstring records that `jax.checkpoint`
**there** was tried and measured to be worth nothing, the intermediates being spread
across the radial kernels rather than concentrated in that one, and it names the fix that
has not been written: a `custom_jvp` carrying `dF/d|G|` in closed form, so that the
transform tapes a vector of length `ngm` instead of a matrix. That measurement was taken
on the **stress**, so it does not transfer to `dE/dq` as a result, but it does transfer
as a prior, and the `custom_jvp` is the candidate with a reason behind it.

**A tabulated augmentation table stays refused**, and the refusal is in
`Calculation.at_spiral_q` rather than in `forces/spiral.py`: the tabulated branch reads
`|shift|` on the host to size its radial table, running past the end is a NaN rather than
a clamp, and a tracer has no such value. It is reached only by a cell whose stored
`Q_ij(G)` is over `AUG_MAX_BYTES`, which is the large-slab regime.

**Size, as a hypothesis.** The memory half is a measurement and an afternoon, and the
floor half may be nothing to fix at all, since 2.812e-07 at `ecutrho = 400` on a gradient
of the order of 1e-3 is a converging cutoff error rather than a missing term. What would
settle it is one more rung.

## 11. Gamma-only storage is substituted, with a warning, and not refused

**What was read.** `gamma_storage_is_consumable`, `_without_gamma_storage` and
`Calculation.__init__` in `scf/driver.py`, `Pseudopotential.is_ultrasoft` in
`pseudo/upf.py`, and `build_augmentation` in `pseudo/augmentation.py`.

**What happens.** `gamma_storage_is_consumable` decides whether the half-sphere storage
can be consumed, and it returns false for any pseudopotential whose `is_ultrasoft` is
true, which by the header rule above includes PAW. `Calculation.__init__` then calls
`_without_gamma_storage`, which warns and rewrites `K_POINTS gamma` into an explicit
single k-point at the origin with the full G sphere. The result is the same physics at
twice the plane waves, and the run says so rather than being quietly refused or quietly
halved. The reason is the same for both datasets: `addusdens` and `newd` need their own
`fact = 2` over the augmentation charge, and `Q_ij(G)` is tabulated on the dense set,
which is a half sphere here too.

**The substitution is not where `AUGMENTATION-NEXT.md` puts it.** Its excluded list
credits `pseudo/augmentation.py`. What is there is `build_augmentation`'s hard
`NotImplementedError` on `gvectors.gamma_only`, which is a backstop a normal
`Calculation` never reaches, because the gate has already turned the storage off before
the augmentation charge is built. The behaviour the excluded list describes is right and
the file it names is not.

**Size, as a hypothesis.** A phase if anyone wants the storage itself, and it buys memory
rather than speed: the halving is on the plane-wave-sized arrays, which on a slab is the
difference between a run that fits and one that does not.

## Where PAW is ahead of ultrasoft

One place, and it is the only one this half of the audit found. A potential-only meta-GGA
runs on a PAW dataset and is permanently refused for an ultrasoft one:
`Calculation._require_meta_supported` in `scf/driver.py` tests
`p.is_ultrasoft and not p.is_paw`, because `tau` on the grid is the smooth states' and
needs a one-centre correction inside the spheres, which PAW's partial waves supply and a
plain ultrasoft dataset has nothing to reconstruct from. `pw.x` refuses both
(`setup.f90`, "Meta-GGA not implemented with USPP/PAW"), so this row is an extension
rather than a deficit.
