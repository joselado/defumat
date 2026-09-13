# Memory audit of defumat

> **Status, 2026-09-13.** The top **four** items are **done**, and B1 with them: A1 (both
> augmentation scan bodies rematted, `bismuthene-soc-small`'s force tape 2.32 GiB -> 0.99
> GiB measured by `memory_analysis()`, force and stress unchanged to one ulp), A2
> (`run_relax`'s two reference drops), A3 (`run_vc_relax`'s one), and **A4 + B1** (the PAW
> one-centre atom axis chunked *and* rematted, which makes its tape flat in the atom count:
> 4.08 GB -> 0.80 GB at the NiBr2 nickel sublattice, and faster). `PERFORMANCE.md` carries
> the numbers. Everything below is the audit as written, including those four; the rest is
> untouched and still a to-do list.
>
> **A4 is also the item that says why an audit is checked rather than applied.** Its
> prescribed one-line fix was measured to be a *regression*, and the correction is inline
> under A4 rather than replacing it, so the wrong reasoning stays legible beside the right
> one.


Static audit, 2026-09-13. Nothing was executed: no pytest, no notebook, no benchmark, no SCF,
nothing that imports `defumat` and allocates. Every number below is arithmetic on shapes read
from source, on counts read from committed UPF headers and QE reference outputs, or on figures
already recorded in `PERFORMANCE.md`. Where a claim needs a measurement, the measurement is
named and not taken.

## 0. How to read the numbers

**A peak is a max over stages, not a sum.** An SCF's Davidson peak, a force's reverse tape and a
relaxation's between-step resident set are three different moments of a run. Figures are given
per stage, at a named cell, and are never added across cells or across stages. The one place a
sum is legitimate is inside one stage: A1, A4 and A11 are all residuals of the *same* backward
pass, so they add.

**Cells used, and why.** A two-atom silicon cell hides all of this, so nothing is sized on one.

| cell | where it comes from | what it shows |
|---|---|---|
| 45-atom NiBr2 slab | `PERFORMANCE.md` P73/P74; `ngm = 3,536,849`, `nbnd = 403`, `npol = 2`, smooth box 200x240x54, measured SCF peak **32.30 GB** | noncollinear + SOC, fully-relativistic PBE PAW; the largest thing measured here |
| `tests/data/qe/bismuthene-soc.in` | committed; `ngm = 139,943` from `reference.out.bismuthene-soc:120`, `nh = 34` | the spinor force `CLAUDE.md` records as not running here at all (P46) |
| `tests/data/qe/bismuthene-soc-small.in` | committed; `ngm = 60,543`, measured at 4.6 GB/iteration (`PERFORMANCE.md:795`) | the stored augmentation route |
| `calculations/nbse2-fermi-surface/nbse2.in` | committed result in `nbse2.json`; `nk = 43`, `nbnd = 24`, `npwx = 9804` | the many-k regime |
| `benchmarks/h40-chain-lsda.in` | committed; `nspin = 2`, `nbnd = 56`, `nk = 1`, box 40x40x640 | `nspin = 2` at accelerator defaults |
| fcc Ni magnon | `PERFORMANCE.md:4581`'s own production case; `nbnd = 30`, `nm = 561`, `nw = 9` | the magnon response |
| 64 k / 200 occupied / 20000 PW | `PERFORMANCE.md:1570-1574`'s own hypothetical | the strain response. **No run of this size exists here** |
| 157-atom FePc/SnTe slab | P67/P68; `nbnd = 1020`, gamma storage | the sizing model |

## 1. Ranked

Ranked by bytes at a realistic cell against risk and effort. The top three change the peak of a
run that currently does not fit on this machine or does not start at all.

| # | item | § | site | cell | saved | effort | cat |
|---|---|---|---|---|---|---|---|
| 1 | Augmentation scan tapes the whole `Q_ij(G)` | A1 | `augmentation.py:530`, `:558` | NiBr2 slab | **76.5 GB** of reverse tape (982 GB at 15 labels); 2.73 GB at bismuthene-soc | 2 lines | a |
| 2 | `vc_relax` holds four Calculations into the final SCF | A3 | `vc_relax.py:341` | NiBr2 scale | **18 GB** resident | 1 line | a |
| 3 | `run_relax` holds the previous step | A2 | `relax.py:434` | NiBr2 scale | **7.7 GB** resident, +24% on a 32.30 GB peak | 3 lines | a |
| 4 | PAW one-centre tape | A4 | `onecenter.py:128` | NiBr2 | **3.28 GB** measured (4.08 -> 0.80), and 0.4 GB forward with it | 1 line -> ~40 | a |
| 5 | Structure factor is a reverse residual | A5 | `potentials.py:224`, `augmentation.py:847` | NiBr2 | **2.5-5.1 GB** of tape (CSE-dependent) | 2 lines | a |
| 6 | `Calculator` and `run_scf` retention | A6, B2 | `calculator.py:524/679/858`, `driver.py:4489` | 64k/200-band **(hypothetical — no run this size exists here)**; NiBr2 | **24.6 GB** retained / 49 GB double-live; 8.8 GB of wavefunction sets; 2.19 GB/k span | 6 lines | a + b |
| 7 | E-field holds three projector-velocity blocks | A7 | `efield.py:329` | P25 yardstick | **1.84 GB** across 18 iterations, *never read* on PAW | 10 lines | a |
| 8 | `spinchi0` stacks the band loop | A8 | `spinchi0.py:472` | fcc Ni | **1.36 GB** — and `PERFORMANCE.md` says this was fixed | 3 lines | a |
| 9 | `_species_charge` doubles the stored route | A9 | `augmentation.py:157` | bismuthene-soc-small | **1.12 GB**; the `AUG_MAX_BYTES` gate measures half the working set | 2 lines + docstring | a |
| 10 | Bootstrap Dyson iterates every frequency | A10 | `dyson.py:122-126` | nm=285/nw=121 | **1.1-1.3 GB** and **88%** of 37.6 s | 25 lines + a flag | a |
| 11 | Stress tapes a real `\|psi\|^2` | A11 | `energy.py:520` | NiBr2-scale spinor stress | **1.10 GB**, a 50% surcharge on psi | 1 line | a |
| 12 | `PawSpecies` materialises a rank-1 outer product | A12 | `onecenter.py:604` | NiBr2 Ni | **0.55 GB/label** (8.29 GB at 15), a floor under `jvp` | 40 lines | a |
| 13 | `vkb` is full-k and outside the dial | A13 | `projectors.py:69` | nbse2 | **355 MB** | 60 lines | a |
| 14 | Velocity holds four full-k blocks | A14 | `velocity.py:302` | AlAs (measured) | **200 MB** | 20 lines | a |
| 15 | `sum_band` vmaps the spin axis | A15 | `density.py:109/118/180` | h40 at accelerator defaults | **0.21 GB** of peak (halves a 2.75 GB stage) | 3 lines | a |
| 16 | PAW one-centre forward set has no dial | B1 | `onecenter.py:128` | NiBr2 | 0.35-0.45 GB, 1.2% of peak — **done with A4**, and it is what makes A4 work | 30 lines | b |

**What legitimately adds.** A force and a stress do not tape the same set, so the groups differ:

- **PAW spinor force on the 45-atom cell**: A1 + A4 + A5 are all residuals of that one backward
  pass, **−79 to −82 GB** of tape (76.5 + 3-5 + 2.5-5.1, the last CSE-dependent).
- **PAW spinor stress on the same cell**: A1 + A4 + A11, plus A5's `_atom_phases` half —
  **−81 to −83 GB**. A11 is stress-only (`kinetic` carries the strain, and under a displacement the
  term is dead-coded); A5's `_structure_factors_at` half is reached by a force.
- **A relaxation's resident set**: A2 + A3 are the same stage, **−25.7 GB**.

Do not add across those groups: they are three different moments of a run.

---

## 2. (a) Wins that need no new physics and break no rule

### A1. The tabulated augmentation charge's `lax.scan` stacks its residuals, so P73 does not reach the backward pass

**Site.** `defumat/pseudo/augmentation.py:521-530` (`_tabulated_charge`'s body and scan) and
`:534-558` (`_tabulated_integrals`, same shape, for any reverse-mode consumer of `newd`).

```
block = jnp.einsum("ijc,ijc->c", build(gcart_chunk), weighted)   # :527
_, blocks = jax.lax.scan(body, None, jnp.arange(nchunks))        # :530
```

**Shape and dtype.** The residual is `(nchunks, nh, nh, chunk)` = `(nh, nh, npad)` complex
(`cell.precision.complex`) — the dense `Q_ij(G)` table reborn on the tape — plus a second
`(nchunks, nat, chunk)` = `(nat, npad)` complex, a stacked copy of the atom phases.

**Mechanism, verified at the jaxpr rather than argued.** Under `at_positions` (driver.py:2150,
augmentation branch :2202-2205) `TabulatedAugmentation.at_positions` (:508-512) replaces only
`phases` and re-pads the *same* `gcart`, and `forces/energy.py:173` differentiates through it. So
inside the body `build(gcart_chunk)` is a known value and `weighted` is not, and transposing
`einsum("ijc,ijc->c", Q, weighted)` with respect to `weighted` needs `Q`. `build(gcart_chunk)`
depends on the scan index through `dynamic_slice`, so it cannot be hoisted as a loop-invariant
residual; partial-eval stacks it. Traced on toy shapes (`make_jaxpr` only, no execution, 3x3x4x5):
the known scan emits `bd:f32[5,3,3,4]` and `bc:f32[5,2,4]`, and wrapping the body in
`jax.checkpoint` removes both from the jaxpr entirely.

**Arithmetic.**

| cell | `nh` | `ngm` | `chunk` | `npad` | residual |
|---|---|---|---|---|---|
| bismuthene-soc, Bi | 34 | 139,943 | 8192 | 147,456 | 1156 x 147,456 x 16 = **2.73 GB** |
| NiBr2 slab, Ni | 34 | 3,536,849 | 8192 | 3,538,944 | 1156 x 3,538,944 x 16 = **65.46 GB** |
| NiBr2 slab, Br | 14 | 3,536,849 | 8192 | 3,538,944 | 196 x 3,538,944 x 16 = **11.10 GB** |

`nh = 34` is `sum(2l+1)` over `Ni.rel-pbe-spn-kjpaw_psl.1.0.0.UPF`'s ten betas at
`l = 0,0,1,1,1,1,2,2,2,2` (`projector_channels`, `projectors.py:53-58`) — not `sum(2j+1)`, which
is `natomwfc`'s rule. `chunk = _aug_chunk(34, ngm)` = `2^floor(log2(256 MiB / (34^2 x 16)))` = 8192.
Br's `nh = 14` is an estimate for a relativistic 4s4p set; that dataset is not committed here, so
the 11.10 GB scales linearly in it. Total for the slab as written with one label per species:
**76.5 GB**. Written the way P73 records the production run — one species per magnetic site —
`charge` runs one scan per species and XLA will not CSE across separate while-loops, so fifteen
Ni scans stack: **982 GB**, which is P73's own deduplication defect reappearing on the tape.

**There is a floor and it is exact.** `build_augmentation:755-758` takes this route only when
`stored_bytes = nh^2 x ngm x 16 > AUG_MAX_BYTES` (2 GiB), and the stacked residual equals
`stored_bytes` to within `npad/ngm = 1.0006`. **The tape is >= 2 GiB by construction on every cell
that takes this path** — the tabulated scheme is chosen exactly when the array it rebuilds is too
large to store.

**Live simultaneously.** Worse than resident: a scan residual is live for the whole backward pass.
`energy_at` (forces/energy.py:407-412) calls `moved.becsum(...)` then `moved.density(...)` inside
one `jax.grad`, so it coexists with psi (2.19 GB at nk=1), `vkb` (2.6 GB), the dense-grid fields
and the rest of the tape.

**Fix.** `jax.lax.scan(jax.checkpoint(body), ...)` at `:530` and `:558`. Under remat the body's
residuals are its inputs; `gcart`, `phases`, `becsum` and `mask` are closed over and loop-invariant,
so they become single arrays rather than stacked ones. What remains is the `(npad, 3)` `gcart`
slices, which are a stored field already.

**Cost.** One extra evaluation of `build` over the padded sphere per backward pass, i.e. one more
`charge` rebuild. The docstring at `:531-538` prices a rebuild at **0.174 s** for `charge` on
`si8-us-1k` against 0.054 s stored and a 5.42 s SCF — fractions of a second there, seconds on a
slab, against a derivative that costs minutes.

**Rules.** `chunk` is `eqx.field(static=True)` and the G set is already padded to `npad` with a mask
killing the padding in the contraction, so remat changes no extent. `jax.checkpoint` is the
sanctioned tool. No host sync, no branch on a traced value, `dynamic_slice` reads unchanged. The
`_tabulated_integrals` carry already takes `phases.real.dtype` (the comment at `:553-557` says why),
so no dtype literal enters. Nothing here is wavefunction-shaped, so R6 is untouched.

**Effort.** 2 lines, plus a docstring paragraph, plus the measurement.

**What it does not settle.** Under a stress `Calculation.at_strain` (driver.py:2426-2429) calls
`build_augmentation` afresh, so `gcart` is traced, `build` becomes unknown and the residual set
changes to its own per-chunk intermediates rather than `Q`. Directionally worse; the size is not
settleable by reading, so do not write "strictly worse."

**Measurement.** `jax.jit(jax.grad(energy_at)).lower(*ShapeDtypeStructs).compile().memory_analysis()`
at those shapes with `DEFUMAT_AUG_MAX_BYTES` forcing each route in turn, before and after. It
compiles and allocates nothing. `tools/gpu/force_memory.py` is the existing instrument.

**Why it is not already done.** `grep -rn "jax.checkpoint\|jax.remat" defumat/` returns exactly one
hit and it is a comment: `augmentation.py:296`, the recorded null for `_qrad_kernel` in the
**stored** route. Nothing in the package is rematted.

---

### A2. `run_relax` holds the previous ionic step's Calculation and SCFResult through the whole of the next step

**Site.** `defumat/workflows/relax.py:434` (`previous = calculation`), never dropped — it is
rebound only at `:434` of the following iteration, so it is live through that iteration's
`run_scf` (`:364`) and `compute_forces` (`:377`). The loop variable `result` is the same: `result
= run_scf(...)` evaluates its right-hand side with the previous `SCFResult` still bound.

**Shape.** `at_positions` (driver.py:2182-2234) shares the basis, both G-sets, the radial tables,
the augmentation table, the PAW and spin-orbit coefficients and `projector_core.columns` by
`copy.copy`, and **eagerly rebuilds** per object: `projectors.vkb` `(nk, npwx, nkb)` complex
(`_apply_phases` returns a materialised array, projectors.py:115-126), `augmentation.phases`
`(nat, npad)` complex (augmentation.py:509-513), `vltot` and `rho_core` on the dense grid,
`rho_core_g` `(ngm,)`, `ewald` and `wfcU`. None of it is lazy. The stale `result` carries
`wavefunctions` `(nspin, nk, nbnd, npol*npwx)` complex plus `density` and `potential`.

**Arithmetic, 45-atom NiBr2 scale** (`nat = 45`, `ngm = 3,536,849`, `nbnd = 403`, `npol = 2`,
`npwx ~ 1.70e5` from `N_smooth/15.28` with `N_smooth = 200x240x54`):

```
phases        45 x 3,536,849 x 16              = 2.547 GB
vkb           1.70e5 x nkb x 16 per k          = 2.77 GB/k   at nkb ~ 1020
vltot + rho_core + rho_core_g                  = 0.17 GB
stale Calculation                              ~ 5.5 GB      at nk = 1
stale SCFResult wavefunctions  403 x 2 x 1.70e5 x 16 = 2.19 GB/k
                                        total  ~ 7.7 GB
```

**The sharp line.** 32.30 + 7.7 = **40.0 GB on a 39 GB machine**. A relaxation of this cell does
not fit where its SCF does, and the reason is dead weight rather than physics.

**Cross-check, independent of the estimate.** `PERFORMANCE.md:4589`'s own bracketed reads put a
whole `Calculation` build at 8.20 GB and "live afterwards" at 9.37 GB, so a stale
Calculation-plus-result of ~7.7 GB is the right order.

**Live simultaneously.** Yes, and under the step's true peak rather than beside it — including
`compute_forces` at `:377`, which for ultrasoft/PAW is the larger of the two per-step peaks
(`PERFORMANCE.md:2401` measures `si8-us`'s reverse pass at 10.49 GB against 1.4 GB for its SCF).

**Fix.** Three host-side reference drops.

```python
# after _extrapolate returns, relax.py:456-458
del previous
# top of the loop body, before relax.py:364
result = None
```

`_extrapolate` (`:470-505`) returns bare arrays and closes over nothing, so dropping `previous`
immediately after it is sound. **Ordering is load-bearing**: `previous` must be dropped *after*
that call, which reads `previous.starting_density()` and `previous.becsum(...)`. `result = None` is
safe against the post-loop `RelaxResult(scf=result)` because the body rebinds it on the next
statement, and `RelaxStep` keeps no reference to the object.

**Cost.** Zero. No recomputation, no recompilation — `at_positions`' output shapes are identical
between steps and executables are keyed on shape.

**Rules.** Host-side Python lifetime management outside every traced path. No shape becomes
data-dependent; no `np.asarray` enters a gradient (the gradient inside `compute_forces` closes over
`calculation`, never over `previous`); no dtype is named; no k axis moves.

**Effort.** 3 lines.

**Caveats, stated.** `nkb ~ 1020` is an estimate (`15 x ~36 + 30 x ~16`), and the `vkb` term scales
linearly in it. The NiBr2 run on record is a helix, which relaxes through `relax_spiral_q` rather
than `run_relax` — this is a cell of that scale, not that run.

**Measurement.** One bracketed `peak_bytes_in_use` read across two ionic steps, the way P74
bracketed the SCF.

---

### A3. `vc_relax` ends holding four Calculations and two SCF states while it starts a fresh SCF

**Site.** `defumat/workflows/vc_relax.py`: `base` at `:245`, `current = base` at `:265`, `previous
= current` at `:326`, `current = _advance(...)` at `:327`, `relaxed = current.system` at `:333`,
`relaxation_scf = result` at `:334`, `final = Calculation(...)` at `:341`, `result = run_scf(...)`
at `:343`. None of `base`, `previous`, `current` or `relaxation_scf` is ever dropped.

**Why each stale object is nearly a whole Calculation.** `_advance` (`:384`) builds every step as
`base.at_cell(at).at_positions(positions)` — cumulative from `base`, which makes `base` a required
retention through the loop and a purely dead one from `:333` on. `at_cell` (driver.py:2236) routes
through `at_strain` (driver.py:2313), which rebuilds far more than `at_positions` does: the
projector core whole (`columns` `(nk, npwx, ncs)` and `kg` `(nk, npwx, 3)`), `kinetic`, the
augmentation whole, `vloc_species`, `vltot`, `rho_core_species` and `rho_core`.

**Arithmetic, NiBr2 scale.** `PERFORMANCE.md:4589` puts one Calculation at **8.20 GB**.

```
base + previous, both dead from :333        ~ 16 GB
relaxation_scf wavefunctions                  2.19 GB/k
                              droppable at :341  ~ 18 GB
```

**Live simultaneously, at the worst instant in the package.** `:341` builds a fourth Calculation
(another ~8.2 GB, sharing nothing, since it comes from a fresh `System`), `:343` runs a complete
SCF from the atomic superposition — the deliberate "nothing carried over" branch — climbing
toward the ~32 GB the same cell's SCF was measured at, and `:346-347` then take a force gradient
and a stress gradient on top.

**Fix.** One line before `:341`:

```python
base = previous = current = None
```

`relaxed` at `:333` has already captured the only thing needed (`current.system`).

**Cost.** Zero.

**Rules.** Host-side drop between compiled calls. No shape, no traced path, no dtype, no k axis.

**Effort.** 1 line. (The public-API half — `relaxation_scf` itself — is B2.)

**Caveats.** `at_strain` refuses a spin spiral (driver.py:2364), so the NiBr2 helix cannot
vc-relax; forces and stress for a noncollinear spinor slab are P46's regime, already known not to
run here. This is a cell of that scale and dataset, not that run.

---

### A4. PAW's one-centre reverse pass holds every atom's XC quadrature, where QE holds one sphere

**Site.** `defumat/paw/onecenter.py:128`:

```python
atom_energy, atom_ddd = jax.vmap(
    partial(onecenter_species, paw, meta_c=meta_c, axis=axis),
    in_axes=1, out_axes=(0, 1),
)(values)
```

reached from `forces/energy.py:412` (`epaw, _ = moved.onecenter(becsum_)`) inside the
differentiated region, with `becsum_` carrying the positions. The `ddd` half of the return is
discarded by `energy_at` and is dead-coded away, so this is the energy path only.

**Shapes.** Inside, `rho_rad` and `density` are `(nspin, nx, mesh)` real (`:253`, `:265`); the
`nspin == 4` branch (`:269-280`) resolves them onto the local spin frame as a 2-channel pair; for
a GGA, in `paw/gradient.py`, `channels` `(2, nx, mesh)` at `:206`, `grad` `(2, 3, nx, mesh)` at
`:209`, `h` at `:216-217`, `h_lm` `(2, 3, nlm_table, mesh)` at `:219`, `out_rad` at `:222`,
`potential_rad` at `:227`. All real at the policy's real dtype. Under the vmap each gains a
leading `nat_t`.

**Quadrature size, derived not assumed.** `Ni.rel-pbe-spn-kjpaw_psl.1.0.0.UPF`: `mesh_size = 1195`,
`l_max_rho = 4`, ten projectors. `build_angular_grid` with `LM_FACTOR = 3`, `XLM = 2`
(angular.py:49,54,100-102) gives `lmax = 3x4 + 2 = 14`, `nphi = 15`, `ntheta = 8`, **`nx = 120`**.
`mesh` is the **full** UPF mesh, not the augmentation cutoff — `_build_paw_species`
(`:575-598`) sets `iraug = min(cutoff_index, mesh)` and *zeroes* past it while keeping the array
at `mesh`.

**Forward arithmetic** (Ni, `nat_t = 15`, `nspin_mag = 4`, `nx = 120`, `mesh = 1195`, float64):

```
grad      (nat,2,3,nx,mesh)      15 x 6 x 120 x 1195 x 8  = 103.2 MB
h_lm      (nat,2,3,49,mesh)      15 x 6 x 49 x 1195 x 8   =  42.2 MB
rho_rad   (nat,4,nx,mesh)        15 x 4 x 120 x 1195 x 8  =  68.8 MB
channels  (nat,2,nx,mesh)        15 x 2 x 120 x 1195 x 8  =  34.4 MB
```

with `h`, `out_rad` and `potential_rad` in the same classes: **0.35-0.45 GB** with all of them
live, against ~30 MB one atom at a time. That is 1.2% of the 32.30 GB peak, and it is B1.

**The tape is the bigger half, and the vmap is not its cause.** Every atom's one-centre reverse
pass is live simultaneously because they are all in one differentiated region — identically so
whether the atom axis is a `vmap` or fifteen width-one vmaps, which is what the
one-species-per-magnetic-site writing gives. ~~Remat is the fix; the dial is not.~~

> **Measured false, 2026-09-13, and the correction is the item's whole content.** Remat alone is a
> **regression**, and the dial alone is worse than either; only the two *together* work. Under a
> `vmap` the rematted backward pass recomputes every atom **simultaneously**, so the recomputation
> is exactly as wide as the tape it removed — the per-atom slope is unchanged and a fixed barrier is
> added on top. Compiled-gradient `temp_size_in_bytes`, Ni `rel-pbe-spn-kjpaw` at `nspin = 4`:
>
> | atoms | `vmap` (as written) | `vmap` + remat | scan | scan + remat |
> |---|---|---|---|---|
> | 1 | 267 MB | 796 MB | 269 MB | 796 MB |
> | 4 | 1087 MB | 1620 MB | 1717 MB | 796 MB |
> | 8 | 2175 MB | 2713 MB | — | 796 MB |
> | 15 | **4078 MB** | — | — | **797 MB** |
>
> So the estimate below (~50 temporaries, 3-5 GB for the Ni sublattice) was right: measured
> **272 MB per atom**, 4.08 GB at fifteen, and 244 grid-sized temporaries rather than 50. Chunked
> *and* rematted the peak goes **flat in the atom count**. The fix is `_paw_atom_batch` +
> `map_axis`, which **closes B1 in the same change** — the chunk is not an alternative to the remat,
> it is what makes the remat pay. `PERFORMANCE.md` has the whole-force numbers and the cost.

Order of the tape: ~50 grid-sized temporaries in the PBE-spin kernel x 34.4 MB ~ **1.7 GB** for
the Ni sublattice, and ~1.3 GB for the 30 Br atoms at `nx = 45` — **3-5 GB**. The count of ~50 is
an estimate and is the whole size claim. It is conservative rather than generous: `spin_potential`
(xc/functional.py:489-518) calls `jax.grad(total)` **twice** — once at `stop_gradient(channels)`
and once at the masked `regular` — so each forward one-centre call already contains two inner
reverse passes with their own tapes, and `spin_energy_density` (`:456-487`) evaluates `raw` twice,
all before the GGA pass on top.

**Fix.** ~~`jax.checkpoint(onecenter_species)` under the vmap.~~ `jax.checkpoint` under a **chunked**
`map_axis` over the atom axis — see the correction above. The per-atom residual falls to that atom's
`becsum`, `4 x 34^2 x 8 = 37 kB`, and the peak becomes one chunk's recompute.

**Cost, measured rather than assumed.** Nothing, above the crossover. Per-iteration SCF on
`si10-paw-pbe` **1.403 -> 1.281 s**, its compiled force **0.505 -> 0.418 s**, its stress
9.41 -> 9.03 s; on the built 4-atom `si4-paw` the force is 0.123 -> 0.120 s. It is *below* the
crossover that it costs, which is why the default is a count: two atoms of the spinor PAW cell are
**0.761 -> 1.286 s** and 547 -> 718 MB, so a sublattice of three or fewer keeps the `vmap`. In
isolation the one-centre gradient alone is 1.63x slower at eight atoms (3.833 -> 6.241 s); the whole
force is *faster*, because the peak allocation halves.

**Do not** swap the vmap for a chunked `map_axis` and stop there — that bounds the forward set and
not the tape, by A1's verified mechanism. Measured: scan without remat is **1717 MB** at four atoms
against the vmap's 1087.

**Rules.** No shape changes, no dtype literal, no host sync, no k axis (`k` does not appear in this
call). `jax.checkpoint` is the identical function re-executed, so differentiability is exact.

**Effort.** 1 line, plus the timing.

**Withdrawn sub-item.** An earlier lens proposed replacing the separate `spin_potential` and
`spin_energy_density` calls at `:270-280` with a fused evaluation. Refuted:
`spin_potential_and_energy_density` (functional.py:521) is itself two calls, and its docstring says
why — "the potential's derivative is masked at a fully polarized point and the energy density's is
not." The fused `potential_and_energy_density` at `scf/potential.py:243` is the `nspin == 1` path.
This is a correctness choice about the saturated point, not waste.

**Measurement.** `jax.jit(jax.grad(lambda b: onecenter_species(paw, b)[1])).lower(*ShapeDtypeStructs)
.compile().memory_analysis()` at the NiBr2 Ni shapes with `nat_t = 1` against `nat_t = 15`. It
allocates nothing, and the same run gives B1's missing forward figure and, from the compiled time,
the `atom_batch = 1` cost that `PERFORMANCE.md` never separated.

---

### A5. The per-atom structure factor is a reverse-mode residual, in two places, on every geometry derivative

**Site.** `defumat/pseudo/potentials.py:224`:

```python
phases = jnp.exp(-1j * (g @ positions.T))   # (ngm, nat)
```

inside `_structure_factors_at` (`:218`), contracted to `(ntyp, ngm)` on the next line. The same
object exists as a *field*: `AugmentationCharge.phases` `(nat, ngm)` complex (augmentation.py:80),
rebuilt by `_atom_phases` (`:847`) on every `at_positions`/`at_strain`.

**Why it is a residual and not a transient.** `exp`'s JVP is `ans * t`, so its VJP saves its own
output. The `(ngm, nat)` complex array is therefore saved by any `jax.grad` with respect to
positions — not something XLA can fuse away.

**Arithmetic, 45-atom NiBr2.** `45 x 3,536,849 x 16 = 2.546 GB` per copy (the augmentation field at
`npad` is 2.548 GB). `Calculation.at_positions` calls `combine_species` twice — driver.py:2203 for
`vloc` and `:2209` for `rho_core`, both against the dense set — so the tape holds **one copy (2.55
GB) if XLA's CSE merges the two identical calls, two (5.1 GB) if not**, with the resident
augmentation field beside them.

**Fix.** `jax.checkpoint` around `_structure_factors_at` and `_atom_phases`. The backward pass then
recomputes one `exp` per `(atom, G)` — trivial arithmetic against an FFT-bound iteration — and the
residual falls to `positions` and the `(ngm, 3)` G set, both of which are held anyway.

**Rules.** Shapes unchanged, dtype unchanged, no host sync, no traced branch, no k axis. Remat is
the identical function re-executed.

**Effort.** 2 lines.

**Explicitly not the eigts factorisation.** QE's `PW/src/struct_fact.f90` stores `strf(ngm, ntyp)`
(113 MB here) plus three 1-D `eigts1/2/3` tables (~0.85 MB), from which any per-atom factor is a
product of three lookups. That would shrink the *resident* 2.55 GB, but it tapes three gathered
`(nat, ngm)` arrays where `exp` tapes one, so it makes the tape worse unless rematted. It is C2,
not this.

**Measurement.** `memory_analysis()` on the compiled force decides 2.55 GB against 5.1 GB.

---

### A6. `Calculator` and `run_scf` retention: four droppable holdings, all host-side

Four sites, one shape of defect — an attribute or local that is live long after its last read, or
that is still bound while its replacement is computed. All four fixes are host-side reference
management between compiled calls: no shape becomes data-dependent, no `np.asarray` enters a
gradient, no dtype is named, no k axis moves, no host sync is added to any inner loop.

**(i) The strain response is cached for the object's lifetime and double-lives on recompute.**
`calculator.py:857-858`:

```python
if self._strain_response is None or options:
    self._strain_response = strain_response(...)
```

The right-hand side evaluates while the attribute still points at the previous object, and `or
options` means **any** non-empty `options` recomputes — so the documented recompute path is
exactly the double-live path. `StrainResponse.dpsi` (response/strain.py:132) is a `(3,3)` object
array made symmetric at `:486`, so **6 distinct** `(nspin, nk, nocc, npwx)` complex blocks;
`nspin` is pinned to 1 by `_require_one_spin_channel` (`:634`). On `PERFORMANCE.md:1570-1574`'s own
hypothetical cell (64 k, 200 occupied, 20000 plane waves) one block is `64 x 200 x 20000 x 16 =
4.096 GB`, so **24.6 GB retained** and 49 GB double-live. With a collinear ultrasoft or PAW dataset
`overlap_derivatives` (`:338`) adds six more at `(nspin, nk, nbnd, ndim)` — `nbnd`, not `nocc` —
for **>= 49 GB retained**. On the 16-atom yardstick (100 k, 32 bands, 3000 PW) one block is 153.6
MB and the six are 0.92 GB. `drho` and `dvscf` at `(3,3,1,n1,n2,n3)` add ~0.1 GB and are noise.
There are only three invalidations (`:333`, `:465`, `:528`) and no release method. Worse,
`:528` sets `_strain_response = None` only *after* `self._scf = run_scf(...)` on `:524`, so a stale
strain response is resident under the whole of the next SCF including its Davidson peak.
**Free fix: `self._strain_response = None` on the line before `:858`.** 1 line.
*The cell is hypothetical — no strain response of this size has been run here.*

**(ii) `_scf` and `_relax` double-live the same way.** `calculator.py:524` and `:679`. Re-running at
a tighter `conv_thr` holds the old wavefunctions through the whole new SCF including its Davidson
peak. **Free fix: `self._scf = None` before `:524`, `self._relax = None` before `:679`.** 2 lines.

**(iii) Four wavefunction sets can coexist on one Calculator.** `_scf` (`:524`), `_seed` (`:1465`,
the parent's converged result, never cleared), `_relax.scf` (`:679`), and for `variable_cell=True`
`_relax.relaxation_scf`. At 2.19 GB/k on a NiBr2-scale spinor cell that is **8.8 GB** — against a
`get_scf` docstring (`:505-509`) that justifies the single slot *precisely* because "what it holds
is the wavefunctions -- the largest arrays in the process -- and a keyed cache would quietly hold
several sets of them." `_scf_options["starting_from"]` is the *same object* as `_seed`, an extra
reference rather than an extra state. Clearing `_seed` is **not** free and is B2.

**(iv) The continuation's promoted span is live for every iteration and read by the first.**
`scf/driver.py:4489`, `starting_wavefunctions = state.wavefunctions`. Its only consumer is
`:4739-4740` inside `if wavefunctions is None:`, and `wavefunctions = None` is assigned exactly once
before the loop at `:4610` (the only other assignment, `:4223`, is in a different function). So the
guard opens once and the span is provably never read after iteration 1. For a 1->4 or 2->4
promotion `promote_wavefunctions` (continuation.py:586-595) builds a **freshly allocated** `(nk,
2*nbnd_src, 2*npwx)` array by concatenating `([up, 0], [0, down])` — the same size as the target
run's own wavefunctions: `403 x 2 x 1.70e5 x 16 = 2.19 GB/k`, live under `diagonalize` at every
iteration, which `PERFORMANCE.md:4589` brackets at 27.23 -> 31.97 GB on this cell. **+6.8% held
from the second iteration on**, on the path `CLAUDE.md` advertises for promoting a collinear result
into a noncollinear run.
**Free fix: `state = starting_wavefunctions = None` after `:4741`**, with `state` initialised to
`None` before the `if starting_from is not None:` branch at `:4474` or the clear raises
`NameError`. 2 lines. Narrow cases where it frees nothing, stated: when source and target share
`npol`, `promote_wavefunctions` returns `psi` or `psi[0]` and allocates nothing new; on a
checkpoint resume `resumed_state` holds it anyway.

**Effort for the free half of all four: 6 lines.**

---

### A7. The electric field holds three `(nk, npwx, nkb)` projector-velocity blocks across the whole loop, and never reads them on PAW

**Site.** `defumat/response/efield.py:313` (`bare, commutators, projector_velocities = [], [], []`),
`:326` (`commutators.append`), `:329` (`projector_velocities.append`), `:333` (`bare.append`). Both
lists are built before the loop at `:323` and next read at `:408-412` (Born charges) or `:420`
(`keep_internals`).

**Shapes.** `commutators`: 3 x `(nspin, nk, nocc, npwx*npol)` complex. `projector_velocities`: 3 x
`(nk, npwx, nkb)` complex (annotated velocity.py:196), no spin axis.

**When `commutators` is free and when it is not.** For a norm-conserving run `position` is never
rebound between `:326` and `:333`, so `commutators[a] is bare[a]` and the list costs nothing. The
split happens only when `_augmentation_dipole` (`:589-596`) is non-None, i.e.
`calculation.is_ultrasoft`, which driver.py:2713-2714 defines as `self.augmentation is not None` —
**true for PAW as well as ultrasoft**.

**Arithmetic on `PERFORMANCE.md`'s own P25 yardstick** (`:1731`: 16 atoms, 100 k-points, 32 bands,
3000 plane waves; ultrasoft at `nh = 8` so `nkb = 128`; `nspin = 1`):

```
one band block            100 x 32 x 3000 x 16       = 153.6 MB
the six the record counts (bare, dpsi, :1573)        = 921.6 MB
commutators               3 x 153.6                  = 460.8 MB
projector_velocities      3 x 100 x 3000 x 128 x 16  =   1.843 GB
                                   true resident total = 3.23 GB
```

The recorded six are **28.6%** of it. The ratio is dataset-driven rather than cell-driven —
`3 nkb / (6 nocc nspin)` — so 2x the six at `nspin = 1`, ~1x at `nspin = 2`, and two to three
times worse again for a PAW Ni or Bi dataset with `nh` in the twenties.

**Live for the longest span in the routine**, through all ~18 iterations (P24's measured count)
during which the CG's four band blocks, `bare`'s and `dpsi`'s three each, and the mixer's history
are also live.

**Dead weight on PAW.** `require_born_charges` (born.py:112, 157-166) raises for
`calculation.is_paw`, so a PAW field run must pass `born_charges=False` — and then neither
`:408-412` nor `:420` touches either list. **2.3 GB carried for eighteen iterations and never
read.**

**Fix.** Gate retention on `born_charges or keep_internals` — pure bookkeeping, zero cost. Where
Born charges *are* wanted, rebuild the three `projector_velocities` at `:396` instead of holding
them: `PERFORMANCE.md:1518` puts all three directions of the velocity `jvp` at **2.9 s** on the
si-epsilon cell, against 1.84 GB held across eighteen iterations.

**Do not recompute `commutators`** — each is a Sternheimer solve, which is the "plus three for the
bare `P_c r|psi>`" `PERFORMANCE.md:1552` counts.

**Rules.** The rebuild is the same `velocity.projectors(direction)` call already made at `:316`, so
it keeps `tau` traced — which matters: velocity.py:233-243 records that `np.asarray(positions)`
there silently drops the atoms from a third derivative, and `:204-218` puts the
about-the-atom-centre convention at 2% if lost.

**Effort.** ~10 lines.

**Honest limit.** The rebuild still holds all three simultaneously at `:396-412`, so what is saved
is the overlap with the CG's working set during the loop, not the peak at the Born-charge assembly.

**One-line measurement that pins the byte figure.** Read `calculation.projectors.nkb` for the
actual dataset.

---

### A8. `spinchi0` stacks `nbnd` copies of the response matrix — and the record says this was fixed

**Site.** `defumat/tddft/spinchi0.py:472-473`:

```python
total = jax.lax.map(one_band, (fields_up, eig_up, occ_up, slope_up))
return jnp.sum(total, axis=0)
```

`one_band` returns one `(nw, nm, nm)`; `lax.map` is a scan whose per-iteration output is stacked
into a full `ys` buffer, and XLA does not promote a while-loop output buffer into a following
reduction. `grep -n "scan\|lax\." defumat/tddft/spinchi0.py` returns exactly two hits, both the
`lax.map` — there is no scan carry anywhere in the file.

**Arithmetic, fcc Ni** (`PERFORMANCE.md:4581`'s own production case: `nbnd = 30`,
`ecut_response = 60` so `nm = 561`, the root-finder's 8 frequencies plus the static point so
`nw = 9`):

```
one (nm, nm)      561^2 x 16                = 5.04 MB
the answer        9 x 5.04                  = 45.3 MB
the stack         30 x 9 x 561 x 561 x 16   = 1.36 GB      -- 30x the result
```

At the record's own `nw = 8` it is **1.21 GB for a 40.3 MB result**, reproducing that entry's
"1.2 GB of intermediate for a 40 MB result on nickel" to the quoted digits — a figure computable
only from the stacking form.

**It is the peak of the module, not a bystander.** The two-channel state block (`npwx ~ 580`,
`nk = 10`, `nbnd = 30`) is 5.6 MB and `fields_up`/`fields_dn` are ~7 MB each — three orders below
the stack and live across it. Under `k_batch = None` (the accelerator default) `sum_k` routes
through `_chunk_sum = jnp.sum(jax.vmap(fn)(chunk), axis=0)`, so the stack gains a k axis:
`nk x 1.36 GB`.

**The tell that the scan once existed:** the map's element parameter is still named `carry` (`:446`).

**Fix.** Call the existing `sum_bands` on the band axis — it delegates to `sum_k`
(batching.py:383-395), which is axis-agnostic. Do **not** write a new `sum_axis` helper.

**Cost.** Zero on flops, transforms and einsums. The accumulation order changes, which is the
~1e-15 round-off `tests/unit/test_batching.py` already pins for `sum_k`.

**Two caveats.** The zero carry must come from `jax.eval_shape` of the body — the pattern `sum_k`
already uses at batching.py:288-289 — never a `jnp.zeros(..., jnp.complex128)` literal. And on an
accelerator `_resolve_band_batch` gives `None`, which routes back to `_chunk_sum` and rebuilds the
stack, so either pass an explicit batch here or state that the accelerator end is the stack by
design.

**Rules.** Carry is a fixed `(nw, nm, nm)`; scan length `nbnd` is static; `lax.scan` differentiates
both ways; the axis walked is the band axis *inside* one k-point's body, so R6 is untouched.

**Effort.** 3 lines — **and `PERFORMANCE.md:4581` must be corrected in the same commit** (§5.1).

---

### A9. `_species_charge` builds a second full-size `(nh, nh, ngm)` beside the resident one, so the stored route's peak is twice what the gate measures

**Site.** `defumat/pseudo/augmentation.py:157`, inside `@jax.jit _species_charge`:

```python
weighted = jnp.einsum("aij,ag->ijg", becsum.astype(phases.dtype), phases)
return jnp.einsum("ijg,ijg->g", qgm, weighted)
```

`weighted` is `(nh, nh, ngm)` complex — the same shape as `qgm` — formed in full and contracted
away one line later.

**Arithmetic, bismuthene-soc-small** (`ngm = 60,543` from `reference.out.bismuthene-soc-small:108`,
`nh = 34`): `1156 x 60,543 x 16 = 1.120 GB`, standing beside the identically sized 1.120 GB `qgm`
that is resident for the run. `PERFORMANCE.md:795` records this exact cell at 14.58 s/iteration and
**4.6 GB per iteration**, so the pair is about half of it.

**The gate is the sharper half.** `build_augmentation:755-758` computes `stored_bytes = nh^2 x ngm
x 16` and tabulates above `AUG_MAX_BYTES`, whose docstring (`:311-315`) says "how much `Q_ij(G)` may
occupy" — while the route's live set at the contraction is **twice** that. A cell sized to sit just
under the 2 GiB default actually runs at 4 GiB. `sizing.py:572-573` lists `augmentation Q_ij(G)
(nh,nh,ngm)` and `augmentation phases (nat,ngm)` and nothing for this.

**Fix.** Reassociate:

```python
t = jnp.einsum("aij,ijg->ag", becsum.astype(phases.dtype), qgm)
return jnp.einsum("ag,ag->g", t, phases)
```

Intermediate `(nat, ngm)` = `2 x 60,543 x 16 = 1.94 MB`, a factor `nh^2/nat = 578`. Called per
species with only that species' atoms (`phases[atoms]` at `:106`), so on a 45-atom NiBr2 written
with one Ni label it is `1156/15 = 77x` and with fifteen labels `1156/1 = 1156x`. Flops go from
`nat nh^2 ngm + nh^2 ngm` to `nat nh^2 ngm + nat ngm` — marginally fewer.

**This is a deliberate departure from "Mirror QE", and that is part of the fix.** The present order
**is** QE's: `addusdens_g` holds `qgm(ngm, nij)` *and* `aux2(ngm, nij)`, so the docstring at
`:89-98` is accurate about it. What makes QE's association non-binding here is that `qgm` is
resident in this code and is not in QE's loop. The docstring has to say that, plus a `sizing.py`
line for the intermediate. Rewriting it is not optional.

**Direction under a derivative.** Under a force `weighted` is a forward transient only, because
`d(block)/d(weighted) = qgm` and the backward pass needs `qgm`, which is resident; the cotangent is
a same-shaped backward transient instead. Under a stress `qgm` itself depends on the cell, so
`weighted` becomes a taped residual.

**Rules.** Both forms are fixed-shape einsums with static extents; pure `jnp`, so the gradient is
unchanged in value; the existing `becsum.astype(phases.dtype)` cast is kept, so no dtype literal
enters; no k axis.

**Effort.** 2 lines, plus the docstring and the `sizing.py` line. It reassociates a sum, so
acceptance is the converged total on `si8-us-1k.in` to 1e-12 Ry and the same iteration count.

**Residual risk, unsettled by reading.** Whether `(nat, ngm) x (nh^2, ngm)` is a worse BLAS shape on
CPU than the present one. Measurement, allocating nothing:
`memory_analysis().temp_size_in_bytes` and the compiled time of `jit(_species_charge)` at
`nh = 34`, `ngm = 60543`, `nat = 2`, before and after.

**Nit at the same site.** `:753` hardcodes `16` for the complex byte width, so under
`precision = 'single'` the gate fires at twice the true size (§6.3).

---

### A10. The bootstrap Dyson loop iterates every frequency when its kernel reads one, and broadcasts `nw` identical copies of `fxc`

**Site.** `defumat/tddft/dyson.py:122-126`. Per pass, all `(nw, nm, nm)` complex: `fxc` (`:122`),
`fxc_x = fxc @ x` (`:125`), the `eps0 - fxc_x` temporary, `jnp.linalg.inv(...)`, `x @ inv` and the
new `epsi` (`:126`) — beside `x` (`:109`) and `eps0` (`:111`), and the previous pass's `epsi`, still
bound while `:126` evaluates. Seven simultaneous. `solve_dyson` is an unjitted Python loop — its
exit test at `:132` calls `complex(...)` on a device array — so each is a separately dispatched
buffer rather than a fusable temporary.

`fxc` is `jnp.broadcast_to(kernel, chi.x.shape)` of a single `(nm, nm)` in `_bootstrap`
(kernels.py:254), `_lrc` (`:150`) and `_alda` (`:174`); `_rpa` (`:121`) returns `jnp.zeros_like(chi.x)`,
`nw` copies of zero. JAX's lazy sublanguage is long gone, so `broadcast_in_dim` allocates.

**Arithmetic.** At `PERFORMANCE.md:3425-3426`'s own row (`nm = 285`, `nw = 121`, quoted at a 2.9 GB
expected peak): one array is `121 x 81,225 x 16 = 157.25 MB`, so seven live is **1.10 GB** plus a
batched-LU workspace of roughly one more — **1.1-1.3 GB**. Extrapolated to 8-atom silicon at the
same `ecut_response = 8` (`nm` scales with cell volume, 115 -> ~451) at 200 frequencies: one array
651 MB, seven live **4.6 GB**, and the bootstrap does nine passes of it.

**Time, from the same rows.** 9 passes x 121 frequencies = 1089 `nm^3` inversions where ~129 are
needed (8 static updates plus one final full solve): **88% of the bootstrap's 37.6 s is discarded
work.**

**Terminal peak.** `workflows/tddft.py:248` calls `solve_dyson` after `independent_response` has
returned, so `chi.x` is live as `x`, `eps0` is a second copy, and the loop's five more sit on top.
`chi_0`'s own assembly has finished, so the two peaks are sequential rather than additive — but
`DysonSolution` then *stores* both `epsilon_inverse` and `fxc` at `(nw, nm, nm)`, so 314 MB
survives the call at `nm = 285` for a matrix that is `nw` identical copies. `:153-154` also
recompute `fxc @ x` over all `nw` to read one scalar.

**Fix.** Run the fixed point on the static slice and solve the frequency axis once at the end.
`_bootstrap` reads only `chi.x[static]` and `epsi[static]`, and the convergence test (`:133-135`)
reads only `fxc_x[static_index, 0, 0]`, so iterating `epsi_s = x[s] @ inv(eps0[s] - fxc @ x[s]) + I`
on `(nm, nm)` matrices converges to the same kernel; one full solve with that converged `fxc`
reproduces the loop's last iteration, because the `epsi` the loop returns was itself computed from
that pass's `fxc`. `fxc` then never needs the `nw` broadcast — `jnp.matmul` broadcasts `(nm, nm)`
against `(nw, nm, nm)` natively. The final solve is also the one place chunking `nw` is free, each
frequency being an independent `inv`.

**Cost.** Strictly negative: less memory and ~6x less time. No recomputation, no approximation.

**The gate, which is a correctness guard and not a trade.** The static-slice iteration is exact only
for a frequency-independent kernel. All five registered kernels qualify today, but an ungated
version would silently truncate a future dynamic one — `_lrc`'s docstring names the `beta omega^2`
term deliberately left out. This needs a `static_kernel` flag on `XCKernel`, not an assumption.

**Breakage.** `tests/regression/test_tddft.py:282` and `:287` index `solve_dyson(...).fxc)[0]`.
Keep the public field broadcast (157 MB once) and remove only the `nw`-wide broadcast inside the
loop, or update those two sites.

**Rules.** Nothing here is differentiated or jitted, so grad, R6 and host-sync rules do not apply.
`static_index` is a compile-time index; the final solve's shapes are the ones already in use;
dtypes follow `chi.x` throughout, as `identity = jnp.eye(..., dtype=x.dtype)` already does.

**Effort.** ~25 lines, plus the flag, plus two test sites.

---

### A11. Every stress tapes a full real copy of `|psi|^2` for the kinetic term, on top of `psi`

**Site.** `defumat/forces/energy.py:520-521`, inside `_kinetic_energy` (`:504`, jitted at `:503`):

```python
density = jnp.real(jnp.conj(psi) * psi)          # (nspin, nk, nbnd, ndim)
per_band = jnp.einsum("skbg,kg->skb", density, kinetic)
```

Transposing that einsum with respect to `kinetic` requires `density`, so it is a residual for the
whole backward pass. Traced on toy shapes: the jaxpr carries `f32[1,2,7,11]` — the real
`|psi|^2` — alongside the `c64[1,2,7,11]` it was built from.

**It is a stress residual, not a force's.** `Calculation.at_strain` rebuilds `kinetic` at
driver.py:2406, so the term carries the strain; under a displacement it has no position dependence
and is dead-coded. `energy_at` passes `moved.state_kinetic` for a noncollinear run (`:452`), which
driver.py:2927 builds as the concatenated `(nk, 2 npwx)`, so `ndim = npol * npwx`.

**Arithmetic, NiBr2-scale spinor stress at `nk = 1`** (`npwx ~ 1.70e5`, `ndim ~ 3.4e5`, `nbnd = 403`):
`403 x 3.4e5 x 8 = 1.096 GB`, against `psi` at `403 x 3.4e5 x 16 = 2.19 GB` — a **50% surcharge on
psi, for one scalar**. It scales as `nspin nk nbnd npol npwx x 8`, so a spinor cell at 8 k-points
pays 8.8 GB.

**Why it has never shown.** On `si8-us` it is `20 x 1607 x 8 = 257 kB` of an 11,105 MB peak.

**Live simultaneously.** A residual of the single `jax.grad` in `stress/autodiff.py:74`, so it
coexists with `psi`, `vkb`, the dense fields and the 10-11 GB of radial-transform tape P11 measured.

**Fix.** `jax.checkpoint` on `_kinetic_energy` — it is already its own jitted unit, so the boundary
is drawn. Residuals become `psi` (held by the caller anyway) and `kinetic` at `(nk, npwx)`.
Recomputation is one elementwise multiply and one reduction over an array the pass already
streams — milliseconds against a stress measured at 30.3 s on `si8-us`.

**The `Re(conj(psi) psi)` form is mandatory** — the `abs`-at-a-forced-zero trap, and this line is
the canonical site and says so in its own comment at `:518-519`. Remat preserves it exactly because
it re-executes the identical code. **A rewrite would not; do not attempt one.**

`_norms` at `:545` builds the same array a second time but is strain- and position-independent and
is dead-coded under both derivatives.

**Effort.** 1 line.

**Measurement.** `memory_analysis()` on the compiled stress of a small spinor cell, looking for a
distinct f64 buffer of shape `(nspin, nk, nbnd, npol*npwx)`. The `@jax.jit` boundary at `:503`
makes XLA's own cross-boundary rematerialisation less likely rather than more.

---

### A12. `PawSpecies.density_ae`/`density_ps` materialise a rank-1 outer product: 553 MB per PAW species that `sizing.py` reports as zero

**Site.** `defumat/paw/onecenter.py:604` and `:606` (with the `+=` at `:616`); declared `:73-80`:

```python
density_ae = np.einsum("lij,ijr->ijlr", coefficients, pfunc[beta_of][:, beta_of])
```

`(nh, nh, nlm, mesh)` float64, two per species, plus `kinetic_ae`/`kinetic_ps` of the same shape
for a meta-GGA (`:524`).

**Arithmetic.** `Ni.rel-pbe-spn-kjpaw_psl.1.0.0.UPF`: `number_of_proj = 10` at
`l = 0,0,1,1,1,1,2,2,2,2` so `nh = 34`; `mesh_size = 1195`; `l_max_rho = 4` so `nlm = 25`;
`PP_AUGMENTATION` carries `nqlc = 5` and `q_with_l = "true"`.

```
density_ae   34 x 34 x 25 x 1195 x 8   = 276.28 MB
the pair                               = 552.57 MB per Ni species
```

`Pt.rel-pbe-n-kjpaw_psl.0.1` (mesh 1277) gives 295.23 MB each, which is exactly the
`f64[25,1277,34,34]` `PERFORMANCE.md:3327` saw XLA constant-fold twice at over 2 s each.

The factors they are built from:

```
coefficients  (25, 34, 34)         =   0.231 MB
pfunc, ptfunc (10, 10, 1195)       =   0.956 MB each
qfuncl        (10, 10, 5, 1195)    =   4.780 MB
                            total  =   6.92 MB      -- a factor of 79.8
```

**Live for the whole run, and a floor rather than a number under a derivative.** `driver.py:1490`
sets `self.paw = build_paw(...)` once in `__init__` and nothing rebuilds it, so both arrays are
device-resident alongside the eigensolver buffer and the dense-grid fields. Under
`jvp(jit(_paw_onecenter))` — every PAW force, stress and response — `PERFORMANCE.md:3330` observes
them as XLA **constants**, so each compiled variant carries its own copy in a cache that never
shrinks.

**Fix: store the factors.** The product form is read straight off the two lines that build it:
`density_ae[i,j,lm,r] = coefficients[lm,i,j] * pfunc[beta_of[i], beta_of[j], r]`, and `density_ps`
the same against `(ptfunc + qfuncl[.,.,floor(sqrt(lm)),.])`. Both consumers factor: `rho_lm` at
`:154` is `einsum("sij,ijlr->slr")`, which becomes `w[s,lm,n,m] = einsum("sij,lij,in,jm->slnm",
becsum, C, P, P)` with `P` the static one-hot `(nh, nbeta)` from `beta_of`, then an einsum over the
`(nbeta, nbeta, mesh)` radial factor; `ddd` at `:179` is the exact transpose.

**Cost: negative.** `nh^2 nlm mesh = 34.5 M` MACs per spin channel falls to
`nbeta^2 nlm mesh = 2.99 M` plus a 1.8 M reindex, a factor `(nh/nbeta)^2 = 11.56`, and the two
multi-second XLA constant folds go away with the constant.

**Proven in-tree by the same package.** `_assemble_qgm`'s docstring at augmentation.py:836-841
refuses the `(nbeta, nbeta) -> (nh, nh)` expansion for exactly this reason and quotes the same 11.56.

**Composes with, and does not duplicate, OPEN.md M2.** M2 removes duplicate species *labels*; this
removes the cost of one label. A texture written as 15 Ni labels is **8.29 GB** today, 0.55 GB with
M2 alone, 103.6 MB with this alone, **6.92 MB with both**.

**Rules.** Every extent (`nh`, `nbeta`, `nlm`, `mesh`) is already an `eqx` static field or fixed by
the dataset, and the one-hot is a constant matrix. The factors are built host-side in numpy exactly
as now and crossed into JAX once with `jnp.asarray` at `:625-626`, so no `np.asarray` moves inside a
differentiated path. `rho_lm` stays linear in `becsum`, which is what makes `ddd` the exact adjoint
the module docstring at `:34-38` relies on. No dtype literal; nothing wavefunction-shaped.

**Effort.** ~40 lines in one file, two contractions. `_kinetic_tensor` (`:523-527`) is passed
`ae[beta_of]` already expanded, so the factored form needs `ae` plus `beta_of` handed in
separately — a caller change at `:643-648`; its `bcoef` term `(nlm, nh, nh)` at 0.23 MB does not
factor and does not need to. It reassociates sums, so acceptance is the converged total to ~1e-12
Ry on `si2-paw-1k.in` and the spinorbit PAW case, not bit-identity.

**Confined.** `grep` across the package and `tests/` shows `density_ae`, `density_ps`,
`kinetic_ae` and `kinetic_ps` are touched nowhere outside `onecenter.py`.

---

### A13. `vkb` is materialised for every k-point and sits outside the `k_batch` dial

**Site.** `defumat/pseudo/projectors.py:69` — `Projectors.vkb` `(nk, npwx, nkb)` complex — built at
`driver.py:1450` and kept as a `Calculation` attribute for the life of the run. Its sources
`ProjectorCore.columns` `(nk, npwx, ncs)` (`:104`) and `kg` `(nk, npwx, 3)` (`:106`) are kept
alongside it at `driver.py:1446`.

**Arithmetic, nbse2** (`nbse2.json`: `nk = 43` irreducible, `nbnd = 24`; `README.md:36`:
`npwx = 9804` and a 167 MB state file; `Nb.pbe-nc-sg15.UPF` and `Se.pbe-nc-sg15.UPF` each
`number_of_proj = 6` at `l = 0,0,1,1,2,2` so `nh = 18`, `nkb = 3 x 18 = 54`, `ncs = 2 x 18 = 36`):

```
vkb        43 x 9804 x 54 x 16      = 364.2 MB
columns    43 x 9804 x 36 x 16      = 242.8 MB
kg         43 x 9804 x 3 x 8        =  10.1 MB
                            resident = 617.1 MB

wavefunctions   43 x 24 x 9804 x 16  = 161.9 MB
Davidson temp (sizing.py's own fit, CPU defaults, nvecx = 96, N_smooth = 30x30x180)
  2.18x96x9804x16 + 4.20x24x9804x16 + 2.00x162000x16 = 53.8 MB
```

`vkb` is **2.2x the wavefunctions and 6.8x the eigensolver's buffer** — the largest single resident
array in that run.

**QE holds `(npwx, nkb)` for one k.** `~/apps/qe-7.4.1/PW/src/c_bands.f90:111` calls `init_us_2`
inside `k_loop`. At nbse2's shapes that is 8.5 MB against 364 MB — a factor of `nk = 43`.

**The dial does not reach it.** `batching.py`'s docstring names "every k-point's Davidson subspace,
and every k-point's band-by-band real-space field" as the cost of `k=None` and never mentions
`vkb`; `sum_k`/`map_k` chunk only what is passed through them, and `hamiltonian/operator.py:210,
:221, :285` read `self.projectors.vkb[ik]` inside the jitted solve.

**Fix.** Build `vkb` for the chunk inside `map_k`'s body. `ProjectorCore.at_positions` already does
`init_us_2`'s job — `_apply_phases` (`:265-275`) is one `jnp.take` of `columns`, one `jnp.take` of
`exp(-i(k+G).tau)`, one mask — so the change is to pass the core plus the chunk's slice into the
Hamiltonian and form `(npwx, nkb)` at the top of each chunk. The build happens once per `map_k`
body invocation and is reused by every step of the `while_loop` inside it — QE's `k_loop` cadence,
not per `h_psi`.

**Corrected saving.** `columns` and `kg` are themselves `(nk, npwx, .)` and are the source the
per-chunk build reads, so they stay resident:

```
617.1 MB  ->  242.8 + 10.1 + one chunk's 8.5 + a (npwx, nat) phases transient 0.5  =  261.9 MB
```

**355 MB saved, 58% of projector storage — not the 96% a first reading suggests.** Reaching 23 MB
requires rebuilding `columns` per k too, which is `PLAN.md:10939`'s per-species scan, recomputes the
radial form factors, and is a different and more expensive fix.

**It is a many-k finding.** On the 45-atom NiBr2 slab `nk` is small (P74's XLA dump shows no k axis
on the band blocks) and at `nk = k_batch` the per-chunk array *is* the whole array, so this saves
essentially nothing there. nbse2 is the cell that carries it.

**Cost in time.** `nat x npwx` complex exponentials and two gathers per k per `map_k` body — on
nbse2, `3 x 9804 ~ 29k` exponentials per k per SCF iteration, against a 57 s iteration (744 s / 13).

**The backward pass is not helped without remat.** Building `vkb` inside a `lax.map` body does not
shrink the reverse tape — reverse mode stores per-iteration residuals, so a force's tape still
holds all `nk` `(npwx, nkb)` slices unless the body is `jax.checkpoint`-ed. That is verbatim
`PLAN.md:10939`'s own note.

**Rules.** The per-chunk array is `(k_batch, npwx, nkb)`, fixed at trace time. `_apply_phases` is
already the differentiated path for forces and is unchanged. No host sync, no traced branch, dtype
still `ProjectorCore.complex_dtype`. **R6 is untouched**: this removes a leading k axis from an
*operator table*, not from a state.

**Effort.** ~60 lines — the largest in (a), and the smallest byte figure of the leading items.

**Already on the books, partly.** `sizing.py:502` already reports the `nk` factor;
`PLAN.md:10939` names the stronger matrix-free form as a deferred phase; OPEN.md M2 asks for the
`ProjectorCore` arrays to get their own sizing line (D5). **What is missing against the memory rule
is the selectable leg**: the trade is stated and measured, and there is no dial. If the refactor is
not taken, the fallback is a declared `DEFUMAT_VKB_PER_K` switch and a `projectors.py` docstring
naming the `nk` factor against QE's `(npwx, nkb)`.

**Measurement.** Sum `jax.live_arrays()` nbytes, or `memory_analysis().temp_size_in_bytes` of the
compiled per-k body, before and after, on `calculations/nbse2-fermi-surface/nbse2.in`.

---

### A14. `VelocityOperator.matrix_elements` holds four full-k blocks to build a matrix `nbnd/ndim` smaller

**Site.** `defumat/response/velocity.py:302`:

```python
jnp.einsum("skmg,skng->skmn", psi.conj(), self.apply(psi, axis))
```

Python evaluates `psi.conj()` before `self.apply(...)`, so `psi`, its conjugate, the `jvp`'s primal
`H|psi>` and its tangent are all `(nspin, nk, nbnd, npwx*npol)` and all live at the instant
`_tangent` (`:255-265`) returns — four blocks, of which `_tangent` keeps one. The output is
`(3, nspin, nk, nbnd, nbnd)`, smaller by `nbnd/ndim`.

**The structural point.** `over_kpoints` (`:387-396`) walks k through `map_k`, but what it returns
is the *stacked* full-k block, so `k_batch` bounds the per-k FFT temporaries and not the block, at
any setting.

**Arithmetic at the one cell where this was measured** (`PERFORMANCE.md:3768-3770`: AlAs, 216 k,
22 bands, `npwx = 869`, `nspin = npol = 1`):

```
one block   216 x 22 x 869 x 16   = 66.1 MB
four live                          = 264 MB
```

against a measured peak resident set of **1.33 GB** for the whole `second_harmonic` call — ~20% of
a measured peak, of which the fix removes three of the four blocks (~200 MB).
`second_matrix_elements`' stored `(3, 3, nspin, nk, nbnd, nbnd)` output is 45 MB there.

*An earlier lens sized this at 3.3-4.4 GB on an 8-atom silicon cell at `nbnd = 200`, `npwx = 1607`.
No such run exists here: the largest band count ever put through `matrix_elements` in the record is
36 (conductivity, `PERFORMANCE.md:3626`) and 60 for `chi_0` on a two-atom cell. The measured cell is
the honest one.*

**Fix.** A second entry point that pushes the `psi.conj() . (.)` einsum into `over_kpoints`' body,
inside the `jvp`. It is exact rather than an approximation because `psi.conj()` does not depend on
`kcart`, so `jvp(einsum(psi*, H(kc)psi)) = einsum(psi*, tangent)` and the contraction is linear in
the tangent. Only a chunk's `(nchunk, nbnd, ndim)` tangent is then live. **`apply()` must keep
returning the full block** — `efield.py:305` consumes it as an array — so this is a second entry
point, not a change to `apply`.

**Direct precedent in this repo.** `PERFORMANCE.md:4337-4352` applied exactly this move to
`workflows/transport.py:_assemble`, because the sum over k is exact term by term.

**Cost.** No per-k work is repeated: `at_kcart`'s docstring (driver.py:2577-2582) says the G sets,
FFT box, stick layout, mask, local potential, augmentation charge and Ewald sum are shared and only
`|k+G|^2` and `vkb(k)` are rebuilt, both per k. What is repeated is the k-independent part of
`moved.hamiltonian(...)` once per chunk.

**This is not the jit proposal.** `PERFORMANCE.md:3842` measured `eqx.filter_jit` on
`matrix_elements` at 1.33 -> 8.96 GB and refused it, correctly. That entry's conclusion
"`k_batch` is the dial" is true of the transient inside `h_psi` and not of the returned block.

**Rules.** Stays inside `map_k`'s `lax.map`/`scan` with k leading (R6), chunk size is the existing
dial, shapes static, nothing becomes `np.asarray`, einsum dtype follows `psi`.

**Effort.** ~20 lines. `second_matrix_elements` (`:356`, forward-over-forward, four blocks per
cartesian pair, six pairs, reached from `photocurrent.py:456`) takes the same treatment.

**Measurement.** `memory_analysis().temp_size_in_bytes` on the contracted body — the same
instrument `PERFORMANCE.md:4370` used for `pair_batch`.

---

### A15. `sum_band` vmaps the spin axis, so both LSDA channels' real-space boxes are in flight at once

**Site.** `defumat/scf/density.py:109`, `:118` and `:180` — `jax.vmap(channel)(psi, weights)`. Under
that vmap every intermediate inside `channel` gains a leading spin axis by construction, so
`g_to_r`'s output is `(nspin, band_batch, n1, n2, n3)` complex and `weight * Re(conj(field)*field)`
is `(nspin, band_batch, n1, n2, n3)` real — one batched FFT producing one buffer, so the two
channels coexist by construction rather than by scheduling luck. For `band_kinetic_density` (`:180`)
the field is `(2, 3, band_batch, n1, n2, n3)`: six boxes per band instead of three.

**Arithmetic, `benchmarks/h40-chain-lsda.in`** (ibrav 6, a = 12 bohr, c = 200 bohr, `ecutwfc = 25`
so `ecutrho = 100` and one grid; `|b1| = 0.5236`, `|b3| = 0.031416`, `Gmax_rho = 10` give
`n1max = 19`, `n3max = 318`, a 40x40x640 = 1,024,000 box; `npwx = 60,800`; `nbnd = 56`; `nk = 1`).
At the accelerator default `band_batch = None`:

```
complex field       2 x 56 x 1,024,000 x 16   = 1.835 GB   (one channel: 0.917)
real accumulator    2 x 56 x 1,024,000 x  8   = 0.917 GB   (one channel: 0.459)
                            density stage      = 2.75 GB   -> 1.38 GB
```

**A peak is a max over stages.** By `sizing.py`'s own fit the Davidson stage at the same dials is
`2.18x224x60800x16 + 4.20x56x60800x16 + 2.00x56x1,024,000x16 = 0.475 + 0.229 + 1.835 = 2.54 GB`.
So the fix takes the run's peak from 2.75 GB to `max(1.38, 2.54) = 2.54 GB` — **0.21 GB, 7.6%**, not
the 1.38 GB the stage figure suggests. At `band_batch = 16` (P74's own NiBr2 setting) the density
stage is 0.79 GB against a Davidson stage of 1.45 GB, so it moves the peak by zero while still
halving the stage. At CPU defaults (`band_batch = 1`) it is 32.8 MB against 16.4 MB.

**Honest headline: a real, undialled factor of two on the density stage for every `nspin = 2` run**,
worth ~8% of the peak at accelerator defaults on h40 and nothing once `band_batch` is below the
crossover. It does not touch NiBr2, which is noncollinear and goes through `spinor_sum_band` with a
leading axis of one.

**Fix.** `jnp.stack([channel(psi[spin], weights[spin]) for spin in range(psi.shape[0])])`. This is
exactly what `becsum` three functions below already does (`density.py:314`), with a comment saying
why: *"One channel at a time rather than a spin axis through the accumulation: QE has no spin axis
here either -- `sum_bec` writes into `becsum(:,:,current_spin)` and its k-list runs over both
channels."* **The inconsistency between the two is the finding.**

**Rules.** `_density_of_bands` is `@partial(jax.jit, static_argnames=("grid", "k_batch"))`
(driver.py:335-337), so `psi.shape[0]` is resolved at trace time — no data-dependent shape, no
Python branch on a traced value; `nspin` is an array rank and static by this project's own
convention. `one_band`'s `Re(conj*field)` form is untouched. k stays the leading independent axis
inside each channel, so R6 holds — the change is above the k axis.

**Effort.** 3 lines. Assert `jnp.array_equal` on the density (spin is a map, no sum crosses the
channel axis), falling back to ~1e-16 relative if the batched and unbatched FFT kernels round
differently.

---

## 3. (b) Wins that require a stated, measured, selectable trade

### B1. An `atom_batch` dial for the PAW one-centre forward set — **done, with A4**

**Site.** `defumat/paw/onecenter.py:128`, as A4. The *forward* live set is 0.35-0.45 GB for the Ni
sublattice of the 45-atom cell, against ~30 MB one atom at a time — about **1.2% of the measured
32.30 GB peak**.

**Fix.** Route the atom axis through the dial that already exists. `batching.py:236`'s `map_axis` is
the generic `lax.map`/`lax.scan` chunker `map_k` and `map_bands` are both built on; add an
`atom_batch` resolved the same way (`DEFUMAT_ATOM_BATCH`, `SHARED_OPTIONS`, a `resolve_*` helper)
and **default it to the present `None`**, so no measured number moves and the dial is there for the
cell that needs it.

> **Landed with A4, and the reason it stopped being a (b) is that the cost turned out to be
> negative.** The dial is `_paw_atom_batch` / `PAW_ATOM_BATCH` / `DEFUMAT_PAW_ATOM_BATCH` in
> `paw/onecenter.py`, on `map_axis` exactly as proposed — but its default is **not** the present
> `None`, because chunking is what makes A4's remat pay and above the crossover it is *faster* as
> well as smaller (per-iteration SCF 1.403 -> 1.281 s on `si10-paw-pbe`). The paragraph below
> predicted the opposite and was right about the mechanism and wrong about which way it points:
> `batch = 1` does turn one dispatch into fifteen, and fifteen dispatches over a peak that has
> halved beat one over a peak that has not. Below the crossover the prediction holds and the default
> keeps the `vmap` there.

**Why this was (b) and not (a): the cost was unknown.** `PERFORMANCE.md:332-338` says "Atoms were
batched from the start (vmap over becsum)", and the measurement that follows — 3.0x to 2.8x against
QE, 0.507 s to 0.477 s per iteration — is the **multipole grouping**, not the atom axis. The 35%
headline at `:323` covers the one-centre batching as a whole and is never decomposed, so what
`atom_batch = 1` costs has never been measured. It could be worse than 35%: on these array sizes
this file's repeated lesson is that the cost is the number of compiled dispatches, and batch = 1
turns one into fifteen.

**Rules.** `map_axis` is `lax.map`/`lax.scan` based, so shapes stay static and the chunked form stays
compiled once and differentiable — the same properties the k and band dials rest on — and its
`batch=1` branch is a plain `lax.map` with no width-one batch dimension. The atom axis is not the k
axis and k does not appear in this call, so R6 is untouched. No dtype introduced. The chunk
reassociates only the sum over atoms in `energy` (`:132`) and leaves `ddd` per-atom, so the move is
small.

**Effort.** ~30 lines, plus the timing. **Add a `sizing.py` line for `PawSpecies` and one for the
quadrature in the same pass** — that half stands on its own regardless of the dial (D4).

**Measurement.** The same compile-only `memory_analysis()` at `nat_t = 1` against `nat_t = 15` as
A4, whose compiled time also gives the missing cost figure.

---

### B2. The `Calculator`'s retention API: a release path, a seed policy, and one contract change

Three things that need a declaration rather than only a drop. The free halves are in A6.

**(i) `Calculator.release()`**, dropping `_seed`, `_relax`, `_strain_response`, `dos_states` and
`pdos_states`, with the class docstring saying what each holds in the shapes A6 gives. Today there
is no way to release any of them short of dropping the Calculator, and the response methods are
called exactly when the seed is dead weight.

**(ii) Clearing `_seed` is not a one-liner, and it carries physics.** `merged.setdefault(
"starting_from", self._seed)` (`:515`) combined with `_same_options`' `set(new) != set(old)` test
(`:1593`) means dropping the key makes every second `get_scf()` a cache miss and reruns the SCF.
Either exclude `starting_from` from the comparison, or put a sentinel token in `_scf_options` in
place of the object. **Never delete the key.** And the seed is not only a speed-up: `CLAUDE.md` P23
has the SCF seeding a magnetization because nothing in the loop breaks spin symmetry on its own, so
on a system with several magnetic minima the seed selects the answer. Clearing it **only once a
converged `_scf` exists** is sound — that `_scf` already carries the texture the seed produced —
and clearing it any earlier is not.

**(iii) `VCRelaxResult.relaxation_scf` is public API.** `pulay_error` (vc_relax.py:157-160) reads
only `.total_energy` from it, so a `keep_relaxation_state=False` default that stores the energy and
drops the arrays is an **API contract change** and must be named as one: a caller wanting the
frozen-basis wavefunctions re-runs one SCF. The Pulay-error contract survives intact, because the
float it needs is then carried explicitly rather than through the array bundle. The cost of today's
behaviour is two wavefunction sets in every returned object, undeclared.

**Plumbing trap.** Options reach `strain_response` through `_defaults_for(...)`, which filters
strictly by named parameter, so a `cache=False` keyword must be popped in `get_strain_response`
before that call rather than passed through.

**Rules.** Entirely host-side object lifetime plus one boolean keyword. No traced path, no shape, no
dtype, no k axis. It keeps the facade rule intact — no physics moves into `calculator.py`, only
reference management — and keeps "Nothing mutates" true, since `with_*` still returns a new
calculator with an empty cache.

**Effort.** ~40 lines, plus docstrings.

---

## 4. (c) One measurement from decidable

### C1. Forward-mode stress. The measurement that looks like it rules forward mode out does not

`stress/autodiff.py:66-78` caches `jax.jit(jax.grad(strained_energy))` — a single reverse pass over
everything `at_strain` rebuilds (`kinetic` at driver.py:2406, the projector core at `:2411`,
`build_augmentation` from scratch at `:2422`), so every radial transform is inside the tape.
`_term_gradients` (`:88`) is `jax.jacfwd`, i.e. `vmap` of `jvp` over **all nine** strain components
at once: one primal plus nine tangent copies. `stress/registry.py` registers only `autodiff` and
`analytic`, and `compute_stress` refuses the latter by name — there is no forward option.

`PERFORMANCE.md`'s P11 table on `si8-us`: SCF 899 MB, stress 11,105 MB, term breakdown 11,049 MB,
and the record concludes *"the breakdown costs nothing over the plain gradient, so the 11 GB is the
reverse pass and not the forward-mode Jacobian."* **That inference does not follow from peak RSS.**
If the two are one process the third column cannot go below the second and says nothing; if they are
separate processes then a nine-tangent `jacfwd` landing on the same 11 GB is exactly what 1 primal +
9 tangents of a ~1 GB forward frontier gives. A consistency check supports the forward reading:
`PERFORMANCE.md` puts `_qrad_kernel`'s intermediate at 300 MB with several temporaries per L, so
~1 GB x 10 plus the 899 MB SCF is the measured 11.0.

**Claim under test.** A sequential loop of `jax.jvp` calls holds one primal and one tangent —
~2 GB, on a ~1 GB forward frontier plus the 899 MB SCF — against ~11 GB of reverse tape. Time, from
the same rows: the 173 s breakdown is nine tangent passes at ~19 s each against 30.3 s reverse, so
nine sequential passes is ~173 s. **Roughly 5-6x the time for roughly 5x less memory**, and on a
cell carrying A1's 65 GB tape the gap is the whole calculation.

**The measurement.** One process running a sequential-`jvp` stress on `si8-us` under
`/usr/bin/time -v`, against one running the existing reverse stress. **Separate processes, nothing
else on the machine** — peak RSS is a within-process high-water mark, which is exactly why the
existing table's columns cannot be compared across modes.

**Three cautions for whoever writes it.** (1) Use **nine** probes, not six: `compute_stress` reports
`rotational_residue = |raw - raw.T|/2` (stress/__init__.py:140-144), and for a spin-orbit run the
energy is not invariant under an antisymmetric strain, so six symmetric probes silently destroy that
diagnostic. (2) Cache the compiled `jvp` the way `_energy_gradient` is cached and key it on the
calculation, or backlog item 6's 0.6 s-per-ionic-step retracing lands on the new path multiplied by
nine. (3) It must be **selectable and not the default**, with the two modes asserted equal to
round-off, because every validated stress number in this project was measured in reverse mode.

Forward mode is `jax.jvp` of the identical function, so static shapes, the precision policy,
differentiability, GPU-readiness and R6 are untouched, and the frozen-sphere Pulay approximation and
the wedge symmetrisation are unchanged because the differentiated function is unchanged.

This is the same target as `PERFORMANCE.md` backlog item 10 by a much cheaper lever — that item
proposes a `custom_jvp` carrying `dF/d|G|` in closed form for every radial transform, which is real
physics coding; this is twenty lines.

---

### C2. The eigts factorisation of the resident structure factor

`AugmentationCharge.phases` `(nat, ngm)` is **2.546 GB resident** on the 45-atom cell.
QE's `PW/src/struct_fact.f90` stores `strf(ngm, ntyp)` — 2 x 3.54e6 x 16 = **113 MB** here — plus
`eigts1/2/3(-nr_i:nr_i, nat)`, three 1-D tables totalling ~**0.85 MB** at this cell's grid, from
which the per-atom factor of any G is a product of three lookups. Every extent is static and the
product of three exponentials of a linear function of the crystal coordinates is exactly
differentiable, with no `abs`.

**The tension that a measurement settles.** Transposing a product of three gathers needs the three
gathered `(nat, ngm)` values, so reverse mode would tape *three* arrays where `exp` tapes one; and
forming the product inside a G-chunk scan without remat stacks residuals by A1's verified mechanism.
So the resident win and the tape pull against each other.

**The measurement.** `memory_analysis()` on a compiled force before and after, with A5's
`jax.checkpoint` in place in **both**. If the three gathers rematerialise cleanly it is a 2.5 GB
resident win; if not it is a 2.5 GB resident win paid for with a larger tape.

**The regime it actually bites in.** On the 45-atom cell this is an order below A1, so it becomes
the dominant position-dependent setup array on a large **norm-conserving** cell, where there is no
augmentation charge at all. Nothing anywhere declares this as a deviation from QE's layout: the only
mention of `eigts` in the whole tree is a one-line docstring at `forces/analytic.py:176`.

---

### C3. The PAW one-centre temporary count

A4's 3-5 GB rests on "~50 grid-sized temporaries in the PBE-spin kernel", which nothing counts.
**Measurement, allocation-free:** `jax.jit(jax.grad(lambda b: onecenter_species(paw, b)[1]))
.lower(*ShapeDtypeStructs).compile().memory_analysis()` at the NiBr2 Ni shapes with `nat_t = 1`
against `nat_t = 15`. The same run gives B1's forward figure and, from the compiled time, the
`atom_batch = 1` cost `PERFORMANCE.md` never separated.

### C4. Whether XLA's CSE merges the two `combine_species` calls

`at_positions` calls it at driver.py:2203 (vloc) and `:2209` (rho_core), both against the dense set.
One residual (2.55 GB) or two (5.1 GB) — A5's range. `memory_analysis()` on the compiled force.

### C5. Whether `jnp.take(columns, column_of, -1)` materialises a second `vkb` in the backward pass

`pseudo/projectors.py:273`. If XLA does not fuse the gather into the backward pass it is a second
`(nk, npwx, nkb)` copy: 876 MB on bi20-soc, 2.6 GB on NiBr2. `memory_analysis()` on the compiled
force answers it.

### C6. The spiral's XLA executable accumulation

`workflows/spiral.py:301` (`run_spiral_scan`) and `:528` (`relax_spiral_q`). `at_spiral_q` rebuilds
the spheres at `k +- q/2`, so `npwx` changes at every step — `PERFORMANCE.md:1146-1148` states this
outright — and every kernel whose signature carries `npwx` is compiled afresh per q and kept for the
life of the process. **Verified: `jax.clear_caches()` appears in ~20 test files and in no module of
`defumat/` (0 hits).**

The only ladder on record (`PERFORMANCE.md:3236-3241`: ten distinct-shape cells in one process at
0.55, 0.81, 1.27, 1.47, 1.60, 1.77, 1.82, 2.14, 2.47, **3.67 GB** monotonic, "none of it the
states", against 0.52...1.45 GB flat with `jax.clear_caches()` between them) gives
`(3.67 - 1.45)/9 = 0.247 GB per distinct shape`, so a 13-point `E(q)` scan would carry ~**2.9 GB**
of compiled code beside the SCF. **Treat that as an upper bound of unknown tightness**: it is
measured on ten full SCF-plus-derivative stacks at ten different ranks, where a q step re-lowers
only the kernels whose signature carries `npwx`.

**The measurement.** A 5-point scan on a small chain with `resource.getrusage` peak RSS read after
each q, with and without the clear — five SCF runs, the same experiment P28b already ran on a test
file.

**If it is taken, it must be behind a keyword and off by default**, because `jax.clear_caches()` is
**process-global** and drops the caller's unrelated compiled kernels too: a notebook or driver script
holding other jitted work pays to re-lower it. The precedent one level down is `at_spiral_q`'s own
explicit gradient-cache pop at driver.py:2662-2663. `PERFORMANCE.md:1146-1152` already declares
per-step recompilation as a deliberate trade, but in **time**, never in resident bytes. The
accompanying `result = None` before `:302` is free and does not collide with `keep_results`.

---

## 5. (d) Model gaps in `sizing.py`

Verified by grep: `paw` 0 hits, `spiral` 0, `mixing_ndim` 0, `mixer` 0, `basis_kpoints` 0.

### D1. Both dials are resolved on the machine running the estimate, not the machine the run is for

`sizing.py:480-483` resolves a `"default"` `band_batch` through `resolve_band_batch` ->
`_band_default()` -> `_platform_default()` -> `1 if _backend() == "cpu" else None`
(batching.py:177); `k_batch` takes the same route from `calculator.py:381-383`. The affected term is

```
_FFT_COEFFICIENT * bands_in_flight * npol * prod(smooth_grid) * zc          # sizing.py:637
bands_in_flight = nbnd if band_batch is None else min(band_batch, nbnd)     # sizing.py:633
```

and it is inside `eigensolver_buffer`, which `peak_bytes` (`:260-275`) is
`resident + max(eigensolver_buffer, setup_transient)` of — so it **is** the number that decides.

**157-atom FePc/SnTe slab** (P67/P68: `nbnd = 1020`, `npol = 1`, gamma storage, `nk = 1`, one grid).
From `sizing.py`'s own calibration (22 GiB at `band_batch = 64`),
`N_smooth = 22 x 2^30 / (2.00 x 64 x 16) = 1.153e7`, i.e. **0.3438 GiB per band in flight**.

| resolved on | `band_batch` | the term | reported peak |
|---|---|---|---|
| this CPU workstation | 1 | 0.34 GiB | 72.4 GiB (david 2) / 95.7 GiB (david 4) — P68's figures |
| an H200 | `None` -> 1020 | **350.6 GiB** | ~420 GiB, against 133.9 GiB of card |

The estimate is not wrong about either machine; it silently answers for the wrong one. Read the
350.6 GiB as "hundreds of GiB": the 1020-band point extrapolates a fit measured at 4-128, and
`sizing.py:60-66` records the fit is non-monotonic in `band_batch` at david = 2.

`report()` (`:315-318`) does print `bands in flight = N`, so the assumption is visible as a number.
What is missing is its **provenance** and a CLI override.

**Fix, zero compute.** (1) Thread a `platform`/`target` argument to both resolvers and have
`report()` say `bands in flight = 1 [cpu default]`. (2) `cli.py:341-349` gives the `size` subparser
`--pseudo-dir`, `--k-batch` and `--david` and **no `--band-batch`**, so the only override for the
largest assumption in the estimate is `DEFUMAT_BAND_BATCH` — add `--band-batch` and `--platform`.
(3) `cli.py:346`'s help for `--k-batch` says "(default: all)", which is wrong on every CPU, where
the resolved default is 1.

*Do not reach for "the estimate touches the device": `Cell.at` is `precision.as_real(at)` =
`jnp.asarray` (config.py:36-37), so constructing the `System` that `estimate_size` takes has already
initialised the backend before `_backend()` is reached. `sizing.py:11`'s "Nothing here touches the
device" is already false for a reason removing the `_backend()` call would not fix.*

### D2. A spin spiral is sized on the unshifted k-list

`sizing.py:451-455` passes `np.asarray(system.kpoints.coords)` to `_plane_wave_counts` and takes
`npwx = max(npw)`. `Calculation.__init__` sets `self.basis_kpoints = spiral_kpoints(...)`
(driver.py:1402-1405) — the **2nk** points `k + q/2` then `k - q/2` (system/spiral.py:71-85) — and
builds the basis (`:1416`), `kinetic` (`:1420`), `fft_index` (`:1421`), `sticks` (`:1437`) and the
projector core (`:1443-1449`) on that list.

Three consequences. `arrays["projectors vkb (nk,npwx,nkb)"]` (`:502`) is a factor of **exactly two**
low, structurally rather than by estimate. `npwx` is a maximum over the wrong point set and biased
**low**, since a displaced point holds at least as many plane waves as Gamma (`CLAUDE.md`'s own
stencil trap). And `kinetic (2nk, npwx)`, `fft_index (2nk, npwx)`, the stick tables and, for a
meta-GGA, `kplusg (2nk, npwx, 3)` appear in no `arrays` entry at all, spiral or otherwise.

The wavefunction line is **correct**: `spiral_q` requires `noncolin` (builder.py:1717-1722), so
`npol = 2` and `ndim = 2 npwx` already carries the up/down pair on its two spheres, and `nk` is the
physical count. A spiral also forces `nosym` (builder.py:1733-1736), so `nk` is larger than a
symmetric run's and the doubled line is amplified.

Order of magnitude on a constructed 15-atom norm-conserving magnetic monolayer (`ecutwfc = 60`,
20 A vacuum, `Omega ~ 1600 bohr^3` so `npwx = 1600 x 464.8/59.22 = 12,558`; 8x8x1 unreduced so
`nk = 64`; `nkb ~ 270`): `vkb` as built `2 x 64 x 12,600 x 270 x 16 = 6.97 GB`, as reported
**3.48 GB**. The factor of 2 is exact and cell-independent; the 3.48 GB anchor is a constructed cell.

**Fix.** When `system.spiral`, build `basis_kpoints = spiral_kpoints(system.kpoints,
system.spiral_q, system.cell)` — already host-side numpy — pass *that* to `_plane_wave_counts`,
report `npw` per basis k-point, and size the `vkb` line at `len(basis_kpoints)`. The estimate then
takes twice as long on a spiral and still allocates nothing large.

**The test that has never run it.** `tests/unit/test_sizing.py:64-67` parametrises
`["nc", "paw-doublegrid", "shifted", "gamma"]` — none noncollinear, none a spiral — so
`estimate.npwx == built.basis.npwx` (`:79`) and `estimate.npw == tuple(built.basis.planewaves.npw)`
(`:89`) have never run against a calculation whose basis k-list differs from its k-point list, and
the second would fail on length alone. `tests/data/qe/h-chain-spiral.in` is a committed one-atom
cell cheap enough to build.

### D3. The Anderson mixer's history is unmodelled

`mixing.py:110-125`: `_densities` and `_residuals`, each up to `history = mixing_ndim` entries
(appended `:121-122`, trimmed `:123-125`). Each entry is `np.asarray(rho_in).ravel()` (`:118`) of the
packed state assembled at driver.py:420-443 — the dense-grid density with `becsum` and, for DFT+U,
`ns` concatenated on, one **real host** vector (the comment at driver.py:429-431 says so). Depth is
wired at driver.py:4508.

Resident cost `2 x mixing_ndim x nspin_mag x prod(dense_grid) x zr`, plus two more whole copies per
iteration at driver.py:444 (`np.concatenate(flat)` and `np.concatenate(flat_out)`) and a third
short-lived `mixed`.

**45-atom NiBr2** (`nspin_mag = 4`; dense points inferred as `3,536,849 x 6/pi = 6.75e6`, which
cross-checks against P74's smooth 200x240x54 = 2.592e6 at a dual of ~7.5):

```
one packed state   4 x 6.75e6 x 8   =  216 MB
default history    16 x 216 MB      = 3.46 GB
per-iteration concatenations        = ~432 MB
```

On the 157-atom slab (`nspin_mag = 2`, ~4e6 points) it is ~64 MB x 16 = **~1.0 GB**.

The dense-point count is inferred rather than read off an input file, so the 3.46 GB carries about
+-10%; `PERFORMANCE.md:4474`'s "~2 000 000" for a 45-atom NiBr2 cycloid disagrees with it, and at the
low figure the history is 1.3 GB. It scales linearly either way.

**It is host memory**, so it is outside P74's device `peak_bytes_in_use` figures — which is why it is
not part of the 4.5 GB that entry leaves unexplained. On this CPU workstation, where
`run_regression.sh` caps each file at 12 GB and the watchdog samples `memory_info().rss`, it is the
same 39 GB pool as everything else, and `np.asarray(rho)` at driver.py:420 forces the device-to-host
copy every iteration on any backend.

**It is a model omission and not an undeclared trade.** The depth is selectable (`mixing_ndim`, and
P78 made it actually reach the mixer) and is already measured for convergence at
`PERFORMANCE.md:4493`. Nothing anywhere puts a memory figure beside it.

**It is also not a deviation from QE.** `mix_rho.f90:217-225` allocates `df(n_iter)`/`dv(n_iter)` as
`mix_type` and `davcio_mix_type` goes through `open_buffer`, which is memory at the default
`io_level`; QE's `mix_type` is `ngm0` complex per channel — within 5% of the same bytes as this
code's real-space packing. (Read on `~/apps/qe-7.4.1`; the vendored 7.5 tree is absent from this
checkout.)

**Fix.** One `arrays` line — `2 * mixing_ndim * nspin_mag * prod(dense_grid) * zr` — with
`mixing_ndim` threaded into `estimate_size` the way `davidson_basis` already is (`:377-391`) and
filled by `Calculator.estimate` from `self.defaults` exactly as calculator.py:372-373 does for
`david` and `nbnd`, since `mixing_ndim` is a `pw.x` input variable and is in the file. Zero compute.

### D4. No PAW term at all

`grep -ic paw defumat/sizing.py` returns **0**. Two arrays a PAW run makes that the model reports as
zero bytes: `PawSpecies.density_ae`/`density_ps` at **552.6 MB per Ni species** (A12), and the
one-centre quadrature at **0.35-0.45 GB** per species sublattice (B1). Both are resident for the
life of the `Calculation`.

### D5. `ProjectorCore.columns` and `kg` are not in `arrays`

`sizing.py:502` counts only `projectors vkb (nk,npwx,nkb)`. `columns` `(nk, npwx, ncs)` is 242.8 MB
at nbse2's shapes and 1.42 GB on a NiBr2 slab written one species per magnetic site; `kg` is 10.1 MB
at nbse2. A **candidate** for part of the 4.5 GB `PERFORMANCE.md` P74 leaves unaccounted (32.30 GB
measured against 27.80 estimated) — a candidate, not an identification. OPEN.md M2 already asks for
this line.

### D6. There is no response estimator

`estimate_size` models the SCF only, so none of A7, A8, A10, A14 or the phonon/strain/
electrostriction working sets is budgetable before a run starts, and `SizeEstimate` cannot warn about
them. A `response=` mode taking `nw`, `nm`, `nbnd` and the perturbation count would have caught A8,
A10 and A7 by arithmetic alone — the same argument `PERFORMANCE.md:4320` makes about setup
allocations missing from the slab estimate (34.78 GB reported against 117.55 measured).

### D7. `Davidson becp+becq (nvecx,nkb)` is a factor of `npol` low on a spinor run

`sizing.py:515` uses `nkb` where the spinor Hamiltonian folds `npol` into that width —
`operator.py:161`'s `s_projections` returns `(nvec, m)` "whatever the spin structure is". 24 MB at
NiBr2's shapes; a model nit, listed for completeness.

### D8. The model counts one of each term where a relaxation holds two or four

`wavefunctions`, `vkb` and `phases` are each counted once (`:501-502`, `:590`); `run_relax` holds two
of each (A2) and `vc_relax` four Calculations (A3). Plausibly part of P74's residual 1.16x.

### D9. The `AUG_MAX_BYTES` gate measures half the stored route's working set

A9, and `:753` hardcodes `16` for the complex byte width, so under `precision = 'single'` it fires at
twice the true size.

---

## 6. Record corrections

This project's rule is that a claim is a number rather than an adjective; a stale claim is therefore
a defect, and these six are stale.

1. **`PERFORMANCE.md:4581` states the `spinchi0` scan carry as done.** The code is `lax.map` +
   `jnp.sum` (spinchi0.py:472-473); `grep -n "scan\|lax\."` on that file returns only the two
   `lax.map` hits, and `git log` shows one commit ever touched it, whose message never mentions a
   scan. The entry's own arithmetic is the tell: `1.2 GB / 40 MB` is exactly
   `nbnd=30 x nw=8 x 5.04 MB`, computable only from the stacking form. Correct it in the commit that
   lands A8.
2. **P11's conclusion that "the 11 GB is the reverse pass and not the forward-mode Jacobian"** does
   not follow from peak RSS, which is a within-process high-water mark (C1).
3. **`velocity.py:51-56` states the module's memory model as "the peak is one extra `vkb`"** and calls
   `vkb` "the largest k-indexed array a calculation has after the wavefunctions themselves". It never
   mentions the primal+tangent band blocks at `:302`, which exceed `vkb` whenever `nbnd > nkb` — every
   consumer in that stack (A14).
4. **OPEN.md's "judged clean" list says of `batching.py` "The only vmaps in the hot path are over the
   spin axis, already measured free at width one."** That is a statement about `nspin = 1` and says
   nothing about `nspin = 2`, where the spin vmap is an undialled factor of two on the density stage
   (A15).
5. **`CLAUDE.md`'s P46 line** — "the backward pass of an ultrasoft or PAW derivative carries the
   augmentation table `Q_ij(G)`, `nh^2 x ngm` per atom ... which is why a bismuthene spinor force does
   not run here at all" — predates P73 and describes the **stored** route. P73 fixed the forward path
   only, so the blocker is still there while looking closed (A1).
6. **`response/strain.py`'s module docstring says ultrasoft and PAW are refused.** `overlap_derivatives`
   (`:338`) is implemented for `is_ultrasoft`, and `require_a_sternheimer_regime` refuses only
   *noncollinear* ultrasoft, so a collinear US/PAW strain response does run and does allocate `ort`
   (A6(i)).

One near-miss worth recording rather than correcting: `augmentation.py:296`'s null — "`jax.checkpoint`
here was tried and measured to be worth nothing" — is about `_qrad_kernel` in the **stored** route. It
is cited correctly, but it reads as a general verdict on remat in that file and is not one.

---

## 7. Dtype-policy breaches found in passing

`CLAUDE.md` and `config.py` both say real and complex dtypes come from the policy object, "never from
literals like `jnp.complex128` or `1.0j`". Correctness claims are only made in float64, so these are
**performance-mode** bugs rather than physics ones — but each doubles a large array in the mode that
exists to halve it.

1. **`pseudo/projectors.py:200`**: `jnp.asarray((-1j) ** np.asarray(l_of))` is a strongly-typed numpy
   complex128, so `ProjectorCore.columns` is complex128 whatever `config.Precision` says. At
   `precision = 'single'` the second-largest resident array of a many-k run (242.8 MB at nbse2, A13)
   stays double-width while `vkb` is cast down around it.
2. **`paw/onecenter.py`**: `build_paw` takes a `cell` argument (`:532`) and never uses it, and
   `_build_species` crosses its arrays into JAX with a bare `jnp.asarray` rather than through
   `cell.precision.real`. Every PAW array — including A12's 552.6 MB pair — is float64 whatever the
   policy says.
3. **`pseudo/augmentation.py:753`**: the `AUG_MAX_BYTES` gate hardcodes `16` for the complex byte
   width, so in single precision it switches routes at twice the true size.

**Checked and *not* a breach**, so that nobody re-reports it: the bare `1j` literals at
`density.py:148/224`, `basis/gradients.py:42/60`, `noncollinear.py:325-326` and `davidson.py:604` are
weakly typed in JAX — `float32 * 1j` gives complex64, not complex128.

---

## 8. Checked and found already done, already selectable, or already recorded

Not re-proposed. Re-reporting a solved item is worse than reporting nothing.

**Done and measured:** P73 (the tabulated `Q_ij(G)` route, `DEFUMAT_AUG_MAX_BYTES`, `_aug_chunk`, the
`_dataset_key` dedup: 76.5 GB -> 8.4 MB) and P74 (the band dial reaching `NoncollinearHamiltonian._local`
and `spinor_band_density`, plus the latent response sites: 78.51 -> 32.30 GB). P77's `LocalRegions`
packing (~700 MB -> ~24 MB). P78's `mixing_ndim` actually reaching the mixer.

**Already a dial:** `k_batch` and `band_batch` (`DEFUMAT_K_BATCH`, `DEFUMAT_BAND_BATCH`, per-platform
defaults, lazy env reads, explicit > env > platform precedence, `nk == 1` short-circuiting before the
chunk size is looked at), `DEFUMAT_AUG_MAX_BYTES`, `DEFUMAT_AUG_CHUNK`, `pair_batch` in
`tddft/chi0.py`, the phonon's `atoms=`/`on_row=` streaming, `run_spiral_scan(keep_results=False)`,
`checkpoint_every`.

**Already lean:** the Davidson subspace (`S|psi>` not stored, the expansion block rebuilt rather than
carried, the robustness retry a host branch); `GVectors` storing only `miller` with everything else a
recomputed property; `_assemble_qgm`'s per-L accumulation and its docstring's refusal to expand
`(nbeta, nbeta)` to `(nh, nh)`; `_interpolate_qrad`'s per-stencil-point sum; `formfactors.py`'s
`CHUNK = 4096`; `_tabulated_integrals`' `(nat, nh, nh)` carry, independent of the chunk;
`spinorbit.py`'s `(nh, nh, 2, 2)` tables at 74 kB; `paw/angular.py`'s `(nx, nlm_table)` tables;
`at_positions`' `copy.copy` sharing; `NSCFResult` carrying no wavefunctions, with
`fixed_density_bands` beside `fixed_density_states` for the energies-only case; `RelaxStep`/
`VCRelaxStep` carrying no state; every `lru_cache(maxsize=None)` in the package, each checked
signature by signature and each keyed on a small scalar (`harmonic_products(lmax)`, `_angular(l)`,
`get_functional(name)`, `_orbital_matrices(l, k)`, `_spin_matrices(p)`,
`moment_matrices(l, real_harmonics)`).

**Already open elsewhere:** OPEN.md M2 (species-label dedup, 2.18 GB — A12 composes with it), OPEN.md
S3 (the `(nat, ngm)` structure factor as a resident line), OPEN.md S4 and backlog item 10
(`_qrad_kernel`'s `(ngm, kkbeta)`, from both ends), OPEN.md D1/D3 (the spectrum and the transport tip
amplitudes), `PLAN.md:10939` (matrix-free projectors as a deferred phase, with its own warning that
matrix-free forward is not matrix-free backward without remat).

**Examined and judged not worth a finding**, recorded so the reasoning is not repeated:

- The Sternheimer CG's docstring (sternheimer.py:118-125) says "four band-blocks"; counting the body
  (`:559-587`) there are six live — `dpsi`, `gradient`, `hold`, `preconditioned`, `direction`,
  `applied` — against QE's five in `cgsolve_all`. A docstring that undercounts by two, not a saving:
  preconditioned CG needs x, r, z, p, Ap, and `hold` is the previous direction QE also keeps.
- `phononq.py:953` keeps the full-`nbnd` `psi_kq` alive where `TwoSphereSolver` slices it to the
  occupied count at `:171` — ~20% of one block, too small beside the `3 nat` pair the same routine
  holds.
- `scf/checkpoint.py:161-171` builds host copies of every array including the wavefunctions and holds
  them all while `np.savez` runs — a second, host-side ~2.19 GB/k. But `_write_checkpoint` runs at the
  *bottom* of an iteration, against the ~9.37 GB live floor rather than the 31.97 GB `diagonalize`
  peak, so it lands around 11.4 GB and never becomes the peak on CPU. It matters on a **GPU**, where
  host and device are different pools and the write is untimed (`PLAN.md:12503-12506`); streaming the
  entries one at a time (`zipfile` + `np.lib.format.write_array`, dropping each host copy after
  writing) would remove it. `checkpoint_every` is a memory knob as well as a time one.
- `Hamiltonian` carries both `potential` `(nspin_mag, n1, n2, n3)` and `potential_wave`, the same field
  transposed (driver.py:3341-3389). Once the stick path is in use, `potential` is read only by
  `diagonal()` for `jnp.mean(self.potential)` — a scalar. 83 MB per copy on NiBr2: too small for a
  finding, but the mean could be precomputed and the untransposed copy dropped.
- `davidson_eigensolver_all`'s retry path holds `fast`, `robust` and the `jnp.where` output — three
  `(nk, nbnd, ndim)` blocks plus a second solve's temp buffer. The docstring states the full-k-set
  retry deliberately ("the shapes are static, so it must"); the memory of it is not stated, but the
  path is rare.
- `response/electrostriction.py` runs six Voigt strains with no `strains=` subset and no `on_row`
  streaming — 21 wavefunction-shaped blocks, 3.2 GB on the 16-atom yardstick, and
  `PERFORMANCE.md:1731` states the trade without making it selectable. Left out because the count is
  **six whatever the cell contains**: it does not grow with the system the way the phonon's `3 nat`
  does, so the gap closes rather than widening. The honest next candidate if one is wanted.
- `hubbard/occupations.py:70` and `:85` declare `k_batch=1` as a signature default rather than
  `"default"`, but `Calculation.occupation_matrix` (driver.py:1908-1913) passes `self.k_batch`, so
  every real call follows the dial and the literal is a fallback for a direct caller.
  `topology/wilson.py:161` and `topology/polarization.py:97` pass `k_batch=1` for a single overlap
  pair, which is correct.
- The starting subspace is `natomwfc` wide, not `nbnd` — 510 against 403 on the NiBr2 cell — and
  `natomwfc` appears nowhere in `sizing.py`. But `PERFORMANCE.md:4589` puts the seed stage at 9.76 GB
  against the Davidson's 31.97, so it does not set the peak there. It would bite only where
  `natomwfc/nbnd` is large *and* `diago_david_ndim` is small — a future `SizeEstimate` line rather than
  a finding.
- **`sum_k`'s `lax.scan` stacks residuals over chunks, so the k dial is inert in every reverse-mode
  derivative.** `forces/spiral.py:265-301` already states this and carries the cure (a Python loop of
  per-chunk `value_and_grad` with zero-weight padding to keep one shape), and
  `tools/gpu/force_memory.py` records the consequence for the force ("this buffer does not have a
  knob"). Not raised as a finding because on every cell where the peak actually bites here — the NiBr2
  slab, bi20-soc, si8-us — `nk = 1`, so it changes nothing. On a many-k PAW cell it would, and the
  spiral's pattern is the fix that already exists. It is the same mechanism as A1, A4 and A5, named
  where it costs.
