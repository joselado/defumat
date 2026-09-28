# What is left to do about GPU memory, sized

A to-do list, written 2026-09-28 on branch `gpu-memory-modes`, right after `memory_mode`
landed (`batching.py`, `scf/streaming.py`; `PERFORMANCE.md` "Two memory modes"). It says
what still costs device memory, what each item scales with, what the fix is, and how to
measure the win on the GTX 1060 in this workstation. `MEMORY-AUDIT.md` is the older,
broader audit; where an item here is already recorded there, its id is given, and nothing
already fixed is listed.

**Where it came from.** A read-only survey: six agents, one per subsystem (SCF core,
derivatives, response, post-SCF workflows, setup and resident arrays, GPU runtime), then
two skeptical verifiers that opened every cited line, checked every size, marked
duplicates and dropped what did not survive. 45 candidates went in; 39 survived and
merge into the 26 items below. Nothing was executed during the survey.

**How to read the numbers.** *Measured* means taken on the GTX 1060 (jax 0.11.1) or with
`memory_analysis()`, and says so. Everything else is **arithmetic on shapes read from the
code**, on a named cell. The standing cells are: eight-atom Si at 20 Ry with `nosym` (the
memory-mode scan), `bn-ldau-noncol.in` (the stress that dies), `pt-soc-paw-nosym.in` and
`bismuthene-soc-small.in` (peaks set by k-independent objects), nbse2 (`nk = 43`,
`npwx = 9804`, `nbnd = 24`, many k) and the 45-atom NiBr2 slab (`nk = 6`, `nbnd = 403`,
`npol = 2`, `ndim = 312692`, smooth box 200x240x54, `natomwfc = 510`).

**Rule for anyone picking one up**: `CLAUDE.md`'s *an identity that closes is not
evidence; an A/B is*. Several fixes below are "remat this" or "chunk that", and
`MEMORY-AUDIT.md` A4 and A11 are two prescribed fixes that measurement refused. Measure
the peak before and after, one run per fresh process (`peak_bytes_in_use` has no reset),
with the compile cache warm (a miss costs *more* device memory on this card).

## Fixed while this list was written

Three survey findings were small enough to fix in the same branch, and are not repeated
below:

* **A streamed resume kept the whole loaded wavefunction set alive for the run**
  (`driver.py`, the release of the source span sat in the non-streamed branch only; it
  reopened `MEMORY-AUDIT.md` A18 on the stream path). Fixed in `ae99566`, with a test that
  holds a weakref to the loaded *array* and fails on the unfixed driver.
* **`at_kpoints` and `at_spiral_q(rebuild_basis=True)` dropped `projectors='rebuild'`**, so
  every band path, derived mesh, spiral scan step and `relax_spiral_q` step stored the
  whole-k `vkb` in memory mode. Both host-side movers now keep the dial.
* **The DFT+U overlap read `projectors.vkb[ik]`**, which on a lazy set builds the whole-k
  array to slice one row, once per k-point (`O(nk^2)` work, a whole-k transient). It reads
  `at_k(ik)` now; the projectors are bit-identical (`test_projector_storage.py`).

## Done since

* **Item 21** (the relaxation loops' retention): `run_vc_relax`, `run_spiral_scan` and
  `relax_spiral_q` drop the previous step's result and `Calculation` (`MEMORY-AUDIT.md`
  A2, extended). No number moves.
* **Item 7, first half**: `local_perturbation` reads `projectors.at_k(ik)` inside its
  guard; a norm-conserving perturbation no longer builds a whole-k `vkb` it never reads.
  The dozen other response sites are still open.
* **Items 13 and 14(b)**: the four radial transforms walk their chunks in a rematted scan
  (`formfactors._scan_rows`). BN's compiled stress temporary on the card **5.574 -> 1.609
  GiB**, and on the CPU 5.74 / 8.52 / 17.81 -> 1.88 / 1.42 / 1.93 GiB at 1 / 9 / 36
  k-points, so the linear growth in nk is gone as well. BN's SCF and stress now run on the
  GTX 1060 at 2052.9 MB peak, the stress within 2.1e-7 Ry/bohr^3 of `pw.x`. Cost: 7.6 per
  cent on `si2-us-1k`'s warm stress. `PERFORMANCE.md`, "The stress tape".

## Suggested order

Cheap and certain first, then the two that decide whether the large cells run in the
default mode:

1. ~~**Retention in the relaxation loops** (item 21)~~ -- done.
2. ~~**One misplaced line in the Sternheimer local perturbation** (item 7, first half)~~ --
   done.
3. ~~**Remat the radial transforms on the derivative path** (item 13)~~ -- done; it was the
   cause of the BN stress death (A/B above).
4. **A budget for the band dial in memory mode** (item 9) -- what stands between the NiBr2
   slab and the default mode.
5. **Stream the NSCF / band-structure solve** (item 1) -- the largest k-mesh term left
   after the SCF.
6. **Factor the PAW one-centre tensors** (item 18) -- 885 MB of Pt's 1.3 GB peak.

Then the rest by priority.

---

## A. What still grows with the k-mesh

Memory mode bounds the SCF. These are the places outside it -- and the resident
bookkeeping inside it -- that still scale with `nk`.

### 1. Stream the fixed-density (NSCF, bands, DOS) solve -- priority 1, medium

`workflows/nscf.py` `fixed_density_states` calls `calculation.diagonalize(...)` and gets
the stacked `(nspin, nk, nbnd, npol*npwx)` set on the device (`nscf.py:278-281`), in
memory mode too; `batching.py`'s own docstring records that bands and NSCF "hold a full-k
store of their own" and "what they would want instead is to never stack the set at all".
It scales with the mesh or the path length, and a non-finite retry holds it three times.
nbse2 at 24x24: about 2.2 GB. `anisotropy.py:577` calls it where only energies are read.

**Fix**, QE's `c_bands_nscf`: when `wfc_store` resolves to `stream`, walk `k_chunks` and
call `eigensolver(h, nbnd, None, ethr, indices=rows)` -- `davidson_eigensolver_all`
already accepts `psi0=None` with `indices`. Keep eigenvalues on the host; either discard
the chunk's states (bands, DOS, nesting, the unprojected force theorem) or write them into
a host numpy store, the same object a streamed `SCFResult.wavefunctions` is. **Risk**:
round-off. **Measure**: `get_scf()` alone against `get_scf()` + `get_bands()` on a 200 and
an 800-point path; the difference should go from ~0.6 MB per k-point to flat.

### 2. Linear response holds its state whole-k and cannot stream -- priority 1, large

`SternheimerSolver` does `self.psi = jnp.asarray(psi)[:, :, :keep]` (`sternheimer.py:317`)
and stacks every block of `dpsi` on the device; `efield.py:296` and `phonon.py:383/385/
430/440` upload the whole store, and a streamed (numpy) store is re-uploaded per call. It
is `(1 + P) nspin nk nocc npwx npol x 16 B` with `P` = 6 for the field (+3 for US Born
charges), `2 x 3nat` for phonons, 12 for the strain response. On eight-atom Si at 216
k-points that is about 3.7 MB per k-point for the field response against the SCF's 0.19.
Recorded as structural in `PERFORMANCE.md` (P24, and the Phase 5 GPU half); no audit id.

**Fix**: keep `psi`, `bare` and `dpsi` as host arrays when streaming; `solve` walks
`k_chunks` like `stream_diagonalize` (the Hamiltonian is indexed by k already); the
response density and `becsum` become per-chunk `jvp`s of `density_at`, accumulated raw and
finished once -- `stream_densities`' structure -- which is exact because both are linear
in the k-sum; build bare perturbations per chunk; turn the assemblies into sums over
chunks. **Risk**: round-off; never hand a whole host array to a `jit` (XLA computes on
the host, silently). **Measure**: a `tools/gpu/response_memory.py`, one property per
process, Si8 at 27/64/125/216 k: SCF only, + dielectric, + one atom's phonon column.

### 3. Forces and stress put the whole k axis on one tape -- priority 2, large

`energy_at` reads `moved.projectors.vkb` and `state.wavefunctions` whole, and the force
and stress are single `jit(grad)` calls (`forces/autodiff.py`, `stress/autodiff.py`).
Force tape: `nk (npwx nkb + npwx nat + nbnd npwx npol) x 16 B`; the stress adds
`nk npwx (16 ncs + 8 kkbeta)`. `MEMORY-AUDIT.md` §8 records the mechanism (the k dial is
inert under reverse mode) and declined it when `nk = 1` was the case that bit; memory mode
targets many-k runs, which reopens it.

**Fix**: generalise `forces/spiral.py`'s `_split_energy_and_gradient`: a forward walk for
raw `becsum`, `ns` and the smooth density; one `value_and_grad` of the global terms
(local, Hartree, XC with core, augmentation, one-centre, Hubbard, Ewald, dispersion) at
the whole sums; a second walk pulling each chunk's `(e_c, b_c, ns_c)` back with cotangent
`(1, g_b, g_ns)`. The spiral route is the working template and its test is the model.
**Risk**: round-off, if every k-coupled term goes through the global function.

### 4. Post-SCF consumers move a streamed store to the device whole -- priority 2, medium

`Calculation.density` has no stream branch (`driver.py:4098`) and `stm.py:205`,
`sfac.py:197` call it with `result.wavefunctions`; `projwfc/projections.py:341-343`,
`angular_momentum.py:301`, `relax.py:520` (every ultrasoft step), `anisotropy.py:780` do
the same. On the NiBr2 slab the streamed SCF keeps 12.10 GB in host RAM and one k-point's
2.0 GB on the card; any of these puts the 12.10 GB back. **Fix**: one helper that routes a
numpy store through `stream_densities` (which returns `becsum`, `rho`, `tau`, `ns`
finished), used at those sites; per-chunk loops for the projections and `<L>`.

The in-loop diagnostics do the same (priority 3, narrow): the PAW orientation torque
differentiates a whole-k `becsum(spin_turned(psi))` (`driver.py:5294-5300`) and the
orientation stepper turns the whole store on the device. Fix: forward mode per chunk
against a `d(pairing)/d(becsum)` computed once.

### 5. The Chern number and the orbital magnetization hold the whole mesh -- priority 2, medium

`chern_number` diagonalises the whole plane at once (`invariants.py:132`,
`topology.py:271-294`) and `build_plane_wave_states` builds the whole-k `vkb` for `becp`
on US/PAW (`states.py:752-775`); the orbital magnetization takes the whole volume mesh
with the Hamiltonian kept. Bismuthene-soc-small at 12x12: 449 MB of solve output, 374 MB
of occupied copies, 424 MB more; about 5.0 GB at 24x24. Wilson-loop Z2 already streams.
**Fix**: diagonalise column by column (`indices=rows`) on one shared sphere and keep two
columns on the device for the overlaps.

### 6. The per-k basis tables stay whole on the card -- priority 2, large

What memory mode leaves resident per k-point (`driver.py:1919-1955`, `:2440`):
`npwx x (16 ncs [core columns] + 24 [kg] + 8 [kinetic] + index arrays + 1 [mask]
+ 16 npol nwfcU [DFT+U wfcU] + 24 [kplusg, meta-GGA])` bytes. Measured on Si8: 0.19 MB per
k-point (22.5 MB of columns and 8.4 MB of `kg` at 216 k). On nbse2 (`ncs = 36`) it is
6.1 MB per k-point, *more* than the 3.9 MB store that streaming moved off the card; with
DFT+U, `wfcU` alone is 18.5 MB against a 44.3 MB store on `ni10-ldau`, and it enters
every Davidson call whole through `HubbardTerm`. Three pieces, in order of cost:

* **Store the projector columns as real** -- priority 2, small. `_species_columns` is
  `ylm (real) x radial (real) x (-i)^l` (`projectors.py:403-407`, `:482-488`), so each
  column is a real array times a fixed phase, stored complex128. Store float64 plus a
  static per-column phase, applied in `_apply_phases`; same for the atomic orbitals'
  `i^l`. Halves the largest per-k resident array (22.5 -> 11.2 MB on Si8 at 216 k,
  242.8 -> 121.4 MB on nbse2). Round-off only.
* **Build the core in k-chunks at setup** -- priority 3, small. `build_projector_core`
  runs `_angular_part`, the form factors and the radial table over the whole k set with
  every intermediate live (`projectors.py:359-415`); at 216 k on Si8 that setup transient
  (90.8 MB, measured) is the run's peak. Chunk it through `_planewaves_rows` /
  `_kpoints_rows`, which already exist for the streamed start.
* **A chunk-local Hamiltonian** -- large. `rows(rows)` methods on `Hamiltonian`,
  `SpinorHamiltonian`, `Projectors`, `HubbardTerm`, `Sticks`, with the per-k leaves in
  host numpy and `device_put` per chunk like the store. `npw` must stay the global static
  tuple (it sets the Davidson subspace). This is what makes memory mode truly flat in
  `nk`.

### 7. `projectors='rebuild'` does not reach the traced movers or the response -- priority 2

**First half done 2026-09-28** (the misplaced line); the rest is open.

`at_strain` (`driver.py:3050`), `at_kcart` (`:3299`) and `at_spiral_q(rebuild_basis=False)`
(`:3412`) still build a stored set; they run under the stress `grad`, the velocity `jvp`
and `dE/dq`, so check with `memory_analysis()` whether a lazy set on the tape saves
anything before switching (`MEMORY-AUDIT.md` C5). The response stack reads the whole-k
`vkb` at about a dozen sites; **the first one is a single misplaced line**:
`local_perturbation` runs `vkb = calculation.projectors.vkb` unconditionally
(`sternheimer.py:880`), outside `apply`, while its only reads sit under
`if coefficients is not None` -- so on a norm-conserving dataset a whole-k `vkb` is built
on every perturbation of every iteration and never read. On Si8 at 216 k that is 179 MB,
8x the core the dial keeps. Move it inside the guard; then pass `Projectors` and use
`at_k(ik)` in `becsum_of`, `_raw_becsum`, `mixed_becsum` and the velocity operator.

### 8. A streamed resume still lands the source set on the device once -- priority 2, small

The resident half is fixed (above). The transient half is not: `load_state` does
`jnp.asarray` on every array including the wavefunctions (`checkpoint.py:89-90`,
`:293-295`) and `promote_wavefunctions` does `jnp.asarray(psi)`, concatenating with zeros
for a 2 -> 4 promotion (`continuation.py:903`, `:960-968`). So a streamed resume or
continuation puts the whole set (12.10 GB on NiBr2, doubled by a promotion) on the device
at iteration 1. **Fix**: keep them numpy when the run streams (`np.load(mmap_mode='r')`
even), and `np.concatenate` for the promotion; `stream_start` already slices the span per
chunk.

---

## B. Per k-point and per band: the next lever on a large cell

### 9. Memory mode never budgets the band dial -- priority 1, medium

Both presets keep `band_batch = all` on a card (`batching.py:482`), a decision measured on
Si8, where a band loop costs 4.3x and buys nothing. On a large cell it is the largest per-k
term: the eigensolver's FFT line is `2 band_batch npol N_smooth x 16 B`
(`sizing.py:801-806`, a measured coefficient), which on the NiBr2 slab is **66.9 GB at
`all` against 2.65 GB at 16**, so one k-point in memory mode needs about 92.9 GB against
28.7 GB at 16 (the cluster runs set `DEFUMAT_BAND_BATCH=16` by hand). The start is worse:
`natomwfc` vectors, and D10's dump names `c128[510,2,200,240,54]` = 42.32 GB. The `nspin =
2` density stage doubles on top (A15, open). Nothing checks memory mode against the card.

**Fix**: in memory mode on an accelerator, resolve `band_batch` from the card -- the
largest `b` (preferring a divisor of `nbnd` and `natomwfc`) whose estimate times the
measured 1.6x fits -- and size the start with `natomwfc` (D10) and the `nspin = 2`
density (A15's three lines). The value needs a carrier, because `map_bands`' `"default"`
never sees the mode: a static field on the Hamiltonian, or a value resolved in the
`Calculation` and passed to `map_bands`/`sum_bands`. **Risk**: none to numbers
(`map_bands` is bit-identical across chunks); time. **Measure**: `h40-chain-lsda.in`
(one k-point, 56 bands, 40x40x640 box) in memory mode at `DEFUMAT_BAND_BATCH` = all, 28,
14, 8, 1: peak and warm time per point.

### 10. Davidson carries two band blocks it could rebuild -- priority 2, medium

The `while_loop` state carries `evc` and `hevc` (`davidson.py:606-614`), so both are live
through the correction block's `h_psi`, which is the step's FFT peak: `2 k_in_flight
nbnd npol npwx x 16 B` -- 4.03 GB of the NiBr2 slab's 28.7 GB per-k stage at
`band_batch = 16` (14 per cent), 2 x 5.8 GiB on the 157-atom slab. `cegterg` carries the
Ritz coefficients `vc` instead. **Fix**: carry the `(nvecx, nbnd)` coefficients and the
live width, form `evc`/`hevc` at the top of the step at the same width (so the GEMM shapes,
hence the bits, are unchanged), and once more after the loop. **Risk**: must be
re-validated bit for bit on the reference cells.

### 11. Donate the streamed Davidson input -- priority 2, small

`davidson.py:977-999` names the robustness retry's reuse of `psi0` as the only thing
blocking `donate_argnums`. On the streamed path `psi0` is a fresh `device_put` of a host
slice that the host still holds, so a retry can re-fetch it. `MEMORY-AUDIT.md` A16
measured that donating one input saves exactly one buffer (`alias_size_in_bytes`) and
removes an alternating large-block allocation it ties to fragmentation. **Fix**: a
donating variant of `_every_k` used only when the caller passes a way to rebuild `psi0`
(`psi0_again=lambda: _to_device(store[spin, rows])`).

### 12. The response's pair and frequency axes -- priority 2

* **`chi_0`'s pair axis defaults to the band dial**, which is `all` on any card in both
  modes (`tddft/chi0.py:468`), so every pair's box is in flight -- the module's own 26 GB
  case comes back on a GPU (`D1` was closed on a CPU, where the band default is 1). Give the
  pair axis its own finite default and measure 1/8/32 on the card. Small.
* **The TDDFT frequency axis has no dial** (`MEMORY-AUDIT.md` A10, open, and
  `PERFORMANCE.md`'s backlog): the `chi_0` assembly holds `(nw, 2 npairs, nm)` per chunk
  and the Dyson loop iterates every frequency with `nw` copies of `f_xc`. Chunk `w`. Medium.
* **Duplicate stacks and dead blocks in the third-derivative drivers**: `b = jnp.stack(
  internals['bare'])` and `u = ...['dpsi']` copy while the lists are still held
  (`response/nonlinear.py:469-470`), the commutator stack copies even where it aliases
  `bare` (`:506`), and `RamanTensors(field=field)` returns the internals regardless of
  `keep_internals` (`:519`). On `PERFORMANCE.md`'s P25 yardstick: 1.38 GB of duplicates and
  7.4 GB of a dead phonon `bare` at the assembly. Small and bit-identical.

---

## C. Derivative tapes: the stress, and the global part of `dE/dq`

### 13. The radial transforms are taped whole under the strain gradient -- priority 1, small

**Done 2026-09-28** (see "Done since"); the text below is the survey's, kept for the record.

`formfactors._chunked` (`formfactors.py:57-63`) is a Python loop of jitted kernels with no
remat, so every chunk's `(chunk, msh)` residuals stay on the reverse tape. `at_strain`
rebuilds `V_loc` and the core charge against the strained `|G|` inside the gradient
(`driver.py:3063-3078`). On BN both datasets carry a core correction and `ngm = 43903`, so
each float64 `(ngm, msh)` array is about 0.6 GB per transform. **This is the leading
candidate for BN's 5.57 GiB stress request** -- consistent with the arithmetic, not yet
proven by an A/B. `OPEN.md` S4 rematted `_qrad_kernel` only; `PERFORMANCE.md` backlog item
11 is the open record.

**Fix**, step 1 (S4's pattern): replace the loop with a `lax.scan` over padded `q` blocks,
body under `jax.checkpoint`, padding with `q = 0` (both kernels are finite there), the
`4 pi / Omega` prefactor outside the scan. The tape then holds the `q` vector and the
backward pass recomputes one block at a time: about 30 MB instead of about 1.2 GB per
array on BN. S4 was bit-identical on `si2-us-1k`. Step 2 (backlog item 11): a
`custom_jvp` returning `F(|G|)` and `dF/d|G|` from the closed forms already transcribed in
`stress/analytic.py`. **Measure**: compile only -- lower `jit(grad(strained_energy))`
with the state as `ShapeDtypeStruct`s at BN's shapes and read
`memory_analysis().temp_size_in_bytes` before and after; then run BN's stress on the card.

### 14. The projector and atomic-orbital transforms are taped per |k+G| -- priority 1, small

**(b) done 2026-09-28** with item 13, and it removed the growth in nk (measured). (a)
and (c) remain, and (a) is now a saving in time rather than in memory.

Under a strain, `build_projector_core` calls `projector_form_factors` on the flattened
`nk x npwx` `|k+G|` (`projectors.py:382-388`), one `_chunked` call per projector, each
with a `(chunk, kkbeta)` intermediate (`formfactors.py:331-334`). So **the stress tape
grows linearly with nk** -- which contradicts `PERFORMANCE.md:1482-1483` ("independent of
nk"), a sentence to correct. **Fix**: (a) one Bessel transform per `(dataset, l)` rather
than per projector (stack the rows of each `l`, as `_qrad_kernel` does -- 3x fewer on BN);
(b) item 13's rematted scan; (c) optionally QE's `tab_beta` on `dq = 0.01` with the
four-point stencil the augmentation already has -- this moves every nonlocal quantity by
the interpolation error, so it stays opt-in. **Measure**: `memory_analysis()` of the
stress on Si8 at 8/27/64/216 k before and after; report MB per k-point.

### 15. The stored augmentation table is assembled on the strain tape -- priority 2, small

**Measured 2026-09-28, and the prescription below does not yet hold.** After items 13/14
this is the largest thing left on BN's stress tape: its 1.42 GiB (CPU) is headed by
`c128[196,43903]` and a dozen `f64[196,1,1,1,43903]`, `nh^2 x ngm` pair blocks. But with
`DEFUMAT_AUG_MAX_BYTES=0`, which sends the table through the tabulated scan named below,
the same executable is **7.47 GiB**, fifteen whole-`ngm` complex pair tables. Locate what
in the tabulated route keeps the whole table on a strain's tape before routing stored
datasets through it.

`at_strain` calls `build_augmentation` from scratch (`driver.py:3054-3057`); below
`AUG_MAX_BYTES` the stored route runs `_assemble_qgm` over strain-dependent `ylm` and
radial blocks (`augmentation.py:1136-1157`), so the tape holds per-`L` `(nh, nh, ngm)`
blocks and `qgm` itself: on bismuthene-soc-small (`nh = 34`, `ngm = 60543`, `qgm` 1.12 GB,
stored) between 69 and 275 MB per `L` plus `qgm`, arithmetic. **Fix**: under a strain,
route stored datasets through the rematted G-chunk scan the tabulated ones use
(`_tabulated_charge`), with an exact builder (S4's `_qrad_block`) instead of the
interpolation.

### 16. Forward-mode stress, and forward-over-forward for piezo and elastic -- priority 2/3

A `forward` stress method -- nine `jax.jvp` with unit strains, compiled once -- holds the
forward working set plus one tangent instead of the reverse tape, for about 9x the work;
`MEMORY-AUDIT.md` C1 is "one measurement from decidable". Select it in memory mode when
the reverse tape would not fit. The piezoelectric and elastic assemblies are
`jvp`-of-`grad` through `at_strain` and carry the whole reverse tape plus its tangent; a
nested-`jvp` mode removes the reverse pass (`MEMORY-AUDIT.md` C1, `OPEN.md` H7). Both
matter less once items 13-15 land -- measure first.

### 17. Compiled gradients still embed k-sized constants -- priority 2, small

`HOISTED_FIELDS = ('paw', 'augmentation')` (`forces/energy.py:109`); the force and stress
`jit`s close over the `Calculation`, so the projector core, `wfcU` and the `(nsym, ngm)`
symmetry maps become executable constants. Hoisting the first two took one-atom Pt's
stress constants from 1062 MB to 10.1 MB (`PERFORMANCE.md`, "A gigabyte of constants");
the rest is 253 MB of core on nbse2. **Fix**: split the `Calculation`'s array leaves with
`eqx.partition` and pass them as an argument, so no large array can become a constant.
Whether XLA:GPU keeps a second resident copy of a constant is unmeasured and decides the
size of the win.

---

## D. k-independent resident objects

### 18. The PAW one-centre tensors are rank-1 products kept whole -- priority 1, medium

`density_ae`, `density_ps` and, since P117, `density_rel` are `einsum('lij,ijr->ijlr')`
outer products put on the device (`paw/onecenter.py:884-925`). For
`Pt.rel-pbe-n-kjpaw_psl.0.1` (`nh = 34`, `nlm = 25`, `mesh = 1277`) each is 295.2 MB:
**885.7 MB of the ~1.3 GB measured Pt spin-orbit PAW peak**, the same in both modes.
`MEMORY-AUDIT.md` A12 is half done (hoisted out of the executables, not factored);
`density_rel` postdates it. **Fix**, QE's `PAW_rho_lm`: store the factors and contract
`becsum` with the coefficients over `(i, j)` first, then with `pfunc` over `(beta_i,
beta_j)`, inside the chunked one-centre kernel; no `(nh, nh, nlm, mesh)` array is formed.
**Risk**: round-off (reassociation); check the PAW stress residue and the `pw.x`
comparisons for spin-orbit PAW Pt. **Measure**: `pt-soc-paw-nosym.in`, peak in both modes,
expected to fall by about 0.88 GB with the energy identical to 1e-12 Ry.

### 19. `Q_ij(G)` is a real table times a known phase -- priority 2, medium

`_assemble_qgm` sums `(-i)^l x` real Gaunt x real `Y_LM` x real radial terms, and real
Gaunt coefficients vanish unless `l_i + l_j + L` is even, so `Q_ij(G) = (-i)^(l_i+l_j)`
times a real number, stored complex128: 1.12 GB for bismuthene's relativistic dataset, and
the assembly holds about 1.9 GB of transient beside it. **Fix**: store the real table with
a static per-pair parity and fold the phase into `becsum`; the charge becomes two real
contractions. The displaced spiral table has the same structure. **Risk**: the forbidden
entries are ~1e-16 rather than 0 (the Gaunt table comes from a matrix inverse), so the
check is round-off, not equality.

### 20. Post-SCF workflows build a second `Calculation` -- priority 2, medium

`get_bands`/`get_nscf`/`get_dos` pass no calculation, so `nscf.py:196-197` builds another
beside the facade's own, which stays alive; `pdos`, `stm`, `sts`, `transport`, `sfac` (with
`kpoints=None`), `DFTSource._base` and `run_spiral_scan` do the same, and the `Q_ij(G)`
dedup is local to one build, so each copy has its own table (1.12 GB on bismuthene). These
setups also do not see the calculator's `memory_mode` or `projectors`
(`docs/features.tex` says so in the amber box; `OPEN.md` Part XVII item 4 mentions
`at_kpoints` in passing). **Fix**: give the workflows a `calculation=` argument
(`fixed_density_states` has one) and pass `self.calculation` where the k-set is the SCF's,
`self.calculation.at_kpoints(k)` elsewhere -- `at_kpoints` now keeps the projector dial.

### 21. The relaxation loops keep the previous step alive -- priority 2, small

**Done 2026-09-28.**

`vc_relax`'s `previous = current` (`:345`) stays bound through the next iteration's SCF,
force and stress; `run_spiral_scan` (`spiral.py:304-316`) and `relax_spiral_q` (`:591`)
keep the previous result through the next SCF. `MEMORY-AUDIT.md` A2's fix was applied to
`run_relax` only. Under the stress, `at_cell` goes through `at_strain`, so `base`,
`previous` and `current` each hold their own augmentation table and core -- at least one
extra 1.12 GB `Q_ij(G)` on bismuthene. **Fix**: A2's lines -- `result = None` at the top of
each loop body, `del previous` after `_advance`. No numbers move.

### 22. In speed mode the cached ground state stays on the device -- priority 3, small

`Calculator._scf` and a derived calculator's seed keep a device store under every later
`get_*` in speed mode (`calculator.py:767`, `:2243`). Park it to host after the SCF (35 ms
per 102 MB each way on this card) once item 4 makes every consumer accept a host store.
The seed half is `MEMORY-AUDIT.md` A6(iii)/B2, open.

---

## E. Runtime and infrastructure

### 23. The device pool is left at JAX's 75 per cent -- priority 2, small

`XLA_CLIENT_MEM_FRACTION` (and its deprecated `XLA_PYTHON_CLIENT_MEM_FRACTION`; setting
both raises) is untouched by the package; on this card the pool is 4.76 GB of 6. About 0.9
gives roughly +0.95 GB here, and the H200 entry in `PERFORMANCE.md` died 5 GiB over the
default limit. **Fix**: in `defumat/__init__.py`, before the backend initialises, set it
to ~0.9 when neither name is set and the mode is not `speed`. The installed jaxlib also
accepts `cuda_async`, `vmm` and `address` allocators, which are the untried half of A16's
fragmentation question. A capacity lever rather than a reduction.

### 24. `sizing.py` misses the stages that set the peak -- priority 3, an enabler

`peak_bytes` is `resident + max(eigensolver buffer, augmentation Bessel transient)`
(`sizing.py:320-335`): the start (D10), the core build, the `qgm` accumulator, `wfcU`, the
symmetry maps and the index arrays are absent, and there is no `wfc_store` argument, so the
1.6x error on this card is covered by `SPEED_HEADROOM = 0.6`. Adding those lines is what
items 9 and 16 need to choose a band batch or a stress method from the card.

### 25. k-axis sharding is a throughput lever, not a memory one -- priority 3, large

`GPU.md` Phase 4 plans sharding the k axis. Once the SCF streams, that buys `nk / ndev`
time and no per-device memory: every device replicates the k-independent set, which is
what sets the bismuthene and Pt peaks in both modes, and a single k-point's solve is not
divided. QE's memory lever is distributing the plane waves (sticks): shard `npwx` and
`ngm` with `jax.sharding`, using the stick layout `basis/sticks.py` already builds, with
one all-to-all between the z and xy passes and a `psum` for the projections. Strategic, to
record in `GPU.md` rather than to start.

### 26. A float32 tier would halve every device array -- priority 3, large

The arithmetic is exact (`zc` 16 -> 8), but no float32 SCF has ever run, so what blocks one
is unknown (`GPU.md` §4 item 2 and Phase 3; `AUDIT-2026-09-18.md` names a Sternheimer carry
mismatch and dtype-less constructions). Build the tier on this card -- eight-atom Si,
`precision=SINGLE`, memory mode -- keeping the density accumulation and the Davidson
overlaps in float64 and `jax_default_matmul_precision='highest'`, before exposing it.
