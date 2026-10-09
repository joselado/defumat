# Plan: the size estimate against the allocator on a spinor PAW slab (`OPEN.md` Part VII item 1)

Written 2026-10-09 by stream D of the OPEN.md split, branch `open-d`. Not implemented: it goes
to a fable review first.

## What the entry is now

The 45-atom NiBr2 spinor PAW slab (`1 6 1`, `nbnd = 403`, `npwx = 156346`, `david 2`, band
batch 16, `k_batch 1`) was sized at a 46.12 GiB peak and measured at 72.30 GiB (79.4 GB) on
2026-09-14. The closure that put 21.40 GiB of the gap on the eigensolver's finiteness guard
was withdrawn the same day by the A/B it predicted (jobs 20252129 against 20244588,
byte-identical per-iteration peaks with and without the guard). What survives from that cell
is the eigensolver's own buffer, 23.18 GiB at `david 2 / band_batch 16`, measured twice
independently (20 per cent above the fit), and the arena read of 57.24 GB in use at the
instant of the failing request. `MEMORY-AUDIT.md` A17 adds that the stage-peak numbers in the
record came from a patch that was never committed, and that the same commit and dials read
65.47 GB on an A100 and 76.51 GB on an H100, an 11.03 GB offset nobody has named.

Everything about the run has moved since: the guard is inside the solve (`0e85a14`), the
memory modes, the streamed store, the rebuilt projectors, the band dial from the card, and
P122's per-iteration orientation torque (2026-09-26) all postdate it. **So the 28 GB describes
code that no longer runs, and no arithmetic on it can be checked against today's head.**

## What was measured here (D22, A2000, 2026-10-09)

The instrument: `tools/gpu/stage_peaks.py` (committed on this branch). It prints
`Calculator.estimate()`'s report, inventories the live device arrays after setup by shape,
then runs the SCF with every stage wrapped (the `Calculation` methods the loop calls, `_mix`,
`_onecenter_torque`, `_paw_deband`, `_magnetization`), each wrapper blocking on its result
and reading `bytes_in_use` and `peak_bytes_in_use`, and a thread writing the largest host
`VmRSS` and the last finished stage to `<out>.host` every half second, so a process killed
inside a stage still names it.

The cell: `plans/open-d-inputs/nii2-3cell-paw.in`, the NiI2 monolayer three times along
`a1`, fully-relativistic PAW Ni and I (`tests/data/pseudo`; no Br dataset is committed), a
120-degree helix in the plane, 46 bohr of vacuum: nine atoms, `nbnd = 116`, `nk = 2`,
`npwx = 27921`, `nkb = 306`, smooth grid 90x30x192, dense 125x45x270, `ngm = 628995`.
Dials: `k_batch 1`, band batch 16, `david 2`, `wfc_store device`, memory mode's `rebuild`
projectors, pool fraction 0.6.

| stage | in use (GiB) | peak (GiB) |
|---|---|---|
| estimate's peak (resident + max of start, eigensolver, setup) | | **2.49** |
| after setup | 0.238 | 0.404 |
| first `potential` | 0.358 | 1.106 |
| `starting_wavefunctions` | 0.607 | 1.627 |
| first `diagonalize` | 0.800 | **2.651** |
| per-iteration orientation torque, whole-set reverse pass (`dce1eeb`) | 5.397 | 5.449 |
| the same, states in chunked forward mode, one-centre still reverse (`8247a0f`) | 4.858 | 4.910 |
| the same, one-centre in forward mode too (`6f7bd7db`) | 0.945 | 2.651 (unchanged) |
| second `diagonalize`, tighter `ethr` (`d96af746`, three iterations run to the end) | 0.832 | **2.767** |
| estimate at the same commit (after S3 removed the phases line) | | 2.409 |

Three findings.

1. **The total is 15 per cent low, and it is two errors of opposite sign, which the stage
   table separates and the total alone could not.** At the second diagonalisation the card
   peaked at 2.767 GiB against an estimate of 2.409. The estimate's resident part is 0.965
   GiB and what was in use going into the solve was 0.639, so the floor is 0.33 GiB *high*
   (its `GGA gradient temporaries`, 0.27 GiB, and the rebuilt projector chunk, 0.13, are
   counted as resident and are transients of other stages); the `diagonalize` stage rose
   2.128 GiB above what was in use. **The eigensolver's own executable is not the error**:
   `tools/gpu/davidson_memory.py` compiled `davidson_eigensolver_all` at these shapes on the
   same card (one k-point, `david 2`, band batch 16) and its temp buffer is **1.45 GiB**,
   the fit's 1.444 to 0.4 per cent, with 0.27 GB of arguments and outputs. So about **0.68
   GiB of the stage is outside that executable and inside `Calculation.diagonalize`**: the
   next bracket goes there. One candidate is named and not measured, `MEMORY-AUDIT.md` A16's
   spin slice and restack (`psi0[spin]` sliced, solved, `jnp.stack`-ed, on the device-store
   path), whose copies of a 0.19 GiB set would be of the right size; that is one equation in
   several unknowns and is evidence of nothing until a bracket holds it.
2. **The term outside the sized unit, on today's code, was the per-iteration orientation
   torque on every noncollinear PAW run**, which `sizing.py` never modelled: its reverse pass
   took the card from 0.8 to 5.4 GiB, more than twice the estimated peak, and 4.2 of the
   4.6 GiB were the one-centre calculation's tape rather than the states'. It is fixed under
   Part XXII item 1 on this branch. It postdates the slab's 28 GB, so it is not that gap.
3. **A host-side term killed the run before anything on the card did.** With every job
   capped at 8 GiB of host memory, the first three attempts were killed by the cap
   (`systemctl --user show <scope> -p Result` reads `oom-kill`) at the end of iteration 1;
   the probe's host sampler put the rise after the torque and before `site_moments`
   returned, at 9.13 GB. It was `scf/locals.py:_minimum_image_distances` at the first
   `site_moments` of the run: an `(npoint, nat, 8, 3)` float64 array, 2.6 GB on this cell's
   1.5 million dense points and about 30 GB at the 45-atom slab's 3.5 million. Walked a
   block of points at a time on this branch (bit-identical, `test_locals.py`), after which
   the three iterations ran with a host peak of 3.40 GB. It is host memory and so not in
   the 28 GB, but it is a term of the same kind: outside every line of the report.

## The plan

**Step 1, the measurement (one Triton job per card, an `sbatch` the user approves).** Run
`stage_peaks.py` on the 45-atom slab at the current head, at the entry's dials (`1 6 1`,
`nbnd 403`, `david 2`, band batch 16, `k_batch 1`, `wfc_store device`) and `projectors`
both `store` and `rebuild`, for **at least 14 iterations**, past the point where `ethr`
reaches 2.3e-6 and the Davidson average goes from 2 inner steps to 73 (iteration 10 on
2026-09-14), since a short run measures the regime the calculation leaves. On an A100 and on
an H100, the same commit and seed, so A17's 11.03 GB offset is either in the stage table of
one card and not the other, or in neither (then it is outside every wrapped stage, which is
itself the finding). Cost: about two hours of card a job at the 2026-09-14 rates (21 s an
iteration loose, 390 s tight).

**Step 2, the attribution, with no arithmetic across stages.** Read the floor and the
solve apart, as finding 1 had to be read: a total within ten per cent made of a floor 0.33
GiB high and a solve 0.68 GiB low is not a model within ten per cent. For each card: the resident
inventory after setup against the estimate's floor, line by line by shape (a resident array
the model lacks shows as a shape with no line); the stage that sets the peak, from the
`peak_rise` column; and that stage's own compiled `memory_analysis()` where it is one
executable (`tools/gpu/davidson_memory.py` for the solve). A sum of three terms that matches
a total is not used as evidence (`CLAUDE.md`'s trap); only a stage whose bracket contains
the rise is.

**Step 3, the change follows the stage.** If the rise is the Davidson at tight `ethr`, the
fit's `david 2` corner is refitted on the card and the coefficient and its backend recorded
in `sizing.py`; if it is a stage the model has no line for, the line is added with a test
against the bytes a real `Calculation` holds, as `test_sizing.py` does for every line; if it
is fragmentation (bytes free and no hole, A16), the report says so rather than adding bytes.

**Acceptance.** The estimate within 10 per cent of the measured peak at the tight end on
both cards, or the remaining gap named by the stage that holds it and either modelled or
stated in the report's text.

**What is not in this plan.** The donation of the wavefunction buffers (A16's blocked fix 1)
and anything that changes the solve: this is a measurement of the model, and a change to
what is measured would make step 1 describe the wrong code again.
