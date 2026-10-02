# What is left to do about GPU speed, sized

A to-do list, written 2026-10-01 on branch `pools-next` (merged to `master` the same day), right after
P125 (`PLAN.md`; `PERFORMANCE.md`, "The endgame on a card is a stall"). It says what is open about
speed on an accelerator, what each item needs first, and how to measure it. `GPU-MEMORY-NEXT.md` is
the memory half; `GPU.md` is the older roadmap, whose Phase 1 and Phase 2 paragraphs carry dated notes
where this work corrected them.

**Where the work stands.** The card's endgame stall is fixed at its cause: the Davidson subspace solve
parked its unused directions at 1000 times the largest diagonal element of `H`, which gave the device
`eigh` a matrix of norm 3e4, and on this card that `eigh`'s error follows the norm (3.4e-12 median on
the stalled call's own matrices) where the host's LAPACK does not (2.9e-15 whatever the parked value).
A factor of 4 (`b418095`) cleared the stalls measured and still left 2 of 72 captured solves over
1e-13; since `90e2f8f` the parked rows sit one above the Gershgorin bound of the reduced live block
(`subspace.generalised_eigh(..., parked=)`), the card's error is at its floor (1.4e-14 at worst), and
every arm of the ladder takes the same steps, sixteen atoms the CPU's in every iteration. The 64-atom
memory-mode SCF on D22's RTX A2000 went from 132 s to 14.5 s at the factor of 4 and 14.0 s at the
bound; the CPU is unchanged to the printed digits. **Everything on a card was measured on that one
float32 card** (float64 at 1/70 of float32). No V100, A100, H100 or H200 was run, and that is the first
item below. `PERFORMANCE.md`, "The endgame on a card is a stall", has the review entry ("The parked
value follows the live block").

**How to read the numbers.** *Measured* means taken on the A2000 with the stated steps. A forecast says
so. The rule that cost four wrong readings: **a time on a card is not a measurement until the Davidson
steps beside it are equal across the arms** (`tools/parallel/time_scf.py` prints them; `CLAUDE.md`,
the trap list). The method that found the cause is in `tools/gpu/replay/README.md`: replay the same
inputs on both machines and swap one component at a time.

**Rules for anyone picking one up.** A Triton job is proposed to the user and not submitted
(`CLAUDE.local.md`); D22 is the development card and is offered for runs. `an identity that closes is not
evidence; an A/B is`. Record the steps with every timing. Two stories that fitted the numbers on the way
(slow genuine convergence and false convergence on stagnation) were both wrong, and so was a workaround
(an accelerator floor under `ethr`) that fitted the symptom; the replay is what settled it.

## 1. Run the stall check on a production card -- priority 1, needs the user

The forecast, not a measurement: the one component that differs between the A2000 and a CPU on the
Davidson pair is the device `eigh`, the random-pair test had the two within 2x, and nothing says
whether an A100's or an H100's `eigh` has the same error on a norm-3e4 matrix.

**Needs first:** a submission the user approves (`sbatch` is never run here). The job is written:
`tools/gpu/stall-check.sbatch` (the captured-solve `eigh` table, the step ladder at the committed
parking, the float32 tier, an `nsys` profile), with the submit line in its header. As first planned, one
short GPU job on one card: `stall_stats.py` on `si16-1k-ecut30` at a commit with the old factor (`b418095^`, 1000) and at
the current one, then `call_xplat.py` export, replay and `HOSTEIGH=eigh`, then the captured-solve error
table of the stall entry (the card's `eigh` against the live block alone), then `time_scf.py` on `si16` and `si64` in the
default memory mode with `nsys profile -t cuda`, then `kern_cats.py`. **Measure:** steps in the last two
iterations at each factor, the share of kernel time in the elementwise chain and the dense solve (the
A100 forecast in `PERFORMANCE.md` says those come first once float64 is fast), and whether the 13x
`conv_thr` entry of `GPU.md` Phase 1 (V100) was this.

## 2, 3 and 4. The margin, the test, and why the device `eigh` differs -- closed 2026-10-01 evening

**2, the margin.** There is no factor left to have a margin. The parked rows sit one above a bound of the
reduced live block, which follows the live spectrum (6 to 9 Ry on sixteen atoms) and not the kinetic
energy at the cutoff, so the norm no longer grows with `ecutwfc`. Why any parked value above the
starting block's largest Ritz value is safe, at any cutoff and on any dataset: the lowest `nbnd` Ritz
values never rise during a call (nested subspaces; the refresh keeps their vectors). The canonical
retry multiplied the old park by 1000 (a norm of 121001 on a Davidson-shaped pair) and parks at a bound
of its kept block since `cc21ad4`.

**3, the test.** `test_solvers.py::test_the_subspace_solve_is_handed_a_matrix_of_the_order_of_the_diagonal`
traces the eigenvalues each route returns inside a real cold Davidson solve and holds the parked value
above every live root and under the largest diagonal element of `H` (7.5 against 11.2; the factor of 4
gave 45.7, and the test fails at 1000). `test_subspace_robustness.py` covers interleaved and trailing
parked rows on both routes and the retry's old compounding.

**4, why.** Not fully: on this card the device `eigh`'s error reaches `eps` times the norm of what it is
handed and LAPACK's does not, and that is what was measured. Where the parked rows sit matters only when
they are interleaved (both platforms then follow the norm, and sorting them last cures it), and
Davidson's are already trailing. Which stage of cuSOLVER (`sytrd`, the divide and conquer, `ormtr`)
carries it was not taken apart, and with the norm at the live block's own it no longer matters here.

## 4a. Memory mode on a k-mesh, and a float32 band side -- done 2026-10-01 evening, two decisions left

Memory mode cost 1.8x against speed mode on a k-mesh of a small cell, nearly all of it the one k-point
per call; `k_batch = 'fit'` sizes the chunk from the card (1.97x faster at 27 k-points on eight-atom
silicon) and the Davidson width ladder now works under a batch over k, which made a 22-point chunk on
sixteen atoms a wash where it had been 1.28x slower. `band_precision = 'single'` is 3.9x to 4.2x on 64
atoms on the A2000 to `conv_thr = 1e-7`; `'mixed'` does not pay. **The user's to decide:** whether
memory mode's default becomes `'fit'`, and whether speed mode's fallback becomes the largest fitting
chunk rather than one k-point. **Open:** a precision switch that keeps the mixer's superlinear step
(`PLAN.md` P126, "What is not done"). Records: `PERFORMANCE.md`, "A k-chunk sized to the card" and "The
band side in single precision".

**The radial transforms compiled for minutes on a card** (fixed 2026-10-02, `formfactors._radial_values`;
`PERFORMANCE.md`, "The radial transforms took minutes to compile on a card"). Card HLO dumps of PAW
silicon, spin-orbit platinum and spin-orbit bismuth found no other loop in a hot path: radial-table chunks
at setup, PAW's one-centre loop over atoms, `newd`'s noncollinear integrals in 591 chunks an SCF
iteration, and the smeared Fermi level's fixed bisection of 100 and 200 trips (QE's `efermig`, left as
it is: a few hundred small launches an iteration, which matters only for a small metal on a card).

## 5. The A100-class profile -- priority 3, needs item 1

On the float32 card the stall-free 64-atom kernel time is 47 per cent FFT, 33 per cent cuBLAS, 12 per
cent elementwise (61,528 launches at 31 us), 7 per cent dense solve (9132 launches at 114.5 us). The
forecast is that the last two lead on a card with a real float64 rate, which is where the options are:
fewer, larger elementwise kernels in the Davidson step; the `narrow` ladder's `lax.switch` branches;
XLA command buffers (a null here: `--xla_gpu_enable_command_buffer` with `WHILE,CONDITIONAL`, 33.0 to
33.5 ms and 70.5 to 71.6 ms, because the A2000 is bound by arithmetic). Nothing to do before item 1.

## 6. The dials that depend on the card -- priority 3, needs a float64 card

* `DEFUMAT_FFT_LAYOUT` (`sticks` default, `box`): equal to round-off. **Re-measured 2026-10-01 evening,
  after the stick fill became a gather** (it had been a scatter the card ran as a loop over sticks): on
  the A2000 in speed mode sticks 63.9 against box 70.3 ms at 16 atoms, 153 against 173 on a 27-point
  mesh of 8 atoms, 1424 against 1423 at 64, and 17.7 against 16.6 at 8 atoms with one k-point; sticks
  9.5 per cent faster at 16 atoms on a CPU. The default stays `sticks`. Decide on a float64 card.
* **The subspace `eigh` by the host's LAPACK, for small matrices on a card** -- now an opt-in dial,
  `DEFUMAT_HOST_EIGH_ROWS` (2026-10-02, measured on whole SCFs: 1.36x on a 27-point mesh in memory mode,
  1.14x at one k-point, 1.00x at 64 atoms, 0.98x in speed mode; `PERFORMANCE.md`, "The host's LAPACK for
  the small subspace solves"); whether it becomes memory mode's default on a card is the user's. The
  device `syevd` is 0.63, 1.11, 3.56 and 6.65 ms a call at 16, 32, 64 and 128 rows on the A2000 against
  0.23, 0.32, 0.71 and 2.21 for LAPACK through `jax.pure_callback`, crossing at about 256; after the two
  fixes of that evening it is 71 per cent of the kernel time of eight-atom silicon at one k-point and 61
  per cent of a 27-point mesh in memory mode, so the host route would be worth about 1.35x to 1.45x
  there. It puts a host round trip inside the Davidson loop, which `CLAUDE.md`'s JAX rules forbid, and
  on a batched solve (speed mode) the device's batched kernels win. jax 0.11 does write an executable
  with host callbacks to the persistent cache (`compiler.compile_or_get_cached`), so that is not the
  objection. A production card's device `syevd` at these sizes is unmeasured.
* The accelerator band-dial default is the whole block. Stall-free at 64 atoms the best is 16 to 32,
  10 per cent under it (1350 to 1359 against 1497 ms), and at 16 atoms the whole block is best. A
  default that follows the cell size would be a third dial; the gain is small on this card.
* `DEFUMAT_BAND_RUNGS` on a large spinor cell (2026-10-02, `bi20-soc`, the cache off): 4, 2 and 1 rungs
  compile the Davidson solve in 78.9, 68.8 and 60.9 s and run three iterations in 74.7, 76.3 and 77.2 s,
  equal steps and energy, so the three extra rungs pay their 18 s back after about 22 iterations. The
  default stays four. **That is a cold-cache figure**: the compile is XLA's autotuning, which the
  persistent cache keeps per matrix-product shape, so with a warm cache two and one rungs compile in 2.8
  and 2.7 s (`PERFORMANCE.md`, "A card's compile is its autotuning").
* **A card job's cache directory should outlive the job** (2026-10-02): 76 of `bi20-soc`'s 79 s of
  Davidson compile is autotuning, kept in `DEFUMAT_CACHE_DIR`'s `xla_gpu_per_fusion_autotune_cache_dir`.
  `--xla_gpu_autotune_level=0` removes it at 1.3 to 5.4 per cent of every iteration, so it is a one-off
  run's setting and not a default.
* Memory mode costs a steady 8 per cent over speed mode at 64 atoms (the rebuild, the streamed store
  and one k-point at a time together) and 7 per cent at 16 atoms (the streamed store alone). The
  user chose memory as the default; this is the price to quote.

## 7. Other items that came up -- priority 4

* **A warm response call on a card is 40 per cent key** (2026-10-02): after `defumat/eager.py` a second
  AlAs dielectric tensor on the A2000 is 7.1 s, of which 2.1 s is tracing the closures and 0.8 s
  printing their jaxprs to hash them. The fix is a perturbation written as a function of explicit
  arrays (a pytree with a static kind), so that `SternheimerSolver.solve` and the response density can be
  module-level `jit`s keyed without a trace; it touches every perturbation the response stack builds
  (`efield`, `phonon`, `phononq`, `strain`, `electrostriction`, `piezo`). Worth doing once the
  response stack runs on cards in production.

* **The 157-atom slab's "12x too many steps"** (`OPEN.md`, the memory notes) is a separate question: its
  iteration 2 is in the loose-`ethr` regime, ten orders above any floor. **It does not reproduce on small
  cells** (2026-10-01 evening, this workstation's CPU, `diago_david_ndim = 2`, the same input through
  both codes): the aluminium slab takes 2.2 Davidson steps per diagonalisation against `pw.x`'s 2.8, the
  spin-polarised hydrogen chain 3.7 against 4.4. The slab's number was taken at `f2be49f`, before the
  per-band thresholds, the band ladder and both parking fixes, and has not been re-measured; its first
  three iterations with `davidson_steps` on an H200 would settle it, which needs a submission.
* **A subspace `eigh` of the lowest `nbnd` only** would cut the back-transformation by about four on a
  large cell, where the full complex128 solve on the A2000 is 1.1 s at `m = 2048` and 7.4 s at 4096 per
  Davidson step; `lax.linalg.eigh(..., subset_by_index=)` raises `NotImplementedError` on CPU and GPU in
  jax 0.11, so it needs a custom call to `cusolver`'s `syevdx`.
* **Old GPU readings without step counts.** Every section of `PERFORMANCE.md` before the stall entry
  that holds a card time, and Phases 0, 1 and 5 of `GPU.md`, now open with a dated marking paragraph
  (2026-10-01): a stall only adds steps, so such a time is an upper bound and a card-against-CPU ratio
  a lower bound on the card's advantage, while a ratio between two card arms can be wrong either way.
  The two A/B readings a default rests on are the GTX 1060's memory-against-speed table (the "1.6x the
  time" of `GPU.md`'s opening) and the V100's `k=all, b=1` 2075 ms (`CLAUDE.md`'s reason that corner is
  in neither preset); the first is re-measured on the A2000 in `PERFORMANCE.md`, "Memory mode on a
  k-mesh, re-measured with the steps beside it" (1.81x and 1.85x, steps equal), the second needs a
  float64 card.
* **The float32 tier** (`GPU-MEMORY-NEXT.md` item 26): the first blocker is named (setup arrays built
  from the radial tables ignore the precision policy, so `H|psi>` comes back complex128 into a
  complex64 basis). On the development cards it is the lever (a `4096^2` product 10.9 ms against 756);
  on a production card its ceiling is about 2x plus halving every device array. Not started.
* **A request tighter than `conv_thr = 1e-10`** no longer stalls on the arms measured (1e-11 and 1e-12
  at sixteen atoms, 1e-11 at 64), and `DEFUMAT_ETHR_MIN` raises the floor if one is met, at a cost in
  accuracy (the force error doubled at 3e-12 on a displaced cell). The warning for a call at the budget
  says when to reach for it.

## What was tried and is not worth repeating

All on the replayed call, 20 seeds, A2000, before the cause was found; each changed the executable and
rolled the same dice again. Projecting the correction out of the Ritz span; `diago_david_ndim` 2 and
3; an exact refresh with a hard restart every 20, 8 and 4 steps; Ritz values recomputed as Rayleigh
quotients (worse, because the parked 3e4 diagonal sits in their numerator); noise injected into the
projected pair on a CPU (does not reproduce the card); an accelerator floor under `ethr` of 1e-12 and
3e-12 (committed, then withdrawn); XLA command-buffer flags (a null). Numbers: `PERFORMANCE.md`.

## Where things are

Tools: `tools/gpu/replay/` (this work), `tools/parallel/time_scf.py` and `gpu_scan.sh` (timing; the
scan takes the FFT layout as a fourth field of an entry), `tools/gpu/` (the older sbatch scripts).
D22: `CLAUDE.local.md`, "The GPU workstation `D22-0161`" and its 2026-10-01 additions (`nsys` path,
how to launch background jobs, the timing environment); its checkout is detached at `4ee01c2`. Memory:
`davidson-stall-at-ethr-floor.md`. Guide: `docs/features.tex`, the batching section (the floor lever, the
budget warning, the layout dial).
