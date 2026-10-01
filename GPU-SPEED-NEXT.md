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

**Needs first:** a submission the user approves (`sbatch` is never run here). One short GPU job on one
card: `stall_stats.py` on `si16-1k-ecut30` at a commit with the old factor (`b418095^`, 1000) and at
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

## 5. The A100-class profile -- priority 3, needs item 1

On the float32 card the stall-free 64-atom kernel time is 47 per cent FFT, 33 per cent cuBLAS, 12 per
cent elementwise (61,528 launches at 31 us), 7 per cent dense solve (9132 launches at 114.5 us). The
forecast is that the last two lead on a card with a real float64 rate, which is where the options are:
fewer, larger elementwise kernels in the Davidson step; the `narrow` ladder's `lax.switch` branches;
XLA command buffers (a null here: `--xla_gpu_enable_command_buffer` with `WHILE,CONDITIONAL`, 33.0 to
33.5 ms and 70.5 to 71.6 ms, because the A2000 is bound by arithmetic). Nothing to do before item 1.

## 6. The dials that depend on the card -- priority 3, needs a float64 card

* `DEFUMAT_FFT_LAYOUT` (`sticks` default, `box`): equal to round-off. On the A2000 box is 11 per cent
  faster at 8 atoms, equal at 16, and at 64 atoms 3 per cent faster at the whole block and 3 to 6 per
  cent slower at `band_batch` 8 and 16; sticks 9.5 per cent faster at 16 atoms on a CPU. No default
  follows the platform. Decide on a float64 card.
* The accelerator band-dial default is the whole block. Stall-free at 64 atoms the best is 16 to 32,
  10 per cent under it (1350 to 1359 against 1497 ms), and at 16 atoms the whole block is best. A
  default that follows the cell size would be a third dial; the gain is small on this card.
* Memory mode costs a steady 8 per cent over speed mode at 64 atoms (the rebuild, the streamed store
  and one k-point at a time together) and 7 per cent at 16 atoms (the streamed store alone). The
  user chose memory as the default; this is the price to quote.

## 7. Other items that came up -- priority 4

* **The 157-atom slab's "12x too many steps"** (`OPEN.md`, the memory notes) is a separate question: its
  iteration 2 is in the loose-`ethr` regime, ten orders above any floor. The parked-direction factor may
  still matter there; run it with `davidson_steps` on Triton before believing either way.
* **Old GPU readings without step counts.** Every section of `PERFORMANCE.md` before the stall entry
  that holds a card time, and Phases 0, 1 and 5 of `GPU.md`, now open with a dated marking paragraph
  (2026-10-01): a stall only adds steps, so such a time is an upper bound and a card-against-CPU ratio
  a lower bound on the card's advantage, while a ratio between two card arms can be wrong either way.
  The two A/B readings a default rests on are the GTX 1060's memory-against-speed table (the "1.6x the
  time" of `GPU.md`'s opening) and the V100's `k=all, b=1` 2075 ms (`CLAUDE.md`'s reason that corner is
  in neither preset); the first is re-measured on the A2000 in the stall entry, the second needs a
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
