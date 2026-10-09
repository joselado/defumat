---
paths:
  - "defumat/**"
  - "tests/**"
  - "MEMORY-AUDIT.md"
  - "GPU-MEMORY-NEXT.md"
---

*Moved out of `CLAUDE.md` verbatim on 2026-10-08; it loads only when files matching the paths above are read. The text is binding exactly as it was.*

## Memory is part of the design

**A design is not finished until its peak working set is known.** A plane-wave code is
memory-bound as often as it is compute-bound, and what decides whether a calculation runs
at all is usually a working set rather than a flop count. Before landing anything that
allocates per k-point, per band, or per G-vector, say what the peak costs in terms of
`nk`, `nbnd`, `npwx`, `npol` and the FFT grid, and put that number against the RAM of a
real machine — the same reflex the performance rule above asks for with time.

**Where QE spends effort to save memory, copy it unless something better is on offer.**
None of these is incidental to the algorithm:

- **One k-point at a time.** `c_bands.f90`'s `k_loop` diagonalises a single k-point and
  `sum_band.f90` accumulates the density inside the same loop, so QE's working set is one
  k-point's whatever `nks` is, and the parallelism over k comes from MPI pools. Batching
  the whole k axis with `vmap` is this code's deliberate deviation — it is what a GPU wants
  — so it is a **dial** (`defumat/batching.py`), set by `memory_mode`: QE's end of it on a
  CPU; in `'memory'`, the accelerator default, the largest k-chunk the card holds
  (`k_batch = 'fit'`) and the wavefunction store streamed through the card from host RAM
  (`wfc_store = 'stream'`) **only when that chunk is smaller than the mesh**, since with
  every k-point in flight the stream saves nothing and cost 1.9x on an H100 (the user's
  decisions of 2026-10-02); the whole batch in `'speed'`, which is checked against the card
  and falls back to `'memory'` when it would not fit. Rule R6 (k leading) is what keeps
  both available.
- **The sphere, not the box.** Wavefunctions live on the G-vectors inside the cutoff and
  are expanded into the FFT box only for the transform, and only over the sticks the
  sphere occupies (`basis/sticks.py`).
- **Two grids.** The smooth grid carries the wavefunctions and the dense one only the
  augmentation charge that needs it, which is most of the point of `ecutrho`.

A deviation is allowed and is sometimes right — this code trades memory for batching the
way QE trades it for MPI ranks, and `becsum` is carried as a full symmetric matrix where
QE packs the upper triangle. The rule is that such a trade is **stated, measured, and made
selectable when it is large**, never arrived at by accident: name it in the module
docstring, and put the number in `PERFORMANCE.md` beside the timing.

**A test file that sweeps many cells is a memory liability, and splitting the file is not
the fix.** The mechanism is accumulation, not any one peak: cells that share no shape each
compile the whole SCF (and, for a derivative, the gradient) stack afresh and **XLA keeps
every executable for the life of the process**, while an unbounded `lru_cache` of converged
states holds their wavefunctions beside it. Both grow monotonically through the file, and
two files have been killed on this machine that way. Two bounds, and they belong on any
file that runs more than about three distinct cells:

- **`jax.clear_caches()` in an autouse fixture**, after the `yield`. The results stay
  cached; only the compiled code is dropped, which trades recompilation for a peak the
  machine can afford. It gets *faster* as well as smaller, which is the tell that the
  process was paging rather than recompiling (`PERFORMANCE.md`, P28b).
- **`lru_cache(maxsize=2)` on the converged-state helper**, never `maxsize=None` — 2 is
  what a comparison between two cells needs and is the largest that is not a leak.

**Splitting the file only pays off under one of the three runners**, which is why it is the
weaker lever: `tools/run_regression.sh` invokes pytest once per file, so a file boundary
there *is* a process boundary; `tools/test-fast.sh` and a plain `pytest -m slow` run
everything in **one** process, where splitting changes nothing at all.

**Running tests is where this stops being a design rule and becomes an operations
problem, and that half is the `test-runs` skill** -- not running demanding suites side by
side on a shared machine, watching `free -g` and being willing to stop, the per-file
cgroup cap, and the RSS watchdog that names the offending test before the kernel acts.
The one sentence that does not need the skill: **run anything long through
`tools/run_regression.sh`, and commit before starting it.**

What neither bound touches is the peak *inside* one test, which is a real cost to be sized
in advance rather than discovered: the backward pass of an ultrasoft or PAW derivative
carries the augmentation table `Q_ij(G)` — `nh^2 x ngm` per atom, and `nh` is in the
twenties for a fully-relativistic dataset. On a **slab** that is tens of GB and is why a
bismuthene spinor force did not run here at all (P46), while the same physics on a small
bulk cell runs in 33 seconds.

**Both halves of that are now fixed, and the second half is the one that had gone quiet.**
P73 kept `Q_ij(G)` off the *forward* working set by rebuilding it a chunk at a time, and
the sentence above then read as closed — but the scan's residuals are stacked under
`jax.grad`, so the dense table came straight back on the tape, at **`>= 2 GiB` by
construction** (the tabulated route is chosen exactly when the array is too large to
store). Both scan bodies are rematted (`pseudo/augmentation.py`), which takes
`bismuthene-soc-small`'s force tape from **2.32 GiB to 0.99 GiB** measured by the
compiler's own `memory_analysis()`, with the force and stress unchanged to one ulp. The
lesson is the general one: **a forward-path memory fix is not a derivative memory fix, and
on this code the derivative is where the peak is.** `MEMORY-AUDIT.md` is where the rest of
that audit lives.
