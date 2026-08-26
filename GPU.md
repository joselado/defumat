# Running on a GPU: what is already there, what is next, and what each step costs

A roadmap for the GPU half of `PLAN.md` P10, written so that the session which picks it
up does not re-derive the constraints. `PLAN.md` stays the phase tracker and
`PERFORMANCE.md` the running log of measurements; this file is about what is *reachable*,
in what order, and at what price.

The organising fact is stated once and everything below follows from it:

> **Nothing here is a port.** JAX already emits GPU code from the same source, and the
> rules that made that true — D1, D2, R6, R7, the `Precision` policy, the two batching
> dials —
> have been binding since P0. What was missing was not a backend but **evidence**: no line of
> this code had ever run on a GPU, the choices that were tuned against a CPU were measured
> against a CPU, and the metric this project reports is a single-core CPU comparison that
> means nothing on an accelerator. The work is to establish the first, revisit the second
> and define the third.

**Phase 0 is done as of 2026-08-25** and the first of those three is settled: the same
source, unmodified, runs on a Tesla V100 and reproduces the CPU energy to 1.6e-13 Ry. It
also returned the two numbers this file was written without — **fp64 costs 1.78–1.98x on a
matmul and 0.85–1.44x on an FFT**, which fixes Phase 3's rank, and **the defaults cost
4.5x**, which turns §1's warning into a measurement. Since then Phase 1 has had a first
pass, the dials' defaults have been made per-platform (§1), and Phase 5's CPU half — the
tape per response property — is measured; **Phases 2, 3 and 4 are unrun**, as is Phase 5's
GPU half.

---

## 1. What is already GPU-ready, by design

None of this needs doing. It is listed so that a future session spends its time on what is
actually missing.

| already true | why it matters on a GPU | where |
|---|---|---|
| **The whole compute path is pure JAX** — no `libxc`, no C callbacks, and (checked) **not one `numba` import in the package** | a host callback in the traced path serialises the device; there are none | D1, `PLAN.md` §6 |
| **`k` is a traced argument of `H(k)`** | no per-k host table to copy in; the velocity operator is a `jvp` | D2 |
| **Plane waves padded to `npwx` with a mask** | static shapes, so one compiled kernel serves every k-point instead of a retrace each | R7 |
| **`k` is the leading independent axis of every wavefunction-shaped array** | the `vmap` axis and the future `sharding` axis already exist and are load-bearing | R6 |
| **Every dtype comes from `config.Precision`** — no `jnp.complex128` or `1.0j` literals in compute code | float32 is a configuration change, not a whole-codebase edit | `config.py` |
| **`batching.py` has *two* dials — `k_batch` and `band_batch` — and `None` on either means one `vmap` over that whole axis** | those two settings *are* the GPU execution mode, and both already exist, are reachable from every entry point, and are tested not to change a result | `batching.py` |
| **State objects are frozen `equinox.Module` pytrees with `eqx.field(static=True)` config** | `jit`/`grad` boundaries are already clean; no mutable module globals to thread to a device | conventions |
| **The SCF's only host sync is its convergence test**, once per iteration | a device stall per iteration is affordable; one per inner step is not | `PLAN.md` §5 |

**Both dials used to default to QE's loop whatever the platform, and on a GPU both
defaults were wrong. Fixed 2026-08-26: the default now follows the platform** — QE's loop
on a CPU, the whole of both axes on anything else (`batching.py`'s `_platform_default`,
tested against a substituted backend so it needs no card). This was the single most
important line in this table, and what made it so is that `k_batch` defaulted to
one k-point at a time and `band_batch` to one band at a time, because that is what a *cache*
wants: `batching.py` measures the band loop at **2.48x faster than the batch** on
`si16-1k-ecut30`, for a reason with nothing to do with JAX — one band's real-space box is
1.5 MB and thirty-two of them are 48 MB, so the batched transform streams from memory where
the looped one stays in cache. A GPU has no such cache and inverts the conclusion, which
that module's own docstring already says: *"the same escape hatch for a GPU, which wants
the batch that a cache does not."* **A GPU run left on the *old* defaults serialised every
FFT into a per-band kernel launch** — close to the worst execution mode available — which
is why this is now a per-platform default rather than a line in a job script. §5's rule is
what shapes the fix: a dial with a per-platform default and both settings tested, never a
rewrite, so nothing moves on a CPU and an explicit argument or `PYPRESSO_*_BATCH` still
beats the platform. Every GPU number below this line was produced with the dials set by
hand in an sbatch script; the same settings are what a bare `run_scf` on a card now picks
on its own.

**One consequence worth stating plainly**: the first GPU run is expected to *work*. (It
did — Phase 0, 2026-08-25.) If it does not, that is information about JAX or about the
machine, not about a missing port.
That is a claim about *crashing*, and it is well supported. It is **not** a claim about the
numbers: §3's Phase 0 has to earn that separately, and the two ways it could fail quietly
are a nondeterministic accumulation (Phase 0, check 5) and a memory reading that measures
the allocator instead of the code (check 4).

## 2. Four constraints that shape the whole roadmap

**2.1 There is no GPU on this machine, so the validation machine is a cluster.** Development
here is CPU-only. That makes every phase below either *testable today* or *blocked on first
contact*, and the roadmap is ordered by that split rather than by expected speedup — which
is the single biggest difference between this file and a GPU plan written by someone who
can run one.

**The cluster's rules are not this project's to set, and they are not in this repository.**
Access policy, account paths, partitions and site limits live in the private notes beside
this checkout (untracked, alongside `CLAUDE.md`'s committed content); read them before
proposing any job. Two consequences bind every phase below: a job is **proposed to the
user rather than submitted**, and **every scheduler value is looked up rather than
remembered**. This file deliberately named no GPU model, memory size or fp64 rate while it
did not know them, because guessing one would have poisoned every measurement that
followed. Phase 0 has since measured them **on one card** — a V100-SXM2-16GB — and they are
recorded there and in `PERFORMANCE.md` as a property of that card, not of GPUs: the same
cluster offers A100, H100, H200, B300 and GH200 partitions, whose fp64 rates and memory
differ by an order of magnitude, so a later phase re-measures rather than inherits.

**2.2 What was measured on a CPU was measured on a CPU.** The standing rule is to mirror
QE's implementation in the performance-critical path, and it has been right more than once.
It also carries its own escape clause — "where QE's fast path is a table lookup, the
differentiable equivalent wins, and that trade is recorded rather than silently taken" —
and a GPU is the other case where it may not hold. The concrete instance is the **stick
FFT** (`basis/sticks.py`): transforming only the sphere's columns along `z` and then doing
a contiguous 2D `xy` pass is QE's layout and beats a fused 3D transform **1.13x on the
eight-atom silicon cell and 1.02x on sixteen atoms** — that module's own docstring. Those
margins are thin, and what buys them is a gather/scatter into the box, which is exactly the
access pattern a GPU punishes and a batched `cuFFT` plan makes unnecessary. **That is a
hypothesis, not a finding**, and §3's Phase 2 is where it gets measured.

**2.3 The metric changes, and reusing the CPU one would flatter the result by a factor
nobody earned.** `CLAUDE.md`'s measurement is *single-core* pypresso against *single-core*
QE on the same input, and `tools/compare_qe.py` pins both to one core precisely so the
comparison is not inflated by core count. A GPU number put against that baseline would be
meaningless in the same way, only more so. The GPU metric is therefore stated separately
and never substituted for it:

* **GPU pypresso against CPU pypresso, same input, same code, per SCF iteration**, with
  compile time reported as its own line rather than amortised away — and **the CPU side
  pinned to a stated core count**, because one core, four and the whole node differ by an
  order of magnitude and an unpinned baseline is not a baseline. That is the entire lesson
  of `tools/compare_qe.py` pinning both sides, applied to a comparison where only one side
  changed;
* the QE comparison stays what it is — a CPU claim — and any table mixing the two says
  which column is which.

**And the comparison this project will eventually owe is pypresso-GPU against QE-GPU.** QE
7.5 has an OpenACC port and it is in the vendored tree — `PW/src/vloc_psi_acc.f90`,
`add_vuspsi_acc.f90`, `stres_har.f90` and others carry `!$acc` directives. Nothing here has
to do that comparison soon, and building QE's GPU path is a project of its own, but a
roadmap that defines an accelerator metric while not knowing the reference exists would be
setting up the same flattering baseline §2.3 is written to prevent.

**2.4 Memory is the binding constraint, not arithmetic — and the mode Phase 0 wants is the
mode already known to exhaust it.** The rule is that a design is not finished until its
peak working set is known *in terms of the parameters*, so here they are rather than a
couple of whole-run numbers:

| term | size | when it dominates |
|---|---|---|
| wavefunctions | `nk · nbnd · npwx · npol · 16 B` (complex128) | always resident |
| Davidson subspace | the above `× nvecx/nbnd` (QE's default `nvecx = 4 nbnd`) | during `c_bands` |
| real-space boxes in flight | `band_batch · n1 n2 n3 · 16 B` per k in flight | the band dial sets this |
| dense-grid fields | `nspin_mag · n1 n2 n3 · 8 B`, several of them, plus the mixer's history | ultrasoft/PAW, big `ecutrho` |
| response tape | reverse-mode through the radial transforms, **not** bounded by the above | stress, and only in reverse mode |

Evaluate those against a card's HBM rather than against this workstation's RAM, and against
the four numbers this repository has already measured:

* the ferromagnetic NiI2 of P31/P32 peaks around **7.4 GB**;
* the sixteen-atom dynamical matrix around **7 GB**;
* **bismuthene — 19 irreducible k-points, two-component spinors, 35 Ry — was killed at
  12.7 GB and still climbing, batched over all its k-points.** That is `k_batch=None`,
  which is exactly what §1 calls "the GPU execution mode" and what Phase 0 sets. **The
  configuration this roadmap prescribes is one already known to exhaust a large-memory
  machine on a committed case**, and the resolution is not to abandon it but to accept that
  the two dials are a *memory-versus-parallelism* trade with a per-platform optimum, to be
  found by measurement and not by taking either end;
* the stress's reverse-mode tape through the radial transforms reaches **11 GB on
  eight-atom ultrasoft silicon — the largest single allocation anywhere in this code**, and
  it is the reverse pass alone.

And **`donate_argnums` is used in exactly zero places** today, despite being a rule since
P0. On a CPU that costs a copy; on a device with a fraction of the memory it decides
whether a cell runs at all.

---

## 3. The phases

> **Fixed 2026-08-26 — and worth reading before trusting a GPU number.**
> `si64-1k-ecut30` converged on an H200 at `conv_thr = 1e-8`, reproducing QE's
> total energy, and returned **`NaN`** at `1e-10`, where four CPU cores converge
> in nine iterations. **The cause was one line, and it was not the GPU's.** As
> Davidson's subspace fills, the vectors it expands with are normalised residuals
> of roots that have *already converged* — amplified round-off — so they go
> linearly dependent, the overlap's smallest eigenvalue lands on the round-off
> floor, and its **sign is arbitrary**: measured on the device,
> `min eig(S) = -4.3e-16` against `max|S| = 1.0`. `jnp.linalg.cholesky` of that
> takes the square root of a negative pivot and **returns `NaN` rather than
> raising**. The CPU landed on the positive side of the same coin flip, which is
> the whole of why it looked device-specific. Cholesky is still the fast path and
> is taken bit-for-bit whenever it works; canonical orthogonalisation runs only
> when the factor comes back non-finite (`solvers/subspace.py`). Verified on the
> device: 9 iterations, QE's energy to 1.6e-8 Ry. Measurements in
> `PERFORMANCE.md`.
>
> **The guard was a `lax.cond` *inside* the per-k solve until 2026-08-26, and
> that cost 2.85x of every multi-k subspace solve on the GPU's own default
> dials** — a `cond` under `vmap` has no branch to take and runs both sides. It
> is now one predicate over the whole k-set, outside `map_k`, returning
> bit-for-bit the same values. Phase 1's continuation below.

### Phase 0 — first contact: does it run, and does it give the same number? ✅ DONE (bar check 5's cross-job form)

**Run 2026-08-25**, Tesla V100-SXM2-16GB on Aalto's Triton, jax 0.11.1 with
`jax-cuda12-plugin` 0.11.1, against four pinned cores of an EPYC Milan.
`tools/gpu/` is the harness; the table and the reasoning are in `PERFORMANCE.md`
under "First contact with a GPU". What the five checks returned:

1. **x64 works**, asserted on a device array. fp64 costs **1.78–1.98x on a
   matmul** and **0.85–1.44x on a batched 3D FFT** over the six records — the
   V100 is a 1:2 fp64 part and the matmul reproduces that, while the transform's
   cost is **grid-dependent** and has to be quoted as a range (1.44x on
   `al10-metal`'s 27x15x72, 0.85x on a 16³ box where the transform is
   latency-bound and the timing is noise). **This settles Phase 3's rank**,
   which this file declined to guess and flagged as provisional: single
   precision's ceiling is ≤2x on the dense algebra and at most ~1.4x on the
   transforms, so Phase 3 goes **after Phase 4**, and the missing float32 *tier*
   stays owed by `PLAN.md` §5 on its own ground rather than as a GPU performance
   item.
2. **The energy matches**, to **1.6e-13 Ry** against the CPU run worst case, and
   `al10-metal` reproduces the **committed QE reference to 1.88e-09 Ry** — the
   same digit the development workstation gets.
3. **Compile is 3.7–14.3 s**, reported as its own line. `PYPRESSO_CACHE_DIR` on
   scratch: one writer per job, so the concurrent-writer question is *still
   open* rather than answered.
4. **Peak device memory 0.40 GB** against an 11.8 GB allocator limit, from
   `memory_stats`. Not binding at this scale; the dial's cost is visible
   (0.10 → 0.40 GB) and is §2.4's trade in miniature.
5. **Determinism holds bit for bit**, every case, run twice in one process. The
   scatter-add-through-atomics hazard did not materialise on this hardware.
   **The cross-job form is not yet run** — the identical *job* twice, which
   catches an irreproducible compilation where one process cannot.

**And the headline, which is §1's prediction turned into a number.** On the GPU
`al10-metal` runs at **801 ms/iteration on the defaults and 177 at `k=all,
b=all`** — 4.5x — where the same change on a CPU costs 1.2x the wrong way. **A
GPU run left on the cache-shaped defaults gives up 4.5x.** One thing the file
did not predict: **`k=all, b=1` is worse than either end** (2075 ms), because
batching k while looping bands multiplies the per-band launches by `nk` and buys
the batched mode's memory with the looped mode's launch count. The two axes move
together or not at all.

**The expectation in §1 held**: the first GPU run worked, and nothing was ported
to make it. Small cells lose (`si-1k` 0.33x, `si8-1k` 0.35x) on launch overhead,
which is the standing two-atom-cell rule in its GPU form.

---

**What.** One SCF, float64, both batching dials set explicitly, on a cluster GPU node,
reproducing the CPU result to the tolerance that case is already committed against.

**Why first, and why this small.** Every later phase is a measurement, and a measurement
needs a machine that works. This phase's deliverable is not a speedup — it is a job script
that runs and a number that matches. `PLAN.md` P34 makes the same argument for the CPU
cluster harness ("build the thin path first"), and this is that path: **one job, run by
hand, not gated on P34 being built.**

**How.** Propose the job script to the user rather than submitting it, writing under the
scratch location the private cluster notes specify, with every scheduler value looked up
there rather than remembered (§2.1).

**Not on the `benchmarks/` inputs alone, and this is a trap the CPU metric sets.** Those
files are **single k-point on purpose** — `CLAUDE.md` says so — because both codes
parallelise over k and a multi-k CPU comparison would measure batching rather than physics.
That choice makes them exactly the wrong first GPU inputs: with `nk = 1`, `k_batch=None` is
a no-op, and the axis §1 calls the GPU execution mode is not exercised at all. So:

* `benchmarks/si-1k.in` first, because two atoms and one k-point is debuggable;
* `si8-1k.in` next, because two atoms show nothing (the standing rule);
* and then **a genuinely multi-k case with a committed reference** — `pw_metal`'s ten
  k-points, or the bismuthene of P14 at nineteen — because that is the first input on which
  either dial means anything.

**Set both dials explicitly**, `k_batch` and `band_batch` — the platform default now
picks the right end on its own (§1), but a measurement says which setting produced it
rather than inheriting one. Expect to run the multi-k case at more than one setting of
each: §2.4 says the fully-batched end is the one already measured to exhaust 12.7 GB.

**What to actually check, in order** — each one can fail independently:

1. **Does x64 work at all, and at what cost?** `jax_enable_x64` is set before any array
   exists and every validated number in this project is float64. Consumer and some
   data-centre cards run fp64 at a small fraction of fp32. If the ratio is bad, that is not
   a blocker — it is the argument for Phase 3 — but it must be *known* before any timing is
   interpreted.
2. **The total energy against the CPU run**, to the committed tolerance. Not "close".
3. **Compile time**, separately, and whether `PYPRESSO_CACHE_DIR` on scratch survives the
   node (its concurrent-writer behaviour on the shared filesystem is unverified — P34 flags
   this too).
4. **Peak device memory — measured with JAX's device memory profiler, not by reading the
   allocator.** XLA preallocates a large fraction of the card on first use, so a naive
   reading reports the size of the pool and not the working set. That would produce a
   *wrong number rather than a failed check*, which under this project's rules is the worse
   outcome of the two. Report it against §2.4's formulas, per dial setting.
5. **Determinism: run the identical job twice and diff the converged results bit for bit.**
   The plan's hardest rule is that no phase may change a validated number, and a single run
   compared once cannot see the way a GPU breaks it. `basis/fft.py` scatters plane waves
   into the FFT box with an **accumulating scatter over deliberately duplicated indices**
   (`flat.at[..., fft_index].add(...)`, and the same in the stick path — the padding entries
   all share the index of `G = 0`), on every wavefunction transform. XLA commonly lowers a
   float scatter-add through atomics, whose summation order is not reproducible run to run.
   *Whether it does so here is unverified* — it is a hazard to test, not an asserted
   outcome — but the test costs one extra job and nothing else in this plan would notice.
   If the two runs differ, settle the determinism policy (XLA's deterministic-ops flag, or
   restructuring the padding so the indices are unique) **before** any GPU number is called
   validated.

**Two settings to get right in the job script, and one of them is not the obvious way
round.**

* **`jax_enable_x64` must be verified at runtime, not assumed.** It is set in
  `pypresso/__init__.py:24`, immediately after `import jax` and before anything else
  touches it, which is structurally correct. But its failure mode — JAX silently running
  the whole thing in single precision — produces a result that is plausible rather than
  wrong-looking, and a *platform change is exactly when a config assumption stops holding*.
  Assert a dtype on the device at the top of the job, rather than trusting the call.
* **`XLA_PYTHON_CLIENT_PREALLOCATE`: leave it alone unless something shares the device.**
  JAX preallocates the bulk of device memory on first use. For a single-tenant pypresso job
  that is *what you want* — it avoids fragmentation over a long SCF — so the reflex to
  disable it is wrong here. It has to be set `false` when another framework shares the
  process or the card, which is the case that bites and is worth knowing before it does.

**Three environment questions to settle on first contact**, none of them answerable from
here, and the first is the one that decides whether any of this roadmap is reachable:

* **does the cluster's JAX build actually have a working CUDA plugin on the card you land
  on?** A broken or absent plugin makes every measurement below unreachable, and it is not
  something this codebase can fix — check it before anything else;
* **can you constrain which GPU generation you get?** Schedulers usually allow requesting a
  compute-capability floor, and it matters here specifically because fp64 rate is check 1
  above and varies by an order of magnitude across generations;
* **is there a short-queue partition for the first job?** First contact should not wait
  behind a long queue, and it needs a CPU run alongside it as §2.3's pinned baseline.

The private cluster notes carry the site-specific answers where they are known.

**Testable without a GPU:** no. This was the gate, and it is open.

### Phase 1 — where the time actually goes 🔶 FIRST PASS DONE (16 atoms, single-k)

**Run 2026-08-25** on a V100-SXM2-32GB against four EPYC Milan cores, on
`si8-1k-ecut30`, `si16-1k` and `si16-1k-ecut30` — **single-k cells on purpose**,
so that no k-parallelism is available and every ratio belongs to the per-k path.
Numbers in `PERFORMANCE.md`; the two findings:

**The diagonalisation does win, and the FFTs win more.** On `si16-1k-ecut30` at
`band_batch = all`: `h_psi` **6.8x**, Davidson **3.9x**, and `v_of_rho` **0.74x**
— the GPU loses on the potential. But the band dial is worth more than the
device is: Davidson goes 2.632 → 0.386 s between `b=1` and `b=all` on the *same*
card, while the same change costs the CPU 1.224 → 1.490 s. On the default band
dial a GPU run does not merely fail to win, it loses at **0.20x**.

**The named pathology is real and it is the SCF's endgame.** This file predicted
that "the small dense `eigh` and matmuls inside a `lax.while_loop`" would be an
accelerator pathology. Measured: tightening `conv_thr` from 1e-8 to 1e-10 costs
the CPU **nothing per iteration** (233 → 237 ms, flat) and costs the GPU **13x**
(36 → 459 ms). QE's adaptive `ethr` schedule means a tighter threshold is more
Davidson steps, each one small dense algebra that does not vectorise — so the
FFTs took their 6.8x and the subspace solve took nothing. The consequence:
**16 atoms keeps 1.80x at a production `conv_thr`, not the 6.5x its loose
iterations suggest**, and any GPU speedup quoted here has to say which
`conv_thr` produced it.

**Thirty-two atoms changes the size of the answer, and softens the second
finding.** On `si32-1k-ecut30` (11781 plane waves, 64 bands) at `band_batch =
all`: `h_psi` **17.9x**, Davidson **13.5x**, `v_of_rho` **1.3x** — the
diagonalisation's advantage more than triples for one doubling of the cell, and
the potential, which lost at 16 atoms, wins. The `ethr` tax falls from 3.6x to
**1.2x** over the same step: the subspace algebra has not got faster, it is just
a smaller share of a bigger cell. So the whole-SCF ladder is **not** monotonic
(2.35x, 1.80x, 13.5x for 8, 16, 32 atoms) because it inherits each case's own
SCF path — **quote the per-stage ratios, which scale cleanly**.

That still makes the subspace algebra the first thing worth attacking, and it
still reorders Phase 2 behind it — but the case for it is "this is what caps a
mid-sized cell", not "this is what caps the GPU".

**A practical finding for anyone running this:** at 32 atoms the default band
dial is not a slowdown, it is a wall — the GPU did not finish two SCF runs at
`band_batch = 1` in the fifteen minutes it had, where four CPU cores do one in
eleven seconds. **The dials are a correctness-of-configuration matter on a GPU,
not tuning.**

**And the 2.6e-09 Ry outlier did not grow**: the same configuration on twice the
cell agrees with the CPU to **-7.4e-13 Ry**. It was particular to
`si16-1k-ecut30`, not a scaling defect in the batched dial.

**Second pass, 2026-08-26: the profile goes one level in, and it found a
regression the first pass could not see.** `tools/gpu/davidson_profile.py`
measures the *parts* of a Davidson step at a case's own shapes, on either
platform. The CPU half is run; the GPU half is `davidson-gpu.sbatch`, proposed
and not submitted. What it returned:

* **the guard the 64-atom `NaN` fix added was inside the k batch, and a `cond`
  under `vmap` is not a branch.** JAX lowers a conditional with a batched
  predicate to `select_n` over *both* branches, and `k_batch=None` — the GPU
  default this file argued for in §1 — is a `vmap` over k. So every multi-k GPU
  subspace solve since `a351005` has computed canonical orthogonalisation as
  well as the Cholesky route it uses: **2.85x** at `si10-nc`'s shapes. The guard
  now lives outside `map_k`, where its predicate is one scalar, and the values
  are bit-for-bit unchanged. The physics sweep ran *downstream* of that commit,
  so its committed ms/iteration is the before column of the fix's own
  measurement, on the same card and the same dials;
* **two more candidates, ranked and not implemented**: sizing the projected
  solve by the live basis (`cegterg`'s `nbase`; ~7% of a step at sixteen atoms,
  aimed at the ~80% the subspace algebra is of a 64-atom solve) and expanding by
  `notcnv` rather than `nbnd` (measured premise: the live roots fall 20 → 13 →
  10 → 0 in the *seeded* regime and never fall at all from a cold start).
  **`lax.switch` is the same trap as `lax.cond`** — every branch runs under
  `vmap` — so both are unbatched-path changes, which is where the large single-k
  cells are;
* **the profile itself is the deliverable that outlives the numbers**: the CPU
  half can be re-run on this workstation whenever a solver change lands, and the
  GPU half is one job.

Numbers in `PERFORMANCE.md`, "The guard was inside the k batch" and "Inside a
Davidson step".

---

**What.** A per-op profile of one SCF iteration on the GPU, on `si8-1k` and `si16-1k`.

**Why those cells.** The standing rule, and it has already caught three wrong conclusions:
a two-atom cell is dominated by fixed overheads and will not show any of this.

**What to look for**, named in advance so the profile is read rather than admired:

* **NumPy arrays passed as runtime arguments to jitted functions.** These are a host-to-
  device copy per call and are invisible on CPU. They are *not* silently wrong — NumPy on a
  traced value raises `TracerArrayConversionError` loudly — so what is left is host-side by
  construction and the question is only how much of it crosses the bus per iteration.
  Several compute modules are roughly half `np.` by line count; that is a profiling target,
  not an audit item.
* **The FFT**, which feeds Phase 2.
* **Kernel launch overhead per k-point chunk**, which is what decides whether `k_batch`'s
  default should differ by platform.
* **The augmentation-charge scatter** for ultrasoft/PAW, the other irregular access pattern
  in the code.
* **The Davidson subspace algebra** — the small dense `eigh` and matmuls inside a
  `lax.while_loop`. `batching.py` measures it at about **a third of a Davidson step** and
  names it as the part that width-one batching punishes; small dense operations inside a
  device loop are a classic accelerator pathology, and this one is already known to be
  large.

**Deliverable.** A `PERFORMANCE.md` section with the GPU-vs-CPU ratio per stage, under
§2.3's metric.

**Testable without a GPU:** partly. The same profile on CPU is worth having as the
comparison baseline and can be produced today with `tools/benchmark.py`.

### Phase 2 — the FFT backend becomes a dial

**What.** `sticks` and a fused 3D `box` transform behind a name registry, selected per
platform, defaulting to whichever the measurement says.

**Why a registry rather than a branch.** It is the project's own pattern for every
pluggable piece — XC functionals, mixers, eigensolvers, smearing, DOS schemes — and the
reason is that a platform-dependent default must not become a growing `if` in the hot path.

**The check is an equivalence with a derived tolerance, not a bit-equality — and the
tempting precedent is a false analogy.** P27 could demand a bit-for-bit identical density
with and without the dispersion correction because that term never enters `v_of_rho`: it is
the *same code path* plus an additive constant outside it. Two FFT decompositions are not
that. A stick transform (`z`, then `xy`) and a fused 3D transform sum in different orders
and will differ at the ulp level; `cuFFT` against `pocketfft` more so. Demanding bit
equality here would either fail spuriously or get quietly relaxed to "close", which is the
outcome this file forbids elsewhere ("Not 'close'", Phase 0). So: **state a tolerance and
derive its floor** from the round-off of an `O(N log N)` summation at the grid size in use,
and treat a departure beyond it as the bug.

**Testable without a GPU:** the *same-platform* comparison, yes, once the second backend
exists. The equality that actually matters — one backend, CPU against GPU — waits for
Phase 0, and it is unverified whether the two CPU backends come out bit-identical or merely
within tolerance.

### Phase 3 — single precision as a performance mode

**What.** `Precision(SINGLE)` end to end, with named reductions kept in float64.

**The policy already exists and the tests do not.** `PLAN.md` §5 says single precision must
stay viable, that it is never the mode a correctness claim is made in, and that "from P5 on
... a small set [of regression tests] also runs in float32 and asserts only the looser
tolerance, so the single-precision path cannot silently rot." **That tier was never built.**
What exists is unit coverage of the `Precision` object itself (`tests/unit/test_config.py`,
`test_cell.py`) — nothing runs an SCF in float32. So Phase 3 begins by building that tier,
not by tuning.

**Which reductions keep float64**, per §5's own instruction to decide this per site and
record it in a comment:

* the density accumulation over bands and k-points;
* the subspace overlap matrices in the Davidson solver;
* and — this file's addition — anything a *response* contracts, because P26 and P35 both
  found that a first-order quantity's error arrives at third order amplified rather than
  attenuated.

**Reduced-precision matmuls are a second, separate thing, and nothing here sets them.**
On NVIDIA hardware a float32 matmul defaults to a reduced-precision path unless
`jax_default_matmul_precision` is pinned, and there is **no such setting anywhere in this
package** (checked). So a float32 tier validated on this CPU does *not* certify float32 on
a GPU, and the operations it would silently affect are exactly the subspace overlaps this
phase already singles out for float64 treatment. Pin it explicitly and say which setting a
number was produced under.

**Testable without a GPU: the tier, yes; the certification, no** — the reduced-precision
matmul path has no CPU equivalent to exercise.

**And this phase's *rank* is unearned until Phase 0 reports the fp64 rate.** Two things are
being conflated whenever float32 is called a priority. Building the missing test tier is
owed by `PLAN.md` §5 whatever hardware exists, and belongs in §4's do-today list on that
ground alone. Phase 3 *as a GPU performance phase* is worth its position only if fp64 is
slow on the card in question: if the GPU partitions are data-centre parts running fp64 at
roughly half of fp32, float32's ceiling is about 2x and this phase belongs **after** Phase
4, whose payoff `PERFORMANCE.md` already measured to be the only parallelism worth having.
That is a datum, not a judgement, and Phase 0 check 1 returns it — this file declines to
guess it (§2.1), but the ordering here is provisional on it and should be revisited rather
than inherited.

### Phase 4 — sharding the k axis, and more than one GPU

**What.** `jax.sharding` with a mesh over the k axis — `PERFORMANCE.md`'s backlog item 1,
and measured there to be *the only parallelism worth having on CPU*: the thread pool gives
15% between one core and four and loses beyond that, while independent k-points are
independent.

**Why it is last and not first.** It is the largest change and the one whose payoff depends
on everything above it. It is also the one whose *correctness* is cheapest to establish.

**Testable without a GPU: yes, and this is the point worth carrying.**
`XLA_FLAGS=--xla_force_host_platform_device_count=N` exposes CPU cores as devices, so the
sharding logic — the mesh, the partition spec, the collectives in `sum_band` — can be
written and validated here, against the rule that already governs `k_batch`: the number of
devices must not be visible in any result beyond round-off.

**Two things that emulation does not reach**, so that "only the speedup needs hardware" is
not read too broadly. Forced host devices share one heap, so per-device memory limits and
the cost of inter-device transfers are unexercised — and §2.4 says memory is the binding
constraint, so that is the half that matters. And *more than one GPU* raises a question of
process topology that changes the code rather than its speed: one process holding several
local devices, or `jax.distributed` across Slurm tasks. Decide that before writing the
mesh, not after.

### Phase 5 — the response path, which is the reason JAX was chosen at all ✅ RUN, BOTH HALVES

**The CPU half is measured (2026-08-26)** and it answers §4 item 3 — the tape
per property, which is the number that decides the phase. `tools/gpu/phase5.py`
runs one converged SCF and one response property per process (peak RSS is a
high-water mark, so two properties in one process report the larger twice); the
table is in `PERFORMANCE.md` under "What each response property's working set
is". What it returned:

* **the mode decides the tape, not the property.** A *forward* response — the
  Sternheimer solves behind `epsilon` and `Z*` — costs **0.7-1.0 GB** over its
  own SCF, and a **forward-over-reverse** one — a `jvp` of a gradient, which
  tapes the inner reverse pass — costs **1.0-2.1 GB**. This file predicted the
  first and explicitly declined to assert the second; it is the same order, not
  the other one. **The dynamical matrix and the Raman tensors are not memory
  problems**, and the spread inside that range is the *property* rather than the
  k-count — the same third derivative costs 2.13 GB on 18 k-points and 2.01 on
  64;
* **the reverse stress still is**: **10.49 GB** on eight-atom ultrasoft silicon,
  +9.53 over its own SCF, measured again here by a different driver than the one
  that recorded 11.1 GB for P11 — **ten times the largest response tape** in the
  table. It is the one row with a fix already in `PERFORMANCE.md`'s backlog
  (item 8) rather than an open question;
* **an ultrasoft response is not worse.** `si-epsilon-us` has 3.4x the `ngm` of
  the norm-conserving cell and costs *less* over its SCF (+0.66 against +0.79),
  because a forward response does not tape the setup at all — the augmentation
  charge is paid for in the SCF's own working set;
* **every value reproduces its committed number** — 13.806646, -0.0757150,
  510.1023 — which is what says the driver measures the calculation the tests
  validate rather than a cheaper one.

**The GPU half ran the same day** — one H200 (`gpu63`) against four EPYC Milan
cores, commit `e562427`, both halves through `phase5.py` twice per property.
`PERFORMANCE.md`, "The response path on a GPU". Four results, and the first is
the one this file would not have guessed:

* **the mode decides the speedup exactly as it decided the tape.** A Sternheimer
  solve gets **1.0x** from an H200 — 1.0 on `epsilon`, 1.0 on the dynamical
  matrix, 1.2-1.4 on the other two — while a reverse-mode gradient gets **24x**
  on a small stress and **339x** on eight-atom ultrasoft silicon. A projected CG
  over occupied bands is Phase 1's small-dense-algebra pathology arriving in the
  response; a gradient through the radial transforms is one enormous dense graph;
* **339x is the warm number and nothing pays it.** A run calls `stress()` once,
  and cold the GPU takes **26.20 s against the CPU's 10.71** — it spends 26 s
  compiling a graph it runs in 9 ms, so on the calculation people actually do it
  **loses by 2.4x**. Quote the cold column for a one-shot property;
* **the card holds a quarter of what the host figure suggested**: the 10.56 GB
  reverse tape is **2.73 GB of HBM**, and every response property is under 0.25
  GB. Against 141 GB, feasibility is answered by two orders of magnitude;
* **three rows are not bit-reproducible on the device** and none is on the CPU —
  check 5's atomics hazard, absent from the SCF in Phase 0, present in the
  response at 1e-13 to 1e-16 relative. Round-off rather than a defect, but it
  means §4a's regression set must compare a response **to a stated tolerance**
  rather than diffing bytes.

**Two things this run did not settle.** `alas-raman raman` died with a
`MemoryError` on the CPU node where 160 G was granted and this workstation runs
the same case in 2.59 GB — unexplained, and the diagnostic job is not run. And
every response case here is a **two-atom cell**, which the standing rule says
shows none of this: the 1.0-1.4x is a fact about these cells, not a ceiling.

**The ten-atom half is measured on a CPU, 2026-08-26, and the GPU job for it is
proposed.** `si10-epsilon.in` — ten silicon atoms, sixteen k-points, `nosym`,
already validated against the vendored `ph.x` to 7.0e-6 — costs **245 s** for
the dielectric constant and **302 s** for the Born charges on four pinned cores,
against 12.5 s and 13.0 s for the two-atom cell. That is the twentyfold larger
response the caveat above asked for, and its **tape does not grow with it**:
+1.30 and +1.58 GB over its own SCF against +0.79 and +0.85 on two atoms, which
is a forward response taping nothing, holding at a size that could have broken
it. `phase5-si10-gpu.sbatch` and its CPU pair are the job; the second one also
carries the `alas-raman` diagnostic, with `JAX_TRACEBACK_FILTERING=off` and the
same 160 G the failure happened under.

**This is the phase that was missing from the first draft of this file, and its absence was
a category error rather than an omission.** `CLAUDE.md`'s "Why JAX" lists autodiff response
as reason *one* and GPU as reason *two*. Everything above is about the SCF. A GPU roadmap
for *this* project that stops at the ground state has scoped out P24 through P28, P35 and
P36 — the dielectric tensor, the Born charges, the phonons, the elastic constants, the
Raman tensors — which are most of what the code has been for since P24.

**And the question there is feasibility before speed**, which is why it cannot simply be
appended to Phase 1:

* **The reverse-mode tape is the largest allocation in the code** — 11 GB on eight-atom
  ultrasoft silicon for the stress (§2.4), and it is the reverse pass alone. On a card with
  a fraction of this workstation's memory, that decides whether the calculation exists, not
  how fast it is. The forward-mode paths (`jvp`) that P25, P26 and P35 are built from are
  much better behaved and that asymmetry should be stated per property rather than assumed
  either way.
* **The linear solves are the run.** `PERFORMANCE.md` puts the Sternheimer stage at **96%**
  of a phonon. Whatever the SCF's profile says, the response's is a different one and it is
  dominated by a projected conjugate gradient — which is a `lax.while_loop` on device, so
  the structure is right, and whose per-perturbation host sync (`sternheimer.py:475`) is on
  the same affordable budget as the SCF's convergence test rather than being per CG step.
  That is worth checking rather than assuming, but the code as written does not look like
  the pathological case.
* **The third derivatives compose tangents**, so any precision decision taken in Phase 3
  arrives at third order amplified. P26 and P35 both found first-order error behaving that
  way.

**Testable without a GPU:** the tape's size, yes — it is measurable here today, per
property, and it is the number that decides the phase.

### Running through all of it: `donate_argnums`

Not a phase, and **not something to defer behind Phase 1's profile** — the first draft of
this file did defer it, which contradicted its own ranking. §2.4 calls memory the binding
constraint; `PLAN.md` §5 already names the buffers ("use `donate_argnums` for the large
wavefunction and density buffers"); those buffers are already known to be the ones that
decide whether a cell fits; and donation is correctness-testable on this CPU today. It
belongs in §4's do-today list, with the working-set change measured into `PERFORMANCE.md`
rather than assumed.

---

## 4. What may be done today, without waiting

In the order they pay off if first contact slips:

1. **`donate_argnums` on the wavefunction and density buffers** — owed since P0, names its
   own targets, and is the cheapest memory win available.
2. **Phase 3's missing float32 test tier** — owed by `PLAN.md` §5 regardless of GPUs. (The
   *tier*; not Phase 3's rank, which waits on Phase 0's fp64 datum.)
3. ✅ **Phase 5's tape measurement**, per response property — *done 2026-08-26*,
   `tools/gpu/phase5.py` and `PERFORMANCE.md`'s "What each response property's working set
   is". It is the number that says whether the response path fits on a card at all, and it
   needed no card to obtain.
4. **Phase 4's sharding logic**, against forced host devices.
5. **Phase 2's backend equivalence test**, which needs only the second backend written.
6. ✅ **The CPU side of Phase 1's profile**, as the pinned baseline the GPU number is
   quoted against (§2.3) — *done 2026-08-26*, `tools/gpu/davidson_profile.py`, and it went
   one level further than this item asked: the sub-stage profile is what found the
   `cond`-under-`vmap` regression, which needed no card either.
7. **Sizing the subspace solve by the live basis, and the expansion by `notcnv`**
   (`PERFORMANCE.md`'s backlog items 2 and 3). Both are unbatched-path changes, both are
   *implementable and correctness-testable today*, and both are worth measuring on a CPU
   before a card decides whether they are worth taking — but their payoff is a GPU one, so
   the order is: measure on the device first, then write.

## 4a. Keeping a GPU number true after the machine goes away

Every number produced on the cluster is produced on a machine that is not in the development
loop, so it starts rotting the moment it is written down — and §5's rule that no phase may
change a validated number cannot be enforced by a test suite that never sees a GPU. So the
cadence is stated rather than left to notice:

* **A GPU regression set**, named explicitly and small: the multi-k case with a committed
  reference, one ultrasoft or PAW case, and one response property. Not the whole suite —
  something rerunnable in a single short job.
* **Rerun it on two triggers**: a phase landing, and a JAX or CUDA-plugin version change.
  The second is the one that will be forgotten, and it is the one most likely to move a
  number without any commit here being responsible.
* **Record the plugin and driver versions beside every GPU number**, the way `PLAN.md` P34
  requires the git commit and environment hash beside every cluster result — and for the
  same reason: version-mixed results are the failure that does not announce itself.

## 5. What must not happen

* **No phase may change a validated number.** The same rule `PERFORMANCE.md`'s backlog
  carries, and it binds harder here because a platform change is exactly the kind of thing
  that moves a result for a reason nobody looks for.
* **No GPU timing quoted against single-core QE** (§2.3).
* **No default flipped on a CPU measurement.** If a choice is platform-dependent, it is a
  dial with a per-platform default and both settings tested — not a rewrite.
* **No speedup claimed in float32 and correctness claimed in float64 in the same sentence.**
  They are different runs and the table says so.
