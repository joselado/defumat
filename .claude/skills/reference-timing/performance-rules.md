# Performance, in full

*Moved out of `CLAUDE.md` verbatim on 2026-10-08.*


**The measurement is single-core defumat against single-core Quantum ESPRESSO on the
same machine and the same input.** That comparison is the starting point of any
performance discussion, not a summary of one:

```bash
python3 tools/compare_qe.py benchmarks/si-1k.in --repeats 5
```

It needs `pw.x` built serially once (`./configure --disable-parallel --disable-openmp &&
make -j pw` inside the vendored tree; the binary is gitignored along with the rest of it).
The tool pins both codes to one core — JAX otherwise uses every core and the comparison
flatters it by the core count — and reads QE's own timing report, so the numbers on the
QE side are QE's, not a stopwatch around it.

The benchmark inputs live in `benchmarks/`, and are **single k-point** on purpose: both
codes parallelise over k, so a multi-k comparison measures batching rather than the cost
of the physics. `si-1k.in` is the test suite's silicon at `ecutwfc = 12`; `si-1k-ecut40.in`
is the same cell at a production cutoff, where scaling starts to show.
`performance/run_performance.py` runs the comparison over a whole set of inputs and
typesets it (`performance/README.md`). `PERFORMANCE.md` is the running log: the comparison,
where the time goes, what each change was worth, and the backlog. **Add a measurement to it
whenever a feature lands or a hot spot moves** — including the QE ratio, not only an
internal timing. `tools/benchmark.py <input>` gives the component breakdown.

**Every feature taken from Quantum ESPRESSO or from Elk is timed against the code it
was taken from, and the pair goes in `PERFORMANCE.md`.** Not the ratio to a previous
version of this code, not an absolute number on its own: the reference implementation's
wall clock beside ours, on the same machine and the same physics, one core each
(`OMP_NUM_THREADS=1` for both; the affinity mask set before JAX is imported, the mechanism
`tools/compare_qe.py` documents). The reason is that **the absolute number is the one worth
having and it is the one nobody measures** — an internal timing says a feature costs 94 s
without saying whether that is what the physics costs or what this implementation costs,
and only the other code answers that.

Two things to state rather than discover, because a comparison against an all-electron
code is never like-for-like and a misleading ratio is worse than no ratio:

- **Say what is not comparable.** LAPW's basis is not a plane-wave sphere; write that
  beside the number.
- **Time the same work, not the same task number.** Codes split a calculation into
  post-processing steps differently — Elk's `dielectric` (task 121) reads momentum matrix
  elements off a file that `writepmat` (task 120) produced, so timing 121 alone against a
  `defumat` call that *builds* `dH/dk` compares a contraction with a contraction plus its
  operator. Add the steps up until both sides start from the same place, usually a
  converged ground state, and say which steps were added.

**Two rules about *this* code's own clock, both of which have produced a confident wrong
number here.** They are the internal counterpart of the `ph.x` warm-scratch artefact.

- **Never time a first call.** It includes compilation, and compilation here is served by
  the on-disk kernel cache (`~/.cache/defumat/jax`, 4 GB and growing), so a first call
  times *whether the cache had that executable* — not the work. A first-call
  `get_stress()` read 8.68 s before a change and 16.09 s after, reproducibly; compiled,
  the two are the same to 0.5 per cent. Call it once to pay the compilation and time the
  next one. The same effect makes the test gate read 7 or 11 minutes for the same suite.
- **A memory figure must say what the cache held, and it points the *other* way.** The
  same rule applies to `%M`, with the sign reversed: a cache **miss is cheaper in memory
  and dearer in time**, because loading a compiled executable expands it. On one spinor
  PAW test, same code, alternating: **10.1 GB with `DEFUMAT_CACHE_DIR=off` against 16.4 GB
  with the cache on**, 57 s against 37 s — a 603 MB cache entry worth 6.3 GB resident
  (`OPEN.md` Part I item 2). The corollary is sharper than the rule: **a bisection across
  commits is a bisection across cache states**, because every new code state is a miss.
  Four commits walked in order read 10.2 / 10.1 / 16.4 / 16.4 GB and looked exactly like a
  regression that did not exist — the first commit re-run warm gives 16.4 too.
- **On a quantity whose expected change is zero, take a median and say how many
  samples.** A best-of-three over samples 2.16/2.21/2.20 against 2.06/2.21/2.27 reads as a
  confident 8 per cent in whichever direction got the lucky draw; the medians are equal to
  0.05 per cent. Best-of-N is right for a change expected to *move* the number and wrong
  for a check that it did not.
