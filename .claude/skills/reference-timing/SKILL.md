---
name: reference-timing
description: Measure defumat against the code a feature was taken from - single-core pw.x or Elk on the same machine and the same physics - and record the pair in PERFORMANCE.md. Encodes the mistakes that have invalidated timings here - a measurement taken beside a test run, a multi-k benchmark that measures batching instead of physics, a two-atom cell that hides every real cost, and an Elk task compared against work it does not do. Use when a feature lands, when a hot spot moves, when asked how fast something is, and before claiming any speed result.
---

# Timing against the reference code

**The measurement is single-core defumat against single-core Quantum ESPRESSO or Elk, on
the same machine and the same input.** That comparison is the starting point of any
performance discussion, not a summary of one. A number without the reference code beside
it does not say whether the cost is what the physics costs or what this implementation
costs, and only the other code answers that.

## Before measuring anything

**Check the machine is quiet.** A timing taken next to a test run is not a timing - a
`projwfc.x` comparison measured beside a background suite read 70% slow and had to be
discarded and repeated. Check for a running pytest, another agent's work, or a second
measurement, and wait rather than measure.

```bash
ps -eo pid,pcpu,etime,comm --sort=-pcpu | head -15
```

If anything substantial is running, say so and stop. Do not measure and caveat it.

## The QE comparison

```bash
python3 tools/compare_qe.py benchmarks/si-1k.in --repeats 5
```

This is the tool; use it rather than a stopwatch around either code. It pins both to one
core - JAX otherwise takes every core and the comparison flatters it by the core count -
and it reads QE's own timing report, so the numbers on the QE side are QE's.

It needs `pw.x` built serially once, inside the vendored tree:
`./configure --disable-parallel --disable-openmp && make -j pw`. The binary is gitignored.

**The benchmark inputs are single k-point on purpose.** Both codes parallelise over k, so
a multi-k comparison measures batching rather than the cost of the physics.

**A two-atom cell will not show you anything.** Fixed overheads dominate it, and three
separate optimisations were invisible there and obvious on eight atoms. Benchmark on
`benchmarks/si8-1k*.in` or `si16-1k*.in` and confirm a change helps *there* before
believing it. `si-1k.in` is the test suite's silicon at `ecutwfc = 12`;
`si-1k-ecut40.in` is the same cell at a production cutoff, where scaling starts to show.

For a component breakdown: `tools/benchmark.py <input>`.
For the whole set, typeset: `performance/run_performance.py` (see `performance/README.md`).

## The Elk comparison

Elk is an all-electron LAPW code, so the comparison is never like-for-like. Two things
must be **stated beside the number**, not discovered by a reader:

- **Say what is not comparable.** LAPW's basis is not a plane-wave sphere. Write that down.
- **Time the same work, not the same task number.** Codes split post-processing
  differently. Elk's `dielectric` task reads momentum matrix elements off a file that a
  *previous* task produced, so timing it alone against a defumat call that **builds**
  `dH/dk` compares a contraction against a contraction plus its operator. Add the steps up
  until both sides start from the same place - usually a converged ground state - and say
  in the entry which steps were added.

Both sides get one core: `OMP_NUM_THREADS=1` for each, and JAX's affinity mask set
**before JAX is imported** (`tools/compare_qe.py` documents the mechanism).

## Memory is half the measurement

A design is not finished until its peak working set is known - a plane-wave code is
memory-bound as often as it is compute-bound, and what decides whether a calculation runs
at all is usually a working set rather than a flop count. Alongside the wall clock, give
the peak in terms of `nk`, `nbnd`, `npwx`, `npol` and the FFT grid, and put it against the
RAM of a real machine.

Run anything long through `tools/run_regression.sh`, which caps each file's memory in its
own cgroup scope and writes a durable summary line carrying that file's **peak RSS**. That
peak is what makes the cap self-calibrating, so record it.

## Writing it down

`PERFORMANCE.md` is the running log: the comparison, where the time goes, what each change
was worth, and the backlog. An entry carries:

- **The pair**: the reference code's wall clock and ours, one core each, same input.
- **The ratio to the reference code** - not only an internal timing, and not only a ratio
  to a previous version of this code.
- **The machine and the date.** Development here is CPU-only; a GPU number is a different
  regime and says so.
- **What was not comparable**, and **which steps were added up**, for an Elk pair.
- **Any deviation from QE's memory strategy that this feature makes**, named and measured.
  Trading memory for batching is allowed and is sometimes right; the rule is that the trade
  is stated, measured, and made selectable when it is large, never arrived at by accident.

Where a quantity has **no counterpart in either code**, the entry says that explicitly. An
absent timing with no explanation is indistinguishable from a forgotten one.

## Two knobs that are not what they look like

- **`ulimit -v` is the wrong cap.** It bounds virtual address space, and XLA reserves
  arenas far larger than it ever resides in - a 16 GB cap failed tests that need a couple
  of GB. Use the cgroup (`DEFUMAT_TEST_MEM_MAX`, through `tools/run_regression.sh`), which
  bounds resident memory and lets a 4 GB cap run work with a 1.0 GB peak.
- **`XLA_PYTHON_CLIENT_MEM_FRACTION` is a GPU knob.** It sizes the PJRT device allocator's
  pool. On CPU there is no such pool and nothing to bound.
