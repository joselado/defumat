# `performance/`

A report you can regenerate by hand, comparing this code against the Fortran it
reimplements:

```bash
python3 performance/run_performance.py
```

That runs the standard set of benchmark inputs through **both** codes, writes
`performance.tex`, compiles `performance.pdf`, and leaves the raw numbers in
`performance.json`.

```bash
python3 performance/run_performance.py --cases si8-1k pt-so-1k   # a subset
python3 performance/run_performance.py --all                     # everything in benchmarks/
python3 performance/run_performance.py --repeats 3               # best of three
python3 performance/run_performance.py --max-seconds 300         # a longer per-case budget
python3 performance/run_performance.py --from-json               # re-typeset, no re-measuring
```

## What it measures, and why that way

The project's metric is **single-core defumat against single-core Quantum
ESPRESSO, same machine, same input, same `conv_thr`** — see `PERFORMANCE.md` for
the running commentary and `tools/compare_qe.py` for the measurement itself,
which this script drives and typesets.

- **Both codes are pinned to one core** by affinity mask. XLA sizes its thread
  pool from that mask and ignores `OMP_NUM_THREADS`, so without the pinning the
  comparison flatters this code by the core count.
- **QE's times are QE's own**, read from the timing report it prints
  (`init_run`, `electrons`), not a stopwatch wrapped around the process.
- **Cold and warm are both reported.** The cold SCF includes XLA compiling every
  kernel, which is what a user meets on a short run; the warm one is the same
  loop with the kernels compiled, which is what predicts a long run or a bigger
  system. QE has no equivalent — there is nothing to compile.
- **Each case runs in its own process, with a timeout** (`--max-seconds`,
  default 120). A case that cannot finish inside its budget is reported as such
  instead of making the whole run long, and one case's compilation cache and
  memory high-water mark cannot colour the next. Peak RSS is reported per case,
  because memory is part of the design here.

The benchmark inputs are **single k-point on purpose**: both codes parallelise
over k, so a multi-k comparison measures batching rather than the cost of the
physics. How many k-points this code holds at once is a separate axis with its
own measurement — `defumat/batching.py`, and the k-axis section of
`PERFORMANCE.md`.

## Requirements

- `pw.x` built serially, once:
  `cd quantum_espresso/qe-7.5-ReleasePack/qe-7.5 && ./configure --disable-parallel --disable-openmp && make -j pw`
  (override the location with `$PW_X`).
- `pdflatex` with `booktabs` and `pgfplots`. Without it the `.tex` is still
  written; pass `--no-pdf` to skip the compile deliberately.

The generated `performance.tex`, `.pdf` and `.json` are **not committed** — they
describe one machine on one day. Keep a copy yourself if you want to diff two
runs, or re-run on the machine that matters.

## The systematic sweep — `sweep.py`, and the two sets

`run_performance.py` above is the typeset CPU-only report over a hand-kept case
list. `performance/sweep.py` is the same measurement made **systematic**: three
legs, two named sets, one record per case per leg, and the same code path on a
laptop and as one task of a Slurm array.

```bash
tools/run_benchmark.sh                      # the fast set, on this machine
tools/run_benchmark.sh complete --resume    # the whole thing, restartable
python3 performance/sweep.py --report performance/results/complete --set complete
python3 performance/sets.py complete --describe    # what is in a set, and why
```

**Three legs, and two of them are a different claim from the third.**

| leg | what runs | what it is compared against |
|---|---|---|
| `qe` | `pw.x`, one core, timed by its own report | — it is the baseline |
| `cpu` | defumat, one core, both batching dials at QE's end | `qe`, per SCF iteration — *the project's metric* |
| `gpu` | defumat, one GPU, both dials batched | `cpu`, per SCF iteration — `GPU.md` §2.3's metric |

The report prints them as two tables and never mixes them. A GPU number put
against single-core QE would be meaningless in the same way an unpinned CPU
number is, only more so, so table B is against the `cpu` leg and its caption
says so. Compile time is its own column there rather than amortised into the
speedup.

**The two sets** are in `performance/sets.py`, which is the single place the
list is written down — the laptop run and the cluster array read the same
module, and the sbatch generator bakes it into a bash array from there rather
than from a retyped copy.

- **`fast`** — 10 cases, one per kind of physics, all single-k and 2–8 atoms.
  The "did anything move?" set, sized to finish while someone waits.
- **`complete`** — 25 cases: `fast`, plus the 8/16/32/64-atom size ladder, plus
  nine ten-atom cells spanning the pseudopotential kinds, a GGA, a metal,
  collinear and noncollinear magnetism, spin-orbit coupling and DFT+U, plus
  three larger cells. Size and physics vary on separate axes so a ratio that
  moves with one can be told from a ratio that moves with the other.

**The kernel cache is off by default** (`--cache`), and that is not incidental:
with `~/.cache/defumat/jax` live, a "cold" run is reading an earlier run's
compilation and the cold/warm split stops being a measurement. On a cluster it
would be a *concurrent* task's compilation.

**A case that cannot fit this machine is refused before it starts.** `bi20-soc`
peaks at about **35 GB** and the workstation this was written on has 31, and an
out-of-memory kill here does not cost a case — it costs whatever else the
terminal was holding, which has happened at least three times. So `sets.py`
carries each large cell's recorded peak, the local runner compares it against
`MemTotal` with 20% headroom, and a case that does not fit is written out as a
*named refusal* rather than started and discovered. `--force-large` overrides
it; the cluster does not need it. The peaks are the GPU sweep's device figures
standing in for host peaks — the right order of magnitude, which is all a guard
needs — and a case with no recorded peak is not assumed to be small.

**Two cells need `max_iterations = 200`**, and this is recorded in `sets.py`
rather than left to the default: `ni10-ldau` and `h40-chain-lsda` converge in
151 and 104 iterations, so the usual cap of 100 reports them as *not converged*
instead of as capped. The first GPU sweep made exactly that mistake.

**One process per (case, leg)**, with a per-unit timeout — which bounds a case
two ways, keeps one case's compilation cache and memory high-water mark off the
next, and is what makes an array task a single unit of this same harness.
`--resume` skips units that already have a successful record, so an interrupted
sweep continues instead of restarting.

## Running the sweep on Triton

```bash
python3 tools/cluster/submit_benchmark.py --set complete
```

That writes three array scripts — one per leg, because `--partition` and
`--gpus` are properties of a job — and **prints the `sbatch` commands for you to
run**. It does not submit, it has no `--submit` flag, and it should not grow
one: the policy beside this checkout is explicit that nothing here submits,
cancels or polls on its own.

One array task is one case, which is the site's "group short tasks into an
array" rule, and each writes one JSON — one small file per unit, not per
iteration. The three legs are joined afterwards by case name, so a leg that
failed or is still queued costs its own column and nothing else.

**The `#SBATCH` values are inherited from `tools/gpu/sweep-qe.sbatch` and
`sweep-gpu.sbatch`**, which have run on this cluster, with their whole-loop time
budgets divided down to one case. Every one is a flag. Check the partition, the
time limit and the `quantum-espresso` module against
<https://scicomp.aalto.fi/triton/> before the first submission rather than
against the generator.

## `gpu-sweep.*` — the GPU report

A second, hand-run report beside the single-core one, and **a different metric**:
defumat on one GPU against Quantum ESPRESSO on one CPU core, across twelve
cases spanning the pseudopotential kinds, a GGA, a metal, collinear and
noncollinear magnetism, spin-orbit coupling and DFT+U, at 10-40 atoms.

```bash
python3 performance/plot_gpu_sweep.py     # figure from gpu-sweep.json
cd performance && pdflatex gpu-sweep.tex  # twice, for the reference
```

| file | what |
|---|---|
| `gpu-sweep.json` | the measurements, one record per case |
| `plot_gpu_sweep.py` | the figure: speedup by physics, and the agreement |
| `gpu-sweep-fig.pdf` / `.png` | that figure |
| `gpu-sweep.tex` / `.pdf` | the report |

**Nothing here re-measures.** The runs are cluster jobs (`tools/gpu/sweep-*.sbatch`)
because this machine has no GPU; `gpu-sweep.json` is what came back, and both the
figure and the report are typeset from it. To refresh it, rerun those jobs and
rebuild the JSON — do not edit the numbers in place.

**Read `PERFORMANCE.md`'s section of the same name before quoting a ratio.**
`GPU.md` §2.3 forbids this comparison by default: one core is the softest
baseline available, `pw.x` saturates by ~16 cores on a single-k cell, and the
two codes take different numbers of SCF iterations. The per-iteration column is
the code comparison; the total is time-to-answer.
