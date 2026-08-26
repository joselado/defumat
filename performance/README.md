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

The project's metric is **single-core pypresso against single-core Quantum
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
own measurement — `pypresso/batching.py`, and the k-axis section of
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

## `gpu-sweep.*` — the GPU report

A second, hand-run report beside the single-core one, and **a different metric**:
pypresso on one GPU against Quantum ESPRESSO on one CPU core, across twelve
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
