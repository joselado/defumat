# Phase 0: first contact with a GPU

`GPU.md` is the roadmap and `PLAN.md` P10 the phase entry; this directory is the
thin path Phase 0 asks for — **one job, run by hand, not gated on P34 being
built** — and nothing more. Read `GPU.md` §2 first: it is what decides how these
scripts are shaped, and above all that *nothing here is a port*. JAX already
emits GPU code from this source. What is missing is evidence.

| file | what it is |
|---|---|
| `phase0.py` | the driver: one case, one setting of each dial, GPU.md's five checks, one JSON |
| `phase0-gpu.sbatch` | the GPU job |
| `phase0-cpu.sbatch` | **the same driver on a CPU node** — §2.3's baseline, and a separate job because CPU-only work does not belong on a GPU partition |
| `phase0_compare.py` | pairs the two sets of JSON and reads the checks off them |

## The cluster's rules are not this project's

`docs/cluster.local.md` (untracked, beside `CLAUDE.md`) carries the binding
policy and it is Aalto SciComp's. Two of its rules govern everything here:
**`sbatch` is proposed to the user rather than run**, and **every `#SBATCH`
value is looked up rather than remembered** — each one in the two job scripts
carries a comment saying which constraint it came from.

## The environment, which is the real cost of the phase

Do it interactively on the login node, not through the queue: iterating on a
broken environment one queue round-trip at a time is the slowest possible way.

**The obvious plan does not work, and `pip install --dry-run` is what says so.**
`scicomp-python-env/2025.2` carries a complete stack — jax 0.7.1 with a working
`jax-cuda12-plugin`, numpy, scipy, numba — and everything but `equinox`, so a
`--system-site-packages` venv adding one package looks like the cheap answer.
It is not: **equinox blocklists that jax by version** (`jax!=0.7.0,!=0.7.1`), so
pip resolves to jax 0.11.1 and installs `jaxlib` 0.11.1 into the venv while
leaving the module's `jax-cuda12-plugin` **0.7.1** visible behind it. That is a
version-mismatched CUDA plugin, and it is the kind of breakage that surfaces as
a confusing runtime failure on a GPU node half an hour into a queue rather than
as an error at install time. Run the dry run; read what it would install.

So the environment is a **self-contained venv**, and no module is loaded by
either job script:

```bash
module load scicomp-python-env/2025.2     # only to get a python3 to build from
W=/scratch/work/ladovj1/calculations/pypresso-gpu
python3 -m venv "$W/venv"                 # NOT --system-site-packages
"$W/venv/bin/pip" install --no-cache-dir "jax[cuda12]==0.11.1" equinox numba scipy pytest
```

That lands jax / jaxlib / `jax-cuda12-pjrt` / `jax-cuda12-plugin` all at
**0.11.1**, one consistent set, with equinox 0.13.8 — the workstation's exact
version — beside jax 0.11.0 there. It costs **5.7 GB and 12k inodes**, which is
the price of not having the overlay; inodes are the quota that bites first on a
shared filesystem, and 12k against a 1048k limit is affordable where a careless
install is not.

`pypresso` itself is **not** pip-installed into the venv. The job scripts put
the checkout on `PYTHONPATH` instead, so the code that runs is the working tree
at the commit the job logs, and there is no second copy to drift.

**The gate is a login-node smoke test before the first `sbatch`:**

```bash
export PYTHONPATH=/scratch/work/ladovj1/apps/pypresso JAX_PLATFORMS=cpu
taskset -c 0-3 "$W/venv/bin/python3" tools/gpu/phase0.py si-1k --k-batch 1 --band-batch 1
taskset -c 0-3 "$W/venv/bin/python3" -m pytest -m unit -q
```

`taskset` because a login node is shared and pypresso otherwise takes four cores
on import. Measured 2026-08-25: the cluster reproduces the workstation's
`si-1k` total energy **bit for bit** at `-15.2544487130 Ry`, across jax 0.11.0
and 0.11.1 on different hardware.

## Both dials are set explicitly, and there is no default

`phase0.py` **requires** `--k-batch` and `--band-batch`. `batching.py` reads
them from the environment at import time and defaults both to one, which is
QE's loop and what a *cache* wants; a GPU has no such cache and inverts the
conclusion. A GPU run left on the defaults serialises every FFT into a per-band
kernel launch — close to the worst execution mode available — so the setting is
part of the job, not a tuning afterthought.

That also means **one process per dial setting**: the dials cannot be changed
after import, so each row of the job script is its own `python3` invocation.

## The cases, and why the benchmarks alone are the wrong first inputs

`benchmarks/` is **single k-point on purpose** — both codes parallelise over k,
so a multi-k CPU comparison would measure batching rather than physics. That
makes those files exactly the wrong place to exercise `k_batch`, which is a
no-op at `nk = 1`. So the job runs:

* `si-1k` — two atoms, one k-point, debuggable; proves the thing runs;
* `al10-metal` — **ten atoms and ten k-points**, `marzari-vanderbilt` smearing,
  and a committed QE reference (`tests/data/qe/reference.out.al10-metal`) that
  the development machine reproduces to **1.9e-9 Ry**. This is the case on which
  either dial means anything, and it is run at three settings;
* `si8-1k` — the smallest cell whose cost is physics rather than fixed overhead
  (the standing rule: a two-atom cell shows none of this), at both band settings.

Measured on the development workstation, `al10-metal` gives **the identical total
energy to the last digit** at `k=1,b=1` and at `k=all,b=all` — the dial is a
round-off-level reordering and nothing else — while the fully batched mode costs
**2.2x** (3726 against 1726 ms per iteration). That is the CPU end of the trade,
and the GPU is expected to invert it. It is the number the GPU run is read
against.

## Reading the result

```bash
python3 tools/gpu/phase0_compare.py out/gpu-*.json --against out/cpu-*.json
```

The tolerance on check 2 is **the case's own `conv_thr`**, not a round number:
an SCF converged to `dr2 < conv_thr` does not define its total energy more
tightly than that.

Three things in the output are the phase's actual findings rather than
decoration:

* **`determinism`** — the same SCF twice in one process, compared bit for bit.
  `basis/fft.py` scatters plane waves into the box with an accumulating scatter
  over *deliberately duplicated* indices, and XLA may lower that through atomics
  whose summation order is not reproducible. Whether it does here is unverified.
  If the two runs differ, the determinism policy is settled **before** any GPU
  number is called validated.
* **`fp64 slowdown`** — measured on a matmul and a batched 3D FFT, which is what
  the Davidson subspace algebra and `h_psi` are actually made of. `GPU.md`
  Phase 3's *rank* waits on this datum: if fp64 runs at about half of fp32,
  float32's ceiling is roughly 2x and that phase belongs after Phase 4.
* **`peak GB`** — from JAX's own allocator accounting, never `nvidia-smi`, which
  reports the preallocated pool. Reading the pool would give a *wrong number
  rather than a failed check*, which is the worse of the two outcomes.

## Check 5 has two forms and this runs both

*Inside* one process, the SCF runs twice and is compared bit for bit — that is
`phase0.py`'s `determinism` line. *Across* two jobs, the density and eigenvalue
fingerprints in the JSON are compared — that is `phase0_compare.py`'s `across`
column, and it catches the failure one process cannot: a compilation that is not
itself reproducible. The fingerprint is a **SHA-256** of the raw bytes and not
Python's `hash`, which is salted per process and would have compared fine inside
one run and meant nothing the moment it was written to a file.

Two different platforms are *not* expected to agree bit for bit — cuFFT and
pocketfft sum in different orders — so `across` is reported for a GPU/CPU pair
and asserted only when the two records come from the same platform. Rerunning
the GPU job and comparing it against itself is therefore a real check and costs
one extra job.
