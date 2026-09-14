---
name: test-runs
description: Run defumat's test suite without getting OOM-killed, and know which group to run. Covers the two groups and where the `slow` line falls, what the push gate costs and why it drifts upward, the per-file cgroup cap and the RSS watchdog that names the test before the kernel acts, and the rule against running demanding suites side by side on this shared machine. Use before running pytest on anything beyond a single file, when deciding whether a new test belongs in the gate or the slow set, when a run was killed or a session died, when asked how long the suite takes or whether the machine is free, and when a memory figure or a timing needs to be taken beside one. Triggers - "run the tests", "run the slow suite", "run the gate", "test-fast", "run_regression", "why was it killed", "OOM", "is this test too slow", "how long does the suite take".
---

# Running defumat's tests

`CLAUDE.md` keeps the command block and one rule: **run anything long through
`tools/run_regression.sh`, and commit before starting it.** This is the rest -- which
group to run, what each costs, and the two bounds that turn an anonymous `SIGKILL`
into a test name.

Every number here is a measurement on this machine, and several were expensive to
get. Do not round them or restate them from memory.

## Which group, and where the line falls

**The suite is two groups and `slow` is the line.** `tools/test-fast.sh` is
`pytest -m "not slow"`: **2609 tests in 10m20s at a 5461 M peak** (measured
2026-09-14, warm cache, idle machine), and it is what runs before a push.

**The gate is held at that size on purpose, and it drifts upward on its own.**
Left alone it reached **2973 tests in 16m18s at 9319 M** by 2026-09-14 -- a
factor of two in time and 1.6x in peak against the figures below, entirely from
tests being added to it rather than from anything getting slower. What brought
it back is in `PERFORMANCE.md`; the rule that keeps it there is worth stating
here, because it is the decision anyone adding a test has to make:

* **A test above about five seconds that is not a direct number against `pw.x`,
  `projwfc.x` or Elk belongs in the slow set.** Identities, refusals, guards and
  internal-consistency checks are the gate's cheapest thing to lose, because the
  slow set still runs them. A reference comparison stays whatever it costs --
  that is what the gate is *for*.
* **Where a parametrised reference comparison has one expensive parameter, mark
  the parameter, not the test** (`pytest.param(..., marks=pytest.mark.slow)`).
  Platinum's PAW spinor was 2.3 GB and 29 s of the gate for the same
  `projwfc.x` comparison its ultrasoft twin makes for neither.
* **Check what a refusal test is paying for before moving it.** Two of them held
  4.5 GB between them, not because the check was expensive but because the
  *cell* was: they built a `Calculation` only so a guard could read one flag off
  it. Swapping a germanene slab for a one-atom hydrogen cell took 3.6 GB and 11 s
  off a test that still raises the byte-identical message.

**Two things to expect when trimming, both measured here.** Removing a test
returns roughly a *third* of its measured seconds, not all of them, because the
first test to reach a cell pays for compiling it and the next one inherits that
bill when it goes -- 99 s of marked tests bought 26 s of wall clock. And the
attribution to fix is the one from `getrusage`'s high-water mark: the process
peak is the sum of the per-test *gains* in it and nothing else, which is what
says two tests out of 2973 held half the peak.

**The suite runs on eight cores where the package runs on four, and that is a
deadlock fix rather than a speed choice.** `defumat` narrows the affinity mask
to four at import because that is fastest for the physics (238 ms per SCF
iteration on the eight-atom benchmark against 411 at eight cores); it is also
the mask that parks a long-lived process with every XLA worker blocked and no
CPU at all, at a rate that follows the mask and nothing else — 8 hangs in 8 runs
at two cores, 6 in 14 at four, none at eight or above. `tests/conftest.py`
therefore sets `DEFUMAT_THREADS=8` before anything imports `defumat`, which
costs the suite 11% of wall clock and, provisionally, 13% of peak RSS — the
memory half was measured in sequence rather than alternating, and peak RSS here
depends on the compiled-kernel cache much more than on the mask. An explicit
setting still wins. `OPEN.md` Part IV item 1 has the measurement and the two theories it
killed.

**The four readings below are the history of that drift**, and they are kept
because the spread inside them is not the test set either -- the same
command read 1892 tests in **7 minutes** at 4.5 GB on 2026-09-12, 1938 in
**11m17s** at 6043 M later on 2026-09-13, and **1941 in 7m22s** at 5903 M later
still the same day, on an idle machine with a **4.0 GB warm**
`~/.cache/defumat/jax`. Three more tests cannot cost minus four minutes, so what
separates the two 2026-09-13 runs is the compiled-kernel cache and whatever else
the machine was doing — which is the same rule the performance section states
about timings taken beside a test run, applied to the gate itself. **Time it
warm and idle, or do not compare it.** The thing genuinely worth watching is the
peak RSS, which moved 4.5 -> 6.0 -> 8.2 GB and did not come back. **Do not
attribute that to the wider mask**: the better candidate is the kernel cache
warming across sessions, so that more of the gate's executables are *loaded*
rather than compiled — and loading one 603 MB cache entry is worth 6.3 GB
resident (`OPEN.md` Part I item 2). The test is one run of
`DEFUMAT_CACHE_DIR=off tools/test-fast.sh`, and it has not been done. 8.2 GB
against a 12 G cap whose watchdog fires at 0.85 is less margin than it reads.

**A fourth figure, and the peak came back down: 1951 passed / 176 skipped in
9m32s at a watchdog-reported 7052 M**, warm cache, idle machine, 2026-09-13
(2127 selected of 3022 collected). The likeliest cause of the 8.2 -> 7.05 GB is
the commit that took a gigabyte of constants out of the force and stress
gradients — it cut one spinor test file's own peak from 16,383 M to 6,232 M — but
that is an inference from one number and the cache-off run is still the
experiment that would settle it. Whatever is in the
gate is paid on every push by someone who is not doing physics at the time,
which is the same argument the notebooks' ten-minute ceiling rests on; the lever
is the `slow` marker, and the question to ask of any test above a few seconds is
whether the gate is where it belongs. (Neither 2026-09-13 figure is the
augmentation remat: the only two gate files that take the rematted route run in
4.18 s together, and every other cell in the gate takes the stored one.) The slow set is ~1280 tests and **over two hours** — it runs when it is
asked for, not on every change. The split cuts across `unit` and `regression`
both, because it is about cost and not about kind: a cheap regression case
against a two-atom reference is in the gate, and an expensive unit test is not.

**The slow set is not optional, it is just not a gate.** Run it before a
release, or when it is explicitly asked for — **not before a push, and not
because a change touched the SCF, the eigensolver or the response stack.**
This project is in heavy development: pushes are frequent, the slow set is over
two hours, and waiting on it stalls the work far more often than it catches
something. Where a change wants more assurance than the gate gives, the cheap
and better instrument is a **targeted numeric check** — the same quantity
computed before and after on one small example that goes through the changed
path, compared to round-off. That is what identified and confirmed every one of
P74's five band-batching sites, and none of it needed the suite. It is two hours precisely
because it is the part that catches what the gate cannot, and the one time it
was run end to end it found **three phases' claims had drifted** — P29's stale
refusal list and its broken BFGS metric, P36's 8.7e-14 wedge agreement, and two
notebooks whose committed outputs no longer matched their code (`PLAN.md` P38).
`tools/run_regression.sh` exists for running it in pieces: one **memory-capped**
pytest invocation per file, a durable summary line each, and a file already in
the summary is skipped, so an interrupted run resumes instead of restarting.

## Sharing the machine, and watching a run

**Do not run demanding suites simultaneously — not in one process, and not in two at
once.** This machine has 39 GB (40906044 kB, read 2026-09-12; it was 30 GB when the
episodes below happened) and both mistakes have killed a session here:

- **Several slow files in one `pytest` invocation** is *one* process, so every file's XLA
  executables accumulate for the whole run — three spinor suites reached 2.4 GB in ninety
  seconds and kept climbing. Run them one at a time, through `tools/run_regression.sh`
  rather than a hand-written loop — it caps each file's memory as well as separating them,
  which is the next subsection.
- **Two test runs in parallel, or a test run beside anything being measured.** A timing
  taken next to a test run is not a timing — a `projwfc.x` comparison measured beside a
  background suite read 70% slow and had to be discarded and repeated.

One habit makes this cheap: write a **durable summary line per file** so a kill costs the
file in flight rather than the whole run, which is what `run_regression.sh` already does.
And **narrow the list before running it**: a `grep` for the inputs that can actually reach
the changed code path is minutes of work and routinely removes most of the suites, where
guessing adds them.

**Watch the memory while it runs, and be willing to stop.** Anything long enough to walk
away from is long enough to check on: read `free -g` between steps rather than only after
a kill, and treat a shrinking `available` column as a reason to act now. When it is
tightening, **kill your own subprocesses first and run what is left serially** — one file
per process, waiting for each to exit before starting the next. Serial is slower and it
finishes; parallel is faster until it takes the session with it. Two specifics, both paid
for here:

- **The default for a long run is one capped process per file, not one big invocation.**
  `tools/test-fast.sh` is the exception, and only on an otherwise idle machine — a single
  2000-test process climbs monotonically for six minutes because XLA never releases a
  compiled executable, and it was killed mid-run on 2026-09-12 for exactly that. Reach for
  `tools/run_regression.sh`, or a loop of `systemd-run --user --scope -p MemoryMax=...
  python3 -m pytest <one file>`, whose peak stays in the low hundreds of megabytes.
- **This machine is shared with other sessions, and memory is the contended resource
  rather than the cores.** A run here killed another session's suite on 2026-09-12. Before
  starting something long, check whether anything else is running; if a peer session says
  it is running one, either wait or cap yourself and say what you are doing. Killing your
  own work is always the right call over letting the kernel choose whose to kill.


### An out-of-memory kill must cost one file, and must name the test that caused it

**This was an open defect rather than a fact of the machine, and it is now bounded from
both sides** — a cgroup cap per file, and a watchdog that names the test before the kernel
acts (the last two subsections below say exactly what each covers). Everything above is a
way of staying under the ceiling; none of it puts a floor under what happens when
something goes over it anyway, and going over it here kills whatever else the terminal was
holding. It has happened at least three
times: `test_ten_site.py` (P28b) and `test_spinor_forces.py` (P46), both measured in
`PERFORMANCE.md`, and again on **2026-09-07** — that one by the user's account rather than
from a log — where it took a session down with a documentation restructuring uncommitted.
**What was in flight the third time is not recorded anywhere, and that is itself the
point** — an unbounded process that dies takes its account of what it was doing with it.

**The mechanism is a cgroup, and it works on this machine** (cgroup v2, user-slice
delegation, probed 2026-09-07). Name the unit rather than quieting it, so
`journalctl --user -u <unit>` can afterwards say the kill was `memory.oom` and not
something else:

```bash
systemd-run --user --unit=reg-<file> -p MemoryMax=12G -p MemorySwapMax=0 --scope \
    python3 -m pytest <file> -q
```

Two measurements, and the second is the one that matters: a plain Python allocation past a
512 MB limit is `SIGKILL`ed at it, exit 137, with the shell untouched; and **the eight
`test_scf.py` energy comparisons run to completion under `MemoryMax=4G`**, peak RSS 1.0 GB,
which is the same class of work that a 16 GB `ulimit -v` failed. That contrast *is* the
argument — XLA's address-space reservations are not charged to a cgroup, so a resident cap
can be set near what the work actually uses.

Two caps that look like this one and are not:

- **`ulimit -v` is not the cap to reach for** — it bounds *virtual* address space, and XLA
  reserves arenas far larger than it ever resides in, so a 16 GB cap fails tests that need
  a couple of GB (`PERFORMANCE.md` has the episode).
- **`XLA_PYTHON_CLIENT_MEM_FRACTION` is a GPU knob.** It sizes the PJRT *device*
  allocator's pool, which is what the 2026-09-04 H200 entry in `PERFORMANCE.md` reads
  against; development here is CPU-only, where there is no such pool and nothing to bound.
  The absence of a CPU equivalent inside JAX is exactly why this item is open.

**`tools/run_regression.sh` is where this is wired, and it is the way to run anything
long.** Each file goes into its own scope, a kill lands there rather than on the loop, and
the loop writes `killed (SIGKILL, cap=…)` as that file's durable summary line and starts
the next one — so a kill costs one file's *result*, which is what the per-file runner was
always for and what the kill taking the runner defeated. Three things it does that are not
obvious and are each a bug that was hit while writing it: the cap is `DEFUMAT_TEST_MEM_MAX`
(`off` for none, and a machine without cgroup delegation says so and runs uncapped); a file
the cap killed is **retried** on the next run rather than skipped, since the reason to
resume after a kill is that something changed; and the unit name carries the run's PID,
because a killed scope stays *loaded* and reusing the name fails with "already loaded",
which reads as a test failure. It also writes an `in-flight.log` line before starting a
file — **what was running is the thing a kill destroys**, and no cap can be trusted to
cover every way that happens.

**Every summary line carries that file's peak RSS**, which is what makes the default cap
self-calibrating: `12G` is a guess, and a file sitting at 11 GB is the next kill whether or
not it has happened yet. The two figures on record so far are `test_scf.py` at **1011 M**
and `test_dos.py` at **1297 M**, both of them slow files running clean, so the first full
pass through the runner is also the measurement that says what the cap should be.

**The named failure is wired in, and its edges are the thing to know.** A `psutil` RSS
watchdog lives in `tests/conftest.py`'s autouse fixture, with all of it in
`tests/memwatch.py`: it writes the running test's nodeid into the runner's own
`in-flight.log` *before* the test starts, samples this process's resident set once a
second in a daemon thread, logs a crossing the moment it happens, and fails that test by
name at teardown once its peak passes **0.85** of `DEFUMAT_TEST_MEM_MAX` — the same
variable and the same `off` sentinel as the cap above, and `run_regression.sh` now exports
the effective value and the log path so the child sees the number actually in force. **It
fails the crossing, not everything after it**: once a test has been named, a later one is
named only if it raises the peak by a further 2% of the cap, because XLA's executable
cache does not shrink and the first offender would otherwise fail every test behind it —
at a deliberately low 200 M cap on `tests/unit/test_config.py` that is 5 named errors
rather than 17. That
turns an anonymous `SIGKILL` into a test name, which is the difference between a lost
afternoon and a bug report. The fraction is the headroom: 15% of a 12 G cap is 1.8 GB and
of a 4 G one 600 MB, which is what a 1 s interval has to cover between two reads. **Unset
is inert** — no thread, no log file, no `psutil` import — so the gate pays nothing; to
watch the gate, which is one process by design and where this is the only form of the
protection that works, give it a cap: `DEFUMAT_TEST_MEM_MAX=12G tools/test-fast.sh`.

**Four things it does not catch, and they are why the log line comes first.** A single
allocation that goes from under the threshold to past the cap **between two samples** is
invisible to any sampler at any interval. The cgroup charges page cache and kernel memory
to the scope while `memory_info().rss` counts this process's anonymous pages, so the
kernel can kill *below* the threshold — the scope's own `memory.current` would match the
kill criterion exactly and is the follow-up, not this. Memory held by **child** processes
is charged to the scope and not counted here. And the failure lands at *teardown*, after
the peak: if the peak is the kill, what survives is the line in `in-flight.log`, not the
failure. So the standing advice does not change — run anything long through
`run_regression.sh`, and **commit before starting it**.
