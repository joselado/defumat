"""A resident-memory watchdog that fails a *named* test before the kernel acts.

Why this exists
---------------
A plane-wave run is memory-bound as often as it is compute-bound, and on this
machine a test file that goes over the ceiling used to take the whole session
with it (``PLAN.md`` P28b, P46, and a session on 2026-09-07 that left no record
of what it had been running). ``tools/run_regression.sh`` bounds the *cost* of
that: each file runs in a cgroup scope with ``MemoryMax``, so the kernel's
``SIGKILL`` lands on one file and the loop survives to write it down. What that
cannot do is say *which test* was holding the memory -- a ``SIGKILL`` destroys
everything the process knew, so the record has to be written before it lands.

This module is the missing half. It samples the process's own resident set in a
low-frequency daemon thread and, when a test's peak crosses a fraction of the
same cap the runner set, it (a) appends a line naming that test to the runner's
``in-flight.log`` *the moment the crossing happens*, and (b) fails that test at
teardown. The first is what survives a kill; the second is what turns an
anonymous exit 137 into a bug report.

Configuration
-------------
The cap is read from ``DEFUMAT_TEST_MEM_MAX``, which is the variable
``tools/run_regression.sh`` already uses, in the same systemd syntax (``12G``,
``512M``, ``50%``) with the same ``off`` sentinel. There is deliberately no
second variable: two caps that could disagree would be worse than none. The
runner exports the *effective* cap so a child sees the number actually in force,
including when the machine has no cgroup delegation and nothing but this
watchdog stands between the process and the machine's OOM killer.

**Unset means inert**, not observational: no thread, no log file, no ``psutil``
import, nothing measured. The reason is that the pre-push gate
(``tools/test-fast.sh``, 1634 tests in one process) pays for anything this
module does on every test, and a peak with no cap to compare it against answers
no question anybody asked. To watch the gate, give it a cap:
``DEFUMAT_TEST_MEM_MAX=12G tools/test-fast.sh``.

The fraction is 0.85, and the headroom is the justification: 15% of a 12 G cap
is 1.8 GB and of a 4 G cap is 600 MB, which is what a 1 s sample interval has to
cover between two reads. Tighter than that and an ordinary ultrasoft backward
pass trips it; looser and the sampler is behind the kernel.

What it cannot catch, stated because a watchdog that is trusted past its range
is worse than none:

* **A single allocation that jumps from under the threshold to past the cap
  between two samples.** No sampler catches that, at any interval.
* **A kill below the RSS threshold.** The cgroup charges page cache and kernel
  memory to the scope, not just anonymous RSS; reading the scope's own
  ``memory.current`` would match the kill criterion exactly and is a follow-up,
  not this.
* **Memory held by child processes**, which the cgroup charges and
  ``memory_info().rss`` does not.
* The failure lands at *teardown*, after the peak. If the peak is the kill, what
  survives is the log line, which is why it is written first.
"""

from __future__ import annotations

import datetime
import os
import threading
from dataclasses import dataclass
from pathlib import Path

#: Fraction of the cap at which a test is failed. See the module docstring.
DEFAULT_FRACTION = 0.85

#: Sampling period, seconds. Low frequency on purpose: this runs beside every
#: test in the gate, so it must cost nothing measurable.
DEFAULT_INTERVAL = 1.0

#: After a crossing has been reported, a later test is reported only if it sets
#: a new high this much of the cap above the last reported one. Without it, one
#: test going over makes every test after it fail too -- XLA's executable cache
#: does not shrink, so the resident set stays where the first offender left it.
DEFAULT_RESTEP = 0.02

#: The variable ``tools/run_regression.sh`` sets, reused rather than duplicated.
CAP_ENV = "DEFUMAT_TEST_MEM_MAX"

#: Where the runner keeps ``in-flight.log``; it exports this so the child writes
#: into the same file the shell loop is already appending to.
LOG_ENV = "DEFUMAT_TEST_INFLIGHT"

_MB = 1024 * 1024
_SUFFIXES = {"": 1, "B": 1, "K": 1024, "M": 1024**2, "G": 1024**3,
             "T": 1024**4, "P": 1024**5, "E": 1024**6}


def parse_mem_max(value: str | None, total_bytes: int | None = None) -> int | None:
    """systemd's ``MemoryMax`` syntax in bytes; ``None`` when there is no cap.

    ``None``/empty/``off``/``infinity`` mean no cap. A percentage needs
    ``total_bytes`` (the machine's physical memory), the way systemd resolves
    it. Anything else raises ``ValueError`` -- the caller warns once and goes
    inert, because a conftest that raises takes the whole run with it.
    """
    if value is None:
        return None
    text = value.strip()
    if text == "" or text.lower() in {"off", "infinity"}:
        return None
    if text.endswith("%"):
        if total_bytes is None:
            raise ValueError(f"{text!r} needs the machine's total memory to resolve")
        percent = float(text[:-1])
        if not 0 < percent <= 100:
            raise ValueError(f"{text!r} is not a percentage between 0 and 100")
        return int(total_bytes * percent / 100)
    number, suffix = text, ""
    if text[-1].upper() in "KMGTPEB":
        number, suffix = text[:-1], text[-1].upper()
        if suffix == "B" and number[-1:].upper() in "KMGTPE":
            number, suffix = number[:-1], number[-1].upper()
    try:
        magnitude = float(number)
    except ValueError:
        raise ValueError(f"{text!r} is not a systemd memory size") from None
    if magnitude < 0:
        raise ValueError(f"{text!r} is negative")
    return int(magnitude * _SUFFIXES[suffix])


def format_bytes(nbytes: float) -> str:
    """``1060135936 -> '1011M'`` -- the unit the runner's summary line uses."""
    return f"{int(nbytes) // _MB}M"


def is_new_high(rss: float, cap: int, prior_reported: float,
                fraction: float = DEFAULT_FRACTION,
                restep: float = DEFAULT_RESTEP) -> bool:
    """Should this resident set be reported against this cap?

    True when ``rss`` is over ``fraction * cap`` *and* is either the first such
    crossing or a new high at least ``restep * cap`` above the last one that was
    reported. The second clause is what stops one offender from failing every
    test that follows it.
    """
    limit = fraction * cap
    if rss < limit:
        return False
    if prior_reported < limit:
        return True
    return rss >= prior_reported + restep * cap


@dataclass
class _Current:
    nodeid: str
    peak: float = 0.0
    crossed: float = 0.0  # the resident set at the crossing, 0 for none


class Watchdog:
    """Samples ``rss_fn`` in a daemon thread and names the test that goes over.

    Everything it needs is injected -- the reader, the clock's period, the log
    path -- so the whole of it is exercised in ``tests/unit/test_memory_watchdog.py``
    without a real cap, a real thread of any duration, or a byte of JAX.
    """

    def __init__(self, cap: int, rss_fn, log_path: Path | None = None,
                 fraction: float = DEFAULT_FRACTION,
                 interval: float = DEFAULT_INTERVAL,
                 restep: float = DEFAULT_RESTEP,
                 cap_label: str | None = None):
        self.cap = int(cap)
        self.rss_fn = rss_fn
        self.log_path = Path(log_path) if log_path is not None else None
        self.fraction = fraction
        self.interval = interval
        self.restep = restep
        self.cap_label = cap_label or format_bytes(cap)
        self.peak = 0.0
        self.peak_nodeid = ""
        self.reported_high = 0.0
        self._current: _Current | None = None
        self._lock = threading.Lock()
        # A separate lock, held only across a write: the sampler thread logs a
        # crossing while the main thread logs the next test's `started` line,
        # and an interleaved line is a corrupted test name -- the one failure
        # this whole feature exists to prevent.
        self._log_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._log = None

    # -- the log, which is the half that survives a SIGKILL ----------------

    def log(self, message: str) -> None:
        """Append one line and flush it. Flushing is enough: a ``SIGKILL``
        loses what is in this process, not what the kernel already holds."""
        if self.log_path is None:
            return
        with self._log_lock:
            try:
                if self._log is None:
                    self.log_path.parent.mkdir(parents=True, exist_ok=True)
                    self._log = self.log_path.open("a", encoding="utf-8")
                self._log.write(message + "\n")
                self._log.flush()
            except OSError:
                self.log_path = None  # an unwritable log must not fail the suite

    @staticmethod
    def _now() -> str:
        # `date -Is`'s shape, so the shell loop's lines and these sort and
        # read as one file.
        return datetime.datetime.now().astimezone().isoformat(timespec="seconds")

    # -- sampling -----------------------------------------------------------

    def sample(self) -> float:
        """Read the resident set once, update the peaks, report a crossing.

        Called both by the daemon thread and directly at teardown, because a
        fixture that only looks after the test misses the peak that killed it.
        """
        try:
            rss = float(self.rss_fn())
        except Exception:
            return 0.0
        with self._lock:
            if rss > self.peak:
                self.peak = rss
                # A sample landing *between* two tests (a session fixture's
                # teardown, a collection) keeps the last name rather than
                # blanking it: an unnamed peak on the summary line is the one
                # thing that line exists to avoid.
                if self._current is not None:
                    self.peak_nodeid = self._current.nodeid
            current = self._current
            if current is None:
                return rss
            if rss > current.peak:
                current.peak = rss
            if is_new_high(rss, self.cap, self.reported_high, self.fraction, self.restep):
                self.reported_high = rss
                current.crossed = rss
                message = (f"{current.nodeid} RSS-HIGH rss={format_bytes(rss)} "
                           f"cap={self.cap_label} threshold={self.fraction:.0%} "
                           f"{self._now()}")
            else:
                return rss
        self.log(message)
        return rss

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="defumat-memwatch",
                                        daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.interval + 1.0)
            self._thread = None
        with self._log_lock:
            if self._log is not None:
                try:
                    self._log.close()
                except OSError:
                    pass
                self._log = None

    # -- the per-test hooks -------------------------------------------------

    def begin(self, nodeid: str) -> None:
        """Name the test *before* it runs. This is requirement one: what a kill
        destroys is the knowledge of what was in flight."""
        with self._lock:
            self._current = _Current(nodeid)
        self.log(f"{nodeid} started {self._now()}")

    def finish(self, nodeid: str) -> str | None:
        """End the test; a message here means fail it by name."""
        self.sample()
        with self._lock:
            current = self._current
            self._current = None
        if current is None or not current.crossed:
            return None
        return (f"memory watchdog: {nodeid} reached {format_bytes(current.crossed)} "
                f"resident, over {self.fraction:.0%} of the {self.cap_label} cap "
                f"({CAP_ENV}). The kernel would kill this process without naming "
                f"a test; this is the name. Peak so far {format_bytes(self.peak)}.")

    def summary_line(self) -> str:
        """One line for the terminal, shaped to compose with the per-file
        summary ``tools/run_regression.sh`` writes. It carries no ``passed`` /
        ``failed`` / ``error`` so the runner's own grep still finds pytest's
        line, and it adds what ``/usr/bin/time -f %M`` cannot: which test."""
        where = f" in {self.peak_nodeid}" if self.peak_nodeid else ""
        return (f"memory watchdog: peak={format_bytes(self.peak)}{where} "
                f"cap={self.cap_label} threshold={self.fraction:.0%}")


def build_watchdog(environ=None, rss_fn=None, total_bytes=None,
                   default_log=None, **kwargs):
    """The watchdog the environment asks for, or ``None`` for inert.

    Returns ``(watchdog, warning)``: ``warning`` is a string to be issued once
    when the cap could not be used. Nothing here raises -- a conftest that
    raises takes the whole run with it -- and ``psutil`` is imported only when a
    cap is actually set, so an unset variable costs one ``os.environ`` lookup.
    """
    environ = os.environ if environ is None else environ
    raw = environ.get(CAP_ENV)
    if raw is None or raw.strip() == "":
        return None, None
    if rss_fn is None or total_bytes is None:
        try:
            import psutil
        except Exception:
            return None, (f"{CAP_ENV}={raw} but psutil is not importable: "
                          "the memory watchdog is off")
        if rss_fn is None:
            process = psutil.Process()
            def rss_fn():  # noqa: E306 -- a closure over one process handle
                return process.memory_info().rss
        if total_bytes is None:
            total_bytes = psutil.virtual_memory().total
    try:
        cap = parse_mem_max(raw, total_bytes)
    except ValueError as exc:
        return None, f"{CAP_ENV}={raw} is unusable ({exc}): the memory watchdog is off"
    if cap is None or cap <= 0:
        return None, None  # `off`, and it is not a warning: it was asked for
    log_path = environ.get(LOG_ENV) or default_log
    return Watchdog(cap, rss_fn, log_path=log_path, cap_label=raw.strip(), **kwargs), None


def guard(watchdog: Watchdog | None, nodeid: str):
    """The body of the autouse fixture, here so the unit tests drive the real
    one rather than a copy of it. Yields once, around the test."""
    import pytest

    if watchdog is None:
        yield
        return
    watchdog.begin(nodeid)
    yield
    message = watchdog.finish(nodeid)
    if message is not None:
        pytest.fail(message, pytrace=False)
