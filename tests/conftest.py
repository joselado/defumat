"""Shared fixtures.

The vendored QE tree is large and is not committed (see ``.gitignore``), so any
test that needs a reference output skips cleanly when it is absent rather than
failing. The pseudopotential files under ``tests/data/pseudo`` *are* committed,
because they are small and nothing is runnable without them.
"""

import os
from pathlib import Path

import pytest

from tests import memwatch  # stdlib + psutil only; never JAX

#: How many cores the suite leaves visible, and why it is not the package
#: default.
#:
#: ``defumat._limit_thread_pool`` narrows the affinity mask to
#: ``DEFAULT_THREADS = 4`` because that is the fastest setting for the physics
#: -- 238 ms per SCF iteration on ``benchmarks/si8-1k-ecut30.in`` against 411 ms
#: at eight cores and 385 at twelve. **It is also the setting that deadlocks**:
#: XLA sizes its CPU thread pool from that mask, and a long-lived process that
#: has compiled many different cells eventually parks with every worker in
#: ``futex_wait_queue`` and no CPU at all. The rate follows the mask and nothing
#: else (``OPEN.md`` Part IV item 1, where the whole measurement is):
#:
#:     2 cores  8 hangs / 8      8 cores   0 / 8
#:     4 cores  6 hangs / 14     all 12    0 / 14
#:
#: So the suite widens the mask and production does not. The suite is not a
#: performance measurement -- nothing may be timed beside a test run anyway --
#: and it pays about 15% on wall clock for a gate that finishes every time
#: instead of one run in three. Eight rather than "off" because this machine is
#: shared, and a gate that takes every core takes them from another session.
#:
#: **This must run before ``defumat`` is first imported**, which is why it is at
#: the top of ``conftest.py`` and not in a fixture: the mask is set once, at
#: import. An explicit ``DEFUMAT_THREADS`` in the environment still wins, so a
#: run that wants to reproduce the hang can ask for it.
TEST_THREADS = "8"
os.environ.setdefault("DEFUMAT_THREADS", TEST_THREADS)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Where the QE source tree is looked for. The vendored location is the default
#: and nothing about it has changed; ``DEFUMAT_QE_ROOT`` points at a QE tree
#: installed somewhere else instead.
#:
#: **Why this exists.** The tree is 285 MB, gitignored and never in history, so a
#: fresh checkout has none -- and about thirty tests skip for that reason alone,
#: including every ``test-suite`` input comparison and the continuation set. A
#: released QE installed outside the repo has the same ``test-suite/`` and the
#: same inputs, so those tests can run against it. **The version is the caveat
#: and it is the caller's to weigh**: the regenerated references committed under
#: ``tests/data/qe`` were produced with the QE this path pointed at when they were
#: made, and ``reference_output`` prefers them over the tree's own benchmark. A
#: minor-version mismatch between the *input* read here and the *output* compared
#: against is therefore possible and is not detected. 7.4.1 has been checked
#: against 7.5 on the whole fast benchmark set and agrees to two digits in every
#: ``dE`` and every iteration count (``PERFORMANCE.md``), which is why it is a
#: usable stand-in rather than a guess.
QE_ROOT = Path(
    os.environ.get(
        "DEFUMAT_QE_ROOT",
        REPO_ROOT / "quantum_espresso" / "qe-7.5-ReleasePack" / "qe-7.5",
    )
)


@pytest.fixture(scope="session")
def qe_testsuite() -> Path:
    """Path to QE's ``test-suite`` directory, or skip if the tree is absent."""
    path = QE_ROOT / "test-suite"
    if not path.is_dir():
        pytest.skip(
            f"QE reference tree not present at {path}. It is gitignored, so a "
            f"fresh checkout has none; set DEFUMAT_QE_ROOT to a QE installation "
            f"to run these against its test-suite instead"
        )
    return path


@pytest.fixture(scope="session")
def pseudo_dir() -> Path:
    """Committed UPF files used by the reference inputs."""
    return Path(__file__).parent / "data" / "pseudo"


#: References regenerated with the vendored pw.x, for cases whose committed
#: benchmark is stale. See ``tools/generate_reference.py``.
GENERATED = Path(__file__).parent / "data" / "qe"


def reference_output(directory: str, input_name: str, testsuite: Path) -> Path | None:
    """The QE output to compare against: the regenerated one if there is one.

    QE's committed benchmarks were produced with release 6.0. For a
    non-symmorphic crystal, QE has since started forcing the FFT dimensions to
    be a multiple of the fractional translations' denominators -- diamond
    silicon's grid went from 15^3 to 16^3 -- which moves the exchange-correlation
    energy in the sixth decimal, since it is evaluated pointwise on that grid.
    Where a regenerated reference exists it is used, so that what is being
    reproduced is the QE that is actually vendored here rather than a release
    from 2016.
    """
    generated = GENERATED / f"reference.out.{directory}-{Path(input_name).stem}"
    if generated.is_file():
        return generated
    committed = testsuite / directory / f"benchmark.out.git.inp={input_name}"
    return committed if committed.is_file() else None


@pytest.fixture(scope="session")
def benchmark(qe_testsuite):
    """``benchmark('pw_scf', 'scf.in')`` -> the QE output to compare against."""

    def _get(directory: str, input_name: str) -> Path:
        path = reference_output(directory, input_name, qe_testsuite)
        if path is None:
            pytest.skip(f"no benchmark for {directory}/{input_name}")
        return path

    return _get


@pytest.fixture(scope="session")
def committed_benchmark(qe_testsuite):
    """QE's own shipped output file, ignoring any regenerated replacement.

    The parser tests transcribe values by hand out of a specific file, so they
    must read *that* file. Everything comparing computed numbers should use
    ``benchmark`` instead, which prefers a regenerated reference.
    """

    def _get(directory: str, input_name: str) -> Path:
        path = qe_testsuite / directory / f"benchmark.out.git.inp={input_name}"
        if not path.is_file():
            pytest.skip(f"no committed benchmark for {directory}/{input_name}")
        return path

    return _get


# ---------------------------------------------------------------------------
# The memory watchdog: fail a *named* test before the kernel kills the process.
#
# `tools/run_regression.sh` puts each file in a cgroup scope with `MemoryMax`,
# so an out-of-memory kill costs one file rather than the session. What it
# cannot do is say which *test* was holding the memory, because a `SIGKILL`
# takes the process's knowledge of that with it. The fixture below writes the
# running test's nodeid into the runner's own `in-flight.log` before the test
# starts, samples the resident set once a second while it runs, and fails the
# test at teardown once its peak crosses 85% of the same `DEFUMAT_TEST_MEM_MAX`
# the runner set. Everything it does is in `tests/memwatch.py`, including what
# it cannot catch; keep this end thin.
#
# It is inert with the variable unset -- no thread, no file, no `psutil` import
# -- so the pre-push gate pays nothing. To watch the gate, give it a cap:
# `DEFUMAT_TEST_MEM_MAX=12G tools/test-fast.sh`.
#
# A teardown failure is reported by pytest as `ERROR at teardown of <test>`
# rather than `FAILED`; that is still the test's name, and it still matches the
# `passed|failed|error` grep the runner's summary line is built from.
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Build the watchdog once, and warn (never raise) if the cap is unusable."""
    watchdog, warning = memwatch.build_watchdog(
        default_log=REPO_ROOT / "regression-results" / "in-flight.log"
    )
    config._memwatch = watchdog
    if warning is not None:
        config.issue_config_time_warning(UserWarning(warning), stacklevel=2)


def pytest_sessionstart(session):
    watchdog = getattr(session.config, "_memwatch", None)
    if watchdog is not None:
        watchdog.start()


def pytest_sessionfinish(session, exitstatus):
    watchdog = getattr(session.config, "_memwatch", None)
    if watchdog is not None:
        watchdog.log(watchdog.summary_line())
        watchdog.stop()


def pytest_terminal_summary(terminalreporter):
    """The peak, in the runner's own `peak=NNNM` shape, plus the test that set
    it -- which is the thing `/usr/bin/time -f %M` around the process cannot
    say."""
    watchdog = getattr(terminalreporter.config, "_memwatch", None)
    if watchdog is not None:
        terminalreporter.write_line(watchdog.summary_line())


@pytest.fixture(autouse=True)
def memory_watchdog(request):
    yield from memwatch.guard(getattr(request.config, "_memwatch", None),
                              request.node.nodeid)
