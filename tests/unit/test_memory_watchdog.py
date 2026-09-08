"""The memory watchdog: threshold logic, sampler, and the named failure.

No JAX, no SCF, nothing that allocates -- the watchdog is driven with an
injected resident-set reader, so a "20 GB" test costs a function call. What is
checked is the thing the feature promises: that a test which crosses a fraction
of ``DEFUMAT_TEST_MEM_MAX`` is *named*, in a log line written before it would
have been killed and in a failure carrying its nodeid.
"""

import datetime
import sys
import threading
import time
from pathlib import Path

import pytest

from tests import memwatch

pytest_plugins = ["pytester"]

GB = 1024**3
MB = 1024**2


# -- the cap, in the runner's own syntax -----------------------------------


@pytest.mark.parametrize("text,expected", [
    ("12G", 12 * GB),
    ("512M", 512 * MB),
    ("4g", 4 * GB),
    ("4GB", 4 * GB),
    ("1048576", MB),
    ("64K", 64 * 1024),
])
def test_parse_mem_max_sizes(text, expected):
    assert memwatch.parse_mem_max(text) == expected


@pytest.mark.parametrize("text", [None, "", "  ", "off", "infinity", "INFINITY"])
def test_parse_mem_max_no_cap(text):
    assert memwatch.parse_mem_max(text) is None


def test_parse_mem_max_percentage_needs_the_machine():
    assert memwatch.parse_mem_max("50%", 32 * GB) == 16 * GB
    with pytest.raises(ValueError):
        memwatch.parse_mem_max("50%")


@pytest.mark.parametrize("text", ["twelve", "12X", "-4G", "200%"])
def test_parse_mem_max_rejects_garbage(text):
    with pytest.raises(ValueError):
        memwatch.parse_mem_max(text, 32 * GB)


def test_format_bytes_matches_the_runner_summary_unit():
    # `/usr/bin/time -f %M` divided by 1024 is what the summary line carries,
    # e.g. `test_scf.py` at 1011 M.
    assert memwatch.format_bytes(1011 * MB) == "1011M"


# -- the threshold ----------------------------------------------------------


def test_below_the_fraction_is_not_a_crossing():
    assert not memwatch.is_new_high(8 * GB, 12 * GB, prior_reported=0.0)


def test_first_crossing_is_reported():
    assert memwatch.is_new_high(10.5 * GB, 12 * GB, prior_reported=0.0)


def test_the_default_fraction_leaves_the_documented_headroom():
    cap = 12 * GB
    assert memwatch.DEFAULT_FRACTION * cap == pytest.approx(cap - 1.8 * GB, rel=0.02)


def test_a_second_test_at_the_same_level_does_not_cascade():
    # XLA's executable cache does not shrink, so without this rule one file's
    # first offender would fail every test after it.
    prior = 10.5 * GB
    assert not memwatch.is_new_high(10.5 * GB, 12 * GB, prior_reported=prior)
    assert not memwatch.is_new_high(10.6 * GB, 12 * GB, prior_reported=prior)


def test_a_new_high_two_percent_of_the_cap_up_is_reported():
    prior = 10.5 * GB
    assert memwatch.is_new_high(prior + 0.025 * 12 * GB, 12 * GB, prior_reported=prior)


# -- the sampler ------------------------------------------------------------


def make_watchdog(tmp_path, values, cap=12 * GB, **kwargs):
    """A watchdog reading a scripted sequence of resident-set values."""
    series = list(values)
    state = {"i": 0}

    def rss_fn():
        i = min(state["i"], len(series) - 1)
        state["i"] += 1
        return series[i]

    return memwatch.Watchdog(cap, rss_fn, log_path=tmp_path / "in-flight.log",
                             cap_label="12G", **kwargs)


def test_the_nodeid_is_written_before_the_test_runs(tmp_path):
    watchdog = make_watchdog(tmp_path, [1 * GB])
    watchdog.begin("tests/regression/test_scf.py::test_energy")
    # The point of the whole feature: the name is on disk while the test is
    # still running, because a SIGKILL takes everything else.
    text = (tmp_path / "in-flight.log").read_text()
    assert "tests/regression/test_scf.py::test_energy started" in text


def test_a_peak_between_samples_is_caught_not_the_teardown_value(tmp_path):
    # 11 GB during the test, 1 GB by the time it ends: a fixture that only
    # looked at teardown would see nothing at all.
    watchdog = make_watchdog(tmp_path, [1 * GB, 11 * GB, 1 * GB])
    watchdog.begin("test_a.py::test_peaky")
    watchdog.sample()          # 1 GB
    watchdog.sample()          # 11 GB -- the peak, as the thread would see it
    message = watchdog.finish("test_a.py::test_peaky")
    assert message is not None
    assert "test_a.py::test_peaky" in message
    assert "11264M" in message


def test_the_crossing_is_logged_when_it_happens(tmp_path):
    watchdog = make_watchdog(tmp_path, [11 * GB])
    watchdog.begin("test_a.py::test_big")
    watchdog.sample()
    text = (tmp_path / "in-flight.log").read_text()
    assert "test_a.py::test_big RSS-HIGH" in text
    assert "cap=12G" in text and "threshold=85%" in text
    # Logged during the test, not at its end.
    assert watchdog.finish("test_a.py::test_big") is not None


def test_the_log_timestamps_match_the_shell_loops_own(tmp_path):
    # `run_regression.sh` writes `<file> started $(date -Is)` into the same
    # file; without the UTC offset the two halves would not sort together.
    watchdog = make_watchdog(tmp_path, [1 * GB])
    watchdog.begin("test_a.py::test_x")
    stamp = (tmp_path / "in-flight.log").read_text().split("started ")[1].strip()
    assert datetime.datetime.fromisoformat(stamp).utcoffset() is not None


def test_a_quiet_test_is_not_failed(tmp_path):
    watchdog = make_watchdog(tmp_path, [2 * GB])
    watchdog.begin("test_a.py::test_small")
    assert watchdog.finish("test_a.py::test_small") is None


def test_only_the_test_that_grows_further_is_failed_again(tmp_path):
    watchdog = make_watchdog(tmp_path, [11 * GB, 11 * GB, 11.6 * GB])
    watchdog.begin("test_a.py::first")
    assert watchdog.finish("test_a.py::first") is not None
    watchdog.begin("test_a.py::second")
    assert watchdog.finish("test_a.py::second") is None
    watchdog.begin("test_a.py::third")
    assert watchdog.finish("test_a.py::third") is not None


def test_the_daemon_thread_samples_and_stops(tmp_path):
    ticks = {"n": 0}
    started = threading.Event()

    def rss_fn():
        ticks["n"] += 1
        started.set()
        return 3 * GB

    watchdog = memwatch.Watchdog(12 * GB, rss_fn, log_path=None, interval=0.005)
    watchdog.begin("test_a.py::test_threaded")
    watchdog.start()
    assert started.wait(2.0), "the sampler thread never ran"
    time.sleep(0.05)
    watchdog.stop()
    assert watchdog._thread is None
    seen = ticks["n"]
    assert seen >= 2
    time.sleep(0.05)
    assert ticks["n"] == seen, "the thread kept sampling after stop()"
    assert watchdog.peak == 3 * GB
    assert watchdog.peak_nodeid == "test_a.py::test_threaded"


def test_a_reader_that_raises_does_not_take_the_suite_down(tmp_path):
    def rss_fn():
        raise OSError("no /proc here")

    watchdog = memwatch.Watchdog(12 * GB, rss_fn)
    watchdog.begin("test_a.py::test_x")
    assert watchdog.sample() == 0.0
    assert watchdog.finish("test_a.py::test_x") is None


def test_an_unwritable_log_is_dropped_not_raised(tmp_path):
    blocked = tmp_path / "afile" / "in-flight.log"
    (tmp_path / "afile").write_text("not a directory")
    watchdog = memwatch.Watchdog(12 * GB, lambda: 1 * GB, log_path=blocked)
    watchdog.begin("test_a.py::test_x")   # must not raise
    assert watchdog.log_path is None


def test_the_summary_line_composes_with_the_runner(tmp_path):
    watchdog = make_watchdog(tmp_path, [1011 * MB])
    watchdog.begin("tests/regression/test_scf.py::test_energy")
    watchdog.sample()
    line = watchdog.summary_line()
    assert line.startswith("memory watchdog: peak=1011M")
    assert "tests/regression/test_scf.py::test_energy" in line
    # `run_regression.sh` picks pytest's own line out of the tail with a
    # `passed|failed|error` grep; this line must not answer to it.
    assert not any(word in line for word in ("passed", "failed", "error"))


# -- what the environment asks for -----------------------------------------


def test_unset_is_inert(monkeypatch):
    monkeypatch.delenv(memwatch.CAP_ENV, raising=False)
    watchdog, warning = memwatch.build_watchdog(environ={})
    assert watchdog is None and warning is None


def test_off_is_inert_and_is_not_a_warning():
    watchdog, warning = memwatch.build_watchdog(environ={memwatch.CAP_ENV: "off"})
    assert watchdog is None and warning is None


def test_a_cap_builds_a_watchdog_on_the_runners_log_path(tmp_path):
    log = tmp_path / "in-flight.log"
    watchdog, warning = memwatch.build_watchdog(
        environ={memwatch.CAP_ENV: "12G", memwatch.LOG_ENV: str(log)},
        rss_fn=lambda: 1 * GB, total_bytes=32 * GB)
    assert warning is None
    assert watchdog.cap == 12 * GB
    assert watchdog.cap_label == "12G"
    assert watchdog.log_path == log
    assert watchdog.fraction == memwatch.DEFAULT_FRACTION


def test_an_unparseable_cap_warns_and_goes_off(tmp_path):
    watchdog, warning = memwatch.build_watchdog(
        environ={memwatch.CAP_ENV: "twelve gigs"},
        rss_fn=lambda: 1 * GB, total_bytes=32 * GB)
    assert watchdog is None
    assert "unusable" in warning and memwatch.CAP_ENV in warning


def test_missing_psutil_disables_the_watchdog_rather_than_the_suite(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)  # `import psutil` -> ImportError
    watchdog, warning = memwatch.build_watchdog(environ={memwatch.CAP_ENV: "12G"})
    assert watchdog is None
    assert "psutil" in warning


def test_psutil_is_there_and_reads_this_process(tmp_path):
    psutil = pytest.importorskip("psutil")
    watchdog, warning = memwatch.build_watchdog(
        environ={memwatch.CAP_ENV: "1024G", memwatch.LOG_ENV: str(tmp_path / "l.log")})
    assert warning is None
    rss = watchdog.sample()
    assert rss == pytest.approx(psutil.Process().memory_info().rss, rel=0.5)
    assert rss > 0


# -- end to end, through the real conftest ---------------------------------


def test_the_real_fixture_fails_the_named_test(pytester, monkeypatch):
    """The deliverable itself: the *copied* project conftest, a 1 M cap that
    this process is certainly over, and a test that is named in the report."""
    conftest = (Path(__file__).parents[1] / "conftest.py").read_text()
    pytester.makeconftest(conftest)
    pytester.makepyfile(test_over="def test_over_the_cap():\n    assert True\n")
    log = pytester.path / "in-flight.log"
    monkeypatch.setenv(memwatch.CAP_ENV, "1M")
    monkeypatch.setenv(memwatch.LOG_ENV, str(log))
    result = pytester.runpytest_inprocess("-p", "no:cacheprovider")
    result.stdout.fnmatch_lines(["*test_over.py::test_over_the_cap*"])
    assert result.ret != 0
    assert "memory watchdog" in result.stdout.str()
    text = log.read_text()
    assert "test_over.py::test_over_the_cap started" in text
    assert "RSS-HIGH" in text


def test_the_real_fixture_leaves_a_passing_test_alone(pytester, monkeypatch):
    conftest = (Path(__file__).parents[1] / "conftest.py").read_text()
    pytester.makeconftest(conftest)
    pytester.makepyfile(test_ok="def test_under_the_cap():\n    assert True\n")
    monkeypatch.setenv(memwatch.CAP_ENV, "1024G")
    monkeypatch.setenv(memwatch.LOG_ENV, str(pytester.path / "in-flight.log"))
    result = pytester.runpytest_inprocess("-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)
    assert "memory watchdog: peak=" in result.stdout.str()
