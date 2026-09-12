"""The benchmark sets name inputs that exist, and the cluster arrays match them.

A set is a list of case names resolved against ``benchmarks/`` and
``tests/data/qe/``. Nothing else checks that those files are still there, and
the failure without this test is a sweep that skips a case quietly -- the
report's missing row reads as "not run yet" rather than as "renamed in 2024".

These are file and string checks only. Nothing here runs an SCF.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from performance import sets  # noqa: E402


def test_every_case_file_exists():
    """Every name in every set resolves to a readable pw.x input."""
    missing = []
    for name, cases in sets.SETS.items():
        for case in cases:
            try:
                assert case.path.is_file()
            except SystemExit:
                missing.append(f"{name}: {case.name}")
    assert not missing, f"benchmark sets name inputs that do not exist: {missing}"


def test_fast_is_contained_in_complete():
    """`complete` is a superset, so a regression seen fast is seen completely."""
    fast = {case.name for case in sets.FAST}
    complete = {case.name for case in sets.COMPLETE}
    assert fast <= complete, f"only in fast: {sorted(fast - complete)}"


def test_no_duplicate_cases():
    """A duplicate would be measured twice and reported once, silently."""
    for name, cases in sets.SETS.items():
        names = [case.name for case in cases]
        assert len(names) == len(set(names)), f"{name} repeats a case"


def test_every_case_has_a_reason():
    """The description is what makes a set reviewable rather than a list."""
    for cases in sets.SETS.values():
        for case in cases:
            assert case.description.strip(), f"{case.name} has no description"


@pytest.mark.parametrize("set_name", sorted(sets.SETS))
def test_generated_sbatch_is_valid_bash_and_covers_the_set(set_name, tmp_path):
    """The array scripts parse, and their bash array is exactly the set.

    The generator bakes the case list into the script rather than importing
    Python at run time, so this is the check that the two cannot drift.
    """
    generator = ROOT / "tools" / "cluster" / "submit_benchmark.py"
    done = subprocess.run(
        [sys.executable, str(generator), "--set", set_name, "--out-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert done.returncode == 0, done.stderr

    expected = [case.name for case in sets.cases(set_name)]
    scripts = sorted(tmp_path.glob("*.sbatch"))
    assert len(scripts) == 3, f"expected one array per leg, got {[p.name for p in scripts]}"

    for script in scripts:
        syntax = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        assert syntax.returncode == 0, f"{script.name}: {syntax.stderr}"

        text = script.read_text()
        listed = text.split("CASES=(", 1)[1].split(")", 1)[0].split()
        assert listed == expected, f"{script.name} case list drifted from {set_name}"
        assert f"--array=0-{len(expected) - 1}%" in text, f"{script.name}: wrong array range"


def test_the_generator_never_submits():
    """It writes scripts and prints commands. It must not call sbatch itself.

    The policy beside this checkout is that nothing in this repository submits,
    cancels or polls a job on its own; this is that rule with a test behind it.
    """
    source = (ROOT / "tools" / "cluster" / "submit_benchmark.py").read_text()
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "print(" in stripped:
            continue
        for forbidden in ("subprocess", "os.system", "os.exec", "popen"):
            assert forbidden not in stripped.lower(), \
                f"the generator must not run anything: {stripped!r}"


# --------------------------------------------------- the two guards on a local run


def test_the_large_cases_carry_a_recorded_peak():
    """A cell big enough to kill this machine must not be unlabelled.

    The guard can only refuse what it knows the size of, and an unmeasured case
    is passed through. So the cells already known to be large -- the twenty- and
    forty-atom ones, and spin-orbit on bismuth -- carry their figure explicitly.
    """
    by_name = {case.name: case for case in sets.COMPLETE}
    for name in ("bi20-soc", "bi10-soc", "h40-chain-lsda"):
        assert by_name[name].peak_gb, f"{name} has no recorded peak for the memory guard"


def test_a_case_too_large_for_the_machine_is_refused():
    """`bi20-soc`'s 35 GB against a 31 GB machine is a refusal, not an attempt."""
    from performance import sweep

    big = next(case for case in sets.COMPLETE if case.name == "bi20-soc")
    small = next(case for case in sets.COMPLETE if case.name == "si-1k")

    assert sweep.too_large_for(big, ram_gb=31.0), "bi20-soc must not start on a 31 GB box"
    assert sweep.too_large_for(big, ram_gb=512.0) is None, "it fits a big node"
    assert sweep.too_large_for(small, ram_gb=31.0) is None, "si-1k fits anything"
    # An unmeasured case, or an unreadable MemTotal, is passed through rather
    # than guessed at in either direction.
    assert sweep.too_large_for(big, ram_gb=None) is None
    assert sweep.too_large_for(sets.Case("x", ""), ram_gb=1.0) is None


def test_the_two_slow_converging_cells_ask_for_more_iterations():
    """A cap of 100 would report these two as not converged. They converge at 104 and 151.

    `PERFORMANCE.md` records the GPU sweep making exactly this mistake and
    reporting an iteration cap as a convergence failure.
    """
    by_name = {case.name: case for case in sets.COMPLETE}
    for name in ("ni10-ldau", "h40-chain-lsda"):
        assert (by_name[name].max_iterations or 0) >= 200, \
            f"{name} converges past 100 iterations and must carry its own cap"
