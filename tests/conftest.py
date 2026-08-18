"""Shared fixtures.

The vendored QE tree is large and is not committed (see ``.gitignore``), so any
test that needs a reference output skips cleanly when it is absent rather than
failing. The pseudopotential files under ``tests/data/pseudo`` *are* committed,
because they are small and nothing is runnable without them.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
QE_ROOT = REPO_ROOT / "quantum_espresso" / "qe-7.5-ReleasePack" / "qe-7.5"


@pytest.fixture(scope="session")
def qe_testsuite() -> Path:
    """Path to QE's ``test-suite`` directory, or skip if the tree is absent."""
    path = QE_ROOT / "test-suite"
    if not path.is_dir():
        pytest.skip(f"QE reference tree not present at {path}")
    return path


@pytest.fixture(scope="session")
def pseudo_dir() -> Path:
    """Committed UPF files used by the reference inputs."""
    return Path(__file__).parent / "data" / "pseudo"


@pytest.fixture(scope="session")
def benchmark(qe_testsuite):
    """``benchmark('pw_scf', 'scf.in')`` -> the committed QE output for that input."""

    def _get(directory: str, input_name: str) -> Path:
        path = qe_testsuite / directory / f"benchmark.out.git.inp={input_name}"
        if not path.is_file():
            pytest.skip(f"no committed benchmark for {directory}/{input_name}")
        return path

    return _get
