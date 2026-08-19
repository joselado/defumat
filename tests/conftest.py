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
