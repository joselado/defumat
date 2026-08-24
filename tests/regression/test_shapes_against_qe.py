"""Every committed reference: the symmetry count and the FFT grid, and nothing else.

This runs no SCF. For each input that has a ``pw.x`` output stored beside it, it
builds the :class:`~pypresso.system.builder.System` and the basis and compares
the two numbers ``pw.x`` prints before it starts iterating -- ``N Sym. Ops.``
and ``FFT dimensions`` -- against what this code chose.

**It is here because those two numbers are where a disagreement in the physics
usually starts**, and because three separate bugs found in P28b were each
visible in one of them long before any energy was computed:

* a lattice point group searched over a fixed ``range(-3, 4)`` window, which on
  a five-cell supercell found 2 operations where ``pw.x`` finds 6;
* a fractional translation accepted whatever its denominator, where ``sgam_at``
  takes only ``1/n`` with ``n in {2, 3, 4, 6}`` -- five-layer graphite kept 12
  operations instead of 6 and got a 20x20x135 FFT grid instead of 20x20x128;
* and, earlier, the supercell rule on fractional translations itself
  (``is_supercell``), which is worth 24 operations against 48 on eight-atom
  silicon.

Each of those cost a real energy difference -- 3.2e-6, 1.7e-4 and 1e-5 Ry -- and
each would have been caught here in a second and a half, against the minutes an
SCF comparison takes. The whole sweep is about ninety seconds for sixty-odd
cases, so it is cheap enough to be the first thing that fails.
"""

import re
from pathlib import Path

import pytest

from pypresso.basis.builder import build_basis
from pypresso.io.pwin import read_pw_input
from pypresso.system import build_system

pytestmark = [pytest.mark.regression]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"

#: Restamped test-suite references are stored as ``reference.out.<dir>-<stem>``
#: and their inputs live in the vendored tree, so they are only reachable when
#: it is present.
RESTAMPED = re.compile(r"^reference\.out\.(pw_[a-z+]+)-(.+)$")


def _committed_cases():
    """``(label, input path or (directory, name), reference path)`` for each."""
    for reference in sorted(CASES.glob("reference.out.*")):
        stem = reference.name[len("reference.out.") :]
        if stem.startswith("ph-"):
            continue  # a ph.x output, which prints neither number
        match = RESTAMPED.match(reference.name)
        if match:
            yield stem, (match.group(1), match.group(2)), reference
        else:
            source = CASES / f"{stem}.in"
            if source.is_file():
                yield stem, source, reference


CASE_LIST = list(_committed_cases())
assert CASE_LIST, "no committed references found"


def _input_path(where, qe_testsuite):
    if isinstance(where, Path):
        return where
    directory, name = where
    path = qe_testsuite / directory / f"{name}.in"
    if not path.is_file():
        pytest.skip(f"{directory}/{name}.in is not in the vendored tree")
    return path


@pytest.mark.parametrize(("label", "where", "reference"), CASE_LIST,
                         ids=[case[0] for case in CASE_LIST])
def test_symmetry_count_and_fft_grid_match_qe(qe_testsuite, label, where, reference):
    text = reference.read_text()
    printed_symmetry = re.search(r"(\d+)\s+Sym\. Ops\.", text)
    printed_grid = re.search(r"FFT dimensions:\s*\(\s*(\d+),\s*(\d+),\s*(\d+)\)", text)
    if printed_symmetry is None and printed_grid is None:
        pytest.skip(f"{reference.name} prints neither number")

    system = build_system(read_pw_input(_input_path(where, qe_testsuite)))

    if printed_symmetry is not None:
        # ``pw.x`` prints the count *after* its own rules have run -- the
        # supercell rule on fractional translations, the crystallographic
        # denominators, and (for a magnetic noncollinear run) the operations
        # that need time reversal.
        assert int(system.symmetry_group().nsym) == int(printed_symmetry.group(1)), label

    if printed_grid is not None:
        grid = tuple(build_basis(system).dense.grid)
        assert grid == tuple(int(n) for n in printed_grid.groups()), label
