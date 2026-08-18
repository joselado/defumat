"""P1 check: cells and k-points reproduce what QE prints in its output header.

Every input in the reference test-suite that has a committed benchmark is built
through pypresso and compared against QE's own header: lattice parameter, cell
volume, crystal and reciprocal axes, and -- where the k-point set is written out
explicitly -- the k-points and their weights.

The ``pw_lattice-ibrav`` directory is the point of this sweep: it exercises
ibrav 1..14 including the negative variants, plus all four ways of specifying a
free cell. Those sign and ordering conventions are exactly what a
"reasonable-looking" reimplementation gets subtly wrong.
"""

import numpy as np
import pytest

from pypresso.io import read_qe_output
from pypresso.io.pwin import read_pw_input
from pypresso.system import build_system
from pypresso.system.kpoints import monkhorst_pack
from tests.conftest import QE_ROOT
from tests.tolerances import GEOMETRY

pytestmark = pytest.mark.regression

#: Directories swept. pw_lattice-ibrav covers the Bravais lattices; the others
#: bring in the systems the later phases target.
SWEPT_DIRECTORIES = ("pw_lattice-ibrav", "pw_scf", "pw_metal", "pw_atom", "pw_lsda")


def _cases():
    """(directory, input name) for every input with a committed benchmark."""
    suite = QE_ROOT / "test-suite"
    if not suite.is_dir():
        return [pytest.param(None, None, marks=pytest.mark.skip(reason="QE tree absent"))]
    found = []
    for directory in SWEPT_DIRECTORIES:
        for benchmark in sorted((suite / directory).glob("benchmark.out.git.inp=*.in")):
            name = benchmark.name.split("inp=", 1)[1]
            if (suite / directory / name).is_file():
                found.append(pytest.param(directory, name, id=f"{directory}/{name}"))
    assert found, "no reference cases discovered"
    return found


CASES = _cases()


#: ATOMIC_POSITIONS crystal_sg gives Wyckoff positions, which need space-group
#: expansion -- a symmetry-phase (P6) feature, not a P1 gap.
NEEDS_SPACE_GROUPS = {"lattice-wyckoff-sio2.in"}


def _build(path):
    if path.name in NEEDS_SPACE_GROUPS:
        pytest.skip("crystal_sg (Wyckoff) positions need space-group expansion (P6)")
    return build_system(read_pw_input(path))


@pytest.mark.parametrize(("directory", "name"), CASES)
def test_cell_matches_reference(qe_testsuite, directory, name):
    system = _build(qe_testsuite / directory / name)
    ref = read_qe_output(qe_testsuite / directory / f"benchmark.out.git.inp={name}")

    # QE prints alat and the volume with 4 decimals and the axes with 6, so
    # agreement is demanded at the precision it actually reports: half of the
    # last printed digit, with a little room for the rounding itself.
    assert system.cell.alat == pytest.approx(ref.alat, abs=1e-4)
    assert float(system.cell.volume) == pytest.approx(ref.volume, abs=1e-4)
    assert np.asarray(system.cell.at_alat) == pytest.approx(ref.at, abs=GEOMETRY)
    assert np.asarray(system.cell.bg_2pi_alat) == pytest.approx(ref.bg, abs=GEOMETRY)

    # The reciprocal cell is defined by this identity; if a lattice were built
    # with a left-handed or permuted basis it would still hold, so it is checked
    # in addition to -- not instead of -- the comparison above.
    identity = np.asarray(system.cell.at_alat @ system.cell.bg_2pi_alat.T)
    assert identity == pytest.approx(np.eye(3), abs=1e-10)


@pytest.mark.parametrize(("directory", "name"), CASES)
def test_kpoints_match_reference(qe_testsuite, directory, name):
    system = _build(qe_testsuite / directory / name)
    pwin = read_pw_input(qe_testsuite / directory / name)
    ref = read_qe_output(qe_testsuite / directory / f"benchmark.out.git.inp={name}")

    if ref.kpoints is None:
        pytest.skip("QE did not print the k-point list for this run")
    if system.nspin == 2:
        # LSDA duplicates every k-point for the two spin channels before
        # printing; comparing that is the spin phase's business, not P1's.
        pytest.skip("nspin=2 prints one k-point list per spin channel")

    card = pwin.card("K_POINTS")
    option = (card.option or "tpiba").lower() if card else "gamma"

    if option == "automatic":
        # QE prints the *irreducible* wedge, so a point-by-point comparison has
        # to wait for symmetry (P6). It cannot even be checked that QE's points
        # lie on our grid: kpoint_grid.f90 reduces using the point group of the
        # Bravais lattice and keeps grid points, but irrek.f90 then maps those
        # representatives into the wedge of the *crystal's* point group by
        # rotating them, and a rotation carries a shifted grid off itself.
        # (Concretely, lattice-ibrav2-kauto prints a k-point at crystal
        # (0.25, 0.5, 0.5) which is not on the shifted 2x2x2 grid it came from.)
        # What is checkable now: the complete grid has the right size, uniform
        # weights, lies in the first BZ, and QE's reduction of it is no larger.
        grid = [int(v) for v in card.lines[0].split()[:6]]
        full, weights = monkhorst_pack(tuple(grid[:3]), tuple(grid[3:6]))

        assert len(full) == grid[0] * grid[1] * grid[2]
        assert weights == pytest.approx(np.full(len(full), 1.0 / len(full)))
        assert np.all(np.abs(full) <= 0.5 + 1e-12), "grid point outside the first BZ"
        assert len(np.unique(np.round(full, 9), axis=0)) == len(full), "duplicate grid point"

        assert len(ref.kpoints) <= len(full)
        assert system.kpoints.nk == len(full)
        assert float(system.kpoints.weights.sum()) == pytest.approx(2.0)
        assert float(np.sum(ref.weights)) == pytest.approx(2.0)
        return

    # Explicit lists and band paths: QE prints exactly what we should produce.
    assert system.kpoints.nk == len(ref.kpoints)
    assert np.asarray(system.kpoints.coords) == pytest.approx(ref.kpoints, abs=1e-6)
    assert np.asarray(system.kpoints.weights) == pytest.approx(ref.weights, abs=1e-6)
    assert float(system.kpoints.weights.sum()) == pytest.approx(2.0)
