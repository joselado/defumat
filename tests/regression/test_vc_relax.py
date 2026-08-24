"""P29 check: variable-cell relaxation against Quantum ESPRESSO.

The cases are QE's own ``pw_vc-relax/`` inputs that use BFGS -- rhombohedral
arsenic at zero pressure and at 500 kbar, the same at ``nspin = 2``, and the
same again with ``treinit_gvecs`` -- plus two cells too large for QE's suite to
have one: an eight-atom cubic silicon supercell under pressure and the ten-atom
five-layer graphite of P28b, whose ``c`` and ``a`` respond to different physics
and so have to move independently.

**What is compared is the relaxed cell, the relaxed positions and the energy of
the final SCF.** Not the trajectory: two BFGS implementations agreeing step for
step would be a stronger statement than either code makes, and a variable-cell
trajectory has a second reason not to -- it is walking downhill on an energy
whose basis was chosen for the starting cell, so where it stops depends on how
far it went before it got there.

**500 kbar is the case that matters and it is not a harder version of 0 kbar.**
At zero pressure the enthalpy is the energy and the ``P Omega`` term is
identically absent; at 500 kbar arsenic compresses by 10% and its two atoms
move from 0.2722 to 0.2500 -- the rhombohedral-to-simple-cubic transition -- so
the cell and the atoms are both doing something, and doing it at once.

**``vc-relax3`` is run with a different symmetry group on each side, and it is
the sharpest case here because of it.** ``symm_base.f90`` tests a fixed
catalogue of rotation matrices written in a canonical cartesian frame, so QE
finds a symmetry only when the crystal is presented in one of those frames;
:func:`~pypresso.system.symmetry.lattice_point_group` here searches for lattice
vectors of matching lengths and angles, which is orientation-free -- its module
docstring has always said so. ``vc-relax3`` and ``vc-relax4`` are the *same*
rhombohedral crystal in two settings, and QE finds **2** operations for the
first and **12** for the second where this code finds 12 for both. So on
``vc-relax3`` the two codes reduce the same 4x4x4 grid to **32** points and
**10**, symmetrise over groups of 2 and 12 -- and agree on the relaxed volume to
3e-5 bohr^3 and on the final energy to **1e-8 Ry**. That is not a weaker
comparison than the others; it is the statement that this grid is closed under
the larger group, which is exactly what P28b found is *not* automatic (a grid
with unequal divisions is not, and there the two codes' answers separate).
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from pypresso.io import read_qe_output
from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.system import build_system
from pypresso.units import RY_TO_KBAR
from pypresso.workflows.vc_relax import run_vc_relax

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
QE_SUITE = "pw_vc-relax"

# The bounds are the ones that were *measured*, not the ones the thresholds
# would allow. Both codes stop when ``|P I - sigma|`` is under
# ``press_conv_thr = 0.5`` kbar, which on arsenic near 500 kbar would permit
# ~4e-3 bohr in a linear dimension; what the four cases actually give is 2.2e-4
# bohr on the cell, 2.4e-6 on a crystal coordinate and 1.5e-5 Ry on the energy,
# so those are what is asserted, with a factor of two or three of margin. A
# bound set from the thresholds instead would pass through a regression that
# moved the answer by an order of magnitude.

#: The relaxed cell, in bohr. Worst measured: 2.2e-4 (``vc-relax6``).
CELL_BOHR = 6e-4
#: A crystal coordinate. Worst measured: 2.4e-6.
POSITION_CRYSTAL = 1e-5
#: The volume, as a fraction. Worst measured: 3.7e-3 bohr^3 on 190.9, i.e. 2e-5.
VOLUME_FRACTION = 6e-5
#: The final SCF's total energy, in Ry. Worst measured: 1.5e-5 (``vc-relax6``,
#: whose grids are rebuilt every step so the two codes' trajectories separate
#: further than they do at a fixed basis).
ENERGY_RY = 5e-5


@lru_cache(maxsize=None)
def _relaxed(name: str, pseudo_dir: Path, suite: bool):
    path = _input_path(name, suite)
    if path is None:
        pytest.skip(f"{name}: needs the vendored QE tree")
    system = build_system(read_pw_input(path))
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    return run_vc_relax(system, pseudos, conv_thr=1e-10)


#: The vendored tree, which is gitignored -- a case that needs it skips.
QE_ROOT = Path(__file__).resolve().parents[2] / "quantum_espresso" / \
    "qe-7.5-ReleasePack" / "qe-7.5"


def _input_path(name: str, suite: bool) -> Path | None:
    if not suite:
        return CASES / f"{name}.in"
    path = QE_ROOT / "test-suite" / QE_SUITE / f"{name}.in"
    return path if path.is_file() else None


def _reference(name: str, suite: bool):
    stem = f"{QE_SUITE}-{name}" if suite else name
    path = CASES / f"reference.out.{stem}"
    if not path.is_file():
        pytest.skip(f"no generated reference for {stem}; run tools/generate_reference.py")
    return read_qe_output(path)


def _crystal(positions, cell):
    return np.asarray(positions) @ np.linalg.inv(np.asarray(cell))


#: The four QE cases that use BFGS. ``vc-relax1`` and ``vc-relax2`` ask for
#: ``cell_dynamics = 'damp-w'`` and are refused by name rather than run.
QE_CASES = ["vc-relax3", "vc-relax4", "vc-relax5", "vc-relax6"]


@pytest.mark.parametrize("name", QE_CASES)
def test_the_relaxed_cell_matches_pw_x(name, pseudo_dir):
    """The nine cell coordinates, against ``pw.x``'s own relaxed cell."""
    result = _relaxed(name, pseudo_dir, True)
    reference = _reference(name, True)
    assert result.converged, f"{name}: pypresso did not converge"
    assert np.abs(result.cell - reference.final_cell).max() < CELL_BOHR


@pytest.mark.parametrize("name", QE_CASES)
def test_the_relaxed_volume_matches_pw_x(name, pseudo_dir):
    """The volume separately: it is what a vc-relax is usually run for."""
    result = _relaxed(name, pseudo_dir, True)
    reference = _reference(name, True)
    expected = abs(float(np.linalg.det(reference.final_cell)))
    assert abs(result.volume - expected) / expected < VOLUME_FRACTION


@pytest.mark.parametrize("name", QE_CASES)
def test_the_relaxed_positions_match_pw_x(name, pseudo_dir):
    """In crystal coordinates, which is where the physics is.

    At 500 kbar the two arsenic atoms move from 0.2722 to 0.2500 -- the
    rhombohedral cell becoming simple cubic -- so this is not a null assertion
    that the atoms stayed where they were put.
    """
    result = _relaxed(name, pseudo_dir, True)
    reference = _reference(name, True)
    expected = _crystal(reference.final_positions, reference.final_cell)
    moved = np.abs(result.positions_crystal - expected).max()
    assert moved < POSITION_CRYSTAL


@pytest.mark.parametrize("name", QE_CASES)
def test_the_final_scf_energy_matches_pw_x(name, pseudo_dir):
    """The energy of the run QE does *after* the relaxation, in its own basis.

    ``reset_gvectors``: the relaxation's own last energy is in a basis chosen
    for the starting cell and is not variational in the cell it is reported at,
    so it is not the number either code quotes.

    The bound is 5e-5 Ry rather than the suite's 1e-6, and it is not slack: the
    two codes stop at slightly different cells (both within ``press_conv_thr``
    of the target pressure) and this is the energy *at* those cells, so what is
    being compared carries the curvature of the enthalpy over that gap. On
    ``vc-relax3``, where both stop at effectively the same cell, they agree to
    1e-8.
    """
    result = _relaxed(name, pseudo_dir, True)
    reference = _reference(name, True)
    assert abs(result.total_energy - reference.final_total_energy) < ENERGY_RY


@pytest.mark.parametrize("name", QE_CASES)
def test_the_relaxed_crystal_carries_the_applied_pressure(name, pseudo_dir):
    """``sigma = P I`` is the stationary point, and the final SCF is where it
    has to hold -- in the basis that belongs to the relaxed cell.

    This shares nothing with ``pw.x``: it is the definition of what was being
    minimised, checked against the run's own output. A relaxation that stopped
    at the wrong cell for a reason both codes share would pass every comparison
    above and fail this one.
    """
    result = _relaxed(name, pseudo_dir, True)
    system = build_system(read_pw_input(_input_path(name, True)))
    target = system.relax.press
    residue = np.abs(result.stress - target / RY_TO_KBAR * np.eye(3)).max()
    assert residue * RY_TO_KBAR < 12.0, (
        f"{name}: the relaxed cell is {residue * RY_TO_KBAR:.1f} kbar from the "
        f"applied {target} kbar in its own basis"
    )


def test_a_different_symmetry_group_reaches_the_same_answer(pseudo_dir):
    """``vc-relax3``: QE reduces to 32 k-points and this code to 10.

    Pinned as its own test because the agreement is the interesting part and
    would otherwise look like a coincidence inside a parametrised sweep. The
    two codes find different groups for the reason in the module docstring, so
    they symmetrise different densities over different orbits and integrate
    different sums -- and land on the same crystal, which says the grid is
    closed under the larger group. P28b's finding is the case where that fails.
    """
    result = _relaxed("vc-relax3", pseudo_dir, True)
    reference = _reference("vc-relax3", True)
    ours = np.asarray(
        build_system(read_pw_input(_input_path("vc-relax3", True))).symmetry_group()
        .rotation_array()
    )
    assert len(ours) == 12, "this code should find the full rhombohedral group"
    expected = abs(float(np.linalg.det(reference.final_cell)))
    assert abs(result.volume - expected) < 1e-3
    assert abs(result.total_energy - reference.final_total_energy) < 1e-6


# --------------------------------------------------------------- bigger cells
def test_eight_atoms_and_a_cell_under_pressure(pseudo_dir):
    """The conventional cubic cell of silicon at 500 kbar, against ``pw.x``.

    Eight atoms is where the cell block and the atom block share a Hessian that
    is not nearly diagonal, and where the exact fractional coordinates make the
    structure factor vanish exactly at a set of G-vectors (P28a) whose new
    consumer here is the cell gradient.
    """
    result = _relaxed("si8-vc-relax", pseudo_dir, False)
    reference = _reference("si8-vc-relax", False)
    assert result.converged
    assert np.abs(result.cell - reference.final_cell).max() < CELL_BOHR


def test_the_relaxed_cubic_cell_is_still_cubic(pseudo_dir):
    """Nine coordinates moved and eight of them had to come back.

    Cubic symmetry is not imposed on the step -- ``cell_dofree = 'all'`` lets
    every entry of ``h`` move -- so the off-diagonal entries returning to zero
    is the symmetrised stress doing its job, and shares nothing with ``pw.x``.
    """
    cell = _relaxed("si8-vc-relax", pseudo_dir, False).cell
    lengths = np.linalg.norm(cell, axis=1)
    assert lengths.max() - lengths.min() < 1e-8, "the cell stopped being cubic"
    metric = cell @ cell.T
    assert np.abs(metric - np.diag(np.diag(metric))).max() < 1e-8


def test_ten_atoms_and_a_cell_that_changes_shape(pseudo_dir):
    """Five-layer graphite, where ``c`` relaxes and ``a`` barely does.

    The case the cell's *nine* coordinates exist for: ``a`` is a covalent bond
    and ``c`` is held only by the D2 correction, so a relaxation that moved
    them together would be wrong in a way no cubic cell can show.
    """
    result = _relaxed("c10-graphite-d2-vc-relax", pseudo_dir, False)
    reference = _reference("c10-graphite-d2-vc-relax", False)
    assert result.converged
    assert np.abs(result.cell - reference.final_cell).max() < CELL_BOHR


def test_graphites_layers_move_and_its_bonds_do_not(pseudo_dir):
    """The physics of the previous test, stated as the thing it is for."""
    result = _relaxed("c10-graphite-d2-vc-relax", pseudo_dir, False)
    system = build_system(read_pw_input(CASES / "c10-graphite-d2-vc-relax.in"))
    start = np.asarray(system.cell.at)
    a_start, c_start = np.linalg.norm(start[0]), np.linalg.norm(start[2])
    a_end, c_end = np.linalg.norm(result.cell[0]), np.linalg.norm(result.cell[2])
    assert abs(a_end - a_start) / a_start < 0.02, "the in-plane bond moved"
    assert abs(c_end - c_start) / c_start > abs(a_end - a_start) / a_start, (
        "the interlayer spacing has to be the soft direction"
    )
