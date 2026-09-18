"""k-point generation: grid construction, folding, band paths, weights."""

import numpy as np
import pytest

from defumat.system.cell import Cell
from defumat.system.kpoints import DEGSPIN, KPoints, expand_band_path, monkhorst_pack

pytestmark = pytest.mark.unit

CUBIC = Cell.from_ibrav(1, [10.0, 0, 0, 0, 0, 0])


@pytest.mark.parametrize("grid", [(1, 1, 1), (2, 2, 2), (3, 2, 1), (4, 4, 4)])
@pytest.mark.parametrize("shift", [(0, 0, 0), (1, 1, 1), (1, 0, 1)])
def test_grid_size_weights_and_uniqueness(grid, shift):
    points, weights = monkhorst_pack(grid, shift)

    assert len(points) == grid[0] * grid[1] * grid[2]
    assert weights.sum() == pytest.approx(1.0)
    assert weights == pytest.approx(np.full(len(points), 1.0 / len(points)))
    assert np.all(np.abs(points) <= 0.5 + 1e-12)
    assert len(np.unique(np.round(points, 9), axis=0)) == len(points)


def test_grid_ordering_matches_quantum_espresso():
    """Last index fastest, as in kpoint_grid.f90's n = (k-1) + (j-1)*nk3 + ..."""
    points, _ = monkhorst_pack((2, 2, 2), (0, 0, 0))
    assert points[0] == pytest.approx([0.0, 0.0, 0.0])
    assert points[1] == pytest.approx([0.0, 0.0, -0.5])  # third index moved first
    assert points[2] == pytest.approx([0.0, -0.5, 0.0])
    assert points[4] == pytest.approx([-0.5, 0.0, 0.0])


def test_half_integer_points_fold_the_fortran_way():
    """NumPy's rint rounds half to even; Fortran's NINT rounds away from zero.

    An unshifted even grid puts points at exactly 0.5, so the two conventions
    disagree on a case that occurs constantly, not on a corner case.
    """
    points, _ = monkhorst_pack((2, 1, 1), (0, 0, 0))
    assert points[1, 0] == pytest.approx(-0.5)  # not +0.5
    assert np.rint(0.5) == 0.0  # the behaviour being worked around


def test_shifted_grid_is_offset_by_half_a_step():
    unshifted, _ = monkhorst_pack((4, 4, 4), (0, 0, 0))
    shifted, _ = monkhorst_pack((4, 4, 4), (1, 1, 1))
    difference = np.sort(shifted, axis=0) - np.sort(unshifted, axis=0)
    assert np.allclose(np.abs(difference), 1.0 / 8.0)


def test_invalid_grids_are_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        monkhorst_pack((0, 1, 1))
    with pytest.raises(ValueError, match="0 or 1"):
        monkhorst_pack((2, 2, 2), (2, 0, 0))


def test_band_path_expansion_counts_and_spacing():
    """1 + sum(counts[:-1]) points; the last count is ignored, as QE does."""
    vertices = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 0.25, 0.25], [0.5, 0.5, 0.5], [0, 0, 0]])
    points, lengths = expand_band_path(vertices, [5, 5, 5, 5, 1])

    assert len(points) == 21
    assert points[0] == pytest.approx([0, 0, 0])
    assert points[1] == pytest.approx([0.2, 0, 0])  # 1/5 of the first segment
    assert points[5] == pytest.approx([1.0, 0, 0])  # the vertex is hit exactly
    assert points[-1] == pytest.approx([0, 0, 0])
    assert np.all(np.diff(lengths) >= -1e-12)  # path length never decreases


def test_band_path_zero_count_is_a_discontinuity():
    """A count of 0 jumps to the next vertex without adding path length."""
    vertices = np.array([[0.0, 0, 0], [0.5, 0, 0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]])
    points, lengths = expand_band_path(vertices, [2, 0, 2, 1])

    assert len(points) == 1 + 2 + 1 + 2
    jump = np.where(np.diff(lengths) == 0.0)[0]
    assert len(jump) == 1, "exactly one discontinuity expected"
    assert points[jump[0] + 1] == pytest.approx([0.0, 0.5, 0.0])


def test_crystal_band_path_keeps_discontinuities_flat():
    """A crystal_b path recomputes lengths in cartesian space; a zero count must
    still add no length, exactly as in the tpiba_b branch."""
    kpoints = KPoints.band_path(
        [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0]], [2, 0, 2, 1], CUBIC, crystal=True
    )
    lengths = np.asarray(kpoints.path_length)
    assert np.all(np.diff(lengths) >= -1e-12)
    assert np.sum(np.diff(lengths) == 0.0) == 1


def test_weights_are_normalised_then_spin_degenerate():
    """QE normalises weights to 1 and multiplies by degspin for nspin=1."""
    kpoints = KPoints.from_cartesian([[0, 0, 0], [0.5, 0, 0]], [1.0, 3.0])
    assert float(kpoints.weights.sum()) == pytest.approx(DEGSPIN)
    assert np.asarray(kpoints.weights) == pytest.approx([0.5, 1.5])

    assert float(KPoints.gamma().weights.sum()) == pytest.approx(DEGSPIN)
    assert KPoints.gamma().gamma_only is True


def test_automatic_grid_round_trips_through_cartesian():
    kpoints = KPoints.automatic((3, 3, 3), (0, 0, 0), CUBIC)
    crystal = np.asarray(kpoints.crystal(CUBIC))
    expected, _ = monkhorst_pack((3, 3, 3), (0, 0, 0))
    assert crystal == pytest.approx(expected, abs=1e-12)
    assert kpoints.grid == (3, 3, 3)


def test_cartesian_conversion_uses_tpiba():
    kpoints = KPoints.from_cartesian([[0.5, 0.0, 0.0]], [1.0])
    assert np.asarray(kpoints.cartesian(CUBIC)) == pytest.approx(
        np.array([[0.5 * CUBIC.tpiba, 0.0, 0.0]])
    )


def test_zero_total_weight_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        KPoints.from_cartesian([[0, 0, 0]], [0.0])


# --- whether a set is a wedge is recorded, not guessed from the weights ---------

FCC = Cell.from_ibrav(2, [10.20, 0, 0, 0, 0, 0])
_E = np.eye(3)[None]
_E_C2Z = np.array([np.eye(3), np.diag([-1.0, -1.0, 1.0])])


@pytest.mark.parametrize("rotations,shift,nk,spread", [
    (_E,     (1, 1, 1), 32, 0.0),
    (_E,     (0, 0, 0), 36, 0.03125),
    (_E_C2Z, (1, 1, 1), 16, 0.0),
    (_E_C2Z, (0, 0, 0), 30, 0.09375),
])
def test_a_shifted_wedge_has_uniform_weights(rotations, shift, nk, spread):
    """The measurement six guards in this package were written against.

    Every "is this k-set a wedge" test read a *spread* in the weights, and a
    symmetry-reduced **shifted** Monkhorst-Pack grid has exactly uniform ones
    whenever the group acts freely on it -- which is what a shift arranges. So
    the guard returned False on half or a quarter of the zone and the refusal
    that protects an axial or rank-3 polar wedge sum never fired, on the most
    ordinary grid QE writes.

    The table is the evidence: the heuristic works unshifted, where it was
    written, and fails shifted. The identity is always in the group and time
    reversal is on by default, so even a P1 cell halves.

    The unshifted spreads are twice what a raw ``irreducible_wedge`` returns,
    because ``KPoints.automatic`` goes through ``_normalise``, which applies
    ``DEGSPIN``. The zeros are zero in either normalisation, which is the whole
    point -- no scaling rescues a spread that is exactly nothing.
    """
    points = KPoints.automatic((4, 4, 4), shift, FCC, rotations=rotations)
    weights = np.asarray(points.weights)
    assert points.nk == nk
    assert np.ptp(weights) == pytest.approx(spread, abs=1e-12)


@pytest.mark.parametrize("rotations,shift,expected", [
    (_E,     (1, 1, 1), True),    # 32 of 64, and the weights do not say so
    (_E,     (0, 0, 0), True),
    (None,   (1, 1, 1), False),   # the whole grid
    (None,   (0, 0, 0), False),
])
def test_the_reduced_flag_says_what_the_weights_cannot(rotations, shift, expected):
    from defumat.system.kpoints import is_reduced

    points = KPoints.automatic((4, 4, 4), shift, FCC, rotations=rotations)
    assert points.reduced is expected
    assert is_reduced(points) is expected


def test_the_weight_spread_survives_as_a_fallback():
    """A hand-built wedge has no flag, and the old test still has to catch it.

    That is why the spread stays *underneath* rather than being replaced: an
    explicit ``K_POINTS`` list with unequal weights is how the closed-grid
    cases in ``tests/data/qe`` are written, and a flag set by
    ``KPoints.automatic`` cannot see one.
    """
    from defumat.system.kpoints import is_reduced

    points = KPoints.from_crystal(
        np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]), np.array([1.0, 3.0]), FCC)
    assert points.reduced is False
    assert is_reduced(points) is True


def test_the_flag_survives_for_spin():
    """It has to reach a polarized run, which is where the guards live."""
    from defumat.system.kpoints import for_spin, is_reduced

    points = KPoints.automatic((4, 4, 4), (1, 1, 1), FCC, rotations=_E)
    assert is_reduced(for_spin(points, 2)) is True
