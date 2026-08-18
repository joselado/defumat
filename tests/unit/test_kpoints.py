"""k-point generation: grid construction, folding, band paths, weights."""

import numpy as np
import pytest

from pypresso.system.cell import Cell
from pypresso.system.kpoints import DEGSPIN, KPoints, expand_band_path, monkhorst_pack

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
