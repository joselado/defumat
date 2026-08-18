"""FFT size selection, checked against the rules rather than against a table."""

import numpy as np
import pytest

from pypresso.basis.fftgrid import (
    allowed_fft_size,
    fft_grid_dimensions,
    gcut_from_ecut,
    good_fft_order,
)
from pypresso.system.cell import Cell

pytestmark = pytest.mark.unit


def test_allowed_sizes_are_products_of_2_3_5():
    """QE accepts factors up to 11 in principle but rejects 7 and 11 for every
    build except IBM ESSL, so the practical rule is 2-3-5 products."""
    allowed = [n for n in range(1, 100) if allowed_fft_size(n)]
    assert allowed[:12] == [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16]
    assert 7 not in allowed and 11 not in allowed and 13 not in allowed
    for n in allowed:
        remainder = n
        for factor in (2, 3, 5):
            while remainder % factor == 0:
                remainder //= factor
        assert remainder == 1
    assert not allowed_fft_size(0)
    assert not allowed_fft_size(-4)


def test_good_fft_order_rounds_up_and_is_idempotent():
    assert good_fft_order(13) == 15
    assert good_fft_order(15) == 15  # already good: unchanged
    assert good_fft_order(17) == 18
    assert good_fft_order(7) == 8
    for n in range(1, 200):
        rounded = good_fft_order(n)
        assert rounded >= n and allowed_fft_size(rounded)
        assert not any(allowed_fft_size(m) for m in range(n, rounded))


def test_good_fft_order_with_a_required_multiple():
    value = good_fft_order(10, multiple_of=4)
    assert value % 4 == 0 and allowed_fft_size(value) and value >= 10


def test_gcut_conversion():
    """gcut is the cutoff in units of (2*pi/alat)^2."""
    alat = 10.2
    assert gcut_from_ecut(48.0, alat) == pytest.approx(48.0 / (2 * np.pi / alat) ** 2)
    assert gcut_from_ecut(0.0, alat) == 0.0


@pytest.mark.parametrize("ibrav", [1, 2, 3, 4, 6, 8, 12, 14])
def test_grid_contains_the_sphere(ibrav):
    """The defining property: every G inside the cutoff must be representable on
    the grid, i.e. its Miller indices must fit in [-(n-1)/2, (n-1)/2]."""
    cell = Cell.from_ibrav(ibrav, [10.0, 1.5, 2.0, 0.1, 0.2, 0.3])
    gcut = gcut_from_ecut(30.0, cell.alat)
    at, bg = np.asarray(cell.at_alat), np.asarray(cell.bg_2pi_alat)
    grid = fft_grid_dimensions(at, bg, gcut)

    half = [(n - 1) // 2 for n in grid]
    ranges = [np.arange(-h - 2, h + 3) for h in half]  # deliberately wider
    i, j, k = np.meshgrid(*ranges, indexing="ij")
    miller = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1)
    inside = miller[np.sum((miller @ bg) ** 2, axis=1) < gcut]

    assert np.all(np.abs(inside).max(axis=0) <= half), "a G inside the cutoff does not fit"
    # And the grid is not wastefully large: shrinking any axis would clip it.
    assert np.all(np.abs(inside).max(axis=0) >= np.array(half) - 2)


def test_silicon_grid_matches_the_known_value():
    cell = Cell.from_ibrav(2, [10.2, 0, 0, 0, 0, 0])
    grid = fft_grid_dimensions(
        np.asarray(cell.at_alat), np.asarray(cell.bg_2pi_alat), gcut_from_ecut(48.0, 10.2)
    )
    assert grid == (15, 15, 15)


def test_absurdly_small_cutoff_is_reported():
    cell = Cell.from_ibrav(1, [10.0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="cutoff"):
        fft_grid_dimensions(np.asarray(cell.at_alat), np.asarray(cell.bg_2pi_alat), 0.0)
