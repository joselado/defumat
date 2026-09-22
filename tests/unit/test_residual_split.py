"""The five-way residual split, and the trip tests its rotation bin needs.

``MAGNETISM-NEXT.md`` F2 says the obvious definition of the rotation bin -- the
``Q = 0`` transverse component of the residual -- is a null that cannot be told
from a pass, because every compensated texture has ``m_{Q = 0} = 0`` and a rigid
rotation leaves it zero. The generator projection is the replacement, and the
tests that matter are the ones that would **fire** on that failure: a rigidly
rotated antiferromagnet, whose net moment is zero at every angle.
"""
import numpy as np
import pytest

from defumat.scf.residual_split import residual_bins


class Cell:
    """The quadrature weight is all :func:`residual_bins` asks of a cell."""

    def __init__(self, volume: float):
        self.volume = volume


CELL = Cell(volume=8.0)
GRID = (4, 4, 4)


def rotate_about_z(magnetization, angle):
    """Turn every moment in the cell by the same angle -- the Goldstone mode."""
    cosine, sine = np.cos(angle), np.sin(angle)
    turned = np.empty_like(magnetization)
    turned[0] = cosine * magnetization[0] - sine * magnetization[1]
    turned[1] = sine * magnetization[0] + cosine * magnetization[1]
    turned[2] = magnetization[2]
    return turned


def density_from(magnetization, charge=1.0):
    """``(nspin_mag, ...)`` with a flat charge channel and this magnetization."""
    return np.concatenate(
        [np.full((1,) + GRID, charge), magnetization], axis=0
    )


def ferromagnet():
    return np.stack([
        np.full(GRID, 0.5), np.zeros(GRID), np.zeros(GRID),
    ])


def antiferromagnet():
    """Two sublattices with opposite moments, so the net moment is exactly zero.

    This is the texture the ``Q = 0`` definition of the rotation bin cannot see:
    it integrates to zero before the rotation and after it, so a bin built on the
    cell's net moment reads zero either way.
    """
    sign = np.ones(GRID)
    sign[::2] = -1.0
    return np.stack([0.5 * sign, np.zeros(GRID), np.zeros(GRID)])


@pytest.mark.parametrize(
    "texture, name",
    [(ferromagnet(), "ferromagnet"), (antiferromagnet(), "antiferromagnet")],
)
def test_a_rigid_rotation_lands_in_the_rotation_bin(texture, name):
    """``R rho - rho`` at a small angle is the rotation and nothing else."""
    angle = 1.0e-3
    rho_in = density_from(texture)
    rho_out = density_from(rotate_about_z(texture, angle))

    bins = residual_bins(rho_in, rho_out, CELL)

    assert bins["charge"] == pytest.approx(0.0, abs=1e-15)
    # The coefficient *is* the angle, which is what the generator normalisation
    # buys: the bin reads in radians rather than in units of the density.
    assert bins["rotation_coefficients"][2] == pytest.approx(angle, rel=1e-5)
    assert bins["rotation_coefficients"][0] == pytest.approx(0.0, abs=1e-12)
    assert bins["rotation_coefficients"][1] == pytest.approx(0.0, abs=1e-12)

    # **The other two bins are second order in the angle rather than zero, and
    # asserting that they are small would be asserting the angle.** Turning a
    # moment by theta leaves its length alone but leaves the *residual*
    # ``R m - m`` a component along ``m`` of ``|m|(cos theta - 1)``, which is
    # ``-|m| theta^2/2``, so at theta = 1e-3 the longitudinal bin reads 5e-4 of
    # the rotation one and a tolerance tight enough to look impressive would
    # simply be wrong. What says the split is right is the *scaling*: halving
    # the angle halves the rotation bin and quarters the other two.
    halved = residual_bins(
        rho_in, density_from(rotate_about_z(texture, angle / 2)), CELL
    )
    assert halved["rotation"] == pytest.approx(bins["rotation"] / 2, rel=1e-3)
    assert halved["longitudinal"] == pytest.approx(
        bins["longitudinal"] / 4, rel=1e-2
    )
    assert halved["transverse"] == pytest.approx(
        bins["transverse"] / 4, rel=1e-2
    )


def test_the_net_moment_definition_would_have_read_zero_here():
    """The falsifier for the bin's definition, on the texture that needs it.

    This is the measurement that says the generator projection is not a
    stylistic preference. The same rotated antiferromagnet, scored the obvious
    way -- the change in the cell's *net* moment -- gives a number that is zero
    to round-off at every angle, so a bin built that way cannot tell a live
    Goldstone mode from a converged one.
    """
    angle = 1.0e-3
    texture = antiferromagnet()
    turned = rotate_about_z(texture, angle)

    weight = CELL.volume / texture[0].size
    net_before = np.array([texture[i].sum() * weight for i in range(3)])
    net_after = np.array([turned[i].sum() * weight for i in range(3)])
    assert np.abs(net_after - net_before).max() < 1e-15

    bins = residual_bins(density_from(texture), density_from(turned), CELL)
    assert bins["rotation"] > 1e-4
    assert bins["rotation_coefficients"][2] == pytest.approx(angle, rel=1e-5)


def test_a_pure_length_change_is_longitudinal_and_not_a_rotation():
    """The other side of the split, so that neither bin absorbs the other."""
    texture = ferromagnet()
    rho_in = density_from(texture)
    rho_out = density_from(texture * 1.01)

    bins = residual_bins(rho_in, rho_out, CELL)

    assert bins["longitudinal"] > 1e-3
    assert bins["rotation"] < 1e-12
    assert bins["transverse"] < 1e-12


def test_a_collinear_run_has_no_rotation_bin_at_all():
    """The other half of the trip test, and it is exact rather than small."""
    rho_in = np.stack([np.full(GRID, 1.0), np.full(GRID, 0.5)])
    rho_out = np.stack([np.full(GRID, 1.0), np.full(GRID, 0.52)])

    bins = residual_bins(rho_in, rho_out, CELL)

    assert bins["rotation"] == 0.0
    assert bins["transverse"] == 0.0
    assert bins["longitudinal"] > 0.0


def test_an_unpolarized_region_is_counted_rather_than_split():
    """A cell with vacuum has no moment direction over most of its grid.

    Those points are reported rather than assigned to either bin, because
    dividing by a vanishing length is how a diagnostic acquires a large number
    that means nothing.
    """
    texture = ferromagnet()
    texture[:, 2:, :, :] = 0.0
    rho_in = density_from(texture)
    rho_out = density_from(texture * 1.01)

    bins = residual_bins(rho_in, rho_out, CELL)

    assert bins["unpolarized_points"] == 2 * GRID[1] * GRID[2]
    assert np.isfinite(bins["longitudinal"])
    assert np.isfinite(bins["rotation"])


def test_a_nonmagnetic_run_still_answers_every_bin_by_name():
    """The twin has to be readable by the code that reads the magnet.

    F2's deconfounder compares a magnetic run against its nonmagnetic twin, so a
    consumer reading a bin by name must get a zero for a channel the regime does
    not have rather than a ``KeyError``. The bins that exist are still real:
    ``charge`` is the whole residual here.
    """
    rho_in = np.full((1,) + GRID, 1.0)
    rho_out = np.full((1,) + GRID, 1.02)

    bins = residual_bins(rho_in, rho_out, CELL)

    assert bins["charge"] > 0.0
    for name in ("longitudinal", "rotation", "transverse"):
        assert bins[name] == 0.0
    assert bins["rotation_coefficients"] == [0.0, 0.0, 0.0]
    assert bins["unpolarized_points"] == 0
