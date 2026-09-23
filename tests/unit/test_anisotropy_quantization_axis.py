"""Turning a magnet's quantization axis is a rigid rotation of its texture.

The force theorem, the torque and the relaxed anisotropy all point a magnet
along a requested direction through ``_with_quantization_axis``. What that
must mean for a cell with more than one moment is a **global** spin rotation:
the angles between moments are a property of the magnetic state, and the
anisotropy is the energy of turning that state as a whole. Two defects broke
it, and both are host-side, so nothing here runs an SCF.

* One pair of angles was written for every species, so the two antiparallel
  sublattices of ``fe2-afm-soc.in`` came out parallel: the antiferromagnet
  was turned into a ferromagnet on every rotation.
* A ``STARTING_MOMENTS`` card overrides the per-species angles and was left
  untouched, so neither the seed nor the fixed axis ever reached the
  requested direction.

The assertions are on the Gram matrix of the moments, which a rigid rotation
leaves invariant and nothing else does, and on where the texture's own axis
ends up.
"""

from pathlib import Path

import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.scf.potential import fixed_quantization_axis
from defumat.system.builder import build_system
from defumat.workflows.anisotropy import _reference_axis, _with_quantization_axis

pytestmark = pytest.mark.unit

AFM = Path(__file__).resolve().parents[1] / "data" / "qe" / "fe2-afm-soc.in"


@pytest.fixture(scope="module")
def afm():
    return build_system(read_pw_input(AFM))


def _gram(moments):
    moments = np.asarray(moments, dtype=float)
    return moments @ moments.T


def test_the_cell_is_the_antiferromagnet_it_says_it_is(afm):
    np.testing.assert_allclose(
        afm.local_moments, [[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]], atol=1.0e-12)


@pytest.mark.parametrize("direction", [
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
    (-1.0, 0.0, 0.0),
    (1.0, 2.0, -0.5),
])
def test_an_antiferromagnet_stays_antiparallel(afm, direction):
    turned = _with_quantization_axis(afm, direction)
    moments = turned.local_moments
    wanted = np.asarray(direction, dtype=float)
    wanted = wanted / np.linalg.norm(wanted)

    # The angles between the moments are those of the original state.
    np.testing.assert_allclose(_gram(moments), _gram(afm.local_moments),
                               atol=1.0e-12)
    # Species one lies along the direction and species two against it.
    np.testing.assert_allclose(moments[0], 0.5 * wanted, atol=1.0e-12)
    np.testing.assert_allclose(moments[1], -0.5 * wanted, atol=1.0e-12)
    # A GGA takes its sign along this axis: it must be the new direction
    # (up to the sign of the first row), not undefined as for a texture.
    axis = fixed_quantization_axis(moments)
    assert axis is not None
    assert abs(abs(float(axis @ wanted)) - 1.0) < 1.0e-10


def test_the_rotation_reaches_a_starting_moments_card(afm):
    canted = afm.with_moments([[0.5, 0.0, 0.0], [0.0, 0.4, 0.3]])
    np.testing.assert_allclose(_reference_axis(canted), (1.0, 0.0, 0.0),
                               atol=1.0e-12)
    turned = _with_quantization_axis(canted, (0.0, 0.0, 1.0))

    # The card itself is rotated, not only the angles the card overrides.
    rows = np.asarray(turned.starting_moments, dtype=float)
    np.testing.assert_allclose(rows[0], (0.0, 0.0, 0.5), atol=1.0e-12)
    np.testing.assert_allclose(_gram(rows), _gram(canted.starting_moments),
                               atol=1.0e-12)
    # ... and the seed the SCF and the symmetry group read is the rotated one.
    np.testing.assert_allclose(turned.local_moments, rows, atol=1.0e-12)
    np.testing.assert_allclose(_reference_axis(turned), (0.0, 0.0, 1.0),
                               atol=1.0e-12)


def test_an_antiparallel_card_stays_antiparallel(afm):
    card = afm.with_moments([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])
    turned = _with_quantization_axis(card, (0.0, 0.0, 1.0))
    np.testing.assert_allclose(
        turned.local_moments, [[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]], atol=1.0e-12)


def test_the_own_direction_is_the_identity(afm):
    # Returned untouched, so a single-direction run rebuilds nothing.
    assert _with_quantization_axis(afm, _reference_axis(afm)) is afm
    card = afm.with_moments([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]])
    assert _with_quantization_axis(card, (0.0, 1.0, 0.0)) is card


# -- the reference axis is where the first magnetic atom points ---------------

def test_a_nonmagnetic_species_one_does_not_set_the_axis(afm):
    """An oxide listed with O first: the axis is the metal's, not O's ``z``.

    Species one carries no moment and keeps its default angles, which point
    along ``z``; species two points along ``x``. The old rule read species
    one's angles and so called ``z`` the system's own direction, and a force
    theorem asked for ``x`` then turned the metal's ``x`` moment onto ``-z``.
    """
    oxide = afm.with_spin(starting_magnetization=(0.0, 0.5),
                          angle1=(0.0, 90.0), angle2=(0.0, 0.0))
    np.testing.assert_allclose(oxide.local_moments,
                               [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], atol=1.0e-12)
    np.testing.assert_allclose(_reference_axis(oxide), (1.0, 0.0, 0.0),
                               atol=1.0e-12)
    # The moment already points along x, so asking for x changes nothing.
    assert _with_quantization_axis(oxide, (1.0, 0.0, 0.0)) is oxide


def test_one_texture_has_one_axis_with_or_without_a_card(afm):
    """The same moments, written per species and as a card, give one axis.

    Both species' ``starting_magnetization`` negative puts atom one along
    ``-x`` and atom two along ``+x``. The card route always took the first
    nonzero row with its sign, and the species route dropped the sign.
    """
    flipped = afm.with_spin(starting_magnetization=(-0.5, -0.5))
    texture = [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]
    np.testing.assert_allclose(flipped.local_moments, texture, atol=1.0e-12)
    card = afm.with_moments(texture)
    np.testing.assert_allclose(_reference_axis(flipped), (-1.0, 0.0, 0.0),
                               atol=1.0e-12)
    np.testing.assert_allclose(_reference_axis(flipped), _reference_axis(card),
                               atol=1.0e-12)

    # A named direction is where atom one ends up, on both routes.
    for system in (flipped, card):
        turned = _with_quantization_axis(system, (0.0, 0.0, 1.0))
        np.testing.assert_allclose(turned.local_moments,
                                   [[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]],
                                   atol=1.0e-12)
