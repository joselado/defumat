"""Properties every Bravais lattice must satisfy, independent of QE's output.

The regression sweep compares against QE's printed axes; these checks assert the
things that must be true of *any* correct implementation, so a failure points at
the transcription rather than at a formatting difference.
"""

import equinox as eqx
import jax
import numpy as np
import pytest

from defumat.config import DOUBLE
from defumat.system.cell import IBRAV_NAMES, Cell, celldm_from_abc, latgen
from defumat.units import TPI

pytestmark = pytest.mark.unit

#: celldm that satisfies every lattice's validity constraints at once.
GENERIC_CELLDM = np.array([10.0, 1.5, 2.0, 0.1, 0.2, 0.3])

ALL_IBRAV = sorted(i for i in IBRAV_NAMES if i != 0)


@pytest.mark.parametrize("ibrav", ALL_IBRAV)
def test_axes_are_right_handed_with_positive_volume(ibrav):
    """latgen.f90 states "all axis sets are right-handed"; a left-handed set
    would break the ultrasoft augmentation boxes later, so it is checked here."""
    at = latgen(ibrav, GENERIC_CELLDM)
    assert np.linalg.det(at) > 0.0, f"ibrav={ibrav} produced a left-handed basis"


@pytest.mark.parametrize("ibrav", ALL_IBRAV)
def test_reciprocal_lattice_is_dual(ibrav):
    cell = Cell.from_ibrav(ibrav, GENERIC_CELLDM)
    overlap = np.asarray(cell.at @ cell.bg.T)
    assert overlap == pytest.approx(TPI * np.eye(3), abs=1e-9)


@pytest.mark.parametrize("ibrav", ALL_IBRAV)
def test_coordinate_round_trips(ibrav):
    cell = Cell.from_ibrav(ibrav, GENERIC_CELLDM)
    rng = np.random.default_rng(ibrav if ibrav > 0 else -ibrav)
    points = rng.uniform(-1.0, 1.0, size=(5, 3))

    assert np.asarray(cell.to_crystal(cell.to_cartesian(points))) == pytest.approx(points)
    assert np.asarray(cell.k_to_crystal(cell.k_to_cartesian(points))) == pytest.approx(points)


def test_known_lattices():
    """Cubic cases whose vectors and volumes are known by hand."""
    simple = Cell.from_ibrav(1, [4.0, 0, 0, 0, 0, 0])
    assert np.asarray(simple.at) == pytest.approx(4.0 * np.eye(3))
    assert float(simple.volume) == pytest.approx(64.0)

    fcc = Cell.from_ibrav(2, [4.0, 0, 0, 0, 0, 0])
    assert float(fcc.volume) == pytest.approx(4.0**3 / 4.0)  # a^3/4 per primitive cell

    bcc = Cell.from_ibrav(3, [4.0, 0, 0, 0, 0, 0])
    assert float(bcc.volume) == pytest.approx(4.0**3 / 2.0)  # a^3/2

    hexagonal = Cell.from_ibrav(4, [4.0, 0, 1.6, 0, 0, 0])
    assert float(hexagonal.volume) == pytest.approx(np.sqrt(3) / 2 * 4.0**3 * 1.6)


def test_bcc_variants_describe_the_same_lattice():
    """ibrav=3 and -3 differ only by the choice of axes."""
    a, b = Cell.from_ibrav(3, [5.0, 0, 0, 0, 0, 0]), Cell.from_ibrav(-3, [5.0, 0, 0, 0, 0, 0])
    assert float(a.volume) == pytest.approx(float(b.volume))
    # The transformation between them must be an integer matrix (same lattice).
    transform = np.asarray(a.at) @ np.linalg.inv(np.asarray(b.at))
    assert transform == pytest.approx(np.rint(transform), abs=1e-9)


def test_from_vectors_defaults_alat_to_first_vector_length():
    at = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    assert Cell.from_vectors(at).alat == pytest.approx(3.0)
    assert Cell.from_vectors(at, alat=1.0).alat == pytest.approx(1.0)


def test_celldm_from_abc_places_cosines_by_ibrav():
    # Triclinic takes all three cosines, in the order (cosbc, cosac, cosab).
    triclinic = celldm_from_abc(14, a=2.0, b=3.0, c=4.0, cosab=0.1, cosac=0.2, cosbc=0.3)
    assert triclinic[1:] == pytest.approx([1.5, 2.0, 0.3, 0.2, 0.1])
    # Monoclinic unique axis b takes only cosac, in slot 5.
    monoclinic = celldm_from_abc(-12, a=2.0, b=3.0, c=4.0, cosab=0.1, cosac=0.2, cosbc=0.3)
    assert monoclinic[3:] == pytest.approx([0.0, 0.2, 0.0])
    # Unique axis c takes cosab, in slot 4.
    assert celldm_from_abc(12, a=2.0, b=3.0, c=4.0, cosab=0.1)[3] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("ibrav", "celldm", "message"),
    [
        (2, [0.0, 0, 0, 0, 0, 0], "celldm\\(1\\)"),
        (4, [10.0, 0, 0.0, 0, 0, 0], "celldm\\(3\\)"),
        (5, [10.0, 0, 0, 1.5, 0, 0], "celldm\\(4\\)"),
        (14, [10.0, 1.0, 1.0, 0.9, 0.9, -0.9], "do not make sense"),
        (99, [10.0, 0, 0, 0, 0, 0], "nonexistent"),
        (0, [10.0, 0, 0, 0, 0, 0], "explicit lattice vectors"),
    ],
)
def test_invalid_parameters_are_rejected(ibrav, celldm, message):
    with pytest.raises(ValueError, match=message):
        latgen(ibrav, celldm)


def test_cell_is_a_pytree_whose_only_leaf_is_the_lattice():
    """Static metadata must not become traced, and `at` must stay traceable --
    stress is a derivative with respect to it."""
    cell = Cell.from_ibrav(2, [10.2, 0, 0, 0, 0, 0])
    leaves = jax.tree_util.tree_leaves(cell)
    assert len(leaves) == 1 and leaves[0].shape == (3, 3)

    volume = eqx.filter_jit(lambda c: c.volume)(cell)
    assert float(volume) == pytest.approx(265.302)

    # d(volume)/d(at) is what a stress calculation needs. The constructor takes
    # the array as given, so a traced lattice flows straight through.
    gradient = jax.grad(lambda at: Cell(at=at, alat=cell.alat).volume)(cell.at)
    expected = float(cell.volume) * np.linalg.inv(np.asarray(cell.at)).T
    assert np.asarray(gradient) == pytest.approx(expected, abs=1e-9)


def test_precision_policy_is_respected():
    from defumat.config import SINGLE

    assert Cell.from_ibrav(2, [10.2, 0, 0, 0, 0, 0], precision=DOUBLE).at.dtype == np.float64
    assert Cell.from_ibrav(2, [10.2, 0, 0, 0, 0, 0], precision=SINGLE).at.dtype == np.float32
