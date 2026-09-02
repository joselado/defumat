"""Space-group detection and density symmetrisation."""

import numpy as np
import pytest

from defumat.basis.gvectors import generate_gvectors
from defumat.system.cell import Cell
from defumat.system.structure import Species, Structure
from defumat.system.symmetry import (
    find_symmetries,
    lattice_point_group,
    symmetrize_density,
)

pytestmark = pytest.mark.unit

SILICON_CELL = Cell.from_ibrav(2, [10.2, 0, 0, 0, 0, 0])
SI = (Species(name="Si", mass=28.086, pseudo_file="Si.pz-vbc.UPF"),)


def _diamond():
    positions = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]) * SILICON_CELL.alat
    return Structure.from_card_units(positions / SILICON_CELL.alat, [0, 0], SI, "alat", SILICON_CELL)


@pytest.mark.parametrize(
    ("ibrav", "celldm", "order"),
    [
        (1, [5.0, 0, 0, 0, 0, 0], 48),  # simple cubic
        (2, [10.2, 0, 0, 0, 0, 0], 48),  # fcc
        (3, [5.0, 0, 0, 0, 0, 0], 48),  # bcc
        (4, [5.0, 0, 1.6, 0, 0, 0], 24),  # hexagonal
        (6, [5.0, 0, 1.6, 0, 0, 0], 16),  # tetragonal
        (8, [5.0, 1.3, 1.7, 0, 0, 0], 8),  # orthorhombic
        (14, [5.0, 1.3, 1.7, 0.1, 0.2, 0.3], 2),  # triclinic: identity + inversion
    ],
)
def test_lattice_point_group_orders(ibrav, celldm, order):
    """The order of a Bravais lattice's point group is fixed by its symmetry."""
    cell = Cell.from_ibrav(ibrav, celldm)
    assert len(lattice_point_group(np.asarray(cell.at))) == order


def test_point_group_elements_are_orthogonal_transformations():
    cell = Cell.from_ibrav(2, [10.2, 0, 0, 0, 0, 0])
    at = np.asarray(cell.at)
    metric = at @ at.T
    for rotation in lattice_point_group(at):
        assert rotation.dtype == int
        assert abs(abs(np.linalg.det(rotation)) - 1.0) < 1e-9
        # The metric is preserved: this is what makes it an isometry.
        assert rotation @ metric @ rotation.T == pytest.approx(metric, abs=1e-9)


def test_diamond_silicon_has_48_operations_and_needs_fractional_translations():
    """Diamond is non-symmorphic: half its operations carry a translation.

    A search that only looks for symmorphic operations finds 24 of the 48, and
    one that transposes the rotation convention finds 12.
    """
    symmetries = find_symmetries(SILICON_CELL, _diamond())
    assert symmetries.nsym == 48
    assert not symmetries.symmorphic

    translations = symmetries.translation_array()
    assert np.any(np.linalg.norm(translations, axis=1) > 1e-6)


def test_a_displaced_atom_lowers_the_symmetry():
    positions = np.array([[0.0, 0.0, 0.0], [0.30, 0.25, 0.25]])
    structure = Structure.from_card_units(positions, [0, 0], SI, "alat", SILICON_CELL)
    assert find_symmetries(SILICON_CELL, structure).nsym < 48


def test_symmetrization_is_a_projection():
    """Symmetrising twice must change nothing the second time."""
    gvectors = generate_gvectors(SILICON_CELL, 48.0)
    symmetries = find_symmetries(SILICON_CELL, _diamond())

    rng = np.random.default_rng(0)
    rho = rng.normal(size=gvectors.ngm) + 1j * rng.normal(size=gvectors.ngm)

    once = symmetrize_density(rho, gvectors, symmetries)
    twice = symmetrize_density(once, gvectors, symmetries)
    assert np.asarray(twice) == pytest.approx(np.asarray(once), abs=1e-12)


def test_symmetrization_preserves_the_average_and_reduces_the_norm():
    gvectors = generate_gvectors(SILICON_CELL, 48.0)
    symmetries = find_symmetries(SILICON_CELL, _diamond())

    rng = np.random.default_rng(1)
    rho = rng.normal(size=gvectors.ngm) + 1j * rng.normal(size=gvectors.ngm)
    symmetrized = np.asarray(symmetrize_density(rho, gvectors, symmetries))

    # G = 0 is invariant under every operation, so the total charge is untouched.
    assert symmetrized[0] == pytest.approx(rho[0])
    # Averaging over a group can only remove components, never add them.
    assert np.linalg.norm(symmetrized) <= np.linalg.norm(rho) + 1e-12


def test_a_symmetric_density_is_unchanged():
    """The structure factor of the crystal is symmetric by construction."""
    from defumat.pseudo.potentials import structure_factors

    gvectors = generate_gvectors(SILICON_CELL, 48.0)
    structure = _diamond()
    symmetries = find_symmetries(SILICON_CELL, structure)

    factors = structure_factors(structure, SILICON_CELL, gvectors)[0]
    symmetrized = symmetrize_density(factors, gvectors, symmetries)
    assert np.asarray(symmetrized) == pytest.approx(np.asarray(factors), abs=1e-10)


def test_symmetry_group_honours_the_nosym_it_takes():
    """The parameter used to be accepted, assigned and never read.

    ``symmetry_group(nosym=True)`` returned the full group and said nothing, so
    a caller asking for ``setup.f90``'s ``nsym = 1`` silently got 48. It
    defaults to ``False`` rather than to ``system.nosym`` on purpose: the group
    is a property of the crystal, and ``basis.builder`` needs the fractional
    translations to size the FFT box whatever the input said about symmetrising.
    """
    from defumat.io.pwin import parse_pw_input
    from defumat.system.builder import build_system

    system = build_system(parse_pw_input(
        "&system\n ibrav=2, celldm(1)=10.2, nat=2, ntyp=1, ecutwfc=12.0,"
        " nosym = .true.\n/\n"
        "ATOMIC_SPECIES\n Si 28.086 Si.pz-vbc.UPF\n"
        "ATOMIC_POSITIONS alat\n Si 0 0 0\n Si 0.25 0.25 0.25\n"
        "K_POINTS gamma\n"
    ))
    assert system.nosym is True
    assert system.symmetry_group().nsym == 48
    trivial = system.symmetry_group(nosym=True)
    assert trivial.nsym == 1
    assert trivial.symmorphic
    assert np.array_equal(trivial.rotation_array()[0], np.eye(3, dtype=int))
