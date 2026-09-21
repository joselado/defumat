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


# --- the acceptance tolerance is QE's, not one ten times tighter -------------


def test_a_translation_written_with_six_decimals_is_a_third():
    """``0.333333`` is a crystallographic third and used to be nothing at all.

    The filter accepts a component when ``1/|residue|`` is close to an integer,
    and ``1/0.333333 = 3.000003`` sits 3.0e-6 from 3 -- inside QE's
    ``eps2 = 1e-5`` (``symm_base.f90:23``, used at ``sgam_at:557-569``) and
    three times outside the 1e-6 this used to apply. So a hexagonal cell whose
    positions are written with six decimals lost **every** non-symmorphic
    operation, and the only other candidate for a sublattice swap is the zero
    translation, which fails the structure match.
    """
    from defumat.system.symmetry import _crystallographic_translation

    exact = np.array([1.0 / 3.0, 2.0 / 3.0, 0.5])
    written = np.array([0.333333, 0.666667, 0.5])
    assert _crystallographic_translation(exact)
    assert _crystallographic_translation(written)
    # ...and a translation that is *not* a crystallographic fraction still is
    # not one, so the loosening did not turn the filter into a pass-through.
    assert not _crystallographic_translation(np.array([0.2, 0.0, 0.0]))
    assert not _crystallographic_translation(np.array([0.142857, 0.0, 0.0]))


def test_hcp_cobalt_finds_the_group_pw_x_finds(pseudo_dir):
    """Four numbers against ``pw.x``'s own header on the same input.

    ``pw.x`` on ``co-hcp-anisotropy-sr.in`` prints ``24 Sym. Ops., with
    inversion, found (12 have fractional translation)``, ``number of k points=
    6`` and a smooth FFT grid of ``(15, 15, 30)``. This read **4**, **8** and
    ``(15, 15, 25)``: the twelve operations that swap the two hcp sublattices
    need ``ft = (1/3, 2/3, 1/2)`` and the filter rejected it, and
    :meth:`Symmetries.fft_factors` then lost the factor of 3 that those
    translations carry.

    **It was a cost rather than an error**, which is worth saying because the
    audit entry forecast otherwise: the four operations still formed a group
    and its eight-point wedge is a valid sampling of the same zone, so the
    density was symmetrised over a proper subgroup and the total came out at
    -148.81512176 Ry against ``pw.x``'s -148.81512173, moving to -148.81512173
    after. What it cost is iterations, **38 against 16**.
    """
    from pathlib import Path

    from defumat.system.builder import system_from_file

    case = Path(__file__).resolve().parents[1] / "data" / "qe"
    system = system_from_file(case / "co-hcp-anisotropy-sr.in")
    group = system.symmetry_group(nosym=system.nosym)
    translations = np.asarray(group.translation_array())

    assert len(group.rotation_array()) == 24
    assert int(np.count_nonzero(np.abs(translations).sum(axis=1) > 1e-5)) == 12
    assert tuple(group.fft_factors()) == (3, 3, 2)
    assert len(np.asarray(system.kpoints.weights)) == 6


def test_the_position_tolerance_has_room_on_every_committed_cell():
    """Loosening ``_maps_structure`` cannot merge two atoms, measured.

    The risk of matching positions at 1e-5 instead of 1e-6 is a cell with two
    atoms closer than that in crystal coordinates. The closest pair anywhere in
    ``tests/data/qe`` is **0.1**, on the two hydrogen chains, which is four
    orders of magnitude of headroom -- and QE itself never goes below 1e-5
    except by halving ``accep`` when the group fails to close.
    """
    import glob
    from pathlib import Path

    from defumat.system.builder import system_from_file
    from defumat.system.symmetry import _POSITION_TOLERANCE

    case = Path(__file__).resolve().parents[1] / "data" / "qe"
    closest = np.inf
    for path in sorted(glob.glob(str(case / "*.in"))):
        try:
            system = system_from_file(path)
        except Exception:                                   # noqa: BLE001
            continue
        crystal = np.asarray(system.structure.positions_crystal(system.cell)) % 1.0
        if len(crystal) < 2:
            continue
        difference = crystal[:, None, :] - crystal[None, :, :]
        difference -= np.rint(difference)
        separation = np.abs(difference).max(axis=-1)
        np.fill_diagonal(separation, np.inf)
        closest = min(closest, float(separation.min()))
    assert closest > 1.0e-3, f"closest pair {closest:.3e} in crystal coordinates"
    assert closest / _POSITION_TOLERANCE > 1.0e3
