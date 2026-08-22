"""P27: Grimme's D2 dispersion correction, checked without an SCF.

The correction is a function of the nuclei and of nothing else, so almost all of
it can be tested with no electronic structure at all -- which is why this is a
unit test and not a regression one. The reference for the energy is QE's own
committed benchmark for ``test-suite/pw_vdw/vdw-d2.in``: ``pw.x`` prints
``Dispersion Correction`` as a line of its energy decomposition, and that number
is a property of the geometry, the coefficient table and ``london_s6`` alone.

What needs an SCF -- that the correction reaches the total energy, the force and
the stress of a real run, and that it does *not* reach the density -- is in
``tests/regression/test_dispersion.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pypresso.io.pwin import parse_pw_input
from pypresso.io.qeref import read_qe_output
from pypresso.system.builder import build_system
from pypresso.system.cell import Cell, latgen
from pypresso.system.elements import atomic_number, element_symbol
from pypresso.system.structure import Species, Structure
from pypresso.vdw import (
    D2_COEFFICIENTS,
    build_grimme_d2,
    build_vdw_correction,
    canonical_vdw_corr,
)
from pypresso.vdw.analytic import dispersion_force, dispersion_stress

pytestmark = pytest.mark.unit

#: ``test-suite/pw_vdw/vdw-d2.in``: graphite, ``ibrav = 4``, four atoms.
GRAPHITE_CELLDM = [4.66, 0.0, 2.60, 0.0, 0.0, 0.0]
GRAPHITE_CRYSTAL = np.array([
    [0.0, 1.0, 0.75],
    [2.0 / 3.0, 1.0 / 3.0, 0.75],
    [0.0, 1.0, 0.25],
    [1.0 / 3.0, 2.0 / 3.0, 0.25],
])


def _graphite(rcut=200.0, **options):
    at = latgen(4, GRAPHITE_CELLDM)
    cell = Cell.from_vectors(at, alat=GRAPHITE_CELLDM[0])
    structure = Structure(
        positions=jnp.asarray(GRAPHITE_CRYSTAL @ np.asarray(cell.at)),
        types=(0, 0, 0, 0),
        species=(Species(name="C", mass=12.0, pseudo_file="C.UPF"),),
    )
    return cell, structure, build_grimme_d2(cell, structure, rcut=rcut, **options)


def test_the_energy_matches_quantum_espressos_own_benchmark(committed_benchmark):
    """``energy_london`` on QE's graphite test case, to the digits it prints.

    The whole of the reference: a cell, four atoms, a table and ``london_s6``.
    No pseudopotential is read and no SCF is run, so a disagreement here is a
    disagreement about the *definition* rather than about the physics -- the
    coefficient table, the damping function, the ``1/2``, or which images the
    sum runs over.
    """
    reference = read_qe_output(committed_benchmark("pw_vdw", "vdw-d2.in"))
    _, structure, dispersion = _graphite()
    energy = float(dispersion.energy(structure.positions))
    assert energy == pytest.approx(reference.energy_terms["dispersion"], abs=1e-8)


def test_the_tabulated_coefficients_match_what_pw_x_prints(committed_benchmark):
    """``print_london``'s table, for the one element the benchmark uses."""
    text = committed_benchmark("pw_vdw", "vdw-d2.in").read_text()
    line = [l for l in text.splitlines() if l.strip().startswith("C ")][0]
    radius, c6 = (float(x) for x in line.split()[1:3])
    assert D2_COEFFICIENTS[atomic_number("C") - 1] == (c6, radius)


def test_the_sum_does_not_notice_a_lattice_translation():
    """The neighbour list is built for the cell, not for one pair's separation.

    ``rgen`` folds each pair's separation into the cell at the origin and builds
    its images around *that*; here one list of lattice translations serves every
    pair and the distance mask does the rest. Folding permutes which translation
    supplies which image and changes no separation, so moving an atom by a
    lattice vector must change nothing at all -- which is also the property that
    lets the same list survive a relaxation.
    """
    cell, structure, dispersion = _graphite(rcut=45.0)
    before = float(dispersion.energy(structure.positions))
    shifted = np.asarray(structure.positions).copy()
    shifted[1] += np.asarray(cell.at)[0] - 2.0 * np.asarray(cell.at)[2]
    assert float(dispersion.energy(jnp.asarray(shifted))) == pytest.approx(
        before, abs=1e-12
    )


def test_the_force_is_the_gradient_of_the_energy():
    """``force_london`` against ``-grad`` of the energy this code differentiates.

    The two share the coefficients and the neighbour list and nothing else: one
    is JAX's reverse mode through a broadcast kernel, the other is QE's
    hand-derived expression transcribed. It is the check that would catch the
    sign of the separation vector, which ``rgen`` returns the other way round.
    """
    _, structure, dispersion = _graphite(rcut=45.0)
    positions = _displaced(structure)
    autodiff = -jax.grad(dispersion.energy)(positions)
    transcribed = dispersion_force(dispersion, positions)
    assert np.abs(np.asarray(autodiff - transcribed)).max() < 1e-14
    # ...and it is not a comparison of two zeros.
    assert np.abs(np.asarray(transcribed)).max() > 1e-5


def test_the_stress_is_the_strain_derivative_of_the_energy():
    """``stres_london`` against ``-(1/Omega) grad`` in the cell coordinate.

    The lattice translations deform with the cell, which is the whole of the
    strain dependence: the sum has no basis, no grid and no reciprocal space, so
    unlike every other term in the stress there is nothing frozen here and no
    Pulay error.
    """
    cell, structure, dispersion = _graphite(rcut=45.0)
    positions = _displaced(structure)
    volume = float(cell.volume)

    def energy(strain):
        deformation = jnp.eye(3) + strain
        return dispersion.at_cell(deformation).energy(positions @ deformation.T)

    autodiff = -jax.grad(energy)(jnp.zeros((3, 3))) / volume
    transcribed = dispersion_stress(dispersion, positions, volume)
    assert np.abs(np.asarray(autodiff - transcribed)).max() < 1e-16
    assert np.abs(np.asarray(transcribed)).max() > 1e-6


def _displaced(structure):
    """The graphite geometry with the symmetry broken, so nothing is zero."""
    generator = np.random.default_rng(0)
    return jnp.asarray(
        np.asarray(structure.positions) + 0.1 * generator.standard_normal((4, 3))
    )


def test_the_correction_is_second_differentiable():
    """It has to be: the elastic constants are a ``jvp`` of the stress (P26).

    A pair sum masked by distance is exactly where a second derivative turns
    into NaN, because ``sqrt`` of a masked zero has an infinite derivative and
    ``0 * inf`` survives one differentiation to appear in the next.
    """
    _, structure, dispersion = _graphite(rcut=45.0)
    positions = _displaced(structure)

    def energy(strain):
        deformation = jnp.eye(3) + strain
        return dispersion.at_cell(deformation).energy(positions @ deformation.T)

    _, second = jax.jvp(jax.grad(energy), (jnp.zeros((3, 3)),), (jnp.eye(3),))
    assert np.all(np.isfinite(np.asarray(second)))
    assert np.abs(np.asarray(second)).max() > 1e-4


def test_a_species_may_override_the_table():
    """``london_c6`` and ``london_rvdw``, with QE's two different sentinels.

    ``init_london`` accepts a *zero* ``C6`` -- a species taken out of the
    correction -- and does not accept a zero radius, which would divide by zero
    in the damping. The two tests are written differently in the Fortran and
    the difference is deliberate, so it is honoured here.
    """
    cell, structure, _ = _graphite(rcut=30.0)
    tabulated = build_grimme_d2(cell, structure, rcut=30.0)
    zeroed = build_grimme_d2(cell, structure, rcut=30.0, c6=[0.0])
    assert float(zeroed.energy(structure.positions)) == 0.0

    doubled = build_grimme_d2(
        cell, structure, rcut=30.0, c6=[2.0 * D2_COEFFICIENTS[5][0]]
    )
    assert float(doubled.energy(structure.positions)) == pytest.approx(
        2.0 * float(tabulated.energy(structure.positions)), rel=1e-12
    )
    # The -1 sentinel in either array means "use the table".
    sentinel = build_grimme_d2(cell, structure, rcut=30.0, c6=[-1.0], rvdw=[-1.0])
    assert float(sentinel.energy(structure.positions)) == pytest.approx(
        float(tabulated.energy(structure.positions)), rel=1e-14
    )


def test_the_scaling_factor_is_linear():
    _, structure, half = _graphite(rcut=30.0, s6=0.375)
    _, _, whole = _graphite(rcut=30.0, s6=0.75)
    assert 2.0 * float(half.energy(structure.positions)) == pytest.approx(
        float(whole.energy(structure.positions)), rel=1e-14
    )


def test_the_cutoff_converges_from_below():
    """A ``1/r^6`` lattice sum truncates with an error falling as ``1/rcut^3``.

    Recorded as a test because it is the justification for QE's default of 200
    bohr, which looks extravagant until the shell count is taken into account.
    """
    _, structure, _ = _graphite()
    energies = [
        float(_graphite(rcut=rcut)[2].energy(structure.positions))
        for rcut in (30.0, 60.0, 120.0)
    ]
    # Monotonically more negative, and each step gains about 1/8 of the last.
    assert energies[0] > energies[1] > energies[2]
    gained = [energies[0] - energies[1], energies[1] - energies[2]]
    assert 4.0 < gained[0] / gained[1] < 16.0


# -- the registry and the input boundary --------------------------------------


@pytest.mark.parametrize(
    "spelling", ["grimme-d2", "Grimme-D2", "DFT-D", "dft-d", " GRIMME-D2 "]
)
def test_every_spelling_set_vdw_corr_accepts_reaches_the_same_correction(spelling):
    assert canonical_vdw_corr(spelling) == "grimme-d2"


@pytest.mark.parametrize(
    "name,reason",
    [("grimme-d3", "coordination number"), ("TS", "Hirshfeld"),
     ("MBD", "coupled-oscillator"), ("xdm", "exchange hole")],
)
def test_the_unimplemented_corrections_are_refused_by_name(name, reason):
    """``set_vdw_corr`` warns and runs on with none; here it stops.

    An input asking for D3 and silently getting plain PBE is a 30 meV error on a
    layered crystal with nothing in the output that looks like an error.
    """
    with pytest.raises(NotImplementedError, match=reason):
        build_vdw_correction(name, None, None)


def test_an_unknown_vdw_corr_is_an_error():
    with pytest.raises(ValueError, match="unknown vdw_corr"):
        canonical_vdw_corr("grimme-d4")


def test_no_correction_is_the_default():
    assert canonical_vdw_corr(None) == "none"
    assert canonical_vdw_corr("none") == "none"
    assert build_vdw_correction("none", None, None) is None


def test_a_species_with_no_tabulated_coefficients_is_refused():
    """The table stops at Z = 86 and QE's ``init_london`` stops there too."""
    cell, structure, _ = _graphite(rcut=10.0)
    thorium = Structure(
        positions=structure.positions,
        types=structure.types,
        species=(Species(name="Th", mass=232.0, pseudo_file="Th.UPF"),),
    )
    with pytest.raises(NotImplementedError, match="Z = 90"):
        build_grimme_d2(cell, thorium, rcut=10.0)


INPUT = """
&control
/
&system
  ibrav = 4, celldm(1) = 4.66, celldm(3) = 2.6, nat = 1, ntyp = 1, ecutwfc = 20
  {extra}
/
ATOMIC_SPECIES
 C 12.0 C.UPF
ATOMIC_POSITIONS crystal
 C 0.0 0.0 0.0
K_POINTS gamma
"""


def test_the_input_variables_are_read():
    system = build_system(parse_pw_input(INPUT.format(
        extra="vdw_corr = 'grimme-d2', london_s6 = 0.7, london_rcut = 60,"
              " london_c6(1) = 55.0, london_rvdw(1) = 3.0"
    )))
    assert system.vdw_corr == "grimme-d2"
    assert system.london_s6 == 0.7
    assert system.london_rcut == 60.0
    assert system.london_c6 == (55.0,)
    assert system.london_rvdw == (3.0,)


def test_the_obsolescent_london_switch_still_works():
    """``input.f90`` still honours ``london = .true.``; so does this."""
    assert build_system(parse_pw_input(
        INPUT.format(extra="london = .true.")
    )).vdw_corr == "grimme-d2"


def test_no_vdw_corr_means_none():
    system = build_system(parse_pw_input(INPUT.format(extra="")))
    assert system.vdw_corr == "none"
    assert system.london_s6 == 0.75
    assert system.london_rcut == 200.0


# -- the element table --------------------------------------------------------


@pytest.mark.parametrize(
    "label,symbol",
    [("C", "C"), ("c", "C"), ("Si", "Si"), ("SI", "Si"), ("C1", "C"),
     ("Fe_up", "Fe"), ("Fe-dw", "Fe"), ("O2", "O")],
)
def test_a_species_label_resolves_to_its_element(label, symbol):
    """``atomic_number.f90``'s rules: a digit, ``-`` or ``_`` ends the symbol."""
    assert element_symbol(label) == symbol


def test_a_label_that_names_no_element_is_an_error():
    with pytest.raises(ValueError):
        atomic_number("Xx")
