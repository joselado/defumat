"""Completing a wedge sum for the site ``<L>`` and ``<S>`` of a nonmagnetic spinor run.

At ``nspin = 4`` with no magnetization (``domag = .false.``) the Hamiltonian
commutes with time reversal ``T = i sigma_y K``, so every state at ``k`` has a
Kramers partner at ``-k`` with the same energy and both site vectors reversed:
the whole-zone ``<L>`` and ``<S>`` are exactly zero on every site.
``KPoints.automatic`` reduces such a grid with ``k -> -k`` unless ``noinv``,
and the spatial axial average carries no time-reversed operation for a
nonmagnetic group, so a wedge keeps the uncancelled half wherever the site
group admits an axial vector -- and at ``nsym = 1`` the function returned
before averaging at all.

The cells are chosen so the premise of each case is a fact about the builder
rather than a hope: three iodine atoms at generic positions have no symmetry
but the identity, so time reversal alone halves the grid; one iodine atom in a
triclinic cell has ``{E, I}``, under which the axial average is the identity
map, so it can neither create nor remove the zero. Every test is host-side:
the symmetrisation is called on a stand-in carrying the real ``System`` and its
group, with no SCF.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from defumat.io.pwin import parse_pw_input
from defumat.projwfc.angular_momentum import _symmetrise_axial
from defumat.system.builder import build_system
from defumat.system.kpoints import is_reduced
from defumat.system.symmetry import (
    atom_mapping,
    symmetrize_atom_cartesian_tensor,
)

pytestmark = pytest.mark.unit


# Three atoms whose pairwise distances (4.72, 4.84, 5.23 bohr) all differ, and
# none of whose separations has a crystal component of 0 or 1/2, so no
# operation of the orthorhombic lattice maps the set onto itself: nsym = 1.
_P1 = """
&control
  calculation = 'scf'
/
&system
  ibrav = 0, nat = 3, ntyp = 1, ecutwfc = 25.0,
  occupations = 'smearing', smearing = 'mv', degauss = 0.02,
  noncolin = .true., lspinorb = .true.
  {extra}
/
&electrons
/
ATOMIC_SPECIES
 I 126.9 I.rel-pbe-nc-dojo.UPF
CELL_PARAMETERS bohr
 8.00 0.00 0.00
 0.00 8.40 0.00
 0.00 0.00 8.90
ATOMIC_POSITIONS crystal
 I 0.00 0.00 0.00
 I 0.45 0.16 0.33
 I 0.17 0.60 0.66
K_POINTS automatic
 3 3 3 0 0 0
"""

# One atom in a triclinic cell: the group is {E, I}, and inversion leaves any
# axial vector alone, so the spatial average returns the wedge sum as given.
_TRICLINIC = """
&control
  calculation = 'scf'
/
&system
  ibrav = 0, nat = 1, ntyp = 1, ecutwfc = 25.0,
  occupations = 'smearing', smearing = 'mv', degauss = 0.02,
  noncolin = .true., lspinorb = .true.
  {extra}
/
&electrons
/
ATOMIC_SPECIES
 I 126.9 I.rel-pbe-nc-dojo.UPF
CELL_PARAMETERS bohr
 5.10 0.00 0.00
 0.70 5.60 0.00
 0.40 0.90 6.20
ATOMIC_POSITIONS crystal
 I 0.0 0.0 0.0
K_POINTS automatic
 4 4 4 0 0 0
"""

_MAGNETIC = "starting_magnetization(1) = 0.5"

# A wedge residue of the size a heavy p shell with spin-orbit coupling leaves.
_ORBITAL = np.array([[0.03, -0.02, 0.05], [-0.01, 0.04, 0.02], [0.02, 0.01, -0.03]])
_SPIN = np.array([[0.10, -0.20, 0.15], [-0.05, 0.12, 0.08], [0.07, 0.03, -0.11]])
_CHARGE = np.array([7.0, 6.9, 7.1])


def _cell(template, extra=""):
    return build_system(parse_pw_input(template.format(extra=extra)))


def _standin(system):
    """What ``Calculation`` carries into ``_symmetrise_axial``, and nothing else."""
    symmetries = system.symmetry_group()
    return SimpleNamespace(
        system=system,
        symmetries=symmetries,
        use_symmetry=bool(not system.nosym and symmetries.nsym > 1),
        nspin=system.nspin,
    )


def _axial_average(vectors, calculation):
    system = calculation.system
    mapping = atom_mapping(system.cell, system.structure, calculation.symmetries)
    return symmetrize_atom_cartesian_tensor(
        vectors, system.cell, calculation.symmetries, mapping, axial=True
    )


def test_a_nonmagnetic_spinor_wedge_with_no_symmetry_is_completed_to_zero():
    """P1: time reversal alone halved the grid, and nothing averages it.

    27 points are Gamma and thirteen Kramers pairs, so the wedge keeps 14 and
    each of the thirteen stands in for a partner whose ``<L>`` and ``<S>`` are
    minus its own. The whole-zone value is zero, and that is what comes back.
    """
    system = _cell(_P1)
    calculation = _standin(system)
    assert system.nspin == 4 and system.nspin_mag == 1, (
        "the premise: a spinor run with no magnetization"
    )
    assert calculation.symmetries.nsym == 1, "the premise: the group is {E}"
    assert is_reduced(system.kpoints) and system.kpoints.nk == 14, (
        "the premise: time reversal alone halved the grid"
    )

    orbital_out, spin_out, charge_out = _symmetrise_axial(
        _ORBITAL, _SPIN, _CHARGE, calculation
    )
    assert orbital_out.shape == _ORBITAL.shape and spin_out.shape == _SPIN.shape
    np.testing.assert_array_equal(orbital_out, 0.0)
    np.testing.assert_array_equal(spin_out, 0.0)
    # The charge is even under time reversal and there is no group to average it.
    np.testing.assert_allclose(charge_out, _CHARGE, atol=1e-14)


def test_a_nonmagnetic_spinor_wedge_is_zeroed_before_the_spatial_average():
    """``{E, I}``: the axial average is the identity, so it cannot complete this.

    The wedge was cut by inversion and time reversal together, both of which
    send ``k`` to ``-k``; the nonmagnetic group has no ``t_rev``, so the average
    returns the wedge sum as given and only the Kramers argument removes it.
    """
    system = _cell(_TRICLINIC)
    calculation = _standin(system)
    assert system.nspin_mag == 1, "the premise: no magnetization"
    assert calculation.symmetries.nsym == 2, "the premise: the group is {E, I}"
    assert calculation.use_symmetry
    assert is_reduced(system.kpoints), "the premise: the k-set is a wedge"

    orbital = _ORBITAL[:1]
    spin = _SPIN[:1]
    # The premise that makes this the case: the average alone keeps the vector.
    np.testing.assert_allclose(_axial_average(orbital, calculation), orbital, atol=1e-14)

    orbital_out, spin_out, charge_out = _symmetrise_axial(
        orbital, spin, _CHARGE[:1], calculation
    )
    np.testing.assert_allclose(orbital_out, 0.0, atol=1e-14)
    np.testing.assert_allclose(spin_out, 0.0, atol=1e-14)
    np.testing.assert_allclose(charge_out, _CHARGE[:1], atol=1e-14)


def test_a_magnetic_spinor_run_keeps_its_moments():
    """``domag``: the grid was reduced without ``k -> -k`` and nothing is zeroed.

    Two cases. The triclinic atom with a moment is a wedge the builder really
    produces -- the magnetic group keeps inversion, which halves the grid on its
    own -- and it must come back as the axial average. The P1 cell with a moment
    is not reduced at all by the builder, so its k-set is replaced with the
    nonmagnetic wedge of the first test: the magnetization is then the only
    thing separating it from that test, and the vectors must come back as given.
    """
    system = _cell(_TRICLINIC, _MAGNETIC)
    calculation = _standin(system)
    assert system.domag and system.nspin_mag == 4, "the premise: a magnetic run"
    assert calculation.symmetries.nsym == 2 and is_reduced(system.kpoints)
    orbital, spin = _ORBITAL[:1], _SPIN[:1]
    orbital_out, spin_out, _ = _symmetrise_axial(
        orbital, spin, _CHARGE[:1], calculation
    )
    for got, given in ((orbital_out, orbital), (spin_out, spin)):
        np.testing.assert_allclose(
            got, _axial_average(given, calculation), atol=1e-14
        )
        np.testing.assert_allclose(got, given, atol=1e-14)

    magnetic = _cell(_P1, _MAGNETIC)
    assert magnetic.domag
    assert not is_reduced(magnetic.kpoints) and magnetic.kpoints.nk == 27, (
        "the premise: a magnetic spinor run is not reduced with k -> -k"
    )
    grafted = dataclasses.replace(magnetic, kpoints=_cell(_P1).kpoints)
    calculation = _standin(grafted)
    assert calculation.symmetries.nsym == 1 and is_reduced(grafted.kpoints)
    orbital_out, spin_out, charge_out = _symmetrise_axial(
        _ORBITAL, _SPIN, _CHARGE, calculation
    )
    np.testing.assert_array_equal(orbital_out, _ORBITAL)
    np.testing.assert_array_equal(spin_out, _SPIN)
    np.testing.assert_array_equal(charge_out, _CHARGE)


def test_noinv_leaves_a_nonmagnetic_spinor_wedge_to_the_spatial_average():
    """Under ``noinv`` the wedge was cut by the rotations alone.

    Inversion still halves the triclinic grid, but time reversal did not take
    part, so the spatial axial average is the whole completion and a vanishing
    moment stays a measured number rather than a zero written in.
    """
    system = _cell(_TRICLINIC, "noinv = .true.")
    calculation = _standin(system)
    assert system.nspin_mag == 1 and system.noinv
    assert is_reduced(system.kpoints), "the premise: inversion alone halved it"
    orbital, spin = _ORBITAL[:1], _SPIN[:1]
    orbital_out, spin_out, _ = _symmetrise_axial(
        orbital, spin, _CHARGE[:1], calculation
    )
    for got, given in ((orbital_out, orbital), (spin_out, spin)):
        np.testing.assert_allclose(
            got, _axial_average(given, calculation), atol=1e-14
        )


def test_the_full_grid_under_nosym_is_left_as_measured():
    """``nosym`` gives the whole grid here, so there is nothing to complete.

    ``pw.x``'s ``nosym`` still halves an automatic grid with ``k -> -k``; this
    code's gives all 27 points, and the site vectors are then the measured
    cancellation between each state and its Kramers partner.
    """
    system = _cell(_P1, "nosym = .true.")
    calculation = _standin(system)
    assert not is_reduced(system.kpoints) and system.kpoints.nk == 27
    orbital_out, spin_out, charge_out = _symmetrise_axial(
        _ORBITAL, _SPIN, _CHARGE, calculation
    )
    np.testing.assert_array_equal(orbital_out, _ORBITAL)
    np.testing.assert_array_equal(spin_out, _SPIN)
    np.testing.assert_array_equal(charge_out, _CHARGE)
