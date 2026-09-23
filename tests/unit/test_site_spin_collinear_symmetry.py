"""Completing a wedge sum for the site ``<S>`` and ``<L>`` in a collinear run.

A collinear run has no spin-orbit coupling, so a spatial operation moves the
orbitals and leaves the spin alone: ``<S_z>`` on a site is a **scalar** that
the group permutes, and the collinear group is built on exactly that rule
(``collinear_symmetries``). Rotating ``(0, 0, S_z)`` as an axial vector under
that group averages it against every operation that turns ``z`` over, and on
bcc iron's 48 it returned ``(0, 0, 0)`` for a ferromagnet.

``<L>`` is the other half. Each collinear channel is a spinless problem with a
real Hamiltonian, so ``<L>_{-k} = -<L>_k`` and the whole-zone value is exactly
zero; a wedge reduced with ``k -> -k`` keeps the uncancelled half, which a
spatial average does not remove on a low-symmetry cell.

Every test here is host-side: the symmetrisation is called directly on a
stand-in carrying the real ``System`` and its group, with no SCF.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from defumat.projwfc.angular_momentum import _symmetrise_axial
from defumat.system.builder import build_system, system_from_file
from defumat.io.pwin import parse_pw_input
from defumat.system.kpoints import is_reduced
from defumat.system.symmetry import (
    atom_mapping,
    symmetrize_atom_cartesian_tensor,
)

pytestmark = pytest.mark.unit

QE = Path(__file__).resolve().parents[1] / "data" / "qe"

# One atom in a triclinic cell: the group is {E, I}, under which any axial
# vector is invariant, so a spatial average leaves a wedge residue of <L>
# exactly where it was.
_TRICLINIC = """
&control
  calculation = 'scf'
/
&system
  ibrav = 0, nat = 1, ntyp = 1, ecutwfc = 20.0,
  occupations = 'smearing', smearing = 'gaussian', degauss = 0.02,
  nspin = {nspin}, starting_magnetization(1) = 0.5
  {extra}
/
&electrons
/
ATOMIC_SPECIES
 Fe 55.847 Fe.pz-nd-rrkjus.UPF
CELL_PARAMETERS bohr
 5.10 0.00 0.00
 0.70 5.60 0.00
 0.40 0.90 6.20
ATOMIC_POSITIONS crystal
 Fe 0.0 0.0 0.0
K_POINTS automatic
 4 4 4 0 0 0
"""


def _standin(system, nspin):
    symmetries = system.symmetry_group()
    return SimpleNamespace(
        system=system,
        symmetries=symmetries,
        use_symmetry=bool(not system.nosym and symmetries.nsym > 1),
        nspin=nspin,
    )


def _triclinic(nspin, extra=""):
    text = _TRICLINIC.format(nspin=nspin, extra=extra)
    return build_system(parse_pw_input(text))


def test_a_collinear_ferromagnet_keeps_its_site_spin_on_the_wedge():
    """bcc iron at ``nspin = 2``: ``S_z = 1.1`` comes back as ``1.1``.

    The group is checked first, because a group that came back smaller than 48
    would leave fewer operations to turn ``z`` over and the old axial average
    could then pass for the wrong reason.
    """
    system = system_from_file(QE / "fe-bcc-sfac.in")
    calculation = _standin(system, nspin=2)
    assert calculation.symmetries.nsym == 48
    assert not calculation.symmetries.magnetic
    assert calculation.use_symmetry

    spin = np.array([[0.0, 0.0, 1.1]])
    orbital = np.zeros((1, 3))
    charge = np.array([7.9])
    orbital_out, spin_out, charge_out = _symmetrise_axial(
        orbital, spin, charge, calculation
    )
    np.testing.assert_allclose(spin_out, [[0.0, 0.0, 1.1]], atol=1e-14)
    np.testing.assert_allclose(orbital_out, 0.0, atol=1e-14)
    np.testing.assert_allclose(charge_out, [7.9], atol=1e-14)


def test_the_collinear_orbital_moment_on_a_time_reversal_wedge_is_zero():
    """A triclinic collinear magnet: the wedge residue of ``<L>`` is completed.

    ``{E, I}`` leaves any axial vector alone, so an axial average returns the
    wedge sum untouched; the half of the zone ``k -> -k`` removed carries
    minus it, and the whole-zone ``<L>`` is zero. ``S_z``, which is even under
    ``k -> -k`` in a collinear channel, is untouched.
    """
    system = _triclinic(nspin=2)
    calculation = _standin(system, nspin=2)
    assert calculation.symmetries.nsym == 2, "the premise: the group is {E, I}"
    assert is_reduced(system.kpoints), "the premise: the k-set is a wedge"

    orbital = np.array([[0.03, -0.02, 0.05]])
    spin = np.array([[0.0, 0.0, 1.2]])
    orbital_out, spin_out, _ = _symmetrise_axial(
        orbital, spin, np.array([8.0]), calculation
    )
    np.testing.assert_allclose(orbital_out, 0.0, atol=1e-14)
    np.testing.assert_allclose(spin_out, spin, atol=1e-14)


def test_noinv_leaves_the_orbital_moment_to_the_spatial_average():
    """Under ``noinv`` the wedge was cut by the rotations alone.

    Nothing then removed ``-k``, so the spatial axial average is the whole
    completion and a quenched ``<L>`` stays a measured number rather than a
    zero written in.
    """
    system = _triclinic(nspin=2, extra="noinv = .true.")
    calculation = _standin(system, nspin=2)
    orbital = np.array([[0.03, -0.02, 0.05]])
    orbital_out, _, _ = _symmetrise_axial(
        orbital, np.zeros((1, 3)), np.array([8.0]), calculation
    )
    np.testing.assert_allclose(orbital_out, orbital, atol=1e-14)


def test_the_spinor_path_is_the_axial_average_it_was():
    """``nspin = 4``: both vectors are axial and nothing is zeroed.

    The same stand-ins with ``nspin = 4`` must give exactly what
    ``symmetrize_atom_cartesian_tensor(axial=True)`` gives, on bcc (where it
    kills a vector) and on the triclinic wedge (where it keeps one), so the
    time-reversal completion above cannot leak into the spinor regime.
    """
    for system in (system_from_file(QE / "fe-bcc-sfac.in"), _triclinic(nspin=2)):
        calculation = _standin(system, nspin=4)
        mapping = atom_mapping(system.cell, system.structure, calculation.symmetries)
        orbital = np.array([[0.03, -0.02, 0.05]])
        spin = np.array([[0.1, 0.2, 1.1]])
        orbital_out, spin_out, _ = _symmetrise_axial(
            orbital, spin, np.array([8.0]), calculation
        )
        for got, given in ((orbital_out, orbital), (spin_out, spin)):
            expected = symmetrize_atom_cartesian_tensor(
                given, system.cell, calculation.symmetries, mapping, axial=True
            )
            np.testing.assert_allclose(got, expected, atol=1e-14)
