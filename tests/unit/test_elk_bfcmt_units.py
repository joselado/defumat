"""Elk's per-atom field ``bfcmt`` is a field coupled through ``cb``, not a Hartree energy.

``genbs.f90`` adds ``cb * (bfcmt + bfieldc)`` to the Kohn-Sham magnetic field,
with ``cb = gfacte/(4 solsc)``, so ``bfcmt`` is converted exactly as ``bfieldc``
is (``PLAN.md`` P86): ``2 cb`` in magnitude, and with the sign flip between
Elk's ``+cb sigma.B`` and this code's ``-B.m``. Reading it as a Hartree energy,
a factor of two alone, is 273.75 times too large and has the wrong sign.

The expected values are built here from Elk's own constants
(``modmain.f90:1238`` and ``:1264``) rather than from the module's, so a wrong
constant in the reader cannot pass by agreeing with itself. Host-side only.
"""

from __future__ import annotations

import numpy as np
import pytest

from defumat.io.elk import read_elk_geometry

#: ``modmain.f90``, both declared ``parameter``; ``solscf`` defaults to 1.
GFACTE = 2.00231930436256
SOL = 137.035999084
CB = GFACTE / (4.0 * SOL)


def _geometry(tmp_path, rows):
    """A ``GEOMETRY.OUT`` with one species and one atom line per row of ``rows``."""
    lines = "".join(
        f"  {0.1 * i:.3f} 0.0 0.0  {b[0]} {b[1]} {b[2]}\n" for i, b in enumerate(rows)
    )
    path = tmp_path / "GEOMETRY.OUT"
    path.write_text(
        "avec\n  6.0 0.0 0.0\n  0.0 6.0 0.0\n  0.0 0.0 6.0\n\n"
        f"atoms\n  1 : nspecies\n'Fe.in'\n  {len(rows)} : natoms; atpos, bfcmt below\n"
        + lines
    )
    return read_elk_geometry(path)


def test_cb_is_the_number_the_record_quotes():
    """``cb = 3.6529e-3`` and ``1/cb = 273.75``, the figures P86 found."""
    assert CB == pytest.approx(3.6529e-3, rel=1e-4)
    assert 1.0 / CB == pytest.approx(273.75, abs=0.01)


def test_the_per_atom_field_gives_elks_zeeman_splitting(tmp_path):
    """The up-down splitting of the converted field is Elk's, in size and in sign.

    Elk's term ``+cb sigma.B`` in Hartree raises the spin-up channel by
    ``cb B_z`` and lowers spin-down by the same, a splitting
    ``e_up - e_down = 2 cb B_z`` Ha = ``4 cb B_z`` Ry. This code's energy is
    ``-B.m``, so its field lowers spin-up by ``B_z`` and the splitting is
    ``-2 B_z`` Ry. Equal splittings is ``B_defumat = -2 cb B_Elk``.
    """
    b_elk = 0.5
    geometry = _geometry(tmp_path, [(0.0, 0.0, b_elk)])
    b = geometry.magnetic_fields
    assert b.shape == (1, 3)

    splitting_elk_ry = 2.0 * (2.0 * CB * b_elk)
    splitting_here_ry = -2.0 * b[0, 2]
    assert splitting_here_ry == pytest.approx(splitting_elk_ry, rel=1e-12)
    assert b[0, :2] == pytest.approx([0.0, 0.0], abs=1e-15)


def test_the_zeeman_energy_of_any_moment_is_elks(tmp_path):
    """``-B_defumat . m`` equals Elk's ``cb m.B`` for every component and atom.

    ``energy.f90:100`` writes the field energy as ``cb momtot.bfieldc`` in
    Hartree, and ``bfcmt`` enters the same Kohn-Sham field through the same
    ``cb``. The check runs over two atoms and a field with all three components
    nonzero and of both signs, so a conversion applied to one component, or a
    sign that depends on the direction, fails it.
    """
    fields_elk = np.array([[0.3, -1.2, 0.7], [-0.4, 0.25, -2.0]])
    geometry = _geometry(tmp_path, fields_elk)
    moments = np.array([[1.1, 0.2, -0.6], [-0.3, 0.9, 1.7]])

    energy_elk_ry = 2.0 * CB * np.einsum("ai,ai->a", moments, fields_elk)
    energy_here_ry = -np.einsum("ai,ai->a", moments, geometry.magnetic_fields)
    assert energy_here_ry == pytest.approx(energy_elk_ry, rel=1e-12)


def test_the_field_is_not_read_as_a_hartree_energy(tmp_path):
    """The old reading, ``2 B_Elk``, is 273.75 times the right field and opposite in sign."""
    b_elk = 1.0
    geometry = _geometry(tmp_path, [(0.0, 0.0, b_elk)])
    hartree_reading = 2.0 * b_elk
    assert geometry.magnetic_fields[0, 2] < 0.0
    assert hartree_reading / abs(geometry.magnetic_fields[0, 2]) == pytest.approx(
        1.0 / CB, rel=1e-12
    )
