"""Atomic positions: the four unit conventions must describe the same atoms."""

import numpy as np
import pytest

from pypresso.system.cell import Cell
from pypresso.system.structure import Species, Structure
from pypresso.units import BOHR_TO_ANGSTROM

pytestmark = pytest.mark.unit

CELL = Cell.from_ibrav(2, [10.2, 0, 0, 0, 0, 0])
SILICON = (Species(name="Si", mass=28.086, pseudo_file="Si.pz-vbc.UPF"),)


def _build(positions, units):
    return Structure.from_card_units(positions, [0, 0], SILICON, units, CELL)


def test_unit_conventions_agree():
    """The same two atoms written four ways must land in the same place."""
    alat = _build([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], "alat")
    expected = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]) * CELL.alat

    assert np.asarray(alat.positions) == pytest.approx(expected)

    bohr = _build(expected, "bohr")
    assert np.asarray(bohr.positions) == pytest.approx(expected)

    angstrom = _build(expected * BOHR_TO_ANGSTROM, "angstrom")
    assert np.asarray(angstrom.positions) == pytest.approx(expected)

    crystal_coordinates = np.asarray(CELL.to_crystal(expected))
    crystal = _build(crystal_coordinates, "crystal")
    assert np.asarray(crystal.positions) == pytest.approx(expected)


def test_crystal_coordinates_round_trip():
    structure = _build([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], "alat")
    back = np.asarray(CELL.to_cartesian(structure.positions_crystal(CELL)))
    assert back == pytest.approx(np.asarray(structure.positions))
    assert np.asarray(structure.positions_alat(CELL)) == pytest.approx(
        np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    )


def test_species_bookkeeping():
    structure = _build([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], "alat")
    assert (structure.nat, structure.ntyp) == (2, 1)
    assert structure.symbols == ("Si", "Si")


def test_inconsistent_input_is_rejected():
    with pytest.raises(ValueError, match="types for"):
        Structure(positions=CELL.precision.as_real(np.zeros((2, 3))), types=(0,), species=SILICON)
    with pytest.raises(ValueError, match="past the end"):
        Structure(positions=CELL.precision.as_real(np.zeros((1, 3))), types=(5,), species=SILICON)
    with pytest.raises(ValueError, match="unknown ATOMIC_POSITIONS units"):
        _build([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], "furlongs")
    with pytest.raises(NotImplementedError, match="space-group"):
        _build([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], "crystal_sg")


def test_positions_are_traceable():
    """Forces are grad of the energy with respect to these, so they must be a
    single differentiable array leaf, not static metadata."""
    import jax

    structure = _build([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], "alat")
    leaves = jax.tree_util.tree_leaves(structure)
    assert len(leaves) == 1 and leaves[0].shape == (2, 3)

    gradient = jax.grad(lambda p: (p**2).sum())(structure.positions)
    assert np.asarray(gradient) == pytest.approx(2.0 * np.asarray(structure.positions))
