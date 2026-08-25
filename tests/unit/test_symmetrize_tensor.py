"""The general-rank cartesian symmetrisers -- ``symmatrix3``/``symtensor3``.

``symme.f90`` writes the same three steps out at rank 2 and again at rank 3
because Fortran has no other way to say it. Here they are one function, so the
rank-4 case P26's elasto-optic tensor needs -- which QE has no counterpart for
-- costs nothing extra. These are the checks that the generalisation did not
change what the rank-2 versions already do, and that the average is what it
claims to be.

The strongest of them needs no reference at all: averaging a **random** rank-3
tensor over the zincblende point group must land exactly on the zincblende
pattern, whose only non-vanishing components are ``xyz`` and its permutations.
That is the property P35's tensor has to have and cannot get from the wedge sum
on its own.
"""

import numpy as np
import pytest

from pypresso.system.cell import Cell
from pypresso.system.structure import Species, Structure
from pypresso.system.symmetry import (
    atom_mapping,
    find_symmetries,
    symmetrize_atom_cartesian_tensor,
    symmetrize_atom_tensor,
    symmetrize_cartesian_tensor,
    symmetrize_matrix,
)

pytestmark = pytest.mark.unit

ALAT = 10.575
CELL = Cell.from_ibrav(2, [ALAT, 0, 0, 0, 0, 0])
#: AlAs: zincblende, ``-43m``, and **not** centrosymmetric -- which is why it is
#: the case P35 runs on and the one that gives a rank-3 average something to do.
ALAS = (
    Species(name="Al", mass=26.98, pseudo_file="Al.pz-vbc.UPF"),
    Species(name="As", mass=74.92, pseudo_file="As.pz-bhs.UPF"),
)
#: Silicon: the same lattice with one species, so the diamond structure and its
#: inversion centre -- where every rank-3 tensor vanishes identically.
SI = (Species(name="Si", mass=28.086, pseudo_file="Si.pz-vbc.UPF"),)

POSITIONS = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])


def _structure(species, types):
    return Structure.from_card_units(POSITIONS, types, species, "alat", CELL)


def _group(species, types):
    structure = _structure(species, types)
    symmetries = find_symmetries(CELL, structure)
    return symmetries, atom_mapping(CELL, structure, symmetries)


@pytest.fixture(scope="module")
def zincblende():
    return _group(ALAS, [0, 1])


@pytest.fixture(scope="module")
def diamond():
    return _group(SI, [0, 0])


def test_the_two_crystals_have_the_groups_they_should(zincblende, diamond):
    """24 operations for ``-43m``, 48 for diamond -- the inversion doubles it."""
    assert zincblende[0].nsym == 24
    assert diamond[0].nsym == 48


# -- the generalisation reproduces what it replaces ---------------------------


def test_rank_two_reproduces_symmatrix(zincblende):
    symmetries, _ = zincblende
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(3, 3))
    assert np.allclose(
        symmetrize_cartesian_tensor(matrix, CELL, symmetries),
        symmetrize_matrix(matrix, CELL, symmetries),
        atol=1e-14,
    )


def test_rank_two_per_atom_reproduces_symtensor(zincblende):
    symmetries, mapping = zincblende
    rng = np.random.default_rng(1)
    tensors = rng.normal(size=(2, 3, 3))
    assert np.allclose(
        symmetrize_atom_cartesian_tensor(tensors, CELL, symmetries, mapping),
        symmetrize_atom_tensor(tensors, CELL, symmetries, mapping),
        atol=1e-14,
    )


def test_it_agrees_with_the_fortran_loops_written_out(zincblende):
    """``symmatrix3``'s six nested loops, transcribed literally, at rank 3.

    The generalisation is index bookkeeping and this is the check on it: the
    conversion to crystal axes and back is done by hand here, in the order
    ``cart_to_crys``/``crys_to_cart`` do it, with no ``tensordot``.
    """
    symmetries, _ = zincblende
    rng = np.random.default_rng(2)
    tensor = rng.normal(size=(3, 3, 3))

    at = np.asarray(CELL.at_alat, dtype=float)
    bg = np.asarray(CELL.bg_2pi_alat, dtype=float)
    crystal = np.einsum("il,jm,kn,lmn->ijk", at, at, at, tensor)
    work = np.zeros((3, 3, 3))
    for rotation in symmetries.rotation_array().astype(float):
        work += np.einsum(
            "il,jm,kn,lmn->ijk", rotation, rotation, rotation, crystal
        )
    work /= symmetries.nsym
    expected = np.einsum("li,mj,nk,lmn->ijk", bg, bg, bg, work)

    assert np.allclose(
        symmetrize_cartesian_tensor(tensor, CELL, symmetries), expected, atol=1e-13
    )


# -- what the average is ------------------------------------------------------


@pytest.mark.parametrize("rank", [2, 3, 4])
def test_averaging_twice_is_averaging_once(zincblende, rank):
    """It is a projection, so it is idempotent -- at every rank."""
    symmetries, _ = zincblende
    rng = np.random.default_rng(3 + rank)
    tensor = rng.normal(size=(3,) * rank)
    once = symmetrize_cartesian_tensor(tensor, CELL, symmetries)
    twice = symmetrize_cartesian_tensor(once, CELL, symmetries)
    assert np.allclose(once, twice, atol=1e-13)


def test_a_symmetric_tensor_is_left_alone(zincblende):
    """``d_ijk`` proportional to ``|eps_ijk|`` is what zincblende allows."""
    symmetries, _ = zincblende
    allowed = np.abs(np.stack([
        np.cross(np.eye(3), row) for row in np.eye(3)
    ]))  # |eps_ijk|
    assert np.allclose(
        symmetrize_cartesian_tensor(allowed, CELL, symmetries), allowed, atol=1e-13
    )


def test_a_random_rank_three_tensor_lands_on_the_zincblende_pattern(zincblende):
    """The free known answer, and the one the wedge sum has to be given.

    ``-43m`` leaves a rank-3 tensor with a single independent component: the
    entries with three *different* indices, all equal. Nothing here imposes
    that -- a random tensor goes in and the point group takes out everything it
    forbids.
    """
    symmetries, _ = zincblende
    rng = np.random.default_rng(7)
    averaged = symmetrize_cartesian_tensor(rng.normal(size=(3, 3, 3)), CELL, symmetries)

    off = [averaged[i, j, k] for i in range(3) for j in range(3) for k in range(3)
           if len({i, j, k}) == 3]
    forbidden = [averaged[i, j, k] for i in range(3) for j in range(3)
                 for k in range(3) if len({i, j, k}) < 3]

    assert np.abs(forbidden).max() < 1e-14
    assert np.abs(np.array(off) - off[0]).max() < 1e-14
    assert abs(off[0]) > 1e-3  # and it did not simply annihilate everything


def test_a_rank_three_tensor_vanishes_in_a_centrosymmetric_crystal(diamond):
    """Diamond has an inversion centre, so ``chi^(2)`` and the Raman tensor die.

    The rank-3 counterpart of the statement that a polar vector vanishes in a
    centrosymmetric crystal, and the reason P35's silicon case is a test that
    the *whole tensor* is zero rather than a comparison against a number.
    """
    symmetries, _ = diamond
    rng = np.random.default_rng(11)
    averaged = symmetrize_cartesian_tensor(rng.normal(size=(3, 3, 3)), CELL, symmetries)
    assert np.abs(averaged).max() < 1e-14


def test_the_atom_axis_is_carried_by_irt(zincblende):
    """A per-atom tensor is averaged **over the atoms an operation connects**.

    In zincblende no operation exchanges Al with As -- the two sublattices are
    different species -- so ``irt`` is the identity here and the per-atom
    average reduces to the pure-cartesian one applied atom by atom. That is
    worth a test precisely because it makes the atom axis look inert: in
    diamond, where the operations *do* exchange the two sites, it is not.
    """
    symmetries, mapping = zincblende
    rng = np.random.default_rng(13)
    tensors = rng.normal(size=(2, 3, 3, 3))
    averaged = symmetrize_atom_cartesian_tensor(tensors, CELL, symmetries, mapping)
    assert np.array_equal(mapping, np.tile(np.arange(2), (symmetries.nsym, 1)))
    for atom in range(2):
        assert np.allclose(
            averaged[atom],
            symmetrize_cartesian_tensor(tensors[atom], CELL, symmetries),
            atol=1e-14,
        )


def test_in_diamond_the_two_sublattices_carry_opposite_tensors(diamond):
    """The atom axis is not inert where operations exchange the sites.

    Diamond's 48 operations include the ones that swap the two silicons, so a
    per-atom rank-3 tensor is averaged across them -- and what survives is
    ``T[0] = -T[1]``, each in the zincblende pattern, **not** zero. That is the
    whole reason silicon has a Raman-active mode at all: the *pure cartesian*
    rank-3 tensor dies in a centrosymmetric crystal, and the per-atom one only
    has to die when summed over atoms, which is P35's translational sum rule.
    An operation taking atom 0 to atom 1 relates the two tensors instead of
    constraining either.
    """
    symmetries, mapping = diamond
    assert (mapping != np.arange(2)).any()
    rng = np.random.default_rng(17)
    tensors = rng.normal(size=(2, 3, 3, 3))
    averaged = symmetrize_atom_cartesian_tensor(tensors, CELL, symmetries, mapping)

    assert np.abs(averaged.sum(axis=0)).max() < 1e-14   # the sum rule
    assert np.allclose(averaged[0], -averaged[1], atol=1e-14)
    assert np.abs(averaged[0]).max() > 1e-3             # and neither is zero

    # ... and each of them is in the same single-component pattern the
    # zincblende case above lands on, since the site group of an atom in
    # diamond is the tetrahedral one.
    forbidden = [averaged[0, i, j, k] for i in range(3) for j in range(3)
                 for k in range(3) if len({i, j, k}) < 3]
    assert np.abs(forbidden).max() < 1e-14
