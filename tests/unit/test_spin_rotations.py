"""The SU(2) representation of a point group, and the axial per-atom average.

``d_spin_ldau`` is the matrix that says how an operation turns a *spinor*, as
against how it turns a position. Everything that keeps both spin indices of a
density matrix needs it -- a noncollinear occupation matrix, a spin-angle
projection -- and symmetrising the ``m`` indices alone without it leaves the
off-diagonal spin blocks in a frame that has moved.

**The matrix is checked before anything is wired to it**, because a wrong one
gives a converged run with its magnetization pointing somewhere else rather than
an error. Three properties, and together they pin it: it is in SU(2), it
represents the rotation it claims to (``U sigma_a U^dagger = R_ba sigma_b``),
and it is a representation of the *group* up to the sign a spin representation
is always free in.
"""

import numpy as np
import pytest

from defumat.system.cell import Cell
from defumat.system.symmetry import (
    _PAULI,
    _su2_from_rotation,
    cartesian_rotations,
    find_symmetries,
    lattice_point_group,
    magnetization_signs,
    spin_rotations,
    symmetrize_atom_cartesian_tensor,
    atom_mapping,
)

pytestmark = pytest.mark.unit


def _random_rotation(rng):
    matrix, upper = np.linalg.qr(rng.normal(size=(3, 3)))
    matrix = matrix * np.sign(np.diag(upper))
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] = -matrix[:, 0]
    return matrix


def _axis_angle(axis, degrees):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    angle = np.deg2rad(degrees)
    cross = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def _symmetries(rotations, t_rev=None):
    """A :class:`Symmetries` carrying just a point group, for the algebra tests."""
    from defumat.system.symmetry import Symmetries

    rotations = [np.asarray(r, dtype=int).tolist() for r in rotations]
    nsym = len(rotations)
    return Symmetries(
        rotations=tuple(tuple(tuple(row) for row in r) for r in rotations),
        translations=tuple(((0.0, 0.0, 0.0),) * nsym),
        time_reversed=tuple(
            int(v) for v in (np.zeros(nsym, dtype=int) if t_rev is None else t_rev)
        ),
    )


def _represents(u, rotation):
    """``U sigma_a U^dagger == R_ba sigma_b`` -- the defining property."""
    rotated = np.array([u @ _PAULI[a] @ u.conj().T for a in range(3)])
    return np.abs(rotated - np.einsum("ba,bij->aij", rotation, _PAULI)).max()


class TestTheLift:
    """``_su2_from_rotation`` on rotations chosen to break a case analysis."""

    def test_a_random_proper_rotation_is_represented(self):
        rng = np.random.default_rng(20260913)
        worst = 0.0
        for _ in range(200):
            rotation = _random_rotation(rng)
            u = _su2_from_rotation(rotation)
            assert abs(np.linalg.det(u) - 1.0) < 1e-12
            assert np.abs(u @ u.conj().T - np.eye(2)).max() < 1e-12
            worst = max(worst, _represents(u, rotation))
        assert worst < 1e-12, worst

    @pytest.mark.parametrize("axis,degrees", [
        ((0, 0, 1), 0),       # identity: the trace branch
        ((1, 0, 0), 180),     # the three 180-degree rotations, where q_0 = 0 and
        ((0, 1, 0), 180),     # QE's ``versor`` needs a branch of its own
        ((0, 0, 1), 180),
        ((1, 1, 0), 180),     # a 180 about a face diagonal
        ((1, 1, 1), 120),     # the cubic three-fold
        ((0, 0, 1), 90),
        ((0, 0, 1), 60),
    ])
    def test_the_special_rotations_a_case_analysis_exists_for(self, axis, degrees):
        rotation = _axis_angle(axis, degrees)
        u = _su2_from_rotation(rotation)
        assert abs(np.linalg.det(u) - 1.0) < 1e-12
        assert _represents(u, rotation) < 1e-12

    def test_a_half_turn_is_traceless_and_a_full_turn_is_minus_one(self):
        """``exp(-i theta/2 n.sigma)`` -- the two-valuedness, stated as a number."""
        half = _su2_from_rotation(_axis_angle((0, 0, 1), 180))
        assert abs(np.trace(half)) < 1e-13
        # Two half turns are the identity rotation and *minus* the identity
        # matrix in spin space: this is the sign every consumer has to cancel.
        assert np.abs(half @ half + np.eye(2)).max() < 1e-13


class TestTheGroupRepresentation:
    """``D(R1) D(R2) = +-D(R1 R2)`` over a whole point group."""

    @pytest.mark.parametrize("at,name", [
        (np.eye(3), "simple cubic, 48 operations"),
        (np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.6]]), "tetragonal"),
        (np.array([[1.0, 0.0, 0.0], [-0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 2.1]]),
         "hexagonal"),
        (np.array([[1.0, 0.0, 0.0], [0.0, 1.3, 0.0], [0.0, 0.0, 1.7]]),
         "orthorhombic"),
    ])
    def test_the_product_of_two_operations_is_the_operation_of_the_product(
        self, at, name
    ):
        """The group property holds **up to sign**, which is what SU(2) gives.

        A faithful representation would have no sign; a spin representation is
        two-valued, so a 2 pi rotation is ``-1`` and the product of two lifts is
        the lift of the product only up to that. Asserting equality without the
        sign would fail on a correct matrix, and asserting nothing would pass on
        a wrong one -- so the assertion is that the residual is zero for **one**
        of the two signs, for every pair in the group.
        """
        at = np.asarray(at, dtype=float) * 5.0
        cell = Cell.from_vectors(at=at, alat=5.0)
        rotations = lattice_point_group(at)
        nsym = len(rotations)
        symmetries = _symmetries(rotations)
        cartesian = cartesian_rotations(cell, symmetries)
        spin = spin_rotations(cell, symmetries)

        # The cartesian matrices close under multiplication; find the index of
        # each product so the two sides can be compared operation by operation.
        # ``+ 0.0`` and not decoration: ``np.round`` turns a -1e-17 into
        # ``-0.0``, whose byte pattern differs from ``+0.0``, so a product that
        # *is* in the group reads as absent. Adding zero normalises the sign.
        index = {}
        for s, matrix in enumerate(cartesian):
            index[(np.round(matrix, 6) + 0.0).tobytes()] = s

        worst = 0.0
        pairs = 0
        for a in range(nsym):
            for b in range(nsym):
                product = cartesian[a] @ cartesian[b]
                key = (np.round(product, 6) + 0.0).tobytes()
                assert key in index, f"{name}: the group is not closed"
                c = index[key]
                lhs = spin[a] @ spin[b]
                # An improper operation acts in spin space as its proper part,
                # so the lift of a product of two improper operations is the
                # lift of the product of the proper parts -- which is the same
                # matrix, and the sign is what absorbs the rest.
                residual = min(
                    np.abs(lhs - spin[c]).max(), np.abs(lhs + spin[c]).max()
                )
                worst = max(worst, residual)
                pairs += 1
        assert pairs == nsym * nsym
        assert worst < 1e-12, f"{name}: worst residual {worst}"

    def test_every_operation_is_unitary_with_unit_determinant(self):
        at = np.eye(3) * 5.0
        rotations = lattice_point_group(at)
        symmetries = _symmetries(rotations)
        spin = spin_rotations(Cell.from_vectors(at=at, alat=5.0), symmetries)
        assert spin.shape == (len(rotations), 2, 2)
        for matrix in spin:
            assert np.abs(matrix @ matrix.conj().T - np.eye(2)).max() < 1e-12
            assert abs(abs(np.linalg.det(matrix)) - 1.0) < 1e-12

    def test_inversion_does_nothing_in_spin_space(self):
        """An improper operation acts as ``-R`` does: the spin is axial."""
        at = np.eye(3) * 5.0
        rotations = lattice_point_group(at)
        symmetries = _symmetries(rotations)
        cell = Cell.from_vectors(at=at, alat=5.0)
        cartesian = cartesian_rotations(cell, symmetries)
        spin = spin_rotations(cell, symmetries)
        inversion = [
            s for s, matrix in enumerate(cartesian)
            if np.abs(matrix + np.eye(3)).max() < 1e-10
        ]
        assert inversion, "the simple cubic group must contain the inversion"
        assert np.abs(spin[inversion[0]] - np.eye(2)).max() < 1e-12


class TestTimeReversal:
    """The ``i sigma_y D*`` twist an operation that needs time reversal carries."""

    def test_the_twist_is_applied_and_stays_in_su2(self):
        at = np.eye(3) * 5.0
        rotations = lattice_point_group(at)
        nsym = len(rotations)
        cell = Cell.from_vectors(at=at, alat=5.0)
        plain = _symmetries(rotations)
        reversed_ = _symmetries(rotations, np.ones(nsym, dtype=int))
        without = spin_rotations(cell, plain)
        with_ = spin_rotations(cell, reversed_)
        i_sigma_y = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=complex)
        assert np.abs(with_ - np.einsum(
            "ij,sjk->sik", i_sigma_y, without.conj()
        )).max() < 1e-14
        for matrix in with_:
            assert np.abs(matrix @ matrix.conj().T - np.eye(2)).max() < 1e-12

    def test_a_time_reversed_operation_reverses_the_spin(self):
        """``U rho^T U^dagger`` carries the moment to ``-det(R) R m``.

        The **transpose is not optional and is the whole content of this test**.
        Time reversal is antiunitary, so the matrix ``spin_rotations`` returns
        is only the unitary half of the operation and the complex conjugation is
        the other half; for a Hermitian block that conjugation is the transpose.
        ``new_ns_nc`` spells it by reading ``nr(m4, m3, is4, is3, nb)`` instead
        of ``nr(m3, m4, is3, is4, nb)``.

        Leaving it out is not a small error in a rare branch: the moment then
        comes out with the **wrong sign** for every time-reversed operation,
        which is most of the group of an antiferromagnet. The two numbers are in
        the assertion below.
        """
        at = np.eye(3) * 5.0
        rotations = lattice_point_group(at)
        nsym = len(rotations)
        cell = Cell.from_vectors(at=at, alat=5.0)
        symmetries = _symmetries(rotations, np.ones(nsym, dtype=int))
        cartesian = cartesian_rotations(cell, symmetries)
        spin = spin_rotations(cell, symmetries)
        plain = spin_rotations(cell, _symmetries(rotations))
        signs = magnetization_signs(cell, symmetries)

        rng = np.random.default_rng(3)
        moment = rng.normal(size=3)
        rho = 0.5 * (np.eye(2) + np.einsum("a,aij->ij", moment, _PAULI))

        def moment_of(matrix):
            return np.array([
                np.real(np.trace(_PAULI[a] @ matrix)) for a in range(3)
            ])

        with_transpose = without_transpose = 0.0
        for s in range(nsym):
            determinant = np.sign(np.linalg.det(cartesian[s]))
            wanted = -determinant * (cartesian[s] @ moment)
            with_transpose = max(with_transpose, np.abs(
                moment_of(spin[s] @ rho.T @ spin[s].conj().T) - wanted
            ).max())
            without_transpose = max(without_transpose, np.abs(
                moment_of(spin[s] @ rho @ spin[s].conj().T) - wanted
            ).max())
            # the axial sign of the same operation is det(R) * (-1)
            assert np.isclose(signs[s], -determinant)
            # and without the time-reversal twist it is det(R) * (+1)
            assert np.abs(
                moment_of(plain[s] @ rho @ plain[s].conj().T)
                - determinant * (cartesian[s] @ moment)
            ).max() < 1e-13
        assert with_transpose < 1e-13, with_transpose
        assert without_transpose > 1.0, (
            "the transpose made no difference, so this test cannot tell a "
            "correct time-reversal branch from a missing one"
        )


class TestTheAxialAtomAverage:
    """``symmetrize_atom_cartesian_tensor(axial=True)`` -- ``<L>`` and ``<S>``."""

    def _structure(self):
        from defumat.system.structure import Structure, Species

        # A **low-symmetry** cell on purpose. On a cubic one the group contains
        # a rotation carrying z onto x, so a z moment averages to zero whether
        # it is read as axial or as polar -- the test would pass on a polar
        # average of an axial vector, which is exactly the defect it exists to
        # catch. Here the group is the identity and the inversion, and those two
        # differ on an axial vector and on nothing else.
        cell = Cell.from_vectors(
            at=np.array([[6.0, 0.0, 0.0], [0.6, 5.3, 0.0], [0.4, 0.9, 4.7]]),
            alat=6.0,
        )
        species = (Species(name="H", mass=1.0, pseudo_file="H.upf"),)
        crystal = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        structure = Structure(
            species=species, types=(0, 0),
            positions=crystal @ np.asarray(cell.at),
        )
        return cell, structure

    def test_an_axial_vector_survives_where_a_polar_one_is_killed(self):
        """The inversion is the discriminator, and it is the whole point.

        A polar vector averages to zero over a group containing the inversion;
        an axial one does not. So on a centrosymmetric cell the two routes give
        different answers for the same input, and reading ``<S>`` through the
        polar one returns a clean, plausible zero.
        """
        cell, structure = self._structure()
        symmetries = find_symmetries(cell, structure)
        assert symmetries.nsym == 2, "the identity and the inversion"
        mapping = atom_mapping(cell, structure, symmetries)
        moments = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

        polar = symmetrize_atom_cartesian_tensor(
            moments, cell, symmetries, mapping, axial=False
        )
        axial = symmetrize_atom_cartesian_tensor(
            moments, cell, symmetries, mapping, axial=True
        )
        assert np.abs(polar).max() < 1e-12
        assert np.abs(axial - moments).max() < 1e-12

    def test_the_default_is_still_polar(self):
        """``symtensor``'s callers must not change under this parameter."""
        cell, structure = self._structure()
        symmetries = find_symmetries(cell, structure)
        mapping = atom_mapping(cell, structure, symmetries)
        tensors = np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3)
        assert np.abs(
            symmetrize_atom_cartesian_tensor(tensors, cell, symmetries, mapping)
            - symmetrize_atom_cartesian_tensor(
                tensors, cell, symmetries, mapping, axial=False
            )
        ).max() == 0.0

    def test_a_rank_two_axial_tensor_takes_the_sign_twice(self):
        """Two axial indices make a polar-signed object, so rank 2 agrees again."""
        cell, structure = self._structure()
        symmetries = find_symmetries(cell, structure)
        mapping = atom_mapping(cell, structure, symmetries)
        rng = np.random.default_rng(7)
        tensors = rng.normal(size=(2, 3, 3))
        assert np.abs(
            symmetrize_atom_cartesian_tensor(
                tensors, cell, symmetries, mapping, axial=True
            )
            - symmetrize_atom_cartesian_tensor(
                tensors, cell, symmetries, mapping, axial=False
            )
        ).max() < 1e-13

    def test_the_average_is_idempotent(self):
        cell, structure = self._structure()
        symmetries = find_symmetries(cell, structure)
        mapping = atom_mapping(cell, structure, symmetries)
        rng = np.random.default_rng(11)
        moments = rng.normal(size=(2, 3))
        once = symmetrize_atom_cartesian_tensor(
            moments, cell, symmetries, mapping, axial=True
        )
        twice = symmetrize_atom_cartesian_tensor(
            once, cell, symmetries, mapping, axial=True
        )
        assert np.abs(twice - once).max() < 1e-13
