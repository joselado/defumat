"""``heisenberg_exchange`` fits one ``J`` per neighbour *shell*, not per vector typed.

The model is ``E(q) - E(0) = m^2 sum_shells J_s sum_{R in s} [1 - cos(q . R)]``,
with the sum over one member of each ``(R, -R)`` bond, and a shell is named by
any one of its members. The scans here are synthetic -- ``E(q)`` written down
from a known ``J`` on a known lattice, no SCF -- so what is tested is only that
the fit returns the ``J`` the surface was built from, whichever representative
names the shell.

The case that failed is fcc with ``q`` along ``b_3``: of the six nearest-neighbour
bonds, three have ``q . R != 0``, so a fit with one column per vector typed
returned 0 for ``[1, 0, 0]`` (orthogonal to ``q``) and ``3 J_1`` for
``[0, 0, 1]``.
"""

import numpy as np
import pytest

from defumat.workflows.spiral import SpiralScan, heisenberg_exchange

pytestmark = pytest.mark.unit

# QE's ibrav = 2, in units of the cubic lattice constant.
FCC = 0.5 * np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, 1.0, 0.0]])


def _crystal(cartesian, at):
    """Cartesian lattice vectors in units of ``alat`` -> crystal coordinates."""
    return np.rint(np.linalg.solve(at.T, np.asarray(cartesian, float).T).T)


def _half_star(cartesian_shell, at):
    """One member of each ``(R, -R)`` pair of a Cartesian shell, in crystal coordinates."""
    bonds = []
    for vector in _crystal(cartesian_shell, at):
        if not any(np.array_equal(-vector, b) for b in bonds):
            bonds.append(vector)
    return np.array(bonds)


# fcc nearest neighbours: the twelve (+-1/2, +-1/2, 0) and permutations.
_NN = np.array([v for v in np.array(np.meshgrid(*[[-0.5, 0.0, 0.5]] * 3)).reshape(3, -1).T
                if np.count_nonzero(v) == 2])
# fcc second neighbours: (+-1, 0, 0) and permutations.
_NNN = np.vstack([np.eye(3), -np.eye(3)])


def _scan(q, energies, moment=1.0):
    q = np.asarray(q, dtype=float)
    return SpiralScan(
        wavevectors=q,
        energies=np.asarray(energies, dtype=float),
        moments=np.tile([0.0, 0.0, moment], (len(q), 1)),
        converged=(True,) * len(q),
    )


def _energy(q, shells_and_j, m=1.0):
    q = np.asarray(q, dtype=float)
    total = np.zeros(len(q))
    for bonds, j in shells_and_j:
        total += j * np.sum(1.0 - np.cos(2.0 * np.pi * q @ bonds.T), axis=1)
    return m**2 * total


def test_fcc_nearest_neighbour_shell_counts():
    """The synthetic lattice itself: 12 and 6 vectors, 6 and 3 bonds."""
    assert len(_NN) == 12 and len(_half_star(_NN, FCC)) == 6
    assert len(_half_star(_NNN, FCC)) == 3


@pytest.mark.parametrize("representative", [[1, 0, 0], [0, 0, 1], [1, -1, 0], [0, 1, -1]])
def test_fcc_nearest_neighbour_j_is_independent_of_the_representative(representative):
    """q along b_3 alone: every member of the nn star returns the same J_1."""
    j1 = 2.5e-4
    q = [[0.0, 0.0, x] for x in np.linspace(0.0, 0.5, 9)]
    energies = _energy(q, [(_half_star(_NN, FCC), j1)])
    (fitted,) = heisenberg_exchange(_scan(q, energies), FCC, [representative])
    assert fitted == pytest.approx(j1, rel=1e-10)


def test_fcc_two_shells_on_a_general_path():
    """J_1 and J_2 together, each named by a member orthogonal to part of the path."""
    j1, j2 = 2.5e-4, -7.0e-5
    q = np.array([[x, 0.3 * x, 0.0] for x in np.linspace(0.0, 0.5, 7)]
                 + [[0.0, 0.0, x] for x in np.linspace(0.05, 0.5, 6)])
    energies = _energy(q, [(_half_star(_NN, FCC), j1), (_half_star(_NNN, FCC), j2)], m=1.7)
    second = _crystal([[0.0, 0.0, 1.0]], FCC)[0]
    for nn in ([1, 0, 0], [0, 0, 1]):
        fitted = heisenberg_exchange(_scan(q, energies, moment=1.7), FCC, [nn, second])
        np.testing.assert_allclose(fitted, [j1, j2], rtol=1e-9)


def test_tetragonal_chain_keeps_one_bond_per_shell():
    """The hydrogen chain's cell (ibrav = 6, c/a = 5/12): the z shell is one bond.

    The star of ``[0, 0, 1]`` is ``+-c`` alone, so ``J`` here is the per-bond
    constant of ``E = m^2 J [1 - cos(2 pi q)]`` exactly as before the fix.
    """
    at = np.diag([1.0, 1.0, 5.0 / 12.0])
    j1, j2 = 1.2e-3, -3.0e-4
    q = [[0.0, 0.0, x] for x in np.linspace(0.0, 0.5, 11)]
    x = np.asarray(q)[:, 2]
    energies = j1 * (1 - np.cos(2 * np.pi * x)) + j2 * (1 - np.cos(4 * np.pi * x))
    fitted = heisenberg_exchange(_scan(q, energies), at, [[0, 0, 1], [0, 0, 2]])
    np.testing.assert_allclose(fitted, [j1, j2], rtol=1e-10)


def test_two_representatives_of_one_shell_are_refused():
    q = [[0.0, 0.0, x] for x in np.linspace(0.0, 0.5, 5)]
    with pytest.raises(ValueError, match="same star"):
        heisenberg_exchange(_scan(q, np.zeros(5)), FCC, [[1, 0, 0], [0, 0, -1]])
