"""The spin-angle projector set: ``atomic_wfc_so`` and its label table.

Nothing here runs an SCF. What is checked is what a ``j``-resolved projection
*is*, independently of any reference:

* the map from real spherical harmonics onto ``|l j m_j>`` is an isometry,
  shell by shell, which is the whole content of ``spinor``/``sph_ind``/
  ``rot_ylm`` being used together correctly;
* the two ``j`` shells of one ``l`` are mutually orthogonal and hold
  ``2(2l+1)`` states between them, ``2l+2`` and ``2l``;
* ``natomwfc`` is **not** twice the scalar count on a fully-relativistic
  dataset, and platinum is the case that shows it;
* the label table agrees column for column with the orbitals, which is the
  thing that goes wrong silently -- a projection with the right total and the
  wrong decomposition.

The comparison against ``projwfc.x`` itself is in
``tests/regression/test_spinor_pdos.py``.
"""

import numpy as np
import pytest

from defumat.pseudo import read_upf
from defumat.pseudo.atomic import (
    _spin_angle_matrix,
    _updown_matrix,
    count_atomic_wavefunctions,
    count_spinor_wavefunctions,
    spinor_orbital_blocks,
)
from defumat.projwfc.channels import projection_channels
from defumat.system.structure import Species, Structure

pytestmark = pytest.mark.unit

PSEUDO = "tests/data/pseudo"


def _one_atom(pseudo_file: str):
    pseudo = read_upf(f"{PSEUDO}/{pseudo_file}")
    structure = Structure(
        positions=np.zeros((1, 3)),
        types=(0,),
        species=(Species(name="X", mass=1.0, pseudo_file=pseudo_file),),
    )
    return (pseudo,), structure


# --------------------------------------------------------------------------
# The map itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_each_spin_angle_shell_is_an_isometry(l):
    """``2j+1`` orthonormal rows: the Clebsch-Gordan and the basis change agree.

    A row is one spin-angle function written on the shell's real harmonics, so
    its norm is ``sum_is spinor(l, j, m, is)^2`` -- one -- only if ``rot_ylm``
    is unitary on the shell *and* is indexed relative to its centre row. Getting
    either wrong leaves a smooth, plausible, non-normalised projector.
    """
    for j in ([0.5] if l == 0 else [l - 0.5, l + 0.5]):
        matrix = _spin_angle_matrix(l, j)
        assert matrix.shape == (int(2 * j + 1), 2, 2 * l + 1)
        flat = matrix.reshape(matrix.shape[0], -1)
        gram = flat @ flat.conj().T
        assert np.abs(gram - np.eye(flat.shape[0])).max() < 1.0e-12


@pytest.mark.parametrize("l", [1, 2, 3])
def test_the_two_j_shells_of_one_l_are_orthogonal_and_complete(l):
    """``2l + 2`` and ``2l`` states, orthogonal, spanning the ``2(2l+1)``.

    This is ``atomic_wfc_so2``'s pair, built from one radial function, so the
    two blocks act on the *same* columns and their orthogonality is a real
    statement rather than a consequence of disjoint support.
    """
    lower = _spin_angle_matrix(l, l - 0.5).reshape(-1, 2 * (2 * l + 1))
    upper = _spin_angle_matrix(l, l + 0.5).reshape(-1, 2 * (2 * l + 1))
    assert lower.shape[0] == 2 * l
    assert upper.shape[0] == 2 * l + 2
    assert np.abs(lower @ upper.conj().T).max() < 1.0e-12
    both = np.concatenate([lower, upper])
    assert np.abs(both @ both.conj().T - np.eye(2 * (2 * l + 1))).max() < 1.0e-12


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_the_updown_map_is_an_isometry(l):
    """``atomic_wfc_nc``: every ``m`` up, then every ``m`` down."""
    matrix = _updown_matrix(l).reshape(2 * (2 * l + 1), -1)
    assert np.abs(matrix @ matrix.conj().T - np.eye(matrix.shape[0])).max() == 0.0


# --------------------------------------------------------------------------
# The count, which is the thing an intuition gets wrong
# --------------------------------------------------------------------------


def test_a_relativistic_dataset_does_not_simply_double():
    """Platinum: 11 scalar columns and **12** spinor ones, not 22.

    ``Pt.rel-pz-n-rrkjus.UPF`` carries ``5D(j=3/2)``, ``5D(j=5/2)`` and ``6S``
    as three separate ``PP_CHI`` entries, so the scalar count already has
    ``2l+1`` for each of the two ``d`` channels where the spinor count has
    ``2j+1``. ``n_atom_wfc``'s noncollinear branch is a doubling only when the
    two ``j`` come out of one radial function.
    """
    pseudos, structure = _one_atom("Pt.rel-pz-n-rrkjus.UPF")
    assert count_atomic_wavefunctions(pseudos, structure) == 11
    assert count_spinor_wavefunctions(pseudos, structure, lspinorb=True) == 12


def test_a_scalar_dataset_does_double():
    """Without a ``j`` to resolve there is one up and one down per harmonic."""
    pseudos, structure = _one_atom("Pt.pbe-n-kjpaw_psl.0.1.UPF")
    scalar = count_atomic_wavefunctions(pseudos, structure)
    assert count_spinor_wavefunctions(pseudos, structure, lspinorb=False) == 2 * scalar


# --------------------------------------------------------------------------
# The labels, which have to agree with the orbitals column for column
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pseudo_file, lspinorb",
    [
        ("Pt.rel-pz-n-rrkjus.UPF", True),
        ("Pt.rel-pbe-n-kjpaw_psl.0.1.UPF", True),
        ("Pt.pbe-n-kjpaw_psl.0.1.UPF", False),
    ],
)
def test_the_label_table_matches_the_orbitals_column_for_column(pseudo_file, lspinorb):
    """``fill_nlmchi`` against ``spinor_orbital_blocks``, shell by shell.

    The two are built in different modules from the same rules, and a
    disagreement is silent: the projection still sums to the right total and
    every column is mislabelled from the first shell whose width differs.
    """
    pseudos, structure = _one_atom(pseudo_file)
    blocks = spinor_orbital_blocks(pseudos, structure, lspinorb)
    channels = projection_channels(pseudos, structure, True, lspinorb)

    assert sum(matrix.shape[0] for _, _, matrix in blocks) == len(channels)

    index = 0
    for _, width, matrix in blocks:
        shell = channels[index:index + matrix.shape[0]]
        # every column of one block belongs to one shell
        assert len({(c.atom, c.wfc, c.l, c.j) for c in shell}) == 1
        assert shell[0].l == (width - 1) // 2
        if lspinorb:
            j = shell[0].j
            assert matrix.shape[0] == int(2 * j + 1)
            # m_j runs over the shell without repeating
            assert len({c.mj for c in shell}) == matrix.shape[0]
            # |m_j| <= j, and m_j differs from j by an integer
            assert all(abs(c.mj) <= j + 1.0e-9 for c in shell)
            assert all(abs((j - c.mj) - round(j - c.mj)) < 1.0e-9 for c in shell)
        index += matrix.shape[0]
    assert index == len(channels)


def test_the_label_path_refuses_a_relativistic_dataset_without_lspinorb():
    """The same refusal on the *other* branch, which is the one a caller hits.

    ``build_atomic_projectors`` reaches the orbital builder only for
    ``spinor_basis="jmj"``; the labels come from
    :func:`~defumat.projwfc.channels.projection_channels` whatever the basis. So
    the check has to be in both, and without it here platinum gets 22 up/down
    labels against the 12 j-averaged columns ``atomic_wfc_so_mag`` builds.
    """
    pseudos, structure = _one_atom("Pt.rel-pz-n-rrkjus.UPF")
    with pytest.raises(NotImplementedError, match="lspinorb"):
        projection_channels(pseudos, structure, noncolin=True, lspinorb=False)


def test_the_two_refusals_agree_on_their_condition():
    """Neither fires on a scalar dataset, which is the branch that is allowed."""
    pseudos, structure = _one_atom("Pt.pbe-n-kjpaw_psl.0.1.UPF")
    blocks = spinor_orbital_blocks(pseudos, structure, lspinorb=False)
    channels = projection_channels(pseudos, structure, True, False)
    assert sum(matrix.shape[0] for _, _, matrix in blocks) == len(channels)


def test_a_relativistic_dataset_without_lspinorb_is_refused():
    """QE dispatches the orbitals on ``has_so`` and the labels on ``lspinorb``.

    That combination builds ``j``-resolved columns and labels them as up/down
    ones -- the counts agree, nothing fails, and every label is wrong. Refused
    rather than reproduced.
    """
    pseudos, structure = _one_atom("Pt.rel-pz-n-rrkjus.UPF")
    with pytest.raises(NotImplementedError, match="lspinorb"):
        spinor_orbital_blocks(pseudos, structure, lspinorb=False)
