"""P62b unit checks: the occupation matrix as a 2x2 matrix in spin space.

The comparison against Quantum ESPRESSO is in
``tests/regression/test_ldau_flavours.py``. What is here needs no SCF, and the
two statements it makes are the ones that cannot be made about a collinear
occupation matrix at all:

* **a spinor with its moment along z reproduces the collinear answer exactly**,
  in the energy and in the potential, for both functionals and both double
  countings -- the check that catches a factor of two in the spin sum;
* **the energy is invariant under a global rotation of the spin frame**, which
  is what the off-diagonal spin blocks are for. Nothing in this functional knows
  about spin-orbit coupling, so turning the moment must cost nothing, and a
  wrong index in the exchange term breaks that where the first check does not
  see it.
"""

from dataclasses import replace

import numpy as np
import pytest

import jax.numpy as jnp

from defumat.hubbard.energy import (
    coefficients_from_setup,
    hubbard_energy,
    hubbard_potential,
)
from defumat.hubbard.interaction import default_racah
from defumat.hubbard.manifold import HubbardSetup, HubbardSpecies
from defumat.units import RY_TO_EV

pytestmark = pytest.mark.unit

U = 4.3 / RY_TO_EV
J = 1.0 / RY_TO_EV


def _setup(kind, noncolin, double_counting="fll", l=2):
    species = HubbardSpecies(n=3, l=l, u=U, occupation=6.0)
    if kind == 1:
        species = replace(species, j=J, racah=default_racah(l, J, (0.0,) if l == 2 else ()))
    width = 2 * l + 1
    return HubbardSetup(
        species=(species,), atoms=(0,), ldims=(width,), ldmx=width, offsets=(0,),
        atomwfc_offsets=(0,), nwfcU=(2 if noncolin else 1) * width, types=(0,),
        kind=kind, noncolin=noncolin, double_counting=double_counting,
    )


def _collinear_pair(width=5, seed=5):
    rng = np.random.default_rng(seed)
    block = rng.normal(size=(2, 1, width, width)) * 0.2
    return jnp.asarray(block + np.swapaxes(block, -1, -2) + np.eye(width) * 0.6)


def _as_spinor_along_z(collinear):
    """The same state written as a spinor: the two channels on the diagonal."""
    spinor = np.zeros((4,) + collinear.shape[1:], dtype=complex)
    spinor[0] = np.asarray(collinear[0])
    spinor[3] = np.asarray(collinear[1])
    return jnp.asarray(spinor)


def _hermitian_spinor(width=5, seed=9):
    rng = np.random.default_rng(seed)
    block = rng.normal(size=(2, 2, 1, width, width)) + 1j * rng.normal(
        size=(2, 2, 1, width, width)
    )
    block = 0.1 * (block + np.conj(np.einsum("stnab->tsnba", block)))
    block = block + np.eye(width) * 0.6 * np.eye(2)[:, :, None, None, None]
    return jnp.asarray(block.reshape((4, 1, width, width)))


CASES = [(0, "fll"), (1, "fll"), (1, "amf")]
IDS = ["simplified", "full/fll", "full/amf"]


@pytest.mark.parametrize("kind,double_counting", CASES, ids=IDS)
def test_a_spinor_along_z_is_the_collinear_answer(kind, double_counting):
    """Two channels or one 2x2 matrix: the same energy and the same potential.

    The off-diagonal spin blocks of the potential must come out **exactly**
    zero, which is the half of the statement a total energy would not catch.
    """
    collinear = _collinear_pair()
    spinor = _as_spinor_along_z(collinear)
    c_col = coefficients_from_setup(_setup(kind, False, double_counting))
    c_spin = coefficients_from_setup(_setup(kind, True, double_counting))

    assert float(hubbard_energy(spinor, c_spin)) == pytest.approx(
        float(hubbard_energy(collinear, c_col)), abs=1e-13
    )
    v_col = np.asarray(hubbard_potential(collinear, c_col))
    v_spin = np.asarray(hubbard_potential(spinor, c_spin))
    assert v_spin[0] == pytest.approx(v_col[0], abs=1e-13)
    assert v_spin[3] == pytest.approx(v_col[1], abs=1e-13)
    assert np.abs(v_spin[1]).max() == pytest.approx(0.0, abs=1e-14)
    assert np.abs(v_spin[2]).max() == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("kind,double_counting", CASES, ids=IDS)
def test_turning_the_moment_costs_nothing(kind, double_counting):
    """A global ``SU(2)`` rotation of the spin frame leaves the energy alone.

    The functional knows nothing about spin-orbit coupling, so the direction of
    the shell's moment cannot enter it. The rotation mixes the four spin blocks
    into each other, so this is the statement that the *exchange* term's spin
    indices are paired the way ``v_hubbard_full_nc`` pairs them and not the
    other way round -- which the check above, on a spin-diagonal matrix, cannot
    distinguish.
    """
    theta, phi = 0.7, 1.1
    rotation = np.array([
        [np.cos(theta / 2), -np.exp(-1j * phi) * np.sin(theta / 2)],
        [np.exp(1j * phi) * np.sin(theta / 2), np.cos(theta / 2)],
    ])
    blocks = np.asarray(_hermitian_spinor()).reshape((2, 2, 1, 5, 5))
    turned = np.einsum("sa,tb,stnmp->abnmp", np.conj(rotation), rotation, blocks)

    coefficients = coefficients_from_setup(_setup(kind, True, double_counting))
    original = float(hubbard_energy(jnp.asarray(blocks.reshape(4, 1, 5, 5)), coefficients))
    rotated = float(hubbard_energy(jnp.asarray(turned.reshape(4, 1, 5, 5)), coefficients))
    assert rotated == pytest.approx(original, abs=1e-13)


@pytest.mark.parametrize("kind,double_counting", CASES, ids=IDS)
def test_the_potential_is_hermitian(kind, double_counting):
    """``v_hub`` is an operator on the ``2(2l+1)`` space and must be Hermitian.

    ``jax.grad`` of a real function of a complex argument returns the
    *conjugate* Wirtinger derivative, which is the transpose of the potential
    rather than the potential. On a Hermitian ``ns`` the two differ by exactly
    that transpose, and nothing about a symmetric test matrix would show it.
    """
    ns = _hermitian_spinor()
    coefficients = coefficients_from_setup(_setup(kind, True, double_counting))
    v = np.asarray(hubbard_potential(ns, coefficients)).reshape((2, 2, 1, 5, 5))
    assert v == pytest.approx(np.conj(np.einsum("stnab->tsnba", v)), abs=1e-13)


def test_the_projector_columns_are_spin_slowest():
    """``offsetU + m + ldim (is - 1)``: all the up columns, then all the down.

    ``atomic_wfc_nc`` emits a channel's ``2l+1`` up columns as a block and its
    down ones after them, and ``new_ns_nc`` reads the occupation matrix at that
    stride. Interleaving them per orbital instead transposes the spin and ``m``
    indices of every block, which is a converged run with a rotated moment.
    """
    setup = _setup(0, True)
    columns = setup.column_map()
    assert columns.shape == (1, 2, 5)
    assert list(columns[0, 0]) == [0, 1, 2, 3, 4]
    assert list(columns[0, 1]) == [5, 6, 7, 8, 9]

    spin, slot, row, column = setup.block_indices()
    # Four quadrants of one block, not four separate matrices.
    assert set(np.unique(spin)) == {0, 1, 2, 3}
    assert row.max() == column.max() == 9
