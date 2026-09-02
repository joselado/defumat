"""P62e unit checks: the occupation matrix decomposed into tensor moments.

Every check here is exact and needs no other code, which is what makes this
basis worth having: it is orthonormal and complete by construction, so the
statements are identities rather than agreements.

* the Wigner 3-j symbols against closed forms, in the **doubled** arguments that
  let the half-integer spin symbols share one implementation;
* the basis is orthonormal to 1e-15 and has ``(2(2l+1))^2`` members, which is
  the dimension of the space it spans -- complete by counting as well as by
  construction;
* the round trip ``N -> w -> N`` returns the matrix;
* the moments that have names carry what they are named for: ``w^{000}`` is the
  shell's charge, ``|w^{011}|`` its moment, and ``L . S`` -- built independently
  from :mod:`defumat.projwfc.angular_momentum` -- decomposes onto ``w^{110}_0``
  and **nothing else**, which is the check that fixes the coupling and the
  basis rotation together.
"""

import numpy as np
import pytest

from defumat.hubbard.tensormoments import (
    MomentLabel,
    compose,
    decompose,
    moment_labels,
    moment_matrices,
    wigner3j,
)
from defumat.projwfc.angular_momentum import orbital_matrices

pytestmark = pytest.mark.unit

PAULI = np.array([
    [[0, 1], [1, 0]],
    [[0, -1j], [1j, 0]],
    [[1, 0], [0, -1]],
], dtype=complex)


def _hermitian(l, seed=3):
    """A random Hermitian occupation matrix on the combined (m, spin) index."""
    width = 2 * l + 1
    rng = np.random.default_rng(seed)
    block = rng.normal(size=(2, 2, width, width)) + 1j * rng.normal(
        size=(2, 2, width, width)
    )
    block = 0.1 * (block + np.conj(np.einsum("stab->tsba", block)))
    return block + np.eye(width) * 0.6 * np.eye(2)[:, :, None, None]


def test_the_wigner_symbols_are_the_closed_forms():
    """Doubled arguments, so ``(1 1 0; 0 0 0)`` is ``wigner3j(2, 2, 0, 0, 0, 0)``."""
    assert wigner3j(2, 2, 0, 0, 0, 0) == pytest.approx(-1 / np.sqrt(3), abs=1e-14)
    assert wigner3j(2, 2, 4, 0, 0, 0) == pytest.approx(np.sqrt(2 / 15), abs=1e-14)
    # A half-integer one, which is why the arguments are doubled at all.
    assert wigner3j(1, 1, 2, 1, -1, 0) == pytest.approx(1 / np.sqrt(6), abs=1e-14)
    # Selection rules.
    assert wigner3j(2, 2, 0, 2, 0, 0) == 0.0     # m's do not sum to zero
    assert wigner3j(2, 2, 8, 0, 0, 0) == 0.0     # triangle rule


@pytest.mark.parametrize("l", [1, 2, 3])
def test_the_basis_is_orthonormal_and_complete(l):
    labels = moment_labels(l)
    size = 2 * (2 * l + 1)
    assert len(labels) == size * size
    basis = moment_matrices(l)
    flat = np.transpose(basis, (0, 1, 3, 2, 4)).reshape(len(labels), size, size)
    gram = np.einsum("iab,jab->ij", np.conj(flat), flat)
    assert gram == pytest.approx(np.eye(len(labels)), abs=1e-12)


@pytest.mark.parametrize("l", [1, 2])
def test_the_decomposition_round_trips(l):
    width = 2 * l + 1
    ns = _hermitian(l).reshape(4, width, width)
    assert compose(decompose(ns, l), l) == pytest.approx(ns, abs=1e-13)


@pytest.mark.parametrize("l", [1, 2])
def test_the_moments_are_real(l):
    """A Hermitian matrix has real tensor moments; that is what the basis is for."""
    width = 2 * l + 1
    w = decompose(_hermitian(l).reshape(4, width, width), l)
    assert w.dtype == np.float64
    assert np.abs(w).max() > 0.1


@pytest.mark.parametrize("l", [1, 2])
def test_the_named_moments_are_what_they_are_named(l):
    """``w^{000}`` is the charge and ``|w^{011}|`` the moment, both over ``sqrt(2(2l+1))``.

    That factor is the basis's own normalisation -- ``Gamma^{000}`` is the
    identity divided by the square root of the dimension -- and is the
    "conventional normalisation" Elk multiplies back in.
    """
    width = 2 * l + 1
    blocks = _hermitian(l)
    w = decompose(blocks.reshape(4, width, width), l)
    index = {tuple(label): n for n, label in enumerate(moment_labels(l))}
    scale = np.sqrt(2.0 * width)

    charge = float(np.real(np.trace(blocks[0, 0]) + np.trace(blocks[1, 1])))
    assert w[index[(0, 0, 0, 0)]] * scale == pytest.approx(charge, abs=1e-12)

    moment = np.array([
        np.real(np.trace(blocks[0, 1]) + np.trace(blocks[1, 0])),
        np.imag(np.trace(blocks[0, 1]) - np.trace(blocks[1, 0])),
        np.real(np.trace(blocks[0, 0]) - np.trace(blocks[1, 1])),
    ])
    triple = np.array([w[index[(0, 1, 1, t)]] for t in (-1, 0, 1)])
    assert np.linalg.norm(triple) * scale == pytest.approx(
        np.linalg.norm(moment), abs=1e-12
    )


@pytest.mark.parametrize("l", [1, 2, 3])
def test_spin_orbit_coupling_is_one_moment_and_nothing_else(l):
    """``L . S`` decomposes onto ``w^{110}_0`` alone.

    The sharpest check here and the one that fixes two conventions at once: the
    ``(k, p) -> r`` coupling, and the rotation of the basis into the **real**
    harmonics the occupation matrix is measured in. ``L`` comes from
    :mod:`defumat.projwfc.angular_momentum`, which shares nothing with this
    module but ``rot_ylm``. The coefficient is
    ``-sqrt(l (l+1) (2l+1) / 2)``, which is a closed form rather than a
    measurement.
    """
    width = 2 * l + 1
    ls = 0.5 * np.einsum("cab,cst->stab", orbital_matrices(l), PAULI)
    w = decompose(ls.reshape(4, width, width), l)
    index = {tuple(label): n for n, label in enumerate(moment_labels(l))}
    spin_orbit = index[(1, 1, 0, 0)]

    assert w[spin_orbit] == pytest.approx(
        -np.sqrt(l * (l + 1) * (2 * l + 1) / 2.0), abs=1e-11
    )
    others = np.delete(w, spin_orbit)
    assert np.abs(others).max() == pytest.approx(0.0, abs=1e-11)


def test_the_labels_cover_the_allowed_ranks():
    labels = moment_labels(2)
    assert MomentLabel((0, 0, 0, 0)) in labels
    assert MomentLabel((4, 1, 5, -5)) in labels
    assert MomentLabel((5, 0, 0, 0)) not in labels     # k > 2l
    assert MomentLabel((1, 1, 3, 0)) not in labels     # r > k + p
    assert MomentLabel((1, 1, 0, 0)).name == "L . S"
    assert str(MomentLabel((0, 1, 1, -1))) == "w^(011)_-1"
