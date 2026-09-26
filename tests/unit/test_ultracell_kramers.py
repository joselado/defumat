"""The Kramers-closed basis on a matrix small enough to know the answer.

The hydrogen helix keeps every direction of the union, so the dropping and the
padding of :func:`~defumat.ultracell.kramers.kramers_closed_basis` are reached
only here: a partner that duplicates a state is dropped, the folded k-point it
belongs to keeps its slot as a zero vector at the sentinel, and what is kept is
orthonormal with the reference's own eigenvalues at the bottom.
"""

import numpy as np
import pytest

import jax.numpy as jnp

from defumat.ultracell.kramers import (
    OVERLAP_FLOOR,
    SENTINEL_SHIFT,
    kramers_closed_basis,
    time_reversed,
)

pytestmark = [pytest.mark.unit]


class _Matrix:
    """``apply`` and ``apply_s`` of one Hermitian matrix per k-point, ``S = 1``."""

    def __init__(self, matrices):
        self.matrices = [jnp.asarray(m) for m in matrices]

    def apply(self, psi, ik):
        return psi @ self.matrices[ik].T

    def apply_s(self, psi, ik):
        return psi


def _hermitian(size, seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    return 0.5 * (a + a.conj().T)


def test_a_duplicate_partner_is_dropped_and_its_slot_padded():
    size, nbnd = 12, 3
    matrices = [_hermitian(size, 0), _hermitian(size, 1)]
    values = [np.linalg.eigh(m) for m in matrices]
    states = np.stack([v[1][:, :nbnd].T for v in values])
    # k-point 0: the partners are new directions; k-point 1: the first one
    # duplicates a state exactly, so the union there has rank 2 nbnd - 1.
    rng = np.random.default_rng(2)
    partners = rng.normal(size=(2, nbnd, size)) + 1j * rng.normal(size=(2, nbnd, size))
    partners /= np.linalg.norm(partners, axis=-1, keepdims=True)
    partners[1, 0] = states[1, 0] * np.exp(0.3j)
    mask = np.ones((2, size), dtype=bool)

    basis = kramers_closed_basis(_Matrix(matrices), states, partners, mask)

    assert basis.ranks.tolist() == [2 * nbnd, 2 * nbnd - 1]
    assert basis.dropped.tolist() == [0, 1]
    assert basis.smallest_overlap > OVERLAP_FLOOR
    vectors = np.asarray(basis.wavefunctions)
    for ik, rank in enumerate(basis.ranks):
        kept = vectors[ik, :rank]
        np.testing.assert_allclose(kept.conj() @ kept.T, np.eye(rank), atol=1e-12)
        # the reference's own states are in the span, so its lowest
        # eigenvalues are the lowest Ritz values
        np.testing.assert_allclose(basis.eigenvalues[ik, :nbnd],
                                   values[ik][0][:nbnd], atol=1e-12)
        # and the Ritz vectors diagonalise the matrix within the span
        h = kept.conj() @ np.asarray(matrices[ik]) @ kept.T
        np.testing.assert_allclose(h, np.diag(basis.eigenvalues[ik, :rank]),
                                   atol=1e-12)
    # the padded slot: a zero vector at the sentinel
    assert np.all(vectors[1, -1] == 0.0)
    top = max(float(basis.eigenvalues[0].max()),
              float(basis.eigenvalues[1, : basis.ranks[1]].max()))
    assert basis.eigenvalues[1, -1] == pytest.approx(top + SENTINEL_SHIFT)


def test_the_mask_zeroes_the_padding_of_both_sets():
    size, nbnd = 8, 2
    matrix = _hermitian(size, 3)
    states = np.linalg.eigh(matrix)[1][:, :nbnd].T[None]
    rng = np.random.default_rng(4)
    partners = rng.normal(size=(1, nbnd, size)).astype(complex)
    mask = np.ones((1, size), dtype=bool)
    mask[0, -2:] = False
    basis = kramers_closed_basis(_Matrix([matrix]), states, partners, mask)
    assert np.all(np.asarray(basis.wavefunctions)[0, :, -2:] == 0.0)


def test_time_reversal_flips_the_magnetization_and_keeps_the_charge():
    rng = np.random.default_rng(5)
    density = rng.normal(size=(4, 3, 3, 3))
    becsum = (rng.normal(size=(4, 2, 5, 5)), None)
    reversed_density, reversed_becsum = time_reversed(density, becsum, 4)
    np.testing.assert_array_equal(np.asarray(reversed_density[0]), density[0])
    np.testing.assert_array_equal(np.asarray(reversed_density[1:]), -density[1:])
    np.testing.assert_array_equal(np.asarray(reversed_becsum[0][0]), becsum[0][0])
    np.testing.assert_array_equal(np.asarray(reversed_becsum[0][1:]), -becsum[0][1:])
    assert reversed_becsum[1] is None
    with pytest.raises(ValueError, match="noncollinear"):
        time_reversed(density[:2], (), 2)
