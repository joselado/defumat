"""The preconditioner's nonlocal diagonal, contracted atom by atom.

``OPEN.md`` Part III M5. ``h_diag`` and ``s_diag`` contracted a ``D`` (or
``q``) that is block-diagonal over atoms as a dense ``(nkb, nkb)`` matrix.
:func:`~defumat.hamiltonian.operator.block_diagonal_form` contracts each atom's
channels with that atom's block only, so it must equal the dense form wherever
``D`` is block-diagonal, to round-off, on either side of the conjugate; and it
must ignore whatever sits *off* the blocks, which is what tells the two forms
apart.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.hamiltonian.operator import atom_blocks, block_diagonal_form

pytestmark = pytest.mark.unit

#: Two species, interleaved and of different sizes, so that one group of
#: atoms is not a contiguous range of channels and the gather path runs too.
NH = (4, 9, 4, 9, 4)
ATOM_OF_CHANNEL = tuple(atom for atom, nh in enumerate(NH) for _ in range(nh))
NPWX = 37


def _hermitian_blocks(rng):
    nkb = len(ATOM_OF_CHANNEL)
    matrix = np.zeros((nkb, nkb), dtype=complex)
    start = 0
    for nh in NH:
        block = rng.standard_normal((nh, nh)) + 1j * rng.standard_normal((nh, nh))
        matrix[start:start + nh, start:start + nh] = block + block.conj().T
        start += nh
    return matrix


def test_the_grouping_is_by_atom_and_by_size():
    groups = atom_blocks(ATOM_OF_CHANNEL)
    assert [group.shape for group in groups] == [(3, 4), (2, 9)]
    assert groups[0][1].tolist() == list(range(13, 17))


@pytest.mark.parametrize("conjugate", ["left", "right"])
def test_the_block_form_is_the_dense_form_on_a_block_diagonal_matrix(conjugate):
    rng = np.random.default_rng(3)
    nkb = len(ATOM_OF_CHANNEL)
    vkb = rng.standard_normal((NPWX, nkb)) + 1j * rng.standard_normal((NPWX, nkb))
    matrix = jnp.asarray(_hermitian_blocks(rng))
    left, right = (vkb.conj(), vkb) if conjugate == "left" else (vkb, vkb.conj())
    left, right = jnp.asarray(left), jnp.asarray(right)

    dense = jnp.einsum("gi,ij,gj->g", left, matrix, right)
    block = jax.jit(block_diagonal_form, static_argnums=3)(left, matrix, right,
                                                           ATOM_OF_CHANNEL)
    scale = np.max(np.abs(np.asarray(dense)))
    assert np.max(np.abs(np.asarray(block - dense))) < 1e-14 * scale
    # With a complex Hermitian matrix the two sides of the conjugate give
    # different real parts, so a helper that conjugated the wrong side would
    # fail the comparison above rather than pass it by symmetry.
    other = jnp.einsum("gi,ij,gj->g", right, matrix, left)
    assert np.max(np.abs(np.real(np.asarray(other - dense)))) > 1e-3 * scale


def test_what_lies_off_the_atom_blocks_never_enters():
    rng = np.random.default_rng(5)
    nkb = len(ATOM_OF_CHANNEL)
    vkb = jnp.asarray(rng.standard_normal((NPWX, nkb)) + 1j * rng.standard_normal((NPWX, nkb)))
    blocks = _hermitian_blocks(rng)
    off = rng.standard_normal((nkb, nkb))
    atoms = np.asarray(ATOM_OF_CHANNEL)
    off[atoms[:, None] == atoms[None, :]] = 0.0
    with_off = jnp.asarray(blocks + off)
    clean = block_diagonal_form(vkb.conj(), jnp.asarray(blocks), vkb, ATOM_OF_CHANNEL)
    assert np.array_equal(
        np.asarray(block_diagonal_form(vkb.conj(), with_off, vkb, ATOM_OF_CHANNEL)),
        np.asarray(clean))
