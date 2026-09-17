"""P88 stage 5: the index bookkeeping the augmented ultracell rests on.

Three tables and one pair of transforms, all of them host-side integer or FFT
arithmetic and none of them needing a calculation -- so they are checked here,
in the gate, rather than only through the supercell comparison that would
notice them eventually and would not say which one had moved.

The pair of transforms is the one worth having a test of its own, and the
first thing to say about it is what it is *not*. ``becsum`` lives on the
Q-difference index and PAW's one-centre terms live on the cell copies, and each
is carried into the other's index -- but the two transforms are not inverses,
because both carry ``e^{+i Q_d . R}``. Testing them as a round trip passes a
factor of ``N`` and a reflection through in silence. What binds them is the
contraction the loop forms out of the pair, which is Parseval and is asserted
below; getting a direction wrong there is a modulation running backwards
through the ultracell, a plausible answer rather than an error.
"""

import numpy as np
import pytest

import jax.numpy as jnp

from defumat.ultracell.augmentation import (
    becsum_per_copy,
    coefficients_per_difference,
)
from defumat.ultracell.grid import Ultracell

SHAPES = [(1, 1, 1), (2, 1, 1), (4, 1, 1), (2, 3, 1), (2, 2, 2)]


@pytest.mark.parametrize("shape", SHAPES)
def test_the_difference_and_sum_tables_are_inverses(shape):
    """``Q_bra + (Q_ket - Q_bra)`` is ``Q_ket``, which is the whole of both tables.

    ``difference[bra, ket]`` is the displaced table a pair reads and
    ``sum_index[q, d]`` is the partner ``becsum`` pairs ``Q_q`` with at
    difference ``d``; the matrix element uses the first and the density the
    second, so they have to agree about what a difference *is* or the two
    halves of the same term disagree by an umklapp.
    """
    ultracell = Ultracell.build(shape, (4, 4, 4))
    triples = ultracell.q_triples
    difference = np.asarray(ultracell.difference_index).T
    sum_index = ultracell.q_index(triples[:, None, :] + triples[None, :, :])

    cells = ultracell.cells
    for bra in range(cells):
        for ket in range(cells):
            assert sum_index[bra, difference[bra, ket]] == ket


@pytest.mark.parametrize("shape", SHAPES)
def test_the_two_transforms_pair_into_one_energy(shape):
    """``sum_R ddd^R becsum^R`` is ``N sum_d DDD(Q_d) becsum(Q_d)``, and that binds them.

    The two transforms are **not** inverses and it would be a mistake to test
    them as though they were: both carry ``e^{+i Q_d . R}``, because both turn
    one index into the other rather than undoing each other. ``becsum`` is given
    on the Q-differences and wanted on the copies, ``ddd`` is given on the
    copies and wanted on the Q-differences, and the same kernel does both.

    What does bind them is the quantity the loop actually forms out of the pair:
    the one-centre contribution to ``deband`` is a contraction over the copies,
    and it has to equal the same contraction in Q-space -- which is Parseval
    with the ``N`` the two conventions differ by sitting on one side. A sign
    flipped in either transform breaks this and nothing else in the gate would.
    """
    cells = int(np.prod(shape))
    rng = np.random.default_rng(0)
    shape_rest = (2, 3, 4, 4)
    becsum_q = jnp.asarray(
        rng.normal(size=(cells,) + shape_rest)
        + 1j * rng.normal(size=(cells,) + shape_rest)
    )
    ddd_r = jnp.asarray(rng.normal(size=(cells,) + shape_rest))

    copies = jnp.fft.ifftn(
        becsum_q.reshape(tuple(shape) + shape_rest), axes=(0, 1, 2)
    ).reshape(becsum_q.shape) * cells
    per_difference = coefficients_per_difference(ddd_r, shape)

    real_space = complex(jnp.sum(jnp.asarray(ddd_r) * copies))
    reciprocal = cells * complex(jnp.sum(per_difference * becsum_q))
    assert abs(real_space - reciprocal) < 1.0e-10 * abs(real_space)


@pytest.mark.parametrize("shape", SHAPES)
def test_a_lattice_periodic_set_is_the_same_on_every_copy(shape):
    """The tiled null in miniature: only ``Q_d = 0`` is alive, so every copy agrees.

    It is the one case the loop starts from -- ``_tiled_becsum`` seeds exactly
    this -- and it fixes the factor ``N`` that
    :func:`~defumat.ultracell.augmentation.becsum_per_copy` carries. Without
    that factor the seed would be the unit cell's occupations divided by ``N``
    and PAW's one-centre energy would be wrong at the first iteration in a way
    the loop would then converge away from.
    """
    cells = int(np.prod(shape))
    rng = np.random.default_rng(2)
    block = rng.normal(size=(1, 2, 3, 3))
    block = 0.5 * (block + np.swapaxes(block, -1, -2))
    values = np.zeros((cells,) + block.shape)
    values[0] = block

    copies, residual = becsum_per_copy(jnp.asarray(values + 0j), shape)
    assert residual < 1.0e-14
    for cell in range(cells):
        assert np.max(np.abs(np.asarray(copies)[cell] - block)) < 1.0e-13


@pytest.mark.parametrize("shape", SHAPES)
def test_a_conjugate_paired_q_set_gives_real_copies(shape):
    """The check the loop runs on itself, on a set built to satisfy it.

    **The relation is conjugation and not Hermiticity, and the difference is
    the symmetrisation.** A copy's projector occupation matrix is Hermitian, so
    its imaginary part is its antisymmetric part and is not zero; what
    :func:`~defumat.ultracell.augmentation.ultracell_becsum` carries is the part
    symmetric in the channel pair, which is all ``Q_ij`` and the one-centre
    tensors ever contract, and on a symmetric block Hermiticity reads
    ``becsum(-Q_d) = conj(becsum(Q_d))``. The transform of such a set is real,
    and the loop warns when it is not -- so this is that guard fed a case it
    must *not* trip. The case that must trip it is the supercell comparison.
    """
    ultracell = Ultracell.build(shape, (4, 4, 4))
    cells = ultracell.cells
    triples = ultracell.q_triples
    minus = ultracell.q_index(-triples)

    rng = np.random.default_rng(1)
    raw = rng.normal(size=(cells, 1, 2, 3, 3)) + 1j * rng.normal(size=(cells, 1, 2, 3, 3))
    raw = 0.5 * (raw + np.swapaxes(raw, -1, -2))
    values = np.empty_like(raw)
    for d in range(cells):
        values[d] = 0.5 * (raw[d] + np.conj(raw[minus[d]]))

    copies, residual = becsum_per_copy(jnp.asarray(values), shape)
    scale = float(np.max(np.abs(np.asarray(copies))))
    assert residual / scale < 1.0e-14


@pytest.mark.parametrize("shape", [s for s in SHAPES if np.prod(s) > 1])
def test_a_broken_pairing_is_caught(shape):
    """The guard fed a case it **must** trip, which is the half that matters.

    The test above shows the check passes on a set built to satisfy it, and on
    its own that is "a check whose null result cannot be told from a pass": a
    diagnostic that returns a clean zero for everything, including for the
    defect it exists to find, reads as agreement rather than as silence. So the
    pairing is broken deliberately -- ``becsum(-Q_d)`` set to something that is
    not the conjugate of ``becsum(Q_d)`` -- and the residual has to come back at
    the scale of the quantity itself rather than at round-off.
    """
    ultracell = Ultracell.build(shape, (4, 4, 4))
    cells = ultracell.cells
    rng = np.random.default_rng(3)
    raw = rng.normal(size=(cells, 1, 2, 3, 3)) + 1j * rng.normal(size=(cells, 1, 2, 3, 3))
    broken = 0.5 * (raw + np.swapaxes(raw, -1, -2))

    _, residual = becsum_per_copy(jnp.asarray(broken), shape)
    assert residual > 1.0e-2, residual
