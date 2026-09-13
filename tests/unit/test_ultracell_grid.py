"""The ultracell index map, which everything in P88 rests on.

Nothing here runs an SCF. The whole of :mod:`defumat.ultracell.grid` is basis
bookkeeping, and it is exactly the kind of bookkeeping that produces a
plausible wrong answer rather than an error -- a mis-signed Miller index puts a
plane wave in the wrong cell of the ultracell, which is a *different physical
state*, not a crash. So each of the identities the module's docstring claims is
asserted here separately.
"""

from pathlib import Path

import numpy as np
import pytest

from defumat.basis.builder import build_basis
from defumat.io.pwin import read_pw_input
from defumat.system import build_system
from defumat.ultracell.grid import Ultracell, folded_kpoints

ROOT = Path(__file__).resolve().parents[2]
SHAPES = [(1, 1, 1), (2, 1, 1), (1, 3, 1), (2, 1, 3), (3, 2, 2)]


@pytest.fixture(scope="module")
def silicon():
    system = build_system(read_pw_input(ROOT / "benchmarks" / "si-1k.in"))
    return system, build_basis(system)


def _ultracell(shape, basis):
    return Ultracell.build(shape, basis.dense.grid)


@pytest.mark.parametrize("shape", SHAPES)
def test_the_box_index_recovers_the_unit_cell_index(shape, silicon):
    """``floor(J_i / n_i)`` is the unit cell's own FFT box index of ``G_i``.

    This is the identity the reciprocal mask and the whole ``G``-wrap argument
    rest on, and it has to hold for ``G_i`` of *either* sign -- the negative
    branch is where an off-by-one would live, because ``n G + q`` is negative
    there and the ``mod`` has to land it in the right cell.
    """
    system, basis = silicon
    ultracell = _ultracell(shape, basis)
    miller = np.asarray(basis.dense.miller)
    assert (miller < 0).any(), "the test is only meaningful with negative G"

    n = np.asarray(ultracell.shape)
    box = np.asarray(ultracell.grid)
    for iq in range(ultracell.cells):
        flat = ultracell.box_index(miller, iq)
        assert flat.min() >= 0 and flat.max() < ultracell.points
        J = np.stack(np.unravel_index(flat, tuple(box)), axis=-1)
        assert np.array_equal(J // n, miller % np.asarray(basis.dense.grid))
        assert np.array_equal(J % n, np.broadcast_to(
            ultracell.q_triples[iq], J.shape
        ))


@pytest.mark.parametrize("shape", SHAPES)
def test_a_plane_wave_at_q_is_the_phase_it_should_be(shape, silicon):
    """A single ``G = 0`` coefficient at ``Q`` transforms to ``e^{iQ.r}``.

    The sharpest check there is on the two conventions agreeing: the reciprocal
    index map, and the claim that ultracell grid point ``J`` sits at unit-cell
    crystal coordinate ``J_i / Nd_i``.
    """
    system, basis = silicon
    ultracell = _ultracell(shape, basis)
    axes = [np.arange(m) / n for m, n in zip(ultracell.grid, ultracell.cell_grid)]
    x = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

    for iq in range(ultracell.cells):
        coefficients = np.zeros(ultracell.points, dtype=complex)
        coefficients[ultracell.box_index(np.zeros((1, 3), int), iq)] = 1.0
        field = np.fft.ifftn(coefficients.reshape(ultracell.grid)) * ultracell.points
        want = np.exp(2j * np.pi * (x @ ultracell.q_crystal[iq]))
        assert np.allclose(field, want, atol=1e-12)


@pytest.mark.parametrize("shape", SHAPES)
def test_a_tiled_field_has_only_q_equals_zero(shape, silicon):
    """``tile`` and ``box_index`` agree: a unit-cell field is pure ``Q = 0``.

    If the real-space tiling and the reciprocal index map used different
    conventions, a tiled field would acquire spurious ``Q != 0`` components --
    a modulation out of nothing, which is precisely the quantity the method
    computes and so the one error that would be invisible.
    """
    system, basis = silicon
    ultracell = _ultracell(shape, basis)
    rng = np.random.default_rng(0)
    cell_field = rng.normal(size=basis.dense.grid)

    spectrum = np.fft.fftn(np.asarray(ultracell.tile(cell_field)))
    spectrum = spectrum.reshape(-1) / ultracell.points

    everything = np.stack(np.meshgrid(
        *[np.arange(m) for m in basis.dense.grid], indexing="ij"
    ), axis=-1).reshape(-1, 3)
    everything = np.where(
        everything > np.asarray(basis.dense.grid) // 2,
        everything - np.asarray(basis.dense.grid), everything,
    )
    at_zero = np.zeros(ultracell.points, dtype=bool)
    at_zero[ultracell.box_index(everything, 0)] = True

    # at N = 1 the ultracell *is* the unit cell and there is no outside
    if not at_zero.all():
        assert np.abs(spectrum[~at_zero]).max() < 1e-13
    reference = np.fft.fftn(cell_field).reshape(-1) / cell_field.size
    assert np.allclose(
        spectrum[ultracell.box_index(np.asarray(basis.dense.miller), 0)],
        reference[np.asarray(basis.dense.fft_index)],
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_the_reciprocal_mask_is_the_repeated_one(shape, silicon):
    """``np.repeat`` of the unit-cell mask is the set of every ``G + Q``."""
    system, basis = silicon
    ultracell = _ultracell(shape, basis)
    cell_mask = np.zeros(basis.dense.grid, dtype=bool)
    cell_mask.reshape(-1)[np.asarray(basis.dense.fft_index)] = True

    mask = ultracell.reciprocal_mask(cell_mask)
    assert mask.sum() == cell_mask.sum() * ultracell.cells

    built = np.zeros(ultracell.points, dtype=bool)
    for iq in range(ultracell.cells):
        built[ultracell.box_index(np.asarray(basis.dense.miller), iq)] = True
    assert np.array_equal(built.reshape(ultracell.grid), mask)


@pytest.mark.parametrize("shape", SHAPES)
def test_g_plus_q_reduces_to_the_unit_cell_at_q_zero(shape, silicon):
    """``|G+Q|^2`` on the box is the unit cell's ``|G|^2`` on the ``Q = 0`` rows."""
    system, basis = silicon
    ultracell = _ultracell(shape, basis)
    g2 = ultracell.g2(system.cell).reshape(-1)
    at_zero = g2[ultracell.box_index(np.asarray(basis.dense.miller), 0)]
    assert np.allclose(at_zero, np.asarray(basis.dense.kinetic(system.cell)), atol=1e-10)
    # exactly one vanishing element, and it is G = Q = 0
    assert (g2 < 1e-12).sum() == 1
    assert g2[0] < 1e-12


@pytest.mark.parametrize("shape", SHAPES)
def test_the_difference_table_is_the_umklapp(shape, silicon):
    """``Q_i - Q_j`` wraps into the Q-set, which is where the ``G`` goes."""
    system, basis = silicon
    ultracell = _ultracell(shape, basis)
    table = ultracell.difference_index
    triples = ultracell.q_triples
    n = np.asarray(ultracell.shape)

    assert table.shape == (ultracell.cells, ultracell.cells)
    assert np.array_equal(np.diag(table), np.zeros(ultracell.cells, dtype=table.dtype))
    assert np.array_equal(triples[table], (triples[:, None, :] - triples[None, :, :]) % n)
    # every row is a permutation: the Q-set is a group under the wrapped
    # difference, which is why every block of the Hamiltonian finds a potential
    for row in table:
        assert sorted(row.tolist()) == list(range(ultracell.cells))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("kgrid", [(1, 1, 1), (2, 1, 2)])
def test_k0_plus_q_tiles_the_unit_cell_grid(shape, kgrid, silicon):
    """The folded set is one Monkhorst-Pack grid of the unit cell, exactly once.

    This is what makes the tiled density an exact fixed point of the ultracell
    loop rather than an approximate one: the ultracell and a unit-cell SCF on
    the ``supercell * kgrid`` mesh integrate the zone over the same points with
    the same weights.
    """
    system, basis = silicon
    ultracell = _ultracell(shape, basis)
    k0, folded = folded_kpoints(ultracell, kgrid, system.cell)

    assert k0.nk == int(np.prod(kgrid))
    assert folded.nk == k0.nk * ultracell.cells

    combined = np.asarray(folded.crystal(system.cell)) % 1.0
    grid = tuple(n * m for n, m in zip(shape, kgrid))
    want = np.stack(np.meshgrid(
        *[np.arange(m) / m for m in grid], indexing="ij"
    ), axis=-1).reshape(-1, 3)

    order = lambda a: a[np.lexsort(a.round(9).T)]
    assert np.allclose(order(combined.round(9)), order(want.round(9)), atol=1e-9)


def test_a_non_integer_ultracell_is_refused():
    """Three positive integers, and Elk's unchecked ``avecu`` is not inherited."""
    with pytest.raises(ValueError, match="three positive integers"):
        Ultracell.build((2, 0, 1), (8, 8, 8))
    with pytest.raises(ValueError, match="three positive integers"):
        Ultracell.build((2, 1), (8, 8, 8))
