"""The magnetization seed: what it does to a density and to a ``becsum``.

Nothing here runs an SCF. :mod:`defumat.ultracell.seed` turns and scales a
magnetization that is handed to it, and every claim it makes is an identity on
an array -- which is the right place to check them, because the two ways this
goes wrong are both silent. A seed that lands on the **wrong copy** of an atom
is a converged, plausible and wrong modulation; and a rotation that is not the
identity where it should be would leave the tiled null failing for a reason that
has nothing to do with the loop. The tiled null the regression suite runs cannot
see the first of those at all, because a tiled seed is the same on every copy.
"""

import warnings

import numpy as np
import pytest

from defumat.ultracell.grid import Ultracell
from defumat.ultracell.seed import (
    reference_axis,
    refuse_an_unmagnetized_reference,
    seeded_becsum,
    seeded_density,
    warn_if_the_seed_leaves_the_closed_sector,
)

AXIS = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)


@pytest.fixture
def ultracell():
    """Four cells along ``x``, three grid points per cell along it."""
    return Ultracell.build((4, 1, 1), (3, 2, 2))


def _tiled(ultracell, values):
    """One unit cell's profile repeated over the box, as the loop's start is.

    It matters that the reference is *tiled* rather than merely random over the
    box: a cell's own moment is then the same in every cell, which is what makes
    a statement about how the direction turns from one cell to the next a
    statement about the seed.
    """
    return np.tile(values, ultracell.shape)


def _collinear(ultracell, rng):
    """``(2, *box)`` up/down densities with a moment everywhere."""
    cell = tuple(ultracell.cell_grid)
    charge = _tiled(ultracell, 1.0 + 0.1 * rng.random(cell))
    moment = _tiled(ultracell, 0.3 * rng.random(cell))
    return np.stack([(charge + moment) / 2, (charge - moment) / 2])


def _noncollinear(ultracell, rng, axis=AXIS):
    """``(4, *box)``, the magnetization collinear along ``axis``."""
    cell = tuple(ultracell.cell_grid)
    charge = _tiled(ultracell, 1.0 + 0.1 * rng.random(cell))
    length = _tiled(ultracell, 0.3 * rng.random(cell))
    return np.concatenate([charge[None], axis[:, None, None, None] * length])


def _on_the_box(ultracell, function, components=None):
    """A seed callable sampled the way the driver samples it."""
    axes = [np.arange(m) / n
            for m, n in zip(ultracell.grid, ultracell.cell_grid)]
    coordinates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    values = np.asarray(function(coordinates))
    return values if components is None else np.moveaxis(values, -1, 0)


def _cell_moments(ultracell, magnetization):
    """``(3, N)`` the moment of each cell, in :meth:`cell_moments`' order."""
    m = np.asarray(magnetization)
    shape, cell_grid = ultracell.shape, ultracell.cell_grid
    m = m.reshape((3, shape[0], cell_grid[0], shape[1], cell_grid[1],
                   shape[2], cell_grid[2]))
    return m.sum(axis=(2, 4, 6)).reshape(3, -1)


def test_a_unit_seed_is_the_tiled_state(ultracell):
    """``s = 1`` and ``s = e_0`` are the identity, to round-off.

    This is what makes the tiled null a statement about the loop rather than
    about the seed: a run given the unit seed reproduces a run given none, so
    anything the null catches afterwards is the machinery. **Round-off and not
    bit for bit in the collinear case**, because that one goes through the
    charge-and-moment representation and back, which is one addition and one
    halving; the noncollinear density is already stored in it and comes back
    unchanged.
    """
    rng = np.random.default_rng(0)
    collinear = _collinear(ultracell, rng)
    got = np.asarray(seeded_density(collinear, np.ones(ultracell.grid), 2, 1.0))
    assert np.abs(got - collinear).max() < 1.0e-16

    noncollinear = _noncollinear(ultracell, rng)
    axis = reference_axis(noncollinear[1:], 1.0)
    unit = np.broadcast_to(axis[:, None, None, None],
                           (3,) + tuple(ultracell.grid))
    got = np.asarray(seeded_density(noncollinear, unit, 4, 1.0, axis=axis))
    assert np.abs(got - noncollinear).max() < 1.0e-16


def test_a_collinear_seed_scales_the_moment_and_leaves_the_charge(ultracell):
    """``m -> s m`` and ``n -> n``, which is the whole collinear case.

    The charge is the half a spin density wave is *not* seeded in: its own
    modulation is a second-order consequence of the spin one and the loop makes
    it, so seeding it would be seeding an answer.
    """
    rng = np.random.default_rng(1)
    density = _collinear(ultracell, rng)
    seed = _on_the_box(ultracell, lambda x: np.cos(np.pi * x[..., 0]))
    got = np.asarray(seeded_density(density, seed, 2, 1.0))
    assert np.abs(got.sum(axis=0) - density.sum(axis=0)).max() < 1.0e-15
    expected = seed * (density[0] - density[1])
    assert np.abs((got[0] - got[1]) - expected).max() < 1.0e-15


@pytest.mark.filterwarnings("ignore:this seed does not turn")
def test_a_noncollinear_seed_turns_the_moment_and_keeps_its_length(ultracell):
    """A unit seed is a rotation, so ``|m(r)|`` is pointwise unchanged.

    The seed is a direction *and* a length, exactly as
    ``starting_magnetization`` with ``angle1``/``angle2`` is, so ``|s| = 1``
    turns the moment and ``|s| = 1/2`` halves it. Both are asserted here because
    the two are one multiplication apart in the code and would be easy to apply
    to the wrong quantity.
    """
    rng = np.random.default_rng(2)
    density = _noncollinear(ultracell, rng)
    axis = reference_axis(density[1:], 1.0)
    pitch = ultracell.shape[0]
    helix = _on_the_box(ultracell, lambda x: np.stack([
        np.cos(2 * np.pi * x[..., 0] / pitch),
        np.sin(2 * np.pi * x[..., 0] / pitch),
        np.zeros(x.shape[:-1]),
    ], axis=-1), components=3)
    got = np.asarray(seeded_density(density, helix, 4, 1.0, axis=axis))
    assert np.abs(got[0] - density[0]).max() < 1.0e-15
    length = np.sqrt((got[1:] ** 2).sum(axis=0))
    assert np.abs(length - np.sqrt((density[1:] ** 2).sum(axis=0))).max() < 1.0e-15

    halved = np.asarray(seeded_density(density, 0.5 * helix, 4, 1.0, axis=axis))
    assert np.abs(halved[1:] - 0.5 * got[1:]).max() < 1.0e-15


def test_a_helix_seed_turns_by_the_pitch_it_was_given(ultracell):
    """Cell to cell, the moment turns by ``360 / n`` degrees and nothing else.

    The seed is evaluated **pointwise**, so a helix turns continuously across a
    unit cell rather than in steps, which is what a spin spiral does. The
    consequence to know is that a cell's *own* moment then sits at the average
    of the phases inside it -- an offset of half a cell's turn, 30 degrees on
    the three points per cell here -- while the **differences** are exactly the
    pitch. A caller who wants each cell rotated rigidly writes ``floor(x)`` into
    the callable.
    """
    rng = np.random.default_rng(3)
    density = _noncollinear(ultracell, rng, axis=np.array([0.0, 0.0, 1.0]))
    axis = reference_axis(density[1:], 1.0)
    pitch = ultracell.shape[0]
    helix = _on_the_box(ultracell, lambda x: np.stack([
        np.cos(2 * np.pi * x[..., 0] / pitch),
        np.sin(2 * np.pi * x[..., 0] / pitch),
        np.zeros(x.shape[:-1]),
    ], axis=-1), components=3)
    moments = _cell_moments(
        ultracell, np.asarray(seeded_density(density, helix, 4, 1.0, axis=axis))[1:]
    )
    angles = np.degrees(np.arctan2(moments[1], moments[0]))
    steps = np.diff(np.unwrap(np.radians(angles)))
    assert np.abs(np.degrees(steps) - 360.0 / pitch).max() < 1.0e-10
    lengths = np.linalg.norm(moments, axis=0)
    assert np.abs(lengths / lengths[0] - 1.0).max() < 1.0e-12


def test_the_antipodal_seed_is_the_reversed_moment(ultracell):
    """``s = -e_0`` is the case a period-2 seed hits in every other cell.

    The rotation taking ``e_0`` to ``-e_0`` is not unique, so this is a branch
    in the code rather than a limit of the general formula -- and it is not an
    edge case to be tolerated, it is the noncollinear antiferromagnet. For a
    reference whose magnetization is collinear along ``e_0`` every choice of
    half turn gives the same ``-m``, which is what is asserted.
    """
    rng = np.random.default_rng(4)
    density = _noncollinear(ultracell, rng)
    axis = reference_axis(density[1:], 1.0)
    reversed_seed = np.broadcast_to(-axis[:, None, None, None],
                                    (3,) + tuple(ultracell.grid))
    got = np.asarray(seeded_density(density, reversed_seed, 4, 1.0, axis=axis))
    assert np.abs(got[0] - density[0]).max() < 1.0e-15
    assert np.abs(got[1:] + density[1:]).max() < 1.0e-15
    assert np.isfinite(got).all()


def test_a_zero_seed_is_no_magnetization_rather_than_a_rotation(ultracell):
    """``|s| = 0`` has no direction, and the answer is zero and not a NaN."""
    rng = np.random.default_rng(5)
    density = _noncollinear(ultracell, rng)
    axis = reference_axis(density[1:], 1.0)
    got = np.asarray(seeded_density(
        density, np.zeros((3,) + tuple(ultracell.grid)), 4, 1.0, axis=axis))
    assert np.abs(got[1:]).max() == 0.0
    assert np.isfinite(got).all()


@pytest.mark.parametrize("nspin_mag", [2, 4])
def test_a_seed_over_one_is_refused(ultracell, nspin_mag):
    """``|s| <= 1``, because ``s`` scales a moment that is already there.

    The reference's own ``|m| <= n`` pointwise, so a factor above one makes a
    channel density negative -- the loop would start from something that is not
    a density. It is ``starting_magnetization``'s range for the same reason.
    """
    rng = np.random.default_rng(6)
    box = tuple(ultracell.grid)
    if nspin_mag == 2:
        density, seed = _collinear(ultracell, rng), 1.5 * np.ones(box)
        axis = None
    else:
        density = _noncollinear(ultracell, rng)
        axis = reference_axis(density[1:], 1.0)
        seed = 1.5 * np.broadcast_to(axis[:, None, None, None], (3,) + box)
    with pytest.raises(ValueError, match="1.5000"):
        seeded_density(density, seed, nspin_mag, 1.0, axis=axis)


def test_a_compensated_reference_has_no_direction_to_turn_from(ultracell):
    """The refusal that says which cells this cannot seed, and why.

    A noncollinear seed turns the unit cell's magnetization *rigidly*, from the
    one direction it points along. An antiferromagnetic unit cell has no such
    direction -- its net moment is a residue of cancellation whose direction is
    round-off -- and the texture would have to be turned sublattice by
    sublattice, which is not written. The test feeds it a cell whose two halves
    are opposite, which is what such a reference is.
    """
    rng = np.random.default_rng(7)
    density = _noncollinear(ultracell, rng)
    # Two sublattices with opposite moments of the same size, which is what an
    # antiferromagnetic unit cell is, and 0.25 so that the cancellation is
    # exact in floating point rather than nearly exact.
    sign = np.ones(tuple(ultracell.cell_grid))
    sign[:, 1::2] = -1.0
    density[1:] = AXIS[:, None, None, None] * 0.25 * _tiled(ultracell, sign)
    net = np.abs(density[1:].reshape(3, -1).sum(axis=1)).max()
    assert net == 0.0
    with pytest.raises(NotImplementedError, match="residue of cancellation"):
        reference_axis(density[1:], 1.0)


def test_a_seed_that_leaves_the_closed_sector_says_so(ultracell):
    """The warning that is worth 290 iterations against 14, and its null.

    The truncated basis closes one sector and its axis is the reference's own
    magnetization, so a seed whose directions sit on a cone about that axis
    stays where it was put and one that does not has to traverse a flat
    manifold to get to the frame the basis prefers. Both halves are checked
    here, because a warning that fires on everything says nothing, and there are
    **two** silent cases rather than one.

    The second of them is the one that nearly got this wrong. What the closed
    sector preserves is the cone's *half-angle*, ``arccos|s-hat . e_0|``, so a
    **staggered** seed of ``+e_0`` and ``-e_0`` in alternate cells belongs to it:
    that state is collinear along ``e_0``, it lies in the up/down span with no
    rotation at all, and its collinear counterpart converges in seven
    iterations. Measured on the *signed* cosine its spread is 1.0 and it would
    warn -- a guard firing on an antiferromagnet, which is the first thing
    anyone tries after a helix.
    """
    pitch = ultracell.shape[0]
    helix = _on_the_box(ultracell, lambda x: np.stack([
        np.cos(2 * np.pi * x[..., 0] / pitch),
        np.sin(2 * np.pi * x[..., 0] / pitch),
        np.zeros(x.shape[:-1]),
    ], axis=-1), components=3)
    staggered = _on_the_box(ultracell, lambda x: (
        np.where(np.floor(x[..., 0]) % 2 == 0, 1.0, -1.0)[..., None] * AXIS
    ), components=3)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert warn_if_the_seed_leaves_the_closed_sector(
            helix, np.array([0.0, 0.0, 1.0])) < 1.0e-15
        assert warn_if_the_seed_leaves_the_closed_sector(staggered, AXIS) == 0.0

    with pytest.warns(UserWarning, match="290 iterations against 14"):
        spread = warn_if_the_seed_leaves_the_closed_sector(helix, AXIS)
    # About 0.25, and how nearly is the sampling: three points per cell here
    # against fifteen on the cell the iteration counts were measured on.
    assert 0.23 < spread < 0.26


@pytest.mark.parametrize("nspin_mag", [2, 4])
def test_an_unmagnetized_reference_is_refused(ultracell, nspin_mag):
    """A seed on a cell with no moment is a no-op that reads as an answer.

    The seed *scales* the reference's magnetization, so on a cell that converged
    unpolarized it scales zero: every iteration is a no-op, the loop converges,
    and what comes back is the unpolarized state the caller was trying to leave.
    That is the "a check whose null result cannot be told from a pass" shape one
    level out -- the calculation returns the null -- so it is refused by name.
    """
    rng = np.random.default_rng(8)
    box = tuple(ultracell.grid)
    charge = _tiled(ultracell, 1.0 + 0.1 * rng.random(ultracell.cell_grid))
    if nspin_mag == 2:
        density = np.stack([charge / 2, charge / 2])
        seed, axis = np.ones(box), None
    else:
        density = np.concatenate([charge[None], np.zeros((3,) + box)])
        axis = np.array([0.0, 0.0, 1.0])
        seed = np.broadcast_to(axis[:, None, None, None], (3,) + box)
    with pytest.raises(ValueError, match="has none"):
        seeded_density(density, seed, nspin_mag, 1.0, axis=axis)


def _tiled_becsum(ultracell, nspin_mag, nh=2, nat=1):
    """One species' occupations, the same on every copy -- the loop's start."""
    values = np.zeros((nspin_mag, nat, nh, nh))
    values[0] = 0.8
    if nspin_mag == 2:
        values[1] = 0.2
    else:
        values[3] = 0.6      # a moment along z, so the axis is z
    return (np.broadcast_to(values, (ultracell.cells,) + values.shape).copy(),)


@pytest.mark.parametrize("nspin_mag", [2, 4])
def test_the_becsum_seed_lands_on_the_copy_its_cell_index_names(
    ultracell, nspin_mag
):
    """Copy ``R`` gets the seed at cell ``R``, and a uniform seed cannot say so.

    The copy index is a lattice translation in C-order over the integer triple
    (:func:`~defumat.ultracell.augmentation.becsum_per_copy` is an ``ifftn``
    over the Q-grid), and a seed landing on the wrong one converges to a
    plausible wrong modulation. So the seed here is deliberately **not** uniform
    -- a different value in every cell, read back copy by copy against the field
    at that cell's own atom.
    """
    cells = ultracell.cells
    wanted = np.array([0.2, -0.4, 0.6, -0.8])[:cells]
    seed = _on_the_box(ultracell,
                       lambda x: wanted[np.floor(x[..., 0]).astype(int)])
    axis = np.array([0.0, 0.0, 1.0])
    if nspin_mag == 4:
        seed = np.stack([np.zeros_like(seed), np.zeros_like(seed), seed])
    becsum = _tiled_becsum(ultracell, nspin_mag)
    got = np.asarray(seeded_becsum(
        becsum, seed, ultracell, np.zeros((1, 3)), ((0,),), nspin_mag,
        axis=axis,
    )[0])
    before = np.asarray(becsum[0])
    if nspin_mag == 2:
        moment = got[:, 0, 0, 0, 0] - got[:, 1, 0, 0, 0]
        reference = before[0, 0, 0, 0, 0] - before[0, 1, 0, 0, 0]
        charge = got[:, 0, 0, 0, 0] + got[:, 1, 0, 0, 0]
        assert np.abs(charge - before[0, :2, 0, 0, 0].sum()).max() < 1.0e-15
    else:
        moment = got[:, 3, 0, 0, 0]
        reference = before[0, 3, 0, 0, 0]
        assert np.abs(got[:, 0] - before[:, 0]).max() < 1.0e-15
        assert np.abs(got[:, 1:3]).max() < 1.0e-15
    assert np.abs(moment / reference - wanted).max() < 1.0e-14


def test_the_becsum_seed_reads_the_atoms_own_point(ultracell):
    """An atom off the cell's origin reads the seed there, not at the origin.

    The becsum and the density are seeded from the **same** evaluated field, so
    the sphere and the grid cannot disagree about the period -- and the place
    that agreement is decided is this index, the atom's own grid point of the
    box. The check moves one atom to the middle of the cell, where a linear seed
    takes a value the origin does not.
    """
    cells = ultracell.cells
    seed = _on_the_box(ultracell, lambda x: np.cos(np.pi * x[..., 0] / cells))
    positions = np.array([[0.0, 0.0, 0.0], [1.0 / 3.0, 0.0, 0.0]])
    becsum = _tiled_becsum(ultracell, 2, nat=2)
    got = np.asarray(seeded_becsum(
        becsum, seed, ultracell, positions, ((0, 1),), 2,
    )[0])
    moment = got[:, 0, :, 0, 0] - got[:, 1, :, 0, 0]
    before = np.asarray(becsum[0])
    reference = before[0, 0, 0, 0, 0] - before[0, 1, 0, 0, 0]
    cell_grid = ultracell.cell_grid[0]
    at_origin = seed[::cell_grid, 0, 0]
    at_middle = seed[1::cell_grid, 0, 0]
    assert np.abs(moment[:, 0] / reference - at_origin).max() < 1.0e-14
    assert np.abs(moment[:, 1] / reference - at_middle).max() < 1.0e-14
    assert np.abs(at_middle - at_origin).min() > 1.0e-3
