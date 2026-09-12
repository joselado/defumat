"""The per-atom integration spheres, and the site moments read out of them.

Two separate things live here. The **layout** (packed against dense) is a
memory choice that must not change a number, so it is checked against the
weights it replaced rather than against a tolerance. The **readout** is the one
quantity a magnetic run reports that can tell a compensated magnet from the
nonmagnetic state it may have collapsed into, so it is checked on a state whose
cell total is zero and whose sites are not.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.pseudo.upf import read_upf
from defumat.scf.driver import Calculation, _report_mag, run_scf
from defumat.scf.locals import (
    _grid_points,
    _minimum_image_distances,
    _qe_weights,
    build_local_regions,
    default_radii,
)
from defumat.system.builder import system_from_file

QE = Path(__file__).resolve().parents[1] / "data" / "qe"
GRID = (20, 20, 40)


@pytest.fixture(scope="module")
def chain():
    return system_from_file(QE / "h10-chain-noncolin.in")


def _dense_reference(system, grid):
    """``_qe_weights`` as it was consumed before the packed layout existed."""
    points = _grid_points(grid)
    positions = np.asarray(system.structure.positions_crystal(system.cell)) % 1.0
    distances = _minimum_image_distances(
        points, positions, np.asarray(system.cell.at, dtype=float)
    )
    types = np.asarray(system.structure.types, dtype=int)
    per_atom = default_radii(system.cell, system.structure)[types]
    nat = len(positions)
    return _qe_weights(distances, per_atom).reshape((nat,) + tuple(grid))


def test_the_packed_layout_is_bit_for_bit_the_dense_one(chain):
    """``pointlist``/``factlist`` throws nothing away, so it must round-trip.

    The packed form exists because the dense one is ``nat x ngrid``: 20 MB on
    this ten-atom chain and an estimated 5 GB on a 157-atom slab, against 3 MB
    and 31 MB packed. That is only worth taking if it is the *same* array, so
    this is an equality and not a tolerance.
    """
    regions = build_local_regions(chain.cell, chain.structure, GRID, scheme="qe")
    assert regions.packed
    assert np.array_equal(
        np.asarray(regions.dense_weights()), _dense_reference(chain, GRID)
    )


def test_the_two_layouts_integrate_a_field_the_same(chain):
    """``segment_sum`` against the ``einsum`` it replaced, on a random field."""
    regions = build_local_regions(chain.cell, chain.structure, GRID, scheme="qe")
    dense = jnp.asarray(_dense_reference(chain, GRID))
    field = jnp.asarray(np.random.default_rng(0).normal(size=(4,) + GRID))
    packed = np.asarray(regions.integrate(field))
    reference = np.asarray(jnp.einsum("anmk,cnmk->ac", dense, field))
    # Summation order, and nothing else: both are sums of the same products.
    assert packed == pytest.approx(reference, abs=1e-12)


def test_the_smooth_scheme_stays_dense(chain):
    """The packed form needs disjointness, which a partition of unity has not.

    A point inside two smoothed spheres contributes to both, so there is no
    single owner to record and the dense layout is not an inefficiency there.
    """
    regions = build_local_regions(chain.cell, chain.structure, GRID, scheme="smooth")
    assert not regions.packed
    assert regions.weights.shape == (10,) + GRID


def test_a_point_in_no_sphere_belongs_to_nobody(chain):
    """``owner = -1`` must contribute to no atom, not to atom 0.

    The packed contraction parks unowned points on atom 0 with a zero taper,
    which is a correct trick and a silent bug if the taper is ever read
    without the mask. A constant field makes the failure visible: every atom's
    integral would pick up the whole interstitial.
    """
    regions = build_local_regions(chain.cell, chain.structure, GRID, scheme="qe")
    owner = np.asarray(regions.owner)
    assert (owner < 0).any(), "this cell has vacuum; some point must be unowned"
    ones = jnp.ones((1,) + GRID)
    integrated = np.asarray(regions.integrate(ones))[:, 0]
    assert integrated.sum() == pytest.approx(
        float(np.asarray(regions.taper)[owner >= 0].sum()), abs=1e-9
    )
    assert integrated.sum() < ones.size, "the interstitial leaked into the spheres"


# --------------------------------------------------------------------------
# the readout
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def antiferromagnet(pseudo_dir):
    """A two-site collinear antiferromagnet, converged.

    ``h-chain-afm.in`` states it with two labels and ``nosym``, which is the
    spelling that works today; what matters here is only that the converged
    state has zero total moment and two nonzero site moments.
    """
    system = system_from_file(QE / "h-chain-afm.in")
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    return system, run_scf(system, pseudos, conv_thr=1e-8, verbose=False)


def test_the_site_moments_see_what_the_cell_total_cannot(antiferromagnet):
    """The whole reason the readout exists.

    ``magnetization`` is ``int (rho_up - rho_dw)`` over the cell and is zero
    for every compensated state -- for this antiferromagnet, and equally for
    the nonmagnetic state a symmetriser can average one into. The site moments
    separate them.
    """
    _, result = antiferromagnet
    assert result.magnetization == pytest.approx(0.0, abs=1e-4)
    assert result.site_moments is not None
    moments = np.asarray(result.site_moments)[:, 0]
    assert moments[0] == pytest.approx(-moments[1], abs=1e-4), "not compensated"
    assert abs(moments[0]) > 0.5, "the moment this test is about is missing"
    charges = np.asarray(result.site_charges)
    assert charges.shape == (2,)
    assert charges.sum() < 2.0, "a sphere integral cannot exceed the cell's charge"


def test_the_site_moments_are_recorded_every_iteration(antiferromagnet):
    """A texture that unwinds does it early and then converges cleanly.

    The converged value is exactly the one that cannot show a collapse, so the
    per-iteration record is the instrument rather than a convenience.
    """
    _, result = antiferromagnet
    assert len(result.history) >= 3
    for entry in result.history:
        assert "site_moments" in entry and "site_charges" in entry
        assert len(entry["site_moments"]) == 2
    first = np.asarray(result.history[0]["site_moments"])[:, 0]
    assert abs(first[0]) > 0.1, "the moment was already gone at iteration 1"


def test_the_printed_block_carries_the_direction(antiferromagnet):
    """``report_mag``'s block, plus the angles QE does not print.

    Reconstructing ``theta`` and ``phi`` by hand from three components for
    every atom is the step that made the one production texture failure
    invisible, so they are printed rather than left to the reader.
    """
    system, _ = antiferromagnet
    calculation = Calculation(
        system,
        tuple(
            read_upf(Path("tests/data/pseudo") / s.pseudo_file)
            for s in system.structure.species
        ),
    )
    charges = np.array([0.77, 0.77])
    moments = np.array([[0.0, 0.0, 0.6], [0.0, 0.0, -0.6]])
    printed = io.StringIO()
    with redirect_stdout(printed):
        _report_mag(system, calculation.local_regions(), charges, moments)
    text = printed.getvalue()
    assert "Magnetic moment per site" in text
    assert "theta =    0.000" in text and "theta =  180.000" in text
    # The pair that says "antiferromagnet" rather than "nonmagnetic".
    assert "sum |m| = 1.200000" in text
