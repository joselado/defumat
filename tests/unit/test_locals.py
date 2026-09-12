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


# --------------------------------------------------------------------------
# what the seeding number means
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "written, expected",
    [
        # Below 1, a fraction of the valence charge -- both codes agree and
        # every committed benchmark lives here, which is why the rule below
        # went unnoticed.
        (0.1, 0.6),
        (0.5, 3.0),
        # At or above 1, pw.x reads Bohr magnetons and divides by Z_v = 6.
        (1.0, 1.0),
        (2.0, 2.0),
        (-2.0, -2.0),
        # 6/6 = 1 exactly: fully polarised, and the clamp is not reached.
        (6.0, 6.0),
    ],
)
def test_the_seeded_moment_is_the_one_pw_x_would_seed(written, expected, pseudo_dir):
    """``input.f90:1448-1449``, on an oxygen atom where ``Z_v = 6``.

    A value at or above 1 is Bohr magnetons in ``pw.x`` and was always a
    fraction here, so the same input file asked the two codes for different
    physics -- by a factor of six on this atom. Invisible to the committed
    benchmark suite because every test-suite value is below 1.

    The clamp at ``:1476-1480`` matters for a second reason that has nothing to
    do with units: without it ``starting_magnetization = 2`` splits this atom
    into **9 up and -3 down** electrons, and a negative channel density is not
    a density.
    """
    import re
    import tempfile
    import warnings

    text = re.sub(
        r"starting_magnetization\(1\)\s*=\s*[-\d.]+",
        f"starting_magnetization(1) = {written}",
        (QE / "o-atom-lsda.in").read_text(),
    )
    directory = Path(tempfile.mkdtemp())
    (directory / "case.in").write_text(text)
    system = system_from_file(directory / "case.in")
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    assert float(pseudos[0].z_valence) == 6.0, "this test is about Z_v != 1"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calculation = Calculation(system, pseudos)
        density = np.asarray(calculation.starting_density())
    scale = system.cell.volume / density[0].size
    up, down = (density.sum(axis=(1, 2, 3)) * scale)[:2]

    assert up - down == pytest.approx(expected, abs=1e-9)
    assert min(up, down) >= -1e-12, "a channel density went negative"
    assert up + down == pytest.approx(6.0, abs=1e-9), "the charge must not move"

    # The reinterpretation is announced, and only when it happens.
    reinterpreted = [w for w in caught if "Bohr magnetons" in str(w.message)]
    assert bool(reinterpreted) == (abs(written) >= 1.0)


def test_the_collinear_and_noncollinear_seeds_use_the_same_rule(pseudo_dir):
    """``spin_weights`` had its own copy of the padding, and so its own units.

    The collinear density and PAW's atomic ``becsum`` both come from
    ``spin_weights``, and it read ``System.starting_magnetization`` directly
    rather than going through the property where QE's rule lives -- so the rule
    reached the noncollinear seed and not the collinear one. Two guesses that
    disagree about how polarized an atom is make iteration 1 contradict itself.
    """
    import re
    import tempfile
    import warnings

    text = re.sub(
        r"starting_magnetization\(1\)\s*=\s*[-\d.]+",
        "starting_magnetization(1) = 3.0",
        (QE / "o-atom-lsda.in").read_text(),
    )
    directory = Path(tempfile.mkdtemp())
    (directory / "case.in").write_text(text)
    system = system_from_file(directory / "case.in")
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculation = Calculation(system, pseudos)
        weights = calculation.spin_weights
        fraction = calculation.starting_magnetization
    assert fraction[0] == pytest.approx(0.5, abs=1e-12)   # 3.0 / 6
    assert weights[0, 0] == pytest.approx(0.75, abs=1e-12)
    assert weights[1, 0] == pytest.approx(0.25, abs=1e-12)


@pytest.mark.parametrize(
    "row, seeded, clamped",
    [(1.0, 1.0, False), (2.0, 2.0, False), (-3.0, -3.0, False),
     (6.0, 6.0, False), (9.0, 6.0, True)],
)
def test_a_card_row_seeds_the_moment_it_names(row, seeded, clamped, pseudo_dir):
    """``STARTING_MOMENTS`` is in Bohr magnetons, which is what it always said.

    It was documented as Bohr magnetons in four places and consumed as the
    per-atom weight on that species' tabulated atomic charge, so a row of
    ``(0, 0, 1.0)`` seeded **6.0** on this oxygen atom and ``(0, 0, 3.0)``
    seeded 18 and a channel of -6 electrons. The rows are now divided by the
    valence charge on the way into the seed, and only there.

    A row larger than the valence charge is clamped with a warning rather than
    refused: an atomic superposition cannot express more moment than the atom
    has electrons, and the alternative is a negative channel density.
    """
    import re
    import tempfile
    import warnings

    text = re.sub(
        r"starting_magnetization\(1\)\s*=\s*[-\d.]+",
        "starting_magnetization(1) = 0.0",
        (QE / "o-atom-lsda.in").read_text(),
    )
    directory = Path(tempfile.mkdtemp())
    (directory / "case.in").write_text(
        text + f"STARTING_MOMENTS\n 0.0 0.0 {row}\n"
    )
    system = system_from_file(directory / "case.in")
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calculation = Calculation(system, pseudos)
        density = np.asarray(calculation.starting_density())
    scale = system.cell.volume / density[0].size
    up, down = (density.sum(axis=(1, 2, 3)) * scale)[:2]

    assert up - down == pytest.approx(seeded, abs=1e-9)
    assert min(up, down) >= -1e-12, "a channel density went negative"
    assert up + down == pytest.approx(6.0, abs=1e-9)
    assert bool([w for w in caught if "valence charge" in str(w.message)]) is clamped

    # **The card itself does not move.** The magnetic symmetry filter and the
    # constraint targets read ``System.local_moments``, and they want the
    # physical moment in Bohr magnetons -- dividing a two-species texture by
    # two different valence charges would give the filter a pattern nothing
    # physical has.
    assert np.asarray(system.local_moments)[0, 2] == pytest.approx(row, abs=1e-12)
