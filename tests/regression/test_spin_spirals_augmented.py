"""A spin spiral on an ultrasoft or PAW dataset, validated against supercells.

The companion of ``test_spin_spirals.py``, which does the same for a
norm-conserving dataset. Everything there about why the validation is an
identity rather than a reference number still holds -- neither ``pw.x`` nor Elk
computes a plane-wave spin spiral -- and this file adds the one thing an
augmented dataset brings with it.

**What is being tested.** A spiral's two spinor components live at ``k + q/2``
and ``k - q/2``, so the transverse block of the density pairs projectors at two
*different* k-points. The augmentation charge that block multiplies is then the
lattice sum ``sum_R e^{-i q.R} Q_ij(r - tau_a - R)`` rather than the periodic
one, and in the rotated frame that is the transform of
``Q_ij(G - q) e^{-i (G - q).tau_a}`` -- the resident table displaced by ``-q``.
The charge and ``m_z`` blocks pair projectors at the same k-point and keep the
resident table. Using the resident table for the transverse block as well
leaves a density that is normalised, plausible and wrong.

**The moment is deliberately in the plane** (``angle1 = 90``), so the whole of
it sits in the transverse channel the displacement governs. With the moment
along ``z`` the transverse block is zero and every test below would pass
without the displaced table existing, which is this repository's
"a check whose null result cannot be told from a pass".

**Why the tolerance is not the norm-conserving one, and why it is not what it
first looked like either.** The identity against the collinear antiferromagnet
leaves a residue that falls with ``ecutrho`` -- -3.61e-07, -5.86e-08,
+6.08e-09 Ry at 200, 300 and 400 -- and is insensitive to ``nbnd`` (10 to 24)
and to ``conv_thr`` (1e-10 to 1e-12), which move it in the fourth digit only.

That looks exactly like a G-sphere truncation, and ``PLAN.md`` P95 records the
argument that says it is one, together with the A/B that refutes it. Running
the *same doubled cell* twice with **no spiral in it at all** -- once as
``nspin = 2`` and once as ``noncolin`` with the two moments antiparallel in the
plane -- gives 7.03e-07, 1.32e-07 and 6.11e-13 at the same three cutoffs, which
is **twice** each residue above, the factor a doubled cell's energy is divided
by. The collinear-referenced identity is therefore measuring the gap between
this code's own collinear and noncollinear paths, and the spiral rides on it.

What is left when that gap vanishes is the spiral's own error, and it is
round-off: 6.08e-09 Ry at ``ecutrho = 400``, against the quarter turn's
**1.65e-09** measured at ``ecutrho = 200`` on a supercell that shares the
noncollinear machinery and so never sees the gap at all.

The tolerance below covers the largest of these at the ``ecutrho = 200`` these
inputs carry, which is PAW's 3.26e-07 on the quarter turn. It is a floor set by
the code's collinear/noncollinear consistency and by PAW's one-centre path, not
by the spiral, and it is not a round number chosen to make a test pass.

**What the tolerance has to be read against.** A tolerance is only a test if
something fails it, so each identity is asserted together with the size of the
error the displacement removes: dropping it and using the resident table for
the transverse block leaves the antiferromagnet identity out by **1.15e-03 Ry**
at the same cutoff, three thousand times the residue that is left with it in.
:func:`test_the_resident_table_is_not_enough` is that measurement.
"""

from functools import lru_cache
from pathlib import Path

import pytest

from defumat.io.pwin import parse_pw_input
from defumat.pseudo import read_upf
from defumat.scf import run_scf
from defumat.system import build_system
from tests.conftest import GENERATED

pytestmark = [pytest.mark.regression, pytest.mark.slow]

#: The consistency floor described in the module docstring, at the
#: ``ecutrho = 200`` these inputs carry -- set by the largest measured residue,
#: PAW's 3.26e-07 on the quarter turn. Not a round-off tolerance and not a
#: round number: ultrasoft reaches 1.65e-09 on the same identity, and what
#: separates the two is PAW's one-centre path rather than anything the spiral
#: does.
#:
#: **Set from one dataset on one cell**, with a factor of six of headroom over
#: the largest measured residue. If PAW's 3.26e-07 is ever traced to a term --
#: ``OPEN.md`` Y4 is the thread -- this wants revisiting downwards rather than
#: leaving as the number that happened to pass.
CONSISTENCY_RY = 2.0e-06

#: What the displaced table is worth. Anything above this and the transverse
#: augmentation charge is simply absent; the measured value is 1.15e-03.
WITHOUT_THE_DISPLACEMENT_RY = 1.0e-04


def _pseudos(system, pseudo_dir: Path):
    return tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)


@lru_cache(maxsize=2)
def _run(text: str, pseudo_dir: Path):
    """SCF on an input given as text, so a test can vary ``spiral_q`` in place.

    ``maxsize = 2`` rather than ``None``, which is this repository's bound on a
    converged-state cache: two is what a comparison between two cells needs and
    is the largest that is not a leak (``CLAUDE.md``, "Memory is part of the
    design"). These cells hold spinor wavefunctions on a 60x60x96 box.
    """
    pwin = parse_pw_input(text)
    system = build_system(pwin)
    return system, run_scf(
        system,
        _pseudos(system, pseudo_dir),
        conv_thr=1e-11,
        mixing_beta=0.3,
        max_iterations=300,
    )


def _at_q(name: str, q3: float) -> str:
    text = (GENERATED / name).read_text()
    return text.replace("spiral_q(3) = 0.25", f"spiral_q(3) = {q3}")


def _electronic(result) -> float:
    """The total energy without the Ewald term, per unit cell of the input.

    Two cells of different size do not compute the same Ewald sum to better than
    QE's own truncation tolerance, and that difference would swamp the identity.
    ``test_spin_spirals.py`` has the full reasoning.
    """
    return result.total_energy - result.energy_terms["ewald"]


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """Keep XLA's executable cache from accumulating across these cells.

    Every cell here has its own shapes, so each compiles the whole SCF stack
    afresh and XLA keeps every executable for the life of the process. The
    results stay cached; only the compiled code is dropped.
    """
    yield
    import jax

    jax.clear_caches()


@pytest.mark.parametrize("dataset", ["us", "paw"])
def test_a_zero_spiral_is_an_ordinary_noncollinear_run(dataset, pseudo_dir):
    """``q = 0``: the displaced table *is* the resident one, term for term.

    The cheapest of the identities and the one that isolates the plumbing --
    the doubled k-list, the per-component projectors, the cross ``becsum``
    between two spheres, and the recombination of the two real transverse
    components into one complex field and back. At ``q = 0`` all of it has to
    collapse onto the ordinary noncollinear ultrasoft path that P17 validated,
    and it does so to round-off rather than to the consistency floor below,
    because both sides are then literally the same calculation: the displaced
    table at ``q = 0`` is the resident one, array for array.

    It says nothing about the *displacement*: at ``q = 0`` a wrong sign, a
    wrong magnitude and a missing structure factor are all invisible. The
    supercell identities below are what see those.
    """
    spiral = _at_q(f"o-chain-spiral-{dataset}.in", 0.0)
    plain = spiral.replace(
        "    spiral_q(1) = 0.0, spiral_q(2) = 0.0, spiral_q(3) = 0.0", ""
    )

    system, result = _run(spiral, pseudo_dir)
    plain_system, reference = _run(plain, pseudo_dir)

    assert system.spiral and not plain_system.spiral
    assert result.converged and reference.converged
    assert result.total_energy == pytest.approx(reference.total_energy, abs=1e-10)


@pytest.mark.parametrize("dataset", ["us", "paw"])
def test_a_quarter_turn_is_the_ninety_degree_supercell(dataset, pseudo_dir):
    """``q = b3/4`` against a four-cell noncollinear supercell.

    **The sharp one**, and the only test here that sees the *sign* of the
    displacement. At ``q = b3/2`` the wavevector is its own negative modulo
    ``b3``, so ``Q_ij(G - q)`` and ``Q_ij(G + q)`` differ by a relabelling of
    the G set and the sign is nearly invisible; at ``q = b3/4`` it is not.
    Both sides are noncollinear and augmented, so nothing differs between them
    except whether the transverse augmentation charge is carried by the
    displaced table or by four cells of supercell.

    Measured: **1.65e-09 Ry** for ultrasoft and **3.26e-07 Ry** for PAW, both
    at ``ecutrho = 200``.
    """
    system, spiral = _run(_at_q(f"o-chain-spiral-{dataset}.in", 0.25), pseudo_dir)
    supercell_system, supercell = _run(
        (GENERATED / f"o-chain-90deg-{dataset}.in").read_text(), pseudo_dir
    )

    assert spiral.converged and supercell.converged
    assert supercell_system.nspin_mag == 4 and not supercell_system.spiral
    assert _electronic(spiral) == pytest.approx(
        _electronic(supercell) / 4.0, abs=CONSISTENCY_RY
    )


@pytest.mark.parametrize("dataset", ["us", "paw"])
def test_moving_the_atom_does_not_move_the_energy(dataset, pseudo_dir):
    """A one-atom cell is translation invariant, displaced table included.

    The structure factor of the transverse block is ``e^{-i (G - q).tau_a}``
    and not ``e^{-i G.tau_a}``, and every other test here puts the atom at the
    origin, where the two are both 1. This is the one that tells them apart:
    with a wrong phase in either the plane-wave augmentation or PAW's
    one-centre terms, the total energy acquires a dependence on where the atom
    sits.

    **It is also the measurement behind a refusal that was withdrawn.** The
    old refusal of this combination named a per-atom phase ``e^{-i q.tau/2}``
    (Elk's ``zqss``) that PAW's transverse one-centre term was said to need.
    It does not need one: ``becsum`` between the two components already carries
    ``e^{i q.tau}`` through its two structure factors, and the one-centre
    energy depends on ``|m|``, which a position-dependent spin rotation leaves
    alone pointwise. This test is what turns that argument into a number --
    **2.9e-12 Ry** for ultrasoft and **1.4e-11** for PAW.

    ``tau_z = 1/3`` rather than an arbitrary displacement, because 1/3 of the
    cell is an exact number of points on both FFT grids here (24 dense, 18
    smooth). At ``tau_z = 0.3``, which is not, the same comparison reads
    4.4e-06 Ry -- the egg-box error of a density sampled on a fixed grid, which
    has nothing to do with the spiral and would otherwise be mistaken for one.
    """
    at_origin = _at_q(f"o-chain-spiral-{dataset}.in", 0.5)
    moved = at_origin.replace(" O 0.0 0.0 0.0", " O 0.0 0.0 0.33333333333333")

    _, here = _run(at_origin, pseudo_dir)
    _, there = _run(moved, pseudo_dir)

    assert here.converged and there.converged
    assert there.total_energy == pytest.approx(here.total_energy, abs=1e-10)


def test_half_a_reciprocal_vector_is_the_antiferromagnet(pseudo_dir):
    """``q = b3/2`` against the collinear LSDA antiferromagnet of the doubled cell.

    Weaker than the quarter turn in two ways worth stating rather than
    discovering: ``q = b3/2`` is its own negative modulo ``b3``, so it is blind
    to the sign of the displacement; and the reference is **collinear**, so the
    two sides do not share the noncollinear machinery and the comparison
    carries that difference as well. It earns its place by tying the augmented
    spiral to the LSDA path rather than to another noncollinear run.

    Measured at 3.61e-07 Ry, and the module docstring's ladder is this identity
    at three cutoffs -- together with the control that shows the ladder is the
    collinear/noncollinear gap rather than the spiral's.
    """
    system, spiral = _run(_at_q("o-chain-spiral-us.in", 0.5), pseudo_dir)
    doubled_system, doubled = _run(
        (GENERATED / "o-chain-afm-us.in").read_text(), pseudo_dir
    )

    assert spiral.converged and doubled.converged
    assert doubled_system.nspin == 2  # the reference is not a spiral at all
    assert _electronic(spiral) == pytest.approx(
        _electronic(doubled) / 2.0, abs=CONSISTENCY_RY
    )


def test_the_resident_table_is_not_enough(pseudo_dir, monkeypatch):
    """The guard fires: without the displacement the identity fails outright.

    A tolerance is only a test if something fails it, and every number above is
    small. This feeds the same calculation a transverse augmentation charge
    built from the **resident** ``Q_ij(G)`` -- which is what the code did before
    the displaced table existed, and what it would silently go back to if the
    ``shift`` were ever dropped on the way through -- and measures how far out
    that is: **1.15e-03 Ry**, against the 3.6e-07 the displacement leaves.
    """
    import defumat.scf.driver as driver
    from defumat.pseudo.augmentation import build_augmentation

    def without_the_shift(*args, **kwargs):
        kwargs.pop("shift", None)
        return build_augmentation(*args, **kwargs)

    monkeypatch.setattr(driver, "build_augmentation", without_the_shift)

    # Not through ``_run``: its cache is keyed on the text alone and would hand
    # back the correctly built state from another test in the same process.
    def run(text):
        pwin = parse_pw_input(text)
        system = build_system(pwin)
        return run_scf(
            system, _pseudos(system, pseudo_dir), conv_thr=1e-11,
            mixing_beta=0.3, max_iterations=300,
        )

    spiral = run(_at_q("o-chain-spiral-us.in", 0.5))
    doubled = run((GENERATED / "o-chain-afm-us.in").read_text())

    assert spiral.converged and doubled.converged
    error = abs(_electronic(spiral) - _electronic(doubled) / 2.0)
    assert error > WITHOUT_THE_DISPLACEMENT_RY
