"""Elk's converged ground state, reconstructed here and used as a seed.

``PLAN.md`` P72. Two kinds of check, and the second is the one the phase turns
on.

**The reconstruction against Elk's own output.** ``RHO3D.OUT`` is what Elk
prints when it evaluates its density on a grid, and it is produced by calling
the very routine transcribed here, so a pointwise comparison over every printed
value says the transcription is right and not merely plausible. ``chgmt`` and
``chgir`` from ``INFO.OUT`` check the two integrals separately, which matters
because they are computed in completely different ways -- a spline quadrature
on a logarithmic radial mesh against a Fourier-truncated step function on a
uniform grid.

**The identity.** A seed is only a seed if the run lands where it would have
landed anyway. Starting from Elk's density and starting from a superposition of
atomic charges must converge to the same state, and the check is the total
energy plus the converged density itself, not the energy alone.

The seed runs on simple-cubic hydrogen at ``a = 3.0`` bohr, one electron, which
is the one element where an all-electron density and a pseudopotential valence
density are the same function -- there is no core to subtract. The
reconstruction is checked on zincblende SiC as well, where the two species have
different radial meshes and the density at the nuclei differs by a factor of
sixteen, so an atom index taken in the wrong order cannot hide.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from defumat import Calculator
from defumat.io.elk import read_elk_state

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "elk" / "h_sc"
PSEUDO = ROOT / "data" / "pseudo"

needs_fixture = pytest.mark.skipif(
    not (FIXTURE / "STATE.OUT").is_file(),
    reason=f"no Elk fixture at {FIXTURE}",
)


@lru_cache(maxsize=2)
def calculator() -> Calculator:
    """A fresh calculator on the fixture's own input.

    ``maxsize=2`` rather than ``None``: what this holds after a run is the
    wavefunctions, and an unbounded cache of those is a leak.
    """
    return Calculator.from_file(FIXTURE / "scf.in", pseudo_dir=PSEUDO)


@lru_cache(maxsize=2)
def state():
    return read_elk_state(FIXTURE)


def read_rho3d(path):
    rows = []
    with open(path) as handle:
        handle.readline()
        for line in handle:
            values = line.split()
            if len(values) == 4:
                rows.append([float(x) for x in values])
    data = np.array(rows)
    return data[:, :3], data[:, 3]


# --- the reconstruction against Elk's own output ------------------------------

@needs_fixture
def test_the_density_matches_elks_own_evaluation_of_it():
    """Every point Elk printed, to the precision Elk printed it at.

    The residual is the file's ``G18.10`` format and nothing else, which is why
    the tolerance is absolute rather than relative: it is the print floor. That
    it *is* the print floor is what says the 4-point Lagrange interpolation was
    transcribed rather than approximated -- a cubic spline on the same
    logarithmic mesh would agree to interpolation error instead, which is
    orders of magnitude larger and impossible to read as either agreement or
    disagreement.

    Both regions are exercised: 1743 of the 4096 points fall inside the muffin
    tin and take the radial-times-harmonic branch, the rest take the Fourier
    sum.
    """
    points, reference = read_rho3d(FIXTURE / "RHO3D.OUT")
    assert len(points) == 4096

    elk = state()
    values = elk.evaluate_at(points, coordinates="cartesian")

    assert np.abs(values - reference).max() < 1e-9
    assert np.abs(values / reference - 1.0).max() < 1e-8

    # Both branches were taken, so neither is untested by accident.
    lattice = points @ np.linalg.inv(elk.geometry.avec)
    folded = lattice - np.rint(lattice)
    radius = np.linalg.norm(folded @ elk.geometry.avec, axis=1)
    inside = int((radius < elk.rmt[0]).sum())
    assert 1000 < inside < 3000


@needs_fixture
def test_the_two_charges_match_the_printed_ones():
    """``chgmt`` and ``chgir``, which are computed in unrelated ways.

    Elk's ``rhonorm`` adds a uniform constant to the interstitial and to the
    ``l = 0`` channel of every sphere so the total comes out exactly right, and
    then updates both numbers -- so these are the post-shift values and they
    sum to the electron count by construction. The printed
    ``total calculated charge`` and its 7e-4 error are pre-shift and are not
    what these describe.
    """
    elk = state()
    chgmt = float(elk.muffin_tin_charges().sum())
    chgir = elk.interstitial_charge()

    assert chgmt == pytest.approx(0.6125761996, abs=1e-9)
    assert chgir == pytest.approx(0.3874238004, abs=1e-9)
    assert chgmt + chgir == pytest.approx(1.0, abs=1e-9)


@needs_fixture
def test_the_printed_interstitial_charge_is_not_a_check_on_a_sharp_sphere():
    """Why ``chgir`` above agrees and a sharp in-or-out sum would not.

    Elk's ``cfunir`` is the *Fourier-truncated* step function, so it rings at
    the sphere boundary and is not zero inside it. Integrating the same
    interstitial density against a sharp step instead gives 0.3847, which is
    0.7 per cent away -- small, but a hundred thousand times the agreement
    above, so an implementation that used the sharp step would look broken
    against ``INFO.OUT`` while being perfectly sensible physics.
    """
    from defumat.io.elk_density import sharp_interstitial_charge

    elk = state()
    sharp = sharp_interstitial_charge(elk)
    truncated = elk.interstitial_charge()

    assert sharp == pytest.approx(0.38465939570920, abs=1e-9)
    assert abs(sharp - truncated) == pytest.approx(2.7644e-3, rel=1e-3)
    assert abs(sharp - truncated) / truncated == pytest.approx(7.14e-3, rel=1e-2)


# --- the seed -----------------------------------------------------------------

@needs_fixture
def test_the_seed_lands_on_the_grid_carrying_the_right_charge():
    """What the reconstruction integrates to before anything is renormalised.

    1.000298 against one electron. The residual is the sharp-versus-truncated
    boundary of the test above, transferred: the muffin tin is written in
    pointwise and the interstitial comes from a Fourier sum, and the two meet
    at a sphere neither of them resolves exactly.

    ``outside_fraction`` is zero here because ``ecutrho = 160`` Ry exceeds
    Elk's own ``gmaxvr = 12`` bohr^-1, i.e. 144 Ry, so nothing at all is
    truncated on the way in.
    """
    report = []
    seed = calculator().get_elk_seed(FIXTURE, report=report)
    measured = report[0]

    assert seed.shape == (1, 15, 15, 15)
    assert measured.integral == pytest.approx(1.000298, abs=1e-5)
    assert measured.nelec == 1.0
    assert measured.ngvec_kept == measured.ngvec == 751
    assert measured.outside_fraction == 0.0

    omega = state().omega
    integral = float(np.asarray(seed).sum()) * omega / np.asarray(seed[0]).size
    assert integral == pytest.approx(1.0, abs=1e-12)


@needs_fixture
def test_a_seeded_run_and_an_atomic_one_converge_to_the_same_state():
    """The phase's central claim: this is a seed and not a perturbation.

    Both runs are converged to ``conv_thr = 1e-10``, so agreement at 1e-13 Ry
    is well inside the tolerance either of them was asked for. The density is
    checked too, because a total energy is stationary and can agree while the
    state behind it does not.

    It buys no iterations, and that is the honest result rather than a
    disappointing one: for one hydrogen atom a superposition of atomic charges
    is already almost the answer, and Elk's all-electron cusp inside the
    pseudisation radius is a place where its density is *further* from the
    pseudo one than the atomic guess is.
    """
    seeded_calc = Calculator.from_file(FIXTURE / "scf.in", pseudo_dir=PSEUDO)
    seed = seeded_calc.get_elk_seed(FIXTURE)
    seeded = seeded_calc.get_scf(starting_density=seed)

    atomic = calculator().get_scf()

    assert seeded.total_energy == pytest.approx(atomic.total_energy, abs=1e-10)
    assert abs(seeded.total_energy - atomic.total_energy) < 1e-12
    assert seeded.iterations == atomic.iterations == 4

    difference = np.abs(np.asarray(seeded.density) - np.asarray(atomic.density))
    assert difference.max() < 1e-6


# --- what it refuses ----------------------------------------------------------

@needs_fixture
def test_a_seed_for_a_different_cell_is_refused():
    """A density is transferable between identical lattices and nothing else.

    There is no interpolation here that would make a transfer between two
    different cells mean anything, so the mismatch is named rather than
    silently accepted on a grid that happens to have the same shape.
    """
    moved = Calculator.from_file(FIXTURE / "scf.in", pseudo_dir=PSEUDO)
    stretched = moved.with_cell(np.asarray(moved.system.cell.at) * 1.05)
    with pytest.raises(ValueError, match="not the same cell"):
        stretched.get_elk_seed(FIXTURE)


@needs_fixture
def test_a_spin_polarized_run_cannot_take_an_unpolarized_seed():
    """Refused before the reconstruction rather than after it.

    Splitting a charge density evenly between two channels would start a
    magnetic run from a non-magnetic guess without saying so, which is a
    physics error dressed as a convenience. Transferring Elk's magnetization is
    a second field with a transfer rule of its own and is not implemented.
    """
    polarized = calculator().with_spin(2, starting_magnetization=0.5)
    with pytest.raises(NotImplementedError, match="nspin_mag"):
        polarized.get_elk_seed(FIXTURE)


# --- a two-species crystal ----------------------------------------------------

TWO_SPECIES = ROOT / "data" / "elk" / "sic_zb"

needs_sic = pytest.mark.skipif(
    not (TWO_SPECIES / "STATE.OUT").is_file(),
    reason=f"no Elk fixture at {TWO_SPECIES}",
)


@needs_sic
def test_the_reconstruction_holds_on_a_two_species_crystal():
    """Zincblende SiC, where the two species have different radial meshes.

    Three residuals with three different causes, and conflating any two of them
    invents a problem that is not there.

    * the **median**, 2.4e-10 relative, is the printed *value's* own rounding.
      ``RHO3D.OUT`` is ``(7G18.10)``, so ten significant digits, and hydrogen's
      7.3e-11 is the same floor. Nothing can be measured below it.
    * the worst **relative** point, 7.2e-9, is the printed *coordinates'*
      rounding times the density's gradient, and it is the only one of the three
      that says anything about the reconstruction's inputs. It sits 0.36 bohr
      from a carbon nucleus where ``|grad rho|`` is 20 e/bohr^4, and 5e-10 bohr
      on two coordinate columns is worth 1.5e-8 there. Hydrogen has no such tail
      because its grid step is ``3/16 = 0.1875`` and every coordinate prints
      exactly, where this cell's is ``4.119225/16`` and needs eleven digits.
    * the worst **absolute**, 3.5e-7, is the value's rounding again seen through
      a large density: it is at the *silicon nucleus*, where rho is 2094 and the
      ten-digit floor is 5e-7. That point is not an interpolation at all --
      ``rfpts`` clamps ``r`` up to ``rsp(1)`` at a nucleus -- so the largest
      absolute disagreement in the file is at its cleanest point.

    So a tighter comparison anywhere needs Elk to print more digits, not a
    better transcription, and hydrogen's number is not the achievable target.
    """
    elk = read_elk_state(TWO_SPECIES)
    points, reference = read_rho3d(TWO_SPECIES / "RHO3D.OUT")
    assert len(points) == 4096

    values = elk.evaluate_at(points, coordinates="cartesian")
    relative = np.abs(values / reference - 1.0)

    assert np.median(relative) < 5e-10
    assert relative.max() < 1e-8
    assert np.abs(values - reference).max() < 1e-6

    # The two nuclei, where the sum is the l = 0 term alone and exact. They
    # differ by a factor of sixteen, which is what makes this fixture -- and
    # not diamond -- the one that can detect a swapped atom index: on diamond
    # the two carbons are related by inversion and print the same value.
    y00 = 1.0 / np.sqrt(4.0 * np.pi)
    assert elk.rhomt[0, 0, 0] * y00 == pytest.approx(129.9024835, abs=1e-6)
    assert elk.rhomt[0, 0, 1] * y00 == pytest.approx(2094.5177730, abs=1e-5)


@needs_sic
def test_the_charges_of_a_two_species_crystal_match_the_printed_ones():
    """Per atom, on meshes of different lengths, against ``INFO.OUT``."""
    elk = read_elk_state(TWO_SPECIES)
    charges = elk.muffin_tin_charges()

    assert charges[0] == pytest.approx(4.832618957, abs=1e-8)
    assert charges[1] == pytest.approx(11.73695755, abs=1e-7)
    assert elk.interstitial_charge() == pytest.approx(3.430423495, abs=1e-8)
    assert float(charges.sum()) + elk.interstitial_charge() == pytest.approx(
        20.0, abs=1e-8
    )
