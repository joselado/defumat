"""Fixed LSDA occupations with a channel the constraint leaves empty.

``occupations = 'fixed'`` with ``nspin = 2`` is ``weights.f90``'s two-Fermi-level
branch: ``iweights`` once per channel, each filling ``NINT(nelup)`` or
``NINT(neldw)`` bands. ``iweights_only`` has no lower bound on that count, so a
channel with no electrons is filled with nothing, and ``iweights`` leaves its
level at the ``Ef = -1.0d+20`` it started from (``iweights.f90:51``). The fully
polarized hydrogen atom, one electron with ``tot_magnetization = 1``, is the case,
and ``pw.x`` runs it.

This package refused it (``OPEN.md`` Part XVI item 4): ``_fixed_occupations_spin``
raised whenever ``NINT(count) < 1``. The refusal is gone, and the one thing the
repair has to get right beyond not raising is the empty channel's HOMO:
``eigenvalues[channel, :, occupied - 1]`` at ``occupied = 0`` is index ``-1``,
which is the **last** band, so deleting the raise alone reports the top of the
computed spectrum as the minority channel's highest occupied level. Every
synthetic spectrum below puts a conspicuous number in that last band so that
reading it cannot pass.

All of it is on synthetic eigenvalues or on a stand-in for the calculation, so
nothing here runs an SCF; the self-consistent hydrogen atom is a regression
check, not a unit test.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from defumat.io.pwin import parse_pw_input
from defumat.scf.driver import Calculation
from defumat.scf.occupations import (
    EMPTY_CHANNEL_LEVEL,
    fixed_occupations,
    spin_electron_counts,
)
from defumat.system.builder import build_system

pytestmark = pytest.mark.unit


#: What the last band of every channel holds, far above everything else. Were the
#: empty channel's HOMO read as ``occupied - 1 = -1``, this is what would come
#: back.
TOP_OF_THE_SPECTRUM = 99.0


def _spectrum():
    """``(2, nk, nbnd)`` levels in Ry, non-degenerate and different per k.

    Channel 0 is the majority, lower in energy; channel 1 the minority. Two
    k-points with unequal weights, so that a maximum or minimum over k is a real
    choice rather than one number.
    """
    eigenvalues = np.array(
        [
            [[-0.60, -0.10, 0.20, TOP_OF_THE_SPECTRUM],
             [-0.55, -0.05, 0.25, TOP_OF_THE_SPECTRUM]],
            [[-0.30, 0.05, 0.30, TOP_OF_THE_SPECTRUM],
             [-0.35, 0.00, 0.35, TOP_OF_THE_SPECTRUM]],
        ]
    )
    weights = np.array([0.25, 0.75])
    return eigenvalues, weights


def test_an_empty_minority_channel_is_filled_with_nothing():
    """``counts = (1, 0)``: one majority band, no minority band, one electron."""
    eigenvalues, weights = _spectrum()
    wg, homo, lumo = fixed_occupations(
        eigenvalues, weights, 1.0, 2, counts=(1.0, 0.0)
    )
    wg, homo, lumo = np.asarray(wg), np.asarray(homo), np.asarray(lumo)

    assert wg.shape == eigenvalues.shape
    # The majority channel's lowest band carries each k-point's whole weight...
    assert np.allclose(wg[0, :, 0], weights)
    assert np.all(wg[0, :, 1:] == 0.0)
    # ...and the minority channel carries none at all.
    assert np.all(wg[1] == 0.0)
    assert wg.sum() == pytest.approx(1.0, abs=1e-14)

    # ``ef_up`` is the majority's highest occupied level, the maximum over k.
    assert homo[0] == pytest.approx(-0.55)
    # ``ef_dw`` is ``iweights``' sentinel, and above all it is not the last band.
    assert homo[1] == EMPTY_CHANNEL_LEVEL
    assert homo[1] != TOP_OF_THE_SPECTRUM
    # The LUMO of each channel is the minimum over k of its first empty band,
    # which for the empty channel is band 0 (``get_homo_lumo`` with kbnd = 0).
    assert lumo[0] == pytest.approx(-0.10)
    assert lumo[1] == pytest.approx(-0.35)
    assert np.all(np.isfinite(homo)) and np.all(np.isfinite(lumo))


def test_an_empty_majority_channel_is_the_mirror_image():
    """``tot_magnetization = -1``: the same atom with the moment the other way."""
    eigenvalues, weights = _spectrum()
    wg, homo, lumo = fixed_occupations(
        eigenvalues, weights, 1.0, 2, counts=(0.0, 1.0)
    )
    wg, homo, lumo = np.asarray(wg), np.asarray(homo), np.asarray(lumo)

    assert np.all(wg[0] == 0.0)
    assert np.allclose(wg[1, :, 0], weights)
    assert wg.sum() == pytest.approx(1.0, abs=1e-14)
    assert homo[0] == EMPTY_CHANNEL_LEVEL
    assert homo[1] == pytest.approx(-0.30)
    assert lumo[0] == pytest.approx(-0.60)
    assert lumo[1] == pytest.approx(0.00)


def test_the_reported_levels_are_the_occupied_channels():
    """What ``Calculation.occupations`` makes of the pair, on a stand-in.

    The scalar HOMO and LUMO are ``get_homo_lumo``'s: the HOMO over both
    channels skips the empty one, so it is the majority's level and not the
    sentinel, and the LUMO is the lowest empty state anywhere, which here is the
    empty channel's band 0 rather than the majority's band 1. The sentinel
    survives only as the empty channel's own ``fermi_energy_down``, and no
    shared ``fermi_energy`` is produced, since fixed occupations have none.
    """
    eigenvalues, weights = _spectrum()
    stand_in = SimpleNamespace(
        system=SimpleNamespace(
            kpoints=SimpleNamespace(weights=weights), occupations="fixed"
        ),
        noncolin=False,
        two_fermi_energies=True,
        nelec=1.0,
        nelup=1.0,
        neldw=0.0,
    )
    wg, levels = Calculation.occupations(stand_in, eigenvalues)

    assert np.asarray(wg).sum() == pytest.approx(1.0, abs=1e-14)
    assert levels["homo"] == pytest.approx(-0.55)
    assert levels["lumo"] == pytest.approx(-0.35)
    assert levels["fermi_energy_up"] == pytest.approx(-0.55)
    assert levels["fermi_energy_down"] == EMPTY_CHANNEL_LEVEL
    assert "fermi_energy" not in levels
    assert all(np.isfinite(value) for value in levels.values())


_HYDROGEN = """
&control
  calculation = 'scf'
/
&system
  ibrav = 1, celldm(1) = 12.0, nat = 1, ntyp = 1, ecutwfc = 10.0
  nspin = 2, occupations = 'fixed', tot_magnetization = {moment}
/
&electrons
/
ATOMIC_SPECIES
 H 1.008 H.pz-vbc.UPF
ATOMIC_POSITIONS (alat)
 H 0.0 0.0 0.0
K_POINTS automatic
 1 1 1 0 0 0
"""


@pytest.mark.parametrize(("moment", "counts"), [(1, (1.0, 0.0)), (-1, (0.0, 1.0))])
def test_the_fully_polarized_hydrogen_atom_reaches_the_fill(moment, counts):
    """The input reader accepts it, and the counts it implies are filled.

    ``input.f90:784-800`` asks fixed LSDA for an integer ``tot_magnetization``
    and an integer charge and nothing else, and the reader mirrors that, so the
    refusal was never on the input side; ``set_nelup_neldw`` then splits the one
    electron as ``(1, 0)`` or ``(0, 1)``, and those are the counts that used to
    stop at the first diagonalisation.
    """
    system = build_system(parse_pw_input(_HYDROGEN.format(moment=moment)))
    assert system.nspin == 2
    assert system.tot_magnetization == float(moment)
    # H.pz-vbc has one valence electron.
    assert spin_electron_counts(1.0, system.tot_magnetization) == counts

    eigenvalues, weights = _spectrum()
    wg, homo, _ = fixed_occupations(eigenvalues, weights, 1.0, 2, counts=counts)
    assert np.asarray(wg).sum() == pytest.approx(1.0, abs=1e-14)
    assert np.sum(np.asarray(homo) == EMPTY_CHANNEL_LEVEL) == 1


@pytest.mark.parametrize("counts", [(2.0, -1.0), (1.5, -0.5), (-1.0, 2.0)])
def test_a_moment_larger_than_the_electron_count_is_still_refused(counts):
    """A negative count is the one case where the total charge goes wrong.

    ``tot_magnetization = 3`` on one electron gives ``(2, -1)`` and ``= 2`` gives
    ``(1.5, -0.5)``; ``pw.x`` fills two majority bands either way, which is two
    electrons in a cell that has one, and stops on the density that follows
    ("charge is wrong", ``electrons.f90:1122-1128``). Here it stays refused at
    the fill, by name.
    """
    eigenvalues, weights = _spectrum()
    with pytest.raises(ValueError, match="larger than the number of electrons"):
        fixed_occupations(eigenvalues, weights, 1.0, 2, counts=counts)
