"""``tot_magnetization`` across :meth:`System.with_spin`, in both directions.

``OPEN.md`` Part XVI item 5. ``tot_magnetization`` fixes ``N_up - N_down`` by
giving each collinear channel its own Fermi level, so it means something only
at ``nspin = 2`` and ``pw.x`` refuses it anywhere else (``input.f90:781-782``).
``with_spin`` carried it unchanged into ``nspin = 1`` and 4, into a ``System``
no input can produce, where ``Calculation.two_fermi_energies`` then read it as
absent: the constraint was released and nothing said so. Going the other way,
entering ``nspin = 2`` with fixed occupations gave a two-channel run with no
rule for how the electrons divide, which the builder refuses from an input and
``with_spin`` let through to the first diagonalisation, and there was no way to
supply the value.

All host-side: a ``System`` from input text, no SCF.
"""

import dataclasses
import os
import warnings

import numpy as np
import pytest

from defumat.io.pwin import parse_pw_input
from defumat.system.builder import build_system

pytestmark = pytest.mark.unit

_SILICON = """
&control
  calculation = '{calculation}'
/
&system
  ibrav = 2, celldm(1) = 10.2, nat = 2, ntyp = 1, ecutwfc = 12.0
  {extra}
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS crystal
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""

#: Silicon's eight electrons split five and three, with the fixed occupations
#: the input defaults to. ``2`` rather than ``0`` so that a dropped value and a
#: zero one cannot be mistaken for each other.
_FIXED_MOMENT = "nspin = 2, starting_magnetization(1) = 0.1, tot_magnetization = 2"


def _system(extra: str = "", calculation: str = "scf"):
    return build_system(parse_pw_input(
        _SILICON.format(extra=extra, calculation=calculation)))


def _fixed_moment():
    system = _system(_FIXED_MOMENT)
    assert system.nspin == 2 and system.occupations == "fixed"
    assert system.tot_magnetization == 2.0
    return system


# --- out of nspin = 2: the carried value is released, out loud ---------------


@pytest.mark.parametrize("nspin, magnetization", [(1, (0.0,)), (4, None)])
def test_leaving_nspin_2_drops_the_fixed_moment_and_says_so(nspin, magnetization):
    """The constraint has no channels to act on, so it is released, with a warning.

    The warning points at the line that called ``with_spin``, which is where a
    reader can do something about it, rather than at ``builder.py`` or at the
    ``BoundMethod`` frame equinox puts between a ``System`` method and its
    caller, which is where a ``stacklevel`` of 2 sends it.
    """
    fixed = _fixed_moment()
    with pytest.warns(RuntimeWarning,
                      match=f"dropped on the way into nspin = {nspin}") as record:
        target = fixed.with_spin(nspin, starting_magnetization=magnetization)
    assert target.nspin == nspin
    assert target.tot_magnetization is None

    dropped = [w for w in record if "tot_magnetization" in str(w.message)]
    assert len(dropped) == 1
    assert os.path.basename(dropped[0].filename) == os.path.basename(__file__)


def test_an_explicit_none_releases_it_without_a_warning():
    """``None`` is the caller saying "drop it", so there is nothing to announce."""
    fixed = _fixed_moment()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        target = fixed.with_spin(4, tot_magnetization=None)
    assert target.tot_magnetization is None
    assert not [w for w in record if "tot_magnetization" in str(w.message)]


def test_staying_at_nspin_2_keeps_it():
    """The one regime where it means something carries it, silently."""
    fixed = _fixed_moment()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        same = fixed.with_spin(2)
        turned = fixed.with_spin(starting_magnetization=(0.2,))
    assert same.tot_magnetization == 2.0
    assert turned.tot_magnetization == 2.0
    assert not [w for w in record if "tot_magnetization" in str(w.message)]


def test_the_release_is_real_so_the_way_back_needs_the_moment_again():
    """Round trip 2 -> 4 -> 2 under fixed occupations: the value is gone.

    Coming back into ``nspin = 2`` is refused until it is supplied, which is
    what shows the constraint was released rather than hidden on a field the
    noncollinear run did not read.
    """
    with pytest.warns(RuntimeWarning, match="dropped on the way into nspin = 4"):
        spinor = _fixed_moment().with_spin(4)
    with pytest.raises(ValueError, match="needs tot_magnetization"):
        spinor.with_spin(2)
    assert spinor.with_spin(2, tot_magnetization=2).tot_magnetization == 2.0


# --- into nspin = 2: the builder's rules, in its own words --------------------


def test_entering_nspin_2_with_fixed_occupations_needs_a_moment():
    """Two channels filled independently need to be told how many each gets.

    The same words as the builder's refusal of the equivalent input, which is
    what "refused here as it is there" means.
    """
    words = "occupations = 'fixed' with nspin = 2 needs tot_magnetization"
    unpolarized = _system()
    assert unpolarized.occupations == "fixed"
    with pytest.raises(ValueError, match=words):
        unpolarized.with_spin(2, starting_magnetization=(0.3,))
    with pytest.raises(ValueError, match=words):
        _system("nspin = 2, starting_magnetization(1) = 0.3")
    # An explicit ``None`` into ``nspin = 2`` is the same request, made by hand.
    with pytest.raises(ValueError, match=words):
        _fixed_moment().with_spin(2, tot_magnetization=None)


def test_the_keyword_supplies_the_moment():
    """And the result is the ``System`` the equivalent input builds."""
    unpolarized = _system()
    polarized = unpolarized.with_spin(2, starting_magnetization=(0.3,),
                                      tot_magnetization=2)
    built = _system("nspin = 2, starting_magnetization(1) = 0.3, "
                    "tot_magnetization = 2")
    assert polarized.nspin == built.nspin == 2
    assert polarized.tot_magnetization == built.tot_magnetization == 2.0
    assert float(np.sum(polarized.kpoints.weights)) == pytest.approx(
        float(np.sum(built.kpoints.weights)))

    # Zero is a moment, not an absence: the nonmagnetic split, asked for.
    zero = unpolarized.with_spin(2, starting_magnetization=(0.3,),
                                 tot_magnetization=0)
    assert zero.tot_magnetization == 0.0


def test_a_fractional_moment_is_refused_under_fixed_occupations_only():
    """A channel fills a whole number of bands; a shared smearing does not care."""
    unpolarized = _system()
    with pytest.raises(ValueError,
                       match="needs an integer tot_magnetization, and this one is 0.5"):
        unpolarized.with_spin(2, starting_magnetization=(0.3,),
                              tot_magnetization=0.5)

    smeared = dataclasses.replace(unpolarized, occupations="smearing",
                                  degauss=0.02)
    assert smeared.with_spin(2, starting_magnetization=(0.3,),
                             tot_magnetization=0.5).tot_magnetization == 0.5


@pytest.mark.parametrize("nspin, magnetization", [(1, (0.0,)), (4, None)])
def test_a_moment_asked_for_outside_nspin_2_is_refused(nspin, magnetization):
    """Dropping is for what was carried, not for what was asked.

    A value passed to ``with_spin`` beside a target that cannot honour it is a
    request the run would silently not apply, so it stops, in the builder's
    words, and no drop warning goes out before the refusal.
    """
    fixed = _fixed_moment()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="tot_magnetization requires nspin = 2"):
            fixed.with_spin(nspin, starting_magnetization=magnetization,
                            tot_magnetization=2)
    assert not [w for w in record if "dropped" in str(w.message)]


def test_a_non_scf_run_is_exempt_from_the_fixed_occupation_rule():
    """``input.f90`` gates the rule on ``lscf``: an ``nscf`` run fills nothing."""
    nscf = _system(calculation="nscf")
    assert nscf.with_spin(2, starting_magnetization=(0.3,)).tot_magnetization is None


def test_the_calculator_forwards_the_keyword(pseudo_dir):
    """``Calculator.with_spin`` hands its options to ``System.with_spin``."""
    from defumat import Calculator

    calculator = Calculator.from_text(
        _SILICON.format(extra="", calculation="scf"), pseudo_dir, announce=False)
    promoted = calculator.with_spin(2, starting_magnetization=(0.3,),
                                    tot_magnetization=2)
    assert promoted.system.nspin == 2
    assert promoted.system.tot_magnetization == 2.0
