"""``System.with_moments``: the Python route to a ``STARTING_MOMENTS`` card.

A texture could only be *stated* in an input file. The machinery to act on one
existed -- ``_respin_kpoints`` already forwards ``per_atom`` -- and no method
reached it, so a Python sweep over magnetic configurations meant writing a file
per configuration or reaching for ``dataclasses.replace``.

**The workaround is the hazard, and it is silent.** ``replace`` swaps the field
and leaves ``kpoints`` reduced with the group the *old* moments had, while
``System.symmetry_group()`` is a property recomputed from the new ones. The run
then symmetrises with a group smaller than the group its k-set was reduced
with, and there is no check anywhere that compares them.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from defumat.io.pwin import parse_pw_input
from defumat.system.builder import build_system

pytestmark = pytest.mark.unit

#: Four hydrogen atoms in a chain, one species, ferromagnetic and noncollinear,
#: with symmetry left on so the k-set is actually reduced. The same file the
#: user guide's ``with_moments`` snippet runs on.
FERROMAGNET = Path("tests/data/qe/h4-chain-ferro.in").read_text()

CYCLOID = np.array(
    [[np.sin(a), 0.0, np.cos(a)] for a in np.deg2rad([0.0, 90.0, 180.0, 270.0])]
) * 0.6


@pytest.fixture(scope="module")
def ferromagnet():
    return build_system(parse_pw_input(FERROMAGNET))


def test_the_replace_workaround_leaves_the_kset_reduced_with_the_wrong_group(
        ferromagnet):
    """The defect this method exists to remove. No SCF needed to see it."""
    assert ferromagnet.symmetry_group().nsym == 16
    assert ferromagnet.kpoints.nk == 9

    naive = dataclasses.replace(
        ferromagnet, starting_moments=tuple(map(tuple, CYCLOID)))
    # The group the density would be symmetrised with has dropped ...
    assert naive.symmetry_group().nsym == 4
    # ... and the k-set is still the ferromagnet's, unchanged.
    assert naive.kpoints.nk == ferromagnet.kpoints.nk


def test_with_moments_rebuilds_the_kpoints_for_the_new_texture(ferromagnet):
    textured = ferromagnet.with_moments(CYCLOID)
    assert textured.symmetry_group().nsym == 4
    assert textured.kpoints.nk == 12
    assert textured.kpoints.nk > ferromagnet.kpoints.nk, (
        "a smaller group needs more k-points, not fewer")
    assert np.allclose(np.asarray(textured.starting_moments), CYCLOID)


def test_dropping_the_card_restores_the_per_species_run(ferromagnet):
    """``None`` means "no card", which is a value rather than "unspecified"."""
    back = ferromagnet.with_moments(CYCLOID).with_moments(None)
    assert back.starting_moments == ()
    assert back.symmetry_group().nsym == ferromagnet.symmetry_group().nsym
    assert back.kpoints.nk == ferromagnet.kpoints.nk


def test_the_shape_is_checked_against_the_atom_count(ferromagnet):
    with pytest.raises(ValueError, match=r"\(4, 3\)"):
        ferromagnet.with_moments(np.zeros((3, 3)))
    with pytest.raises(ValueError, match=r"\(4, 3\)"):
        ferromagnet.with_moments(np.zeros(4))


def test_a_collinear_run_still_refuses_a_transverse_row(ferromagnet):
    """The refusal ``local_moments`` already makes: ``nspin = 2`` has one
    component, so an x or y part is an error rather than a projection."""
    collinear = build_system(parse_pw_input(
        FERROMAGNET.replace("noncolin = .true.", "nspin = 2")))
    collinear.with_moments(np.array([[0.0, 0.0, m] for m in (1, -1, 1, -1)]))
    with pytest.raises(ValueError):
        collinear.with_moments(CYCLOID)


def test_the_calculator_forwards_it_and_keeps_no_stale_answer():
    """A different texture is a different calculation, so the cache is empty
    and the converged state crosses as a seed rather than as an answer."""
    from defumat import Calculator

    calculator = Calculator.from_text(FERROMAGNET, pseudo_dir="tests/data/pseudo")
    textured = calculator.with_moments(CYCLOID)
    assert np.allclose(np.asarray(textured.system.starting_moments), CYCLOID)
    assert textured.system.kpoints.nk == 12
    assert textured._scf is None
