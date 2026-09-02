"""The structure factor machinery: the H-vector set, the two conventions in
``F(H)``, and the guards.

Nothing here runs an SCF. What is checked is the part a Fourier convention can
get wrong -- the sign of the phase, the ``1/Omega``, the wrap of a negative
index onto the top of an FFT axis -- and the reduction of a star, which is the
one piece of the enumeration that is not a loop over a box.

The anchor is :func:`structure_factors_of_field` with ``method="direct"``,
which is the definition summed over grid points and shares no index arithmetic
with the transform. A field built out of one cosine has an ``F`` that can be
written down, and that is what pins the convention absolutely rather than
against another implementation of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from defumat.diffraction.structure_factor import (
    conventional_transform,
    h_vectors,
    structure_factors_of_field,
    symmorphic_rotations,
)
from defumat.io.pwin import parse_pw_input
from defumat.system.builder import build_system

pytestmark = pytest.mark.unit


SILICON = """
 &control
    calculation = 'scf'
 /
 &system
    ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1,
    ecutwfc = 12.0,
 /
 &electrons
 /
ATOMIC_SPECIES
 Si  28.086  Si.pz-vbc.UPF
ATOMIC_POSITIONS (alat)
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS {automatic}
 2 2 2 0 0 0
"""


def silicon():
    return build_system(parse_pw_input(SILICON))


class _Result:
    """The little of an ``SCFResult`` a structure factor consumes."""

    def __init__(self, density):
        self.density = density


# --- the H-vector set --------------------------------------------------------


def test_every_enumerated_vector_is_inside_the_cutoff_and_none_is_missing():
    """The box of Miller indices searched must cover the sphere exactly.

    ``|H . a_i| = 2 pi |m_i|`` bounds each index by ``hmax |a_i| / 2 pi``, and
    a bound that is too tight loses reflections silently -- they simply do not
    appear in the table. The check is against a brute-force enumeration over a
    box two indices wider in every direction.
    """
    cell = silicon().cell
    hmax = 4.0
    vectors = h_vectors(cell, hmax, reduce=False)
    assert np.all(vectors.length <= hmax + 1.0e-9)

    bg = np.asarray(cell.bg)
    span = range(-8, 9)
    brute = [(i, j, k) for i in span for j in span for k in span
             if np.linalg.norm(np.array([i, j, k]) @ bg) <= hmax + 1.0e-9]
    assert len(brute) == len(vectors)
    assert {tuple(row) for row in vectors.miller} == set(brute)


def test_the_set_is_sorted_by_length_and_starts_at_the_origin():
    vectors = h_vectors(silicon().cell, 3.0, reduce=False)
    assert tuple(vectors.miller[0]) == (0, 0, 0)
    assert np.all(np.diff(vectors.length) >= -1.0e-12)


def test_the_multiplicities_account_for_every_vector():
    """A reduced set plus its multiplicities is the unreduced set.

    This is the whole content of the reduction: no reflection may be dropped
    and none counted twice.
    """
    system = silicon()
    whole = h_vectors(system.cell, 4.0, reduce=False)
    reduced = h_vectors(system.cell, 4.0, symmetries=system.symmetry_group(),
                        reduce=True)
    assert reduced.multiplicity.sum() == len(whole)
    assert len(reduced) < len(whole)


def test_only_the_symmorphic_operations_reduce_a_star():
    """Diamond's group is half symmorphic, and the other half must not be used.

    An operation with a fractional translation ``f`` sends ``F(H)`` to
    ``e^{-i 2 pi H.f} F(H)``: equal in modulus, different in phase, so
    collapsing a star with it averages numbers that are not equal. Silicon has
    48 operations of which 24 carry the diamond glide's ``(1/4, 1/4, 1/4)``.
    """
    symmetries = silicon().symmetry_group()
    kept = symmorphic_rotations(symmetries)
    translations = symmetries.translation_array()
    moving = sum(1 for t in translations
                 if np.any(np.minimum(np.abs(t), np.abs(1.0 - t)) > 1.0e-6))
    assert symmetries.nsym == 48
    assert moving == 24
    assert len(kept) == 24


def test_the_conventional_transform_labels_the_cubic_reflections():
    """A primitive fcc triple is not a Miller index anyone would recognise.

    The transform is Elk's ``vhmat``, and on a cubic Bravais lattice it must
    come out integer -- every primitive reciprocal vector is a conventional
    one. The (111) shell is what identifies it: ``|H| = 2 pi sqrt(3) / a``.
    """
    cell = silicon().cell
    matrix = conventional_transform(cell)
    assert np.allclose(matrix, np.rint(matrix))

    vectors = h_vectors(cell, 1.5, reduce=False, transform=matrix)
    assert np.allclose(vectors.indices, np.rint(vectors.indices))
    shell = np.abs(vectors.length - 2 * np.pi * np.sqrt(3) / float(cell.alat)) < 1e-9
    assert shell.sum() == 8
    for index in vectors.indices[shell]:
        assert sorted(np.abs(index).tolist()) == [1.0, 1.0, 1.0]


# --- the two conventions in F(H) --------------------------------------------


def test_one_cosine_has_a_structure_factor_that_can_be_written_down():
    """``f(r) = c (1 + cos(2 pi H0.s))`` has ``F(H0) = c Omega / 2``.

    The absolute pin on the convention: an amplitude, a factor of two and the
    volume, none of which survives a wrong normalisation, and a *real* field
    whose two coefficients sit at ``+H0`` and ``-H0`` -- so a sign error in the
    phase is invisible here and is caught by the plane wave below.
    """
    cell = silicon().cell
    grid = (12, 12, 12)
    axes = np.meshgrid(*[np.arange(n) / n for n in grid], indexing="ij")
    h0 = np.array([1, -2, 3])
    field = 3.0 * (1.0 + np.cos(2 * np.pi * sum(h0[i] * axes[i] for i in range(3))))
    miller = np.array([[0, 0, 0], h0, -h0])

    volume = float(cell.volume)
    factors = structure_factors_of_field(field, cell, miller)
    assert np.allclose(factors[0], 3.0 * volume)
    assert np.allclose(factors[1], 1.5 * volume)
    assert np.allclose(factors[2], 1.5 * volume)


def test_the_phase_has_the_crystallographers_sign():
    """``F(H) = int f e^{+iH.r}``, which is the conjugate of ``rho(H)``.

    ``sfacrho.f90`` prints ``Omega * conj(zftrf)`` because a crystallographic
    transform carries the positive phase. A complex field is what tells the two
    apart: for ``f = e^{i 2 pi H0.s}`` the positive-phase transform puts the
    whole cell on ``-H0`` and nothing on ``+H0``.
    """
    cell = silicon().cell
    grid = (9, 9, 9)
    axes = np.meshgrid(*[np.arange(n) / n for n in grid], indexing="ij")
    h0 = np.array([2, 1, 0])
    field = np.exp(2j * np.pi * sum(h0[i] * axes[i] for i in range(3)))
    miller = np.array([h0, -h0])

    factors = structure_factors_of_field(field, cell, miller)
    assert abs(factors[0]) < 1.0e-10
    assert np.allclose(factors[1], float(cell.volume))


def test_the_definition_and_the_transform_agree_on_a_random_density():
    """``method="direct"`` is the definition and shares no indexing with the FFT.

    A random *complex* field has no symmetry to hide behind, and it is complex
    on purpose: conjugating the coefficient and transforming with the positive
    phase are the same operation on a real field and different ones here, so a
    density would not tell them apart.
    """
    cell = silicon().cell
    rng = np.random.default_rng(20260902)
    field = rng.normal(size=(8, 9, 10)) + 1j * rng.normal(size=(8, 9, 10))
    vectors = h_vectors(cell, 2.5, reduce=False)
    fast = structure_factors_of_field(field, cell, vectors.miller, method="fft")
    slow = structure_factors_of_field(field, cell, vectors.miller, method="direct")
    assert np.abs(fast - slow).max() < 1.0e-12


def test_a_real_field_gives_conjugate_factors_at_plus_and_minus_H():
    cell = silicon().cell
    rng = np.random.default_rng(7)
    field = rng.normal(size=(8, 8, 8))
    miller = np.array([[1, 2, -3], [-1, -2, 3], [2, 0, 1], [-2, 0, -1]])
    factors = structure_factors_of_field(field, cell, miller)
    assert np.allclose(factors[0], np.conj(factors[1]))
    assert np.allclose(factors[2], np.conj(factors[3]))


def test_the_leading_axes_of_a_vector_field_are_carried_through():
    """The three components of a magnetization go through in one call."""
    cell = silicon().cell
    rng = np.random.default_rng(11)
    field = rng.normal(size=(3, 6, 6, 6))
    vectors = h_vectors(cell, 2.0, reduce=False)
    together = structure_factors_of_field(field, cell, vectors.miller)
    assert together.shape == (3, len(vectors))
    for component in range(3):
        alone = structure_factors_of_field(field[component], cell, vectors.miller)
        assert np.abs(together[component] - alone).max() < 1.0e-14


def test_a_frequency_the_grid_does_not_carry_is_refused():
    cell = silicon().cell
    field = np.zeros((6, 6, 6))
    with pytest.raises(ValueError, match="does not carry"):
        structure_factors_of_field(field, cell, np.array([[4, 0, 0]]))


# --- the guards on the workflow ---------------------------------------------


def test_a_cutoff_past_the_density_sphere_is_refused():
    """Past ``sqrt(ecutrho)`` the box carries aliasing, not the density.

    The one guard that cannot be replaced by a check on the answer: the
    coefficients out there are small, smooth and entirely plausible.
    """
    from defumat.workflows.sfac import run_structure_factors

    system = silicon()  # ecutwfc = 12, so ecutrho = 48 and sqrt is 6.93
    result = _Result(np.zeros((1, 16, 16, 16)))
    with pytest.raises(ValueError, match="beyond the density's own cutoff"):
        run_structure_factors(system, (), result, hmax=8.0)


def test_an_empty_or_inverted_energy_window_is_refused():
    from defumat.workflows.sfac import run_structure_factors

    system = silicon()
    result = _Result(np.zeros((1, 16, 16, 16)))
    result.wavefunctions = np.zeros((1, 2, 4, 10))
    result.eigenvalues = np.zeros((2, 4))
    result.occupations = np.ones((2, 4))
    with pytest.raises(ValueError, match="hi > lo"):
        run_structure_factors(system, (), result, window=(1.0, -1.0))
    with pytest.raises(ValueError, match="window is empty"):
        run_structure_factors(system, (), result, window=(10.0, 20.0))
