"""The Raman/IR assembly, as arithmetic -- ``RamanIR`` without a crystal under it.

``pypresso.response.spectra.mode_activities`` is a pure function of arrays, so
everything about it can be checked without solving anything: the
eigendisplacement normalisation, the two rotational invariants, the unit
factors, and -- the one that matters for how the results may be *used* -- what a
degenerate multiplet does to them.

The regression test beside this one runs the vendored ``dynmat.x`` on tensors
this code computed and compares the printed table. These are the checks that do
not need the Fortran.
"""

import numpy as np
import pytest

from pypresso.response.spectra import (
    degenerate_manifolds,
    eigendisplacements,
    mode_activities,
)
from pypresso.units import AMU_TO_RY

pytestmark = pytest.mark.unit

MASSES = np.array([26.98, 74.92])
VOLUME = 295.6
EPSILON = np.eye(3) * 12.9674


def _orthonormal(rng, n):
    matrix, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return matrix


def _zincblende_raman(value):
    """``d(chi)/d(tau)``, ``[atom, cart, i, j]``, in the only allowed form.

    ``-43m`` leaves one independent component: ``d(chi_yz)/d(tau_x)`` and its
    permutations. The two atoms carry opposite signs, which is the translational
    sum rule of :func:`~pypresso.response.nonlinear.translational_residue`.
    """
    tensor = np.zeros((2, 3, 3, 3))
    for cart in range(3):
        i, j = (cart + 1) % 3, (cart + 2) % 3
        tensor[0, cart, i, j] = tensor[0, cart, j, i] = value
        tensor[1, cart, i, j] = tensor[1, cart, j, i] = -value
    return tensor


def test_eigendisplacements_are_normalised_in_the_mass_metric(rng=None):
    """``<z|M|z> = 1`` with ``M`` in Rydberg mass units -- ``dyndiag``'s rule."""
    rng = np.random.default_rng(0)
    vectors = _orthonormal(rng, 6)
    z = eigendisplacements(vectors, MASSES)
    weighted = np.einsum("nac,a,nac->n", z, MASSES * AMU_TO_RY, z)
    assert np.allclose(weighted, 1.0, atol=1e-12)


def test_the_mode_index_is_the_column_of_the_eigenvector_matrix():
    """``eigenvectors[:, nu]`` is mode ``nu``, which is what ``eigh`` returns.

    Getting this transposed is silent -- the array is square and the result is
    still a plausible-looking spectrum -- so it is pinned with a mode that has a
    single nonzero entry.
    """
    vectors = np.zeros((6, 6))
    vectors[4, 1] = 1.0   # mode 1 moves atom 1 along y and nothing else
    z = eigendisplacements(vectors, MASSES)
    assert np.abs(z[1]).argmax() == 4
    assert z[1, 1, 1] == pytest.approx(1.0 / np.sqrt(MASSES[1] * AMU_TO_RY))
    assert np.abs(z[0]).max() == 0.0


def test_manifolds_are_runs_of_equal_frequencies():
    """The acoustic triplet is one multiplet and the optical one is another."""
    labels = degenerate_manifolds(np.array([-1.0, -0.9, -0.6, 500.0, 500.0, 500.0]))
    assert list(labels) == [0, 0, 0, 1, 1, 1]
    # The tolerance is a **gap between neighbours**, not a spread: three modes
    # 0.9 cm^-1 apart in a chain are one multiplet even though the outer two are
    # 1.8 apart. That is deliberate -- what it has to separate is a
    # symmetry-imposed degeneracy from a real splitting, and real splittings in
    # these cells are hundreds of cm^-1.
    assert list(degenerate_manifolds(np.array([0.0, 0.9, 1.8]))) == [0, 0, 0]
    assert list(degenerate_manifolds(np.array([0.0, 1.5, 3.0]))) == [0, 1, 2]


# -- the degeneracy rule ------------------------------------------------------


def test_a_degenerate_multiplet_is_only_comparable_as_a_sum():
    """The finding this module's docstring turns into a rule.

    Rotating the three members of a degenerate multiplet into each other is a
    change of basis the eigensolver was free to make (rule D4). Both invariants
    are *quadratic* in the mode's Raman tensor, so the multiplet's **sum** of
    activities is unchanged and its individual entries are not -- which is why
    :meth:`~pypresso.response.spectra.VibrationalSpectrum.by_manifold` exists
    and why a per-mode number inside a multiplet must never be compared against
    another code.
    """
    rng = np.random.default_rng(3)
    vectors = _orthonormal(rng, 6)
    frequencies = np.array([0.0, 0.0, 0.0, 500.0, 500.0, 500.0])
    # A tensor with no particular symmetry, so that nothing makes the per-mode
    # numbers accidentally invariant the way silicon's T_2g does.
    raman = rng.normal(size=(2, 3, 3, 3))
    raman = 0.5 * (raman + raman.transpose(0, 1, 3, 2))
    born = rng.normal(size=(2, 3, 3))

    def spectrum(basis):
        return mode_activities(
            frequencies, basis, MASSES, EPSILON, VOLUME, raman=raman, born=born
        )

    plain = spectrum(vectors)
    mixed = np.array(vectors)
    mixed[:, 3:] = vectors[:, 3:] @ _orthonormal(rng, 3)
    turned = spectrum(mixed)

    optical = slice(3, 6)
    # The sum over the multiplet is invariant ...
    assert turned.raman_activity[optical].sum() == pytest.approx(
        plain.raman_activity[optical].sum(), rel=1e-12
    )
    assert turned.infrared[optical].sum() == pytest.approx(
        plain.infrared[optical].sum(), rel=1e-12
    )
    # ... and the individual members are not, which is the half that makes the
    # first half a rule rather than a curiosity.
    assert not np.allclose(
        turned.raman_activity[optical], plain.raman_activity[optical], rtol=1e-6
    )
    assert not np.allclose(
        turned.depolarisation[optical], plain.depolarisation[optical], rtol=1e-6
    )
    assert [entry[1] for entry in turned.by_manifold()] == pytest.approx(
        [entry[1] for entry in plain.by_manifold()], rel=1e-12
    )


# -- the invariants themselves ------------------------------------------------


def test_a_purely_off_diagonal_tensor_is_fully_depolarised():
    """``alpha = 0`` gives ``3 beta^2/(4 beta^2) = 3/4``, the maximum.

    Zincblende's single Raman-active component is off-diagonal, so this is what
    every mode of AlAs and of silicon comes out at -- and a depolarisation ratio
    that is not 0.75 on such a crystal is an index error, not physics.
    """
    rng = np.random.default_rng(5)
    spectrum = mode_activities(
        np.array([0.0, 0.0, 0.0, 353.0, 353.0, 353.0]),
        _orthonormal(rng, 6), MASSES, EPSILON, VOLUME,
        raman=_zincblende_raman(0.4),
    )
    assert np.allclose(spectrum.alpha, 0.0, atol=1e-14)
    assert np.allclose(spectrum.depolarisation, 0.75, atol=1e-12)


def test_an_isotropic_tensor_has_no_anisotropy_and_no_depolarisation():
    """``R = c I`` gives ``beta^2 = 0``, so the ratio is ``0`` and not ``0/0``."""
    raman = np.zeros((2, 3, 3, 3))
    raman[0, 0] = np.eye(3)
    spectrum = mode_activities(
        np.zeros(6), np.eye(6), MASSES, EPSILON, VOLUME, raman=raman
    )
    assert np.allclose(spectrum.beta2, 0.0, atol=1e-20)
    assert np.allclose(spectrum.depolarisation, 0.0, atol=1e-20)


def test_a_vanishing_tensor_gives_zero_rather_than_a_nan():
    """The ``0/0`` the depolarisation ratio is at an inactive mode."""
    spectrum = mode_activities(
        np.zeros(6), np.eye(6), MASSES, EPSILON, VOLUME,
        raman=np.zeros((2, 3, 3, 3)), born=np.zeros((2, 3, 3)),
    )
    assert np.all(np.isfinite(spectrum.depolarisation))
    assert np.abs(spectrum.raman_activity).max() == 0.0
    assert np.abs(spectrum.infrared).max() == 0.0


def test_the_polarizability_table_is_the_susceptibility_in_cubic_angstrom():
    """``Omega chi/(4 pi)``, which is the block ``dynmat.x`` prints first."""
    from pypresso.units import BOHR_TO_ANGSTROM, FPI

    spectrum = mode_activities(
        np.zeros(6), np.eye(6), MASSES, EPSILON, VOLUME
    )
    expected = 11.9674 * BOHR_TO_ANGSTROM**3 * VOLUME / FPI
    assert np.diag(spectrum.polarizability) == pytest.approx(expected, rel=1e-12)
    assert spectrum.clausius_mossotti == pytest.approx(
        3.0 / (2.0 + 12.9674), rel=1e-12
    )
    assert spectrum.raman_activity is None and spectrum.infrared is None


def test_the_translational_sum_rule_kills_the_acoustic_modes():
    """A rigid translation changes neither ``chi`` nor the dipole.

    The acoustic modes at ``Gamma`` *are* rigid translations, so with tensors
    that obey the sum rule exactly -- as :func:`_zincblende_raman`'s do by
    construction -- their activity is identically zero. On a real calculation it
    is the sum rule's own residue instead, which is what
    :attr:`~pypresso.response.nonlinear.RamanTensors.translational_residue`
    reports.
    """
    # The three acoustic modes of a diatomic cell: both atoms move together,
    # mass-weighted, so ``u_(a,c) ~ sqrt(M_a)``.
    amplitude = np.sqrt(np.repeat(MASSES, 3) * AMU_TO_RY)
    vectors = np.zeros((6, 6))
    for cart in range(3):
        pattern = np.zeros(6)
        pattern[cart::3] = amplitude[cart::3]
        vectors[:, cart] = pattern / np.linalg.norm(pattern)
    spectrum = mode_activities(
        np.zeros(6), vectors, MASSES, EPSILON, VOLUME,
        raman=_zincblende_raman(0.4),
        born=np.stack([np.eye(3) * 2.0, np.eye(3) * -2.0]),
    )
    assert np.abs(spectrum.raman_activity[:3]).max() < 1e-24
    assert np.abs(spectrum.infrared[:3]).max() < 1e-24
