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

from pypresso.response.phonon import _diagonalize
from pypresso.response.spectra import (
    degenerate_manifolds,
    eigendisplacements,
    loto_modes,
    mode_activities,
    nonanal,
)
from pypresso.units import (
    AMU_SI, AMU_TO_RY, BOHR_RADIUS_SI, E2, ELECTRON_SI, EPSILON0_SI, FPI, PI,
    RY_TO_THZ,
)

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


# --------------------------------------------------------------------------
# P55: the long-range electric field -- LO-TO splitting and the static
# dielectric constant. Both are contractions of the same two ingredients
# (``Z*`` and ``eps_infinity``), which is what lets them check each other: the
# Lyddane-Sachs-Teller relation ties the frequencies to the permittivities, and
# neither routine can satisfy it by accident because each supplies one side.
# --------------------------------------------------------------------------

#: A diatomic cubic crystal, made up rather than computed: two atoms pulled
#: together by one spring constant, opposite Born charges, an isotropic
#: ``eps_infinity``. Everything below is exact for it, so the tolerances are
#: double precision rather than a physics tolerance.
SPRING = 0.18
CHARGE = 2.1
EPS_INFINITY = 9.5


def _diatomic():
    """``(force_constants, born, epsilon)`` for the model above."""
    phi = np.zeros((2, 3, 2, 3))
    for cart in range(3):
        phi[0, cart, 0, cart] = phi[1, cart, 1, cart] = SPRING
        phi[0, cart, 1, cart] = phi[1, cart, 0, cart] = -SPRING
    born = np.stack([CHARGE * np.eye(3), -CHARGE * np.eye(3)])
    return phi, born, EPS_INFINITY * np.eye(3)


def test_the_non_analytic_term_raises_one_mode_and_leaves_the_others():
    """The field is longitudinal: one LO mode, two TO modes, three acoustic.

    The rank-one structure of ``nonanal`` is the physical content -- an optical
    mode builds a macroscopic field only if it has a dipole along ``q`` -- and
    it is visible in the spectrum without any reference to compare against.
    """
    phi, born, epsilon = _diatomic()
    transverse, _ = _diagonalize(phi, MASSES)
    longitudinal, _ = loto_modes(phi, MASSES, born, epsilon, (1, 0, 0), VOLUME)

    # The three acoustic modes stay at zero: ``sum_a Z*_a = 0``, so the term
    # annihilates a rigid translation exactly as the force constants do. Both
    # sets are zero to the eigensolver's own residue, which for a frequency is
    # the square root of it -- hence a bound rather than a comparison.
    assert np.max(np.abs(longitudinal[:3])) < 1e-4
    assert np.max(np.abs(transverse[:3])) < 1e-4
    # Two of the three optical modes are untouched and one is raised.
    assert np.allclose(longitudinal[3:5], transverse[3:5], atol=1e-9)
    assert longitudinal[5] > transverse[5] + 30.0


def test_the_non_analytic_term_is_blind_to_the_length_of_q():
    """It is homogeneous of degree zero: only the *direction* enters."""
    phi, born, epsilon = _diatomic()
    one, _ = loto_modes(phi, MASSES, born, epsilon, (1, 0, 0), VOLUME)
    long_ = loto_modes(phi, MASSES, born, epsilon, (57.0, 0, 0), VOLUME)[0]
    assert np.allclose(one, long_, atol=1e-12)


def test_a_non_polar_crystal_has_no_splitting():
    """``Z* = 0`` and the term vanishes identically -- silicon, in a fixture."""
    phi, _, epsilon = _diatomic()
    born = np.zeros((2, 3, 3))
    split, _ = loto_modes(phi, MASSES, born, epsilon, (1, 1, 1), VOLUME)
    assert np.allclose(split, _diagonalize(phi, MASSES)[0], atol=1e-12)


def test_a_direction_with_no_length_is_refused():
    """``nonanal`` writes a message and returns the matrix unchanged.

    Silently giving TO modes back to a caller who asked for LO ones is the kind
    of failure this project refuses instead.
    """
    phi, born, epsilon = _diatomic()
    with pytest.raises(ValueError, match="direction"):
        nonanal(phi, born, epsilon, (0.0, 0.0, 0.0), VOLUME)


def test_lyddane_sachs_teller():
    """``eps_0/eps_inf = (omega_LO/omega_TO)^2``, the check neither half passes alone.

    The left-hand side comes from :func:`polar_mode_permittivity` and the right
    from :func:`nonanal`, and the two share no line of code: one divides the
    mode dipole by ``omega^2`` and the other adds a rank-one term to a matrix
    before it is diagonalised. For a diatomic cubic crystal the relation is an
    identity, so this holds to double precision and any error in either
    prefactor -- a factor of two in ``e^2``, a missing ``4 pi``, the volume in
    the wrong place -- breaks it.
    """
    phi, born, epsilon = _diatomic()
    transverse, vectors = _diagonalize(phi, MASSES)
    longitudinal, _ = loto_modes(phi, MASSES, born, epsilon, (1, 0, 0), VOLUME)
    spectrum = mode_activities(
        transverse, vectors, MASSES, epsilon, VOLUME, born=born
    )

    ratio = spectrum.static_permittivity[0, 0] / EPS_INFINITY
    splitting = (longitudinal[5] / transverse[5]) ** 2
    assert ratio == pytest.approx(splitting, rel=1e-12)
    # ... and the static tensor is isotropic, with nothing imposing it.
    off_diagonal = spectrum.static_permittivity - np.diag(
        np.diag(spectrum.static_permittivity)
    )
    assert np.max(np.abs(off_diagonal)) < 1e-14


def test_the_ionic_permittivity_constant_is_qes():
    """The one constant written down here rather than transcribed.

    ``polar_mode_permittivity`` in ``LR_Modules/dynmat_sub.f90`` builds its
    prefactor from ``e/sqrt(eps_0 a_0^3 amu)`` and converts ``omega`` to THz;
    in Rydberg units the same number is ``4 pi e^2`` with the mode dipole in
    ``e/sqrt(Ry mass)``. A wrong constant here gives a plausible dielectric
    constant rather than a broken one, so the two chains are compared directly.
    """
    plasma = ELECTRON_SI / np.sqrt(EPSILON0_SI * BOHR_RADIUS_SI**3 * AMU_SI)
    qe = plasma**2 / (FPI * PI) * 1.0e-24 / RY_TO_THZ**2
    assert FPI * E2 / AMU_TO_RY == pytest.approx(qe, rel=1e-10)


def test_a_soft_mode_is_left_out_of_the_static_permittivity():
    """An imaginary frequency has no oscillator strength to report.

    ``polar_mode_permittivity``'s ``w2 > eps8``, verbatim: a crystal at a saddle
    point of its energy has no static dielectric constant, and dividing by a
    negative ``omega^2`` would return one with the wrong sign rather than
    saying so.
    """
    phi, born, epsilon = _diatomic()
    unstable = -phi  # every mode imaginary
    frequencies, vectors = _diagonalize(unstable, MASSES)
    spectrum = mode_activities(
        frequencies, vectors, MASSES, epsilon, VOLUME, born=born
    )
    assert np.allclose(spectrum.ionic_permittivity, 0.0, atol=0.0)
    assert np.allclose(spectrum.static_permittivity, epsilon)
