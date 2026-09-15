"""The augmentation charge inside a transverse matrix element.

``<psi_{nk,up}| e^{-i(q+G).r} |psi_{m k+q,dn}>`` is the plane-wave overlap of
the pseudo states only when all of the charge is in ``|psi|^2``. With an
ultrasoft dataset it is not, and what is missing is

    sum_a sum_ij Q^a_ij(q+G) <psi_{nk}|beta^k_i> <beta^{k+q}_j|psi_{m k+q}>,

the augmentation charge at the *same* wavevector the exponent carries. Nothing
is derived for it: ``q^a_ij(b)`` at an arbitrary wavevector is the primitive the
ultrasoft Berry phase already uses (:mod:`defumat.topology.augmentation`,
validated by ``b -> 0`` reproducing the projectors' own ``qq`` and by a Chern
number coming out an exact integer), called once per ``G`` of the response
sphere.

**The check is the Goldstone identity and it must fail without the term.**
``X_0(q = 0, omega = 0) B_xc = m`` says a spin wave of infinite wavelength is a
rigid rotation and costs nothing, and it is a statement about the *whole*
moment. Most of a transition metal's moment lives in the d shell, which is
exactly where an ultrasoft dataset puts its augmentation, so dropping the term
does not degrade the identity -- it destroys it.

**What the bare number is, said exactly, because it is not a fair fight and
pretending otherwise would overclaim.** :func:`goldstone_residual` reads ``m``
off the dense-grid density, which for an ultrasoft run already carries
``addusdens``'s augmentation charge. So the bare run compares an unaugmented
``X_0`` against an augmented ``m``, and 0.984 is the augmentation's **share of
the moment** rather than an independent measure of the error. That is still the
guard firing -- it says the term is most of the quantity -- and the number that
measures the assembly is the augmented one.

**And the augmented residual is truncation rather than a missing term**, which
was checked rather than assumed: over ``ecut_response`` of 6, 8, 12, 16, 24 and
48 Ry it reads 0.1063, 0.0706, 0.0992, 0.0987, 0.0854 and **0.0128**. It is *not* monotone
-- the 8 Ry point is the workflow's default and sits below the next two, and the
sequence reproduces bit for bit -- so the tolerance below is a bound on the
default setting and not a claim about a trend. The other axis of the same double
truncation is flat here: ``nbnd`` of 14, 20 and 30 give 0.070562, 0.070589 and
0.070582.
"""

from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest

import defumat.tddft.spinchi0 as spinchi0
from defumat import Calculator

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: ``omega = 0`` first, with no broadening: the identity is exact only there.
FREQUENCIES = np.linspace(0.0, 0.05, 6)


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _calculator(stem):
    calc = Calculator.from_file(CASES / stem, pseudo_dir=PSEUDO)
    calc.get_scf()
    return calc


def _goldstone(calc, augmented, **options):
    original = spinchi0.augmentation_factors
    if not augmented:
        spinchi0.augmentation_factors = lambda *args, **kwargs: None
    try:
        return calc.get_spin_susceptibility(
            (0.0, 0.0, 0.0), FREQUENCIES, goldstone=True, **options
        ).goldstone
    finally:
        spinchi0.augmentation_factors = original


def test_the_augmentation_is_what_makes_the_identity_hold():
    calc = _calculator("ni-fcc-magnon-us.in")
    assert calc.calculation.is_ultrasoft

    with_term = _goldstone(calc, augmented=True)
    without = _goldstone(calc, augmented=False)

    # Measured: 0.984 bare against 0.071 augmented, at the default sphere and
    # nbnd = 14. The bare number is not a degraded identity, it is no identity:
    # almost the whole moment of a d shell is in the augmentation.
    assert without > 0.5
    assert with_term < 0.15
    assert with_term < 0.2 * without


def test_a_norm_conserving_run_is_untouched():
    """The term must be exactly absent where there is no augmentation charge,
    rather than small: a norm-conserving dataset has none and the code path
    that carries it must make no difference at all."""
    calc = _calculator("h-fcc-magnon.in")
    assert not calc.calculation.is_ultrasoft
    assert spinchi0.augmentation_factors(
        calc.calculation, (0.0, 0.0, 0.0),
        spinchi0.spin_response_sphere(calc.calculation, 4.0),
    ) is None


def test_the_wavevector_convention_matches_the_berry_phase():
    """Which sign of ``b`` in ``Q_ij(b)``, pinned against validated code.

    **Neither obvious check can see this.** The Goldstone identity is at
    ``q = 0``, where the response sphere is symmetric under ``G -> -G`` so a
    flipped sign only permutes the ``G`` index; and reciprocal-lattice
    periodicity is a relabelling that holds for either sign -- measured, on this
    cell, at 6.8e-16 with ``+b`` and 5.5e-16 with ``-b``. A test built on either
    would report a pass it could not have failed.

    What pins it is a *structural* correspondence with code that is already
    validated. :meth:`defumat.topology.states.PlaneWaveStates.overlap` builds
    the same augmented object for a pair of k-points, takes its wavevector as
    ``k_ket - k_bra`` (its ``_difference``), and is pinned by a Chern number
    coming out an exact integer on an ultrasoft dataset. The transverse matrix
    element has its bra at ``k`` and its ket at ``k + q``, so the same rule
    gives ``b = +(q + G)``, and the ``G = 0`` entry of the factors must be the
    *identical array* the Berry phase would use for that pair.
    """
    from defumat.topology.augmentation import augmentation_at_q

    calc = _calculator("ni-fcc-magnon-us.in")
    calculation = calc.calculation
    cell = calc.system.cell
    sphere = spinchi0.spin_response_sphere(calculation, 4.0)
    q = (0.0, 0.0, 0.25)

    factors = spinchi0.augmentation_factors(calculation, q, sphere)
    assert factors is not None
    # ``G = 0`` is the sphere's first entry by construction.
    assert np.abs(np.asarray(sphere.miller)[0]).max() == 0

    # The Berry phase's own wavevector for the pair (k, k + q): ket minus bra,
    # in 1/bohr, which is what ``KPoints.cartesian`` and ``cell.bg`` share.
    difference = np.asarray(q, dtype=float) @ np.asarray(cell.bg)
    reference = np.asarray(augmentation_at_q(calculation, difference))
    assert np.abs(np.asarray(factors[0]) - reference).max() < 1.0e-14

    # And the guard fires: the other sign is a genuinely different array, so
    # this comparison is capable of failing.
    flipped = np.asarray(augmentation_at_q(calculation, -difference))
    assert np.abs(flipped - reference).max() > 1.0e-6


def test_paw_is_still_refused():
    """Lifting the ultrasoft refusal must not lift PAW's, which is about the
    kernel rather than the matrix element."""
    calc = Calculator.from_file(CASES / "o2-paw-texture.in", pseudo_dir=PSEUDO)
    with pytest.raises(NotImplementedError, match="PAW"):
        spinchi0.require_a_transverse_regime(calc.calculation)
