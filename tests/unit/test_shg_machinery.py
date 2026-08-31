"""The pieces of ``chi^(2)(-2w; w, w)``, checked without an SCF.

Two of these are the tests that actually found something, and they are worth
reading in that light.

**The literal loop.** ``nonlinopt.f90``'s triple sum accumulates into five
coefficient matrices at three *different* index pairs -- ``cc1`` is written at
``(m, l)`` and at ``(l, n)`` while ``cc2`` is written at ``(m, n)`` -- and the
vectorised form turns each of those into a reduction over a different axis
followed by a transposition or not. Getting one of them wrong is invisible in
every physical check: the tensor still comes out exactly the crystal's class,
still vanishes on a centrosymmetric cell and still has its resonances in the
right places. So the test is a literal, indexed, triple-nested transcription of
the Fortran, run against the vectorised one on random Hermitian matrices. It
found two transposed occupation factors, ``f(n, m)`` written where ``f(m, n)``
was meant, which between them flipped the sign of two of the three parts.

**The multiplet average.** ``Delta^a`` is built out of the diagonal of the
velocity operator, and the diagonal of an operator is not invariant under the
rotation a degenerate eigensolver is free to apply inside a multiplet -- rule
D4. The test rotates one and requires the answer not to move, which the bare
diagonal fails and the block average passes.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import unitary_group

import jax.numpy as jnp

from pypresso.response.photocurrent import dipole_matrix
from pypresso.response.shg import (
    CHI2_AU_TO_PM_PER_V,
    OCCUPATION_TOL,
    band_velocity_difference,
    shg_coefficients,
)

pytestmark = pytest.mark.unit

SWIDTH = 0.05


def _ingredients(seed: int = 7, nb: int = 7, degenerate: bool = True):
    """Random energies, a Hermitian velocity, and the ``r``/``Delta`` from them."""
    rng = np.random.default_rng(seed)
    energies = np.sort(rng.normal(size=nb))
    if degenerate:
        energies[4] = energies[3] + 1.0e-12
    velocity = (rng.normal(size=(3, nb, nb))
                + 1j * rng.normal(size=(3, nb, nb)))
    velocity = 0.5 * (velocity + np.conj(np.swapaxes(velocity, -1, -2)))
    filling = np.array([1.0] * (nb // 2) + [0.0] * (nb - nb // 2))
    r = np.asarray(dipole_matrix(energies[None], velocity[:, None], SWIDTH))[:, 0]
    delta = np.asarray(band_velocity_difference(
        jnp.asarray(energies), jnp.asarray(velocity), SWIDTH))
    return energies, velocity, filling, r, delta


def _elk_triple_loop(energies, r, delta, filling, a, b, c, swidth):
    """``nonlinopt.f90``'s sum over states, indexed exactly as the Fortran indexes it.

    Deliberately a slow, literal, four-space-indented transcription: it is the
    reference, so it must be readable against the Fortran line by line and must
    share no machinery at all with what it checks.
    """
    nb = len(energies)
    e = energies[:, None] - energies[None, :]
    f = filling[:, None] - filling[None, :]
    eps = OCCUPATION_TOL
    cc1 = np.zeros((nb, nb), complex)
    cc2 = np.zeros((nb, nb), complex)
    ce1 = np.zeros((nb, nb), complex)
    ce2 = np.zeros((nb, nb), complex)
    cs1 = np.zeros((nb, nb), complex)
    for n in range(nb):
        for m in range(nb):
            for l in range(nb):
                z1 = 0.5 * r[a][n, m] * (r[b][m, l] * r[c][l, n]
                                         + r[c][m, l] * r[b][l, n])
                t1 = e[l, n] - e[m, l]
                if abs(t1) > swidth:                       # Eq. (B4)
                    z2 = z1 / t1
                    if abs(f[n, m]) > eps:
                        cc2[m, n] += 2.0 * f[n, m] * z2
                    if abs(f[m, l]) > eps:
                        cc1[m, l] += f[m, l] * z2
                    if abs(f[l, n]) > eps:
                        cc1[l, n] += f[l, n] * z2
                z2 = z1 * e[m, n]
                if abs(f[n, l]) > eps:                     # Eq. (B13b)
                    t1 = e[l, n]
                    if abs(t1) > swidth:
                        ce1[l, n] += f[n, l] * z2 / t1**2
                if abs(f[l, m]) > eps:
                    t1 = e[m, l]
                    if abs(t1) > swidth:
                        ce1[m, l] -= f[l, m] * z2 / t1**2
                if abs(f[n, m]) > eps:
                    t1 = e[m, n]
                    if abs(t1) > swidth:
                        t1 = 1.0 / t1**2
                        ce2[m, n] += 2.0 * f[n, m] * (e[m, l] - e[l, n]) * t1 * z1
                        z1b = (                            # Eq. (B17)
                            e[n, l] * r[a][l, m]
                            * (r[b][m, n] * r[c][n, l] + r[c][m, n] * r[b][n, l])
                            - e[l, m] * r[a][n, l]
                            * (r[b][l, m] * r[c][m, n] + r[c][l, m] * r[b][m, n])
                        )
                        cs1[m, n] += 0.25 * f[n, m] * t1 * z1b
            if abs(f[n, m]) > eps:                         # the two double sums
                t1 = e[m, n]
                if abs(t1) > swidth:
                    t1 = 1.0 / t1**2
                    z1b = r[a][n, m] * (delta[b][m, n] * r[c][m, n]
                                        + delta[c][m, n] * r[b][m, n])
                    ce2[m, n] += 4.0 * f[n, m] * t1 * (-1j * z1b)   # (B12a)
                    z1b = r[a][n, m] * (r[b][m, n] * delta[c][m, n]
                                        + r[c][m, n] * delta[b][m, n])
                    cs1[m, n] += 0.25 * f[n, m] * t1 * (1j * z1b)   # (B16b)
    return cc1, cc2, ce1, ce2, cs1


@pytest.mark.parametrize("triple", [(0, 1, 2), (0, 0, 0), (2, 0, 1), (1, 1, 2)])
def test_the_coefficients_reproduce_a_literal_transcription_of_the_triple_loop(triple):
    """The vectorised assembly against ``nonlinopt.f90``'s own loop.

    Five matrices, four cartesian triples, a degenerate pair in the spectrum,
    and no tolerance worth the name: these are the same floating-point
    operations in a different order.
    """
    energies, _, filling, r, delta = _ingredients()
    a, b, c = triple
    reference = _elk_triple_loop(energies, r, delta, filling, a, b, c, SWIDTH)
    got = shg_coefficients(jnp.asarray(energies), jnp.asarray(r),
                           jnp.asarray(delta), jnp.asarray(filling),
                           a, b, c, SWIDTH)
    for name, want, have in zip("cc1 cc2 ce1 ce2 cs1".split(), reference, got):
        scale = max(np.max(np.abs(want)), 1.0e-30)
        assert np.max(np.abs(want - np.asarray(have))) / scale < 1.0e-13, name


def test_the_coefficients_are_symmetric_in_the_two_field_labels():
    """``chi^abc = chi^acb`` before any symmetrisation, by construction."""
    energies, _, filling, r, delta = _ingredients()
    args = (jnp.asarray(energies), jnp.asarray(r), jnp.asarray(delta),
            jnp.asarray(filling))
    for a, b, c in [(0, 1, 2), (2, 0, 1), (1, 0, 2)]:
        one = shg_coefficients(*args, a, b, c, SWIDTH)
        other = shg_coefficients(*args, a, c, b, SWIDTH)
        for x, y in zip(one, other):
            assert np.allclose(np.asarray(x), np.asarray(y), atol=1e-14)


def test_the_band_velocity_difference_is_invariant_inside_a_multiplet():
    """Rule D4 on ``Delta^a``, and the reason it is not just the diagonal.

    A degenerate eigensolver is free to hand back any unitary combination of a
    multiplet's members. The bare diagonal of the velocity operator moves under
    that rotation; the multiplet's block average does not, and it is the block
    average that is the physical band velocity -- the value a symmetry-adapted
    basis would have given.

    This is P51's finding for the Drude weight, one order up. It is worth four
    orders of magnitude on silicon's ``chi^(2)``, where the two ``Delta`` terms
    fail to cancel at the mesh's high-symmetry points, and **no symmetry check
    sees it**: the tensor stays exactly the crystal's class either way.
    """
    rng = np.random.default_rng(11)
    nb = 6
    energies = np.array([-0.9, -0.4, -0.4, -0.4, 0.7, 1.3])  # a threefold multiplet
    velocity = (rng.normal(size=(3, nb, nb)) + 1j * rng.normal(size=(3, nb, nb)))
    velocity = 0.5 * (velocity + np.conj(np.swapaxes(velocity, -1, -2)))

    rotation = np.eye(nb, dtype=complex)
    rotation[1:4, 1:4] = unitary_group.rvs(3, random_state=5)
    rotated = np.einsum("pn,ipq,qm->inm", np.conj(rotation), velocity, rotation)

    before = np.asarray(band_velocity_difference(
        jnp.asarray(energies), jnp.asarray(velocity), 0.05))
    after = np.asarray(band_velocity_difference(
        jnp.asarray(energies), jnp.asarray(rotated), 0.05))
    assert np.max(np.abs(before - after)) < 1.0e-12

    # And the check that the test is testing something: the bare diagonal,
    # which is what a literal reading of ``nonlinopt.f90`` would use, moves.
    def bare(v):
        d = np.real(np.einsum("inn->in", v))
        return d[:, :, None] - d[:, None, :]

    assert np.max(np.abs(bare(velocity) - bare(rotated))) > 1.0e-3


def test_a_band_that_stands_alone_keeps_its_own_velocity():
    """The multiplet average is not an approximation where there is no multiplet."""
    rng = np.random.default_rng(2)
    nb = 5
    energies = np.linspace(-1.0, 1.0, nb)          # no two within the tolerance
    velocity = (rng.normal(size=(3, nb, nb)) + 1j * rng.normal(size=(3, nb, nb)))
    velocity = 0.5 * (velocity + np.conj(np.swapaxes(velocity, -1, -2)))
    diagonal = np.real(np.einsum("inn->in", velocity))
    want = diagonal[:, :, None] - diagonal[:, None, :]
    got = np.asarray(band_velocity_difference(
        jnp.asarray(energies), jnp.asarray(velocity), 0.05))
    assert np.max(np.abs(want - got)) < 1.0e-14


def test_the_atomic_unit_of_chi2_is_what_its_derivation_says():
    """``e^3 / (E_h^2 eps_0)`` in pm/V, from CODATA rather than from memory.

    The one constant every other check in this corner is blind to: a
    ``chi^(2)`` wrong by a factor of two is still exactly the crystal's class,
    still zero on silicon and still resonant in the right places, which is
    P50's trap in this module's coordinates.
    """
    e = 1.602176634e-19
    hartree = 4.3597447222e-18
    epsilon0 = 8.8541878128e-12
    expected = (e**3 / hartree**2) / epsilon0 * 1.0e12
    assert abs(expected - CHI2_AU_TO_PM_PER_V) < 1.0e-3
