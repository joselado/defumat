"""The pieces of the shift current, each checked without a crystal where it can be.

``defumat.response.photocurrent`` is three things stacked: the second
derivative of ``H(k)``, the generalised derivative built from it by the
Aversa-Sipe sum rule, and a spectrum. This file checks the first two, plus the
refusals; the crystal-scale statements -- the complete-basis identity against a
parallel-transport finite difference, and the ``-43m`` form of AlAs -- are in
``tests/regression/test_photocurrent.py``, because they need an SCF.

**The one piece of algebra that is in neither reference gets its own test.**
Both IATS18 and Wannier90's ``berry_get_sc_klist`` write the intermediate sum
of the sum rule as an explicit sum over ``p`` with ``p != n, m`` excluded.
Here it is two matrix products and a correction, because the exclusions
collapse into a commutator -- which is what makes the assembly ``O(nbnd^3)``
instead of a Python loop over triples. That collapse is a claim about indices
and nothing else, so it is checked on random Hermitian matrices where the
explicit triple loop is affordable and no physics can hide a mistake.
"""

from __future__ import annotations

import numpy as np
import pytest

from defumat.response.photocurrent import (
    DEGENERACY_TOL,
    dipole_matrix,
    generalized_derivative,
    require_a_shift_current_regime,
    shift_integrand,
)

pytestmark = pytest.mark.unit


def _random_ingredients(nb: int, seed: int = 0, degenerate: bool = False):
    """``(energies, v, w)`` of the right shapes and symmetries, one k-point."""
    rng = np.random.default_rng(seed)

    def hermitian(*lead):
        a = rng.normal(size=(*lead, nb, nb)) + 1j * rng.normal(size=(*lead, nb, nb))
        return a + np.conj(np.swapaxes(a, -1, -2))

    energies = np.sort(rng.normal(size=(1, nb)))
    if degenerate:
        energies[0, 2] = energies[0, 1]
    v = hermitian(3, 1)
    w = hermitian(3, 3, 1)
    w = 0.5 * (w + np.swapaxes(w, 0, 1))  # w^ab = w^ba, as the operator is
    return energies, v, w


def _explicit_intermediate(energies, v, a: int, c: int, tol=DEGENERACY_TOL):
    """The sum over ``p != n, m`` of IATS18's Eq. (32), written as a loop.

    The literal transcription of what the vendored ``berry.F90`` does, so that
    the vectorised commutator form is compared with the thing it replaces.
    """
    e = np.asarray(energies)[0]
    nb = len(e)
    out = np.zeros((nb, nb), dtype=complex)

    def ratio(x, gap):
        return 0.0 if abs(gap) <= tol else x / gap

    for n in range(nb):
        for m in range(nb):
            total = 0.0 + 0.0j
            for p in range(nb):
                if p == n or p == m:
                    continue
                total += ratio(v[c, 0, n, p] * v[a, 0, p, m], e[p] - e[m])
                total -= ratio(v[a, 0, n, p] * v[c, 0, p, m], e[n] - e[p])
            out[n, m] = total
    return out


def _commutator_form(energies, v, a: int, c: int, tol=DEGENERACY_TOL):
    """What :func:`generalized_derivative` computes for the same sum."""
    e = np.asarray(energies)[0]
    gap = e[:, None] - e[None, :]
    finite = np.abs(gap) > tol
    g = np.where(finite, v[a, 0] / np.where(finite, gap, 1.0), 0.0)
    diagonal = np.real(np.diagonal(v[c, 0]))
    delta = diagonal[:, None] - diagonal[None, :]
    return (v[c, 0] @ g - g @ v[c, 0]) - g * delta


# --- the sum rule's one piece of original algebra ----------------------------


@pytest.mark.parametrize("degenerate", [False, True])
def test_the_exclusions_collapse_into_a_commutator(degenerate):
    """``sum_{p != n,m}`` written as ``[v^c, g^a] - g^a D^c``, against the loop.

    The exclusion of ``p = m`` from the first sum and ``p = n`` from the second
    is free -- the ``1/w_pm`` and ``1/w_np`` they carry are already zero there
    -- and the two that remain are exactly the ``D^c`` term. Nothing but index
    bookkeeping, which is why it is checked here and not on a crystal.

    The degenerate case is not decoration: a degenerate pair is dropped from
    ``g`` by :data:`DEGENERACY_TOL`, and the loop has to drop the same terms
    for the two forms to agree at all.
    """
    energies, v, _ = _random_ingredients(9, seed=3, degenerate=degenerate)
    for a, c in ((0, 1), (2, 2), (1, 0)):
        loop = _explicit_intermediate(energies, v, a, c)
        fast = _commutator_form(energies, v, a, c)
        assert np.max(np.abs(loop)) > 1.0  # not a vacuous comparison
        assert np.max(np.abs(loop - fast)) < 1.0e-12


def test_the_generalized_derivative_matches_a_transcribed_sum_rule():
    """The whole of Eq. (32), assembled a second way from the explicit sum."""
    energies, v, w = _random_ingredients(8, seed=7)
    e = np.asarray(energies)[0]
    gap = e[:, None] - e[None, :]
    finite = np.abs(gap) > DEGENERACY_TOL

    def safe(x):
        return np.where(finite, x / np.where(finite, gap, 1.0), 0.0)

    gen = np.asarray(generalized_derivative(energies, v, w))
    for a, c in ((0, 1), (1, 1), (2, 0)):
        da = np.real(np.diagonal(v[a, 0]))
        dc = np.real(np.diagonal(v[c, 0]))
        delta_a = da[:, None] - da[None, :]
        delta_c = dc[:, None] - dc[None, :]
        bracket = (
            safe(v[c, 0] * delta_a + v[a, 0] * delta_c)
            - w[a, c, 0]
            + _explicit_intermediate(energies, v, a, c)
        )
        assert np.max(np.abs(gen[a, c, 0] - 1j * safe(bracket))) < 1.0e-12


# --- the dipole ---------------------------------------------------------------


def test_the_dipole_is_interband_only_and_hermitian():
    """``r^a_nn = 0``, every degenerate pair is dropped, and ``r`` is Hermitian.

    The diagonal is the *intraband* Berry connection, which is gauge dependent
    and must appear nowhere; a degenerate pair is a genuine singularity of
    ``1/w_nm`` and rule D4 says its individual elements are not defined. Both
    are structural, and both are what stops a gauge-dependent number reaching
    the answer.
    """
    energies, v, _ = _random_ingredients(7, seed=11, degenerate=True)
    r = np.asarray(dipole_matrix(energies, v))
    for a in range(3):
        assert np.max(np.abs(np.diagonal(r[a, 0]))) == 0.0
        assert abs(r[a, 0, 1, 2]) == 0.0 and abs(r[a, 0, 2, 1]) == 0.0
        assert np.max(np.abs(r[a, 0] - np.conj(r[a, 0].T))) < 1.0e-12


def test_the_integrand_is_real_and_symmetric_in_the_two_field_labels():
    """``Im[r^b_mn g^{c;a}_nm + r^c_mn g^{b;a}_nm]`` is real and ``b <-> c`` even.

    A shift current is a *rectified* response: it has no phase, and the two
    field labels are the same field. Both fall out of the assembly rather than
    being imposed, so a transposed index would break them.
    """
    energies, v, w = _random_ingredients(6, seed=5)
    integrand = np.asarray(shift_integrand(energies, v, w))
    assert integrand.dtype.kind == "f"
    assert np.max(np.abs(integrand - np.swapaxes(integrand, 1, 2))) < 1.0e-14
    assert np.max(np.abs(integrand)) > 1.0e-3


def test_the_generalized_derivative_is_hermitian_in_the_band_pair():
    """``(r^{c;a}_nm)* = r^{c;a}_mn`` -- a covariant derivative of a Hermitian object.

    This one is worth having because it is the *structural* check that survives
    the sum rule: every term of Eq. (32) is antisymmetric or symmetric under
    ``n <-> m`` in a way that has to conspire, and a wrong sign in the
    commutator breaks it while leaving the magnitudes plausible.
    """
    energies, v, w = _random_ingredients(8, seed=13)
    gen = np.asarray(generalized_derivative(energies, v, w))
    for a in range(3):
        for c in range(3):
            block = gen[a, c, 0]
            assert np.max(np.abs(block - np.conj(block.T))) < 1.0e-11


# --- the refusals, on the guard itself ---------------------------------------


class _Stub:
    """The four attributes :func:`require_a_shift_current_regime` reads.

    A stand-in and not a :class:`~defumat.scf.driver.Calculation`, because the
    branch under test is unreachable from any committed input: every DFT+U case
    in this suite uses an ultrasoft dataset, so the ultrasoft refusal fires
    first and the Hubbard message is never seen. The guard is four independent
    conditions on four flags, and this checks the one a crystal cannot reach.
    """

    class _System:
        occupations = "fixed"

        class _KPoints:
            weights = np.ones(4) / 4

        kpoints = _KPoints()

    def __init__(self, **flags):
        self.is_ultrasoft = flags.get("ultrasoft", False)
        self.is_paw = flags.get("paw", False)
        self.is_hubbard = flags.get("hubbard", False)
        self.spiral = flags.get("spiral", False)
        self.system = self._System()


@pytest.mark.parametrize(
    "flags, message",
    [
        ({"ultrasoft": True}, "ultrasoft or PAW"),
        ({"paw": True}, "ultrasoft or PAW"),
        ({"spiral": True}, "spin spiral"),
        ({"hubbard": True}, r"DFT\+U"),
    ],
)
def test_each_refusal_names_its_own_missing_term(flags, message):
    """Four flags, four refusals, each naming what it lacks rather than failing late."""
    with pytest.raises(NotImplementedError, match=message):
        require_a_shift_current_regime(_Stub(**flags))


def test_a_regime_the_assembly_covers_is_not_refused():
    """The control: with every flag clear, the guard passes.

    Without it the parametrisation above is satisfied by a function that raises
    unconditionally.
    """
    require_a_shift_current_regime(_Stub())
