"""Second-harmonic generation: ``chi^(2)(-2 omega; omega, omega)``.

Shine light of frequency ``omega`` on a crystal with no inversion centre and
some of it comes back out at ``2 omega``. The tensor that says how much is the
second-order optical susceptibility

    P^a(2w) = chi^abc(-2w; w, w) E_b(w) E_c(w),

a polar rank-3 tensor, symmetric in its last two labels, and identically zero
in every centrosymmetric crystal. It is written here following Sipe and
Ghahramani (PRB **48**, 11705 (1993)) and Hughes and Sipe (PRB **53**, 10751
(1996)), in the corrected form Elk's ``nonlinopt.f90`` carries after Gonze's
2022 corrections -- which is the assembly this module transcribes, and the one
place in this corner of the package where a real reference implementation
exists.

The tensor is a sum of three physically distinct pieces, and they are kept
**separate** in the result rather than added silently:

* ``chi_II``  -- the pure interband term, Hughes-Sipe Eq. (B4);
* ``eta_II``  -- the intraband modulation of the interband transition,
  Eqs. (B12a) and (B13);
* ``sigma_II``-- the modulation of the transition energy by the intraband
  motion, Eqs. (B16b) and (B17), carried as ``i/(2w) sigma_II``.

Elk writes the three to three separate files, and keeping them apart is what
makes a disagreement localisable instead of a single number to stare at -- the
lesson P43 records for the five partial derivatives of ``d(eps)/d(tau)``.

**Why this is reachable while P35's ``chi^(2)`` refusal stands.** That refusal
is a statement about the **Sternheimer stack**: the field enters only through a
source term, so a 2n+1 expression with two field labels has nothing to build
``<u_i|r_k|u_j>`` from, and it is 42% of the answer. A **sum over states** has
no such problem. It needs the interband dipole ``r^a_nm = -i v^a_nm / w_nm``,
which is *exact* for an eigenstate of the full ``H(k)`` -- no truncation, and
the reason a plane-wave code has no counterpart to Wannier90's ``AA_R``. So
this follows P53's route and not P35's, and `NONLINEAR.md` §7 says so.

**The momentum matrix element is the wrong operator and this code never forms
one.** ``[H, r] = p`` only for a *local* Hamiltonian; with a nonlocal
pseudopotential ``v = dH/dk`` and ``p`` differ by the projector term. Elk's
``getpmat`` is right in an all-electron LAPW basis and is the one line
``CLAUDE.md`` records as not to be transcribed, so
:class:`~defumat.response.velocity.VelocityOperator` takes its place. That is
the only substitution made in an otherwise literal transcription.

**What is *not* needed here, and it is P53's most expensive object.** The
triple sum over ``l`` below *is* the sum-rule expansion of the generalised
derivative ``r^{c;a}``, written out rather than collapsed, so
``second_matrix_elements`` -- the second derivative of ``H(k)`` -- never
appears. What does carry over from P53 unchanged is the *cost* of that sum: it
is an identity only over a complete basis, so the band count is the convergence
parameter of the whole quantity and :attr:`SecondHarmonic.truncation` reports
it.

**The degeneracy threshold is the broadening, not 1e-8**, which is Elk's own
rule (``if (abs(t1) > swidth)``) and P53's finding arriving again: a linear
conductivity carries ``1/w_nm`` once and this carries it three times.

Scope
-----

Norm-conserving, ``nspin = 1`` or a spinor run, an **insulator** with fixed
occupations, on a k-grid closed under the point group. Refused by name, each
for its own missing term, in :func:`require_an_shg_regime`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from defumat.batching import resolve_k_batch, sum_k
from defumat.response.photocurrent import (
    DEGENERACY_TOL,
    _kpoints_are_reduced,
    _safe_ratio,
    dipole_matrix,
)
from defumat.response.velocity import VelocityOperator

__all__ = [
    "SecondHarmonic",
    "CHI2_AU_TO_PM_PER_V",
    "band_velocity_difference",
    "shg_coefficients",
    "second_harmonic",
    "require_an_shg_regime",
]

#: An occupation difference smaller than this counts as zero -- Elk's
#: ``epsocc``. With fixed occupations ``f_nm`` is exactly 0 or exactly +-1, so
#: this only skips terms that would contribute nothing; it is kept because the
#: transcription is literal and because a smeared run would need it to mean
#: something.
OCCUPATION_TOL = 1.0e-8

#: One atomic unit of ``chi^(2)`` in **pm/V**, in the SI convention
#: ``P = eps_0 chi^(2) E E``.
#:
#: **Derived rather than looked up**, because this is the constant every other
#: check in the module is blind to -- P50's trap in this module's coordinates,
#: and P53's: a ``chi^(2)`` wrong by a factor of two is still exactly ``-43m``,
#: still zero on silicon, still vanishing below the two-photon edge, and still
#: has its resonances in the right places.
#:
#: Second-order perturbation theory in ``H' = e E.r`` gives a polarization
#: ``P = (e^3 r^3 / (V dE^2)) E^2``, so the natural unit of the sum below is
#: ``e^3 / E_h^2`` (the ``r^3`` and the ``1/V`` cancel), which is
#: ``(1.602176634e-19)^3 / (4.3597447222e-18)^2 = 2.1637e-22 C/V^2``. Dividing
#: by ``eps_0 = 8.8541878128e-12`` puts it in m/V:
#: **24.4377 pm/V per atomic unit**.
#:
#: **The Rydberg factor is separate and is a factor of four.** Every term of
#: the assembly carries exactly *two* energy denominators -- one inside the
#: coefficient and one in the frequency contraction -- so evaluating it with
#: energies in Ry rather than Hartree makes it four times too small, and
#: :func:`second_harmonic` multiplies it back. Counting those denominators is
#: the check that they are the same two in all three of ``chi_II``, ``eta_II``
#: and ``sigma_II``, which they are.
#:
#: **The convention is declared because the literature has two of them**, the
#: same ambiguity P53 records for the shift current. What is computed is
#: ``chi^(2)``, defined by ``P^a(2w) = chi^abc E_b E_c``; the nonlinear-optics
#: literature more often quotes ``d^abc = chi^abc / 2``, so a number compared
#: against a ``d`` coefficient without halving it is wrong by exactly two.
CHI2_AU_TO_PM_PER_V = 24.4377


def band_velocity_difference(energies, velocity, tol: float):
    """``Delta^a_mn = v^a_mm - v^a_nn``, ``(3, nb, nb)``, in Ry bohr.

    Elk's ``d(m, n, i)``, with **one thing added that Elk does not do and that
    a plane-wave code cannot skip**: the diagonal is averaged over each
    degenerate multiplet before the difference is taken.

    ``Delta`` is built entirely out of the *diagonal* of an operator, and the
    diagonal of an operator is not invariant under the unitary rotation a
    degenerate eigensolver is free to apply inside a multiplet -- rule D4, and
    P51's finding for the Drude weight one order up. What is invariant is the
    multiplet's block trace, so each member takes the block's average, which is
    what a symmetry-adapted basis would have given it and which reduces to
    ``v^a_mm`` exactly wherever a band stands alone.

    **It is worth four orders of magnitude and no symmetry check sees it.**
    Silicon is centrosymmetric, so every part of ``chi^(2)`` must vanish; with
    the bare diagonal the two ``Delta`` terms -- Eqs. (B12a) and (B16b), the
    only two places ``Delta`` appears -- come out at **1499** and **238** on a
    4x4x4 mesh where the other three sit at 0.09, and with the multiplet
    average they fall to **0.10** and **0.055**, which is the same floor. The
    high-symmetry points of the mesh are what does it: at ``Gamma`` silicon's
    valence top is threefold degenerate, its block trace of ``v`` is zero by
    symmetry, and an arbitrary basis inside it gives three nonzero diagonal
    entries that cancel only in that sum.

    Elk does not need this at 42x42x42 with a shifted mesh that misses the
    symmetry points, which is why ``nonlinopt.f90`` has no counterpart to it
    and why this is not a transcription bug.

    Args:
        energies: ``(nb,)`` in Ry -- what decides which bands are one multiplet.
        velocity: ``(3, nb, nb)``, ``<n|dH/dk_a|m>`` in Ry bohr.
        tol: bands closer than this in Ry are one multiplet. It is the
            broadening, for :data:`DEGENERACY_TOL`'s reason.
    """
    diagonal = jnp.real(jnp.diagonal(jnp.asarray(velocity), axis1=-2, axis2=-1))
    multiplet = (jnp.abs(energies[:, None] - energies[None, :]) < tol)
    weight = multiplet.astype(diagonal.dtype)
    averaged = (diagonal @ weight.T) / jnp.sum(weight, axis=1)[None, :]
    return averaged[:, :, None] - averaged[:, None, :]


class _Shared(NamedTuple):
    """The part of the sum that does not depend on the cartesian triple.

    Two kinds of thing. The energy denominators and the occupation differences
    are functions of ``energies`` and ``filling`` alone, so all eighteen
    triples share them; ``resonant`` is the only one of them that is ``nb^3``,
    and it is the only ``nb^3`` denominator left. And ``x``/``y`` are the
    eighteen matrix products the intermediate sum collapses into -- see
    :func:`_coefficients` for why that collapse exists.
    """

    energies: jnp.ndarray
    e: jnp.ndarray          # e[m, n] = E_m - E_n
    f: jnp.ndarray          # f[m, n]
    esum: jnp.ndarray       # E_m + E_n, as [m, n]
    inv2: jnp.ndarray       # 1 / e[m, n]^2, masked on |e[m, n]|
    resonant: jnp.ndarray   # 1 / (e[l, n] - e[m, l]), as [n, m, l]
    x: jnp.ndarray          # x[i, j] = r^i r^j          (3, 3, nb, nb)
    y: jnp.ndarray          # y[i, j] = r^i diag(E) r^j  (3, 3, nb, nb)


def _shared(energies, r, filling, swidth, epsocc) -> _Shared:
    """The triple-independent half of :func:`shg_coefficients`."""
    e = energies[:, None] - energies[None, :]          # e[m, n] = E_m - E_n
    f = filling[:, None] - filling[None, :]            # f[m, n]
    f = jnp.where(jnp.abs(f) > epsocc, f, 0.0)

    def inverse(x):
        return _safe_ratio(jnp.ones_like(x), x, swidth)

    scaled = r * energies[None, None, :]               # r^i[m, l] E_l
    return _Shared(
        energies=energies,
        e=e,
        f=f,
        esum=energies[:, None] + energies[None, :],
        inv2=inverse(e) ** 2,
        resonant=inverse(e.T[:, None, :] - e[None, :, :]),
        x=jnp.einsum("iml,jln->ijmn", r, r),
        y=jnp.einsum("iml,jln->ijmn", scaled, r),
    )


def _coefficients(s: _Shared, r, delta, a: int, b: int, c: int, q):
    """The five coefficient matrices for one triple, from the shared work.

    **Only chi_II keeps a sum over the intermediate state.** Its denominator
    ``2E_l - E_n - E_m`` couples all three band indices and does not factorise,
    so ``cc1`` and ``cc2`` are genuinely ``O(nb^3)`` in memory as well as in
    flops, which is what ``q`` carries.

    The other three do factorise, and that is this module's one departure from
    ``nonlinopt.f90``'s loop structure. Every ``l``-dependent factor in
    ``eta_II`` and ``sigma_II`` is either linear in ``E_l`` or independent of
    it, so each sum over ``l`` is a pair of matrix products:
    ``sum_l r^i[.,l] r^j[l,.]`` and ``sum_l r^i[.,l] E_l r^j[l,.]``, which are
    ``s.x`` and ``s.y``. Eighteen of those cover all eighteen triples, so what
    was three ``nb^3`` arrays per triple becomes eighteen ``nb^3`` matrix
    multiplications per k-point -- BLAS work rather than strided elementwise
    work, and no ``nb^3`` temporary at all.

    **The rule that says mirror the reference's loop structure is why this is
    written down rather than just done.** The re-association is exact, the
    literal transcription of the Fortran stays in the tests as the reference
    both forms are checked against, and it is measured: one core, 200
    frequencies, against the form that materialises every sum, **23.3 ms per
    k-point to 8.9 at 22 bands and 156.3 to 31.6 at 40**, with the answer
    unchanged to 3e-15. The gap widens with the band count because what it
    removes is the ``nb^3`` term.
    """
    ra, rb, rc = r[a], r[b], r[c]
    raT = ra.T
    x, y, energies = s.x, s.y, s.energies

    # -- chi_II, Hughes-Sipe Eq. (B4): the one sum that does not factorise.
    # Its summand is 0.5 r^a_nm q[n, m, l] / (2E_l - E_n - E_m), and that
    # denominator couples all three band indices, so ``q`` -- which is
    # 0.5 (r^b_ml r^c_ln + r^c_ml r^b_ln), symmetric in b <-> c, which is where
    # the tensor's own symmetry in its two field labels comes from -- stays an
    # ``nb^3`` object. It is passed in already divided and already summed over
    # ``l``, both of which depend on the field pair alone.
    #
    # The summand itself is never formed. ``cc2`` needs only the sum over
    # ``l``, which carries no ``a`` at all; the two ``cc1`` accumulations are
    # contractions over ``n`` and over ``m``, which is what the two ``einsum``
    # calls are. Materialising it instead is one ``nb^3`` temporary per triple
    # rather than one per field pair.
    divided, over_l = q
    # ``cc2`` carries neither the 2 of Eq. (B4) nor the 0.5 of ``z1``: they
    # cancel, and a reader checking this against the Fortran will look for
    # both.
    cc2 = (s.f * ra * over_l).T                                 # -> [m, n]
    cc1 = 0.5 * s.f * jnp.einsum("nm,nml->ml", ra, divided)     # -> [m, l]
    cc1 = cc1 + (0.5 * s.f.T
                 * jnp.einsum("nm,nml->nl", ra, divided)).T     # -> [l, n]

    # -- eta_II, Eq. (B13b), both terms.  ``E_n`` runs along the first axis of
    # a ``[n, l]`` matrix and along the second of an ``[m, n]`` one, which is
    # the whole of the index care needed here.
    down, across = energies[:, None], energies[None, :]
    first = 0.5 * (rc.T * (y[a, b] - down * x[a, b])
                   + rb.T * (y[a, c] - down * x[a, c]))         # [n, l]
    second = 0.5 * (rb * (down * x[c, a].T - y[c, a].T)
                    + rc * (down * x[b, a].T - y[b, a].T))      # [m, l]
    ce1 = (s.f * first * s.inv2.T).T - s.f.T * second * s.inv2

    # -- eta_II, Eq. (B13a), and i/2w sigma_II, Eq. (B17): both gated by the
    # same |e(m, n)| > swidth, which is what ``inv2`` carries.
    ce2 = 2.0 * s.f.T * s.inv2 * (
        0.5 * raT * (s.esum * (x[b, c] + x[c, b]) - 2.0 * (y[b, c] + y[c, b]))
    )
    head = (rb * (across * x[c, a].T - y[c, a].T)
            + rc * (across * x[b, a].T - y[b, a].T))
    tail = (rc * (y[a, b].T - down * x[a, b].T)
            + rb * (y[a, c].T - down * x[a, c].T))
    cs1 = 0.25 * s.f.T * s.inv2 * (head - tail)

    # -- the two double sums, Eqs. (B12a) and (B16b).  The same product of
    # matrix elements appears in both, ``r^a_nm (D^b_mn r^c_mn + D^c_mn r^b_mn)``;
    # they differ only by the phase and the weight, which is why Elk computes
    # it twice and this computes it once.
    # ``f.T`` and not ``f``: these accumulate into ``[m, n]`` while the weight
    # is still ``f(n, m)``, the order trap the module docstring records.
    pair = raT * (delta[b] * rc + delta[c] * rb)                # [m, n]
    ce2 = ce2 + 4.0 * s.f.T * s.inv2 * (-1j * pair)
    cs1 = cs1 + 0.25 * s.f.T * s.inv2 * (1j * pair)

    return cc1, cc2, ce1, ce2, cs1


def _bracket(s: _Shared, r, b: int, c: int):
    """chi_II's bracket over the intermediate state, for one field pair.

    ``(r^b[m, l] r^c[l, n] + r^c[m, l] r^b[l, n]) / (2E_l - E_n - E_m)`` as
    ``[n, m, l]``, together with its sum over ``l``. This is the one ``nb^3``
    object the assembly still forms, and it depends on the two *field* labels
    only, so the three current directions of a field pair share it: six
    constructions per k-point rather than eighteen, and the division and the
    ``l``-sum come with them.
    """
    rb, rc = r[b], r[c]
    divided = (rb[None, :, :] * rc.T[:, None, :]
               + rc[None, :, :] * rb.T[:, None, :]) * s.resonant
    return divided, jnp.sum(divided, axis=2)


def shg_coefficients(energies, r, delta, filling, a: int, b: int, c: int,
                     swidth: float, epsocc: float = OCCUPATION_TOL):
    """The five frequency-independent coefficient matrices, each ``(nb, nb)``.

    Elk's ``cc1``, ``cc2``, ``ce1``, ``ce2`` and ``cs1``, for one cartesian
    triple. Everything expensive is here: the sum over the intermediate state
    ``l`` is done **once**, before the frequency axis is touched. That is
    ``nonlinopt.f90``'s own layout and the reason a 200-point spectrum costs no
    more than a single frequency; writing the sum inside the frequency loop
    instead would be ``nw`` times the cost for the same answer.

    **This is the single-triple entry point and the one the tests drive.**
    :func:`_chi_at_k` does not call it, because the work splits into three
    pieces shared over different things -- :func:`_shared` across all eighteen
    triples, :func:`_bracket` across the three current directions of one field
    pair, and :func:`_coefficients` over neither -- and hoisting them is what
    makes the sum affordable at a production band count. The split is
    arithmetically empty: this function is those three calls in order.

    Args:
        energies: ``(nb,)`` in Ry at one k-point, **after** any scissors shift.
        r: ``(3, nb, nb)`` position matrix elements in bohr, ``r[i][m, n]``,
           built from the **unshifted** gaps -- see :func:`second_harmonic`.
        delta: ``(3, nb, nb)``, ``Delta^i_mn = v^i_mm - v^i_nn`` in Ry bohr.
        filling: ``(nb,)`` occupations in ``[0, 1]``.
        a, b, c: the cartesian labels; the result is symmetric in ``b <-> c``.
        swidth: the broadening in Ry, and also the threshold below which an
            energy denominator is unresolvable and the term is dropped.

    Returns:
        ``(cc1, cc2, ce1, ce2, cs1)``, each ``(nb, nb)`` indexed ``[m, n]`` as
        Elk indexes them.
    """
    shared = _shared(energies, r, filling, swidth, epsocc)
    return _coefficients(shared, r, delta, a, b, c,
                         _bracket(shared, r, b, c))


# -- the spectrum --------------------------------------------------------------


#: How much larger the answer is in Hartree than in Rydberg. **Every** term of
#: the assembly carries exactly two energy denominators -- one inside the
#: coefficient matrix and one in the frequency contraction -- so a sum done in
#: Ry is four times too small. Counting them is also the check that the same
#: two appear in all three of ``chi_II``, ``eta_II`` and ``sigma_II``, which
#: they do: ``chi_II`` divides by ``2E_l - E_n - E_m`` and then by the
#: resonance, ``eta_II`` and ``sigma_II`` by ``e_mn^2`` and then multiply an
#: energy back in.
RYDBERG_TO_HARTREE_SQUARED = 4.0


@dataclass
class SecondHarmonic:
    """What :func:`second_harmonic` returns.

    Attributes:
        frequencies: ``(nw,)`` in Ry -- the *fundamental* photon energy
            ``hbar omega``, not the second harmonic. The response at entry
            ``i`` is the light emitted at ``2 omega_i``.
        chi: ``(nw, 3, 3, 3)`` complex, in **pm/V**, indexed ``[w, a, b, c]``:
            the polarization along ``a`` from fields along ``b`` and ``c``, in
            which it is symmetric by construction. The sum of the three parts
            below.
        chi_ii, eta_ii, sigma_ii: the same shape and units -- the interband
            term, the intraband modulation of it, and the modulation of the
            transition energy, kept apart because Elk writes them to three
            separate files and because a disagreement in one of them is
            localisable where a disagreement in their sum is not.
        chi_au: :attr:`chi` in atomic units, which is what
            ``nonlinopt.f90`` writes and therefore what a comparison against
            Elk uses with no conversion at all.
        volume: the cell volume in bohr^3.
        broadening: the Lorentzian width in Ry. It is also the threshold below
            which an energy denominator is treated as unresolvable, which is
            Elk's rule rather than ``dielectric.f90``'s 1e-8 -- P53's finding,
            and it matters more here than there.
        scissor: the rigid shift applied to the empty states, in Ry.
        nbnd: how many bands the sums ran over. **It is the convergence
            parameter**, and more so than for a linear spectrum: the sum over
            the intermediate state ``l`` is the sum-rule expansion of the
            generalised derivative and is an identity only over a complete
            basis.
        truncation: the largest change in :attr:`chi` when the top **quarter**
            of the bands is dropped, relative to ``max|chi|``. A quarter and
            not one band, for P53's reason -- this sum does not converge band
            by band, so its last term is not its tail, and dropping one band
            under-reports by orders of magnitude. **Read it before believing a
            number.**
        band_cut_gap: ``min_k (e_(nbnd+1) - e_nbnd)`` in Ry when the caller
            diagonalised one extra band to measure it, else ``nan``.
    """

    frequencies: np.ndarray
    chi: np.ndarray
    chi_ii: np.ndarray
    eta_ii: np.ndarray
    sigma_ii: np.ndarray
    chi_au: np.ndarray
    volume: float
    broadening: float
    scissor: float
    nbnd: int
    truncation: float = float("nan")
    band_cut_gap: float = float("nan")

    @property
    def frequencies_ev(self) -> np.ndarray:
        """The fundamental photon energy in eV."""
        from defumat.units import RY_TO_EV

        return np.asarray(self.frequencies) * RY_TO_EV

    def component(self, a: int, b: int, c: int) -> np.ndarray:
        """``chi^abc(-2w; w, w)`` against frequency, ``(nw,)`` complex, pm/V."""
        return np.asarray(self.chi)[:, a, b, c]

    def d_coefficient(self, a: int, b: int, c: int) -> np.ndarray:
        """``d^abc = chi^abc / 2``, which is what the literature usually quotes.

        The convention trap P53 records for the shift current, in this
        module's coordinates: a number compared against a published ``d``
        without halving it is wrong by exactly two, and nothing about its
        shape, its symmetry or its zeros would say so.
        """
        return self.component(a, b, c) / 2.0

    def static(self, a: int, b: int, c: int) -> complex:
        """``chi^abc`` at the lowest frequency on the axis, in pm/V."""
        return complex(np.asarray(self.chi)[0, a, b, c])

    def __repr__(self) -> str:  # pragma: no cover - display only
        peak = float(np.max(np.abs(self.chi))) if np.size(self.chi) else 0.0
        return (
            f"SecondHarmonic(nw={len(self.frequencies)}, nbnd={self.nbnd}, "
            f"peak={peak:.4g} pm/V, truncation={self.truncation:.2e})"
        )


#: The 18 independent cartesian triples. ``chi^abc`` is symmetric in ``b`` and
#: ``c`` by construction -- ``z1``'s ``r^b r^c + r^c r^b`` -- so the other nine
#: are copies rather than a symmetry imposed afterwards.
_TRIPLES = [(a, b, c) for a in range(3)
            for b in range(3) for c in range(b, 3)]


def _chi_at_k(energies, energies_bare, velocity, filling, weight,
              frequencies, eta, swidth, epsocc):
    """One k-point's contribution, ``(3, nw, 3, 3, 3)`` -- the three parts.

    Laid out in the two stages ``nonlinopt.f90`` uses, and for its reason: the
    coefficient matrices are ``O(nb^3)`` and frequency-independent, so they are
    built once and only then contracted against the two resonances, which are
    ``O(nb^2 nw)``. The loop is over the six independent **field pairs** on the
    outside and the three current directions inside, because the ``nb^3``
    products depend on the field pair alone.
    """
    r = dipole_matrix(energies_bare[None], velocity[:, None], swidth)[:, 0]
    delta = band_velocity_difference(energies_bare, velocity, swidth)
    d = _shared(energies, r, filling, swidth, epsocc)

    # Elk's two resonances. The broadening **doubles** in the second-harmonic
    # channel: ``e - 2(w - i eta)``, not ``e - 2w + i eta``.
    zv1 = 1.0 / (d.e[None] - frequencies[:, None, None] + 1j * eta)
    zv2 = 1.0 / (d.e[None] - 2.0 * (frequencies[:, None, None] - 1j * eta))

    table = {}
    for b in range(3):
        for c in range(b, 3):
            bracket = _bracket(d, r, b, c)
            for a in range(3):
                table[(a, b, c)] = _coefficients(d, r, delta, a, b, c, bracket)

    # One contraction per resonance rather than one per coefficient per
    # triple: ninety small ``einsum`` calls become two. Worth 17 to 24 per
    # cent once the coefficients themselves are cheap -- and worth nothing
    # before that, which is why it is measured here rather than assumed.
    first = jnp.stack([  # the three parts that resonate at w
        jnp.stack([table[t][i] for t in _TRIPLES]) for i in (0, 2, 4)
    ])
    second = jnp.stack([  # the two that resonate at 2w
        jnp.stack([table[t][i] for t in _TRIPLES]) for i in (1, 3)
    ])
    at_omega = jnp.einsum("ptmn,wmn->ptw", first, zv1)
    at_two_omega = jnp.einsum("qtmn,wmn->qtw", second, zv2)
    parts = jnp.concatenate([
        at_omega[:2] + at_two_omega, at_omega[2:]
    ])  # (3, 18, nw), ordered chi_II, eta_II, sigma_II

    return weight * jnp.stack([_expand(columns) for columns in parts])


def _expand(columns):
    """The 18 independent triples out to the full ``(nw, 3, 3, 3)``.

    ``chi^abc = chi^acb`` holds by construction -- :func:`_bracket` builds
    both orders into one bracket -- so the other nine components are copies
    rather than a symmetry imposed after the fact.
    """
    table = {}
    for (a, b, c), value in zip(_TRIPLES, columns):
        table[(a, b, c)] = value
        table[(a, c, b)] = value
    stacked = jnp.stack([
        jnp.stack([jnp.stack([table[(a, b, c)] for c in range(3)])
                   for b in range(3)])
        for a in range(3)
    ])  # (3, 3, 3, nw)
    return jnp.moveaxis(stacked, -1, 0)


def second_harmonic(
    calculation,
    wavefunctions,
    eigenvalues,
    v_scf,
    *,
    frequencies=None,
    window: float = 0.6,
    nw: int = 200,
    broadening: float = 0.003,
    scissor: float = 0.0,
    band_cut_gap: float = float("nan"),
    k_batch: int | None | str = "default",
    degeneracy_tol: float | None = None,
) -> SecondHarmonic:
    """``chi^(2)(-2 omega; omega, omega)`` in pm/V.

    Args:
        calculation: the fixed-density :class:`~defumat.scf.driver.
            Calculation` the states came from.
        wavefunctions: ``(nspin, nk, nbnd, ndim)`` -- **every** diagonalised
            band, not the occupied manifold: the sum over the intermediate
            state needs them.
        eigenvalues: ``(nspin, nk, nbnd)`` in Ry.
        v_scf: the self-consistent potential the Hamiltonian was built with.
        frequencies: an explicit *fundamental* photon-energy axis in Ry;
            ``None`` builds a uniform one of ``nw`` points up to ``window``.
        broadening: the Lorentzian width in Ry -- Elk's ``swidth``, which is
            the inverse lifetime and also the degeneracy threshold.
        scissor: a rigid shift (Ry) of the empty states. **The dipoles are
            built from the unshifted gaps and the denominators from the shifted
            ones**, which is exactly what ``getpmat.f90`` achieves by scaling
            the momentum matrix elements by ``e/(e -+ scissor)``: the position
            matrix element is the physical object and a scissors correction
            does not move it. Doing it here instead of on ``pmat`` is the same
            arithmetic with one fewer place to lose a sign.
        degeneracy_tol: how close in Ry two bands may be before every
            ``1/w_nm`` between them is dropped. ``None`` -- the default -- is
            ``broadening``, which is Elk's rule.

    The frequency axis carries the **fundamental** ``hbar omega`` in Ry, so a
    semiconductor's two-photon absorption edge sits at half its gap.
    """
    require_an_shg_regime(calculation)

    eigenvalues = jnp.asarray(eigenvalues)
    wavefunctions = jnp.asarray(wavefunctions)
    if eigenvalues.ndim == 2:
        eigenvalues, wavefunctions = eigenvalues[None], wavefunctions[None]
    if eigenvalues.shape[0] != 1:
        raise NotImplementedError(
            "the second-harmonic tensor of a collinear spin-polarized run is "
            "not implemented: the two channels are two band structures whose "
            "susceptibilities add, which is a loop this assembly does not have"
        )

    precision = calculation.system.cell.precision
    volume = float(calculation.system.cell.volume)
    nbnd = int(eigenvalues.shape[-1])

    velocity = VelocityOperator(calculation, v_scf)
    v = velocity.matrix_elements(wavefunctions)[:, 0]  # (3, nk, nb, nb)

    wg, _ = calculation.occupations(eigenvalues)
    wg = jnp.asarray(wg)[0]
    wk = jnp.asarray(calculation.system.kpoints.weights)
    filling = wg / wk[:, None]  # in [0, 1] per spin channel

    bare = jnp.asarray(eigenvalues)[0]
    if scissor:
        # Which bands are empty is read from the occupations rather than from
        # a Fermi level, because this branch runs only for fixed occupations.
        empty = filling < 0.5
        shifted = bare + jnp.where(empty, precision.as_real(scissor), 0.0)
    else:
        shifted = bare

    if frequencies is None:
        frequencies = np.linspace(0.0, float(window), int(nw))
    frequencies = jnp.asarray(np.asarray(frequencies, dtype=float))

    eta = precision.as_real(broadening)
    if degeneracy_tol is None:
        degeneracy_tol = max(float(broadening), DEGENERACY_TOL)
    batch = resolve_k_batch(k_batch)

    scale = (RYDBERG_TO_HARTREE_SQUARED / volume)

    def total(bands: int):
        def one_k(arrays):
            e_k, e0_k, v_k, f_k, wk_k = arrays
            return _chi_at_k(
                e_k[:bands], e0_k[:bands], v_k[:, :bands, :bands],
                f_k[:bands], wk_k, frequencies, eta, degeneracy_tol,
                OCCUPATION_TOL,
            )

        arrays = (shifted, bare, jnp.moveaxis(v, 0, 1), filling, wk)
        return np.asarray(sum_k(one_k, arrays, batch=batch)) * scale

    parts = total(nbnd)
    chi_au = parts.sum(axis=0)
    chi = chi_au * CHI2_AU_TO_PM_PER_V

    reference = float(np.max(np.abs(chi)))
    truncation = float("nan")
    dropped = max(1, nbnd // 4)
    if nbnd - dropped > 1 and reference > 0.0:
        coarse = total(nbnd - dropped).sum(axis=0) * CHI2_AU_TO_PM_PER_V
        truncation = float(np.max(np.abs(coarse - chi)) / reference)

    return SecondHarmonic(
        frequencies=np.asarray(frequencies),
        chi=chi,
        chi_ii=parts[0] * CHI2_AU_TO_PM_PER_V,
        eta_ii=parts[1] * CHI2_AU_TO_PM_PER_V,
        sigma_ii=parts[2] * CHI2_AU_TO_PM_PER_V,
        chi_au=chi_au,
        volume=volume,
        broadening=float(broadening),
        scissor=float(scissor),
        nbnd=nbnd,
        truncation=truncation,
        band_cut_gap=float(band_cut_gap),
    )


# -- the refusals --------------------------------------------------------------


def require_an_shg_regime(calculation) -> None:
    """The same five refusals :func:`~defumat.response.photocurrent.
    require_a_shift_current_regime` makes, and for the same reasons.

    This module is the same velocity matrix elements contracted a different
    way, so it inherits every one of them: an ultrasoft or PAW ``dS/dk``, a
    spiral's two spheres, a wedge that a polar rank-3 tensor is not summed
    over, a Hubbard term with a velocity of its own, and a metal whose
    partially filled bands put a vanishing denominator in every term.
    """
    from defumat.response.photocurrent import require_a_shift_current_regime

    try:
        require_a_shift_current_regime(calculation)
    except NotImplementedError as error:
        message = str(error).replace(
            "the shift current", "second-harmonic generation"
        ).replace("sigma^abc", "chi^abc")
        raise NotImplementedError(message) from None
