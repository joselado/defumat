"""Vertical transmission from the nonlocal Green's function.

The quantity is the conductance an electron sees on its way *through* a
two-dimensional material: in at a point ``r`` (the tip), out anywhere in a plane
below (the substrate). In the tunnelling regime the Landauer-Buettiker
transmission is

    T = Tr[Gamma_t G^r Gamma_s G^a]

and a **point** tip makes ``Gamma_t`` rank one, which collapses the trace to an
integral of the nonlocal Green's function over the exit region:

    T(r; E) = gamma_t^2 gamma_s^2 int_plane d^2 r' |G^r(r, r'; E)|^2,

    G^r(r, r'; E) = sum_k w_k sum_n psi_kn(r) psi*_kn(r') / (E - e_kn + i eta).

The two couplings are unfixed prefactors -- what is computed here is the map and
its contrast, not an absolute conductance.

**The amplitude is the on-shell one, and that is a finding rather than a
choice.** Writing ``G`` with its literal energy denominator
``1/(E - e_kn + i eta)`` is the exact Landauer expression, and a sum over states
**cannot evaluate it**: the denominator has a long tail, so the states far from
``E`` carry the barrier's evanescent decay, and that decay is built entirely out
of cancellation between them. Measured on a cell small enough to diagonalise
completely -- a hydrogen sheet at ``ecutwfc = 15``, 367 plane waves and
therefore 367 exact bands -- the running sum for one ``G(r_tip, r_exit)``
wanders over more than an order of magnitude (2.4e-4, 7.2e-3, 1.2e-3, 5.1e-4,
1.0e-3, 4.5e-4 at 8, 16, 32, 128, 200 and 367 bands) and only lands when the
basis is *complete*, at a **cancellation ratio of 349**: the sum of the moduli
of the terms is 0.157 and their sum is 4.5e-4. There is no band count at which
a truncated version of it is right, and on a real cell the complete basis is a
dense diagonalisation.

What converges is the **modulus** of that denominator with its phase held
constant, which is the weak-coupling (tunnelling) limit of the same Landauer
expression -- Bardeen's golden rule, in which the sample's states are visited
on shell:

    a_kn(r) = psi_kn(r) sqrt( delta(E - e_kn) )

Two things make this the right object rather than a retreat to a cruder one.
It is *exactly* the resolvent's modulus: for a Lorentzian delta,
``|1/(E - e + i eta)| = sqrt(pi delta_eta(E - e)/eta)``, so the only thing
dropped is the arctan the resolvent's phase sweeps through across a resonance.
And the interference it keeps is the one that matters -- between bands
**degenerate at the tip energy**, which are exactly the paths a real experiment
lets interfere, and which no local density of states can represent. The
off-resonant part it drops is direct tunnelling through the barrier without
going on shell in the sample, which is what a weak-coupling tunnelling geometry
is defined by not having.

The delta defaults to a **Gaussian** and that is what makes the sum finite:
a Lorentzian's square root falls off only as ``1/|E - e|``, where a Gaussian's
is another Gaussian. ``method = "resolvent"`` is available and warns, with the
measurement above as its reason.

**The contraction.** With the exit integral diagonal in ``k``
(:mod:`defumat.transport.substrate`), everything reduces to one quadratic form
per k-point:

    T(r; E) = sum_k w_k  a_k(r)^dagger S_k a_k(r)

``S_k`` being a Gram matrix, each term is ``|B a|^2`` for some ``B``, so **the
transmission is non-negative by construction** -- unlike a tunnelling density
built from a smeared delta, which a Methfessel-Paxton weight can drive negative
(``PLAN.md`` P52, P65).

**The Tersoff-Hamann limit is exact.** Widen the substrate from a plane to the
whole cell and ``S_k -> delta_nn'`` by orthonormality, leaving

    T(r; E) = sum_kn w_k |psi_kn(r)|^2 delta(E - e_kn)

which is the tunnelling density of states of ``PLAN.md`` P65 -- the *same*
number :func:`defumat.workflows.stm.run_stm` returns, with no factor between
them. So the claim "on graphene this is the STM image" is a theorem rather than
an observation, and it is a validation route that shares no line of code with
the assembly it checks.

**Interference is the off-diagonal of ``S_k``, and it is separable.** Dropping
it leaves an incoherent sum in which every channel tunnels independently --
taken in the basis that diagonalises ``S_k`` inside each degenerate multiplet,
because the *diagonal* of an operator is not invariant under the rotation a
degenerate eigensolver is free in where the quadratic form above is
(:func:`channel_basis`). The difference between the two maps is the
interference, and it costs nothing extra to report because both come from the
same three arrays.

**Energy is nearly free.** ``psi_kn(r)`` and ``S_k`` do not depend on ``E``;
only the per-state amplitude weight does. A whole transport ``dI/dV`` curve at every pixel is
therefore one ``nk x nbnd^2 x npts`` contraction per energy on top of a
sampling step that is paid once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["VerticalTransport", "transmission", "amplitude_weights",
           "channel_basis"]


def amplitude_weights(eigenvalues, energy: float, broadening: float,
                      method: str = "spectral", smearing: str = "gaussian"):
    """The per-state factor multiplying ``psi_kn(r)`` before anything is squared.

    ``"spectral"`` is ``sqrt(delta(E - e)/eta)`` -- the on-shell amplitude, real
    and non-negative, and the one the module docstring establishes is the only
    convergent form. Normalised so that the whole-cell exit region reproduces
    :func:`defumat.stm.image.tunnelling_weights` exactly rather than up to a
    factor: ``|a|^2 = |psi|^2 delta(E - e)/eta`` is P65's weight.

    ``"resolvent"`` is the literal ``1/(E - e + i eta)``. It is the exact
    Landauer denominator and it is **not summable over a truncated band set**;
    it is here so that the statement can be measured rather than asserted.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if broadening <= 0.0:
        raise ValueError(f"the broadening must be positive, got {broadening}")
    name = method.strip().lower()
    if name == "resolvent":
        return 1.0 / (float(energy) - eigenvalues + 1.0j * float(broadening))
    if name != "spectral":
        raise ValueError(
            f"unknown method {method!r}: use 'spectral' (the on-shell "
            "amplitude) or 'resolvent' (the exact denominator, which a band "
            "sum cannot converge)"
        )
    from defumat.stm.image import smeared_delta

    delta = smeared_delta((float(energy) - eigenvalues) / float(broadening),
                          smearing) / float(broadening)
    # A delta that goes negative on its wings has no square root, which is the
    # same objection PLAN.md P52 and P65 make to using one for a quantity with
    # a sign -- here it is sharper, because an amplitude is what is being taken.
    if np.min(delta) < 0.0:
        raise ValueError(
            f"the {smearing!r} delta is negative on its wings and an amplitude "
            "is its square root: use a positive delta ('gaussian', the "
            "default, or 'fermi-dirac'), not a Methfessel-Paxton or cold one"
        )
    return np.sqrt(delta).astype(complex)


#: Two eigenvalues closer than this (Ry) are one multiplet. QE prints six
#: decimals in eV, so this is well inside what a degeneracy is resolvable to.
DEGENERACY_TOL = 1.0e-6


def channel_basis(overlaps, eigenvalues, tol: float = DEGENERACY_TOL):
    """A per-k unitary that diagonalises ``S_k`` inside each degenerate block.

    **The incoherent map needs this and the coherent one does not**, which is
    rule D4 arriving in a *diagnostic* rather than in an answer. ``T_coh`` is a
    quadratic form and is invariant under the rotation a degenerate eigensolver
    is free in; ``sum_n |a_n|^2 S[n,n]`` is a **diagonal**, and the diagonal of
    an operator is not. Reported as it stands, "how much interference there is"
    would depend on which basis the solver happened to return inside a
    multiplet -- P51's Drude weight and P54's band-velocity difference, one
    layer out.

    Diagonalising ``S_k`` within each multiplet fixes it and gives the quantity
    its meaning back: the incoherent map becomes *the substrate's own channels,
    each taken on its own*, and the interference becomes the part that runs
    **between** multiplets, which no choice of basis can rotate away.

    Where it makes no difference is worth knowing, because it is the common
    case and it is not luck: at a symmetry point the little group acts
    irreducibly on the multiplet, so by Schur's lemma any invariant operator
    restricted to it is a multiple of the identity, and every basis is already
    a channel basis. Graphene's Dirac pair at ``K`` measures
    ``diag(0.05405086, 0.05405084)`` with off-diagonals at 1e-8.

    Returns ``(nk, nbnd, nbnd)``; ``psi'_i = sum_n U_ni psi_n``.
    """
    overlaps = np.asarray(overlaps)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    nk, nbnd = eigenvalues.shape
    rotation = np.zeros((nk, nbnd, nbnd), dtype=complex)
    for ik in range(nk):
        rotation[ik] = np.eye(nbnd)
        start = 0
        order = np.argsort(eigenvalues[ik], kind="stable")
        while start < nbnd:
            stop = start + 1
            while (stop < nbnd
                   and eigenvalues[ik][order[stop]]
                   - eigenvalues[ik][order[start]] < tol):
                stop += 1
            block = order[start:stop]
            if block.size > 1:
                sub = overlaps[ik][np.ix_(block, block)]
                _, vectors = np.linalg.eigh(0.5 * (sub + sub.conj().T))
                rotation[ik][np.ix_(block, block)] = vectors
            start = stop
    return rotation


def transmission(amplitudes, overlaps, kweights, weights, coherent: bool = True,
                 eigenvalues=None, tol: float = DEGENERACY_TOL):
    """``T(r)`` at every tip point, for one set of per-state amplitude weights.

    Args:
        amplitudes: ``(nk, nbnd, npts)`` complex, ``psi_kn`` at the tip points.
        overlaps: ``(nk, nbnd, nbnd)`` complex, the exit-plane Gram matrices.
        kweights: ``(nk,)`` the run's own weights, carrying the spin
            degeneracy exactly as they do everywhere else here.
        weights: ``(nk, nbnd)`` the per-state factor from
            :func:`amplitude_weights`.
        coherent: keep the off-diagonal of ``S_k``. ``False`` is the incoherent
            map in which every channel tunnels on its own; the difference
            between the two is the interference.
        eigenvalues: ``(nk, nbnd)`` in Ry. Needed only for ``coherent=False``,
            to find the degenerate multiplets whose basis the diagonal would
            otherwise depend on -- see :func:`channel_basis`.
        tol: what counts as degenerate, in Ry.

    Returns ``(npts,)`` real and non-negative.
    """
    amplitudes = np.asarray(amplitudes)
    overlaps = np.asarray(overlaps)
    kweights = np.asarray(kweights, dtype=float)
    weights = np.asarray(weights)

    nk, nbnd, npts = amplitudes.shape
    if overlaps.shape != (nk, nbnd, nbnd):
        raise ValueError(
            f"the overlaps are {overlaps.shape} and the amplitudes want "
            f"{(nk, nbnd, nbnd)}"
        )
    if weights.shape != (nk, nbnd) or kweights.shape != (nk,):
        raise ValueError(
            f"state weights {weights.shape} and k weights {kweights.shape} do "
            f"not match {nk} k-points and {nbnd} bands"
        )

    a = amplitudes * weights[:, :, None]
    if coherent:
        # **The conjugation belongs on the second factor, not the first.**
        # ``G(r, r') = sum_n a_n(r) psi*_n(r')`` carries psi *conjugated* in the
        # exit variable, so the plane integral of its modulus squared is
        # ``sum_nm a_n a*_m S[n,m]`` with ``S[n,m] = int psi*_n psi_m`` -- and
        # that is ``a^T S a*``, not ``a^dagger S a``. The two differ by a
        # transpose of a Hermitian matrix, so the wrong one is real,
        # non-negative, exactly reproduces the Tersoff-Hamann limit (where
        # ``S`` is the identity and the two coincide), passes the sum rule, is
        # blind to a degenerate rotation, and is wrong wherever ``S`` has an
        # off-diagonal -- which is to say wherever the interference this is for
        # actually lives. It is P54's transposed occupation factor again, and
        # only a literal check against the definition catches it.
        sa = np.einsum("kij,kjp->kip", overlaps, a.conj(), optimize=True)
        term = np.real(np.einsum("kip,kip->kp", a, sa, optimize=True))
    else:
        if eigenvalues is not None:
            # Into the substrate's own channels first, so that "each band on
            # its own" is a statement and not a choice of basis.
            u = channel_basis(overlaps, eigenvalues, tol)
            a = np.einsum("kni,knp->kip", u, a, optimize=True)
            overlaps = np.einsum("kni,knm,kmj->kij", u.conj(), overlaps, u,
                                 optimize=True)
        diagonal = np.real(np.einsum("knn->kn", overlaps))
        term = np.einsum("kn,knp->kp", diagonal, np.real(a.conj() * a),
                         optimize=True)
    return kweights @ term


@dataclass
class VerticalTransport:
    """A vertical-transport map and everything needed to read it."""

    #: ``(nE, n1, n2)`` -- or ``(n1, n2)`` when one energy was asked for. The
    #: transmission in arbitrary units: the tip and substrate couplings are
    #: unfixed prefactors, so what this carries is the map and its contrast.
    values: np.ndarray
    #: The tip plane the map was sampled on.
    plane: object
    #: ``(nE,)`` the energies in Ry, relative to nothing -- absolute.
    energies: np.ndarray
    #: The Lorentzian broadening in Ry.
    broadening: float
    #: The exit plane's crystal coordinate and the axis it is normal to.
    exit_height: float = 0.0
    exit_axis: int = 2
    #: The same map with the off-diagonal of every ``S_k`` dropped: every band
    #: tunnelling independently. ``None`` unless it was asked for.
    incoherent: np.ndarray | None = None
    #: The Fermi level of the k-set actually used, in Ry.
    fermi_energy: float | None = None
    #: The substrate's spin acceptance, if it had one.
    spin: object = None
    polarization: float = 1.0
    #: The k-set, when a denser one was solved for.
    grid: tuple[int, int, int] | None = None
    #: Diagnostics: the smallest eigenvalue of any ``S_k`` (must not be
    #: negative -- a Gram matrix is positive semi-definite), and how much of
    #: each ``S_k`` sits off the diagonal, which is how much interference there
    #: is to have.
    least_eigenvalue: float = 0.0
    offdiagonal_weight: float = 0.0
    #: What the bands could not represent: the plane-restricted trace against
    #: the number of bands, a band-count truncation measure.
    notes: dict = field(default_factory=dict)

    @property
    def image(self) -> np.ndarray:
        """``(n1, n2)`` at the first energy -- what to plot."""
        return self.values if self.values.ndim == 2 else self.values[0]

    @property
    def interference(self) -> np.ndarray | None:
        """Coherent minus incoherent: the part no local picture can give."""
        if self.incoherent is None:
            return None
        return self.values - self.incoherent

    @property
    def coordinates(self) -> np.ndarray:
        """``(n1, n2, 2)`` in-plane cartesian coordinates in bohr."""
        return self.plane.coordinates

    def extent(self) -> tuple[float, float, float, float]:
        """``(x0, x1, y0, y1)`` in bohr, for ``imshow``."""
        x, y = self.coordinates[..., 0], self.coordinates[..., 1]
        return (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
