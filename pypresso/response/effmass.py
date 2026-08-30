"""The effective mass tensor: one central difference of an *analytic* velocity.

    (1/m*)_ab = (1/2) d^2 eps_n(k) / dk_a dk_b

in units of ``1/m_e``, the factor of a half being Rydberg atomic units: a free
electron has ``eps = |k|^2`` with ``k`` in 1/bohr, because ``hbar^2/2m_e`` is
exactly 1 Ry bohr^2. So the free-electron tensor is the **identity**, and that
is a statement about the units rather than a fitted normalisation.

**Why this is a difference at all, in a package whose thesis is autodiff.** The
first derivative is exact and analytic here -- ``d(eps_n)/dk`` is the
generalised Hellmann-Feynman expression ``<n|dH/dk - eps_n dS/dk|n>`` built from
one ``jvp`` of ``H(k)`` (:mod:`pypresso.response.velocity`, rule D2). The
*second* is not available the same way, and the reason is worth writing down
because it looks as though it should be:

- Differentiating that expression again **at frozen states** gives only the
  first of the two terms in

      d^2 eps_n/dk_a dk_b = <n| d^2H/dk_a dk_b |n>
                          + 2 Re <dn/dk_a| dH/dk_b - eps_n dS/dk_b |n> + ...

  and the missing piece is the whole ``k.p`` sum -- for silicon's lowest
  conduction band at ``Gamma`` it is not a correction, it is most of the answer.
- The first-order state ``|dn/dk>`` is what :mod:`pypresso.response.sternheimer`
  solves for, and it cannot supply this one: its projector ``P_c`` removes the
  whole occupied manifold, which is right for a density response (the
  occupied-occupied pairs cancel there) and wrong for one eigenvalue's second
  derivative, where they do not. And the band whose mass is wanted is usually
  **empty** -- an electron mass at a conduction minimum -- where ``H - eps_n S``
  is indefinite and the projected CG has nothing to converge to.
- Differentiating through the eigensolver is rule D4's prohibition and P22's
  measurement: 109% wrong from a cold start.

So the honest construction is the one implemented: **the first derivative is a
``jvp`` and the second is one central difference of it**. That is not a
concession -- it is strictly better than differencing eigenvalues, which is what
Elk's ``effmass.f90`` does. Six stencil points instead of twenty-seven, an
``O(h^2)`` error on a quantity that is exact at each point rather than on a
second difference, and no polynomial fit.

**The eigenvalue route is here too, and it is not decoration.** ``method =
"eigenvalue"`` is Elk's: a second difference of ``eps_n(k)`` alone, on a stencil
that -- unlike Elk's -- excludes the centre, for the basis-set reason
:func:`_by_eigenvalue` measures. It shares nothing with the velocity operator -- no ``jvp``, no
``dH/dk``, no ``dS/dk`` -- so agreement between the two is an independent check
on the operator itself, in regimes where nothing else checks it. A spinor
``dH/dk`` is the case in point: P47's Kubo curvature validated the operator at
``nspin = 1`` only, and a two-component platinum band through both routes is
the first thing that tests it.

**Degeneracies, and what is reported instead.** A per-band second derivative is
not a property of the band inside a degenerate multiplet: the eigensolver's
arbitrary rotation within the manifold rotates it, exactly as it rotates the
band velocity (rule D4). Such bands are **refused by name** rather than
returned, and what is offered in their place is the multiplet *sum*
``sum_n d^2 eps_n/dk_a dk_b``, which is the trace over the manifold and is
invariant under that rotation. Silicon's valence top at ``Gamma`` is threefold
and is the case: its heavy and light hole masses are not separable this way,
and their sum is.

**Cost.** Thirteen NSCF k-points for the velocity route and forty-nine for
the eigenvalue one, both at a *single* k-point of the crystal and both halved
without the Richardson step -- there is no self-consistency and no
integration over the zone, so this is the cheapest derived quantity in the
package. **Memory** is one NSCF's: ``(nspin, nk, nbnd, npwx)`` with ``nk`` the
stencil size, and the stencil is the only thing this adds.

**The truncation is removed and then reported, rather than tuned away.** Both
routes are ``O(h^2)``, and at Elk's own ``deltaem = 0.025`` that error is not
small: silicon's ``Gamma_2'`` conduction band comes out at 5.131 against a
converged 5.303, **3 per cent**, and the eigenvalue route at 5.216. Halving
``h`` divides the two routes' disagreement by exactly four, which is what makes
a **Richardson step** the right default here rather than a smaller ``h``: the
stencil is run at ``h`` and ``h/2`` and combined as ``(4 M(h/2) - M(h)) / 3``,
which is ``O(h^4)`` and costs six more NSCF k-points. What the two levels differ
by is kept as :attr:`EffectiveMass.truncation`, so the error that remains is a
number the caller can read (P47's rule, and P37's before it) instead of a
convergence study nobody runs.

**Precision.** The difference divides by ``h``, so it amplifies the
eigensolver's own scatter by ``1/h`` (and by ``1/h^2`` for the eigenvalue
route). :mod:`pypresso.response.velocity` records that a ``conv_thr`` of 1e-6
moves an individual band velocity by 0.5 Ry bohr near a multiplet, so the
default here is tight (1e-12) and ``delta`` is exposed -- Elk's ``deltaem`` is
0.025 and is the default here too, which with the Richardson step is a
combination Elk does not have.

No QE counterpart: ``grep -ri "effective mass"`` over ``PW/src`` and ``PP/src``
finds nothing. Elk's task 25 (``effmass.f90``) is the reference for the
algorithm and the local binary for the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "EffectiveMass",
    "Multiplet",
    "effective_mass",
    "MASS_METHODS",
]

#: The two constructions. ``"velocity"`` differences the analytic ``d(eps)/dk``;
#: ``"eigenvalue"`` differences ``eps`` itself, which is Elk's ``effmass.f90``.
MASS_METHODS = ("velocity", "eigenvalue")

#: Elk's ``deltaem``, in 1/bohr. Small enough that ``O(h^2)`` is small and large
#: enough that the eigensolver's noise divided by ``h`` still is.
DEFAULT_DELTA = 0.025

#: Two eigenvalues closer than this (Ry) at the centre are one multiplet. It is
#: loose on purpose: a near-degeneracy is contaminated by the same rotation an
#: exact one is, only continuously.
DEFAULT_DEGENERACY_TOL = 1.0e-5


@dataclass(frozen=True)
class Multiplet:
    """A set of bands degenerate at the centre k-point.

    A one-band multiplet is the ordinary case and carries a mass; anything wider
    carries only :attr:`inverse_mass_sum`, because the individual tensors are
    basis-dependent.
    """

    #: Band indices, contiguous and ascending.
    bands: tuple[int, ...]
    #: The shared eigenvalue, Ry (the mean over the multiplet).
    eigenvalue: float
    #: ``sum_n (1/m*)_ab`` over the multiplet, ``(3, 3)`` in 1/m_e. Invariant
    #: under the eigensolver's rotation inside the manifold; equal to the band's
    #: own tensor when the multiplet holds one band.
    inverse_mass_sum: np.ndarray

    @property
    def degenerate(self) -> bool:
        return len(self.bands) > 1


@dataclass
class EffectiveMass:
    """``(1/m*)_ab`` at one k-point, per band and per spin channel."""

    #: Crystal coordinates of the k-point the tensor is taken at.
    kpoint: np.ndarray
    #: ``(nspin, nbnd, 3, 3)`` in 1/m_e. Bands inside a degenerate multiplet are
    #: ``nan`` -- see :attr:`multiplets` for what is defined there instead.
    inverse_mass_by_spin: np.ndarray
    #: ``(nspin, nbnd)`` in Ry, the eigenvalues at the centre.
    eigenvalues_by_spin: np.ndarray
    #: One entry per spin channel, each a tuple of :class:`Multiplet`.
    multiplets: tuple
    #: ``(nspin, nbnd)``: ``||M - M^T|| / ||M||`` before symmetrisation. The
    #: tensor is a second derivative and so is symmetric exactly; what this
    #: measures is the stencil's own error, and it is free.
    asymmetry_by_spin: np.ndarray
    #: ``(nspin, nbnd)`` in 1/m_e: ``|M(h/2) - M(h)| / 3``, the ``O(h^2)``
    #: truncation the Richardson step removed, kept as an estimate of what is
    #: left. ``None`` when ``richardson = False``, where the error is the whole
    #: of that difference and is not measured.
    truncation_by_spin: np.ndarray | None
    #: The stencil's outermost displacement, 1/bohr.
    delta: float
    method: str
    nspin: int = 1
    #: Bands whose tensor is ``nan`` because they sit in a degenerate multiplet.
    refused: tuple = field(default_factory=tuple)

    # -- the squeezed views, following ``SCFResult``'s convention -----------

    @property
    def inverse_mass(self) -> np.ndarray:
        """``(nbnd, 3, 3)``, or ``(2, nbnd, 3, 3)`` for a collinear run."""
        return (
            self.inverse_mass_by_spin
            if self.nspin == 2
            else self.inverse_mass_by_spin[0]
        )

    @property
    def eigenvalues(self) -> np.ndarray:
        return (
            self.eigenvalues_by_spin
            if self.nspin == 2
            else self.eigenvalues_by_spin[0]
        )

    @property
    def truncation(self) -> np.ndarray | None:
        """``(nbnd,)``, or ``(2, nbnd)`` for a collinear run."""
        if self.truncation_by_spin is None:
            return None
        return (
            self.truncation_by_spin
            if self.nspin == 2
            else self.truncation_by_spin[0]
        )

    @property
    def mass_by_spin(self) -> np.ndarray:
        """The tensor inverted band by band, ``(nspin, nbnd, 3, 3)`` in ``m_e``.

        A band with a vanishing curvature in some direction has no mass there,
        so a singular tensor comes back as ``inf`` rather than raising -- which
        is the physical answer, and the reason the inverse mass is what is
        computed and stored.
        """
        out = np.full_like(self.inverse_mass_by_spin, np.nan)
        for spin in range(self.inverse_mass_by_spin.shape[0]):
            for band in range(self.inverse_mass_by_spin.shape[1]):
                tensor = self.inverse_mass_by_spin[spin, band]
                if not np.all(np.isfinite(tensor)):
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    values, vectors = np.linalg.eigh(tensor)
                    inverted = vectors @ np.diag(1.0 / values) @ vectors.T
                out[spin, band] = inverted
        return out

    @property
    def mass(self) -> np.ndarray:
        return self.mass_by_spin if self.nspin == 2 else self.mass_by_spin[0]

    def principal(self, band: int, spin: int = 0):
        """``(masses, axes)``: the mass tensor's eigenvalues and eigenvectors.

        The three principal masses in ``m_e`` and the cartesian directions they
        belong to. For a cubic crystal at a non-degenerate band at ``Gamma`` the
        three come out equal with nothing imposing it, which is the symmetry
        check this quantity carries for free.
        """
        tensor = self.inverse_mass_by_spin[spin, band]
        if not np.all(np.isfinite(tensor)):
            raise ValueError(
                f"band {band} sits in a degenerate multiplet at this k-point, "
                f"so it has no tensor of its own: the eigensolver's rotation "
                f"inside the manifold rotates it. Read .multiplets for the "
                f"multiplet's summed inverse mass, which is invariant"
            )
        values, vectors = np.linalg.eigh(tensor)
        with np.errstate(divide="ignore"):
            return 1.0 / values, vectors

    def density_of_states_mass(self, band: int, spin: int = 0) -> float:
        """``(m_x m_y m_z)^(1/3)``: the geometric mean of the principal masses.

        What a density of states or a carrier concentration actually sees.
        """
        masses, _ = self.principal(band, spin)
        return float(np.cbrt(np.prod(masses)))


# ---------------------------------------------------------------------------


def effective_mass(
    calculation,
    result,
    kpoint,
    bands=None,
    delta: float = DEFAULT_DELTA,
    method: str = "velocity",
    nbnd: int | None = None,
    conv_thr: float = 1.0e-12,
    k_batch="default",
    richardson: bool = True,
    degeneracy_tol: float = DEFAULT_DEGENERACY_TOL,
) -> EffectiveMass:
    """``(1/m*)_ab = (1/2) d^2 eps_n/dk_a dk_b`` at one k-point, in 1/m_e.

    Args:
        calculation: the :class:`~pypresso.scf.driver.Calculation` the ground
            state was converged with -- its pseudopotentials and cell are what
            the stencil's NSCF runs are built from.
        result: the converged :class:`~pypresso.scf.driver.SCFResult`. The
            density is frozen and the whole mixed state crosses with it
            (``becsum``, ``ns``, ``tau``, the converged field), because a
            fixed-density run needs all of it and forwarding some of it is the
            defect P38 closed in three places.
        kpoint: the centre, in **crystal** coordinates -- ``(0, 0, 0)`` for
            ``Gamma``, ``(0.5, 0.5, 0.5)`` for ``L`` on an fcc cell. The tensor
            that comes back is cartesian.
        bands: which band indices to report; ``None`` is all of them. It
            slices :attr:`EffectiveMass.inverse_mass_by_spin` and its
            companions, but **not** :attr:`~EffectiveMass.multiplets` or
            :attr:`~EffectiveMass.refused`, which keep the original band
            numbering because that is what they are a statement about. The
            degeneracy masking happens before the slice, so a refused band is
            still ``nan`` after it; :meth:`EffectiveMass.principal` indexes the
            *sliced* array, so pass ``bands=None`` if the two have to line up.
        delta: the stencil's outermost displacement in 1/bohr (Elk's
            ``deltaem``). Both routes reach exactly this far, so their
            agreement is a statement about the operator and not about how far
            each one sampled.
        method: ``"velocity"`` (default) or ``"eigenvalue"``; see the module
            docstring for why both exist.
        nbnd: how many bands the stencil's diagonalisation resolves. A mass is
            usually wanted for a conduction band, which the ground state did not
            carry, so this is normally larger than the SCF's own count.
        conv_thr: that diagonalisation's threshold. The default is tight because
            a difference divides the eigensolver's scatter by ``delta``.
        richardson: run the stencil at ``delta`` and ``delta/2`` and combine
            them as ``(4 M(h/2) - M(h)) / 3``, which removes the ``O(h^2)``
            truncation and measures what it was. At Elk's ``deltaem`` that
            error is 3% on silicon's conduction band, so this is on by default.
        degeneracy_tol: eigenvalues closer than this at the centre are one
            multiplet, and the bands in it are refused individually.

    Returns:
        :class:`EffectiveMass`.
    """
    if method not in MASS_METHODS:
        raise ValueError(
            f"unknown effective-mass method {method!r}; expected one of "
            f"{MASS_METHODS}"
        )
    if delta <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}")
    if calculation.spiral:
        raise NotImplementedError(
            "an effective mass on a spin spiral is not implemented: the two "
            "spinor components sit on spheres centred at k + q/2 and k - q/2, "
            "so a k-stencil moves both and the band index does not survive it"
        )

    centre = np.asarray(kpoint, dtype=float).reshape(3)
    # ``h`` and ``h/2`` in one stencil, so the Richardson step is one NSCF over
    # more k-points rather than two runs.
    deltas = (delta, 0.5 * delta) if richardson else (delta,)

    build = _by_velocity if method == "velocity" else _by_eigenvalue
    curvatures, eigenvalues = build(
        calculation, result, centre, deltas, nbnd, conv_thr, k_batch
    )

    # ``(1/m*)_ab = (1/2) d^2 eps / dk_a dk_b`` -- Rydberg atomic units, so a
    # free electron is the identity.
    tensors = [0.5 * curvature for curvature in curvatures]
    if richardson:
        coarse, fine = tensors
        inverse_mass = (4.0 * fine - coarse) / 3.0
        truncation = np.linalg.norm(fine - coarse, axis=(-2, -1)) / 3.0
    else:
        inverse_mass, truncation = tensors[0], None

    asymmetry = _asymmetry(inverse_mass)
    inverse_mass = 0.5 * (inverse_mass + np.swapaxes(inverse_mass, -1, -2))

    multiplets, refused = _multiplets(eigenvalues, inverse_mass, degeneracy_tol)
    reported = _mask_degenerate(inverse_mass, refused)
    if bands is not None:
        wanted = np.asarray(bands, dtype=int)
        reported = reported[:, wanted]
        eigenvalues = eigenvalues[:, wanted]
        asymmetry = asymmetry[:, wanted]
        if truncation is not None:
            truncation = truncation[:, wanted]

    return EffectiveMass(
        kpoint=centre,
        inverse_mass_by_spin=reported,
        eigenvalues_by_spin=np.asarray(eigenvalues),
        multiplets=multiplets,
        asymmetry_by_spin=asymmetry,
        truncation_by_spin=truncation,
        delta=float(delta),
        method=method,
        nspin=calculation.nspin,
        refused=refused,
    )


# -- the two constructions ---------------------------------------------------


def _by_velocity(calculation, result, centre, deltas, nbnd, conv_thr, k_batch):
    """``[v_b(k + h e_a) - v_b(k - h e_a)] / 2h``, one curvature per ``h``.

    The centre carries the eigenvalues the multiplet structure is read off, and
    six axial points per stencil width carry the velocities. They all go through
    :func:`~pypresso.response.velocity.band_velocities` in **one** call, so a
    Richardson pair is one NSCF over thirteen k-points rather than two runs.
    """
    from pypresso.response.velocity import band_velocities

    offsets, index = [np.zeros(3)], {}
    for h in deltas:
        for axis in range(3):
            for sign in (+1, -1):
                step = np.zeros(3)
                step[axis] = sign * h
                index[(h, axis, sign)] = len(offsets)
                offsets.append(step)

    kpoints = _stencil_kpoints(calculation, centre, offsets)
    velocities = band_velocities(
        calculation, result, kpoints=kpoints, nbnd=nbnd, conv_thr=conv_thr,
        k_batch=k_batch,
    )
    # ``(nspin, nk, nbnd, 3)`` -> ``(nk, nspin, nbnd, 3)``: the stencil index
    # has to lead for the differences below, and the spin axis is kept whole.
    v = np.moveaxis(np.asarray(velocities.velocities_by_spin), 1, 0)
    eps = np.moveaxis(np.asarray(velocities.eigenvalues_by_spin), 1, 0)

    curvatures = [
        np.stack([
            (v[index[(h, axis, +1)]] - v[index[(h, axis, -1)]]) / (2.0 * h)
            for axis in range(3)
        ], axis=-2)  # (nspin, nbnd, 3 = a, 3 = b)
        for h in deltas
    ]
    return curvatures, eps[0]


def _by_eigenvalue(calculation, result, centre, deltas, nbnd, conv_thr, k_batch):
    """Elk's route: a second difference of ``eps_n(k)`` alone, per ``h``.

    **The centre k-point is not in the difference, and that is not a detail.**
    The obvious three-point formula ``[eps(+h) - 2 eps(0) + eps(-h)] / h^2`` is
    wrong here for a reason that has nothing to do with the physics: the
    plane-wave sphere is rebuilt at every k, and a *high-symmetry* centre is
    exactly where a whole shell of ``G`` sits on the cutoff. Measured on
    two-atom silicon at ``ecutwfc = 30``: ``Gamma`` holds **725** plane waves
    and every displaced point holds **733**, whatever the displacement. So the
    centre eigenvalue is variationally higher by a fixed basis-set offset
    ``delta ~ 1.2e-6 Ry``, the numerator inherits ``-2 delta``, and the
    curvature inherits ``-delta/h^2`` -- an error that **grows** as the stencil
    shrinks. It was measured growing by exactly four per halving, from 2.1e-4 at
    ``h = 0.05`` to 3.0e-2 at ``h = 0.00625``, while the velocity route
    converged. Neither route looks wrong on its own; what says so is having two.

    This is the sphere-membership discontinuity :mod:`pypresso.response.velocity`
    records for ``dH/dk``, met from the other side. The velocity route is immune
    to it for free -- its ``jvp`` freezes the sphere, and its difference is
    between two *displaced* points that hold the same 733 -- so this is the
    price the eigenvalue route pays for sharing no machinery with it.

    The cure is a stencil that never touches the centre. The diagonal entries
    come from the four axial points at ``+-h`` and ``+-2h``,

        d^2 eps/dk_a^2 = [eps(2h) + eps(-2h) - eps(h) - eps(-h)] / 3h^2,

    which is ``O(h^2)`` like the three-point form with five times its
    coefficient -- paid back by the Richardson step -- and the off-diagonal ones
    from the four-point mixed difference

        d^2 eps/dk_a dk_b = [eps(++) - eps(+-) - eps(-+) + eps(--)] / 4h^2,

    which was already centre-free. Every point in both is displaced, so the
    basis-set offset is common to all of them and cancels.

    The centre is still diagonalised, because the eigenvalues the multiplet
    structure is read off are its own; it is simply never differenced.
    """
    from pypresso.workflows.nscf import fixed_density_bands

    # ``delta`` is the stencil's **outermost** displacement in both routes, so
    # that the two sample the same range of ``k`` and their agreement is about
    # the operator rather than about how far each one reached. This stencil's
    # outermost point is ``2h``, so its ``h`` is half. Skipping this is worth
    # 0.16 on silicon's ``Gamma_1`` band, where ``+-0.05`` is far enough to
    # leave the parabolic region -- and the truncation estimate says so, which
    # is how it was found.
    deltas = [0.5 * d for d in deltas]

    offsets, axial, mixed = [np.zeros(3)], {}, {}
    for h in deltas:
        for axis in range(3):
            for scale in (1, 2):
                for sign in (+1, -1):
                    step = np.zeros(3)
                    step[axis] = sign * scale * h
                    axial[(h, axis, scale, sign)] = len(offsets)
                    offsets.append(step)
        for a in range(3):
            for b in range(a + 1, 3):
                for sa in (+1, -1):
                    for sb in (+1, -1):
                        step = np.zeros(3)
                        step[a], step[b] = sa * h, sb * h
                        mixed[(h, a, b, sa, sb)] = len(offsets)
                        offsets.append(step)

    kpoints = _stencil_kpoints(calculation, centre, offsets)
    _, _, eigenvalues = fixed_density_bands(
        result.system, calculation.pseudos, result.density,
        kpoints=kpoints, nbnd=nbnd, conv_thr=conv_thr, k_batch=k_batch,
        ns=result.ns, tau=getattr(result, "tau", None),
        becsum=result.becsum or (),
        field=result.magnetic_field, field_scale=result.field_scale,
    )
    eps = _as_spin_first(np.asarray(eigenvalues), calculation.nspin)
    eps = np.moveaxis(eps, 1, 0)  # (npoint, nspin, nbnd)

    nspin, nbnd = eps.shape[1], eps.shape[2]
    curvatures = []
    for h in deltas:
        curvature = np.zeros((nspin, nbnd, 3, 3))
        for a in range(3):
            near = eps[axial[(h, a, 1, +1)]] + eps[axial[(h, a, 1, -1)]]
            far = eps[axial[(h, a, 2, +1)]] + eps[axial[(h, a, 2, -1)]]
            curvature[:, :, a, a] = (far - near) / (3.0 * h ** 2)
        for a in range(3):
            for b in range(a + 1, 3):
                value = (
                    eps[mixed[(h, a, b, +1, +1)]] - eps[mixed[(h, a, b, +1, -1)]]
                    - eps[mixed[(h, a, b, -1, +1)]] + eps[mixed[(h, a, b, -1, -1)]]
                ) / (4.0 * h ** 2)
                curvature[:, :, a, b] = value
                curvature[:, :, b, a] = value
        curvatures.append(curvature)
    return curvatures, eps[0]


def _stencil_kpoints(calculation, centre, offsets):
    """The stencil as a :class:`~pypresso.system.kpoints.KPoints`.

    ``centre`` is crystal, ``offsets`` are cartesian in 1/bohr, and what comes
    back is in QE's ``xk`` units of ``2 pi / alat`` -- the conversion is here so
    that the tensor is cartesian while the k-point a user names is not.
    """
    from pypresso.system.kpoints import KPoints

    cell = calculation.system.cell
    base = np.asarray(cell.k_to_cartesian(centre[None]))[0]  # 2 pi / alat
    points = np.stack([
        base + np.asarray(offset) / cell.tpiba for offset in offsets
    ])
    weights = np.full(len(points), 1.0 / len(points))
    return KPoints.from_cartesian(
        points, weights, precision=calculation.system.kpoints.precision
    )


# -- reporting ---------------------------------------------------------------


def _as_spin_first(values, nspin: int):
    """Give a squeezed ``(nk, nbnd)`` array its leading spin axis back."""
    return values if values.ndim == 3 else values[None]


def _asymmetry(tensor):
    """``||M - M^T|| / ||M||`` per band -- the stencil's own error, for free."""
    skew = tensor - np.swapaxes(tensor, -1, -2)
    norm = np.linalg.norm(tensor, axis=(-2, -1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            norm > 0.0, np.linalg.norm(skew, axis=(-2, -1)) / norm, 0.0
        )


def _multiplets(eigenvalues, inverse_mass, tol):
    """Group the bands degenerate at the centre, and sum each group's tensor.

    Returns ``(per_spin_multiplets, refused)``, the second being the
    ``(spin, band)`` pairs whose individual tensor is not a property of the band.
    """
    per_spin, refused = [], []
    for spin in range(eigenvalues.shape[0]):
        groups, start = [], 0
        eps = np.asarray(eigenvalues[spin])
        for band in range(1, len(eps) + 1):
            if band == len(eps) or eps[band] - eps[band - 1] > tol:
                indices = tuple(range(start, band))
                groups.append(Multiplet(
                    bands=indices,
                    eigenvalue=float(np.mean(eps[list(indices)])),
                    inverse_mass_sum=np.sum(
                        inverse_mass[spin, list(indices)], axis=0
                    ),
                ))
                if len(indices) > 1:
                    refused += [(spin, index) for index in indices]
                start = band
        per_spin.append(tuple(groups))
    return tuple(per_spin), tuple(refused)


def _mask_degenerate(inverse_mass, refused):
    """``nan`` where the tensor is not a property of the band it is indexed by.

    A refusal rather than a plausible number: inside a degenerate multiplet the
    per-band tensor is whatever basis the eigensolver happened to return, and it
    is smooth, symmetric and entirely wrong.
    """
    out = np.array(inverse_mass, copy=True)
    for spin, band in refused:
        out[spin, band] = np.nan
    return out
