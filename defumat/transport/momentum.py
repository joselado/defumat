"""Which k-points carry the tunnelling current: a *planar* tip, resolved in k.

:mod:`defumat.transport.green` asks where on a surface the current goes -- a
**point** tip, a map over tip position, every k-point summed into every pixel.
This asks the conjugate question. Replace the point tip by a plane parallel to
the surface and the map collapses; what is left is one number per k-point, and
that number is the answer to "which pocket of the Fermi surface does an
electron actually tunnel out of".

The two are the same object seen twice, and that is a theorem rather than a
coincidence: integrating the point-tip map over the tip plane gives exactly the
sum over k of what is here. The derivation is one line. :mod:`.green` has

    T(r) = sum_k w_k sum_{n,m} a_kn(r) S^exit_k[n, m] a*_km(r),

    S^exit_k[n, m] = int_exit psi*_kn(r') psi_km(r') d^2 r',

with the tip amplitudes ``a_kn(r) = g_kn psi_kn(r)``. Integrating over one
cell's worth of the tip plane turns ``int a_kn(r) a*_km(r) d^2 r`` into
``g_kn g_km S^tip_k[m, n]``, where ``S^tip`` is the *same* Gram matrix
(:func:`~defumat.transport.substrate.exit_overlap`) evaluated at the tip's
height instead of the substrate's. So

    W(k) = w_k  Tr[ D_k S^exit_k D_k S^tip_k ],   D_k = diag(g_kn),

one trace of a **product** of two Hermitian matrices per k-point. Nothing is
sampled on a real-space grid, so this costs a small fraction of the map: two
:func:`exit_overlap` calls and an ``nbnd^2`` contraction, against an
``nk x nbnd x npoints`` sampling.

**The index order in that trace is the whole difficulty, and it is P66's trap
one level up.** The elementwise-looking alternative,
``sum_{nm} g_n g_m S^exit[n, m] S^tip[n, m] = Tr[D S^exit D (S^tip)^T]``, is
real, non-negative, invariant under the rotation a degenerate eigensolver is
free in, and **exactly equal to the right answer whenever either Gram matrix is
diagonal** -- so it agrees in the Tersoff-Hamann limit, agrees on a
single-band metal, passes every check that does not involve two distinct planes
at once, and is wrong precisely where the interference this quantity is for
actually lives. Only a literal evaluation of the double plane integral, on two
different planes, separates them. Transposing *both* conventions together is
harmless; transposing one is the bug. (Independently confirmed against elkpy's
Elk task 9007, which contracts `zgemm('C','N')` Gram matrices the same way and
records the same trap.)

**Three things fall out of the correct form and each is worth stating.**

*It is non-negative and real by construction.* Both Gram matrices are positive
semi-definite and ``D`` is a real non-negative diagonal, so ``D S D`` is
positive semi-definite too, and ``Tr[XY] >= 0`` for a pair of them. There is no
place for a sign to come from, unlike a tunnelling density built from a smeared
delta.

*It is blind to the rotation a degenerate eigensolver is free in* -- rule D4,
and here it is free. ``Tr[U^dagger X U U^dagger Y U] = Tr[XY]``, so a multiplet
may come back in any basis and ``W(k)`` does not move. The *incoherent* column
below is a diagonal and is not, which is why it is taken in the substrate's own
channel basis (:func:`~defumat.transport.green.channel_basis`) rather than as
the solver returns it.

*Tersoff-Hamann is the ``S^exit -> 1`` limit, exactly.* Widen the substrate from
a plane to the whole cell and orthonormality makes ``S^exit`` the identity,
leaving ``Tr[D^2 S^tip] = sum_n g_kn^2 int_tip |psi_kn|^2``, which is the local
density of states integrated over the tip plane -- P65's quantity, with no
factor between them. So the columns are computed here from the same two arrays
and reported side by side rather than from separate code, and the third beside
them is the plain Fermi surface ``sum_n delta(E - e_kn)``, the further limit in
which the tip plane stops discriminating too. **The physics this is for is the
ratio between those columns**: a large-``|k_par|`` pocket has a faster-decaying
vacuum tail than a zone-centre one, so it is suppressed in the tunnelling
column while carrying most of the bare one.

**The decay constants are the check, and they are a check on the physics rather
than on the code.** A state at in-plane momentum ``k_par`` decays into the
vacuum as ``exp(-sqrt(kappa_0^2 + |k_par|^2) z)``, so two pockets measured at
two tip heights give two decay constants obeying

    kappa_1^2 - kappa_2^2 = |k_1|^2 - |k_2|^2

with no reference to either code's machinery. :func:`decay_constants` fits them
and :func:`decay_identity` reports the residual. ``kappa`` is the **amplitude**
decay and ``W`` is quadratic in the wavefunction on each plane, so a sweep of
the tip plane alone has ``d ln W / dz = -2 kappa`` and a sweep of *both* planes
outward has ``-4 kappa``; getting that factor wrong makes the identity come out
four times too large, which is what it is for.

**What it does not do.** The plane is a plane: a tip with structure of its own
is a different ``Gamma_t``, and this is the featureless limit of it. A
*magnetic* plane tip is refused rather than approximated -- it contracts the two
spinor components through its own 2x2 projector, which the plane integral does
not collapse the same way; the substrate's acceptance does go in, inside the
Gram matrix, exactly as it does for the map. And the k-set must be the whole
grid rather than a wedge, for the same reason
:func:`~defumat.workflows.transport.run_vertical_transport` needs it -- the
answer is a function *of* k, so there is nothing for a symmetry sum to do to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["MomentumTransport", "momentum_weights", "decay_constants",
           "decay_identity", "fold_into_zone", "pocket_mask"]


def momentum_weights(exit_overlaps, tip_overlaps, kweights, weights,
                     eigenvalues=None, tol=None):
    """``W(k) = w_k Tr[D S^exit D S^tip]``, and the limits it is read against.

    Args:
        exit_overlaps: ``(nk, nbnd, nbnd)`` complex, the substrate plane's Gram
            matrices from :func:`~defumat.transport.substrate.exit_overlap`.
        tip_overlaps: ``(nk, nbnd, nbnd)`` complex, the *same function* at the
            tip's height.
        kweights: ``(nk,)`` the run's own k-point weights, carrying the spin
            degeneracy exactly as they do everywhere else here.
        weights: ``(nk, nbnd)`` real, the per-state amplitude factor from
            :func:`~defumat.transport.green.amplitude_weights` -- ``sqrt(delta(E
            - e)/eta)``, so that ``weights**2`` is ``delta/eta``.
        eigenvalues: ``(nk, nbnd)`` in Ry. Only for the incoherent column, to
            find the degenerate multiplets whose basis a diagonal would
            otherwise depend on. ``None`` leaves that column out.
        tol: what counts as degenerate, in Ry; defaults to
            :data:`~defumat.transport.green.DEGENERACY_TOL`.

    Returns:
        A dict of ``(nk,)`` real, non-negative columns.

        * ``weight`` -- the transmission through the material, ``W(k)`` above.
        * ``tersoff_hamann`` -- its ``S^exit -> 1`` limit, ``w_k Tr[D^2
          S^tip]``: the tip plane's local density of states, with the substrate
          made structureless.
        * ``bare`` -- the further limit ``w_k Tr[D^2]``, in which the tip plane
          stops discriminating as well: the plain Fermi-surface weight ``w_k
          sum_n delta(E - e_kn)/eta``, which is
          :func:`~defumat.response.nesting.fermi_surface_weights` divided by
          ``eta``.
        * ``incoherent`` -- only with ``eigenvalues``: every channel tunnelling
          on its own, ``w_k sum_n g_n^2 S^tip[n,n] S^exit[n,n]`` in the basis
          that diagonalises ``S^exit`` inside each degenerate multiplet. The
          difference from ``weight`` is the interference, which is the part no
          local density of states can carry.

    The columns share ``D`` and differ only in what they contract it against, so
    a ratio between two of them is exactly what the extra structure is worth.
    """
    from defumat.transport.green import DEGENERACY_TOL, channel_basis

    exit_overlaps = np.asarray(exit_overlaps)
    tip_overlaps = np.asarray(tip_overlaps)
    kweights = np.asarray(kweights, dtype=float)
    weights = np.asarray(weights)

    if exit_overlaps.shape != tip_overlaps.shape:
        raise ValueError(
            f"the exit overlaps are {exit_overlaps.shape} and the tip overlaps "
            f"{tip_overlaps.shape}; they are the same function at two heights "
            "and must have the same shape"
        )
    nk, nbnd = exit_overlaps.shape[0], exit_overlaps.shape[1]
    if exit_overlaps.shape != (nk, nbnd, nbnd):
        raise ValueError(
            f"the overlaps are {exit_overlaps.shape}, which is not "
            f"(nk, nbnd, nbnd)"
        )
    if weights.shape != (nk, nbnd) or kweights.shape != (nk,):
        raise ValueError(
            f"state weights {weights.shape} and k weights {kweights.shape} do "
            f"not match {nk} k-points and {nbnd} bands"
        )

    # ``D S D``: the amplitude weight is real and non-negative, so this is
    # still positive semi-definite and the trace below cannot go negative.
    amplitude = np.real(weights).astype(float)
    scaled = exit_overlaps * amplitude[:, :, None] * amplitude[:, None, :]

    # ``Tr[X Y]`` -- the second index of one against the *first* of the other.
    # ``einsum("kij,kij->k")`` would be the transposed form the module docstring
    # names, which differs only in the interference and only where the exit
    # plane has an off-diagonal.
    columns = {
        "weight": kweights * np.real(
            np.einsum("kij,kji->k", scaled, tip_overlaps, optimize=True)),
        "tersoff_hamann": kweights * np.real(
            np.einsum("kn,knn->k", amplitude**2, tip_overlaps, optimize=True)),
        "bare": kweights * (amplitude**2).sum(axis=1),
    }

    if eigenvalues is not None:
        # The substrate's own channels first, so that "each band on its own" is
        # a statement rather than a choice of basis: rule D4 arriving in a
        # diagnostic. ``green.transmission`` does exactly this for its own
        # incoherent map.
        u = channel_basis(exit_overlaps, np.asarray(eigenvalues),
                          DEGENERACY_TOL if tol is None else float(tol))
        rotated_exit = np.einsum("kni,knm,kmj->kij", u.conj(), exit_overlaps, u,
                                 optimize=True)
        rotated_tip = np.einsum("kni,knm,kmj->kij", u.conj(), tip_overlaps, u,
                                optimize=True)
        columns["incoherent"] = kweights * np.einsum(
            "kn,kn,kn->k", amplitude**2,
            np.real(np.einsum("knn->kn", rotated_tip)),
            np.real(np.einsum("knn->kn", rotated_exit)), optimize=True)
    return columns


@dataclass(frozen=True)
class MomentumTransport:
    """``W(k)`` on the whole grid, with the limits it is read against."""

    #: ``(nE, nk)`` -- or ``(nk,)`` when one energy was asked for. The
    #: transmission per k-point, in the same arbitrary units as
    #: :class:`~defumat.transport.green.VerticalTransport`: the tip and
    #: substrate couplings are unfixed prefactors. **The k-point weight is in
    #: it**, so that summing the column is the total; :attr:`per_kpoint`
    #: divides it back out, which is the form a per-k comparison wants.
    weight: np.ndarray
    #: Same shape. The ``S^exit -> 1`` limit: the tip plane's local density of
    #: states, which is what a Tersoff-Hamann picture would give.
    tersoff_hamann: np.ndarray
    #: Same shape. The plain Fermi-surface weight ``sum_n delta(E - e_kn)/eta``
    #: times the k-point weight -- the surface before any tunnelling.
    bare: np.ndarray
    #: ``(nk, 3)`` crystal coordinates, in ``monkhorst_pack`` order.
    kpoints: np.ndarray
    #: ``(nk, 3)`` cartesian, in 1/bohr, without the ``2 pi``.
    kcartesian: np.ndarray
    #: ``(nk,)`` the k-point weights that are folded into the columns.
    kweights: np.ndarray
    #: ``(nE,)`` the energies in Ry, absolute.
    energies: np.ndarray
    #: The delta's width in Ry.
    broadening: float
    #: Which delta the on-shell amplitude is the square root of. Carried
    #: because a Gaussian and a Fermi-Dirac delta of the same width differ by a
    #: factor 2.1 in full width, so a number without its name is not comparable.
    smearing: str = "gaussian"
    #: Same shape as :attr:`weight`, with every channel tunnelling on its own.
    #: ``None`` unless it was asked for.
    incoherent: np.ndarray | None = None
    #: The tip plane's crystal coordinate, or the tuple of them when a sweep
    #: was asked for -- in which case every column has a leading height axis.
    height: float | tuple = 0.0
    exit_height: float = 0.0
    exit_axis: int = 2
    grid: tuple[int, int, int] | None = None
    fermi_energy: float | None = None
    #: Diagnostics, as :class:`~defumat.transport.green.VerticalTransport` has
    #: them: the smallest eigenvalue of any Gram matrix (a Gram matrix is
    #: positive semi-definite, so a negative one is a bug) and the worst
    #: departure from Hermiticity.
    least_eigenvalue: float = 0.0
    hermiticity: float = 0.0
    notes: dict = field(default_factory=dict)

    def column(self, name: str = "weight", index: int = 0) -> np.ndarray:
        """``(nk,)`` of one column -- what to plot.

        The stored arrays are ``(nheights, nenergies, nk)`` with each of the
        leading axes squeezed away when a scalar was asked for, so this walks
        down to one k-axis. ``index`` picks along whichever leading axes remain,
        outermost first: on a height sweep at one energy it is the height.
        """
        values = getattr(self, name)
        if values is None:
            raise ValueError(f"this result carries no {name!r} column")
        while values.ndim > 1:
            values = values[index]
        return values

    @property
    def heights(self) -> np.ndarray:
        """``(nh,)`` the tip heights, whether one was asked for or a sweep."""
        return np.atleast_1d(np.asarray(self.height, dtype=float))

    @property
    def map(self) -> np.ndarray:
        """``(nk,)`` of the transmission at the first height and energy."""
        return self.column("weight")

    def sweep(self, name: str = "weight") -> np.ndarray:
        """``(nh, nk)`` of one column across the heights, at the first energy.

        What :func:`decay_constants` is fed. A single height comes back as one
        row rather than as an error, so the same code reads both.
        """
        values = getattr(self, name)
        if values is None:
            raise ValueError(f"this result carries no {name!r} column")
        if values.ndim == 1:
            return values[None]
        while values.ndim > 2:
            values = values[:, 0]
        return values

    @property
    def interference(self) -> np.ndarray | None:
        """Coherent minus incoherent: the part no local picture can give."""
        if self.incoherent is None:
            return None
        return self.weight - self.incoherent

    def per_kpoint(self, name: str = "weight", index: int = 0) -> np.ndarray:
        """One column with the k-point weight divided out.

        The columns carry ``w_k`` because every sum in this package does, and
        because the total is what a conductance is. A **comparison against
        another code** wants the other form: a per-k density, whose value at one
        k-point does not depend on how many k-points there are. Elk's task 9007
        reports that one and applies its weights only at the end.
        """
        values = self.column(name, index)
        return values / np.asarray(self.kweights, dtype=float)

    def as_grid(self, name: str = "weight", index: int = 0) -> np.ndarray:
        """One column reshaped to ``(n1, n2, n3)``, for a Brillouin-zone image.

        The k-points are in ``monkhorst_pack`` order, last index fastest, so
        this is a reshape and not a permutation.
        """
        if self.grid is None:
            raise ValueError(
                "this result did not come from a Monkhorst-Pack grid, so it "
                "has no shape to be reshaped to"
            )
        return self.column(name, index).reshape(self.grid)

    def shares(self, mask, index: int = 0) -> dict:
        """What a region of the zone is worth in each column, and the ratio.

        Args:
            mask: ``(nk,)`` boolean, the k-points of the region -- a pocket.
                :func:`pocket_mask` builds the zone-corner one.

        Returns each column's **share** of its own total over the region, and
        ``suppression``, the bare share divided by the tunnelling share: how
        much less of the current the pocket carries than its size on the Fermi
        surface would suggest. Shares rather than absolute weights, because the
        lead couplings are unfixed prefactors and a ratio of shares survives
        them.
        """
        mask = np.asarray(mask, dtype=bool)
        out = {}
        for name in ("bare", "tersoff_hamann", "weight", "incoherent"):
            if getattr(self, name) is None:
                continue
            values = self.column(name, index)
            total = float(values.sum())
            out[name] = float(values[mask].sum()) / total if total else 0.0
        out["suppression"] = (out["bare"] / out["weight"]
                              if out["weight"] else float("inf"))
        return out

    def reweighting(self, mask, index: int = 0) -> float:
        """``(W_out/W_in) / (D_out/D_in)`` -- how the junction re-ranks two regions.

        The contrast a momentum-resolved tunnelling experiment reports is not
        the ratio of two pockets' weights, which depends on their sizes; it is
        how much that ratio has *moved* relative to the plain density of
        states. Both factors are ratios of shares, so both are free of the
        unfixed prefactors, and the density-of-states half is
        height-independent -- which is what makes this quantity inherit the
        pure exponential of the vacuum decay and makes the ``kappa`` identity
        bite on it.
        """
        share = self.shares(mask, index)
        inside_w, inside_d = share["weight"], share["bare"]
        if inside_w <= 0.0 or inside_d <= 0.0 or inside_d >= 1.0:
            raise ValueError(
                "the reweighting needs both regions to carry weight in both "
                f"columns; got tunnelling share {inside_w} and bare share "
                f"{inside_d}"
            )
        return ((1.0 - inside_w) / inside_w) / ((1.0 - inside_d) / inside_d)


# -- the zone, and the pockets -------------------------------------------------


def fold_into_zone(kcartesian, cell, axis: int = 2) -> np.ndarray:
    """Every k-point moved to its shortest periodic image, ``(nk, 3)`` cartesian.

    The nine in-plane reciprocal-lattice translations ``n1 b1 + n2 b2`` with
    ``n1, n2`` in ``{-1, 0, 1}`` are enough for a Monkhorst-Pack grid in
    ``[0, 1)``: the shortest image of such a point is never more than one cell
    away in either direction. The axis normal to the layer is left alone -- a
    two-dimensional material has one k-point along it.
    """
    kcartesian = np.asarray(kcartesian, dtype=float)
    bg = np.asarray(cell.bg, dtype=float)
    plane = [i for i in (0, 1, 2) if i != axis]
    shifts = np.array([n1 * bg[plane[0]] + n2 * bg[plane[1]]
                       for n1 in (-1, 0, 1) for n2 in (-1, 0, 1)])
    candidates = kcartesian[:, None, :] + shifts[None, :, :]
    best = np.argmin(np.linalg.norm(candidates, axis=2), axis=1)
    return candidates[np.arange(kcartesian.shape[0]), best]


#: The six corners of a hexagonal Brillouin zone, in reciprocal-lattice
#: coordinates. ``K`` and ``K'`` alternate around it.
HEXAGONAL_CORNERS = (
    (1 / 3, 1 / 3, 0.0), (-1 / 3, -1 / 3, 0.0),
    (2 / 3, -1 / 3, 0.0), (-2 / 3, 1 / 3, 0.0),
    (1 / 3, -2 / 3, 0.0), (-1 / 3, 2 / 3, 0.0),
)


def pocket_mask(kcartesian, cell, corners=HEXAGONAL_CORNERS,
                axis: int = 2) -> np.ndarray:
    """``(nk,)`` boolean: which k-points belong to the zone-corner pockets.

    A k-point is a corner point when it is **closer to a corner than to the
    zone centre**, once folded to its shortest image. That is a partition of
    the whole zone rather than a mask on a contour: every k-point belongs to
    exactly one side, so the two shares add to one and the delta does the
    energy selection on its own. Masking on the weight instead would make the
    shares depend on where the threshold was put.

    ``corners`` are in reciprocal-lattice coordinates; the default is the
    hexagon's six, which is what a transition-metal dichalcogenide's ``K`` and
    ``K'`` valleys sit on.
    """
    folded = fold_into_zone(kcartesian, cell, axis)
    bg = np.asarray(cell.bg, dtype=float)
    corner = np.asarray(corners, dtype=float) @ bg
    to_corner = np.linalg.norm(folded[:, None, :] - corner[None, :, :],
                               axis=2).min(axis=1)
    return to_corner < np.linalg.norm(folded, axis=1)


# -- the vacuum decay ----------------------------------------------------------


def decay_constants(heights, weights, planes: str = "tip") -> np.ndarray:
    """``kappa`` in 1/bohr from a height sweep, one per column of ``weights``.

    Args:
        heights: ``(nh,)`` tip heights in **bohr**, measured from wherever --
            only differences enter.
        weights: ``(nh, ...)`` the tunnelling weight at each height.
        planes: ``"tip"`` if only the tip plane moved and the exit plane stayed
            put, ``"both"`` if the two moved outward together by the same
            distance.

    A vacuum tail is ``psi ~ exp(-kappa z)`` and the weight is quadratic in the
    wavefunction **on each plane**, so ``log W = const - 2 kappa z`` for a tip
    sweep and ``- 4 kappa z`` when both planes move. This returns ``kappa``, not
    ``2 kappa``. **That factor is the one convention worth checking against an
    identity rather than against another code**: get it wrong and every
    ``kappa`` is out by two, which :func:`decay_identity` sees as a factor of
    four.

    A straight line is fitted through *all* the heights given, so the caller
    trims the sweep to its straight part first -- past the point where a
    plane-wave basis's own floor takes over, the weight stops decaying at all,
    and a fit through that is a fit to an artefact rather than to a tail.
    """
    factors = {"tip": 2.0, "both": 4.0}
    if planes not in factors:
        raise ValueError(
            f"planes must be 'tip' (only the tip moved) or 'both', got {planes!r}"
        )
    heights = np.asarray(heights, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if heights.ndim != 1 or heights.shape[0] != weights.shape[0]:
        raise ValueError(
            f"{heights.shape[0]} heights and {weights.shape[0]} rows of weights"
        )
    if heights.shape[0] < 2:
        raise ValueError("a decay constant needs at least two heights")
    if np.any(weights <= 0.0):
        raise ValueError(
            "a decay constant is fitted to log W and some weight is not "
            "positive; trim the sweep to the range where the tail is resolved"
        )
    flat = weights.reshape(heights.shape[0], -1)
    slope = np.polyfit(heights, np.log(flat), 1)[0]
    return (-slope / factors[planes]).reshape(weights.shape[1:])


def decay_identity(kappa_1: float, kappa_2: float, k_1, k_2) -> dict:
    """``kappa_1^2 - kappa_2^2`` against ``|k_1|^2 - |k_2|^2``.

    The check that shares no machinery with the calculation: a state at in-plane
    momentum ``k_par`` tunnels through the vacuum as ``exp(-sqrt(kappa_0^2 +
    |k_par|^2) z)``, with ``kappa_0`` set by the barrier height alone, so the
    difference of two pockets' squared decay constants is the difference of
    their squared momenta and nothing else. Both arguments are in 1/bohr.

    Returns the two sides and the relative residual. A residual near **3**
    rather than near zero is :func:`decay_constants`'s factor of two applied on
    one side only.

    **The residual has a sign and the sign means something.** ``k_1`` and
    ``k_2`` are the pockets' *centres*, and a pocket of finite radius sits at a
    larger ``|k_par|`` than its centre on the side facing outward -- so a
    zone-centre *hole* pocket of radius ``k_0`` raises ``kappa_2`` and makes the
    measured difference come out **low** by about ``k_0^2``. Reading a small
    deficit as a failure of the identity, rather than as a measurement of that
    radius, is the mistake this note is here to prevent.
    """
    k_1 = float(np.linalg.norm(np.asarray(k_1, dtype=float)))
    k_2 = float(np.linalg.norm(np.asarray(k_2, dtype=float)))
    measured = float(kappa_1) ** 2 - float(kappa_2) ** 2
    expected = k_1**2 - k_2**2
    deficit = expected - measured
    return {
        "measured": measured,
        "expected": expected,
        "relative_residual": (abs(measured - expected) / abs(expected)
                              if expected else float("inf")),
        # What a zone-centre pocket of this radius would account for, in the
        # units of ``k_1``/``k_2``. Meaningless when the residual is positive.
        "implied_pocket_radius": float(np.sqrt(deficit)) if deficit > 0 else 0.0,
        "kappa": (float(kappa_1), float(kappa_2)),
        "k_parallel": (k_1, k_2),
    }
