"""Z2 invariants from the flow of Wannier charge centres.

The construction is Yu, Qi, Bernevig, Fang and Dai's (PRB 84, 075119 (2011)):
take a closed loop of k-points across the zone in one direction, form the
Wilson loop

    W = M(k_0, k_1) M(k_1, k_2) ... M(k_{N-1}, k_0 + b),

and read the phases of its eigenvalues. Those are the hybrid Wannier charge
centres of the occupied manifold, localised along the loop direction and
resolved by the perpendicular momentum. Sweeping that perpendicular momentum --
the *pumping* parameter -- from a time-reversal-invariant plane to the next one
traces a flow, and the Z2 invariant is the **parity of the number of times that
flow crosses an arbitrary reference line**. Soluyanov and Vanderbilt (PRB 83,
235401 (2011)) make the reference line canonical by putting it in the largest
gap between the centres, which is what makes the count well-defined without a
band-by-band identification.

Why this and not a Berry phase or a Pfaffian. It needs nothing of the crystal
except time-reversal symmetry -- no inversion centre, no special gauge, no
Pfaffian of the sewing matrix at the TRIM, and no continuous gauge across the
zone, which is the thing a Z2-nontrivial band structure makes impossible in the
first place. What it does need is spin-orbit coupling: without it every band is
doubly degenerate in spin, the two copies have opposite Chern numbers and the
flow crosses in pairs, so ``nu`` is trivially zero.

**Three conventions are load-bearing** and are taken from ``elkpy``'s
``parsers/wilson.py``, so that a number computed here can be compared with one
computed there without a translation table:

1. **Each link is unitarised before the product** (polar decomposition), not
   just at the end. A product of non-unitary factors has eigenvalues off the
   unit circle and their phases stop being charge centres.
2. **The pumping parameter runs over half the zone, ending exactly at 1/2** --
   ``t_m = m / (2 (npump - 1))``. Both ends must be TRI planes, which is what
   makes the centres return to themselves as a *set* and the crossing count a
   Z2 quantity. See :func:`pypresso.topology.mesh.pumping_mesh`.
3. **The charge-centre angles are the raw phases of the Wilson loop's
   eigenvalues** -- not routed through
   :func:`pypresso.topology.links.berry_phase`, so they carry the opposite sign
   to a Berry phase computed here. That reflects every curve at once, which a
   crossing parity does not see. Negating them would be equally correct and
   would disagree with ``elkpy`` plot for plot.

**In three dimensions** the same 2D invariant is computed on the six planes
``k_i = 0`` and ``k_i = 1/2``, and :func:`combine_3d` assembles them:
``nu_0 = z(k_i = 0) XOR z(k_i = 1/2)`` for *any* ``i`` -- an algebraic identity,
so the three answers must agree and a disagreement is a bug rather than physics
-- and the weak indices are ``nu_i = z(k_i = 1/2)`` (Fu, Kane and Mele, PRL 98,
106803 (2007)).

**A caution that the reference implementation earned the hard way.** A Wilson
loop resolves an anticrossing only if the loop mesh resolves it. ``elkpy``
records graphene with a genuine 15 meV gap whose anticrossing is ~1e-3 wide in
fractional coordinates returning a confident, wrong ``z = 0`` at 24 loop points,
and bulk Bi2Se3 -- correctly ``(1;000)`` by the parity route -- returning
``nu_0 = 0`` from a well-gapped band structure on an 8-point loop. **Refine the
mesh until the answer stops moving, and cross-check against the parity route
wherever there is an inversion centre.** That is what
:func:`pypresso.topology.parity.fu_kane_z2` is for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from pypresso.topology.links import unitarize

__all__ = [
    "WannierFlow",
    "wilson_loop",
    "wannier_centers",
    "wilson_z2_from_loops",
    "z2_from_centers",
    "largest_gap_center",
    "combine_3d",
    "Z2Invariant3D",
]


@dataclass
class WannierFlow:
    """The charge-centre flow over half the zone, and the Z2 it implies."""

    #: ``(npump,)`` the pumping parameter, from 0 to 1/2 inclusive.
    pump: np.ndarray
    #: ``(npump, nbnd)`` charge-centre angles, sorted, in ``(-pi, pi]``.
    centers: np.ndarray
    z2: int
    #: ``(npump,)`` the largest-gap reference line the crossings are counted
    #: against -- worth plotting beside the centres, because a flow whose gap
    #: line jumps erratically is a flow the mesh has not resolved.
    gap_center: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def centers_fractional(self) -> np.ndarray:
        """The centres as positions in the unit cell, in ``[0, 1)``."""
        return (self.centers / (2.0 * np.pi)) % 1.0

    @property
    def gap_step(self) -> float:
        """The largest single-step motion of the reference line, as a fraction.

        **The convergence diagnostic of this method**, and the one number to
        look at before believing the integer. The crossing count asks which
        charge centres the reference line swept past between two pumping steps;
        if the line moves a fifth of the way round the circle in one step, the
        sweep is not resolved and the count is a guess. Measured on bismuthene:
        0.20 on a 7-point pumping mesh, where the answer disagreed with the
        exact parity product, and it is the *only* thing in the output that says
        so. There is no upper bound that is safe in general -- refine until the
        integer stops moving.
        """
        if len(self.gap_center) < 2:
            return 0.0
        step = np.diff(self.gap_center)
        return float(np.max(np.abs((step + np.pi) % (2.0 * np.pi) - np.pi)) / (2.0 * np.pi))


def wilson_loop(states, closing_shift=None, k_batch="default") -> np.ndarray:
    """Charge-centre angles of one closed loop of k-points.

    ``states`` holds the occupied manifold at the ``N`` points of the loop **in
    order**; the loop is closed from the last point back to the first, displaced
    by ``closing_shift`` -- the reciprocal lattice vector that makes them
    neighbours. Omitting that shift is the classic error: the last link is then
    an overlap between two points a whole zone apart, and the centres come out
    of a matrix that is not a Wilson loop.

    Returns ``(nbnd,)`` sorted angles in ``(-pi, pi]``.
    """
    n = states.nk
    if n < 2:
        raise ValueError("a Wilson loop needs at least two k-points")
    zero = np.zeros(3, dtype=int)
    shift = zero if closing_shift is None else np.asarray(closing_shift, dtype=int)

    interior = [(i, i + 1, zero) for i in range(n - 1)]
    matrices = list(states.overlaps(interior, k_batch=k_batch))
    matrices.append(states.overlaps([(n - 1, 0, shift)], k_batch=1)[0])

    product = jnp.eye(states.nbnd, dtype=matrices[0].dtype)
    for matrix in matrices:
        product = product @ unitarize(matrix)
    return np.sort(np.angle(np.linalg.eigvals(np.asarray(product))))


def wannier_centers(loops, pump=None) -> np.ndarray:
    """The charge centres of a sequence of loops, stacked as ``(npump, nbnd)``.

    ``loops`` is an iterable of ``(state set, closing shift)``. Taking an
    iterable rather than one mesh-shaped state set is the memory decision that
    makes a real calculation possible: the states of one loop are built,
    reduced to ``nbnd`` angles, and dropped before the next loop is
    diagonalised, so the working set is one loop's -- ``nloop * nbnd * npol *
    npwx * 16`` bytes -- instead of the whole half-zone mesh's.
    """
    rows = [wilson_loop(states, shift) for states, shift in loops]
    return np.array(rows)


def largest_gap_center(angles: np.ndarray) -> float:
    """The middle of the largest gap between charge centres on the circle.

    Sort, take the differences including the wrap from the last back to the
    first plus ``2 pi``, and return the midpoint of the largest. This is
    Soluyanov and Vanderbilt's reference line: it is the one place on the circle
    guaranteed to be as far as possible from every centre, so a centre crossing
    it is a real crossing and not a numerical wobble.
    """
    ordered = np.sort(np.asarray(angles))
    extended = np.append(ordered, ordered[0] + 2.0 * np.pi)
    gaps = np.diff(extended)
    index = int(np.argmax(gaps))
    center = ordered[index] + gaps[index] / 2.0
    return float(((center + np.pi) % (2.0 * np.pi)) - np.pi)


def _orientation(a: float, b: float, c: float) -> float:
    """Signed area of the circular triangle ``(a, b, c)``.

    ``sin(b-a) + sin(c-b) + sin(a-c)``: positive when the three points are in
    counter-clockwise order, so its sign says which side of the arc swept by the
    reference line between two pumping steps a charge centre lies on. Exactly
    zero is left alone -- a centre sitting on the reference line has not crossed
    it, and counting it either way would make the invariant depend on round-off.
    """
    return np.sin(b - a) + np.sin(c - b) + np.sin(a - c)


def z2_from_centers(centers: np.ndarray) -> tuple[int, np.ndarray]:
    """Count the crossings of the charge-centre flow. Returns ``(z2, gap line)``.

    For each consecutive pair of pumping steps, the reference line moves from
    ``z(t_m)`` to ``z(t_{m+1})``; every charge centre **of the later step** that
    lies inside the arc swept flips the parity. Sweeping over half the zone and
    taking the parity is the Z2 invariant.

    **The count is only meaningful modulo two, and that is not a caveat -- it is
    what makes it robust.** The signed-area test is not symmetric in the
    direction the line moves: for a *backwards* step it selects the complement
    of the arc rather than the arc, so it returns ``n_bands - n`` where a
    forwards step would return ``n``. Those differ by an even number whenever
    the manifold has an even number of bands, which a Z2 calculation always does
    -- spin-orbit coupling makes every level a Kramers doublet. So the parity is
    the same either way, and an odd band count is refused rather than counted,
    because there the two disagree and neither is the invariant.
    """
    centers = np.atleast_2d(np.asarray(centers))
    if centers.shape[1] % 2:
        raise ValueError(
            f"{centers.shape[1]} charge centres is odd; a Z2 invariant is a "
            "property of a Kramers-degenerate manifold, so the band count must "
            "be even. An odd one means the window has cut a doublet in half, or "
            "that the calculation has no spin-orbit coupling"
        )
    gap = np.array([largest_gap_center(row) for row in centers])
    parity = 1
    for m in range(len(centers) - 1):
        for theta in centers[m + 1]:
            if np.sign(_orientation(gap[m], gap[m + 1], theta)) < 0:
                parity *= -1
    return (0 if parity > 0 else 1), gap


def wilson_z2_from_loops(loops, pump=None) -> WannierFlow:
    """The whole 2D Z2 calculation, from a sequence of loops to an integer."""
    centers = wannier_centers(loops)
    z2, gap = z2_from_centers(centers)
    if pump is None:
        pump = np.arange(len(centers)) / (2.0 * (len(centers) - 1))
    return WannierFlow(
        pump=np.asarray(pump), centers=centers, z2=int(z2), gap_center=gap
    )


@dataclass
class Z2Invariant3D:
    """The four indices ``(nu0; nu1 nu2 nu3)`` and the planes they came from."""

    nu0: int
    nu: tuple[int, int, int]
    #: ``nu0`` computed from each axis separately. They must agree.
    nu0_by_axis: tuple[int, int, int]
    planes: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"({self.nu0}; {self.nu[0]}{self.nu[1]}{self.nu[2]})"


def combine_3d(planes: dict) -> Z2Invariant3D:
    """Assemble the four 3D indices from the six planes' 2D invariants.

    ``planes`` maps ``(axis, offset)`` -- ``axis`` in ``0, 1, 2`` and ``offset``
    in ``0.0, 0.5`` -- to that plane's ``z2``.

    ``nu_0 = z(k_i = 0) XOR z(k_i = 1/2)`` holds for every ``i`` as an algebraic
    identity, so the three values are computed and compared: a disagreement
    means one of the six planes was not resolved, and reporting a majority
    verdict would hide exactly the failure that matters.
    """
    missing = [
        key for key in ((a, o) for a in range(3) for o in (0.0, 0.5)) if key not in planes
    ]
    if missing:
        raise ValueError(f"the 3D indices need all six planes; missing {missing}")
    by_axis = tuple(
        (int(planes[(a, 0.0)]) + int(planes[(a, 0.5)])) % 2 for a in range(3)
    )
    if len(set(by_axis)) != 1:
        raise ValueError(
            f"the strong index disagrees across axes: {by_axis}. This is an "
            "algebraic identity, not a physical quantity, so a disagreement "
            "means at least one plane's Wilson loop is unconverged -- refine "
            "the loop and pump meshes"
        )
    weak = tuple(int(planes[(a, 0.5)]) for a in range(3))
    return Z2Invariant3D(
        nu0=int(by_axis[0]), nu=weak, nu0_by_axis=by_axis, planes=dict(planes)
    )
