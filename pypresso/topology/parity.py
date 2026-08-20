"""The Fu-Kane parity criterion: Z2 from eight k-points.

When a crystal has an inversion centre, Fu and Kane (PRB 76, 045302 (2007))
showed that the Z2 invariants are products of the *parity eigenvalues* of the
occupied bands at the time-reversal-invariant momenta. At a TRIM the Bloch
states are eigenstates of inversion, spin-orbit coupling makes every level a
Kramers doublet whose two partners share a parity, and

    delta(k) = prod over occupied Kramers pairs  xi_2m(k)  =  (-1)^(N_- / 2)

with ``N_-`` the number of occupied states of odd parity. Then, in three
dimensions, ``(-1)^nu_0`` is the product of all eight deltas and
``(-1)^nu_i`` the product of the four with ``k_i = 1/2``; in two dimensions
``(-1)^nu`` is the product of the four.

**Why this is worth having beside the Wilson loop.** It is exact -- there is no
mesh, so there is nothing to converge -- and it costs *eight* diagonalisations
where a Wilson loop costs a whole half-zone. And it shares no machinery with the
Wilson loop beyond the state set, so the two agreeing is a real check on both.
The reference implementation records two cases where the Wilson route returned a
confident wrong integer on an unresolved mesh and the parity route did not
(``elkpy``: graphene with a narrow anticrossing, bulk Bi2Se3 on an 8-point
loop). It needs an inversion centre, which is why it cannot replace the other.

**The halving is not cosmetic.** With spin-orbit coupling the product over *all*
occupied states is identically ``+1``, because the two members of every Kramers
doublet carry the same parity. Taking ``N_- // 2`` is what recovers the
information. An odd ``N_-``, or an odd number of occupied states, means the band
window cut a Kramers pair in half -- the manifold is not the full occupied one
-- and is raised rather than rounded.

**Individual deltas are not physical; only the products are.** Moving the
inversion centre by ``t`` multiplies ``delta(k)`` by ``(e^{2 pi i k . 2t})^N``,
so symmetry-related TRIM can carry different deltas for an odd number of
occupied Kramers pairs. Every invariant here is a product over an *even* number
of TRIM, which is exactly what makes it immune to that -- and to a global sign
error in the parity matrix, and to a different choice of where the occupied
window starts.

The parity matrix itself is built in
:meth:`pypresso.topology.states.PlaneWaveStates.parity_matrix`, where inversion
is a permutation of the plane-wave sphere onto itself. It is diagonalised rather
than read off the diagonal: a TRIM spectrum is heavily degenerate and the basis
inside a multiplet is whatever the eigensolver left there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ParityInvariant",
    "parity_eigenvalues",
    "trim_delta",
    "fu_kane_z2",
    "inversion_centre",
]

#: How far a parity eigenvalue may sit from +-1, and the parity matrix from
#: Hermitian, before the manifold is declared not to be an inversion eigenspace.
#:
#: Tight on purpose. Inversion maps the plane-wave sphere onto itself exactly,
#: so ``P`` is built by permuting coefficients and multiplying by phases and
#: nothing is approximated: measured on silicon, ``|P - P^H| ~ 1e-16`` and
#: ``|P^2 - 1| ~ 5e-11``, limited only by how tightly the eigensolver converged.
#: A number anywhere near the middle between ``+1`` and ``-1`` means the band
#: window is not closed under inversion, and rounding it to the nearer sign is
#: exactly how a confident wrong invariant is produced. (``elkpy`` uses ``5e-2``
#: because Elk's overlaps carry a ~1e-3 real-space truncation floor; there is no
#: such floor here, so the tolerance that hid it is not wanted.)
TOLERANCE = 1.0e-6


@dataclass
class ParityInvariant:
    """The Fu-Kane result: the indices, and the deltas they were built from."""

    #: 2 or 3.
    dimension: int
    nu0: int
    nu: tuple[int, int, int] | None
    #: ``{crystal k-point (as a tuple) -> delta}``.
    deltas: dict = field(default_factory=dict)
    #: ``{k-point -> sorted parity eigenvalues}``, for inspection.
    eigenvalues: dict = field(default_factory=dict)

    @property
    def z2(self) -> int:
        return self.nu0

    def __str__(self) -> str:
        if self.dimension == 2:
            return f"nu = {self.nu0}"
        return f"({self.nu0}; {self.nu[0]}{self.nu[1]}{self.nu[2]})"


def parity_eigenvalues(matrix, tolerance: float = TOLERANCE) -> np.ndarray:
    """The ``+-1`` parity eigenvalues of an occupied manifold, sorted.

    ``matrix`` is ``P_mn = <u_m|S P|u_n>`` over the manifold. It must be
    Hermitian (inversion is a Hermitian, unitary, involutive operator and the
    manifold is closed under it) and its eigenvalues must be ``+-1``; both are
    checked, because either failing means the manifold is not what it was taken
    to be -- most often that the band window cut through a degenerate group.
    """
    matrix = np.asarray(matrix)
    deviation = float(np.max(np.abs(matrix - matrix.conj().T)))
    if deviation > tolerance:
        raise ValueError(
            f"the parity matrix is not Hermitian (max |P - P^H| = {deviation:.2e}); "
            "the band window is not closed under inversion, or the k-point is "
            "not a TRIM"
        )
    values = np.linalg.eigvalsh((matrix + matrix.conj().T) / 2.0)
    off = float(np.max(np.abs(np.abs(values) - 1.0)))
    if off > tolerance:
        raise ValueError(
            f"a parity eigenvalue is {off:.2e} away from +-1; the occupied "
            "manifold is not an eigenspace of inversion. Check that the window "
            "ends at a gap and that the calculation really has an inversion "
            "centre"
        )
    return np.sort(np.sign(values))


def trim_delta(eigenvalues, require_kramers: bool = True) -> int:
    """``delta = (-1)^(N_- / 2)`` at one TRIM.

    ``require_kramers`` asserts what spin-orbit coupling guarantees: an even
    number of states, and an even number of odd-parity ones. Both fail when the
    window splits a Kramers pair, which is the one way to get a wrong answer
    here that still looks like an answer.
    """
    values = np.asarray(eigenvalues)
    negative = int(np.sum(values < 0))
    if require_kramers:
        if len(values) % 2:
            raise ValueError(
                f"{len(values)} occupied states at this TRIM is odd; with "
                "spin-orbit coupling every level is a Kramers doublet, so the "
                "window has cut one in half"
            )
        if negative % 2:
            raise ValueError(
                f"{negative} odd-parity states at this TRIM is odd; the two "
                "members of a Kramers doublet share a parity, so the window "
                "has cut one in half"
            )
    return 1 if (negative // 2) % 2 == 0 else -1


def fu_kane_z2(deltas: dict, dimension: int = 3) -> ParityInvariant:
    """Assemble the invariants from the delta at each TRIM.

    ``deltas`` maps a crystal k-point (a 3-tuple, components 0 or 1/2) to its
    ``delta``. Four entries in two dimensions, eight in three.
    """
    keys = {tuple(round(float(x) % 1.0, 6) for x in k): int(v) for k, v in deltas.items()}
    expected = 4 if dimension == 2 else 8
    if len(keys) != expected:
        raise ValueError(
            f"{dimension}D needs {expected} distinct TRIM, got {len(keys)}"
        )
    for key in keys:
        if not all(abs(x) < 1e-6 or abs(x - 0.5) < 1e-6 for x in key):
            raise ValueError(f"{key} is not a time-reversal-invariant momentum")

    total = int(np.prod(list(keys.values())))
    nu0 = 0 if total == 1 else 1
    if dimension == 2:
        return ParityInvariant(dimension=2, nu0=nu0, nu=None, deltas=keys)
    weak = []
    for axis in range(3):
        product = int(
            np.prod([v for k, v in keys.items() if abs(k[axis] - 0.5) < 1e-6])
        )
        weak.append(0 if product == 1 else 1)
    return ParityInvariant(
        dimension=3, nu0=nu0, nu=tuple(weak), deltas=keys
    )


def inversion_centre(symmetries) -> np.ndarray:
    """Where the inversion centre of a crystal is, in crystal coordinates.

    The space group's operations act as ``r -> r R + t``; for ``R = -1`` the
    fixed point is ``t / 2``. Any other choice differing by half a lattice
    vector is equally valid -- it flips individual deltas and leaves every
    product alone -- but it must be *a* centre, so a crystal without inversion
    is refused rather than defaulted to the origin.
    """
    rotations = symmetries.rotation_array()
    translations = symmetries.translation_array()
    for rotation, translation in zip(rotations, translations):
        if np.allclose(rotation, -np.eye(3), atol=1e-8):
            return np.asarray(translation, dtype=float) / 2.0
    raise ValueError(
        "this crystal has no inversion centre, so the Fu-Kane parity criterion "
        "does not apply; use the Wilson-loop method, which needs only "
        "time-reversal symmetry"
    )
