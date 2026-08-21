"""The tetrahedron method: Brillouin-zone integration without a smearing width.

Transcribed from ``PW/src/tetra.f90`` (Bloechl's method and Kawamura's linear
and optimised ones), ``PW/src/sumkt.f90``, ``PW/src/efermit.f90`` and
``PP/src/dos.f90``. The three variants QE offers map onto ``occupations`` as
``PW/src/set_occupations.f90`` decides -- read there rather than guessing, since
the names do not say which algorithm they select:

===================== ============ =========================================
``occupations``       ``tetra_type`` algorithm
===================== ============ =========================================
``tetrahedra``        0            Bloechl, PRB **49**, 16223 (1994), with
                                   his O(1/N^2) curvature correction
``tetrahedra-lin``    1            plain linear tetrahedra, no correction
``tetrahedra-opt``    2            Kawamura's optimised tetrahedra, PRB
                                   **89**, 094515 (2014): the corner energies
                                   are a 20-point stencil rather than the 4
                                   corners themselves
===================== ============ =========================================

The two families also disagree about **which** tetrahedra a microcell is cut
into. ``tetra_init`` (Bloechl) hardwires one decomposition; ``opt_tetra_init``
picks the shortest of the microcell's four body diagonals as the shaft, which is
what keeps the tetrahedra from becoming needles on an anisotropic grid. Using
one decomposition with the other's weights gives an answer that looks plausible
and is wrong in the third decimal of the Fermi energy.

Everything here is per spin channel: the eigenvalues come in as ``(nk, nbnd)``
and the weights come out the same shape. The spin degeneracy is read off the
k-point weights, which QE normalises to ``degspin`` (2 unpolarised, 1 per
channel otherwise) -- so ``nspin == 1 -> wg * 2`` at the end of every routine in
``tetra.f90`` is here just ``sum(wk)``.

What is *not* JAX: building the tetrahedra. That is integer index work on the
full grid, done once, and it produces the static ``(ntetra, nntetra)`` table the
compiled path only gathers through.
"""

from __future__ import annotations



import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from pypresso.config import DEFAULT_PRECISION, Precision
from pypresso.system.kpoints import grid_equivalence

__all__ = [
    "Tetrahedra",
    "TETRAHEDRON_KINDS",
    "tetrahedron_kind",
    "build_tetrahedra",
    "tetrahedra_for",
    "tetrahedron_occupations",
    "tetrahedron_occupations_spin",
    "tetrahedron_weights_at",
    "tetrahedron_fermi_level",
    "tetrahedron_dos",
    "tetrahedron_projected_dos",
    "PROJECTED_ENERGY_CHUNK",
    "integrated_states",
]


#: ``occupations`` keyword -> the name used here. QE's ``tetra_type``, spelled.
TETRAHEDRON_KINDS = {
    "tetrahedra": "bloechl",
    "tetrahedra_lin": "linear",
    "tetrahedra-lin": "linear",
    "tetrahedra_opt": "optimized",
    "tetrahedra-opt": "optimized",
}


def tetrahedron_kind(occupations: str) -> str:
    """Which algorithm an ``occupations`` keyword selects."""
    try:
        return TETRAHEDRON_KINDS[occupations.lower()]
    except KeyError as error:
        raise ValueError(
            f"unknown tetrahedron scheme {occupations!r}; "
            f"expected one of {sorted(TETRAHEDRON_KINDS)}"
        ) from error


#: Two energies closer than this are the same energy. ``opt_tetra_weights_only``
#: uses exactly this number to zero the ratios it would otherwise form 0/0 from.
_DEGENERATE = 1.0e-12

#: Bands within this of each other share their weight (``opt_tetra_weights_only``).
_BAND_DEGENERATE = 1.0e-6

#: Bisection steps for the Fermi level. As in ``scf/occupations.py``: 200
#: halvings put the bracket far below float64 resolution, so the result is the
#: exact root of the discretised count and does not depend on the number.
BISECTION_STEPS = 200


# --------------------------------------------------------------------------
# Construction: host-side integer work, done once per k-grid.
# --------------------------------------------------------------------------

#: Corner offsets of a microcell, in the order ``tetra_init`` calls n1..n8.
_CUBE = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
     [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
)

#: The six tetrahedra ``tetra_init`` cuts a microcell into, as 0-based indices
#: into :data:`_CUBE`. Bloechl's decomposition is fixed: it always runs along the
#: n1-n8 diagonal, which is why it is the variant that suffers on a grid whose
#: microcell is far from cubic.
_BLOECHL_TETRAHEDRA = np.array(
    [[1, 2, 3, 6], [2, 3, 4, 6], [1, 3, 5, 6],
     [3, 4, 6, 8], [3, 6, 7, 8], [3, 5, 6, 7]]
) - 1

#: Kawamura's 20-point stencil weights, ``wlsm`` in ``opt_tetra_init``. Rows are
#: the four interpolation targets, columns the 20 points; every row sums to
#: 1260/1260 = 1, which is what makes the integrated DOS a partition of unity.
_OPTIMIZED_WLSM = np.array([
    [1440,    0,   30,    0,  -38,    7,   17,  -28,  -56,    9,
      -46,    9,  -38,  -28,   17,    7,  -18,  -18,   12,  -18],
    [   0, 1440,    0,   30,  -28,  -38,    7,   17,    9,  -56,
        9,  -46,    7,  -38,  -28,   17,  -18,  -18,  -18,   12],
    [  30,    0, 1440,    0,   17,  -28,  -38,    7,  -46,    9,
      -56,    9,   17,    7,  -38,  -28,   12,  -18,  -18,  -18],
    [   0,   30,    0, 1440,    7,   17,  -28,  -38,    9,  -46,
        9,  -56,  -28,   17,    7,  -38,  -18,   12,  -18,  -18],
], dtype=float) / 1260.0


def _optimized_offsets(bg: np.ndarray, grid) -> np.ndarray:
    """``ivvec``: the 20 stencil points of each of the six tetrahedra.

    ``opt_tetra_init``. The first four points are the tetrahedron's own corners;
    the remaining sixteen surround it and are what turns the piecewise-linear
    interpolation into Kawamura's higher-order one.

    The shaft is the shortest of the microcell's four body diagonals. QE encodes
    the choice in a fourth, unused component of ``ivvec0``/``divvec``: selecting
    diagonal 4 (the all-plus one) leaves the first three components untouched and
    recovers the plain decomposition, while selecting diagonal ``ii < 4`` moves
    the origin to corner ``ii`` and reverses that axis.
    """
    nk = np.asarray(grid, dtype=float)
    bvec2 = np.asarray(bg, dtype=float) / nk[:, None]  # rows: b_i / nk_i
    bvec3 = np.array([
        -bvec2[0] + bvec2[1] + bvec2[2],
        bvec2[0] - bvec2[1] + bvec2[2],
        bvec2[0] + bvec2[1] - bvec2[2],
        bvec2[0] + bvec2[1] + bvec2[2],
    ])
    shaft = int(np.argmin(np.einsum("ij,ij->i", bvec3, bvec3)))

    origin = np.zeros(4, dtype=int)
    steps = np.eye(4, dtype=int)  # steps[:, i] is QE's divvec(1:4, i)
    origin[shaft] = 1
    steps[shaft, shaft] = -1

    corners = np.zeros((6, 20, 3), dtype=int)
    itet = 0
    for i1 in range(3):
        for i2 in range(3):
            if i2 == i1:
                continue
            for i3 in range(3):
                if i3 in (i1, i2):
                    continue
                corners[itet, 0] = origin[:3]
                corners[itet, 1] = corners[itet, 0] + steps[:3, i1]
                corners[itet, 2] = corners[itet, 1] + steps[:3, i2]
                corners[itet, 3] = corners[itet, 2] + steps[:3, i3]
                itet += 1
    assert itet == 6

    v = corners  # shorthand; v[:, :4] are the corners, in QE's 1-based order
    extra = [
        (0, 1), (1, 2), (2, 3), (3, 0),        # ivvec  5..8
        (0, 2), (1, 3), (2, 0), (3, 1),        # ivvec  9..12
        (0, 3), (1, 0), (2, 1), (3, 2),        # ivvec 13..16
    ]
    for n, (a, b) in enumerate(extra):
        v[:, 4 + n] = 2 * v[:, a] - v[:, b]
    triples = [(3, 0, 1), (0, 1, 2), (1, 2, 3), (2, 3, 0)]  # ivvec 17..20
    for n, (a, b, c) in enumerate(triples):
        v[:, 16 + n] = v[:, a] - v[:, b] + v[:, c]
    return v


class Tetrahedra(eqx.Module):
    """A Brillouin-zone tetrahedron decomposition, indexed into the IBZ.

    ``corners[t, n]`` is the irreducible k-point at the ``n``-th stencil point of
    tetrahedron ``t``, and ``wlsm[c, n]`` how much that point contributes to
    corner energy ``c``. For everything but the optimised method ``wlsm`` is the
    identity and there are only four stencil points per tetrahedron.
    """

    corners: jnp.ndarray  # (ntetra, nntetra), int
    wlsm: jnp.ndarray  # (4, nntetra)
    kind: str = eqx.field(static=True)

    @property
    def ntetra(self) -> int:
        return self.corners.shape[0]

    @property
    def nntetra(self) -> int:
        return self.corners.shape[1]


def build_tetrahedra(
    kind: str,
    grid: tuple[int, int, int],
    shift: tuple[int, int, int],
    rotations: np.ndarray,
    bg: np.ndarray,
    time_reversal: bool = True,
    precision: Precision = DEFAULT_PRECISION,
) -> Tetrahedra:
    """Cut the Monkhorst-Pack grid into tetrahedra and index them into the IBZ.

    Args:
        kind: ``"bloechl"``, ``"linear"`` or ``"optimized"``.
        grid, shift: the automatic k-grid the calculation runs on.
        rotations: the crystal's symmetries in crystal axes, the same ones the
            wedge was reduced with -- the tetrahedra corners are looked up in
            *that* reduced list.
        bg: reciprocal lattice vectors as rows; only their relative lengths
            matter, and only for the optimised/linear shaft choice.
    """
    nk1, nk2, nk3 = (int(n) for n in grid)
    equiv = grid_equivalence(grid, shift, rotations, time_reversal)

    if kind == "bloechl":
        offsets = np.broadcast_to(_CUBE[_BLOECHL_TETRAHEDRA], (6, 4, 3))
        wlsm = np.eye(4)
    elif kind in ("linear", "optimized"):
        offsets = _optimized_offsets(bg, (nk1, nk2, nk3))
        wlsm = np.eye(4) if kind == "linear" else _OPTIMIZED_WLSM
        offsets = offsets[:, : wlsm.shape[1]]
    else:  # pragma: no cover - guarded by tetrahedron_kind
        raise ValueError(f"unknown tetrahedron kind {kind!r}")

    # The microcell loop is (i, j, k) with k fastest, matching monkhorst_pack's
    # ordering, so `equiv` can be indexed directly by the flattened corner.
    i, j, k = np.meshgrid(np.arange(nk1), np.arange(nk2), np.arange(nk3), indexing="ij")
    base = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1)  # (ncell, 3)
    counts = np.array([nk1, nk2, nk3])
    # (ncell, 6, nn, 3) -> flat grid index, periodic in every direction
    ikv = (base[:, None, None, :] + offsets[None, :, :, :]) % counts
    flat = (ikv[..., 0] * nk2 + ikv[..., 1]) * nk3 + ikv[..., 2]
    corners = equiv[flat].reshape(-1, offsets.shape[1])

    return Tetrahedra(
        corners=jnp.asarray(corners, dtype=jnp.int32),
        wlsm=precision.as_real(wlsm),
        kind=kind,
    )


def tetrahedra_for(occupations, kpoints, symmetries, cell) -> Tetrahedra:
    """The tetrahedra of a calculation, from the objects a driver already holds.

    Refuses an explicit k-point list the way ``PP/src/dos.f90`` does: the method
    needs the grid the points came from, and there is no way to recover one.
    """
    if kpoints.grid is None:
        raise ValueError(
            "the tetrahedron method needs an automatic k-point grid "
            "(K_POINTS automatic); an explicit list carries no tetrahedra"
        )
    return build_tetrahedra(
        tetrahedron_kind(occupations),
        kpoints.grid,
        kpoints.shift or (0, 0, 0),
        symmetries.rotation_array(),
        np.asarray(cell.bg_2pi_alat),
        precision=kpoints.precision,
    )


# --------------------------------------------------------------------------
# The integrated density of states, N(E). Everything else follows from it.
# --------------------------------------------------------------------------


def _positive(x):
    """``x``, floored away from zero *before* anything divides by it.

    Corner energies are degenerate all the time -- at a high-symmetry point, in
    a flat band, whenever two grid points are related by a symmetry that the
    wedge reduction left in place. QE never notices, because its ``IF/ELSEIF``
    chain only ever evaluates the selected branch and the selected branch's
    denominators are provably nonzero (a branch spanning ``e_i <= E < e_j``
    is empty exactly when ``e_i == e_j``). Here every branch is evaluated and
    then selected, so the *dead* branches divide by zero; the forward value
    survives the ``where`` but ``grad`` does not -- ``where`` hands the dead
    branch a zero cotangent and ``0 * inf`` is NaN. Clamping the denominator
    before the division, rather than masking the quotient afterwards, is the
    fix, and it is exact because the clamp never fires on a live branch.
    """
    return jnp.where(x > _DEGENERATE, x, _DEGENERATE)


def _occupied_fraction(e: jnp.ndarray, energy) -> jnp.ndarray:
    """The fraction of one tetrahedron lying below ``energy``.

    ``e`` is ``(..., 4)``, sorted ascending. This is ``sumkt``'s integrand and,
    identically, the ``dosint`` of both ``tetra_dos_t`` and ``opt_tetra_dos_t``:
    the piecewise cubic that linear interpolation of the band inside a
    tetrahedron implies.

    Only this is written down. The density of states is ``jax.grad`` of it --
    QE hand-codes ``dost`` beside ``dosint``, four matched branches each, and
    the two are exact derivatives of one another (on ``e3 <= E < e4``,
    ``d/dE [1 - (e4-E)^3/((e4-e1)(e4-e2)(e4-e3))]`` is QE's ``dost`` line for
    line). Differentiating instead of transcribing halves the surface for a sign
    slip and makes the sum rule ``int D = N`` hold by construction. Same pattern
    as ``xc/functional.py``, where only the energy is written down.
    """
    e1, e2, e3, e4 = (e[..., i] for i in range(4))

    below4 = e4 - energy
    above1 = energy - e1
    above2 = energy - e2

    # e1 <= E < e2
    first = above1**3 / _positive((e2 - e1) * (e3 - e1) * (e4 - e1))
    # e2 <= E < e3
    second = (
        (e2 - e1) ** 2
        + 3.0 * (e2 - e1) * above2
        + 3.0 * above2**2
        - (e3 - e1 + e4 - e2) / _positive((e3 - e2) * (e4 - e2)) * above2**3
    ) / _positive((e3 - e1) * (e4 - e1))
    # e3 <= E < e4
    third = 1.0 - below4**3 / _positive((e4 - e1) * (e4 - e2) * (e4 - e3))

    fraction = jnp.where(energy < e2, first, jnp.where(energy < e3, second, third))
    fraction = jnp.where(energy < e1, jnp.zeros_like(fraction), fraction)
    return jnp.where(energy >= e4, jnp.ones_like(fraction), fraction)


def _corner_energies(tetra: Tetrahedra, eigenvalues: jnp.ndarray) -> jnp.ndarray:
    """``(ntetra, nbnd, 4)``: the four interpolation energies of each tetrahedron.

    For the linear and Bloechl methods these are the corner eigenvalues; for the
    optimised method they are Kawamura's 20-point combinations of them.
    """
    gathered = eigenvalues[tetra.corners]  # (ntetra, nntetra, nbnd)
    return jnp.einsum("cn,tnb->tbc", tetra.wlsm, gathered)


def _sorted_corners(tetra: Tetrahedra, eigenvalues: jnp.ndarray):
    """Corner energies sorted ascending, with the permutation that did it.

    QE calls ``hpsort`` per (tetrahedron, band); here it is one ``argsort`` over
    a length-4 axis, vectorised over both. Ties need no stable sort: every
    formula below is symmetric under exchanging two equal corner energies.
    """
    e = _corner_energies(tetra, eigenvalues)
    order = jnp.argsort(e, axis=-1)
    return jnp.take_along_axis(e, order, axis=-1), order


def integrated_states(e_sorted: jnp.ndarray, energy, ntetra: int, spin) -> jnp.ndarray:
    """``sumkt``: the number of electrons below ``energy``."""
    return spin / ntetra * jnp.sum(_occupied_fraction(e_sorted, energy))


# --------------------------------------------------------------------------
# The Fermi level.
# --------------------------------------------------------------------------


def _bisect(e_sorted, ntetra: int, nelec, spin):
    """``efermit``: bisection on ``sumkt(E) = nelec``, as a device-side loop.

    ``N(E)`` is non-decreasing, so bisection is exact; the loop runs a fixed
    number of steps with a ``where`` for its branch and therefore never leaves
    the device. This mirrors ``scf/occupations.py``'s ``bisect_fermi``, which
    cannot be reused directly because its count function is ``wgauss`` -- the
    reason for repeating the loop is the count, not the search.
    """
    low = jnp.min(e_sorted)
    high = jnp.max(e_sorted)

    def step(_, bracket):
        low, high = bracket
        middle = 0.5 * (low + high)
        too_many = integrated_states(e_sorted, middle, ntetra, spin) > nelec
        return jnp.where(too_many, low, middle), jnp.where(too_many, middle, high)

    low, high = jax.lax.fori_loop(0, BISECTION_STEPS, step, (low, high))
    return 0.5 * (low + high)


# --------------------------------------------------------------------------
# Occupation weights.
# --------------------------------------------------------------------------


def _bloechl_weights(e: jnp.ndarray, ef) -> jnp.ndarray:
    """``tetra_weights_only``: Bloechl's weights with his curvature correction.

    ``e`` is ``(..., 4)`` sorted ascending; the result is ``(..., 4)``, the
    contribution of this tetrahedron to each of its four corners, still to be
    divided by ``ntetra``.

    The correction is the ``dosef * (e1+e2+e3+e4 - 4 e_i) / 40`` term. It is what
    makes the method converge as ``1/N^2`` instead of ``1/N``, and it is why the
    weights are not simply the derivative of the integrated DOS with respect to
    a corner energy.
    """
    e1, e2, e3, e4 = (e[..., i] for i in range(4))
    total = e1 + e2 + e3 + e4
    correction = jnp.stack([total - 4.0 * e[..., i] for i in range(4)], axis=-1) / 40.0

    full = jnp.broadcast_to(jnp.asarray(0.25, e.dtype), e.shape)

    # e3 <= ef < e4
    d4 = _positive((e4 - e1) * (e4 - e2) * (e4 - e3))
    c4 = 0.25 * (e4 - ef) ** 3 / d4
    dos_third = 3.0 * (e4 - ef) ** 2 / d4
    inv = jnp.stack(
        [1.0 / _positive(e4 - e1), 1.0 / _positive(e4 - e2), 1.0 / _positive(e4 - e3)],
        axis=-1,
    )
    third = jnp.stack(
        [
            0.25 - c4 * (e4 - ef) * inv[..., 0],
            0.25 - c4 * (e4 - ef) * inv[..., 1],
            0.25 - c4 * (e4 - ef) * inv[..., 2],
            0.25 - c4 * (4.0 - (e4 - ef) * jnp.sum(inv, axis=-1)),
        ],
        axis=-1,
    ) + dos_third[..., None] * correction

    # e2 <= ef < e3
    c1 = 0.25 * (ef - e1) ** 2 / _positive((e4 - e1) * (e3 - e1))
    c2 = 0.25 * (ef - e1) * (ef - e2) * (e3 - ef) / _positive(
        (e4 - e1) * (e3 - e2) * (e3 - e1)
    )
    c3 = 0.25 * (ef - e2) ** 2 * (e4 - ef) / _positive((e4 - e2) * (e3 - e2) * (e4 - e1))
    dos_second = (
        3.0 * (e2 - e1)
        + 6.0 * (ef - e2)
        - 3.0 * (e3 - e1 + e4 - e2) * (ef - e2) ** 2 / _positive((e3 - e2) * (e4 - e2))
    ) / _positive((e3 - e1) * (e4 - e1))
    second = jnp.stack(
        [
            c1
            + (c1 + c2) * (e3 - ef) / _positive(e3 - e1)
            + (c1 + c2 + c3) * (e4 - ef) / _positive(e4 - e1),
            c1
            + c2
            + c3
            + (c2 + c3) * (e3 - ef) / _positive(e3 - e2)
            + c3 * (e4 - ef) / _positive(e4 - e2),
            (c1 + c2) * (ef - e1) / _positive(e3 - e1)
            + (c2 + c3) * (ef - e2) / _positive(e3 - e2),
            (c1 + c2 + c3) * (ef - e1) / _positive(e4 - e1)
            + c3 * (ef - e2) / _positive(e4 - e2),
        ],
        axis=-1,
    ) + dos_second[..., None] * correction

    # e1 <= ef < e2
    d1 = _positive((e2 - e1) * (e3 - e1) * (e4 - e1))
    c4first = 0.25 * (ef - e1) ** 3 / d1
    dos_first = 3.0 * (ef - e1) ** 2 / d1
    inv1 = jnp.stack(
        [1.0 / _positive(e2 - e1), 1.0 / _positive(e3 - e1), 1.0 / _positive(e4 - e1)],
        axis=-1,
    )
    first = jnp.stack(
        [
            c4first * (4.0 - (ef - e1) * jnp.sum(inv1, axis=-1)),
            c4first * (ef - e1) * inv1[..., 0],
            c4first * (ef - e1) * inv1[..., 1],
            c4first * (ef - e1) * inv1[..., 2],
        ],
        axis=-1,
    ) + dos_first[..., None] * correction

    empty = jnp.zeros_like(full)
    weights = jnp.where(
        (ef < e2)[..., None],
        first,
        jnp.where((ef < e3)[..., None], second, third),
    )
    weights = jnp.where((ef < e1)[..., None], empty, weights)
    return jnp.where((ef >= e4)[..., None], full, weights)


def _linear_weights(e: jnp.ndarray, ef) -> jnp.ndarray:
    """``opt_tetra_weights_only``'s ``wg0``: weights with no curvature correction.

    Written in Kawamura's ratio form ``a_ij = (E_F - e_j) / (e_i - e_j)``, whose
    own degeneracy guard -- ``a_ij = 0`` when the two energies coincide -- is the
    same clamp the rest of this module uses, and here it is QE's own.
    """
    delta = e[..., :, None] - e[..., None, :]  # a[i, j] denominator e_i - e_j
    degenerate = jnp.abs(delta) < _DEGENERATE
    a = jnp.where(
        degenerate, 0.0, (ef - e[..., None, :]) / jnp.where(degenerate, 1.0, delta)
    )

    def at(i, j):
        return a[..., i - 1, j - 1]

    # e1 <= ef < e2
    c = at(2, 1) * at(3, 1) * at(4, 1) * 0.25
    first = jnp.stack(
        [
            c * (1.0 + at(1, 2) + at(1, 3) + at(1, 4)),
            c * at(2, 1),
            c * at(3, 1),
            c * at(4, 1),
        ],
        axis=-1,
    )

    # e2 <= ef < e3
    c1 = at(4, 1) * at(3, 1) * 0.25
    c2 = at(4, 1) * at(3, 2) * at(1, 3) * 0.25
    c3 = at(4, 2) * at(3, 2) * at(1, 4) * 0.25
    second = jnp.stack(
        [
            c1 + (c1 + c2) * at(1, 3) + (c1 + c2 + c3) * at(1, 4),
            c1 + c2 + c3 + (c2 + c3) * at(2, 3) + c3 * at(2, 4),
            (c1 + c2) * at(3, 1) + (c2 + c3) * at(3, 2),
            (c1 + c2 + c3) * at(4, 1) + c3 * at(4, 2),
        ],
        axis=-1,
    )

    # e3 <= ef < e4
    c = at(1, 4) * at(2, 4) * at(3, 4)
    third = 0.25 * jnp.stack(
        [
            1.0 - c * at(1, 4),
            1.0 - c * at(2, 4),
            1.0 - c * at(3, 4),
            1.0 - c * (1.0 + at(4, 1) + at(4, 2) + at(4, 3)),
        ],
        axis=-1,
    )

    e1, e2, e3, e4 = (e[..., i] for i in range(4))
    full = jnp.broadcast_to(jnp.asarray(0.25, e.dtype), e.shape)
    weights = jnp.where(
        (ef < e2)[..., None], first, jnp.where((ef < e3)[..., None], second, third)
    )
    weights = jnp.where((ef < e1)[..., None], jnp.zeros_like(full), weights)
    return jnp.where((ef >= e4)[..., None], full, weights)


def _average_degenerate(wg: jnp.ndarray, eigenvalues: jnp.ndarray) -> jnp.ndarray:
    """Share the weight of degenerate bands equally (``opt_tetra_weights_only``).

    Kawamura's weights are not symmetric between two bands that cross inside a
    tetrahedron, so QE averages over each degenerate group afterwards. Its own
    version is a sequential scan that compares every band to the *first* of the
    group it is building; this one is the symmetric equivalent -- each band takes
    the mean over every band within ``1e-6`` Ry of it. The two agree whenever the
    relation is transitive, which at that tolerance it is unless a band structure
    is degenerate in a chain, and the operation is weight-preserving either way,
    so nothing downstream can see the difference.
    """
    same = jnp.abs(eigenvalues[:, :, None] - eigenvalues[:, None, :]) < _BAND_DEGENERATE
    same = same.astype(wg.dtype)
    return jnp.einsum("kij,kj->ki", same, wg) / jnp.sum(same, axis=-1)


def _scatter(tetra: Tetrahedra, order, contributions, nk: int, nbnd: int):
    """Add each tetrahedron's corner weights back onto the irreducible k-points.

    ``contributions`` is ``(ntetra, nbnd, 4)`` in *sorted* corner order. For the
    optimised method the four sorted corners are spread back over all twenty
    stencil points through ``wlsm``, exactly as QE's
    ``DOT_PRODUCT(wlsm(itetra(1:4), ii), wg0(1:4))`` does -- the permutation
    indexes ``wlsm``'s first axis, not its second, and getting that backwards is
    the one transcription error here that still produces a plausible number.
    """
    if tetra.kind == "optimized":
        spread = jnp.einsum("tbjn,tbj->tbn", tetra.wlsm[order], contributions)
        targets = jnp.broadcast_to(tetra.corners[:, None, :], spread.shape)
    else:
        spread = contributions
        targets = jnp.take_along_axis(
            jnp.broadcast_to(tetra.corners[:, None, :], order.shape), order, axis=-1
        )
    bands = jnp.broadcast_to(jnp.arange(nbnd)[None, :, None], spread.shape)
    return jnp.zeros((nk, nbnd), dtype=spread.dtype).at[targets, bands].add(spread)


@jax.jit
def tetrahedron_weights_at(
    tetra: Tetrahedra, eigenvalues: jnp.ndarray, weights: jnp.ndarray, ef
):
    """``tetra_weights_only``: the weights of one channel at a *given* ``ef``.

    Split out of :func:`tetrahedron_occupations` because that is how
    ``tetra.f90`` splits it, and for the same reason: with two spin channels the
    Fermi level is found once from both of them together and then handed to each
    channel separately (``tetra_weights`` calls ``efermit`` and then
    ``tetra_weights_only``). Solving per channel would be a different physical
    problem -- see :func:`tetrahedron_occupations_spin`.
    """
    nk, nbnd = eigenvalues.shape
    spin = jnp.sum(weights)
    e_sorted, order = _sorted_corners(tetra, eigenvalues)

    if tetra.kind == "bloechl":
        contributions = _bloechl_weights(e_sorted, ef)
    else:
        contributions = _linear_weights(e_sorted, ef)
    wg = _scatter(tetra, order, contributions, nk, nbnd) * (spin / tetra.ntetra)

    if tetra.kind != "bloechl":
        wg = _average_degenerate(wg, eigenvalues)
    return wg


@jax.jit
def tetrahedron_occupations(
    tetra: Tetrahedra, eigenvalues: jnp.ndarray, weights: jnp.ndarray, nelec
):
    """Occupation weights ``wg`` and the Fermi level, per spin channel.

    ``eigenvalues`` is ``(nk, nbnd)`` on the irreducible k-points, ``weights``
    their integration weights -- used only for their *sum*, which is the spin
    degeneracy: the tetrahedra already carry the Brillouin-zone measure, which
    is why ``tetra.f90`` never looks at ``wk``.
    """
    spin = jnp.sum(weights)
    e_sorted, _ = _sorted_corners(tetra, eigenvalues)
    ef = _bisect(e_sorted, tetra.ntetra, nelec, spin)
    return tetrahedron_weights_at(tetra, eigenvalues, weights, ef), ef


# --------------------------------------------------------------------------
# Two spin channels.
# --------------------------------------------------------------------------
#
# Everything above is per channel, and deliberately so: nothing in the
# tetrahedron construction or in the weight formulas knows about spin. What spin
# changes is *how many Fermi levels there are*, and that is decided here.
#
# ``sumkt`` is where to read it off. With ``is = 0`` and ``nspin = 2`` it loops
# over both channels accumulating ``1/ntetra`` per (tetrahedron, band) from
# each, and applies its factor of two **only** when ``nspin == 1``:
#
#     IF ( nspin == 1 ) sumkt = sumkt * 2.0_DP
#
# So the count whose root is the Fermi level is the *sum over channels*, with
# each channel weighted by one. That is not the same as solving each channel for
# half the electrons: a magnetic metal moves electrons between the channels
# until one number, not two, is stationary, and the whole magnetization comes
# out of the imbalance that a single shared level produces.
#
# The degeneracy bookkeeping needs no special case, because
# :func:`tetrahedron_occupations` already reads it off ``sum(weights)``: an
# unpolarized run's k-point weights sum to 2 and a polarized channel's to 1,
# which is exactly the factor ``sumkt`` applies and withholds.


def _stacked_corners(tetra: Tetrahedra, eigenvalues: jnp.ndarray) -> jnp.ndarray:
    """Sorted corner energies of every channel, as one array.

    ``integrated_states`` sums over *all* axes of what it is given and ``_bisect``
    brackets with a global min and max, so stacking the channels turns the
    per-channel count into the summed one with no arithmetic of its own. That is
    the whole of ``sumkt``'s ``DO ns = 1, nspin_lsda`` loop.
    """
    return jnp.stack([_sorted_corners(tetra, channel)[0] for channel in eigenvalues])


def tetrahedron_occupations_spin(
    tetra: Tetrahedra,
    eigenvalues: jnp.ndarray,
    weights: jnp.ndarray,
    nelec,
    counts=None,
):
    """``wg`` and the Fermi level(s) for ``(nspin, nk, nbnd)`` eigenvalues.

    Args:
        eigenvalues: ``(nspin, nk, nbnd)``.
        weights: ``(nk,)`` k-point weights, shared by the channels; their sum is
            the spin degeneracy (2 unpolarized, 1 per channel polarized).
        counts: ``(nelup, neldw)`` to constrain the magnetization, or ``None``
            for the single shared Fermi level of an unconstrained run.

    Returns ``(wg, ef)`` with ``wg`` the shape of ``eigenvalues`` and ``ef``
    either a scalar or, when constrained, a pair.

    ``weights.f90`` supports both, and which one it takes is ``two_fermi_energies``:
    unconstrained it calls ``tetra_weights(..., nelec, ef, wg, 0, isk)``, which
    finds one level from both channels; constrained it calls the same routine
    twice with ``nelup``/``neldw`` and ``is = 1``/``is = 2``, which is genuinely
    two independent problems. So a constrained magnetization is *not* refused
    here -- it is the second branch, and it is the one place where solving each
    channel separately is right.
    """
    eigenvalues = jnp.asarray(eigenvalues)
    weights = jnp.asarray(weights)

    if counts is not None:
        solved = [
            tetrahedron_occupations(tetra, eigenvalues[spin], weights, counts[spin])
            for spin in range(eigenvalues.shape[0])
        ]
        return jnp.stack([wg for wg, _ in solved]), tuple(ef for _, ef in solved)

    ef = _bisect(
        _stacked_corners(tetra, eigenvalues),
        tetra.ntetra,
        nelec,
        jnp.sum(weights),
    )
    wg = jnp.stack([
        tetrahedron_weights_at(tetra, channel, weights, ef) for channel in eigenvalues
    ])
    return wg, ef


def tetrahedron_fermi_level(
    tetra: Tetrahedra, eigenvalues: jnp.ndarray, weights: jnp.ndarray, nelec
) -> jnp.ndarray:
    """The Fermi level alone, without building the weights."""
    e_sorted, _ = _sorted_corners(tetra, eigenvalues)
    return _bisect(e_sorted, tetra.ntetra, nelec, jnp.sum(weights))


# --------------------------------------------------------------------------
# The density of states.
# --------------------------------------------------------------------------

#: How many energy points to evaluate at once. A 20^3 grid is 48000 tetrahedra;
#: a ``(nE, ntetra, nbnd)`` intermediate at ``nE = 1000`` and ``nbnd = 8`` is
#: 3 GB before the four branches are counted, so the energy axis is mapped over
#: in blocks rather than broadcast.
ENERGY_CHUNK = 32


def tetrahedron_dos(
    tetra: Tetrahedra,
    eigenvalues: jnp.ndarray,
    weights: jnp.ndarray,
    energies: jnp.ndarray,
    chunk: int = ENERGY_CHUNK,
):
    """``(dos, integrated)`` on an energy grid, in states/Ry and states.

    ``tetra_dos_t`` / ``opt_tetra_dos_t``, except that only ``dosint`` is
    transcribed and ``dost`` is its derivative -- see :func:`_occupied_fraction`.
    """
    energies = jnp.asarray(energies)
    e_sorted, _ = _sorted_corners(tetra, eigenvalues)
    spin = jnp.sum(weights)
    ntetra = tetra.ntetra

    def at(energy):
        return jax.value_and_grad(
            lambda e: integrated_states(e_sorted, e, ntetra, spin)
        )(energy)

    n = energies.shape[0]
    chunk = max(1, min(int(chunk), n))
    padded = jnp.concatenate(
        [energies, jnp.full((-n) % chunk, energies[-1], energies.dtype)]
    )
    integrated, dos = jax.lax.map(jax.vmap(at), padded.reshape(-1, chunk))
    return dos.reshape(-1)[:n], integrated.reshape(-1)[:n]


#: The projected version carries a ``(chunk, ntetra, nbnd, 4)`` intermediate --
#: four times :func:`tetrahedron_dos`'s, because the corner weights are kept
#: separately instead of being summed into one fraction. The chunk is smaller by
#: the same factor so the working set is not.
PROJECTED_ENERGY_CHUNK = 8


def tetrahedron_projected_dos(
    tetra: Tetrahedra,
    eigenvalues: jnp.ndarray,
    weights: jnp.ndarray,
    projections: jnp.ndarray,
    energies: jnp.ndarray,
    chunk: int = PROJECTED_ENERGY_CHUNK,
):
    """``(pdos, integrated)``, both ``(nE, nproj)``, in states/Ry and states.

    ``opt_tetra_partialdos``. ``projections[k, b, p]`` is the weight band ``b``
    at k-point ``k`` carries in channel ``p`` -- ``|<phi_p|S|psi_kb>|^2`` for a
    projected density of states, but nothing here knows that.

    Only the *integral* is written down, as everywhere else in this module:

        N_p(E) = sum_kb w_kb(E) proj[k, b, p]

    with ``w_kb(E)`` the occupation weights :func:`tetrahedron_weights_at` would
    give for a Fermi level at ``E``, and ``D_p(E)`` is its derivative. With
    ``proj = 1`` that is ``integrated_states`` identically, so summing the
    projected density of states over a *complete* set of channels reproduces the
    total one to round-off rather than approximately -- which is the sum rule,
    and it is a property of this construction rather than a coincidence to test.

    Two departures from :func:`tetrahedron_weights_at`, both QE's:

    * **the degenerate-band average is not applied.** ``opt_tetra_partialdos``
      has no counterpart of ``opt_tetra_weights_only``'s averaging loop, and
      applying it here would move weight between two crossing bands that carry
      *different* projections, which is a change to the answer rather than a
      symmetrisation of it.
    * **Bloechl's correction is refused.** ``do_projwfc`` silently promotes
      ``tetra_type = 0`` to 1, i.e. runs the linear method for the projected
      density of states whatever the SCF used, and the two families do not even
      cut a microcell into the same tetrahedra -- so the substitution has to
      happen where the tetrahedra are *built*, not here.
    """
    if tetra.kind == "bloechl":
        raise ValueError(
            "a projected density of states cannot use Bloechl's corrected "
            "weights: their curvature term is not the derivative of an "
            "occupation, and do_projwfc substitutes the linear method for it "
            "(tetra_type 0 -> 1). Build the tetrahedra with kind='linear'"
        )
    energies = jnp.asarray(energies)
    projections = jnp.asarray(projections)
    nk, nbnd = eigenvalues.shape
    e_sorted, order = _sorted_corners(tetra, eigenvalues)
    scale = jnp.sum(weights) / tetra.ntetra

    def projected(energy):
        contributions = _linear_weights(e_sorted, energy)
        wg = _scatter(tetra, order, contributions, nk, nbnd) * scale
        return jnp.einsum("kb,kbp->p", wg, projections)

    def at(energy):
        return jax.jvp(projected, (energy,), (jnp.ones_like(energy),))

    n = energies.shape[0]
    chunk = max(1, min(int(chunk), n))
    padded = jnp.concatenate(
        [energies, jnp.full((-n) % chunk, energies[-1], energies.dtype)]
    )
    integrated, dos = jax.lax.map(jax.vmap(at), padded.reshape(-1, chunk))
    nproj = projections.shape[-1]
    return dos.reshape(-1, nproj)[:n], integrated.reshape(-1, nproj)[:n]
