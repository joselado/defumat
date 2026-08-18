"""Bravais lattices and the unit cell.

``latgen`` is a transcription of the reference ``Modules/latgen.f90``. The
lattice vectors it produces are not merely *a* valid basis for each Bravais
lattice -- they are QE's particular choice, including sign and ordering
conventions, because every later comparison (G-vector ordering, symmetry
operations, k-point coordinates) depends on matching them exactly.

Units: ``latgen`` works in bohr, as QE does. :class:`Cell` stores ``at`` in bohr
and exposes the ``alat``-scaled views QE prints, so a test can compare against an
output header without unit guesswork.

:class:`Cell` holds JAX arrays rather than NumPy ones even though it is built
during setup: the cell is what stress differentiates with respect to (a strain
derivative of the total energy), so it has to sit on the traced side.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from pypresso.config import DEFAULT_PRECISION, Precision
from pypresso.units import ANGSTROM_TO_BOHR, TPI

__all__ = ["Cell", "latgen", "celldm_from_abc", "IBRAV_NAMES"]

_SR2 = math.sqrt(2.0)
_SR3 = math.sqrt(3.0)

#: ibrav -> description, from the table at the top of ``latgen.f90``.
IBRAV_NAMES = {
    0: "free (lattice vectors given explicitly)",
    1: "cubic P (sc)",
    2: "cubic F (fcc)",
    3: "cubic I (bcc)",
    -3: "cubic I (bcc), symmetric axes",
    4: "hexagonal / trigonal P",
    5: "trigonal R, 3-fold axis c",
    -5: "trigonal R, 3-fold axis (111)",
    6: "tetragonal P (st)",
    7: "tetragonal I (bct)",
    8: "orthorhombic P",
    9: "base-centred orthorhombic (C type)",
    -9: "base-centred orthorhombic (C type), alternate",
    91: "base-centred orthorhombic (A type)",
    10: "face-centred orthorhombic",
    11: "body-centred orthorhombic",
    12: "monoclinic P, unique axis c",
    -12: "monoclinic P, unique axis b",
    13: "base-centred monoclinic, unique axis c",
    -13: "base-centred monoclinic, unique axis b",
    14: "triclinic P",
}


def latgen(ibrav: int, celldm) -> np.ndarray:
    """Lattice vectors for a Bravais lattice index, in bohr.

    Args:
        ibrav: QE's Bravais lattice index (see :data:`IBRAV_NAMES`).
        celldm: the six cell parameters, 1-based in QE's numbering, passed here
            as a length-6 sequence: ``celldm(1)`` is ``celldm[0]``.

    Returns:
        ``(3, 3)`` array whose *rows* are ``a1, a2, a3`` in bohr. Note QE stores
        them as columns of ``at``; rows are used throughout pypresso so that
        ``at[i]`` is a vector.
    """
    celldm = np.asarray(celldm, dtype=float)
    if celldm.shape != (6,):
        raise ValueError(f"celldm must have 6 entries, got shape {celldm.shape}")
    if ibrav != 0 and celldm[0] <= 0.0:
        raise ValueError(f"ibrav={ibrav} requires celldm(1) > 0, got {celldm[0]}")
    if ibrav not in IBRAV_NAMES:
        raise ValueError(f"nonexistent bravais lattice ibrav={ibrav}")

    a = celldm[0]
    at = np.zeros((3, 3))

    def _require(index: int, condition: bool) -> None:
        if not condition:
            raise ValueError(f"ibrav={ibrav}: wrong celldm({index}) = {celldm[index - 1]}")

    if ibrav == 0:
        raise ValueError("ibrav=0 takes explicit lattice vectors; use Cell.from_vectors")

    if ibrav == 1:  # simple cubic
        at = np.diag([a, a, a])

    elif ibrav == 2:  # fcc
        term = a / 2.0
        at = np.array([[-term, 0.0, term], [0.0, term, term], [-term, term, 0.0]])

    elif abs(ibrav) == 3:  # bcc
        term = a / 2.0
        at = np.full((3, 3), term)
        if ibrav < 0:  # more symmetric choice of axes
            at[0, 0], at[1, 1], at[2, 2] = -term, -term, -term
        else:
            at[1, 0], at[2, 0], at[2, 1] = -term, -term, -term

    elif ibrav == 4:  # hexagonal
        _require(3, celldm[2] > 0.0)
        at = np.array([[a, 0.0, 0.0], [-a / 2.0, a * _SR3 / 2.0, 0.0], [0.0, 0.0, a * celldm[2]]])

    elif abs(ibrav) == 5:  # trigonal R
        _require(4, -0.5 < celldm[3] < 1.0)
        term1 = math.sqrt(1.0 + 2.0 * celldm[3])
        term2 = math.sqrt(1.0 - celldm[3])
        if ibrav == 5:  # 3-fold axis along c
            a1x = a * term2 / _SR2
            a3z = a * term1 / _SR3
            at = np.array(
                [
                    [a1x, -a1x / _SR3, a3z],
                    [0.0, _SR2 * a * term2 / _SR3, a3z],
                    [-a1x, -a1x / _SR3, a3z],
                ]
            )
        else:  # 3-fold axis along (111)
            u = a * (term1 - 2.0 * term2) / 3.0
            v = a * (term1 + term2) / 3.0
            at = np.array([[u, v, v], [v, u, v], [v, v, u]])

    elif ibrav == 6:  # simple tetragonal
        _require(3, celldm[2] > 0.0)
        at = np.diag([a, a, a * celldm[2]])

    elif ibrav == 7:  # body-centred tetragonal
        _require(3, celldm[2] > 0.0)
        half, cz = a / 2.0, celldm[2] * a / 2.0
        at = np.array([[half, -half, cz], [half, half, cz], [-half, -half, cz]])

    elif ibrav == 8:  # simple orthorhombic
        _require(2, celldm[1] > 0.0)
        _require(3, celldm[2] > 0.0)
        at = np.diag([a, a * celldm[1], a * celldm[2]])

    elif abs(ibrav) == 9:  # base-centred orthorhombic, C type
        _require(2, celldm[1] > 0.0)
        _require(3, celldm[2] > 0.0)
        half = 0.5 * a
        if ibrav == 9:  # historical PWscf description
            at = np.array(
                [[half, half * celldm[1], 0.0], [-half, half * celldm[1], 0.0], [0.0, 0.0, a * celldm[2]]]
            )
        else:  # alternate description
            at = np.array(
                [[half, -half * celldm[1], 0.0], [half, half * celldm[1], 0.0], [0.0, 0.0, a * celldm[2]]]
            )

    elif ibrav == 91:  # base-centred orthorhombic, A type
        _require(2, celldm[1] > 0.0)
        _require(3, celldm[2] > 0.0)
        by, bz = a * celldm[1] * 0.5, -a * celldm[2] * 0.5
        at = np.array([[a, 0.0, 0.0], [0.0, by, bz], [0.0, by, -bz]])

    elif ibrav == 10:  # face-centred orthorhombic
        _require(2, celldm[1] > 0.0)
        _require(3, celldm[2] > 0.0)
        half = 0.5 * a
        at = np.array(
            [
                [half, 0.0, half * celldm[2]],
                [half, half * celldm[1], 0.0],
                [0.0, half * celldm[1], half * celldm[2]],
            ]
        )

    elif ibrav == 11:  # body-centred orthorhombic
        _require(2, celldm[1] > 0.0)
        _require(3, celldm[2] > 0.0)
        x, y, z = 0.5 * a, 0.5 * a * celldm[1], 0.5 * a * celldm[2]
        at = np.array([[x, y, z], [-x, y, z], [-x, -y, z]])

    elif ibrav == 12:  # monoclinic P, unique axis c
        _require(2, celldm[1] > 0.0)
        _require(3, celldm[2] > 0.0)
        _require(4, abs(celldm[3]) < 1.0)
        sen = math.sqrt(1.0 - celldm[3] ** 2)
        at = np.array(
            [[a, 0.0, 0.0], [a * celldm[1] * celldm[3], a * celldm[1] * sen, 0.0], [0.0, 0.0, a * celldm[2]]]
        )

    elif ibrav == -12:  # monoclinic P, unique axis b
        _require(2, celldm[1] > 0.0)
        _require(3, celldm[2] > 0.0)
        _require(5, abs(celldm[4]) < 1.0)
        sen = math.sqrt(1.0 - celldm[4] ** 2)
        at = np.array(
            [[a, 0.0, 0.0], [0.0, a * celldm[1], 0.0], [a * celldm[2] * celldm[4], 0.0, a * celldm[2] * sen]]
        )

    elif ibrav == 13:  # base-centred monoclinic, unique axis c
        _require(2, celldm[1] > 0.0)
        _require(3, celldm[2] > 0.0)
        _require(4, abs(celldm[3]) < 1.0)
        sen = math.sqrt(1.0 - celldm[3] ** 2)
        half = 0.5 * a
        at = np.array(
            [
                [half, 0.0, -half * celldm[2]],
                [a * celldm[1] * celldm[3], a * celldm[1] * sen, 0.0],
                [half, 0.0, half * celldm[2]],
            ]
        )

    elif ibrav == -13:  # base-centred monoclinic, unique axis b
        _require(2, celldm[1] > 0.0)
        _require(3, celldm[2] > 0.0)
        _require(5, abs(celldm[4]) < 1.0)
        sen = math.sqrt(1.0 - celldm[4] ** 2)
        half = 0.5 * a
        at = np.array(
            [
                [half, half * celldm[1], 0.0],
                [-half, half * celldm[1], 0.0],
                [a * celldm[2] * celldm[4], 0.0, a * celldm[2] * sen],
            ]
        )

    elif ibrav == 14:  # triclinic
        _require(2, celldm[1] > 0.0)
        _require(3, celldm[2] > 0.0)
        for index in (4, 5, 6):
            _require(index, abs(celldm[index - 1]) < 1.0)
        singam = math.sqrt(1.0 - celldm[5] ** 2)
        term = (
            1.0
            + 2.0 * celldm[3] * celldm[4] * celldm[5]
            - celldm[3] ** 2
            - celldm[4] ** 2
            - celldm[5] ** 2
        )
        if term < 0.0:
            raise ValueError("ibrav=14: celldm do not make sense, check your data")
        term = math.sqrt(term / (1.0 - celldm[5] ** 2))
        at = np.array(
            [
                [a, 0.0, 0.0],
                [a * celldm[1] * celldm[5], a * celldm[1] * singam, 0.0],
                [
                    a * celldm[2] * celldm[4],
                    a * celldm[2] * (celldm[3] - celldm[4] * celldm[5]) / singam,
                    a * celldm[2] * term,
                ],
            ]
        )

    return at


def celldm_from_abc(
    ibrav: int,
    a: float,
    b: float = 0.0,
    c: float = 0.0,
    cosab: float = 0.0,
    cosac: float = 0.0,
    cosbc: float = 0.0,
) -> np.ndarray:
    """Crystallographic parameters (A, B, C in angstrom, cosines) -> celldm.

    Transcribed from ``abc2celldm`` in ``latgen.f90``. Which cosine lands in
    which celldm slot depends on ``ibrav``, which is the whole reason this
    conversion cannot be done inline at the call site.
    """
    if a <= 0.0:
        raise ValueError("incorrect lattice parameter A")
    for name, value in (("cosab", cosab), ("cosac", cosac), ("cosbc", cosbc)):
        if abs(value) > 1.0:
            raise ValueError(f"incorrect lattice parameter {name}")

    celldm = np.zeros(6)
    celldm[0] = a * ANGSTROM_TO_BOHR
    celldm[1] = b / a
    celldm[2] = c / a
    if ibrav in (0, 14):
        celldm[3], celldm[4], celldm[5] = cosbc, cosac, cosab
    elif ibrav in (-12, -13):
        celldm[4] = cosac
    elif ibrav in (5, -5, 12, 13):
        celldm[3] = cosab
    return celldm


class Cell(eqx.Module):
    """A unit cell: lattice vectors in bohr, plus the derived reciprocal cell.

    ``at[i]`` is the i-th lattice vector in bohr. ``alat`` is QE's length unit
    for the cell, kept because QE prints k-points and positions in terms of it.

    Derived quantities are properties rather than stored fields so that a strain
    derivative differentiates through them: perturbing ``at`` must move the
    volume and the reciprocal vectors with it.
    """

    at: jnp.ndarray
    alat: float = eqx.field(static=True)
    ibrav: int = eqx.field(static=True, default=0)
    precision: Precision = eqx.field(static=True, default=DEFAULT_PRECISION)

    @classmethod
    def from_ibrav(cls, ibrav: int, celldm, precision: Precision = DEFAULT_PRECISION) -> "Cell":
        celldm = np.asarray(celldm, dtype=float)
        at = latgen(ibrav, celldm)
        return cls(at=precision.as_real(at), alat=float(celldm[0]), ibrav=ibrav, precision=precision)

    @classmethod
    def from_vectors(
        cls,
        at,
        alat: float | None = None,
        ibrav: int = 0,
        precision: Precision = DEFAULT_PRECISION,
    ) -> "Cell":
        """Build from explicit lattice vectors in bohr (rows ``a1, a2, a3``).

        With ``alat=None``, QE's ``ibrav=0`` rule applies: ``alat`` becomes the
        length of the first lattice vector.
        """
        at = np.asarray(at, dtype=float)
        if at.shape != (3, 3):
            raise ValueError(f"lattice vectors must be (3,3), got {at.shape}")
        if np.linalg.norm(at, axis=1).min() == 0.0:
            raise ValueError("a lattice vector has zero length")
        if alat is None:
            alat = float(np.linalg.norm(at[0]))
        return cls(at=precision.as_real(at), alat=float(alat), ibrav=ibrav, precision=precision)

    # --- derived quantities ---------------------------------------------------
    @property
    def volume(self) -> jnp.ndarray:
        """Cell volume in bohr^3 (QE's ``omega``). Signed determinant magnitude."""
        return jnp.abs(jnp.linalg.det(self.at))

    @property
    def bg(self) -> jnp.ndarray:
        """Reciprocal lattice vectors in 1/bohr, rows ``b1, b2, b3``.

        Defined with the 2*pi convention, so ``a_i . b_j = 2*pi delta_ij``.
        """
        return TPI * jnp.linalg.inv(self.at).T

    @property
    def at_alat(self) -> jnp.ndarray:
        """Lattice vectors in units of alat -- what QE prints as ``crystal axes``."""
        return self.at / self.alat

    @property
    def bg_2pi_alat(self) -> jnp.ndarray:
        """Reciprocal vectors in units of 2*pi/alat -- QE's ``reciprocal axes``."""
        return self.bg * self.alat / TPI

    @property
    def tpiba(self) -> float:
        """2*pi/alat, the unit QE expresses k-points and G-vectors in."""
        return TPI / self.alat

    # --- coordinate conversions ----------------------------------------------
    def to_cartesian(self, crystal) -> jnp.ndarray:
        """Crystal (fractional) coordinates -> cartesian in bohr."""
        return jnp.asarray(crystal, dtype=self.at.dtype) @ self.at

    def to_crystal(self, cartesian) -> jnp.ndarray:
        """Cartesian coordinates in bohr -> crystal (fractional)."""
        return jnp.asarray(cartesian, dtype=self.at.dtype) @ jnp.linalg.inv(self.at)

    def k_to_cartesian(self, crystal) -> jnp.ndarray:
        """k in crystal coordinates -> cartesian in units of 2*pi/alat.

        This is QE's convention for ``xk``: ``cryst_to_cart`` with ``bg``, whose
        columns are the reciprocal vectors in units of 2*pi/alat.
        """
        return jnp.asarray(crystal, dtype=self.at.dtype) @ self.bg_2pi_alat

    def k_to_crystal(self, cartesian) -> jnp.ndarray:
        """k in units of 2*pi/alat -> crystal coordinates."""
        return jnp.asarray(cartesian, dtype=self.at.dtype) @ jnp.linalg.inv(self.bg_2pi_alat)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        name = IBRAV_NAMES.get(self.ibrav, "?")
        return f"Cell(ibrav={self.ibrav} [{name}], alat={self.alat:.6f} bohr, volume={self.volume:.4f})"
