"""The cell as nine more coordinates: its gradient, and what may move.

``Modules/cell_base.f90`` -- ``cell_force`` and ``init_dofree`` -- plus the two
constraints ``move_ions.f90`` applies afterwards. What this module owns is the
translation between the two languages the cell is spoken in: the *stress*, which
is what a plane-wave code computes and what a user reads, and ``dH/dh``, which is
what an optimizer needs.

**The cell gradient is the stress, rearranged, and the rearrangement is exact.**
Write the cell as ``h``, the matrix whose *columns* are the lattice vectors
(QE's convention, and the transpose of this package's row-wise ``at``). A
deformation ``h -> (1 + epsilon) h`` gives ``epsilon = dh h^-1``, so

    dE = tr[ (dE/d epsilon)^T dh h^-1 ]   =>   dE/dh = (dE/d epsilon) h^-T,

and with ``sigma = -(1/Omega) dE/d epsilon`` (:mod:`pypresso.stress`) and the
volume's own derivative ``d Omega/dh = Omega h^-T``, the gradient of the
**enthalpy** ``H = E + P Omega`` is

    dH/dh = Omega (P I - sigma) h^-T,

which is ``cell_force`` line for line. Two things fall out of it and both are
used: the stationary point is ``sigma = P I`` -- the crystal is relaxed when its
stress *equals the applied pressure*, not when the stress vanishes -- and
``(dH/dh) h^T / Omega = P I - sigma`` recovers the stress from the gradient,
which is how QE reports the cell's convergence error in kbar.

**Why the enthalpy and not the energy.** At fixed pressure the cell will always
lower its energy by expanding into vacuum; what is stationary is the enthalpy,
and ``P Omega`` is the only term that makes ``press = 0`` different from "no
cell term at all" -- at zero pressure the two coincide, which is why
``vc-relax3`` is the case that tests the gradient and ``vc-relax4`` the case
that tests the ``P Omega``.

**What ``cell_dofree`` is.** A ``(3, 3)`` mask on ``h``, QE's ``iforceh``, with
``iforceh(i, j)`` the *i*-th cartesian component of the *j*-th lattice vector.
It is applied to the gradient and re-applied after every product with the
inverse Hessian (:mod:`pypresso.relax.bfgs`). The masks that also impose a
*nonlinear* constraint -- ``'shape'`` and ``'2Dshape'``, which fix the volume or
the area through ``impose_deviatoric_strain``, ``'volume'``, which forces the
step isotropic, and ``'ibrav'``, which reimposes a Bravais lattice through
``remake_cell`` -- are refused by name rather than approximated by their mask
alone, since the mask without the constraint is a different calculation that
would converge and report success.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cell_force", "cell_dofree_mask", "CELL_DOFREE"]


def cell_force(
    stress: np.ndarray, h: np.ndarray, omega: float, pressure: float = 0.0
) -> np.ndarray:
    """``dH/dh = Omega (P I - sigma) h^-T`` -- ``cell_base.f90``'s ``cell_force``.

    Args:
        stress: ``(3, 3)`` cartesian stress in Ry/bohr^3, with
            :mod:`pypresso.stress`'s sign (a compressed crystal has a positive
            ``tr sigma / 3``).
        h: ``(3, 3)`` cell with the lattice vectors as **columns**, in bohr --
            the transpose of :attr:`pypresso.system.cell.Cell.at`.
        omega: the cell volume in bohr^3.
        pressure: the target pressure in Ry/bohr^3.

    QE writes this with ``ainv = -bg^T/alat``, which is ``-h^-1``, and the two
    minus signs that produces are the reason the expression looks as though the
    stress enters with the wrong sign. It does not: expanding a crystal whose
    stress is positive (compressed) must lower the enthalpy, and it does.
    """
    stress = np.asarray(stress, dtype=float)
    h = np.asarray(h, dtype=float)
    target = pressure * np.eye(3) - stress
    return omega * target @ np.linalg.inv(h).T


#: ``cell_dofree`` -> the ``iforceh`` mask, for the values whose whole content
#: *is* the mask. ``init_dofree`` in ``Modules/cell_base.f90``.
CELL_DOFREE: dict[str, np.ndarray] = {}


def _register(name: str, free) -> None:
    mask = np.zeros((3, 3))
    for i, j in free:
        mask[i, j] = 1.0
    CELL_DOFREE[name] = mask


_ALL = [(i, j) for i in range(3) for j in range(3)]
_register("all", _ALL)
_register("default", _ALL)
# 'a', 'b', 'c' free everything but one diagonal entry; 'fixa'/'fixb'/'fixc'
# free everything but a whole lattice vector (a *column* of h).
for _axis, _name in enumerate("abc"):
    _register(_name, [(i, j) for i, j in _ALL if (i, j) != (_axis, _axis)])
    _register(f"fix{_name}", [(i, j) for i, j in _ALL if j != _axis])
# 'x', 'y', 'z' and their combinations free only diagonal entries: the cell
# keeps its shape and stretches along the named cartesian axes.
_register("x", [(0, 0)])
_register("y", [(1, 1)])
_register("z", [(2, 2)])
_register("xy", [(0, 0), (1, 1)])
_register("xz", [(0, 0), (2, 2)])
_register("yz", [(1, 1), (2, 2)])
_register("xyz", [(0, 0), (1, 1), (2, 2)])
# '2Dxy' frees the whole xy block, where 'xy' frees only its diagonal -- QE
# leaves the off-diagonal lines commented out in one and writes them in the
# other, and the difference is whether the xy plane may shear.
_register("2Dxy", [(0, 0), (0, 1), (1, 0), (1, 1)])
# The epitaxial constraints hold two lattice vectors and free the third.
_register("epitaxial_ab", [(0, 2), (1, 2), (2, 2)])
_register("epitaxial_ac", [(0, 1), (1, 1), (2, 1)])
_register("epitaxial_bc", [(0, 0), (1, 0), (2, 0)])

#: The values ``init_dofree`` accepts whose content is *not* only a mask. Each
#: is refused by name with what it would additionally need.
_CONSTRAINED_DOFREE = {
    "shape": (
        "fix_volume, which projects the deviatoric part out of the step "
        "(move_ions.f90's impose_deviatoric_strain)"
    ),
    "2Dshape": (
        "fix_area, the two-dimensional counterpart "
        "(impose_deviatoric_strain_2d)"
    ),
    "volume": (
        "isotropic = .true., which averages the diagonal of the cell force so "
        "that only the volume moves, and which QE allows for ibrav = 1 alone"
    ),
    "ibrav": (
        "enforce_ibrav, which pushes the stepped cell back onto its Bravais "
        "lattice each step (remake_cell) and rescales alat with it"
    ),
}


def cell_dofree_mask(cell_dofree: str | None) -> np.ndarray:
    """``iforceh`` for a ``cell_dofree`` value; ``init_dofree``'s SELECT CASE."""
    name = (cell_dofree or "all").strip().strip("'\"")
    if name in CELL_DOFREE:
        return CELL_DOFREE[name].copy()
    base = name.split("+", 1)[0]
    if base in _CONSTRAINED_DOFREE or name in _CONSTRAINED_DOFREE:
        needs = _CONSTRAINED_DOFREE.get(name) or _CONSTRAINED_DOFREE[base]
        raise NotImplementedError(
            f"cell_dofree = {name!r} is not implemented -- it needs {needs}. "
            "It is refused rather than run as its mask alone, which is a "
            "different constraint that would converge and report success"
        )
    raise ValueError(
        f"unknown cell_dofree {name!r}; expected one of "
        f"{', '.join(sorted(CELL_DOFREE))}"
    )
