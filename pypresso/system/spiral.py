"""Spin spirals: the generalized Bloch theorem, and what it does to the k-points.

A flat spin spiral is a magnetization that turns by ``q . R`` from one cell to
the next,

    m(r + R) = Rot_z(q . R) m(r),

which is not periodic and would seem to need a supercell -- one commensurate
with ``q``, which for a general ``q`` does not exist. It does not need one.
Without spin-orbit coupling the Hamiltonian is invariant under a lattice
translation *combined* with the matching spin rotation, and the eigenstates can
be labelled by ``k`` again (Sandratskii, J. Phys. Condens. Matter 3, 8565
(1991)). Elk's manual (§5.146, ``vqlss``) writes the resulting spinor as

    Psi^q_k(r) = ( U_up(r) e^{i(k + q/2).r},  U_dn(r) e^{i(k - q/2).r} )

with ``U_up``, ``U_dn`` lattice periodic, and the magnetization it produces as

    m^q(r) = ( m_x(r) cos(q.r),  m_y(r) sin(q.r),  m_z(r) )

with ``m_x``, ``m_y``, ``m_z`` lattice periodic. **Everything the SCF touches is
one of those periodic functions**, so the density, the potential, the
exchange-correlation functional and the mixer are untouched by the spiral; what
changes is which plane waves each spinor component is built from.

**That is the whole implementation.** ``gengkqvec.f90`` in Elk puts the up
component at ``k + q/2`` and the down at ``k - q/2``, each with its own ``G+k``
set, and this module produces exactly that: one k-list of length ``2 nk``, the
up component's points first. Building both halves in *one* call to
``build_plane_wave_basis`` is not an optimisation -- it is what gives the two
components a common ``npwx``, so rule R7's padding, the ``vmap`` over k and the
stick layout all keep working unchanged.

**The shifted points are kept literal.** ``k + q/2`` is not wrapped back into
the first Brillouin zone: the wrapped point is the same physics through a
different G-index map, and the phase that relates them is exactly the trap P16
records for the zone edge (``u_{k+b}(G) = u_k(G+b)``). Nothing here needs the
wrapped form, so nothing wraps it.

**Symmetry is refused rather than reduced.** A spiral breaks the ordinary space
group: only the operations with ``S^T q = q`` survive at all (Elk checks this in
``findsymlat.f90``), and even those act on the rotated-frame magnetization with a
spin rotation of their own -- the spin space group, which is not written here.
Time reversal is not available either, since it sends ``q`` to ``-q``. So a
spiral run needs the full k-grid, and :func:`invariant_operations` exists to say
how much would be gained by writing the spin space group rather than to be used.
"""

from __future__ import annotations

import numpy as np

from pypresso.system.cell import Cell
from pypresso.system.kpoints import KPoints
from pypresso.system.symmetry import Symmetries

__all__ = ["spiral_kpoints", "invariant_operations", "spiral_cartesian"]


def spiral_cartesian(q_crystal, cell: Cell) -> np.ndarray:
    """The spiral wavevector in cartesian units of ``2 pi / alat``.

    The input is in *lattice* coordinates, as Elk's ``vqlss`` is: a spiral at
    ``q = (0, 0, 1/2)`` doubles the magnetic period along the third lattice
    vector whatever the cell's shape, which is what makes that the useful way to
    say it.
    """
    return np.asarray(cell.k_to_cartesian(np.asarray(q_crystal, dtype=float)))


def spiral_kpoints(kpoints: KPoints, q_crystal, cell: Cell) -> KPoints:
    """The ``2 nk`` shifted points a spiral needs: ``k + q/2`` then ``k - q/2``.

    The weights are carried through unchanged in both halves. They are not used
    as integration weights on this list -- the *state* at ``k`` is one object
    with two components, and its weight is the original one -- but keeping them
    means the doubled list is a perfectly ordinary :class:`KPoints` that
    ``build_plane_wave_basis`` and the projectors accept without a special case.
    """
    half = 0.5 * spiral_cartesian(q_crystal, cell)
    coords = np.asarray(kpoints.coords)
    shifted = np.concatenate([coords + half, coords - half])
    weights = np.concatenate([np.asarray(kpoints.weights)] * 2)
    return KPoints.from_cartesian(shifted, weights, precision=kpoints.precision)


def invariant_operations(symmetries: Symmetries, q_crystal) -> Symmetries:
    """The operations that leave the spiral wavevector alone, ``S^T q = q``.

    Elk's check, in ``findsymlat.f90``, and the *first* condition a spin space
    group imposes -- not the only one, which is why this is not yet enough to
    reduce a k-set with. ``q`` is a reciprocal-space vector in crystal
    coordinates, so it transforms like a Miller index: ``q' = M q``.
    """
    q = np.asarray(q_crystal, dtype=float)
    kept, translations, t_rev = [], [], []
    for index, rotation in enumerate(symmetries.rotation_array()):
        if np.max(np.abs(rotation @ q - q)) > 1.0e-6:
            continue
        kept.append(symmetries.rotations[index])
        translations.append(symmetries.translations[index])
        t_rev.append(int(symmetries.t_rev_array()[index]))
    return Symmetries(
        rotations=tuple(kept),
        translations=tuple(translations),
        time_reversed=tuple(t_rev),
    )
