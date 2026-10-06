"""The substrate: the plane the electron leaves through, as a Gram matrix.

An electron that enters at the tip leaves into the substrate, and "the
substrate" here is an **infinite, featureless plane** below the material -- a
metal the material sits on, structureless at the scale of the surface cell.
That assumption is what the whole phase rests on and it is worth stating as a
physical claim rather than as a convenience, because it is what makes the
calculation cheap: a plane that is invariant under every lateral lattice
translation conserves the lateral momentum, so states at different ``k`` cannot
interfere with each other on their way out.

Formally, the exit integral runs over the plane's *whole infinite area*, and

    int_{all cells} d^2 r' psi*_kn(r') psi_k'n'(r')
        = int d^2 r' e^{i(k'-k).r'} u*_kn u_k'n'
        ~ delta(k'_par - k_par + G_par)

which on a Brillouin-zone mesh forces ``k' = k``. What is left is one Hermitian
matrix per k-point,

    S_k[n, n'] = int_{plane inside one cell} d^2 r'  psi*_kn(r') psi_kn'(r')

-- a **Gram matrix of the bands restricted to the plane**. Two things follow
from that word and both are load-bearing downstream: it is positive
semi-definite, so the transmission built from it is non-negative by
construction rather than by luck; and if the "plane" is widened to the whole
cell it becomes the identity by orthonormality, which is the exact statement
that this quantity reduces to Tersoff-Hamann.

**It is computed exactly, with no quadrature.** For a plane at crystal
coordinate ``s3`` along one lattice vector, spanning the other two, the
in-plane integral is an orthogonality relation between Miller indices:

    S_k[n, n'] = (A / Omega) sum_{h_par} b*_n(h_par) b_n'(h_par),
    b_n(h_par) = sum_{h3} c_n(h_par, h3) e^{2 pi i h3 s3}

with ``A = |a_i x a_j|`` the area of the surface cell. One gather and one
``nbnd x n_hpar`` product per k-point; nothing is sampled and nothing converges.
A plane that is *not* spanned by two lattice vectors has no such relation and
would need a quadrature, which is why a tilted exit plane is refused rather
than approximated -- a two-dimensional material's substrate never is one.
"""

from __future__ import annotations

import numpy as np

__all__ = ["exit_overlap", "volume_overlap", "surface_area", "spin_projector",
           "plane_groups", "fold_onto_plane"]


def surface_area(cell, axis: int) -> float:
    """``|a_i x a_j|`` in bohr^2: the area of the cell's face normal to ``axis``.

    The measure of the exit integral. Not ``Omega / |a_axis|`` -- that is the
    same number only for an orthogonal cell, and a two-dimensional material's
    cell is routinely hexagonal.
    """
    at = np.asarray(cell.at, dtype=float)
    first, second = [i for i in (0, 1, 2) if i != axis]
    return float(np.linalg.norm(np.cross(at[first], at[second])))


def spin_projector(direction, polarization: float = 1.0,
                   what: str = "substrate") -> np.ndarray:
    """``(1 + P n.sigma)/2``: which spins a lead accepts.

    A spin-polarized substrate (or, read the other way round, a magnetic
    counter-electrode) does not take every state equally. This is the same
    ``[rho + P n.m]/2`` that :func:`defumat.stm.image.project_spin` applies to a
    tunnelling density, written one level lower as the 2x2 matrix in spin space
    it comes from -- because here it has to sit *inside* the overlap integral,
    between two different bands, where a density has already been squared.

    **Both leads use it.** For the substrate it goes inside
    :func:`exit_overlap`, between two bands; for the tip it multiplies the 2x2
    matrix :func:`defumat.transport.green.spin_transmission` builds. ``what``
    only names the lead in the error message.
    """
    from defumat.stm.image import _unit_vector

    unit = _unit_vector(direction)
    p = float(polarization)
    if not -1.0 <= p <= 1.0:
        raise ValueError(f"the {what} polarization must be in [-1, 1], got {p}")
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    dotted = unit[0] * sigma_x + unit[1] * sigma_y + unit[2] * sigma_z
    return 0.5 * (np.eye(2, dtype=complex) + p * dotted)


def plane_groups(miller, axis: int):
    """``(group, inplane)``: which in-plane index each plane wave carries.

    ``group`` is ``(npw,)``, the position of each plane wave's in-plane pair in
    ``inplane``, which is ``(ngroups, 3)`` integer Miller indices with the
    ``axis`` column zero -- the sphere's shadow on the surface reciprocal
    lattice, written as a sphere in its own right so that
    :func:`~defumat.basis.sample.sample_wavefunctions` takes it unchanged. It
    depends on the k-point's sphere and not on a height, so a caller folding at
    two heights (the exit plane and a flat tip plane) builds it once.
    """
    miller = np.asarray(miller, dtype=int)
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
    inplane = np.delete(miller, axis, axis=1)
    unique, group = np.unique(inplane, axis=0, return_inverse=True)
    rows = np.insert(np.asarray(unique, dtype=int), axis, 0, axis=1)
    return np.asarray(group).reshape(-1), rows.reshape((-1, 3))


def fold_onto_plane(coefficients, miller, height: float, axis: int, *,
                    npol: int = 1, mask=None, k_along: float = 0.0,
                    groups=None):
    """The sum over the Miller index normal to a plane, done before anything else.

    On a plane at crystal coordinate ``s`` along ``axis`` every plane wave with
    the same in-plane index ``h_par`` has the same lateral dependence, so

        b_n(h_par) = sum_{h3} c_n(h_par, h3) e^{2 pi i (k3 + h3) s}

    is everything a band carries there. :func:`exit_overlap` integrates two of
    these over the plane, where ``k3`` is common to every band at the k-point
    and cancels (it is passed as 0); a flat **tip** plane evaluates one of them
    at points, where ``k3`` is kept so that the result is ``psi`` and not ``psi``
    up to a phase. The sphere of ``npw`` plane waves becomes ``ngroups`` of
    them, which is the number of columns the sphere has along ``axis``: 1415 of
    26695 on a twenty-atom bismuth slab at ``ecutwfc = 20``.

    Args:
        coefficients: ``(nbnd, npol * npw)`` complex, one k-point's bands.
        miller: ``(npw, 3)`` the sphere's integer Miller indices.
        height: the plane's crystal coordinate along ``axis``.
        axis: which lattice vector the plane is normal to.
        npol: 1 or 2; a spinor's components are the two halves of the row.
        mask: ``(npw,)`` the padding mask. A padded entry points at ``G = 0``
            and is dropped rather than trusted to be zero.
        k_along: the k-point's crystal coordinate along ``axis``.
        groups: :func:`plane_groups` of ``miller`` and ``axis``, when the caller
            already has it.

    Returns ``(b, inplane)``: ``(nbnd, npol, ngroups)`` complex and the
    ``(ngroups, 3)`` in-plane sphere from :func:`plane_groups`.
    """
    coefficients = np.asarray(coefficients)
    miller = np.asarray(miller, dtype=int)
    npw = miller.shape[0]
    if coefficients.shape[-1] != npol * npw:
        raise ValueError(
            f"the coefficients are {coefficients.shape[-1]} long and the sphere "
            f"has {npw} plane waves times npol = {npol}"
        )
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
    group, inplane = plane_groups(miller, axis) if groups is None else groups

    nbnd = coefficients.shape[0]
    blocks = coefficients.reshape((nbnd, npol, npw))
    # The h3 sum, which carries the plane's height and collapses the sphere onto
    # its shadow on the surface reciprocal lattice. The mask goes into the phase
    # rather than onto the coefficients, so the whole block is copied once.
    phase = np.exp(2.0j * np.pi * (miller[:, axis] + float(k_along))
                   * float(height))
    if mask is not None:
        phase = phase * np.asarray(mask, dtype=bool)
    b = np.zeros((nbnd, npol, inplane.shape[0]), dtype=complex)
    np.add.at(b, (slice(None), slice(None), group), blocks * phase[None, None, :])
    return b, inplane


def exit_overlap(coefficients, miller, height: float, axis: int, cell,
                 mask=None, npol: int = 1, projector=None,
                 groups=None) -> np.ndarray:
    """``S_k``: the bands' Gram matrix on the exit plane, ``(nbnd, nbnd)``.

    Args:
        coefficients: ``(nbnd, npol * npwx)`` complex, one k-point's bands. A
            spinor's two components are the two halves of the row, which is
            how the whole package stores them.
        miller: ``(npwx, 3)`` integer Miller indices of this k-point's sphere.
        height: the plane's crystal coordinate along ``axis``.
        axis: which lattice vector the plane is normal to -- the stacking axis
            of a two-dimensional material, 2 in every ordinary slab.
        cell: for the surface area and the volume.
        mask: ``(npwx,)`` padding mask.
        npol: 1 or 2.
        projector: ``(2, 2)`` complex, the substrate's spin acceptance, or
            ``None`` for a substrate that takes both spins equally. Only for a
            spinor run; a collinear one selects a channel outside this.
        groups: :func:`plane_groups` of ``miller`` and ``axis``, when the caller
            already has it.

    Hermitian to round-off and positive semi-definite. Both are asserted by the
    tests rather than imposed here, because imposing them would hide the bug
    they are there to catch.
    """
    # The h3 sum, done first (:func:`fold_onto_plane`). ``k3`` is common to
    # every band at this k-point and cancels in ``b* b``, so it is left out.
    b, _ = fold_onto_plane(coefficients, miller, height, axis, npol=npol,
                           mask=mask, groups=groups)

    scale = surface_area(cell, axis) / float(cell.volume)
    if npol == 1:
        return scale * (b[:, 0].conj() @ b[:, 0].T)
    if projector is None:
        return scale * sum(b[:, s].conj() @ b[:, s].T for s in range(npol))
    projector = np.asarray(projector, dtype=complex)
    if projector.shape != (2, 2):
        raise ValueError(f"a spin projector is 2x2, got {projector.shape}")
    return scale * sum(
        projector[s, t] * (b[:, s].conj() @ b[:, t].T)
        for s in range(2) for t in range(2)
    )


def volume_overlap(coefficients, mask=None, npol: int = 1, overlap=None):
    """The same Gram matrix with the *whole cell* as the exit region.

    Which is the identity, by orthonormality -- and that is the point of it.
    Widening the substrate from a plane to everything turns this calculation
    into Tersoff-Hamann exactly (:mod:`defumat.transport.green` derives it), so
    running the same assembly with this in place of :func:`exit_overlap` and
    comparing against :func:`defumat.workflows.stm.run_stm` checks the
    k-weights, the spin degeneracy, the normalisation and the tip sampling in
    one number, sharing no code with the reduction it is checking.

    ``overlap`` applies ``S`` -- :meth:`defumat.scf.driver.Calculation._overlap`
    or its spinor twin. **It is not optional for an ultrasoft or PAW dataset**:
    orthonormality there is ``<psi|S|psi> = delta``, and the plain
    ``sum_G c* c`` is short of the augmentation charge by 9 per cent on an
    ultrasoft carbon sheet and 3 per cent on a PAW silicon one -- which reads
    exactly like an error in the assembly and is not one. It does not arise for
    :func:`exit_overlap`, whose plane is in the vacuum where a
    pseudo-wavefunction is the true one.
    """
    coefficients = np.asarray(coefficients)
    if mask is not None:
        npwx = np.asarray(mask).shape[0]
        nbnd = coefficients.shape[0]
        blocks = coefficients.reshape((nbnd, npol, npwx))
        coefficients = (blocks * np.asarray(mask, dtype=bool)).reshape(
            (nbnd, npol * npwx))
    other = coefficients if overlap is None else np.asarray(overlap(coefficients))
    return coefficients.conj() @ other.T
