"""The Hubbard projector functions ``wfcU``, in the plane-wave basis.

``PW/src/orthoUwfc.f90``. The occupation matrix is measured by projecting the
Kohn-Sham states onto a set of localised orbitals, and *which* set is the single
most consequential choice in a DFT+U calculation -- the same U on two different
projector sets is two different calculations.

Two of QE's choices are implemented, and both start from the pseudo-atomic
orbitals of :mod:`pypresso.pseudo.atomic`:

``atomic``
    ``wfcU = S phi``. **The overlap operator is applied even here**, and this
    is the trap: the option is called "atomic" and the natural reading is that
    the orbitals are used as they come out of the file. They are not. QE runs
    ``s_psi`` on them unconditionally (``orthoUwfc``'s call sequence), so what
    the projection ``<wfcU|psi>`` computes is ``<phi|S|psi>`` -- the correct
    projection in the generalised metric an ultrasoft dataset lives in. With a
    norm-conserving dataset ``S`` is the identity and the distinction vanishes,
    which is why the mistake survives testing on silicon.

``ortho-atomic``
    Löwdin-orthogonalised: ``wfcU = O^{-1/2} S phi`` with
    ``O_ij = <phi_i|S|phi_j>``. **The orthogonalisation runs over every atomic
    orbital in the crystal**, not over the Hubbard manifold alone -- the 4s of
    an iron atom is orthogonalised against the 3d of its neighbour, and only
    afterwards are the Hubbard columns selected out. Restricting ``O`` to the
    Hubbard columns is a different (and much better conditioned) matrix, and it
    gives a different occupation matrix.

``norm-atomic`` is the same code path with the off-diagonal entries of ``O``
zeroed, which normalises the orbitals without orthogonalising them.

The transposition in the transform is QE's and is kept: ``ortho_swfc`` builds
the *transpose* of ``O^{-1/2}`` and applies it as
``Sphi_I = sum_J O^{-1/2}_{JI} Sphi_J``. For a real ``O`` this is the same as
the untransposed form; at a general k-point ``O`` is complex Hermitian and it is
not.

Memory: ``wfcU`` is ``(nk, npwx, nwfcU)`` complex, held for the whole run --
five or ten columns per correlated atom, so a few percent of one k-point's
wavefunctions. QE keeps it in the ``iunhub`` buffer, one k-point at a time,
because it stores it alongside ``evc`` on disk; here the whole k axis is in
memory already (rule R6) and this follows it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from pypresso.hubbard.manifold import PROJECTOR_TYPES
from pypresso.pseudo.atomic import atomic_wavefunctions

__all__ = [
    "build_hubbard_projectors",
    "build_atomic_projectors",
    "lowdin_transform",
]


def lowdin_transform(overlap: jnp.ndarray, normalize_only: bool = False) -> jnp.ndarray:
    """``O^{-1/2}`` transposed, as ``ortho_swfc`` builds it.

    ``overlap[i, j] = <phi_i|S|phi_j>`` is Hermitian positive definite, so the
    inverse square root is an eigendecomposition. The result is stored with its
    indices swapped relative to the usual convention, matching the Fortran, and
    :func:`_apply_transform` consumes it in the matching order.
    """
    if normalize_only:
        overlap = jnp.diag(jnp.diag(overlap))
    values, vectors = jnp.linalg.eigh(overlap)
    inverse_sqrt = (vectors * jax.lax.rsqrt(values)) @ vectors.conj().T
    return inverse_sqrt.T


def _apply_transform(transform: jnp.ndarray, wfc: jnp.ndarray) -> jnp.ndarray:
    """``new[:, j] = sum_k O^{-1/2}[k, j] old[:, k]``, the ZGEMM in ``ortho_swfc``."""
    return wfc @ transform.T


def build_hubbard_projectors(
    setup,
    pseudos,
    structure,
    cell,
    gvectors,
    planewaves,
    kpoints,
    apply_s,
) -> jnp.ndarray:
    """``(nk, npwx, nwfcU)``: the Hubbard projectors at every k-point.

    ``apply_s(psi, ik)`` is the overlap operator of the calculation, taking
    ``(..., npwx)`` states -- :meth:`pypresso.hamiltonian.operator.Hamiltonian.apply_s`
    or anything with that signature.

    ``orthoUwfc_k``'s ``lflag = .TRUE.`` variant -- ``O^{-1/2} phi``, the
    orthogonalisation without the trailing ``S`` -- is deliberately absent. It
    exists in QE only for the hand-derived force and stress expressions, and the
    force here is ``jax.grad`` of the energy through this function
    (:mod:`pypresso.forces.energy`), so nothing would consume it.

    A thin selection of columns out of :func:`build_atomic_projectors`, which is
    the same construction over *every* atomic orbital -- the projected density of
    states (:mod:`pypresso.projwfc`) is the other caller and it keeps them all.
    """
    columns = (
        np.concatenate([
            np.arange(offset, offset + ldim)
            for offset, ldim in zip(setup.atomwfc_offsets, setup.ldims)
        ]) if setup.nwfcU else np.zeros(0, dtype=int)
    )
    return build_atomic_projectors(
        pseudos, structure, cell, gvectors, planewaves, kpoints, apply_s,
        kind=setup.projectors, columns=columns,
    )


def build_atomic_projectors(
    pseudos,
    structure,
    cell,
    gvectors,
    planewaves,
    kpoints,
    apply_s,
    kind: str = "ortho-atomic",
    columns=None,
) -> jnp.ndarray:
    """``(nk, npwx, ncolumns)``: projector functions built from ``chi``.

    ``kind`` is one of :data:`pypresso.hubbard.manifold.PROJECTOR_TYPES` and
    ``columns`` selects which of the ``natomwfc`` orbitals to keep *after* the
    orthogonalisation -- ``None`` keeps all of them, which is what
    ``projwfc.x`` projects onto and what a Löwdin charge is defined against.

    The orthogonalisation always runs over the whole set whatever is kept, for
    the reason in the module docstring: restricting ``O`` to a sub-manifold is a
    different matrix and a different answer.
    """
    atomic = atomic_wavefunctions(
        pseudos, structure, cell, gvectors, planewaves, kpoints
    )  # (nk, natomwfc, npwx)
    nk, natomwfc = atomic.shape[0], atomic.shape[1]
    if columns is None:
        columns = np.arange(natomwfc)
    columns = jnp.asarray(np.asarray(columns, dtype=int))

    if kind not in PROJECTOR_TYPES:
        raise ValueError(
            f"unknown projector set {kind!r}; expected one of {PROJECTOR_TYPES}"
        )
    orthogonalize = kind in ("ortho-atomic", "norm-atomic")
    normalize_only = kind == "norm-atomic"

    def one_kpoint(ik: int) -> jnp.ndarray:
        phi = atomic[ik]  # (natomwfc, npwx)
        sphi = apply_s(phi, ik)
        if not orthogonalize:
            return jnp.transpose(sphi[columns], (1, 0))
        # ``O_ij = <phi_i|S|phi_j> = <phi_i|sphi_j>``, over every atomic
        # orbital of the crystal.
        overlap = jnp.conj(phi) @ sphi.T
        transform = lowdin_transform(overlap, normalize_only)
        return _apply_transform(transform, jnp.transpose(sphi, (1, 0)))[:, columns]

    # A Python loop over k rather than a ``vmap``: this runs once per geometry,
    # the eigendecomposition inside is ``natomwfc`` cubed and tiny, and batching
    # it would hold every k-point's ``natomwfc x natomwfc`` overlap at once for
    # no gain.
    return jnp.stack([one_kpoint(ik) for ik in range(nk)])
