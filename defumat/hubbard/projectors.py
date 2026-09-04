"""The Hubbard projector functions ``wfcU``, in the plane-wave basis.

``PW/src/orthoUwfc.f90``. The occupation matrix is measured by projecting the
Kohn-Sham states onto a set of localised orbitals, and *which* set is the single
most consequential choice in a DFT+U calculation -- the same U on two different
projector sets is two different calculations.

Two of QE's choices are implemented, and both start from the pseudo-atomic
orbitals of :mod:`defumat.pseudo.atomic`:

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

from defumat.hubbard.manifold import PROJECTOR_TYPES
from defumat.pseudo.atomic import atomic_wavefunctions, spinor_atomic_wavefunctions

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
    kcart=None,
) -> jnp.ndarray:
    """``(nk, npwx, nwfcU)``: the Hubbard projectors at every k-point.

    ``apply_s(psi, ik)`` is the overlap operator of the calculation, taking
    ``(..., npwx)`` states -- :meth:`defumat.hamiltonian.operator.Hamiltonian.apply_s`
    or anything with that signature.

    ``kcart`` replaces the k-points' cartesian coordinates while keeping this
    basis, exactly as it does for the nonlocal projectors. It exists for the
    stress: ``KPoints.coords`` are in units of ``2 pi / alat`` and ``alat`` is
    static, so under a strain the k-points do **not** follow the reciprocal cell
    on their own -- and a Hubbard projector built at the wrong k is silently
    wrong rather than an error.

    ``orthoUwfc_k``'s ``lflag = .TRUE.`` variant -- ``O^{-1/2} phi``, the
    orthogonalisation without the trailing ``S`` -- is deliberately absent. It
    exists in QE only for the hand-derived force and stress expressions, and the
    force here is ``jax.grad`` of the energy through this function
    (:mod:`defumat.forces.energy`), so nothing would consume it.

    A thin selection of columns out of :func:`build_atomic_projectors`, which is
    the same construction over *every* atomic orbital -- the projected density of
    states (:mod:`defumat.projwfc`) is the other caller and it keeps them all.
    """
    width = setup.npol
    columns = (
        np.concatenate([
            np.arange(offset, offset + width * ldim)
            for offset, ldim in zip(setup.atomwfc_offsets, setup.ldims)
        ]) if setup.nwfcU else np.zeros(0, dtype=int)
    )
    return build_atomic_projectors(
        pseudos, structure, cell, gvectors, planewaves, kpoints, apply_s,
        kind=setup.projectors, columns=columns, kcart=kcart,
        noncolin=setup.noncolin,
    )


def _spinor_channels(pseudos, structure) -> list:
    """How to build each spinor channel out of the scalar orbital list.

    Each entry is ``(width, [(start, weight), ...])``: the columns to combine
    and with what weight, ``start`` being where that scalar channel begins.

    **A fully-relativistic dataset needs the combination and it is not
    optional.** ``atomic_wfc_so_mag`` returns immediately for the
    ``j = l - 1/2`` channel and, for ``j = l + 1/2``, builds the *average*

        chi = [ (l + 1) chi_{j = l + 1/2} + l chi_{j = l - 1/2} ] / (2l + 1)

    which is the "averaged j = l +- 1/2 radial WFs" ``plus_u_full.f90``'s header
    advertises. So the two ``j`` channels of one shell collapse into **one**
    spinor channel of ``2 (2l+1)`` columns, not two of them. Doubling every
    entry of the scalar list instead gives twice as many columns, shifts every
    offset after the first ``p`` shell, and makes the Hubbard manifold a mixture
    of the two ``j`` channels rather than their average: on relativistic BN that
    is the whole occupation matrix wrong by a factor near 1.85, an SCF that
    converges, and a total energy 0.025 Ry out.

    The average is taken on the *built* orbitals rather than on the radial
    tables, which is exact: the two channels share ``i^l``, the structure factor
    and the harmonic, and differ only in the radial function they multiply.
    """
    channels = []
    start = 0
    for species in structure.types:
        pseudo = pseudos[species]
        kept = [o for o in pseudo.orbitals if o.occupation >= 0.0]
        starts, previous = [], None
        for orbital in kept:
            starts.append(start)
            start += 2 * orbital.l + 1
        relativistic = any(getattr(o, "j", None) is not None for o in kept)
        for index, orbital in enumerate(kept):
            width = 2 * orbital.l + 1
            l, j = orbital.l, getattr(orbital, "j", None)
            if not relativistic or l == 0 or j is None:
                channels.append((width, [(starts[index], 1.0)]))
                continue
            if abs(j - l + 0.5) < 1.0e-4:
                continue  # the j = l - 1/2 partner; folded into the one below
            partner = next(
                (n for n, other in enumerate(kept)
                 if other.l == l and other.j is not None
                 and abs(other.j - l + 0.5) < 1.0e-4),
                None,
            )
            if partner is None:
                channels.append((width, [(starts[index], 1.0)]))
                continue
            channels.append((width, [
                (starts[index], (l + 1.0) / (2 * l + 1.0)),
                (starts[partner], l / (2 * l + 1.0)),
            ]))
    return channels


def _spinor_expand(atomic: jnp.ndarray, channels, npwx: int) -> jnp.ndarray:
    """``(nk, 2 nchannel_orbitals, 2 npwx)`` from scalar orbitals, in QE's order.

    ``atomic_wfc_nc``'s ``updown`` branch: each spatial orbital appears twice,
    once with a pure spin-up spinor and once with a pure spin-down one, and the
    two copies of a *channel* are emitted as blocks -- all ``2l+1`` up columns,
    then all ``2l+1`` down ones -- rather than interleaved per orbital. That
    order is not cosmetic: ``new_ns_nc`` reads the occupation matrix at
    ``offsetU + m + ldim (is - 1)``, so interleaving would transpose the spin
    and ``m`` indices of every block.

    ``channels`` comes from :func:`_spinor_channels` and says which scalar
    columns each spinor channel is built from.
    """
    nk = atomic.shape[0]
    blocks = []
    for width, parts in channels:
        block = sum(
            weight * atomic[:, start:start + width, :] for start, weight in parts
        )
        pad = jnp.zeros_like(block)
        blocks.append(jnp.concatenate([block, pad], axis=-1))   # spin up
        blocks.append(jnp.concatenate([pad, block], axis=-1))   # spin down
        start = 0
    if not blocks:
        return jnp.zeros((nk, 0, 2 * npwx), dtype=atomic.dtype)
    return jnp.concatenate(blocks, axis=1)


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
    kcart=None,
    noncolin: bool = False,
    spinor_basis: str = "updown",
) -> jnp.ndarray:
    """``(nk, npwx, ncolumns)``: projector functions built from ``chi``.

    ``kind`` is one of :data:`defumat.hubbard.manifold.PROJECTOR_TYPES` and
    ``columns`` selects which of the ``natomwfc`` orbitals to keep *after* the
    orthogonalisation -- ``None`` keeps all of them, which is what
    ``projwfc.x`` projects onto and what a Löwdin charge is defined against.

    The orthogonalisation always runs over the whole set whatever is kept, for
    the reason in the module docstring: restricting ``O`` to a sub-manifold is a
    different matrix and a different answer.

    ``kcart`` replaces the k-points' cartesian coordinates, as it does for the
    nonlocal projectors, and exists for the stress -- see
    :func:`build_hubbard_projectors`.

    ``spinor_basis`` picks *which* spinor set ``noncolin`` builds, and the two
    are different bases rather than different spellings:

    * ``"updown"`` is ``atomic_wfc_so_mag`` -- the two ``j`` radial functions of
      a shell averaged, filling pure up and down spinors. It is what a
      noncollinear SCF starts from and what DFT+U's ``wfcU`` is, and it is the
      default so that nothing already validated moves.
    * ``"jmj"`` is ``atomic_wfc_so`` -- the spin-angle functions themselves,
      ``|l j m_j>``, which is what ``atomic_wfc_nc_proj`` gives ``projwfc.x``
      and what a ``j``-resolved projection is a decomposition on.

    QE reaches the two through ``starting_spin_angle``, ``.FALSE.`` for the SCF
    and ``.TRUE.`` for the projection.
    """
    if spinor_basis not in ("updown", "jmj"):
        raise ValueError(
            f"unknown spinor basis {spinor_basis!r}; expected 'updown' or 'jmj'"
        )
    if noncolin and spinor_basis == "jmj":
        atomic = spinor_atomic_wavefunctions(
            pseudos, structure, cell, gvectors, planewaves, kpoints,
            lspinorb=True, kcart=kcart,
        )  # (nk, natomwfc_spinor, 2 npwx)
    else:
        atomic = atomic_wavefunctions(
            pseudos, structure, cell, gvectors, planewaves, kpoints, kcart
        )  # (nk, natomwfc, npwx)
        if noncolin:
            atomic = _spinor_expand(
                atomic, _spinor_channels(pseudos, structure), planewaves.npwx
            )
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
