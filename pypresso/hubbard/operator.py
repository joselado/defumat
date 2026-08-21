"""The Hubbard potential as an operator on the wavefunctions.

``PW/src/vhpsi.f90``:

    V_U |psi> = sum_{I,m1,m2} |phi^I_{m1}> v^I_{m1 m2} <phi^I_{m2}|psi>

which is separable in exactly the way the nonlocal pseudopotential term is --
the same shape of contraction, with the Hubbard projectors in place of the beta
functions and the Hubbard potential in place of ``D_ij``. That is why it costs
what it costs: two matrix products per band, no transform, and a matrix that is
five or ten wide per correlated atom rather than the whole plane-wave sphere.

The potential is block diagonal over atoms -- there are no off-site terms in
``lda_plus_u_kind = 0`` -- so it is assembled once per SCF iteration into a
single ``(nwfcU, nwfcU)`` matrix and the whole sum becomes one contraction.
QE keeps the blocks separate and issues one ``ZGEMM`` pair per atom; here the
block matrix is a scatter with static indices, which is what keeps the term
inside ``jit`` and differentiable.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

__all__ = ["HubbardTerm", "block_potential"]


def block_potential(v_ns: jnp.ndarray, indices) -> jnp.ndarray:
    """``(nspin, nwfcU, nwfcU)``: the per-atom blocks laid on the diagonal.

    ``indices`` is ``(slot, row, column, block_row, block_column, nwfcU)`` --
    the static scatter :meth:`~pypresso.hubbard.manifold.HubbardSetup.block_indices`
    produces, paired with the positions inside each padded block.
    """
    slot, row, column, block_row, block_column, nwfcU = indices
    nspin = v_ns.shape[0]
    values = v_ns[:, slot, block_row, block_column]  # (nspin, nentries)
    empty = jnp.zeros((nspin, nwfcU, nwfcU), dtype=v_ns.dtype)
    return empty.at[:, row, column].set(values)


class HubbardTerm(eqx.Module):
    """The Hubbard term of one spin channel's Hamiltonian.

    ``wfcU`` is ``(nk, npwx, nwfcU)`` -- the same layout as ``vkb``, so that the
    contraction below is the nonlocal term's with two names changed. ``vns`` is
    the block matrix for *this* channel; a second channel is a second
    :class:`HubbardTerm` sharing the same projectors, which is how the two
    Hamiltonians of an LSDA run already differ.
    """

    wfcU: jnp.ndarray
    vns: jnp.ndarray  # (nwfcU, nwfcU), real

    @property
    def nwfcU(self) -> int:
        return self.wfcU.shape[-1]

    def apply(self, psi: jnp.ndarray, ik: int) -> jnp.ndarray:
        """``V_U|psi>`` for ``psi`` of shape ``(..., npwx)``."""
        columns = self.wfcU[ik]
        proj = jnp.einsum("gi,...g->...i", jnp.conj(columns), psi)
        return jnp.einsum("gi,...i->...g", columns, proj @ self.vns.T.astype(columns.dtype))

    def matrix(self, ik: int) -> jnp.ndarray:
        """The term as an explicit ``(npwx, npwx)`` matrix, for the dense solve."""
        columns = self.wfcU[ik]
        return columns @ self.vns.astype(columns.dtype) @ jnp.conj(columns).T

    def diagonal(self, ik: int) -> jnp.ndarray:
        """``<k+G|V_U|k+G>``, the preconditioner's share of the term."""
        columns = self.wfcU[ik]
        return jnp.real(
            jnp.einsum("gi,ij,gj->g", jnp.conj(columns), self.vns.astype(columns.dtype), columns)
        )
