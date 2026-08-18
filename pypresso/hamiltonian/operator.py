"""The Kohn-Sham Hamiltonian applied to wavefunctions.

``H|psi>`` has three parts, following ``PW/src/h_psi.f90``:

* **kinetic**, diagonal in the plane-wave basis: ``|k+G|^2``;
* **local**, diagonal in real space: transform to the grid, multiply by
  ``V(r)``, transform back -- the reason a plane-wave code needs FFTs at all;
* **nonlocal**, a sum of separable projector terms.

This is the hot path: it is applied once per band per iteration of the
eigensolver, so it is the natural unit to ``jit`` and to ``vmap`` over bands and
k-points. Everything here accepts arbitrary leading axes on ``psi`` for exactly
that reason.

``apply_s`` is the overlap operator. For norm-conserving pseudopotentials it is
the identity, but it exists from the start (rule R5) because ultrasoft
pseudopotentials make it a genuine operator, and the eigensolver is written for
the generalised problem either way.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from pypresso.basis.fft import g_to_r, gather_from_box
from pypresso.pseudo.projectors import Projectors

__all__ = ["Hamiltonian"]


class Hamiltonian(eqx.Module):
    """A Kohn-Sham Hamiltonian at fixed potential.

    ``potential`` is the total local potential on the real-space grid, in Ry:
    the local pseudopotential plus Hartree plus exchange-correlation.
    """

    kinetic: jnp.ndarray  # (nk, npwx), Ry
    potential: jnp.ndarray  # (n1, n2, n3), Ry, real
    fft_index: jnp.ndarray  # (nk, npwx)
    mask: jnp.ndarray  # (nk, npwx)
    projectors: Projectors
    grid: tuple[int, int, int] = eqx.field(static=True)

    @property
    def nk(self) -> int:
        return self.kinetic.shape[0]

    @property
    def npwx(self) -> int:
        return self.kinetic.shape[1]

    def apply(self, psi: jnp.ndarray, ik: int) -> jnp.ndarray:
        """``H|psi>`` for ``psi`` of shape ``(..., npwx)``."""
        psi = jnp.where(self.mask[ik], psi, 0.0)

        result = self.kinetic[ik] * psi
        result = result + self._local(psi, ik)
        result = result + self._nonlocal(psi, ik)
        return jnp.where(self.mask[ik], result, 0.0)

    def apply_s(self, psi: jnp.ndarray, ik: int) -> jnp.ndarray:
        """``S|psi>``. The identity for norm-conserving pseudopotentials."""
        return jnp.where(self.mask[ik], psi, 0.0)

    def _local(self, psi: jnp.ndarray, ik: int) -> jnp.ndarray:
        """``V(r) psi``, evaluated by a round trip through the FFT grid."""
        field = g_to_r(psi, self.fft_index[ik], self.grid)
        product = field * self.potential
        n = self.grid[0] * self.grid[1] * self.grid[2]
        box = jnp.fft.fftn(product, axes=(-3, -2, -1)) / n
        return gather_from_box(box, self.fft_index[ik])

    def _nonlocal(self, psi: jnp.ndarray, ik: int) -> jnp.ndarray:
        """``sum_ij |beta_i> D_ij <beta_j|psi>``."""
        if self.projectors.nkb == 0:
            return jnp.zeros_like(psi)
        vkb = self.projectors.vkb[ik]  # (npwx, nkb)
        becp = jnp.einsum("gk,...g->...k", vkb.conj(), psi)  # <beta|psi>
        return jnp.einsum("gk,...k->...g", vkb, becp @ self.projectors.dij.T)

    def matrix(self, ik: int) -> jnp.ndarray:
        """The Hamiltonian as an explicit matrix, for reference diagonalization.

        Built by applying the operator to every basis vector: correct by
        construction and independent of any matrix-element formula, at the cost
        of ``npwx`` FFTs. Only usable for small systems -- which is the point,
        since it is the ground truth a fast eigensolver is checked against.
        """
        identity = jnp.eye(self.npwx, dtype=self.projectors.vkb.dtype)
        columns = self.apply(identity, ik)  # row b holds H e_b
        matrix = columns.T
        return 0.5 * (matrix + matrix.conj().T)
