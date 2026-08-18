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
    #: Whether the density grid resolves every difference ``G - G'`` of two
    #: wavefunction plane waves, i.e. whether ``ecutrho >= 4 ecutwfc``. It is
    #: what makes :meth:`matrix` exact; see there.
    resolves_differences: bool = eqx.field(static=True, default=True)

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

    def diagonal(self, ik: int) -> jnp.ndarray:
        """``<k+G|H|k+G>``: the diagonal of the Hamiltonian, ``(npwx,)`` and real.

        This is what an iterative solver preconditions with -- QE builds it in
        ``usnldiag`` as ``|k+G|^2 + V(G=0)`` plus the diagonal of the nonlocal
        term. ``V(G=0)`` is the average of the local potential over the cell,
        which is what the mean of the grid values is.
        """
        diagonal = self.kinetic[ik] + jnp.mean(self.potential)
        if self.projectors.nkb:
            vkb = self.projectors.vkb[ik]
            dij = self.projectors.dij.astype(vkb.dtype)
            diagonal = diagonal + jnp.real(
                jnp.einsum("gi,ij,gj->g", vkb.conj(), dij, vkb)
            )
        return jnp.where(self.mask[ik], diagonal, 0.0)

    def matrix_by_application(self, ik: int) -> jnp.ndarray:
        """The Hamiltonian as an explicit matrix, built by applying it.

        Correct by construction and independent of any matrix-element formula,
        at the cost of ``npwx`` FFTs. That independence is the point: this is the
        reference that :meth:`matrix` -- which does use a formula -- is checked
        against, and through it the whole operator.
        """
        identity = jnp.eye(self.npwx, dtype=self.projectors.vkb.dtype)
        columns = self.apply(identity, ik)  # row b holds H e_b
        matrix = columns.T
        return 0.5 * (matrix + matrix.conj().T)

    def matrix(self, ik: int) -> jnp.ndarray:
        """The Hamiltonian as an explicit matrix, from its matrix elements.

        Same result as :meth:`matrix_by_application`, at a small fraction of the
        cost. The local part is a plain gather,

            <k+G_i| V |k+G_j> = V(G_i - G_j)

        so the whole matrix needs *one* FFT of the potential rather than one per
        basis vector -- which on the reference silicon cell is 180 FFTs replaced
        by 1, and is most of what a dense SCF iteration was spending its time on.

        The gather is exact only if the grid can represent every difference
        ``G_i - G_j`` without aliasing. Differences of two vectors inside the
        wavefunction sphere reach ``2 sqrt(ecutwfc)``, so the condition is
        ``ecutrho >= 4 ecutwfc`` -- which is exactly why QE's default dual is 4.
        When it does not hold, this falls back to applying the operator.
        """
        if not self.resolves_differences:
            return self.matrix_by_application(ik)

        n1, n2, n3 = self.grid
        n = n1 * n2 * n3
        potential_g = jnp.fft.fftn(self.potential, axes=(-3, -2, -1)).reshape(n) / n

        # Box coordinates of each plane wave, from the flat index it was stored
        # as. Their difference modulo the grid is the box coordinate of the
        # difference vector, which is the wrap the FFT layout already uses.
        index = self.fft_index[ik]
        a, b, c = index // (n2 * n3), (index // n3) % n2, index % n3
        difference = (
            (jnp.mod(a[:, None] - a[None, :], n1) * n2 + jnp.mod(b[:, None] - b[None, :], n2)) * n3
            + jnp.mod(c[:, None] - c[None, :], n3)
        )
        matrix = potential_g[difference]

        # Kinetic energy is diagonal, and the nonlocal part is separable: both
        # are matrices already, with no transform involved.
        matrix = matrix + jnp.diag(self.kinetic[ik].astype(matrix.dtype))
        if self.projectors.nkb:
            vkb = self.projectors.vkb[ik]
            matrix = matrix + vkb @ self.projectors.dij.astype(vkb.dtype) @ vkb.conj().T

        mask = self.mask[ik]
        matrix = jnp.where(mask[:, None] & mask[None, :], matrix, 0.0)
        return 0.5 * (matrix + matrix.conj().T)
