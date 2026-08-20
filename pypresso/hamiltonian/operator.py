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

``apply_s`` is the overlap operator,

    S = 1 + sum_{a,ij} |beta_i^a> q_ij^a <beta_j^a|

the identity for norm-conserving pseudopotentials and a genuine operator once a
species is ultrasoft, where ``q_ij`` is the integral of the augmentation charge
(``PW/src/s_psi.f90``). Everything downstream was written for the generalised
problem from the start (rule R5), so switching it on is a matter of ``qq``
ceasing to be ``None`` rather than of new call sites.

The nonlocal coefficients differ between the two cases in the same way. For a
norm-conserving potential ``D_ij`` is the file's, fixed for the run; for an
ultrasoft one it is ``D_ij^(0) + int V_eff Q_ij``, which depends on the
potential and therefore on the atom and on the SCF iteration. ``deeq`` carries
the rebuilt matrix when there is one, and the projectors' own ``dij`` is used
when there is not.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from pypresso.basis.fft import g_to_r, gather_from_box, r_to_sticks, sticks_to_r
from pypresso.batching import map_bands
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
    #: The stick layout, and the potential stored to match it -- see
    #: :meth:`_local`. ``None`` falls back to transforming the whole box.
    sticks: object = None
    potential_wave: jnp.ndarray | None = None
    #: Whether the density grid resolves every difference ``G - G'`` of two
    #: wavefunction plane waves, i.e. whether ``ecutrho >= 4 ecutwfc``. It is
    #: what makes :meth:`matrix` exact; see there.
    resolves_differences: bool = eqx.field(static=True, default=True)
    #: The nonlocal coefficients, when they are not the projectors' own -- i.e.
    #: ``deeq`` rebuilt by ``newd`` from the current potential. ``None`` means
    #: "use ``projectors.dij``", which is the norm-conserving case.
    deeq: jnp.ndarray | None = None

    @property
    def nk(self) -> int:
        return self.kinetic.shape[0]

    @property
    def npwx(self) -> int:
        return self.kinetic.shape[1]

    @property
    def npol(self) -> int:
        """Spinor components per state: one. See :mod:`pypresso.hamiltonian.noncollinear`."""
        return 1

    @property
    def ndim(self) -> int:
        """The dimension of the space a state lives in, ``npol * npwx``.

        The eigensolvers are written against this rather than against ``npwx``
        so that a spinor Hamiltonian -- whose states are twice as long -- is
        another operator rather than another solver.
        """
        return self.npwx

    @property
    def dtype(self):
        return self.projectors.vkb.dtype

    @property
    def state_mask(self) -> jnp.ndarray:
        """``(nk, ndim)``: which entries of a state vector are real basis functions."""
        return self.mask

    @property
    def state_kinetic(self) -> jnp.ndarray:
        """``(nk, ndim)``: ``|k+G|^2`` laid out like a state vector.

        Only the random starting guess uses it, to damp the high-kinetic
        components; it is a property so that a spinor state gets one copy per
        component without the solver knowing there are two.
        """
        return self.kinetic

    def s_projections(self, vectors: jnp.ndarray, ik: int):
        """``(<beta|psi>, q <beta|psi>)`` for a block of states, both flattened.

        The pair the Davidson subspace carries: ``becp`` builds the projected
        overlap and ``becq`` reconstructs ``S|psi>`` from a rotation of vectors
        already stored, so ``S`` is never applied to the whole subspace. Both
        come back as ``(nvec, m)`` matrices whatever the spin structure is, so
        the solver's rotations stay single matrix products.
        """
        if not self.has_overlap:
            width = 0
            empty = self.projectors.vkb[ik][:, :width]
            becp = vectors @ empty.conj()
            return becp, becp
        vkb = self.projectors.vkb[ik]
        becp = vectors @ vkb.conj()
        return becp, becp @ self.projectors.qq.astype(vkb.dtype).T

    def s_correction(self, becq: jnp.ndarray, ik: int) -> jnp.ndarray:
        """``(S - 1)|psi>`` from the stored ``q <beta|psi>``."""
        if not self.has_overlap:
            return jnp.zeros(becq.shape[:-1] + (self.ndim,), dtype=self.dtype)
        return becq @ self.projectors.vkb[ik].T

    @property
    def coefficients(self) -> jnp.ndarray:
        """``D_ij``: the rebuilt ultrasoft ones if present, the file's if not."""
        return self.projectors.dij if self.deeq is None else self.deeq

    @property
    def has_overlap(self) -> bool:
        """Whether ``S`` differs from the identity -- i.e. whether ``qq`` exists."""
        return self.projectors.qq is not None

    def apply(self, psi: jnp.ndarray, ik: int) -> jnp.ndarray:
        """``H|psi>`` for ``psi`` of shape ``(..., npwx)``."""
        psi = jnp.where(self.mask[ik], psi, 0.0)

        result = self.kinetic[ik] * psi
        result = result + self._local(psi, ik)
        result = result + self._nonlocal(psi, ik)
        return jnp.where(self.mask[ik], result, 0.0)

    def apply_s(self, psi: jnp.ndarray, ik: int) -> jnp.ndarray:
        """``S|psi>``. The identity for norm-conserving pseudopotentials."""
        psi = jnp.where(self.mask[ik], psi, 0.0)
        if not self.has_overlap:
            return psi
        vkb = self.projectors.vkb[ik]
        becp = jnp.einsum("gk,...g->...k", vkb.conj(), psi)
        qq = self.projectors.qq.astype(vkb.dtype)
        result = psi + jnp.einsum("gk,...k->...g", vkb, becp @ qq.T)
        return jnp.where(self.mask[ik], result, 0.0)

    def overlap_diagonal(self, ik: int) -> jnp.ndarray:
        """``<k+G|S|k+G>``, the preconditioner's ``s_diag`` (``usnldiag``)."""
        if not self.has_overlap:
            return jnp.where(self.mask[ik], 1.0, 0.0)
        vkb = self.projectors.vkb[ik]
        qq = self.projectors.qq.astype(vkb.dtype)
        diagonal = 1.0 + jnp.real(jnp.einsum("gi,ij,gj->g", vkb.conj(), qq, vkb))
        return jnp.where(self.mask[ik], diagonal, 0.0)

    def _local(self, psi: jnp.ndarray, ik: int) -> jnp.ndarray:
        """``V(r) psi``, evaluated by a round trip through the FFT grid.

        Two ways of doing the same thing. The stick path is QE's: the ``z``
        transform runs only over the columns the wavefunction sphere occupies
        (under a fifth of them) and the field is held with its ``xy`` plane
        contiguous so the 2D pass is cheap. The fallback transforms the whole
        box in one fused call, which is what everything did before the layout
        existed and is still what the dense-grid quantities use.

        **The bands are walked, not batched**, because a band's real-space box
        is the working set and a block of them is not -- ``vloc_psi_k``'s
        ``DO ibnd = 1, m`` is worth 2.5x on the sixteen-atom cell for that
        reason alone. :func:`pypresso.batching.map_bands` is the dial, and it
        changes nothing but the order the transforms are issued in.
        """
        if self.sticks is None:
            n = self.grid[0] * self.grid[1] * self.grid[2]

            def block(states):
                field = g_to_r(states, self.fft_index[ik], self.grid)
                box = jnp.fft.fftn(field * self.potential, axes=(-3, -2, -1)) / n
                return gather_from_box(box, self.fft_index[ik])

            return map_bands(block, psi)

        columns, index = self.sticks.columns[ik], self.sticks.index[ik]

        def block(states):
            field = sticks_to_r(states, self.sticks, columns, index)
            return r_to_sticks(field * self.potential_wave, self.sticks, columns, index)

        return map_bands(block, psi)

    def _nonlocal(self, psi: jnp.ndarray, ik: int) -> jnp.ndarray:
        """``sum_ij |beta_i> D_ij <beta_j|psi>``."""
        if self.projectors.nkb == 0:
            return jnp.zeros_like(psi)
        vkb = self.projectors.vkb[ik]  # (npwx, nkb)
        becp = jnp.einsum("gk,...g->...k", vkb.conj(), psi)  # <beta|psi>
        dij = self.coefficients.astype(vkb.dtype)
        return jnp.einsum("gk,...k->...g", vkb, becp @ dij.T)

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
            dij = self.coefficients.astype(vkb.dtype)
            diagonal = diagonal + jnp.real(
                jnp.einsum("gi,ij,gj->g", vkb.conj(), dij, vkb)
            )
        return jnp.where(self.mask[ik], diagonal, 0.0)

    def overlap_matrix(self, ik: int) -> jnp.ndarray:
        """``S`` as an explicit matrix, for the reference dense solve.

        The identity plus a rank-``nkb`` correction, so it is cheap however
        large ``npwx`` is. Padding rows and columns are left as the identity,
        which keeps the generalised problem positive definite -- a zero there
        would make the Cholesky factorisation fail rather than merely give a
        spurious eigenvalue.
        """
        mask = self.mask[ik]
        identity = jnp.eye(self.npwx, dtype=self.projectors.vkb.dtype)
        if not self.has_overlap:
            return identity
        vkb = self.projectors.vkb[ik]
        qq = self.projectors.qq.astype(vkb.dtype)
        correction = vkb @ qq @ vkb.conj().T
        correction = jnp.where(mask[:, None] & mask[None, :], correction, 0.0)
        return identity + 0.5 * (correction + correction.conj().T)

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
            matrix = matrix + vkb @ self.coefficients.astype(vkb.dtype) @ vkb.conj().T

        mask = self.mask[ik]
        matrix = jnp.where(mask[:, None] & mask[None, :], matrix, 0.0)
        return 0.5 * (matrix + matrix.conj().T)
