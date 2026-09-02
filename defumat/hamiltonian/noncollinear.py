"""The Kohn-Sham Hamiltonian of a two-component spinor.

Everything a noncollinear calculation changes about ``H|psi>`` is here. A state
is no longer a function of ``G`` but a pair of them,

    |psi> = ( psi_up(G), psi_down(G) ),

stored as one vector of length ``2 npwx`` exactly as QE stores it
(``evc(npwx*npol, nbnd)``), so that the eigensolvers see a larger vector space
and not a new kind of problem. The three parts of the operator each grow a spin
structure, and each grows a different one:

* **kinetic** is spin-independent -- the same ``|k+G|^2`` on both components;
* **local** becomes a 2x2 matrix at every point of the grid,
  ``V(r) = v_0(r) I + m(r) . sigma`` (``vloc_psi_nc``). When the calculation
  carries no magnetization it collapses back to a multiple of the identity, and
  then the only thing that has changed from a collinear run is that there are
  two components to multiply;
* **nonlocal** is where spin-orbit coupling lives. ``D_ij`` becomes a 2x2 matrix
  in spin space, complex and *not* diagonal, built in
  :mod:`defumat.pseudo.spinorbit` from the ``j``-resolved projectors
  (``add_vuspsi_nc``, ``newd_so``). The overlap operator ``S`` gains the same
  structure through ``qq_so`` (``s_psi_nc``).

The last of these is the whole physics: a spinor Hamiltonian with a
spin-diagonal ``D`` describes two independent copies of the same scalar problem
and every eigenvalue comes out doubly degenerate. It is the off-diagonal blocks
of ``D`` that mix the components and split the bands.

**Why a separate class.** The collinear :class:`~defumat.hamiltonian.operator.Hamiltonian`
is the hot path of every calculation this code does, and threading ``npol``
through it would put a branch in the middle of it. The two present the same
surface to the eigensolvers instead -- ``apply``, ``apply_s``, ``ndim``,
``state_mask``, ``diagonal``, ``s_projections`` -- and neither knows the other
exists.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from defumat.basis.fft import g_to_r, gather_from_box, r_to_sticks, sticks_to_r
from defumat.pseudo.projectors import Projectors

__all__ = ["SpinorHamiltonian"]


class SpinorHamiltonian(eqx.Module):
    """A noncollinear Kohn-Sham Hamiltonian at fixed potential.

    ``potential`` is ``(nspin_mag, n1, n2, n3)`` in Ry: one component when the
    calculation carries no magnetization, four -- ``(v_0, m_x, m_y, m_z)`` -- when
    it does.
    """

    kinetic: jnp.ndarray  # (nk, npwx), Ry
    potential: jnp.ndarray  # (nspin_mag, n1, n2, n3), Ry, real
    fft_index: jnp.ndarray  # (nk, npwx)
    mask: jnp.ndarray  # (nk, npwx)
    projectors: Projectors
    #: ``deeq_nc``: ``(2, 2, nkb, nkb)`` complex, block diagonal over atoms.
    #: Present always, unlike the collinear ``deeq``, because even a
    #: norm-conserving spin-orbit run has a genuinely 2x2 ``D`` -- ``dvan_so`` --
    #: with nothing to fall back on.
    deeq: jnp.ndarray
    grid: tuple[int, int, int] = eqx.field(static=True)
    #: A **spin spiral** (P19): the two spinor components live on different
    #: plane-wave spheres, ``k + q/2`` and ``k - q/2``, so every k-indexed array
    #: here has ``2 nk`` rows -- the up component's ``nk`` first, the down
    #: component's after them -- rather than ``nk``. False is the ordinary case,
    #: where both components share one sphere and the arrays have ``nk`` rows.
    #: Static, because it decides an array's *length*.
    spiral: bool = eqx.field(static=True, default=False)
    sticks: object = None
    potential_wave: jnp.ndarray | None = None
    resolves_differences: bool = eqx.field(static=True, default=True)
    #: ``qq_so``: ``(2, 2, nkb, nkb)``. ``None`` for a norm-conserving
    #: calculation, where ``S`` is the identity.
    qq: jnp.ndarray | None = None

    @property
    def nk(self) -> int:
        rows = self.kinetic.shape[0]
        return rows // 2 if self.spiral else rows

    def _rows(self, ik):
        """The rows of the k-indexed arrays the two components read.

        The same row twice in the ordinary case -- the two components share a
        sphere, a kinetic energy and a set of projectors -- and two different
        ones for a spin spiral, where they do not.
        """
        return (ik, ik + self.nk) if self.spiral else (ik, ik)

    def _pair(self, array, ik):
        """``array[ik]``, or the two components' rows stacked, for a spiral.

        Shaped so that it broadcasts against ``(..., 2, npwx)`` either way:
        ``(npwx,)`` when the components share a row and ``(2, npwx)`` when they
        do not.
        """
        if not self.spiral:
            return array[ik]
        return jnp.stack([array[ik], array[ik + self.nk]])

    @property
    def npwx(self) -> int:
        return self.kinetic.shape[1]

    @property
    def npol(self) -> int:
        return 2

    @property
    def ndim(self) -> int:
        return 2 * self.npwx

    @property
    def nspin_mag(self) -> int:
        return self.potential.shape[0]

    @property
    def dtype(self):
        return self.projectors.vkb.dtype

    @property
    def has_overlap(self) -> bool:
        return self.qq is not None

    @property
    def state_mask(self) -> jnp.ndarray:
        return self._as_state(self.mask)

    @property
    def state_kinetic(self) -> jnp.ndarray:
        return self._as_state(self.kinetic)

    def _as_state(self, array: jnp.ndarray) -> jnp.ndarray:
        """A ``(nk, npwx)`` (or ``(2 nk, npwx)``) array as ``(nk, 2 npwx)``.

        The solvers index this by k-point and expect one row per spinor, so the
        spiral's two blocks of rows are concatenated along the plane-wave axis
        rather than left stacked.
        """
        if not self.spiral:
            return jnp.concatenate([array, array], axis=-1)
        nk = self.nk
        return jnp.concatenate([array[:nk], array[nk:]], axis=-1)

    # --- the operator ---------------------------------------------------------

    def _split(self, psi: jnp.ndarray, ik: int) -> jnp.ndarray:
        """``(..., 2 npwx)`` -> ``(..., 2, npwx)``, with the padding zeroed.

        The two components are stored one after the other, which is QE's
        ``evc(1:npw)`` / ``evc(npwx+1:npwx+npw)`` layout -- each component padded
        to ``npwx`` in its own right, so the same mask applies to both.
        """
        components = psi.reshape(psi.shape[:-1] + (2, self.npwx))
        return jnp.where(self._pair(self.mask, ik), components, 0.0)

    @staticmethod
    def _join(components: jnp.ndarray) -> jnp.ndarray:
        """``(..., 2, npwx)`` -> ``(..., 2 npwx)``."""
        return components.reshape(components.shape[:-2] + (2 * components.shape[-1],))

    def apply(self, psi: jnp.ndarray, ik: int) -> jnp.ndarray:
        """``H|psi>`` for ``psi`` of shape ``(..., 2 npwx)``."""
        components = self._split(psi, ik)
        result = self._pair(self.kinetic, ik) * components
        result = result + self._local(components, ik)
        result = result + self._nonlocal(components, ik)
        return self._join(jnp.where(self._pair(self.mask, ik), result, 0.0))

    def apply_s(self, psi: jnp.ndarray, ik: int) -> jnp.ndarray:
        """``S|psi>``; the identity for a norm-conserving calculation."""
        components = self._split(psi, ik)
        if not self.has_overlap:
            return self._join(jnp.where(self._pair(self.mask, ik), components, 0.0))
        becp = self._project(components, ik)
        becq = jnp.einsum("abij,...bj->...ai", self.qq.astype(self.dtype), becp)
        result = components + self._unproject(becq, ik)
        return self._join(jnp.where(self._pair(self.mask, ik), result, 0.0))

    def _project(self, components: jnp.ndarray, ik: int) -> jnp.ndarray:
        """``<beta_i|psi^a>``, shaped ``(..., 2, nkb)``.

        For a spiral the two components are projected on *different* projectors
        -- ``vkb(k + q/2)`` and ``vkb(k - q/2)`` -- which is the only way the
        nonlocal term differs from the ordinary noncollinear one.
        """
        if not self.spiral:
            return jnp.einsum(
                "gk,...ag->...ak", self.projectors.vkb[ik].conj(), components
            )
        up, down = self._rows(ik)
        vkb = jnp.stack([self.projectors.vkb[up], self.projectors.vkb[down]])
        return jnp.einsum("agk,...ag->...ak", vkb.conj(), components)

    def _unproject(self, coefficients: jnp.ndarray, ik: int) -> jnp.ndarray:
        """``sum_i |beta_i> c^a_i``, shaped ``(..., 2, npwx)``."""
        if not self.spiral:
            return jnp.einsum("gk,...ak->...ag", self.projectors.vkb[ik], coefficients)
        up, down = self._rows(ik)
        vkb = jnp.stack([self.projectors.vkb[up], self.projectors.vkb[down]])
        return jnp.einsum("agk,...ak->...ag", vkb, coefficients)

    def _local(self, components: jnp.ndarray, ik: int) -> jnp.ndarray:
        """``V(r) psi`` with ``V`` a 2x2 matrix at each point of the grid.

        ``vloc_psi_nc``. The two components are transformed independently -- the
        FFT knows nothing about spin -- and mixed pointwise in between, which is
        the only place in ``H|psi>`` where the magnetization enters at all.
        """
        up, down = self._rows(ik)
        if self.sticks is None:
            if not self.spiral:
                field = g_to_r(components, self.fft_index[ik], self.grid)
            else:
                field = jnp.stack([
                    g_to_r(components[..., 0, :], self.fft_index[up], self.grid),
                    g_to_r(components[..., 1, :], self.fft_index[down], self.grid),
                ], axis=-4)
            product = self._multiply(field, self.potential)
            n = self.grid[0] * self.grid[1] * self.grid[2]
            box = jnp.fft.fftn(product, axes=(-3, -2, -1)) / n
            if not self.spiral:
                return gather_from_box(box, self.fft_index[ik])
            return jnp.stack([
                gather_from_box(box[..., 0, :, :, :], self.fft_index[up]),
                gather_from_box(box[..., 1, :, :, :], self.fft_index[down]),
            ], axis=-2)

        if not self.spiral:
            columns, index = self.sticks.columns[ik], self.sticks.index[ik]
            field = sticks_to_r(components, self.sticks, columns, index)
            product = self._multiply(field, self.potential_wave)
            return r_to_sticks(product, self.sticks, columns, index)

        # A spiral transforms each component with its own stick layout. The
        # layouts are rows of one ``Sticks`` built over the concatenated k-list,
        # so they share ``nsticks`` -- which is what keeps the compiled shapes
        # the same for both components.
        field = jnp.stack([
            sticks_to_r(components[..., 0, :], self.sticks,
                        self.sticks.columns[up], self.sticks.index[up]),
            sticks_to_r(components[..., 1, :], self.sticks,
                        self.sticks.columns[down], self.sticks.index[down]),
        ], axis=-4)
        product = self._multiply(field, self.potential_wave)
        return jnp.stack([
            r_to_sticks(product[..., 0, :, :, :], self.sticks,
                        self.sticks.columns[up], self.sticks.index[up]),
            r_to_sticks(product[..., 1, :, :, :], self.sticks,
                        self.sticks.columns[down], self.sticks.index[down]),
        ], axis=-2)

    @staticmethod
    def _multiply(field: jnp.ndarray, potential: jnp.ndarray) -> jnp.ndarray:
        """``sum_b V_{ab}(r) psi^b(r)`` for ``field`` shaped ``(..., 2, ...grid)``.

        With one potential component this is a scalar multiplication of both
        spinor components -- the ``.NOT. domag`` branch of ``vloc_psi_nc``, and
        the case a nonmagnetic spin-orbit calculation is in.
        """
        if potential.shape[0] == 1:
            return field * potential[0]
        v0, mx, my, mz = potential[0], potential[1], potential[2], potential[3]
        up, down = field[..., 0, :, :, :], field[..., 1, :, :, :]
        return jnp.stack(
            [
                up * (v0 + mz) + down * (mx - 1j * my),
                down * (v0 - mz) + up * (mx + 1j * my),
            ],
            axis=-4,
        )

    def _nonlocal(self, components: jnp.ndarray, ik: int) -> jnp.ndarray:
        """``sum_{ab,ij} |beta_i a> D^{ab}_ij <beta_j b|psi>``.

        ``add_vuspsi_nc``: four matrix products where the collinear case has
        one, and the two off-diagonal ones are what spin-orbit coupling adds.
        """
        if self.projectors.nkb == 0:
            return jnp.zeros_like(components)
        becp = self._project(components, ik)
        ps = jnp.einsum("abij,...bj->...ai", self.deeq.astype(self.dtype), becp)
        return self._unproject(ps, ik)

    # --- what the solvers ask for --------------------------------------------

    def s_projections(self, vectors: jnp.ndarray, ik: int):
        """``(<beta|psi>, q <beta|psi>)``, both flattened over ``(spin, channel)``.

        Flattening is what lets Davidson keep treating them as plain matrices:
        the rotation ``coefficients.T @ becq`` and the overlap block
        ``becp^H becq`` are the same single matrix products they are in the
        collinear case, with the spin sum folded into the contracted index.
        """
        components = self._split(vectors, ik)
        becp = self._project(components, ik)
        if not self.has_overlap:
            flat = becp.reshape(becp.shape[:-2] + (2 * becp.shape[-1],))[..., :0]
            return flat, flat
        becq = jnp.einsum("abij,...bj->...ai", self.qq.astype(self.dtype), becp)
        return (
            becp.reshape(becp.shape[:-2] + (2 * becp.shape[-1],)),
            becq.reshape(becq.shape[:-2] + (2 * becq.shape[-1],)),
        )

    def s_correction(self, becq: jnp.ndarray, ik: int) -> jnp.ndarray:
        """``(S - 1)|psi>`` from the stored ``q <beta|psi>``."""
        if not self.has_overlap:
            return jnp.zeros(becq.shape[:-1] + (self.ndim,), dtype=self.dtype)
        coefficients = becq.reshape(becq.shape[:-1] + (2, becq.shape[-1] // 2))
        return self._join(self._unproject(coefficients, ik))

    def diagonal(self, ik: int) -> jnp.ndarray:
        """``<k+G a|H|k+G a>``, the preconditioner's ``h_diag``, ``(2 npwx,)``.

        ``usnldiag_nc``: the *diagonal* spin blocks of ``D`` only, and the
        charge component of the local potential -- the preconditioner does not
        have to be the operator, and the off-diagonal blocks would cost as much
        as they are worth.
        """
        rows = self._rows(ik)
        average = jnp.mean(self.potential[0])
        blocks = [self.kinetic[row] + average for row in rows]
        if self.projectors.nkb:
            for spin, row in enumerate(rows):
                vkb = self.projectors.vkb[row]
                d = self.deeq[spin, spin].astype(self.dtype)
                blocks[spin] = blocks[spin] + jnp.real(
                    jnp.einsum("gi,ij,gj->g", vkb, d, vkb.conj())
                )
        return jnp.concatenate(
            [jnp.where(self.mask[row], b, 0.0) for row, b in zip(rows, blocks)]
        )

    def overlap_diagonal(self, ik: int) -> jnp.ndarray:
        """``<k+G a|S|k+G a>``, ``s_diag`` in ``usnldiag_nc``."""
        if not self.has_overlap:
            return jnp.where(self.state_mask[ik], 1.0, 0.0)
        blocks = []
        for spin, row in enumerate(self._rows(ik)):
            vkb = self.projectors.vkb[row]
            q = self.qq[spin, spin].astype(self.dtype)
            value = 1.0 + jnp.real(jnp.einsum("gi,ij,gj->g", vkb, q, vkb.conj()))
            blocks.append(jnp.where(self.mask[row], value, 0.0))
        return jnp.concatenate(blocks)

    # --- explicit matrices, for the reference dense solver --------------------

    def matrix(self, ik: int) -> jnp.ndarray:
        """The Hamiltonian as a ``(2 npwx, 2 npwx)`` matrix.

        Assembled block by block from the same three terms
        :meth:`apply` applies, using the gather of ``V(G - G')`` that
        :meth:`defumat.hamiltonian.operator.Hamiltonian.matrix` documents. The
        block structure is the only new part: the local term contributes
        ``V_{ab}(G - G')`` to block ``(a, b)`` and the kinetic term only to the
        diagonal ones.
        """
        if not self.resolves_differences:
            return self.matrix_by_application(ik)

        n1, n2, n3 = self.grid
        n = n1 * n2 * n3
        component_rows = self._rows(ik)

        def unpack(index):
            return index // (n2 * n3), (index // n3) % n2, index % n3

        def difference(row_index, column_index):
            """``G - G'`` as a flat box index, between two (possibly different) spheres."""
            a, b, c = unpack(self.fft_index[row_index])
            d, e, f = unpack(self.fft_index[column_index])
            return (
                (jnp.mod(a[:, None] - d[None, :], n1) * n2
                 + jnp.mod(b[:, None] - e[None, :], n2)) * n3
                + jnp.mod(c[:, None] - f[None, :], n3)
            )

        transformed = [
            jnp.fft.fftn(component, axes=(-3, -2, -1)).reshape(n) / n
            for component in self.potential
        ]

        def local_block(row, column):
            """``V_{ab}(G - G')`` for the ``(row, column)`` spin block."""
            offsets = difference(component_rows[row], component_rows[column])
            if self.nspin_mag == 1:
                return transformed[0][offsets] if row == column else None
            v0, mx, my, mz = (t[offsets] for t in transformed)
            return [[v0 + mz, mx - 1j * my], [mx + 1j * my, v0 - mz]][row][column]

        blocks = []
        for row, row_index in enumerate(component_rows):
            columns = []
            for column, column_index in enumerate(component_rows):
                block = local_block(row, column)
                block = jnp.zeros((self.npwx, self.npwx), self.dtype) if block is None \
                    else block.astype(self.dtype)
                if row == column:
                    block = block + jnp.diag(self.kinetic[row_index].astype(self.dtype))
                if self.projectors.nkb:
                    d = self.deeq[row, column].astype(self.dtype)
                    block = block + (
                        self.projectors.vkb[row_index] @ d
                        @ self.projectors.vkb[column_index].conj().T
                    )
                pair = self.mask[row_index][:, None] & self.mask[column_index][None, :]
                columns.append(jnp.where(pair, block, 0.0))
            blocks.append(jnp.concatenate(columns, axis=1))
        matrix = jnp.concatenate(blocks, axis=0)
        return 0.5 * (matrix + matrix.conj().T)

    def matrix_by_application(self, ik: int) -> jnp.ndarray:
        """The Hamiltonian built by applying it to every basis vector."""
        identity = jnp.eye(self.ndim, dtype=self.dtype)
        matrix = self.apply(identity, ik).T
        return 0.5 * (matrix + matrix.conj().T)

    def overlap_matrix(self, ik: int) -> jnp.ndarray:
        """``S`` as a ``(2 npwx, 2 npwx)`` matrix, identity on the padding."""
        identity = jnp.eye(self.ndim, dtype=self.dtype)
        if not self.has_overlap:
            return identity
        component_rows = self._rows(ik)
        blocks = []
        for row, row_index in enumerate(component_rows):
            columns = []
            for column, column_index in enumerate(component_rows):
                q = self.qq[row, column].astype(self.dtype)
                block = (
                    self.projectors.vkb[row_index] @ q
                    @ self.projectors.vkb[column_index].conj().T
                )
                pair = self.mask[row_index][:, None] & self.mask[column_index][None, :]
                columns.append(jnp.where(pair, block, 0.0))
            blocks.append(jnp.concatenate(columns, axis=1))
        correction = jnp.concatenate(blocks, axis=0)
        return identity + 0.5 * (correction + correction.conj().T)
