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

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from defumat.basis.fft import (
    g_to_r, g_to_r_gamma, gamma_inner, gather_from_box, r_to_sticks, sticks_local,
    sticks_to_r,
)
from defumat.batching import map_bands
from defumat.pseudo.projectors import Projectors

__all__ = ["Hamiltonian", "to_planes", "from_planes", "planes_inner", "planes_norm2",
           "planes_zero_term", "planes_force_real_g0", "twice"]


# --- half-sphere states as real planes -----------------------------------------
#
# ``regterg`` and ``calbec_gamma`` treat a gamma-point state as a *real* array of
# length ``2 npw`` and contract it with DGEMM, two real multiply-adds an element,
# where a complex product spends four on a result whose imaginary half is
# discarded (``regterg.f90:204``, ``:318-338``, ``:403-428``, ``:511``;
# ``becmod.f90:236-241``; ``add_vuspsi.f90:115``). XLA has no free view of a
# complex buffer as reals -- ``bitcast_convert_type`` refuses complex operands and
# ``ndarray.view`` lowers to two scatters, a copy -- so the gamma eigensolver
# carries its states in this layout for the whole solve instead: the real plane
# in the first ``npwx`` entries and the imaginary plane in the next ``npwx``.
# ``G = 0`` is then entries ``0`` and ``npwx``. Interleaving the two would put the
# contraction on two axes and cost a transpose before every product.


def to_planes(states: jnp.ndarray) -> jnp.ndarray:
    """``(..., n)`` complex to ``(..., 2n)`` real: the real plane, then the imaginary one."""
    return jnp.concatenate([states.real, states.imag], axis=-1)


def from_planes(planes: jnp.ndarray) -> jnp.ndarray:
    """The inverse of :func:`to_planes`: ``(..., 2n)`` real to ``(..., n)`` complex."""
    n = planes.shape[-1] // 2
    return jax.lax.complex(planes[..., :n], planes[..., n:])


def twice(values: jnp.ndarray) -> jnp.ndarray:
    """A real ``(..., n)`` quantity laid out over both planes, ``(..., 2n)``.

    A mask, a kinetic energy or a preconditioner's diagonal multiplies the real
    and the imaginary part of a coefficient alike.
    """
    return jnp.concatenate([values, values], axis=-1)


def planes_zero_term(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """``Re(conj(a_0) b_0)`` on planes, ``a_0 b_0`` summed over both planes, as an outer form.

    ``a`` is ``(..., 2n)`` and ``b`` ``(m, 2n)``; the result is ``(..., m)``.
    **Both planes, where ``regterg``'s ``MYDGER`` subtracts the real one alone**
    and relies on ``Im psi(1) = 0`` (``regterg.f90:174``, ``:375``). The two
    agree exactly while :func:`planes_force_real_g0` holds, which it does for
    every vector the solver stores; the form kept is the one the complex code
    applied, ``Re(conj(a_0) b_0)``, so that nothing but the representation moves.
    """
    n = a.shape[-1] // 2
    return a[..., :1] * b[:, :1].T + a[..., n:n + 1] * b[:, n:n + 1].T


def planes_inner(rows: jnp.ndarray, columns: jnp.ndarray) -> jnp.ndarray:
    """``<rows_i|columns_j>`` over a half sphere held as planes, ``(..., m)`` real.

    ``2 sum_stored - (G = 0)``, the rule :func:`~defumat.basis.fft.gamma_inner`
    applies to complex coefficients, written as one real product over ``2 npwx``
    -- ``regterg``'s ``DGEMM('T', 'N', ..., npw2, 2.D0, ...)`` followed by its
    ``MYDGER(..., -1.D0, ...)``.
    """
    return 2.0 * (rows @ columns.T) - planes_zero_term(rows, columns)


def planes_norm2(planes: jnp.ndarray) -> jnp.ndarray:
    """``<x|x>`` per row of a planes block, ``(..., 1)``: ``regterg.f90:360-361``."""
    n = planes.shape[-1] // 2
    zero = planes[..., :1] * planes[..., :1] + planes[..., n:n + 1] * planes[..., n:n + 1]
    return 2.0 * jnp.sum(planes * planes, axis=-1, keepdims=True) - zero


def planes_force_real_g0(planes: jnp.ndarray) -> jnp.ndarray:
    """:func:`~defumat.basis.fft.force_real_g0` on planes: ``Im c(0)``, entry ``npwx``, set to zero.

    A select rather than a scatter, for the reason ``force_real_g0`` gives: it
    sits inside an elementwise chain that a scatter would split.
    """
    n = planes.shape[-1] // 2
    return jnp.where(jnp.arange(planes.shape[-1]) == n, 0.0, planes)


def smallest_sphere(npw) -> int | None:
    """The one plane-wave count an operator keeps: ``min_k npw``, or ``None``.

    The converter of both operators' ``npw`` field. A per-k list is reduced to
    its minimum, because that minimum is all the eigensolver reads, and an
    empty list or ``None`` -- a chunk of a force pass carries ``()`` -- is
    ``None``, which leaves the bound at the padded width.
    """
    if npw is None:
        return None
    if isinstance(npw, (int, np.integer)):
        return int(npw)
    npw = tuple(npw)
    return int(min(npw)) if npw else None


def conjugated_contraction(projectors, states, subscripts: str) -> jnp.ndarray:
    """``einsum(subscripts, conj(projectors), states)``, conjugating the smaller operand.

    ``<beta|psi>`` needs one of its two operands conjugated, and conjugating
    ``vkb`` materialises an ``(npwx, nkb)`` copy of it at every call: XLA does
    not fold an elementwise op into a dot's operand on a CPU, and in the
    compiled Davidson the copy sits inside the loop body, once for each rung of
    the band ladder (``OPEN.md`` Part III M4).
    ``conj(einsum(projectors, conj(states)))`` is the same number **to the last
    bit**: each product in the sum has both factors' imaginary parts negated,
    which negates its imaginary part exactly and leaves its real part alone,
    the operands keep their places in the dot, and round-to-nearest is
    symmetric under negation. So which one is conjugated is decided by size
    alone, from static shapes: the band block where it is the smaller -- a
    Davidson block of at most ``nbnd`` states against ``nkb`` projector
    channels -- and the projectors where they are.
    """
    if math.prod(states.shape) < math.prod(projectors.shape):
        return jnp.einsum(subscripts, projectors, states.conj()).conj()
    return jnp.einsum(subscripts, projectors.conj(), states)


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
    #: The DFT+U term, ``sum |phi> v_ns <phi|`` (``PW/src/vhpsi.f90``), or
    #: ``None`` when no species carries a Hubbard U. It is separable in exactly
    #: the way the nonlocal term is and enters in the same three places --
    #: :meth:`apply`, :meth:`matrix` and :meth:`diagonal`. Leaving it out of
    #: :meth:`matrix` alone would make the dense reference fixture solve a
    #: *different* Hamiltonian from the one Davidson solves.
    hubbard: object | None = None
    #: ``(nk, npwx)``: the flat index of ``-(k+G)`` -- QE's ``nlm`` -- and
    #: ``None`` unless this is a gamma-only run. Its presence *is* the switch:
    #: see :attr:`gamma_only`.
    fft_index_minus: jnp.ndarray | None = None
    #: How many plane waves the *smallest* k-point's sphere holds, ``min_k
    #: npw``, as against ``npwx``, which is the padded maximum over k. **Static,
    #: because it bounds an array's length**: the Davidson subspace cannot be
    #: larger than the smallest space it is built in, and the k-points that go
    #: singular are precisely the ones *below* ``npwx`` -- see
    #: :func:`~defumat.solvers.davidson.davidson_eigensolver_all`. ``None``
    #: leaves the bound at ``npwx``, which is QE's own ``ipw``
    #: (``c_bands.f90:286``) and is what a Hamiltonian built without its basis
    #: gets.
    #:
    #: **One number and not the per-k list**, which is what it held until
    #: ``OPEN.md`` Part XXIII item 9: a static field is part of the treedef, so
    #: two spheres with the same padded width and the same smallest sphere --
    #: two wavevectors of a spin-spiral scan, typically -- were two treedefs,
    #: and the Davidson solve and the Rayleigh-Ritz start compiled again at
    #: every wavevector for counts nothing read. A list passed in is reduced to
    #: its minimum by :func:`smallest_sphere`.
    npw: int | None = eqx.field(static=True, default=None, converter=smallest_sphere)
    #: How many bands :meth:`_local` puts through the grid at once --
    #: :func:`~defumat.batching.map_bands`'s dial, carried here so that the
    #: value the :class:`~defumat.scf.driver.Calculation` resolved (from the
    #: card, in memory mode) is the one every ``h_psi`` uses. **Static**,
    #: because it sets the shape of the block the transform is compiled for;
    #: ``"default"`` defers to the environment and the platform, which is what
    #: a Hamiltonian built outside a calculation gets.
    band_batch: int | None | str = eqx.field(static=True, default="default")
    #: How many ``z`` planes :meth:`_local` takes through its ``xy`` transforms
    #: and product at once on the stick path
    #: (:func:`~defumat.basis.fft.sticks_local`), or ``None`` for the whole box
    #: (:func:`~defumat.batching.resolve_plane_chunk`). **Static**, since it is a
    #: shape. ``None`` by default, so that a Hamiltonian built outside a
    #: calculation keeps the whole-box path it always had.
    plane_chunk: int | None = eqx.field(static=True, default=None)

    @property
    def gamma_only(self) -> bool:
        """Whether states are stored on the half sphere (QE's ``gamma_only``).

        At ``k = 0`` a state can be chosen real, so ``c(-G) = conj(c(G))`` and
        half the sphere carries all of it. What that changes here is three
        things and no more: the field is rebuilt from both halves
        (:meth:`_local`), every plane-wave sum doubles and corrects ``G = 0``
        (:func:`~defumat.basis.fft.gamma_inner`), and ``Im c(0)`` must stay zero.
        """
        return self.fft_index_minus is not None

    @property
    def nk(self) -> int:
        return self.kinetic.shape[0]

    @property
    def npwx(self) -> int:
        return self.kinetic.shape[1]

    @property
    def npol(self) -> int:
        """Spinor components per state: one. See :mod:`defumat.hamiltonian.noncollinear`."""
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
    def space(self) -> int:
        """The smallest space a state is solved in, ``npol * min_k npw``.

        What the Davidson subspace has to fit inside. It is the *minimum* over
        k rather than ``npwx`` because a shell sitting on the cutoff makes some
        k-points hold fewer plane waves than others, and those are the ones an
        oversized subspace goes singular at: on silicon at ``ecutwfc = 12``
        folded to a 32 k-point ultracell the spheres run from 169 to 192, so a
        bound at ``npwx`` would leave every k-point below 192 oversubscribed.
        :attr:`npw` already holds that minimum.
        """
        if self.npw is None:
            return self.ndim
        return self.npol * self.npw

    @property
    def dtype(self):
        return self.projectors.dtype

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

    def _becp(self, vectors: jnp.ndarray, vkb: jnp.ndarray) -> jnp.ndarray:
        """``<beta|psi>`` for a block of states -- ``calbec``.

        ``(..., npwx) x (npwx, nkb) -> (..., nkb)``.

        **The gamma branch is ``calbec_gamma``** (``Modules/becmod.f90:321``):
        ``DGEMM`` with a factor ``2`` over the real and imaginary parts as one
        real vector, then ``betapsi -= beta(1,:) psi(1,:)`` for the rank that
        holds ``G = 0``. Both ``beta(r)`` and ``psi(r)`` are real there, so the
        result is real -- it is cast back to the complex dtype only because the
        reconstruction below multiplies ``vkb``.

        The factor belongs to *this* sum and to nothing after it: rebuilding
        ``|beta> D <beta|psi>`` is an expansion in the stored basis, not a sum
        over it, and doubling it too is the classic way to get an energy that
        is nearly right.

        The conjugate goes on the smaller operand (:func:`conjugated_contraction`).
        """
        product = conjugated_contraction(vkb, vectors, "gk,...g->...k")
        if not self.gamma_only:
            return product
        zero = vkb[0].conj() * vectors[..., :1]
        return (2.0 * product.real - zero.real).astype(vkb.dtype)

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
            empty = self.projectors.at_k(ik)[:, :width]
            becp = vectors @ empty.conj()
            return becp, becp
        vkb = self.projectors.at_k(ik)
        becp = self._becp(vectors, vkb)
        return becp, becp @ self.projectors.qq.astype(vkb.dtype).T

    def s_correction(self, becq: jnp.ndarray, ik: int) -> jnp.ndarray:
        """``(S - 1)|psi>`` from the stored ``q <beta|psi>``."""
        if not self.has_overlap:
            return jnp.zeros(becq.shape[:-1] + (self.ndim,), dtype=self.dtype)
        return becq @ self.projectors.at_k(ik).T

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
        return self._applied(jnp.where(self.mask[ik], psi, 0.0), ik)

    def apply_projected(self, psi: jnp.ndarray, ik: int):
        """``(H|psi>, <beta|psi>, q <beta|psi>)`` from one ``calbec``.

        What :meth:`apply` and :meth:`s_projections` return, in one call, for
        the solver that needs both of the same block. ``h_psi`` computes
        ``becp`` once (``h_psi.f90:231``) and ``s_psi`` reads it
        (``s_psi.f90:15``); calling the two methods separately computed it
        twice, once on the masked block inside :meth:`_nonlocal` and once on
        the block as passed, and XLA cannot merge two products whose operands
        are different arrays even where their values are equal. Here the
        nonlocal term and the pair are built from the same ``becp``, taken on
        the masked block as :meth:`apply` takes it, so ``H|psi>`` is the same
        expression as :meth:`apply`'s and the pair is :meth:`s_projections`'
        whenever the block arrives masked, which every Davidson block does.

        Without an augmentation charge the pair is zero-width and nothing is
        shared: this is :meth:`apply` and :meth:`s_projections` as they were.
        """
        if not self.has_overlap:
            return (self.apply(psi, ik), *self.s_projections(psi, ik))
        psi = jnp.where(self.mask[ik], psi, 0.0)
        vkb = self.projectors.at_k(ik)
        becp = self._becp(psi, vkb)
        return (self._applied(psi, ik, becp), becp,
                becp @ self.projectors.qq.astype(vkb.dtype).T)

    # --- the gamma eigensolver's operator, on real planes -------------------------
    #
    # The same three products and the same transform as :meth:`apply_projected`,
    # :meth:`s_projections` and :meth:`s_correction`, for a half-sphere block held
    # as real planes (:func:`to_planes`). Every plane-wave contraction here is a
    # real product over ``2 npwx``; the complex block exists only per band chunk,
    # inside :meth:`_local_planes`, where the transform needs it.

    def _projector_planes(self, ik: int) -> jnp.ndarray:
        """``vkb`` as ``(2 npwx, nkb)`` real planes, ``calbec_gamma``'s view of ``beta``.

        **A copy of one ``vkb`` per call** where the projectors are stored, which
        they are on a CPU by default: there is no free real view of a complex
        buffer, and the store itself is ``pseudo/projectors.py``'s. Where they
        are rebuilt per call the planes are the build's own output.
        """
        vkb = self.projectors.at_k(ik)
        return jnp.concatenate([vkb.real, vkb.imag], axis=0)

    def _becp_planes(self, planes: jnp.ndarray, vkb: jnp.ndarray) -> jnp.ndarray:
        """``calbec_gamma`` on planes: ``2 psi^T beta - (G = 0)``, real ``(..., nkb)``.

        ``becmod.f90:236-241``: ``MYDGEMM('C', 'N', nkb, m, 2*npw, 2.0_DP, ...)``
        and ``MYDGER(..., -1.0_DP, beta, ..., psi, ...)``. The ``G = 0`` term is
        :meth:`_becp`'s, ``Re(conj(beta_0) psi_0)``, over both planes.
        """
        n = planes.shape[-1] // 2
        product = jnp.einsum("...g,gk->...k", planes, vkb)
        zero = planes[..., :1] * vkb[0] + planes[..., n:n + 1] * vkb[n]
        return 2.0 * product - zero

    def _real_coefficients(self, matrix: jnp.ndarray, dtype) -> jnp.ndarray:
        """A projector-space matrix in the planes' real dtype.

        ``D_ij`` and ``q_ij`` of a collinear run are real; the complex dtype the
        complex path casts them to carries a zero imaginary part.
        """
        return jnp.real(matrix).astype(dtype)

    def _nonlocal_planes(self, planes: jnp.ndarray, vkb: jnp.ndarray,
                         becp: jnp.ndarray) -> jnp.ndarray:
        """``sum_ij |beta_i> D_ij <beta_j|psi>`` on planes: ``add_vuspsi_gamma``.

        ``add_vuspsi.f90:115``: ``DGEMM('N', 'N', 2*n, m, nkb, 1.D0, vkb, ...)``,
        the expansion in the stored basis, which takes no factor of two.
        """
        dij = self._real_coefficients(self.coefficients, planes.dtype)
        return jnp.einsum("gk,...k->...g", vkb, becp @ dij.T)

    def _local_planes(self, planes: jnp.ndarray, ik: int) -> jnp.ndarray:
        """:meth:`_local`'s gamma branch on planes, the complex block rebuilt per band chunk.

        The coefficients are reassembled and split again *inside* the body
        :func:`~defumat.batching.map_bands` walks, so the only complex block is
        the one chunk in flight. Reassembling the whole block before the walk
        would materialise the ``(m, npwx)`` complex array the planes exist to
        replace. The transform is the one :meth:`_local` applies, so each band's
        product is the same number.
        """
        n = self.grid[0] * self.grid[1] * self.grid[2]
        minus = self.fft_index_minus[ik]

        def block_gamma(states):
            field = g_to_r_gamma(from_planes(states), self.fft_index[ik], minus, self.grid)
            box = jnp.fft.fftn(field * self.potential, axes=(-3, -2, -1)) / n
            return to_planes(gather_from_box(box, self.fft_index[ik]))

        return map_bands(block_gamma, planes, batch=self.band_batch)

    def _applied_planes(self, planes: jnp.ndarray, ik: int, vkb, becp) -> jnp.ndarray:
        """:meth:`_applied` on a masked planes block, in the same order of terms."""
        result = twice(self.kinetic[ik]) * planes
        result = result + self._local_planes(planes, ik)
        if self.projectors.nkb:
            result = result + self._nonlocal_planes(planes, vkb, becp)
        if self.hubbard is not None:
            # Not reached: a Hubbard run does not consume the half sphere
            # (``gamma_storage_is_consumable``). A round trip keeps it correct.
            result = result + to_planes(self.hubbard.apply(from_planes(planes), ik))
        return jnp.where(twice(self.mask[ik]), result, 0.0)

    def apply_projected_planes(self, planes: jnp.ndarray, ik: int):
        """:meth:`apply_projected` for a half-sphere block held as real planes.

        ``(H|psi>, <beta|psi>, q <beta|psi>)``, all real: ``H|psi>`` as planes
        ``(..., 2 npwx)`` and the projections as ``(..., nkb)``, or zero-width
        without an augmentation charge, as :meth:`apply_projected` returns them.
        One ``calbec`` serves the nonlocal term and the pair.
        """
        if not self.gamma_only:
            raise ValueError("planes are the half-sphere layout; this operator "
                             "stores the whole sphere")
        planes = jnp.where(twice(self.mask[ik]), planes, 0.0)
        if not self.projectors.nkb:
            return (self._applied_planes(planes, ik, None, None),
                    *self.s_projections_planes(planes, ik))
        vkb = self._projector_planes(ik)
        becp = self._becp_planes(planes, vkb)
        applied = self._applied_planes(planes, ik, vkb, becp)
        if not self.has_overlap:
            return (applied, *self.s_projections_planes(planes, ik))
        qq = self._real_coefficients(self.projectors.qq, planes.dtype)
        return applied, becp, becp @ qq.T

    def s_projections_planes(self, planes: jnp.ndarray, ik: int):
        """:meth:`s_projections` for a planes block: real, zero-width without ``S``."""
        if not self.has_overlap:
            empty = jnp.zeros(planes.shape[:-1] + (0,), planes.dtype)
            return empty, empty
        becp = self._becp_planes(planes, self._projector_planes(ik))
        qq = self._real_coefficients(self.projectors.qq, planes.dtype)
        return becp, becp @ qq.T

    def s_correction_planes(self, becq: jnp.ndarray, ik: int) -> jnp.ndarray:
        """:meth:`s_correction` on planes: ``(S - 1)|psi>`` from the stored ``q <beta|psi>``."""
        if not self.has_overlap:
            return jnp.zeros(becq.shape[:-1] + (2 * self.ndim,), dtype=becq.dtype)
        return becq @ self._projector_planes(ik).T

    def _applied(self, psi: jnp.ndarray, ik: int, becp=None) -> jnp.ndarray:
        """``H|psi>`` of an already masked block; ``becp`` is its ``<beta|psi>`` if known."""
        result = self.kinetic[ik] * psi
        result = result + self._local(psi, ik)
        result = result + self._nonlocal(psi, ik, becp)
        if self.hubbard is not None:
            result = result + self.hubbard.apply(psi, ik)
        return jnp.where(self.mask[ik], result, 0.0)

    def apply_s(self, psi: jnp.ndarray, ik: int) -> jnp.ndarray:
        """``S|psi>``. The identity for norm-conserving pseudopotentials."""
        psi = jnp.where(self.mask[ik], psi, 0.0)
        if not self.has_overlap:
            return psi
        vkb = self.projectors.at_k(ik)
        becp = self._becp(psi, vkb)
        qq = self.projectors.qq.astype(vkb.dtype)
        result = psi + jnp.einsum("gk,...k->...g", vkb, becp @ qq.T)
        return jnp.where(self.mask[ik], result, 0.0)

    def overlap_diagonal(self, ik: int) -> jnp.ndarray:
        """``<k+G|S|k+G>``, the preconditioner's ``s_diag`` (``usnldiag``)."""
        if not self.has_overlap:
            return jnp.where(self.mask[ik], 1.0, 0.0)
        vkb = self.projectors.at_k(ik)
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
        reason alone. :func:`defumat.batching.map_bands` is the dial, and it
        changes nothing but the order the transforms are issued in.
        """
        if self.gamma_only:
            # **The half sphere is rebuilt into a whole box and the product is
            # gathered back out of half of it.** ``V`` is real and ``psi(r)`` is
            # real, so ``V psi`` is real and its own coefficients obey
            # ``c(-G) = conj(c(G))`` -- which is why gathering the stored half
            # loses nothing and no factor appears here. The doubling belongs to
            # *sums* over the stored coefficients, never to the transform.
            #
            # The stick path is deliberately not taken: a half sphere occupies
            # half the columns and the conjugate fill needs the others, so
            # ``sticks_to_r`` would transform the wrong set. That is a speed
            # question, not a memory one, and this is the memory phase.
            n = self.grid[0] * self.grid[1] * self.grid[2]
            minus = self.fft_index_minus[ik]

            def block_gamma(states):
                field = g_to_r_gamma(states, self.fft_index[ik], minus, self.grid)
                box = jnp.fft.fftn(field * self.potential, axes=(-3, -2, -1)) / n
                return gather_from_box(box, self.fft_index[ik])

            return map_bands(block_gamma, psi, batch=self.band_batch)

        if self.sticks is None:
            n = self.grid[0] * self.grid[1] * self.grid[2]

            def block(states):
                field = g_to_r(states, self.fft_index[ik], self.grid)
                box = jnp.fft.fftn(field * self.potential, axes=(-3, -2, -1)) / n
                return gather_from_box(box, self.fft_index[ik])

            return map_bands(block, psi, batch=self.band_batch)

        columns, index = self.sticks.columns[ik], self.sticks.index[ik]

        if self.plane_chunk is not None:
            # The same round trip a chunk of ``z`` planes at a time, with the
            # product fused in, so that a band's box never streams through the
            # shared cache whole (:func:`~defumat.basis.fft.sticks_local`).
            def block(states):
                return sticks_local(states, self.sticks, columns, index,
                                    jnp.multiply, self.potential_wave, self.plane_chunk)

            return map_bands(block, psi, batch=self.band_batch)

        def block(states):
            field = sticks_to_r(states, self.sticks, columns, index)
            return r_to_sticks(field * self.potential_wave, self.sticks, columns, index)

        return map_bands(block, psi, batch=self.band_batch)

    def _nonlocal(self, psi: jnp.ndarray, ik: int, becp=None) -> jnp.ndarray:
        """``sum_ij |beta_i> D_ij <beta_j|psi>``, from ``becp`` when it is given."""
        if self.projectors.nkb == 0:
            return jnp.zeros_like(psi)
        vkb = self.projectors.at_k(ik)  # (npwx, nkb)
        if becp is None:
            becp = self._becp(psi, vkb)  # <beta|psi>
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
        if self.hubbard is not None:
            diagonal = diagonal + self.hubbard.diagonal(ik)
        if self.projectors.nkb:
            vkb = self.projectors.at_k(ik)
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
        identity = jnp.eye(self.npwx, dtype=self.projectors.dtype)
        if not self.has_overlap:
            return identity
        vkb = self.projectors.at_k(ik)
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
        identity = jnp.eye(self.npwx, dtype=self.projectors.dtype)
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
            vkb = self.projectors.at_k(ik)
            matrix = matrix + vkb @ self.coefficients.astype(vkb.dtype) @ vkb.conj().T
        if self.hubbard is not None:
            matrix = matrix + self.hubbard.matrix(ik)

        mask = self.mask[ik]
        matrix = jnp.where(mask[:, None] & mask[None, :], matrix, 0.0)
        return 0.5 * (matrix + matrix.conj().T)
