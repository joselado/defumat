"""Bloch states on a k-list, and the one operation every invariant is built from.

Berry curvature, the Chern number, Wilson loops and the Fu-Kane parity products
are all built from a single primitive:

    M_mn(k, k') = <u_mk | S | u_nk'>

-- the overlap of the *cell-periodic* parts of two occupied manifolds, through
the pseudopotential's overlap operator. Everything downstream is determinants
and phases of that matrix, so this module is where the physics-specific work
lives and the rest of the subpackage is convention.

**Why overlaps and not ``jacfwd`` of H(k).** PLAN.md D2 says the velocity
operator should come from differentiating the Hamiltonian with respect to ``k``,
and for a *smooth* curvature it should -- that is the Kubo route registered as
``"kubo"`` in :mod:`defumat.topology.berry`. It is the wrong tool for an
invariant, for two independent reasons. First, D4: the Kubo expression divides
by ``E_n - E_m``, so it is singular exactly where crystals are degenerate by
symmetry, and it needs the eigenvectors band by band, which are not even
well-defined in a degenerate multiplet. The overlap route needs only the
*subspace*, through ``det M``, which is invariant under any unitary mixing
inside it. Second, quantisation: a lattice of link variables gives a Chern
number that is an **exact integer** on any mesh (Fukui, Hatsugai and Suzuki,
J. Phys. Soc. Jpn. 74, 1674 (2005)), because the phase of each plaquette is
taken on the principal branch and the sum telescopes. A Riemann sum of a
pointwise curvature -- however that curvature was obtained -- converges to an
integer and never equals one. Measured on the Haldane model in
``tests/unit/test_topology_chern.py``: the link construction gives 1 to 1e-15 on
a 6x6 mesh, the Kubo sum needs a 40x40 mesh to reach 1e-3.

So: **invariants from overlaps, curvature *plots* from either.**

Two kinds of state set implement the primitive.

``ArrayStates``
    Coefficients in an abstract orthonormal basis -- the eigenvectors of a
    model Hamiltonian, or any family of states parameterised by k. ``S`` is the
    identity. This is what the tests use to reach exact answers cheaply, and
    what a tight-binding cross-check plugs into.
``PlaneWaveStates``
    Kohn-Sham states from a defumat run. Two things make it different from a
    plain inner product and both are silent when wrong:

    * **The spheres differ.** ``u_k`` and ``u_k'`` are stored on the plane waves
      inside the cutoff at *their own* k-point, which are different sets of G.
      The coefficients have to be aligned by Miller index before they are
      contracted.
    * **The zone wraps.** On a closed mesh the neighbour of the last point is
      the first one displaced by a reciprocal lattice vector ``b``, and the
      periodic gauge ``u_{k+b}(G) = u_k(G + b)`` makes that a *shift* of the
      Miller index, not a relabelling of the k-point. This is the classic
      Chern-number bug: without it the plaquette product is not a closed loop
      and the answer is smooth and non-integer.

    On top of which ``S`` is not the identity for an ultrasoft or PAW dataset,
    and between two k-points it is not ``qq`` either but ``q_ij(b)`` --
    :mod:`defumat.topology.augmentation`.

**Memory.** A state set holds ``(nk, nbnd, npol * npwx)`` complex numbers, plus
``(nk, nbnd, npol, nkb)`` projections and ``(nk, npwx)`` integer keys. The
wavefunctions dominate: 16 bytes each, so ``nk * nbnd * npol * npwx * 16``. On
the bismuthene reference (``npwx ~ 2700``, ``npol = 2``, ``nbnd = 30``) that is
2.6 MB per k-point -- a 24-point Wilson loop is 62 MB and a 24x13 mesh would be
810 MB. Which is why the workflows build a state set **one loop or one mesh row
at a time** rather than materialising the whole mesh, and why ``nbnd`` here is
the occupied count and not the diagonalised one.
"""

from __future__ import annotations

import weakref

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "StateSet",
    "ArrayStates",
    "ModelStates",
    "PlaneWaveStates",
    "build_plane_wave_states",
]

#: Half-width of the Miller-index box a lookup key is packed into. Any G-vector
#: of any runnable cutoff is far inside it; the packing only has to be
#: collision-free, and an int64 has room for three 21-bit fields.
_KEY_OFFSET = 1 << 20
_KEY_BASE = 1 << 21


class StateSet(eqx.Module):
    """A manifold of states at each of ``nk`` k-points.

    Subclasses implement :meth:`overlap`. Everything else in the subpackage
    goes through it, which is what lets the same Chern-number code run on a
    tight-binding model and on a plane-wave calculation.
    """

    def overlap(self, i: int, j: int, shift=None) -> jnp.ndarray:
        """``<u_m(k_i) | S | u_n(k_j + shift)>``, an ``(nbnd, nbnd)`` matrix.

        ``shift`` is an integer triple in crystal coordinates: the reciprocal
        lattice vector that has to be added to ``k_j`` for it to be the physical
        neighbour of ``k_i``. ``None`` means zero.
        """
        raise NotImplementedError

    def overlaps(self, pairs, k_batch: int | None | str = "default") -> jnp.ndarray:
        """``(npair, nbnd, nbnd)``: :meth:`overlap` for a list of neighbour pairs.

        ``pairs`` is a sequence of ``(i, j, shift)``. This is the entry point
        every algorithm here uses, rather than :meth:`overlap` one at a time,
        because it is the k-axis walk: subclasses that can chunk it do
        (:class:`PlaneWaveStates` through :func:`defumat.batching.map_k`), and
        the base implementation is the ``batch = 1`` behaviour written out.
        """
        return jnp.stack([self.overlap(i, j, shift) for i, j, shift in pairs])

    def select(self, index) -> "StateSet":
        """The same states restricted to a subset of the k-points, in order.

        This is what lets one diagonalisation of a whole mesh be split into the
        loops a Wilson calculation walks -- and what a memory-bound calculation
        does *not* use, because there the loops are diagonalised one at a time
        and never coexist.
        """
        raise NotImplementedError

    @property
    def nk(self) -> int:
        raise NotImplementedError

    @property
    def nbnd(self) -> int:
        raise NotImplementedError


class ArrayStates(StateSet):
    """States as coefficient vectors in a fixed orthonormal basis.

    ``coefficients[k, n, a]`` is the amplitude of state ``n`` at k-point ``k`` on
    basis function ``a``; the overlap is the plain inner product, which is what
    a tight-binding model in the periodic (cell) gauge wants -- there
    ``H(k + b) = H(k)`` exactly, so a wrap is not a shift of anything and
    ``shift`` is ignored.

    ``orbital_positions``, if given, switches to the atomic (Bloch) gauge, where
    ``u_{k+b}`` and ``u_k`` differ by ``diag(e^{-2 pi i b . r_a})``. Passing the
    positions of a model whose Hamiltonian was built in the periodic gauge is
    exactly as wrong as omitting them for one that was not, and neither shows up
    anywhere except in the invariant.
    """

    coefficients: jnp.ndarray  # (nk, nbnd, dim)
    orbital_positions: jnp.ndarray | None = None  # (dim, 3), crystal coordinates

    @property
    def nk(self) -> int:
        return self.coefficients.shape[0]

    @property
    def nbnd(self) -> int:
        return self.coefficients.shape[1]

    def overlap(self, i: int, j: int, shift=None) -> jnp.ndarray:
        return _array_overlap(
            self.coefficients[i], self.coefficients[j], shift, self.orbital_positions
        )

    def select(self, index) -> "ArrayStates":
        return ArrayStates(
            coefficients=self.coefficients[jnp.asarray(index)],
            orbital_positions=self.orbital_positions,
        )


def _array_overlap(bra, ket, shift, orbital_positions):
    """``<u_m|u_n>`` in a fixed basis, with the atomic-gauge wrap if asked for."""
    if shift is not None and orbital_positions is not None:
        shift = jnp.asarray(shift, dtype=orbital_positions.dtype)
        phase = jnp.exp(-2j * jnp.pi * (orbital_positions @ shift))
        ket = ket * phase
    return jnp.einsum("ma,na->mn", bra.conj(), ket)


class ModelStates(StateSet):
    """The occupied eigenvectors of a model ``H(k)``, keeping ``H`` around.

    Exists for two reasons. It is what a tight-binding cross-check needs -- the
    Haldane, Kane-Mele and lattice Dirac models whose invariants are known
    exactly, which is how the conventions in this subpackage are pinned without
    running a DFT calculation. And it is the only state set that can answer the
    ``kubo`` curvature method, because it carries ``H(k)`` as a differentiable
    JAX function and that is what ``jacfwd`` needs (PLAN.md D2).

    ``hamiltonian`` takes a k-point in **crystal** coordinates and returns a
    Hermitian matrix; the Bloch convention is the model's own, and only the
    caller knows whether that is the periodic gauge (no ``orbital_positions``)
    or the atomic one.
    """

    coefficients: jnp.ndarray
    hamiltonian: object = eqx.field(static=True)
    energies: jnp.ndarray | None = None
    orbital_positions: jnp.ndarray | None = None
    #: The matrix representing spatial inversion on the basis, if the model has
    #: an inversion centre. At a TRIM it commutes with ``H(k)``, so the parity
    #: matrix is just its representation in the occupied manifold.
    inversion: jnp.ndarray | None = None

    @property
    def nk(self) -> int:
        return self.coefficients.shape[0]

    @property
    def nbnd(self) -> int:
        return self.coefficients.shape[1]

    def overlap(self, i: int, j: int, shift=None) -> jnp.ndarray:
        return _array_overlap(
            self.coefficients[i], self.coefficients[j], shift, self.orbital_positions
        )

    def select(self, index) -> "ModelStates":
        index = jnp.asarray(index)
        return ModelStates(
            coefficients=self.coefficients[index],
            hamiltonian=self.hamiltonian,
            energies=None if self.energies is None else self.energies[index],
            orbital_positions=self.orbital_positions,
            inversion=self.inversion,
        )

    def parity_matrix(self, i: int, centre=None) -> jnp.ndarray:
        """``<u_m|P|u_n>`` at a TRIM, from the model's inversion representation.

        ``centre`` is ignored: a model's inversion is given as a matrix on the
        basis, and where its centre sits is already encoded in that matrix.
        """
        if self.inversion is None:
            raise ValueError(
                "this model carries no inversion representation, so it has no "
                "parity eigenvalues; pass one to ModelSource(inversion=...) or "
                "use the Wilson-loop method"
            )
        coefficients = self.coefficients[i]
        return coefficients.conj() @ jnp.asarray(self.inversion) @ coefficients.T

    @classmethod
    def solve(cls, hamiltonian, points, nocc: int, orbital_positions=None,
              inversion=None):
        """Diagonalise ``H(k)`` at every point and keep the lowest ``nocc`` bands.

        ``eigh`` is *called* here and never differentiated through -- D4 forbids
        the latter, not the former. What is differentiated, in the ``kubo``
        method, is ``H(k)`` itself.
        """
        points = jnp.asarray(points, dtype=float)
        energies, vectors = jax.vmap(lambda k: jnp.linalg.eigh(hamiltonian(k)))(points)
        # ``eigh`` returns eigenvectors as columns; a state set stores them as
        # rows, so that band is the second axis everywhere in this subpackage.
        coefficients = jnp.swapaxes(vectors, -1, -2)[:, :nocc]
        return cls(
            coefficients=coefficients,
            hamiltonian=hamiltonian,
            energies=energies,
            orbital_positions=orbital_positions,
            inversion=inversion,
        )


class PlaneWaveStates(StateSet):
    """Kohn-Sham spinors or scalars on a k-list, with the ultrasoft overlap.

    ``coefficients`` is ``(nk, nbnd, npol * npwx)`` in the layout the
    eigensolvers use -- for ``npol = 2`` the two spinor components one after the
    other, each padded to ``npwx``, as QE's ``evc(npwx*npol, nbnd)``.

    ``keys`` packs each retained plane wave's Miller index into one integer, and
    ``order`` sorts them, so that aligning two k-points' spheres is a
    ``searchsorted`` rather than a dictionary. ``kcart`` is needed because the
    augmentation term depends on the *geometric* difference ``k' - k``, wrap
    included.
    """

    coefficients: jnp.ndarray  # (nk, nbnd, npol * npwx)
    keys: np.ndarray = eqx.field(static=False)  # (nk, npwx) int64 Miller keys
    valid: np.ndarray = eqx.field(static=False)  # (nk, npwx) bool
    miller: np.ndarray = eqx.field(static=False)  # (nk, npwx, 3) int
    order: np.ndarray = eqx.field(static=False)  # (nk, npwx) argsort of keys
    kcart: np.ndarray = eqx.field(static=False)  # (nk, 3), 1/bohr
    bg: np.ndarray = eqx.field(static=False)  # (3, 3) reciprocal basis, 1/bohr
    at: np.ndarray = eqx.field(static=False)  # (3, 3) direct basis, bohr
    npol: int = eqx.field(static=True)
    #: ``<beta_i|psi>`` at each k, ``(nk, nbnd, npol, nkb)``. ``None`` when the
    #: calculation is norm-conserving and ``S`` is the identity.
    becp: jnp.ndarray | None = None
    #: The calculation the augmentation factors are rebuilt from. Static: it is
    #: setup, never traced.
    calculation: object = eqx.field(static=True, default=None)
    #: ``(nk, npwx, nkb)`` projectors, kept only when a parity operation needs
    #: to reproject a transformed state. Large -- see the module docstring.
    vkb: jnp.ndarray | None = None
    #: ``(nk, nband)`` eigenvalues in Ry, *including* the empty bands above the
    #: manifold -- which is what makes the gap above it checkable.
    energies: jnp.ndarray | None = None
    #: ``(nk, nband, npol * npwx)`` -- **every** diagonalised band, where
    #: :attr:`coefficients` is the occupied manifold alone. Kept only for the
    #: ``kubo`` curvature, which is a sum over empty states and is the one
    #: thing here that needs them; it doubles the state set's memory, so it is
    #: off unless asked for.
    all_coefficients: jnp.ndarray | None = None
    #: The :class:`~defumat.response.velocity.VelocityOperator` built on
    #: :attr:`calculation` at *these* k-points, for the ``kubo`` curvature.
    #: Static: it holds the frozen potential and the calculation, neither of
    #: which is traced through a state set.
    velocity: object = eqx.field(static=True, default=None)

    @property
    def nk(self) -> int:
        return self.coefficients.shape[0]

    @property
    def nbnd(self) -> int:
        return self.coefficients.shape[1]

    @property
    def npwx(self) -> int:
        return self.keys.shape[1]

    def _alignment(self, i: int, j: int, shift):
        """Where each of ``i``'s plane waves sits in ``j``'s list, after ``shift``.

        Returns ``(gather, found)``: an index array of length ``npwx`` into
        ``j``'s plane waves, and a boolean saying whether the Miller index was
        there at all. A plane wave inside ``i``'s sphere and outside ``j``'s
        contributes nothing, which is correct -- the coefficient it would
        multiply is zero -- and is what ``found`` masks.
        """
        shift = np.zeros(3, dtype=int) if shift is None else np.asarray(shift, dtype=int)
        target = _pack(self.miller[i] + shift)
        order = self.order[j]
        sorted_keys = self.keys[j][order]
        position = np.searchsorted(sorted_keys, target)
        position = np.clip(position, 0, len(sorted_keys) - 1)
        gather = order[position]
        found = (sorted_keys[position] == target) & self.valid[i] & self.valid[j][gather]
        return jnp.asarray(gather), jnp.asarray(found)

    def _difference(self, i: int, j: int, shift) -> np.ndarray:
        """``k_j + shift - k_i`` in cartesian 1/bohr, the *unwrapped* difference."""
        shift = np.zeros(3) if shift is None else np.asarray(shift, dtype=float)
        return self.kcart[j] + shift @ self.bg - self.kcart[i]

    def overlap(self, i: int, j: int, shift=None) -> jnp.ndarray:
        gather, found = self._alignment(i, j, shift)
        matrix = _aligned_overlap(
            self.coefficients[i], self.coefficients[j], gather, found, self.npol
        )
        if self.becp is None:
            return matrix
        factors = _cached_augmentation(self.calculation, self._difference(i, j, shift))
        if factors is None:
            return matrix
        return matrix + _augmentation_term(self.becp[i], self.becp[j], factors)

    def overlaps(self, pairs, k_batch: int | None | str = "default") -> jnp.ndarray:
        """The batched overlap, walked over the pair axis by ``map_k``.

        Every pair in one call must share the same geometric ``k' - k``, which
        is what a uniform mesh gives: the step is ``b_d / n_d`` at every point
        including the one that wraps, where the neighbour is the first point
        plus a whole reciprocal lattice vector and the difference comes out the
        same fraction of it. That is what lets the augmentation factors be
        computed once rather than per pair, and it is asserted rather than
        assumed -- a pair list mixing directions would silently use one
        direction's ``q_ij(b)`` for both.

        **Memory.** The wavefunctions are closed over, not gathered, so nothing
        of size ``(npair, nbnd, npol * npwx)`` is ever built: the chunk in
        flight is ``k_batch`` k-points' worth. What the call does allocate is
        the ``(npair, nbnd, nbnd)`` result and ``(npair, npwx)`` integer
        alignment maps -- kilobytes against the megabytes per k-point of the
        states themselves.
        """
        from defumat.batching import map_k, resolve_k_batch

        pairs = list(pairs)
        differences = np.stack([self._difference(i, j, s) for i, j, s in pairs])
        if not np.allclose(differences, differences[0], atol=1e-10):
            raise ValueError(
                "every pair in one overlaps() call must have the same k' - k; "
                "group the pairs by mesh direction"
            )
        alignments = [self._alignment(i, j, s) for i, j, s in pairs]
        gather = jnp.stack([g for g, _ in alignments])
        found = jnp.stack([f for _, f in alignments])
        index_i = jnp.asarray([i for i, _, _ in pairs])
        index_j = jnp.asarray([j for _, j, _ in pairs])

        coefficients = self.coefficients
        npol = self.npol
        factors = None
        if self.becp is not None:
            factors = _cached_augmentation(self.calculation, differences[0])
        becp = self.becp

        def body(entry):
            ci = jnp.take(coefficients, entry["i"], axis=0)
            cj = jnp.take(coefficients, entry["j"], axis=0)
            matrix = _aligned_overlap(ci, cj, entry["gather"], entry["found"], npol)
            if factors is None:
                return matrix
            return matrix + _augmentation_term(
                jnp.take(becp, entry["i"], axis=0),
                jnp.take(becp, entry["j"], axis=0),
                factors,
            )

        return map_k(
            body,
            {"i": index_i, "j": index_j, "gather": gather, "found": found},
            batch=resolve_k_batch(k_batch),
        )

    def parity_matrix(self, i: int, centre) -> jnp.ndarray:
        """``<u_m | S P | u_n>`` at a time-reversal-invariant momentum.

        ``P`` is inversion about ``centre`` (crystal coordinates). At a TRIM
        ``2k`` is a reciprocal lattice vector ``G_k``, so ``G -> -G - G_k`` maps
        the plane-wave sphere onto *itself*: the operation is a permutation of
        the stored coefficients with a phase, and every Miller index must be
        found. It is asserted rather than masked -- a missing partner means the
        k-point is not a TRIM, and silently dropping it would return a
        confident, meaningless parity.
        """
        gk = np.rint(2.0 * self.kcrystal[i]).astype(int)
        if not np.allclose(2.0 * self.kcrystal[i], gk, atol=1e-8):
            raise ValueError(
                f"k = {self.kcrystal[i]} is not a time-reversal-invariant momentum; "
                "2k must be a reciprocal lattice vector for a parity eigenvalue "
                "to exist"
            )
        target = _pack(-self.miller[i] - gk)
        order = self.order[i]
        sorted_keys = self.keys[i][order]
        position = np.clip(np.searchsorted(sorted_keys, target), 0, len(order) - 1)
        gather = order[position]
        found = (sorted_keys[position] == target) | ~self.valid[i]
        if not np.all(found):
            raise ValueError(
                "the plane-wave sphere is not mapped onto itself by inversion; "
                f"{int((~found).sum())} of {int(self.valid[i].sum())} plane waves "
                "have no partner -- the k-point is not a TRIM, or the G-vector "
                "set was truncated inconsistently"
            )
        # (P u)(G) = u(-G - G_k) e^{-2 i (k + G) . r0}, with r0 the inversion
        # centre in cartesian coordinates.
        r0 = np.asarray(centre, dtype=float) @ self.at
        kg = self.kcart[i] + self.miller[i] @ self.bg  # (npwx, 3)
        phase = jnp.asarray(np.exp(-2j * (kg @ r0)) * self.valid[i])

        transformed = _permute(self.coefficients[i], jnp.asarray(gather), phase, self.npol)
        matrix = jnp.einsum(
            "ma,na->mn", self.coefficients[i].conj(), transformed
        )
        if self.becp is None or self.vkb is None:
            return matrix
        factors = _cached_augmentation(self.calculation, np.zeros(3))
        if factors is None:
            return matrix
        becp = _project(transformed, self.vkb[i], self.npol)
        return matrix + _augmentation_term(self.becp[i], becp, factors)

    def select(self, index) -> "PlaneWaveStates":
        index = np.asarray(index)
        return PlaneWaveStates(
            coefficients=self.coefficients[jnp.asarray(index)],
            keys=self.keys[index],
            valid=self.valid[index],
            miller=self.miller[index],
            order=self.order[index],
            kcart=self.kcart[index],
            bg=self.bg,
            at=self.at,
            npol=self.npol,
            becp=None if self.becp is None else self.becp[jnp.asarray(index)],
            calculation=self.calculation,
            vkb=None if self.vkb is None else self.vkb[jnp.asarray(index)],
            energies=None if self.energies is None else self.energies[jnp.asarray(index)],
            # **Dropped, not sliced.** The velocity operator is built on a
            # ``Calculation`` at the *whole* k-list -- its ``vkb(k)`` and
            # ``|k+G|^2`` are indexed by the original k-axis -- so a selected
            # subset no longer lines up with it. A later ``kubo`` call on the
            # selection then refuses by name rather than differentiating at the
            # wrong k-points, which is the failure that would be silent.
            all_coefficients=None,
            velocity=None,
        )

    @property
    def kcrystal(self) -> np.ndarray:
        """The k-points in crystal coordinates, from the cartesian ones."""
        return self.kcart @ np.linalg.inv(self.bg)


#: ``{calculation -> {rounded b -> q_ij(b)}}``, held **weakly**.
#:
#: A uniform mesh has exactly **two** distinct ``b`` -- one per direction,
#: ``b_d / n_d`` -- however many k-points it has, including at the wrap, where
#: the last point's neighbour is the first plus a whole reciprocal lattice
#: vector and the difference comes out the same fraction of it. Recomputing the
#: Bessel transforms per k-point would be the dominant cost of an otherwise
#: trivial contraction, so they are computed once per direction.
#:
#: **Weakly, and that is not a detail.** A ``Calculation`` owns ``Q_ij(G)`` on
#: the dense grid -- a gigabyte on the bismuthene reference -- and the whole
#: point of streaming a Wilson loop one row at a time is that the previous
#: row's calculation is dropped. An ordinary ``lru_cache`` keyed on the
#: calculation would pin every one of them and turn a 200 MB working set into
#: fourteen gigabytes, with nothing in any answer to show for it.
_AUGMENTATION_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _cached_augmentation(calculation, qcart):
    """``q_ij(b)``, memoised on ``b`` for as long as the calculation lives."""
    from defumat.topology.augmentation import augmentation_at_q

    key = tuple(np.round(np.asarray(qcart, dtype=float), 10))
    entries = _AUGMENTATION_CACHE.setdefault(calculation, {})
    if key not in entries:
        entries[key] = augmentation_at_q(calculation, np.asarray(key))
    return entries[key]


def _pack(miller: np.ndarray) -> np.ndarray:
    """One int64 key per Miller index, collision-free over any runnable box."""
    shifted = np.asarray(miller, dtype=np.int64) + _KEY_OFFSET
    return (shifted[..., 0] * _KEY_BASE + shifted[..., 1]) * _KEY_BASE + shifted[..., 2]


@eqx.filter_jit
def _aligned_overlap(ci, cj, gather, found, npol: int):
    """``sum_G conj(c_i(G)) c_j(G + b)`` with the second set gathered onto the first."""
    ci = ci.reshape(ci.shape[0], npol, -1)
    cj = cj.reshape(cj.shape[0], npol, -1)
    picked = jnp.where(found, cj[:, :, gather], 0.0)
    return jnp.einsum("mag,nag->mn", ci.conj(), picked)


@eqx.filter_jit
def _permute(coefficients, gather, phase, npol: int):
    """``c(G) -> c(sigma(G)) * phase(G)``, applied to every spinor component."""
    reshaped = coefficients.reshape(coefficients.shape[0], npol, -1)
    moved = reshaped[:, :, gather] * phase
    return moved.reshape(coefficients.shape)


@eqx.filter_jit
def _project(coefficients, vkb, npol: int):
    """``<beta|psi>`` shaped ``(nbnd, npol, nkb)``."""
    reshaped = coefficients.reshape(coefficients.shape[0], npol, -1)
    return jnp.einsum("gk,nag->nak", vkb.conj(), reshaped)


@eqx.filter_jit
def _augmentation_term(becp_i, becp_j, factors):
    """``sum q_ij(b) <psi_m|beta_i><beta_j|psi_n>`` for either spin structure."""
    if factors.ndim == 2:  # (nkb, nkb): one spin channel, or none
        return jnp.einsum(
            "mai,ij,naj->mn", becp_i.conj(), factors.astype(becp_i.dtype), becp_j
        )
    return jnp.einsum(
        "mai,abij,nbj->mn", becp_i.conj(), factors.astype(becp_i.dtype), becp_j
    )


def build_plane_wave_states(
    calculation,
    coefficients: jnp.ndarray,
    nbnd: int | None = None,
    keep_projectors: bool = False,
    energies: jnp.ndarray | None = None,
    velocity=None,
) -> PlaneWaveStates:
    """Wrap a diagonalisation's output as a :class:`PlaneWaveStates`.

    ``coefficients`` is ``(nk, nband, npol * npwx)`` as
    :meth:`~defumat.scf.driver.Calculation.diagonalize` returns it for one spin
    channel; ``nbnd`` truncates it to the occupied manifold, which is what every
    invariant here is a property of and what keeps the working set small.

    ``keep_projectors`` retains ``vkb``, which the parity operation needs and
    nothing else does. It is ``(nk, npwx, nkb)`` complex -- megabytes per
    k-point on a real cell -- so it is off by default and the four TRIM of a
    parity calculation are the only place it is worth paying.

    ``velocity`` is a :class:`~defumat.response.velocity.VelocityOperator`
    built on the same ``calculation``; passing one also retains the *whole*
    band set as :attr:`PlaneWaveStates.all_coefficients`, because the ``kubo``
    curvature is a sum over empty states and the truncation is exactly the
    bands the eigensolver did not resolve. Both are off by default for the same
    memory reason ``keep_projectors`` is.
    """
    basis = calculation.basis
    gvectors = basis.smooth if hasattr(basis, "smooth") else basis.dense
    planewaves = basis.planewaves
    cell = calculation.system.cell

    indices = np.asarray(planewaves.indices)
    mask = np.asarray(planewaves.mask)
    miller = np.asarray(gvectors.miller)[indices]  # (nk, npwx, 3)
    # Padding entries all repeat G = 0, which would collide with the real G = 0;
    # ``valid`` is what keeps them out of every lookup.
    keys = _pack(miller)
    order = np.argsort(keys, axis=1, kind="stable")

    # The whole diagonalised set is kept only where a sum over empty states
    # asks for it; ``nbnd`` is the occupied manifold and everything else here
    # is a property of that alone.
    all_coefficients = coefficients if velocity is not None else None
    coefficients = coefficients[:, :nbnd] if nbnd is not None else coefficients
    npol = int(calculation.npol)

    becp = None
    vkb = None
    if calculation.augmentation is not None:
        projectors = calculation.projectors.vkb
        becp = jax.vmap(lambda c, v: _project(c, v, npol))(coefficients, projectors)
        if keep_projectors:
            vkb = projectors

    return PlaneWaveStates(
        coefficients=coefficients,
        keys=keys,
        valid=mask,
        miller=miller,
        order=order,
        kcart=np.asarray(calculation.system.kpoints.cartesian(cell)),
        bg=np.asarray(cell.bg),
        at=np.asarray(cell.to_cartesian(np.eye(3))),
        npol=npol,
        becp=becp,
        calculation=calculation,
        vkb=vkb,
        energies=energies,
        all_coefficients=all_coefficients,
        velocity=velocity,
    )
