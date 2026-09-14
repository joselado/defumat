"""The ultracell, its Q-vectors, and the one index map everything rests on.

``PLAN.md`` P88. An **ultracell** is ``N = n1 n2 n3`` copies of the unit cell,
and it exists so that the density, the magnetization and the potential can be
modulated over a length no unit cell has while the atoms stay exactly where the
unit cell puts them. Its reciprocal lattice vectors inside the unit cell's
Brillouin zone are ``N`` wavevectors ``Q = sum_i q_i b_i / n_i``, and every
long-ranged quantity is

    f(r) = sum_Q f_Q(r) e^{iQ.r},     f_Q lattice periodic,

which is the same object as one lattice-periodic ``f`` per cell ``R`` of the
ultracell.

**Everything in this subpackage lives on the ultracell FFT box**, and that one
decision is what makes the rest short. The box has shape
``(n1 Nd1, n2 Nd2, n3 Nd3)`` where ``Nd`` is the unit cell's *dense* grid, so it
has the same real-space resolution as the unit cell and its reciprocal grid is
exactly the ``G + Q`` set. A plane wave with unit-cell Miller index ``G``,
belonging to a k-point that carries ``Q``, sits at

    J_i = (n_i G_i + q_i)  mod  (n_i Nd_i)

and that single line removes three pieces of machinery the phase was planned
around:

* **the ``G``-wrap of ``Q - Q'``.** Two ultracell basis functions whose ``Q``
  differ by more than the Q-set couple through ``V_{Q-Q'-G} e^{iG.r}``, and in
  a unit-cell formulation that phase has to be carried explicitly. On the box
  the wrap is what ``mod`` already did, so no pair of ``Q`` values needs a
  phase factor and no Miller-index alignment (P16's ``_alignment``) is needed
  at all.
* **the Coulomb Green's function at ``G + Q``.** Elk builds ``gclgq`` per ``Q``
  (``gengclgq``) because its G-set is the unit cell's; here the Hartree term is
  ``4 pi / |G_u|^2`` over the box's own reciprocal grid and there is nothing to
  assemble. Elk's ``q0cut`` row is, on the box, the set of ``J != 0`` whose
  ``floor(J_i/n_i)`` is zero in every direction -- the ``G = 0, Q != 0``
  elements, which are the ones that screen a long-wavelength potential and are
  kept by default.
* **the per-cell exchange-correlation call.** ``potxcu`` evaluates ``potxc``
  once per cell ``R`` on that cell's own density; on the box an LDA is simply
  evaluated pointwise on the whole ultracell grid, which is the same thing for
  a local functional and is *not* the same thing for a GGA (see
  :mod:`defumat.ultracell.potential`).

The two facts that make the index map work are worth stating because both are
used without comment downstream:

* ``floor(J_i / n_i)`` is the unit cell's own FFT box index of ``G_i``, for
  ``G_i`` of either sign. So a unit-cell reciprocal mask becomes the ultracell
  one by ``np.repeat`` along each axis, with no arithmetic.
* the ultracell real-space grid point ``J`` sits at unit-cell crystal
  coordinate ``x_i = J_i / Nd_i``, so a unit-cell real-space field becomes the
  tiled ultracell one by ``np.tile``, and the two conventions agree:
  ``exp(i G_u . r)`` at ``J' = n G`` reduces to the unit cell's own kernel.

Reference: Elk's ``src/modulr.f90``, ``genkpakq.f90``, ``initulr.f90``; the
method is T. Mueller, S. Sharma, E. K. U. Gross and J. K. Dewhurst, *Extending
solid-state calculations to ultra long-range length scales*, Phys. Rev. Lett.
**125**, 256402 (2020).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from defumat.config import DEFAULT_PRECISION, Precision
from defumat.system.cell import Cell
from defumat.system.kpoints import KPoints

__all__ = ["Ultracell", "folded_kpoints"]


@dataclasses.dataclass(frozen=True, eq=False)
class Ultracell:
    """``n1 x n2 x n3`` unit cells, and the maps between the two grids.

    Built host-side and then fixed. **Not** an ``eqx.Module``, and the reason
    is rule R1 read the other way: nothing here crosses a ``jit`` or ``grad``
    boundary. Every field is an integer triple or a NumPy index array, all of it
    is basis bookkeeping decided before any array is traced, and a plain frozen
    dataclass says that where a pytree-registered module would invite the maps
    to be closed over as tracers.
    """

    #: How many unit cells along each lattice vector.
    shape: tuple[int, int, int]
    #: The unit cell's dense FFT grid, which the box is built on.
    cell_grid: tuple[int, int, int]
    #: ``(N, 3)`` integer ``q`` triples, ``q_i`` in ``[0, n_i)``, in the order
    #: the Q index uses (C-order over the triple).
    q_triples: np.ndarray
    precision: Precision = DEFAULT_PRECISION

    @staticmethod
    def build(shape, cell_grid, precision: Precision = DEFAULT_PRECISION) -> "Ultracell":
        shape = tuple(int(n) for n in shape)
        if len(shape) != 3 or any(n < 1 for n in shape):
            raise ValueError(f"the ultracell must be three positive integers, got {shape}")
        cell_grid = tuple(int(n) for n in cell_grid)
        q = np.stack(
            np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1
        ).reshape(-1, 3)
        return Ultracell(
            shape=shape, cell_grid=cell_grid,
            q_triples=q.astype(np.int64), precision=precision,
        )

    # -- counts and shapes ---------------------------------------------------

    @property
    def cells(self) -> int:
        """``N``: how many unit cells, and how many Q-vectors."""
        return int(np.prod(self.shape))

    @property
    def grid(self) -> tuple[int, int, int]:
        """The ultracell FFT box: the unit cell's dense grid, ``n_i`` times."""
        return tuple(n * m for n, m in zip(self.shape, self.cell_grid))

    @property
    def points(self) -> int:
        return int(np.prod(self.grid))

    def volume(self, cell: Cell) -> float:
        return float(cell.volume) * self.cells

    def lattice(self, cell: Cell) -> np.ndarray:
        """The ultracell lattice vectors, rows, in the unit cell's own units."""
        return np.asarray(cell.at_alat) * np.asarray(self.shape)[:, None]

    # -- the Q-vectors -------------------------------------------------------

    @property
    def q_crystal(self) -> np.ndarray:
        """``(N, 3)`` Q-vectors in the *unit cell's* reciprocal crystal basis."""
        return self.q_triples / np.asarray(self.shape, dtype=float)

    def q_index(self, triples) -> np.ndarray:
        """Q index of an integer triple, wrapped into ``[0, n_i)``.

        The wrap is the umklapp: ``Q - Q'`` leaves the Q-set by a unit-cell
        reciprocal lattice vector, and *this* is where that is absorbed. On the
        ultracell box the ``e^{iG.r}`` it would otherwise leave behind is
        already in the index map, so nothing downstream carries a phase.
        """
        triples = np.asarray(triples, dtype=np.int64) % np.asarray(self.shape)
        n2, n3 = self.shape[1], self.shape[2]
        return triples[..., 0] * (n2 * n3) + triples[..., 1] * n3 + triples[..., 2]

    @property
    def difference_index(self) -> np.ndarray:
        """``(N, N)``: the Q index of ``Q_i - Q_j``, wrapped.

        The table the Hamiltonian's potential blocks are read through.
        """
        return self.q_index(self.q_triples[:, None, :] - self.q_triples[None, :, :])

    # -- the two grid maps ---------------------------------------------------

    def box_index(self, miller, q_index) -> np.ndarray:
        """Flat ultracell box index of unit-cell Miller indices at one ``Q``.

        Args:
            miller: ``(..., 3)`` integer Miller indices ``G``.
            q_index: the Q index those plane waves belong to; a scalar, or an
                array broadcasting against ``miller[..., 0]``.

        ``J_i = (n_i G_i + q_i) mod (n_i Nd_i)``, flattened C-order over the
        box, which is the layout ``jnp.fft.fftn`` on the last three axes uses.
        """
        miller = np.asarray(miller, dtype=np.int64)
        q = self.q_triples[np.asarray(q_index, dtype=np.int64)]
        n = np.asarray(self.shape, dtype=np.int64)
        box = np.asarray(self.grid, dtype=np.int64)
        j = (n * miller + q) % box
        return j[..., 0] * (box[1] * box[2]) + j[..., 1] * box[2] + j[..., 2]

    def miller_at(self, flat_index) -> np.ndarray:
        """``(..., 3)`` ultracell Miller indices of flat box indices.

        The triple ``h`` standing for ``sum_i h_i b_i / n_i``, folded to the
        interval centred on zero -- the convention :meth:`g2` enumerates in and
        the one the unit cell's own G-vectors are enumerated in. It is computed
        **from the indices asked for** rather than read out of a table over the
        whole box, and that is a memory decision rather than a style one: a
        slab's ultracell box runs to 10^8 points, where a stored table of
        triples would be gigabytes, and both callers ask for a small subset of
        it -- a wavefunction sphere (``N npwx`` entries) or the dense ``keep``
        set.

        ``miller_at(box_index(G, q))`` is ``n G + q`` on the **wavefunction**
        sphere, which is what lets an ultracell state be handed to a routine
        written for a unit-cell one. It is not so on the dense one and does not
        need to be: ``n G + q`` leaves the centred interval once it passes
        ``n Nd / 2``, which the dense sphere's outer shell does from ``n = 3``
        (23 against a half box of 22 at ``n = 3, G_i = 7``), and there what
        comes back is the *other* representative of the same box entry -- which
        is the frequency that entry actually carries and therefore the label a
        sum over the box wants. The wavefunction sphere has a factor of two in
        hand (``4n - 1`` against ``7.5n`` on the test-suite silicon) and is
        exact at every entry.
        """
        flat = np.asarray(flat_index, dtype=np.int64)
        box = np.asarray(self.grid, dtype=np.int64)
        j = np.stack([flat // (box[1] * box[2]),
                      (flat // box[2]) % box[1],
                      flat % box[2]], axis=-1)
        return np.where(j > box // 2, j - box, j)

    def tile(self, field):
        """A unit-cell real-space field on the ultracell grid, ``N`` copies.

        Works on any array whose last three axes are the unit cell's dense
        grid, which is what a spin-resolved potential is.
        """
        reps = (1,) * (field.ndim - 3) + self.shape
        return np.tile(field, reps) if isinstance(field, np.ndarray) else _jtile(field, reps)

    def reciprocal_mask(self, cell_mask) -> np.ndarray:
        """A unit-cell reciprocal mask, on the ultracell box.

        ``keep[J] = cell_mask[floor(J_1/n_1), floor(J_2/n_2), floor(J_3/n_3)]``,
        which is exactly ``np.repeat`` along each axis. This is the set
        ``{G + Q : G inside the unit cell's dense sphere}`` -- Elk's choice
        (``ngvec`` G-vectors per ``Q``) rather than a sphere in the ultracell's
        own reciprocal space, and it is what makes the ``N = 1`` limit reduce to
        the unit cell exactly.
        """
        mask = np.asarray(cell_mask)
        for axis, n in enumerate(self.shape):
            mask = np.repeat(mask, n, axis=axis)
        return mask

    def g2(self, cell: Cell) -> np.ndarray:
        """``|G + Q|^2`` in 1/bohr^2 on the whole ultracell box.

        The ultracell reciprocal index ``J`` stands for ``sum_i J_i b_i / n_i``,
        folded to the interval centred on zero so that the box's upper half is
        the negative frequencies -- the same convention ``jnp.fft`` uses and the
        same one the unit-cell G-vectors are enumerated in.
        """
        box = self.grid
        folded = [
            np.where(np.arange(m) > m // 2, np.arange(m) - m, np.arange(m))
            for m in box
        ]
        fractional = np.stack(
            np.meshgrid(*[f / n for f, n in zip(folded, self.shape)], indexing="ij"),
            axis=-1,
        )
        cartesian = fractional @ (np.asarray(cell.bg_2pi_alat) * float(cell.tpiba))
        return np.sum(cartesian ** 2, axis=-1)


def _jtile(field, reps):
    import jax.numpy as jnp

    return jnp.tile(field, reps)


def folded_kpoints(
    ultracell: Ultracell,
    kgrid,
    cell: Cell,
    precision: Precision = DEFAULT_PRECISION,
) -> tuple[KPoints, KPoints]:
    """The ultracell BZ k-set and the unit-cell k-set it folds from.

    Returns ``(k0, folded)``. ``k0`` samples the *ultracell's* Brillouin zone,
    which is ``N`` times smaller than the unit cell's, on a ``kgrid`` mesh; the
    folded set is every ``k0 + Q``, in the order ``ik = ik0 * N + iq``, and it
    is the set the frozen unit-cell states are computed on.

    **The two are a single Monkhorst-Pack grid of the unit cell.** With
    ``kgrid = m0`` the k0 points are ``j / (n_i m0_i)`` in unit-cell crystal
    coordinates for ``j`` in ``[0, m0_i)``, so ``k0 + Q`` sweeps
    ``{0, ..., n_i m0_i - 1} / (n_i m0_i)`` exactly once. That is why a unit-cell
    SCF run on the unshifted ``n * m0`` grid with symmetry off has the tiled
    density as an *exact* fixed point of the ultracell loop: the two integrate
    the Brillouin zone over the same points with the same weights.

    Unshifted and unreduced, both deliberately: a modulation breaks the
    crystal's point group, and the group of the ultracell is not written (Elk
    sets ``reducek = 0`` for the whole run, ``gndstulr.f90``).
    """
    kgrid = tuple(int(m) for m in kgrid)
    if any(m < 1 for m in kgrid):
        raise ValueError(f"the ultracell k-grid must be three positive integers, got {kgrid}")
    folded_grid = tuple(n * m for n, m in zip(ultracell.shape, kgrid))

    j = np.stack(
        np.meshgrid(*[np.arange(m) for m in kgrid], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    k0_crystal = j / np.asarray(folded_grid, dtype=float)
    weights = np.full(len(k0_crystal), 1.0 / len(k0_crystal))

    q = ultracell.q_crystal
    combined = (k0_crystal[:, None, :] + q[None, :, :]).reshape(-1, 3)

    k0 = KPoints.from_crystal(k0_crystal, weights, cell, precision=precision,
                              grid=kgrid, shift=(0, 0, 0))
    folded = KPoints.from_crystal(
        combined, np.full(len(combined), 1.0 / len(combined)), cell,
        precision=precision, grid=folded_grid, shift=(0, 0, 0),
    )
    return k0, folded
