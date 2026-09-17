"""An ultracell state as one plane-wave vector, which is what makes it visible.

``PLAN.md`` P89. An ultracell state at an ultracell Brillouin zone point ``k0``
is

    Psi_j(r) = sum_{Q,n} a^j_{Q,n} psi_{k0+Q,n}(r),

and :mod:`defumat.ultracell.density` builds its *square* by scattering the
frozen coefficients into the ultracell box and transforming. Everything that
looks at a wavefunction rather than at a density -- the value of ``Psi`` at a
tip point, its Gram matrix on an exit plane -- needs ``Psi`` itself, and the
useful fact is that it is already there, with no transform at all:

**the union over ``Q`` of the ``N`` unit-cell spheres at ``k0 + Q`` is the
ultracell's own plane-wave sphere at ``k0``.** A plane wave with unit-cell
Miller index ``G`` belonging to ``Q`` is the ultracell plane wave
``h_u = n G + q`` (:meth:`~defumat.ultracell.grid.Ultracell.box_index`), and no
two of them collide: reducing ``n(G - G') + (q - q') = 0 mod n Nd`` modulo
``n`` forces ``q = q'`` because both lie in ``[0, n)``, and then ``G = G'``
because the sphere sits inside the dense grid. So the ``(Q, p)`` list *is* a
sphere of ``N npwx`` plane waves, the mixing above is one contraction over the
band index, and

    C_j[q, p] = sum_n a^j_{q,n} c_{k0+Q_q, n}[p]

is an ordinary coefficient vector on it. Hand that, the ultracell's Miller
indices, the ultracell's cell and ``k0`` in the ultracell's own reciprocal
coordinates to a routine written for a unit-cell wavefunction --
:func:`~defumat.basis.sample.sample_wavefunctions`,
:func:`~defumat.transport.substrate.exit_overlap` -- and it is an ultracell
routine, with no second implementation of anything.

**The padding mask is not optional here.** Every plane-wave array is padded to
a common ``npwx`` and a padded entry points at ``G = 0``, so within one ``Q``
all of them land on that ``Q``-row's own ``G = 0`` slot -- 21 of 190 entries on
the two-atom silicon cell at ``ecutwfc = 12``. The coefficients there are zero,
but ``exit_overlap`` groups plane waves by their in-plane index before it ever
looks at a coefficient, so the pad has to be masked rather than trusted.

**Normalisation.** The frozen states are normalised over the *unit* cell and
``sum_j |a_j|^2 = 1``, so ``sum_{q,p} |C[q,p]|^2 = 1`` and

    Psi_j(r) = Omega_u^{-1/2} sum_{q,p} C_j[q,p] e^{i(k0 + G_u).r}

is normalised over the **ultracell**, ``Omega_u = N Omega_cell``. That is the
convention :func:`~defumat.basis.sample.sample_wavefunctions` takes when it is
given ``volume = Omega_u``, and it is the one that makes an ultracell weight
``w_k0 f_j`` rather than ``w_k0 f_j / N``: the density module divides by the
unit cell's volume and :func:`~defumat.ultracell.driver._occupy` divides the
occupations by ``N``, which is the same number written the other way round.

**What the peak costs.** One ``k0``'s block is ``nstate x npol N npwx``
complex, and with ``nstate = N nbnd`` that is ``N^2 nbnd npwx npol`` numbers --
4.7 MB for eight cells of silicon at ``nbnd = 24``, and 0.42 GB for a 21-cell
slab at ``nbnd = 30, npwx = 2000``. It is built **per ``k0``, on demand**
(:meth:`UltracellStates.block`) for that reason: materialising the whole
``(blocks, nk0, ...)`` array would be ``nk0`` times it, and the ``N^2`` is the
same one the direct route's matrix build already pays in time.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from defumat.system.cell import Cell
from defumat.ultracell.density import (
    spinor_ultracell_density,
    ultracell_density,
)
from defumat.ultracell.grid import Ultracell

__all__ = ["UltracellStates", "ultracell_band_density"]


@dataclasses.dataclass(frozen=True, eq=False)
class UltracellStates:
    """The frozen states and the envelope amplitudes, and the sphere they live on.

    A plain frozen dataclass and not an ``eqx.Module``, for
    :class:`~defumat.ultracell.grid.Ultracell`'s own reason: nothing here
    crosses a ``jit`` or a ``grad`` boundary, it is basis bookkeeping and the
    two arrays a finished loop happens to still hold.

    It is also a **stand-in for a wavefunction array**: ``states[ispin, ik0]``
    is one ``k0``'s ``(nstate, npol N npwx)`` block and ``states.shape`` is what
    a ``(nspin, nk, nbnd, npol npwx)`` array would report, so a routine that
    walks a wavefunction array k-point by k-point takes this without knowing
    the difference.
    """

    #: ``(blocks, nk0, N, nbnd, npol npwx)`` the frozen unit-cell states at
    #: ``k0 + Q``, padding already zeroed.
    coefficients: jnp.ndarray
    #: ``(blocks, nk0, N nbnd, N nbnd)`` the envelope amplitudes, column ``j``
    #: the state ``a^j_{Q,n}`` in ``(Q, n)`` C-order.
    vectors: jnp.ndarray
    #: ``(blocks, nk0, N nbnd)`` the ultracell eigenvalues in Ry.
    eigenvalues: np.ndarray
    #: ``(nk0, N, npwx)`` flat ultracell box index of every plane wave.
    box_index: np.ndarray
    #: ``(nk0, N, npwx)`` the padding mask of the folded spheres.
    padding: np.ndarray
    #: ``(nk0, 3)`` the k0 points in the **unit cell's** crystal basis.
    k0_crystal: np.ndarray
    #: ``(nk0,)`` their weights, through ``for_spin`` already.
    weights: np.ndarray
    ultracell: Ultracell
    cell: Cell
    npol: int = 1
    nspin: int = 1

    # -- counts --------------------------------------------------------------

    @property
    def blocks(self) -> int:
        """Independent matrices per ``k0``: two collinear channels, or one."""
        return int(self.coefficients.shape[0])

    @property
    def nk0(self) -> int:
        return int(self.coefficients.shape[1])

    @property
    def cells(self) -> int:
        return int(self.ultracell.cells)

    @property
    def nbnd(self) -> int:
        """Frozen bands per folded k-point."""
        return int(self.coefficients.shape[3])

    @property
    def nstate(self) -> int:
        """``N nbnd``: how many ultracell states there are at one ``k0``."""
        return self.cells * self.nbnd

    @property
    def npwx(self) -> int:
        """The *unit cell's* padded plane-wave count."""
        return int(self.coefficients.shape[4]) // int(self.npol)

    @property
    def width(self) -> int:
        """``npol N npwx``: the length of one ultracell state's vector."""
        return int(self.npol) * self.cells * self.npwx

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """What a wavefunction array of these states would report."""
        return (self.blocks, self.nk0, self.nstate, self.width)

    # -- the ultracell's own geometry ----------------------------------------

    @property
    def ultracell_cell(self) -> Cell:
        """The ultracell as a :class:`~defumat.system.cell.Cell`.

        Its lattice vectors are ``n_i a_i``, so its volume is ``N Omega`` and
        its surface area ``n_i n_j |a_i x a_j|`` -- the two numbers
        :func:`~defumat.transport.substrate.exit_overlap` scales by, and the
        reason the exit integral over the whole modulation comes out right
        without a factor being written anywhere.
        """
        return Cell.from_vectors(
            np.asarray(self.cell.at) * np.asarray(self.ultracell.shape)[:, None],
            alat=float(self.cell.alat), precision=self.cell.precision,
        )

    @property
    def kcrystal(self) -> np.ndarray:
        """``(nk0, 3)`` the k0 points in the **ultracell's** crystal basis.

        Which is ``n_i`` times the unit cell's, because the ultracell's
        reciprocal vectors are ``b_i / n_i``: a Bloch phase written against the
        ultracell's Miller indices has to be written against its k-points too.
        """
        return np.asarray(self.k0_crystal, dtype=float) * np.asarray(
            self.ultracell.shape, dtype=float)

    def miller(self, ik0: int) -> np.ndarray:
        """``(N npwx, 3)`` ultracell Miller indices of one ``k0``'s sphere."""
        return self.ultracell.miller_at(
            np.asarray(self.box_index)[int(ik0)].reshape(-1))

    def mask(self, ik0: int) -> np.ndarray:
        """``(N npwx,)`` which of them are plane waves rather than padding."""
        return np.asarray(self.padding)[int(ik0)].reshape(-1)

    # -- the states themselves -----------------------------------------------

    def block(self, ispin: int, ik0: int) -> jnp.ndarray:
        """``(nstate, npol N npwx)`` the ultracell states at one ``k0``.

        The band index is contracted away and the ``(Q, p)`` pair is flattened
        into one sphere; a spinor keeps its two components as the two halves of
        the row, which is how every other wavefunction in this package is laid
        out and what :func:`~defumat.transport.substrate.exit_overlap` and
        :func:`~defumat.basis.sample.sample_wavefunctions` both expect.
        """
        cells, nbnd, npol, npwx = self.cells, self.nbnd, int(self.npol), self.npwx
        amplitudes = jnp.asarray(self.vectors)[int(ispin), int(ik0)].T.reshape(
            self.nstate, cells, nbnd)
        parts = jnp.asarray(self.coefficients)[int(ispin), int(ik0)].reshape(
            cells, nbnd, npol, npwx)
        mixed = jnp.einsum("jqn,qnap->jaqp", amplitudes, parts)
        return mixed.reshape(self.nstate, npol * cells * npwx)

    def __getitem__(self, index):
        ispin, ik0 = index
        return self.block(ispin, ik0)


    # -- what they make ------------------------------------------------------

    def density(self, weights, nspin_mag: int = 1, batch: int | None = 1,
                dtype=None) -> jnp.ndarray:
        """``(nspin_mag, *box)`` the density these states make with ``weights``.

        The self-consistent loop calls this with the occupations and an STM
        image calls it with a smeared delta at the tip energy, which is the
        whole of what a Tersoff-Hamann image is -- Elk's ``wfplot.f90``
        overwrites ``occsv`` and calls ``rhomagv`` again, and this is the same
        sentence with one function name in it. ``weights`` is
        ``(blocks, nk0, N nbnd)`` either way.

        **On an ultrasoft or PAW dataset this is the smooth density and not the
        whole one**: the augmentation charge is built from ``becsum`` rather
        than from the states, so the loop adds it separately
        (:func:`~defumat.ultracell.augmentation.ultracell_augmentation_charge`)
        and an image does not. That is right rather than a shortfall for the
        one thing that asks for it -- a tip sits in the vacuum, where the
        augmentation charge is zero and a pseudo-wavefunction is the true one --
        and it is why ``run_ultracell_stm`` and ``run_ultracell_sts`` refuse a
        tip inside a sphere rather than correcting for one.
        """
        return ultracell_band_density(
            self.coefficients, self.vectors, weights, self.box_index,
            self.ultracell.grid, float(self.cell.volume), npol=int(self.npol),
            nspin_mag=int(nspin_mag), batch=batch,
            dtype=dtype or self.ultracell.precision.real,
        )


def ultracell_band_density(coefficients, vectors, weights, box_index, grid,
                           volume, *, npol: int = 1, nspin_mag: int = 1,
                           batch: int | None = 1, dtype=None) -> jnp.ndarray:
    """Accumulate :mod:`defumat.ultracell.density` over every block and ``k0``.

    **A collinear channel produces one component and a spinor state produces
    all of them**, which is why the two accumulations are different functions
    rather than one with a flag: the four components of ``(n, m_x, m_y, m_z)``
    are four bilinears in the *same* pair of transformed spinor components, and
    there is no channel index to add them into.

    Args:
        coefficients: ``(blocks, nk0, N, nbnd, npol npwx)`` frozen states.
        vectors: indexable as ``vectors[block][ik0]`` -- an array or the loop's
            own list of lists -- each ``(N nbnd, nstate)``.
        weights: ``(blocks, nk0, nstate)``. The occupations ``w_k0 f_j / N``,
            or any other per-state weight on the same scale.
        box_index: ``(nk0, N, npwx)`` flat ultracell box index.
        grid: the ultracell box shape.
        volume: the **unit cell** volume in bohr^3, which is what makes the
            density per unit cell rather than per ultracell.
        batch: ultracell states in flight at once; one by default, because each
            one holds a whole ultracell box. Nothing here ever materialises the
            ``(nstate, N npwx)`` block :meth:`UltracellStates.block` builds --
            the state axis is streamed, which is the ``N^2`` this side does not
            pay.
    """
    blocks, nk0 = int(np.shape(coefficients)[0]), int(np.shape(coefficients)[1])
    total = jnp.zeros((int(nspin_mag),) + tuple(grid), dtype=dtype)
    for block in range(blocks):
        for ik in range(nk0):
            if int(npol) == 2:
                total = total + spinor_ultracell_density(
                    coefficients[block, ik], vectors[block][ik],
                    jnp.asarray(weights[block][ik]), box_index[ik], grid,
                    volume, nspin_mag=int(nspin_mag), batch=batch,
                )
            else:
                total = total.at[block].add(ultracell_density(
                    coefficients[block, ik], vectors[block][ik],
                    jnp.asarray(weights[block][ik]), box_index[ik], grid,
                    volume, batch=batch,
                ))
    return total
