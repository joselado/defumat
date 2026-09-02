"""The stick decomposition of the FFT box, following QE's FFT layout.

A wavefunction occupies a sphere of plane waves that fills only a few percent of
the FFT box, and the sphere's shadow on the ``xy`` plane -- the set of columns
along ``z`` that contain any of it, QE's **sticks** -- is under a fifth of the
columns. A transform that knows this does the ``z`` pass on the sticks alone and
only then fills the box, which is what ``FFTXlib`` does: ``cft_1z`` over the
sticks, then ``cft_2xy`` over the planes.

**The layout is the whole point.** QE's arrays are Fortran-ordered with ``x``
fastest, so an ``xy`` plane is contiguous and its 2D transform is cheap. A
C-ordered ``(n1, n2, n3)`` box has ``z`` fastest instead, and the same 2D
transform runs over the two *strided* axes -- on a 36x36x72 box that costs more
on its own (107 ms) than the entire fused 3D transform (68 ms). Reproducing QE's
speed therefore means reproducing QE's layout: this module's transforms hold the
field as ``(n3, n1, n2)``, with ``xy`` contiguous, and the potential is stored to
match.

Measured against a fused 3D transform of the whole box, on silicon: 1.13x for
the eight-atom cell at 2950 plane waves, 1.02x for sixteen atoms at 5900. The
saving is the ``z`` pass shrinking to a fifth; what it buys back is spent on the
scatter into the box, which both layouts have to pay.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

__all__ = ["Sticks", "build_sticks"]


class Sticks(eqx.Module):
    """Where each plane wave lives in the stick layout, for every k-point.

    ``columns[ik, s]`` is the flat ``xy`` index of the ``s``-th stick at k-point
    ``ik``, and ``index[ik, n]`` places the ``n``-th plane wave inside the
    compact ``(nsticks, n3)`` array. Both are padded to a common width so that
    the k axis stays a ``vmap`` axis: surplus sticks point at columns the sphere
    does not use, where writing zeros is what should happen anyway.
    """

    columns: jnp.ndarray  # (nk, nsticks) int, into n1*n2
    index: jnp.ndarray  # (nk, npwx) int, into nsticks*n3
    grid: tuple[int, int, int] = eqx.field(static=True)
    nsticks: int = eqx.field(static=True)

    @property
    def nk(self) -> int:
        return self.columns.shape[0]


def build_sticks(fft_index, mask, grid: tuple[int, int, int]) -> Sticks:
    """Work out the stick layout from the box indices of each plane wave.

    Host-side integer bookkeeping over a fixed G list, done once -- the
    definition of setup work.

    Args:
        fft_index: ``(nk, npwx)`` flat indices into the ``(n1, n2, n3)`` box.
        mask: ``(nk, npwx)``, false on padding.
        grid: the box dimensions.
    """
    n1, n2, n3 = (int(n) for n in grid)
    fft_index = np.asarray(fft_index)
    mask = np.asarray(mask)

    per_k = []
    for indices, keep in zip(fft_index, mask):
        column = indices // n3  # the flat xy index, since z is fastest
        per_k.append(np.unique(column[keep]))
    nsticks = max(len(s) for s in per_k)

    columns = np.zeros((len(per_k), nsticks), dtype=np.int64)
    index = np.zeros(fft_index.shape, dtype=np.int64)

    for ik, (indices, keep, sticks) in enumerate(zip(fft_index, mask, per_k)):
        # Pad with columns the sphere does not occupy, so that the surplus
        # entries of the compact array scatter zeros into empty parts of the box
        # rather than over a stick that carries something.
        spare = np.setdiff1d(np.arange(n1 * n2), sticks, assume_unique=True)
        columns[ik] = np.concatenate([sticks, spare[: nsticks - len(sticks)]])

        position = np.full(n1 * n2, -1, dtype=np.int64)
        position[sticks] = np.arange(len(sticks))
        # Padding plane waves share the index of the first slot, and are zero
        # there, so an accumulating scatter leaves it alone -- the same
        # convention scatter_to_box uses.
        where = np.where(keep, position[indices // n3] * n3 + indices % n3, 0)
        index[ik] = where

    return Sticks(
        columns=jnp.asarray(columns),
        index=jnp.asarray(index),
        grid=(n1, n2, n3),
        nsticks=int(nsticks),
    )
