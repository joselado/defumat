"""The ultracell Hamiltonian, in the frozen unit-cell states at ``k0 + Q``.

``PLAN.md`` P88. One ultracell Brillouin zone point ``k0``, one dense matrix:

    H_{(Q,m),(Q',n)} = delta_{QQ'} delta_{mn} eps_{k0+Q,n}
                       + <psi_{k0+Q,m}| dV |psi_{k0+Q',n}>

with the integral over the **ultracell** and ``dV`` the difference potential
:func:`~defumat.ultracell.potential.delta_potential`. The diagonal is the unit
cell's own eigenvalue and nothing else, because ``psi_{k0+Q,n}`` is an exact
eigenstate of the unmodulated Hamiltonian -- the kinetic energy, the nonlocal
pseudopotential and the whole unit-cell potential are in that one number, which
is why the atoms are allowed to stay where they are and why no ``h_psi`` runs
inside this loop.

**This is the direct route** (``PLAN.md`` P88's stage 1): the bra and the ket
are the states at ``k0 + Q`` and ``k0 + Q'`` themselves, so the only
approximation between this matrix and the exact ``N``-cell supercell is the
truncation to ``nbnd`` bands per folded k-point. Elk cannot write it -- LAPW
cannot apply ``H(k0+Q)`` to a state expressed at ``k0`` -- and represents every
state in one basis instead (``genhmlu``, ``hdbulrk``), which is cheaper by a
factor ``N`` and is a second approximation. That route is stage 2 and is not
here.

**Normalisation.** The basis functions are normalised over the *unit* cell, so
``<Q,m|Q',n>`` over the ultracell is ``N delta delta`` and dividing by ``N``
makes the overlap the identity: the matrix below is the unit-cell integral, and
there is no ``N`` anywhere in it. The ``e^{i(Q''+Q'-Q).r}`` that the ultracell
sum would leave behind is absorbed by the box index map -- see
:mod:`defumat.ultracell.grid`, and it is the reason no pair of ``Q`` values
needs a phase factor here.

**How many blocks there are is the whole difference between the regimes.**
Without spin-orbit coupling a collinear potential is diagonal in spin, so the
``(2 N nbnd)`` matrix is block diagonal and its two blocks are this function
called twice, with ``dV[0]`` and ``dV[1]``: there is no spin axis in the build
at all, and that is physics rather than an omission. A **spinor** run is one
block on a space twice as large. Its basis function is a two-component object,
the potential acts on it as the 2x2 matrix ``v_0 I + B . sigma``, and the two
components are transformed independently and mixed pointwise -- ``vloc_psi_nc``,
and :func:`~defumat.hamiltonian.noncollinear.spin_multiply` is the one place
that algebra is written. ``npol`` selects between the two.

**What a spinor block is *not* is two collinear blocks of a doubled size.** The
off-diagonal Pauli terms couple the components at every point, so nothing about
the matrix factorises, and the states it is built from hold **one** electron
each rather than two.

**Cost.** Two ultracell transforms per ket basis function and per spinor
component, so ``2 npol N nbnd`` ultracell FFTs per ``k0`` per iteration, which
is ``2 npol N^2 nbnd`` unit-cell transforms. That ``N^2`` is the price of the
direct route and is what stage 2 removes; the ``npol`` is the price of the
spin, and a spinor run pays it twice over because ``nbnd`` counts spinor bands,
of which a given electron count needs twice as many.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from defumat.batching import map_k
from defumat.hamiltonian.noncollinear import spin_multiply

__all__ = ["ultracell_matrix", "multiplet_cut"]


def ultracell_matrix(
    coefficients: jnp.ndarray,
    eigenvalues: jnp.ndarray,
    box_index: jnp.ndarray,
    delta_v: jnp.ndarray,
    grid: tuple[int, int, int],
    batch: int | None = 1,
    npol: int = 1,
    augmentation: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """The ``(N nbnd, N nbnd)`` matrix at one ``k0``.

    Args:
        coefficients: ``(N, nbnd, npol npwx)`` frozen states at ``k0 + Q``,
            padding entries already zero. A spinor state holds its two
            components one after the other, as
            :class:`~defumat.hamiltonian.noncollinear.SpinorHamiltonian` stores
            them.
        eigenvalues: ``(N, nbnd)`` in Ry.
        box_index: ``(N, npwx)`` flat ultracell box index of every plane wave.
            **One map for both spinor components**: they live on the same
            sphere, which is what makes a spinor different from a spiral.
        delta_v: the real difference potential in Ry -- ``(*grid)`` for one
            collinear channel, ``(nspin_mag, *grid)`` for a spinor block, where
            the components are ``(v_0, B_x, B_y, B_z)`` and one component alone
            means a nonmagnetic spin-orbit run.
        grid: the ultracell box shape.
        batch: how many ket basis functions are in flight at once. **One by
            default and never ``None`` by accident**: each one holds a whole
            ultracell box, so a ``vmap`` over the basis axis would materialise
            ``N nbnd`` of them at once, which is the peak this method is
            supposed not to have. A spinor ket holds ``npol`` boxes.
        npol: spinor components per basis function, 2 noncollinear and 1 not.
        augmentation: the ``(N nbnd, N nbnd)`` augmentation term of an
            ultrasoft or PAW dataset, from
            :func:`~defumat.ultracell.augmentation.augmentation_matrix`, or
            ``None``. It is added **inside** this function rather than by the
            caller so that the one Hermitian symmetrisation below covers both
            halves of ``dV``: the two are the smooth and the augmented parts of
            a single matrix element, and nothing downstream should be able to
            see them apart.

    The state index is ``(Q, n)`` flattened C-order, so band ``n`` of ``Q``
    is row ``Q * nbnd + n`` -- the same order
    :func:`~defumat.ultracell.density.ultracell_density` and
    :func:`~defumat.ultracell.density.spinor_ultracell_density` unpack.
    """
    cells, nbnd, width = coefficients.shape
    npwx = width // int(npol)
    points = int(grid[0]) * int(grid[1]) * int(grid[2])
    flat_index = box_index.reshape(-1)
    potential = delta_v if npol == 2 else None
    if npol == 2 and delta_v.ndim == 3:
        potential = delta_v[None]
    delta_flat = None if npol == 2 else delta_v.reshape(-1)
    bras = jnp.conj(coefficients).reshape(cells, nbnd, npol, npwx)

    def column(ket):
        """``<bra | dV | ket>`` for every bra, one ultracell state as the ket.

        The scatter, the two transforms and the gather are the same operations
        whatever ``npol`` is -- an FFT knows nothing about spin. The only place
        the regimes differ is the pointwise multiply between them, which is a
        real number for a collinear channel and a 2x2 matrix mixing the
        components for a spinor.
        """
        components = ket["c"].reshape(npol, npwx)
        scattered = jnp.zeros((npol, points), dtype=coefficients.dtype)
        scattered = scattered.at[:, box_index[ket["q"]]].add(components)
        field = jnp.fft.ifftn(
            scattered.reshape((npol,) + tuple(grid)), axes=(-3, -2, -1)
        ) * points
        if npol == 2:
            product = spin_multiply(field, potential)
        else:
            product = delta_flat.reshape(grid)[None] * field
        acted = jnp.fft.fftn(product, axes=(-3, -2, -1)).reshape(npol, -1) / points
        gathered = acted[:, flat_index].reshape((npol,) + box_index.shape)
        return jnp.einsum("qmap,aqp->qm", bras, gathered).reshape(-1)

    kets = {
        "c": coefficients.reshape(cells * nbnd, -1),
        "q": jnp.repeat(jnp.arange(cells), nbnd),
    }
    columns = map_k(column, kets, batch=batch)

    matrix = columns.T + jnp.diag(eigenvalues.reshape(-1).astype(columns.dtype))
    if augmentation is not None:
        matrix = matrix + jnp.asarray(augmentation).astype(matrix.dtype)
    # Hermitian by construction -- ``dV`` is real -- so this only removes the
    # round-off asymmetry the two transforms leave, and it is what ``eigh``
    # would silently impose anyway by reading one triangle.
    return 0.5 * (matrix + jnp.conj(matrix).T)


def multiplet_cut(eigenvalues: jnp.ndarray, nbnd: int) -> float:
    """The gap the band truncation falls in, at its tightest over the folded k-set.

    ``min_k (eps[nbnd] - eps[nbnd-1])`` in Ry: the distance from the last
    retained band to the first discarded one, so ``eigenvalues`` has to carry
    **at least** ``nbnd + 1`` bands on its last axis -- the driver solves once
    more with one band more than it keeps for exactly this reason
    (:func:`~defumat.ultracell.driver.run_ultracell`).

    **Why this is checked rather than assumed.** The ultracell basis is the span
    of the retained states, and a span is invariant under any unitary mixing
    inside it -- unless the truncation *cuts a degenerate multiplet*, in which
    case which member of the multiplet survives is whatever the eigensolver
    happened to return, and the basis is arbitrary. That is rule D4 at the
    ``nbnd`` boundary: the answer moves and no symmetry check sees it. The fix
    is to raise ``nbnd`` until the cut falls in a gap, and this is the number
    that says whether it does.

    **The gap has to straddle the cut, and the first form of this did not.**
    It read ``eps[nbnd-1] - eps[nbnd-2]``, the spacing between the last two
    *retained* bands, because band ``nbnd`` was never solved. That is a
    different number and it fails in both directions: silicon at ``Gamma`` is
    ``-0.394, 0.499 x3, 0.672 x3, 0.758`` Ry, and ``nbnd = 5``, which splits the
    ``Gamma_15`` triplet, read 0.17 Ry, while ``nbnd = 4``, which cuts in a
    0.17 Ry gap, read zero.
    """
    eigenvalues = jnp.asarray(eigenvalues)
    nbnd = int(nbnd)
    if nbnd < 1:
        raise ValueError(f"nbnd must be at least 1, got {nbnd}")
    if eigenvalues.shape[-1] <= nbnd:
        raise ValueError(
            f"the gap across an nbnd = {nbnd} cut needs band {nbnd + 1} as "
            f"well, and only {eigenvalues.shape[-1]} bands were passed: solve "
            f"one band more than is kept"
        )
    gaps = eigenvalues[..., nbnd] - eigenvalues[..., nbnd - 1]
    return float(jnp.min(gaps))
