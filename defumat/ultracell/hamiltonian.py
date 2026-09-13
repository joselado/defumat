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

**Cost.** Two ultracell transforms per ket basis function, so ``2 N nbnd``
ultracell FFTs per ``k0`` per iteration, which is ``2 N^2 nbnd`` unit-cell
transforms. That ``N^2`` is the price of the direct route and is what stage 2
removes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from defumat.batching import map_k

__all__ = ["ultracell_matrix", "multiplet_cut"]


def ultracell_matrix(
    coefficients: jnp.ndarray,
    eigenvalues: jnp.ndarray,
    box_index: jnp.ndarray,
    delta_v: jnp.ndarray,
    grid: tuple[int, int, int],
    batch: int | None = 1,
) -> jnp.ndarray:
    """The ``(N nbnd, N nbnd)`` matrix at one ``k0``.

    Args:
        coefficients: ``(N, nbnd, npwx)`` frozen states at ``k0 + Q``, padding
            entries already zero.
        eigenvalues: ``(N, nbnd)`` in Ry.
        box_index: ``(N, npwx)`` flat ultracell box index of every plane wave.
        delta_v: ``(*grid)`` real difference potential in Ry.
        grid: the ultracell box shape.
        batch: how many ket basis functions are in flight at once. **One by
            default and never ``None`` by accident**: each one holds a whole
            ultracell box, so a ``vmap`` over the basis axis would materialise
            ``N nbnd`` of them at once, which is the peak this method is
            supposed not to have.

    The state index is ``(Q, n)`` flattened C-order, so band ``n`` of ``Q``
    is row ``Q * nbnd + n`` -- the same order
    :func:`~defumat.ultracell.density.ultracell_density` unpacks.
    """
    cells, nbnd, _ = coefficients.shape
    points = int(grid[0]) * int(grid[1]) * int(grid[2])
    flat_index = box_index.reshape(-1)
    delta_flat = delta_v.reshape(-1)

    def column(ket):
        """``<bra | dV | ket>`` for every bra, one ultracell state as the ket."""
        scattered = jnp.zeros(points, dtype=coefficients.dtype)
        scattered = scattered.at[box_index[ket["q"]]].add(ket["c"])
        field = jnp.fft.ifftn(scattered.reshape(grid)) * points
        acted = jnp.fft.fftn(delta_flat.reshape(grid) * field).reshape(-1) / points
        gathered = acted[flat_index].reshape(box_index.shape)
        return jnp.einsum("qmp,qp->qm", jnp.conj(coefficients), gathered).reshape(-1)

    kets = {
        "c": coefficients.reshape(cells * nbnd, -1),
        "q": jnp.repeat(jnp.arange(cells), nbnd),
    }
    columns = map_k(column, kets, batch=batch)

    matrix = columns.T + jnp.diag(eigenvalues.reshape(-1).astype(columns.dtype))
    # Hermitian by construction -- ``dV`` is real -- so this only removes the
    # round-off asymmetry the two transforms leave, and it is what ``eigh``
    # would silently impose anyway by reading one triangle.
    return 0.5 * (matrix + jnp.conj(matrix).T)


def multiplet_cut(eigenvalues: jnp.ndarray, tolerance: float = 1.0e-5) -> float:
    """The smallest gap the band truncation opens, over the folded k-set.

    ``min_k (eps[nbnd-1] - eps[nbnd-2])`` in Ry: how far the last retained band
    is from the first discarded one's neighbour at the point where the cut is
    tightest.

    **Why this is checked rather than assumed.** The ultracell basis is the span
    of the retained states, and a span is invariant under any unitary mixing
    inside it -- unless the truncation *cuts a degenerate multiplet*, in which
    case which member of the multiplet survives is whatever the eigensolver
    happened to return, and the basis is arbitrary. That is rule D4 at the
    ``nbnd`` boundary: the answer moves and no symmetry check sees it. The fix
    is to raise ``nbnd`` until the cut falls in a gap, and this is the number
    that says whether it does.
    """
    if eigenvalues.shape[-1] < 2:
        return float("inf")
    gaps = eigenvalues[..., -1] - eigenvalues[..., -2]
    return float(jnp.min(gaps))
