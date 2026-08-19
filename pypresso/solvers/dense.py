"""Reference eigensolver: build the Hamiltonian and diagonalise it exactly.

This is not how a plane-wave code should solve for its bands -- it costs
``O(npw^3)`` time and ``O(npw^2)`` memory, and ``npw`` runs to tens of thousands
in a real calculation. It exists because it is *unambiguously correct*: an
iterative solver that disagrees with it has a bug, and one that agrees can be
trusted on systems too large to check this way.

For the small cells the project validates against (silicon at 12 Ry has 186
plane waves) it is also perfectly fast.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from pypresso.hamiltonian.operator import Hamiltonian
from pypresso.solvers.subspace import generalised_eigh

__all__ = ["dense_eigensolver", "dense_eigensolver_all"]


def dense_eigensolver(hamiltonian: Hamiltonian, ik: int, nbnd: int):
    """The ``nbnd`` lowest eigenpairs at k-point ``ik``.

    Returns ``(eigenvalues, eigenvectors)`` with eigenvalues in Ry, ascending,
    and eigenvectors as ``(nbnd, npwx)`` -- bands first, matching how the rest of
    the code carries wavefunctions.

    Padded plane waves are projected out first. They would otherwise appear as
    spurious zero eigenvalues sitting in the middle of the spectrum.
    """
    matrix = hamiltonian.matrix(ik)
    mask = hamiltonian.state_mask[ik]

    # Push padding rows/columns far above the physical spectrum instead of
    # deleting them, so the matrix keeps its static shape.
    shift = jnp.max(jnp.abs(matrix)) * 1000.0 + 1.0
    matrix = jnp.where(mask[:, None] & mask[None, :], matrix, 0.0)
    matrix = matrix + jnp.diag(jnp.where(mask, 0.0, shift))

    if not hamiltonian.has_overlap:
        eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
        return eigenvalues[:nbnd], eigenvectors[:, :nbnd].T

    # Ultrasoft: the problem is H v = e S v, and S is built explicitly for the
    # same reason H is -- this solver exists to be obviously right, not fast.
    eigenvalues, eigenvectors = generalised_eigh(matrix, hamiltonian.overlap_matrix(ik))
    return eigenvalues[:nbnd].real, eigenvectors[:, :nbnd].T


@partial(jax.jit, static_argnames=("nbnd",))
def dense_eigensolver_all(hamiltonian: Hamiltonian, nbnd: int, psi0=None, ethr=None):
    """Every k-point at once: the same solver, ``vmap``-ed over ``ik``.

    ``psi0`` and ``ethr`` are accepted and ignored: a direct solve has no use for
    a starting guess and no threshold to converge to, but the registry gives
    every solver the same signature so that the driver does not have to know
    which kind it is holding.

    The k index is the leading axis of every wavefunction-shaped array (rule
    R6), so batching over it is a ``vmap`` and nothing else. Doing it this way
    rather than in a Python loop turns ``nk`` separate compilations-and-launches
    of each kernel into one, which is where most of the loop's cost was: the
    arithmetic per k-point is unchanged.

    Returns ``(eigenvalues, wavefunctions)`` of shapes ``(nk, nbnd)`` and
    ``(nk, nbnd, npwx)``.
    """
    def solve(ik):
        return dense_eigensolver(hamiltonian, ik, nbnd)

    return jax.vmap(solve)(jnp.arange(hamiltonian.nk))
