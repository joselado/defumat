"""Exact diagonalisation, for tests only.

Correctness in this project is established against Quantum ESPRESSO: the same
input through both codes, compared term by term. That is what the regression
suite does, and it is why the package ships **one** eigensolver -- QE's block
Davidson -- rather than keeping a dense one alongside it as an internal
reference. A dense solve costs ``O(npw^3)`` time and ``O(npw^2)`` memory, which
is the largest single allocation a plane-wave code can make and the one thing
the whole design of an iterative solver exists to avoid; shipping it invites it
to be selected.

An exact answer is still the right check for the *fast* tests, though. Davidson
fails by quietly not converging -- plausible numbers, wrong in the fourth
decimal -- and on a cell of a couple of hundred plane waves, forming ``H`` and
handing it to ``eigh`` settles the question in milliseconds without a QE run.
So the dense solve lives here, where it is a test fixture and cannot be reached
from a calculation.

It is three lines of algebra on top of the Hamiltonian, which still builds its
own matrix (``matrix``, ``overlap_matrix``) for exactly this sort of use.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from pypresso.solvers.subspace import generalised_eigh

__all__ = ["exact_eigenpairs", "exact_eigenpairs_all"]


def exact_eigenpairs(hamiltonian, ik: int, nbnd: int):
    """The ``nbnd`` lowest eigenpairs at k-point ``ik``, by forming ``H``.

    Returns ``(eigenvalues, eigenvectors)`` with eigenvalues in Ry, ascending,
    and eigenvectors as ``(nbnd, npwx)`` -- bands first, matching how the rest
    of the code carries wavefunctions.

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
    # same reason H is -- this is meant to be obviously right, not fast.
    eigenvalues, eigenvectors = generalised_eigh(matrix, hamiltonian.overlap_matrix(ik))
    return eigenvalues[:nbnd].real, eigenvectors[:, :nbnd].T


def exact_eigenpairs_all(hamiltonian, nbnd: int):
    """Every k-point, ``vmap``-ed. Only for cells small enough to form ``H``."""
    return jax.vmap(lambda ik: exact_eigenpairs(hamiltonian, ik, nbnd))(
        jnp.arange(hamiltonian.nk)
    )
