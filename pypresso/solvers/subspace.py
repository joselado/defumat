"""Working in a subspace: the generalised eigenproblem and Rayleigh-Ritz.

Both iterative diagonalisation and the choice of starting wavefunctions come
down to the same operation -- project the Hamiltonian onto a small set of trial
vectors, diagonalise there, and rotate the vectors onto the result. Davidson
does it once per iteration on a growing subspace; ``wfcinit`` does it once on
the pseudo-atomic orbitals. Shared here so there is one implementation to get
right.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

__all__ = ["generalised_eigh", "rayleigh_ritz"]


def generalised_eigh(h, s):
    """Eigenpairs of ``H v = e S v`` for Hermitian ``H`` and positive ``S``.

    QE's ``cdiaghg``: reduce to a standard problem through the Cholesky factor
    of the overlap. Writing ``S = L L^H``, the substitution ``v = L^-H u`` gives
    ``(L^-1 H L^-H) u = e u``, which is Hermitian, and the eigenvectors come back
    with one triangular solve.
    """
    factor = jnp.linalg.cholesky(s)
    reduced = solve_triangular(factor, h, lower=True)
    reduced = solve_triangular(factor, reduced.conj().T, lower=True).conj().T
    reduced = 0.5 * (reduced + reduced.conj().T)

    values, vectors = jnp.linalg.eigh(reduced)
    return values, solve_triangular(factor.conj().T, vectors, lower=False)


@partial(jax.jit, static_argnames=("nbnd",))
def rayleigh_ritz(hamiltonian, ik, vectors, nbnd: int):
    """The ``nbnd`` best approximate eigenpairs inside the span of ``vectors``.

    QE's ``rotate_wfc``. Given trial vectors that are not orthonormal and not
    eigenvectors -- pseudo-atomic orbitals, typically -- this returns the
    combinations of them that diagonalise the Hamiltonian within their span.
    It is the difference between handing an iterative solver a pile of atomic
    orbitals and handing it something that already looks like the answer.

    Args:
        vectors: ``(nvec, npwx)`` trial vectors, with ``nvec >= nbnd``.

    Returns ``(eigenvalues, wavefunctions)`` shaped ``(nbnd,)`` and
    ``(nbnd, npwx)``.
    """
    mask = hamiltonian.state_mask[ik]
    vectors = jnp.where(mask, vectors, 0.0)
    applied = hamiltonian.apply(vectors, ik)

    h = vectors.conj() @ applied.T
    # The overlap is <psi|S|psi>, not <psi|psi>. With an ultrasoft
    # pseudopotential the atomic orbitals are not S-orthonormal -- their
    # S-norms are off by tens of percent -- so using the plain inner product
    # here returns combinations that are not even approximately eigenvectors,
    # and the first Davidson call starts further from the answer than a random
    # guess would. ``rotate_wfc`` calls ``s_psi`` for exactly this reason.
    s = vectors.conj() @ hamiltonian.apply_s(vectors, ik).T
    h = 0.5 * (h + h.conj().T)
    s = 0.5 * (s + s.conj().T)

    values, coefficients = generalised_eigh(h, s)
    return values[:nbnd].real, coefficients[:, :nbnd].T @ vectors
