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


#: A direction of the overlap below this fraction of its largest eigenvalue is
#: treated as outside the subspace rather than inside it. The failure it guards
#: is not marginal-looking: the eigenvalue measured there was **-4.3e-16**
#: against a largest of 1.0 -- zero to round-off, and negative.
OVERLAP_FLOOR = 1.0e-12


def _cholesky_route(h, s):
    """QE's ``cdiaghg``: reduce through the Cholesky factor of the overlap.

    Writing ``S = L L^H``, the substitution ``v = L^-H u`` gives
    ``(L^-1 H L^-H) u = e u``, which is Hermitian, and the eigenvectors come
    back with one triangular solve. This is the fast path and the one every
    number in this project was produced with.
    """
    factor = jnp.linalg.cholesky(s)
    reduced = solve_triangular(factor, h, lower=True)
    reduced = solve_triangular(factor, reduced.conj().T, lower=True).conj().T
    reduced = 0.5 * (reduced + reduced.conj().T)

    values, vectors = jnp.linalg.eigh(reduced)
    return values, solve_triangular(factor.conj().T, vectors, lower=False)


def _canonical_route(h, s):
    """Canonical orthogonalisation, for an overlap that is no longer positive.

    Diagonalise ``S = U w U^H`` and work in ``X = U w^-1/2``, which is the
    standard construction; the directions whose ``w`` has fallen to the round-off
    floor are *projected out* rather than inverted, by parking them at an energy
    above the spectrum so they cannot enter the lowest roots. Shapes stay static,
    which is why they are parked rather than dropped.
    """
    w, u = jnp.linalg.eigh(s)
    keep = w > OVERLAP_FLOOR * jnp.max(w)
    safe = jnp.where(keep, w, 1.0)
    x = u / jnp.sqrt(safe)[None, :]

    reduced = x.conj().T @ h @ x
    reduced = 0.5 * (reduced + reduced.conj().T)
    # Above anything physical, by the same argument the Davidson driver uses for
    # its own inactive directions: the diagonal bounds the spectrum well enough.
    shift = jnp.max(jnp.abs(jnp.diagonal(reduced).real)) * 1000.0 + 1.0
    pair = keep[:, None] & keep[None, :]
    reduced = jnp.where(pair, reduced, 0.0) + jnp.diag(
        jnp.where(keep, 0.0, shift).astype(reduced.dtype))

    values, vectors = jnp.linalg.eigh(reduced)
    return values, x @ vectors


def generalised_eigh(h, s):
    """Eigenpairs of ``H v = e S v`` for Hermitian ``H`` and positive ``S``.

    **``S`` stops being positive, and when it does JAX does not say so.** As
    Davidson's subspace fills, the vectors it expands with are normalised
    residuals of roots that have already converged -- which is amplified
    round-off, and goes linearly dependent. The overlap's smallest eigenvalue
    then sits *at* the round-off floor and its sign is arbitrary:
    ``jnp.linalg.cholesky`` of a matrix with a tiny negative eigenvalue takes
    the square root of a negative pivot and **returns NaN rather than raising**,
    so the failure travels silently into the density, the mixer and the total
    energy. QE's ``cdiaghg`` hits the same wall and stops with "problems
    computing cholesky"; returning ``NaN`` is strictly worse than stopping.

    Measured (`PERFORMANCE.md`): 64 atoms at ``ecutwfc = 30``, on a GPU,
    ``min eig(S) = -4.3e-16`` against ``max|S| = 1.0`` at the ninetieth call --
    after which ``S`` arrived non-finite 452 times and the run ended in ``NaN``
    having converged happily to ``conv_thr = 1e-8`` on the way. The identical
    input on a CPU converges: at that size the eigenvalue is *zero to round-off*
    and which side of zero it lands on is a coin flip, which is the whole of why
    this looked GPU-specific.

    So Cholesky stays the fast path -- it is what QE does and what every
    validated number here was produced with, and it is taken bit-for-bit
    whenever it works -- and the canonical-orthogonalisation route is used only
    when it has failed. ``lax.cond`` traces both and runs one.
    """
    factor = jnp.linalg.cholesky(s)
    return jax.lax.cond(
        jnp.all(jnp.isfinite(factor)),
        lambda: _cholesky_route(h, s),
        lambda: _canonical_route(h, s),
    )


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
