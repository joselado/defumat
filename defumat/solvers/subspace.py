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

from defumat.basis.fft import force_real_g0
from defumat.config import subspace_dtype
from jax.lax.linalg import EighImplementation, eigh as _lax_eigh
from jax.scipy.linalg import solve_triangular

__all__ = ["generalised_eigh", "rayleigh_ritz"]


#: A direction of the overlap below this fraction of its largest eigenvalue is
#: treated as outside the subspace rather than inside it. The failure it guards
#: is not marginal-looking: the eigenvalue measured there was **-4.3e-16**
#: against a largest of 1.0 -- zero to round-off, and negative.
OVERLAP_FLOOR = 1.0e-12


def _eigh(x):
    """``jnp.linalg.eigh`` with cuSOLVER's ``syevd`` asked for by name.

    Left to choose, jaxlib takes cuSOLVER's Jacobi solver for a matrix of 32 or
    fewer rows and ``syevd`` above, and at Davidson's first rungs the Jacobi one
    is the slow and the inaccurate one. One complex128 matrix on an RTX A2000:
    6.00 ms against 0.63 for ``syevd`` at ``m = 16``, 6.94 against 1.11 at 32,
    with an eigenvalue error of 2e-13 against 2e-14 there; above 32 the two are
    the same call. Profiled on a 27-k-point eight-atom SCF in memory mode, one
    k-point a call, the subspace solve was 1.9 s of 3.2 s of kernel time, most
    of it the Jacobi kernels. On a CPU every route is LAPACK's ``heevd``, so
    nothing there changes.
    """
    vectors, values = _lax_eigh(x, implementation=EighImplementation.QR)
    return values, vectors


def _park_above(reduced, parked):
    """``reduced`` with its ``parked`` rows decoupled and one above everything else.

    The value is one above the Gershgorin bound of the rest, its largest absolute
    row sum, which no eigenvalue of it can exceed, so the parked directions sort
    last whatever the live spectrum is. It is a device and not physics, so it
    carries no gradient.
    """
    live = jnp.logical_not(parked)
    reduced = jnp.where(live[:, None] & live[None, :], reduced, 0.0)
    bound = jax.lax.stop_gradient(jnp.max(jnp.sum(jnp.abs(reduced), axis=1)))
    return reduced + jnp.diag(jnp.where(parked, bound + 1.0, 0.0).astype(reduced.dtype))


def _cholesky_route(h, s, parked=None):
    """QE's ``cdiaghg``: reduce through the Cholesky factor of the overlap.

    Writing ``S = L L^H``, the substitution ``v = L^-H u`` gives
    ``(L^-1 H L^-H) u = e u``, which is Hermitian, and the eigenvectors come
    back with one triangular solve. This is the fast path and the one every
    number in this project was produced with.

    ``parked`` marks rows the caller has decoupled in both matrices, with a unit
    overlap, so that the factor is the identity there and the reduction leaves
    them alone; they are put above the reduced live block here
    (:func:`generalised_eigh` says why it is done after the reduction).
    """
    factor = jnp.linalg.cholesky(s)
    reduced = solve_triangular(factor, h, lower=True)
    reduced = solve_triangular(factor, reduced.conj().T, lower=True).conj().T
    reduced = 0.5 * (reduced + reduced.conj().T)
    if parked is not None:
        reduced = _park_above(reduced, parked)

    values, vectors = _eigh(reduced)
    return values, solve_triangular(factor.conj().T, vectors, lower=False)


def _canonical_route(h, s, parked=None):
    """Canonical orthogonalisation, for an overlap that is no longer positive.

    Diagonalise ``S = U w U^H`` and work in ``X = U w^-1/2``, which is the
    standard construction; the directions whose ``w`` has fallen to the round-off
    floor are *projected out* rather than inverted, by parking them at an energy
    above the spectrum so they cannot enter the lowest roots. Shapes stay static,
    which is why they are parked rather than dropped.

    **Where they are parked sets the norm of the matrix ``eigh`` is handed, and
    with it the absolute error of every eigenvalue.** They sit one above the
    Gershgorin bound of the kept block, its largest absolute row sum, which no
    eigenvalue of that block can exceed, so they sort last whatever the kept
    spectrum is, and the norm stays of the order of the kept block's own. They
    were at 1000 times the largest diagonal element of ``reduced``, and while
    Davidson parked its own idle directions on the diagonal of ``H`` (at 4 times
    its largest element) that diagonal carried them, so the retry handed
    ``eigh`` a matrix of norm 4000 times that of ``H``'s diagonal: 1.2e5 at
    30 Ry, four times the norm that made a card's Davidson stall at the ``ethr``
    floor (``PERFORMANCE.md``, "The endgame on a card is a stall"). The kept block's eigenvalues do not depend on
    the parked value.

    A caller's ``parked`` rows are made null directions of the overlap, so that
    they are dropped and parked with the rest. Leaving them at a unit overlap
    would not do: a refreshed Davidson block has an overlap of exactly the
    identity, degenerate with theirs, and ``eigh(S)`` is free to rotate the two
    together, after which no row of ``X`` is a parked one.
    """
    if parked is not None:
        live = jnp.logical_not(parked)
        pair = live[:, None] & live[None, :]
        h, s = jnp.where(pair, h, 0.0), jnp.where(pair, s, 0.0)
    w, u = _eigh(s)
    keep = w > OVERLAP_FLOOR * jnp.max(w)
    safe = jnp.where(keep, w, 1.0)
    x = u / jnp.sqrt(safe)[None, :]

    reduced = x.conj().T @ h @ x
    reduced = 0.5 * (reduced + reduced.conj().T)
    reduced = _park_above(reduced, jnp.logical_not(keep))

    values, vectors = _eigh(reduced)
    return values, x @ vectors


def _route(route, h, s, parked):
    """``route(h, s)``, with the parked rows when there are any."""
    return route(h, s) if parked is None else route(h, s, parked)


def generalised_eigh(h, s, robust: bool | None = None, parked=None):
    """Eigenpairs of ``H v = e S v`` for Hermitian ``H`` and positive ``S``.

    **``parked`` marks rows that are not in the problem**, decoupled by the caller
    in both matrices with a unit overlap: Davidson's subspace directions not in
    use, under the one shape a ``while_loop`` allows. They come back as the top
    eigenvalues, one above the Gershgorin bound of the reduced live block, and
    the live eigenpairs are those of the live block alone. **Where they sit is
    the norm of the matrix ``eigh`` is handed, and a device ``eigh``'s error
    follows that norm** where LAPACK's does not: on the 72 captured solves of
    sixteen-atom silicon at 30 Ry that had parked rows, a card's error in the
    lowest 32 roots was 3.4e-12 Ry median with them at 1000 times the largest
    diagonal element of ``H`` (norm 2.9e4), 5.2e-15 median and 1.6e-13 at worst
    at 4 times it (norm 118), and 2.6e-15 median and 1.4e-14 at worst one above
    the live block's bound (its norm 6 to 9 Ry), against 2.4e-15 and 9.3e-15 for
    the live block alone and 2.9e-15 and 9.3e-15 for the host at any of them
    (``PERFORMANCE.md``, "The endgame on a card is a stall"). The bound is taken
    after the reduction, on the matrix ``eigh`` sees, because the live block of
    ``H`` is not what bounds the spectrum when ``S`` is not the identity.

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

    **Except under ``vmap``, where it runs both, and that is what ``robust``
    exists for.** A ``cond`` whose predicate is batched has no branch to take:
    JAX's batching rule lowers it to ``select_n`` over the results of *both*
    branches. ``k_batch=None`` -- the default on an accelerator since the dials
    became per-platform -- is exactly a ``vmap`` over the k axis, so on a GPU
    every multi-k Davidson step has been paying the canonical route as well as
    the Cholesky one, on top of the solve it actually uses. Measured on this
    workstation at ``si10-nc``'s own shapes (80 x 80, seven k-points):
    **42.5 ms against 14.9 for the Cholesky route alone, 2.85x**, where
    unbatched the two are within a percent of each other. It is a lowering fact
    rather than a hardware one, which is why a CPU can measure it.

    ``robust`` therefore selects the route **statically**, so that a caller in a
    batched hot loop can take the fast one with no ``cond`` in the graph at all
    and handle the failure where the predicate is *not* batched -- which is what
    :func:`~defumat.solvers.davidson.davidson_eigensolver_all` does, one level
    outside ``map_k``:

    * ``None`` (the default) keeps the guard, for callers that solve once --
      ``rayleigh_ritz``, the exact-reference fixture -- where 2.85x of one small
      solve per SCF is not worth a second code path;
    * ``False`` is the Cholesky route alone, and is bit-for-bit what the guarded
      version returns whenever the guard passes;
    * ``True`` is canonical orthogonalisation alone, which is the retry.
    """
    if robust is True:
        return _route(_canonical_route, h, s, parked)
    if robust is False:
        return _route(_cholesky_route, h, s, parked)
    factor = jnp.linalg.cholesky(s)
    return jax.lax.cond(
        jnp.all(jnp.isfinite(factor)),
        lambda: _route(_cholesky_route, h, s, parked),
        lambda: _route(_canonical_route, h, s, parked),
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
    # ``rotate_wfc``'s gamma counterpart (``rotate_wfc_gamma``) works in real
    # arithmetic, so the trial vectors enter with ``Im c(0) = 0`` and the two
    # subspace matrices are real. Without the first the rotated vectors carry a
    # complex ``G = 0`` into the SCF; without the second the atomic guess is
    # rotated by the wrong combinations, which costs iterations rather than
    # correctness -- the run still converges, to the same answer, more slowly.
    gamma_only = hamiltonian.gamma_only
    vectors = force_real_g0(vectors, gamma_only)
    applied = hamiltonian.apply(vectors, ik)

    h = vectors.conj() @ applied.T
    # The overlap is <psi|S|psi>, not <psi|psi>. With an ultrasoft
    # pseudopotential the atomic orbitals are not S-orthonormal -- their
    # S-norms are off by tens of percent -- so using the plain inner product
    # here returns combinations that are not even approximately eigenvectors,
    # and the first Davidson call starts further from the answer than a random
    # guess would. ``rotate_wfc`` calls ``s_psi`` for exactly this reason.
    s = vectors.conj() @ hamiltonian.apply_s(vectors, ik).T
    if gamma_only:
        # ``2 Re(...) - (G = 0)``, the same correction every plane-wave sum
        # under half-sphere storage takes.
        h = 2.0 * h.real - jnp.real(
            vectors[:, :1].conj() * applied[:, :1].T
        )
        s = 2.0 * s.real - jnp.real(
            vectors[:, :1].conj()
            * hamiltonian.apply_s(vectors, ik)[:, :1].T
        )
        h = h.astype(vectors.dtype)
        s = s.astype(vectors.dtype)
    h = 0.5 * (h + h.conj().T)
    s = 0.5 * (s + s.conj().T)
    # solved in the subspace precision, as Davidson's projected problem is
    wide = subspace_dtype(h.dtype)
    values, coefficients = generalised_eigh(h.astype(wide), s.astype(wide))
    rotated = coefficients[:, :nbnd].T.astype(vectors.dtype) @ vectors
    return values[:nbnd].real, force_real_g0(rotated, gamma_only)
