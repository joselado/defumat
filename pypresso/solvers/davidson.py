"""Block Davidson: the eigensolver a plane-wave code is supposed to use.

Transcribed from ``KS_Solvers/Davidson/cegterg.f90``, which is QE's default.
The idea is the one every iterative eigensolver shares -- never form ``H``,
only apply it -- but Davidson's particular choice is what to apply it to: given
the current estimate of an eigenpair ``(e, psi)``, the residual ``(H - e) psi``
points at what the estimate is missing, and adding it to the subspace and
re-diagonalising there is a step towards the true eigenvector.

Why this matters more than any other optimisation here: building the matrix
costs ``O(npw^2)`` memory and diagonalising it ``O(npw^3)`` time, and ``npw`` is
tens of thousands in a real calculation. Davidson touches only the ``nbnd``
lowest states, so the cost is ``nbnd`` applications of ``H`` per iteration and a
dense solve in a subspace of a few times ``nbnd``. On the reference silicon cell
that is a 16x16 solve instead of 180x180.

Three deliberate departures from the Fortran, all of them forced by the rule
that shapes inside ``jit`` are static:

* **The subspace is masked, not resized.** The work arrays are always
  ``(nvecx, npwx)``; which of their rows are in play is a boolean mask, and the
  masked-out part of the projected problem is set to ``shift * I`` against an
  identity overlap, which puts its eigenvalues far above the physical spectrum
  instead of leaving a singular block.
* **Unconverged roots are compacted by sorting, not by resizing.** ``cegterg``
  moves them to the front so that the block it works on shrinks; the same
  reordering here is a stable ``argsort`` on the convergence flags, which is a
  fixed-shape operation, and the subspace then grows by the number of
  unconverged roots rather than by the full block. Both halves of that matter.
  Keeping a converged root in the expansion means normalising a residual of
  order 1e-14, which turns round-off into a basis vector and makes the overlap
  matrix singular; growing by the full block regardless means the subspace fills
  up and is collapsed long before a stubborn root -- in practice always the
  highest band, which has nothing above it to mix with -- has had a deep enough
  space to converge in. Each was found the same way: the top band of the silicon
  band structure sitting a few meV above the reference.
* **The loop is ``lax.while_loop``**, so it stays on device: no host round trip
  per Davidson iteration, which would otherwise cost more than the arithmetic.

The convergence test is QE's -- two consecutive estimates of a root differing by
less than ``ethr`` -- and the preconditioner is ``g_psi.f90``'s, including its
``TEST_NEW_PRECONDITIONING`` branch, which is the one QE compiles by default.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from pypresso.hamiltonian.operator import Hamiltonian

__all__ = ["davidson_eigensolver", "davidson_eigensolver_all", "DAVID_NDIM",
           "MAX_ITERATIONS", "ETHR", "ETHR_MIN", "RESIDUAL_THRESHOLD"]

#: QE's ``diago_david_ndim``: the subspace may grow to this many times ``nbnd``
#: before it is collapsed back onto the current eigenvector estimates.
DAVID_NDIM = 4

#: Total budget of Davidson steps, matching QE's.
#:
#: ``cegterg``'s own ``maxter`` is 20, but ``c_bands.f90`` re-enters it up to
#: five times (``ntry <= 5`` in ``test_exit_cond``), each time seeded with the
#: current estimate -- so QE's real budget is 100 steps. Re-entering is the same
#: operation as the subspace collapse this loop already performs when the basis
#: fills up, so one loop of 100 reproduces it without the outer level. It costs
#: nothing when the solve converges early, which inside an SCF it always does:
#: the budget is only reached on a cold start, and on the band-structure runs
#: where there is no SCF to spread the convergence over.
MAX_ITERATIONS = 100

#: Root-improvement threshold used when the caller does not supply one -- a
#: standalone solve, with no SCF around it to say how accurate is accurate
#: enough. Inside the SCF the driver schedules it the way ``electrons.f90`` does,
#: from 1e-2 down to the error in the density.
ETHR = 1.0e-12

#: The floor QE puts under ``ethr``: below this the iterative diagonalisation
#: becomes unstable rather than more accurate (``electrons.f90``).
ETHR_MIN = 1.0e-13

#: Optional extra test on the residual norm ``|(H - e)psi|``, per band, on top of
#: QE's test on the change in the eigenvalue. ``None`` -- the default -- is QE's
#: behaviour exactly.
#:
#: It exists because the change in an eigenvalue is a weak proxy for its error:
#: the eigenvalue is variational, so the error goes as ``|r|^2 / gap`` while the
#: change between two steps can already be tiny. That matters when the solver
#: stalls -- which it did before the expansion was restricted to unconverged
#: roots -- and it is a useful thing to be able to demand of a standalone solve.
#: Inside the SCF it is left off: ``ethr`` is scheduled against the error in the
#: density, and demanding more than that of the eigenvalues is exactly the waste
#: this schedule exists to remove.
RESIDUAL_THRESHOLD = None


def _generalised_eigh(h, s):
    """Lowest eigenpairs of ``H v = e S v`` for Hermitian ``H`` and positive ``S``.

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


def _precondition(residual, diagonal, energies):
    """``g_psi.f90``: an approximate inverse of ``H - e`` from its diagonal.

    The naive ``1/(H_ii - e)`` is unbounded where the shift meets the diagonal.
    QE's default branch replaces it with ``(1 + x + sqrt(1 + (x-1)^2)) / 2``,
    which agrees with ``x`` for large ``x`` and saturates at 1 near the pole --
    so a plane wave nearly resonant with the eigenvalue is damped rather than
    amplified.
    """
    x = diagonal[None, :] - energies[:, None]
    denominator = 0.5 * (1.0 + x + jnp.sqrt(1.0 + (x - 1.0) ** 2))
    return residual / denominator


def davidson_eigensolver(
    hamiltonian: Hamiltonian,
    ik: int,
    nbnd: int,
    psi0=None,
    ethr=None,
    residual_threshold=RESIDUAL_THRESHOLD,
    david: int = DAVID_NDIM,
    max_iterations: int = MAX_ITERATIONS,
):
    """The ``nbnd`` lowest eigenpairs at k-point ``ik``, iteratively.

    Args:
        hamiltonian: the operator; only ``apply`` and ``diagonal`` are used.
        ik: k-point index. May be traced, so this ``vmap``s over k.
        psi0: ``(nbnd, npwx)`` starting vectors -- normally the previous SCF
            iteration's wavefunctions, which is what makes later iterations
            converge in one or two Davidson steps. ``None`` starts from QE's
            random guess.
        ethr: convergence threshold on the change in each eigenvalue, in Ry.
            ``None`` uses :data:`ETHR`; the SCF driver passes its scheduled
            value, which starts loose and tightens as the density converges.

    Returns ``(eigenvalues, eigenvectors)`` with eigenvalues ascending in Ry and
    eigenvectors ``(nbnd, npwx)``, matching :func:`~pypresso.solvers.dense`.
    """
    ethr = ETHR if ethr is None else ethr
    npwx = hamiltonian.npwx
    nvecx = david * nbnd
    mask = hamiltonian.mask[ik]
    kinetic = hamiltonian.kinetic[ik]
    diagonal = hamiltonian.diagonal(ik)
    dtype = hamiltonian.projectors.vkb.dtype

    start = _starting_vectors(psi0, nbnd, npwx, kinetic, mask, dtype)

    # Inactive subspace directions are given this eigenvalue, which has to sit
    # above anything physical: the diagonal bounds the spectrum from above well
    # enough for that.
    shift = jnp.max(jnp.abs(diagonal)) * 1000.0 + 1.0

    psi = jnp.zeros((nvecx, npwx), dtype).at[:nbnd].set(start)
    hpsi = jnp.zeros((nvecx, npwx), dtype).at[:nbnd].set(hamiltonian.apply(start, ik))
    first = jnp.arange(nvecx) < nbnd

    state = (
        psi,
        hpsi,
        start,                                    # current eigenvector estimate
        jnp.full((nbnd,), jnp.inf, dtype=diagonal.dtype),  # previous eigenvalues
        first,                                    # which rows are in the subspace
        nbnd,                                     # where the next block is written
        0,                                        # iteration
        jnp.array(False),                         # converged
    )

    def unconverged(state):
        return jnp.logical_and(jnp.logical_not(state[7]), state[6] < max_iterations)

    def step(state):
        psi, hpsi, _, previous, active, nbase, iteration, _ = state

        # ... the projection of H and of the overlap onto the current subspace
        pair = active[:, None] & active[None, :]
        inactive = jnp.where(active, 0.0, 1.0).astype(dtype)
        hc = jnp.where(pair, psi.conj() @ hpsi.T, 0.0) + jnp.diag(shift * inactive)
        sc = jnp.where(pair, psi.conj() @ psi.T, 0.0) + jnp.diag(inactive)
        hc = 0.5 * (hc + hc.conj().T)
        sc = 0.5 * (sc + sc.conj().T)

        values, vectors = _generalised_eigh(hc, sc)
        energies = values[:nbnd].real
        coefficients = vectors[:, :nbnd]  # (nvecx, nbnd)

        # ... the estimate in the plane-wave basis, and H applied to it, both
        # rotations of vectors already computed -- no extra application of H.
        evc = coefficients.T @ psi
        hevc = coefficients.T @ hpsi

        # ... the residual is both what the basis is expanded with and how
        # convergence is judged; see RESIDUAL_THRESHOLD for why QE's test on the
        # eigenvalue alone is not enough once the solver is seeded.
        residual = hevc - energies[:, None].astype(dtype) * evc
        settled = jnp.abs(energies - previous) < ethr
        if residual_threshold is not None:
            settled = jnp.logical_and(
                settled,
                jnp.sqrt(jnp.sum(jnp.abs(residual) ** 2, axis=1)) < residual_threshold,
            )
        converged = jnp.all(settled)

        residual = _precondition(residual, diagonal, energies)
        residual = jnp.where(mask, residual, 0.0)
        norm = jnp.sqrt(jnp.sum(jnp.abs(residual) ** 2, axis=1, keepdims=True))
        residual = jnp.where(settled[:, None], 0.0, residual / jnp.where(norm > 0.0, norm, 1.0))

        # ... unconverged roots first, so that the block written below starts
        # with exactly the vectors worth keeping and ``notcnv`` of them are
        # marked active. The sort is stable, so roots keep their relative order.
        order = jnp.argsort(settled)
        residual = residual[order]
        notcnv = jnp.sum(jnp.logical_not(settled))

        # ... collapse onto the current estimates when the subspace is full,
        # which is the only place the basis ever shrinks (cegterg's "refresh")
        full = nbase + nbnd > nvecx
        psi, hpsi, active, nbase = jax.lax.cond(
            full,
            lambda: (
                jnp.zeros_like(psi).at[:nbnd].set(evc),
                jnp.zeros_like(hpsi).at[:nbnd].set(hevc),
                first,
                nbnd,
            ),
            lambda: (psi, hpsi, active, nbase),
        )

        psi = jax.lax.dynamic_update_slice(psi, residual, (nbase, 0))
        hpsi = jax.lax.dynamic_update_slice(hpsi, hamiltonian.apply(residual, ik), (nbase, 0))
        active = jax.lax.dynamic_update_slice(active, jnp.arange(nbnd) < notcnv, (nbase,))

        return psi, hpsi, evc, energies, active, nbase + notcnv, iteration + 1, converged

    _, _, evc, energies, *_ = jax.lax.while_loop(unconverged, step, state)
    return energies, jnp.where(mask, evc, 0.0)


def _starting_vectors(psi0, nbnd, npwx, kinetic, mask, dtype):
    """The trial vectors: the caller's, or QE's random guess.

    ``wfcinit``'s ``starting_wfc = 'random'`` draws random coefficients damped by
    ``1/(1 + |k+G|^2)``, so that the guess is concentrated on the low-kinetic
    plane waves where the occupied states live. The damping is what matters; the
    particular random numbers are not, so a fixed key is used and the result is
    reproducible.
    """
    if psi0 is not None:
        return jnp.where(mask, psi0.astype(dtype), 0.0)

    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    real = jax.random.uniform(keys[0], (nbnd, npwx)) - 0.5
    imaginary = jax.random.uniform(keys[1], (nbnd, npwx)) - 0.5
    guess = (real + 1j * imaginary).astype(dtype) / (1.0 + kinetic)
    return jnp.where(mask, guess, 0.0)


@partial(jax.jit, static_argnames=("nbnd", "david", "max_iterations"))
def davidson_eigensolver_all(
    hamiltonian: Hamiltonian,
    nbnd: int,
    psi0=None,
    ethr=None,
    residual_threshold=RESIDUAL_THRESHOLD,
    david: int = DAVID_NDIM,
    max_iterations: int = MAX_ITERATIONS,
):
    """Every k-point at once: the same solver under ``vmap`` over ``ik``."""

    def solve(ik, start):
        return davidson_eigensolver(
            hamiltonian, ik, nbnd, start, ethr=ethr,
            residual_threshold=residual_threshold, david=david,
            max_iterations=max_iterations,
        )

    indices = jnp.arange(hamiltonian.nk)
    if psi0 is None:
        return jax.vmap(lambda ik: solve(ik, None))(indices)
    return jax.vmap(solve)(indices, psi0)
