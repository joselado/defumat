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

**The problem is generalised**, ``H v = e S v``, because an ultrasoft
pseudopotential makes ``S`` a genuine operator. ``cegterg`` tracks ``S|psi>``
alongside ``H|psi>`` in a second ``(nvecx, npw)`` array; this does not, and the
reason is worth stating. ``S`` differs from the identity only inside the
projector subspace,

    S|psi> = |psi> + sum_kl |beta_k> q_kl <beta_l|psi>

so everything the algorithm needs from it is a function of the small
``(nvecx, nkb)`` array of projections ``<beta|psi>`` -- the projected overlap is
``psi^H psi + becp^H q becp``, and ``S`` applied to a Ritz vector is one
``(nkb, npw)`` product away. Carrying ``becp`` instead of ``S|psi>`` is the same
arithmetic in a fraction of the memory, and it makes the norm-conserving path
free rather than merely cheap: with no augmentation charge the tracked array has
zero columns, and every expression involving it disappears at compile time.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from pypresso.batching import map_k, resolve_k_batch
from pypresso.hamiltonian.operator import Hamiltonian
from pypresso.solvers.subspace import generalised_eigh

__all__ = ["davidson_eigensolver", "davidson_eigensolver_all", "DAVID_NDIM",
           "MAX_ITERATIONS", "ETHR", "ETHR_MIN", "RESIDUAL_THRESHOLD",
           "starting_vectors"]

#: QE's ``diago_david_ndim``: the subspace may grow to this many times ``nbnd``
#: before it is collapsed back onto the current eigenvector estimates.
#:
#: **Four, as QE has it.** Three was tried, and the episode is worth recording
#: because both halves of it were surprises. ``INPUT_PW.txt`` says to use four
#: "if the time spent in subspace diagonalization is small compared to the time
#: spent in ``h_psi``", and at the time it was not -- that algebra is
#: ``O(nvecx nbnd npw)`` matrix products sized by ``nvecx``, since the shapes
#: must be static, where ``cegterg`` sizes its ZGEMMs by the *live* basis. Three
#: was then worth 12% of a whole SCF on the eight-atom cell and 7% on the
#: sixteen-atom one.
#:
#: It is four anyway, for two independent reasons.
#:
#: It **changed a validated number**. A band-structure run has no SCF around it:
#: the SCF re-seeds this solver from the previous iteration and re-runs it ten
#: times on a tightening ``ethr``, so a root left slightly short is corrected on
#: the next pass, while ``non_scf`` gets one attempt. Combined with a
#: convergence test that watches the *change* in an eigenvalue rather than its
#: residual -- QE's test, and a weak proxy (see :data:`RESIDUAL_THRESHOLD`) -- a
#: smaller workspace collapses more often, consecutive estimates sit closer
#: together, and the test fires while the error is larger. That showed up
#: exactly where it should: on the bismuthene spin-orbit path, Kramers pairs are
#: degenerate by symmetry and so measure nothing but solver error, and they went
#: from below 1e-6 eV to 5.9e-6. Every SCF regression passed; only the one-shot
#: solve moved.
#:
#: And by the time that was understood the **speed was gone too**. The 12% was
#: measured before ``h_psi`` began walking its bands one at a time
#: (:mod:`pypresso.batching`); with that in, three and four are within the
#: run-to-run spread of each other on both cells. The saving had been in the
#: cache, not in the flop count, and the band loop had already collected it.
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


def _extend_projection(hc, sc, psi, hpsi, becp, becq, offset, block):
    """Add one block of rows and columns to the projected H and overlap.

    The projected matrices grow by a block of vectors per Davidson step, so all
    but the newest rows are unchanged from the step before. ``cegterg`` computes
    only the new ones -- its ZGEMM writes into ``hc(nb1, n_start)`` -- and keeps
    the rest, and so does this.

    It is not a small saving. Recomputing costs ``O(nvecx^2 npw)`` per step
    against ``O(nvecx nbnd npw)`` for the update, a factor of ``nvecx/nbnd``,
    four at the default subspace size. On an eight-atom silicon cell at 2950
    plane waves that is 7.7 ms per step against 2.2; on a two-atom cell it is
    half a millisecond either way, which is why it took a bigger system to
    notice.

    The Hermitian counterpart of each new row is written at the same time, so
    the stored matrices stay full rather than triangular.

    ``becp``/``becq`` carry the augmentation part of the overlap: ``becq`` is
    ``q <beta|psi>``, so ``becp^H becq`` is the ``<psi|S - 1|psi>`` block. They
    have zero columns when there is no augmentation charge.
    """
    rows = jax.lax.dynamic_slice(psi, (offset, 0), (block, psi.shape[1]))
    row_h = rows.conj() @ hpsi.T
    row_s = rows.conj() @ psi.T
    row_b = jax.lax.dynamic_slice(becp, (offset, 0), (block, becp.shape[1]))
    row_s = row_s + row_b.conj() @ becq.T

    hc = jax.lax.dynamic_update_slice(hc, row_h, (offset, 0))
    sc = jax.lax.dynamic_update_slice(sc, row_s, (offset, 0))
    hc = jax.lax.dynamic_update_slice(hc, row_h.conj().T, (0, offset))
    sc = jax.lax.dynamic_update_slice(sc, row_s.conj().T, (0, offset))
    return hc, sc


def _precondition(residual, diagonal, overlap_diagonal, energies):
    """``g_psi.f90``: an approximate inverse of ``H - e S`` from its diagonal.

    The naive ``1/(H_ii - e S_ii)`` is unbounded where the shift meets the
    diagonal. QE's default branch replaces it with
    ``(1 + x + sqrt(1 + (x-1)^2)) / 2``, which agrees with ``x`` for large ``x``
    and saturates at 1 near the pole -- so a plane wave nearly resonant with the
    eigenvalue is damped rather than amplified.

    ``overlap_diagonal`` is ``usnldiag``'s ``s_diag``, identically one without an
    augmentation charge.
    """
    x = diagonal[None, :] - energies[:, None] * overlap_diagonal[None, :]
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
    eigenvectors ``(nbnd, npwx)`` -- bands first, as the rest of the code
    carries wavefunctions.
    """
    ethr = ETHR if ethr is None else ethr
    ndim = hamiltonian.ndim
    nvecx = david * nbnd
    mask = hamiltonian.state_mask[ik]
    kinetic = hamiltonian.state_kinetic[ik]
    diagonal = hamiltonian.diagonal(ik)
    s_diagonal = hamiltonian.overlap_diagonal(ik)
    dtype = hamiltonian.dtype

    # The projections S is built from, asked of the operator rather than
    # assembled here. With no augmentation charge they are zero-width arrays and
    # every expression below that touches them is a no-op -- which is how the
    # norm-conserving path stays exactly what it was -- and with a spinor
    # Hamiltonian they carry the spin index folded into their width, so nothing
    # in this routine has to know how many components a state has.
    def project(vectors):
        """``<beta|psi>`` and ``q <beta|psi>`` for a block of vectors."""
        return hamiltonian.s_projections(vectors, ik)

    start = starting_vectors(psi0, nbnd, ndim, kinetic, mask, dtype)

    # Inactive subspace directions are given this eigenvalue, which has to sit
    # above anything physical: the diagonal bounds the spectrum from above well
    # enough for that.
    shift = jnp.max(jnp.abs(diagonal)) * 1000.0 + 1.0

    psi = jnp.zeros((nvecx, ndim), dtype).at[:nbnd].set(start)
    hpsi = jnp.zeros((nvecx, ndim), dtype).at[:nbnd].set(hamiltonian.apply(start, ik))
    becp0, becq0 = project(start)
    nkb = becp0.shape[1]
    becp = jnp.zeros((nvecx, nkb), dtype).at[:nbnd].set(becp0)
    becq = jnp.zeros((nvecx, nkb), dtype).at[:nbnd].set(becq0)
    first = jnp.arange(nvecx) < nbnd
    empty = jnp.zeros((nvecx, nvecx), dtype)
    hc0, sc0 = _extend_projection(empty, empty, psi, hpsi, becp, becq, 0, nbnd)

    def solve(psi, hpsi, becq, active, hc_raw, sc_raw, previous):
        """Diagonalise in the current subspace and measure what is left."""
        pair = active[:, None] & active[None, :]
        inactive = jnp.where(active, 0.0, 1.0).astype(dtype)
        hc = jnp.where(pair, hc_raw, 0.0) + jnp.diag(shift * inactive)
        sc = jnp.where(pair, sc_raw, 0.0) + jnp.diag(inactive)

        values, vectors = generalised_eigh(0.5 * (hc + hc.conj().T),
                                           0.5 * (sc + sc.conj().T))
        energies = values[:nbnd].real
        coefficients = vectors[:, :nbnd]

        # ... the estimate in the plane-wave basis, and H applied to it, both
        # rotations of vectors already computed -- no extra application of H.
        evc = coefficients.T @ psi
        hevc = coefficients.T @ hpsi
        # S|evc> without ever storing S|psi>: the Ritz vector's projections are
        # the same rotation of the stored ones.
        sevc = evc + hamiltonian.s_correction(coefficients.T @ becq, ik)

        residual = hevc - energies[:, None].astype(dtype) * sevc
        settled = jnp.abs(energies - previous) < ethr
        if residual_threshold is not None:
            settled = jnp.logical_and(
                settled,
                jnp.sqrt(jnp.sum(jnp.abs(residual) ** 2, axis=1)) < residual_threshold,
            )

        # ... the preconditioned, normalised correction, with the unconverged
        # roots sorted to the front so the block written next starts with
        # exactly the vectors worth keeping. The sort is stable, so roots keep
        # their relative order.
        correction = _precondition(residual, diagonal, s_diagonal, energies)
        correction = jnp.where(mask, correction, 0.0)
        norm = jnp.sqrt(jnp.sum(jnp.abs(correction) ** 2, axis=1, keepdims=True))
        correction = jnp.where(settled[:, None], 0.0,
                               correction / jnp.where(norm > 0.0, norm, 1.0))
        correction = correction[jnp.argsort(settled)]

        return (energies, evc, hevc, correction,
                jnp.sum(jnp.logical_not(settled)), jnp.all(settled))

    energies0, evc0, hevc0, correction0, notcnv0, converged0 = solve(
        psi, hpsi, becq, first, hc0, sc0,
        jnp.full((nbnd,), jnp.inf, dtype=diagonal.dtype),
    )

    state = (
        psi, hpsi, becp, becq, first, hc0, sc0,  # the subspace and its projections
        nbnd,                              # where the next block is written
        evc0, hevc0, energies0,            # current estimate, and H applied to it
        correction0, notcnv0,              # what to expand with, decided already
        0, converged0,                     # iteration, converged
    )

    def unconverged(state):
        return jnp.logical_and(jnp.logical_not(state[14]), state[13] < max_iterations)

    def step(state):
        (psi, hpsi, becp, becq, active, hc_raw, sc_raw, nbase,
         evc, hevc, energies, correction, notcnv, iteration, _) = state

        # ... collapse onto the current estimates when the subspace is full,
        # which is the only place the basis ever shrinks (cegterg's "refresh").
        # The projected matrices come along for free: the Ritz vectors
        # diagonalise H within the span and are S-orthonormal by construction,
        # so the retained block is diag(energies) against the identity.
        full = nbase + nbnd > nvecx
        blank = jnp.zeros_like(hc_raw)
        evc_becp, evc_becq = project(evc)
        psi, hpsi, becp, becq, active, nbase, hc_raw, sc_raw = jax.lax.cond(
            full,
            lambda: (
                jnp.zeros_like(psi).at[:nbnd].set(evc),
                jnp.zeros_like(hpsi).at[:nbnd].set(hevc),
                jnp.zeros_like(becp).at[:nbnd].set(evc_becp),
                jnp.zeros_like(becq).at[:nbnd].set(evc_becq),
                first,
                nbnd,
                blank.at[:nbnd, :nbnd].set(jnp.diag(energies.astype(dtype))),
                blank.at[:nbnd, :nbnd].set(jnp.eye(nbnd, dtype=dtype)),
            ),
            lambda: (psi, hpsi, becp, becq, active, nbase, hc_raw, sc_raw),
        )

        # ... expand, and only then diagonalise and test. This ordering is
        # cegterg's, and it is not cosmetic: testing after expanding, as an
        # earlier version did, means every call ends by applying H to a block of
        # residuals that are all zero because every root has just converged.
        # That was one wasted h_psi per Davidson call -- 7 of the 23 steps a
        # whole eight-atom run takes.
        new_becp, new_becq = project(correction)
        psi = jax.lax.dynamic_update_slice(psi, correction, (nbase, 0))
        hpsi = jax.lax.dynamic_update_slice(hpsi, hamiltonian.apply(correction, ik), (nbase, 0))
        becp = jax.lax.dynamic_update_slice(becp, new_becp, (nbase, 0))
        becq = jax.lax.dynamic_update_slice(becq, new_becq, (nbase, 0))
        active = jax.lax.dynamic_update_slice(active, jnp.arange(nbnd) < notcnv, (nbase,))
        hc_raw, sc_raw = _extend_projection(hc_raw, sc_raw, psi, hpsi, becp, becq, nbase, nbnd)
        nbase = nbase + notcnv

        energies, evc, hevc, correction, notcnv, converged = solve(
            psi, hpsi, becq, active, hc_raw, sc_raw, energies
        )
        return (psi, hpsi, becp, becq, active, hc_raw, sc_raw, nbase,
                evc, hevc, energies, correction, notcnv, iteration + 1, converged)

    final = jax.lax.while_loop(unconverged, step, state)
    evc, energies = final[8], final[10]
    return energies, jnp.where(mask, evc, 0.0)


def starting_vectors(psi0, nbnd, ndim, kinetic, mask, dtype):
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
    real = jax.random.uniform(keys[0], (nbnd, ndim)) - 0.5
    imaginary = jax.random.uniform(keys[1], (nbnd, ndim)) - 0.5
    guess = (real + 1j * imaginary).astype(dtype) / (1.0 + kinetic)
    return jnp.where(mask, guess, 0.0)


@partial(jax.jit, static_argnames=("nbnd", "david", "max_iterations", "k_batch"))
def davidson_eigensolver_all(
    hamiltonian: Hamiltonian,
    nbnd: int,
    psi0=None,
    ethr=None,
    residual_threshold=RESIDUAL_THRESHOLD,
    david: int = DAVID_NDIM,
    max_iterations: int = MAX_ITERATIONS,
    k_batch: int | None | str = "default",
):
    """Every k-point, ``k_batch`` of them at a time.

    This is where the k-axis working set is largest: each k-point in flight
    holds ``david * nbnd`` subspace vectors of length ``npol * npwx``, three of
    them (``psi``, ``hpsi``, ``spsi``), plus whatever ``h_psi`` needs to
    transform a block of bands. Multiplying that by ``nk`` is what a ``vmap``
    over the whole axis does, and it is why the default here is QE's loop --
    ``c_bands.f90`` calls ``diag_bands`` on one ``ik`` at a time. See
    :mod:`pypresso.batching`.

    The chunking cannot change the answer: the k-points are independent here,
    and each is solved by exactly the same function either way.
    """

    def solve(ik, start):
        return davidson_eigensolver(
            hamiltonian, ik, nbnd, start, ethr=ethr,
            residual_threshold=residual_threshold, david=david,
            max_iterations=max_iterations,
        )

    batch = resolve_k_batch(k_batch)
    indices = jnp.arange(hamiltonian.nk)
    if psi0 is None:
        return map_k(lambda ik: solve(ik, None), indices, batch=batch)
    return map_k(lambda pair: solve(*pair), (indices, psi0), batch=batch)
