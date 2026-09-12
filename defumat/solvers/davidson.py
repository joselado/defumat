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

**``ethr`` is a per-band vector, not a scalar, and that is ``cegterg`` too.**
``cegterg.f90:556-563`` tests band ``i`` against ``ethr`` when ``btype(i) == 1``
and against ``empty_ethr = MAX(5 ethr, 1e-5)`` (``:129``) otherwise, where
``btype`` is 0 for a band whose fractional occupation is below 0.01
(``sum_band.f90:118-128``). An empty band is converged more loosely because
nothing reads it: it carries no charge, so the density, the total energy and
every derivative of them are blind to it, and the states it holds up are the
occupied ones the SCF is actually solving for. The threshold is data of shape
``(nbnd,)`` per k-point -- a traced argument, so it changes no shape and forces
no recompilation -- and :func:`~defumat.scf.driver.band_thresholds` is where the
occupations become one. Two properties of the Fortran that the transcription
keeps: the flag is **not sticky** (recomputed from scratch every step, so a band
that met the test once may fail it later, unlike ``crmmdiagg.f90:1093``'s
latched ``conv = conv .OR. ...``), and a loosened band is **not locked out of
the loop** -- ``notcnv`` counts every band and the exit is still that all of
them pass, so the only thing ``btype`` changes is when a given root stops being
expanded. What it buys is that plus a narrower expansion block, which is the
1.47x ``pw.x`` itself shows when ``diago_full_acc`` turns it off.

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

**The working set is one contiguous XLA buffer and it is what decides whether a
large run starts at all**, so it is written down here rather than discovered.
Fitted to ``memory_analysis().temp_size_in_bytes`` of the compiled solve, in
bytes of one complex number ``zc``::

    2.18 nvecx npwx zc  +  4.20 nbnd npwx zc  +  2.00 band_batch npol N_smooth zc

The first term is ``psi`` and ``hpsi``, the two arrays this carries; the second
is the ``(nbnd, npwx)`` blocks -- ``evc``, ``hevc`` and the expansion chain; and
the third is ``h_psi``'s FFT boxes, two per band in flight and ``npol`` fields
per band, which is the term nothing in this file controls. On a 157-atom slab at ``ecutwfc = 60`` with
``nbnd = 1020`` and ``diago_david_ndim = 4`` that is 46 + 24 + 24 GiB, and an
H200 with the default 75 per cent preallocation refuses it.

Two of those band blocks were **carried and did not need to be**, and removing
them is worth 11.6 GiB there:

* the expansion block is a pure function of ``evc``, ``hevc``, ``sbec`` and the
  eigenvalues, all of which the loop already carries, so it is rebuilt at the
  top of the step that consumes it rather than held across the loop boundary
  (:func:`~defumat.solvers.davidson.davidson_eigensolver`'s ``expansion``);
* and the robustness retry was a ``lax.cond`` over a *static* flag, which put
  two copies of the whole ``while_loop`` in one executable -- XLA shares the
  ``(nvecx, npwx)`` carries between them and gives the ``(nbnd, npwx)`` one a
  slot each. The retry is a host branch now
  (:func:`davidson_eigensolver_all`).

Neither changes an eigenvalue: seven of eight reference cells came back
bit-for-bit and the eighth moved by 1.1e-15 Ry, from an unrelated rewrite of
``force_real_g0`` on the same pass. **The measurement that shows it is the
ratio ``nbnd npwx / (band_batch N_smooth)``**, which is 0.5 on that slab: at a
small cell's 0.06 the same change reads as 1 per cent, and only at a
production-like ratio does it read as the 10 per cent it is (``si64`` at
``band_batch = 16``: 932.6 MB to 837.1).
"""

from __future__ import annotations

from functools import partial

import warnings

import jax
import jax.numpy as jnp
import numpy as np

from defumat.basis.fft import force_real_g0, gamma_inner
from defumat.batching import map_k, resolve_k_batch
from defumat.hamiltonian.operator import Hamiltonian
from defumat.solvers.subspace import generalised_eigh

__all__ = ["davidson_eigensolver", "davidson_eigensolver_all", "DAVID_NDIM",
           "MAX_ITERATIONS", "ETHR", "ETHR_MIN", "EMPTY_ETHR_FLOOR",
           "RESIDUAL_THRESHOLD", "empty_band_threshold", "starting_vectors"]

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
#: (:mod:`defumat.batching`); with that in, three and four are within the
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

#: The floor under the threshold an *empty* band is converged to
#: (``cegterg.f90:129``: ``empty_ethr = MAX( ( ethr * 5.D0 ), 1.D-5 )``).
#:
#: **The floor is the half that matters, and writing ``5 ethr`` alone misses
#: most of the effect.** Five times a loose ``ethr`` is still loose, so early in
#: an SCF the two thresholds barely differ; it is late, once ``ethr`` has fallen
#: below 2e-6, that the constant takes over and the empty bands stop being
#: converged at all. At QE's ``ethr`` floor of 1e-13 the two differ by eight
#: orders of magnitude.
EMPTY_ETHR_FLOOR = 1.0e-5


def empty_band_threshold(ethr):
    """``cegterg.f90:129``'s ``empty_ethr``, for one scalar ``ethr`` in Ry."""
    return max(5.0 * float(ethr), EMPTY_ETHR_FLOOR)


def _extend_projection(hc, sc, psi, hpsi, becp, becq, offset, block,
                       gamma_only: bool = False):
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

    **The new rows are sliced back out of ``psi`` rather than passed in**, and
    that is the measured choice rather than the obvious one. The caller has
    just written them there and still holds them, so handing them over looks
    like one ``(nbnd, npwx)`` block saved -- and it costs one instead, on both
    backends and at every band batch tried. A ``dynamic_slice`` of a buffer the
    consumer only reads is free to XLA; passing the block keeps the caller's
    copy live *across* the two matrix products below, where otherwise it dies
    into the ``dynamic_update_slice`` that wrote it.
    """
    rows = jax.lax.dynamic_slice(psi, (offset, 0), (block, psi.shape[1]))
    if gamma_only:
        # ``regterg``'s ``MYDGER(..., -1.D0, psi, ..., hr, ...)``: the stored
        # half is doubled and ``G = 0`` -- its own conjugate partner rather than
        # half of a pair -- is then counted once. Both matrices are **real**
        # here, which is the point of the branch: a complex overlap is round-off
        # that ``generalised_eigh`` turns into an arbitrary phase per
        # eigenvector, hence a complex ``c(0)``, hence a state that is no longer
        # real. `regterg` never sees one because it works in real arithmetic.
        row_h = 2.0 * (rows.conj() @ hpsi.T).real - (
            rows[:, :1].conj() * hpsi[:, :1].T
        ).real
        row_s = 2.0 * (rows.conj() @ psi.T).real - (
            rows[:, :1].conj() * psi[:, :1].T
        ).real
        row_h = row_h.astype(psi.dtype)
        row_s = row_s.astype(psi.dtype)
    else:
        row_h = rows.conj() @ hpsi.T
        row_s = rows.conj() @ psi.T
    row_b = jax.lax.dynamic_slice(becp, (offset, 0), (block, becp.shape[1]))
    # ``becp^H becq`` is a sum over *projector* channels, not over plane waves,
    # so it carries no gamma factor -- the factor is already inside ``becp``,
    # which ``calbec_gamma`` returned.
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
    robust: bool = False,
    return_steps: bool = False,
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
            A scalar applies to every band; an ``(nbnd,)`` array is QE's
            per-band threshold, ``ethr`` for an occupied band and
            ``empty_ethr`` for an empty one (``cegterg.f90:556-563``, and
            :func:`~defumat.scf.driver.band_thresholds` for where the vector
            comes from). ``None`` uses :data:`ETHR`; the SCF driver passes its
            scheduled value, which starts loose and tightens as the density
            converges.
        return_steps: also return how many Davidson steps the solve took and
            how many bands were still unsettled when it stopped. Both are
            already computed inside the loop -- they are its trip counter and
            its ``notcnv`` -- so this only widens the return, and it is
            *static*, read at trace time, so the two-value form compiles to
            exactly what it did. The count is the number of passes of the
            step function, and the initial subspace solve happens outside the
            loop, so a solve that arrives already converged reports 0. Read the
            pair together: a count at ``max_iterations`` with a small
            ``notcnv`` is a straggler, with a large one it is a stall, and a
            *short* count can also mean the eigenvalues stopped being finite
            (see ``unconverged``).
        robust: which route the subspace solve takes, *statically*. ``False`` is
            the Cholesky one, which is what every validated number here was
            produced with; ``True`` is canonical orthogonalisation, for an
            overlap that has gone indefinite. The choice is not made per step
            because this function is ``vmap``ped over k, where a ``cond`` runs
            both branches -- see :func:`davidson_eigensolver_all`, which makes
            it outside the batch.

    Returns ``(eigenvalues, eigenvectors)`` with eigenvalues ascending in Ry and
    eigenvectors ``(nbnd, npwx)`` -- bands first, as the rest of the code
    carries wavefunctions -- and, with ``return_steps``, the step count and the
    number of unsettled bands after them.
    """
    ethr = ETHR if ethr is None else ethr
    gamma_only = hamiltonian.gamma_only
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
    # ``regterg.f90:174``: ``psi(1,k) = CMPLX(DBLE(psi(1,k)), 0)`` for every
    # vector that enters the subspace. A random start has an imaginary part at
    # ``G = 0`` and it makes the rebuilt field complex.
    start = force_real_g0(start, gamma_only)

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
    hc0, sc0 = _extend_projection(empty, empty, psi, hpsi, becp, becq, 0, nbnd,
                                  gamma_only)

    def solve(psi, hpsi, becq, active, hc_raw, sc_raw, previous):
        """Diagonalise in the current subspace and measure what is left."""
        pair = active[:, None] & active[None, :]
        inactive = jnp.where(active, 0.0, 1.0).astype(dtype)
        hc = jnp.where(pair, hc_raw, 0.0) + jnp.diag(shift * inactive)
        sc = jnp.where(pair, sc_raw, 0.0) + jnp.diag(inactive)

        if gamma_only:
            # Real symmetric, as ``regterg``'s ``hr``/``sr`` are. Taking the
            # real part is not a truncation: the imaginary part is round-off in
            # a quantity that is real by construction, and leaving it in is what
            # gives each eigenvector an arbitrary phase.
            hc, sc = hc.real, sc.real
        values, vectors = generalised_eigh(0.5 * (hc + hc.conj().T),
                                           0.5 * (sc + sc.conj().T),
                                           robust=robust)
        vectors = vectors.astype(psi.dtype)
        energies = values[:nbnd].real
        coefficients = vectors[:, :nbnd]

        # ... the estimate in the plane-wave basis, and H applied to it, both
        # rotations of vectors already computed -- no extra application of H.
        evc = coefficients.T @ psi
        hevc = coefficients.T @ hpsi
        # ``q <beta|evc>``, the only thing S needs beyond ``evc`` itself: the
        # Ritz vector's projections are the same rotation of the stored ones,
        # so ``S|psi>`` is never formed. ``(nbnd, nkb)``, and zero-width
        # without an augmentation charge.
        sbec = coefficients.T @ becq

        settled = jnp.abs(energies - previous) < ethr
        if residual_threshold is not None:
            # The only consumer of the residual inside this function, and it is
            # off by default (:data:`RESIDUAL_THRESHOLD`). :func:`expansion`
            # builds its own from the same inputs.
            sevc = (evc + hamiltonian.s_correction(sbec, ik)
                    if hamiltonian.has_overlap else evc)
            residual = hevc - energies[:, None].astype(dtype) * sevc
            settled = jnp.logical_and(
                settled,
                jnp.sqrt(jnp.sum(jnp.abs(residual) ** 2, axis=1)) < residual_threshold,
            )

        return (energies, evc, hevc, sbec, settled,
                jnp.sum(jnp.logical_not(settled)), jnp.all(settled))

    def expansion(evc, hevc, sbec, energies, settled):
        """The block the subspace grows by: preconditioned, normalised, sorted.

        The unconverged roots are sorted to the front so that the block written
        next starts with exactly the vectors worth keeping. The sort is stable,
        so roots keep their relative order.

        **Split out of :func:`solve` and evaluated at the top of the step that
        consumes it, rather than at the bottom of the one that could have
        produced it.** It is a pure function of things the loop already carries,
        and computing it here means it is *not* a loop carry -- which is one
        ``(nbnd, npwx)`` block, 5.8 GiB on a 157-atom slab, live across the
        whole of the next step for the sake of an elementwise chain that costs
        nothing to rebuild. ``cegterg`` carries ``evc`` and ``hevc`` and no
        such block either. The arithmetic is unchanged, so the vectors are
        bit-for-bit what the carried version produced.

        ``sbec`` is ``<beta|evc>`` already multiplied by ``q``, carried as the
        ``(nbnd, nkb)`` array it is rather than recomputed: recomputing it from
        ``project(evc)`` would be the same quantity by a different summation and
        would move the answer in the last bits.
        """
        if hamiltonian.has_overlap:
            sevc = evc + hamiltonian.s_correction(sbec, ik)
        else:
            sevc = evc
        residual = hevc - energies[:, None].astype(dtype) * sevc
        correction = _precondition(residual, diagonal, s_diagonal, energies)
        correction = jnp.where(mask, correction, 0.0)
        correction = force_real_g0(correction, gamma_only)
        # ``regterg.f90:361``: ``ew(n) = ew(n) - DBLE(psi(1,n) psi(1,n))`` --
        # the same doubling and the same ``G = 0`` correction as every other
        # plane-wave sum here.
        norm = jnp.sqrt(gamma_inner(correction, correction, gamma_only,
                                    keepdims=True).real
                        if gamma_only else
                        jnp.sum(jnp.abs(correction) ** 2, axis=1, keepdims=True))
        # **The sort comes before the normalisation, and only the buffer cares.**
        # Each row is still divided by its own norm and zeroed by its own flag,
        # so the value is what it was; but a gather whose consumer is elementwise
        # fuses into that consumer, where a gather *after* the division is a
        # second ``(nbnd, npwx)`` block -- one before it and one after. The
        # permutation is applied to the two ``(nbnd,)`` vectors instead, which
        # is free.
        order = jnp.argsort(settled)
        norm = norm[order]
        return jnp.where(
            settled[order][:, None], 0.0,
            correction[order] / jnp.where(norm > 0.0, norm, 1.0),
        )

    energies0, evc0, hevc0, sbec0, settled0, notcnv0, converged0 = solve(
        psi, hpsi, becq, first, hc0, sc0,
        jnp.full((nbnd,), jnp.inf, dtype=diagonal.dtype),
    )

    state = (
        psi, hpsi, becp, becq, first, hc0, sc0,  # the subspace and its projections
        nbnd,                              # where the next block is written
        evc0, hevc0, energies0,            # current estimate, and H applied to it
        sbec0, settled0, notcnv0,          # what the expansion is built from
        0, converged0,                     # iteration, converged
    )

    def unconverged(state):
        # ... and stop the moment the eigenvalues stop being finite. Without
        # this a subspace solve that has returned ``NaN`` runs the full budget
        # of steps before the retry outside can see it, since ``settled`` --
        # a comparison against ``NaN`` -- is false for every root forever.
        # It costs one reduction over ``nbnd`` per step and it is what makes
        # the retry in :func:`davidson_eigensolver_all` cheap enough to be the
        # guard rather than a ``cond`` inside the batch.
        alive = jnp.all(jnp.isfinite(state[10]))
        return jnp.logical_and(
            jnp.logical_and(jnp.logical_not(state[15]), state[14] < max_iterations),
            alive,
        )

    def step(state):
        (psi, hpsi, becp, becq, active, hc_raw, sc_raw, nbase,
         evc, hevc, energies, sbec, settled, notcnv, iteration, _) = state

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
        # ... the block to expand with, rebuilt here rather than carried across
        # the loop boundary, and built *after* the collapse rather than before
        # it: the collapse leaves ``evc``, ``hevc``, ``sbec`` and ``energies``
        # untouched, so the value is the same either way, and putting it here
        # keeps the block out of the ``cond``'s live range -- which is where the
        # FFT boxes of the next ``h_psi`` are. See :func:`expansion`.
        correction = expansion(evc, hevc, sbec, energies, settled)
        new_becp, new_becq = project(correction)
        psi = jax.lax.dynamic_update_slice(psi, correction, (nbase, 0))
        hpsi = jax.lax.dynamic_update_slice(hpsi, hamiltonian.apply(correction, ik), (nbase, 0))
        becp = jax.lax.dynamic_update_slice(becp, new_becp, (nbase, 0))
        becq = jax.lax.dynamic_update_slice(becq, new_becq, (nbase, 0))
        active = jax.lax.dynamic_update_slice(active, jnp.arange(nbnd) < notcnv, (nbase,))
        hc_raw, sc_raw = _extend_projection(hc_raw, sc_raw, psi, hpsi, becp, becq,
                                            nbase, nbnd, gamma_only)
        nbase = nbase + notcnv

        energies, evc, hevc, sbec, settled, notcnv, converged = solve(
            psi, hpsi, becq, active, hc_raw, sc_raw, energies
        )
        return (psi, hpsi, becp, becq, active, hc_raw, sc_raw, nbase,
                evc, hevc, energies, sbec, settled, notcnv, iteration + 1,
                converged)

    final = jax.lax.while_loop(unconverged, step, state)
    evc, energies = final[8], final[10]
    if return_steps:
        # Both are loop carries already: nothing is measured that was not
        # measured before, and nothing is read on the host inside the loop.
        return energies, jnp.where(mask, evc, 0.0), final[14], final[13]
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


@partial(jax.jit, static_argnames=("nbnd", "david", "max_iterations", "k_batch",
                                   "robust", "return_steps"))
def _every_k(
    hamiltonian: Hamiltonian,
    nbnd: int,
    psi0,
    ethr,
    residual_threshold,
    david: int,
    max_iterations: int,
    k_batch: int | None | str,
    robust: bool,
    *,
    return_steps: bool = False,
):
    """One compiled solve of the whole k-set, by one of the two routes.

    ``return_steps`` is keyword-only and last on purpose: ``tools/gpu``'s memory
    tool lowers this unit by position, so a new positional parameter would break
    it with no test to notice.
    """
    def solve(ik, start):
        # The threshold rides the traced ``ethr`` slot as an ``(nk, nbnd)``
        # array and is gathered here with the same ``ik`` the solver already
        # uses for ``state_mask[ik]``. It is *not* a leaf of ``map_k``'s
        # pytree: keeping it closed over means the chunked, the scanned and the
        # ``vmap``ped branch all read the same rows, so the chunk size still
        # cannot change the answer. ``jnp.ndim`` is static on a tracer, so the
        # branch below is taken at trace time.
        row = ethr if jnp.ndim(ethr) < 2 else ethr[ik]
        return davidson_eigensolver(
            hamiltonian, ik, nbnd, start, ethr=row,
            residual_threshold=residual_threshold, david=david,
            max_iterations=max_iterations, robust=robust,
            return_steps=return_steps,
        )

    batch = resolve_k_batch(k_batch)
    indices = jnp.arange(hamiltonian.nk)
    if psi0 is None:
        return map_k(lambda ik: solve(ik, None), indices, batch=batch)
    return map_k(lambda pair: solve(*pair), (indices, psi0), batch=batch)


def davidson_eigensolver_all(
    hamiltonian: Hamiltonian,
    nbnd: int,
    psi0=None,
    ethr=None,
    residual_threshold=RESIDUAL_THRESHOLD,
    david: int = DAVID_NDIM,
    max_iterations: int = MAX_ITERATIONS,
    k_batch: int | None | str = "default",
    robust_retry: bool = True,
    return_steps: bool = False,
):
    """Every k-point, ``k_batch`` of them at a time.

    This is where the k-axis working set is largest: each k-point in flight
    holds ``david * nbnd`` subspace vectors of length ``npol * npwx``, three of
    them (``psi``, ``hpsi``, ``spsi``), plus whatever ``h_psi`` needs to
    transform a block of bands. Multiplying that by ``nk`` is what a ``vmap``
    over the whole axis does, and it is why the default here is QE's loop --
    ``c_bands.f90`` calls ``diag_bands`` on one ``ik`` at a time. See
    :mod:`defumat.batching`.

    The chunking cannot change the answer: the k-points are independent here,
    and each is solved by exactly the same function either way.

    **This is also where the indefinite-overlap guard lives, and it is here
    rather than inside the solve for a reason that is entirely about batching.**
    ``generalised_eigh``'s ``lax.cond`` between the Cholesky route and canonical
    orthogonalisation is a real branch only where its predicate is a scalar; one
    level down, inside a ``vmap`` over k, the predicate is batched and JAX
    lowers the ``cond`` to a ``select_n`` over **both** branches. That is 2.85x
    of the subspace solve on ``si10-nc``'s shapes, paid on every step of every
    multi-k run in the mode an accelerator defaults to.

    So the batched solve takes the Cholesky route with no ``cond`` in it at all,
    and the guard is applied *here*, where ``jnp.all(jnp.isfinite(...))`` over
    the whole k-set -- of the eigenvalues **and** of the wavefunctions -- is one
    scalar and ``lax.cond`` is a branch again. A clean
    solve pays one reduction; a solve that has gone non-finite -- the 64-atom
    ``NaN`` -- is repeated for every k-point with canonical orthogonalisation,
    which is strictly more work than the old per-step ``cond`` in exactly the
    case that was already broken. The per-step loop exits as soon as the
    eigenvalues stop being finite (see ``unconverged``), so the wasted half of
    that retry is a few steps rather than the whole budget.

    The values returned are bit-for-bit what the guarded version returned
    whenever the guard passed, because the guard passing *is* taking the
    Cholesky route's own answer.

    **The guard is a Python branch and not a ``lax.cond``, and that is a memory
    decision.** ``robust`` is static -- it picks between two different subspace
    routines -- so a ``cond`` over it puts *two* copies of the whole Davidson
    ``while_loop`` in one executable. XLA shares the big ``(nvecx, npwx)``
    carries between them, but not the ``(nbnd, npwx)`` correction: the buffer
    assignment gives ``while.4`` and ``while.5`` a slot each, which is one band
    block of pure duplication -- 5.8 GiB on a 157-atom slab -- and twice the
    HLO for the rematerialisation pass to work on, which is where a large run
    actually fails. Branching on the host costs one scalar transfer per call,
    at a point where :meth:`~defumat.scf.driver.Calculation.diagonalize`'s
    caller synchronises on the Fermi level a few lines later anyway.

    ``robust_retry = False`` keeps the whole thing inside one ``jit`` for a
    caller that has to trace through it; ``clear_cache`` reaches the compiled
    unit, so a test that monkeypatches the subspace route still works.

    ``ethr`` may be a scalar, an ``(nbnd,)`` vector applied at every k-point, or
    an ``(nk, nbnd)`` one. **It is broadcast to ``(nk, nbnd)`` here, on the
    host, and the reason is compilation rather than convenience**: the value is
    traced, so no value of it ever recompiles, but a ``()`` aval and an
    ``(nk, nbnd)`` aval are two signatures -- and an SCF whose first iteration
    has no occupations yet, hence a scalar, and whose second has a vector would
    compile the whole Davidson stack twice.

    ``return_steps`` adds the per-k step count and unsettled-band count to the
    return, ``(nk,)`` each. It is off by default so that every existing caller
    still unpacks two values.
    """
    ethr = jnp.broadcast_to(
        jnp.asarray(ETHR if ethr is None else ethr,
                    dtype=hamiltonian.kinetic.dtype),
        (hamiltonian.nk, nbnd),
    )
    arguments = (hamiltonian, nbnd, psi0, ethr, residual_threshold, david,
                 max_iterations, k_batch)
    fast = _every_k(*arguments, robust=False, return_steps=return_steps)
    if not robust_retry:
        return fast
    # Both halves, not just the eigenvalues. A Cholesky factor that has gone
    # non-finite does not necessarily poison every root -- the first regression
    # test written for the 64-atom NaN passed on the *unfixed* code precisely
    # because the failure sat in a triangle nothing read -- so a guard that
    # watches only the energies can pass a wavefunction with a NaN in it
    # straight into the density. Two reductions against a solve is not a cost.
    #
    # **Per k-point, not over the whole set.** This reduction used to be one
    # scalar ``jnp.isfinite(...).all()``, and the retry it gated replaced
    # *every* k-point's result with the robust route's. That is wrong twice
    # over on a dense mesh, where one bad k-point in ten is ordinary rather
    # than exceptional: the converged Cholesky answers at the other nine were
    # discarded and recomputed from a fresh random start, and if the robust
    # pass then hit its iteration budget -- which nothing downstream checks --
    # the caller received *less* converged wavefunctions than the ones thrown
    # away. Measured on a 1H-NbSe2 mesh at ``ethr = 4e-9``: k-points 0-4 all
    # finite, k-point 9 non-finite, and the whole-set retry turned 509 s into
    # 1368 s while replacing four good solves.
    per_k = (jnp.isfinite(fast[0]).all(axis=1)
             & jnp.isfinite(fast[1]).reshape(fast[1].shape[0], -1).all(axis=1))
    failed = ~np.asarray(per_k)
    if not failed.any():
        return fast
    warnings.warn(
        f"{int(failed.sum())} of {failed.size} k-points came back non-finite "
        f"from the Cholesky route ({np.flatnonzero(failed).tolist()[:8]}"
        f"{' ...' if failed.sum() > 8 else ''}) and are being re-solved with "
        "canonical orthogonalisation. A non-finite overlap here is usually a "
        "solve that stalled rather than a bad Hamiltonian -- check the step "
        "counts, and loosen ethr (conv_thr) before trusting the result",
        stacklevel=2,
    )
    robust = _every_k(*arguments, robust=True, return_steps=return_steps)
    # Keep what the fast route already converged. The robust pass still runs
    # over the whole k-set -- the shapes are static, so it must -- but its
    # answer is taken only where the fast one has none.
    take = jnp.asarray(failed)
    return tuple(
        jnp.where(take.reshape((-1,) + (1,) * (jnp.ndim(quick) - 1)), sturdy, quick)
        for quick, sturdy in zip(fast, robust)
    )


davidson_eigensolver_all.clear_cache = _every_k.clear_cache
