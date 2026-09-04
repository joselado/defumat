"""The Sternheimer equation: the first-order wavefunctions, without a sum over states.

The response of a Kohn-Sham state to a perturbation ``dV`` is, formally,

    |dpsi_n> = sum_{m != n} |psi_m> <psi_m|dV|psi_n> / (eps_n - eps_m),

which needs every empty state and divides by a gap that closes at every
degeneracy. The Sternheimer form asks the same question as a linear system,

    (H - eps_n S + alpha Q) |dpsi_n> = -P_c^+ dV |psi_n>,

and answers it with a projected conjugate-gradient solve per occupied band. It
needs no empty states at all, and it never divides by ``eps_n - eps_m`` -- which
is rule D4's requirement, not a convenience: a crystal is degenerate everywhere
by symmetry.

``LR_Modules/cgsolve_all.f90`` (the solver), ``ch_psi_all.f90`` (the operator),
``orthogonalize.f90`` (the projector), ``h_prec.f90`` (the preconditioner) and
``setup_alpha_pv.f90`` (the level shift) are transcribed here. The Fortran's
dynamic repacking of the unconverged bands becomes a **mask at a static shape**,
following the same rule ``cegterg`` followed into
:mod:`defumat.solvers.davidson`: a converged band stays in the block and takes a
zero step.

**What is not transcribed is everything the response is *of*.** The density is
already written down once as a differentiable function of the states, and the
nonlocal coefficients as one of the potential, so the quantities QE builds a
routine apiece for are derivatives of code that exists:

===============================================  ==========================================
QE                                               here
===============================================  ==========================================
``incdrhoscf``, ``addusdbec``, ``lr_addusddens``  one ``jvp`` of the density w.r.t. the states
``newdq``'s ``int3``, ``adddvscf``                one ``jvp`` of ``newd`` w.r.t. the potential
``PAW_dpotential``                                one ``jvp`` of ``onecenter`` w.r.t. ``becsum``
===============================================  ==========================================

which is why **ultrasoft and PAW work here without a second implementation**.
:meth:`SternheimerSolver.response_density` and :func:`local_perturbation` are
those three lines and the bookkeeping around them.

**What ``alpha Q`` is for, and why it is kept even though the right-hand side is
already projected.** ``Q = alpha_pv S P_occ S`` shifts the occupied manifold up
by ``alpha_pv`` so that ``H - eps_n S + alpha Q`` is positive definite on the
whole space rather than only on the conduction subspace. Without it the solution
is undetermined along the occupied directions and the numerical leakage into
them is amplified rather than damped. ``alpha_pv = 2 (eps_max^occ - eps_min)``
is ``setup_alpha_pv``'s insulator value.

**This is P22c and D3's backward pass, not only a step towards a dielectric
constant.** ``chi_0 = drho/dV`` is the exact independent-particle susceptibility,
and the SCF Jacobian is ``chi_0 K`` with ``K = dV_scf/drho`` already free from
``jax.grad`` of ``v_of_rho`` (rule D1). P22 measured the alternative -- forward
mode straight through Davidson's ``lax.while_loop`` -- at 109% wrong from a cold
start and 4-7x slower than a finite difference from a warm one. The solve here is
exact rather than accurate to ``ethr``, and on two-atom silicon at
``ecutwfc = 12`` it costs **0.5 s** against that route's **3.5 s**. That it should
also *scale* better -- a projected CG over the occupied bands against a Davidson
subspace of ``nvecx = 4 nbnd`` -- is an expectation rather than a measurement: it
has not been timed on a cell where the two would separate.

**Metals are in** (``PLAN.md`` P24c), and what they change is the projector
rather than the solve. For an insulator ``P_c^+`` is a projector: a band is
occupied or it is not. For a metal there is no such partition, and
``orthogonalize``'s smearing branch replaces the sharp step with a pair of
weights -- one on the right-hand side and one, ``wwg``, on each overlap, built
from the *difference* of two occupations
(:meth:`SternheimerSolver._smeared_projection`). Three consequences run through
this module:

* **every band is kept**, because ``nbnd_eff = nbnd`` there. QE truncates the
  solve at ``setup_nbnd_occ``'s count; here the block stays whole at a static
  shape and the bands past it carry an occupation of zero.
* **``alpha_pv`` is measured to where the smearing dies**, ``ef + xmax
  degauss``, since the occupied manifold has no top (``setup_alpha_pv``).
* **the density response is accumulated with ``wk``, not ``wg``.** The
  occupation is applied to ``dpsi`` itself, so weighting the density by it again
  would count it twice. The two coincide for an insulator, which is why nothing
  before this had to tell them apart -- and why getting it wrong would have been
  invisible on every case in the suite.

What a metal still needs from its *caller* is ``ef_shift``: a perturbation at
``q = 0`` changes the number of states below the Fermi level, so the level moves
and the response density has to be corrected by the local density of states at
it. That is :func:`fermi_level_shift`, and it belongs to the self-consistent
loop rather than to the solve.

**Refused rather than approximated**, each by name:

* **the tetrahedron occupations**, whose ``orthogonalize`` branch reads
  ``dfpt_tetra_beta`` -- a response weight per band *pair*, which the
  tetrahedron machinery here does not build. The smearing family is what is
  implemented.
* **noncollinear magnetism and spin-orbit coupling**, whose ``incdrhoscf_nc``
  and ``set_int3_nc`` are a second implementation rather than a spin axis on
  this one.
* **DFT+U**, whose induced potential has a ``dns`` of its own
  (``adddvhubscf.f90``) that is not a function of ``drho``.
* **a potential-only meta-GGA** (``tb09``, ``bj06``), which has no total energy
  and so nothing for the response to be the second derivative of. The refusal
  is :func:`defumat.forces.energy.reject_potential_only`, reused rather than
  restated.
* **a ``tot_magnetization`` with a smearing** -- two Fermi levels, where
  :class:`Smearing` carries one scalar ``ef``.
* **a ``nspin = 2`` filling that cuts a degenerate multiplet**
  (:data:`DEGENERATE_CUT_RY`).

**Collinear spin is *not* on that list any more.** The occupied-band count is
one number per channel (:func:`occupied_counts`), the sharp projector masks the
deficient channel's extra bands, and ``chi_0`` is validated against a central
difference of the density for a spin-polarized metal and a spin-polarized
insulator alike. What the *callers* do with it is a separate claim and is what
``spin_polarized`` in :func:`require_a_sternheimer_regime` gates.

**Memory.** The CG carries four band-blocks -- the gradient, its
preconditioning, the search direction and the previous one -- at
``(nk, nbnd_occ, ndim)`` complex each if the whole k axis is in flight, or one
k-point's worth at ``k_batch = 1``. That is the same working set the Davidson
subspace has at ``nvecx = 4 nbnd``, and it goes through the same dial
(:mod:`defumat.batching`).

**A metal pays ``nbnd`` where an insulator pays ``nbnd_occ``**, and that is a
deliberate trade rather than an oversight: QE truncates the solve at
``setup_nbnd_occ``'s per-k count, and a per-k count is a dynamic shape, which
rule R2 does not allow inside a compiled loop. So the block stays whole and the
bands past the cutoff carry an occupation of zero -- exact, and paying their
share of the CG for nothing. On the aluminium of ``al-metal.in`` that is 8 bands
against a per-k ``nbnd_occ`` of 1 to 3, so the factor is real; it is bounded by
``nbnd/nbnd_occ`` and it shrinks as a cell grows, because ``nbnd`` is chosen a
fixed margin above the occupied count rather than a multiple of it. The way down
if it ever matters is the same as ``cegterg``'s: a mask that compacts, not a
shape that changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import lax

import jax

from defumat.basis.fft import g_to_r
from defumat.basis.interpolate import to_dense, to_smooth
from defumat.batching import map_k
from defumat.scf.density import becsum as becsum_of, sum_band
from defumat.scf.occupations import smearing_order, w0gauss, wgauss
from defumat.forces.energy import reject_potential_only
from defumat.scf.potential import as_potential_components

__all__ = [
    "Smearing",
    "SternheimerSolver",
    "SternheimerResult",
    "local_perturbation",
    "make_sternheimer",
    "occupied_counts",
    "paw_response",
    "require_a_sternheimer_regime",
    "smearing_of",
]

#: ``cgsolve_all``'s own ceiling on the CG iterations.
MAX_ITERATIONS = 400

#: The default target for the preconditioned residual norm of one band. QE
#: schedules this against the self-consistency of the response (``dfpt_kernels``:
#: ``thresh = min(0.1 sqrt(dr2), 1e-2)``); a solver used on its own, as
#: ``chi_0``, wants it tight from the start.
THRESHOLD = 1.0e-11

#: The smallest gap, in Ry, between the last occupied and the first empty band
#: of a spin channel that the sliced (insulator) branch will accept for
#: ``nspin = 2``. Below it the occupied manifold *cuts a degenerate multiplet*
#: and there is no response to compute: which member of the multiplet the
#: eigensolver returned is arbitrary, so ``P_c`` is built for an arbitrary
#: subspace of a degenerate shell and the answer is a property of the
#: eigensolver rather than of the density. It is the same multivaluedness the
#: *residual* solver is diagnosed for (``GAPS.md`` section 3, the closed
#: ``occupations = 'fixed'`` + ``nspin = 2`` entry), one layer up.
#:
#: **And it fails silently, which is why this is a refusal and not a warning.**
#: Measured on the oxygen atom of ``o-atom-fixed-lsda`` (``neldw = 2``, cutting
#: the triply degenerate 2p shell, the two levels 1.4e-14 Ry apart): the CG
#: *converges* -- 42 iterations to a residual of 5e-12 against a 1e-11
#: threshold -- and ``chi_0`` then disagrees with a central difference of the
#: density by **100 per cent** (1.24, 1.02, 1.01 and 1.01 relative for probes
#: at Miller (1,0,0), (0,0,1), (1,1,0) and (1,1,1)). The difference re-selects
#: which member of the shell falls below the cut, because the perturbation
#: splits it at first order; the solve keeps the member it was handed. Nothing
#: in the solve can see that, which is exactly why it is refused here.
#:
#: Checked only for ``nspin = 2``, which is the branch this rule is new for --
#: an unpolarized insulator's fillings have been validated for many phases and
#: are left bit-for-bit alone.
DEGENERATE_CUT_RY = 1.0e-5


#: ``setup_nbnd_occ``/``setup_alpha_pv``'s cutoff on the smeared occupation: a
#: band is "occupied" for the response if ``eps < ef + xmax degauss``, where
#: ``xmax`` is where the smearing function has fallen to ``small``.
_SMALL = 6.9626525973374e-5


@dataclass(frozen=True)
class Smearing:
    """The ground state's smearing, which the metal branch of the projector needs.

    ``ef`` and ``degauss`` in Ry, ``ngauss`` QE's integer smearing order (the
    same one :func:`defumat.scf.occupations.wgauss` takes). Present on a
    :class:`SternheimerSolver` exactly when the run is a metal; ``None`` selects
    ``orthogonalize``'s insulator branch, where the projector is sharp.
    """

    ef: float
    degauss: float
    ngauss: int

    @property
    def cutoff(self) -> float:
        """``ef + xmax degauss`` -- ``setup_nbnd_occ``'s ``target``."""
        if self.ngauss == -99:
            factor = 1.0 / np.sqrt(_SMALL)
            xmax = 2.0 * np.log(0.5 * (factor + np.sqrt(factor * factor - 4.0)))
        else:
            xmax = np.sqrt(-np.log(np.sqrt(np.pi) * _SMALL))
        return self.ef + xmax * self.degauss


@dataclass
class SternheimerResult:
    """The first-order wavefunctions and what the solve cost."""

    #: ``(nspin, nk, nbnd_occ, ndim)`` complex -- the ``dpsi`` of ``dfpt``.
    dpsi: jnp.ndarray
    #: How many CG iterations the *worst* band needed, per spin channel.
    iterations: int
    #: The largest preconditioned residual norm left over any band.
    residual: float

    @property
    def converged(self) -> bool:
        return self.iterations < MAX_ITERATIONS


class SternheimerSolver:
    """``(H - eps S + alpha Q) dpsi = -P_c^+ dV psi`` at a fixed ground state.

    Built once for a converged calculation and then applied to as many
    perturbations as the caller has: the operator, the projector and the
    preconditioner all depend on the ground state alone.
    """

    def __init__(
        self,
        calculation,
        hamiltonians,
        psi,
        eigenvalues,
        weights,
        nocc,
        threshold: float = THRESHOLD,
        max_iterations: int = MAX_ITERATIONS,
        v_scf=None,
        becsum=(),
        smearing: Smearing | None = None,
        kpoint_weights=None,
    ):
        self.calculation = calculation
        self.hamiltonians = tuple(hamiltonians)
        # The ground state's potential and projector occupations. Not needed to
        # *solve* -- the Hamiltonians carry everything for that -- but needed to
        # build an ultrasoft or PAW perturbation, whose nonlocal part moves with
        # the potential (``int3``) and whose one-centre part moves with
        # ``becsum`` (``PAW_dpotential``).
        self.v_scf = None if v_scf is None else jnp.asarray(v_scf)
        self.becsum = becsum
        self.ddd_paw = calculation.onecenter(becsum)[1] if becsum else None
        # Only the occupied bands are solved for. The empty ones are what the
        # sum-over-states form would need and what this form exists to avoid;
        # they are kept out of ``psi`` here so that no shape carries them.
        #
        # **A metal keeps all of them**, because "occupied" is not a count there:
        # ``orthogonalize``'s smearing branch sums over every band (its
        # ``nbnd_eff = nbnd``) and the occupation rides inside ``dpsi`` as a
        # weight rather than deciding which bands exist. QE truncates the *solve*
        # at ``setup_nbnd_occ``'s ``nbnd_occ``; here the block stays whole and
        # the bands past that point are multiplied by an occupation of zero,
        # which is the same answer at a static shape (rule R2) and costs the
        # empty bands' share of the CG.
        self.smearing = smearing
        # **The occupied-band count is one number per spin channel**, because a
        # magnetic insulator's channels are filled to different depths. QE never
        # meets this: LSDA doubles ``nks`` there, so its spin channels are
        # separate k-points and ``setup_nbnd_occ`` writes one ``nbnd_occ(ik)``
        # per k-point that already carries the channel. Here the channel is an
        # axis of one array, so what QE gets from its k index is got from a
        # per-channel count and a mask (rule R2: the shape stays static and the
        # deficient channel's extra bands are masked, not removed).
        counts = _normalized_counts(nocc, psi.shape[0])
        self.occupied_counts = counts
        keep = psi.shape[2] if smearing is not None else max(counts)
        self.psi = jnp.asarray(psi)[:, :, :keep]
        self.eigenvalues = jnp.asarray(eigenvalues)[:, :, :keep]
        self.weights = jnp.asarray(weights)[:, :, :keep]
        self.nocc = int(keep)
        self.threshold = float(threshold)
        self.max_iterations = int(max_iterations)
        self.alpha_pv = _alpha_pv(np.asarray(eigenvalues), counts, smearing)
        # ``nbnd_occ``: which bands the ``alpha_pv`` projector runs over, and
        # which ones ``orthogonalize``'s ``jbnd <= nbnd_occ(ikq)`` admits into
        # the level-shift correction. A boolean mask rather than a per-k count,
        # for the same reason ``cegterg``'s repacking became one.
        if smearing is None:
            bands = jnp.arange(self.eigenvalues.shape[2])
            self.projector_mask = jnp.broadcast_to(
                (bands[None, :] < jnp.asarray(counts)[:, None])[:, None, :],
                self.eigenvalues.shape,
            )
            if len(counts) > 1:
                _require_a_gap_at_the_cut(np.asarray(eigenvalues), counts)
        else:
            self.projector_mask = self.eigenvalues < smearing.cutoff
        # **The weight the density response is built with is not ``wg``.** In the
        # smearing branch the occupation is applied to ``dpsi`` itself
        # (``dvpsi = wg1 dvpsi``), so accumulating ``drho`` with ``wg`` on top
        # would count it twice; ``incdrhoscf`` is called with ``wk`` and this is
        # that ``wk``. For an insulator the two coincide -- a filled band has
        # ``wg = wk`` -- which is why nothing before this had to tell them apart.
        if smearing is None or kpoint_weights is None:
            self.density_weights = self.weights
        else:
            self.density_weights = jnp.broadcast_to(
                jnp.asarray(kpoint_weights)[None, :, None], self.weights.shape
            )

    @property
    def nspin(self) -> int:
        return self.psi.shape[0]

    # -- the pieces of ``ch_psi_all`` -------------------------------------

    def _operator(self, vectors, ik, spin):
        """``(H - eps S + alpha_pv S P_occ S) |h>`` -- ``ch_psi_all``."""
        hamiltonian = self.hamiltonians[spin]
        occupied = self.psi[spin][ik]
        eps = self.eigenvalues[spin][ik][:, None]

        h = hamiltonian.apply(vectors, ik)
        s = hamiltonian.apply_s(vectors, ik)
        out = h - eps * s

        # The level shift. ``S P_occ S``: project the *overlapped* vector onto
        # the occupied manifold, then apply ``S`` again -- which for a
        # norm-conserving dataset is the identity twice and the plain projector.
        # ``ch_psi_all`` runs this sum over ``nbnd_occ``, which for a metal is
        # not every band in the block -- hence the mask.
        overlaps = jnp.einsum("mg,ng->mn", jnp.conj(occupied), s)
        overlaps = jnp.where(self.projector_mask[spin][ik][:, None], overlaps, 0.0)
        lifted = jnp.einsum("mn,mg->ng", overlaps, occupied)
        return out + self.alpha_pv * hamiltonian.apply_s(lifted, ik)

    def project(self, rhs, ik, spin):
        """``-P_c^+ rhs`` where ``P_c^+ = 1 - S |psi_occ><psi_occ|``.

        ``orthogonalize.f90``, **including its sign**: the routine returns minus
        the projected vector, because the right-hand side of the Sternheimer
        equation is ``-P_c^+ dV|psi>``. The insulator branch is the sharp
        projector above; :meth:`_smeared_projection` is the other one.
        """
        hamiltonian = self.hamiltonians[spin]
        occupied = self.psi[spin][ik]
        s_occupied = hamiltonian.apply_s(occupied, ik)
        if self.smearing is not None:
            overlaps = jnp.einsum("mg,ng->mn", jnp.conj(occupied), rhs)
            rhs, overlaps = self._smeared_projection(rhs, overlaps, ik, spin)
            return -(rhs - jnp.einsum("mn,mg->ng", overlaps, s_occupied))
        # The sharp branch, and the mask is the identity except in the deficient
        # channel of a magnetic insulator. It does two different jobs there, and
        # only the first is about conditioning:
        #
        # * the **row** mask zeroes the right-hand side of a band this channel
        #   does not occupy, so the CG returns ``dpsi = 0`` at iteration zero
        #   instead of solving ``H - eps_n S`` for an *empty* ``n``, where the
        #   operator is not positive definite and the level shift does not make
        #   it so;
        # * the **column** mask (over ``m``) keeps ``P_c^+`` projecting out this
        #   channel's own occupied manifold. Without it the projector would also
        #   remove the bands the *other* channel fills -- which is precisely the
        #   subspace the deficient channel's response lives in.
        keep = self.projector_mask[spin][ik][:, None]
        rhs = jnp.where(keep, rhs, 0.0)
        overlaps = jnp.where(
            keep, jnp.einsum("mg,ng->mn", jnp.conj(occupied), rhs), 0.0
        )
        return -(rhs - jnp.einsum("mn,mg->ng", overlaps, s_occupied))

    def _smeared_projection(self, rhs, overlaps, ik, spin):
        """``orthogonalize``'s metal branch: the sharp step becomes ``wwg``.

        For an insulator ``P_c^+`` is a projector -- a band is in the occupied
        manifold or it is not. For a metal there is no such partition, and what
        replaces it is a pair of *weights* (``PRB 51, 6773 (1995)``, Eq. 75):

            dvpsi_i  <-  wg1_i dvpsi_i
            ps_(j,i) <-  wwg_(j,i) <psi_j|dvpsi_i>

        with ``wg1_i = theta(ef - eps_i)`` the smeared occupation of the band
        being solved for, and

            wwg_(j,i) = wg1_i (1 - t) + wg1_j t
                        + alpha_pv t (wg1_j - wg1_i) / (eps_j - eps_i)

        where ``t = theta((eps_j - eps_i)/degauss)`` is a *second* smeared step,
        this one of the energy difference. Two things about that expression are
        worth keeping in view. The ``0/0`` at a degeneracy is taken to its limit
        ``-alpha_pv t w0gauss_i`` rather than guarded, which matters because a
        crystal is degenerate everywhere by symmetry (rule D4) -- and the branch
        is written with :func:`jnp.where` on a *safe* denominator, so the unused
        side never produces a NaN that would poison the gradient. And the
        ``alpha_pv`` piece is admitted only for ``j`` inside ``nbnd_occ``, which
        is :attr:`projector_mask`; without that cut the empty bands contribute a
        term QE does not have.

        Everything here reduces to the insulator branch when the smearing is
        narrow: ``wg1`` becomes a step, ``t`` becomes a step, and ``wwg`` becomes
        the sharp ``1`` on the occupied block that :meth:`project` applies above.
        """
        smearing = self.smearing
        eps = self.eigenvalues[spin][ik]
        degauss, ngauss = smearing.degauss, smearing.ngauss
        occupation = wgauss((smearing.ef - eps) / degauss, ngauss)      # wg1
        delta = w0gauss((smearing.ef - eps) / degauss, ngauss) / degauss

        difference = eps[:, None] - eps[None, :]                        # eps_j - eps_i
        step = wgauss(difference / degauss, 0)                          # theta
        mixed = occupation[None, :] * (1.0 - step) + occupation[:, None] * step

        close = jnp.abs(difference) <= 1.0e-5
        safe = jnp.where(close, 1.0, difference)
        ratio = (occupation[:, None] - occupation[None, :]) / safe
        shift = self.alpha_pv * step * jnp.where(close, -delta[None, :], ratio)
        weights = mixed + jnp.where(self.projector_mask[spin][ik][:, None], shift, 0.0)

        return occupation[:, None] * rhs, weights * overlaps

    def _preconditioner(self, ik, spin):
        """``h_prec``: ``1 / max(1, |k+G|^2 / eprec_n)``, ``eprec = 1.35 <T>``."""
        hamiltonian = self.hamiltonians[spin]
        occupied = self.psi[spin][ik]
        kinetic = hamiltonian.state_kinetic[ik]
        eprec = 1.35 * jnp.real(
            jnp.einsum("ng,g,ng->n", jnp.conj(occupied), kinetic, occupied)
        )
        return 1.0 / jnp.maximum(1.0, kinetic[None, :] / eprec[:, None])

    # -- the solver -------------------------------------------------------

    def solve_at(self, rhs, ik, spin):
        """``cgsolve_all`` for one k-point: the projected CG, masked by band.

        ``rhs`` is the *already projected* right-hand side -- what
        :meth:`project` returns. Returns ``(dpsi, iterations, residual)``.
        """
        hamiltonian = self.hamiltonians[spin]
        mask = hamiltonian.state_mask[ik]
        precondition = self._preconditioner(ik, spin)
        threshold = self.threshold

        def operator(vectors):
            return jnp.where(mask, self._operator(vectors, ik, spin), 0.0)

        def dot(a, b):
            """``MYDDOTV3``: the real part of the Hermitian product, per band."""
            return jnp.real(jnp.einsum("ng,ng->n", jnp.conj(a), b))

        rhs = jnp.where(mask, rhs, 0.0)
        dpsi = jnp.zeros_like(rhs)
        gradient = operator(dpsi) - rhs

        state = (
            dpsi,
            gradient,
            jnp.zeros_like(rhs),                      # hold: the previous step
            jnp.zeros(rhs.shape[0]),                  # rhoold
            jnp.zeros(rhs.shape[0]),                  # rho, for the report
            jnp.array(0),
            jnp.zeros(rhs.shape[0], dtype=bool),      # conv
        )

        def body(state):
            dpsi, gradient, hold, rhoold, _, iteration, conv = state
            preconditioned = precondition * gradient
            rho = dot(preconditioned, gradient)
            conv = conv | (jnp.sqrt(jnp.abs(rho)) < threshold)

            # ``dcgamma = rho/rhoold``, zero on the first iteration -- which is
            # what ``hold = 0`` makes it here, rather than a branch.
            gamma = jnp.where(rhoold > 0.0, rho / jnp.where(rhoold > 0.0, rhoold, 1.0), 0.0)
            direction = -preconditioned + gamma[:, None] * hold
            applied = operator(direction)

            a = dot(direction, gradient)
            c = dot(direction, applied)
            # A converged band takes a zero step and stops moving; this is the
            # mask that replaces ``cgsolve_all``'s repacking of the block.
            safe = jnp.where(jnp.abs(c) > 0.0, c, 1.0)
            step = jnp.where(conv | (jnp.abs(c) == 0.0), 0.0, -a / safe)

            return (
                dpsi + step[:, None] * direction,
                gradient + step[:, None] * applied,
                jnp.where(conv[:, None], hold, direction),
                jnp.where(conv, rhoold, rho),
                rho,
                iteration + 1,
                conv,
            )

        def keep_going(state):
            iteration, conv = state[5], state[6]
            return (iteration < self.max_iterations) & ~jnp.all(conv)

        final = lax.while_loop(keep_going, body, state)
        dpsi, _, _, _, rho, iterations, _ = final
        return jnp.where(mask, dpsi, 0.0), iterations, jnp.sqrt(jnp.max(jnp.abs(rho)))

    def solve(self, perturbation) -> SternheimerResult:
        """Solve at every k-point and spin channel.

        ``perturbation(psi, ik, spin)`` returns ``dV|psi>`` for the occupied
        block at one k-point -- unprojected, since :meth:`project` is applied
        here. It is a function rather than an array because the perturbations
        this module serves are not all local potentials: an electric field is a
        commutator (:mod:`defumat.response.efield`).
        """
        batch = self.calculation.k_batch
        blocks, iterations, residuals = [], [], []
        for spin in range(self.nspin):
            def one_k(ik, spin=spin):
                rhs = self.project(
                    perturbation(self.psi[spin][ik], ik, spin), ik, spin
                )
                return self.solve_at(rhs, ik, spin)

            dpsi, steps, residual = map_k(
                one_k, jnp.arange(self.psi.shape[1]), batch=batch
            )
            blocks.append(dpsi)
            iterations.append(int(jnp.max(steps)))
            residuals.append(float(jnp.max(residual)))
        return SternheimerResult(
            dpsi=jnp.stack(blocks),
            iterations=max(iterations),
            residual=max(residuals),
        )

    # -- the density it produces -------------------------------------------

    def density_at(self, states, weights=None) -> jnp.ndarray:
        """``rho`` from a set of occupied states, as the SCF builds it.

        :meth:`~defumat.scf.driver.Calculation.density` **without the
        symmetrisation**: a response is symmetrised as a vector, not as a scalar
        (:meth:`~defumat.scf.driver.Calculation.symmetrize_directional`), so the
        caller does it. Everything else is the same function the SCF uses --
        including ``becsum`` and the augmentation charge, which is what makes
        the derivative below carry them.
        """
        calculation = self.calculation
        smooth, dense = calculation.basis.smooth, calculation.basis.dense
        weights = self.density_weights if weights is None else weights
        becsum_ = self._raw_becsum(states, weights)
        rho = sum_band(
            states, calculation.fft_index, smooth.grid, weights,
            calculation.system.cell, calculation.k_batch,
        )
        return calculation.augmented(to_dense(rho, smooth, dense), becsum_)

    def response_density(self, dpsi) -> jnp.ndarray:
        """``drho``: the first-order density, as one ``jvp`` of :meth:`density_at`.

        ``incdrhoscf`` plus ``addusdbec`` plus ``lr_addusddens``, and none of the
        three is transcribed. The density is a *quadratic* function of the states
        that this code already writes down once, so its response to
        ``psi -> psi + lambda dpsi`` is the directional derivative of that
        function -- and every term QE adds by hand for an ultrasoft dataset comes
        with it:

        * the smooth part, ``sum_kn wg 2 Re[psi* dpsi]/Omega``, which is the
          derivative of ``|psi|^2`` (``incdrhoscf``);
        * ``dbecsum``, the derivative of ``becsum``, which is bilinear in the
          projections (``addusdbec``);
        * the augmentation charge's own response, ``sum_ij Q_ij(r) dbecsum_ij``,
          which is the derivative of ``addusdens`` -- linear in ``becsum``, so it
          is that same sum with ``dbecsum`` in it (``lr_addusddens``).

        The one thing this needed was for ``|psi|^2`` to be written as
        ``Re(conj(psi) psi)`` rather than ``abs(psi)**2``, whose derivative is
        ``0/0`` at a node (:func:`defumat.scf.density.band_density`).

        On the **dense** grid, like every other density here.
        """
        _, drho = jax.jvp(self.density_at, (self.psi,), (jnp.asarray(dpsi),))
        return drho

    def response_becsum(self, dpsi) -> tuple:
        """``dbecsum``: the projector occupations' response, on its own.

        The augmentation charge's share of ``drho`` is already inside
        :meth:`response_density`; this is the same derivative kept separately,
        because **PAW's one-centre terms are a function of ``becsum`` and not of
        the density** and there is nowhere else to get them from
        (``PAW_dpotential``).
        """
        _, dbecsum = jax.jvp(
            self._raw_becsum, (self.psi,), (jnp.asarray(dpsi),)
        )
        return dbecsum

    def _raw_becsum(self, states, weights=None) -> tuple:
        """``becsum`` **without** the symmetrisation :meth:`Calculation.becsum` applies.

        For PAW that method ends with ``PAW_symmetrize``, and a *response* must
        not go through it: the three response densities are symmetrised together
        as a polar vector afterwards
        (:meth:`~defumat.scf.driver.Calculation.symmetrize_directional`), and
        pre-averaging each one as a scalar is the same wrong-symmetry mistake in
        a place where it is much harder to see. It is worth **1.6e-2** on the
        dielectric constant of PAW silicon, against the 5e-5 the rest of the
        machinery reaches.
        """
        calculation = self.calculation
        if not calculation.is_ultrasoft:
            return ()
        return becsum_of(
            states, calculation.projectors.vkb,
            self.density_weights if weights is None else weights,
            calculation.species_channels, calculation.k_batch,
        )

    def local_density_of_states(self):
        """``localdos``: ``(ldos, becsum1, dos_ef)`` at the Fermi level.

        ``ldos(r) = sum_kn w_k delta(ef - eps_kn) |psi_kn(r)|^2`` and
        ``dos_ef = sum_kn w_k delta(ef - eps_kn)``. **It is the same density
        builder with a different weight** -- the smeared delta in place of the
        smeared step -- so ``localdos.f90``'s hundred lines, including its
        ``becsum1``, are :meth:`density_at` called once more.

        Only a metal has one; an insulator raises, because ``delta(ef - eps)``
        is zero everywhere there and the caller asking for it has confused two
        regimes.
        """
        if self.smearing is None:
            raise ValueError(
                "a local density of states at the Fermi level is a metallic "
                "quantity: this run has occupations='fixed' and no Fermi surface"
            )
        weights = self._delta_weights()
        return self.density_at(self.psi, weights), float(jnp.sum(weights))

    def _delta_weights(self):
        """``w_k delta(ef - eps_kn)``, the weight :meth:`local_density_of_states` uses."""
        smearing = self.smearing
        x = (smearing.ef - self.eigenvalues) / smearing.degauss
        return self.density_weights * w0gauss(x, smearing.ngauss) / smearing.degauss

    def fermi_level_shift(self, drho):
        """``ef_shift``: the Fermi level moves, and the response density with it.

        ``PW/src`` has no counterpart; this is ``LR_Modules/efermi_shift.f90``.
        A perturbation at ``q = 0`` is not orthogonal to the identity, so it
        changes the number of states below the Fermi level and the level itself
        has to move to keep the electron count. The shift is fixed by that count,

            def = - (integral of drho) / N(ef),

        and what it does to the density is fill or empty the Fermi surface:
        ``drho <- drho + def ldos``. Both pieces come from
        :meth:`local_density_of_states`.

        Returns ``(corrected, def)``. **The corrected density integrates to
        zero**, which is the identity this is checked against and is not
        imposed: ``ldos`` integrates to ``dos_ef`` by construction, so the
        correction removes exactly the charge the uncorrected response invented.

        ``drho`` is ``(..., nspin_mag, n1, n2, n3)`` with any number of leading
        perturbation axes; the shift is computed per perturbation.
        """
        ldos, dos_ef = self.local_density_of_states()
        calculation = self.calculation
        cell = calculation.system.cell
        element = cell.volume / int(np.prod(ldos.shape[1:]))
        leading = drho.ndim - ldos.ndim
        axes = tuple(range(leading, drho.ndim))
        delta_n = jnp.sum(drho, axis=axes) * element
        shift = jnp.where(jnp.abs(dos_ef) > 1.0e-18, -delta_n / dos_ef, 0.0)
        return drho + shift[(...,) + (None,) * ldos.ndim] * ldos, shift

    def fermi_level_shift_states(self, dpsi, shift):
        """``ef_shift_wfc``: the same shift applied to the first-order states.

        ``dpsi_n <- dpsi_n + (1/2) def delta(ef - eps_n) psi_n``. The half is
        QE's and is not a typo: the density is quadratic in the states, so half
        the shift on each of ``psi`` and its conjugate reproduces the whole of
        the ``def ldos`` that :meth:`fermi_level_shift` added to the density.
        It matters only where ``dpsi`` is consumed as a *tangent* rather than
        through the density it already carries -- which is the second derivative
        (:mod:`defumat.response.phonon`), not ``chi_0``.
        """
        smearing = self.smearing
        x = (smearing.ef - self.eigenvalues) / smearing.degauss
        delta = w0gauss(x, smearing.ngauss) / smearing.degauss
        return dpsi + 0.5 * shift * delta[..., None] * self.psi

    def perturbation(self, dv, dddd_paw=None):
        """``dH|psi>`` for a change ``dv`` in the potential -- see
        :func:`local_perturbation`, with this solver's ground state filled in."""
        return local_perturbation(
            self.calculation, dv, self.v_scf, self.ddd_paw, dddd_paw
        )

    def chi0(self, dv) -> jnp.ndarray:
        """``chi_0 dV``: the independent-particle density response to a potential.

        ``dv`` is ``(nspin, n1, n2, n3)`` real on the **dense** grid, the shape a
        potential has. The result is a density of the same shape.

        **PAW's one-centre response is deliberately not in it.** ``chi_0`` is the
        response at a *frozen* one-centre potential, which is what an
        independent-particle susceptibility means; the term that makes
        ``ddd_paw`` move with ``becsum`` is part of the self-consistent loop and
        is added there (:mod:`defumat.response.efield`).
        """
        return self.response_density(self.solve(self.perturbation(dv)).dpsi)


def local_perturbation(calculation, dv, v_scf=None, ddd_paw=None, dddd_paw=None):
    """``dH|psi>`` for a change ``dv`` in the self-consistent potential.

    For a norm-conserving dataset this is ``dV(r)|psi>`` and nothing else. For an
    ultrasoft one it is not, and the missing piece is invisible in the answer:
    ``D_ij = D_ij^(0) + int V_eff(r) Q_ij(r) dr`` depends on the potential, so a
    change in the potential changes the *nonlocal* operator as well,

        dH |psi> = dV(r) |psi> + sum_ij |beta_i> int3_ij <beta_j|psi>,
        int3_ij  = int dV(r) Q_ij(r) dr,

    which is ``newdq.f90`` and ``adddvscf.f90``. Neither is transcribed:
    ``int3`` is one ``jvp`` of :meth:`~defumat.scf.driver.Calculation.
    coefficients`, which is ``newd`` and is already a differentiable function of
    the potential.

    ``dddd_paw`` is PAW's one-centre response, which is *not* a function of the
    potential and so cannot come from the same derivative -- it is
    ``PAW_dpotential``, obtained from ``becsum``'s response
    (:func:`paw_response`) and simply added to ``int3`` here, exactly as
    ``add_paw_to_deeq`` adds ``ddd_paw`` to ``deeq``.

    The potential arrives on the dense grid and is interpolated to the smooth one
    exactly as ``Calculation.hamiltonian`` interpolates the self-consistent
    potential: the wavefunctions live there, and applying a dense-grid field to
    them would be applying a different operator.
    """
    dense, smooth = calculation.basis.dense, calculation.basis.smooth
    grid = smooth.grid
    fft_index = calculation.fft_index
    mask = calculation.basis.planewaves.mask
    n = grid[0] * grid[1] * grid[2]
    dv = jnp.asarray(dv)
    fields = jnp.stack([
        to_smooth(dv[spin], dense, smooth) for spin in range(dv.shape[0])
    ])
    coefficients = _perturbed_coefficients(
        calculation, dv, v_scf, ddd_paw, dddd_paw
    )
    vkb = calculation.projectors.vkb

    def apply(states, ik, spin):
        index = fft_index[ik]
        box = jnp.fft.fftn(
            g_to_r(states, index, grid) * fields[spin], axes=(-3, -2, -1)
        ) / n
        out = jnp.take(box.reshape(box.shape[:-3] + (-1,)), index, axis=-1)
        if coefficients is not None:
            projectors = vkb[ik]
            projections = jnp.einsum("gk,...g->...k", projectors.conj(), states)
            dij = coefficients[spin].astype(projectors.dtype)
            out = out + jnp.einsum("gk,...k->...g", projectors, projections @ dij.T)
        return jnp.where(mask[ik], out, 0.0)

    return apply


def _perturbed_coefficients(calculation, dv, v_scf, ddd_paw, dddd_paw):
    """``int3 + d(ddd_paw)``: how ``D_ij`` moves when the potential does.

    ``None`` for a norm-conserving dataset, where ``D_ij`` is the file's and
    nothing moves it.
    """
    if not calculation.is_ultrasoft:
        return None
    if v_scf is None:
        raise ValueError(
            "an ultrasoft or PAW perturbation needs the ground-state potential "
            "as well as its change: D_ij depends on V_eff, so dH carries "
            "int3_ij = int dV Q_ij dr (newdq/adddvscf) and that derivative is "
            "taken at the converged potential"
        )
    components = as_potential_components(calculation.vltot, calculation.nspin_mag)
    _, int3 = jax.jvp(
        lambda potential: calculation.coefficients(potential, ddd_paw),
        (jnp.asarray(v_scf) + components,),
        (dv,),
    )
    return int3 if dddd_paw is None else int3 + dddd_paw


def paw_response(calculation, dbecsum, becsum_):
    """``PAW_dpotential``: the one-centre coefficients' response to ``dbecsum``.

    ``paw_onecenter.f90``. PAW's ``ddd_paw`` is a function of ``becsum`` and of
    nothing else, so its response is the directional derivative of
    :meth:`~defumat.scf.driver.Calculation.onecenter` along ``dbecsum`` -- one
    ``jvp``, where QE writes a second radial routine beside the first.

    ``None`` when no species is PAW, which is the norm-conserving and plain
    ultrasoft case both.
    """
    if calculation.paw is None:
        return None
    _, ddd = jax.jvp(
        lambda parts: calculation.onecenter(parts)[1], (becsum_,), (dbecsum,)
    )
    return ddd


def _nint(value) -> int:
    """Fortran's ``NINT`` for a non-negative count: half rounds *away* from zero.

    Python's :func:`round` is banker's rounding -- ``round(2.5)`` is 2 -- and
    ``_fixed_occupations_spin`` fills ``int(floor(count + 1/2))`` bands, so the
    two disagree at exactly a half electron. That happens: an **odd** electron
    count with an **even** ``tot_magnetization`` gives a half-integer ``nelup``,
    and QE rounds it up rather than refusing (``GAPS.md`` section 3). Using
    ``round`` here would build a mask one band shallower than the weights the
    density was made with, in exactly that case and no other.
    """
    return int(np.floor(float(value) + 0.5))


def _normalized_counts(nocc, nspin: int) -> tuple:
    """``nocc`` as one count per spin channel.

    Accepts a single number -- every earlier caller passes one -- and repeats it
    across the channels, which is what an unpolarized run and a shared-Fermi
    metal both want. A sequence is taken as it stands and must have one entry
    per channel.
    """
    if np.ndim(nocc) == 0:
        return (_nint(nocc),) * nspin
    counts = tuple(_nint(n) for n in nocc)
    if len(counts) == 1:
        return counts * nspin
    if len(counts) != nspin:
        raise ValueError(
            f"the Sternheimer solver was given {len(counts)} occupied-band "
            f"counts for {nspin} spin channels"
        )
    return counts


def occupied_counts(calculation) -> tuple:
    """How many bands each spin channel occupies -- ``setup_nbnd_occ``'s ``nbnd_occ``.

    One number per channel, because a magnetic insulator's channels are filled
    to different depths and the three perturbation modules used to derive
    ``nelec / 2`` themselves. The rule follows
    :func:`~defumat.scf.occupations.spin_electron_counts`, which is QE's
    ``set_nelup_neldw``:

    * ``nspin = 1``: ``nelec / 2`` (``nelec`` for a spinor, whose band holds one
      electron rather than two);
    * ``nspin = 2``: ``(NINT(nelup), NINT(neldw))``. With ``occupations =
      'fixed'`` those are what ``iweights_only`` fills -- QE refuses fixed LSDA
      without ``tot_magnetization`` and requires an integer one
      (``input.f90:784-800``), so the pair is always defined and always integral
      on the branch that slices. With a smearing they are ``nelec / 2`` twice
      and unused: the metal branch keeps every band and masks by ``eps <
      ef + xmax degauss`` instead.

    Returned as a tuple so that a caller can pass it straight to
    :class:`SternheimerSolver`.
    """
    if calculation.nspin != 2:
        degeneracy = 1 if calculation.noncolin else 2
        # ``fixed_occupations``' unpolarized branch uses ``round`` here and
        # refuses a non-integral filling outright, so the two agree.
        return (int(round(calculation.nelec / degeneracy)),)
    return (_nint(calculation.nelup), _nint(calculation.neldw))


def _require_a_gap_at_the_cut(eigenvalues, counts) -> None:
    """Refuse a filling whose boundary lands inside a degenerate multiplet.

    See :data:`DEGENERATE_CUT_RY`. The check needs a band above the cut to
    measure against; where the run carries exactly ``nbnd = max(counts)`` there
    is nothing to compare and the caller is trusted, as it was before.
    """
    for spin, count in enumerate(counts):
        if count <= 0 or count >= eigenvalues.shape[2]:
            continue
        gap = float(np.min(eigenvalues[spin][:, count] - eigenvalues[spin][:, count - 1]))
        if gap < DEGENERATE_CUT_RY:
            raise NotImplementedError(
                f"spin channel {spin} occupies {count} bands and the gap to the "
                f"first empty one is {gap:.2e} Ry: the filling cuts a degenerate "
                "multiplet, so there is no response to compute. Which member of "
                "the multiplet the eigensolver returned is arbitrary, so P_c is "
                "built for an arbitrary subspace of a degenerate shell and the "
                "answer is a property of the eigensolver rather than of the "
                "density -- measured at 100 per cent against a finite difference "
                "on the oxygen atom, with the CG converging normally, which is "
                "why this is refused rather than warned about. It is the physics "
                "rather than a missing term (a Hund's-rule atom is exactly this "
                "case); occupy the whole shell, or use a smearing"
            )


def _alpha_pv(eigenvalues, counts, smearing=None) -> float:
    """``setup_alpha_pv``: the level shift that makes the operator positive definite.

    Insulator: ``2 (eps_max^occ - eps_min)``. **Metal**: ``emax - emin`` with
    ``emax = ef + xmax degauss``, the same cutoff ``setup_nbnd_occ`` uses -- the
    occupied manifold has no top there, so the shift is measured to where the
    smearing function has died instead. Both are floored at 1e-2, as QE floors
    them.

    ``counts`` is one occupied-band count per spin channel, and the insulator's
    ``emax`` is the largest over the channels -- which is what QE's own
    ``setup_alpha_pv`` computes, since its LSDA ``et`` array runs over ``nks``
    k-points that already include both channels. The metal branch does not read
    it at all: ``ef + xmax degauss`` is one number for a shared Fermi level.
    """
    if smearing is not None:
        emin = float(np.min(eigenvalues))
        return max(smearing.cutoff - emin, 1.0e-2)
    emin = float(np.min(eigenvalues))
    emax = max(
        float(np.max(eigenvalues[spin][..., :count]))
        for spin, count in enumerate(counts) if count > 0
    )
    return max(2.0 * (emax - emin), 1.0e-2)


def require_a_sternheimer_regime(
    calculation, metals: bool = False, spin_polarized: bool = False
) -> None:
    """Refuse, by name, every regime whose response needs machinery not here.

    A separate function because several entry points need the same list -- this
    module's :func:`make_sternheimer`, the electric field's
    :func:`~defumat.response.efield.dielectric_tensor`, the phonons' -- and a
    refusal stated twice is a refusal that will eventually be stated
    differently. See the module docstring for what each case would need.

    ``metals = True`` says the *caller's* quantity exists for a metal. The solve
    does: ``orthogonalize``'s smearing branch is implemented
    (:meth:`SternheimerSolver._smeared_projection`). ``epsilon_infinity`` and
    the Born charges do not, and are refused here rather than in three places.

    ``spin_polarized = True`` says the same thing about ``nspin = 2``: the
    *solve* now takes one occupied-band count per channel
    (:func:`occupied_counts`) and is validated for both regimes, but a caller's
    **assembly** on top of it is a separate claim -- the third derivatives of
    :mod:`defumat.response.electrostriction` and
    :mod:`defumat.response.nonlinear` have never been run with a spin axis and
    are still refused here. The flag is deliberately opt-in for that reason,
    exactly as ``metals`` is.
    """
    if getattr(calculation, "gamma_only", False):
        raise NotImplementedError(
            "a response quantity is not implemented for gamma-only storage. "
            "The SCF consumes the half sphere (P68) and this stack does not: "
            "every inner product here -- the projector in orthogonalize, the "
            "CG's own products in cgsolve_all, the response density -- is a sum "
            "over plane waves and needs 2 Re(...) minus the G = 0 term, and "
            "none of them has it. **It does not fail, which is why this refusal "
            "exists**: silicon's dielectric constant comes out 285.4/229.4/228.3 "
            "against 190.8/190.8/190.8, half again too large and not even cubic "
            "on a cubic crystal. Run the same cell with an explicit k = 0 "
            "(K_POINTS automatic, 1 1 1 0 0 0), which is the same physics on the "
            "whole sphere"
        )
    system = calculation.system
    reject_potential_only(calculation)
    if calculation.noncolin:
        raise NotImplementedError(
            "the Sternheimer response is not implemented for a noncollinear or "
            "spin-orbit calculation: incdrhoscf_nc and set_int3_nc are a second "
            "implementation rather than a spin axis on this one"
        )
    if calculation.is_hubbard:
        raise NotImplementedError(
            "the Sternheimer response with a Hubbard U is not implemented: the "
            "induced potential carries a dns of its own (adddvhubscf.f90), "
            "which is not a function of drho"
        )
    scheme = system.occupations
    if scheme.startswith("tetrahedra"):
        # ``orthogonalize``'s other metallic branch reads ``dfpt_tetra_beta``,
        # a per-pair weight the tetrahedron machinery builds for the response
        # and which nothing here computes. The smearing family is implemented;
        # this one is refused rather than silently run with a smeared weight.
        raise NotImplementedError(
            f"occupations={scheme!r}: the Sternheimer response of a metal is "
            "implemented for the smearing family only. The tetrahedron branch "
            "needs dfpt_tetra_beta -- a response weight per band *pair*, which "
            "the tetrahedron occupations here do not build. Re-run the ground "
            "state with a smearing"
        )
    if scheme == "from_input":
        raise NotImplementedError(
            "occupations='from_input': the response projector needs occupations "
            "that are a differentiable function of a Fermi level, and an "
            "OCCUPATIONS card is neither"
        )
    if scheme != "fixed" and not metals:
        raise NotImplementedError(
            f"occupations={scheme!r}: this response is refused for a metal. "
            "The Sternheimer solve itself handles one (orthogonalize's smearing "
            "branch is implemented), but the quantity being asked for is not "
            "defined there -- a metal has no epsilon_infinity and no Born "
            "effective charge, which is why pw.x refuses epsil for one too"
        )
    if calculation.spiral:
        raise NotImplementedError("the Sternheimer response of a spin spiral is not implemented")
    if calculation.nspin == 2:
        # **The solve is spin-polarized now** (:func:`occupied_counts`), and
        # what is left is a per-quantity claim rather than the old blanket one.
        if not spin_polarized:
            raise NotImplementedError(
                "this response quantity is not implemented for nspin = 2. The "
                "Sternheimer *solve* is -- it takes one occupied-band count per "
                "spin channel (occupied_counts) and chi_0 is validated against a "
                "finite difference of the density in both regimes -- but the "
                "assembly on top of it in this module has never been run with a "
                "spin axis, so it is refused rather than reported"
            )
        if calculation.two_fermi_energies and scheme != "fixed":
            # ``smearing_of`` would build a :class:`Smearing` from
            # ``result.fermi_energy``, which for a constrained magnetization is
            # the **mean** of the two levels -- a number QE prints only so that
            # the field is not NaN. Every weight in ``_smeared_projection``
            # would then be evaluated at a level neither channel has, and the
            # result would be smooth, plausible and wrong.
            raise NotImplementedError(
                "the Sternheimer response with tot_magnetization and a smearing "
                "is not implemented: the two channels have separate Fermi levels "
                "there and Smearing carries a single scalar ef, so the projector "
                "would be evaluated at the mean of the two -- which is the number "
                "QE prints only to avoid a NaN. Giving Smearing.ef a spin axis "
                "(and smearing_of the pair off the result) is the missing piece. "
                "Drop tot_magnetization for a shared Fermi level, or use "
                "occupations='fixed', whose channels need no level at all"
            )


def smearing_of(calculation, result) -> Smearing | None:
    """The :class:`Smearing` of a converged run, or ``None`` if it is an insulator.

    ``result`` is anything carrying a ``fermi_energy``. The level is **taken
    from the run and not re-derived**: it is what the occupations the density was
    built from were evaluated at, and re-deriving it would differ by the
    bisection tolerance -- which the projector would feel as an inconsistency
    between its ``wg1`` and the ``wg`` the ground state used. Where a caller has
    no ``SCFResult`` to hand (:func:`defumat.response.phonon.dynamical_matrix`
    is given a density and eigenvalues instead) it re-runs
    :meth:`~defumat.scf.driver.Calculation.occupations` on those same
    eigenvalues, which is the same call the SCF made on the same numbers and so
    the same level, not an independent bisection.
    """
    scheme = calculation.system.occupations
    if scheme == "fixed":
        return None
    return Smearing(
        ef=float(result.fermi_energy),
        degauss=float(calculation.system.degauss),
        ngauss=smearing_order(calculation.system.smearing),
    )


def make_sternheimer(calculation, result, threshold: float = THRESHOLD,
                     metals: bool = False, spin_polarized: bool = False):
    """A solver for a converged :class:`~defumat.scf.driver.SCFResult`."""
    require_a_sternheimer_regime(
        calculation, metals=metals, spin_polarized=spin_polarized
    )
    eigenvalues = jnp.asarray(result.eigenvalues)
    if eigenvalues.ndim == 2:
        eigenvalues = eigenvalues[None]
    weights, _ = calculation.occupations(eigenvalues)
    weights = jnp.asarray(weights)

    nocc = occupied_counts(calculation)
    potential = calculation.potential(result.density)
    # PAW's one-centre coefficients come from ``becsum``, which is part of the
    # mixed state and not a function of the density -- which is why
    # ``SCFResult`` carries it and why a fixed-density run without it is refused
    # elsewhere.
    _, ddd_paw = calculation.onecenter(result.becsum)
    hamiltonians = calculation.hamiltonian(potential.v_scf, ddd_paw)
    return SternheimerSolver(
        calculation, hamiltonians, result.wavefunctions, eigenvalues, weights,
        nocc, threshold, v_scf=potential.v_scf, becsum=result.becsum,
        smearing=smearing_of(calculation, result),
        kpoint_weights=calculation.system.kpoints.weights,
    )
