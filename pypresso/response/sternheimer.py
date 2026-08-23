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
:mod:`pypresso.solvers.davidson`: a converged band stays in the block and takes a
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

**Refused rather than approximated**, each by name:

* **metals.** ``orthogonalize``'s smearing branch replaces the sharp projector
  with the occupation-difference weights ``wwg``, and the Fermi level itself
  shifts (``ef_shift``). The insulator projector applied to a metal is silently
  wrong, so a smeared run raises here. (The Fermi-level term itself already
  exists -- P22 wrote ``bisect_fermi``'s ``custom_jvp`` -- so this is the
  projector's gap, not the level's.)
* **noncollinear magnetism and spin-orbit coupling**, whose ``incdrhoscf_nc``
  and ``set_int3_nc`` are a second implementation rather than a spin axis on
  this one.
* **DFT+U**, whose induced potential has a ``dns`` of its own
  (``adddvhubscf.f90``) that is not a function of ``drho``.

**Memory.** The CG carries four band-blocks -- the gradient, its
preconditioning, the search direction and the previous one -- at
``(nk, nbnd_occ, ndim)`` complex each if the whole k axis is in flight, or one
k-point's worth at ``k_batch = 1``. That is the same working set the Davidson
subspace has at ``nvecx = 4 nbnd``, and it goes through the same dial
(:mod:`pypresso.batching`).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import lax

import jax

from pypresso.basis.fft import g_to_r
from pypresso.basis.interpolate import to_dense, to_smooth
from pypresso.batching import map_k
from pypresso.scf.density import becsum as becsum_of, sum_band
from pypresso.scf.potential import as_potential_components

__all__ = [
    "SternheimerSolver",
    "SternheimerResult",
    "local_perturbation",
    "make_sternheimer",
    "paw_response",
    "require_a_sternheimer_regime",
]

#: ``cgsolve_all``'s own ceiling on the CG iterations.
MAX_ITERATIONS = 400

#: The default target for the preconditioned residual norm of one band. QE
#: schedules this against the self-consistency of the response (``dfpt_kernels``:
#: ``thresh = min(0.1 sqrt(dr2), 1e-2)``); a solver used on its own, as
#: ``chi_0``, wants it tight from the start.
THRESHOLD = 1.0e-11


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
        nocc: int,
        threshold: float = THRESHOLD,
        max_iterations: int = MAX_ITERATIONS,
        v_scf=None,
        becsum=(),
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
        self.psi = jnp.asarray(psi)[:, :, :nocc]
        self.eigenvalues = jnp.asarray(eigenvalues)[:, :, :nocc]
        self.weights = jnp.asarray(weights)[:, :, :nocc]
        self.nocc = int(nocc)
        self.threshold = float(threshold)
        self.max_iterations = int(max_iterations)
        self.alpha_pv = _alpha_pv(np.asarray(eigenvalues), nocc)

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
        overlaps = jnp.einsum("mg,ng->mn", jnp.conj(occupied), s)
        lifted = jnp.einsum("mn,mg->ng", overlaps, occupied)
        return out + self.alpha_pv * hamiltonian.apply_s(lifted, ik)

    def project(self, rhs, ik, spin):
        """``-P_c^+ rhs`` where ``P_c^+ = 1 - S |psi_occ><psi_occ|``.

        ``orthogonalize.f90``'s insulator branch, **including its sign**: the
        routine returns minus the projected vector, because the right-hand side
        of the Sternheimer equation is ``-P_c^+ dV|psi>``.
        """
        hamiltonian = self.hamiltonians[spin]
        occupied = self.psi[spin][ik]
        overlaps = jnp.einsum("mg,ng->mn", jnp.conj(occupied), rhs)
        s_occupied = hamiltonian.apply_s(occupied, ik)
        return -(rhs - jnp.einsum("mn,mg->ng", overlaps, s_occupied))

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
        commutator (:mod:`pypresso.response.efield`).
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

    def density_at(self, states) -> jnp.ndarray:
        """``rho`` from a set of occupied states, as the SCF builds it.

        :meth:`~pypresso.scf.driver.Calculation.density` **without the
        symmetrisation**: a response is symmetrised as a vector, not as a scalar
        (:meth:`~pypresso.scf.driver.Calculation.symmetrize_directional`), so the
        caller does it. Everything else is the same function the SCF uses --
        including ``becsum`` and the augmentation charge, which is what makes
        the derivative below carry them.
        """
        calculation = self.calculation
        smooth, dense = calculation.basis.smooth, calculation.basis.dense
        becsum_ = self._raw_becsum(states)
        rho = sum_band(
            states, calculation.fft_index, smooth.grid, self.weights,
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
        ``0/0`` at a node (:func:`pypresso.scf.density.band_density`).

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

    def _raw_becsum(self, states) -> tuple:
        """``becsum`` **without** the symmetrisation :meth:`Calculation.becsum` applies.

        For PAW that method ends with ``PAW_symmetrize``, and a *response* must
        not go through it: the three response densities are symmetrised together
        as a polar vector afterwards
        (:meth:`~pypresso.scf.driver.Calculation.symmetrize_directional`), and
        pre-averaging each one as a scalar is the same wrong-symmetry mistake in
        a place where it is much harder to see. It is worth **1.6e-2** on the
        dielectric constant of PAW silicon, against the 5e-5 the rest of the
        machinery reaches.
        """
        calculation = self.calculation
        if not calculation.is_ultrasoft:
            return ()
        return becsum_of(
            states, calculation.projectors.vkb, self.weights,
            calculation.species_channels, calculation.k_batch,
        )

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
        is added there (:mod:`pypresso.response.efield`).
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
    ``int3`` is one ``jvp`` of :meth:`~pypresso.scf.driver.Calculation.
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
    :meth:`~pypresso.scf.driver.Calculation.onecenter` along ``dbecsum`` -- one
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


def _alpha_pv(eigenvalues, nocc: int) -> float:
    """``setup_alpha_pv``'s insulator value: ``2 (eps_max^occ - eps_min)``."""
    emin = float(np.min(eigenvalues))
    emax = float(np.max(eigenvalues[..., :nocc]))
    return max(2.0 * (emax - emin), 1.0e-2)


def require_a_sternheimer_regime(calculation) -> None:
    """Refuse, by name, every regime whose response needs machinery not here.

    A separate function because two entry points need the same list -- this
    module's :func:`make_sternheimer` and the electric field's
    :func:`~pypresso.response.efield.dielectric_tensor` -- and a refusal stated
    twice is a refusal that will eventually be stated differently. See the
    module docstring for what each case would need.
    """
    system = calculation.system
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
    if system.occupations != "fixed":
        raise NotImplementedError(
            f"occupations={system.occupations!r}: the Sternheimer response here "
            "is the insulator one. A metal needs orthogonalize's smearing "
            "branch, where the sharp projector becomes the occupation-difference "
            "weights, and the Fermi level shifts with the perturbation "
            "(ef_shift). Applying the insulator projector to a metal is "
            "silently wrong, so it is refused"
        )
    if calculation.spiral:
        raise NotImplementedError("the Sternheimer response of a spin spiral is not implemented")
    if calculation.nspin == 2:
        # ``SternheimerSolver`` takes *one* occupied-band count and slices
        # ``psi[:, :, :nocc]`` across the spin axis with it. Both callers derive
        # it as ``nelec / 2``, which is right for an unpolarized insulator and
        # wrong for a magnetic one -- its channels are occupied to different
        # depths, so ``P_c^+`` and ``alpha_pv`` would be built for the wrong
        # manifold in at least one of them, with no shape error and no failed
        # convergence to show for it.
        raise NotImplementedError(
            "the Sternheimer response is not implemented for nspin = 2: the "
            "occupied-band count here is one number for both spin channels "
            "(nelec/2), and a spin-polarized insulator's channels are occupied "
            "to different depths, so the response would be solved for the wrong "
            "bands in one of them without any sign of it"
        )


def make_sternheimer(calculation, result, threshold: float = THRESHOLD):
    """A solver for a converged :class:`~pypresso.scf.driver.SCFResult`."""
    require_a_sternheimer_regime(calculation)
    eigenvalues = jnp.asarray(result.eigenvalues)
    if eigenvalues.ndim == 2:
        eigenvalues = eigenvalues[None]
    weights, _ = calculation.occupations(eigenvalues)
    weights = jnp.asarray(weights)

    nocc = int(round(calculation.nelec / (1 if calculation.noncolin else 2)))
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
    )
