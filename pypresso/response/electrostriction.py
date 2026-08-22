"""Electrostriction: ``d(chi)/d(strain)`` as one mixed third derivative.

``PLAN.md`` P26. Electrostriction is the quadratic electromechanical coupling
every dielectric has -- a strain proportional to the *square* of an applied
field -- and the four tensors that name it are, by a thermodynamic identity
(Tanner, Bousquet and Janolin, `arXiv:2012.03841
<https://arxiv.org/abs/2012.03841>`_, Eqs. 2), derivatives of the dielectric
susceptibility with respect to a mechanical variable::

    eps0 d(chi_ij)/d(X_kl) =  2 M_ijkl        (1/eps0) d(eta_ij)/d(X_kl) = -2 Q_ijkl
    eps0 d(chi_ij)/d(x_kl) = -2 m_ijkl        (1/eps0) d(eta_ij)/d(x_kl) =  2 q_ijkl

with ``x`` the strain, ``X`` the stress and ``eta = chi^-1`` the dielectric
stiffness. That is the route this module takes, and taking it rather than
applying a finite field is not a preference: a finite field puts a ceiling on
the k-point density that depends on the band gap, entangles electrostriction
with non-linear piezoelectricity in a non-centrosymmetric crystal, and needs a
relaxation under constrained ``E`` or ``D``. A derivative of ``chi`` needs none
of that.

**The derivative is taken, not differenced.** ``chi`` is already a second
derivative of the energy with respect to a field, so ``d(chi)/dx`` is a *third*
derivative -- and the point of this phase is that it is one ``jvp`` of code that
exists rather than a sweep of re-converged calculations. What makes that
possible is the 2n+1 theorem in the only form it takes here: **the second-order
energy is stationary in the first-order wavefunctions**, so it may be
differentiated with them held fixed. It is P15's envelope argument one order up,
and it is the same sentence P25 makes about the force.

Write the second-order energy as a functional of the field response ``u_i``,
the ground state, the density and the position operator::

    F_ij[x; psi, rho, b, u] = sum_kn w [ <u_i|H(x)|u_j> - Lambda_mn <u_i,m|u_j,n>
                                        + 2 Re <u_i|P_c b_j> ]
                            + (1/2) int drho_i K(x) drho_j

whose stationary point in ``u`` is exactly the self-consistent Sternheimer
solution P24 already computes, and whose stationary *value* is
``sum w Re <b_i|u_j>`` -- the expression ``dielec.f90`` assembles. Then

    d(eps_ij)/dx = jvp( F_ij )( x, psi, rho, b ; e_x, dpsi/dx, drho/dx, db/dx )

with **no tangent for** ``u``. The three tangents that are needed come from
:mod:`pypresso.response.strain` (the ground state's own response to a strain)
and from one further Sternheimer solve for ``db/dx`` below.

**The trap of the phase is that ``u`` is frozen and the space it lives in is
not.** The Sternheimer solution is *constrained* to be orthogonal to the
occupied manifold, and that manifold moves with the strain, so the variable of
the functional is ``P_c(psi) u`` and never the stored array. Writing ``u``
changes no value -- ``P_c u = u`` at the point everything is evaluated -- and it
destroys the stationarity the whole construction rests on: varying an
unrestricted ``u`` gives ``A u + P_c b + (K drho) psi = 0`` where the
self-consistent loop solves the same equation with the screening term projected
as well, and the two differ by the occupied component of ``(K drho) psi``. The
envelope theorem's hypothesis is then false, and the error is **2%** of
``d(eps)/dx`` on silicon -- large, systematic, and invisible to every check that
does not difference the functional itself at a *re-converged* strained cell. It
survived the value identity against ``dielec.f90``, the cubic form of the rank-4
tensor, and a finite-difference check of each of the four tangents separately;
what found it was splitting the disagreement in two, ``jvp`` against a
difference of ``F`` at frozen ``u``, and that difference against the true
``epsilon``. The first pair agreed to 1e-4 and the second did not.

**Four more things had to be got right and each is silent too.**

*The multiplier is a matrix, not the eigenvalues.* Writing ``eps_n`` where
``Lambda_mn = <psi_m|H|psi_n>`` stands would be exact in value and wrong in
derivative, twice over: it would need ``d(eps_n)/dx`` supplied as a separate
tangent, and inside a degenerate multiplet -- which a crystal has everywhere --
the diagonal element is basis-dependent where the matrix is not (rule D4). In
the ``H - Lambda`` form both come from the same ``jvp`` and neither is an input.

*The ``G = 0`` ambiguity has to cancel, and it does so only in that form.* The
mean electrostatic potential of a periodic solid is undefined, so ``dH/dx``
carries an arbitrary constant (:mod:`pypresso.response.strain`). It enters
``<u_i|dH|u_j>`` and ``dLambda <u_i|u_j>`` with opposite signs and cancels
exactly -- which is a *check*, not a hope: the answer must not move when the
constant does.

*The level shift and the projector are not the same kind of term.* ``alpha Q``
is dropped from ``F`` altogether: at the solution ``u`` is orthogonal to the
occupied manifold, so both its value and its derivative vanish through a factor
``<psi|u> = 0``. ``P_c`` in the source term is **not** droppable for the same
reason -- ``<u|P_c b>`` equals ``<u|b>`` in value, and their derivatives differ
by ``<u|dpsi_m><psi_m|b>``, which is first order and does not vanish.

*A strain response is not a wedge sum.* This module refuses a symmetry-reduced
k-set by name. The field response inside ``F`` is symmetrised as a polar vector
(P24) and the ground state's strain response as a rank-2 tensor
(:mod:`pypresso.response.strain`), but the *mixed* object being differentiated
here would need the two labels averaged together -- a rank-3 symmetriser that is
not written. An **unshifted** Monkhorst-Pack grid is closed under the point
group and needs none of it, which is the route P24 already uses as its
independent check.

**Clamped-ion, and refused by name otherwise.** What is computed is the
electronic (clamped-ion) susceptibility's strain derivative. The relaxed-ion
coefficients that experiment measures add the lattice's own contribution,
``chi_ion ~ (1/Omega) sum_m (Z* e_m)^2 / omega_m^2``, whose strain derivative
needs ``dZ*/dx`` and ``d(omega^2)/dx`` -- two more third derivatives of the same
family, each again a ``jvp`` of an assembly that exists (P24's ``Z*`` and P25's
force constants) along the strain tangent this module already builds. They are a
phase, not a function, and until they are here the coefficients reported are the
clamped-ion ones and say so.

Norm-conserving, ``nspin = 1``, insulators: everything
:mod:`pypresso.response.strain` and
:func:`~pypresso.response.sternheimer.require_a_sternheimer_regime` refuse is
refused here too.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from pypresso.basis.interpolate import to_dense
from pypresso.response.elastic import elastic_constants
from pypresso.response.efield import (
    _solve_stored,
    dielectric_tensor,
    require_a_symmetrisable_response,
)
from pypresso.response.phonon import require_norm_conserving
from pypresso.response.sternheimer import require_a_sternheimer_regime
from pypresso.response.strain import StrainResponse, strain_response, strain_tangent
from pypresso.response.velocity import VelocityOperator, over_kpoints
from pypresso.scf.density import sum_band
from pypresso.units import EPSILON0_SI, FPI

__all__ = [
    "Electrostriction",
    "electrostriction",
    "require_converged_responses",
    "refined_states",
    "second_order_energy",
    "susceptibility_strain_derivative",
]

#: The six independent components of a symmetric strain, in Voigt order.
VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))

#: ``ethr`` for the re-diagonalisation :func:`refined_states` performs. It has
#: to be far tighter than the SCF's own, for the reason that function documents.
REFINE_ETHR = 1.0e-13


@dataclass
class Electrostriction:
    """``d(chi)/dx`` and the electrostriction tensors that follow from it."""

    #: ``(3, 3)`` -- the clamped-ion electronic susceptibility ``chi``, in SI
    #: (dimensionless, ``eps_r = 1 + chi``).
    susceptibility: np.ndarray
    #: ``(3, 3, 3, 3)`` -- ``d(chi_ij)/dx_kl`` in the **tensor** strain
    #: convention: ``delta chi_ij = sum_kl (dchi/dx)_ijkl x_kl`` over all nine
    #: pairs, with no factors of two anywhere. Voigt's engineering shear is
    #: applied only by :attr:`m_voigt` and :attr:`q_voigt`.
    dchi_dstrain: np.ndarray
    #: ``(3, 3, 3, 3)`` -- ``m_ijkl = -(eps0/2) d(chi_kl)/dx_ij``, in
    #: **pN/V^2**, the unit Tanner et al. tabulate. **The first pair is the
    #: stress index and the second the field one**, which is the order
    #: ``X_ij = m_ijkl E_k E_l`` defines and is the *transpose* of the order the
    #: paper's Eq. (2) is printed in. The two agree for a cubic crystal, where
    #: ``m_1122 = m_2211``, and not in general; the derivation settles it --
    #: ``X_ij = df/dx_ij`` differentiates the strain index.
    m: np.ndarray
    #: ``(3, 3, 3, 3)`` -- ``q_ijkl = (1/2 eps0) d(eta_kl)/dx_ij`` with
    #: ``eta = chi^-1``, in **GN m^2/C^2**; same index order as ``m``.
    q: np.ndarray
    #: ``(3, 3)`` -- the dielectric tensor the derivative was taken at.
    epsilon: np.ndarray
    #: ``(3, 3, 3, 3)`` -- the **elasto-optic (photoelastic) tensor**
    #: ``p_ijkl``, defined by ``delta(eps^-1)_ij = p_ijkl x_kl``, hence
    #: ``p = -eps^-1 (d eps/dx) eps^-1``: pure algebra on ``dchi_dstrain``, and
    #: the one quantity in this module with a *measured* value to sit against
    #: (silicon: ``p_11 = -0.094``, ``p_12 = +0.017``, ``p_44 = -0.051``,
    #: Biegelsen, `PRL 32, 1196 (1974)
    #: <https://doi.org/10.1103/PhysRevLett.32.1196>`_). Dimensionless.
    #:
    #: **Clamped-ion, and for ``p_11`` and ``p_12`` that is the whole answer**:
    #: in the diamond structure no internal displacement is compatible with a
    #: tetragonal strain, so those two are directly comparable with experiment.
    #: ``p_44`` carries a Kleinman internal-displacement term that is *not*
    #: computed here, exactly as ``C_44`` does.
    photoelastic: np.ndarray | None = None
    #: ``(3, 3, 3, 3)`` -- ``M_ijkl``, the *strain* per field squared, in
    #: **pm^2/V^2**. ``M = -S : m`` with ``S`` the elastic compliance: at zero
    #: total stress ``C : x + m E^2 = 0``. ``None`` when the elastic constants
    #: were not asked for.
    M: np.ndarray | None = None
    #: ``(3, 3, 3, 3)`` -- ``Q_ijkl``, the strain per polarization squared, in
    #: **m^4/C^2**, from ``Q = -S : q``.
    Q: np.ndarray | None = None
    #: The clamped-ion elastic constants ``M`` and ``Q`` were built from.
    elastic: object | None = None
    #: The ground state's strain response, carried so that its convergence and
    #: its own checks are available to the caller.
    strain: StrainResponse | None = None
    #: ``(3, 3, 3, 3)`` -- ``d(eps_ij)/dx_kl``, before the ``4 pi``.
    depsilon_dstrain: np.ndarray | None = None

    @property
    def m_voigt(self) -> np.ndarray:
        """``m`` as ``(6, 6)``, with Voigt's factor of two on the shears."""
        return _to_voigt(self.m)

    @property
    def q_voigt(self) -> np.ndarray:
        return _to_voigt(self.q)

    @property
    def photoelastic_voigt(self) -> np.ndarray:
        """``p`` as ``(6, 6)`` -- ``p_11 = p[0,0]``, ``p_12``, ``p_44 = p[3,3]``."""
        return _to_voigt(self.photoelastic)

    @property
    def M_voigt(self) -> np.ndarray:
        return None if self.M is None else _to_voigt(self.M)

    @property
    def Q_voigt(self) -> np.ndarray:
        return None if self.Q is None else _to_voigt(self.Q)

    @property
    def hydrostatic(self) -> tuple[float, float]:
        """``(m_h, q_h) = X_11 + 2 X_12`` -- what a hydrostatic strain gives.

        The combination experiment quotes for a cubic crystal, and the one
        Tanner et al. obtain in a single calculation by straining
        hydrostatically rather than axially.
        """
        m, q = _to_voigt(self.m), _to_voigt(self.q)
        return (float(m[0, 0] + 2 * m[0, 1]), float(q[0, 0] + 2 * q[0, 1]))

    @property
    def hydrostatic_Mq(self) -> tuple[float, float] | None:
        """``(M_h, Q_h) = X_11 + 2 X_12``, the pair experiment quotes."""
        if self.M is None:
            return None
        big_m, big_q = _to_voigt(self.M), _to_voigt(self.Q)
        return (float(big_m[0, 0] + 2 * big_m[0, 1]),
                float(big_q[0, 0] + 2 * big_q[0, 1]))


def refined_states(calculation, result, ethr: float = REFINE_ETHR):
    """Re-diagonalise at the converged density -- ``(eigenvalues, psi)``.

    **This is not a tidying step and leaving it out is silent.** The
    wavefunctions an SCF returns are eigenvectors of the Hamiltonian built from
    the *input* density of its last iteration, not of ``H[rho_out]``; the two
    differ by the mixing step, so ``<psi_m|H|psi_n>`` is diagonal only to the
    accuracy of that difference -- 1.6e-7 Ry on two-atom silicon converged to
    ``conv_thr = 1e-12``.

    A first derivative never notices. A **third** one does, because the
    quantity that multiplies the error is ``<u|u>``, the norm of a first-order
    wavefunction, which is of order 10^3 here: the variational identity
    ``F_ij = sum w Re<b_i|u_j>`` then fails by 7e-7 *relative*, and the failure
    is a systematic offset rather than noise -- it does not shrink when the
    response's own thresholds are tightened, which is how it was found.

    Re-diagonalising to ``ethr = 1e-13`` takes it to 3.5e-15 and the identity to
    9e-10 relative. It costs one Davidson pass from an excellent starting guess.
    """
    density = jnp.asarray(result.density)
    nbnd = result.wavefunctions.shape[2]
    v_scf = calculation.potential(density).v_scf
    _, ddd_paw = calculation.onecenter(result.becsum)
    eigenvalues, psi = calculation.diagonalize(
        calculation.hamiltonian(v_scf, ddd_paw), nbnd,
        result.wavefunctions, ethr,
    )
    return jnp.asarray(eigenvalues), psi


# -- the variational second-order energy -------------------------------------


def _second_order_energy_at(moved, psi, rho, b, u, weights):
    """``F_ij`` on an already-strained calculation. See :func:`second_order_energy`."""
    batch = moved.k_batch
    hamiltonians = moved.hamiltonian(moved.potential(rho).v_scf, None)
    nspin = psi.shape[0]

    def apply_h(states):
        return jnp.stack([
            over_kpoints(hamiltonians[spin], states[spin], batch)
            for spin in range(nspin)
        ])

    # ``Lambda_mn = <psi_m|H|psi_n>``: the orthonormality multipliers as a
    # matrix. Diagonal and equal to the eigenvalues at the solution
    # (:func:`refined_states`), so the *value* of ``F`` is what it would be with
    # the eigenvalues written in; its derivative is not, and that is the point.
    lambdas = jnp.einsum("skmg,skng->skmn", jnp.conj(psi), apply_h(psi))

    # ``P_c b``. Its value equals ``b`` and so does its derivative once both
    # ``<psi|u> = 0`` and ``<psi|b> = 0``; it is written because that is the
    # functional, not because the two terms are large.
    overlaps = jnp.einsum("skmg,askng->askmn", jnp.conj(psi), b)
    pcb = b - jnp.einsum("askmn,skmg->askng", overlaps, psi)

    # **``u`` is frozen, but the subspace it is constrained to live in is not.**
    # The Sternheimer solution is required to be orthogonal to the occupied
    # manifold, and that manifold moves with the strain, so the variable of the
    # functional is ``P_c(psi) u`` and not the stored array. Writing ``u``
    # changes no value -- ``P_c u = u`` at the point everything is evaluated --
    # and it breaks the *stationarity*: varying an unrestricted ``u`` gives
    # ``A u + P_c b + K drho psi = 0`` where the loop solves the same equation
    # with the screening term projected too, and the two differ by the occupied
    # component of ``(K drho) psi``. That is the envelope theorem's hypothesis
    # failing, not a small term: it is worth **2%** of ``d(eps)/dx`` on silicon,
    # it is invisible at zeroth order, and it survives every check that does not
    # difference the functional itself at a *re-converged* strained cell.
    pcu = _project_conduction(psi, u)

    hu = jnp.stack([apply_h(pcu[axis]) for axis in range(3)])

    def raw_density(states):
        smooth, dense = moved.basis.smooth, moved.basis.dense
        density = sum_band(
            states, moved.fft_index, smooth.grid, weights,
            moved.system.cell, moved.k_batch,
        )
        return moved.augmented(to_dense(density, smooth, dense), ())

    drho = jnp.stack([
        jax.jvp(raw_density, (psi,), (pcu[axis],))[1] for axis in range(3)
    ])
    # ``dv_of_drho``, rebuilt at the **strained** cell: the Hartree kernel's
    # ``4 pi / G^2`` moves with it and ``f_xc`` moves with ``rho``, so both
    # belong inside the derivative rather than in a table computed once.
    kernel = jnp.stack([
        jax.jvp(lambda r: moved.potential(r).v_scf, (rho,), (drho[axis],))[1]
        for axis in range(3)
    ])
    rho = jnp.asarray(rho)
    measure = moved.system.cell.volume / rho.size * rho.shape[0]

    rows = []
    for i in range(3):
        row = []
        for j in range(3):
            band = jnp.sum(weights * jnp.real(
                jnp.einsum("skng,skng->skn", jnp.conj(pcu[i]), hu[j])
            ))
            # ``sum_mn Lambda_mn <u_i,n|u_j,m>`` -- and the index order is
            # the trap. Writing ``Lambda_mn <u_i,m|u_j,n>`` instead gives the
            # *same number* whenever ``Lambda`` is diagonal, so it passes the
            # identity against ``dielec.f90`` exactly; but it is
            # ``Tr(Lambda Ov^T)``, which is not invariant under the unitary
            # mixing a degenerate multiplet is defined only up to, where this is
            # ``Tr(Lambda Ov)``, which is. The symptom is a ``d(chi)/dx`` that
            # is not cubic on a cubic crystal -- 11% of the scale in components
            # the point group forbids -- and it is invisible at zeroth order.
            # The same order is what makes the stationary equation the standard
            # ``H u_n - sum_m u_m Lambda_mn = -P_c source_n``.
            multiplier = jnp.sum(weights * jnp.real(
                jnp.einsum("skmn,skng,skmg->skn", lambdas, jnp.conj(pcu[i]), pcu[j])
            ))
            source = jnp.sum(weights * jnp.real(
                jnp.einsum("skng,skng->skn", jnp.conj(pcu[i]), pcb[j])
                + jnp.einsum("skng,skng->skn", jnp.conj(pcu[j]), pcb[i])
            ))
            screening = 0.5 * measure * jnp.sum(drho[i] * kernel[j])
            row.append(band - multiplier + source + screening)
        rows.append(jnp.stack(row))
    return jnp.stack(rows)


def second_order_energy(calculation, strain, psi, rho, b, u, weights):
    """``F_ij``: the variational second-order energy of a uniform field, in Ry.

    A ``(3, 3)`` array, and a differentiable function of ``strain``, ``psi``,
    ``rho`` and ``b``. ``u`` is closed over rather than passed as a
    differentiable argument on purpose -- the functional is *stationary* in it,
    which is the whole of the 2n+1 theorem here (module docstring).

    At the stationary point its value is ``sum_kn w Re <b_i|u_j>``, which is
    what ``dielec.f90`` assembles, and that identity is a committed test.

    Args:
        psi: ``(nspin, nk, nocc, npwx)`` -- occupied bands only, and eigenvectors
            of the Hamiltonian built here (:func:`refined_states`).
        rho: the converged density on the dense grid.
        b: ``(3, nspin, nk, nocc, npwx)`` -- ``P_c r_a|psi>``, P24's ``bare``.
        u: ``(3, nspin, nk, nocc, npwx)`` -- the self-consistent field response.
        weights: ``(nspin, nk, nocc)`` -- ``wg`` for the occupied bands.
    """
    return _second_order_energy_at(
        calculation.at_strain(strain), psi, rho, b, u, weights
    )


def _epsilon_at(calculation, strain, psi, rho, b, u, weights):
    """``eps_ij = delta_ij - 16 pi F_ij / Omega`` at a strain -- ``dielec.f90``.

    The volume is the **strained** one, which is why this is not a constant
    times :func:`second_order_energy`: a hydrostatic strain changes ``eps``
    through ``Omega`` as well as through ``F``.
    """
    moved = calculation.at_strain(strain)
    energies = _second_order_energy_at(moved, psi, rho, b, u, weights)
    return jnp.eye(3) - 4.0 * FPI * energies / moved.system.cell.volume


# -- the position operator's own strain derivative ---------------------------


def _position_response(calculation, solver, rho, b, tangent, dpsi, drho):
    """``P_c db/dx``: one further Sternheimer solve per cartesian direction.

    ``b_a = P_c r_a|psi>`` is not written down anywhere -- it is *defined* by a
    linear equation, ``(H - eps_n) b_a = P_c^+ c_a`` with
    ``c_a = -i dH/dk_a |psi>`` (``dvpsi_e``) -- so unlike every other argument of
    ``F`` it has no closed form to differentiate. Differentiating its equation
    instead gives one of the same shape::

        (H - eps_n) d(b_a) = d[ P_c^+ c_a - (H - eps_n) b_a ]   at frozen b_a

    where the right-hand side is a ``jvp`` of an explicit expression and the
    operator is the one the solver already builds. Two things make it exact
    rather than nearly so:

    * **only the conduction part is wanted.** ``b_a(x)`` is orthogonal to the
      occupied manifold at every strain, so ``db_a`` has an occupied component
      (``<psi|db> = -<dpsi|b>``); it is annihilated in ``F``, where ``db``
      appears only as ``<u_i|P_c db_j>`` and ``u`` is orthogonal to that
      manifold. Solving the projected equation is therefore not an
      approximation.
    * **the level shift can be left out of the differentiated operator.**
      ``alpha Q = alpha S P_occ S`` contributes ``alpha sum_m |psi_m><dpsi_m|b>``,
      which is *purely occupied* and is removed by the same projection.

    **And the operator is written with the multiplier matrix, not ``eps_n``** --
    ``H b_n - sum_m b_m Lambda_mn`` where ``Lambda_mn = <psi_m|H|psi_n>``. This
    is :func:`second_order_energy`'s rule in a second place and it is *not*
    cosmetic here either: the first-order eigenvalue ``d(eps_n)`` is
    basis-dependent inside a degenerate multiplet, which silicon's valence bands
    are at most k-points, and the parallel-transport gauge the Sternheimer
    solutions live in is not the one that diagonalises the perturbation. Using
    the scalar gives a ``d(chi)/dx`` that is **not cubic on a cubic crystal** --
    20% of the diagonal appears in components the point group forbids, which is
    how this was found. In the matrix form the off-diagonal ``dLambda_mn b_m``
    terms are there and the tensor comes out cubic to 1e-4.

    ``c_a`` is the direction-``a`` cartesian velocity at the strained cell, and
    the strained ``kcart`` is the trap there: ``KPoints.coords`` do not move
    under a strain, so the operator is built on the ``kcart``
    :meth:`~pypresso.scf.driver.Calculation.at_strain` recorded.
    """
    calculation_zero = jnp.zeros((3, 3))
    psi = solver.psi
    frozen_b = b
    directions = np.eye(3)

    def residual(strain, states, density):
        moved = calculation.at_strain(strain)
        v_scf = moved.potential(density).v_scf
        velocity = VelocityOperator(moved, v_scf, None)
        hamiltonians = moved.hamiltonian(v_scf, None)
        batch = moved.k_batch

        def apply_h(block):
            return jnp.stack([
                over_kpoints(hamiltonians[spin], block[spin], batch)
                for spin in range(block.shape[0])
            ])

        lambdas = jnp.einsum("skmg,skng->skmn", jnp.conj(states), apply_h(states))

        out = []
        for axis in range(3):
            commutator = -1j * velocity.apply(states, directions[axis])
            overlaps = jnp.einsum("skmg,skng->skmn", jnp.conj(states), commutator)
            projected = commutator - jnp.einsum(
                "skmn,skmg->skng", overlaps, states
            )
            applied = apply_h(frozen_b[axis]) - jnp.einsum(
                "skmn,skmg->skng", lambdas, frozen_b[axis]
            )
            out.append(projected - applied)
        return jnp.stack(out)

    _, rhs = jax.jvp(
        residual, (calculation_zero, psi, rho), (tangent, dpsi, drho)
    )
    # One solve per cartesian direction: ``_solve_stored`` takes a right-hand
    # side that is already an array and applies ``orthogonalize``'s sign, which
    # is the projection onto the conduction space the derivation above needs.
    return jnp.stack([_solve_stored(solver, rhs[axis]) for axis in range(3)])


# -- the third derivative ----------------------------------------------------


def susceptibility_strain_derivative(
    calculation, solver, rho, b, u, response: StrainResponse, verbose: bool = False
):
    """``d(eps_ij)/dx_kl``: one ``jvp`` of ``F`` per independent strain.

    Returns ``(3, 3, 3, 3)``, symmetric in its last two indices by construction
    -- the six Voigt strains are computed and mirrored.
    """
    psi = solver.psi
    weights = solver.weights
    zero = jnp.zeros((3, 3))
    rho = jnp.asarray(rho)

    def epsilon(strain, states, density, position):
        return _epsilon_at(
            calculation, strain, states, density, position, u, weights
        )

    out = np.zeros((3, 3, 3, 3))
    for (k, l) in VOIGT:
        tangent = strain_tangent(k, l)
        dpsi = jnp.asarray(response.dpsi[k, l])
        drho = jnp.asarray(response.drho[k, l])
        db = _position_response(calculation, solver, rho, b, tangent, dpsi, drho)
        _, column = jax.jvp(
            epsilon, (zero, psi, rho, b), (tangent, dpsi, drho, db)
        )
        out[:, :, k, l] = out[:, :, l, k] = np.asarray(column)
        if verbose:
            print(f"  d(eps)/dx_{k}{l}: trace/3 = {np.trace(column) / 3:.6f}")
    return out


def require_converged_responses(field, strain) -> None:
    """Refuse to build a third derivative on a first-order solution that diverged.

    **This is not defensive programming; it is the lesson of running the phase
    on a slab.** Bilayer graphene's strain response diverges at QE's default
    ``alpha_mix = 0.7`` -- ``|ddv_scf|^2`` grows by 1.34 per iteration, from
    1.7e7 to 8.9e9 in twenty-five -- and the loop then simply runs out of
    iterations and returns what it has. Everything downstream consumed that
    without complaint and produced an elastic tensor that was not even symmetric
    under ``C_ijkl = C_klij``: 49817 GPa against -243233 for the same pair of
    indices, on a crystal whose stiffest constant is 859. Nothing in the numbers
    said "unconverged"; only the identity did.

    Why a slab and not silicon: the induced Hartree potential is ``4 pi e^2/G^2``
    against the induced charge, and a cell with 14 bohr of vacuum has its
    smallest nonzero ``G_z`` at ``2 pi/c``, where that kernel is two orders
    larger than anything a compact cell reaches. Simple linear mixing of a map
    whose Jacobian has an eigenvalue that large is unstable for
    ``alpha_mix`` above roughly ``2/(1 + |lambda|)``. Measured on this cell:
    0.7 diverges at 1.34 per iteration, **0.3 converges at 0.5 per iteration**
    and reaches ``tr2 = 1e-14`` in 68. That is the dial to turn, and
    it is the same stiffness the ground-state SCF meets on a slab and answers
    with Kerker preconditioning (`PERFORMANCE.md`); the response loop has no
    preconditioner of its own, which is why the mixing parameter is the whole of
    the remedy here.
    """
    for name, response, dial in (
        ("the electric-field response", field, "alpha_mix"),
        ("the strain response", strain, "alpha_mix"),
    ):
        if response is None or response.converged:
            continue
        last = response.history[-1] if response.history else float("nan")
        raise ValueError(
            f"{name} did not converge: |ddv_scf|^2 = {last:.3e} after "
            f"{len(response.history)} iterations, against the requested tr2. A "
            "third derivative built on it is meaningless -- and silently so, "
            "since the answer it produces looks like a tensor. Lower "
            f"`{dial}` (QE's 0.7 diverges on a slab, 0.3 converges) and raise "
            "`max_iterations`, or pass allow_unconverged=True if this is a "
            "diagnostic run"
        )


def electrostriction(
    calculation,
    result,
    strain: StrainResponse | None = None,
    elastic: bool = True,
    verbose: bool = False,
    allow_unconverged: bool = False,
    **response_options,
) -> Electrostriction:
    """The clamped-ion electrostriction tensors of a converged insulator.

    Args:
        calculation: the :class:`~pypresso.scf.driver.Calculation` the run used.
            A symmetry-reduced k-set is **refused**: see the module docstring
            for the rank-3 symmetriser that would be needed.
        result: the converged :class:`~pypresso.scf.driver.SCFResult`. Its
            states are re-diagonalised first (:func:`refined_states`).
        strain: a strain response computed earlier, if there is one. Recomputed
            otherwise; it is the expensive half.
        elastic: also compute the clamped-ion elastic constants, and with them
            ``M`` and ``Q`` -- the coefficients that give a *strain* rather than
            a stress, which is what experiment quotes. They cost six more
            ``jvp``s of the stress and reuse the same strain response
            (:mod:`pypresso.response.elastic`).
        allow_unconverged: return an answer even when one of the two
            self-consistent responses under this one did not converge. Off, and
            it is off because the alternative is what
            :func:`require_converged_responses` documents.
        response_options: passed to
            :func:`~pypresso.response.efield.dielectric_tensor` and
            :func:`~pypresso.response.strain.strain_response`. ``alpha_mix`` is
            the one to reach for on a **slab**: QE's default of 0.7 diverges
            where a bulk crystal converges (see
            :func:`require_converged_responses`).
    """
    require_a_symmetrisable_response(calculation)
    require_a_sternheimer_regime(calculation)
    require_norm_conserving(calculation)
    _require_a_closed_grid(calculation)

    eigenvalues, psi = refined_states(calculation, result)
    density = jnp.asarray(result.density)

    field = dielectric_tensor(
        calculation, psi, eigenvalues, density,
        born_charges=False, keep_internals=True, verbose=verbose,
        **response_options,
    )
    internals = field.internals
    solver = internals["solver"]
    # **Projected onto the conduction manifold before anything else.** Both are
    # orthogonal to the occupied states *by definition* -- the Sternheimer
    # right-hand side is projected and the operator preserves the split -- but
    # a few hundred CG steps without reorthogonalisation leave ``<psi|b>`` at
    # about 1e-6 of ``|b|``. Nothing at first order notices; here the leakage
    # multiplies ``<u|dpsi>``, and it is cheaper to remove it than to argue
    # about how large the product is.
    b = _project_conduction(solver.psi, jnp.stack(internals["bare"]))
    u = _project_conduction(solver.psi, jnp.stack(internals["dpsi"]))

    if strain is None:
        strain = strain_response(
            calculation, psi, eigenvalues, density, verbose=verbose,
            **response_options,
        )

    if not allow_unconverged:
        require_converged_responses(field, strain)

    depsilon = susceptibility_strain_derivative(
        calculation, solver, density, b, u, strain, verbose=verbose
    )

    # Gaussian ``eps = 1 + 4 pi chi_G`` to SI ``eps_r = 1 + chi``: the
    # susceptibility Tanner et al.'s equations are written in is the SI one, and
    # the conversion happens here, at the boundary, and once. ``d(eps)/dx`` and
    # ``d(chi_SI)/dx`` are then the *same* array, since the two differ by an
    # additive constant.
    epsilon = np.asarray(field.epsilon)
    chi = epsilon - np.eye(3)
    dchi = depsilon

    # ``eta = chi^-1``, so ``d(eta) = -eta d(chi) eta`` -- pure algebra, and the
    # reason ``q`` costs nothing beyond ``m``.
    eta = np.linalg.inv(chi)
    deta = -np.einsum("ia,abkl,bj->ijkl", eta, dchi, eta)

    # The index order: ``X_ij = m_ijkl E_k E_l`` puts the **strain** pair first,
    # and ``dchi`` has the field pair first, so the pairs are swapped here.
    m = -0.5 * EPSILON0_SI * dchi.transpose(2, 3, 0, 1) * 1.0e12
    q = 0.5 / EPSILON0_SI * deta.transpose(2, 3, 0, 1) * 1.0e-9

    # The elasto-optic tensor: the same inversion as ``eta``, one level up --
    # on ``eps`` rather than on ``chi``. It costs nothing and it is the only
    # thing here that a laboratory has measured directly.
    inverse = np.linalg.inv(epsilon)
    photoelastic = -np.einsum("ia,abkl,bj->ijkl", inverse, depsilon, inverse)

    constants = big_m = big_q = None
    if elastic:
        constants = elastic_constants(
            calculation, psi, eigenvalues, density, strain,
            allow_unconverged=allow_unconverged,
        )
        # ``C : x + m E^2 = 0`` at zero total stress, so ``M = -S : m`` -- and
        # the same relation with ``q`` gives ``Q``, because the polarization
        # enters the elastic Gibbs function exactly where the field enters the
        # free energy. Done in Voigt, where the compliance is an ordinary
        # matrix inverse (``sigma_I = C_IJ e_J`` with the engineering shear).
        compliance = _compliance_tensor(constants.compliance / 1.0e9)  # 1/Pa
        big_m = -np.einsum(
            "ijmn,mnkl->ijkl", compliance, m * 1.0e-12
        ) * 1.0e24                                     # m^2/V^2 -> pm^2/V^2
        big_q = -np.einsum("ijmn,mnkl->ijkl", compliance, q * 1.0e9)

    return Electrostriction(
        susceptibility=chi,
        dchi_dstrain=dchi,
        m=m,
        q=q,
        epsilon=epsilon,
        photoelastic=photoelastic,
        M=big_m,
        Q=big_q,
        elastic=constants,
        strain=strain,
        depsilon_dstrain=depsilon,
    )


def _project_conduction(psi, block):
    """``P_c`` applied to a ``(3, nspin, nk, nocc, npwx)`` block. ``S = 1``."""
    overlaps = jnp.einsum("skmg,askng->askmn", jnp.conj(psi), block)
    return block - jnp.einsum("askmn,skmg->askng", overlaps, psi)


def _compliance_tensor(voigt: np.ndarray) -> np.ndarray:
    """``S_ijkl`` from the ``(6, 6)`` compliance -- Voigt's halves, undone.

    ``S`` in Voigt form relates *engineering* strains to stresses
    (``e_I = S_IJ sigma_J`` with ``e_4 = 2 x_23``), so going back to a tensor
    that contracts as ``x_ij = S_ijkl X_kl`` over all nine pairs divides by two
    for each shear index: 1, 1/2 or 1/4 according to how many of ``I``, ``J``
    are 4, 5 or 6. It is written out rather than done in Voigt because *every*
    factor of two in this module lives here, in one place, instead of being
    spread over four tensors with different index meanings.
    """
    out = np.zeros((3, 3, 3, 3))
    for row, (i, j) in enumerate(VOIGT):
        for col, (k, l) in enumerate(VOIGT):
            value = voigt[row, col]
            value /= 2.0 if row >= 3 else 1.0
            value /= 2.0 if col >= 3 else 1.0
            for (a, b) in {(i, j), (j, i)}:
                for (c, d) in {(k, l), (l, k)}:
                    out[a, b, c, d] = value
    return out


def _to_voigt(tensor: np.ndarray) -> np.ndarray:
    """A ``(3,3,3,3)`` tensor as ``(6, 6)`` -- a plain relabelling of the pairs.

    **No factors of two.** They are tempting and wrong here: for
    ``delta chi_ij = sum_kl G_ijkl x_kl`` summed over all nine pairs, the two
    off-diagonal terms combine into ``G_23 e_4`` with ``e_4 = 2 x_23``, so the
    Voigt coefficient *is* the tensor component. What Voigt's doubling belongs
    to is the compliance (:func:`_compliance_tensor`) and to a strain-valued
    *output* row, and neither is done in this form.
    """
    out = np.zeros((6, 6))
    for row, (i, j) in enumerate(VOIGT):
        for col, (k, l) in enumerate(VOIGT):
            out[row, col] = tensor[i, j, k, l]
    return out


def _require_a_closed_grid(calculation) -> None:
    """Refuse a symmetry-reduced k-set, which is what P26 has no average for."""
    # The condition is the one the symmetrisers themselves test -- ``nosym``
    # leaves ``symmetries`` in place (the group is wanted for other things) and
    # sets the density maps to ``None``, which is what says no average happens.
    if getattr(calculation, "_symmetry_maps", None) is not None:
        raise NotImplementedError(
            "electrostriction on a symmetry-reduced k-set is not implemented: "
            "the object being differentiated carries a field label and a strain "
            "label at once, so completing the wedge sum needs a rank-3 average "
            "(R_ai R_bk R_cl) that is not written. Run the whole grid instead -- "
            "an **unshifted** Monkhorst-Pack grid with nosym is closed under the "
            "point group and needs no average at all, which is the route P24 "
            "already uses as its independent check"
        )
