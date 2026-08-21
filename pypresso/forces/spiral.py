"""``dE/dq``: the gradient of the energy with respect to a spin spiral's wavevector.

A spin spiral's ground state depends on ``q`` exactly as it depends on where the
atoms are (P19, :mod:`pypresso.system.spiral`): ``q`` is a coordinate of the
calculation, ``E(q)`` is a surface over it, and the pitch a magnet actually
adopts is that surface's minimum. Finding it by scanning
(:func:`~pypresso.workflows.spiral.run_spiral_scan`) costs one SCF per grid
point of a three-dimensional space; finding it by following a gradient costs one
SCF per *step*, and the gradient is what this module produces.

**It is the same construction as the force, term for term.** The energy is
written down as a function of ``q`` at *frozen* wavefunctions, occupations and
eigenvalues, and the gradient is ``jax.grad`` of it -- no expression is derived
for any contribution, and there is no Fortran counterpart to transcribe, because
``pw.x`` has no spin spiral at all. What makes it correct rather than merely
convenient is the same stationarity argument :mod:`pypresso.forces.energy`
makes for the positions, and it needs one extra step here:

**The frozen quantity is the periodic part of the spinor, and that is the right
one.** The generalized Bloch theorem writes the state as

    Psi^q_k(r) = ( U_up(r) e^{i(k + q/2).r},  U_dn(r) e^{i(k - q/2).r} )

with ``U_up``, ``U_dn`` lattice periodic. The stored coefficients *are* those
periodic parts -- the plane-wave sphere carries the ``e^{i(k +- q/2).r}``
factor -- so holding the coefficients fixed while ``q`` moves holds ``U`` fixed
and lets the spiral turn, which is precisely the variational parameter the SCF
minimised over. The orthonormality constraint ``<U|U> = 1`` has no ``q`` in it
(``S`` is the identity: ultrasoft and PAW spirals are refused), so unlike the
ultrasoft force there is no Pulay term to carry, and the total derivative is the
partial one at frozen state.

**What actually depends on ``q``, and what visibly does not.** In the rotated
frame the density is lattice periodic and is built from the same coefficients on
the same FFT box, so at frozen state it *does not move with* ``q`` at all --
neither does the Hartree energy, the exchange-correlation energy, the local
pseudopotential term or the Ewald sum. The whole gradient comes from the two
places the shifted spheres appear: ``|k +- q/2 + G|^2`` and
``vkb(k +- q/2)``. The energy below is nonetheless written out in full, for two
reasons: evaluated at the converged geometry it must reproduce the SCF total
energy to round-off, which is the only check there is on the rest of it; and
when the augmentation charge between the two components is eventually threaded
through, the density *will* depend on ``q`` and the term will appear in the
gradient by itself rather than needing to be remembered.

**The plane-wave sphere is held fixed while differentiating, and that loses
nothing.** Which plane waves satisfy ``|k +- q/2 + G|^2 <= ecutwfc`` is a
host-side decision and cannot be traced -- but it is also *piecewise constant*
in ``q``, so on each piece the frozen-sphere derivative is the exact derivative.
What it does not see is the jump at the isolated wavevectors where a plane wave
crosses the cutoff, and that jump is the Pulay-like error of a finite basis: the
crossing coefficient is by construction the one at the cutoff, so it is of the
size of the basis-set incompleteness and it shrinks with ``ecutwfc``. It is the
floor on how tightly ``dE/dq`` can be driven to zero, and
:func:`~pypresso.workflows.spiral.relax_spiral_q` is what has to know that.

**A magnetic field or a constrained moment is refused rather than corrected.**
The field's own energy is deliberately outside the reported total
(:mod:`pypresso.scf.fields`), so the converged state minimises total *plus*
field while the number being differentiated is the total alone -- the
stationarity the whole method rests on does not hold, and the missing term is
invisible.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from pypresso.forces.energy import FrozenState, state_from_result
from pypresso.scf.potential import total_charge

__all__ = ["SpiralGradient", "spiral_energy", "compute_spiral_gradient"]


@dataclass
class SpiralGradient:
    """``dE/dq`` at one spiral wavevector, and the step direction it implies.

    Two coordinate systems, because the two are used for different things.
    :attr:`gradient` is with respect to ``q`` in **lattice** coordinates -- the
    units ``spiral_q`` is given in, so it says how the energy responds to
    turning the spiral by a fraction of a reciprocal lattice vector.
    :attr:`gradient_cartesian` is with respect to ``q`` in units of
    ``2 pi / alat``, which is the metric an optimizer has to measure a step in:
    a fractional coordinate means a different distance along each axis of a cell
    that is not cubic.
    """

    #: The wavevector it was evaluated at, in lattice coordinates.
    wavevector: np.ndarray
    #: ``(3,)`` ``dE/dq`` with ``q`` in lattice coordinates, Ry.
    gradient: np.ndarray
    #: ``(3,)`` ``dE/dq`` with ``q`` in units of ``2 pi / alat``, Ry.
    gradient_cartesian: np.ndarray
    #: The total energy the gradient was taken at, in Ry -- recomputed from the
    #: frozen state rather than copied from the SCF, so that comparing the two
    #: is the identity check on the functional being differentiated.
    total_energy: float

    @property
    def force(self) -> np.ndarray:
        """``-dE/dq`` in units of ``2 pi / alat``: the direction ``q`` moves in."""
        return -self.gradient_cartesian

    @property
    def max_gradient(self) -> float:
        """``max |dE/dq|`` in Ry per ``2 pi / alat`` -- what convergence tests."""
        return float(np.abs(self.gradient_cartesian).max())


def spiral_energy(calculation, q_crystal, state: FrozenState):
    """The total energy at spiral wavevector ``q_crystal``, state frozen.

    ``calculation`` supplies everything ``q`` does not decide; it is moved with
    :meth:`~pypresso.scf.driver.Calculation.at_spiral_q` at a *frozen basis*, so
    the result is a differentiable function of ``q``. A scalar in Ry.
    """
    _require_a_differentiable_spiral(calculation)

    moved = calculation.at_spiral_q(q_crystal, rebuild_basis=False)
    psi, weights = state.wavefunctions, state.weights

    # The density, and everything built from it. None of it moves with ``q`` at
    # frozen coefficients -- the rotated frame is what makes it lattice periodic
    # -- so ``grad`` finds zero here and never differentiates through an FFT or
    # the exchange-correlation functional. It is written out because the
    # functional has to *be* the total energy before it is differentiated.
    becsum_ = moved.becsum(psi, weights)
    rho = moved.density(psi, weights, becsum_)
    potential = moved.potential(rho)
    volume = calculation.system.cell.volume
    local = volume / rho[0].size * jnp.sum(moved.vltot * total_charge(rho))

    kinetic = _kinetic_energy(psi, moved.state_kinetic, weights)
    nonlocal_ = _nonlocal_energy(
        psi, moved.projectors.vkb, moved.dvan_so, weights, calculation.system.kpoints.nk
    )
    # ``S`` is the identity for the norm-conserving datasets a spiral is
    # restricted to, so the orthonormality constraint carries no ``q`` and
    # contributes nothing to the gradient. It is here for the same reason the
    # density is: the identity against the SCF total energy is the check.
    norm = jnp.sum(weights * state.eigenvalues * (_norms(psi) - 1.0))

    return (
        kinetic
        + nonlocal_
        + local
        + potential.ehart
        + potential.etxc
        + moved.ewald
        - norm
        + state.entropy
    )


def compute_spiral_gradient(calculation, result_or_state) -> SpiralGradient:
    """``dE/dq`` for ``calculation`` in its converged state.

    Args:
        calculation: the :class:`~pypresso.scf.driver.Calculation` the state
            belongs to -- its ``spiral_q`` is where the gradient is evaluated.
        result_or_state: an :class:`~pypresso.scf.driver.SCFResult`, or the
            :class:`~pypresso.forces.energy.FrozenState` taken from one.
    """
    _require_a_differentiable_spiral(calculation)
    state = (
        result_or_state
        if isinstance(result_or_state, FrozenState)
        else state_from_result(result_or_state)
    )

    q = jnp.asarray(calculation.system.spiral_q, dtype=float)
    energy, gradient = _energy_and_gradient(calculation)(q, state)

    cell = calculation.system.cell
    # ``q_cart = q_cryst @ B``, so ``dE/dq_cryst = B dE/dq_cart`` and the
    # cartesian gradient comes back through the same matrix. Doing it this way
    # rather than differentiating a cartesian parameterisation keeps one
    # definition of what ``q`` is -- the lattice coordinates ``spiral_q`` is
    # written in, and the only ones the rest of the code names.
    bg = np.asarray(cell.bg_2pi_alat)
    gradient = np.asarray(gradient, dtype=float)
    cartesian = np.linalg.solve(bg, gradient)

    return SpiralGradient(
        wavevector=np.asarray(calculation.system.spiral_q, dtype=float),
        gradient=gradient,
        gradient_cartesian=cartesian,
        total_energy=float(energy),
    )


def _energy_and_gradient(calculation):
    """``(E, dE/dq)`` in one pass, compiled once per plane-wave sphere.

    ``value_and_grad`` rather than ``grad``: the energy is wanted anyway -- it
    is what checks the functional against the SCF total -- and the forward pass
    is shared.

    The compiled function is cached on the calculation, and unlike the force's
    cache it is **dropped whenever the sphere changes**
    (:meth:`~pypresso.scf.driver.Calculation.at_spiral_q`). A moved ``q`` is a
    different set of plane waves and therefore a different ``npwx``, so the
    cache buys nothing across the steps of a relaxation and would be actively
    wrong if it were carried: the closure holds the sphere the gradient was
    compiled with. Each accepted step pays one compilation, which is small
    beside the SCF it goes with.
    """
    cached = calculation.__dict__.get("_spiral_gradient")
    if cached is None:
        cached = jax.jit(jax.value_and_grad(
            lambda q, state: spiral_energy(calculation, q, state)
        ))
        calculation._spiral_gradient = cached
    return cached


def _require_a_differentiable_spiral(calculation) -> None:
    """The three things that make ``dE/dq`` at frozen state not be the answer."""
    if not calculation.spiral:
        raise ValueError(
            "dE/dq needs a spin spiral: set spiral_q, which is the coordinate "
            "being differentiated with respect to"
        )
    if calculation.is_ultrasoft:
        # Unreachable through ``Calculation``, which refuses the combination
        # outright, and stated here because this is the term that would be
        # missing: the augmentation charge *between* the two components carries
        # its own ``q`` dependence.
        raise NotImplementedError(
            "dE/dq for an ultrasoft or PAW spiral is not implemented, for the "
            "same reason the spiral itself is not: q_ij between the two "
            "components is not threaded through"
        )
    if calculation.magnetic_field is not None:
        raise NotImplementedError(
            "dE/dq with a magnetic field or a constrained moment is not "
            "implemented: the field's energy is deliberately outside the "
            "reported total (see pypresso.scf.fields), so the converged state "
            "is stationary for a different functional than the one being "
            "differentiated and the missing term would be silent"
        )


@jax.jit
def _kinetic_energy(psi, kinetic, weights):
    """``sum w f <psi| |k +- q/2 + G|^2 |psi>``.

    ``kinetic`` is already in the layout a *spinor* is stored in -- the up
    component's row and the down component's row laid end to end -- so the two
    components meet their own sphere without the sum having to know there are
    two of them.
    """
    return jnp.sum(weights * jnp.einsum("skbg,kg->skb", jnp.abs(psi) ** 2, kinetic))


@jax.jit
def _norms(psi):
    return jnp.sum(jnp.abs(psi) ** 2, axis=-1)


@partial(jax.jit, static_argnames=("nk",))
def _nonlocal_energy(psi, vkb, dvan_so, weights, nk):
    """``sum w f <psi|V_NL|psi>`` with the projectors of both shifted spheres.

    ``add_vuspsi_nc``'s quadratic form: ``D`` is a 2x2 matrix in spin space
    (``dvan_so``, which for a scalar-relativistic dataset is the ordinary
    ``dion`` on each diagonal block), and the two spinor components are
    projected onto *different* projectors -- ``vkb(k + q/2)`` and
    ``vkb(k - q/2)`` -- which is the only place the spiral enters the nonlocal
    term and one of only two places it enters the energy at all.

    The indices below are ``s`` spin (one, for a spinor run), ``k`` k-point,
    ``n`` band, ``a``/``b`` spinor component, ``i``/``j`` projector channel.
    """
    if vkb.shape[-1] == 0:
        return jnp.zeros(())
    npwx = vkb.shape[1]
    components = psi.reshape(psi.shape[:-1] + (2, npwx))
    # The spiral's two blocks of rows -- the up component's ``nk`` first --
    # stacked into a spinor axis, which is what ``SpinorHamiltonian._project``
    # does one k-point at a time.
    pair = jnp.stack([vkb[:nk], vkb[nk:]], axis=1)  # (nk, 2, npwx, nkb)
    becp = jnp.einsum("kagi,sknag->sknai", pair.conj(), components)
    bands = jnp.real(jnp.einsum(
        "sknai,abij,sknbj->skn",
        becp.conj(), dvan_so.astype(becp.dtype), becp, optimize=True,
    ))
    return jnp.sum(weights * bands)
