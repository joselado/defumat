"""The energy as a function of a strain, at frozen wavefunctions.

**It is P15's construction with the cell in place of the atoms.** The total
energy is written down once (:func:`defumat.forces.energy.energy_at`) and the
derivative is ``jax.grad`` of it; what changes here is only which coordinate the
calculation is moved along -- :meth:`~defumat.scf.driver.Calculation.at_strain`
instead of :meth:`~defumat.scf.driver.Calculation.at_positions`. No expression
is derived for any contribution, and QE's ``PW/src/stress.f90`` and the eight
routines it calls are a *check* on this (:mod:`defumat.stress.analytic`), not
the way it is done.

The strain is the ordinary one,

    h -> (1 + epsilon) h,   tau_a -> (1 + epsilon) tau_a,   G -> (1 + eps)^-T G,

with the atoms carried along in crystal coordinates and the reciprocal lattice
following from the Miller indices being what is stored. The stress is

    sigma_ab = -(1/Omega) dE/d(epsilon_ab)      [Ry/bohr^3]

and the sign is QE's: an isotropic ``epsilon = e I`` gives
``tr sigma / 3 = -dE/dV``, which is the pressure.

**What makes the partial derivative the total one** is the same stationarity
argument :mod:`defumat.forces.energy` makes: at the converged solution the
energy is stationary with respect to the wavefunctions subject to
``<psi|S|psi> = 1``, so with the coefficients, the occupations and the
eigenvalues held fixed the partial derivative with respect to *any* external
parameter is the total one. Two consequences worth stating because they are not
obvious at a glance:

* **The frozen quantity is the coefficient vector, not the wavefunction.** A
  strain moves the plane waves ``e^{i(k+G).r}`` with the cell, so freezing the
  coefficients holds the state fixed in *crystal* coordinates -- which is the
  variational parameter the SCF minimised over, exactly as freezing the periodic
  part of a spinor is for a spiral (P21).
* **The orthonormality constraint carries no strain.** ``<psi|psi>`` is a sum
  over the sphere of ``|c_G|^2``, and the sphere is a set of integers; the
  ultrasoft part ``qq_ij`` is ``int Q_ij(r) dr`` over all space and has no cell
  in it either. So unlike the force, whose Pulay term is the constraint's own
  derivative, the stress's Pulay term is not this -- it is the cutoff, below.

**The plane-wave sphere is held fixed while differentiating.** Which plane waves
satisfy ``|k + G|^2 <= ecutwfc`` is a host-side decision that cannot be traced,
and it is piecewise constant in ``epsilon``, so on each piece the frozen-sphere
derivative is the exact derivative. What it misses is the jump at the strains
where a plane wave crosses the cutoff, and that jump is the **Pulay stress** --
the same trade :mod:`defumat.forces.spiral` documents for ``q``, and a larger
one here, because a strain changes ``|k+G|`` for *every* plane wave at once
rather than shifting one sphere. It is the reason a stress at ``ecutwfc = 12``
disagrees with a re-converged finite difference of the SCF energy by ~1e-3
Ry/bohr^3 while agreeing with QE (which makes the same approximation) to 1e-9.
`PLAN.md`'s P11 section carries the measurement against the cutoff.

**A magnetic field or a constrained moment is refused rather than corrected**,
for P21's reason verbatim: the field's own energy is deliberately outside the
reported total (:mod:`defumat.scf.fields`), so the converged state is
stationary for a different functional than the one being differentiated and the
missing term would be silent.
"""

from __future__ import annotations

import jax.numpy as jnp

from defumat.forces.energy import (
    FrozenState,
    energy_at,
    reject_potential_only,
    reject_spinors,
)

__all__ = ["strained_energy", "strained_energy_terms", "require_a_differentiable_cell"]


def strained_energy(calculation, strain, state: FrozenState, spinors: bool = False):
    """The total energy of ``calculation`` under ``strain``, state frozen.

    Args:
        strain: ``(3, 3)``. Zero is the calculation's own cell.

    A scalar in Ry, and a differentiable function of ``strain``.
    """
    require_a_differentiable_cell(calculation)
    reject_potential_only(calculation)
    if not spinors:
        reject_spinors(calculation)
    return energy_at(calculation.at_strain(strain), state, spinors=spinors)


def strained_energy_terms(calculation, strain, state: FrozenState,
                          spinors: bool = False) -> dict:
    """The same energy as a dict of contributions rather than their sum.

    Differentiating this instead gives the stress term by term, which is what
    the comparison against QE's own decomposition needs. The regrouping is not
    one-to-one and the mapping is in :mod:`defumat.stress.analytic`.
    """
    require_a_differentiable_cell(calculation)
    reject_potential_only(calculation)
    if not spinors:
        reject_spinors(calculation)
    return energy_at(calculation.at_strain(strain), state, terms=True, spinors=spinors)


def require_a_differentiable_cell(calculation) -> None:
    """The things that make ``dE/d(epsilon)`` at frozen state not be the answer."""
    if calculation.spiral:
        raise NotImplementedError(
            "the stress of a spin spiral is not implemented: q is given in "
            "lattice coordinates, so a strain turns the spiral as well and the "
            "generalized Bloch theorem's own term would be missing"
        )
    if calculation.magnetic_field is not None:
        raise NotImplementedError(
            "the stress with a magnetic field or a constrained moment is not "
            "implemented: the field's energy is deliberately outside the "
            "reported total (see defumat.scf.fields), so the converged state "
            "is stationary for a different functional than the one being "
            "differentiated and the missing term would be silent"
        )


def strain_of(deformation) -> jnp.ndarray:
    """``epsilon`` such that ``1 + epsilon`` is the given deformation matrix."""
    deformation = jnp.asarray(deformation)
    return deformation - jnp.eye(3, dtype=deformation.dtype)
