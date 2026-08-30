"""The stress as ``-(1/Omega) grad`` of the total energy with respect to strain.

The whole method is four lines, which is the point: the kinetic term's
``(k+G)_a (k+G)_b``, the Hartree term's ``G_a G_b / G^2``, the local
pseudopotential's ``dV_loc/d|G|``, the exchange-correlation functional's
``-(e_xc - v_xc rho)`` diagonal *and* its gradient correction, the core charge,
the Ewald sum and the augmentation charge's own strain derivative all come out
of differentiating :func:`~pypresso.stress.energy.strained_energy`, with no
expression written for any of them. ``PW/src/stress.f90`` and the eight routines
it calls are the alternative, and the ones that are written here
(:mod:`pypresso.stress.analytic`) exist as a check on this, not as the way it is
done.

**Reverse mode for the total, forward mode for the terms.** The energy is a
scalar of nine inputs, so one reverse pass gives the whole tensor and is what
:func:`autodiff_stress` uses. The *decomposition* is eleven scalars of the same
nine inputs, where reverse mode would cost eleven passes and forward mode costs
nine and delivers every term at once -- so :func:`autodiff_stress_terms` is a
``jacfwd``. On silicon at ``ecutwfc = 12`` that is the difference between 0.9 s
and 5 s.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from pypresso.forces.energy import FrozenState
from pypresso.stress.energy import strained_energy, strained_energy_terms

__all__ = ["autodiff_stress", "autodiff_stress_terms"]


def autodiff_stress(calculation, state: FrozenState) -> jnp.ndarray:
    """``(3, 3)`` stress in Ry/bohr^3, before symmetrisation.

    ``calculation`` fixes everything the cell does not; the derivative is taken
    at *its* cell, i.e. at ``epsilon = 0``.
    """
    gradient = _energy_gradient(calculation)(_zero(), state)
    return -gradient / calculation.system.cell.volume


def autodiff_stress_terms(calculation, state: FrozenState) -> dict:
    """The same tensor split into the energy's contributions.

    One ``(3, 3)`` array per term of
    :func:`~pypresso.stress.energy.strained_energy_terms`, each already divided
    by the volume and negated, so that they sum to :func:`autodiff_stress`.
    """
    gradients = _term_gradients(calculation)(_zero(), state)
    volume = calculation.system.cell.volume
    return {name: -value / volume for name, value in gradients.items()}


def _zero() -> jnp.ndarray:
    """``epsilon = 0``: the calculation's own cell."""
    return jnp.zeros((3, 3))


def _energy_gradient(calculation):
    """``grad`` of the strained energy, compiled once per calculation.

    Cached on the calculation the way the force's gradient is, and **keyed on
    the calculation it closed over**. The strain is an argument, but the cell it
    strains and the positions it moves are the captured calculation's, so an
    entry inherited through :meth:`~pypresso.scf.driver.Calculation.at_strain`
    or :meth:`~pypresso.scf.driver.Calculation.at_positions` -- both of which
    copy the instance dict -- would answer at the geometry it was compiled at
    and say nothing. Rebuilt when the entry does not belong to this calculation.
    """
    cached = calculation.__dict__.get("_strain_gradient")
    if cached is None or cached[0] is not calculation:
        cached = (calculation, jax.jit(jax.grad(
            lambda eps, state: strained_energy(calculation, eps, state, spinors=True)
        )))
        calculation._strain_gradient = cached
    return cached[1]


def _term_gradients(calculation):
    """``jacfwd`` of the term dict, compiled once per calculation.

    Keyed on that calculation for the reason :func:`_energy_gradient` gives.
    """
    cached = calculation.__dict__.get("_strain_term_gradients")
    if cached is None or cached[0] is not calculation:
        cached = (calculation, jax.jit(jax.jacfwd(
            lambda eps, state: strained_energy_terms(calculation, eps, state, spinors=True)
        )))
        calculation._strain_term_gradients = cached
    return cached[1]
