"""Forces as ``-grad`` of the total energy with respect to the atomic positions.

The whole method is four lines, which is the point: the Hellmann-Feynman term,
the ultrasoft Pulay terms, the augmentation charge's own derivative and the
Ewald sum all come out of differentiating
:func:`~pypresso.forces.energy.frozen_energy`, with no expression written for
any of them. ``PW/src/forces.f90`` and the six routines it calls are the
alternative, and they are also implemented here
(:mod:`pypresso.forces.analytic`) -- as a check on this one, not as the way it
is done.

What makes it correct rather than merely convenient is stationarity: see the
module docstring of :mod:`pypresso.forces.energy` for which terms exist because
of it, and which are absent for the same reason.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from pypresso.forces.energy import FrozenState, frozen_energy

__all__ = ["autodiff_forces"]


def autodiff_forces(calculation, state: FrozenState) -> jnp.ndarray:
    """``(nat, 3)`` cartesian forces in Ry/bohr, before symmetrisation.

    ``calculation`` fixes everything the geometry does not; the derivative is
    taken at *its* positions.
    """
    positions = calculation.system.structure.positions
    gradient = _energy_gradient(calculation)(positions, state)
    return -gradient


def _energy_gradient(calculation):
    """``grad`` of the frozen energy, compiled once per calculation.

    The compiled function is cached on the calculation, and a calculation moved
    with :meth:`~pypresso.scf.driver.Calculation.at_positions` inherits the
    cache: the energy depends on the geometry only through the ``positions``
    argument -- everything position-dependent on the object itself is rebuilt
    inside -- so the same compiled kernel serves every step of a relaxation.
    """
    cached = getattr(calculation, "_energy_gradient", None)
    if cached is None:
        cached = jax.jit(jax.grad(
            lambda tau, state: frozen_energy(calculation, tau, state, spinors=True)
        ))
        calculation._energy_gradient = cached
    return cached
