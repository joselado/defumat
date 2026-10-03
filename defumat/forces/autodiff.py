"""Forces as ``-grad`` of the total energy with respect to the atomic positions.

The whole method is four lines, which is the point: the Hellmann-Feynman term,
the ultrasoft Pulay terms, the augmentation charge's own derivative and the
Ewald sum all come out of differentiating
:func:`~defumat.forces.energy.frozen_energy`, with no expression written for
any of them. ``PW/src/forces.f90`` and the six routines it calls are the
alternative, and they are also implemented here
(:mod:`defumat.forces.analytic`) -- as a check on this one, not as the way it
is done.

What makes it correct rather than merely convenient is stationarity: see the
module docstring of :mod:`defumat.forces.energy` for which terms exist because
of it, and which are absent for the same reason.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from defumat.forces.chunked import chunked_gradient, wants_chunks
from defumat.forces.energy import FrozenState, frozen_energy, geometry_compiled, hoisted

__all__ = ["autodiff_forces"]


def autodiff_forces(calculation, state: FrozenState) -> jnp.ndarray:
    """``(nat, 3)`` cartesian forces in Ry/bohr, before symmetrisation.

    ``calculation`` fixes everything the geometry does not; the derivative is
    taken at *its* positions.
    """
    positions = calculation.system.structure.positions
    if wants_chunks(calculation, state):
        # Memory mode, or a state in host memory: the k axis is walked rather
        # than taped whole (``GPU-MEMORY-NEXT.md`` item 3).
        return -chunked_gradient(calculation, state, "positions", positions)[1]
    gradient, geometry = _energy_gradient(calculation)
    return -gradient(positions, state, hoisted(calculation), geometry)


def _energy_gradient(calculation):
    """``(grad of the frozen energy, its geometry arguments)``, compiled once per run.

    The compiled function takes the positions, the large fields
    (:data:`~defumat.forces.energy.HOISTED_FIELDS`) and every other array the
    geometry moves (:data:`~defumat.forces.energy.GEOMETRY_FIELDS`) as
    arguments, and is cached on the calculation under a key of everything it
    still closes over (:class:`~defumat.forces.energy.GeometryKey`). A
    calculation moved with :meth:`~defumat.scf.driver.Calculation.at_positions`
    or :meth:`~defumat.scf.driver.Calculation.at_cell` inherits the cache and
    matches the key, so one compiled kernel serves every step of a relaxation,
    the variable-cell one included.
    """
    def build(key):
        def energy(tau, state, big, geometry):
            here = key.rebuild(geometry, big)
            return frozen_energy(here, tau, state, spinors=True)

        return jax.jit(jax.grad(energy))

    return geometry_compiled(calculation, "_energy_gradient", build)
