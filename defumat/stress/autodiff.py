"""The stress as ``-(1/Omega) grad`` of the total energy with respect to strain.

The whole method is four lines, which is the point: the kinetic term's
``(k+G)_a (k+G)_b``, the Hartree term's ``G_a G_b / G^2``, the local
pseudopotential's ``dV_loc/d|G|``, the exchange-correlation functional's
``-(e_xc - v_xc rho)`` diagonal *and* its gradient correction, the core charge,
the Ewald sum and the augmentation charge's own strain derivative all come out
of differentiating :func:`~defumat.stress.energy.strained_energy`, with no
expression written for any of them. ``PW/src/stress.f90`` and the eight routines
it calls are the alternative, and the ones that are written here
(:mod:`defumat.stress.analytic`) exist as a check on this, not as the way it is
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

from defumat.forces.energy import FrozenState, geometry_compiled, hoisted
from defumat.forces.chunked import chunked_gradient, wants_chunks
from defumat.stress.energy import (
    require_a_differentiable_cell, strained_energy, strained_energy_terms,
)

__all__ = ["autodiff_stress", "autodiff_stress_terms"]


def _hoisted_note():
    """The big arrays this module's gradients take as **arguments**.

    See :data:`~defumat.forces.energy.HOISTED_FIELDS`, which carries the
    measurement: capturing them instead put 1062 MB of constants into the
    compiled stress gradient, of which the largest was an outer product that
    no longer exists at all (:class:`~defumat.paw.symmetry.BecsumSymmetry`).
    """


def autodiff_stress(calculation, state: FrozenState) -> jnp.ndarray:
    """``(3, 3)`` stress in Ry/bohr^3, before symmetrisation.

    ``calculation`` fixes everything the cell does not; the derivative is taken
    at *its* cell, i.e. at ``epsilon = 0``.
    """
    if wants_chunks(calculation, state):
        # Memory mode, or a state in host memory: the k axis is walked rather
        # than taped whole (``GPU-MEMORY-NEXT.md`` item 3).
        require_a_differentiable_cell(calculation)
        gradient = chunked_gradient(calculation, state, "strain", _zero())[1]
    else:
        compiled, geometry = _energy_gradient(calculation)
        gradient = compiled(_zero(), state, hoisted(calculation), geometry)
    return -gradient / calculation.system.cell.volume


def autodiff_stress_terms(calculation, state: FrozenState) -> dict:
    """The same tensor split into the energy's contributions.

    One ``(3, 3)`` array per term of
    :func:`~defumat.stress.energy.strained_energy_terms`, each already divided
    by the volume and negated, so that they sum to :func:`autodiff_stress`.
    """
    compiled, geometry = _term_gradients(calculation)
    gradients = compiled(_zero(), state, hoisted(calculation), geometry)
    volume = calculation.system.cell.volume
    return {name: -value / volume for name, value in gradients.items()}


def _zero() -> jnp.ndarray:
    """``epsilon = 0``: the calculation's own cell."""
    return jnp.zeros((3, 3))


def _energy_gradient(calculation):
    """``(grad of the strained energy, its geometry arguments)``, compiled once per run.

    Cached on the calculation the way the force's gradient is, and **keyed on
    what it closes over** (:class:`~defumat.forces.energy.GeometryKey`). The
    strain is an argument, and so are the cell it strains, the positions it
    moves and every other array a geometry carries
    (:data:`~defumat.forces.energy.GEOMETRY_FIELDS`), so an entry inherited
    through :meth:`~defumat.scf.driver.Calculation.at_cell` or
    :meth:`~defumat.scf.driver.Calculation.at_positions` -- both of which copy
    the instance dict -- answers at the new geometry. It used to be keyed on the
    calculation's identity instead, which made every step of a variable-cell
    relaxation compile it again (`OPEN.md` Part XXIII item 7).
    """
    def build(key):
        def energy(eps, state, big, geometry):
            here = key.rebuild(geometry, big)
            return strained_energy(here, eps, state, spinors=True)

        return jax.jit(jax.grad(energy))

    return geometry_compiled(calculation, "_strain_gradient", build)


def _term_gradients(calculation):
    """``jacfwd`` of the term dict and its geometry arguments, keyed as :func:`_energy_gradient` is."""
    def build(key):
        def terms(eps, state, big, geometry):
            here = key.rebuild(geometry, big)
            return strained_energy_terms(here, eps, state, spinors=True)

        return jax.jit(jax.jacfwd(terms))

    return geometry_compiled(calculation, "_strain_term_gradients", build)
