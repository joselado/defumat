"""The magnetic torque: ``dE/dtheta`` as the moment turns.

**Why a torque rather than a difference of energies.** A magnetocrystalline
anisotropy is 1e-5 Ry against a total energy of 1e2 -- seven digits of
cancellation -- so taking it as ``E(n_1) - E(n_2)`` asks two calculations to
agree far inside their own convergence. The torque does not: it is a *first*
derivative evaluated once, at one angle, and nothing cancels. That is the whole
reason the method exists (Wang, Wu, Wang and Freeman, PRB 54, 61 (1996)), and
for a uniaxial magnet it gives the anisotropy constant directly --

    E(theta) = K1 sin^2(theta)   =>   dE/dtheta = K1 sin(2 theta),

so a single calculation at **45 degrees** returns ``K1``.

**It is the same construction as the force, term for term**
(:mod:`defumat.forces.spiral` says this of ``dE/dq`` and it is as true here):
the energy is written as a function of the angle at *frozen* wavefunctions and
the gradient is ``jax.grad`` of it. No expression is derived for any
contribution. The literature's torque *is* such an expression --
``<psi| dH_SO/dtheta |psi>``, differentiated by hand -- so what this module adds
to a known technique is the same thing P15 added to the force.

**Only one term carries the angle, and knowing which makes this cheap.**
``dvan_so`` is the spin-orbit matrix in the *crystal* frame and does not depend
on where the moment points; neither does ``qq_so``, the kinetic term or the
local pseudopotential. Turning the moment turns the **exchange field** and
nothing else, so ``dH/dtheta`` lives entirely in the self-consistent potential
built from the rotated density. Everything else differentiates to zero on its
own, and the gradient finds that without being told.

**What is frozen and why the answer is still right.** ``sum_n w_n <psi_n|H|psi_n>``
over the occupied manifold is stationary with respect to the states at fixed
``H``, so differentiating at frozen ``psi`` gives the same answer as
differentiating through the eigenproblem -- the Hellmann-Feynman argument, and
the same envelope argument P15 and P25 make. It is checked rather than asserted:
:func:`band_energy_at_angle` evaluated at the angle its states came from must
reproduce ``sum w eps``, which is one line and catches a wrong contraction, a
lost weight or a mis-shaped spinor at once.

The remaining angle dependence of the *total* energy is nothing: the Hartree
term sees only the charge, the exchange-correlation energy only ``|m|``, the
Ewald sum neither, and ``deband``'s ``int rho v = n v_0 + |m| |b|`` is invariant
too. So ``dE_total/dtheta = dE_band/dtheta`` exactly, which is the force
theorem's own statement one derivative down.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["band_energy_at_angle", "rotated_density", "torque_at_angle"]


def rotated_density(density, direction):
    """:func:`~defumat.scf.continuation.nc_magnetization_from_lsda`, traceable.

    The same rotation written in ``jnp`` so that ``direction`` may be a tracer.
    The original takes its direction through ``np.asarray`` and a ``float()``
    norm, which is right for a workflow argument and cannot be differentiated
    through; it stays as it is rather than being loosened, because it is on
    P58's validated path. The two are checked against each other on concrete
    inputs in ``tests/unit/test_torque.py``.
    """
    density = jnp.asarray(density)
    direction = jnp.asarray(direction)
    direction = direction / jnp.sqrt(jnp.sum(direction**2))
    channels = density.shape[0]
    if channels == 2:
        charge = density[0] + density[1]
        scalar = density[0] - density[1]
    elif channels == 4:
        charge = density[0]
        # Projected onto the direction it already has, which is what makes this
        # idempotent on a state that is already noncollinear.
        moment = density[1:4]
        norm = jnp.sqrt(jnp.sum(jnp.sum(moment**2, axis=0)))
        scalar = jnp.where(norm > 0.0, jnp.sum(moment**2, axis=0) ** 0.5, 0.0)
    else:
        raise ValueError(
            f"rotated_density wants a magnetic density, got {channels} channels"
        )
    shaped = direction.reshape((3,) + (1,) * scalar.ndim)
    return jnp.concatenate([charge[None], shaped * scalar[None]])


def _direction(angle, first, second):
    """``cos(angle) e1 + sin(angle) e2``: the moment turning in one plane."""
    first = jnp.asarray(first, dtype=float)
    second = jnp.asarray(second, dtype=float)
    return jnp.cos(angle) * first + jnp.sin(angle) * second


def band_energy_at_angle(calculation, states, weights, density, plane, angle):
    """``sum_n w_n <psi_n | H(theta) | psi_n>`` at frozen ``states``.

    ``plane`` is the orthonormal pair ``(e1, e2)`` the moment turns in, so that
    ``angle = 0`` points along ``e1``. ``states`` is ``(1, nk, nbnd, 2 npwx)``
    -- a spinor run has one density channel whatever else it has -- and
    ``weights`` is the matching ``wg``.

    This is the quantity :func:`torque_at_angle` differentiates, and evaluating
    it *at* the angle its states came from is the check that it is assembled
    right (see the module docstring).
    """
    direction = _direction(angle, plane[0], plane[1])
    rotated = rotated_density(density, direction)
    potential = calculation.potential(rotated, 1.0, None)
    hamiltonian = calculation.hamiltonian(potential.v_scf)[0]

    psi = jnp.asarray(states)[0]
    occupation = jnp.asarray(weights)[0]

    total = 0.0
    for ik in range(psi.shape[0]):
        applied = hamiltonian.apply(psi[ik], ik)
        bands = jnp.real(jnp.sum(jnp.conj(psi[ik]) * applied, axis=-1))
        total = total + jnp.sum(occupation[ik] * bands)
    return total


def torque_at_angle(calculation, states, weights, density, plane, angle):
    """``-dE/dtheta``: the torque on the moment, in Ry per radian.

    The sign is the mechanical one -- a positive torque turns the moment
    towards larger ``theta`` -- so for ``E = K1 sin^2(theta)`` this returns
    ``-K1 sin(2 theta)`` and a measurement at ``pi/4`` gives ``-K1``.
    """
    def energy(value):
        return band_energy_at_angle(
            calculation, states, weights, density, plane, value
        )

    return -float(jax.grad(energy)(jnp.asarray(float(angle))))
