"""QE's hand-derived D2 force and stress, transcribed as the cross-check.

``mm_dispersion.f90``'s ``force_london`` and ``stres_london``, which this
project does **not** use to compute anything: the force and the stress come from
``jax.grad`` of :meth:`pypresso.vdw.grimme.GrimmeD2.energy` in the two
coordinates, exactly as every other term's do. They are here for the reason the
transcribed Ewald, local and core forces are (:mod:`pypresso.forces.analytic`) --
two implementations that share no machinery, checked against each other, is what
catches a sign or a missing chain-rule factor that a self-consistent single
implementation would carry silently.

**The one thing to keep straight is which way the separation points.** ``rgen``
returns ``r = R - (tau_a - tau_b)``, the vector from atom ``a`` to atom ``b``'s
image, where the kernel here broadcasts ``s = tau_a - tau_b + R``, its negative.
The stress is quadratic in it and does not notice; the force is linear in it and
changes sign, which is a bug that shows up only as a relaxation walking uphill.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from pypresso.vdw.grimme import _folded

__all__ = ["dispersion_force", "dispersion_stress"]


def dispersion_force(dispersion, positions: jnp.ndarray) -> jnp.ndarray:
    """``force_london``: ``(nat, 3)`` in Ry/bohr."""
    coefficients, separations = _pair_coefficients(dispersion, positions)
    return jnp.einsum("abt,abtc->ac", coefficients, separations)


def dispersion_stress(dispersion, positions: jnp.ndarray, volume: float) -> jnp.ndarray:
    """``stres_london``: ``(3, 3)`` in Ry/bohr^3.

    QE builds the lower triangle and mirrors it; the outer product here is
    symmetric by construction, so there is nothing to mirror.
    """
    coefficients, separations = _pair_coefficients(dispersion, positions)
    outer = jnp.einsum("abt,abtc,abtd->cd", coefficients, separations, separations)
    return outer / (2.0 * volume)


@jax.jit
def _pair_coefficients(dispersion, positions):
    """The scalar every pair contributes, and the separation it multiplies.

    ``force_london`` and ``stres_london`` differ in QE only in what the same
    bracket is contracted with -- one power of the separation or two -- so it is
    written once. The bracket is

        coeff = s6 (C6/d^6) [ (beta/R_sum) f (1 - f) - 6 f / d ] / d

    with ``f`` the damping function, where QE writes ``-par exp/(1+exp) + 6/d``
    inside ``scal6/(1+expval) * fac * (...)`` and carries the opposite sign
    because its separation points the other way. ``exp/(1 + exp)`` is ``1 - f``,
    which is how the two forms are the same and why nothing here has to guard an
    exponential against overflow.

    :func:`pypresso.vdw.grimme._dispersion_kernel`'s masking rule, verbatim:
    sanitise the *square* before the square root, or the gradient of this
    function is NaN wherever an atom sits on a lattice point.
    """
    separations = (
        _folded(positions, dispersion.reciprocal, dispersion.lattice)[:, :, None, :]
        + dispersion.translations[None, None, :, :]
    )
    square = jnp.sum(separations**2, axis=-1)
    keep = (square > 1.0e-16) & (square <= dispersion.rcut**2)
    distances = jnp.sqrt(jnp.where(keep, square, 1.0))

    r_sum = dispersion.r_sum[:, :, None]
    damping = jax.nn.sigmoid(dispersion.beta * (distances / r_sum - 1.0))
    bracket = (
        dispersion.beta / r_sum * damping * (1.0 - damping)
        - 6.0 * damping / distances
    )
    coefficients = (
        dispersion.s6 * dispersion.c6[:, :, None] / distances**6 * bracket / distances
    )
    return jnp.where(keep, coefficients, 0.0), separations
