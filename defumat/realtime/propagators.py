"""One time step of ``i d psi/dt = H(t) psi``, behind a name registry.

A propagator here is a function ``step(apply, psi, dt, centre)`` returning
``exp(-i H dt) psi`` to its order, with ``apply(psi) = H psi`` for the
Hamiltonian at the time the step is centred on (the driver passes the one at
``t + dt/2``, which makes every entry the exponential midpoint rule, second
order in ``dt`` whatever the order of its exponential), ``dt`` in ``hbar/Ry``
and ``centre`` an energy in Ry subtracted from ``H`` first. Subtracting a
constant is a global phase per k-point, which neither the current nor the
energy sees, and it halves the spectral radius the step has to be stable for,
since the spectrum of a plane-wave Hamiltonian is one-sided.

**The Taylor expansion of the exponential to fourth order** is the one entry,
four Hamiltonian applications a step. It is stable on the imaginary axis for
``|dt rho| <= 2 sqrt 2``, ``rho`` the spectral radius of ``H - centre``:
``|R(iy)|^2 = 1 - y^6/72 + y^8/576`` for ``R(z) = sum_{n<=4} z^n/n!``, which is
at most one exactly when ``y^2 <= 8``, and inside that it loses norm at
``y^6/144`` a step, so the norm drift a run reports is the measure of how far
inside the bound it is. That bound is the imaginary-axis interval of the
polynomial and was not taken from a paper. :func:`stable_step` is what the
driver refuses a step against.

A split-operator step (kinetic in G, local potential in real space, the
separable nonlocal term per atom) is the candidate the plan names for a second
entry: one transform pair a band instead of four and exactly unitary, with an
error that has to be measured against this one on the current before it can be
a default (``HARMONICS-NEXT.md``, "Speed").
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

__all__ = ["taylor4", "register_propagator", "get_propagator",
           "propagator_names", "stable_step", "TAYLOR4_BOUND"]

#: ``|dt rho|`` at which the fourth-order Taylor step stops being stable.
TAYLOR4_BOUND = 2.0 * math.sqrt(2.0)


def taylor4(apply, psi, dt, centre):
    """``sum_{n=0}^{4} (-i dt (H - centre))^n / n! psi``, four applications.

    ``-i`` is applied by exchanging the real and imaginary parts rather than by
    a complex literal, so the dtype of the step is the dtype of ``psi``
    (``config.py``'s policy, single precision included).
    """
    # ``dt`` and ``centre`` in the state's own real precision: a strongly typed
    # float64 scalar would promote a complex64 state to complex128 (measured).
    real = psi.real.dtype
    dt = jnp.asarray(dt, dtype=real)
    centre = jnp.asarray(centre, dtype=real)
    out = psi
    term = psi
    for n in range(1, 5):
        shifted = apply(term) - centre * term
        term = jax.lax.complex(shifted.imag, -shifted.real) * (dt / n)
        out = out + term
    return out


def stable_step(radius: float, bound: float = TAYLOR4_BOUND) -> float:
    """The largest ``dt`` (``hbar/Ry``) stable for a spectral radius in Ry."""
    return bound / float(radius)


_PROPAGATORS = {"taylor4": (taylor4, TAYLOR4_BOUND)}


def register_propagator(name: str, step, bound: float) -> None:
    """Make ``step`` reachable by name, with the ``|dt rho|`` it is stable to."""
    _PROPAGATORS[name.lower()] = (step, float(bound))


def get_propagator(name: str):
    """``(step, bound)`` for the propagator called ``name``."""
    key = name.lower()
    if key not in _PROPAGATORS:
        raise ValueError(
            f"unknown propagator {name!r}; available: {propagator_names()}")
    return _PROPAGATORS[key]


def propagator_names() -> tuple[str, ...]:
    return tuple(sorted(_PROPAGATORS))
