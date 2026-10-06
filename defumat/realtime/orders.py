"""The perturbative orders of the current, by nested ``jax.jvp`` through the propagation.

The orders are defined by scaling the field, ``kappa(t) = lam a(t)``, and
expanding the current, ``J(t) = sum_n lam^n J^(n)(t)`` with
``J^(n) = (1/n!) d^n J/d lam^n`` at ``lam = 0``. This is a derivative of code
that exists, the propagation of :mod:`defumat.realtime.propagate`, so it is
taken by forward-mode differentiation through every step rather than by
fitting runs at several amplitudes, and the orders come out separated exactly
rather than up to the next order in the field (``HARMONICS-NEXT.md``, "The
physics").

With the adiabatic switch-on ``a(t) = exp(eta t) cos(w t)`` on ``[-T, 0]``
(:class:`~defumat.realtime.pulse.Adiabatic`) the response of order ``n`` is
``exp(n eta t)`` times a periodic function, so

    J^(n)(t) = exp(n eta t) sum_m J_(n,m) exp(-i m w t),

and ``J_(n,m)`` is the response at the complex frequency ``m w + i n eta``
(:func:`fourier_component` projects it over the last period). The third
harmonic is ``J_(3,3)``, the intensity-dependent index ``J_(3,1)``, the
second harmonic ``J_(2,2)`` and the optical rectification ``J_(2,0)``.

**The cost of the nesting.** Order ``n`` of nested forward mode carries ``2^n``
copies of each state, since every level doubles the pytree and nothing tells
JAX that the tangents of one direction taken twice coincide: eight at third
order, and about that many times the Hamiltonian applications. Taylor-mode
propagation (``jax.experimental.jet``) would carry four; that is a measurement
to make with this route as its test, not a default.

The reference these are checked against is the dense frequency-domain
hierarchy of :mod:`defumat.realtime.dense`, which shares ``H(k)`` with this
route and nothing else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from defumat.eager import compiled_function
from defumat.realtime.propagate import (
    _Chunk, _block_function, _check_growth, _chunk_weights, _padded_grid, _prepare,
    _warn_damping, time_grid)

__all__ = ["OrdersResult", "propagate_orders", "fourier_component"]


@dataclass
class OrdersResult:
    """What :func:`propagate_orders` returns.

    Attributes:
        times: ``(nt,)`` in Hartree atomic units, ending at ``t = 0`` for an
            adiabatic switch-on.
        currents: ``(order + 1, nt, 3)``, ``J^(n)(t)`` in Hartree atomic units
            per unit of ``lam`` to the ``n``: the current density of order ``n``
            in the amplitude of ``a(t)``, which is ``kappa`` in 1/bohr.
        shape: the pulse at unit amplitude, ``a(t)``.
        dt: the step.
        volume: the cell volume in bohr^3.
    """

    times: np.ndarray
    currents: np.ndarray
    shape: object
    dt: float
    volume: float
    #: ``max |<u|u> - 1|`` of the unperturbed states at the end: the zeroth order
    #: of the tower, which a step that damps the bands would show here first.
    norm_drift: float = float("nan")

    def component(self, n: int, m: int, axis=None, periods: int = 1):
        """``J_(n,m)``, :func:`fourier_component` of order ``n`` at harmonic ``m``.

        ``axis`` selects a cartesian component (an index or a vector); the
        whole ``(3,)`` vector is returned without it.
        """
        value = fourier_component(self.times, self.currents[n], n, m,
                                  self.shape.omega, self.shape.eta, periods)
        if axis is None:
            return value
        if np.ndim(axis) == 0:
            return value[int(axis)]
        return np.asarray(axis, dtype=float) @ value


def fourier_component(times, current, n: int, m: int, omega: float, eta: float,
                      periods: int = 1):
    """``mean over the last periods of exp(-n eta t) J(t) exp(i m w t)``, ``(3,)`` complex.

    The samples are the last ``periods`` periods of the grid, ending at its last
    time, with the endpoint one period back excluded, so for a step that divides
    the period the mean is the exact Fourier coefficient of a trigonometric
    polynomial of degree below the number of samples.
    """
    times = np.asarray(times)
    dt = float(times[1] - times[0])
    exact = 2.0 * math.pi / omega / dt
    per = int(round(exact))
    if abs(exact - per) > 1e-6 * exact:
        # A window that is not a whole period leaks every other harmonic into
        # the one projected: 1.5e-2 on J_(3,3) beside a J_(3,1) a hundred times
        # larger at dt = 0.3 (found in review), where it is exact to 1.7e-14 at
        # a step that divides the period.
        raise ValueError(
            f"the step does not divide the period: 2 pi / (w dt) = {exact:.6f}. "
            "Run the orders with dt = period / an integer, as run_harmonic_orders does")
    take = slice(len(times) - per * periods, len(times))
    t = times[take]
    factor = np.exp(-n * eta * t) * np.exp(1j * m * omega * t)
    return np.mean(np.asarray(current)[take] * factor[:, None], axis=0)


def _lift(f, depth: int):
    """``f(states, lam) -> (states, out)`` carried through ``depth`` nested ``jvp`` in ``lam``.

    The states become a tower: ``(primal, tangent)`` at one level, each of them
    a tower one level down. The tangent of ``lam`` is one at every level and
    ``lam`` itself is linear, so the innermost tangent of the output after
    ``n`` levels is ``d^n out/d lam^n``.
    """
    if depth == 0:
        return f
    inner = _lift(f, depth - 1)

    def lifted(tower, lam):
        primal, tangent = tower
        (states, out), (states_dot, out_dot) = jax.jvp(
            inner, (primal, lam), (tangent, jnp.ones_like(lam)))
        return (states, states_dot), (out, out_dot)

    return lifted


def _tower(states, depth: int):
    """The states at ``lam = 0`` with every derivative zero, ``depth`` levels deep."""
    if depth == 0:
        return states
    lower = _tower(states, depth - 1)
    return (lower, jax.tree_util.tree_map(jnp.zeros_like, lower))


def _derivative(tower, k: int, depth: int):
    """``d^k/d lam^k`` out of an output tower of ``depth`` levels."""
    for level in range(depth):
        tower = tower[1 if level < k else 0]
    return tower


def propagate_orders(calculation, states, weights, v_scf, shape, *, dt: float,
                     order: int = 3, duration: float | None = None,
                     start: float | None = None, propagator: str = "taylor4",
                     k_batch="default", block_steps: int = 400,
                     kcart=None, potential: str = "frozen", corrector: int = 1,
                     symmetrise=None, density_symmetry=None) -> OrdersResult:
    """``J^(n)(t)`` for ``n <= order`` under ``kappa(t) = lam * shape(t)``, at ``lam = 0``.

    ``shape`` is a :class:`~defumat.realtime.pulse.Pulse` whose amplitude is the
    unit ``lam`` multiplies, usually an :class:`~defumat.realtime.pulse.Adiabatic`
    of amplitude one. The other arguments are
    :func:`~defumat.realtime.propagate.propagate`'s, ``symmetrise`` included:
    the currents of every order are averaged over those rotations, a polar
    vector each. With ``potential`` other than ``'frozen'`` the orders are those
    of the self-consistent propagation
    (:func:`~defumat.realtime.selfconsistent.propagate_orders_self_consistent`),
    where ``density_symmetry`` completes the wedge's density.
    """
    if potential != "frozen":
        from defumat.realtime.selfconsistent import propagate_orders_self_consistent

        return propagate_orders_self_consistent(
            calculation, states, weights, v_scf, shape, dt=dt, order=order,
            duration=duration, start=start, propagator=propagator, k_batch=k_batch,
            block_steps=block_steps, kcart=kcart, symmetrise=symmetrise,
            density_symmetry=density_symmetry, potential=potential,
            corrector=corrector)
    states = np.asarray(states)
    weights = np.asarray(weights, dtype=float)
    nk = states.shape[0]
    times, a_t, a_mid, _ = time_grid(shape, dt, duration, start)
    nsteps = len(times) - 1
    # The primal is the unperturbed evolution: the field is zero at lam = 0, so
    # the spectrum the step has to be stable for is that of H(k).
    setup = _prepare(calculation, states, weights, v_scf, 0.0, dt, propagator,
                     k_batch, kcart)
    real = setup.real
    a_mid_p, a_end_p, dts, nblocks = _padded_grid(a_mid, a_t[1:], nsteps, block_steps,
                                                  setup.dt_ry)
    base = _block_function(setup.step_fn, setup.centre)
    depth = int(order)

    def lifted(chunk, w, tower, lam, amid, aend, steps):
        def f(psi, l):
            return base(chunk, w, psi, l * amid, l * aend, steps)
        return _lift(f, depth)(tower, lam)

    def initial(chunk, w, tower, lam, a0):
        def f(psi, l):
            return psi, jax.grad(chunk.kappa_energy)(l * a0, psi, w)
        return _lift(f, depth)(tower, lam)[1]

    raw = np.zeros((depth + 1, nsteps + 1, 3))
    drift = 0.0
    _warn_damping(setup, nsteps)
    run = start_run = None
    lam = jnp.zeros((), dtype=real)
    for index, (rows, live) in enumerate(setup.chunks):
        chunk = setup.first if index == 0 else _Chunk.build(
            calculation, rows, setup.terms, setup.table, setup.kcart)
        w = _chunk_weights(weights, rows, live, real)
        tower = _tower(jnp.asarray(states[rows]), depth)
        a0 = jnp.asarray(a_t[0], dtype=real)
        if run is None:
            args = (chunk, w, tower, lam, jnp.asarray(a_mid_p[:block_steps], dtype=real),
                    jnp.asarray(a_end_p[:block_steps], dtype=real),
                    jnp.asarray(dts[:block_steps], dtype=real))
            run = compiled_function(lifted, *args)
            start_run = compiled_function(initial, chunk, w, tower, lam, a0)
        out = start_run(chunk, w, tower, lam, a0)
        for k in range(depth + 1):
            raw[k, 0] += np.asarray(_derivative(out, k, depth))
        for b in range(nblocks):
            sl = slice(b * block_steps, (b + 1) * block_steps)
            tower, out = run(chunk, w, tower, lam, jnp.asarray(a_mid_p[sl], dtype=real),
                             jnp.asarray(a_end_p[sl], dtype=real),
                             jnp.asarray(dts[sl], dtype=real))
            stop = min((b + 1) * block_steps, nsteps)
            for k in range(depth + 1):
                raw[k, b * block_steps + 1:stop + 1] += np.asarray(
                    _derivative(out, k, depth))[:stop - b * block_steps]
            norms = _check_growth(_derivative(tower, 0, depth), live,
                                  f"after step {stop} of k-chunk {index}")
        drift = max(drift, float(np.abs(norms - 1.0).max()))

    factorials = np.asarray([math.factorial(k) for k in range(depth + 1)], dtype=float)
    currents = -raw / (2.0 * setup.volume) / factorials[:, None, None]
    if symmetrise is not None:
        rotations = np.asarray(symmetrise, dtype=float)
        currents = np.einsum("sab,ntb->nta", rotations, currents) / len(rotations)
    return OrdersResult(times=times, currents=currents, shape=shape, dt=float(dt),
                        volume=setup.volume, norm_drift=drift)
