"""The magnetic torque: ``dE/dtheta`` as the moment turns.

**Why a torque rather than a difference of energies.** A magnetocrystalline
anisotropy is 1e-5 Ry against a total energy of 1e2 -- seven digits of
cancellation -- so taking it as ``E(n_1) - E(n_2)`` asks two calculations to
agree far inside their own convergence. The torque does not: it is a *first*
derivative evaluated once, at one angle, and nothing cancels. That is the whole
reason the method exists (Wang, Wu, Wang and Freeman, PRB 54, 61 (1996)), and
for a uniaxial magnet it gives the anisotropy constant directly --

    E(theta) = K1 sin^2(theta)   =>   -dE/dtheta = -K1 sin(2 theta),

so a single calculation at **45 degrees** returns ``-K1``.

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

from defumat.scf.continuation import _axis, _collinear_axis

__all__ = ["band_energy_at_angle", "rotated_density", "torque_at_angle"]


def rotated_density(density, direction):
    """:func:`~defumat.scf.continuation.nc_magnetization_from_lsda`, traceable.

    The same rotation written in ``jnp`` so that ``direction`` may be a tracer.
    ``density`` may not be: a four-channel one has its axis read on the host by
    the same ``_collinear_axis`` the original uses, so the two lay the moment
    down with the same sign at every point.

    The original takes its direction through ``np.asarray`` and a ``float()``
    norm, which is right for a workflow argument and cannot be differentiated
    through; it stays as it is rather than being loosened, because it is on
    P58's validated path. The two are checked against each other pointwise, on
    a signed antiferromagnetic density and along several axes, in
    ``tests/unit/test_torque_signed_moment.py``, and on the concrete inputs of
    ``tests/regression/test_anisotropy.py``.
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
        moment = density[1:4]
        # **Signed, along the axis the states were rotated off.** The states
        # this energy is evaluated in come from ``nc_magnetization_from_lsda``,
        # which writes ``m . n`` with ``n`` from ``_collinear_axis`` -- the
        # dominant eigenvector of ``int m_a m_b``, which an antiferromagnet
        # does not zero. The potential has to be built from the same rotation.
        # This branch used to take ``|m|`` per point instead (through
        # ``safe_modulus``), which agrees only where every point is parallel to
        # the axis: an antiferromagnet came out ferromagnetic, so the potential
        # was not the one the states had been diagonalised in. The
        # axis is read on the host from the *density*, which is a constant of
        # the derivative -- only ``direction`` is traced -- and a genuinely
        # noncollinear density is refused there by name, as it is on the path
        # that built the states. A projection is linear in ``m``, so no
        # modulus and no guard at a vanishing moment are needed.
        along = _collinear_axis(density)
        scalar = jnp.sum(
            _axis(along or (0.0, 0.0, 1.0), moment.ndim) * moment, axis=0
        )
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


def _chunked_energy_and_slope(calculation, states, weights, density, plane,
                              angle, k_batch: int):
    """``(E, dE/dtheta)`` accumulated over ``k_batch`` k-points at a time.

    A Python loop of per-chunk ``value_and_grad`` calls, **not** one
    ``value_and_grad`` around a ``lax.map``, for the reason
    :func:`defumat.forces.spiral._chunked_energy_and_gradient` states: reverse
    mode through a scan stacks every chunk's residuals for the backward pass, so
    the mapped form holds the peak the single pass does. The loop discards each
    chunk's tape before the next one starts.

    **It is exact rather than an approximation**, and what makes it so is that
    the potential comes from the ``density`` argument rather than from the
    states: ``E(theta) = sum_k w_k <psi_k|H(theta)|psi_k>`` has no term
    coupling two k-points, so the chunk sums add and so do their derivatives.
    Every chunk is padded to exactly ``k_batch`` with a repeat of its own first
    k-point at **zero weight**, so all chunks share one shape and therefore one
    compilation, and the padding contributes nothing to either number.
    """
    psi = jnp.asarray(states)[0]
    occupation = jnp.asarray(weights)[0]
    nk = int(psi.shape[0])

    def chunk(value, indices, live):
        direction = _direction(value, plane[0], plane[1])
        rotated = rotated_density(density, direction)
        potential = calculation.potential(rotated, 1.0, None)
        hamiltonian = calculation.hamiltonian(potential.v_scf)[0]
        total = 0.0
        for slot in range(k_batch):
            ik = indices[slot]
            applied = hamiltonian.apply(psi[ik], ik)
            bands = jnp.real(jnp.sum(jnp.conj(psi[ik]) * applied, axis=-1))
            total = total + live[slot] * jnp.sum(occupation[ik] * bands)
        return total

    compiled = jax.jit(jax.value_and_grad(chunk))
    energy, slope = 0.0, 0.0
    for start in range(0, nk, k_batch):
        ks = np.arange(start, min(start + k_batch, nk))
        pad = k_batch - len(ks)
        indices = jnp.asarray(np.concatenate([ks, np.full(pad, ks[0], dtype=int)]))
        live = jnp.asarray(np.concatenate([np.ones(len(ks)), np.zeros(pad)]))
        value, derivative = compiled(jnp.asarray(float(angle)), indices, live)
        energy = energy + float(value)
        slope = slope + float(derivative)
    return energy, slope


def torque_at_angle(calculation, states, weights, density, plane, angle,
                    k_batch: int | None | str = "default"):
    """``-dE/dtheta``: the torque on the moment, in Ry per radian.

    The sign is the mechanical one -- a positive torque turns the moment
    towards larger ``theta`` -- so for ``E = K1 sin^2(theta)`` this returns
    ``-K1 sin(2 theta)`` and a measurement at ``pi/4`` gives ``-K1``.

    ``k_batch`` bounds the backward pass. The energy above walks the k axis with
    a Python loop, which is right for evaluating it and wrong for
    differentiating it: the tape then holds, **simultaneously for every
    k-point**, the real-space block ``SpinorHamiltonian._local_block`` builds,
    which its own docstring sizes at ``nbnd x 2 x N_smooth`` -- 33 GB for one
    k-point of the P74 cell. No dial reached it. ``k_batch`` stops at the NSCF
    that produced the states and ``DEFUMAT_BAND_BATCH`` reaches ``map_bands``
    inside the operator, where a scan stacks its residuals under ``jax.grad``
    just the same, so a run that was given ``k_batch = 1`` to fit inside a
    machine reached this function and asked for the whole axis anyway.
    ``None`` is that behaviour, kept as the default of the *whole-axis* path and
    reachable on purpose; an integer, or the dial's own ``"default"``, chunks.
    """
    from defumat.batching import resolve_k_batch

    resolved = resolve_k_batch(k_batch)
    nk = int(jnp.asarray(states).shape[1])
    if resolved is None or resolved >= nk:
        def energy(value):
            return band_energy_at_angle(
                calculation, states, weights, density, plane, value
            )

        return -float(jax.grad(energy)(jnp.asarray(float(angle))))

    _, slope = _chunked_energy_and_slope(
        calculation, states, weights, density, plane, angle, int(resolved)
    )
    return -float(slope)
