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

**The torque as a vector: every spin turned by one rotation**
(``ORIENTATION-NEXT.md``, Route A). The angle above turns a collinear moment in
one plane. :func:`orientation_torque` turns a whole texture by a rotation ``R``,
``m(r) -> R m(r)`` with the charge kept, and returns ``-dF/dw`` for the three
generators at once, ``w`` being the rotation vector about the current
orientation. For a collinear texture it contains the plane torque as one
component, ``torque . (e1 x e2)``, and its component along the moment is zero
because a rotation about the moment moves nothing. For a texture that is not
collinear nothing is read off an axis: :func:`rotate_texture` is ``R m`` on the
three magnetization channels, linear in ``R``, so the charge, ``|m|`` and every
angle between moments are kept exactly.

The derivative is only ever taken at ``w = 0``, where the rotation is written as
``(1 + [w]x) R0`` (:func:`rotation_near`). That agrees with ``exp([w]x) R0`` in
value and in first derivative there, which is all a gradient at ``w = 0`` reads,
and it contains no norm of ``w``: Rodrigues' formula puts ``sqrt(sum w^2)`` at
the origin, whose gradient is ``0/0`` exactly where every call evaluates it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from defumat.scf.continuation import _axis, _collinear_axis

__all__ = [
    "band_energy_at_angle",
    "band_energy_at_rotation",
    "cross_matrix",
    "orientation_torque",
    "rotate_texture",
    "rotated_density",
    "rotation_near",
    "torque_at_angle",
]


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
    return _band_energy(calculation, states, weights,
                        rotated_density(density, direction))


def _hamiltonian_of(calculation, density, becsum=()):
    """The spinor Hamiltonian built from ``density`` and, on PAW, from ``becsum``.

    A PAW Hamiltonian has two representations of its potential, the grid one
    from the density and the one-centre coefficients ``ddd_paw`` from
    ``becsum``, and turning the texture turns both. Rebuilding ``ddd_paw`` here,
    inside the function a gradient is taken of, is what puts the one-centre
    field's share of the torque into the same ``jax.grad`` as the grid's; built
    once outside, it would be a constant the gradient passes through, and the
    torque would miss it.
    """
    potential = calculation.potential(density, 1.0, None)
    ddd_paw = calculation.onecenter(tuple(becsum))[1] if becsum else None
    return calculation.hamiltonian(potential.v_scf, ddd_paw)[0]


def _band_energy(calculation, states, weights, density, becsum=()):
    """``sum_n w_n <psi_n | H[density, becsum] | psi_n>`` over every k-point at once."""
    hamiltonian = _hamiltonian_of(calculation, density, becsum)

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
    """``(E, dE/dtheta)`` over ``k_batch`` k-points at a time, in one plane."""
    def build(value):
        return rotated_density(density, _direction(value, plane[0], plane[1])), ()

    energy, slope = _chunked_value_and_grad(
        calculation, states, weights, build, jnp.asarray(float(angle)), k_batch
    )
    return energy, float(slope)


def _chunked_value_and_grad(calculation, states, weights, build, parameter,
                            k_batch: int):
    """``(E, dE/dp)`` accumulated over ``k_batch`` k-points at a time.

    ``build(p)`` is the pair ``(density, becsum)`` the Hamiltonian is made
    from, ``becsum`` empty except on PAW, and ``p`` is a scalar angle or a
    rotation vector; the gradient comes back with its shape.

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
        hamiltonian = _hamiltonian_of(calculation, *build(value))
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
        value, derivative = compiled(parameter, indices, live)
        energy = energy + float(value)
        slope = slope + np.asarray(derivative)
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


def rotate_texture(field, rotation):
    """``R m`` on a four-channel field, its charge channel kept.

    ``field`` is ``(4, ...)``, charge first and the three cartesian
    magnetization components after it: a density on the grid, or one species'
    ``becsum``, which carries its components on the same leading axis
    (``ultracell/kramers.py:time_reversed`` negates exactly those three, which
    is this with ``-1`` in place of ``R``). Linear in ``rotation``, which may be
    a tracer; nothing is read off the field, so a texture that is not collinear
    turns as it is, with every angle between its moments kept.
    """
    field = jnp.asarray(field)
    if field.shape[0] != 4:
        raise ValueError(
            f"rotate_texture wants a four-channel field (charge and three "
            f"magnetization components), got {field.shape[0]} channels"
        )
    rotation = jnp.asarray(rotation, dtype=field.real.dtype)
    moment = jnp.tensordot(rotation, field[1:4], axes=(1, 0))
    return jnp.concatenate([field[:1], moment])


def cross_matrix(vector):
    """``[w]x``, the antisymmetric matrix with ``[w]x v = w x v``."""
    w = jnp.asarray(vector)
    zero = jnp.zeros_like(w[0])
    return jnp.stack([
        jnp.stack([zero, -w[2], w[1]]),
        jnp.stack([w[2], zero, -w[0]]),
        jnp.stack([-w[1], w[0], zero]),
    ])


def rotation_near(omega, base):
    """``(1 + [w]x) R0``: the rotation ``w`` about the orientation ``R0``.

    Exact at ``w = 0`` in value and in first derivative, which is where every
    caller evaluates it (the module docstring says why a norm of ``w`` must not
    appear). It is not a rotation away from ``w = 0`` and is not used there.
    """
    base = jnp.asarray(base)
    return (jnp.eye(3, dtype=base.dtype) + cross_matrix(omega)) @ base


def band_energy_at_rotation(calculation, states, weights, texture, base, omega,
                            becsum=()):
    """``sum_n w_n <psi_n | H(R) | psi_n>`` at frozen ``states``, ``R = (1 + [w]x) R0``.

    ``texture`` is the four-channel density at the reference orientation, the
    one ``R = 1`` means, and ``base`` is ``R0``, the orientation the states were
    diagonalised at. ``becsum`` is, on PAW, each species' one-centre occupations
    at that same reference orientation, turned with the density. Evaluated at
    ``omega = 0`` it must reproduce ``sum w eps`` of those states, which is the
    same check :func:`band_energy_at_angle` makes.
    """
    texture = jnp.asarray(texture)
    omega = jnp.asarray(omega, dtype=texture.real.dtype)
    rotation = rotation_near(omega, jnp.asarray(base, dtype=texture.real.dtype))
    return _band_energy(calculation, states, weights,
                        rotate_texture(texture, rotation),
                        _turned_becsum(becsum, rotation))


def _turned_becsum(becsum, rotation) -> tuple:
    """Every species' ``becsum`` turned by ``rotation``; ``()`` stays ``()``."""
    return tuple(None if values is None else rotate_texture(values, rotation)
                 for values in becsum)


def orientation_torque(calculation, states, weights, texture, base,
                       k_batch: int | None | str = "default", becsum=()):
    """``-dF/dw`` at ``w = 0``: the torque on the whole texture, in Ry per radian.

    Three cartesian components, one per generator of a rigid rotation about the
    orientation ``base``. The sign is :func:`torque_at_angle`'s, so for a
    collinear texture turning in the plane ``(e1, e2)`` that function's torque
    is ``orientation_torque(...) . (e1 x e2)``. It is the derivative of the free
    energy ``sum w eps - TS``, for the reason P60 measured: a Hellmann-Feynman
    derivative at frozen occupations does not see the occupations' own change,
    which the entropy cancels.

    ``k_batch`` chunks the backward pass exactly as it does for
    :func:`torque_at_angle`, and ``None`` takes the whole k axis in one pass.
    ``becsum`` is, on PAW, the one-centre occupations at the reference
    orientation, turned with the density inside the differentiated energy so
    that the one-centre coefficients' share of the torque is in it.
    """
    from defumat.batching import resolve_k_batch

    texture = jnp.asarray(texture)
    base = jnp.asarray(base, dtype=texture.real.dtype)
    origin = jnp.zeros(3, dtype=texture.real.dtype)
    resolved = resolve_k_batch(k_batch)
    nk = int(jnp.asarray(states).shape[1])
    if resolved is None or resolved >= nk:
        def energy(value):
            return band_energy_at_rotation(
                calculation, states, weights, texture, base, value, becsum
            )

        return -np.asarray(jax.grad(energy)(origin))

    def build(value):
        rotation = rotation_near(value, base)
        return rotate_texture(texture, rotation), _turned_becsum(becsum, rotation)

    _, slope = _chunked_value_and_grad(
        calculation, states, weights, build, origin, int(resolved)
    )
    return -np.asarray(slope)
