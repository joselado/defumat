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
the derivative is taken of it by automatic differentiation. No expression is
derived for any contribution. The literature's torque *is* such an expression --
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

**Forward mode, one ``jax.jvp`` per generator** (``OPEN.md`` Part XXII item 2,
the user's decision of 2026-10-10). The reverse pass this module used until then
kept a tape of every band's real-space block in flight, and on the fifteen-cell
NiBr2 helix it asked an H200 for 33.37 GiB more at one k-point; on nine atoms of
spinor PAW NiI2 it ran out of a 6.96 GiB pool on D22's card. A ``jvp`` holds one
primal and one tangent of each intermediate and frees both as it goes, so the
derivative costs the working set of the energy it differentiates, twice over,
for each of the three generators (one for the angle). Both modes differentiate
the same function, and nothing on its path carries a ``custom_vjp`` or a host
callback, while every ``custom_jvp`` on it (``pseudo/formfactors.py``,
``pseudo/augmentation.py``) defines the forward rule reverse mode transposes, so
the two agree to round-off whatever those rules are; what can catch a wrong
rule is the closed form and a central difference, which the tests hold the
forward torque to. On PAW the input ``becsum`` is turned as a vector inside the
differentiated energy on every route of this module; turning it through the
states is :func:`defumat.scf.driver._onecenter_torque`'s, with the coupling on,
and has no counterpart here, which is exact for the source Route A takes at
``soc_scale = 0`` (2.8e-16, P122).

**Where the peak is, and the dial that sets it.** Under a ``jvp`` the k axis is
a ``sum_k`` at ``k_batch`` and the band axis is :func:`~defumat.batching.map_bands`
inside the Hamiltonian at its ``band_batch``, and neither stacks anything, so
both dials are live, where under ``jax.grad`` the band dial was inert (the scan
stacked its residuals). The largest term is the local potential's real-space
block, primal and tangent: on nine atoms of NiI2 (116 spinor bands, a 90x30x192
grid, one k-point, D22's A2000) the derivative raised the card's peak by 1.40,
1.88 and 2.89 GiB at a band batch of 16, 32 and 58, about 36 MiB a band, which
is 2.3 spinor fields of ``npol x N_smooth`` complex numbers a band in flight, so
roughly ``2.3 x k_batch x band_batch x npol x N_smooth`` complex numbers on top
of the energy's own fixed share. ``band_batch`` is the one the
:class:`~defumat.scf.driver.Calculation` the states came from carries, chosen
to fit the card in memory mode (58 of 116 there), or ``DEFUMAT_BAND_BATCH``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from defumat.scf.continuation import _axis, _collinear_axis
from defumat.batching import sum_k, upload
from defumat.eager import compiled, compiled_function

__all__ = [
    "band_energy_at_angle",
    "band_energy_at_rotation",
    "cross_matrix",
    "frozen_generators",
    "orientation_torque",
    "project_out",
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
    field's share of the torque into the same derivative as the grid's; built
    once outside, it would be a constant the gradient passes through, and the
    torque would miss it.
    """
    potential = calculation.potential(density, 1.0, None)
    ddd_paw = calculation.onecenter(tuple(becsum))[1] if becsum else None
    return calculation.hamiltonian(potential.v_scf, ddd_paw)[0]


def _band_energy(calculation, states, weights, density, becsum=()):
    """``sum_n w_n <psi_n | H[density, becsum] | psi_n>`` over every k-point at once."""
    hamiltonian = _hamiltonian_of(calculation, density, becsum)

    psi = upload(states)[0]
    occupation = jnp.asarray(weights)[0]

    def one(ik):
        applied = hamiltonian.apply(psi[ik], ik)
        bands = jnp.real(jnp.sum(jnp.conj(psi[ik]) * applied, axis=-1))
        return jnp.sum(occupation[ik] * bands)

    # Called outside any trace once per angle, with a new Hamiltonian each
    # time, so compiled by its structure (:mod:`defumat.eager`); the k axis is
    # a ``sum_k`` rather than a Python loop, which would unroll one Hamiltonian
    # application per k-point into the kept program.
    return compiled(
        lambda indices: sum_k(one, indices, batch=calculation.k_batch),
        jnp.arange(psi.shape[0]))


def _value_and_slopes(calculation, states, weights, build, parameter, tangents,
                      k_batch: int | None):
    """``(E, [dE/dp . t for t in tangents])`` by forward mode, the k axis at ``k_batch``.

    ``build(p)`` is the pair ``(density, becsum)`` the Hamiltonian is made
    from, ``becsum`` empty except on PAW, and ``p`` is a scalar angle or a
    rotation vector. ``tangents`` is a stack of directions of ``p``'s shape, the
    unit angle or the three generators, in the dtype of ``parameter``.

    **One ``jax.jvp`` per tangent, each over the whole k axis.** The energy
    ``sum_k w_k <psi_k|H(p)|psi_k>`` has no term coupling two k-points, since the
    potential comes from ``build`` and not from the states, so it is a
    ``sum_k`` whose chunk is the k dial, and under a ``jvp`` a ``sum_k``
    carries the running primal and tangent and stacks nothing: the reason this
    function used to be a Python loop of per-chunk reverse passes, each
    discarding its tape before the next, is gone, and the Hamiltonian is built
    once a tangent rather than once a chunk. ``None`` maps the whole axis at
    once, as everywhere else.

    The derivative is kept once a process (:mod:`defumat.eager`) with the
    tangent an **argument**, so the three generators and a later call on a new
    ``calculation`` of the same shapes reuse one program; a tangent closed over
    would be a nested constant of the key and three programs.
    """
    psi = upload(states)[0]
    occupation = jnp.asarray(weights)[0]
    nk = int(psi.shape[0])

    def energy(value):
        hamiltonian = _hamiltonian_of(calculation, *build(value))

        def one(ik):
            applied = hamiltonian.apply(psi[ik], ik)
            bands = jnp.real(jnp.sum(jnp.conj(psi[ik]) * applied, axis=-1))
            return jnp.sum(occupation[ik] * bands)

        return sum_k(one, jnp.arange(nk), batch=k_batch)

    def derivative(value, tangent):
        return jax.jvp(energy, (value,), (tangent,))

    program = compiled_function(derivative, parameter, tangents[0])
    value, slopes = None, []
    for tangent in tangents:
        value, slope = program(parameter, tangent)
        slopes.append(np.asarray(slope))
    return float(value), np.stack(slopes)


def _resolved_chunk(k_batch, nk: int) -> int | None:
    """The k dial for the forward derivative: ``None`` or a chunk below ``nk``."""
    from defumat.batching import resolve_k_batch

    resolved = resolve_k_batch(k_batch)
    if resolved is None or resolved >= nk:
        return None
    return int(resolved)


def torque_at_angle(calculation, states, weights, density, plane, angle,
                    k_batch: int | None | str = "default"):
    """``-dE/dtheta``: the torque on the moment, in Ry per radian.

    The sign is the mechanical one -- a positive torque turns the moment
    towards larger ``theta`` -- so for ``E = K1 sin^2(theta)`` this returns
    ``-K1 sin(2 theta)`` and a measurement at ``pi/4`` gives ``-K1``.

    ``k_batch`` is how many k-points the derivative holds at once
    (:func:`_value_and_slopes`); ``None`` takes the whole axis in one ``vmap``
    and an integer, or the dial's own ``"default"``, walks it in chunks. The
    derivative is forward mode, so the chunk and the Hamiltonian's band dial
    both bound its working set (the module docstring has the formula); until
    2026-10-10 it was a reverse pass whose tape no band dial reached, which is
    what ``OPEN.md`` Part XXII item 2 records.
    """
    nk = int(np.shape(states)[1])
    dtype = jnp.asarray(density).real.dtype

    def build(value):
        return rotated_density(density, _direction(value, plane[0], plane[1])), ()

    _, slope = _value_and_slopes(
        calculation, states, weights, build, jnp.asarray(float(angle), dtype=dtype),
        jnp.ones((1,), dtype=dtype), _resolved_chunk(k_batch, nk),
    )
    return -float(slope[0])


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

    ``k_batch`` sets the k-points in flight exactly as it does for
    :func:`torque_at_angle`, and ``None`` takes the whole k axis at once; the
    three components are three ``jax.jvp``, one per generator.
    ``becsum`` is, on PAW, the one-centre occupations at the reference
    orientation, turned with the density inside the differentiated energy so
    that the one-centre coefficients' share of the torque is in it.
    """
    texture = jnp.asarray(texture)
    dtype = texture.real.dtype
    base = jnp.asarray(base, dtype=dtype)
    nk = int(np.shape(states)[1])

    def build(value):
        rotation = rotation_near(value, base)
        return rotate_texture(texture, rotation), _turned_becsum(becsum, rotation)

    _, slopes = _value_and_slopes(
        calculation, states, weights, build, jnp.zeros(3, dtype=dtype),
        jnp.eye(3, dtype=dtype), _resolved_chunk(k_batch, nk),
    )
    return -np.asarray(slopes)


def frozen_generators(density, phase: bool = True, tolerance: float = 1.0e-3) -> np.ndarray:
    """The rotations a texture's orientation relaxation should not move along, ``(k, 3)``.

    Read from the moment tensor ``M_ab = integral of m_a m_b`` (``ORIENTATION-NEXT.md``,
    the coordinate): a **collinear** texture (the two smaller eigenvalues below
    ``tolerance`` of the largest) returns its axis, the turn about which moves
    nothing; a **coplanar** one (the smallest below it) returns its plane's
    normal when ``phase`` is set, the turn about which is the texture's phase in
    its own plane, flat for a spiral and nearly flat for a commensurate one, and
    a coordinate an optimizer lets noise push; a noncoplanar one returns none.
    Unit vectors in the frame of ``density``.
    """
    moment = np.asarray(jnp.real(jnp.asarray(density)[1:4])).reshape(3, -1)
    values, vectors = np.linalg.eigh(moment @ moment.T)
    if values[2] <= 0.0:
        return np.zeros((0, 3))
    if values[0] + values[1] <= tolerance * values[2]:
        return vectors[:, 2:3].T.copy()
    if phase and values[0] <= tolerance * values[2]:
        return vectors[:, 0:1].T.copy()
    return np.zeros((0, 3))


def project_out(vector, directions) -> np.ndarray:
    """``vector`` with its components along each unit row of ``directions`` removed."""
    vector = np.asarray(vector, dtype=float).copy()
    for direction in np.asarray(directions, dtype=float).reshape(-1, 3):
        vector -= float(vector @ direction) * direction
    return vector
