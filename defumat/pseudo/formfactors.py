"""Radial pseudopotential data transformed into reciprocal space.

Every pseudopotential quantity enters the plane-wave code as a function of
``|G|`` (or ``|k+G|``) obtained by a radial integral. QE precomputes each of
these on a uniform ``q`` grid with spacing ``dq = 0.01`` and interpolates with a
cubic polynomial; here the integral is evaluated directly at the ``|G|`` actually
needed.

That choice is deliberate. It is slightly more accurate (no interpolation error),
and more importantly it keeps the result a differentiable function of ``q`` --
and therefore of ``k`` and of the cell. A spline table would break the chain that
makes the velocity operator fall out of ``jacfwd`` of ``H(k)`` (rule D2). The
price is arithmetic, mitigated by chunking; if it ever matters, the replacement
is a *differentiable* interpolation, not a lookup.

Conventions follow ``upflib``: ``vloc_mod.f90``, ``rhoat_mod.f90``,
``rhoc_mod.f90`` and ``beta_mod.f90``. Units are Rydberg atomic units, ``q`` in
1/bohr, volumes in bohr^3.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf

from defumat.pseudo.radial import simpson_weights, spherical_bessel
from defumat.pseudo.upf import Pseudopotential
from defumat.units import E2, FPI

__all__ = [
    "local_potential_of_g",
    "atomic_charge_of_g",
    "core_charge_of_g",
    "projector_form_factors",
    "atomic_form_factors",
    "CHUNK",
    "RADIAL_CHUNK_BYTES",
    "radial_chunk",
]

#: The most q values transformed at once. The intermediate is (chunk, mesh); the
#: chunk a transform actually takes is :func:`radial_chunk`'s, which this caps.
CHUNK = 4096

#: What one ``(chunk, mesh)`` float64 integrand may occupy, from which
#: :func:`radial_chunk` sizes the chunk against the dataset's mesh.
#:
#: **Sized against the derivatives, not the integrand.** A transform's forward
#: pass holds a few such arrays, and that is what "a few tens of MB" at
#: :data:`CHUNK` was; the strain's derivatives hold many more of them at once
#: (the spherical Bessel function's temporaries, its tangent, the rematted body's
#: recomputation). Measured on ultrasoft AlAs at ``ecutrho = 200`` (841-point
#: mesh, 14211 dense G-vectors), the compiled temporaries of the augmented
#: density's strain derivatives, CPU, ``memory_analysis()``: at 4096 values the
#: ``jvp`` 270.0 MB, the gradient 924.7, the ``jvp`` of the gradient 1550.9; at
#: 1024 values 130.1, 102.8 and 293.7; at 256 values 128.0, 102.8 and 185.6. The
#: local potential's ``jvp`` of the gradient 457.1, 135.7 and 39.6
#: (``PERFORMANCE.md``, "The radial transforms' chunk, sized from the mesh").
#:
#: Chosen on an RTX A2000 (5 to 8 per cent of a strained call's time there) and a
#: CPU (none). A float64 card may find the smaller transforms' launches cost more
#: than that; ``DEFUMAT_RADIAL_CHUNK`` is the dial.
RADIAL_CHUNK_BYTES = 8 * 1024**2

#: The fewest q values a chunk takes, whatever the mesh.
MIN_CHUNK = 256


def radial_chunk(mesh: int) -> int:
    """How many ``q`` values one radial transform takes at a time, for a ``mesh``.

    :data:`RADIAL_CHUNK_BYTES` over one row of the ``(chunk, mesh)`` float64
    integrand, between :data:`MIN_CHUNK` and :data:`CHUNK`: 1246 values on an
    841-point mesh, 1054 on bismuth's 995. ``DEFUMAT_RADIAL_CHUNK`` overrides it
    with a count, read when a kernel is first traced. Python arithmetic on a
    static shape, so a compiled kernel sees a constant.
    """
    import os

    value = os.environ.get("DEFUMAT_RADIAL_CHUNK")
    if value:
        return max(1, int(value))
    return int(min(CHUNK, max(MIN_CHUNK, RADIAL_CHUNK_BYTES // (8 * max(1, mesh)))))

# The kernels below are module-level and jitted rather than closures defined
# per call. Each radial transform is ~30 elementwise operations on a (nq, mesh)
# intermediate; dispatched eagerly, XLA compiles and launches every one of them
# separately, which is where most of a cold run's setup time went. As one
# compiled unit they fuse into a single pass over the intermediate, and the
# compilation is cached across species and across calculations -- a closure
# would be a new callable each time and so a new compilation each time.


def _radial_values(values: jnp.ndarray) -> jnp.ndarray:
    """``values``, kept out of the reduction that consumes them.

    Every radial transform here is an elementwise integrand -- a spherical
    Bessel function of ``q r`` times a tabulated function -- reduced against the
    quadrature weights, and XLA's GPU backend fuses the whole integrand into the
    reduction as one ``input_reduce_fusion``, which it then takes minutes to
    compile. On an RTX A2000, bismuth's relativistic dataset at 28572 values of
    ``q`` on a 995-point mesh: the projector transform compiled in 74.6 s at
    ``l = 0`` and 198 s at ``l = 1``, and a 20-atom spin-orbit cell spent over ten
    minutes of its setup there. Through an ``optimization_barrier`` the integrand
    is one elementwise kernel and the reduction a matrix-vector product: 0.3 and
    0.1 s to compile, 21.1 against 21.9 and 35.7 against 37.5 ms to run, the same
    numbers to the last bit; on a CPU, where both compile in a fraction of a
    second, the same bits and the same time. The barrier is the identity and
    differentiates as one.
    """
    return jax.lax.optimization_barrier(values)


def _scan_rows(block, q: jnp.ndarray, mesh: int) -> jnp.ndarray:
    """``block(q)`` in pieces of :func:`radial_chunk` values, walked by a rematted scan.

    Called *inside* a jitted kernel, so ``block`` may close over that kernel's
    arguments. ``block`` maps ``(n,)`` values of ``q`` to ``(..., n)``.

    **Why a scan and a remat rather than a loop.** These transforms were a
    Python loop of jitted calls, which bounds the forward pass and nothing
    else: under ``jax.grad`` every chunk's ``(chunk, mesh)`` residuals stay on
    the tape, and ``Calculation.at_strain`` rebuilds ``V_loc``, the core charge
    and the projectors against the strained ``|G|`` inside the stress's
    gradient. With the body under ``jax.checkpoint`` the tape holds the ``q``
    chunks and the backward pass recomputes one chunk at a time -- the pattern
    ``augmentation._qrad_kernel`` took first (``OPEN.md`` S4, ``PLAN.md``
    P112). Measured on ``bn-ldau-noncol.in``'s stress gradient on the GTX 1060:
    see ``PERFORMANCE.md``, "The stress tape".

    **The chunks are exactly** :func:`radial_chunk` **wide**, and only the last
    is padded, with ``q = 0`` -- a value every transform here
    already meets at ``G = 0`` and is finite at, in value and derivative. So
    every row but the last chunk's is computed at the shape it always was --
    which is not bit-identity: inside the scan XLA fuses the body differently,
    and on N, B and Si datasets at 100 to 43903 values of ``q`` 53 of 68
    arrays came out identical and the rest within 2.2e-16 relative, a gradient
    through all of them within 4.9e-15.
    """
    nq = q.shape[0]
    chunk = radial_chunk(mesh)
    if nq <= chunk:
        return block(q)
    nchunks = -(-nq // chunk)
    padded = jnp.pad(q, (0, nchunks * chunk - nq)).reshape(nchunks, chunk)

    @jax.checkpoint
    def body(carry, rows):
        return carry, block(rows)

    _, blocks = jax.lax.scan(body, None, padded)  # (nchunks, ..., CHUNK)
    values = jnp.moveaxis(blocks, 0, -2)
    return values.reshape(values.shape[:-2] + (-1,))[..., :nq]


def _truncated(pseudo: Pseudopotential):
    """The mesh QE integrates over, with its Simpson weights."""
    msh = pseudo.msh
    r = jnp.asarray(pseudo.r[:msh])
    weights = simpson_weights(jnp.asarray(pseudo.rab[:msh]))
    return r, weights, msh


def local_potential_of_g(pseudo: Pseudopotential, q, omega: float) -> jnp.ndarray:
    """Fourier transform of the local potential, ``V_loc(q)`` in Ry.

    The bare potential is long-ranged (``-Z e^2 / r``) and its transform diverges
    as ``1/q^2``, so it cannot be integrated numerically. QE's trick, reproduced
    here, is to add ``Z e^2 erf(r)/r`` inside the integral -- making the
    integrand short-ranged -- and subtract that function's analytic transform
    ``4 pi Z e^2 exp(-q^2/4) / (Omega q^2)`` outside it.

    At ``q = 0`` the remaining integral is the ``alpha Z`` term: the average of
    the potential over the cell, finite only because the divergence cancels
    against the Hartree and Ewald ``G = 0`` terms.
    """
    r, weights, msh = _truncated(pseudo)
    vloc = jnp.asarray(pseudo.vloc[:msh])
    z = pseudo.z_valence

    # Short-ranged integrand for q > 0: r^2 [V(r) + Z e^2 erf(r) / r]
    short = r * vloc + z * E2 * erf(r)
    # The q = 0 term is *not* the q -> 0 limit of that expression, and QE says so
    # in as many words. The erf is part of the splitting that makes the q > 0
    # integral converge; at q = 0 what is wanted is the average of the potential
    # with its bare Coulomb tail removed, so the screening function is 1, not
    # erf(r). Using the erf form here shifts every eigenvalue by a constant --
    # a convincingly self-consistent calculation with the wrong absolute energy.
    at_zero = r * (r * vloc + z * E2)

    return _vloc_kernel(jnp.atleast_1d(jnp.asarray(q)), r, weights, short, at_zero, z, omega)


@jax.jit
def _vloc_kernel(q, r, weights, short, at_zero, z, omega):
    return _scan_rows(
        lambda qq: _vloc_block(qq, r, weights, short, at_zero, z, omega), q,
        r.shape[0],
    )


def _vloc_block(qq, r, weights, short, at_zero, z, omega):
    qq = qq[:, None]
    small = qq[:, 0] < 1e-8
    safe = jnp.where(qq < 1e-8, 1.0, qq)
    integrand = jnp.where(small[:, None], at_zero[None, :],
                          short[None, :] * jnp.sin(safe * r[None, :]) / safe)
    value = _radial_values(integrand) @ weights * FPI / omega

    analytic = FPI / omega * z * E2 * jnp.exp(-safe[:, 0] ** 2 * 0.25) / safe[:, 0] ** 2
    return jnp.where(small, value, value - analytic)


def atomic_charge_of_g(pseudo: Pseudopotential, q, omega: float) -> jnp.ndarray:
    """Transform of the atomic charge density used to start the SCF.

    ``PP_RHOATOM`` is tabulated as ``4 pi r^2 rho(r)``, so the transform is a
    plain ``j_0`` integral and ``rho(q=0) = Z_valence / Omega``.
    """
    if pseudo.rho_atom is None:
        raise ValueError(f"{pseudo.element}: the UPF file has no PP_RHOATOM section")

    r, weights, msh = _truncated(pseudo)
    rho = jnp.asarray(pseudo.rho_atom[:msh])

    return _rhoat_kernel(jnp.atleast_1d(jnp.asarray(q)), r, weights, rho, omega)


@jax.jit
def _rhoat_kernel(q, r, weights, rho, omega):
    return _scan_rows(lambda qq: _rhoat_block(qq, r, weights, rho, omega), q,
                      r.shape[0])


def _rhoat_block(qq, r, weights, rho, omega):
    qq = qq[:, None]
    small = qq[:, 0] < 1e-8
    safe = jnp.where(qq < 1e-8, 1.0, qq)
    argument = safe * r[None, :]
    integrand = jnp.where(
        small[:, None], rho[None, :], rho[None, :] * spherical_bessel(0, argument)
    )
    return _radial_values(integrand) @ weights / omega


def core_charge_of_g(pseudo: Pseudopotential, q, omega: float) -> jnp.ndarray:
    """Transform of the nonlinear core-correction charge (``PP_NLCC``).

    Unlike ``PP_RHOATOM`` this is tabulated as ``rho_c(r)`` itself, so the
    ``4 pi r^2`` measure appears explicitly.
    """
    if pseudo.rho_core is None:
        raise ValueError(f"{pseudo.element}: the UPF file has no PP_NLCC section")

    r, weights, msh = _truncated(pseudo)
    rho = jnp.asarray(pseudo.rho_core[:msh])

    return _rhocore_kernel(jnp.atleast_1d(jnp.asarray(q)), r, weights, rho, omega)


@jax.jit
def _rhocore_kernel(q, r, weights, rho, omega):
    return _scan_rows(lambda qq: _rhocore_block(qq, r, weights, rho, omega), q,
                      r.shape[0])


def _rhocore_block(qq, r, weights, rho, omega):
    argument = qq[:, None] * r[None, :]
    integrand = FPI * r[None, :] ** 2 * rho[None, :] * spherical_bessel(0, argument)
    return _radial_values(integrand) @ weights / omega


def projector_form_factors(pseudo: Pseudopotential, q, omega: float) -> jnp.ndarray:
    """Radial parts ``f_l(q)`` of the nonlocal projectors, shaped ``(nbeta, nq)``.

    ``PP_BETA`` is tabulated as ``r beta_l(r)``, so the transform is
    ``4 pi / sqrt(Omega) * int dr r beta_l(r) j_l(qr)``. The ``1/sqrt(Omega)``
    rather than ``1/Omega`` is because the projectors multiply wavefunctions,
    which carry their own normalisation.

    Every projector is integrated over the *same* range, the species-wide
    ``kkbeta`` of ``upflib/beta_mod.f90``, rather than each over its own
    ``cutoff_radius_index``. The two differ for a PAW dataset, where ``kkbeta``
    is widened to cover the augmentation sphere -- and the tabulated ``beta``
    is truncated abruptly rather than tapering to zero, so it is still of order
    1e-3 at its own cutoff. Integrating over the shorter range would drop a
    contribution QE keeps.
    """
    # ``jnp`` rather than ``np``, for the reason :func:`atomic_form_factors`
    # gives: the stress differentiates through this and ``omega`` arrives as a
    # tracer (P11), and a ``np.sqrt`` of a tracer is a ``TypeError`` at best and
    # a frozen constant at worst.
    prefactor = FPI / jnp.sqrt(omega)
    q = jnp.atleast_1d(jnp.asarray(q))

    cutoff = pseudo.kkbeta
    r = jnp.asarray(pseudo.r[:cutoff])
    weights = simpson_weights(jnp.asarray(pseudo.rab[:cutoff]))

    rows = []
    for projector in pseudo.projectors:
        beta = jnp.asarray(projector.beta[:cutoff])
        l = projector.l

        rows.append(_beta_kernel(q, r, weights, beta, prefactor, l))

    if not rows:
        return jnp.zeros((0,) + q.shape)
    return jnp.stack(rows, axis=0)


def projector_origin_slopes(pseudo: Pseudopotential, omega) -> jnp.ndarray:
    """``lim_{q -> 0} f_l(q) / q^l`` for every radial projector, shaped ``(nbeta,)``.

    The small-``q`` limit of :func:`projector_form_factors`, taken analytically
    rather than by evaluating the transform at a small ``q``. ``j_l(x)`` goes as
    ``x^l / (2l+1)!!``, so

        f_l(q) -> (4 pi / sqrt(Omega)) q^l / (2l+1)!! int dr (r beta)(r) r^{l+1},

    on the same ``kkbeta`` range and with the same Simpson weights the transform
    itself uses, so the two agree by construction rather than by luck. Measured
    on ``Si.pz-vbc``'s ``l = 1`` channel: **0.2291291689** here against
    0.2291291689 from a straight-line fit to the table at ``q = 1e-5`` to
    ``4e-5``, agreeing to 4.5e-11.

    **What wants this is the derivative and not the value.** At ``q = 0`` the
    product ``f_l(q) Y_lm(qhat)`` is zero for every ``l > 0`` and the transform
    gives that correctly; what it cannot give is the product's *tangent*, since
    both factors are guarded at the origin separately (see
    ``projectors._origin_tangent``). For ``l = 1`` that tangent is
    ``sqrt(3/4pi)`` times the number returned here.

    ``jnp`` throughout, for the reason :func:`projector_form_factors` gives:
    ``omega`` arrives as a tracer under a stress derivative.
    """
    return FPI / jnp.sqrt(omega) * _origin_integrals(pseudo)


def _origin_integrals(pseudo: Pseudopotential) -> jnp.ndarray:
    """The cell-independent half of :func:`projector_origin_slopes`.

    ``int dr (r beta)(r) r^(l+1) / (2l+1)!!`` for every projector, as **one**
    matrix-vector product rather than a loop of them, because ``at_kcart``
    rebuilds the projectors inside a ``jvp`` once per velocity call and the
    dispatch count here is paid on a hot differentiable path.

    **Nothing this reads is ever a tracer, so nothing forces** ``jnp`` **here.**
    Every array is a NumPy array from the UPF file, on every path:
    :class:`~defumat.pseudo.upf.Pseudopotential` is a frozen dataclass and not
    a pytree, so it crosses a ``jit`` or a ``grad`` as a closed-over constant,
    and ``Calculation.at_strain`` hands ``self.pseudos`` to its builders
    untouched. The witness is on this function's own path: inside that trace,
    ``build_projector_core`` fingerprints each dataset with
    ``projectors._projector_dataset_key``, which is ``np.asarray`` of
    ``pseudo.r``, ``pseudo.rab`` and every ``beta``, a few lines before it
    reaches this function through ``_origin_slopes``, and it could not do that
    if any of them were a tracer. What a stress derivative traces is the cell,
    and the volume enters outside this function, in
    :func:`projector_origin_slopes` and in ``projectors._origin_slopes``, where
    ``4 pi / sqrt(Omega)`` is traced under ``at_strain`` and carries no
    tangent on the velocity ``jvp``. So the reason :func:`projector_form_factors` is
    written in ``jnp``, a traced ``omega`` and a traced ``q``, does not reach
    here, and a version written wholly in NumPy would run on every path.

    **It stays** ``jnp`` **so that its bytes do not move**: NumPy's matmul and
    XLA's dot need not sum in the same order, and every velocity tangent at
    ``k + G = 0`` is built from these slopes. The one constraint is to be all
    one or all the other. A *mixed* form, NumPy applied to a ``jnp``
    intermediate, is what most likely failed the stress leg of the
    bit-identity check that commit 5a8d267 describes: the stress gradient is
    ``jax.jit(jax.grad(...))`` (``stress/autodiff.py``'s ``_energy_gradient``),
    and under a ``jit`` a ``jnp`` operation is staged even on constant
    arguments, so ``np.asarray`` of its result is a
    ``TracerArrayConversionError`` although nothing upstream is traced. That
    mechanism is inferred from how JAX stages a ``jit`` and from the commit
    message, which blames traced pseudopotentials; it has not been reproduced.
    """
    cutoff = pseudo.kkbeta
    if not pseudo.projectors:
        return jnp.zeros((0,))
    r = jnp.asarray(pseudo.r[:cutoff])
    weights = simpson_weights(jnp.asarray(pseudo.rab[:cutoff]))
    # ``l`` is static -- it comes from the file's header, not from an array --
    # so the powers and the double factorials are host constants.
    ls = np.asarray([projector.l for projector in pseudo.projectors])
    factorials = np.asarray([
        float(np.prod(np.arange(3, 2 * l + 2, 2))) for l in ls
    ])
    beta = jnp.stack([
        jnp.asarray(projector.beta[:cutoff]) for projector in pseudo.projectors
    ])
    # ``r ** (l + 1)`` and not ``r ** l``: ``j_l(qr)`` contributes ``r^l`` and
    # the transform carries an ``r`` of its own beside ``(r beta)``, which is
    # the extra factor in ``_beta_kernel``'s integrand.
    powers = r[None, :] ** jnp.asarray(ls + 1)[:, None]
    return (beta * powers) @ weights / jnp.asarray(factorials)


def atomic_form_factors(pseudo: Pseudopotential, q, omega) -> jnp.ndarray:
    """Radial parts of the pseudo-atomic orbitals, shaped ``(nwfc, nq)``.

    ``upflib/atwfc_mod.f90``. ``PP_CHI`` is tabulated as ``r chi(r)``, exactly as
    ``PP_BETA`` is, so this is the same integral as
    :func:`projector_form_factors` with the same prefactor -- the difference is
    the mesh it runs over (QE's 10-bohr truncation rather than each projector's
    own cutoff) and that orbitals with negative occupation are skipped, as QE
    skips them.
    """
    # ``jnp`` rather than ``np``: the DFT+U force differentiates through this
    # (the Hubbard projectors are atomic orbitals, and they move with the
    # atoms), and inside that trace ``cell.volume`` is a tracer -- a nested
    # ``jax.jit`` is inlined into the enclosing trace, so even a constant cell
    # arrives abstract.
    prefactor = FPI / jnp.sqrt(omega)
    q = jnp.atleast_1d(jnp.asarray(q))
    r, weights, _ = _truncated(pseudo)

    rows = [
        _beta_kernel(q, r, weights, jnp.asarray(orbital.chi[: r.shape[0]]),
                     prefactor, orbital.l)
        for orbital in pseudo.orbitals
        if orbital.occupation >= 0.0
    ]
    if not rows:
        return jnp.zeros((0,) + q.shape)
    return jnp.stack(rows, axis=0)


@partial(jax.jit, static_argnames=("l",))
def _beta_kernel(q, r, weights, beta, prefactor, l):
    return _scan_rows(
        lambda qq: _beta_block(qq, r, weights, beta, prefactor, l), q,
        r.shape[0],
    )


def _beta_block(qq, r, weights, beta, prefactor, l):
    argument = qq[:, None] * r[None, :]
    integrand = beta[None, :] * spherical_bessel(l, argument) * r[None, :]
    return _radial_values(integrand) @ weights * prefactor
