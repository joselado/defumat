"""Occupation numbers and the Fermi level.

Two regimes, following ``PW/src/weights.f90`` and its callees:

* **fixed** occupations for a system with a gap: the lowest ``nelec/2`` bands
  are full and the rest empty.
* **smeared** occupations for a metal, where an integer filling would make the
  density discontinuous as bands cross the Fermi level between iterations. The
  Fermi level is then found by bisection on the total electron count.

Weights returned here are QE's ``wg``: the k-point weight times the occupation,
summing to the number of electrons.

**Spin.** Eigenvalues and weights carry a leading channel axis. With one Fermi
level the search runs over both channels at once -- QE simply passes its
``2 nks`` k-list, whose weights already sum to one per channel, so nothing about
the bisection changes. With ``tot_magnetization`` given there are *two*
independent Fermi levels, one per channel, each solving for its own electron
count, and the ``-TS`` term is the sum of the two (``weights.f90``).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf, erfc

from pypresso.units import SQRT_PI

__all__ = ["fixed_occupations", "smeared_occupations", "fermi_level", "bisect_fermi",
           "smearing_entropy", "spin_electron_counts",
           "input_occupations", "wgauss", "w0gauss", "w1gauss",
           "smearing_order", "tetrahedra_for", "tetrahedron_occupations"]


def fixed_occupations(eigenvalues: jnp.ndarray, weights: jnp.ndarray, nelec: float):
    """Fill the lowest ``nelec/2`` bands completely.

    Returns ``(wg, homo, lumo)``. Raises if the electron count is odd, which
    means the system needs either spin polarisation or smearing.
    """
    nspin, _, nbnd = eigenvalues.shape
    if nspin != 1:
        raise NotImplementedError(
            "occupations='fixed' with nspin = 2 is not implemented; QE fills the "
            "two channels from tot_magnetization or from a shared Fermi level, "
            "and no committed benchmark exercises either, so it is refused "
            "rather than guessed. Use occupations='from_input' or 'smearing'"
        )
    occupied = nelec / 2.0
    if abs(occupied - round(occupied)) > 1e-8:
        raise ValueError(
            f"{nelec} electrons cannot fill spin-degenerate bands; use smearing or nspin=2"
        )
    occupied = int(round(occupied))
    if occupied > nbnd:
        raise ValueError(f"{nelec} electrons need {occupied} bands but only {nbnd} were computed")

    occupation = jnp.arange(nbnd) < occupied
    wg = weights[None, :, None] * occupation[None, None, :]

    homo = jnp.max(eigenvalues[0, :, occupied - 1])
    lumo = jnp.min(eigenvalues[0, :, occupied]) if occupied < nbnd else None
    return wg, homo, lumo


def wgauss(x: jnp.ndarray, ngauss: int) -> jnp.ndarray:
    """QE's ``wgauss``: the occupation as a function of ``x = (E_F - e)/degauss``.

    Transcribed from ``Modules/wgauss.f90``, including its sign convention --
    ``x`` is positive for an occupied state. Writing it the other way round and
    flipping signs by hand works for the symmetric Gaussian and silently breaks
    the asymmetric Methfessel-Paxton and cold smearings.

    ``ngauss``: 0 Gaussian, n > 0 Methfessel-Paxton of order n, -1 cold
    (Marzari-Vanderbilt), -99 Fermi-Dirac.
    """
    if ngauss == -99:  # Fermi-Dirac
        return 1.0 / (1.0 + jnp.exp(jnp.clip(-x, -200.0, 200.0)))

    if ngauss == -1:  # cold smearing
        xp = x - 1.0 / np.sqrt(2.0)
        arg = jnp.minimum(200.0, xp**2)
        return 0.5 * erf(xp) + 1.0 / np.sqrt(2.0 * np.pi) * jnp.exp(-arg) + 0.5

    occupation = 0.5 * erfc(-x)
    if ngauss == 0:
        return occupation

    # Methfessel-Paxton: Gaussian plus Hermite corrections. Two details worth
    # copying exactly rather than reconstructing: the coefficient recursion
    # divides by the loop index i (not by the Hermite counter ni, which they
    # coincide with only on the first pass), and the correction is *subtracted*.
    hd = jnp.zeros_like(x)
    arg = jnp.minimum(200.0, x**2)
    hp = jnp.exp(-arg)
    ni, a = 0, 1.0 / SQRT_PI
    for i in range(1, ngauss + 1):
        hd = 2.0 * x * hp - 2.0 * ni * hd
        ni += 1
        a = -a / (i * 4.0)
        occupation = occupation - a * hd
        hp = 2.0 * x * hd - 2.0 * ni * hp
        ni += 1
    return occupation


def w1gauss(x: jnp.ndarray, ngauss: int) -> jnp.ndarray:
    """QE's ``w1gauss``: the generalised entropy integrand, from ``w1gauss.f90``.

    ``degauss * sum_k w_k sum_i w1gauss(x_ik)`` is the ``-TS`` term QE prints.
    """
    if ngauss == -99:  # Fermi-Dirac
        f = jnp.clip(1.0 / (1.0 + jnp.exp(jnp.clip(-x, -200.0, 200.0))), 1e-300, 1.0 - 1e-16)
        return f * jnp.log(f) + (1.0 - f) * jnp.log(1.0 - f)

    if ngauss == -1:  # cold smearing
        xp = x - 1.0 / np.sqrt(2.0)
        arg = jnp.minimum(200.0, xp**2)
        return 1.0 / np.sqrt(2.0 * np.pi) * xp * jnp.exp(-arg)

    arg = jnp.minimum(200.0, x**2)
    value = -0.5 * jnp.exp(-arg) / SQRT_PI
    if ngauss == 0:
        return value

    hd = jnp.zeros_like(x)
    hp = jnp.exp(-arg)
    ni, a = 0, 1.0 / SQRT_PI
    for i in range(1, ngauss + 1):
        hd = 2.0 * x * hp - 2.0 * ni * hd
        ni += 1
        hpm1 = hp
        hp = 2.0 * x * hd - 2.0 * ni * hp
        ni += 1
        a = -a / (i * 4.0)
        value = value - a * (0.5 * hp + ni * hpm1)
    return value


#: Input-file smearing names -> QE's ``ngauss``.
SMEARING_ORDER = {
    "gaussian": 0, "gauss": 0,
    "methfessel-paxton": 1, "m-p": 1, "mp": 1,
    "marzari-vanderbilt": -1, "cold": -1, "m-v": -1, "mv": -1,
    "fermi-dirac": -99, "f-d": -99, "fd": -99,
}


def smearing_order(smearing: str) -> int:
    try:
        return SMEARING_ORDER[smearing.lower()]
    except KeyError as error:
        raise ValueError(
            f"unknown smearing {smearing!r}; expected one of {sorted(SMEARING_ORDER)}"
        ) from error


#: Bisection steps. 200 halvings take any physical bracket far below the
#: resolution of a float64, so the result is the exact root of the discretised
#: count function and does not depend on this number.
BISECTION_STEPS = 200

#: Newton steps for the Methfessel-Paxton and cold refinement. QE allows 300;
#: Newton on a smooth one-dimensional function converges quadratically and a
#: converged step is a fixed point, so the two counts cannot disagree about
#: where it lands -- only about how much arithmetic is wasted afterwards.
NEWTON_STEPS = 100

#: ``efermig``'s two tolerances: ``eps`` on the electron count throughout, and
#: ``eps_cold_MP`` for deciding that the Newton refinement was good enough.
FERMI_EPS = 1.0e-10
FERMI_EPS_COLD_MP = 1.0e-2


def _count(eigenvalues, weights, degauss, ngauss, ef):
    """``sumkg``: the electron count a given Fermi level would give."""
    occupation = wgauss((ef - eigenvalues) / degauss, ngauss)
    return jnp.sum(weights[:, None] * occupation)


def _bisect(eigenvalues, weights, nelec, degauss, ngauss):
    """Plain bisection on the count function, as a device-side loop.

    The loop runs a fixed number of steps and its branch is a ``where``, so it
    is a ``fori_loop`` and never leaves the device. Written as a Python loop
    with a ``float()`` comparison it cost 200 host round trips *per SCF
    iteration* -- by far the most expensive thing about a metal.

    The bracket is ``+-10 degauss`` beyond the extreme eigenvalues, which is
    ``efermig``'s.
    """

    def step(_, bracket):
        low, high = bracket
        middle = 0.5 * (low + high)
        too_many = _count(eigenvalues, weights, degauss, ngauss, middle) > nelec
        return jnp.where(too_many, low, middle), jnp.where(too_many, middle, high)

    low = jnp.min(eigenvalues) - 10.0 * degauss
    high = jnp.max(eigenvalues) + 10.0 * degauss
    low, high = jax.lax.fori_loop(0, BISECTION_STEPS, step, (low, high))
    return 0.5 * (low + high)


def _newton(eigenvalues, weights, nelec, degauss, ngauss, start):
    """``efermig``'s ``newton_minimization`` of ``(N(Ef) - nelec)^2``.

    The first and second derivatives of the count -- QE's ``sumkg1`` and
    ``sumkg2``, each a hand-written sum over its own tabulated ``w0gauss`` /
    ``w0gauss'`` -- come from ``jax.grad`` of the count itself, so they cannot
    disagree with the occupation function they are supposed to differentiate.

    The step is Newton's on the *squared* residual with the second derivative's
    absolute value in the denominator, which is what makes it a descent step
    towards the nearest root rather than a Newton step on ``N - nelec`` that
    would run away wherever ``N'`` changes sign -- and ``N'`` changing sign is
    the whole problem here.
    """
    residual = lambda ef: _count(eigenvalues, weights, degauss, ngauss, ef) - nelec
    first = jax.grad(residual)
    second = jax.grad(first)

    def step(_, state):
        x, done = state
        value, slope, curvature = residual(x), first(x), second(x)
        numerator = 2.0 * value * slope
        denominator = jnp.abs(2.0 * (slope**2 + value * curvature))
        singular = denominator <= FERMI_EPS
        safe = jnp.where(singular, 1.0, denominator)
        candidate = x - numerator / safe
        moved = jnp.where(done | singular, x, candidate)
        stop = (
            done
            | singular
            | (jnp.abs(moved - x) < FERMI_EPS)
            | (jnp.abs(residual(moved)) < FERMI_EPS)
        )
        return moved, stop

    refined, _ = jax.lax.fori_loop(
        0, NEWTON_STEPS, step, (start, jnp.zeros((), dtype=bool))
    )
    return refined


@partial(jax.jit, static_argnames=("ngauss",))
def bisect_fermi(
    eigenvalues: jnp.ndarray,
    weights: jnp.ndarray,
    nelec: float,
    degauss: float,
    ngauss: int,
) -> jnp.ndarray:
    """The Fermi level, by ``PW/src/efermig.f90``'s algorithm.

    **Bisection alone is not enough, and this is the trap.** Methfessel-Paxton
    and cold occupations *overshoot*: a cold-smeared level reaches 1.07 before
    settling at 1, so the electron count is not monotonic in ``E_F`` and
    ``N(E_F) = nelec`` has several roots. Which one a bisection finds depends on
    its bracket, and the wrong root gives the same occupations to 1e-5 while
    putting ``-TS`` out by 3e-4 Ry -- an error that is invisible in the density
    and shows up only in the total energy. It bites hardest where the count is
    nearly flat, which is exactly a half-metallic channel: nickel's majority
    spin, with ``tot_magnetization`` fixing six electrons in it.

    So QE does what is transcribed here: bisect with a **Gaussian**, which is
    monotonic and has one root, then refine that guess with Newton's method on
    the actual occupation function. The Gaussian level is what selects the
    physical root; the refinement moves it to where the real count is right.
    Bisection with the true function survives only as the fallback QE takes when
    the refinement misses by more than 1e-2 electrons.
    """
    eigenvalues = jnp.asarray(eigenvalues)
    weights = jnp.asarray(weights)

    # Fermi-Dirac is monotonic, so its own bisection is both the guess and the
    # answer; everything else is guessed at with the Gaussian.
    guess_smearing = ngauss if ngauss == -99 else 0
    guess = _bisect(eigenvalues, weights, nelec, degauss, guess_smearing)
    if ngauss in (0, -99):
        return guess

    refined = _newton(eigenvalues, weights, nelec, degauss, ngauss, guess)
    missed = (
        jnp.abs(_count(eigenvalues, weights, degauss, ngauss, refined) - nelec)
        >= FERMI_EPS_COLD_MP
    )
    fallback = _bisect(eigenvalues, weights, nelec, degauss, ngauss)
    return jnp.where(missed, fallback, refined)


def fermi_level(
    eigenvalues: jnp.ndarray,
    weights: jnp.ndarray,
    nelec: float,
    degauss: float,
    smearing: str = "gaussian",
) -> float:
    """Find ``E_F`` by bisection on the total electron count."""
    return float(bisect_fermi(eigenvalues, weights, nelec, degauss, smearing_order(smearing)))


def smeared_occupations(
    eigenvalues: jnp.ndarray,
    weights: jnp.ndarray,
    nelec: float,
    degauss: float,
    smearing: str = "gaussian",
    counts=None,
):
    """Occupation weights and the Fermi level(s) for a smeared calculation.

    Args:
        eigenvalues: ``(nspin, nk, nbnd)``.
        weights: ``(nk,)`` k-point weights, summing to 1 per channel when
            ``nspin = 2`` and to 2 when it is 1.
        counts: ``(nelup, neldw)`` to constrain the magnetization -- QE's
            ``two_fermi_energies``. ``None`` shares one Fermi level between the
            channels, which is the unconstrained case.

    Returns ``(wg, ef)`` with ``ef`` a scalar, or ``(wg, (ef_up, ef_dw))`` when
    the magnetization is constrained. The shared search is literally the
    unpolarized one run on a k-list twice as long, which is how QE gets it: it
    stores both channels in one array of ``2 nks`` points and never writes a
    spin-aware Fermi search at all.
    """
    ngauss = smearing_order(smearing)
    eigenvalues = jnp.asarray(eigenvalues)
    weights = jnp.asarray(weights)
    nspin = eigenvalues.shape[0]

    if counts is None:
        flat, tiled = _flatten_spin(eigenvalues, weights)
        wg, ef = _smeared(flat, tiled, nelec, degauss, ngauss)
        return wg.reshape(eigenvalues.shape), ef

    channels = [
        _smeared(eigenvalues[spin], weights, counts[spin], degauss, ngauss)
        for spin in range(nspin)
    ]
    return (
        jnp.stack([wg for wg, _ in channels]),
        tuple(ef for _, ef in channels),
    )


def _flatten_spin(eigenvalues, weights):
    """The ``2 nks`` k-list QE builds with ``set_kup_and_kdw``, as arrays."""
    nspin = eigenvalues.shape[0]
    return (
        eigenvalues.reshape(-1, eigenvalues.shape[-1]),
        jnp.tile(weights, nspin),
    )


@partial(jax.jit, static_argnames=("ngauss",))
def _smeared(eigenvalues, weights, nelec, degauss, ngauss):
    """Fermi level and weights in one compiled unit; ``ef`` stays on device."""
    ef = bisect_fermi(eigenvalues, weights, nelec, degauss, ngauss)
    occupation = wgauss((ef - eigenvalues) / degauss, ngauss)
    return weights[:, None] * occupation, ef


def smearing_entropy(
    eigenvalues: jnp.ndarray,
    weights: jnp.ndarray,
    ef,
    degauss: float,
    smearing: str = "gaussian",
) -> jnp.ndarray:
    """The ``-TS`` term QE prints as "smearing contrib.".

    It is what makes a smeared total energy variational: the quantity being
    minimised is a free energy, not the energy at fictitious occupations.

    ``ef`` is a scalar for one shared Fermi level, or a pair for a constrained
    magnetization -- in which case ``demet`` is ``demet_up + demet_dw``, each
    channel measured against its own level.
    """
    eigenvalues = jnp.asarray(eigenvalues)
    weights = jnp.asarray(weights)
    ngauss = smearing_order(smearing)

    if isinstance(ef, (tuple, list)):
        return sum(
            _entropy(eigenvalues[spin], weights, level, degauss, ngauss)
            for spin, level in enumerate(ef)
        )
    flat, tiled = _flatten_spin(eigenvalues, weights)
    return _entropy(flat, tiled, ef, degauss, ngauss)


@partial(jax.jit, static_argnames=("ngauss",))
def _entropy(eigenvalues, weights, ef, degauss, ngauss):
    return degauss * jnp.sum(weights[:, None] * w1gauss((ef - eigenvalues) / degauss, ngauss))


def input_occupations(card_values, eigenvalues: jnp.ndarray, weights: jnp.ndarray):
    """Occupations read from an ``OCCUPATIONS`` card (``occupations='from_input'``).

    ``weights.f90``: ``wg(:,ik) = f_inp(:, isk(ik)) * wk(ik)``, halved **only**
    when ``nspin == 1``. The halving is not a normalisation choice -- for one
    channel the card gives occupations in [0, 2] against a k-point weight that
    already carries the spin degeneracy, while for two it gives one row per
    channel in [0, 1] against a weight that does not.

    Args:
        eigenvalues: ``(nspin, nk, nbnd)``.
        weights: ``(nk,)`` k-point weights.
    """
    nspin, _, nbnd = eigenvalues.shape
    values = np.asarray(card_values, dtype=float)
    if values.size < nspin * nbnd:
        raise ValueError(
            f"OCCUPATIONS card gives {values.size} values but {nspin} x {nbnd} "
            "are needed (one row per spin channel)"
        )
    rows = values[: nspin * nbnd].reshape(nspin, nbnd)
    occupation = jnp.asarray(rows if nspin == 2 else rows / 2.0)
    return weights[None, :, None] * occupation[:, None, :]


def spin_electron_counts(nelec: float, tot_magnetization: float | None):
    """``(nelup, neldw)`` -- ``set_nelup_neldw`` in ``Modules/electrons_base.f90``.

    Transcribed rather than reasoned out, because of the ``INT(nelec)`` in the
    constrained branch: with an integer charge *and* an integer magnetization QE
    truncates the electron count before adding the magnetization, so an input
    whose two are of opposite parity gets a non-integer split and a warning
    rather than a refusal. The unconstrained default is likewise not
    ``nelec/2``: it is ``INT(nelec + 1)/2``, which puts the odd electron in the
    up channel.
    """
    integer_charge = abs(nelec - round(nelec)) < 1e-8

    if tot_magnetization is None:
        if integer_charge:
            up = float(int(nelec + 1) // 2)
            return up, nelec - up
        return nelec / 2.0, nelec / 2.0

    integer_magnetization = abs(tot_magnetization - round(tot_magnetization)) < 1e-8
    if integer_charge and integer_magnetization:
        return (
            (int(nelec) + tot_magnetization) / 2.0,
            (int(nelec) - tot_magnetization) / 2.0,
        )
    return (nelec + tot_magnetization) / 2.0, (nelec - tot_magnetization) / 2.0


def w0gauss(x: jnp.ndarray, ngauss: int) -> jnp.ndarray:
    """QE's ``w0gauss``: the smeared delta function, ``d wgauss / dx``.

    ``Modules/w0gauss.f90``'s own docstring is "the derivative of wgauss", and
    :func:`wgauss` above is already pure JAX, elementwise, and static in
    ``ngauss`` -- so the derivative is taken rather than transcribed. That is
    exact, it is one line instead of forty, and it makes it impossible for the
    delta function to drift out of step with the occupation function it is
    supposed to be the derivative of. (The same trade `xc/functional.py` makes:
    write the energy, differentiate for the potential.)

    The argument convention is the one ``PP/src/dosg.f90`` uses when it forms a
    density of states, ``x = (E - e)/degauss`` -- the same ``x`` :func:`wgauss`
    takes with ``E`` in the role of the Fermi level, positive for a state below
    ``E``. It matters: the Methfessel-Paxton and cold smearings are not
    symmetric in ``x``, so a sign flip here is invisible for a Gaussian and
    wrong for everything else.

    ``sum_k w_k w0gauss((E - e_k)/degauss) / degauss`` is the smearing DOS.
    """
    x = jnp.asarray(x)
    return jax.jvp(lambda t: wgauss(t, ngauss), (x,), (jnp.ones_like(x),))[1]


def tetrahedra_for(occupations: str, kpoints, symmetries, cell):
    """The tetrahedron decomposition of a calculation's k-grid.

    Re-exported here so that the occupation schemes are reachable from one
    place; the implementation is :mod:`pypresso.scf.tetrahedra`, which is
    imported lazily because it needs ``system.kpoints`` and this module is
    imported from everywhere.
    """
    from pypresso.scf.tetrahedra import tetrahedra_for as _build

    return _build(occupations, kpoints, symmetries, cell)


def tetrahedron_occupations(tetrahedra, eigenvalues, weights, nelec):
    """Occupation weights and the Fermi level by the tetrahedron method.

    Returns ``(wg, ef)`` like :func:`smeared_occupations`, with no ``-TS`` term:
    the tetrahedron method integrates the true step function, so there is no
    entropy to subtract and QE prints no "smearing contrib." for these runs.
    """
    from pypresso.scf.tetrahedra import tetrahedron_occupations as _occupations

    return _occupations(tetrahedra, eigenvalues, weights, nelec)
