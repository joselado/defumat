"""Occupation numbers and the Fermi level.

Two regimes, following ``PW/src/weights.f90`` and its callees:

* **fixed** occupations for a system with a gap: the lowest ``nelec/2`` bands
  are full and the rest empty.
* **smeared** occupations for a metal, where an integer filling would make the
  density discontinuous as bands cross the Fermi level between iterations. The
  Fermi level is then found by bisection on the total electron count.

Weights returned here are QE's ``wg``: the k-point weight times the occupation,
summing to the number of electrons.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf, erfc

from pypresso.units import SQRT_PI

__all__ = ["fixed_occupations", "smeared_occupations", "fermi_level", "bisect_fermi",
           "smearing_entropy",
           "input_occupations", "wgauss", "w0gauss", "w1gauss", "smearing_order",
           "tetrahedra_for", "tetrahedron_occupations"]


def fixed_occupations(eigenvalues: jnp.ndarray, weights: jnp.ndarray, nelec: float):
    """Fill the lowest ``nelec/2`` bands completely.

    Returns ``(wg, homo, lumo)``. Raises if the electron count is odd, which
    means the system needs either spin polarisation or smearing.
    """
    nbnd = eigenvalues.shape[1]
    occupied = nelec / 2.0
    if abs(occupied - round(occupied)) > 1e-8:
        raise ValueError(
            f"{nelec} electrons cannot fill spin-degenerate bands; use smearing or nspin=2"
        )
    occupied = int(round(occupied))
    if occupied > nbnd:
        raise ValueError(f"{nelec} electrons need {occupied} bands but only {nbnd} were computed")

    occupation = jnp.arange(nbnd) < occupied
    wg = weights[:, None] * occupation[None, :]

    homo = jnp.max(eigenvalues[:, occupied - 1])
    lumo = jnp.min(eigenvalues[:, occupied]) if occupied < nbnd else None
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


@partial(jax.jit, static_argnames=("ngauss",))
def bisect_fermi(
    eigenvalues: jnp.ndarray,
    weights: jnp.ndarray,
    nelec: float,
    degauss: float,
    ngauss: int,
) -> jnp.ndarray:
    """The bisection itself, as a device-side loop.

    Bisection rather than Newton because Methfessel-Paxton and cold occupations
    are not monotonic in the energy, so a derivative-based search can step out
    of the bracket entirely.

    The loop runs a fixed number of steps and its branch is a ``where``, so it
    is a ``fori_loop`` and never leaves the device. Written as a Python loop
    with a ``float()`` comparison it cost 200 host round trips *per SCF
    iteration* -- by far the most expensive thing about a metal.
    """
    eigenvalues = jnp.asarray(eigenvalues)
    weights = jnp.asarray(weights)

    def count(ef):
        occupation = wgauss((ef - eigenvalues) / degauss, ngauss)
        return jnp.sum(weights[:, None] * occupation)

    def step(_, bracket):
        low, high = bracket
        middle = 0.5 * (low + high)
        too_many = count(middle) > nelec
        return jnp.where(too_many, low, middle), jnp.where(too_many, middle, high)

    low = jnp.min(eigenvalues) - 20.0 * degauss
    high = jnp.max(eigenvalues) + 20.0 * degauss
    low, high = jax.lax.fori_loop(0, BISECTION_STEPS, step, (low, high))
    return 0.5 * (low + high)


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
):
    """Occupation weights and the Fermi level for a smeared calculation."""
    ngauss = smearing_order(smearing)
    return _smeared(jnp.asarray(eigenvalues), jnp.asarray(weights), nelec, degauss, ngauss)


@partial(jax.jit, static_argnames=("ngauss",))
def _smeared(eigenvalues, weights, nelec, degauss, ngauss):
    """Fermi level and weights in one compiled unit; ``ef`` stays on device."""
    ef = bisect_fermi(eigenvalues, weights, nelec, degauss, ngauss)
    occupation = wgauss((ef - eigenvalues) / degauss, ngauss)
    return weights[:, None] * occupation, ef


def smearing_entropy(
    eigenvalues: jnp.ndarray,
    weights: jnp.ndarray,
    ef: float,
    degauss: float,
    smearing: str = "gaussian",
) -> jnp.ndarray:
    """The ``-TS`` term QE prints as "smearing contrib.".

    It is what makes a smeared total energy variational: the quantity being
    minimised is a free energy, not the energy at fictitious occupations.
    """
    return _entropy(jnp.asarray(eigenvalues), jnp.asarray(weights), ef, degauss,
                    smearing_order(smearing))


@partial(jax.jit, static_argnames=("ngauss",))
def _entropy(eigenvalues, weights, ef, degauss, ngauss):
    return degauss * jnp.sum(weights[:, None] * w1gauss((ef - eigenvalues) / degauss, ngauss))


def input_occupations(card_values, eigenvalues: jnp.ndarray, weights: jnp.ndarray):
    """Occupations read from an ``OCCUPATIONS`` card (``occupations='from_input'``).

    QE applies the same list at every k-point, and the values are occupations
    per band in [0, 2] for an unpolarised calculation -- so the weight that
    multiplies them is the k-point weight divided by the spin degeneracy.
    """
    values = np.asarray(card_values, dtype=float)
    nbnd = eigenvalues.shape[1]
    if values.size < nbnd:
        raise ValueError(f"OCCUPATIONS card gives {values.size} values but {nbnd} bands are computed")
    occupation = jnp.asarray(values[:nbnd])
    return weights[:, None] * occupation[None, :] / 2.0


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
