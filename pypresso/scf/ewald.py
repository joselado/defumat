"""Ewald summation: the electrostatic energy of the point ion cores.

The ion-ion interaction is a lattice sum of ``1/r`` terms, conditionally
convergent in both real and reciprocal space. Ewald's method splits it with a
Gaussian screening parameter ``alpha``: the short-ranged remainder is summed in
real space and the smooth part in reciprocal space, and the total is independent
of ``alpha``.

Transcribed from ``PW/src/ewald.f90``, including its choice of ``alpha`` (start
at 2.9 and reduce until the reciprocal-space truncation error is below 1e-7) and
its real-space cutoff of ``4/sqrt(alpha)``, at which the neglected terms are
``erfc(4) ~ 2e-8``. Matching those choices is not required for correctness --
the sum converges either way -- but it makes agreement with QE exact rather than
approximate, which is what makes the comparison a real test.

The result is one of the terms QE prints in its energy decomposition, so it can
be checked on its own long before there is an SCF to put it in.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erfc
from scipy.special import erfc as erfc_host

from pypresso.basis.gvectors import GVectors
from pypresso.system.cell import Cell
from pypresso.system.structure import Structure
from pypresso.units import E2, TPI

__all__ = ["ewald_energy", "ewald_alpha"]


def ewald_alpha(charge: float, gcut: float, tpiba2: float, tolerance: float = 1.0e-7) -> float:
    """QE's choice of screening parameter: the largest that still converges.

    ``upperbound`` is a bound on the error made by truncating the reciprocal
    sum at ``gcut``; ``alpha`` is stepped down from 2.9 until it is small enough.
    """
    alpha = 2.9
    while alpha > 0.0:
        alpha -= 0.1
        if alpha <= 0.0:
            break
        upperbound = (
            2.0
            * charge**2
            * np.sqrt(2.0 * alpha / TPI)
            * float(erfc_host(np.sqrt(tpiba2 * gcut / 4.0 / alpha)))
        )
        if upperbound <= tolerance:
            return alpha
    raise ValueError("optimal Ewald alpha not found; is the G cutoff absurdly small?")


def ewald_energy(
    cell: Cell,
    structure: Structure,
    gvectors: GVectors,
    charges: np.ndarray,
) -> jnp.ndarray:
    """Electrostatic energy of the ion cores, in Ry.

    Args:
        cell: the unit cell.
        structure: atomic positions (cartesian, bohr).
        gvectors: the dense G-vector set, whose cutoff sets ``alpha``.
        charges: valence charge ``Z`` of each *atom* (not each species).
    """
    charges = np.asarray(charges, dtype=float)
    if len(charges) != structure.nat:
        raise ValueError(f"{len(charges)} charges for {structure.nat} atoms")

    tpiba2 = cell.tpiba**2
    gcut = gvectors.ecut / tpiba2
    alpha = ewald_alpha(float(charges.sum()), gcut, tpiba2)

    reciprocal = _reciprocal_term(cell, structure, gvectors, charges, alpha)
    real = _real_term(cell, structure, charges, alpha)
    return 0.5 * E2 * (reciprocal + real)


def _reciprocal_term(cell, structure, gvectors, charges, alpha) -> jnp.ndarray:
    """The smooth part, summed over G, plus the two ``G = 0`` constants."""
    factor = 2.0 if gvectors.gamma_only else 1.0
    return _reciprocal_kernel(
        gvectors.cartesian(cell), structure.positions, jnp.asarray(charges),
        alpha, cell.volume, factor,
    )


@jax.jit
def _reciprocal_kernel(g, tau, charges, alpha, volume, factor):
    # rho(G) = sum_a Z_a conj(S_a(G)); only |rho|^2 is used.
    rho = jnp.sum(charges * jnp.exp(1j * (g @ tau.T)), axis=1)
    g2 = jnp.sum(g**2, axis=1)

    charge = jnp.sum(charges)
    total = -(charge**2) / alpha / 4.0

    # G = 0 is index 0 by construction and is excluded from the sum.
    nonzero = g2[1:]
    total = total + factor * jnp.sum(
        jnp.abs(rho[1:]) ** 2 * jnp.exp(-nonzero / alpha / 4.0) / nonzero
    )

    total = 2.0 * TPI / volume * total

    # The self-interaction of each Gaussian, removed once per atom.
    return total - jnp.sum(charges**2) * jnp.sqrt(8.0 / TPI * alpha)


def _real_term(cell, structure, charges, alpha) -> jnp.ndarray:
    """The short-ranged part: erfc(sqrt(alpha) r)/r over neighbouring images.

    The set of lattice translations is enumerated on the host -- it is integer
    bookkeeping over a fixed cell, the definition of setup work -- and the sum
    over ``(atom, atom, translation)`` is then one broadcast kernel rather than a
    Python double loop over atom pairs. Beyond the speed, this is what makes the
    term differentiable with respect to the atomic positions, which the forces
    will need: the neighbour list is a constant, the distances computed from it
    are not.
    """
    at = np.asarray(cell.at)
    tau = np.asarray(structure.positions)
    rmax = 4.0 / np.sqrt(alpha)

    translations = _lattice_translations(at, rmax + _max_separation(tau, at))
    return _real_kernel(structure.positions, jnp.asarray(charges),
                        jnp.asarray(translations), alpha, rmax)


@jax.jit
def _real_kernel(tau, charges, translations, alpha, rmax):
    # (nat, nat, ntrans, 3): every pair, every image.
    separations = tau[:, None, None, :] - tau[None, :, None, :] + translations[None, None, :, :]
    distances = jnp.sqrt(jnp.sum(separations**2, axis=-1))

    # The self term (r = 0) and images past the cutoff are dropped by weight, not
    # by indexing, so the shape stays static. The distance fed to the divide is
    # sanitised first: masking the result of a division by zero afterwards still
    # leaves a NaN in the gradient.
    keep = (distances > 1.0e-8) & (distances <= rmax)
    safe = jnp.where(keep, distances, 1.0)
    terms = jnp.where(keep, erfc(jnp.sqrt(alpha) * safe) / safe, 0.0)

    pairs = charges[:, None] * charges[None, :]
    return jnp.sum(pairs * jnp.sum(terms, axis=-1))


def _max_separation(tau: np.ndarray, at: np.ndarray) -> float:
    """Largest distance between two atoms, to widen the translation search."""
    if len(tau) < 2:
        return 0.0
    differences = tau[:, None, :] - tau[None, :, :]
    return float(np.linalg.norm(differences, axis=-1).max())


def _lattice_translations(at: np.ndarray, radius: float) -> np.ndarray:
    """All lattice vectors ``n . at`` with ``|n . at| <= radius``.

    The search box is bounded by projecting the radius onto each reciprocal
    direction, which is the smallest box guaranteed to contain the sphere for a
    skewed cell -- a cubic guess on ``|a_i|`` misses corners of oblique lattices.
    """
    reciprocal = np.linalg.inv(at).T  # rows: b_i / 2pi
    bounds = np.ceil(radius * np.linalg.norm(reciprocal, axis=1)).astype(int) + 1

    ranges = [np.arange(-n, n + 1) for n in bounds]
    i, j, k = np.meshgrid(*ranges, indexing="ij")
    integers = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1)
    vectors = integers @ at
    return vectors[np.linalg.norm(vectors, axis=1) <= radius]
