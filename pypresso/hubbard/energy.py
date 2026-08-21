"""The Hubbard energy, and the potential as its derivative.

The simplified rotationally-invariant functional of Dudarev *et al.*,
Phys. Rev. B **57**, 1505 (1998), extended with the ``J0`` and ``beta`` terms of
Himmetoglu *et al.*, Phys. Rev. B **84**, 115108 (2011) -- QE's
``lda_plus_u_kind = 0``, ``v_hubbard`` in ``PW/src/v_of_rho.f90``:

    E_U = sum_{I,s} [ (alpha_I + Ueff_I/2) Tr n^{Is}
                      - (Ueff_I/2) Tr (n^{Is} n^{Is}) ]
        + sum_{I,s} [ sgn(s) beta_I Tr n^{Is}
                      + (J0_I/2) Tr (n^{Is} n^{I,-s}) ]

with ``Ueff = U - J0``. **Only the energy is written down**; the potential
``v^{Is}_{m1 m2} = dE_U / dn^{Is}_{m2 m1}`` is ``jax.grad`` of it, which is this
project's rule (`PLAN.md` D1) and the same arrangement
:mod:`pypresso.scf.fields` uses for the magnetic field. QE's hand-derived
``v_hubbard`` is transcribed below as :func:`qe_hubbard_potential` and is a
*test*, not a second implementation.

**The factor of two for an unpolarized run is not in the potential.** QE halves
``ns`` in ``new_ns`` when ``nspin = 1`` -- so ``ns`` always means the occupation
*of one spin channel* -- and then doubles ``eth`` and the ``deband`` correction
at the end, because there are two identical channels. It does **not** double
``v_hub``, which acts on one channel at a time. So the function differentiated
here is the per-channel sum, and the doubling is applied to the reported energy
afterwards (:func:`hubbard_energy`). Differentiating the doubled energy instead
gives a potential twice too large, an SCF that converges, and a total energy
wrong in the second decimal.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "coefficients_from_setup",
    "hubbard_energy",
    "hubbard_potential",
    "ns_ddot",
    "qe_hubbard_potential",
]


def coefficients_from_setup(setup) -> dict:
    """The per-slot coefficients of the four terms, already branched.

    QE's ``v_hubbard`` guards each pair of terms with a test on the *input*
    parameters -- the ``U``/``alpha`` pair runs only when one of them is
    nonzero, the ``J0``/``beta`` pair likewise -- and ``Ueff = U - J0`` only
    when ``J0`` is nonzero. Those are host-side branches on static numbers, so
    they are resolved here, once, into arrays; nothing downstream branches.

    A plain ``dict`` of arrays, which is a pytree, so it crosses ``jit`` and
    ``grad`` boundaries as the constant it is.
    """
    u = setup.parameter("u")
    j0 = setup.parameter("j0")
    alpha = setup.parameter("alpha")
    beta = setup.parameter("beta")

    effective = np.where(j0 != 0.0, u - j0, u)
    first = (u != 0.0) | (alpha != 0.0)
    second = (j0 != 0.0) | (beta != 0.0)
    return {
        "linear": jnp.asarray(np.where(first, alpha + 0.5 * effective, 0.0)),
        "quadratic": jnp.asarray(np.where(first, 0.5 * effective, 0.0)),
        "beta": jnp.asarray(np.where(second, beta, 0.0)),
        "exchange": jnp.asarray(np.where(second, 0.5 * j0, 0.0)),
        # ``ns_ddot`` uses the bare U, not ``Ueff``.
        "u_metric": jnp.asarray(0.5 * u),
    }


def _per_channel_energy(ns: jnp.ndarray, coefficients) -> jnp.ndarray:
    """The bracketed sum above, **without** the ``nspin = 1`` doubling.

    ``ns`` is ``(nspin, nslot, ldmx, ldmx)``; padded rows of a manifold shorter
    than ``ldmx`` are zero and contribute nothing to either trace.
    """
    nspin = ns.shape[0]
    trace = jnp.einsum("snmm->sn", ns)
    square = jnp.einsum("snab,snba->sn", ns, ns)

    energy = jnp.sum(coefficients["linear"] * trace)
    energy = energy - jnp.sum(coefficients["quadratic"] * square)

    sign = jnp.asarray([1.0, -1.0])[:nspin] if nspin == 2 else jnp.asarray([1.0])
    energy = energy + jnp.sum(sign[:, None] * coefficients["beta"] * trace)
    # ``isop``: the opposite channel when there are two, the same one when there
    # is one. QE writes it as an index; reversing the spin axis is the same map.
    opposite = ns[::-1] if nspin == 2 else ns
    energy = energy + jnp.sum(
        coefficients["exchange"] * jnp.einsum("snab,snba->sn", ns, opposite)
    )
    return energy


def hubbard_energy(ns: jnp.ndarray, coefficients) -> jnp.ndarray:
    """``eth``: the Hubbard energy in Ry, as ``electrons.f90`` adds it to ``etot``.

    Doubled for ``nspin = 1``, because ``ns`` is then one channel of two.
    """
    energy = _per_channel_energy(ns, coefficients)
    return 2.0 * energy if ns.shape[0] == 1 else energy


def hubbard_potential(ns: jnp.ndarray, coefficients) -> jnp.ndarray:
    """``v_hub``: ``(nspin, nslot, ldmx, ldmx)``, the derivative of the energy.

    Not of :func:`hubbard_energy` -- of the per-channel sum. See the module
    docstring.
    """
    return jax.grad(_per_channel_energy)(ns, coefficients)


def qe_hubbard_potential(ns: np.ndarray, setup) -> np.ndarray:
    """``v_hubbard`` transcribed literally from ``PW/src/v_of_rho.f90``.

    The cross-check for :func:`hubbard_potential`, written in the Fortran's own
    loop structure so that the two implementations share nothing but the input.
    """
    ns = np.asarray(ns)
    nspin, nslot, ldmx, _ = ns.shape
    v_hub = np.zeros_like(ns)
    sgn = (1.0, -1.0)

    for slot in range(nslot):
        item = setup.species[setup.types[slot]]
        ldim = setup.ldims[slot]
        if item.u != 0.0 or item.alpha != 0.0:
            eff = item.u - item.j0 if item.j0 != 0.0 else item.u
            for spin in range(nspin):
                for m1 in range(ldim):
                    v_hub[spin, slot, m1, m1] += item.alpha + 0.5 * eff
                    for m2 in range(ldim):
                        v_hub[spin, slot, m1, m2] -= eff * ns[spin, slot, m2, m1]
        if item.j0 != 0.0 or item.beta != 0.0:
            for spin in range(nspin):
                other = 1 - spin if nspin == 2 else 0
                for m1 in range(ldim):
                    v_hub[spin, slot, m1, m1] += sgn[spin] * item.beta
                    for m2 in range(ldim):
                        v_hub[spin, slot, m1, m2] += (
                            item.j0 * ns[other, slot, m2, m1]
                        )
    return v_hub


def ns_ddot(residual: jnp.ndarray, coefficients) -> jnp.ndarray:
    """``ns_ddot`` (``PW/src/scf_mod.f90``): the ``ns`` part of the mixing metric.

    ``U/2 sum |dns|^2``, doubled for ``nspin = 1``, added to ``rho_ddot`` to
    make ``dr2`` an estimate of the self-consistency error in the *whole*
    functional rather than in its plane-wave part alone. Leaving it out does not
    change the converged answer; it changes ``dr2``, and with it the ``ethr``
    schedule and the iteration count, so QE's numbers would no longer be
    reproducible step for step.
    """
    value = jnp.sum(coefficients["u_metric"] * jnp.sum(residual**2, axis=(2, 3)))
    return 2.0 * value if residual.shape[0] == 1 else value
