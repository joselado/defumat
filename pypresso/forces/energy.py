"""The energy functional whose gradient with respect to the positions is the force.

QE computes forces from six hand-derived expressions (``force_lc``, ``force_us``,
``force_ew``, ``force_cc``, ``addusforce``, ``force_corr``). This module writes
down the *one* object those expressions are the derivative of, so that the force
can be ``jax.grad`` of it instead -- which is the reason this project is written
in JAX at all (`CLAUDE.md`, rule D5).

**What is differentiated, and why it is not the SCF loop.** Backpropagating
through the iterations would cost memory proportional to their number and would
differentiate a fixed-point *search* rather than the fixed point (rule D3), and
differentiating the eigenvalues is worse still, since ``eigh``'s derivative is
singular at the degeneracies a crystal has everywhere (rule D4). Neither is
needed. At the converged solution the total energy is *stationary* with respect
to the wavefunctions -- subject to their being orthonormal in the metric ``S`` --
and with respect to the density, so the total derivative with respect to the
positions equals the **partial** derivative at fixed wavefunctions, occupations
and eigenvalues:

    E[psi, tau] = sum_kn w f <psi|T|psi>                                  (1)
                + sum_kn w f <psi|V_NL(tau)|psi>       with the *bare* D_ij (2)
                + int vltot(tau) rho                                      (3)
                + E_H[rho] + E_xc[rho + rho_core(tau)]                     (4)
                + E_Ewald(tau) + E_onecentre[becsum]                       (5)
                - sum_kn w f eps (<psi|S(tau)|psi> - 1)                    (6)

    rho = sum_kn w f |psi|^2 + rho_aug(becsum(psi, vkb(tau)), tau)

Term (6) is the Lagrange multiplier of the orthonormality constraint, with the
multipliers at their converged values ``w f eps``. It is identically zero at the
solution and its *derivative* is not: it is what QE writes as the
``- eps <psi|dS/dtau|psi>`` half of ``force_us``, and leaving it out costs an
ultrasoft or PAW force its Pulay term. For a norm-conserving run ``S`` is the
identity, the term has no position dependence at all, and the force is pure
Hellmann-Feynman.

Two things are deliberately *not* separate terms. ``D_ij`` is not an input: the
self-consistent part of it is ``int V_eff Q_ij``, which is already in (3) and (4)
through the augmentation charge inside ``rho``, so (2) takes the bare ``dion``
from the pseudopotential file and nothing is counted twice. And there is no
"SCF correction" term: QE's ``force_corr`` exists because its density is not
exactly converged, and it vanishes at the fixed point this functional assumes.

**The identity that checks all of this** -- and it is checked, in the test
suite -- is that (1)+(2)+(3) is QE's ``eband + deband``. Substituting
``eps = <psi|T + vltot + v_scf + V_NL^deeq|psi>`` and
``deband = -int rho v_scf - sum ddd_paw becsum`` into QE's decomposition gives
back exactly the terms above, so :func:`frozen_energy` evaluated at the
converged geometry must reproduce the SCF total energy to round-off.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from pypresso.scf.potential import total_charge

__all__ = ["FrozenState", "frozen_energy", "state_from_result"]


class FrozenState(eqx.Module):
    """The converged electronic state, held fixed while the atoms move.

    ``weights`` are QE's ``wg``: the k-point weight times the occupation, so a
    sum over ``(spin, k, band)`` of ``weights * anything`` is already the
    Brillouin-zone integral. ``entropy`` is the smearing term ``-TS``, which has
    no position dependence at fixed occupations and is carried only so that
    :func:`frozen_energy` reproduces the total energy rather than the total
    energy minus one term.
    """

    wavefunctions: jnp.ndarray  # (nspin, nk, nbnd, npwx)
    weights: jnp.ndarray  # (nspin, nk, nbnd) -- wg
    eigenvalues: jnp.ndarray  # (nspin, nk, nbnd), Ry
    #: The smearing term ``-TS``. A *traced* leaf rather than static metadata,
    #: which matters for a metal: it changes at every geometry, and a static
    #: field changing is a new pytree structure, so the compiled gradient would
    #: be retraced at every ionic step of a relaxation.
    entropy: jnp.ndarray = eqx.field(converter=jnp.asarray, default=0.0)
    #: ``V[rho_out] - V[rho_in]`` at the last SCF iteration, if the run kept it.
    #: Only QE's ``force_corr`` uses it, and only the analytic force has that
    #: term: it corrects for a density that stopped short of the fixed point the
    #: differentiated functional assumes.
    potential_change: jnp.ndarray | None = None
    #: The converged density and the total local potential ``vltot + v_scf``
    #: that go with it. The **analytic** force consumes them the way
    #: ``forces.f90`` consumes ``rho%of_r`` and ``v%of_r`` -- as state the SCF
    #: already produced -- which saves rebuilding them and their transforms.
    #: The autodiff force cannot use them and does not: it needs the density as
    #: a *function* of the positions, which is the whole point of it. ``None``
    #: makes the analytic path rebuild them, so a hand-built state still works.
    density: jnp.ndarray | None = None
    potential: jnp.ndarray | None = None


def state_from_result(result) -> FrozenState:
    """The frozen state of a finished :class:`~pypresso.scf.driver.SCFResult`."""
    nspin = result.nspin
    weights = result.occupations if nspin == 2 else result.occupations[None]
    eigenvalues = result.eigenvalues if nspin == 2 else result.eigenvalues[None]
    return FrozenState(
        wavefunctions=result.wavefunctions,
        weights=jnp.asarray(weights),
        eigenvalues=jnp.asarray(eigenvalues),
        entropy=float(result.energy_terms.get("smearing", 0.0)),
        potential_change=getattr(result, "potential_change", None),
        density=result.density,
        potential=result.potential,
    )


def frozen_energy(calculation, positions: jnp.ndarray, state: FrozenState):
    """The total energy at ``positions``, with the electronic state frozen.

    ``calculation`` supplies everything that does not depend on where the atoms
    are; :meth:`~pypresso.scf.driver.Calculation.at_positions` rebuilds what
    does. The result is a scalar in Ry and a differentiable function of
    ``positions``.
    """
    if calculation.noncolin:
        # The spinor constraint term needs ``qq_so`` rather than ``qq``, and the
        # nonlocal term ``dvan_so`` rather than ``dij`` -- both complex 2x2
        # matrices in spin space. Neither is written here, and a force computed
        # with the scalar expressions would be wrong by the whole spin-orbit
        # part of the nonlocal energy while looking entirely plausible.
        raise NotImplementedError(
            "forces for a noncollinear or spin-orbit calculation are not "
            "implemented; nspin = 1 and nspin = 2 are, on norm-conserving, "
            "ultrasoft and PAW pseudopotentials"
        )

    moved = calculation.at_positions(positions)
    psi, weights = state.wavefunctions, state.weights

    # The density: the bands' own charge plus, for an ultrasoft or PAW dataset,
    # the augmentation charge that moves with the atoms. Symmetrised exactly as
    # the SCF symmetrises it -- the functional has to be the same function of
    # the positions the SCF minimised, not a tidier one.
    becsum_ = moved.becsum(psi, weights)
    rho = moved.density(psi, weights, becsum_)
    potential = moved.potential(rho)
    epaw, _ = moved.onecenter(becsum_)

    volume = calculation.system.cell.volume
    local = volume / rho[0].size * jnp.sum(moved.vltot * total_charge(rho))

    kinetic = _kinetic_energy(psi, calculation.kinetic, weights)
    nonlocal_, overlap = _projector_energies(
        psi, moved.projectors.vkb, moved.projectors.dij, moved.projectors.qq,
        weights, state.eigenvalues,
    )
    # <psi|psi> - 1 is position-independent (the plane-wave basis does not move
    # with the atoms), so it contributes nothing to the force; it is here
    # because the term is the constraint, and writing half of it would make the
    # energy identity above only approximately true.
    norm = jnp.sum(weights * state.eigenvalues * (_norms(psi) - 1.0))

    return (
        kinetic
        + nonlocal_
        + local
        + potential.ehart
        + potential.etxc
        + moved.ewald
        + epaw
        - overlap
        - norm
        + state.entropy
    )


@jax.jit
def _kinetic_energy(psi, kinetic, weights):
    """``sum w f <psi| |k+G|^2 |psi>``.

    The plane-wave basis does not move with the atoms, so this term has no
    position dependence and contributes nothing to the force. It is here because
    the functional has to *be* the total energy before it is differentiated --
    that identity is the only check there is on the rest of it.
    """
    density = jnp.abs(psi) ** 2  # (nspin, nk, nbnd, npwx)
    return jnp.sum(weights * jnp.einsum("skbg,kg->skb", density, kinetic))


@jax.jit
def _norms(psi):
    return jnp.sum(jnp.abs(psi) ** 2, axis=-1)


@jax.jit
def _projector_energies(psi, vkb, dij, qq, weights, eigenvalues):
    """The two sums over projector channels: ``<psi|V_NL|psi>`` and ``<psi|S-1|psi>``.

    Computed together because both are quadratic forms in the same projections
    ``<beta|psi>``, which are the expensive part -- ``(nspin, nk, nbnd, nkb)``
    from an ``npwx``-long contraction. The overlap term is weighted by the
    eigenvalue as well as the occupation, which is QE's ``deff = deeq - eps qq``
    with the two halves kept apart.
    """
    if vkb.shape[-1] == 0:
        return jnp.zeros(()), jnp.zeros(())
    becp = jnp.einsum("skbg,kgc->skbc", psi.conj(), vkb).conj()  # <beta|psi>
    bands = jnp.real(jnp.einsum("skbi,ij,skbj->skb", becp.conj(), dij.astype(becp.dtype), becp))
    nonlocal_ = jnp.sum(weights * bands)
    if qq is None:
        return nonlocal_, jnp.zeros(())
    overlap = jnp.real(
        jnp.einsum("skbi,ij,skbj->skb", becp.conj(), qq.astype(becp.dtype), becp)
    )
    return nonlocal_, jnp.sum(weights * eigenvalues * overlap)
