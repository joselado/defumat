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

from pypresso.hubbard.energy import hubbard_energy
from pypresso.scf.potential import total_charge

__all__ = ["FrozenState", "frozen_energy", "energy_at", "reject_spinors",
           "state_from_result"]


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


def frozen_energy(
    calculation, positions: jnp.ndarray, state: FrozenState, density=None,
    becsum=None, multipliers=None,
):
    """The total energy at ``positions``, with the electronic state frozen.

    ``calculation`` supplies everything that does not depend on where the atoms
    are; :meth:`~pypresso.scf.driver.Calculation.at_positions` rebuilds what
    does. The result is a scalar in Ry and a differentiable function of
    ``positions``.

    ``density`` and ``becsum`` override the two members of the mixed state this
    would otherwise build from ``state`` -- see :func:`_mixed_state_part` and
    :func:`energy_at`, which is where the reason lives.
    """
    # Refused **before** the calculation is moved, not after. Moving it is real
    # work -- new projectors, a new local potential, a new Ewald sum -- and a
    # refusal that arrives on the far side of it is both slower and, for a
    # caller holding a partially built calculation, a different exception
    # entirely.
    reject_spinors(calculation)
    return energy_at(
        calculation.at_positions(positions), state, density=density, becsum=becsum,
        multipliers=multipliers,
    )


def reject_spinors(calculation) -> None:
    """The one regime this functional does not cover, refused by name.

    The spinor constraint term needs ``qq_so`` rather than ``qq``, and the
    nonlocal term ``dvan_so`` rather than ``dij`` -- both complex 2x2 matrices
    in spin space. Neither is written here, and a force or a stress computed
    with the scalar expressions would be wrong by the whole spin-orbit part of
    the nonlocal energy while looking entirely plausible.

    Called from the entry points *before* they move the calculation, and again
    inside :func:`energy_at` so that a caller reaching it directly cannot slip
    past.
    """
    if calculation.noncolin:
        raise NotImplementedError(
            "forces and stress for a noncollinear or spin-orbit calculation are "
            "not implemented; nspin = 1 and nspin = 2 are, on norm-conserving, "
            "ultrasoft and PAW pseudopotentials"
        )


def _mixed_state_part(override, build, moved, *arguments):
    """One member of the mixed state -- the density or ``becsum`` -- built or given.

    ``None`` builds it the way the SCF builds it. An **array** (or, for
    ``becsum``, a tuple) replaces it with a constant, which is
    :mod:`pypresso.response.phonon`'s use: the response has already been
    symmetrised outside and is handed in whole. A **callable** replaces the
    *builder* instead, and is called as ``f(moved, ...)`` -- which is what a
    mixed derivative with respect to the positions needs, because the part of
    the mixed state that moves with the atoms (the augmentation charge, and
    ``becsum`` through the projectors) has to stay a function of ``moved`` for
    the chain rule to reach it. A constant cannot: it freezes exactly the
    dependence an ultrasoft second derivative is made of.
    """
    if override is None:
        return build(*arguments)
    if callable(override):
        return override(moved, *arguments)
    return override


def energy_at(moved, state: FrozenState, terms: bool = False, density=None,
              becsum=None, multipliers=None):
    """The frozen-state energy of an already-moved calculation.

    Split out from :func:`frozen_energy` because the *coordinate* being
    differentiated is the only thing that separates a force from a stress: the
    functional above is written in terms of a calculation, and which of
    :meth:`~pypresso.scf.driver.Calculation.at_positions` and
    :meth:`~pypresso.scf.driver.Calculation.at_strain` produced it is not its
    business (:mod:`pypresso.stress.energy` is the other caller).

    **Everything here reads ``moved``, never the calculation it came from.**
    Under a displacement the two share the cell, the kinetic term and the
    plane-wave basis, so it makes no difference; under a strain they share none
    of them, and a single ``calculation.`` left in this function is a term
    silently evaluated at the undeformed cell.

    ``terms = True`` returns the contributions as a dict instead of their sum,
    which is what makes a term-by-term stress available without writing the
    decomposition twice.

    **``density`` makes the density an independent argument** instead of a
    function of ``state``, and exists for the *second* derivative
    (:mod:`pypresso.response.phonon`). The reason is the symmetrisation. The
    density built below is symmetrised as a **scalar**, which is right for the
    ground state -- it is how a wedge sum is completed to the whole Brillouin
    zone, and the functional has to be the one the SCF minimised. It is wrong
    for a *response*: displacing one atom breaks the crystal's symmetry, and
    averaging that perturbation over the full group of the undisplaced crystal
    projects most of it away. A second derivative differentiates this functional
    with respect to the **states**, so the chain rule would push the state
    tangent straight through that scalar average. Supplying the density (and
    hence its tangent) from outside is what lets the caller symmetrise the
    response the way a response must be symmetrised -- ``symdvscf``, as a
    displacement-labelled vector field -- and hand in the result. It is
    :meth:`~pypresso.response.sternheimer.SternheimerSolver.density_at`'s rule
    one level up, and it changes nothing for a force or a stress, where the
    argument is left at ``None``.

    **``becsum`` is the same argument for the other half of the mixed state**,
    and it exists because :meth:`~pypresso.scf.driver.Calculation.becsum` ends
    with ``PAW_symmetrize``. That average is right for the ground state and
    wrong for a response for exactly the reason above, one level down --
    ``PAW_dusymmetrize`` against ``PAW_symmetrize``, worth 1.6e-2 on PAW
    silicon's dielectric constant where the rest of the machinery reaches 5e-5
    (``PLAN.md`` P24a). Nothing before
    :mod:`pypresso.response.born` needed it, because P25's second derivative is
    norm-conserving and a norm-conserving dataset has no ``becsum`` at all.

    Both arguments accept a **callable** as well as a constant, and for a
    coordinate derivative that distinction is the whole of the matter: see
    :func:`_mixed_state_part`.
    """
    reject_spinors(moved)

    psi, weights = state.wavefunctions, state.weights

    # The density: the bands' own charge plus, for an ultrasoft or PAW dataset,
    # the augmentation charge that moves with the atoms. Symmetrised exactly as
    # the SCF symmetrises it -- the functional has to be the same function of
    # the coordinate the SCF minimised at, not a tidier one.
    becsum_ = _mixed_state_part(becsum, moved.becsum, moved, psi, weights)
    rho = _mixed_state_part(
        density, moved.density, moved, psi, weights, becsum_
    )
    potential = moved.potential(rho)
    epaw, _ = moved.onecenter(becsum_)

    volume = moved.system.cell.volume
    local = volume / rho[0].size * jnp.sum(moved.vltot * total_charge(rho))

    # DFT+U. The occupation matrix is measured through the *moved* projectors,
    # which is the whole of ``force_hub``: the atomic orbitals are centred on
    # the atoms, so ``ns`` depends on where they are, and for ortho-atomic
    # projectors so does the ``O^{-1/2}`` that orthogonalises them. Both
    # dependences are inside this one call and neither is written down.
    hubbard = jnp.asarray(0.0)
    if moved.is_hubbard:
        hubbard = hubbard_energy(
            moved.occupation_matrix(psi, weights), moved.hubbard_coefficients
        )

    kinetic = _kinetic_energy(psi, moved.kinetic, weights)
    nonlocal_, overlap = _projector_energies(
        psi, moved.projectors.vkb, moved.projectors.dij, moved.projectors.qq,
        weights, state.eigenvalues,
    )
    # <psi|psi> - 1 is position-independent (the plane-wave basis does not move
    # with the atoms), so it contributes nothing to the force; it is here
    # because the term is the constraint, and writing half of it would make the
    # energy identity above only approximately true. Under a *strain* it is
    # equally inert -- the coefficients are frozen and the basis is a set of
    # Miller indices -- which is the statement that a norm-conserving stress has
    # no Pulay term of this kind either.
    norm = jnp.sum(weights * state.eigenvalues * (_norms(psi) - 1.0))
    if multipliers is not None:
        # The same constraint with the multipliers off the diagonal -- see
        # :func:`_constraint_energy`. Identical at the ground state, where
        # ``Lambda = diag(w eps)``, and the two terms above are what it replaces.
        overlap = jnp.zeros(())
        norm = _constraint_energy(
            psi, moved.projectors.vkb, moved.projectors.qq, multipliers
        )

    contributions = {
        "kinetic": kinetic,
        "nonlocal": nonlocal_,
        "local": local,
        "hartree": potential.ehart,
        "xc": potential.etxc,
        "ewald": moved.ewald,
        # The van der Waals correction, a pair sum over the nuclei and nothing
        # else (:mod:`pypresso.vdw`). It is a term of the energy like any other
        # here, which is the whole of ``force_london`` and ``stres_london``:
        # both are ``grad`` of this entry, in the coordinate the caller chose.
        "dispersion": moved.dispersion,
        "onecentre": epaw,
        "hubbard": hubbard,
        "overlap": -overlap,
        "constraint": -norm,
        "smearing": state.entropy,
    }
    if terms:
        return contributions
    return sum(contributions.values())


@jax.jit
def _kinetic_energy(psi, kinetic, weights):
    """``sum w f <psi| |k+G|^2 |psi>``.

    The plane-wave basis does not move with the atoms, so this term has no
    position dependence and contributes nothing to the force. It is here because
    the functional has to *be* the total energy before it is differentiated --
    that identity is the only check there is on the rest of it.
    """
    # ``Re(conj(psi) psi)`` and not ``abs(psi)**2`` -- the two are the same
    # number to a rounding and only one is differentiable, ``abs``'s derivative
    # being ``0/0`` at a coefficient that vanishes. It does not matter for a
    # force, which differentiates this with respect to the *positions*; it
    # matters the moment the same functional is differentiated with respect to
    # the **states**, which is what the second derivative does
    # (:mod:`pypresso.response.phonon`). The trap is
    # :func:`pypresso.scf.density.band_density`'s, in a third place.
    density = jnp.real(jnp.conj(psi) * psi)  # (nspin, nk, nbnd, npwx)
    return jnp.sum(weights * jnp.einsum("skbg,kg->skb", density, kinetic))


@jax.jit
def _norms(psi):
    """``<psi|psi>``, with :func:`_kinetic_energy`'s rule about ``abs``."""
    return jnp.sum(jnp.real(jnp.conj(psi) * psi), axis=-1)


@jax.jit
def _constraint_energy(psi, vkb, qq, multipliers):
    """``Tr[Lambda (<psi|S|psi> - 1)]``: the constraint with a *matrix* multiplier.

    The diagonal form it replaces is the same expression at
    ``Lambda_mn = delta_mn w_n eps_n``, which is what the multipliers are at a
    converged ground state -- so a force, a stress and a ``Gamma`` phonon never
    need this one and do not pay for it (an ``nbnd x nbnd`` Gram matrix per
    k-point where the diagonal form is a vector).

    **What needs it is a second derivative in which the multipliers themselves
    move**, which is ``psidspsi`` in ``zstar_eu_us.f90``. Stationarity gives
    ``Lambda_mp = w_m <psi_p|H|psi_m>``, diagonal at the ground state and *not*
    diagonal to first order in a perturbation, so a diagonal-only tangent drops
    the off-diagonal block of ``dLambda`` -- and that block multiplies
    ``<psi_m|dS/du|psi_p>``, which vanishes only for a norm-conserving dataset.
    Writing the constraint with the full matrix is what lets one ``jvp`` carry
    the term instead of a second routine computing it
    (:mod:`pypresso.response.born`).

    ``Lambda`` is ``(nspin, nk, nbnd, nbnd)`` and Hermitian, and it is
    contracted as a trace -- ``sum_mn Lambda_mn (G - 1)_nm`` -- so the result is
    real for any Hermitian pair and the index order is the one stationarity
    fixes rather than a convention.
    """
    gram = jnp.einsum("skmg,skng->skmn", psi.conj(), psi)
    if vkb.shape[-1] != 0 and qq is not None:
        becp = jnp.einsum("skbg,kgc->skbc", psi.conj(), vkb).conj()
        gram = gram + jnp.einsum(
            "skmi,ij,sknj->skmn", becp.conj(), qq.astype(becp.dtype), becp
        )
    identity = jnp.eye(gram.shape[-1], dtype=gram.dtype)
    return jnp.real(jnp.einsum("skmn,sknm->", multipliers, gram - identity))


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
