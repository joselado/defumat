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

**A noncollinear or spin-orbit run is the same functional with two matrices
and a layout** (P46). The state is a two-component spinor, which QE stores as
*one* coefficient vector of length ``2 npwx`` (``evc(npwx*npol, nbnd)``) -- so
``FrozenState.wavefunctions`` is ``(1, nk, nbnd, 2 npwx)`` and the kinetic term
(1) reads :attr:`~defumat.scf.driver.Calculation.state_kinetic`, which is
``|k+G|^2`` in that vector's own layout. Term (2) takes the pseudopotential's
bare ``dvan_so``, a complex 2x2 matrix in spin space, where the collinear
branch takes ``dij``, and the constraint (6) takes ``qq_so`` where it takes
``qq``; everything else -- the density, ``becsum``, the potential, the
one-centre terms -- already handles the regime, because the SCF runs it. Keep
``nspin``, ``npol`` and ``nspin_mag`` apart here as everywhere: a spin-orbit run
without a magnetization has ``nspin_mag = 1`` and still stores two components
per band. The spinor path is **opt-in** (``spinors=True``) rather than simply
available, and :func:`reject_spinors` says which consumers cannot ask for it.

**The identity that checks all of this** -- and it is checked, in the test
suite -- is that (1)+(2)+(3) is QE's ``eband + deband``. Substituting
``eps = <psi|T + vltot + v_scf + V_NL^deeq|psi>`` and
``deband = -int rho v_scf - sum ddd_paw becsum`` into QE's decomposition gives
back exactly the terms above, so :func:`frozen_energy` evaluated at the
converged geometry must reproduce the SCF total energy to round-off.
"""

from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp

from defumat.hubbard.energy import hubbard_energy
from defumat.scf.potential import total_charge

__all__ = ["FrozenState", "frozen_energy", "energy_at", "reject_spinors", "reject_potential_only",
           "reject_magnetic_field", "reject_spinor_spiral",
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

    #: ``(nspin, nk, nbnd, npwx)``, and ``(1, nk, nbnd, 2 npwx)`` for a
    #: noncollinear run -- one channel of *spinors*, each a single vector
    #: with the up component's coefficients followed by the down one's.
    #: ``nspin`` is not ``npol``: a spin-orbit run without a magnetization
    #: has ``nspin_mag = 1`` and still stores ``2 npwx`` per band.
    wavefunctions: jnp.ndarray
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
    """The frozen state of a finished :class:`~defumat.scf.driver.SCFResult`."""
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
    becsum=None, multipliers=None, spinors: bool = False,
):
    """The total energy at ``positions``, with the electronic state frozen.

    ``calculation`` supplies everything that does not depend on where the atoms
    are; :meth:`~defumat.scf.driver.Calculation.at_positions` rebuilds what
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
    reject_potential_only(calculation)
    if not spinors:
        # ``spinors`` is opt-in here for the same reason it is on
        # :func:`energy_at`: the *force* asks for it
        # (:func:`defumat.forces.autodiff.autodiff_forces`), and
        # :mod:`defumat.response.phonon` and :mod:`defumat.response.born`
        # call this same function for a **second** derivative, whose first-order
        # wavefunctions come from a Sternheimer solve that has no spinor form.
        # Their own guard is
        # :func:`~defumat.response.sternheimer.require_a_sternheimer_regime`;
        # this keeps that from being the only one.
        reject_spinors(calculation)
    reject_spinor_spiral(calculation)
    return energy_at(
        calculation.at_positions(positions), state, density=density, becsum=becsum,
        multipliers=multipliers, spinors=spinors,
    )


def reject_spinors(calculation) -> None:
    """Refuse a two-component calculation, for the paths that have no form for it.

    **This is no longer the whole functional's refusal.** :func:`energy_at`
    grew a spinor branch (P46) -- the nonlocal quadratic form with ``dvan_so``
    and the constraint with ``qq_so``, both complex 2x2 matrices in spin space,
    on the ``2 npwx``-long coefficient vector a spinor actually is -- so the
    *force* and the *stress* run for ``noncolin = .true.``, with or without
    ``lspinorb``, on norm-conserving, ultrasoft and PAW datasets. They ask for
    it with ``spinors=True``.

    What still calls this, and why each one has to:

    * :func:`defumat.stress.analytic.analytic_terms` and
      :func:`defumat.forces.analytic.analytic_forces` (which states it in its
      own words). Those are transcriptions of ``force_us``/``stres_knl`` and
      the rest, they share no machinery with the functional, and none of them
      has a spinor form -- ``force_us`` alone would need ``deeq_nc``,
      ``becsum_nc`` and ``qq_so`` threaded through a second time.
    * :mod:`defumat.response.elastic`, which reaches :func:`energy_at`
      directly and would otherwise inherit a spinor path its first-order
      wavefunctions do not have. The Sternheimer solver refuses
      ``noncolin`` in its own guard
      (:func:`~defumat.response.sternheimer.require_a_sternheimer_regime`);
      the default of ``spinors=False`` is what keeps that refusal from being
      the *only* one, since a caller who never solves a Sternheimer equation
      would sail past it.

    So the rule is: the functional does spinors when asked; everything that
    consumes it has to ask, and the ones that cannot are the ones listed here.
    """
    if calculation.noncolin:
        raise NotImplementedError(
            "this path is not implemented for a noncollinear or spin-orbit "
            "calculation; the force and the stress are (defumat.forces, "
            "defumat.stress), on norm-conserving, ultrasoft and PAW "
            "pseudopotentials"
        )
    reject_potential_only(calculation)


def reject_spinor_spiral(calculation) -> None:
    """A spin spiral has two spheres, and this functional writes down one.

    ``noncolin`` alone is now carried (see :func:`reject_spinors`), and a
    spiral is ``noncolin`` with ``spiral_q`` on top -- so without this the
    force of a spiral would walk into :func:`_spinor_projector_energies` with a
    ``(2 nk, npwx, nkb)`` ``vkb`` against ``nk`` rows of coefficients and die on
    an einsum shape rather than on a refusal. The missing piece is real and is
    named in :mod:`defumat.forces.spiral`: the up component lives at
    ``k + q/2`` and the down at ``k - q/2``, each with its own projectors, so
    the nonlocal term needs the pair -- which ``spiral_energy`` does write, in
    the *other* coordinate. ``dE/dq`` is what a spiral has instead.
    """
    if getattr(calculation, "spiral", False):
        raise NotImplementedError(
            "the force on an atom of a spin spiral is not implemented: the two "
            "spinor components live on different plane-wave spheres, so the "
            "nonlocal term needs vkb(k + q/2) and vkb(k - q/2) as a pair (see "
            "defumat.forces.spiral, which writes exactly that for dE/dq). The "
            "stress of a spiral is refused for its own reason -- q is in "
            "lattice coordinates, so a strain turns the spiral"
        )


def reject_potential_only(calculation) -> None:
    """A potential-only functional has no energy to differentiate.

    Refused rather than approximated, and it is not a missing term: with
    Tran-Blaha there is no ``E_x[rho]`` at all, so the quantity this module
    differentiates is the *correlation* energy plus the electrostatics and the
    band term -- an expression the SCF did not minimise. Its gradient would be a
    smooth, plausible, entirely meaningless force, and the run would report it
    without complaint.

    This is the one refusal in the package that cannot be lifted by writing more
    code: a force under this functional needs a definition of the total energy
    that Tran and Blaha's potential does not come with. (There is a literature
    on assigning one -- fitting a functional whose derivative approximates the
    potential -- and none of it is implemented.)

    Reached from every consumer that comes through :func:`energy_at` -- the
    stress tensor, the dynamical matrix, the elastic constants and the
    Sternheimer response. **The analytic force expressions do not**: they are a
    transcription of ``force_lc``/``force_cc``/``force_ew``/... and share no
    machinery with this functional, so ``method='analytic'`` reached a
    Tran-Blaha run and returned a number until
    :func:`defumat.forces.compute_forces` was made to call this before
    dispatching. That is why the call is in three places and not one.
    """
    if getattr(calculation, "functional", None) is not None and calculation.functional.is_meta:
        raise NotImplementedError(
            f"the {calculation.functional.name} functional is a potential and "
            "not the derivative of an energy, so there is no total energy to "
            "differentiate: forces, stress, phonons and linear response are "
            "refused for it. The band structure, the density of states and the "
            "density itself are unaffected -- they are what it is for"
        )


def reject_magnetic_field(calculation) -> None:
    """A field or a constrained moment makes the frozen state non-stationary.

    The twin of :func:`defumat.stress.energy.require_a_differentiable_cell`'s
    second check, which has refused the same combination for the *stress* since
    P18 and states the reason: ``add_bfield`` is called from inside
    ``v_of_rho``, so ``deband`` removes the field's energy again and ``etcon``
    is printed and never added (Elk excludes its external field's energy by the
    same convention). The converged state is therefore stationary for a
    *different* functional than :func:`energy_at` writes down, and the missing
    term -- ``-int B . dm/dtau``, plus the penalty's own derivative -- is
    silent: the force comes back finite, sums to zero over the atoms, and
    obeys the symmetry of the crystal.

    The force path had no such check while the stress path did, which is the
    same guard-on-one-sibling shape as the rest of this pass. Lifting it means
    writing the constraint's energy into ``energy_at`` and its derivative with
    it -- :mod:`defumat.scf.fields` already has the energy, so it is a real
    term rather than a structural obstacle.
    """
    if getattr(calculation, "magnetic_field", None) is not None:
        raise NotImplementedError(
            "forces with a magnetic field or a constrained moment are not "
            "implemented: the field's energy is deliberately outside the "
            "reported total (see defumat.scf.fields), so the converged state "
            "is stationary for a different functional than the one being "
            "differentiated and the missing term would be silent. The stress "
            "refuses the same combination for the same reason"
        )


def _mixed_state_part(override, build, moved, *arguments):
    """One member of the mixed state -- the density or ``becsum`` -- built or given.

    ``None`` builds it the way the SCF builds it. An **array** (or, for
    ``becsum``, a tuple) replaces it with a constant, which is
    :mod:`defumat.response.phonon`'s use: the response has already been
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
              becsum=None, multipliers=None, spinors: bool = False):
    """The frozen-state energy of an already-moved calculation.

    Split out from :func:`frozen_energy` because the *coordinate* being
    differentiated is the only thing that separates a force from a stress: the
    functional above is written in terms of a calculation, and which of
    :meth:`~defumat.scf.driver.Calculation.at_positions` and
    :meth:`~defumat.scf.driver.Calculation.at_strain` produced it is not its
    business (:mod:`defumat.stress.energy` is the other caller).

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
    (:mod:`defumat.response.phonon`). The reason is the symmetrisation. The
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
    :meth:`~defumat.response.sternheimer.SternheimerSolver.density_at`'s rule
    one level up, and it changes nothing for a force or a stress, where the
    argument is left at ``None``.

    **``becsum`` is the same argument for the other half of the mixed state**,
    and it exists because :meth:`~defumat.scf.driver.Calculation.becsum` ends
    with ``PAW_symmetrize``. That average is right for the ground state and
    wrong for a response for exactly the reason above, one level down --
    ``PAW_dusymmetrize`` against ``PAW_symmetrize``, worth 1.6e-2 on PAW
    silicon's dielectric constant where the rest of the machinery reaches 5e-5
    (``PLAN.md`` P24a). Nothing before
    :mod:`defumat.response.born` needed it, because P25's second derivative is
    norm-conserving and a norm-conserving dataset has no ``becsum`` at all.

    Both arguments accept a **callable** as well as a constant, and for a
    coordinate derivative that distinction is the whole of the matter: see
    :func:`_mixed_state_part`.
    """
    if moved.noncolin and not spinors:
        # The default, and it is the guard that stands for every consumer which
        # has not been validated in this regime -- :mod:`defumat.response.
        # elastic` above all, which calls this function directly and never sees
        # the Sternheimer solver's own ``noncolin`` refusal. See
        # :func:`reject_spinors`.
        reject_spinors(moved)
    reject_potential_only(moved)
    if moved.noncolin:
        reject_spinor_spiral(moved)
        if multipliers is not None:
            raise NotImplementedError(
                "the matrix orthonormality multipliers are not implemented for "
                "a spinor: _constraint_energy contracts the scalar qq, where a "
                "spinor's metric is qq_so and Lambda carries a spin pair as "
                "well. Nothing but the ultrasoft second derivative "
                "(defumat.response.born) asks for them, and that path is "
                "refused for noncolin already"
            )

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

    # Half-sphere storage: every plane-wave sum below doubles and corrects
    # ``G = 0``. A spinor run never has it (``gamma_storage_is_consumable``
    # takes only a norm-conserving collinear run), so the branch above is
    # untouched.
    gamma_only = bool(getattr(moved, "gamma_only", False))
    if moved.noncolin:
        # A spinor is one coefficient vector of length ``2 npwx``, not two
        # states: ``npol`` and ``nspin`` are different numbers (`CLAUDE.md`),
        # and every array here follows ``npol``. ``state_kinetic`` is
        # ``|k+G|^2`` laid out in that vector's own layout, and the two
        # quadratic forms take the complex 2x2-in-spin ``dvan_so`` and
        # ``qq_so`` where the collinear ones take ``dij`` and ``qq``.
        kinetic = _kinetic_energy(psi, moved.state_kinetic, weights)
        nonlocal_, overlap = _spinor_projector_energies(
            psi, moved.projectors.vkb, moved.dvan_so, moved.qq_so,
            weights, state.eigenvalues,
        )
    else:
        kinetic = _kinetic_energy(psi, moved.kinetic, weights, gamma_only)
        nonlocal_, overlap = _projector_energies(
            psi, moved.projectors.vkb, moved.projectors.dij, moved.projectors.qq,
            weights, state.eigenvalues, gamma_only,
        )
    # <psi|psi> - 1 is position-independent (the plane-wave basis does not move
    # with the atoms), so it contributes nothing to the force; it is here
    # because the term is the constraint, and writing half of it would make the
    # energy identity above only approximately true. Under a *strain* it is
    # equally inert -- the coefficients are frozen and the basis is a set of
    # Miller indices -- which is the statement that a norm-conserving stress has
    # no Pulay term of this kind either.
    norm = jnp.sum(weights * state.eigenvalues * (_norms(psi, gamma_only) - 1.0))
    if multipliers is not None:
        # The same constraint with the multipliers off the diagonal -- see
        # :func:`_constraint_energy`. Identical at the ground state, where
        # ``Lambda = diag(w eps)``, and the two terms above are what it replaces.
        overlap = jnp.zeros(())
        norm = _constraint_energy(
            psi, moved.projectors.vkb, moved.projectors.qq, multipliers, gamma_only
        )

    contributions = {
        "kinetic": kinetic,
        "nonlocal": nonlocal_,
        "local": local,
        "hartree": potential.ehart,
        "xc": potential.etxc,
        "ewald": moved.ewald,
        # The van der Waals correction, a pair sum over the nuclei and nothing
        # else (:mod:`defumat.vdw`). It is a term of the energy like any other
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


@partial(jax.jit, static_argnames=("gamma_only",))
def _kinetic_energy(psi, kinetic, weights, gamma_only: bool = False):
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
    # (:mod:`defumat.response.phonon`). The trap is
    # :func:`defumat.scf.density.band_density`'s, in a third place.
    density = jnp.real(jnp.conj(psi) * psi)  # (nspin, nk, nbnd, npwx)
    per_band = jnp.einsum("skbg,kg->skb", density, kinetic)
    if gamma_only:
        # ``2 x - (the G = 0 term)``. ``kinetic`` is ``(nk, npwx)`` against a
        # ``(nspin, nk, nbnd, npwx)`` density, so the G = 0 slice has to be
        # broadcast back onto the band axes explicitly.
        per_band = 2.0 * per_band - density[..., 0] * kinetic[None, :, None, 0]
    return jnp.sum(weights * per_band)


# Every plane-wave sum in this module takes ``2 x - (the G = 0 term)`` under
# half-sphere storage. They are written out at each site rather than funnelled
# through one helper, because the shapes differ (a band index here, a projector
# index there) and a helper that took every layout would be harder to read than
# the two lines it replaced. What they share is the *comment*: the correction is
# not optional and not small, and a **force** is where dropping it shows --
# these terms are the functional the force is the gradient of, so an energy that
# is right at the ground state can still have a wrong derivative. Measured: with
# the ``G = 0`` term dropped the total energy was right to 3e-12 and the force
# was wrong by 0.4 Ry/bohr on a force of 0.06.


@partial(jax.jit, static_argnames=("gamma_only",))
def _norms(psi, gamma_only: bool = False):
    """``<psi|psi>``, with :func:`_kinetic_energy`'s rule about ``abs``."""
    density = jnp.real(jnp.conj(psi) * psi)
    total = jnp.sum(density, axis=-1)
    if gamma_only:
        total = 2.0 * total - density[..., 0]
    return total


def _becp_gamma(psi, vkb, gamma_only: bool):
    """``<beta|psi>``, halved sphere or whole -- ``calbec`` / ``calbec_gamma``.

    The same expression :meth:`defumat.hamiltonian.operator.Hamiltonian._becp`
    computes for the operator. It is written twice because the two live on
    opposite sides of a ``jax.grad``: this one is differentiated with respect to
    the atomic positions through ``vkb``, and importing the operator's would
    drag a whole ``Hamiltonian`` into the functional.
    """
    becp = jnp.einsum("skbg,kgc->skbc", psi.conj(), vkb).conj()
    if not gamma_only:
        return becp
    zero = jnp.einsum("skbg,kgc->skbc", psi[..., :1].conj(), vkb[:, :1]).conj()
    return (2.0 * becp.real - zero.real).astype(becp.dtype)


@partial(jax.jit, static_argnames=("gamma_only",))
def _constraint_energy(psi, vkb, qq, multipliers, gamma_only: bool = False):
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
    (:mod:`defumat.response.born`).

    ``Lambda`` is ``(nspin, nk, nbnd, nbnd)`` and Hermitian, and it is
    contracted as a trace -- ``sum_mn Lambda_mn (G - 1)_nm`` -- so the result is
    real for any Hermitian pair and the index order is the one stationarity
    fixes rather than a convention.
    """
    gram = jnp.einsum("skmg,skng->skmn", psi.conj(), psi)
    if gamma_only:
        gram = 2.0 * gram.real - jnp.real(
            psi[..., :1].conj() * jnp.swapaxes(psi[..., :1], -1, -2)
        )
        gram = gram.astype(psi.dtype)
    if vkb.shape[-1] != 0 and qq is not None:
        becp = _becp_gamma(psi, vkb, gamma_only)
        gram = gram + jnp.einsum(
            "skmi,ij,sknj->skmn", becp.conj(), qq.astype(becp.dtype), becp
        )
    identity = jnp.eye(gram.shape[-1], dtype=gram.dtype)
    return jnp.real(jnp.einsum("skmn,sknm->", multipliers, gram - identity))


@partial(jax.jit, static_argnames=("gamma_only",))
def _projector_energies(psi, vkb, dij, qq, weights, eigenvalues,
                        gamma_only: bool = False):
    """The two sums over projector channels: ``<psi|V_NL|psi>`` and ``<psi|S-1|psi>``.

    Computed together because both are quadratic forms in the same projections
    ``<beta|psi>``, which are the expensive part -- ``(nspin, nk, nbnd, nkb)``
    from an ``npwx``-long contraction. The overlap term is weighted by the
    eigenvalue as well as the occupation, which is QE's ``deff = deeq - eps qq``
    with the two halves kept apart.
    """
    if vkb.shape[-1] == 0:
        return jnp.zeros(()), jnp.zeros(())
    becp = _becp_gamma(psi, vkb, gamma_only)  # <beta|psi>
    bands = jnp.real(jnp.einsum("skbi,ij,skbj->skb", becp.conj(), dij.astype(becp.dtype), becp))
    nonlocal_ = jnp.sum(weights * bands)
    if qq is None:
        return nonlocal_, jnp.zeros(())
    overlap = jnp.real(
        jnp.einsum("skbi,ij,skbj->skb", becp.conj(), qq.astype(becp.dtype), becp)
    )
    return nonlocal_, jnp.sum(weights * eigenvalues * overlap)


@jax.jit
def _spinor_projector_energies(psi, vkb, dvan_so, qq_so, weights, eigenvalues):
    """:func:`_projector_energies` for a two-component spinor.

    ``add_vuspsi_nc`` and ``s_psi_nc``: the coefficient vector is ``2 npwx``
    long and splits into the two components QE stores as ``evc(1:npw)`` and
    ``evc(npwx+1:npwx+npw)``, both projected on the *same* projectors -- the
    spiral is the one case where they are not, and it is refused
    (:func:`reject_spinor_spiral`). What changes is the matrix between them:
    ``D`` and the overlap's ``q`` each carry a spin pair, complex and not
    diagonal, and the off-diagonal blocks are the whole of spin-orbit coupling.

    ``dvan_so`` is the **bare** ``D`` from the pseudopotential -- the spinor
    twin of ``dij`` and for the same reason: the self-consistent
    ``int V_eff Q_ij`` part of ``deeq_nc`` is already inside the augmented
    density, so taking ``deeq_nc`` here would count it twice. Note that the
    spin transform is not a detour around that: ``newd_nc`` sandwiches the
    scalar integrals between ``fcoef`` and *adds* them to ``dvan_so``
    (:func:`defumat.scf.driver._newd_noncollinear`), so the split is the same
    split one spin index up.

    Indices: ``s`` the (single) density channel the state axis carries, ``k``
    k-point, ``n`` band, ``a``/``b`` spinor component, ``i``/``j`` projector.
    """
    if vkb.shape[-1] == 0:
        return jnp.zeros(()), jnp.zeros(())
    npwx = vkb.shape[1]
    components = psi.reshape(psi.shape[:-1] + (2, npwx))
    # ``<beta_i|psi^a>``, the same contraction ``SpinorHamiltonian._project``
    # does one k-point at a time.
    becp = jnp.einsum("kgi,sknag->sknai", vkb.conj(), components)
    bands = jnp.real(jnp.einsum(
        "sknai,abij,sknbj->skn",
        becp.conj(), dvan_so.astype(becp.dtype), becp, optimize=True,
    ))
    nonlocal_ = jnp.sum(weights * bands)
    if qq_so is None:
        return nonlocal_, jnp.zeros(())
    overlap = jnp.real(jnp.einsum(
        "sknai,abij,sknbj->skn",
        becp.conj(), qq_so.astype(becp.dtype), becp, optimize=True,
    ))
    return nonlocal_, jnp.sum(weights * eigenvalues * overlap)
