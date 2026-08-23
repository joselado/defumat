"""Quantum ESPRESSO's forces, term by term.

``PW/src/forces.f90`` and the five routines it calls, transcribed. This is the
*other* way to get the force -- the one that writes down each derivative by hand
-- and it exists here for three reasons: it is what the reference implementation
does, it is a check on the autodiff force that shares none of its machinery, and
it has one term (``force_corr``) that a force obtained by differentiating the
energy at the fixed point cannot have.

The terms, each with the file it comes from:

``force_ew``   ``force_ew.f90``   the ion-ion electrostatics
``force_lc``   ``force_lc.f90``   the local pseudopotential moving through the density
``force_cc``   ``force_cc.f90``   the nonlinear core charge moving through ``v_xc``
``force_us``   ``force_us.f90``   the projectors moving, with ``deff = deeq - eps qq``
``addusforce`` ``addusforce.f90`` the augmentation charge ``Q_ij(r - tau)`` moving
``force_corr`` ``force_corr.f90`` the correction for a density that stopped short

Three conventions carry over from the Fortran and are worth stating once, because
every term uses them. QE stores ``g`` in units of ``2 pi / alat`` and ``tau`` in
units of ``alat``, so its ``arg = tpi * g . tau`` is the cartesian ``G . tau``
used below, and a factor ``tpiba`` accompanying a ``g`` is what makes it a
cartesian ``G``. Every term's sum starts at ``gstart``, i.e. skips ``G = 0``,
where the derivative of a structure factor vanishes. And each of them ends up as
the same contraction, ``sum_G G Im[f(G) e^{i G.tau_a}]``, because differentiating
``e^{-i G . tau}`` is all any of them does.

**The spin representation is a trap.** ``sum_band.f90`` converts the density to
``(rho, m)`` before returning, so the ``rho%of_r(:,1)`` that ``force_lc`` is
handed is the **total charge**, not the up channel. This code stores
``(up, down)`` throughout, so the transcription sums the channels; using one of
them would halve the local force in an LSDA run and change nothing in any
unpolarized test. ``force_corr`` shows the other side of the same coin: it
*averages* its two channels, because a potential is stored as ``(v_up, v_down)``
whatever the density is.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from pypresso.basis.fft import r_to_g
from pypresso.scf.potential import (
    as_potential_components,
    exchange_correlation,
    gradient_correction,
    total_charge,
    with_core,
)
from pypresso.units import E2, TPI
from pypresso.vdw.analytic import dispersion_force

__all__ = ["analytic_forces"]


def analytic_forces(calculation, state):
    """QE's forces on the atoms of ``calculation``, plus the term breakdown.

    Returns ``((nat, 3) array, {term: (nat, 3) array})`` in Ry/bohr.

    The six terms are assembled inside **one** compiled function rather than
    evaluated one at a time. That is not a detail: each of them is a handful of
    contractions over the G-vector sphere, and run eagerly they cost more in
    dispatch and in intermediate buffers than in arithmetic. The compiled
    version is cached on the calculation, keyed on the calculation it closed
    over, so a moved or strained one recompiles instead of answering at the old
    geometry.
    """
    if calculation.noncolin:
        raise NotImplementedError(
            "forces for a noncollinear or spin-orbit calculation are not "
            "implemented; nspin = 1 and nspin = 2 are, on norm-conserving, "
            "ultrasoft and PAW pseudopotentials"
        )
    if calculation.is_hubbard:
        # ``force_hub`` is 2552 lines of Fortran -- the derivative of ``ns``
        # with respect to a displacement, which for ortho-atomic projectors
        # carries the derivative of ``O^{-1/2}`` as well. The autodiff force
        # gets all of it by differentiating through the projectors, so
        # transcribing it would be a second implementation of something already
        # validated; a *silent* omission, on the other hand, would leave the
        # force wrong by the whole Hubbard term. Hence the refusal.
        raise NotImplementedError(
            "the analytic force expressions do not include force_hub; use the "
            "autodiff method, which differentiates through the Hubbard "
            "projectors and gets the term for free"
        )
    terms = _compiled_terms(calculation)(state)
    total = sum(terms.values())
    return total, terms


def _compiled_terms(calculation):
    """``jit`` of :func:`_terms` with the calculation captured, cached on it.

    The captured calculation is a *constant* of the compiled function -- its
    positions, its projectors, its local potential -- so the cache is only valid
    for the calculation it was built from. ``at_positions`` copies the instance
    dict, so the entry carries the calculation it belongs to and is rebuilt when
    it does not match; without that the force of every geometry after the first
    is the first one's, silently.
    """
    cached = calculation.__dict__.get("_analytic_terms")
    if cached is None or cached[0] is not calculation:
        cached = (calculation, jax.jit(partial(_terms, calculation)))
        calculation._analytic_terms = cached
    return cached[1]


def _terms(calculation, state) -> dict:
    """Every contribution QE's ``forces()`` sums, as a dict of ``(nat, 3)``."""
    system = calculation.system
    cell, structure = system.cell, system.structure
    dense = calculation.basis.dense
    gcart = dense.cartesian(cell)
    phases = _phases(gcart, structure.positions)  # e^{-i G . tau_a}

    psi, weights = state.wavefunctions, state.weights
    becsum_ = calculation.becsum(psi, weights)

    # The density and the total local potential come from the converged state
    # when it carries them, which is how ``forces.f90`` gets them: they are what
    # the SCF just finished producing, and rebuilding them here would cost four
    # transforms to arrive at the same arrays.
    rho = state.density
    if rho is None:
        rho = calculation.density(psi, weights, becsum_)
    potential = state.potential
    if potential is None:
        scf_potential = calculation.potential(rho)
        potential = scf_potential.v_scf + as_potential_components(
            calculation.vltot, calculation.nspin_mag
        )
    rho_g = jax.vmap(r_to_g, in_axes=(0, None))(rho, dense.fft_index)
    _, ddd_paw = calculation.onecenter(becsum_)

    types = np.asarray(structure.types)
    volume = cell.volume

    terms = {
        "ewald": _force_ew(calculation, gcart, phases),
        "local": _force_lc(
            _species_of_atom(calculation.vloc_species, types),
            total_charge(rho_g), gcart, phases, volume,
        ),
        "core": _force_cc(calculation, rho, gcart, phases, types, volume),
        "nonlocal": _force_us(calculation, state, potential, ddd_paw),
    }
    if calculation.is_ultrasoft:
        terms["augmentation"] = _addusforce(
            calculation, becsum_, potential, gcart, phases, volume
        )
    if calculation.dispersion_sum is not None:
        # ``force_london``, transcribed in :mod:`pypresso.vdw.analytic`. It is
        # the one term here that does not touch the density or the plane-wave
        # basis at all.
        terms["dispersion"] = dispersion_force(
            calculation.dispersion_sum, structure.positions
        )
    if state.potential_change is not None:
        terms["scf_correction"] = _force_corr(
            calculation, state.potential_change, gcart, phases, types, volume
        )
    return terms


def _phases(gcart, positions):
    """``e^{-i G . tau_a}``, ``(nat, ngm)`` -- QE's ``eigts1 * eigts2 * eigts3``."""
    return jnp.exp(-1j * (positions @ gcart.T))


def _species_of_atom(per_species, types) -> jnp.ndarray:
    """A per-species ``(ngm,)`` table repeated onto the atoms, ``(nat, ngm)``."""
    return jnp.stack([per_species[t] for t in types])


@jax.jit
def _structure_derivative(radial, field, gcart, phases, prefactor):
    """``prefactor * sum_G G Im[field(G) e^{i G.tau_a}] * radial_a(|G|)``.

    Every one of QE's reciprocal-space force terms is this contraction; what
    changes between them is which radial table sits on the atom and which field
    it is paired with. Writing it once is not only shorter -- it is what makes
    the ``G = 0`` exclusion and the sign of the phase impossible to get right in
    one term and wrong in another.
    """
    weight = radial * jnp.imag(field[None, :] * jnp.conj(phases))  # (nat, ngm)
    weight = weight.at[:, 0].set(0.0)  # G = 0 contributes nothing
    return prefactor * (weight @ gcart)


def _force_lc(radial, rho_g, gcart, phases, volume):
    """``force_lc``: the local pseudopotential dragged through the density."""
    return _structure_derivative(radial, rho_g, gcart, phases, volume)


def _force_cc(calculation, rho, gcart, phases, types, volume):
    """``force_cc``: the nonlinear core charge dragged through ``v_xc``.

    The exchange-correlation potential here is the **whole** of it, gradient
    correction included. That is worth stating because ``force_cc`` calls
    ``v_xc`` and the gradient correction looks as though it is added separately
    by ``v_of_rho`` -- it is not: ``gradcorr`` is called from *inside* ``v_xc``
    (``PW/src/v_of_rho.f90``, line 607, which is within the routine that starts
    at line 440), and it is the only place in ``PW/src`` that calls it. Building
    this term from the local part alone changes silicon's core force by 9e-4
    Ry/bohr, which is a thousand times the agreement the rest of the terms
    reach -- and it is what the comparison against the autodiff force says,
    since that one differentiates the energy and cannot leave a piece out.
    """
    if calculation.rho_core is None:
        return jnp.zeros((len(types), 3))
    v_xc = _xc_potential(calculation, rho)
    # "the exchange-correlation potential" for a two-channel run is the average
    # of the channels: the core charge is shared equally between them.
    v_xc = jnp.mean(v_xc, axis=0)
    v_g = r_to_g(v_xc, calculation.basis.dense.fft_index)
    radial = _species_of_atom(calculation.rho_core_species, types)
    return _structure_derivative(radial, v_g, gcart, phases, volume)


def _xc_potential(calculation, rho):
    """``v_xc`` as QE's routine of that name computes it, gradient part included.

    The same assembly :func:`~pypresso.scf.potential.v_of_rho` does, minus the
    Hartree term: the functional is evaluated at ``rho + rho_core`` and, for a
    gradient-corrected functional, the divergence term is added on top.
    """
    dense = calculation.basis.dense
    cell = calculation.system.cell
    functional = calculation.functional
    v_xc, _ = exchange_correlation(rho, cell, calculation.rho_core, functional)
    if not functional.is_gradient:
        return v_xc

    nspin = rho.shape[0]
    density_r = jnp.real(rho)
    density_g = jax.vmap(r_to_g, in_axes=(0, None))(rho, dense.fft_index)
    if calculation.rho_core is not None:
        density_r = density_r + with_core(jnp.real(calculation.rho_core), nspin)
        density_g = density_g + with_core(calculation.rho_core_g, nspin)
    v_gradient, _ = gradient_correction(density_r, density_g, dense, cell, functional)
    return v_xc + v_gradient


def _force_corr(calculation, potential_change, gcart, phases, types, volume):
    """``force_corr``: the term that vanishes at self-consistency.

    Chan, Bohnen and Ho's correction (PRB 47, 4771): the true density is
    approximated by a superposition of atomic charges, and its displacement is
    paired with ``V_out - V_in``, the potential change the last SCF step did not
    make. It is the one term of the six that a force obtained by differentiating
    the energy *at the fixed point* cannot have, and comparing it against zero is
    the cheapest statement of how converged a run is.
    """
    change = jnp.mean(potential_change, axis=0)  # (v_up + v_down) / 2
    v_g = r_to_g(change, calculation.basis.dense.fft_index)
    radial = _species_of_atom(calculation.rho_atomic_species, types)
    return _structure_derivative(radial, v_g, gcart, phases, volume)


def _force_ew(calculation, gcart, phases):
    """``force_ew``: the ion-ion term, in reciprocal and in real space.

    QE picks a different ``alpha`` here than ``ewald.f90`` does for the energy
    (1.1 stepping down against a 1e-6 bound, rather than 2.9 against 1e-7) and a
    real-space cutoff of ``5/sqrt(alpha)`` rather than ``4/sqrt(alpha)``. The sum
    is independent of both, so this uses the energy's -- which is what
    :class:`~pypresso.scf.ewald.EwaldSum` already fixed for the cell, neighbour
    list included.
    """
    ewald = calculation.ewald_sum
    charges, alpha = ewald.charges, ewald.alpha
    cell = calculation.system.cell
    positions = calculation.system.structure.positions

    g2 = jnp.sum(gcart**2, axis=1)
    safe = jnp.where(g2 > 1e-12, g2, 1.0)
    # rho_ion(G)* = sum_a Z_a e^{+i G tau_a}, screened
    ionic = jnp.sum(charges[:, None] * jnp.conj(phases), axis=0)
    screened = jnp.where(g2 > 1e-12, jnp.exp(-safe / alpha / 4.0) / safe, 0.0)
    field = ionic * screened

    weight = jnp.imag(field[None, :] * phases)  # (nat, ngm)
    reciprocal = -(charges[:, None] * (weight @ gcart)) * 2.0 * E2 * TPI / cell.volume

    return reciprocal + _ewald_real_force(
        positions, charges, ewald.translations, alpha, ewald.rmax
    )


@jax.jit
def _ewald_real_force(tau, charges, translations, alpha, rmax):
    """``-d/dtau`` of ``e2/2 sum erfc(sqrt(alpha) r)/r`` over pairs and images."""
    from jax.scipy.special import erfc

    separations = (
        tau[:, None, None, :] - tau[None, :, None, :] + translations[None, None, :, :]
    )
    square = jnp.sum(separations**2, axis=-1)
    keep = (square > 1.0e-16) & (square <= rmax**2)
    r = jnp.sqrt(jnp.where(keep, square, 1.0))

    # d/dr [erfc(sqrt(a) r)/r] = -[erfc(sqrt(a) r)/r^2 + 2 sqrt(a/pi) e^{-a r^2}/r]
    radial = erfc(jnp.sqrt(alpha) * r) / r**2 + 2.0 * jnp.sqrt(
        alpha / jnp.pi
    ) * jnp.exp(-alpha * r**2) / r
    radial = jnp.where(keep, radial / r, 0.0)  # the extra 1/r makes it r_vec/|r|

    pairs = charges[:, None] * charges[None, :]
    return E2 * jnp.einsum("ab,abt,abtc->ac", pairs, radial, separations)


def _force_us(calculation, state, total, ddd_paw):
    """``force_us``: the projectors moving under the atoms they belong to.

    ``deff = deeq - eps qq`` is the whole of the ultrasoft story in one line:
    the first half is the nonlocal potential's own derivative, the second is the
    derivative of the orthonormality constraint ``<psi|S|psi> = 1``, which for a
    norm-conserving dataset is absent because ``S`` is the identity. It is what
    the autodiff force gets from the Lagrange term of
    :func:`~pypresso.forces.energy.frozen_energy`.
    """
    projectors = calculation.projectors
    if projectors.nkb == 0:
        return jnp.zeros((calculation.system.structure.nat, 3))

    deeq = calculation.coefficients(total, ddd_paw)
    if deeq is None:
        deeq = jnp.broadcast_to(projectors.dij, (calculation.nspin,) + projectors.dij.shape)
    qq = projectors.qq
    if qq is None:
        qq = jnp.zeros_like(projectors.dij)

    kg = calculation.projector_core.kg  # (nk, npwx, 3) -- k + G
    kpoints = calculation.system.kpoints.cartesian(calculation.system.cell)
    g_of_pw = kg - kpoints[:, None, :]  # QE differentiates with G, not k + G

    channels = jnp.asarray(calculation.projectors.atom_of_channel)
    nat = calculation.system.structure.nat
    return _projector_force(
        state.wavefunctions, projectors.vkb, g_of_pw, deeq, qq,
        state.weights, state.eigenvalues, channels, nat,
    )


@partial(jax.jit, static_argnames=("nat",))
def _projector_force(psi, vkb, gcart_of_pw, deeq, qq, weights, eigenvalues,
                     atom_of_channel, nat):
    """``-2 sum_kn w f deff_ij Re[<psi|beta_i>* <beta_j| i G |psi>]``.

    ``<beta|psi>`` differentiated with respect to the atom's position brings down
    an ``i G`` inside the integral, which is why QE builds ``vkb1 = -i G vkb``
    and projects the wavefunctions on it a second time. Only ``G`` appears, not
    ``k + G``: the ``k`` piece is antisymmetric in ``(i, j)`` where ``deff`` is
    symmetric, so it cancels in the sum.

    **Memory.** ``deeq`` is contracted with the *derivative* projections before
    they meet the ordinary ones, so nothing of size ``nkb^2`` per band is ever
    formed: the working set is three copies of ``becp``, i.e.
    ``3 nspin nk nbnd nkb`` complex numbers.
    """
    becp = jnp.einsum("skbg,kgc->skbc", psi, jnp.conj(vkb))
    becd = jnp.einsum("skbg,kgc,kgx->xskbc", psi, jnp.conj(vkb), 1j * gcart_of_pw)

    deff_becd = jnp.einsum("sij,xskbj->xskbi", deeq.astype(becd.dtype), becd)
    overlap_becd = jnp.einsum("ij,xskbj->xskbi", qq.astype(becd.dtype), becd)
    per_channel = jnp.real(
        jnp.conj(becp)[None] * (deff_becd - eigenvalues[None, ..., None] * overlap_becd)
    )
    summed = jnp.einsum("xskbi,skb->xi", per_channel, weights)

    onto_atoms = jnp.zeros((3, nat)).at[:, atom_of_channel].add(summed)
    return -2.0 * onto_atoms.T


def _addusforce(calculation, becsum_, total, gcart, phases, volume):
    """``addusforce``: the augmentation charge moving with its atom.

    ``F = sum_G i G V*(G) Q_ij(G) e^{-i G tau} becsum_ij``. The potential is the
    **total** effective one, ``vltot`` included -- the same ``V_eff`` that
    ``newd`` integrates ``Q`` against, since this term is the position
    derivative of that integral.
    """
    augmentation = calculation.augmentation
    dense = calculation.basis.dense
    nat = calculation.system.structure.nat
    forces = jnp.zeros((nat, 3))

    for spin in range(calculation.nspin_mag):
        v_g = r_to_g(total[spin], dense.fft_index)
        for t, atoms in enumerate(augmentation.species_atoms):
            qgm = augmentation.qgm[t]
            if qgm.shape[0] == 0 or not atoms:
                continue
            index = jnp.asarray(atoms)
            # becsum is (nspin, nat_t, nh, nh) whatever nspin is -- the spin
            # axis is never squeezed on the way in, only on the way out.
            contribution = _augmentation_force(
                qgm, becsum_[t][spin], v_g, phases[index], gcart, volume,
            )
            forces = forces.at[index].add(contribution)
    return forces


@jax.jit
def _augmentation_force(qgm, becsum, v_g, phases, gcart, volume):
    """One species' contribution: ``(nat_t, 3)``."""
    # sum_ij becsum_ij Q_ij(G) -- the same contraction addusdens does, per atom
    charge = jnp.einsum("aij,ijg->ag", becsum.astype(qgm.dtype), qgm)
    field = jnp.conj(charge) * (v_g * jnp.conj(phases))
    weight = jnp.imag(field)
    weight = weight.at[:, 0].set(0.0)
    return volume * (weight @ gcart)
