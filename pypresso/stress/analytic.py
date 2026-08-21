"""Quantum ESPRESSO's stress, term by term.

``PW/src/stress.f90`` and the routines it calls, transcribed. This is the
*other* way to get the stress -- the one that writes each derivative down by
hand -- and it exists here for the reason the analytic forces do: it is what the
reference implementation does, and it is a check on the autodiff stress that
shares none of its machinery.

The terms, each with the file it comes from:

``kinetic``   ``stres_knl.f90``      ``e2/Omega sum w f (k+G)_a (k+G)_b |c|^2``
``hartree``   ``stres_har.f90``      ``G_a G_b/(G^2)^2`` against ``|rho(G)|^2``
``local``     ``stres_loc.f90``      ``dV_loc/dG^2`` through the structure factor
``xc``        ``stress.f90``, inline the diagonal ``-(etxc - vtxc)/Omega``
``core``      ``stres_cc.f90``       the core charge moving inside ``v_xc``
``gradcorr``  ``stres_gradcorr.f90`` a GGA's non-diagonal ``v2 grad_a grad_b``
``ewald``     ``stres_ewa.f90``      the ion-ion sum, both halves

**Two are missing and it is deliberate.** ``stres_us.f90`` -- the projectors'
own strain derivative -- needs ``gen_us_dj`` (the radial form factor
differentiated with respect to ``|k+G|``) and ``gen_us_dy`` (``dylmr2``, the
spherical harmonics differentiated with respect to direction), which together
are a transcription the size of everything above; ``addusstress.f90`` needs the
same pair for ``Q_ij(G)``. Neither is here, so **this module offers no total**:
the sum of what is written would be missing the whole nonlocal pseudopotential,
which on silicon is a third of the pressure and looks entirely plausible.
:func:`analytic_terms` returns a dict, and
:func:`~pypresso.stress.compute_stress` refuses ``method = 'analytic'`` by name.

**How the two decompositions line up**, because it is not one for one and the
cross-check needs the mapping stated:

* the autodiff ``xc`` term is the *whole* exchange-correlation strain
  derivative, and on QE's side that is ``xc + core + gradcorr`` -- the diagonal,
  the core charge and the gradient correction together, because all three come
  from differentiating one ``etxc``;
* the autodiff ``kinetic`` term is ``stres_knl``'s ``sigmakin`` alone, its
  nonlocal half having gone into ``nonlocal``, which is the term with no
  counterpart here;
* for an ultrasoft dataset the autodiff ``local``, ``hartree`` and ``xc`` terms
  each contain a piece of QE's ``addusstress``, since the augmentation charge
  lives inside ``rho`` here and is a separate routine there.

The first two comparisons are what the cross-check test makes, and it makes them
on a norm-conserving cell for the third's sake.

**Conventions carried over from the Fortran**, stated once because every term
uses them. QE's ``g`` is in units of ``2 pi / alat`` and ``gg`` its square, so a
``tpiba``/``tpiba2`` beside a ``g`` is what makes it cartesian; this module works
in cartesian 1/bohr throughout, and the ``tpiba2`` factors of the Fortran are
therefore absent rather than forgotten. Every reciprocal-space term skips
``G = 0`` (QE's ``gstart``), where the structure factor's derivative vanishes and
the Hartree kernel does not exist. Each term is symmetric by construction, so QE
fills the upper triangle from the lower; here the full contraction is written
and the symmetrisation over the point group is the caller's
(:mod:`pypresso.stress`).

**The density is the total charge**, ``rho%of_r(:,1)`` -- which for an LSDA run
is the *sum* of this code's two channels and not its first one. That is P15's
trap 3 in a second place, and it is the reason ``total_charge`` appears in every
term below.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf, erfc

from pypresso.basis.fft import r_to_g
from pypresso.basis.gradients import gradient
from pypresso.basis.gvectors import modulus
from pypresso.pseudo.formfactors import local_potential_of_g
from pypresso.pseudo.potentials import structure_factors
from pypresso.pseudo.radial import simpson_weights, spherical_bessel
from pypresso.scf.potential import (
    exchange_correlation,
    gradient_correction,
    total_charge,
    with_core,
)
from pypresso.units import E2, FPI, TPI

__all__ = ["analytic_terms", "ANALYTIC_TERMS"]

#: The contributions this module writes down. Named here so a test can assert
#: that the list has not silently shrunk, and so the mapping above has something
#: to refer to.
ANALYTIC_TERMS = ("kinetic", "hartree", "local", "xc", "core", "gradcorr", "ewald")

#: QE's ``eps8``: below this a G-vector is the origin.
_EPS = 1.0e-8


def analytic_terms(calculation, state) -> dict:
    """QE's stress contributions for ``calculation`` in state ``state``.

    Returns ``{term: (3, 3) array}`` in Ry/bohr^3, **unsymmetrised** and
    **incomplete** -- the nonlocal pseudopotential is not among them, see the
    module docstring.
    """
    if calculation.noncolin:
        raise NotImplementedError(
            "the analytic stress expressions are written for nspin = 1 and "
            "nspin = 2; a spinor run needs stres_knl's noncollinear branch"
        )
    cell = calculation.system.cell
    dense = calculation.basis.dense
    volume = float(cell.volume)

    psi, weights = state.wavefunctions, state.weights
    rho = state.density
    if rho is None:
        rho = calculation.density(psi, weights, calculation.becsum(psi, weights))
    rho_g = r_to_g(total_charge(jnp.real(rho)), dense.fft_index)

    gcart = dense.cartesian(cell)
    factors = structure_factors(calculation.system.structure, cell, dense)

    terms = {
        "kinetic": _kinetic(
            psi, weights, calculation.basis.planewaves,
            calculation.basis.smooth.cartesian(cell),
            calculation.system.kpoints.cartesian(cell), volume,
        ),
        "hartree": _hartree(rho_g, gcart, volume),
        "local": _local(calculation, rho_g, gcart, factors, volume),
        "ewald": _ewald(calculation, cell, gcart, volume),
    }
    terms.update(
        _exchange_correlation(calculation, rho, gcart, factors, volume)
    )
    return {name: np.asarray(value, dtype=float) for name, value in terms.items()}


# --- stres_knl ---------------------------------------------------------------
def _kinetic(psi, weights, planewaves, gcart, kcart, volume):
    """``stres_knl``'s ``sigmakin``.

    ``sigma_ab = (e2/Omega) sum_kn w f sum_G (k+G)_a (k+G)_b |c_G|^2``, which is
    ``-1/Omega`` times the strain derivative of ``sum w f |k+G|^2 |c|^2``,
    because ``|k+G|^2`` is where ``G -> (1 + eps)^-T G`` lands. The ``e2 = 2``
    is the ``2`` of ``d(x^2)/dx`` rather than a unit conversion -- the energy is
    already in Rydberg here -- and it is a coincidence of QE's units that the
    two are the same number.
    """
    kg = kcart[:, None, :] + gcart[planewaves.indices]  # (nk, npwx, 3)
    kg = jnp.where(planewaves.mask[..., None], kg, 0.0)
    occupied = jnp.einsum("skn,skng->kg", weights, jnp.abs(psi) ** 2)
    return E2 / volume * jnp.einsum("kg,kga,kgb->ab", occupied, kg, kg)


# --- stres_har ---------------------------------------------------------------
def _hartree(rho_g, gcart, volume):
    """``stres_har``.

    Two pieces and a sign flip: ``2 G_a G_b / (G^2)^2`` weighted by
    ``|rho(G)|^2``, minus ``E_H/Omega`` on the diagonal, the whole negated --
    which is what turns a derivative into a stress.
    """
    g2 = jnp.sum(gcart**2, axis=1)
    inverse = jnp.where(g2 > _EPS, 1.0 / jnp.where(g2 > _EPS, g2, 1.0), 0.0)

    weight = jnp.abs(rho_g) ** 2 * inverse
    tensor = 2.0 * jnp.einsum("g,ga,gb->ab", weight * inverse, gcart, gcart)
    energy = 0.5 * volume * E2 * FPI * jnp.sum(weight)
    return -(0.5 * FPI * E2 * tensor - jnp.eye(3) * energy / volume)


# --- stres_loc ---------------------------------------------------------------
def _local(calculation, rho_g, gcart, factors, volume):
    """``stres_loc``.

    ``sigma_ab = sum_G Re[conj(rho(G)) S_t(G)] 2 dV_t/dG^2 G_a G_b`` with
    ``evloc`` on the diagonal. ``evloc`` is ``E_loc/Omega`` -- the *value* of the
    term rather than a derivative, and it is there because ``V_loc(G)`` carries
    an explicit ``1/Omega`` that a strain differentiates on its own.
    """
    gmod = modulus(gcart)
    derivatives = jnp.stack([
        _dvloc_of_g2(pseudo, gmod, volume) for pseudo in calculation.pseudos
    ])  # (ntyp, ngm), dV/d(G^2)
    values = jnp.stack([
        local_potential_of_g(pseudo, gmod, volume) for pseudo in calculation.pseudos
    ])

    weight = 2.0 * jnp.sum(
        jnp.real(jnp.conj(rho_g)[None, :] * factors) * derivatives, axis=0
    )
    tensor = jnp.einsum("g,ga,gb->ab", weight, gcart, gcart)
    evloc = jnp.sum(jnp.real(jnp.conj(rho_g)[None, :] * factors) * values)
    return tensor + jnp.eye(3) * evloc


def _dvloc_of_g2(pseudo, q, omega):
    """``dV_loc/d(G^2)`` -- ``upflib/vloc_mod.f90``'s ``dvloc_of_g``.

    The same splitting :func:`~pypresso.pseudo.formfactors.local_potential_of_g`
    uses, differentiated: the erf-screened remainder under the integral
    (``d/dq [sin(qr)/q] = [r cos(qr) - sin(qr)/q] / q``) and the analytic
    Coulomb transform in closed form, ``4 pi Z e2 e^{-u/4}(u/4 + 1)/(Omega u^2)``
    with ``u = q^2``. QE returns ``dV/d(q^2)`` rather than ``dV/dq`` because that
    is what its contraction against ``g_a g_b`` wants, and the transcription
    keeps the convention so that the two can be compared line by line.

    ``G = 0`` is zero by fiat, as it is there: the term it belongs to is the
    ``evloc`` diagonal, not this one.
    """
    r = jnp.asarray(pseudo.r[: pseudo.msh])
    weights = simpson_weights(jnp.asarray(pseudo.rab[: pseudo.msh]))
    vloc = jnp.asarray(pseudo.vloc[: pseudo.msh])
    short = r * vloc + pseudo.z_valence * E2 * erf(r)
    return _dvloc_kernel(
        jnp.atleast_1d(q), r, weights, short, float(pseudo.z_valence), omega
    )


@jax.jit
def _dvloc_kernel(q, r, weights, short, z, omega):
    small = q < _EPS
    safe = jnp.where(small, 1.0, q)[:, None]
    argument = safe * r[None, :]
    integrand = short[None, :] * (
        r[None, :] * jnp.cos(argument) / safe - jnp.sin(argument) / safe**2
    )
    # dV_short/dq, then /(2q) for QE's dV/d(q^2).
    value = (integrand @ weights) * FPI / omega / (2.0 * safe[:, 0])

    u = safe[:, 0] ** 2
    long_range = FPI / omega * z * E2 * jnp.exp(-u * 0.25) * (u * 0.25 + 1.0) / u**2
    return jnp.where(small, 0.0, value + long_range)


# --- the exchange-correlation family ----------------------------------------
def _exchange_correlation(calculation, rho, gcart, factors, volume) -> dict:
    """``sigmaxc`` (diagonal), ``stres_cc`` and ``stres_gradcorr``.

    The first is ``stress.f90``'s own two lines, ``sigma_ll =
    -(etxc - vtxc)/Omega``, and it needs ``vtxc`` -- which nothing in this code
    stores, because the SCF never wants it -- so the potential is recomputed
    here and integrated against the valence density.

    **``vtxc`` is against the valence density and ``etxc`` is over the total.**
    That asymmetry is the same one :func:`~pypresso.scf.potential.exchange_correlation`
    documents, and it is exactly what makes ``stres_cc``'s diagonal the missing
    half: the two together are ``-(etxc - int v_xc (rho + rho_core))/Omega``,
    which is what differentiating ``etxc`` through the explicit ``1/Omega`` of
    *both* densities gives.

    **The potential includes the gradient correction.** ``v_of_rho`` calls
    ``gradcorr`` from *inside* ``v_xc`` (line 607 of ``v_of_rho.f90``, within the
    ``v_xc`` that starts at 440), so the ``vxc`` ``stres_cc`` transforms is the
    full one and so is the ``vtxc`` on the diagonal. This is P15's trap 2 in a
    third place; taking the local part alone leaves a GGA stress wrong in the
    third decimal.
    """
    functional = calculation.functional
    valence = jnp.real(rho)
    nspin = valence.shape[0]
    rho_core = calculation.rho_core
    n = valence[0].size

    v_xc, etxc = exchange_correlation(valence, calculation.system.cell,
                                      rho_core, functional)

    density_r = valence
    density_g = jax.vmap(r_to_g, in_axes=(0, None))(valence, calculation.basis.dense.fft_index)
    if rho_core is not None:
        density_r = density_r + with_core(jnp.real(rho_core), nspin)
        density_g = density_g + with_core(calculation.rho_core_g, nspin)

    gradcorr = jnp.zeros((3, 3))
    if functional.is_gradient:
        v_gradient, e_gradient = gradient_correction(
            density_r, density_g, calculation.basis.dense,
            calculation.system.cell, functional,
        )
        v_xc = v_xc + v_gradient
        etxc = etxc + e_gradient
        gradcorr = _gradcorr(density_r, density_g, calculation, functional)

    vtxc = volume / n * jnp.sum(v_xc * valence)
    diagonal = -jnp.eye(3) * (etxc - vtxc) / volume
    core = _core_charge(calculation, v_xc, gcart, factors, volume)
    return {"xc": diagonal, "core": core, "gradcorr": gradcorr}


def _gradcorr(density_r, density_g, calculation, functional):
    """``stres_gradcorr``: ``(1/N) sum_r sum_s h_{s,a} (grad rho_s)_b``.

    The non-diagonal part a gradient-corrected functional adds. It comes from
    ``grad -> (1 + eps)^-T grad``, so what survives is the outer product of the
    gradient with the energy's derivative with respect to it.

    QE writes that derivative as ``e2 (v2x + v2c)`` per channel plus a
    hand-added cross term ``v2c_ud``, because correlation depends on the *total*
    density's gradient; here ``h`` is already the derivative with respect to the
    gradient **field**, so the cross term is part of it and the pairing of
    channels cannot be got the wrong way round -- the same argument
    :func:`~pypresso.scf.potential.gradient_correction` makes for the potential.
    """
    gvectors, cell = calculation.basis.dense, calculation.system.cell
    nspin = density_r.shape[0]
    grad = jax.vmap(gradient, in_axes=(0, None, None))(density_g, gvectors, cell)
    n = density_r[0].size

    if nspin == 1:
        sigma = jnp.sum(grad[0] * grad[0], axis=0)
        _, v2 = functional.gradient_potentials(density_r[0], sigma)
        flat = grad[0].reshape(3, -1)
        return jnp.einsum("r,ar,br->ab", v2.ravel(), flat, flat) / n

    _, h = functional.spin_gradient_terms(density_r, grad)
    return jnp.einsum(
        "sar,sbr->ab", h.reshape(nspin, 3, -1), grad.reshape(nspin, 3, -1)
    ) / n


def _core_charge(calculation, v_xc, gcart, factors, volume):
    """``stres_cc``.

    The nonlinear core charge moves with the cell, so ``E_xc`` picks up
    ``int v_xc d(rho_core)/d(eps)``: a diagonal ``sum_G Re[conj(v_xc) rho_c]``
    from the ``1/Omega`` inside the transform, and the radial derivative
    ``d rho_c/d|G|`` contracted with ``G_a G_b / |G|``.

    With two channels ``v_xc`` enters as the **average** of them, because the
    core charge is unpolarized -- ``stres_cc`` writes
    ``vxc(:,1) = (vxc(:,1) + vxc(:,2))/2`` in as many words, and it is the same
    rule ``force_cc`` follows (P15 trap 3, the half of it that averages).
    """
    if calculation.rho_core is None:
        return jnp.zeros((3, 3))
    gvectors = calculation.basis.dense
    nspin = v_xc.shape[0]
    vxc_r = jnp.mean(jnp.real(v_xc), axis=0) if nspin > 1 else jnp.real(v_xc[0])
    vxc_g = r_to_g(vxc_r, gvectors.fft_index)

    gmod = modulus(gcart)
    inverse = jnp.where(gmod > _EPS, 1.0 / jnp.where(gmod > _EPS, gmod, 1.0), 0.0)
    derivatives = jnp.stack([
        _drhoc_of_g(pseudo, gmod, volume) if pseudo.has_nlcc else jnp.zeros_like(gmod)
        for pseudo in calculation.pseudos
    ])
    values = jnp.stack([
        _rhoc_of_g(pseudo, gmod, volume) if pseudo.has_nlcc else jnp.zeros_like(gmod)
        for pseudo in calculation.pseudos
    ])

    projected = jnp.real(jnp.conj(vxc_g)[None, :] * factors)
    diagonal = jnp.eye(3) * jnp.sum(projected * values)
    weight = jnp.sum(projected * derivatives, axis=0) * inverse
    return diagonal + jnp.einsum("g,ga,gb->ab", weight, gcart, gcart)


def _rhoc_of_g(pseudo, q, omega):
    from pypresso.pseudo.formfactors import core_charge_of_g

    return core_charge_of_g(pseudo, q, omega)


def _drhoc_of_g(pseudo, q, omega):
    """``d rho_core/d|G|`` -- ``upflib/rhoc_mod.f90``'s ``drhoc``.

    The ``j_0`` transform differentiated, using ``d j_0(x)/dx = -j_1(x)``.
    """
    r = jnp.asarray(pseudo.r[: pseudo.msh])
    weights = simpson_weights(jnp.asarray(pseudo.rab[: pseudo.msh]))
    rho = jnp.asarray(pseudo.rho_core[: pseudo.msh])
    return _drhoc_kernel(jnp.atleast_1d(q), r, weights, rho, omega)


@jax.jit
def _drhoc_kernel(q, r, weights, rho, omega):
    argument = q[:, None] * r[None, :]
    integrand = -FPI * r[None, :] ** 3 * rho[None, :] * spherical_bessel(1, argument)
    return (integrand @ weights) / omega


# --- stres_ewa ---------------------------------------------------------------
def _ewald(calculation, cell, gcart, volume):
    """``stres_ewa``, both halves.

    The reciprocal sum differentiated through ``G -> (1 + eps)^-T G`` -- which
    is where the ``(g2a + 1)`` comes from, one from ``1/G^2`` and ``g2a`` from
    the Gaussian -- and the real sum through ``r -> (1 + eps) r``, with the
    ``G = 0`` and self-interaction pieces collected in ``sdewald`` on the
    diagonal. The whole is negated at the end, as ``stres_har`` is.

    ``alpha`` and the image list are this calculation's own
    (:class:`~pypresso.scf.ewald.EwaldSum`), not re-derived: the split is exact
    for any ``alpha``, and reusing the one the energy was computed with is what
    makes the two consistent to the last digit rather than to the truncation.
    """
    ewald = calculation.ewald_sum
    positions = calculation.system.structure.positions
    return _ewald_kernel(
        gcart, positions, ewald.charges, ewald.translations,
        ewald.alpha, ewald.rmax, volume,
    )


@jax.jit
def _ewald_kernel(gcart, tau, charges, translations, alpha, rmax, volume):
    charge = jnp.sum(charges)
    g2 = jnp.sum(gcart**2, axis=1)

    # rho*(G) = sum_a Z_a e^{i G.tau_a} / Omega, and only |rho*|^2 is used.
    rhostar = jnp.sum(charges * jnp.exp(1j * (gcart @ tau.T)), axis=1) / volume
    g2a = g2 / (4.0 * alpha)
    keep = g2 > _EPS
    safe = jnp.where(keep, g2, 1.0)
    sewald = jnp.where(
        keep, TPI * E2 * jnp.exp(-g2a) / safe * jnp.abs(rhostar) ** 2, 0.0
    )

    # sdewald: the G = 0 term, which for a neutral cell is what remains of the
    # divergence, minus the whole reciprocal sum.
    sdewald = TPI * E2 / (4.0 * alpha) * (charge / volume) ** 2 - jnp.sum(sewald)
    reciprocal = 2.0 * jnp.einsum(
        "g,ga,gb->ab", sewald * (g2a + 1.0) / safe, gcart, gcart
    ) + jnp.eye(3) * sdewald

    # The real-space half: every pair, every image, the self term and everything
    # past ``rmax`` dropped by weight rather than by indexing so the shape stays
    # static. The mask is on the *squared* distance, before the square root --
    # P15's trap 1, and it bites here through the strain rather than through a
    # position.
    separations = (
        tau[:, None, None, :] - tau[None, :, None, :] + translations[None, None, :, :]
    )
    square = jnp.sum(separations**2, axis=-1)
    inside = (square > 1.0e-16) & (square <= rmax**2)
    distance = jnp.sqrt(jnp.where(inside, square, 1.0))
    pairs = charges[:, None] * charges[None, :]
    factor = jnp.where(
        inside,
        -E2 / (2.0 * volume) * pairs[:, :, None] / distance**3
        * (
            erfc(jnp.sqrt(alpha) * distance)
            + distance * jnp.sqrt(8.0 * alpha / TPI) * jnp.exp(-alpha * distance**2)
        ),
        0.0,
    )
    real = jnp.einsum("ijn,ijna,ijnb->ab", factor, separations, separations)

    return -(reciprocal + real)
