"""The PAW one-centre corrections.

PAW is ultrasoft plus a pair of bookkeeping corrections. The plane-wave grid
carries a *pseudo* density that is smooth by construction and therefore wrong
inside the augmentation spheres; PAW fixes that by recomputing the Hartree and
exchange-correlation energy twice on each atom's own radial mesh -- once from
the all-electron partial waves and once from the pseudo ones -- and adding the
difference:

    E = E[pseudo density on the grid]
      + sum_a ( E_Hxc[rho^1_a] - E_Hxc[rho~^1_a] )

Neither one-centre term is small (silicon's is -67 Ry against a total of -89),
but the difference is what the grid could not represent, and it is local to each
sphere, which is what makes the scheme work at a cutoff an ultrasoft potential
would need anyway.

Following ``PW/src/paw_onecenter.f90``. The three pieces are:

* **the on-site densities**, ``rho^1_lm(r) = sum_ij becsum_ij phi_i phi_j``
  resolved into multipoles with the same coupling coefficients the augmentation
  charge uses (``PAW_rho_lm``). The pseudo one additionally carries the
  augmentation charge ``Q_ij(r)``, since that is what was added to the grid
  density and has to be subtracted back here;
* **Hartree**, one radial Poisson solve per ``lm`` (:mod:`pypresso.paw.hartree`);
* **exchange-correlation**, evaluated pointwise on a spherical quadrature
  (:mod:`pypresso.paw.angular`) because it does not act multipole by multipole.

Out of it come ``epaw``, which QE prints as its own energy term, and
``ddd_paw``, which is added to the nonlocal coefficients exactly as the
ultrasoft ``int V Q`` term is (``add_paw_to_deeq``).

**What is linear in ``becsum``, and why that matters here.** ``rho^1_lm`` is,
which is what lets ``ddd_paw`` -- defined as the derivative of the one-centre
energy with respect to ``becsum`` -- be one contraction rather than the
``nh(nh+1)/2`` separate density rebuilds ``PAW_potential`` does with its
``becfake`` trick. The same tensor serves both directions.

**Spin.** ``becsum`` and therefore ``rho_lm``, the potential and ``ddd`` all
carry a channel axis. The two one-centre terms split the way they do on the
grid, and for the same reason: **Hartree is solved once, for the summed
density, and copied to both channels** (``PAW_h_potential`` sums over
``nspin_lsda`` before calling the radial Poisson solver, and
``PAW_potential`` copies its answer into ``savedv_lm`` for every spin), while
exchange-correlation is evaluated per channel on the sphere.
"""

from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from pypresso.paw.angular import AngularGrid, build_angular_grid
from pypresso.paw.gradient import onecenter_gradient_correction
from pypresso.paw.hartree import radial_hartree
from pypresso.pseudo.coupling import harmonic_products
from pypresso.pseudo.projectors import projector_channels
from pypresso.pseudo.radial import simpson_weights
from pypresso.pseudo.upf import Pseudopotential
from pypresso.units import E2, FPI
from pypresso.xc.functional import Functional

__all__ = ["PawSpecies", "PawCorrections", "build_paw", "onecenter_species"]


class PawSpecies(eqx.Module):
    """Everything one PAW species contributes, precomputed on its radial mesh.

    ``density_ae`` and ``density_ps`` are the tensors that turn ``becsum`` into
    on-site multipole densities: ``rho_lm = einsum('ij,ijlr->lr', becsum, t)``,
    holding ``r^2 rho_lm(r)`` as QE's ``rho_lm`` does. They are ``(nh, nh, nlm,
    mesh)`` and symmetric in their first two indices.
    """

    density_ae: jnp.ndarray
    density_ps: jnp.ndarray
    core_ae: jnp.ndarray  # (mesh,) the true core charge
    core_ps: jnp.ndarray  # (mesh,) the NLCC core charge
    r: jnp.ndarray
    r2: jnp.ndarray
    sqr: jnp.ndarray
    #: Simpson weights over the whole mesh (energies) and over ``kkbeta``
    #: (the ``ddd`` integrals) -- QE uses different ranges for the two.
    weights_full: jnp.ndarray
    weights_core: jnp.ndarray
    angular: AngularGrid
    functional: Functional = eqx.field(static=True)
    dx: float = eqx.field(static=True)
    nlm: int = eqx.field(static=True)
    nh: int = eqx.field(static=True)


class PawCorrections(eqx.Module):
    """The PAW species of a structure, and the atoms each applies to."""

    species: tuple  # PawSpecies or None, one per species
    species_atoms: tuple = eqx.field(static=True)

    def energy_and_coefficients(self, becsum: tuple):
        """``(epaw, ddd)`` from the current ``becsum``.

        ``ddd`` matches ``becsum``'s layout -- one ``(nspin, nat_t, nh_t, nh_t)``
        array per species -- so it drops straight into the same block assembly
        the ultrasoft ``int V Q`` term uses.
        """
        energy = jnp.asarray(0.0)
        coefficients = []
        for paw, atoms, values in zip(self.species, self.species_atoms, becsum):
            if paw is None or not atoms:
                coefficients.append(
                    None if values is None else jnp.zeros_like(values)
                )
                continue
            # over atoms: becsum is (nspin, nat, nh, nh) and the atom axis is
            # the one that batches, so it is moved to the front for the map.
            atom_energy, atom_ddd = jax.vmap(
                partial(onecenter_species, paw), in_axes=1, out_axes=(0, 1)
            )(values)
            energy = energy + jnp.sum(atom_energy)
            coefficients.append(atom_ddd)
        return energy, tuple(coefficients)


def onecenter_species(paw: PawSpecies, becsum: jnp.ndarray):
    """One atom's one-centre energy and ``ddd``, from its ``becsum``.

    ``becsum`` is the full symmetric ``(nspin, nh, nh)`` matrix. QE carries the
    packed upper triangle with off-diagonals doubled; the two contract
    identically against a tensor symmetric in the same pair, and the full form is
    what the rest of this code already holds.
    """
    nspin = becsum.shape[0]
    energy = jnp.asarray(0.0)
    ddd = jnp.zeros((nspin, paw.nh, paw.nh))

    for tensor, core, sign in (
        (paw.density_ae, paw.core_ae, 1.0),
        (paw.density_ps, paw.core_ps, -1.0),
    ):
        # (nspin, nlm, mesh), holding r^2 rho_lm per channel
        rho_lm = jnp.einsum("sij,ijlr->slr", becsum, tensor)

        v_hartree, e_hartree = _hartree(jnp.sum(rho_lm, axis=0), paw)
        v_xc, e_xc = _exchange_correlation(rho_lm, core, paw)

        # The Hartree potential is the same in both channels: it is a functional
        # of the total on-site density and of nothing else.
        potential = v_hartree[None] + v_xc
        energy = energy + sign * (e_hartree + e_xc)
        # ddd is the derivative of that energy with respect to becsum, and
        # because rho_lm is linear in becsum it is the same tensor contracted
        # against the potential instead of against becsum. QE gets it by
        # rebuilding rho_lm once per (ih, jh) pair with a unit becsum.
        ddd = ddd + sign * jnp.einsum(
            "ijlr,slr->sij", tensor, potential * paw.weights_core[None, None, :]
        )

    return energy, ddd


def _hartree(rho_lm, paw: PawSpecies):
    """One radial Poisson solve per multipole, and the energy that goes with it.

    ``PAW_h_potential``. The ``l`` of each ``lm`` decides both the prefactor and
    the equation solved, so the solves are grouped by ``l`` -- inside a group
    they are the same compiled function under ``vmap``, and there are at most
    ``2 lmax + 1`` groups.

    ``rho_lm`` here is already summed over spin: electrostatics does not
    distinguish the channels, which is why QE sums before calling this and
    copies the single answer back into both.
    """
    blocks = []
    for l in range(int(np.sqrt(paw.nlm - 1)) + 1):
        rows = slice(l * l, min((l + 1) ** 2, paw.nlm))
        source = E2 * FPI / (2 * l + 1) * rho_lm[rows]
        # The 2l+1 components of one multipole solve the *same* equation, so
        # they go through as one batched call; ``l`` is static, which is why the
        # grouping is by l rather than a single vmap over every lm.
        blocks.append(
            jax.vmap(
                lambda f, l=l: radial_hartree(
                    f, paw.r, paw.r2, paw.sqr, paw.dx, l, 2 * l + 2
                )
            )(source)
        )
    potential = jnp.concatenate(blocks)  # (nlm, mesh)
    energy = 0.5 * jnp.sum(potential * rho_lm * paw.weights_full)
    return potential, energy


def _exchange_correlation(rho_lm, core, paw: PawSpecies):
    """``PAW_xc_potential``: onto the sphere, evaluate, and project back.

    The two asymmetries here are QE's and are the same ones the plane-wave
    ``v_xc`` has: the functional sees the **total** density, valence plus core,
    while the potential that comes back out is integrated against the valence
    density alone downstream.
    """
    nspin = rho_lm.shape[0]
    # ... onto the angular grid. rho_lm holds r^2 rho, so dividing by r^2 gives
    # the density the functional wants; the core charge is tabulated directly
    # and, being unpolarized, is shared equally between the channels -- which is
    # what ``arho(:,1) = rho_rad(sum) + rho_core`` says with the magnetization
    # left alone.
    rho_rad = jnp.einsum(
        "xl,slr->sxr", paw.angular.ylm[:, : paw.nlm], rho_lm
    )  # (nspin, nx, mesh)
    density = rho_rad / paw.r2 + core / nspin

    if nspin == 1:
        potential_rad = paw.functional.potential(density[0])[None]
        energy_density = paw.functional.energy_density(density[0])
    else:
        potential_rad = paw.functional.spin_potential(density)
        energy_density = paw.functional.spin_energy_density(density)

    # ... the energy integrates e_xc against the total r^2 rho, direction by
    # direction, with the quadrature weights folded in.
    integrand = energy_density * (jnp.sum(rho_rad, axis=0) + core * paw.r2)
    energy = jnp.sum(
        paw.angular.weights[:, None] * integrand * paw.weights_full[None, :]
    )

    # ... and back onto the multipoles.
    potential = jnp.einsum(
        "xl,sxr->slr", paw.angular.weighted_ylm[:, : paw.nlm], potential_rad
    )

    # A gradient-corrected functional adds a second pass over the same sphere,
    # this time needing the density's gradient there (``PAW_gcxc_potential``).
    if paw.functional.is_gradient:
        v_gradient, e_gradient = onecenter_gradient_correction(rho_lm, rho_rad, core, paw)
        potential = potential + v_gradient
        energy = energy + e_gradient

    return potential, energy


def build_paw(pseudos, structure, functional: Functional, cell=None) -> PawCorrections | None:
    """Precompute the one-centre tensors. ``None`` if no species is PAW."""
    if not any(p.is_paw for p in pseudos):
        return None

    types = np.asarray(structure.types)
    species = []
    for pseudo in pseudos:
        species.append(_build_species(pseudo, functional) if pseudo.is_paw else None)

    return PawCorrections(
        species=tuple(species),
        species_atoms=tuple(
            tuple(int(a) for a in np.flatnonzero(types == t))
            for t in range(structure.ntyp)
        ),
    )


def _build_species(pseudo: Pseudopotential, functional: Functional) -> PawSpecies:
    paw = pseudo.paw
    augmentation = pseudo.augmentation
    if paw is None or augmentation is None or augmentation.qfuncl is None:
        raise ValueError(f"{pseudo.element}: incomplete PAW data in the UPF file")
    # ``PP_AUGMENTATION``'s ``shape`` attribute ('PSQ', 'GAUSS', 'BESSEL', ...)
    # records how the *generator* pseudized Q, and this used to be refused for
    # anything but 'PSQ' on the assumption that the other shapes needed a
    # reconstruction. They do not: grepping the whole of QE, ``upf%paw%augshape``
    # is read, broadcast, printed by ``summary.f90`` and written back out, and
    # never used in a calculation. What PW consumes is the tabulated
    # ``PP_QIJL``, which every shape supplies, so the refusal only kept out
    # perfectly readable datasets -- ``O.pz-kjpaw.UPF`` among them.

    channels = projector_channels(pseudo)
    nh = len(channels)
    lmax_rho = int(pseudo.header.get("l_max_rho", 2 * pseudo.lmax) or 2 * pseudo.lmax)
    nlm = (lmax_rho + 1) ** 2
    ap = harmonic_products(pseudo.lmax)
    if ap.shape[0] < nlm:
        raise ValueError(
            f"{pseudo.element}: l_max_rho={lmax_rho} exceeds what the projectors couple to"
        )

    mesh = pseudo.mesh
    iraug = min(paw.cutoff_index or mesh, mesh)

    # pfunc / ptfunc, as read_upf_new builds them: products of the tabulated
    # r*phi, cut off at the augmentation radius. The cut matters -- the
    # all-electron partial waves do not decay, they oscillate out to the edge of
    # the mesh, and only inside the sphere is their difference meaningful.
    ae = np.asarray(paw.ae_wfc)
    ps = np.asarray(paw.ps_wfc)
    pfunc = np.einsum("nr,mr->nmr", ae, ae)
    ptfunc = np.einsum("nr,mr->nmr", ps, ps)
    if paw.ae_wfc_rel is not None:
        # A fully-relativistic dataset solves the Dirac equation, so its
        # all-electron partial waves have a *small* component as well as a large
        # one, and both carry charge. ``read_upf_new`` adds the small component's
        # density into ``pfunc`` inside the augmentation sphere -- for every
        # calculation, not only a magnetic one; ``pfunc_rel`` is kept separately
        # only for the magnetization term. Leaving it out is worth about 1e-3 Ry
        # on platinum: small enough to look like a convergence difference, large
        # enough to be wrong.
        small = np.asarray(paw.ae_wfc_rel)
        pfunc[:, :, :iraug] += np.einsum("nr,mr->nmr", small, small)[:, :, :iraug]
    pfunc[:, :, iraug:] = 0.0
    ptfunc[:, :, iraug:] = 0.0

    beta_of = np.array([nb for nb, _, _ in channels])
    lm_of = np.array([lm for _, _, lm in channels])
    coefficients = ap[:nlm, lm_of[:, None], lm_of[None, :]]  # (nlm, nh, nh)

    density_ae = np.einsum("lij,ijr->ijlr", coefficients, pfunc[beta_of][:, beta_of])
    pseudo_density = ptfunc[beta_of][:, beta_of].copy()
    density_ps = np.einsum("lij,ijr->ijlr", coefficients, pseudo_density)

    # ... plus the augmentation charge, which the pseudo on-site density carries
    # because it is what was added to the grid density in the first place. Its
    # L is fixed by the lm, not free: rho_lm picks up Q^L for L = floor(sqrt(lm)).
    qfuncl = augmentation.qfuncl
    for lm in range(nlm):
        l = int(np.sqrt(lm))
        if l >= qfuncl.shape[2]:
            continue
        density_ps[:, :, lm, :] += (
            coefficients[lm][:, :, None] * qfuncl[beta_of][:, beta_of, l]
        )

    core_ps = pseudo.rho_core
    core_ae = paw.ae_rho_core
    zero = np.zeros(mesh)

    return PawSpecies(
        density_ae=jnp.asarray(density_ae),
        density_ps=jnp.asarray(density_ps),
        core_ae=jnp.asarray(zero if core_ae is None else core_ae[:mesh]),
        core_ps=jnp.asarray(zero if core_ps is None else core_ps[:mesh]),
        r=jnp.asarray(pseudo.r),
        r2=jnp.asarray(pseudo.r**2),
        sqr=jnp.asarray(np.sqrt(pseudo.r)),
        weights_full=simpson_weights(jnp.asarray(pseudo.rab)),
        weights_core=_truncated_weights(pseudo.rab, pseudo.kkbeta, mesh),
        angular=build_angular_grid(lmax_rho, nlm, functional.is_gradient),
        functional=functional,
        dx=float(pseudo.dx),
        nlm=nlm,
        nh=nh,
    )


def _truncated_weights(rab, cutoff: int, mesh: int) -> jnp.ndarray:
    """Simpson weights over ``kkbeta``, padded with zeros to the full mesh.

    ``PAW_potential`` integrates the ``ddd`` contributions only to ``kkbeta``
    while integrating the energies over the whole mesh. Padding rather than
    slicing keeps every array in this module the same length, which is what lets
    the whole thing be one traced shape.
    """
    weights = np.zeros(mesh)
    weights[:cutoff] = np.asarray(simpson_weights(jnp.asarray(rab[:cutoff])))
    return jnp.asarray(weights)
