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
* **Hartree**, one radial Poisson solve per ``lm`` (:mod:`defumat.paw.hartree`);
* **exchange-correlation**, evaluated pointwise on a spherical quadrature
  (:mod:`defumat.paw.angular`) because it does not act multipole by multipole.

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

from defumat.paw.angular import AngularGrid, build_angular_grid
from defumat.paw.gradient import onecenter_gradient_correction, radial_derivative
from defumat.paw.hartree import radial_hartree
from defumat.pseudo.coupling import harmonic_products
from defumat.pseudo.projectors import projector_channels
from defumat.pseudo.radial import simpson_weights
from defumat.pseudo.upf import Pseudopotential
from defumat.units import E2, FPI
from defumat.xc.functional import Functional, local_spin_frame

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
    #: The density tensors' counterparts for the **kinetic energy density**, or
    #: ``None`` when the functional is not a meta-GGA:
    #: ``tau_lm = einsum('ij,ijlr->lr', becsum, t)``, holding ``r^2 tau_lm`` in
    #: **Ry** exactly as the density tensors hold ``r^2 rho_lm``. Built by
    #: :func:`_kinetic_tensor`, and the reason a potential-only meta-GGA can
    #: have PAW at all.
    kinetic_ae: jnp.ndarray | None = None
    kinetic_ps: jnp.ndarray | None = None


class PawCorrections(eqx.Module):
    """The PAW species of a structure, and the atoms each applies to."""

    species: tuple  # PawSpecies or None, one per species
    species_atoms: tuple = eqx.field(static=True)

    def energy_and_coefficients(self, becsum: tuple, meta_c=None, axis=None):
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
                partial(onecenter_species, paw, meta_c=meta_c, axis=axis),
                in_axes=1, out_axes=(0, 1),
            )(values)
            energy = energy + jnp.sum(atom_energy)
            coefficients.append(atom_ddd)
        return energy, tuple(coefficients)


def onecenter_species(paw: PawSpecies, becsum: jnp.ndarray, meta_c=None, axis=None):
    """One atom's one-centre energy and ``ddd``, from its ``becsum``.

    ``becsum`` is the full symmetric ``(nspin, nh, nh)`` matrix. QE carries the
    packed upper triangle with off-diagonals doubled; the two contract
    identically against a tensor symmetric in the same pair, and the full form is
    what the rest of this code already holds.
    """
    nspin = becsum.shape[0]
    energy = jnp.asarray(0.0)
    ddd = jnp.zeros((nspin, paw.nh, paw.nh))

    for tensor, kinetic, core, sign in (
        (paw.density_ae, paw.kinetic_ae, paw.core_ae, 1.0),
        (paw.density_ps, paw.kinetic_ps, paw.core_ps, -1.0),
    ):
        # (nspin, nlm, mesh), holding r^2 rho_lm per channel
        rho_lm = jnp.einsum("sij,ijlr->slr", becsum, tensor)

        v_hartree, e_hartree = _hartree(_charge_channel(rho_lm), paw)
        v_xc, e_xc = _exchange_correlation(rho_lm, core, paw, axis)
        if kinetic is not None:
            # The one-centre Tran-Blaha potential. It adds to ``v_xc`` and
            # **nothing to the energy**: the potential is not the derivative of
            # one, which is the whole character of this functional. That the
            # ``ddd`` contraction below is still right for it is the point of
            # the phase -- see :func:`_meta_exchange_onecenter`.
            v_xc = v_xc + _meta_exchange_onecenter(
                rho_lm, jnp.einsum("sij,ijlr->slr", becsum, kinetic), paw, meta_c
            )

        # The Hartree potential is the same in both channels: it is a functional
        # of the total on-site density and of nothing else. In the
        # ``(n, m_x, m_y, m_z)`` representation "both channels" means the charge
        # component alone -- ``PAW_h_potential`` is called on
        # ``rho_lm(:,:,1:nspin_lsda)``, and ``nspin_lsda`` is 1 there.
        potential = _as_potential(v_hartree, nspin) + v_xc
        energy = energy + sign * (e_hartree + e_xc)
        # ddd is the derivative of that energy with respect to becsum, and
        # because rho_lm is linear in becsum it is the same tensor contracted
        # against the potential instead of against becsum. QE gets it by
        # rebuilding rho_lm once per (ih, jh) pair with a unit becsum.
        ddd = ddd + sign * jnp.einsum(
            "ijlr,slr->sij", tensor, potential * paw.weights_core[None, None, :]
        )

    return energy, ddd


def _charge_channel(rho_lm: jnp.ndarray) -> jnp.ndarray:
    """The charge out of however many spin components ``rho_lm`` carries.

    ``(up, down)`` sums; ``(n, m_x, m_y, m_z)`` is the first component alone.
    :func:`defumat.scf.potential.total_charge`'s rule, on the radial mesh.
    """
    return rho_lm[0] if rho_lm.shape[0] == 4 else jnp.sum(rho_lm, axis=0)


def _as_potential(scalar: jnp.ndarray, nspin: int) -> jnp.ndarray:
    """A spin-independent potential laid out over ``nspin`` components.

    :func:`defumat.scf.potential.as_potential_components`' rule, on the radial
    mesh: every channel of an ``(up, down)`` potential feels it in full, and only
    the charge component of an ``(n, m)`` one does.
    """
    if nspin == 4:
        zero = jnp.zeros_like(scalar)
        return jnp.stack([scalar, zero, zero, zero])
    return jnp.broadcast_to(scalar, (nspin,) + scalar.shape)


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


def _exchange_correlation(rho_lm, core, paw: PawSpecies, axis=None):
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
    # The core charge is unpolarized, and what that means depends on the
    # representation: shared equally between ``(up, down)``, all of it in the
    # charge component of ``(n, m)``. :func:`defumat.scf.potential.with_core`'s
    # rule, and dividing by four instead loses three quarters of it.
    core_weights = (
        jnp.array([1.0, 0.0, 0.0, 0.0]) if nspin == 4
        else jnp.full((nspin,), 1.0 / nspin)
    )
    density = rho_rad / paw.r2 + core_weights[:, None, None] * core

    if nspin == 1:
        potential_rad = paw.functional.potential(density[0])[None]
        energy_density = paw.functional.energy_density(density[0])
    elif nspin == 4:
        # The same local spin frame the plane-wave ``v_xc`` uses, on the sphere:
        # ``PAW_xc_potential`` calls ``xc(..., 4, 2, ...)`` and recombines the
        # two channel potentials into ``v_0`` and a splitting along ``m-hat``.
        channels, _, direction = local_spin_frame(density[0], density[1:])
        potentials = paw.functional.spin_potential(channels)
        v0 = 0.5 * (potentials[0] + potentials[1])
        vs = 0.5 * (potentials[0] - potentials[1])
        potential_rad = jnp.concatenate([v0[None], vs[None] * direction])
        energy_density = paw.functional.spin_energy_density(channels)
    else:
        potential_rad = paw.functional.spin_potential(density)
        energy_density = paw.functional.spin_energy_density(density)

    # ... the energy integrates e_xc against the total r^2 rho, direction by
    # direction, with the quadrature weights folded in.
    integrand = energy_density * (_charge_channel(rho_rad) + core * paw.r2)
    energy = jnp.sum(
        paw.angular.weights[:, None] * integrand * paw.weights_full[None, :]
    )

    # ... and back onto the multipoles.
    potential = jnp.einsum(
        "xl,sxr->slr", paw.angular.weighted_ylm[:, : paw.nlm], potential_rad
    )

    # A gradient-corrected functional adds a second pass over the same sphere,
    # this time needing the density's gradient there (``PAW_gcxc_potential``).
    # Its ``nspin = 4`` branch resolves each direction onto the local spin axis
    # first (``compute_rho_spin_lm``) -- see :func:`_noncollinear_gradient`.
    if paw.functional.is_gradient:
        v_gradient, e_gradient = onecenter_gradient_correction(
            rho_lm, rho_rad, core, paw, axis
        )
        potential = potential + v_gradient
        energy = energy + e_gradient

    return potential, energy




def _radial_laplacian(f_lm, paw: PawSpecies):
    """``lap rho`` on the sphere from its multipoles, ``(nx, mesh)``.

    Args:
        f_lm: ``(nlm, mesh)`` holding ``r^2 rho_lm``, the storage everything
            here uses.

    For a multipole expansion the Laplacian is diagonal in ``lm``:

        lap rho = sum_lm [ rho_lm'' + (2/r) rho_lm' - l(l+1) rho_lm / r^2 ] Y_lm,

    so it costs two radial derivatives per multipole and no angular work at all
    -- the angular part is the ``-l(l+1)/r^2`` and nothing else. **QE has no
    counterpart**: its one-centre XC never needs a Laplacian, because none of
    the functionals it reaches on the sphere is one that asks for it.

    The ``r^2`` has to come off first: differentiating the stored ``r^2 rho_lm``
    and correcting afterwards would need three more terms and give the same
    answer less accurately near the origin.
    """
    r, r2 = paw.r, paw.r2
    safe = jnp.where(r2 > 0.0, r2, 1.0)
    rho_lm = jnp.where(r2 > 0.0, f_lm / safe, 0.0)
    first = radial_derivative(rho_lm, r)
    second = radial_derivative(first, r)
    l_of = jnp.asarray(np.array([int(np.sqrt(lm)) for lm in range(paw.nlm)]), dtype=r.dtype)
    centrifugal = (l_of * (l_of + 1.0))[:, None] * jnp.where(r2 > 0.0, rho_lm / safe, 0.0)
    radial = second + 2.0 * jnp.where(r > 0.0, first / jnp.where(r > 0.0, r, 1.0), 0.0)
    return jnp.einsum("xl,lr->xr", paw.angular.ylm[:, : paw.nlm], radial - centrifugal)


def _meta_exchange_onecenter(rho_lm, tau_lm, paw: PawSpecies, meta_c=None):
    """The Tran-Blaha potential on one PAW sphere, projected back to multipoles.

    Args:
        rho_lm: ``(nspin, nlm, mesh)`` holding ``r^2 rho_lm``.
        tau_lm: ``(nspin, nlm, mesh)`` holding ``r^2 tau_lm``, in Ry.

    Returns ``(nspin, nlm, mesh)`` -- a potential and **no energy**, which is
    what makes this branch different from every other one-centre term here.

    **Why the ordinary ``ddd`` contraction is still the right thing to do with
    it, even though there is no energy to differentiate.** For a local
    functional ``ddd = d E / d becsum`` and, because ``rho_lm`` is linear in
    ``becsum``, that derivative is numerically the same contraction as the
    matrix element ``<phi_i| v |phi_j>``. Here only the second reading survives:
    the mBJ potential is *multiplicative*, so what the Hamiltonian must receive
    is exactly ``<phi_i| v |phi_j> - <phi~_i| v~ |phi~_j>``, and that is the
    contraction ``onecenter_species`` already performs. Nothing had to be added
    for it -- which is the reason PAW is reachable for this functional at all,
    and the reason the *energy* returned here is identically zero rather than
    small.

    **The frozen core is left out of this term, and it has to be.** Every other
    one-centre functional here sees ``rho_valence + rho_core``; this one sees the
    valence density alone, on both the all-electron and the pseudo side. The
    reason is that a UPF PAW dataset carries a core *charge* (``PP_AE_NLCC``)
    and **no core kinetic energy density** -- nothing in the format has one, and
    QE never needs one because it reaches no functional that asks. Feeding the
    all-electron core into ``rho`` with no matching ``tau`` is not a small
    inconsistency inside a sphere: the core dominates ``rho`` and ``lap rho``
    near the nucleus while contributing nothing to ``tau``, so ``2 tau -
    |grad rho|^2/4 rho`` and ``sqrt(2 tau / rho)`` are both evaluated on
    mismatched halves.

    It is measurable, and it inverts the functional. With the core included, the
    silicon gap *falls* monotonically as ``c`` rises -- 1.249, 0.934, 0.348 eV at
    ``c = 1.0, 1.1, 1.28`` -- where every norm-conserving cell has it rise. With
    the core left out it rises: 0.917, 1.260, 1.991 eV. The sign of
    ``d(gap)/dc`` is the diagnostic, and it is what says which of the two is the
    functional and which is an artefact.

    (VASP's meta-GGA PAW datasets carry the core kinetic energy density as a
    separate tabulated field for exactly this reason. Reading one is what it
    would take to include the core here; no UPF has it to read.)
    """
    from defumat.paw.gradient import _gradient

    nspin = rho_lm.shape[0]
    r2 = paw.r2
    rho_rad = jnp.einsum("xl,slr->sxr", paw.angular.ylm[:, : paw.nlm], rho_lm)
    density = rho_rad / r2
    tau_rad = jnp.einsum("xl,slr->sxr", paw.angular.ylm[:, : paw.nlm], tau_lm) / r2

    # ``c`` is a cell average and belongs to the *calculation*, not to one
    # sphere: recomputing it from the on-site density would give each atom a
    # different functional. The plane-wave value is used, and for a fixed ``c``
    # (``mbj_c``, or BJ06) it is simply that constant.
    c = paw.functional.meta_coefficient if meta_c is None else meta_c
    if c is None:
        raise ValueError(
            "the Tran-Blaha c is an average over the whole unit cell, so a PAW "
            "sphere cannot compute its own: it has to be passed down from the "
            "plane-wave part (Calculation.onecenter(meta_c=...))"
        )

    def channel_potential(channels, taus):
        """``v_x`` for a set of channel densities already on the sphere.

        The multipoles of each channel are projected back out of its *grid*
        values rather than assembled from ``rho_lm``. That is not tidiness: for
        ``nspin = 4`` the channels come from a rotation into the local spin
        frame, which involves ``|m|`` and is not linear in the components, so
        their combination is not the expansion of the result. It is the trap
        ``gradcorr`` documents on the plane-wave side, and the Laplacian
        inherits it -- so the projection is done for every regime, since it
        costs one quadrature and removes the special case.
        """
        weighted = paw.angular.weighted_ylm[:, : paw.nlm]
        channel_lm = jnp.einsum("xl,sxr->slr", weighted, channels * r2)
        gradients = jax.vmap(_gradient, in_axes=(0, 0, None))(
            channel_lm, channels, paw
        )
        sigma = jnp.sum(gradients * gradients, axis=1)
        laplacian = jax.vmap(_radial_laplacian, in_axes=(0, None))(channel_lm, paw)
        return jax.vmap(paw.functional.meta_exchange_potential,
                        in_axes=(0, 0, 0, 0, None))(
            channels, sigma, laplacian, taus / E2, c
        )

    if nspin == 1:
        # One stored channel is the *total*, so the functional's argument is
        # half of it -- the same halving :func:`meta_exchange` does.
        potential_rad = channel_potential(0.5 * density, 0.5 * tau_rad)
    elif nspin == 4:
        channels, _, direction = local_spin_frame(density[0], density[1:])
        projected = jnp.sum(tau_rad[1:] * direction, axis=0)
        taus = 0.5 * jnp.stack([tau_rad[0] + projected, tau_rad[0] - projected])
        potentials = channel_potential(channels, taus)
        v0 = 0.5 * (potentials[0] + potentials[1])
        vs = 0.5 * (potentials[0] - potentials[1])
        potential_rad = jnp.concatenate([v0[None], vs[None] * direction])
    else:
        potential_rad = channel_potential(density, tau_rad)

    return jnp.einsum(
        "xl,sxr->slr", paw.angular.weighted_ylm[:, : paw.nlm], potential_rad
    )


def _kinetic_tensor(waves, coefficients, angular, lm_of, r, iraug, nlm):
    """The ``becsum -> r^2 tau_lm`` tensor for one set of partial waves.

    Args:
        waves: ``(nh, mesh)`` the tabulated ``u_i(r) = r phi_i(r)``, already
            selected per projector channel.
        coefficients: ``(nlm, nh, nh)`` the Clebsch-Gordan expansion of
            ``Y_i Y_j``, which the density tensor uses too.
        lm_of: ``(nh,)`` which harmonic each channel carries.

    The kinetic energy density of a set of partial waves is

        tau(r) = sum_ij becsum_ij  grad phi_i . grad phi_j,

    and in spherical coordinates that gradient has a radial part and an angular
    one which do **not** share an angular structure:

        grad phi_i . grad phi_j
            = R'_i R'_j  Y_i Y_j
            + (R_i R_j / r^2) (grad_Omega Y_i . grad_Omega Y_j),

    with ``R_i = u_i / r``. The first term expands on exactly the same
    multipoles the *density* does -- it is ``Y_i Y_j`` again -- so it reuses
    ``coefficients``. The second does not, and its expansion is computed here by
    quadrature on the angular grid:

        Bcoef[lm, i, j] = sum_x w_x Y_lm(x) (grad_Omega Y_i . grad_Omega Y_j)_x.

    **Which of the two angular derivative tables carries the ``1/sin(theta)``
    is not something to infer from a variable name.** The versor components are
    ``dylmt`` and ``dylmp`` and the modulus is their plain sum of squares --
    established by the exact identity ``int |grad_Omega Y_lm|^2 dOmega =
    l(l+1)``, which ``dylmt^2 + dylmp^2`` reproduces to every digit on two grid
    sizes and ``dylmt^2 + dylmp^2/sin^2`` does not (4.58 against 2 for
    ``Y_1,+-1``). :func:`~defumat.paw.gradient.onecenter_gradient_correction`
    forms ``sigma`` the same way, and its lone ``divide(sin_theta)`` belongs to
    the *divergence*'s input convention and not to the modulus.

    Returned in ``r^2 tau_lm`` form, in Ry: the radial term carries the ``r^2``
    explicitly and the angular one gets it from ``u_i u_j = r^2 R_i R_j``
    already. No factor of one half, matching
    :func:`defumat.scf.density.band_kinetic_density` and QE's ``rho%kin_r``.

    **The relativistic small component is in the density and not here.**
    ``_build_species`` adds ``pfunc_rel`` into the all-electron *charge* for a
    fully-relativistic dataset, because ``read_upf_new`` does; there is no
    counterpart for ``tau``, and the large component's ``u_i`` alone is what
    this builds from. It is the same family of mismatch as the missing core
    kinetic energy density (:func:`_meta_exchange_onecenter`), and far smaller:
    the small component carries order ``(Z alpha)^2`` of the valence weight
    where the core carries all of it. Recorded rather than corrected.
    """
    import numpy as _np

    safe_r = _np.where(r > 0.0, r, 1.0)
    radial = _np.asarray(waves) / safe_r                      # R_i(r)
    radial = _np.where(r > 0.0, radial, 0.0)
    derivative = _np.asarray(
        radial_derivative(jnp.asarray(radial), jnp.asarray(r))
    )                                                          # dR_i/dr

    # (nh, nh, mesh): the two radial products, each already carrying its r^2.
    radial_product = _np.einsum("ir,jr->ijr", derivative, derivative) * (r**2)
    angular_product = _np.einsum("ir,jr->ijr", radial, radial)

    dylmt = _np.asarray(angular.dylmt)
    dylmp = _np.asarray(angular.dylmp)
    overlap = (
        _np.einsum("xi,xj->ijx", dylmt[:, lm_of], dylmt[:, lm_of])
        + _np.einsum("xi,xj->ijx", dylmp[:, lm_of], dylmp[:, lm_of])
    )
    bcoef = _np.einsum("xl,ijx->lij", _np.asarray(angular.weighted_ylm)[:, :nlm], overlap)

    tensor = (
        _np.einsum("lij,ijr->ijlr", coefficients, radial_product)
        + _np.einsum("lij,ijr->ijlr", bcoef, angular_product)
    )
    tensor[:, :, :, iraug:] = 0.0
    return jnp.asarray(tensor)


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
        # A meta-GGA needs the harmonics' angular derivatives for the same
        # reason a GGA does -- ``tau``'s angular part is built from them.
        angular=(angular_grid := build_angular_grid(
            lmax_rho, nlm, functional.is_gradient or functional.is_meta
        )),
        functional=functional,
        dx=float(pseudo.dx),
        nlm=nlm,
        nh=nh,
        kinetic_ae=None if not functional.is_meta else _kinetic_tensor(
            ae[beta_of], coefficients, angular_grid, lm_of, pseudo.r, iraug, nlm
        ),
        kinetic_ps=None if not functional.is_meta else _kinetic_tensor(
            ps[beta_of], coefficients, angular_grid, lm_of, pseudo.r, iraug, nlm
        ),
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
