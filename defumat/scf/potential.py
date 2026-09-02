"""Building the self-consistent potential from the density.

``v_of_rho``: given the electron density, produce the Hartree and
exchange-correlation potentials and their energies. Following
``PW/src/v_of_rho.f90``, with the gradient correction of ``gradcorr.f90``.

The Hartree term is diagonal in G space -- ``V_H(G) = 4 pi e^2 rho(G) / G^2`` --
with the ``G = 0`` component set to zero. That divergence is not an error: it
cancels against the corresponding divergences in the Ewald sum and in the local
pseudopotential, and the three ``G = 0`` terms are only finite together, for a
neutral cell.

**The spin axis.** Densities and potentials here are ``(nspin, ...)`` and each
channel is that spin's own density -- so ``nspin = 1`` is the total density in a
single channel and needs no special case anywhere. QE instead stores the pair as
(total, magnetization) and converts back and forth (``rhoz_or_updw``); the two
conventions meet where its formulas are written in one or the other, which is
exactly three places: the Hartree term and ``dr2`` want the total, and the
exchange-correlation functional wants the total and ``zeta``.

The physics of the split is the whole of LSDA: **Hartree is a functional of the
total density alone** -- an electron does not care about the spin of the charge
repelling it -- while exchange-correlation is not, and that asymmetry is why the
two channels see different potentials at all.

**Noncollinear.** With four components the convention changes again, and this
time it is QE's throughout: ``(n, m_x, m_y, m_z)``, the charge and the
magnetization *vector*, because there is no "up" and "down" axis to project on.
The exchange-correlation functional is still the collinear one -- LSDA evaluated
along the local magnetization direction, which is what makes the potential a 2x2
matrix ``v_0 I + v_s m-hat . sigma`` rather than a new functional. A run whose
starting magnetization is zero has ``nspin_mag = 1``: one component, and every
routine here takes the unpolarized path it always did.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from defumat.basis.fft import g_to_r, r_to_g
from defumat.basis.gradients import divergence, gradient, laplacian
from defumat.basis.gvectors import GVectors
from defumat.system.cell import Cell
from defumat.units import E2, FPI
from defumat.xc.functional import Functional, get_functional, local_spin_frame

__all__ = ["Potential", "v_of_rho", "hartree", "exchange_correlation",
           "gradient_correction", "meta_exchange", "scf_accuracy", "total_charge",
           "with_core", "as_potential_components",
           "DEFAULT_FUNCTIONAL"]

#: What a calculation uses when nothing names a functional. QE has no such
#: default -- it takes the functional from the pseudopotentials and stops if
#: they disagree -- so this is only for callers that construct a potential
#: directly, and every path through the SCF driver passes one explicitly.
DEFAULT_FUNCTIONAL = "PZ"


class Potential(eqx.Module):
    """The self-consistent potential and the energies that go with it.

    ``v_scf`` excludes the local pseudopotential: that part is fixed, while this
    is what changes from iteration to iteration and what ``deband`` subtracts.
    """

    v_scf: jnp.ndarray  # (nspin, n1, n2, n3), Ry -- Hartree + XC
    ehart: jnp.ndarray  # Ry
    etxc: jnp.ndarray  # Ry
    #: ``-int B . m`` for a field put in by hand, and ``etcon`` for a
    #: constrained moment. Both are *inside* ``v_scf`` -- ``add_bfield`` is
    #: called from within ``v_of_rho`` -- and neither is added to the total
    #: energy, which is QE's convention and Elk's; see
    #: :mod:`defumat.scf.fields`.
    e_field: jnp.ndarray = 0.0
    e_constraint: jnp.ndarray = 0.0
    #: The Tran-Blaha coefficient this potential was built with, or ``0`` when
    #: the functional is not a meta one. Carried on the potential rather than
    #: recomputed by whoever wants to print it, because it is a *cell average*
    #: of the density: two call sites recomputing it from two densities is how
    #: a reported ``c`` stops describing the run it is reported for.
    meta_c: jnp.ndarray = 0.0

    @property
    def nspin(self) -> int:
        return self.v_scf.shape[0]


def total_charge(rho_g: jnp.ndarray) -> jnp.ndarray:
    """The charge density out of however many components are stored.

    The sum of the channels when they are ``(up, down)``, and the first
    component alone when they are ``(n, m_x, m_y, m_z)`` -- adding those would
    put the magnetization into the Hartree term, which is a mistake that leaves
    the run self-consistent and wrong rather than failing.
    """
    return rho_g[0] if rho_g.shape[0] == 4 else jnp.sum(rho_g, axis=0)


def hartree(rho_g: jnp.ndarray, gvectors: GVectors, cell: Cell):
    """Hartree potential in G space and its energy.

    Returns ``(v_hartree(G), E_hartree)``.
    """
    g2 = gvectors.kinetic(cell)  # |G|^2 in 1/bohr^2
    inverse = jnp.where(g2 > 1e-12, 1.0 / jnp.where(g2 > 1e-12, g2, 1.0), 0.0)

    v = E2 * FPI * rho_g * inverse
    energy = 0.5 * cell.volume * E2 * FPI * jnp.sum(jnp.abs(rho_g) ** 2 * inverse)
    if gvectors.gamma_only:
        # Only half the sphere is stored; every G != 0 stands for a +-G pair.
        energy = 2.0 * energy
    return v, energy


def scf_accuracy(residual_r: jnp.ndarray, gvectors: GVectors, cell: Cell) -> jnp.ndarray:
    """QE's ``dr2``: how far from self-consistency the density still is, in Ry.

    ``rho_ddot`` in ``PW/src/scf_mod.f90``, evaluated on the residual with
    itself. It is the *Hartree energy of the density error* -- literally the same
    expression as :func:`hartree`'s energy, applied to ``rho_out - rho_in`` --
    which is why it is an estimate of the error in the total energy rather than
    just a size of the density change: it weights long-wavelength errors by
    ``1/G^2``, and those are the ones that cost energy.

    This is the quantity QE compares against ``conv_thr`` and the quantity its
    diagonalisation threshold is scheduled from, so both of those now mean the
    same thing here as they do there.

    With two spin channels ``rho_ddot`` gains a second piece, and it is not a
    second Hartree energy: the magnetization enters with a **G-independent**
    weight ``e2 4 pi / (2 pi)^2`` (QE's comment says ``lambda = 1 a.u.``) and
    with its ``G = 0`` component *included*, where the Hartree half excludes it.
    An error in the total charge is expensive in proportion to its wavelength; an
    error in the magnetization is expensive at every wavelength equally, and a
    uniform shift of the magnetization is a real error where a uniform shift of
    the charge is forbidden by neutrality.
    """
    residual_g = r_to_g(residual_r, gvectors.fft_index)
    total = hartree(total_charge(residual_g), gvectors, cell)[1]
    if residual_g.shape[0] == 1:
        return total

    magnetization = (
        residual_g[1:] if residual_g.shape[0] == 4
        else (residual_g[0] - residual_g[1])[None]
    )
    weight = E2 * FPI / (2.0 * jnp.pi) ** 2
    contribution = jnp.sum(jnp.abs(magnetization) ** 2)
    if gvectors.gamma_only:
        # Only half the sphere is stored, and unlike the Hartree half the G = 0
        # term is counted here -- so the doubling applies to the rest of it.
        contribution = 2.0 * contribution - jnp.sum(jnp.abs(magnetization[:, 0]) ** 2)
    return total + 0.5 * cell.volume * weight * contribution


def exchange_correlation(
    rho_r: jnp.ndarray,
    cell: Cell,
    rho_core: jnp.ndarray | None = None,
    functional: Functional | None = None,
):
    """The local part of the XC potential on the grid, and its energy.

    ``rho_core`` is the nonlinear core correction (``PP_NLCC``): the frozen core
    charge a pseudopotential leaves out of the valence density but which the
    exchange-correlation functional is nonlinear in, so that evaluating ``e_xc``
    at the valence density alone is wrong wherever the core overlaps the
    valence. QE's ``v_xc`` handles it in the way transcribed here, and the two
    halves are deliberately asymmetric (``PW/src/v_of_rho.f90``):

    * the functional -- both ``e_xc`` and ``v_xc`` -- is evaluated at the
      **total** density ``rho + rho_core``;
    * the energy integrates ``(rho + rho_core) e_xc`` over the total density,
      but ``vtxc`` and hence ``deband`` integrate ``v_xc`` against the
      **valence** density only, because it is only the valence density that the
      SCF varies.

    Getting the second point backwards leaves the total energy self-consistent
    and wrong, which is the same failure mode as the ``vloc`` ``G = 0`` trap.

    With a gradient-corrected functional this is only half the story: QE calls
    ``v_xc`` for the local part and then ``gradcorr`` for the rest, and so does
    :func:`v_of_rho`.
    """
    functional = functional or get_functional(DEFAULT_FUNCTIONAL)
    valence = jnp.real(rho_r)
    nspin = valence.shape[0]
    density = valence if rho_core is None else valence + with_core(jnp.real(rho_core), nspin)
    n = density[0].size

    # ``v_xc`` and ``e_xc`` come out of one evaluation of the functional rather
    # than two: the potential is the derivative of ``rho e_xc`` and its forward
    # value carries ``e_xc`` along with it. QE gets both from one call to
    # ``xc_lda`` for the same reason, having written the derivative down by
    # hand.
    if nspin == 1:
        potential, energy_density = functional.potential_and_energy_density(density[0])
        v = potential[None]
        energy = cell.volume / n * jnp.sum(density[0] * energy_density)
        return v, energy

    if nspin == 4:
        return _noncollinear_xc(density, cell, functional)

    v, energy_density = functional.spin_potential_and_energy_density(density)
    energy = cell.volume / n * jnp.sum(jnp.sum(density, axis=0) * energy_density)
    return v, energy


#: ...and below this charge there is nothing to build a potential from at all.
VANISHING_CHARGE = 1.0e-10


def _noncollinear_xc(density: jnp.ndarray, cell: Cell, functional: Functional):
    """``v_xc``'s ``nspin == 4`` branch: LSDA along the local spin axis.

    There is no new functional here, and that is the physical content of the
    local spin-density approximation applied to a noncollinear state: at each
    point the magnetization picks out an axis, the density is resolved into the
    two spin projections on *that* axis,

        rho_up = (n + |m|) / 2,     rho_down = (n - |m|) / 2,

    and the ordinary spin-polarized functional is evaluated there. What comes
    back is a scalar potential and a splitting,

        v_0 = (v_up + v_down) / 2,   v_s = (v_up - v_down) / 2,

    and the splitting is attached to the *direction* of the magnetization, so
    the potential the spinor Hamiltonian sees is ``v_0 I + v_s m-hat . sigma``.
    That is why the vector part of ``v`` is parallel to ``m`` by construction:
    a functional of ``|m|`` alone cannot produce a torque, which is exactly the
    known limitation of this approximation rather than an artefact of it.

    ``jax.grad`` is not used for the axis: the functional's own
    ``spin_potential`` already gives ``v_up`` and ``v_down``, and the projection
    onto ``m-hat`` is algebra on top of it.
    """
    n = density[0].size
    charge = density[0]
    absolute = jnp.abs(charge)

    # The local spin frame is shared with PAW's one-centre XC, which does the
    # same thing on the radial sphere -- see :func:`defumat.xc.functional.local_spin_frame`.
    channels, _, direction = local_spin_frame(charge, density[1:])

    potentials = functional.spin_potential(channels)
    v0 = 0.5 * (potentials[0] + potentials[1])
    vs = 0.5 * (potentials[0] - potentials[1])

    v = jnp.concatenate([v0[None], vs[None] * direction])
    # Nothing to polarise where there is no charge; QE zeroes all four
    # components there rather than dividing.
    v = jnp.where(absolute >= VANISHING_CHARGE, v, 0.0)

    energy = (
        cell.volume / n
        * jnp.sum(absolute * functional.spin_energy_density(channels))
    )
    return v, energy


def fixed_quantization_axis(moments: np.ndarray) -> np.ndarray | None:
    """``compute_ux``: a fixed axis to take the sign of the magnetization along.

    A gradient-corrected noncollinear run resolves the density onto the local
    spin axis before evaluating the functional, and the naive resolution
    ``(n +- |m|)/2`` has a **kink** wherever ``m`` passes through zero -- an
    antiferromagnet has one on every plane between two atoms, and a kink in the
    density is a divergence in its gradient. QE avoids it whenever the starting
    moments are all parallel to one direction: it then keeps the *signed*
    projection on that direction, so up stays up across the node.

    Returns the unit axis, or ``None`` when the starting moments are not all
    parallel (QE's ``lsign = .FALSE.``), in which case ``|m|`` is used and the
    kink is accepted -- there is no single axis to take a sign along.
    """
    moments = np.asarray(moments, dtype=float)
    norms = np.linalg.norm(moments, axis=1)
    nonzero = np.flatnonzero(norms > 1.0e-12)
    if not len(nonzero):
        return None
    axis = moments[nonzero[0]] / norms[nonzero[0]]
    for index in nonzero[1:]:
        direction = moments[index] / norms[index]
        if np.linalg.norm(np.cross(axis, direction)) > 1.0e-6:
            return None
    return axis


def _noncollinear_gradient_correction(
    rho_r: jnp.ndarray,
    gvectors: GVectors,
    cell: Cell,
    functional: Functional,
    rho_core: jnp.ndarray | None,
    axis: jnp.ndarray | None,
):
    """``gradcorr``'s ``nspin == 4 .AND. domag`` branch.

    Three steps, and the middle one is the ordinary collinear code:

    1. **Rotate.** ``compute_rho`` resolves ``(n, m)`` onto the local spin axis,
       ``rho_up/dw = (n +- s |m|) / 2`` with ``s = sign(m . ux)`` when there is a
       fixed axis (:func:`fixed_quantization_axis`) and ``s = 1`` when there is not.
       The core charge is added *after* the rotation, half to each channel, as
       an unpolarized density is in the ``(up, down)`` representation.
    2. **Evaluate**, with :func:`gradient_correction` and ``nspin = 2``. The
       rotated density's transform is taken afresh rather than assembled from
       ``rho(G)``: the rotation involves ``|m|``, which is not linear in the
       components, so their combination is not the transform of the result.
    3. **Rotate back.** The charge component gets ``(v_up + v_dw)/2`` and the
       vector part gets ``s (v_up - v_dw)/2`` along ``m-hat`` -- the same
       attachment to the direction of ``m`` the LDA part makes, with the sign
       carried through.
    """
    charge = rho_r[0]
    magnetization = rho_r[1:]
    modulus = jnp.sqrt(jnp.sum(magnetization**2, axis=0))
    if axis is None:
        sign = jnp.ones_like(modulus)
    else:
        projection = jnp.tensordot(jnp.asarray(axis), magnetization, axes=(0, 0))
        sign = jnp.where(projection >= 0.0, 1.0, -1.0)

    signed = sign * modulus
    rotated = 0.5 * jnp.stack([charge + signed, charge - signed])
    if rho_core is not None:
        rotated = rotated + with_core(jnp.real(rho_core), 2)
    rotated_g = jax.vmap(r_to_g, in_axes=(0, None))(rotated, gvectors.fft_index)

    v, energy = gradient_correction(rotated, rotated_g, gvectors, cell, functional)

    v0 = 0.5 * (v[0] + v[1])
    vs = 0.5 * (v[0] - v[1])
    direction = jnp.where(
        modulus > VANISHING_GRADIENT_MAGNETIZATION,
        magnetization / jnp.where(modulus > 0.0, modulus, 1.0),
        0.0,
    )
    return jnp.concatenate([v0[None], (sign * vs)[None] * direction]), energy


#: ``gradcorr`` leaves the vector part of its potential alone below this
#: magnetization -- a looser threshold than the LDA part's ``vanishing_mag``,
#: and QE's.
VANISHING_GRADIENT_MAGNETIZATION = 1.0e-12


def as_potential_components(scalar: jnp.ndarray, nspin: int) -> jnp.ndarray:
    """A spin-independent *potential* laid out like an ``nspin``-component one.

    ``set_vrs``: the local pseudopotential is added to **every** channel of an
    ``(up, down)`` potential -- both spins feel all of it -- and to the **first
    component only** of an ``(n, m_x, m_y, m_z)`` one, where the other three are
    a magnetic field and a spin-independent potential contributes nothing to
    them.

    Not the same rule as :func:`with_core`, and the difference is not cosmetic.
    A *density* that is unpolarized splits equally between the two channels; a
    *potential* that is spin-independent is felt in full by both. Sharing the
    potential would run the whole calculation at half the local
    pseudopotential -- which converges, and is wrong by tens of eV.
    """
    if nspin == 4:
        zero = jnp.zeros_like(scalar)
        return jnp.stack([scalar, zero, zero, zero])
    return jnp.broadcast_to(scalar, (nspin,) + scalar.shape)


def with_core(rho_core: jnp.ndarray, nspin: int) -> jnp.ndarray:
    """The core charge laid out like an ``nspin``-component density.

    The core charge is unpolarized, and what that means for a *density* depends
    on which representation the components are in. In ``(up, down)`` it is
    shared equally, half to each; in ``(n, m_x, m_y, m_z)`` it is all charge and
    no magnetization, so it goes entirely into the first component. Those are
    the same statement, and dividing by four in the second case -- which is what
    a single ``/ nspin`` would do -- silently loses three quarters of the core
    charge from the exchange-correlation energy.

    See :func:`as_potential_components` for the rule a *potential* follows,
    which is deliberately different.
    """
    if nspin == 4:
        zero = jnp.zeros_like(rho_core)
        return jnp.stack([rho_core, zero, zero, zero])
    return jnp.broadcast_to(rho_core / nspin, (nspin,) + rho_core.shape)


def gradient_correction(
    density_r: jnp.ndarray,
    density_g: jnp.ndarray,
    gvectors: GVectors,
    cell: Cell,
    functional: Functional,
):
    """``gradcorr``: what a GGA adds to the potential and to the energy.

    The functional derivative of an energy that depends on ``grad rho`` has two
    terms,

        v = d e / d rho  -  div ( d e / d grad rho )
          = v1           -  div ( v2 grad rho ),

    and the second is why a gradient-corrected potential costs four more
    transforms per iteration than a local one: three to build ``grad rho`` and
    one to take the divergence back. QE assembles exactly this, storing the
    vector field ``h = v2 grad rho`` and calling ``fft_graddot`` on it.

    With two channels there is a third term. Correlation depends on the
    **total** density's gradient, so ``d e / d(grad rho_up)`` picks up
    ``grad rho_dw`` as well; QE calls that cross term ``v2c_ud`` and adds it by
    hand. Here ``h`` is the derivative of the energy with respect to the
    gradient *field* rather than with respect to ``|grad rho|^2``, so the cross
    term is simply part of it and its pairing with the two channels cannot be
    got the wrong way round.

    Args:
        density_r: ``rho + rho_core/nspin`` on the grid, ``(nspin, ...)`` -- the
            density the functional sees, as in :func:`exchange_correlation`.
        density_g: the same density on the G-vector sphere. Passed in rather
            than transformed here because the caller already holds ``rho(G)``
            for the Hartree term, and the core charge's own G components come
            straight from the pseudopotential.

    Returns ``(v, energy)`` in Ry and Ry/bohr^3 respectively -- the energy
    already integrated over the cell, so that it adds to ``etxc``.
    """
    nspin = density_r.shape[0]
    grad = jax.vmap(gradient, in_axes=(0, None, None))(density_g, gvectors, cell)
    n = density_r[0].size

    if nspin == 1:
        sigma = jnp.sum(grad[0] * grad[0], axis=0)
        v1, v2 = functional.gradient_potentials(density_r[0], sigma)
        v = v1 - divergence(v2[None, ...] * grad[0], gvectors, cell)
        energy = (
            cell.volume / n * jnp.sum(functional.gradient_energy(density_r[0], sigma))
        )
        return v[None], energy

    v1, h = functional.spin_gradient_terms(density_r, grad)
    v = v1 - jax.vmap(divergence, in_axes=(0, None, None))(h, gvectors, cell)
    energy = (
        cell.volume / n * jnp.sum(functional.spin_gradient_energy(density_r, grad))
    )
    return v, energy


def meta_exchange(
    density_r: jnp.ndarray,
    density_g: jnp.ndarray,
    tau_r: jnp.ndarray,
    gvectors: GVectors,
    cell: Cell,
    functional: Functional,
):
    """The Tran-Blaha exchange potential on the grid, and the ``c`` it used.

    Args:
        density_r: ``(nspin, ...)`` -- the total density when unpolarized, and
            the ``(up, down)`` pair when not, which is this package's storage in
            both regimes (:func:`with_core`, :func:`exchange_correlation`) -- with
            the core charge already folded in, as :func:`exchange_correlation`
            folds it.
        density_g: the same density on the sphere.
        tau_r: ``(nspin, ...)`` kinetic energy density in **Ry**, per spin
            channel, in the same layout: the whole of ``tau`` when unpolarized
            and ``(tau_up, tau_down)`` when not.

    Returns ``(v, c)``: the potential per channel in Ry, and the coefficient.

    **The unpolarized case still halves.** One stored channel is the *total*
    density, and this functional -- like every exchange functional -- acts on
    one spin channel at a time, so what it is handed is ``rho/2`` and
    ``tau/2``, and the potential that comes back is already the one both
    channels feel. It is the spin-scaling relation
    ``E_x[n_up, n_dw] = (E_x[2 n_up] + E_x[2 n_dw])/2`` in its potential form,
    and it is the same halving :meth:`Functional._spin_energy_density` does for
    exchange.

    **There is no energy to return**, which is the whole character of this
    branch. :func:`gradient_correction` hands back a potential *and* the energy
    it is the derivative of; here the second is absent, ``etxc`` keeps only the
    correlation term, and the run's total energy is not the value of a
    functional the SCF minimised. Everything downstream that differentiates the
    total energy is refused rather than allowed to return a plausible number.

    **The Laplacian is what QE does not have.** ``xc_wrapper_mgga.f90`` passes
    zeros for it to every libxc call, and it enters Becke-Roussel's ``Q``
    directly. Here it is ``-G^2 rho(G)``, one transform per channel.
    """
    nspin = density_r.shape[0]
    tau_r = jnp.asarray(tau_r)
    if nspin > 2:
        raise ValueError(
            "the noncollinear branch is _noncollinear_meta_exchange, which "
            "rotates into the local spin frame before calling this"
        )

    # One channel means the total density, so the functional's argument is half
    # of it; two channels are already the pair it wants.
    scale = 0.5 if nspin == 1 else 1.0
    channels_g = scale * density_g
    channels_r = scale * jnp.real(density_r)
    channels_tau = scale * tau_r
    total_r = jnp.real(density_r[0]) if nspin == 1 else jnp.sum(jnp.real(density_r), axis=0)

    grad = jax.vmap(gradient, in_axes=(0, None, None))(channels_g, gvectors, cell)
    lap = jax.vmap(laplacian, in_axes=(0, None, None))(channels_g, gvectors, cell)
    sigma = jnp.sum(grad * grad, axis=1)
    # The total density's gradient without a fourth transform: the gradient is
    # linear and the channels already carry it, so it is twice one channel's
    # when unpolarized and their sum when not. Three FFTs per potential build.
    total_grad = 2.0 * grad[0] if nspin == 1 else grad[0] + grad[1]

    # ``c`` is an average over the cell of the **total** density's ratio, so it
    # is one number for the whole calculation and not one per channel -- and it
    # is the total, not the majority channel: Tran and Blaha's Eq. (3) has no
    # spin index on it, and giving each channel its own ``c`` would make the two
    # potentials belong to different functionals.
    c = functional.meta_c(total_r, total_grad)

    v = jax.vmap(functional.meta_exchange_potential, in_axes=(0, 0, 0, 0, None))(
        channels_r, sigma, lap, channels_tau / E2, c
    )
    return v, c


def _noncollinear_meta_exchange(
    rho_r, gvectors, cell, functional, rho_core, tau_r, axis,
):
    """``meta_exchange`` for ``nspin_mag = 4``: the local spin frame again.

    The same three steps as :func:`_noncollinear_gradient_correction`, and
    deliberately the same three: rotate onto the local axis, run the *collinear*
    functional there, rotate the answer back. What is new is only that a second
    field rotates with the density.

    1. **Rotate.** The density gives the axis -- ``m-hat`` and the sign against a
       fixed quantization axis, if there is one -- and both fields are resolved
       on it:

           rho_up/dw = (n +- s|m|) / 2,
           tau_up/dw = (tau_0 +- s (tau_vec . m-hat)) / 2.

       **The axis is the density's, not ``tau``'s.** They are not parallel in
       general -- ``tau_vec`` is the Pauli expectation of a *gradient*, and
       nothing makes it collinear with the magnetization -- so the projection
       ``tau_vec . m-hat`` is a genuine projection and its transverse part is
       discarded. That is not an approximation this code invents: it is what
       "evaluate the collinear functional in the local frame" means, and the
       LSDA and GGA branches discard the same transverse information (a
       functional of ``|m|`` alone cannot produce a torque). It is stated here
       because for ``tau`` it is easier to miss.

    2. **Evaluate**, with ``nspin = 2``. The rotated channel densities are
       transformed afresh -- the rotation involves ``|m|`` and is not linear in
       the components, so the gradient *and the Laplacian* have to be taken from
       the rotated field's own transform rather than assembled from ``rho(G)``.
       That trap is ``gradcorr``'s, and the Laplacian inherits it unchanged.

    3. **Rotate back**, attaching the splitting to ``m-hat``:
       ``v = v_0 I + s (v_up - v_dw)/2 m-hat . sigma``.

    Returns ``(v, c)`` with ``v`` of shape ``(4, ...)``.
    """
    charge = rho_r[0]
    magnetization = rho_r[1:]
    modulus = jnp.sqrt(jnp.sum(magnetization**2, axis=0))
    if axis is None:
        sign = jnp.ones_like(modulus)
    else:
        projection = jnp.tensordot(jnp.asarray(axis), magnetization, axes=(0, 0))
        sign = jnp.where(projection >= 0.0, 1.0, -1.0)
    safe = jnp.where(modulus > 0.0, modulus, 1.0)
    direction = jnp.where(
        modulus > VANISHING_GRADIENT_MAGNETIZATION, magnetization / safe, 0.0
    )

    signed = sign * modulus
    rotated = 0.5 * jnp.stack([charge + signed, charge - signed])
    if rho_core is not None:
        rotated = rotated + with_core(jnp.real(rho_core), 2)
    rotated_g = jax.vmap(r_to_g, in_axes=(0, None))(rotated, gvectors.fft_index)

    tau_r = jnp.asarray(tau_r)
    if tau_r.shape[0] == 1:
        # A nonmagnetic spin-orbit run: the density has one component and so
        # does tau, and the "rotation" is the unpolarized halving.
        rotated_tau = 0.5 * jnp.stack([tau_r[0], tau_r[0]])
    else:
        projected = sign * jnp.sum(tau_r[1:] * direction, axis=0)
        rotated_tau = 0.5 * jnp.stack([tau_r[0] + projected, tau_r[0] - projected])

    v, c = meta_exchange(rotated, rotated_g, rotated_tau, gvectors, cell, functional)

    v0 = 0.5 * (v[0] + v[1])
    vs = 0.5 * (v[0] - v[1])
    return jnp.concatenate([v0[None], (sign * vs)[None] * direction]), c


def v_of_rho(
    rho_r: jnp.ndarray,
    gvectors: GVectors,
    cell: Cell,
    rho_core: jnp.ndarray | None = None,
    functional: Functional | None = None,
    rho_core_g: jnp.ndarray | None = None,
    quantization_axis: jnp.ndarray | None = None,
    tau: jnp.ndarray | None = None,
) -> Potential:
    """The full self-consistent potential from a real-space density.

    The Hartree term sees the valence density alone -- the core charge is a
    device for the exchange-correlation functional and carries no Hartree
    energy, since the pseudopotential already contains the core's electrostatics.

    ``rho_core_g`` is the core charge on the G-vector sphere, needed only by a
    gradient-corrected functional: its gradient is taken in G space along with
    the valence density's, which is why ``set_rhoc`` keeps ``rhog_core`` around
    rather than only its transform.

    ``tau`` is the kinetic energy density in Ry, per spin channel, and is
    required by -- and only by -- a meta-GGA functional. It is the one
    ingredient of the potential that is not a function of the density: it comes
    from the *states*, so a run under such a functional carries it beside the
    density all the way through the SCF.
    """
    functional = functional or get_functional(DEFAULT_FUNCTIONAL)
    rho_r = jnp.asarray(rho_r)
    if rho_r.ndim == 3:
        # A bare grid means one spin channel. Accepted so that callers holding a
        # plain density -- the tests, and the band-structure workflow -- do not
        # have to know about the axis.
        rho_r = rho_r[None]
    nspin = rho_r.shape[0]
    rho_g = jax.vmap(r_to_g, in_axes=(0, None))(rho_r, gvectors.fft_index)

    # The Hartree term sees the total charge and nothing else, so it is the same
    # potential in both channels -- ``v_h`` is called on ``rho%of_g(:,1)`` and
    # added to every component of ``v%of_r``.
    v_hartree_g, ehart = hartree(total_charge(rho_g), gvectors, cell)
    v_hartree_r = jnp.real(g_to_r(v_hartree_g, gvectors.fft_index, gvectors.grid))

    v_xc, etxc = exchange_correlation(rho_r, cell, rho_core, functional)
    meta_c = jnp.asarray(0.0)

    if functional.is_meta:
        if tau is None:
            raise ValueError(
                f"the {functional.name} functional is a meta-GGA: its potential "
                "depends on the kinetic energy density, which has to be passed "
                "as tau (Ry, per spin channel). Calculation.potential supplies "
                "it from the states; a bare v_of_rho call has to as well"
            )
        # The same density the local part saw -- core charge folded in, because
        # the functional is as nonlinear in it here as it is there. QE's meta
        # branch is *not* consistent about this: ``v_xc_meta`` adds the core to
        # the gradient it builds and passes ``rho%of_r`` -- valence only -- as
        # the density. That asymmetry is visible only with a nonlinear core
        # correction and is not reproduced.
        density_r = jnp.real(rho_r)
        density_g = rho_g
        if rho_core is not None and nspin != 4:
            density_r = density_r + with_core(jnp.real(rho_core), nspin)
            core_g = (
                r_to_g(jnp.real(rho_core), gvectors.fft_index)
                if rho_core_g is None
                else rho_core_g
            )
            density_g = density_g + with_core(core_g, nspin)
        if nspin == 4:
            # The local spin frame, exactly as the gradient correction does it.
            # The core charge is added *inside*, after the rotation, so it is
            # passed rather than folded in above.
            v_meta, meta_c = _noncollinear_meta_exchange(
                jnp.real(rho_r), gvectors, cell, functional, rho_core, tau,
                quantization_axis,
            )
        else:
            v_meta, meta_c = meta_exchange(
                density_r, density_g, tau, gvectors, cell, functional
            )
        # No ``etxc`` term: there is no exchange *energy* to add.
        v_xc = v_xc + v_meta

    if functional.is_gradient:
        if nspin == 4:
            # ``gradcorr``'s noncollinear branch: rotate into the local spin
            # frame, run the *collinear* gradient correction there, rotate the
            # answer back. Kept in its own function because the rotation is
            # nonlinear -- the rotated density's transform cannot be assembled
            # from the components of ``rho(G)`` and has to be taken afresh.
            v_gradient, e_gradient = _noncollinear_gradient_correction(
                jnp.real(rho_r), gvectors, cell, functional, rho_core,
                quantization_axis,
            )
            return Potential(
                v_scf=as_potential_components(v_hartree_r, nspin) + v_xc + v_gradient,
                ehart=ehart,
                etxc=etxc + e_gradient,
                meta_c=meta_c,
            )
        density_r = jnp.real(rho_r)
        density_g = rho_g
        if rho_core is not None:
            density_r = density_r + with_core(jnp.real(rho_core), nspin)
            core_g = (
                r_to_g(jnp.real(rho_core), gvectors.fft_index)
                if rho_core_g is None
                else rho_core_g
            )
            density_g = density_g + with_core(core_g, nspin)
        v_gradient, e_gradient = gradient_correction(
            density_r, density_g, gvectors, cell, functional
        )
        v_xc = v_xc + v_gradient
        etxc = etxc + e_gradient

    # The Hartree potential is spin-independent, so it follows the *potential*
    # rule and not the density's: both channels of an ``(up, down)`` potential
    # feel all of it, and only the charge component of an ``(n, m)`` one does
    # (``v_h``'s own ``IF (nspin == 4)`` at the end of ``v_of_rho.f90``).
    # Broadcasting it over four components instead adds ``v_H`` to all three
    # magnetization components -- an enormous spurious magnetic field, which
    # converges perfectly well and is wrong by more than a Rydberg per electron.
    return Potential(
        v_scf=as_potential_components(v_hartree_r, nspin) + v_xc,
        ehart=ehart,
        etxc=etxc,
        meta_c=meta_c,
    )
