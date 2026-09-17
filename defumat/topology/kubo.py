"""The Kubo Berry curvature of a *plane-wave* calculation.

:mod:`defumat.topology.berry` has carried a ``kubo`` method since P16, and it
has only ever been reachable from a tight-binding
:class:`~defumat.topology.states.ModelStates`, where ``H(k)`` is a small dense
matrix that ``jacfwd`` can differentiate and ``eigh`` can diagonalise. Its
refusal for a real crystal said the velocity operator "needs ``d(vkb)/dk`` and
the k-dependence of the plane-wave sphere" and pointed at P11. **P24 wrote
both**: :class:`~defumat.response.velocity.VelocityOperator` is one
``jax.jvp`` of ``H(k)`` at a frozen sphere, and the sphere being frozen is
exactly the right thing -- membership is piecewise constant in ``k`` and on
each piece the frozen-sphere derivative is the exact one. This module is that
operator cashed in one quantity further along.

**Nothing dense is formed, and that is the whole design constraint.** The
obvious adaptor -- give ``PlaneWaveStates`` a ``hamiltonian(k)`` returning a
matrix, so that the existing ``_kubo_point`` runs unchanged -- is forbidden:
``H`` as a matrix is ``npw^2``, which ``CLAUDE.md`` rules out outright ("a
dense solve is a test fixture, never a ``diagonalization`` a run can select").
So the sum is written as **band matrix elements between the states an NSCF
already produced**: one ``jvp`` per crystal direction gives ``v_a|psi_n>`` for
every band at every k-point, and the Kubo expression is a contraction of that
against the same states.

The expression, for a **generalised** eigenproblem ``H|n> = e_n S|n>``:

    Omega_n^{12}(k) = -2 Im sum_{m != n} A^1_{nm} A^2_{mn} / (e_n - e_m)^2,
    A^a_{nm} = <psi_n| dH/dk_a - e_n dS/dk_a |psi_m>.

The ``e_n dS/dk`` piece is not decoration and its index is not free: it comes
from differentiating ``H|m> = e_m S|m>`` and projecting on ``<n|``, which gives
``<n|S|d_a m> = <n|(dH/dk_a - e_m dS/dk_a)|m> / (e_m - e_n)``. Carrying that
through the curvature leaves ``e_n`` -- the band whose curvature is being
computed -- in **both** factors, not ``e_n`` in one and ``e_m`` in the other.

**And it is not the whole of what a moving overlap does**, which is the one
thing an ultrasoft or PAW dataset adds here and the reason this was refused
until P98. With ``S = T^dagger T`` the states a Berry phase is about are
``T|psi>``, so the connection carries ``<psi_n|T^dag dT/dk_a|psi_m>`` beside
``<psi_n|S d_a psi_m>``, and that piece is a function of ``T`` rather than of
``S``: ``U(k) T`` leaves ``S`` alone and moves the physical states, so no
arrangement of ``dS/dk`` can supply it. :func:`augmentation_connection` builds
it, ``kubo_from_matrices`` takes it as ``k1`` and ``k2``, and the two factors
take **different** blocks -- ``L = K^dagger`` in the first and ``K`` in the
second, with ``L + K = dS``. A norm-conserving dataset has ``T = 1``, so
everything in this paragraph is identically zero there and no norm-conserving
validation can see any of it; what pins it instead is an exact identity, the
``k``-derivative of the two-point overlap ``S(k, k')`` at frozen coefficients,
which is Fukui-Hatsugai-Suzuki's own object and shares nothing with this sum.

A **spinor** augmented run is still refused, and by a term rather than by
caution: its overlap carries ``qq_so`` and its dipole the same ``fcoef``
transform, which is not written.

**Two honest numbers come out with the curvature**, because both are ways this
answer can be quietly wrong.

* ``truncation``. The sum over ``m`` runs over the bands the eigensolver was
  asked for and stops. That is a real approximation -- the Sternheimer stack
  exists precisely to avoid it (P24) -- and it is reported the way P37 reports
  ``static_residual``: the shift in the zone-summed curvature when the highest
  empty band is dropped, relative to the curvature's own scale. It is a
  diagnostic to read, not a knob to tune until a test passes.
* ``curvature_by_band``. Band by band the Kubo curvature is gauge invariant
  only for a **non-degenerate** band; inside a degenerate multiplet the
  eigensolver's arbitrary rotation moves the members' values and only their sum
  is defined. The manifold total this module returns as ``curvature`` never has
  that problem -- it is restricted to occupied/empty pairs, so an
  intra-manifold degeneracy never enters it -- while the per-band array does,
  and says so.

**It is still not what an invariant is read from.** The ``1/(e_n - e_m)^2``
denominator is what design rule D4 forbids, the Brillouin-zone sum is an
ordinary Riemann sum that converges to an integer without ever being one, and
Fukui-Hatsugai-Suzuki is exact on any mesh. ``fhs`` stays the default and stays
the only method a Chern number should be taken from. What this is for is the
smooth ``Omega(k)`` map -- the anomalous-Hall picture -- that a plaquette sum
can only give as a cell average.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from defumat.batching import map_k
from defumat.pseudo.augmentation import augmentation_dipole_blocks

__all__ = [
    "DEGENERACY_TOL",
    "plane_wave_kubo",
    "velocity_matrices",
    "augmentation_connection",
    "kubo_from_matrices",
]

#: Below this gap (Ry) a pair of bands is treated as degenerate and dropped
#: from the sum. For the occupied/empty pairs the total is built from, a gap
#: this small is a genuine singularity and ``DFTSource._check_gap`` has already
#: refused it; what the mask is actually for is the *intra*-manifold
#: degeneracies the per-band array runs into (a zincblende crystal has a
#: three-fold one at Gamma).
DEGENERACY_TOL = 1.0e-6


def velocity_matrices(states, direction):
    """``<psi_m|dH/dk_a|psi_n>`` and ``<psi_m|dS/dk_a|psi_n>``, both ``(nk, nb, nb)``.

    ``direction`` is a **cartesian** 3-vector in 1/bohr; a crystal direction
    ``d`` is reached by handing it the reciprocal lattice vector ``bg[d]``,
    which is what makes the result a derivative with respect to the crystal
    coordinate the mesh is spanned by.

    One ``jvp`` produces both tangents (``VelocityOperator.both``): the whole
    cost is rebuilding ``vkb(k)`` as a differentiable function of ``k``, and
    the overlap's tangent rides along for nothing.
    """
    velocity = states.velocity
    psi = jnp.asarray(states.all_coefficients)[None]  # (1, nk, nband, ndim)
    dh, ds = velocity.both(psi, jnp.asarray(direction))
    bra = psi.conj()
    dh_mat = jnp.einsum("skmg,skng->skmn", bra, dh)[0]
    ds_mat = jnp.einsum("skmg,skng->skmn", bra, ds)[0]
    return dh_mat, ds_mat


def augmentation_connection(states, direction):
    """``<psi_n| T^dagger dT/dk_a |psi_m>``, ``(nk, nb, nb)``, or ``None``.

    The piece of the Berry connection that a moving overlap adds and that
    ``dH/dk`` and ``dS/dk`` between them cannot supply. With
    ``S = T^dagger T`` the states a Berry phase is about are ``T|psi>``, so

        <Psi_n|d_a Psi_m> = <psi_n|S|d_a psi_m> + <psi_n|T^dag d_a T|psi_m>,

    and only the first term is a function of ``S``. The second is what this
    builds. ``U(k) T`` leaves ``S`` unchanged and moves the physical states, so
    no arrangement of ``dS/dk`` can stand in for it; ``PLAN.md`` P94 measures
    the omission at 18 per cent of the curvature on a model where the exact
    answer is free.

    **Where the expression comes from.** Expand the overlap between two
    neighbouring k-points, which is the object Fukui-Hatsugai-Suzuki already
    uses (:func:`~defumat.topology.augmentation.augmentation_at_q`),

        S(k, k') = 1 + sum_ij e^{-i b.tau} Q_ij(b) |beta^k_i><beta^k'_j|,
        b = k' - k,

    to first order in ``b``. Two objects appear: ``Q_ij(b) = q_ij - i b_a
    dpqq^a_ij``, the augmentation dipole, and the ket projector's own motion.
    Their ``tau`` terms cancel **exactly** -- the structure factor of
    ``e^{-i b.tau}`` against the ``-i tau_a`` inside ``d(vkb)/dk_a`` -- and what
    is left is

        T^dag d_a T = sum_ij q_ij |beta_i><d_a beta_j| - i sum_ij dpqq^a_ij
                      |beta_i><beta_j|,

    with the projector derivative the one about the atom's **own centre**. That
    is the same convention pair ``adddvepsi_us`` runs on
    (:func:`~defumat.response.efield.ultrasoft_position`), and it is why
    :meth:`~defumat.response.velocity.VelocityOperator.projectors` and
    :func:`~defumat.pseudo.augmentation.augmentation_dipole_blocks` are the two
    ingredients rather than the full ``d(vkb)/dk``.

    ``direction`` is the same cartesian tangent :func:`velocity_matrices` takes,
    so the dipole is contracted with it and the result is a derivative with
    respect to the crystal coordinate the mesh is spanned by.

    Returns ``None`` for a norm-conserving dataset, where ``T`` is the identity
    and there is nothing to add.
    """
    calculation = getattr(states, "calculation", None)
    if calculation is None or getattr(calculation, "augmentation", None) is None:
        return None
    projectors = calculation.projectors
    if projectors.nkb == 0 or projectors.qq is None:
        return None

    direction = jnp.asarray(direction)
    dipole = augmentation_dipole_blocks(calculation)  # (3, nkb, nkb), bohr
    psi = jnp.asarray(states.all_coefficients)  # (nk, nb, ndim)
    vkb = projectors.vkb                        # (nk, npwx, nkb)
    dvkb = states.velocity.projectors(direction)
    qq = projectors.qq.astype(vkb.dtype)
    along = jnp.einsum(
        "a,aij->ij", direction.astype(dipole.dtype), dipole
    ).astype(vkb.dtype)

    def one_k(ik):
        # becp[n, i] = <beta_i|psi_n>; dbecp[m, j] = <d_a beta_j|psi_m>.
        becp = jnp.einsum("gi,ng->ni", vkb[ik].conj(), psi[ik])
        dbecp = jnp.einsum("gj,ng->nj", dvkb[ik].conj(), psi[ik])
        moving = jnp.einsum("ni,ij,mj->nm", becp.conj(), qq, dbecp)
        static = jnp.einsum("ni,ij,mj->nm", becp.conj(), along, becp)
        return moving - 1j * static

    return map_k(one_k, jnp.arange(psi.shape[0]), batch=calculation.k_batch)


def kubo_from_matrices(
    dh1, ds1, dh2, ds2, energies, nocc: int, nbnd: int | None = None,
    degeneracy_tol: float = DEGENERACY_TOL, k1=None, k2=None,
):
    """``(Omega(k), Omega_n(k))`` from the two directions' velocity blocks.

    ``Omega`` is ``(nk,)``, the manifold total, summed over occupied ``n`` and
    **empty** ``m`` only -- which is the same number as the sum over all
    ``m != n`` (the occupied/occupied terms cancel in pairs under the
    imaginary part) and is the form that never divides by an intra-manifold
    degeneracy. ``Omega_n`` is ``(nk, nocc)`` and does sum over all ``m != n``,
    because that is what the per-band curvature *is*; see the module docstring
    for what it means inside a multiplet.

    ``nbnd`` truncates the sum, which is how the truncation diagnostic is
    computed without a second ``jvp``.

    ``k1`` and ``k2`` are :func:`augmentation_connection`'s blocks for the two
    directions, and they are ``None`` for a norm-conserving dataset. They do
    **not** enter the two factors the same way, which is the whole content of
    the term: the first takes ``L = K^dagger`` and the second takes ``K``, each
    multiplied by the gap, so no single object of the form ``dS/dk`` can stand
    for the pair. ``L + K = dS`` exactly, and that identity is what a
    plane-wave implementation is checked against.
    """
    energies = jnp.asarray(energies)
    if nbnd is not None:
        dh1, ds1 = dh1[:, :nbnd, :nbnd], ds1[:, :nbnd, :nbnd]
        dh2, ds2 = dh2[:, :nbnd, :nbnd], ds2[:, :nbnd, :nbnd]
        energies = energies[:, :nbnd]
        k1 = None if k1 is None else k1[:, :nbnd, :nbnd]
        k2 = None if k2 is None else k2[:, :nbnd, :nbnd]
    e = energies
    # A^1_{nm} = <n|dH_1 - e_n dS_1|m>; A^2_{mn} = <m|dH_2 - e_n dS_2|n>, the
    # transpose taken *before* the e_n subtraction so that the multiplier is
    # the outer band's energy in both factors.
    a1 = dh1 - e[:, :, None] * ds1
    a2 = jnp.swapaxes(dh2, -1, -2) - e[:, :, None] * jnp.swapaxes(ds2, -1, -2)
    gap = e[:, :, None] - e[:, None, :]
    # The moving overlap. ``gap[n, m] = e_n - e_m`` here, and the second factor
    # is transposed above, so the same ``+ gap *`` reaches ``L`` in the first
    # and ``K^T`` in the second -- which is the asymmetry the model check in
    # ``tests/unit/test_topology_curvature.py`` pins to 4.5e-16.
    if k1 is not None:
        a1 = a1 + gap * jnp.conj(jnp.swapaxes(k1, -1, -2))
    if k2 is not None:
        a2 = a2 + gap * jnp.swapaxes(k2, -1, -2)
    finite = jnp.abs(gap) > degeneracy_tol
    weight = jnp.where(finite, 1.0 / jnp.where(finite, gap, 1.0) ** 2, 0.0)
    terms = -2.0 * jnp.imag(a1 * a2) * weight

    nband = e.shape[1]
    occupied = jnp.arange(nband) < nocc
    inter = occupied[:, None] & ~occupied[None, :]
    total = jnp.sum(jnp.where(inter, terms, 0.0), axis=(1, 2))
    off_diagonal = occupied[:, None] & ~jnp.eye(nband, dtype=bool)
    by_band = jnp.sum(jnp.where(off_diagonal, terms, 0.0), axis=2)[:, :nocc]
    return total, by_band


def plane_wave_kubo(
    states, mesh, axes, nocc: int | None = None,
    degeneracy_tol: float = DEGENERACY_TOL, **_,
):
    """``Omega(k)`` on a plane mesh, from ``VelocityOperator``.

    ``axes`` is the pair of **crystal** directions the mesh spans; the tangents
    handed to the velocity operator are the corresponding reciprocal lattice
    vectors, so the curvature comes out in the same crystal-``k`` units the
    ``fhs`` path's flux does and the two are directly comparable.
    """
    from defumat.topology.berry import BerryCurvature

    _require_velocity(states)
    _refuse_augmented(states)

    nocc = states.nbnd if nocc is None else int(nocc)
    energies = jnp.asarray(states.energies)
    nband = int(energies.shape[1])
    if nband <= nocc:
        raise ValueError(
            f"the Kubo curvature is a sum over empty states and this "
            f"diagonalisation resolved {nband} bands with {nocc} of them "
            "occupied, so the sum is empty. Raise nbnd -- and read the "
            "reported truncation, because the sum is truncated wherever it "
            "stops"
        )
    if int(states.all_coefficients.shape[1]) != nband:
        raise ValueError(
            "the state set's band count and its eigenvalues disagree; the "
            "Kubo sum needs the whole diagonalised set, not the occupied "
            "manifold"
        )

    d1, d2 = axes
    bg = np.asarray(states.bg)
    dh1, ds1 = velocity_matrices(states, bg[d1])
    dh2, ds2 = velocity_matrices(states, bg[d2])
    # ``None`` for a norm-conserving dataset, where ``T`` is the identity.
    k1 = augmentation_connection(states, bg[d1])
    k2 = augmentation_connection(states, bg[d2])

    total, by_band = kubo_from_matrices(
        dh1, ds1, dh2, ds2, energies, nocc, degeneracy_tol=degeneracy_tol,
        k1=k1, k2=k2,
    )
    dropped, _ = kubo_from_matrices(
        dh1, ds1, dh2, ds2, energies, nocc, nbnd=nband - 1,
        degeneracy_tol=degeneracy_tol, k1=k1, k2=k2,
    )
    total = np.asarray(total)
    dropped = np.asarray(dropped)
    scale = float(np.max(np.abs(total)))
    shift = float(np.max(np.abs(total - dropped)))
    truncation = shift / scale if scale > 0.0 else float("nan")

    n1, n2 = mesh.shape
    return BerryCurvature(
        mesh=mesh,
        curvature=total.reshape(n1, n2),
        flux=None,
        method="kubo",
        curvature_by_band=np.asarray(by_band).reshape(n1, n2, nocc),
        nbnd=nband,
        nocc=nocc,
        truncation=truncation,
        truncation_abs=shift,
    )


def _require_velocity(states) -> None:
    if getattr(states, "velocity", None) is None or getattr(
        states, "all_coefficients", None
    ) is None:
        raise ValueError(
            "the Kubo curvature of a plane-wave calculation needs the velocity "
            "operator and the *whole* diagonalised band set, and this state "
            "set carries neither: build it with "
            "DFTSource.states(points, keep_velocity=True), which is what "
            "run_berry_curvature(method='kubo') does. The occupied manifold "
            "alone is enough for 'fhs' and not for a sum over empty states"
        )


def _refuse_augmented(states) -> None:
    """An augmented **spinor**, refused by name; the scalar case runs (P98).

    The scalar ultrasoft and PAW refusal is gone: what it named as missing is
    :func:`augmentation_connection`. What is left is the spin structure, and it
    is a term rather than plumbing -- a fully-relativistic dataset's overlap
    carries ``qq_so``, a 2x2 matrix in spin space built through ``fcoef``, and
    its dipole is the same transform applied to ``dpqq``, which
    :func:`~defumat.pseudo.augmentation.augmentation_dipole_blocks` does not do.
    """
    calculation = states.calculation
    if calculation is None or getattr(calculation, "augmentation", None) is None:
        return
    if int(getattr(calculation, "npol", 1)) == 1:
        return
    raise NotImplementedError(
        "the Kubo Berry curvature of a *spinor* ultrasoft or PAW run is not "
        "implemented: the scalar case is (PLAN.md P98), and what the spinor "
        "one needs beyond it is the spin structure of the overlap. S carries "
        "qq_so, the fcoef transform of qq into a 2x2 matrix in spin space "
        "(transform_qq_so), and the connection's augmentation dipole is that "
        "same transform applied to dpqq -- which augmentation_dipole_blocks "
        "does not build. Use method='fhs', which carries the whole thing as "
        "q_ij(b) at every npol and is what an invariant needs anyway"
    )
