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
It vanishes identically for a norm-conserving dataset, where ``S`` is the
identity and has no ``k`` in it at all, which is why an ultrasoft or PAW run is
refused here rather than trusted: nothing in a norm-conserving validation can
see the term, and an off-diagonal element with a moving ``S`` is easy to get
wrong in a way no symmetry check catches.

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

__all__ = [
    "DEGENERACY_TOL",
    "plane_wave_kubo",
    "velocity_matrices",
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


def kubo_from_matrices(
    dh1, ds1, dh2, ds2, energies, nocc: int, nbnd: int | None = None,
    degeneracy_tol: float = DEGENERACY_TOL,
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
    """
    energies = jnp.asarray(energies)
    if nbnd is not None:
        dh1, ds1 = dh1[:, :nbnd, :nbnd], ds1[:, :nbnd, :nbnd]
        dh2, ds2 = dh2[:, :nbnd, :nbnd], ds2[:, :nbnd, :nbnd]
        energies = energies[:, :nbnd]
    e = energies
    # A^1_{nm} = <n|dH_1 - e_n dS_1|m>; A^2_{mn} = <m|dH_2 - e_n dS_2|n>, the
    # transpose taken *before* the e_n subtraction so that the multiplier is
    # the outer band's energy in both factors.
    a1 = dh1 - e[:, :, None] * ds1
    a2 = jnp.swapaxes(dh2, -1, -2) - e[:, :, None] * jnp.swapaxes(ds2, -1, -2)
    gap = e[:, :, None] - e[:, None, :]
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

    total, by_band = kubo_from_matrices(
        dh1, ds1, dh2, ds2, energies, nocc, degeneracy_tol=degeneracy_tol
    )
    dropped, _ = kubo_from_matrices(
        dh1, ds1, dh2, ds2, energies, nocc, nbnd=nband - 1,
        degeneracy_tol=degeneracy_tol,
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
    """Ultrasoft and PAW, refused by name with the unvalidated term said out loud."""
    calculation = states.calculation
    if calculation is None or getattr(calculation, "augmentation", None) is None:
        return
    raise NotImplementedError(
        "the Kubo Berry curvature with an ultrasoft or PAW pseudopotential is "
        "not implemented: the *off-diagonal* element <psi_n|dS/dk_a|psi_m> "
        "enters the velocity as -e_n dS/dk and is identically zero for a "
        "norm-conserving dataset, so no norm-conserving check can see whether "
        "the convention is right. The term is written (VelocityOperator."
        "apply_s, one jvp of s_psi) and unvalidated, and there is a second one "
        "beside it -- the augmentation charge's own k-derivative, which the "
        "overlap between two *different* k-points needs as q_ij(b) rather than "
        "qq (topology/augmentation.py). Use method='fhs', which carries both "
        "correctly and is what an invariant needs anyway"
    )
