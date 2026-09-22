"""Which of the four directions a noncollinear SCF is still moving in.

``MAGNETISM-NEXT.md`` F2 reframes "the magnet takes twice the iterations
``pw.x`` takes" as a question about the *spectrum* of the map the mixer damps
rather than about the mixer: a density error ``e`` comes back as
``(1 - beta) e + beta J e`` with ``J = chi_0 K``, and ``J`` goes wrong at both
ends for different reasons. In a noncollinear cell there are four such
directions with four different physical origins, and only the first has anything
acting on it:

* **long-wavelength charge**, amplified by ``q_TF^2/q^2`` through the Hartree
  kernel, which is charge sloshing and which Kerker and ``local-TF`` divide out;
* **the longitudinal magnetization**, the length of ``m(r)``, where the kernel is
  local so there is no long-wavelength divergence and Kerker has nothing to say,
  and what there is instead is the Stoner enhancement ``chi_0/(1 - I chi_0)``,
  which amplifies a uniform change at every wavelength equally;
* **the rigid rotation of every moment together**, a Goldstone mode of the broken
  spin-rotation symmetry whose restoring force is exactly zero, so the fixed
  point is a two-parameter family and the residual has no component along it --
  the mixer is *stuck* there rather than unstable, and the manifold has to be
  traversed;
* **the transverse channel at finite q**, a slow twist of the direction of
  ``m(r)``, whose restoring force vanishes as ``D q^2`` so that a large cell is
  worse than a small one, and where ``D`` is negative the iteration is not slow
  but genuinely divergent.

**Nothing here has ever been told apart on which of the four it is slow in**, and
that is what this module is for. It is Option 0 of that item, and the item's own
rule is that none of the options below it can be chosen without this dump.

**The rotation bin is the one that has to be defined carefully, because the
obvious definition is a null that cannot be told from a pass.** Taking it as the
``Q = 0`` transverse component is right for a ferromagnet and is identically zero
for every compensated texture: an antiferromagnet, a spiral and a helix all have
``m_{Q = 0} = 0``, and a rigid rotation leaves it zero, so the bin would read zero
whether or not the mode is live. The definition that works for any texture is the
projection of the residual on the three **generators** of the rotation,

    c_a = int dm(r) . (e_a x m_in(r)) dr / int |e_a x m_in(r)|^2 dr,

which reduces to the transverse ``Q = 0`` component for a ferromagnet and costs
the same one pass over the grid.

**A diagnostic must not change the run it is diagnosing**, which is why this is
computed from ``rho_out - rho_in`` after the mixer has been handed the same
array, feeds nothing, and is off unless asked for: ``accuracy`` drives the
``ethr`` schedule and one ulp there moves every eigenvalue.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

__all__ = ["residual_bins", "becsum_residual", "ROTATION_GENERATORS"]

#: The three generators of a rigid spin rotation, as the antisymmetric action
#: ``e_a x m``. Written as index triples rather than as matrices because the
#: cross product of a fixed axis with a vector field is two multiplications and
#: a subtraction per point, and a 3x3 contraction over a grid is not.
ROTATION_GENERATORS = ((1, 2), (2, 0), (0, 1))

#: Below this the pointwise direction of ``m_in`` is not defined and the
#: longitudinal/transverse split of the residual at that point is meaningless.
#: Such points are reported in their own bin rather than assigned to either, so
#: that a cell with vacuum -- where most of the grid is here -- cannot quietly
#: inflate one of the two.
MAGNETIZATION_FLOOR = 1.0e-10


def _norm(field, weight) -> float:
    """The volume-weighted L2 norm of a real field, so the bins are comparable."""
    return float(jnp.sqrt(jnp.sum(jnp.asarray(field) ** 2) * weight))


def residual_bins(rho_in, rho_out, cell, becsum_in=None, becsum_out=None) -> dict:
    """Split one iteration's residual into the five bins F2 asks for.

    Args:
        rho_in: ``(nspin_mag, n1, n2, n3)`` the density the iteration started
            from, which is what the magnetization direction is taken from.
        rho_out: the same shape, the density it produced.
        cell: for the quadrature weight.
        becsum_in, becsum_out: the augmentation occupations, if the run has
            them. Their magnetic part is the fifth bin and is the blind spot the
            item is about: ``becsum`` is mixed at the plain ``beta`` and appears
            in no convergence measure at all, since both halves of ``accuracy``
            are of the smooth density and what reaches them is only what
            ``addusdens`` already put on the grid.

    Returns:
        A dict of scalars. ``charge``, ``longitudinal``, ``rotation``,
        ``transverse`` and ``becsum`` are the five bins; ``rotation_coefficients``
        are the three ``c_a`` themselves, whose *signs* say which way the texture
        is turning where the norm only says how fast.
    """
    rho_in = jnp.asarray(rho_in)
    rho_out = jnp.asarray(rho_out)
    residual = rho_out - rho_in
    weight = cell.volume / rho_in[0].size

    bins = {"charge": _norm(residual[0], weight)}
    if rho_in.shape[0] < 4:
        # A collinear run has no direction to project on: the whole magnetic
        # residual is longitudinal by construction, and the rotation bin is
        # exactly zero, which is the other half of the trip test. A run with no
        # magnetization at all has neither, and **every key is still present**,
        # because a consumer that reads a bin by name should get a zero for a
        # channel the regime does not have rather than a KeyError -- the whole
        # point of the dump is to compare a magnetic run against its nonmagnetic
        # twin, so both have to be readable by the same code.
        bins["longitudinal"] = (
            _norm(residual[1], weight) if rho_in.shape[0] == 2 else 0.0
        )
        bins["rotation"] = 0.0
        bins["transverse"] = 0.0
        bins["rotation_coefficients"] = [0.0, 0.0, 0.0]
        bins["unpolarized_points"] = 0
        return _with_becsum(bins, becsum_in, becsum_out)

    magnetization = rho_in[1:4]
    dm = residual[1:4]
    length = jnp.sqrt(jnp.sum(magnetization**2, axis=0))
    live = length > MAGNETIZATION_FLOOR
    # ``jnp.where`` on the denominator rather than a clip on the length: the
    # tangent of a clip at its boundary is the trap this project has paid for
    # four times, and although nothing differentiates this diagnostic, writing
    # the guarded division the other way would be a copy of that pattern for a
    # later reader to inherit.
    safe = jnp.where(live, length, 1.0)
    direction = magnetization / safe

    parallel = jnp.sum(dm * direction, axis=0)
    perpendicular = dm - parallel * direction
    bins["longitudinal"] = _norm(jnp.where(live, parallel, 0.0), weight)

    # The rotation bin: project the residual on the three generators. The
    # normalisation is the generator's own norm, so ``c_a`` is an angle in
    # radians -- the rigid rotation that best fits this residual -- rather than
    # an amplitude in whatever units the density carries.
    coefficients, generators = [], []
    for first, second in ROTATION_GENERATORS:
        generator = jnp.zeros_like(magnetization)
        generator = generator.at[first].set(-magnetization[second])
        generator = generator.at[second].set(magnetization[first])
        overlap = jnp.sum(dm * generator) * weight
        norm = jnp.sum(generator**2) * weight
        coefficients.append(jnp.where(norm > 0.0, overlap / jnp.where(norm > 0.0, norm, 1.0), 0.0))
        generators.append(generator)

    rotation = sum(
        coefficient * generator
        for coefficient, generator in zip(coefficients, generators)
    )
    bins["rotation"] = _norm(rotation, weight)
    bins["rotation_coefficients"] = [float(value) for value in coefficients]
    # What is left of the transverse residual once the rigid rotation is out of
    # it, which is the finite-q twist. The two are not summed to the
    # perpendicular norm -- the generators are not orthogonal to each other for
    # a general texture -- so this is a residue rather than a difference.
    bins["transverse"] = _norm(perpendicular - rotation, weight)
    bins["unpolarized_points"] = int(np.asarray(jnp.sum(~live)))
    return _with_becsum(bins, becsum_in, becsum_out)


def becsum_residual(becsum_in, becsum_out) -> tuple:
    """``(total, magnetic)``: how far the augmentation occupations moved.

    **This is the half of the residual that `accuracy` cannot see**, and it is
    reported because of that rather than fed to anything. `scf_accuracy_split`
    takes the *smooth density* residual, so what reaches the convergence number
    from ``becsum`` is only what ``addusdens`` already put on the grid,
    ``Q_ij(r) becsum``, at the charge half's ``1/|G|^2`` weight; PAW's one-centre
    piece, the all-electron minus pseudo Hartree and exchange-correlation on the
    radial grids, is not in it at all.

    **That matches `pw.x` exactly and is not a deviation**, which is also why
    this number must stay outside the convergence test rather than being added
    to it. QE writes the term and comments it out (``PW/src/scf_mod.f90:843``):

    .. code-block:: fortran

        ! Beware: commented out because it yields too often negative values
        ! IF (okpaw)  rho_ddot = rho_ddot + paw_ddot(rho1%bec, rho2%bec)

    ``paw_ddot`` is not positive definite, so a convergence measure built on it
    can go negative. What is wrong is not that the term is missing from `dr2`; it
    is that nothing anywhere reported it, so a run stalling inside it looked
    exactly like a run converging.

    **Two numbers rather than one**, because on a magnet the moment lives in the
    d-shell occupations: the total, and the part carried by the magnetization
    channels alone, which is the one that can move while the charge sits still.
    Channel 0 is the charge and is the piece ``addusdens`` already routed onto
    the grid.

    The norm is a plain Euclidean one over the occupations, whose units are
    electrons rather than the density's, so **it is comparable with itself across
    iterations and not with `accuracy`**. Quoting it beside `dr2` as though the
    two were the same size is the mistake this docstring exists to prevent.
    """
    if not becsum_in or not becsum_out:
        return None, None
    total = magnetic = 0.0
    for before, after in zip(becsum_in, becsum_out):
        before = jnp.asarray(before)
        after = jnp.asarray(after)
        if before.ndim == 0 or before.size == 0:
            continue
        difference = after - before
        total += float(jnp.sum(difference**2))
        if before.shape[0] >= 2:
            magnetic += float(jnp.sum(difference[1:] ** 2))
    return float(np.sqrt(total)), float(np.sqrt(magnetic))


def _with_becsum(bins: dict, becsum_in, becsum_out) -> dict:
    """The fifth bin, which is :func:`becsum_residual`'s magnetic half."""
    _, magnetic = becsum_residual(becsum_in, becsum_out)
    bins["becsum"] = magnetic
    return bins
