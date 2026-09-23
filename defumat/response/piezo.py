"""The clamped-ion piezoelectric tensor as the mixed second derivative it is.

``PLAN.md`` P50. ``e_(k)ij = dP_k/d(eps_ij) = d(sigma_ij)/dE_k``: the
polarization a strain induces, which is the same number as the stress an
electric field induces, because both are

    e_(k)ij = -(1/Omega) d^2 E / d(eps_ij) dE_k

and a mixed second derivative does not care which leg is taken first. Baroni,
de Gironcoli, Dal Corso and Giannozzi's review (`Rev. Mod. Phys. 73, 515
<https://arxiv.org/abs/cond-mat/0012092>`_, §II.C.2) states the equivalence and
records that the stress-under-a-field route is the one de Gironcoli, Baroni and
Resta took for the III-V compounds (PRL **62**, 2853 (1989)).

**It is P24b's construction with one coordinate changed.** A Born effective
charge is ``Z* = dF/dE``, and :mod:`defumat.response.born` computes it as one
``jvp`` of the *force* along the field's response. The force is ``jax.grad`` of
the frozen-state energy in the atomic positions; the **stress** is ``jax.grad``
of the same functional in a strain (:mod:`defumat.stress.energy`). So the
piezoelectric tensor is one ``jvp`` of the stress along the same field
response, per field direction -- three of them and the tensor is complete, at
the cost of the dielectric constant that was going to be solved anyway.

**For a norm-conserving dataset the strain leg has no orthonormality term, and
that is what makes this cheaper than the Born charge it copies.**
``<psi|S|psi>`` is then a sum over the plane-wave sphere of ``|c_G|^2`` and the
sphere is a set of *integers*, so the constraint is strain-independent, its
strain derivative vanishes identically as a function of the states too, and the
multiplier response ``dLambda`` that :mod:`defumat.response.born` needs (QE's
``psidspsi``, ``add_dkmds``, ``add_for_charges``) has nothing to contribute.
What is left is one term::

    Omega e_(k)ij = -d/dE_k [ dE/d(eps_ij) ] = -(d_psi d_eps E) . dpsi_k

*This paragraph used to say that without its first three words, and it was
false on an augmented dataset by* **-0.0055 C/m^2** *on ultrasoft AlAs.*
``qq_ij = int Q_ij(r) d3r`` *has no cell in it, which is true, and the
conclusion does not follow:* ``S = 1 + sum |beta> qq <beta|`` *also carries*
``vkb``, *which is* ``beta(|k+G|)`` *and deforms with the cell like every other
radial transform. So an augmented dataset has both halves of* ``dLambda``
*after all, and* :func:`clamped_ion_piezoelectric` *carries them
(*``solver``, ``field_perturbations`` *and* ``commutators`` *together switch
them on) while a norm-conserving run skips them exactly rather than
approximately, since both are contracted with a* ``dS/d(eps)`` *that is
identically zero.*

*Units.* The field response is the one :mod:`defumat.response.born` divides
into the force to get a charge in units of ``e``, so the same tangent through a
gradient in the dimensionless strain gives ``e bohr``; divided by the cell
volume that is ``e/bohr^2``, and :data:`~defumat.units.E_BOHR2_TO_C_M2` takes
it to C/m^2. Nothing about the field's normalisation has to be known here, and
that is deliberate: :func:`born_charges_from_stress_route` runs this module's
own assembly with :meth:`~defumat.scf.driver.Calculation.at_positions` in
place of :meth:`~defumat.scf.driver.Calculation.at_strain` and must reproduce
the Born charges, which are validated against ``ph.x`` to every digit it
prints. That is the test the scale and the sign rest on.

**Proper against improper, and why every target here is a crystal where they
coincide.** The tensor above is the *improper* one -- the bare mixed second
derivative. What a measurement sees is Vanderbilt's proper piezoelectric
response (`J. Phys. Chem. Solids 61, 147 (2000)
<https://doi.org/10.1016/S0022-3697(99)00273-5>`_; QE cites it in
``PW/src/bp_c_phase.f90`` and computes nothing from it), and the two differ by
terms built from the polarization itself,

    e^proper_(k)ij = e^improper_(k)ij + delta_ki P_j - delta_ij P_k,

which arise because a strain carries the charge distribution with the cell
(``d(Omega P_k)/d(eps_ij)|_frozen = delta_ki Omega P_j``) and changes the volume
that divides it. **Both corrections vanish identically whenever the two
Cartesian labels they pair are different**, so the shear components with all
three indices distinct -- ``e_14`` of a zincblende crystal, which is its only
independent component -- carry no ambiguity at all. And they vanish for *every*
component of a crystal whose spontaneous polarization is zero, which any
non-pyroelectric class is: the class ``-43m`` of AlAs has no invariant vector.
A **polar crystal is refused by name** rather than corrected, because the
correction needs ``P`` itself and a Berry-phase polarization is not implemented
in this package (:func:`require_a_nonpolar_crystal`).

**What is left out and is not an approximation.** This is the *clamped-ion*
constant: the atoms are carried along by the strain in crystal coordinates and
are not allowed to relax. The measured constant adds the internal-strain term
``sum_a Z*_a (C^-1)_a,b Lambda_b`` of the review's Eq. (111), and the review is
also where the warning about it belongs -- the two contributions "are often of
opposite sign and close in absolute value, so that a well converged calculation
is needed in order to extract a reliable value for their sum". Every ingredient
of that term is in this package (``Z*`` from P24b, the force constants from
P25, and ``Lambda = -d^2E/du d(eps)`` which is this module's ``jvp`` with the
strain response as its tangent instead of the field's), and it is the next step
rather than part of this one.

**The peak working set is the tape of a strained gradient, and it is large.**
Forward-over-reverse through :func:`~defumat.stress.energy.strained_energy`
holds every radial and reciprocal-space intermediate the *setup* rebuilds when
the cell moves -- ``V_loc(|G|)``, ``rho_core(|G|)``, ``f_l(|k+G|)``,
``Q^L_nm(|G|)`` -- which a displacement does not touch at all. Measured on
two-atom AlAs at ``ecutwfc = 10`` with 64 k-points: **4.2 GB** against the Born
charge's 1.13 on the same field response, and 6.4 s against 0.8. It does **not**
move with ``k_batch``, because what the tape holds is not the k axis.
:func:`piezoelectric_zstar_eu_style` is the same number for 4.0 s and no extra
memory, and on a cell where this one does not fit it is the route to reach for.
`PERFORMANCE.md` carries the table.

**Elk is the only established code with a piezoelectric tensor and it takes the
expensive route.** ``piezoelt.f90`` (task 380) runs a full ground state per
strain tensor, computes the Berry-phase polarization of each, and finite-
differences them with a ``2 pi`` branch fix-up between the two -- ``nstrain``
self-consistent calculations where this is one. ``pw.x`` has no piezoelectric
tensor at all: the only occurrence of the word in the vendored tree is a
citation of Vanderbilt's paper in a comment in ``PW/src/bp_c_phase.f90``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from defumat.batching import map_k
from defumat.forces.energy import FrozenState, energy_at
from defumat.response.born import _raw_mixed_state
from defumat.response.efield import (
    dielectric_tensor,
    require_a_symmetrisable_response,
)
from defumat.response.electrostriction import (
    refined_states,
    require_converged_responses,
)
from defumat.response.sternheimer import require_a_sternheimer_regime
from defumat.stress.energy import require_a_differentiable_cell
from defumat.units import E_BOHR2_TO_C_M2

__all__ = [
    "PiezoelectricTensor",
    "piezoelectric_tensor",
    "clamped_ion_piezoelectric",
    "born_charges_from_stress_route",
    "piezoelectric_zstar_eu_style",
    "piezoelectric_from_strain_response",
    "require_a_piezoelectric_tensor",
    "KMESH_STEP",
    "ULTRASOFT_MESH",
    "require_a_measured_dataset",
    "require_a_nonpolar_crystal",
    "polar_direction",
    "to_voigt",
]

#: ``(i, j)`` of each Voigt index, in the standard order ``xx yy zz yz xz xy``.
VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


@dataclass(frozen=True)
class PiezoelectricTensor:
    """What :func:`piezoelectric_tensor` returns.

    Attributes:
        e: ``(3, 3, 3)`` in ``e/bohr^2``, indexed ``e[k, i, j]`` -- the
            polarization along ``k`` induced by the strain component ``ij``,
            symmetric in the last two indices.
        voigt: ``(3, 6)`` in **C/m^2**, the form a table quotes ``e_14`` in:
            the columns are ``xx yy zz yz xz xy`` and there are no factors of
            two, for :class:`~defumat.response.elastic.ElasticConstants`'
            reason -- the engineering convention puts the two on the *strain*
            (``eps_4 = 2 eps_23``), so ``P_i = e_iJ eps_J`` needs none on the
            coefficient.
        dielectric: the :class:`~defumat.response.efield.DielectricTensor` the
            field response was taken from, so that ``epsilon_infinity`` and (if
            it was asked for) the Born charges come back with it rather than
            costing a second solve.
        converged: whether that response converged.
        nk: how many k-points the response was integrated over, and ``grid``
            the Monkhorst-Pack divisions behind them when the set came from a
            grid rather than from an explicit list. **This is on the result
            because it is the one parameter of this quantity that nothing
            inside it can see**: the three routes share one field response and
            the ``Z*`` anchor is the same assembly in another coordinate, so
            every internal check moves with the mesh instead of catching it.
        kmesh_drift: the relative change of ``e`` across the last step of a
            k-mesh ladder, when one was run
            (:func:`~defumat.workflows.piezo_ladder.piezoelectric_kmesh_ladder`),
            and ``None`` when the mesh was taken on trust. It is a *step* and
            not an error: on the one cell where both are known the remaining
            distance to the Berry-phase value was about three times the last
            step.
    """

    e: np.ndarray
    voigt: np.ndarray
    dielectric: object = None
    converged: bool = True
    nk: int | None = None
    grid: tuple | None = None
    kmesh_drift: float | None = None

    @property
    def e14(self) -> float:
        """The single independent component of a zincblende crystal, in C/m^2."""
        return float(self.voigt[0, 3])


def to_voigt(e: np.ndarray) -> np.ndarray:
    """``(3, 3, 3)`` -> ``(3, 6)``: ``e_iJ = e_(i)jk``, no factors of two.

    The engineering convention puts the two on the **strain**
    (``eps_4 = 2 eps_23``), so that ``P_i = sum_jk e_(i)jk eps_jk`` is
    ``sum_J e_iJ eps_J`` with the coefficient untouched -- and equivalently
    ``e_iJ = d(sigma_J)/dE_i``, where a stress in Voigt form carries no two
    either. It is :class:`~defumat.response.elastic.ElasticConstants`'
    convention one rank down, and it is the one every table of ``e_14`` is in.
    """
    e = np.asarray(e)
    return np.stack([e[:, i, j] for i, j in VOIGT], axis=1)


#: The last ladder step below which :func:`piezoelectric_tensor` stops warning.
#:
#: **It is a threshold on the step and the error is larger than the step**, which
#: is the one thing to know before reading a silent run as a converged one. On
#: zincblende AlAs, where the ladder and an independent Berry-phase value are
#: both known, ``e_14`` moves 10 per cent from ``4 4 4`` to ``6 6 6``, 2.1 per
#: cent to ``8 8 8`` and 0.44 per cent to ``10 10 10``, while the ``10 10 10``
#: rung is still **1.2 per cent** from the Berry value: the remaining distance
#: runs about three times the last step, because the approach is a slow tail
#: rather than a geometric one. So one per cent here means a few per cent of
#: error on that cell, and it is a place to stop warning rather than a claim of
#: convergence.
KMESH_STEP = 0.01

#: ``e_14`` in C/m^2 on zincblende AlAs at ``4 4 4``, ``6 6 6``, ``8 8 8`` and
#: ``10 10 10``, against a Berry-phase finite difference of -0.6614 to -0.6620
#: on the same cell (``PLAN.md`` P50, Triton ``20337789``). Quoted in the
#: warning because a user with no ladder of their own still needs a number.
KMESH_CURVE = (-0.763786, -0.687475, -0.672897, -0.669907)


def _kmesh_of(calculation):
    """``(nk, grid)`` of the set the response will be integrated over."""
    kpoints = calculation.system.kpoints
    grid = getattr(kpoints, "grid", None)
    return int(kpoints.nk), (None if grid is None else tuple(int(n) for n in grid))


def _warn_about_the_kmesh(calculation, drift) -> None:
    """Say that nothing in this assembly can see its own Brillouin-zone sum.

    **The one guard this quantity had no version of.** Every internal check it
    carries -- the three routes against each other, the symmetry statements, the
    wedge against the closed grid, the ``Z*`` anchor -- is insensitive to the
    k-mesh, because the routes share the field response and the anchor is the
    same assembly in the position coordinate. So a run on a mesh that would be
    ample for a total energy returns a tensor that is thirteen per cent out and
    announces nothing, which is what happened here on the committed ``4 4 4``
    input (``OPEN.md`` Part XIII item 3). A warning is the honest answer rather
    than a refusal: which mesh is enough is a property of the crystal, and the
    ladder that measures it is one call away.
    """
    if drift is not None and drift < KMESH_STEP:
        return
    nk, grid = _kmesh_of(calculation)
    where = f"{nk} k-points" + ("" if grid is None else f" ({grid[0]} {grid[1]} {grid[2]})")
    if drift is None:
        measured = (
            "this run's own k-convergence has not been measured. On zincblende "
            "AlAs, against a Berry-phase finite difference that shares no "
            "machinery with it, e_14 reads "
            + ", ".join(f"{value:.4f}" for value in KMESH_CURVE)
            + " C/m^2 at 4 4 4, 6 6 6, 8 8 8 and 10 10 10 against the Berry "
              "value's -0.6614 to -0.6620, so a committed-quality 4 4 4 mesh "
              "is 13 per cent out there and 8 8 8 is 1.6"
        )
    else:
        measured = (
            f"the last step of the ladder that was run moved the tensor by "
            f"{drift * 100:.1f} per cent, which is above the {KMESH_STEP * 100:.0f} "
            "per cent this stops warning at, and the remaining error is larger "
            "than the step rather than equal to it"
        )
    warnings.warn(
        f"the piezoelectric tensor was integrated over {where} and {measured}. "
        "Nothing inside this quantity can see that: the three routes share one "
        "field response and the Z* anchor is the same assembly in the position "
        "coordinate, so every internal check moves with the mesh instead of "
        "catching it. Measure this crystal's own curve with "
        "defumat.workflows.piezo_ladder.piezoelectric_kmesh_ladder "
        "(Calculator.get_piezoelectric_kmesh_ladder), which reruns the ground "
        "state and the response on a ladder of meshes and puts the last step on "
        "the result; pass kmesh_warning=False to silence this once the mesh is "
        "known to be enough",
        RuntimeWarning,
        stacklevel=3,
    )


# -- the refusals ------------------------------------------------------------


def polar_direction(calculation) -> np.ndarray:
    """The projector onto the directions a spontaneous polarization may point.

    ``(1/N) sum_R R`` over the crystal's point group: a vector survives it only
    if every operation leaves it alone, which is exactly the condition for the
    class to be polar. Zero for ``-43m`` (AlAs) and for any centrosymmetric
    class (silicon), rank one for ``6mm`` (wurtzite), rank three for ``P1``.

    **Taken from the crystal rather than from the run.** A response is often
    computed with ``nosym``, whose symmetry list is the identity alone and
    would call every crystal polar; what decides whether a polarization can
    exist is the crystal, so the operations are searched for here.
    """
    from defumat.system.symmetry import cartesian_rotations, find_symmetries

    cell = calculation.system.cell
    symmetries = find_symmetries(cell, calculation.system.structure)
    return cartesian_rotations(cell, symmetries).mean(axis=0)


def require_a_nonpolar_crystal(calculation) -> None:
    """A polar class needs ``P`` itself, which this package does not have.

    The improper-to-proper correction is ``delta_ki P_j - delta_ij P_k`` and
    every term of it is built from the spontaneous polarization. A Berry-phase
    polarization is not implemented here -- :mod:`defumat.topology` has the
    k-string overlaps it would be built from and no polarization on top of them
    -- so a crystal whose class *permits* a polarization is refused rather than
    reported with a term missing that no symmetry check would catch.
    """
    projector = polar_direction(calculation)
    if np.abs(projector).max() > 1.0e-6:
        directions = np.linalg.matrix_rank(projector, tol=1.0e-6)
        raise NotImplementedError(
            "the piezoelectric tensor of a polar crystal is not implemented: "
            f"this crystal's point group leaves {directions} direction(s) "
            "invariant, so it may carry a spontaneous polarization, and the "
            "proper piezoelectric response then differs from the mixed second "
            "derivative computed here by delta_ki P_j - delta_ij P_k "
            "(Vanderbilt, J. Phys. Chem. Solids 61, 147 (2000)). P is a "
            "Berry-phase quantity and defumat.topology has the overlaps but "
            "no polarization built on them. A non-polar class -- zincblende, "
            "diamond, rocksalt -- has no such term"
        )


#: The Monkhorst-Pack division an **ultrasoft** run needs in each direction.
#:
#: **Where it comes from, and it is one crystal's curve.** The ladder on
#: zincblende AlAs reads ``e_14`` at 15.63, 4.11, 1.26 and 0.57 per cent from an
#: independent Berry-phase value at ``4 4 4``, ``6 6 6``, ``8 8 8`` and
#: ``10 10 10``, against the norm-conserving calibration's 13.33, 3.71, 1.62 and
#: 1.19 on the same meshes. Eight is where the ultrasoft deficit first falls
#: *below* the calibration's, which is the statement that matters here: from
#: there on the dataset is not the largest error, the mesh is, and the mesh has
#: a warning of its own. Below it the two are not separable and the dataset has
#: never been measured, so the run is refused rather than warned about.
#:
#: A division is a coarse instrument on a cell that is not cubic, which is why
#: a **measured** ladder overrides it: ``kmesh_drift`` below :data:`KMESH_STEP`
#: is evidence where a threshold is a guess generalized from one crystal.
ULTRASOFT_MESH = 8


def require_a_measured_dataset(calculation, drift=None,
                               allow_a_coarse_mesh: bool = False) -> None:
    """PAW always, and an ultrasoft dataset below a measured mesh.

    Nothing in the assembly is norm-conserving. The density and ``becsum`` are
    handed to the functional as builders that carry the strain, which is what
    the strain response already needed, and the *displacement* leg of the same
    assembly is validated on all three pseudopotential kinds, being the Born
    charge.

    **One sentence that stood here is false and was measured to be false on
    2026-09-19.** It said that "``qq_ij`` has no cell in it, so the constraint
    stays strain-independent for an ultrasoft dataset exactly as it is for a
    norm-conserving one". ``qq_ij`` indeed has no cell in it and the conclusion
    does not follow: ``S = 1 + sum_ij |beta_i> q_ij <beta_j|`` also carries
    ``vkb``, which is ``beta(|k+G|)`` and moves with the cell like every other
    radial transform, so ``<psi|S|psi>`` deforms under a strain whatever
    ``q_ij`` does. :func:`~defumat.response.strain.overlap_derivatives` is the
    ``jvp`` that says so, and it is *not* zero here:
    :func:`_multiplier_strain_term`, which is that object contracted with the
    multipliers' own response, is **-0.00325 C/m^2** on ``alas-piezo.in`` at 64
    k-points (Triton ``20339308_0``). A term measured at a fifth of the
    disagreement between two routes is not a term that vanishes.

    *All three of those terms were written on 2026-09-19 and the two routes now
    agree to 3.4e-08 on a wedge and 1.0e-07 at 64 k-points; what the paragraph
    below describes is what the refusal was protecting against until then, and
    the refusal stays for the separate reason the last paragraph gives.*

    **Two things about PAW, and both are reasons this refusal is not a
    formality.** :func:`piezoelectric_zstar_eu_style` refuses it outright
    (:func:`require_a_norm_conserving_transcription`), and in the taped route
    :func:`~defumat.response.born._full_zone_becsum_response` is now **live in
    the strain coordinate and exercised by nothing**: the wedge completion runs
    for PAW as well as for ultrasoft, and no non-centrosymmetric PAW cell is
    committed, so the one dataset kind whose ``becsum`` reaches the energy
    outside the dense grid has neither of its two new paths under a test. That is
    a hole in the coverage rather than a known defect, and it is written here
    because this refusal is what stands between it and a user.

    **The consequence is larger than the sentence.** If ``S`` deforms then this
    assembly, carrying only the states tangent -- no multipliers, no
    ``add_for_charges`` sandwich, no full-zone shift -- was missing on an
    ultrasoft dataset exactly what
    :func:`~defumat.response.born.born_effective_charges` supplies in the
    position coordinate, and :func:`born_charges_from_stress_route`'s docstring
    has said as much all along: run in the position coordinate this assembly is
    ``Z*`` *minus the constraint term an ultrasoft dataset adds*. So the
    refusal below is not a formality waiting on a measurement; **both** routes
    to this tensor are incomplete for an augmented dataset, and the one
    reference that is not is the Berry-phase finite difference.

    **What is missing is a measurement, and what stood here named the wrong
    obstacle.** The claim was that "every ultrasoft and PAW dataset committed
    here belongs to a centrosymmetric crystal", and it was already false on the
    day it was written: ``Al.pbe-n-rrkjus_psl.1.0.0.UPF`` and
    ``As.pbe-n-rrkjus_psl.1.0.0.UPF`` are committed and
    ``tests/data/qe/alas-magnetoelectric-nosoc.in`` already builds zincblende
    from them, which is exactly the non-polar, non-centrosymmetric class
    :func:`require_a_nonpolar_crystal` handles. A reader who believed the
    sentence would have gone looking for a pseudopotential; the case exists and
    ``tests/data/qe/alas-piezo.in`` is now the nonmagnetic version of it.

    **What blocks it is a missing measurement, not a missing term.** An earlier
    form of this docstring, and of the message below, said that the strain leg
    goes through :func:`~defumat.response.strain.strain_response` and that
    *that* refuses ultrasoft and PAW because ``Q_ij(r)`` is a function of the
    cell. It does not: P41 lifted that refusal, ``overlap_derivatives`` and
    ``density_of_strained_states`` are the term, and
    ``test_electrostriction.py`` pins them against a central difference of the
    converged density at 4.6e-4 (ultrasoft) and 4.7e-4 (PAW) beside a
    norm-conserving control at 1.9e-4. Sending a reader to write a term that has
    existed since P41 is wasted work, which is why the cause is corrected here
    while the refusal stays.

    **What stays true is that the quantity has never been measured on an
    augmented dataset.** Whether *this* assembly, a ``jvp`` of the stress rather
    than of the density, needs anything beyond what ``strain_response`` already
    carries is a separate question, and a plausible argument about the strain
    coordinate is exactly what P44 falsified by measurement on the third
    derivative: two of its ingredients transferred, the residue did not, and it
    was localised only because it could be measured.

    The one attempt so far does not count (Triton `20336374`, 2026-09-19):
    ``tests/data/qe/alas-piezo.in`` gave ``e_14 = +0.815929`` against a
    Berry-phase finite difference's ``+0.687757``, 15.7 per cent, but the same
    comparison on the norm-conserving calibration cell is 13.4 per cent out at
    the same ``4 4 4`` mesh and falls to 1.6 per cent at ``8 8 8``, so that
    number measures this quantity's own k-convergence and not the dataset (see
    ``PLAN.md`` P50). Lifting the refusal is the ultrasoft tensor at a converged
    mesh against a Berry-phase value on the same cell, and the cell is
    committed.
    """
    if not calculation.is_ultrasoft:
        return
    if calculation.is_paw:
        raise NotImplementedError(
            "the piezoelectric tensor is not implemented for a PAW dataset. "
            "The ultrasoft half of this refusal was lifted on 2026-09-20 by a "
            "ladder against a Berry-phase finite difference on zincblende AlAs "
            "-- 0.57 per cent at 10 10 10, against the norm-conserving "
            "calibration's 1.19 on the same mesh -- and no such comparison "
            "exists for PAW: no non-centrosymmetric PAW crystal has been run "
            "against an independent reference at all. What is measured for PAW "
            "is internal only, the wedge completion's one-centre half against "
            "its own closed grid (1.58e-06, a factor of 440 over the same run "
            "with the completion off, tests/regression/test_piezoelectric_paw.py). "
            "An internal identity is not a reference. Use an ultrasoft or "
            "norm-conserving dataset"
        )
    if allow_a_coarse_mesh or (drift is not None and drift < KMESH_STEP):
        return
    _, grid = _kmesh_of(calculation)
    if grid is not None and min(grid) >= ULTRASOFT_MESH:
        return
    where = "an explicit k-point list" if grid is None else (
        f"a {grid[0]} {grid[1]} {grid[2]} grid"
    )
    raise NotImplementedError(
        f"the piezoelectric tensor of an ultrasoft dataset needs a denser "
        f"k-mesh than {where}: the dataset was measured against a Berry-phase "
        f"finite difference only at and above {ULTRASOFT_MESH} divisions in "
        "each direction, where its disagreement (1.26 per cent at 8 8 8, 0.57 "
        "at 10 10 10 on zincblende AlAs) is smaller than the norm-conserving "
        "calibration's own on the same mesh. Below that the two are not "
        "separable: at 4 4 4 the same cell reads 15.6 per cent out, almost all "
        "of it k-convergence, which is why a coarse ultrasoft run is refused "
        "here rather than warned about. Raise the mesh, or measure this "
        "crystal's own curve with "
        "defumat.workflows.piezo_ladder.piezoelectric_kmesh_ladder and pass its "
        "drift as kmesh_drift, which is evidence where the division count is a "
        "threshold taken from one cubic crystal"
    )


@dataclass(frozen=True)
class _Regime:
    """The regime a ``System`` will run in, read off it and its pseudopotentials.

    **What the refusals below read, and nothing else**, so that
    :func:`require_a_piezoelectric_tensor` can answer from the input rather
    than from a built :class:`~defumat.scf.driver.Calculation`. Every guard in
    the chain duck-types its argument by attribute -- the spin regime, the
    occupations and the k-set off ``system``, the symmetry off ``symmetries``,
    which kind of dataset off ``is_ultrasoft``/``is_paw`` -- and none of them
    reads the G sphere, either FFT grid, ``vkb`` or ``Q_ij(G)``, which are what
    the constructor spends its time and memory on.

    **Each field is the expression ``Calculation.__init__`` computes it with**,
    so that the two cannot disagree on any ``System`` a calculation can be
    built from. Three of them are not the builder called and tested for
    ``None``, and each is that builder's own early return transcribed: ``is_ultrasoft``
    is :func:`~defumat.pseudo.augmentation.build_augmentation`'s, which runs
    over the *atoms'* species, and ``is_paw`` is
    :func:`~defumat.paw.onecenter.build_paw`'s, which runs over every species
    given -- the two are written differently there and are not unified here.
    ``magnetic_field`` is ``_build_magnetic_field``'s early return and carries
    only **whether** there is a field or a constrained moment (``True``) or
    not (``None``): the rest of that method builds integration spheres on the
    dense grid, and every guard asks ``is not None`` and nothing more.

    ``system`` is the input as given, **without** the ``K_POINTS gamma``
    substitution the constructor makes when it cannot consume the half-sphere
    storage. That substitution is ``dataclasses.replace(kpoints,
    gamma_only=False)`` (:func:`~defumat.scf.driver._without_gamma_storage`),
    so the shift, the grid and the count -- everything a guard here reads off
    the set -- are the same either way, and making it would only issue the
    warning the calculation built afterwards issues again. ``gamma_only``
    itself is the constructor's own predicate.
    """

    system: object
    symmetries: object
    functional: object
    nspin: int
    noncolin: bool
    spiral: bool
    gamma_only: bool
    two_fermi_energies: bool
    is_ultrasoft: bool
    is_paw: bool
    is_hubbard: bool
    magnetic_field: object

    @classmethod
    def of(cls, system, pseudos) -> "_Regime":
        """Build it the way ``Calculation.__init__`` builds each attribute."""
        from defumat.hubbard.manifold import build_hubbard_setup
        from defumat.scf.driver import gamma_storage_is_consumable
        from defumat.xc.functional import resolve_functional

        pseudos = tuple(pseudos)
        nspin = int(system.nspin)
        noncolin = bool(system.noncolin)
        field = not (
            system.constrained_magnetization == "none"
            and not np.asarray(system.b_field, dtype=float).any()
            and not system.atomic_b_field
        )
        return cls(
            system=system,
            symmetries=system.symmetry_group(),
            functional=resolve_functional(
                [pseudo.functional for pseudo in pseudos], system.input_dft
            ).with_meta_coefficient(getattr(system, "mbj_c", None)),
            nspin=nspin,
            noncolin=noncolin,
            spiral=bool(system.spiral),
            gamma_only=gamma_storage_is_consumable(system, pseudos),
            two_fermi_energies=(
                nspin == 2 and system.tot_magnetization is not None
            ),
            is_ultrasoft=any(
                pseudos[t].is_ultrasoft for t in system.structure.types
            ),
            is_paw=any(pseudo.is_paw for pseudo in pseudos),
            # The builder rather than ``system.hubbard is not None``: a card
            # whose parameters are all zero resolves to ``None`` there, and
            # that is the case where the two would disagree.
            is_hubbard=build_hubbard_setup(
                system.hubbard, system.structure, pseudos, noncolin=noncolin
            ) is not None,
            magnetic_field=True if field else None,
        )


def require_a_piezoelectric_tensor(calculation, drift=None,
                                   allow_a_coarse_mesh: bool = False,
                                   *, pseudos=None) -> None:
    """Everything that makes the mixed derivative above not be the answer.

    ``calculation`` is a :class:`~defumat.scf.driver.Calculation`, or a
    :class:`~defumat.system.builder.System` when ``pseudos`` is given, and
    **the second form is the one to refuse with.** Nothing in this chain reads
    what a calculation builds; it reads the spin regime, the occupations, the
    k-set, the symmetry and which kind of dataset each species is, and all of
    those are properties of the input (:class:`_Regime`). A refusal that had to
    allocate the calculation it was refusing was a refusal at the wrong
    boundary (``OPEN.md`` Part III X3): a caller holding only the input --
    :meth:`~defumat.calculator.Calculator.get_piezoelectric_tensor`,
    :func:`~defumat.workflows.piezo_ladder.piezoelectric_kmesh_ladder` -- now
    refuses before it runs a ground state rather than after one.

    ``pseudos`` is keyword-only because ``drift`` is the second positional
    argument (:func:`piezoelectric_tensor` passes it that way), and a tuple of
    pseudopotentials slid into its place would not fail.

    **What the ``System`` form does not do is the constructor's own
    refusals** -- a fully-relativistic dataset without ``lspinorb``, a spiral
    with symmetry, a functional with no spin-polarized form, a field on an
    ``nspin = 1`` run. Those are still met when the calculation is built, so
    for a ``System`` no calculation can be built from, this may name a
    refusal of its own where the constructor would have named another; for
    every other ``System`` the two forms raise the same one.
    """
    if pseudos is not None:
        if hasattr(calculation, "system"):
            raise TypeError(
                "pass a Calculation, or a System together with its "
                "pseudopotentials, and not both: a Calculation already "
                "carries the pseudopotentials it was built with"
            )
        calculation = _Regime.of(calculation, pseudos)
    elif not hasattr(calculation, "system"):
        raise TypeError(
            "a System needs its pseudopotentials to be refused on (pass "
            "pseudos=...): whether the dataset is ultrasoft or PAW, and "
            "which functional runs, are read off them"
        )
    require_a_symmetrisable_response(calculation)
    # Bare, not ``metals=True``/``spin_polarized=True``: the *solve* runs for a
    # metal and for two spin channels, and this assembly on top of it has been
    # run with neither.
    require_a_sternheimer_regime(calculation)
    require_a_differentiable_cell(calculation)
    require_a_measured_dataset(calculation, drift,
                               allow_a_coarse_mesh=allow_a_coarse_mesh)
    require_a_nonpolar_crystal(calculation)


# -- the assembly ------------------------------------------------------------


def _frozen_energy_of(calculation, psi, eigenvalues, weights, density, becsum):
    """``E(coordinate, states)`` with the mixed state a *function* of both.

    The density and ``becsum`` are handed over as builders rather than as
    arrays, and :mod:`defumat.response.born` gives the two reasons in full:
    the SCF's *scalar* symmetrisation must not stand inside a chain rule whose
    tangent is a response, and for an ultrasoft dataset the density itself
    carries the coordinate. Under a strain there is a third reason of the same
    kind -- the density is stored on a grid that does not move and carries a
    factor ``1/Omega``, so it responds to a strain even at frozen states.

    **This is the norm-conserving assembly's energy**, with no multipliers and no
    wedge completion; :func:`_shifted_frozen_energy_of` is the augmented one.
    It carried an optional ``multipliers`` argument until the augmented branch
    needed two more and moved to that function, at which point nothing passed it
    any more.
    """
    positions = jnp.asarray(calculation.system.structure.positions)
    density_of, becsum_of = _raw_mixed_state(
        calculation, positions, psi, weights, density, becsum
    )

    def energy(moved, states):
        return energy_at(
            moved,
            FrozenState(
                wavefunctions=states, weights=weights, eigenvalues=eigenvalues
            ),
            density=density_of, becsum=becsum_of,
        )

    return energy


def _shifted_frozen_energy_of(calculation, psi, eigenvalues, weights, density,
                              becsum):
    """:func:`_frozen_energy_of` with the wedge completion as two more arguments.

    Returns ``(energy, density_of, becsum_of)``: the energy takes
    ``(moved, states, multipliers, shift, becsum_shift)``, and the two shifts
    are added to the builders' output. **Their primal is zero and only their
    tangent is used**, so the value of the density is untouched and what moves
    is the *derivative* -- which is the whole construction, since what has to be
    the full-zone object is a factor inside a term quadratic in a per-k tangent
    rather than the density itself.

    The builders come back because
    :func:`~defumat.response.born._full_zone_field_response` needs them to build
    the shifts in the first place. This is
    :func:`~defumat.response.born.born_effective_charges`' arrangement one
    coordinate over, and the two are deliberately the same shape: the term being
    completed is the same one.
    """
    positions = jnp.asarray(calculation.system.structure.positions)
    density_of, becsum_of = _raw_mixed_state(
        calculation, positions, psi, weights, density, becsum
    )

    def energy(moved, states, multipliers, shift, becsum_shift):
        def shifted_becsum(inner, inner_states, occupations):
            return tuple(
                None if part is None else part + offset
                for part, offset in zip(
                    becsum_of(inner, inner_states, occupations), becsum_shift
                )
            )

        return energy_at(
            moved,
            FrozenState(
                wavefunctions=states, weights=weights, eigenvalues=eigenvalues
            ),
            density=lambda inner, inner_states, occupations, parts: (
                density_of(inner, inner_states, occupations, parts) + shift
            ),
            becsum=shifted_becsum, multipliers=multipliers,
        )

    return energy, density_of, becsum_of


def _field_column(gradient, coordinate, psi, dpsi, nocc):
    """One ``jvp`` of a coordinate gradient along one field response.

    The tangent is the field's first-order wavefunction in the occupied block
    and zero in the coordinate, so what comes back is
    ``d/dE_k [dE/d(coordinate)]`` and nothing else. JAX's ``jvp`` of a
    real-valued function of complex primals is the real-linear tangent map,
    which is where the ``+ c.c.`` of the hand-derived expression comes from.

    **This is the states tangent alone, and that is the whole derivative only
    for a norm-conserving dataset.** An augmented one moves three more things
    with the coordinate -- the multipliers, the constraint's own sandwich and
    the wedge completion of the density -- and
    :func:`clamped_ion_piezoelectric` carries its own loop for them rather than
    growing three more optional arguments here. It used to grow two, and the
    branch went unreachable the moment the third arrived, which is the reason it
    is not four now.
    """
    states = jnp.zeros_like(psi).at[:, :, :nocc].set(dpsi)
    _, column = jax.jvp(
        gradient, (coordinate, psi), (jnp.zeros_like(coordinate), states)
    )
    return np.asarray(column)


def constraint_strain_term(calculation, solver, weights, commutator) -> np.ndarray:
    """``sum_n w_n <psi_n| dS/d(eps_ab) | P_c r_k psi_n>`` -- ``(3, 3)`` complex.

    :func:`~defumat.response.born.constraint_position_term`'s sandwich with
    :meth:`~defumat.scf.driver.Calculation.at_strain` where it has
    ``at_positions``, and it is here for the same reason it is there:
    ``dLambda_mn = w_n <psi_m|X|psi_n>`` wants the occupied-occupied block of
    the position operator, and ``<psi_m|r|psi_n>`` is the Berry connection,
    gauge-dependent and not a matrix element of any operator in a periodic
    cell. :func:`~defumat.response.born._multiplier_response` therefore reaches
    only the part ``P_c^+ r|psi>`` carries; what is left is finite only in the
    combination it appears in, contracted with ``<psi_n|dS/d(eps)|psi_m>``.
    ``add_for_charges.f90`` is that combination and this is it in the strain
    coordinate.

    **It is worth 0.55 on ultrasoft silicon in the position coordinate**, the
    difference between +0.47 and -0.079, so it is not a correction to be left
    for later; and it is identically zero for a norm-conserving dataset, where
    ``S = 1`` and does not deform.

    ``commutator`` is ``P_c r|psi>`` **before** ``S`` and before
    ``adddvepsi_us``, which is what ``dielectric_tensor`` keeps in
    ``internals["commutators"]`` for exactly this use.
    """
    if not calculation.is_ultrasoft:
        return np.zeros((3, 3))
    from defumat.response.strain import strain_tangent

    occupied = solver.psi
    nocc = solver.nocc
    batch = calculation.k_batch
    noncolin = bool(calculation.noncolin)

    def sandwich(strain):
        moved = calculation.at_strain(strain)
        vkb = moved.projectors.vkb
        npwx = vkb.shape[1]
        qq = jnp.asarray(
            moved.qq_so if noncolin else moved.projectors.qq
        ).astype(vkb.dtype)
        total = jnp.zeros((), dtype=vkb.dtype)
        for spin in range(occupied.shape[0]):
            def one_k(ik, spin=spin):
                if noncolin:
                    shape = occupied[spin][ik].shape[:-1] + (2, npwx)
                    left = jnp.einsum(
                        "gc,nag->nac", vkb[ik].conj(),
                        occupied[spin][ik].reshape(shape),
                    )
                    right = jnp.einsum(
                        "gc,nag->nac", vkb[ik].conj(),
                        commutator[spin][ik].reshape(shape),
                    )
                    return jnp.einsum("nai,abij,nbj->n", left.conj(), qq, right)
                left = jnp.einsum("gc,ng->nc", vkb[ik].conj(), occupied[spin][ik])
                right = jnp.einsum("gc,ng->nc", vkb[ik].conj(), commutator[spin][ik])
                return jnp.einsum("ni,ij,nj->n", left.conj(), qq, right)

            values = map_k(one_k, jnp.arange(occupied.shape[1]), batch=batch)
            total = total + jnp.sum(weights[spin][:, :nocc] * values)
        return total

    zero = jnp.zeros((3, 3))
    out = np.zeros((3, 3), dtype=complex)
    for a in range(3):
        for b in range(a, 3):
            _, derivative = jax.jvp(sandwich, (zero,), (strain_tangent(a, b),))
            out[a, b] = out[b, a] = complex(derivative)
    return out


def clamped_ion_piezoelectric(
    calculation, psi, eigenvalues, weights, density, becsum, dpsi, nocc,
    solver=None, field_perturbations=None, commutators=None,
    full_zone: bool = True,
) -> np.ndarray:
    """``(3, 3, 3)`` in ``e/bohr^2``: ``e[k, i, j]``, symmetrised.

    Args:
        calculation: the one the states belong to, at its own cell.
        psi: ``(nspin, nk, nbnd, ndim)``, all bands.
        eigenvalues, weights: ``(nspin, nk, nbnd)``.
        density, becsum: the converged mixed state.
        dpsi: three ``(nspin, nk, nocc, ndim)`` field responses.
        nocc: how many bands they cover -- the solver's own ``nocc``, which is
            one number across the spin channels.
        solver, field_perturbations, commutators: what an **augmented** dataset
            needs and a norm-conserving one has no use for. Given all three,
            the derivative gains three terms that do not exist when ``S`` is the
            identity and the density does not move with the cell at frozen
            states: the multipliers' own first-order change, carried as a third
            tangent the way
            :func:`~defumat.response.born.born_effective_charges` carries it;
            :func:`constraint_strain_term`, which is ``add_for_charges.f90``;
            and :func:`~defumat.response.born._full_zone_field_response`, which
            completes the one factor of this derivative that is **quadratic** in
            a per-k tangent and which no average of the finished tensor can
            repair. Omit them and the assembly is the states-tangent-only one it
            has always been, which is complete for a norm-conserving dataset and
            short of all three otherwise.
        full_zone: keep the third of those terms. **It exists to be switched
            off in a test and for nothing else.** The completion is identically
            zero on a run with no symmetry, so the only way to see that it is
            doing anything is to run a wedge with it and without it, and a
            wedge test that cannot fail is the trap ``CLAUDE.md`` calls a check
            whose null result cannot be told from a pass. Off, the wedge and the
            closed grid disagree by a thousand times more.

    **Why the augmented branch is skipped exactly and not approximately.** The
    first two extra terms are contracted with ``dS/d(eps)``, and ``S`` is the
    identity for a norm-conserving dataset; the third multiplies the
    frozen-state density response, which at frozen plane-wave coefficients is
    ``-delta_ab rho`` and therefore zero for every traceless strain. All of them
    are identically zero on a norm-conserving shear rather than small -- which
    is why this assembly agreed with the transcribed one to 6.2e-15 on a
    norm-conserving cell while both were missing all three. The cost of
    switching them on is real: the matrix-multiplier constraint is an
    ``nbnd x nbnd`` Gram per k-point where the diagonal form is a vector, which
    is why :func:`~defumat.forces.energy._constraint_energy` exists separately
    and why a force, a stress and a ``Gamma`` phonon do not pay for it.
    """
    energy = _frozen_energy_of(
        calculation, psi, eigenvalues, weights, density, becsum
    )
    zero = jnp.zeros((3, 3))
    volume = calculation.system.cell.volume

    # **The augmented branch, and it is skipped exactly rather than
    # approximately for a norm-conserving dataset.** Both extra terms are
    # contracted with ``dS/d(eps)``, which is identically zero when ``S`` is the
    # identity, so not paying for them there is a statement about the physics
    # and not a tolerance -- and ``_constraint_energy``'s matrix form is an
    # ``nbnd x nbnd`` Gram per k-point where the diagonal one is a vector.
    augmented = (
        calculation.is_ultrasoft
        and solver is not None
        and field_perturbations is not None
        and commutators is not None
    )
    if not augmented:
        gradient = jax.grad(
            lambda strain, states: energy(calculation.at_strain(strain), states),
            argnums=0,
        )
        tensor = np.stack([
            -_field_column(gradient, zero, psi, dpsi[axis], nocc) / volume
            for axis in range(3)
        ])
        return calculation.symmetrize_cartesian_tensor(tensor)

    from defumat.response.born import (
        _full_zone_field_response, _ground_state_multipliers,
        _multiplier_response,
    )

    # ``_frozen_energy_of`` again, with the wedge completion as two further
    # arguments whose primal is zero: see the comment on ``shifts`` below.
    energy, density_of, becsum_of = _shifted_frozen_energy_of(
        calculation, psi, eigenvalues, weights, density, becsum
    )
    positions = jnp.asarray(calculation.system.structure.positions)
    states_by_axis = [
        jnp.zeros_like(psi).at[:, :, :nocc].set(dpsi[axis]) for axis in range(3)
    ]
    # **The one term of this assembly that is quadratic in a per-k tangent**, and
    # therefore the one a rank-3 average of the finished tensor cannot complete:
    # on an augmented dataset the density moves with the coordinate at frozen
    # states, so the mixed derivative carries ``int (drho/d(eps)) K (drho/dE)``
    # and a wedge sum of a product is not the product of the full-zone objects.
    # P36's rule is that *one* factor has to be made whole and the other left as
    # the raw wedge sum; the field response is the one made whole, because an
    # induced charge density is a polar vector field and ``symdvscf``'s average
    # is already written for it. It is identically zero on a run with no
    # symmetry, which is what makes the wedge and the closed grid a real check of
    # each other. Measured on the committed tiny cell: without it the wedge gives
    # 1.475427270 C/m^2 against the closed grid's 1.474377366, **1.05e-03**,
    # where the contracted route -- which needs none of this, its own screening
    # factor being full-zone already -- splits by 4.8e-06.
    shifts, becsum_shifts = _full_zone_field_response(
        calculation, positions, psi, weights, density_of, becsum_of,
        states_by_axis,
    )
    if not full_zone:
        # The falsifier, and it is a *tangent* that is zeroed rather than a term
        # deleted: everything else about the assembly, including the two
        # constraint terms, stays exactly where it is, so what the pair measures
        # is this completion and not a different calculation.
        shifts = tuple(jnp.zeros_like(shift) for shift in shifts)
        becsum_shifts = tuple(
            tuple(None if offset is None else jnp.zeros_like(offset)
                  for offset in per_axis)
            for per_axis in becsum_shifts
        )

    gradient = jax.grad(
        lambda strain, states, mult, shift, becsum_shift: energy(
            calculation.at_strain(strain), states, mult, shift, becsum_shift
        ),
        argnums=0,
    )
    ground = _ground_state_multipliers(weights, eigenvalues, psi.dtype)
    unshifted = jnp.zeros_like(shifts[0])
    no_becsum_shift = tuple(
        None if offset is None else jnp.zeros_like(offset)
        for offset in becsum_shifts[0]
    )
    columns = []
    for axis in range(3):
        multipliers = _multiplier_response(
            solver, field_perturbations[axis], weights, psi.shape[2], nocc
        )
        _, column = jax.jvp(
            gradient, (zero, psi, ground, unshifted, no_becsum_shift),
            (jnp.zeros_like(zero), states_by_axis[axis], multipliers,
             shifts[axis], becsum_shifts[axis]),
        )
        column = np.asarray(column)
        # ``born_effective_charges`` subtracts this beside its own column, for
        # the reason :func:`constraint_strain_term` gives: it is the half of
        # ``dLambda`` that the occupied-occupied block cannot carry.
        column = column + np.real(
            constraint_strain_term(calculation, solver, weights, commutators[axis])
        )
        columns.append(-column / volume)
    tensor = np.stack(columns)
    # ``symmatrix3``: a wedge sum is exact for a scalar and not for a rank-3
    # tensor. Everything left in this assembly by the time the average is taken
    # *is* linear in the response, which is why the average is the whole of the
    # completion -- the one term that was not is the shift above, and it is
    # completed before it enters rather than after.
    return calculation.symmetrize_cartesian_tensor(tensor)


def born_charges_from_stress_route(
    calculation, psi, eigenvalues, weights, density, becsum, dpsi, nocc,
) -> np.ndarray:
    """The same ``jvp``, with the atoms as the coordinate: ``(nat, 3, 3)``.

    **This exists to be compared, not to be used.** Run in the position
    coordinate, :func:`clamped_ion_piezoelectric`'s assembly is the *electronic*
    half of a Born effective charge -- ``Z*`` minus its bare ionic term and
    minus the constraint term an ultrasoft dataset adds -- so for a
    norm-conserving crystal ``Z_a delta_ij - this`` is
    :func:`~defumat.response.born.born_effective_charges` exactly. Since that
    number is validated against ``ph.x`` to every digit it prints, the
    comparison fixes the sign, the field's normalisation and the volume factor
    of the piezoelectric tensor, none of which any symmetry check would catch.

    Returned unsymmetrised and indexed ``[a, k, j]``: field along ``k``, atom
    ``a`` displaced along ``j``.
    """
    energy = _frozen_energy_of(
        calculation, psi, eigenvalues, weights, density, becsum
    )
    positions = jnp.asarray(calculation.system.structure.positions)
    gradient = jax.grad(
        lambda pos, states: energy(calculation.at_positions(pos), states),
        argnums=0,
    )
    return np.stack([
        _field_column(gradient, positions, psi, dpsi[axis], nocc)
        for axis in range(3)
    ], axis=1)


def require_a_norm_conserving_transcription(calculation) -> None:
    """:func:`piezoelectric_zstar_eu_style` refuses **PAW**, and only PAW.

    **This is a separate refusal from :func:`require_a_measured_dataset` and it
    has to be**, because the two are about different things: that one says the
    *quantity* has never been measured against an independent reference on an
    augmented dataset, and this one says whether this particular assembly is the
    one to measure it with. Lifting the first must not lift the second.

    **Ultrasoft was refused here until 2026-09-19 and is not any more, because
    the two terms it was missing were found and both are in.**
    ``zstar_eu.f90`` line 90 is ``if (okvan) call zstar_eu_us``, three hundred
    further lines this transcription does not have, and what those lines are
    worth here turned out to be two contractions rather than three hundred
    lines, for the reason :mod:`defumat.response.born` gives one coordinate
    over: **only one leg of this derivative moves ``S``**, so the case has the
    Born charge's shape and not a phonon's. They are
    :func:`_multiplier_strain_term`, the multipliers' own response against
    ``<psi_m|dS/d(eps)|psi_n>`` together with the ``add_for_charges`` sandwich
    beside it, and :func:`_screened_strain_term`, which is not about the
    constraint at all but about what "bare" means when the coordinate is a
    strain -- ``_bare_strains`` freezes the density *array*, and the density is
    a function of the cell as well as of the states.

    **The measurement, and the cell it was taken on is the point.** The two
    routes are made to agree on ``tests/data/qe/alas-piezo-tiny.in``, ultrasoft
    AlAs at 8 k-points -- a cell chosen for cost and not for physics, which is
    legitimate because **two assemblies of the same mixed second derivative must
    agree at any cutoff and on any mesh**, so their disagreement is a defect and
    not a convergence question. Two rungs of it were run and **each number below
    belongs to one of them**, the committed ``ecutwfc = 10, ecutrho = 44`` and
    the ``12 / 96`` the identity was first read at:

    ===========  ============  ===========  ==========  ==========
    rung         gap before    the term     gap after   peak
    ===========  ============  ===========  ==========  ==========
    ``12 / 96``  -0.023777619  -0.023777621  2.6e-09    18.9 GiB
    ``10 / 44``  (not taken)   -0.022727213  1.7e-07    10.1 GiB
    ===========  ============  ===========  ==========  ==========

    on values of 1.473304 and 1.474377, so the defect is 1.61 and 1.54 per cent
    and the cutoff moves neither it nor ``e_14``. What is left in each case is
    the Sternheimer solve's own threshold rather than round-off, since the two
    routes contract differently-converged intermediates. The same two terms were
    measured together on ``alas-piezo.in`` at 64 k-points (Triton
    ``20339831``), where the gap was 1.8 per cent.

    **What the norm-conserving agreement was worth, which is less than it
    looked.** The two routes agree to ``6.2e-15`` on ``alas-raman.in`` and did
    so while both were missing all of this, because every missing term is
    contracted either with ``dS/d(eps)``, identically zero when ``S`` is the
    identity, or with the frozen-state density response, which for a traceless
    strain at frozen plane-wave coefficients is identically zero as well. A
    zincblende crystal's only independent component is the shear ``e_14``, so
    the calibration cell could not have seen any of it. That agreement is still
    the check that the new terms cannot *break* a norm-conserving answer, and
    it is nothing more.

    **PAW is refused and is not an inherited refusal.**
    :func:`_screened_strain_term` integrates the field's induced potential
    against the frozen-state density response on the dense grid, and for an
    ultrasoft dataset that is the whole coupling, because ``becsum`` reaches the
    energy only through the augmentation charge that is on that grid. PAW adds a
    one-centre energy that is a function of ``becsum`` directly, so its cross
    term with the field's ``dbecsum`` is on no grid at all and is not in this
    route. :func:`clamped_ion_piezoelectric` has it because it differentiates
    the energy rather than an operator, which is why the default stays there.
    """
    if calculation.is_paw:
        raise NotImplementedError(
            "method='zstar_eu' refuses a PAW dataset: the term that screens "
            "the frozen-state density response is integrated on the dense "
            "grid, which is the whole coupling for ultrasoft and not for PAW, "
            "whose one-centre energy is a function of becsum directly and "
            "whose cross term with the field's dbecsum is on no grid. "
            "Ultrasoft runs here and agrees with method='autodiff' to 2.6e-09 "
            "C/m^2 on alas-piezo-tiny.in at ecutwfc = 12. Use "
            "method='autodiff', which "
            "differentiates the energy and therefore carries the one-centre "
            "term too"
        )


def piezoelectric_zstar_eu_style(
    calculation, solver, density, dpsi,
    field_perturbations=None, band_weights=None, nocc=None, commutators=None,
    field_dvscf=None,
) -> np.ndarray:
    """``zstar_eu.f90``'s contraction with the strain in place of the atom.

    **Ultrasoft runs here and PAW is refused** --
    :func:`require_a_norm_conserving_transcription` is the guard and carries
    the measurement. ``zstar_eu.f90`` line 90 is ``if (okvan) call
    zstar_eu_us``, and what those three hundred further lines are worth in this
    coordinate is the two contractions below, :func:`_multiplier_strain_term`
    and :func:`_screened_strain_term`, because only one leg of this derivative
    moves ``S``. With both in, the two routes agree to **2.6e-09** C/m^2 on
    ultrasoft AlAs at ``ecutwfc = 12`` where they were 1.6 per cent apart, and
    to 1.7e-07 at the ``ecutwfc = 10`` the test cell is committed at.

    **The transcribed expression put beside the differentiated one**, in this
    project's usual arrangement. QE writes a Born charge as a bare perturbation
    against the field's self-consistent response,

        Z*_(a)ij = Z_a delta_ij - 2 sum_n w_n Re <dpsi^(E_i)_n | dV/du_(a)j psi_n>

    (:func:`~defumat.response.efield.born_charges_zstar_eu`), and a
    piezoelectric constant is the same object with ``d/d(eps_ab)`` where that
    has ``d/du_(a)j``::

        e_(k)ab = -(2/Omega) sum_n w_n Re <dpsi^(E_k)_n | dH/d(eps_ab) psi_n>

    with no ionic term, because the frozen polarization's own strain derivative
    is ``delta_ki Omega P_j`` and vanishes for the non-polar crystals this is
    allowed on. The bare strain perturbation is
    :func:`~defumat.response.strain._bare_strains` -- the same ``jvp`` through
    :meth:`~defumat.scf.driver.Calculation.at_strain` P26 drives its response
    with -- so **this route needs no strain response at all**: it is three
    ``jvp`` calls of ``H|psi>`` and a contraction, and it shares with
    :func:`clamped_ion_piezoelectric` only the field response both consume.

    **The factor is 2 and not 4, and the difference is Rydberg's ``e^2``.**
    :func:`~defumat.response.efield._assemble` builds ``epsilon`` from the same
    field response with a 4 in front, because a *susceptibility* is a
    Coulomb-normalised quantity and carries ``e^2 = 2`` where a bare mixed
    second derivative does not -- and the piezoelectric constant is a mixed
    derivative, in units of ``e/bohr^2``, exactly as the Born charge it copies
    is in units of ``e``. Taking the 4 gives a tensor that is right in every
    symmetry and twice too large, which no symmetry check sees; what says so is
    that ``zstar_eu.f90``'s own constant, on the leg that is validated against
    ``ph.x``, is 2.
    """
    from defumat.response.strain import _bare_strains

    volume = calculation.system.cell.volume
    weights = solver.weights
    bare = _bare_strains(calculation, solver, density)
    tensor = np.zeros((3, 3, 3))
    for k in range(3):
        for a in range(3):
            for b in range(a, 3):
                overlap = jnp.einsum(
                    "skng,skng->skn", jnp.conj(dpsi[k]), bare[a, b]
                )
                value = -2.0 * float(
                    jnp.sum(weights * jnp.real(overlap))
                ) / volume
                tensor[k, a, b] = tensor[k, b, a] = value
    constraint = _multiplier_strain_term(
        calculation, solver, field_perturbations, band_weights, nocc
    )
    if constraint is not None:
        tensor = tensor + constraint
    # The screening of the frozen-state density response, which is what
    # ``_bare_strains``' frozen density *array* leaves out and the taped route
    # generates through its density builder. Zero for a shear on a
    # norm-conserving dataset, and it is the whole of the two routes' remaining
    # disagreement on an augmented one (:func:`_screened_strain_term`).
    screened = _screened_strain_term(calculation, solver, field_dvscf)
    if screened is not None:
        tensor = tensor + screened
    # The other half of the same ``dLambda``, which the occupied-occupied block
    # cannot carry -- ``add_for_charges.f90``. Zero for a norm-conserving
    # dataset, and in the position coordinate it is the larger of the two by an
    # order of magnitude, so a route with one and not the other is worse placed
    # than a route with neither.
    if commutators is not None and calculation.is_ultrasoft:
        volume = calculation.system.cell.volume
        for k in range(3):
            tensor[k] = tensor[k] - np.real(constraint_strain_term(
                calculation, solver, band_weights, commutators[k]
            )) / volume
    return calculation.symmetrize_cartesian_tensor(tensor)


def _screened_strain_term(calculation, solver, field_dvscf) -> np.ndarray | None:
    """``(3, 3, 3)``: the screening of the frozen-state density response.

    **This is the second thing ``zstar_eu.f90`` hands to ``zstar_eu_us.f90``
    for**, and unlike :func:`_multiplier_strain_term` it is not about the
    constraint at all -- it is about what "bare" means when the coordinate is a
    strain.

    The transcribed route contracts the field response against
    :func:`~defumat.response.strain._bare_strains`, which rebuilds the potential
    from the converged density **array** at the deformed cell, so what it
    differentiates is ``dH/d(eps)`` at frozen ``rho``. The derivative the tensor
    is is taken at frozen *states*, and the density is a function of the states
    **and** of the cell, so the chain rule has one more link::

        dH/d(eps)|_psi = dH/d(eps)|_rho + K . (drho/d(eps))|_psi

    with ``K = dV_Hxc/drho``. Contracted with the field response and using that
    ``K`` is symmetric, the second link is

        -(1/Omega) int dV_scf^(E_k)(r) [drho/d(eps_ab)]_psi (r) d3r

    -- no factor of two, because the ``2 Re sum_n w_n`` of the main term is
    exactly what turns ``<dpsi|K.x|psi>`` into ``int x drho^(E)``, and the
    volume cancels against the ``Omega/N`` of the quadrature, leaving a mean
    over the grid. :func:`clamped_ion_piezoelectric` needs none of this because
    it hands the density to the energy as a *builder* that carries the strain,
    so its ``jvp`` generates the link itself.

    **Why a norm-conserving cell cannot see it, and that is a statement about
    the cell rather than a tolerance.** At frozen plane-wave coefficients the
    smooth density in crystal coordinates does not move under a strain at all --
    the exponentials are indexed by integers -- so ``[drho/d(eps_ab)]_psi`` is
    ``-delta_ab rho`` exactly, the volume's own ``1/Omega`` and nothing else,
    and it **vanishes for every traceless strain**. A zincblende crystal's only
    independent component is ``e_14``, a pure shear, so on the calibration cell
    this term is zero to 1e-15 and the two routes agreed to 6.2e-15 while one of
    them was missing it. An augmented dataset is where it exists: the
    augmentation charge ``Q_ij(r)`` deforms with the cell, so a shear moves the
    density even at frozen states.

    **Measured, and it is the whole of what was left** (``alas-piezo-tiny.in``,
    ultrasoft AlAs at 8 k-points, **at the ``ecutwfc = 12, ecutrho = 96`` rung
    rather than the committed ``10 / 44``**): the two routes were
    **-0.023777619** C/m^2 apart on ``e_14`` and this term is **-0.023777621**,
    agreeing to **2.6e-09**, which is the Sternheimer solve's own threshold
    rather than round-off -- the two routes contract differently-converged
    intermediates. At the committed rung the same pair reads **-0.022727213**
    and **1.7e-07**. Every other component of the term is 1e-15 at both.

    **PAW is refused above rather than approximated here**
    (:func:`require_a_norm_conserving_transcription`). For an ultrasoft dataset
    ``becsum`` reaches the energy only through the augmented density on the
    dense grid, which ``moved`` below already carries, so the grid integral is
    the whole coupling; PAW adds a one-centre energy that is a function of
    ``becsum`` directly, and its cross term with the field's ``dbecsum`` is not
    on any grid.

    ``None`` when the field's converged induced potential was not handed over,
    which is what the norm-conserving cross-check in ``test_piezoelectric.py``
    passes and what keeps that comparison at 6.2e-15.
    """
    if field_dvscf is None:
        return None
    from defumat.response.strain import _frozen_density_response

    # ``ort = None``: what is wanted is the derivative at frozen *states*, and
    # the orthogonality block is a change of the states. The function returns
    # the ``moved`` half separately for exactly this reason.
    _, moved, _, _ = _frozen_density_response(
        calculation, solver, solver.weights, None
    )
    points = int(np.prod(np.asarray(moved.shape[-3:])))
    out = np.zeros((3, 3, 3))
    for k in range(3):
        for a in range(3):
            for b in range(a, 3):
                value = -float(
                    jnp.sum(field_dvscf[k] * moved[a, b])
                ) / points
                out[k, a, b] = out[k, b, a] = value
    return out


def _multiplier_strain_term(
    calculation, solver, field_perturbations, band_weights, nocc,
) -> np.ndarray | None:
    """``(3, 3, 3)``: what the multipliers' own response adds, or ``None``.

    **This is the term ``zstar_eu.f90`` delegates to ``zstar_eu_us.f90`` for**,
    and it is one contraction here rather than three hundred lines there,
    because only *one* leg of this derivative moves ``S``. The reasoning is
    :mod:`defumat.response.born`'s, one coordinate over: the stationary
    functional carries the orthonormality constraint with matrix multipliers as
    ``-Re Tr[Lambda (G - 1)]`` (``forces/energy.py:_constraint_energy``, entering
    the total as ``-norm``), so the mixed second derivative picks up

        (d_Lambda d_eps E) . dLambda^E = -Re sum_mn dLambda^E_mn S'_nm

    with ``S'_nm = d<psi_n|S|psi_m>/d(eps_ab)``, and since
    ``e_(k)ab = -(1/Omega) d^2E/d(eps_ab) d(E_k)`` the tensor gains
    ``+(1/Omega) Re sum_mn dLambda^E_mn S'_nm``. **There is no factor of two**:
    the ``+ c.c.`` that doubles the main term is what ``jvp`` of a real function
    of complex primals produces for the *states* tangent, while ``Lambda``
    enters ``_constraint_energy`` linearly under an explicit ``Re``.

    Both factors already existed and both are ``None``/zero for a
    norm-conserving dataset, which is why the two routes agree to 6.2e-15 there
    and why this term cannot perturb that agreement:
    :func:`~defumat.response.strain.overlap_derivatives` returns ``None`` when
    ``S`` does not deform, and
    :func:`~defumat.response.born._multiplier_response` multiplies exactly the
    object that vanishes with it. The index order is the one trap and is not a
    convention -- ``_multiplier_response``'s own docstring records that the
    weight belongs to the *column* and that transposing it costs 0.28 on
    ultrasoft silicon while costing nothing at all on a norm-conserving cell.

    ``field_perturbations`` are the three callables the last Sternheimer solve
    was driven by, rebuilt at the converged ``dV_scf``; ``None`` skips the term,
    which is what the norm-conserving cross-check in
    ``test_piezoelectric.py`` passes.

    **Measured, and it is right and not sufficient** (Triton `20339308_0`,
    ultrasoft AlAs on the whole ``4 4 4`` grid, 64 k-points). Adding it moves
    the transcribed route from ``+0.830702`` to ``+0.827448`` where
    :func:`clamped_ion_piezoelectric` reads ``+0.815802``: the right sign,
    ``21.8 per cent`` of the gap, and the disagreement falls from 1.79 to 1.41
    per cent. So this is one piece of what ``zstar_eu_us.f90`` adds and not all
    of it, and the refusal stays.

    **The target it was aimed at is not the truth, which is why a fifth was
    all it could close.** ``+0.815802`` is
    :func:`clamped_ion_piezoelectric`, and that assembly carries only the
    states tangent: ``multipliers``, ``constraint_position_term``,
    ``commutator`` and ``_full_zone`` appear nowhere in it, where
    :func:`~defumat.response.born.born_effective_charges` -- the function that
    actually matches ``ph.x`` on an ultrasoft cell -- carries all of them.
    :func:`born_charges_from_stress_route` says the same thing from the other
    side: in the position coordinate this tape is ``Z*`` *minus the constraint
    term an ultrasoft dataset adds*. **So on an augmented dataset both routes
    are incomplete and this term belongs in both**, and the only reference here
    that needs none of it is the Berry-phase finite difference, ``+0.692986``.

    The piece still missing from *both* is
    :func:`~defumat.response.born.constraint_position_term`'s, which is
    ``sum_n w_n <psi_n| dS/d(eps) | P_c r_k psi_n>`` in this coordinate:
    ``_multiplier_response`` reaches only what the occupied-occupied block
    carries, and its docstring says so, while ``add_for_charges`` is the rest
    and is **worth 0.55 on ultrasoft silicon**, the difference between +0.47
    and -0.079. ``internals["commutators"]`` is the ``P_c r|psi>`` it needs and
    the sandwich is ``constraint_position_term``'s with
    :meth:`~defumat.scf.driver.Calculation.at_strain` in place of
    ``at_positions``.
    """
    if field_perturbations is None or band_weights is None:
        return None
    if not isinstance(nocc, (int, np.integer)):
        raise TypeError(
            f"nocc must be the single occupied-band count, got {nocc!r}. "
            "``SternheimerSolver.nocc`` is that number; the response's "
            "``internals['nocc']`` is the per-spin tuple beside it"
        )
    from defumat.response.born import _multiplier_response
    from defumat.response.strain import overlap_derivatives

    derivatives = overlap_derivatives(calculation, solver)
    if derivatives is None:
        return None

    volume = calculation.system.cell.volume
    nbnd = derivatives[0, 0].shape[-1]
    out = np.zeros((3, 3, 3))
    for k in range(3):
        dlambda = _multiplier_response(
            solver, field_perturbations[k], band_weights, nbnd, nocc
        )
        for a in range(3):
            for b in range(a, 3):
                value = float(jnp.real(jnp.einsum(
                    "skmn,sknm->", dlambda, derivatives[a, b]
                ))) / volume
                out[k, a, b] = out[k, b, a] = value
    return out


def piezoelectric_from_strain_response(calculation, solver, bare, strain) -> np.ndarray:
    """The same tensor with the two perturbations interchanged -- ``(3, 3, 3)``.

    A mixed second derivative can be contracted either way round, and this is
    the other one: the **strain's** self-consistent response against the
    field's bare perturbation::

        e_(k)ab = -(2/Omega) sum_n w_n Re <b_k | dpsi^(ab)_n>

    where ``b_k`` is ``P_c r_k|psi>``, the array
    :func:`~defumat.response.efield._assemble` builds ``epsilon`` from, and
    ``dpsi^(ab)`` is P26's strain response. It costs six more Sternheimer
    solves, which :func:`piezoelectric_zstar_eu_style` does not, and it is here
    because it is the only route that puts the **strain** response on the
    screened side -- so agreement between the three is a statement about that
    response as well as about the assembly.

    The factor is 2 for :func:`piezoelectric_zstar_eu_style`'s reason.
    """
    volume = calculation.system.cell.volume
    weights = solver.weights
    tensor = np.zeros((3, 3, 3))
    for k in range(3):
        for a in range(3):
            for b in range(3):
                overlap = jnp.einsum(
                    "skng,skng->skn", jnp.conj(bare[k]), strain.dpsi[a, b]
                )
                tensor[k, a, b] = -2.0 * float(
                    jnp.sum(weights * jnp.real(overlap))
                ) / volume
    return calculation.symmetrize_cartesian_tensor(tensor)


# -- the driver --------------------------------------------------------------


#: The routes :func:`piezoelectric_tensor` will assemble the tensor with. Both
#: consume the *same* field response, so neither is a check on the other's
#: solve; what they differ in is what happens above it. See the module
#: docstring's cost table for why the default is the dear one.
PIEZOELECTRIC_METHODS = ("autodiff", "zstar_eu")


def piezoelectric_tensor(
    calculation,
    result,
    verbose: bool = False,
    allow_unconverged: bool = False,
    method: str = "autodiff",
    kmesh_warning: bool = True,
    kmesh_drift: float | None = None,
    allow_a_coarse_mesh: bool = False,
    **response_options,
) -> PiezoelectricTensor:
    """The clamped-ion piezoelectric tensor of a converged insulator.

    Args:
        calculation: the :class:`~defumat.scf.driver.Calculation` the run used.
            An ordinary symmetry-reduced wedge is fine and is averaged as a
            rank-3 tensor; a ``nosym`` run is accepted only on an unshifted
            grid, which is :mod:`defumat.response.efield`'s own rule.
        result: the converged :class:`~defumat.scf.driver.SCFResult`. Its
            states are re-diagonalised first
            (:func:`~defumat.response.electrostriction.refined_states`).
        allow_unconverged: return an answer even when the field response did
            not converge. Off by default.
        method: ``'autodiff'`` (default), the ``jvp`` of the stress that this
            module is built around, or ``'zstar_eu'``, the transcribed
            contraction of the field response against the bare strain
            perturbation. **They are the same number and not the same cost**:
            on the two-atom AlAs cell the second is 4.0 s against 6.4 and
            carries no extra memory at all, where the first holds a
            forward-over-reverse tape of every radial and reciprocal-space
            intermediate the cell derivative rebuilds. That is worth reaching
            for on a large cell or a dense mesh -- measured on the ultrasoft
            cell, the autodiff route peaks at **139.6 GiB** at 64 k-points and
            scales at 1.6 GiB a point, which puts ``6 6 6`` on a whole node.
            The default stays ``'autodiff'`` because it is the route that
            extends, and because the transcribed one is only as good as the
            transcription, which is why both run in the regression file.
        kmesh_warning: warn that the Brillouin-zone sum is the one parameter
            of this quantity no check inside it can see
            (:func:`_warn_about_the_kmesh`). On by default, and the reason it
            is a warning rather than a refusal is that which mesh is enough is
            a property of the crystal.
        kmesh_drift: the last step of a k-mesh ladder, when one was run, which
            is what
            :func:`~defumat.workflows.piezo_ladder.piezoelectric_kmesh_ladder`
            passes. Below :data:`KMESH_STEP` it silences the warning; above it,
            the warning quotes it instead of quoting AlAs.
        allow_a_coarse_mesh: run an **ultrasoft** dataset below
            :data:`ULTRASOFT_MESH` divisions, which
            :func:`require_a_measured_dataset` otherwise refuses. It exists for
            :func:`~defumat.workflows.piezo_ladder.piezoelectric_kmesh_ladder`,
            whose coarse rungs are how the mesh gets measured in the first
            place, and the tensor a coarse rung returns is a point on a curve
            rather than an answer.
        response_options: passed to the field response.
    """
    require_a_piezoelectric_tensor(
        calculation, kmesh_drift, allow_a_coarse_mesh=allow_a_coarse_mesh
    )
    if kmesh_warning:
        _warn_about_the_kmesh(calculation, kmesh_drift)
    if method not in PIEZOELECTRIC_METHODS:
        raise ValueError(
            f"unknown piezoelectric method {method!r}; expected one of "
            f"{', '.join(PIEZOELECTRIC_METHODS)}"
        )
    if method == "zstar_eu":
        require_a_norm_conserving_transcription(calculation)

    eigenvalues, psi = refined_states(calculation, result)
    density = jnp.asarray(result.density)
    field = dielectric_tensor(
        calculation, psi, eigenvalues, density, result.becsum,
        born_charges=False, keep_internals=True, verbose=verbose,
        **response_options,
    )
    if not allow_unconverged:
        require_converged_responses(field, None)

    internals = field.internals
    # The three callables the last Sternheimer solve was driven by, rebuilt at
    # the converged ``dV_scf`` exactly as
    # :func:`~defumat.response.efield.dielectric_tensor` rebuilds them for the
    # Born charges. They are what the multipliers' own response is a matrix
    # element of, so nothing new is solved -- only contracted differently. Both
    # routes need them on an augmented dataset and neither uses them otherwise.
    from defumat.response.efield import _bare_plus_induced as _perturbation_of

    _onecentre = internals["onecentre"]
    _perturbations = [
        _perturbation_of(
            internals["solver"], internals["bare"][axis],
            internals["dvscf"][axis],
            None if _onecentre is None else _onecentre[axis], True,
        )
        for axis in range(3)
    ]
    if method == "zstar_eu":
        # The three callables the last Sternheimer solve was driven by, rebuilt
        # at the converged ``dV_scf`` exactly as
        # :func:`~defumat.response.efield.dielectric_tensor` rebuilds them for
        # the Born charges. They are what the multipliers' own response is a
        # matrix element of, so nothing new is solved -- only contracted
        # differently (:func:`_multiplier_strain_term`).
        from defumat.response.efield import _bare_plus_induced

        onecentre = internals["onecentre"]
        perturbations = [
            _bare_plus_induced(
                internals["solver"], internals["bare"][axis],
                internals["dvscf"][axis],
                None if onecentre is None else onecentre[axis], True,
            )
            for axis in range(3)
        ]
        e = piezoelectric_zstar_eu_style(
            calculation, internals["solver"], density, internals["dpsi"],
            commutators=internals["commutators"],
            field_perturbations=perturbations,
            band_weights=jnp.asarray(internals["weights"]),
            # ``solver.nocc`` and *not* ``internals["nocc"]``: the latter is the
            # per-spin tuple ``(4,)`` and the former the one number across the
            # channels, which is what slices an array.
            # :func:`clamped_ion_piezoelectric` is handed the same thing one
            # line further down, and getting it wrong here could not be caught
            # on a norm-conserving cell, where
            # :func:`_multiplier_strain_term` returns before it is used.
            nocc=internals["solver"].nocc,
            # The converged induced potential of the field response, which is
            # the ``K . drho^(E)`` half of :func:`_screened_strain_term` and is
            # already built -- nothing further is solved for it either.
            field_dvscf=internals["dvscf"],
        )
    else:
        e = clamped_ion_piezoelectric(
            calculation, psi, eigenvalues, jnp.asarray(internals["weights"]),
            density, result.becsum, internals["dpsi"], internals["solver"].nocc,
            solver=internals["solver"], field_perturbations=_perturbations,
            commutators=internals["commutators"],
        )
    nk, grid = _kmesh_of(calculation)
    return PiezoelectricTensor(
        e=e,
        voigt=to_voigt(e) * E_BOHR2_TO_C_M2,
        dielectric=field,
        converged=bool(field.converged),
        nk=nk,
        grid=grid,
        kmesh_drift=kmesh_drift,
    )
