"""P50: the clamped-ion piezoelectric tensor, and the three routes to it.

``e_(k)ij = dP_k/d(eps_ij) = d(sigma_ij)/dE_k`` is a mixed second derivative of
the energy, and the whole content of this phase is that it is the Born charge's
construction with a strain where ``Z*`` has a displacement. What that buys is
that **the same assembly, run in the position coordinate, is the Born charge**
-- which is validated against the vendored ``ph.x`` to every digit it prints --
so the sign, the field's normalisation and the volume factor are anchored to a
reference without one existing for the piezoelectric tensor itself.

There is no such reference. ``pw.x`` computes no piezoelectric tensor at all
(the word occurs once in the vendored tree, in a citation in a comment in
``PW/src/bp_c_phase.f90``), and Elk's task 380 reaches it by a finite difference
of the Berry-phase polarization over one full ground state per strain. So the
validation is four internal statements, each of which fails for a different
reason if the assembly is wrong:

* silicon is centrosymmetric and its whole tensor must vanish, with nothing
  imposing that on a ``nosym`` run;
* AlAs is ``-43m`` and the only components a rank-3 tensor may have are
  ``e_14 = e_25 = e_36``, again with nothing imposing it;
* the wedge and the closed grid must agree, which is what the rank-3
  symmetriser has to earn;
* and the mixed derivative must be the same number contracted either way
  round -- the field's response against the strain's bare perturbation, and
  the strain's response against the field's.

The last of those is where this phase's trap lives, and it is a factor of two:
``dielec.f90``'s contraction of the *same* field response carries a 4 because a
susceptibility is Coulomb-normalised and a Rydberg-unit code puts ``e^2 = 2``
there, while a bare mixed second derivative carries 2. Taking the 4 gives a
tensor that is exactly zincblende, exactly symmetric, vanishes on silicon, and
is twice too large.
"""

from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.response.efield import dielectric_tensor
from defumat.response.electrostriction import refined_states
from defumat.response.piezo import (
    born_charges_from_stress_route,
    clamped_ion_piezoelectric,
    piezoelectric_from_strain_response,
    piezoelectric_tensor,
    piezoelectric_zstar_eu_style,
    require_a_nonpolar_crystal,
    to_voigt,
)
from defumat.response.strain import strain_response
from defumat.scf import Calculation, run_scf
from defumat.system import build_system
from defumat.units import E_BOHR2_TO_C_M2

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: What a symmetry forbids is round-off here: nothing in these runs imposes the
#: crystal class, and the measured residue is 1.7e-14 on AlAs.
FORBIDDEN = 1e-10

#: ``ph.x``'s own Born charges for this cell, from
#: ``reference.out.ph-alas-raman`` -- printed to five decimals, which is what
#: the comparison can use.
PH_ZSTAR = (1.92461, -3.18098)


@pytest.fixture(autouse=True)
def _bounded_compilation():
    """Drop XLA's executables between cases -- ``CLAUDE.md``'s memory rule.

    Three cells that share no shape each compile the whole response stack
    afresh and the backend keeps every one of them for the life of the process.
    The converged states stay cached below; only the compiled code goes.
    """
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _converged(case: str):
    """The ground state, on QE's ``k + G = 0`` convention.

    **This file runs on Quantum ESPRESSO's convention for the ``l = 1`` tangent
    at ``k + G = 0``** (``origin_tangent=False``). Every external anchor here is
    a ``ph.x`` number on an AlAs cell whose mesh is ``4 4 4 0 0 0`` and so holds
    Gamma, and QE zeroes that row where this code carries it by default
    (``commutator_Hx_psi.f90:113-118``, ``dylmr2.f90:88-92``; `PLAN.md` P24).
    Carrying it moves ``Z*`` by **2.0e-3** on 1.92461 and the dielectric
    constant by **3.7e-2** on 12.9673, both past their tolerances, while every
    *route identity* in this file -- one assembly against another at 1e-12, the
    driver against its own solve at 1e-8 -- holds on either leg, which is what
    says the term is applied consistently and only the reference moved.
    """
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos, origin_tangent=False)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                     max_iterations=100)
    return system, pseudos, calculation, result


@lru_cache(maxsize=2)
def _field(case: str):
    """The field response and the states it was solved at."""
    _, _, calculation, result = _converged(case)
    eigenvalues, psi = refined_states(calculation, result)
    density = jnp.asarray(result.density)
    field = dielectric_tensor(
        calculation, psi, eigenvalues, density, result.becsum,
        born_charges=True, keep_internals=True,
    )
    return calculation, result, eigenvalues, psi, density, field


@lru_cache(maxsize=2)
def _piezo(case: str):
    """The tensor itself, in ``e/bohr^2``."""
    calculation, result, eigenvalues, psi, density, field = _field(case)
    internals = field.internals
    return clamped_ion_piezoelectric(
        calculation, psi, eigenvalues, jnp.asarray(internals["weights"]),
        density, result.becsum, internals["dpsi"], internals["solver"].nocc,
    )


def _forbidden(voigt: np.ndarray) -> float:
    """The largest component a ``-43m`` crystal may not have."""
    allowed = [(0, 3), (1, 4), (2, 5)]
    return max(
        abs(voigt[i, j])
        for i in range(3) for j in range(6) if (i, j) not in allowed
    )


def test_zincblende_has_one_independent_component_and_nothing_else():
    """``e_14 = e_25 = e_36`` and every other component is zero.

    ``-43m`` permits exactly one independent component of a rank-3 tensor
    symmetric in two of its indices. The run is ``nosym`` on the whole closed
    grid, so no average is applied anywhere and the crystal class is not
    imposed on the answer -- it is what the answer comes out as.
    """
    voigt = to_voigt(_piezo("alas-raman")) * E_BOHR2_TO_C_M2
    assert _forbidden(voigt) < FORBIDDEN
    assert abs(voigt[0, 3]) > 0.5
    assert abs(voigt[1, 4] - voigt[0, 3]) < FORBIDDEN
    assert abs(voigt[2, 5] - voigt[0, 3]) < FORBIDDEN


def test_a_centrosymmetric_crystal_has_no_piezoelectric_response():
    """Silicon's tensor vanishes, and this is what AlAs's not vanishing means.

    An inversion centre forbids every component of an odd-rank tensor. Nothing
    in the run imposes it -- ``nosym``, no symmetrisation -- so what is left is
    the response solver's own floor, and it is four orders below the AlAs value
    computed by the same code on the same day.
    """
    voigt = to_voigt(_piezo("si-electrostriction")) * E_BOHR2_TO_C_M2
    assert np.abs(voigt).max() < 1e-3


def test_the_wedge_reproduces_the_closed_grid():
    """P36's rank-3 symmetriser, on a tensor that is *linear* in the response.

    Eight k-points averaged as a rank-3 tensor against sixty-four run whole.
    Unlike the Raman tensor, where the screening term is quadratic in a
    per-k object and a wedge sum has to be completed *inside* the functional,
    this assembly is linear in ``dpsi``, so the average of the assembled tensor
    is the whole of the completion.
    """
    closed = to_voigt(_piezo("alas-raman"))[0, 3]
    wedge = to_voigt(_piezo("alas-raman-wedge"))[0, 3]
    assert abs(wedge - closed) < 1e-7 * abs(closed)


def test_the_transcribed_contraction_reproduces_the_differentiated_one():
    """``zstar_eu.f90``'s expression with a strain label, beside the ``jvp``.

    The two share the field response and nothing else: one is a derivative of
    the energy functional taken by JAX, the other is QE's hand-derived
    contraction of a bare perturbation against that response. This is the test
    that fixes the factor of two the module docstring records -- with
    ``dielec.f90``'s 4 in place of ``zstar_eu.f90``'s 2 it fails by exactly a
    factor of two and every other test in this file still passes.
    """
    calculation, result, eigenvalues, psi, density, field = _field("alas-raman")
    transcribed = piezoelectric_zstar_eu_style(
        calculation, field.internals["solver"], density,
        field.internals["dpsi"],
    )
    differentiated = _piezo("alas-raman")
    assert np.abs(transcribed - differentiated).max() < 1e-12


def test_the_constraint_term_is_wired_and_inert_on_this_dataset():
    """The three arguments the entry point now builds, exercised on this cell.

    The transcribed route takes the field perturbations, the band weights,
    ``nocc`` and the field's converged induced potential on top of what it used
    to, so that :func:`~defumat.response.piezo._multiplier_strain_term` can add
    the multipliers' own response and
    :func:`~defumat.response.piezo._screened_strain_term` the screening of the
    frozen-state density response -- the two terms ``zstar_eu.f90`` delegates to
    ``zstar_eu_us.f90`` for. This builds them the way
    :func:`~defumat.response.piezo.piezoelectric_tensor` does and asserts the
    answer does not move.

    **What this cell can and cannot say.** It cannot say the terms are *right*:
    ``overlap_derivatives`` returns ``None`` when ``S`` does not deform, so the
    multiplier term returns before it touches any of the three, and a wrong
    argument there is invisible -- which is exactly how ``internals['nocc']``,
    the per-spin tuple, got passed where ``solver.nocc``, the number, was
    wanted. The screened term is not skipped here at all and still changes
    nothing, for a different and sharper reason: at frozen plane-wave
    coefficients the smooth density in crystal coordinates does not move under a
    **traceless** strain, so it is identically zero on the one component a
    zincblende crystal has. What this says, then, is that the construction runs
    and that adding both terms has not perturbed an agreement which was 6.2e-15
    before either existed -- measured at 1.9e-14 after, with ``e_14`` itself
    unchanged at -0.763786071 in every digit. The numbers that check the terms
    are on an ultrasoft cell, in ``test_piezoelectric_augmented.py``.
    """
    from defumat.response.efield import _bare_plus_induced

    calculation, _, _, _, density, field = _field("alas-raman")
    internals = field.internals
    onecentre = internals["onecentre"]
    perturbations = [
        _bare_plus_induced(
            internals["solver"], internals["bare"][axis],
            internals["dvscf"][axis],
            None if onecentre is None else onecentre[axis], True,
        )
        for axis in range(3)
    ]
    wired = piezoelectric_zstar_eu_style(
        calculation, internals["solver"], density, internals["dpsi"],
        field_perturbations=perturbations,
        band_weights=jnp.asarray(internals["weights"]),
        nocc=internals["solver"].nocc,
        field_dvscf=internals["dvscf"],
    )
    assert np.abs(wired - _piezo("alas-raman")).max() < 1e-12


def test_the_same_mixed_derivative_contracted_the_other_way_round():
    """The strain's *response* against the field's bare perturbation.

    The only route that puts the strain response on the screened side, so it
    costs six more Sternheimer solves and is the one statement here that says
    anything about that response. It agrees to 1.3e-7 relative, which is the
    strain response's own convergence rather than the assembly's.
    """
    calculation, result, eigenvalues, psi, density, field = _field("alas-raman")
    strain = strain_response(
        calculation, psi, eigenvalues, density, result.becsum,
    )
    assert strain.converged
    swapped = piezoelectric_from_strain_response(
        calculation, field.internals["solver"], field.internals["bare"], strain,
    )
    reference = _piezo("alas-raman")
    assert abs(swapped[0, 1, 2] - reference[0, 1, 2]) < 1e-6 * abs(reference[0, 1, 2])


def test_the_same_assembly_in_the_position_coordinate_is_the_born_charge():
    """The anchor: change ``at_strain`` to ``at_positions`` and get ``Z*``.

    Nothing external validates a piezoelectric tensor here, and this is what
    stands in for one. The module's assembly is a ``jvp`` of a *coordinate*
    gradient of the frozen energy along the field response; run with the atoms
    as that coordinate it is the electronic half of a Born effective charge,
    and the bare ionic charge completes it for a norm-conserving dataset. The
    result is ``ph.x``'s, so a wrong sign, a wrong field normalisation or a
    missing volume in the piezoelectric tensor would show here.
    """
    calculation, result, eigenvalues, psi, density, field = _field("alas-raman")
    system, pseudos, _, _ = _converged("alas-raman")
    internals = field.internals
    electronic = born_charges_from_stress_route(
        calculation, psi, eigenvalues, jnp.asarray(internals["weights"]),
        density, result.becsum, internals["dpsi"], internals["solver"].nocc,
    )
    valence = np.array([float(p.z_valence) for p in pseudos])
    types = np.asarray(system.structure.types)
    charges = np.stack([valence[t] * np.eye(3) for t in types]) - electronic

    assert np.abs(charges - field.born_charges).max() < 1e-12
    for atom, reference in enumerate(PH_ZSTAR):
        assert abs(np.trace(charges[atom]) / 3 - reference) < 5e-4


def test_the_driver_reports_the_dielectric_constant_it_already_solved():
    """One field response serves both, which is what makes this NSCF-scale."""
    _, _, calculation, result = _converged("alas-raman")
    tensor = piezoelectric_tensor(calculation, result)
    assert tensor.converged
    assert abs(tensor.e14 - to_voigt(_piezo("alas-raman"))[0, 3] * E_BOHR2_TO_C_M2) < 1e-8
    assert abs(np.trace(tensor.dielectric.epsilon) / 3 - 12.9674) < 1e-3


def test_a_polar_crystal_is_refused_by_name():
    """The improper-to-proper correction needs ``P``, and there is no ``P`` here.

    Displacing one atom of AlAs along ``z`` leaves a point group with one
    invariant direction, so the crystal may carry a spontaneous polarization
    and ``e^proper - e^improper = delta_ki P_j - delta_ij P_k`` is not zero.
    Nothing in the tensor's symmetry would say so, which is why the refusal is
    by name.
    """
    _, _, calculation, _ = _converged("alas-raman")
    positions = np.asarray(calculation.system.structure.positions)
    displaced = calculation.at_positions(
        jnp.asarray(positions + np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.0]]))
    )
    require_a_nonpolar_crystal(calculation)
    with pytest.raises(NotImplementedError, match="polar crystal"):
        require_a_nonpolar_crystal(displaced)


@pytest.mark.unit
def test_the_ultrasoft_refusal_names_the_term_and_not_the_pseudopotentials(
    pseudo_dir,
):
    """The refusal said the obstacle was the *dataset library*, and it was not.

    "Every ultrasoft and PAW case committed here is a centrosymmetric crystal"
    was already false when it was written: ``Al.pbe-n-rrkjus_psl`` and
    ``As.pbe-n-rrkjus_psl`` are committed and ``alas-magnetoelectric-nosoc.in``
    already builds zincblende from them -- ``ibrav = 2`` with the basis at
    ``(0,0,0)`` and ``(1/4,1/4,1/4)``, which is the non-polar,
    non-centrosymmetric class ``require_a_nonpolar_crystal`` names as handled.
    A reader who believed the sentence would have gone looking for a
    pseudopotential file. What actually blocks it is one term: ``Q_ij(r)`` is a
    function of the cell, so ``dbecsum`` gains a strain term of its own.

    This builds the case, checks it really is what the old claim denied
    existed, and checks the refusal names a real obstacle.

    **The words it pins changed twice and both reasons are worth keeping.** On
    2026-09-19 it stopped blaming a missing strain term in
    ``response/strain.py``, which P41 had written and ``test_electrostriction``
    pins at 4.6e-4 (``AUDIT-2026-09-18.md`` ``drift.3``). On 2026-09-20 the
    ultrasoft half of the refusal was **lifted**: the ladder put this quantity
    0.57 per cent from a Berry-phase finite difference at ``10 10 10`` against
    the norm-conserving calibration's 1.19 on the same mesh, so what is left is
    a condition on the **mesh** and not on the dataset. This cell carries
    ``4 4 4``, which is below it, so it still raises -- and what the message has
    to do now is say which of the two it is, since a reader who reads "ultrasoft"
    and stops will change their pseudopotentials when what they need is a denser
    grid.
    """
    from defumat.response.piezo import require_a_measured_dataset

    system = build_system(read_pw_input(
        Path(__file__).parent.parent / "data" / "qe" / "alas-piezo.in"
    ))
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)

    # The case exists, and it is the one the old text said did not.
    assert calculation.is_ultrasoft
    assert system.nspin == 1 and not system.noncolin
    require_a_nonpolar_crystal(calculation)  # non-polar, and not centrosymmetric
    assert calculation.symmetries.nsym == 24  # Td, which has no inversion
    assert not any(
        np.allclose(np.asarray(r), -np.eye(3))
        for r in calculation.symmetries.rotation_array()
    )

    with pytest.raises(NotImplementedError) as raised:
        require_a_measured_dataset(calculation)
    message = str(raised.value)
    # The reader is not sent to find a pseudopotential, which is what the
    # original text did, and not to write a term that already exists either.
    assert "centrosymmetric" not in message
    assert "response/strain.py refuses" not in message
    # What blocks *this* run is the mesh, and the message says so with the
    # number that lifted the dataset half and the two ways past it.
    assert "denser" in message and "k-mesh" in message
    assert "8 divisions" in message or "8 8 8" in message
    assert "piezoelectric_kmesh_ladder" in message

    # And the same cell on a mesh where the dataset *was* measured is allowed,
    # which is the half of this that a refusal test usually leaves out. The
    # k-set is substituted rather than rebuilt, because what the guard reads is
    # the grid.
    import dataclasses

    from defumat.system.kpoints import KPoints, for_spin

    dense = dataclasses.replace(system, kpoints=for_spin(KPoints.automatic(
        (8, 8, 8), (0, 0, 0), system.cell,
        rotations=system.symmetry_group().rotation_array(),
    ), system.nspin))
    require_a_measured_dataset(Calculation(dense, pseudos))
