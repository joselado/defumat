"""The two piezoelectric routes made to agree on an **augmented** dataset.

``test_piezoelectric.py`` checks everything this phase can check on a
norm-conserving cell, and the whole of what it could not is here. The two
assemblies -- :func:`~defumat.response.piezo.clamped_ion_piezoelectric`, one
``jvp`` of the stress along the field response, and
:func:`~defumat.response.piezo.piezoelectric_zstar_eu_style`, the same mixed
second derivative contracted term by term -- agreed there to ``6.2e-15`` for
two years while **both** were missing terms, because every one of them is
contracted either with ``dS/d(eps)``, identically zero when ``S`` is the
identity, or with the frozen-state density response, which for a traceless
strain at frozen plane-wave coefficients is identically zero as well. A
zincblende crystal's only independent component is the shear ``e_14``, so that
cell could not have seen any of it.

**Two cells, and the second is not a duplicate.** ``alas-piezo-tiny.in`` is the
whole unshifted grid with no symmetry and says the two assemblies are the same
derivative; ``alas-piezo-tiny-wedge.in`` is the same k-sample reduced to three
points and says the *wedge sum* has been completed, which on an augmented
dataset is a separate statement because one term of this derivative is quadratic
in a per-k tangent and no average of the finished tensor can repair it. Each
cell caught something the other could not: without the second, the taped route
was 1.05e-03 out on a wedge while agreeing with the contracted one to 1.7e-07 on
the closed grid.

**Why a cell chosen for cost is legitimate here, where it would not be for a
number.** Two assemblies of the same mixed second derivative must agree at any
cutoff and on any mesh; their disagreement is an assembly defect, not a
convergence question. So the identity can be read on
``tests/data/qe/alas-piezo-tiny.in``, ultrasoft AlAs at ``ecutwfc = 10`` and 8
k-points, where ``alas-piezo.in`` at 64 k-points peaks at **139.6 GiB** in the
taped route and needs a whole node. The cell's own ``e_14`` of 1.474 C/m^2 is
not physics and nothing here reads it as such.

**The test asserts that the guard fires**, which is ``CLAUDE.md``'s rule about
a check whose null result cannot be told from a pass: the screened term is
measured first and required to be *large*, and only then are the two routes
required to agree. An agreement between two routes that are both missing the
same term is exactly what the norm-conserving cell showed for two years.
"""

from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.response.efield import _bare_plus_induced, dielectric_tensor
from defumat.response.electrostriction import refined_states
from defumat.response.piezo import (
    _multiplier_strain_term,
    _screened_strain_term,
    clamped_ion_piezoelectric,
    piezoelectric_zstar_eu_style,
    to_voigt,
)
from defumat.scf import Calculation, run_scf
from defumat.system import build_system
from defumat.units import E_BOHR2_TO_C_M2

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: What the screened term was measured at on this cell, in C/m^2 on ``e_14``.
#: It is asserted as an order of magnitude rather than as a digit, because what
#: the test needs from it is that it is not zero -- the cutoff may be lowered
#: again against the runner's memory cap without that mattering.
SCREENED_E14 = -0.0227272

#: ``tools/run_regression.sh`` caps each file at 12 GiB and this one measured
#: **10.1 GiB** at ``ecutwfc = 10, ecutrho = 44``, 2m44 -- which is why it is a
#: file of its own rather than three more tests in ``test_piezoelectric.py``: a
#: file boundary is a process boundary under that runner, and the peak here is
#: the taped route's forward-over-reverse tape rather than anything cached.
CASE = "alas-piezo-tiny"

#: The same cell and the same unshifted grid with the point group kept, so that
#: eight k-points reduce to three. The pair is what says the wedge sum has been
#: completed, and on an augmented dataset that is a statement about a term
#: quadratic in a per-k tangent rather than about the rank-3 symmetriser.
WEDGE = "alas-piezo-tiny-wedge"

#: ``e_14`` on the closed grid, C/m^2, from
#: :func:`test_the_two_routes_agree_on_an_augmented_dataset` -- quoted rather
#: than recomputed so that the wedge test costs one ground state and not two.
CLOSED_GRID_E14 = 1.474377366


@pytest.fixture(autouse=True)
def _bounded_compilation():
    """Drop XLA's executables between cases -- ``CLAUDE.md``'s memory rule."""
    yield
    jax.clear_caches()


@lru_cache(maxsize=1)
def _field(case: str = CASE):
    """One converged ground state and one field response.

    ``maxsize=1`` and not 2, deliberately: the two cells here are a 10.1 GiB and
    a 9.0 GiB peak and the runner caps the file at 12, so holding both alive is
    exactly the accumulation ``CLAUDE.md`` names. The second cell evicts the
    first and pays for its own SCF, which is seconds.
    """
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-10,
                     max_iterations=100)
    eigenvalues, psi = refined_states(calculation, result)
    density = jnp.asarray(result.density)
    field = dielectric_tensor(
        calculation, psi, eigenvalues, density, result.becsum,
        born_charges=False, keep_internals=True,
    )
    assert field.converged
    return calculation, result, eigenvalues, psi, density, field


def _both_routes(case: str):
    """``e_14`` in C/m^2 from the taped and the contracted assembly."""
    calculation, result, eigenvalues, psi, density, field = _field(case)
    internals = field.internals
    perturbations = _perturbations(internals)
    weights = jnp.asarray(internals["weights"])
    taped = clamped_ion_piezoelectric(
        calculation, psi, eigenvalues, weights, density, result.becsum,
        internals["dpsi"], internals["solver"].nocc,
        solver=internals["solver"], field_perturbations=perturbations,
        commutators=internals["commutators"],
    )
    contracted = piezoelectric_zstar_eu_style(
        calculation, internals["solver"], density, internals["dpsi"],
        field_perturbations=perturbations, band_weights=weights,
        nocc=internals["solver"].nocc, commutators=internals["commutators"],
        field_dvscf=internals["dvscf"],
    )
    scale = E_BOHR2_TO_C_M2
    return to_voigt(taped)[0, 3] * scale, to_voigt(contracted)[0, 3] * scale


def _perturbations(internals):
    """The three callables the last Sternheimer solve was driven by."""
    onecentre = internals["onecentre"]
    return [
        _bare_plus_induced(
            internals["solver"], internals["bare"][axis],
            internals["dvscf"][axis],
            None if onecentre is None else onecentre[axis], True,
        )
        for axis in range(3)
    ]


def test_the_screened_term_is_large_and_lives_only_in_the_shear():
    """Measured first, so that the agreement below cannot be read as a null.

    ``_screened_strain_term`` is what the transcribed route was missing after
    the multipliers' own response went in: ``_bare_strains`` freezes the density
    *array* and the density is a function of the cell as well as of the states,
    so ``dH/d(eps)`` is short of ``K . (drho/d(eps))|_psi``.

    Two things are asserted and the first is the one that matters. It is
    **1.5 per cent of e_14** on this cell, so a route without it is wrong by
    that much and the next test's agreement is a real statement. And it sits in
    the shear alone: ``[drho/d(eps)]_psi`` is ``-delta_ab rho`` plus the
    augmentation charge's own deformation, and the trace part contracts with
    the field's induced potential to a vector, which a non-polar class has none
    of -- which is why every other component is 1e-15 rather than small.
    """
    calculation, _, _, _, _, field = _field()
    internals = field.internals
    term = _screened_strain_term(
        calculation, internals["solver"], internals["dvscf"]
    )
    assert term is not None
    voigt = to_voigt(term) * E_BOHR2_TO_C_M2
    assert voigt[0, 3] == pytest.approx(SCREENED_E14, rel=1e-3)
    assert abs(voigt[0, 3]) > 1e-2
    # e_14 = e_25 = e_36 and nothing else, with nothing imposing it here.
    assert np.abs(np.delete(voigt, (3, 4, 5), axis=1)).max() < 1e-10


def test_the_two_routes_agree_on_an_augmented_dataset():
    """The identity, and the whole reason this file exists.

    Before :func:`~defumat.response.piezo._screened_strain_term` the two routes
    were **0.0238 C/m^2** apart on ``e_14`` on this cell at ``ecutwfc = 12``,
    1.6 per cent, and the same pair was 1.8 per cent apart on ``alas-piezo.in``
    at 64 k-points (Triton ``20339831``). What is left is the Sternheimer
    solve's own threshold rather than round-off, because the two routes contract
    differently-converged intermediates: **2.6e-09** at ``ecutwfc = 12`` and
    1.7e-07 at the ``ecutwfc = 10`` committed here.

    The tolerance is set against the *defect* rather than against round-off, and
    deliberately: at 1e-5 it is three orders below what a missing term costs and
    two above what the response solve delivers, so it fails on the physics and
    not on a re-tuned mixing parameter.
    """
    calculation, result, eigenvalues, psi, density, field = _field()
    internals = field.internals
    perturbations = _perturbations(internals)
    weights = jnp.asarray(internals["weights"])

    taped = clamped_ion_piezoelectric(
        calculation, psi, eigenvalues, weights, density, result.becsum,
        internals["dpsi"], internals["solver"].nocc,
        solver=internals["solver"], field_perturbations=perturbations,
        commutators=internals["commutators"],
    )
    contracted = piezoelectric_zstar_eu_style(
        calculation, internals["solver"], density, internals["dpsi"],
        field_perturbations=perturbations, band_weights=weights,
        nocc=internals["solver"].nocc, commutators=internals["commutators"],
        field_dvscf=internals["dvscf"],
    )
    gap = np.abs(to_voigt(taped - contracted) * E_BOHR2_TO_C_M2).max()
    assert gap < 1e-5, f"the two routes disagree by {gap:.3e} C/m^2"
    # ... and not because both collapsed to nothing.
    assert abs(to_voigt(taped)[0, 3] * E_BOHR2_TO_C_M2) > 1.0


def test_dropping_either_augmented_term_reopens_the_gap():
    """Each of the two terms is separately worth more than the tolerance.

    This is the test that would have failed against the code of 2026-09-19, and
    it fails in two different ways: withholding ``field_dvscf`` puts the
    transcribed route back where it was that morning, and withholding the field
    perturbations takes the multipliers' own response out on top of that. Both
    are far outside the 1e-5 the test above passes at, which is what says that
    the agreement there is carried by the terms rather than by the tolerance.
    """
    calculation, _, _, _, density, field = _field()
    internals = field.internals
    perturbations = _perturbations(internals)
    weights = jnp.asarray(internals["weights"])

    def contracted(**switches):
        return to_voigt(piezoelectric_zstar_eu_style(
            calculation, internals["solver"], density, internals["dpsi"],
            band_weights=weights, nocc=internals["solver"].nocc,
            commutators=internals["commutators"], **switches,
        ))[0, 3] * E_BOHR2_TO_C_M2

    whole = contracted(field_perturbations=perturbations,
                       field_dvscf=internals["dvscf"])
    without_screening = contracted(field_perturbations=perturbations)
    without_either = contracted()

    assert abs(whole - without_screening) == pytest.approx(
        abs(SCREENED_E14), rel=1e-3
    )
    multiplier = to_voigt(_multiplier_strain_term(
        calculation, internals["solver"], perturbations, weights,
        internals["solver"].nocc,
    ))[0, 3] * E_BOHR2_TO_C_M2
    assert abs(multiplier) > 1e-3
    assert abs(without_screening - without_either) == pytest.approx(
        abs(multiplier), rel=1e-6
    )


def test_the_wedge_completes_and_the_taped_route_is_where_it_did_not():
    """The pair that says a wedge sum of a *product* has been completed.

    On an augmented dataset the density moves with the strain at frozen states,
    so the mixed second derivative carries ``int (drho/d(eps)) K (drho/dE)`` --
    a product of two per-k tangents. A wedge sum of a product is not the product
    of the full-zone objects, so averaging the finished rank-3 tensor cannot
    repair it: one factor has to be made whole *before* it is contracted, which
    is :func:`~defumat.response.born._full_zone_field_response` and is P36's
    rule. The two routes need it in different amounts and that is the point of
    asserting both: the contracted one's screening factor is the field's
    converged ``dvscf``, which ``dielectric_tensor`` mixes from the
    **symmetrised** density response and which is therefore full-zone already,
    while the taped one builds its own from unsymmetrised builders.

    Measured against the code of this morning, on this cell: the taped route
    gave 1.475427270 C/m^2 on the wedge against 1.474377366 on the closed grid,
    **1.05e-03** apart, where the contracted one split by 4.8e-06 -- a factor of
    219, which is what says the defect was the taped route's and not the
    symmetriser's. With the completion in, the two agree **on the wedge** to
    3.4e-08 and both sit 4.8e-06 from the closed grid, which is the residue the
    rank-3 average leaves on the factor that stays a raw wedge sum.

    The closed-grid value is quoted rather than recomputed so this costs one
    ground state; it is the number
    :func:`test_the_two_routes_agree_on_an_augmented_dataset` asserts on the
    cell beside this one.
    """
    taped, contracted = _both_routes(WEDGE)
    # The identity again, and on three k-points it is the sharper of the two
    # statements: this is what was 1.05e-03 before the completion went in.
    assert abs(taped - contracted) < 1e-6, (
        f"the two routes disagree on the wedge by {taped - contracted:.3e} C/m^2"
    )
    # ... and the wedge is the closed grid, which is what the pair exists for.
    assert abs(taped - CLOSED_GRID_E14) < 1e-4
    assert abs(contracted - CLOSED_GRID_E14) < 1e-4
    # Three points, not eight: if the reduction stopped happening the test above
    # would still pass and would be comparing the cell with itself.
    calculation, *_ = _field(WEDGE)
    assert calculation.system.kpoints.nk == 3
