"""P35: the Raman tensor as a mixed third derivative, and what it is refused for.

The interesting thing about validating this phase is that **the reference it
would naturally use is broken**. QE reaches the Raman tensor through ``ph.x``
with ``lraman``, and the vendored 7.5 build does not reproduce its own committed
example (``PHonon/examples/example05``, generated with v6.0): -1.8681 against
-0.78497 for the Raman tensor, 157.87 against 40.4578 for the electro-optic one.
Its own internal check fails too -- ``dhdrhopsi`` prints the dielectric constant
implied by its finite-difference k-derivative beside the analytic one, and where
v6.0 has 8.8116 against 8.8147 the vendored build gives **-0.288** against
8.8143. Tightening ``eth_rps``/``eth_ns`` by four orders moves it by 1e-2.

What validates the phase instead is a **finite difference of the dielectric
tensor over re-converged displaced geometries**, which shares nothing with the
third derivative but the linear response underneath both -- and which P24
validated against ``ph.x`` independently. It is the route P26 used for
``d(chi)/d(strain)`` for the same reason.

The second half of the file is about the tensor that is **refused**. ``chi^(2)``
would be the same functional differentiated along a third *field*, and this code
has no ``dH/dE`` to differentiate: the field lives in the source term and in the
density. The tests below measure how much that is worth (42% of the answer, via
its displacement counterpart) and -- the part worth having in a test suite --
that **every symmetry check passes without it**, which is why the refusal is by
name rather than by tolerance.
"""

from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.response.efield import dielectric_tensor
from pypresso.response.electrostriction import _project_conduction, refined_states
from pypresso.response.nonlinear import (
    permutation_asymmetry,
    raman_tensors,
    require_a_complete_third_derivative,
    susceptibility_displacement_derivative,
    susceptibility_field_derivative,
    translational_residue,
)
from pypresso.response.phonon import _bare_displacements, self_consistent_response
from pypresso.scf import Calculation, run_scf
from pypresso.system import build_system

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: Nothing imposes the crystal class or the permutation symmetry -- the runs are
#: ``nosym`` on a closed grid and no tensor is averaged anywhere -- so what a
#: symmetry forbids is round-off. Measured: 3e-13 relative on both cases.
EXACT_TOLERANCE = 1e-10

#: The finite-difference reference is the loose one, not the analytic route.
#: Measured **1.0e-5** relative at ``h = 0.02`` bohr on both atoms; the
#: tolerance is a factor of 20 above it, and deliberately not the comfortable 5%
#: a finite difference invites -- the trap this machinery already has a name for
#: (P26's frozen ``u`` and its moving constraint surface) was worth 2%, so a
#: test that admits 2% is one that would have passed with it in place.
FINITE_DIFFERENCE_TOLERANCE = 2e-4

#: ``max |sum_atoms d(eps)/d(tau)|`` over the tensors' own scale. Measured
#: 2.8e-4 on AlAs; the vendored ``ph.x`` gives 0.43 on the same input.
SUM_RULE_TOLERANCE = 1e-3

#: The displacement for the finite difference, in bohr.
FD_STEP = 0.02


@lru_cache(maxsize=None)
def _converged(case: str):
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                     max_iterations=100)
    return system, pseudos, calculation, result


@lru_cache(maxsize=None)
def _raman(case: str):
    _, _, calculation, result = _converged(case)
    return raman_tensors(calculation, result)


@lru_cache(maxsize=None)
def _first_order(case: str):
    """The two first-order responses and the solver, shared by the tests below."""
    _, _, calculation, result = _converged(case)
    eigenvalues, psi = refined_states(calculation, result)
    density = jnp.asarray(result.density)
    field = dielectric_tensor(
        calculation, psi, eigenvalues, density,
        born_charges=False, keep_internals=True,
    )
    internals = field.internals
    solver = internals["solver"]
    return (calculation, density, solver,
            _project_conduction(solver.psi, jnp.stack(internals["bare"])),
            _project_conduction(solver.psi, jnp.stack(internals["dpsi"])),
            jnp.asarray(field.induced_density), internals)


@lru_cache(maxsize=None)
def _field_derivative(case: str):
    calculation, density, solver, b, u, drho, _ = _first_order(case)
    return susceptibility_field_derivative(
        calculation, solver, density, b, u, drho
    )


@lru_cache(maxsize=None)
def _epsilon_displaced(case: str, atom: int, cart: int, step: float):
    """``eps`` at a displaced geometry, re-converged from scratch."""
    system, pseudos, _, _ = _converged(case)
    positions = np.asarray(system.structure.positions).copy()
    positions[atom, cart] += step
    moved = Calculation(system, pseudos).at_positions(jnp.asarray(positions))
    result = run_scf(system, pseudos, calculation=moved, conv_thr=1e-12,
                     max_iterations=200)
    eigenvalues, psi = refined_states(moved, result)
    tensor = dielectric_tensor(
        moved, psi, eigenvalues, jnp.asarray(result.density), born_charges=False,
    )
    return np.asarray(tensor.epsilon)


# -- the ground state the whole phase stands on -------------------------------


def test_the_alas_ground_state_matches_pw_x():
    """The case is new, so the SCF underneath it gets checked before anything else.

    ``reference.out.alas-raman`` is the vendored ``pw.x`` on the same input. The
    ground state is the half of QE's Raman calculation that has *not* regressed
    -- ``ph.x``'s linear response agrees here too (12.9673 against 12.9674) --
    which is what makes it possible to say that what has is the third-derivative
    branch specifically.
    """
    from pypresso.io.qeref import read_qe_output

    _, _, _, result = _converged("alas-raman")
    reference = read_qe_output(CASES / "reference.out.alas-raman")
    assert result.total_energy == pytest.approx(reference.total_energy, abs=1e-7)


# -- the Raman tensor against a finite difference -----------------------------


@pytest.mark.parametrize("atom", [0, 1])
def test_the_raman_tensor_matches_a_finite_difference(atom):
    """``d(eps)/d(tau)`` against a central difference of ``eps`` itself.

    The end-to-end reference of the phase, and the only one there is (module
    docstring). The two routes share the linear response and nothing else: this
    one differentiates the variational second-order energy once more, that one
    re-converges the SCF at two displaced geometries and solves the dielectric
    response at each.
    """
    result = _raman("alas-raman")
    plus = _epsilon_displaced("alas-raman", atom, 0, FD_STEP)
    minus = _epsilon_displaced("alas-raman", atom, 0, -FD_STEP)
    reference = (plus - minus) / (2 * FD_STEP)

    analytic = result.raman[atom, 0]
    scale = np.abs(analytic).max()
    assert scale > 1.0
    error = np.abs(analytic - reference).max() / scale
    assert error < FINITE_DIFFERENCE_TOLERANCE, (
        f"analytic {analytic[1, 2]:.6f} against finite difference "
        f"{reference[1, 2]:.6f} ({error:.2e} relative)"
    )


def test_the_raman_tensors_obey_the_translational_sum_rule():
    """Translating the crystal cannot change ``eps``, so the sum over atoms is 0.

    P25's acoustic sum rule one derivative up, and like it **reported rather
    than imposed** -- the residue measures the responses' own convergence. The
    vendored ``ph.x`` violates it by 43% on this input, which is one of the two
    things that showed its Raman branch has regressed.
    """
    result = _raman("alas-raman")
    scale = np.abs(result.raman).max()
    assert translational_residue(result.raman) < SUM_RULE_TOLERANCE * scale
    assert result.sum_rule_relative < SUM_RULE_TOLERANCE


def test_the_raman_tensors_have_the_zincblende_form():
    """``T_d`` leaves one independent component, and the run imposes none of it.

    AlAs is ``-43m``: the only non-vanishing components of a rank-3 tensor are
    those with all three indices different, and they are all equal. The
    calculation is ``nosym`` on the whole closed grid with no average applied
    anywhere, so this is a measurement of every index convention in the phase at
    once -- which is what the cubic form of ``d(chi)/dx`` was for P26.
    """
    tensors = _raman("alas-raman").raman
    for atom in range(tensors.shape[0]):
        reference = tensors[atom, 0, 1, 2]
        assert abs(reference) > 1.0
        for cart in range(3):
            for i in range(3):
                for j in range(3):
                    entry = tensors[atom, cart, i, j]
                    if len({cart, i, j}) == 3:
                        assert abs(abs(entry) - abs(reference)) < 1e-9 * abs(reference)
                    else:
                        assert abs(entry) < EXACT_TOLERANCE * abs(reference)


def test_the_raman_tensor_of_silicon_is_the_one_nonzero_third_derivative():
    """Silicon has no ``chi^(2)`` and does have a Raman tensor.

    The two statements are the same symmetry read twice: an inversion centre
    kills a tensor odd in the field, and the optical mode of the diamond
    structure is Raman-active because the two atoms move *against* each other,
    which the inversion does not leave alone. It is worth a test because it is
    the case where a phase that silently returned zero would look plausible.
    """
    result = _raman("si-epsilon-unshifted-nosym")
    assert np.abs(result.raman[0, 0, 1, 2]) > 1.0
    assert result.sum_rule_relative < SUM_RULE_TOLERANCE


# -- the tensor that is refused, and why the refusal is by name ---------------


def test_the_explicit_operator_term_is_a_large_part_of_the_answer():
    """Zeroing ``dH/d(parameter)`` in the Raman derivative is worth 42%.

    This is the *measurement* behind the refusal of ``chi^(2)``: the field
    derivative has no ``dH/dE`` to carry, and this puts the displacement
    derivative -- the one validated against a finite difference above -- in
    exactly that position. It moves ``d(eps_yz)/d(tau)`` from -3.1183 to
    -1.8100, so what the field derivative is missing is of the same order as
    everything it does compute.
    """
    calculation, density, solver, b, u, _, internals = _first_order("alas-raman")
    positions = jnp.asarray(calculation.system.structure.positions)
    bare = _bare_displacements(calculation, solver, internals["v_scf"], positions)
    dpsi, drho, *_ = self_consistent_response(calculation, solver, bare, density)
    without = susceptibility_displacement_derivative(
        calculation, solver, density, b, u, positions, dpsi, drho,
        geometry_tangent=False,
    )
    full = _raman("alas-raman").raman
    ratio = abs(full[0, 0, 1, 2] / without[0, 0, 1, 2])
    assert 1.5 < ratio < 2.0, f"the omitted term changes the answer by {ratio:.3f}"


def test_the_incomplete_field_derivative_passes_every_symmetry_check():
    """And that is the point: symmetry cannot see the term that is missing.

    ``chi^(2)`` is refused because a term of the 2n+1 expression has nothing in
    this code to build it from -- **not** because a check failed. This test
    records that the checks pass anyway: the incomplete tensor vanishes
    identically in a centrosymmetric crystal, comes out exactly zincblende in
    AlAs, and is symmetric under every permutation of its three labels. A phase
    that shipped it on the strength of those would have been wrong by 40%.
    """
    silicon = _field_derivative("si-epsilon-unshifted-nosym")
    assert np.abs(silicon).max() < EXACT_TOLERANCE

    alas = _field_derivative("alas-raman")
    scale = np.abs(alas).max()
    assert scale > 1.0
    assert permutation_asymmetry(alas) < EXACT_TOLERANCE * scale
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if len({i, j, k}) != 3:
                    assert abs(alas[i, j, k]) < EXACT_TOLERANCE * scale


def test_chi2_is_refused_by_name():
    """With the missing term named, in the project's usual arrangement."""
    with pytest.raises(NotImplementedError, match="dvpsi_e2"):
        require_a_complete_third_derivative()


def test_a_symmetry_reduced_kset_is_refused():
    """Two field labels and an atom need a rank-3 average that is not written.

    P26's refusal, reached through the same function and for the same reason:
    ``symme.f90`` has ``symtensor3`` for exactly this, and until it is here the
    wedge sum is incomplete. The escape is the one P24 already uses -- an
    unshifted grid is closed under the point group, so running it whole needs no
    average at all.
    """
    _, _, calculation, result = _converged("si-epsilon")
    with pytest.raises(NotImplementedError, match="rank-3"):
        raman_tensors(calculation, result)
