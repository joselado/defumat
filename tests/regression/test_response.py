"""P24: linear response by autodiff -- the velocity operator and what stands on it.

The stages of the phase, each checked against something that shares no
machinery with it:

* **the velocity operator** (:mod:`pypresso.response.velocity`) against a
  central difference of the band structure. ``dH/dk`` comes from one ``jvp`` of
  ``H(k)`` at a frozen sphere and the reference comes from diagonalising at
  ``k +- h``, so the only thing the two have in common is the Hamiltonian
  itself.

The finite-difference reference has one failure mode worth knowing about, and it
is the reference's rather than the operator's: eigenvalues come back **sorted**,
so a step that straddles a band crossing compares two different bands and gives
a difference of order the band width. It is why the step below is small and why
the k-point is a generic one -- at a symmetry point every band is degenerate and
a diagonal expectation value is basis-dependent anyway (rule D4).
"""

from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import jax

from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.response import (
    VelocityOperator,
    local_perturbation,
    make_sternheimer,
)
from pypresso.scf import run_scf
from pypresso.system import build_system
from pypresso.system.kpoints import KPoints
from pypresso.workflows.nscf import fixed_density_states

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"

#: A generic k-point in units of ``2 pi/alat``: no symmetry operation fixes it,
#: so no band is degenerate and every diagonal velocity is well defined.
GENERIC_K = np.array([[0.13, 0.27, 0.41]])

#: Central-difference step in 1/bohr. At ``1e-4`` the truncation error is a few
#: parts in ``1e-7`` of a velocity of order 1 Ry bohr, which is what the
#: comparison below is measuring.
STEP = 1.0e-4

#: How far the operator and the difference may disagree. This is the finite
#: difference's own error, not the operator's -- see the step size above.
VELOCITY_RY_BOHR = 5e-6


@lru_cache(maxsize=None)
def _converged(case: str):
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudo_dir = Path(__file__).resolve().parents[1] / "data" / "pseudo"
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    result = run_scf(system, pseudos, conv_thr=1e-10, max_iterations=80)
    return system, pseudos, result


def _states_at(case: str, coords, nbnd: int = 8):
    """Diagonalise at ``coords`` on the converged density."""
    system, pseudos, result = _converged(case)
    kpoints = KPoints(coords=jnp.asarray(coords), weights=jnp.ones(len(coords)))
    calculation, _, eigenvalues, psi = fixed_density_states(
        system, pseudos, result.density, kpoints=kpoints, nbnd=nbnd,
        conv_thr=1e-12,
    )
    return calculation, np.asarray(eigenvalues), psi


@pytest.mark.parametrize("case", ["si2-nc-force", "si2-us"])
def test_band_velocity_matches_a_finite_difference(case):
    """``<psi|dH/dk - eps dS/dk|psi>`` against ``(eps(k+h) - eps(k-h))/2h``.

    Both cases run, and the ultrasoft one is the point of the pair: ``S(k)``
    is built from the same ``vkb(k)`` the nonlocal potential is, so it carries a
    velocity of its own, and an operator that dropped it would pass on
    ``si2-nc-force`` and fail here.
    """
    system, _, result = _converged(case)
    tpiba = float(system.cell.tpiba)

    calculation, eigenvalues, psi = _states_at(case, GENERIC_K)
    operator = VelocityOperator(
        calculation, calculation.potential(result.density).v_scf
    )
    velocities = operator.band_velocities(psi, eigenvalues).velocities

    reference = np.zeros_like(velocities)
    for axis in range(3):
        step = np.zeros(3)
        # ``coords`` is in units of 2 pi/alat and the step is in 1/bohr.
        step[axis] = STEP / tpiba
        _, plus, _ = _states_at(case, GENERIC_K + step)
        _, minus, _ = _states_at(case, GENERIC_K - step)
        reference[..., axis] = (plus - minus) / (2.0 * STEP)

    assert np.abs(velocities - reference).max() < VELOCITY_RY_BOHR


def test_the_overlap_carries_a_velocity_only_when_it_is_not_the_identity():
    """``dS/dk`` is exactly zero for a norm-conserving dataset and is not for USPP.

    The pair is what makes the ultrasoft comparison above mean something: if
    ``dS/dk`` were being dropped, this test says by how much the band velocity
    would then be wrong.
    """
    directions = jnp.eye(3)

    def largest_ds(case):
        _, _, result = _converged(case)
        calculation, _, psi = _states_at(case, GENERIC_K)
        operator = VelocityOperator(
            calculation, calculation.potential(result.density).v_scf
        )
        return max(
            float(jnp.abs(operator.apply_s(psi, directions[axis])).max())
            for axis in range(3)
        )

    assert largest_ds("si2-nc-force") == 0.0
    assert largest_ds("si2-us") > 1e-3


def test_the_band_velocity_vanishes_at_gamma():
    """A symmetry statement, and the check on the convenience entry point.

    Silicon has an inversion centre, so at ``Gamma`` every eigenstate has a
    definite parity and ``<psi|v|psi>`` is zero exactly. Nothing in the operator
    knows that: it is built from ``dH/dk`` at a k-point like any other, and what
    is left over is the eigensolver's own tolerance. It also exercises
    :func:`~pypresso.response.band_velocities`, which is the entry point the
    README names and which the tests above bypass.
    """
    from pypresso.response import band_velocities as compute_velocities
    from pypresso.scf import Calculation

    system = build_system(read_pw_input(CASES / "si2-nc-force.in"))
    pseudo_dir = Path(__file__).resolve().parents[1] / "data" / "pseudo"
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-10,
                     max_iterations=80)

    gamma = KPoints(coords=jnp.zeros((1, 3)), weights=jnp.ones(1))
    velocities = compute_velocities(calculation, result, kpoints=gamma).velocities
    assert np.abs(velocities).max() < 1e-3


# ---------------------------------------------------------------------------
# Stage 2: the Sternheimer solve, and the susceptibility built from it.
# ---------------------------------------------------------------------------

#: The perturbing potential used below: ``cos(2 pi G.r)`` for one short
#: reciprocal-lattice vector. A single Fourier component is the cleanest probe
#: there is -- it is smooth, real, periodic, and it breaks the crystal's
#: symmetry, which is what makes the comparison below a comparison of
#: *unsymmetrised* densities on both sides.
PROBE_MILLER = (1, 0, 0)

#: How far ``chi_0 dV`` may sit from a central difference of the density. The
#: bound is the difference's own floor -- see the test.
CHI0_RELATIVE = 1e-5

#: How far ``chi_0 K`` may sit from P22's finite-difference SCF Jacobian, at
#: that difference's own optimal step. The two share no machinery at all.
JACOBIAN_RELATIVE = 2e-3

#: The step at which P22's ``jvp_finite_difference`` is most accurate on this
#: cell -- the bottom of the usual U between truncation and noise, measured
#: rather than assumed (0.3 -> 8.3e-2, 0.1 -> 9.7e-3, 0.03 -> 8.0e-4,
#: **0.01 -> 4.0e-4**, 0.003 -> 1.0e-3). The default step is chosen for a
#: gradient rather than for this comparison and is two orders below it, deep in
#: the noise, where the same numbers disagree by 11%.
JACOBIAN_STEP = 1e-2


def _probe_potential(calculation):
    """``cos(2 pi G.r)`` on the dense grid, ``(nspin, n1, n2, n3)``."""
    grid = calculation.basis.dense.grid
    axes = [np.arange(n) / n for n in grid]
    positions = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    field = np.cos(2.0 * np.pi * (positions @ np.asarray(PROBE_MILLER)))
    return jnp.asarray(field[None])


@lru_cache(maxsize=None)
def _silicon():
    """The reference two-atom silicon of ``pw_scf/scf.in``, converged tightly."""
    from pypresso.scf import Calculation

    testsuite = (
        Path(__file__).resolve().parents[2]
        / "quantum_espresso" / "qe-7.5-ReleasePack" / "qe-7.5" / "test-suite"
    )
    if not testsuite.is_dir():
        pytest.skip("QE reference tree not present")
    system = build_system(read_pw_input(testsuite / "pw_scf" / "scf.in"))
    pseudo_dir = Path(__file__).resolve().parents[1] / "data" / "pseudo"
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12)
    return system, pseudos, calculation, result


def test_chi0_matches_a_finite_difference_of_the_density():
    """``chi_0 dV`` against ``(rho[V + h dV] - rho[V - h dV]) / 2h``.

    The two share the Hamiltonian and nothing else: one solves a projected
    linear system per occupied band, the other diagonalises twice. Neither side
    symmetrises -- the probe potential breaks the crystal's point group, so a
    symmetrised comparison would be comparing two different quantities.
    """
    from pypresso.basis.interpolate import to_dense
    from pypresso.scf.density import sum_band

    system, _, calculation, result = _silicon()
    solver = make_sternheimer(calculation, result)
    dv = _probe_potential(calculation)

    solution = solver.solve(local_perturbation(calculation, dv))
    assert solution.converged
    drho = np.asarray(solver.response_density(solution.dpsi))

    smooth, dense = calculation.basis.smooth, calculation.basis.dense
    v_scf = calculation.potential(result.density).v_scf
    nbnd = result.wavefunctions.shape[2]

    def density_at(scale):
        hamiltonians = calculation.hamiltonian(v_scf + scale * dv)
        eigenvalues, psi = calculation.diagonalize(hamiltonians, nbnd, None, 1e-13)
        weights, _ = calculation.occupations(eigenvalues)
        rho = sum_band(
            psi, calculation.fft_index, smooth.grid, jnp.asarray(weights),
            system.cell, calculation.k_batch,
        )
        return np.asarray(to_dense(rho, smooth, dense))

    step = 1e-4
    reference = (density_at(step) - density_at(-step)) / (2.0 * step)
    relative = np.abs(drho - reference).max() / np.abs(drho).max()
    assert relative < CHI0_RELATIVE


def test_the_exact_jacobian_agrees_with_the_finite_difference_one():
    """P22c: ``chi_0 K`` is the SCF Jacobian P22 could only difference.

    ``F`` maps a density to the density its Hamiltonian produces, so
    ``dF/drho = chi_0 K`` with ``K = dV_scf/drho`` -- and ``K`` is free, being
    one ``jvp`` of ``v_of_rho`` (rule D1). The comparison is against
    :meth:`~pypresso.scf.residual.ScfResidual.jvp_finite_difference`, which
    evaluates ``F`` twice and shares nothing with the linear solve.

    ``F`` symmetrises its output density and on a symmetry-reduced k-set that is
    not a no-op, so the exact route has to symmetrise too.
    """
    from pypresso.scf.residual import make_residual

    _, _, calculation, result = _silicon()
    rho = jnp.asarray(result.density)

    generator = np.random.default_rng(0)
    direction = generator.standard_normal(rho.shape)
    # A density perturbation conserves the electron count, which is the only
    # direction the SCF Jacobian is ever asked in.
    direction -= direction.mean()
    direction = jnp.asarray(direction / np.linalg.norm(direction))

    solver = make_sternheimer(calculation, result)
    _, kernel = jax.jvp(
        lambda r: calculation.potential(r).v_scf, (rho,), (direction,)
    )
    exact = np.asarray(calculation.symmetrize(
        solver.response_density(
            solver.solve(local_perturbation(calculation, kernel)).dpsi
        )
    ))

    nbnd = result.wavefunctions.shape[2]
    residual = make_residual(calculation, nbnd, ethr=1e-13)
    packed = residual.pack(np.asarray(result.density), result.becsum, result.ns)
    flat = np.asarray(direction).ravel()
    # ``r = F - x``, so ``dF/dx . v`` is ``J v + v``.
    reference = (
        residual.jvp_finite_difference(
            packed, flat, result.wavefunctions, epsilon=JACOBIAN_STEP
        )
        + flat
    ).reshape(rho.shape)

    relative = np.abs(exact - reference).max() / np.abs(exact).max()
    assert relative < JACOBIAN_RELATIVE


@pytest.mark.parametrize(
    "case, expected",
    [
        ("si2-us", "ultrasoft"),
        ("al-tetrahedra", "occupations"),
    ],
)
def test_the_regimes_without_a_response_here_are_refused_by_name(case, expected):
    """A response this module cannot compute raises rather than approximating.

    Both would otherwise be silently wrong: an ultrasoft ``drho`` missing the
    augmentation charge's own response looks entirely plausible, and the
    insulator projector applied to a metal is the wrong operator with no symptom.
    """
    from pypresso.scf import Calculation

    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudo_dir = Path(__file__).resolve().parents[1] / "data" / "pseudo"
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    with pytest.raises(NotImplementedError, match=expected):
        make_sternheimer(calculation, None)


# ---------------------------------------------------------------------------
# Stage 3: the electric field -- epsilon_infinity and the Born charges.
# ---------------------------------------------------------------------------

#: What ``ph_base/si.phG.in``'s committed benchmark prints for this cell, in
#: cartesian axes. The input is copied to ``si-epsilon.in`` so these tests run
#: without the vendored tree; the numbers are the benchmark's.
QE_EPSILON = 13.806375297
QE_BORN = -0.07568
QE_TOTAL_ENERGY = -15.84452726

#: How far the dielectric constant may sit from QE's. The measured difference is
#: 2.7e-4 -- two parts in 1e5 -- and what is left is the same thing that puts a
#: floor under the eigenvalues (``tests/tolerances.py``): QE interpolates every
#: radial form factor from a ``dq = 0.01`` table where this code integrates it
#: directly.
EPSILON_TOLERANCE = 1e-3

#: The Born charges are printed to five decimals, so this is the last digit.
BORN_TOLERANCE = 1e-4

#: The tensor is cubic by symmetry and nothing here imposes that, so its
#: departure from a scalar is round-off.
CUBIC_TOLERANCE = 1e-9


@lru_cache(maxsize=None)
def _dielectric(case: str):
    """The electric-field response of one of the committed silicon inputs."""
    from pypresso.response.efield import dielectric_tensor
    from pypresso.scf import Calculation

    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudo_dir = Path(__file__).resolve().parents[1] / "data" / "pseudo"
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                     max_iterations=80)
    response = dielectric_tensor(
        calculation, result.wavefunctions, result.eigenvalues, result.density,
        born_charges=(case == "si-epsilon"),
    )
    return result, response


def test_the_dielectric_constant_matches_quantum_espresso():
    """``epsilon_infinity`` of silicon against ``ph_base/si.phG.in``'s benchmark.

    Everything on this side comes from differentiating: the commutator that
    makes ``P_c r|psi>`` computable is ``dH/dk`` from a ``jvp`` (stage 1), the
    screening kernel ``dv_of_drho`` is a ``jvp`` of ``v_of_rho``, and the
    exchange-correlation kernel QE tabulates in ``setup_dmuxc`` is the second
    derivative of the energy this code writes down once. What is transcribed is
    the linear solve, the projector and the assembly.
    """
    result, response = _dielectric("si-epsilon")
    assert result.total_energy == pytest.approx(QE_TOTAL_ENERGY, abs=1e-7)
    assert response.converged
    assert response.isotropic == pytest.approx(QE_EPSILON, abs=EPSILON_TOLERANCE)


def test_the_dielectric_tensor_comes_out_cubic():
    """Nothing here imposes the crystal class, so this is a measurement.

    ``symmetrize_directional`` and ``symmatrix`` average over the group, which
    is not the same as projecting onto the cubic form: a wrong rotation
    convention, a missing fractional translation or an axial sign where a polar
    one belongs would all survive the average and show up here.
    """
    _, response = _dielectric("si-epsilon")
    assert response.anisotropy < CUBIC_TOLERANCE


def test_the_born_charges_match_quantum_espresso():
    """``Z*`` against the same benchmark -- the sharper of the two numbers.

    Silicon's Born charge is zero by symmetry in a converged calculation, so
    what ``-0.07568`` measures is the residue of a difference of large numbers:
    ``Z_val = 4`` against an electronic part near ``4.076``. Reproducing it to
    the printed digits means the bare displacement perturbation, the
    self-consistent field response and the weights are all right.
    """
    _, response = _dielectric("si-epsilon")
    charges = response.born_charges
    assert charges.shape == (2, 3, 3)
    for atom in range(2):
        diagonal = np.diag(charges[atom])
        assert np.allclose(diagonal, QE_BORN, atol=BORN_TOLERANCE)
        off = charges[atom] - np.diag(diagonal)
        assert np.abs(off).max() < CUBIC_TOLERANCE


@pytest.mark.slow
def test_the_symmetrised_wedge_and_the_closed_grid_give_one_answer():
    """The two routes to a response, on the same k-sample and sharing nothing.

    A **shifted** Monkhorst-Pack grid is not closed under the point group, so
    the wedge route needs ``symdvscf``'s average and a whole-grid route is not
    available. An **unshifted** grid *is* closed: the same sample can be run
    reduced to 8 points with the response symmetrised, or whole at 64 points
    with the symmetrisation switched off entirely. Nothing is shared between the
    two but the Sternheimer solve, so agreeing is the check on
    ``symmetrize_directional`` -- and it is the only check there is, since QE
    computes only the first of them.
    """
    reduced, wedge = _dielectric("si-epsilon-unshifted")
    whole, closed = _dielectric("si-epsilon-unshifted-nosym")

    assert reduced.total_energy == pytest.approx(whole.total_energy, abs=1e-9)
    assert wedge.isotropic == pytest.approx(closed.isotropic, abs=1e-8)
    assert wedge.anisotropy < CUBIC_TOLERANCE
    # The unsymmetrised route has no group to make it cubic; the grid does.
    assert closed.anisotropy < CUBIC_TOLERANCE


def test_a_shifted_grid_with_no_symmetry_is_refused():
    """The combination that is silently wrong, and how it was found.

    With ``nosym`` there is no group to average the response over, so the grid
    has to carry the symmetry itself -- and a shifted one does not. Run anyway,
    this cell gives a dielectric tensor with a diagonal of 13.848 and
    off-diagonal entries of 3.77 that cubic symmetry forbids. It is P6's trap
    (``si2-nc-shifted-nosym.in``) in a second place.
    """
    from pypresso.response.efield import dielectric_tensor
    from pypresso.scf import Calculation

    system = build_system(read_pw_input(CASES / "si2-nc-shifted-nosym.in"))
    pseudo_dir = Path(__file__).resolve().parents[1] / "data" / "pseudo"
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    with pytest.raises(NotImplementedError, match="shifted"):
        dielectric_tensor(calculation, None, np.zeros((1, 1, 1)), None)
