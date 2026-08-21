"""P24: linear response by autodiff -- the velocity operator and what stands on it.

The stages of the phase, each checked against something that shares no
machinery with it:

* **the velocity operator** (:mod:`pypresso.response.velocity`) against a
  central difference of the band structure. ``dH/dk`` comes from one ``jvp`` of
  ``H(k)`` at a frozen sphere and the reference comes from diagonalising at
  ``k +- h``, so the only thing the two have in common is the Hamiltonian
  itself. It is checked once more against a *symmetry* statement, which shares
  nothing with either: at ``Gamma`` an inversion-symmetric crystal has states of
  definite parity, so every band velocity is exactly zero.
* **the Sternheimer solve** (:mod:`pypresso.response.sternheimer`) against a
  central difference of the density under the same perturbation, and its
  composition with the screening kernel against P22's own finite-difference SCF
  Jacobian -- at *that* difference's optimal step, which is not its default one.
* **the electric field** (:mod:`pypresso.response.efield`) against the
  **vendored** ``ph.x``, regenerated -- ``ph_base``'s committed benchmark dates
  from release 6.0 and has drifted by six times the disagreement being measured
  -- on norm-conserving, ultrasoft and PAW silicon and on ultrasoft carbon; and
  against itself on the one k-grid where the symmetrisation it needs can be
  switched off.

The finite-difference reference has one failure mode worth knowing about, and it
is the reference's rather than the operator's: eigenvalues come back **sorted**,
so a step that straddles a band crossing compares two different bands and gives
a difference of order the band width. It is why the step below is small and why
the k-point is a generic one -- at a symmetry point every band is degenerate and
a diagonal expectation value is basis-dependent anyway (rule D4).
"""

from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.response import VelocityOperator, make_sternheimer
from pypresso.scf import run_scf
from pypresso.system import build_system
from pypresso.system.kpoints import KPoints

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
    """Diagonalise at ``coords`` on the converged density.

    Not through :func:`fixed_density_states`, which refuses PAW because *it*
    cannot rebuild ``becsum`` from a density. Here the converged ``becsum`` is in
    hand, so the one-centre coefficients are built from it and the same helper
    serves all three datasets.
    """
    system, pseudos, result = _converged(case)
    from pypresso.scf import Calculation

    calculation = Calculation(system, pseudos)
    kpoints = KPoints(coords=jnp.asarray(coords), weights=jnp.ones(len(coords)))
    moved = calculation.at_kpoints(kpoints)
    v_scf = moved.potential(result.density).v_scf
    _, ddd_paw = moved.onecenter(result.becsum)
    eigenvalues, psi = moved.diagonalize(
        moved.hamiltonian(v_scf, ddd_paw), nbnd, None, 1e-13
    )
    return moved, np.asarray(eigenvalues), psi


@pytest.mark.parametrize("case", ["si2-nc-force", "si2-us", "si2-paw"])
def test_band_velocity_matches_a_finite_difference(case):
    """``<psi|dH/dk - eps dS/dk|psi>`` against ``(eps(k+h) - eps(k-h))/2h``.

    Three datasets, each adding one thing. The **ultrasoft** case is the point of
    the first pair: ``S(k)`` is built from the same ``vkb(k)`` the nonlocal
    potential is, so it carries a velocity of its own, and an operator that
    dropped it would pass on ``si2-nc-force`` and fail here. The **PAW** case
    adds the one-centre coefficients, which are not a function of the density and
    which multiply ``vkb(k)`` -- see
    :func:`test_paw_without_its_one_centre_coefficients_is_refused` for what
    leaving them out costs.
    """
    system, _, result = _converged(case)
    tpiba = float(system.cell.tpiba)

    calculation, eigenvalues, psi = _states_at(case, GENERIC_K)
    _, ddd_paw = calculation.onecenter(result.becsum)
    operator = VelocityOperator(
        calculation, calculation.potential(result.density).v_scf, ddd_paw
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
        _, ddd_paw = calculation.onecenter(result.becsum)
        operator = VelocityOperator(
            calculation, calculation.potential(result.density).v_scf, ddd_paw
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


def test_paw_without_its_one_centre_coefficients_is_refused():
    """The silent 2% error the guard exists to prevent, measured.

    PAW's ``ddd_paw`` is built from ``becsum`` -- a property of the
    wavefunctions, not recoverable from the density -- and it multiplies
    ``vkb(k)``, so it is part of ``dH/dk`` and not only of ``H``. Omitting it
    leaves a velocity that is wrong by 2% and looks entirely ordinary, which is
    why the constructor raises instead. This test does both halves: that it
    raises, and that what it is refusing is worth refusing.
    """
    _, _, result = _converged("si2-paw")
    calculation, eigenvalues, psi = _states_at("si2-paw", GENERIC_K)
    v_scf = calculation.potential(result.density).v_scf
    _, ddd_paw = calculation.onecenter(result.becsum)
    assert ddd_paw is not None

    with pytest.raises(ValueError, match="one-centre"):
        VelocityOperator(calculation, v_scf)

    # What the refusal is worth: the same operator built by hand without them.
    right = VelocityOperator(calculation, v_scf, ddd_paw)
    wrong = object.__new__(VelocityOperator)
    wrong.__dict__.update(right.__dict__)
    wrong.ddd_paw = None
    difference = np.abs(
        right.band_velocities(psi, eigenvalues).velocities
        - wrong.band_velocities(psi, eigenvalues).velocities
    ).max()
    assert difference > 1e-3


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
#: bound is the difference's own floor -- measured at 2.6e-7 norm-conserving,
#: 8.8e-7 ultrasoft and 1.4e-6 PAW.
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


@pytest.mark.parametrize("case", ["si2-nc-force", "si2-us", "si2-paw"])
def test_chi0_matches_a_finite_difference_of_the_density(case):
    """``chi_0 dV`` against ``(rho[V + h dV] - rho[V - h dV]) / 2h``.

    The two share the Hamiltonian and nothing else: one solves a projected
    linear system per occupied band, the other diagonalises twice. Neither side
    symmetrises -- the probe potential breaks the crystal's point group, so a
    symmetrised comparison would be comparing two different quantities.

    All three datasets, and the last two are the point: an ultrasoft response
    carries ``dbecsum`` and the augmentation charge's own response inside
    ``drho`` and ``int3`` inside the perturbation, and none of those is
    transcribed -- they come from differentiating the density builder and
    ``newd``, which already knew about them. The reference builds its density the
    same way, so it carries them too.
    """
    from pypresso.basis.interpolate import to_dense
    from pypresso.scf import Calculation
    from pypresso.scf.density import becsum as becsum_of, sum_band

    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudo_dir = Path(__file__).resolve().parents[1] / "data" / "pseudo"
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                     max_iterations=80)

    solver = make_sternheimer(calculation, result)
    dv = _probe_potential(calculation)
    solution = solver.solve(solver.perturbation(dv))
    assert solution.converged
    drho = np.asarray(solver.response_density(solution.dpsi))

    smooth, dense = calculation.basis.smooth, calculation.basis.dense
    v_scf = calculation.potential(result.density).v_scf
    _, ddd_paw = calculation.onecenter(result.becsum)
    nbnd = result.wavefunctions.shape[2]
    weights = solver.weights

    def density_at(scale):
        hamiltonians = calculation.hamiltonian(v_scf + scale * dv, ddd_paw)
        _, psi = calculation.diagonalize(hamiltonians, nbnd, None, 1e-13)
        psi = psi[:, :, : solver.nocc]
        # The *unsymmetrised* becsum, which is what the response uses: the probe
        # potential breaks the crystal's point group, so PAW's ``PAW_symmetrize``
        # would average this reference over operations the perturbed system does
        # not have. Comparing against it instead is worth 0.45 relative -- which
        # is what this line looked like before, and it failed loudly rather than
        # quietly, because the two sides then compute different quantities.
        becsum_ = becsum_of(
            psi, calculation.projectors.vkb, weights,
            calculation.species_channels, calculation.k_batch,
        ) if calculation.is_ultrasoft else ()
        rho = sum_band(
            psi, calculation.fft_index, smooth.grid, weights,
            system.cell, calculation.k_batch,
        )
        return np.asarray(
            calculation.augmented(to_dense(rho, smooth, dense), becsum_)
        )

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
        solver.response_density(solver.solve(solver.perturbation(kernel)).dpsi)
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


@pytest.mark.parametrize("case, expected", [("al-tetrahedra", "occupations")])
def test_the_regimes_without_a_response_here_are_refused_by_name(case, expected):
    """A response this module cannot compute raises rather than approximating.

    The insulator projector applied to a metal is the wrong operator with no
    symptom at all: ``orthogonalize``'s smearing branch replaces the sharp
    projector with occupation-difference weights, and the Fermi level shifts.
    (Ultrasoft and PAW *were* on this list and are not any more -- they are
    implemented and checked above.)
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

#: What the **vendored** ``ph.x`` prints for each case, regenerated with
#: ``epsil = .true.`` and committed as ``reference.out.ph-*``. The committed
#: ``ph_base`` benchmarks are *not* used: they date from release 6.0 and have
#: drifted -- 13.806375297 against 13.806689470 on silicon, 5.756035041 against
#: 5.756181864 on carbon -- which is exactly what
#: :func:`tests.conftest.reference_output` already documents for ``pw.x``.
QE_DIELECTRIC = {
    "si-epsilon": 13.806689470,
    "si-epsilon-us": 14.325269631,
    "si-epsilon-paw": 14.320176984,
    "c-epsilon": 5.756181864,
}
QE_BORN = -0.07571
QE_TOTAL_ENERGY = -15.84452726

#: How far the dielectric constant may sit from ``ph.x``'s. The measured
#: differences are 4.3e-5, 5.2e-5, 3.4e-5 and 1.2e-4, and what is left is the
#: same thing that puts a floor under the eigenvalues
#: (``tests/tolerances.py``): QE interpolates every radial form factor from a
#: ``dq = 0.01`` table where this code integrates it directly.
EPSILON_TOLERANCE = 5e-4

#: The Born charges are printed to five decimals, so this is the last digit.
BORN_TOLERANCE = 1e-4

#: The tensor is cubic by symmetry and nothing here imposes that, so its
#: departure from a scalar is round-off.
CUBIC_TOLERANCE = 1e-9


@lru_cache(maxsize=None)
def _dielectric(case: str):
    """The electric-field response of one of the committed inputs."""
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
        result.becsum,
        # Born charges are norm-conserving only, and refused by name otherwise.
        born_charges=not calculation.is_ultrasoft,
    )
    return result, response


@pytest.mark.parametrize("case", list(QE_DIELECTRIC))
def test_the_dielectric_constant_matches_quantum_espresso(case):
    """``epsilon_infinity`` against the vendored ``ph.x``, on all three datasets.

    Everything on this side comes from differentiating: the commutator that
    makes ``P_c r|psi>`` computable is ``dH/dk`` from a ``jvp``, the screening
    kernel ``dv_of_drho`` is a ``jvp`` of ``v_of_rho``, the response density
    (``incdrhoscf`` + ``addusdbec`` + ``lr_addusddens``) is a ``jvp`` of the
    density builder, ``int3`` is a ``jvp`` of ``newd``, and ``PAW_dpotential`` is
    a ``jvp`` of ``onecenter``. What is transcribed is the linear solve, the
    projector, the augmentation dipole and the assembly.

    The four cases add one thing each: **norm-conserving silicon** is the base;
    **ultrasoft silicon** adds the augmentation charge to all of `drho`, `D_ij`
    and the position operator; **PAW silicon** adds the one-centre terms on top;
    and **ultrasoft carbon** is the independent check -- a different element, a
    different cutoff pair and a different lattice constant, so agreeing on it is
    not agreeing twice on the same arithmetic.
    """
    result, response = _dielectric(case)
    if case == "si-epsilon":
        assert result.total_energy == pytest.approx(QE_TOTAL_ENERGY, abs=1e-7)
    assert response.converged
    assert response.isotropic == pytest.approx(
        QE_DIELECTRIC[case], abs=EPSILON_TOLERANCE
    )


@pytest.mark.parametrize("case", list(QE_DIELECTRIC))
def test_the_dielectric_tensor_comes_out_cubic(case):
    """Nothing here imposes the crystal class, so this is a measurement.

    ``symmetrize_directional``, ``PAW_dusymmetrize`` and ``symmatrix`` average
    over the group, which is not the same as projecting onto the cubic form: a
    wrong rotation convention, a missing fractional translation, or an axial sign
    where a polar one belongs would all survive the average and show up here.
    """
    _, response = _dielectric(case)
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


def test_born_charges_are_refused_for_an_ultrasoft_dataset():
    """The one thing the ultrasoft path does *not* do, refused rather than returned.

    ``zstar_eu.f90`` is the whole story for a norm-conserving dataset;
    ``zstar_eu_us.f90`` adds five stages for an ultrasoft one. Without them the
    norm-conserving expression is wrong in sign as well as size -- **+0.1625**
    against ``ph.x``'s **-0.07945** on this cell -- while the dielectric constant
    from the *same* run is right to 5e-5. Two quantities out of one field
    response, one of them complete and one not, is exactly the situation a
    refusal is for.
    """
    from pypresso.response.efield import dielectric_tensor
    from pypresso.scf import Calculation

    system = build_system(read_pw_input(CASES / "si-epsilon-us.in"))
    pseudo_dir = Path(__file__).resolve().parents[1] / "data" / "pseudo"
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    with pytest.raises(NotImplementedError, match="zstar_eu_us"):
        dielectric_tensor(
            calculation, None, np.zeros((1, 1, 1)), None, born_charges=True
        )


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
