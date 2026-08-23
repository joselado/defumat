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


#: fcc aluminium's ``chi_0`` against the same finite difference, and the step it
#: is measured at. The error scales as ``h^2`` -- 2.5e-7, 1.4e-6, 1.3e-5 at
#: 3e-4, 1e-3, 3e-3 -- so this is the difference's truncation and not the
#: solve's.
METAL_CHI0_STEP = 3.0e-4
METAL_CHI0_RELATIVE = 1e-6


@lru_cache(maxsize=None)
def _metal():
    """The converged aluminium of ``al-metal.in`` -- QE's own ``pw_metal`` cell."""
    from pypresso.scf import Calculation

    system = build_system(read_pw_input(CASES / "al-metal.in"))
    pseudo_dir = Path(__file__).resolve().parents[1] / "data" / "pseudo"
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, nbnd=METAL_NBND, calculation=calculation,
                     conv_thr=1e-12, max_iterations=100)
    return system, calculation, result


#: Enough bands that the smearing tail is resolved: at ``degauss = 0.05`` the
#: occupation is below 1e-4 well before the eighth band, so both sides of the
#: comparison see the same Fermi surface.
METAL_NBND = 8


def test_chi0_matches_a_finite_difference_for_a_metal():
    """P24c: ``orthogonalize``'s smearing branch, against a difference of densities.

    The reference re-occupies at the **same** Fermi level rather than
    re-converging it, and that is not a shortcut -- it is what the quantity being
    computed is. The Sternheimer response of a metal is the response at fixed
    ``ef``; the level's own motion is a separate correction
    (:meth:`~pypresso.response.sternheimer.SternheimerSolver.fermi_level_shift`,
    ``ef_shift``), and testing the two together would test neither.

    What the metal branch changes, and what this therefore checks at once: the
    sharp projector becomes the occupation-difference weights ``wwg``, with the
    ``0/0`` at a degeneracy taken to its limit; every band stays in the block
    with ``nbnd_occ`` a mask rather than a count; ``alpha_pv`` is measured to
    ``ef + xmax degauss``; and the density is accumulated with ``wk`` and not
    ``wg``, because the occupation is already inside ``dpsi``. Any one of the
    four wrong moves this by far more than the truncation error below -- the
    weight convention alone would double-count the occupations.
    """
    from pypresso.basis.interpolate import to_dense
    from pypresso.scf.density import sum_band
    from pypresso.scf.occupations import smearing_order, wgauss

    system, calculation, result = _metal()
    solver = make_sternheimer(calculation, result, metals=True)
    assert solver.smearing is not None
    dv = _probe_potential(calculation)

    solution = solver.solve(solver.perturbation(dv))
    assert solution.converged
    drho = np.asarray(solver.response_density(solution.dpsi))

    smooth, dense = calculation.basis.smooth, calculation.basis.dense
    v_scf = calculation.potential(result.density).v_scf
    ngauss = smearing_order(system.smearing)
    kweights = jnp.asarray(system.kpoints.weights)

    def density_at(scale):
        hamiltonians = calculation.hamiltonian(v_scf + scale * dv, None)
        eigenvalues, psi = calculation.diagonalize(hamiltonians, METAL_NBND, None, 1e-13)
        occupation = wgauss(
            (result.fermi_energy - eigenvalues) / system.degauss, ngauss
        )
        rho = sum_band(
            psi, calculation.fft_index, smooth.grid,
            occupation * kweights[None, :, None], system.cell, calculation.k_batch,
        )
        return np.asarray(to_dense(rho, smooth, dense))

    step = METAL_CHI0_STEP
    reference = (density_at(step) - density_at(-step)) / (2.0 * step)
    relative = np.abs(drho - reference).max() / np.abs(drho).max()
    assert relative < METAL_CHI0_RELATIVE


def test_the_fermi_level_shift_restores_charge_neutrality():
    """``ef_shift``, checked against an identity rather than against a number.

    A perturbation at ``q = 0`` is not orthogonal to the identity, so the
    independent-particle response moves charge in or out of the cell: the
    uncorrected ``drho`` integrates to 0.21 electrons on this probe, which is
    not a response at all but a change in the electron count. Letting the Fermi
    level move by ``def = -(integral) / N(ef)`` and filling the Fermi surface
    with ``def ldos`` removes exactly that, because ``ldos`` integrates to
    ``N(ef)`` by construction -- so the corrected density integrating to zero is
    an identity the two halves satisfy together and neither imposes.

    Two probes, because one of them could be zero by accident: the ``(1,0,0)``
    and ``(1,1,1)`` cosines give shifts of opposite sign (-0.036 and +0.009 Ry)
    and both come back neutral.
    """
    system, calculation, result = _metal()
    solver = make_sternheimer(calculation, result, metals=True)
    ldos, dos_ef = solver.local_density_of_states()
    element = system.cell.volume / int(np.prod(np.asarray(ldos).shape[1:]))
    # ``ldos`` integrates to the density of states at the Fermi level, which is
    # what makes the correction exact rather than approximate.
    assert abs(float(jnp.sum(ldos)) * element - dos_ef) < 1e-10

    for miller in ((1, 0, 0), (1, 1, 1)):
        grid = calculation.basis.dense.grid
        axes = [np.arange(n) / n for n in grid]
        coordinates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        dv = jnp.asarray(np.cos(2.0 * np.pi * (coordinates @ np.asarray(miller)))[None])
        drho = solver.response_density(solver.solve(solver.perturbation(dv)).dpsi)
        before = float(jnp.sum(drho)) * element
        corrected, shift = solver.fermi_level_shift(drho)
        assert abs(before) > 1e-3, "the probe has to move charge for this to test anything"
        assert abs(float(jnp.sum(corrected)) * element) < 1e-12


def test_a_metal_is_refused_the_quantities_it_does_not_have():
    """The solve handles a metal; ``epsilon_infinity`` and ``Z*`` do not exist for one.

    ``pw.x`` refuses ``epsil`` for a metal for the same reason, and the refusal
    is here rather than three times over because
    :func:`~pypresso.response.sternheimer.require_a_sternheimer_regime` takes a
    flag saying whether the *caller's* quantity survives a Fermi surface.
    """
    from pypresso.response.efield import dielectric_tensor

    _, calculation, _ = _metal()
    with pytest.raises(NotImplementedError, match="epsilon_infinity"):
        dielectric_tensor(
            calculation, None, np.zeros((1, 1, 1)), None, born_charges=False
        )


def test_ef_shift_on_the_states_reproduces_its_effect_on_the_density():
    """``ef_shift_wfc`` against ``ef_shift``, which is where its factor of a half lives.

    The Fermi level's motion has to reach the first-order *states* as well as
    the density, because the second derivative consumes ``dpsi`` as a tangent
    rather than through the density it builds. QE writes the correction as
    ``dpsi_n += (1/2) def delta(ef - eps_n) psi_n`` and the half is not
    decoration: the density is quadratic in the states, so half on each of
    ``psi`` and its conjugate is what reproduces the whole ``def ldos`` that the
    density correction adds. Asserting that equality is what pins the factor --
    the two routines share the shift and nothing else, one going through the
    density builder and one not.
    """
    system, calculation, result = _metal()
    solver = make_sternheimer(calculation, result, metals=True)
    dv = _probe_potential(calculation)
    dpsi = solver.solve(solver.perturbation(dv)).dpsi

    ldos, _ = solver.local_density_of_states()
    _, shift = solver.fermi_level_shift(solver.response_density(dpsi))
    corrected = solver.fermi_level_shift_states(dpsi, shift)
    through_states = (
        solver.response_density(corrected) - solver.response_density(dpsi)
    )
    through_density = shift * ldos
    scale = float(jnp.abs(through_density).max())
    assert float(jnp.abs(through_states - through_density).max()) < 1e-10 * scale


def test_the_dynamical_matrix_of_a_metal_is_not_refused_any_more():
    """P28 lifted it, and this asserts the *door* rather than the answer.

    This was a refusal for one phase and the reason was a weight: a metal's
    ``dpsi`` carries its own occupation -- which is what the ``wk`` in the
    density response encodes -- while the energy functional the force constants
    differentiate weights its states by ``wg = wk f``, so a single ``jvp``
    counted ``f`` twice. Two-atom aluminium came out at 155.7, 155.7, 155.7,
    198.0, 198.0, 309.3 cm^-1 against ``ph.x``'s 1.1, 1.8, 1.9, 146.7, 146.7,
    311.0, from a run that converged and returned a symmetric matrix.
    Splitting the ``jvp`` -- the frozen Hessian at ``wg``, the electronic half
    at ``wk``, which is what ``dynmat_us.f90`` and ``drhodvnl.f90`` are --
    puts them at 1.088, 1.559, 1.559, 146.711240, 146.711240 and 311.033545.

    What is asserted here is only that **neither guard the front door runs
    raises for a smeared metal**. The frequencies themselves belong to
    ``tests/regression/test_phonons.py``, which has the ``ph.x`` reference and
    the acoustic sum rule beside them; repeating them here would be a second
    place for the same numbers to go stale. The tetrahedron occupations stay
    refused and are covered by
    :func:`test_the_regimes_without_a_response_here_are_refused_by_name`.
    """
    from pypresso.response.phonon import require_norm_conserving
    from pypresso.response.sternheimer import require_a_sternheimer_regime

    _, calculation, _ = _metal()
    assert calculation.system.occupations != "fixed", "this cell must be a metal"
    require_a_sternheimer_regime(calculation, metals=True)
    require_norm_conserving(calculation)


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
#: What the vendored ``ph.x`` prints for ``Z*``, per case. Silicon's is zero by
#: symmetry in a converged calculation, so the number is a *residue* -- 4 against
#: an electronic part near 4.076 -- which makes it a sharper check of the
#: machinery than the dielectric constant. Carbon's has the opposite **sign**,
#: on a different element at different cutoffs, so agreeing on both is not
#: agreeing twice on the same arithmetic.
QE_BORN = {
    "si-epsilon": -0.07571,
    "si-epsilon-us": -0.07945,
    "c-epsilon": 0.04179,
}
QE_TOTAL_ENERGY = -15.84452726

#: How far the dielectric constant may sit from ``ph.x``'s. The measured
#: differences are 4.3e-5, 5.2e-5, 3.4e-5 and 1.2e-4, and what is left is the
#: same thing that puts a floor under the eigenvalues
#: (``tests/tolerances.py``): QE interpolates every radial form factor from a
#: ``dq = 0.01`` table where this code integrates it directly.
EPSILON_TOLERANCE = 5e-4

#: How far ``Z*`` may sit from ``ph.x``'s, per case. Silicon's two are at the
#: printed digit; **carbon's is three times looser and for the same reason its
#: dielectric constant is** -- 1.2e-4 there against silicon's 4.3e-5, the
#: radial form factors' interpolation floor (``tests/tolerances.py``). ``Z*`` is
#: a residue of ``4`` against ``3.958``, so that floor arrives amplified by the
#: cancellation: 2.3e-4 on the residue is 6e-5 relative to the 4 it came from.
BORN_TOLERANCE = {"si-epsilon": 1e-4, "si-epsilon-us": 1e-4, "c-epsilon": 3e-4}

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
        # Born charges are norm-conserving and ultrasoft; PAW is refused.
        born_charges=not calculation.is_paw,
        # The transcribed ``zstar_eu`` cross-check reads ``dpsi`` and the solver
        # back out of the same run rather than paying for a second one.
        keep_internals=True,
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


@pytest.mark.parametrize("case", list(QE_BORN))
def test_the_born_charges_match_quantum_espresso(case):
    """``Z*`` against the same benchmark -- the sharper of the two numbers.

    Silicon's Born charge is zero by symmetry in a converged calculation, so
    what ``-0.07571`` measures is the residue of a difference of large numbers:
    ``Z_val = 4`` against an electronic part near ``4.076``. Reproducing it to
    the printed digits means the bare displacement perturbation, the
    self-consistent field response and the weights are all right.

    **The ultrasoft cases are the point** (``PLAN.md`` P24b). ``Z*`` is
    ``dF/dE``, a mixed second derivative, and it is computed as one -- one
    ``jvp`` of the force along the field response
    (:mod:`pypresso.response.born`) -- so four of the five stages
    ``zstar_eu_us.f90`` adds are terms of the same derivative rather than five
    more routines. The norm-conserving expression on this ultrasoft cell gives
    **+0.1625**, wrong in sign as well as size, which is what the machinery here
    has to beat and does: -0.079442 against -0.07945.
    """
    _, response = _dielectric(case)
    charges = response.born_charges
    assert charges.shape == (2, 3, 3)
    for atom in range(2):
        diagonal = np.diag(charges[atom])
        assert np.allclose(diagonal, QE_BORN[case], atol=BORN_TOLERANCE[case])
        off = charges[atom] - np.diag(diagonal)
        assert np.abs(off).max() < CUBIC_TOLERANCE


def test_the_mixed_derivative_reproduces_the_transcribed_zstar_eu():
    """The regression gate on the whole assembly, and it is an *equality*.

    ``zstar_eu.f90`` is transcribed beside the mixed derivative
    (:func:`~pypresso.response.efield.born_charges_zstar_eu`) and shares only the
    field response ``dpsi`` with it -- one contracts the bare displacement
    perturbation by hand, the other differentiates the force. On a
    norm-conserving dataset they must agree exactly, and that is what says every
    term :mod:`pypresso.response.born` adds for an ultrasoft dataset switches
    itself off when ``S = 1``: the matrix multipliers, the augmentation charge's
    share of the frozen polarization, and ``add_for_charges``' ``dS/du`` are all
    identically zero there, and any one of them leaking would show up here long
    before it showed up against ``ph.x``'s five printed digits.
    """
    from pypresso.response.efield import born_charges_zstar_eu

    _, response = _dielectric("si-epsilon")
    internals = response.internals
    transcribed = born_charges_zstar_eu(
        internals["calculation"], internals["solver"], internals["v_scf"],
        internals["dpsi"],
    )
    assert np.abs(transcribed - response.born_charges).max() < 1e-9


def test_born_charges_are_refused_for_a_paw_dataset():
    """The one dataset the mixed derivative does not finish, refused by name.

    Everything in :mod:`pypresso.response.born` gets PAW to **1.3e-3** --
    -0.078293 against ``ph.x``'s -0.07961, where the ultrasoft case of the same
    assembly reaches 8e-6 -- and what is left is QE's last stage, ``int3_paw``
    against ``becsumort``: the one-centre twin of ``add_for_charges``, pairing
    the field's response of the one-centre coefficients with the displacement's
    orthogonality ``becsum``. 1.3e-3 is sixteen times the last digit ``ph.x``
    prints, so it is refused rather than returned. The dielectric constant from
    the *same* run is right to 3.4e-5 and is not refused.
    """
    from pypresso.response.efield import dielectric_tensor
    from pypresso.scf import Calculation

    system = build_system(read_pw_input(CASES / "si-epsilon-paw.in"))
    pseudo_dir = Path(__file__).resolve().parents[1] / "data" / "pseudo"
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    with pytest.raises(NotImplementedError, match="becsumort"):
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
