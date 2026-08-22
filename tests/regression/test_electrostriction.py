"""P26: electrostriction as a mixed third derivative, and what stands under it.

The phase has four pieces and each is checked against something that shares no
machinery with it:

* **the strain response** (:mod:`pypresso.response.strain`) against a central
  difference of a *re-converged* SCF at the same frozen sphere. The analytic
  route is a ``jvp`` through ``at_strain`` plus a projected CG solve; the
  reference re-runs the whole SCF loop at ``+-h``, so the two share only the
  Hamiltonian. Measured: 1.6e-5 relative at the difference's optimal step, with
  the error falling as ``h^2`` above it and as ``1/h`` below.
* **the variational second-order energy** against ``dielec.f90``'s own
  assembly. ``F_ij`` is a different expression built from the same ``u`` -- four
  terms where the assembly has one -- and at the stationary point they are equal
  identically. It is the sharpest cheap check in the phase, and it is what found
  that the SCF's wavefunctions are eigenvectors of the *previous* iteration's
  Hamiltonian (:func:`~pypresso.response.electrostriction.refined_states`).
* **the cubic form of ``d(chi)/dx``**. Nothing here imposes the crystal class:
  the k-grid is unshifted and closed under the point group, and no average is
  applied anywhere. So the components a cubic crystal forbids are a measurement
  of every index convention in the phase at once -- and they are what found the
  multiplier's transposed band indices, which are invisible in the value.
* **the third derivative itself** against a central difference of the dielectric
  constant over re-converged strained cells, which is the route Tanner,
  Bousquet and Janolin take (`arXiv:2012.03841 <https://arxiv.org/abs/2012.03841>`_)
  and the only end-to-end reference there is: ``ph.x`` has no strain
  perturbation.

The unit chain has a reference of its own that needs no DFT at all: the paper's
MgO table, where ``m``, ``q``, the elastic compliance and ``M``, ``Q`` are all
tabulated, so the conversion between them can be checked on published numbers.
"""

from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.response.electrostriction import (
    _compliance_tensor,
    _epsilon_at,
    _project_conduction,
    electrostriction,
    refined_states,
)
from pypresso.response.strain import strain_response, strain_tangent
from pypresso.scf import Calculation, run_scf
from pypresso.system import build_system

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))

#: The k-grid is closed under the point group and nothing averages over it, so
#: a component cubic symmetry forbids is round-off. Measured: 3e-14 relative.
CUBIC_TOLERANCE = 1e-10

#: The finite-difference references are the loose ones here, not the analytic
#: route: see the module docstring for the step-size study behind each.
STRAIN_DENSITY_TOLERANCE = 5e-4

#: Measured agreement is **2e-4** on all three independent components. The
#: tolerance is deliberately only a factor of 25 above it rather than the
#: comfortable 5% a finite-difference comparison invites: the bug this phase
#: spent longest on -- the frozen ``u`` and its moving constraint surface -- was
#: worth 2%, so a tolerance that admits 2% is a test that would have passed with
#: the phase's own headline error in place.
THIRD_DERIVATIVE_TOLERANCE = 5e-3
ELASTIC_TOLERANCE = 5e-3


@lru_cache(maxsize=None)
def _converged(case: str):
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                     max_iterations=100)
    eigenvalues, psi = refined_states(calculation, result)
    return system, pseudos, calculation, result, eigenvalues, psi


@lru_cache(maxsize=None)
def _strain(case: str):
    _, _, calculation, result, eigenvalues, psi = _converged(case)
    return strain_response(
        calculation, psi, eigenvalues, jnp.asarray(result.density)
    )


@lru_cache(maxsize=None)
def _electrostriction(case: str):
    _, _, calculation, result, _, _ = _converged(case)
    return electrostriction(calculation, result, strain=_strain(case))


def _reconverged(case: str, strain):
    """The SCF re-run in a cell deformed by ``strain``, same frozen sphere.

    Cached on the strain's *values*, because three tests ask for the same
    deformations and each one is a full SCF.
    """
    return _reconverged_at(case, tuple(np.asarray(strain).ravel().tolist()))


@lru_cache(maxsize=None)
def _reconverged_at(case: str, values: tuple):
    system, pseudos, calculation, result, _, _ = _converged(case)
    moved = Calculation(system, pseudos).at_strain(
        jnp.asarray(values).reshape(3, 3)
    )
    return moved, run_scf(
        system, pseudos, calculation=moved, conv_thr=1e-12, max_iterations=200,
        starting_density=result.density,
        starting_wavefunctions=result.wavefunctions,
    )


# -- the strain response ------------------------------------------------------


@pytest.mark.parametrize("component", [(0, 0), (0, 1)])
def test_the_strain_response_matches_a_finite_difference(component):
    """``drho/dx`` against a central difference of the converged density.

    The step is the difference's own optimum, measured: at ``h = 1e-2`` the
    truncation error dominates and falls as ``h^2`` (1.9e-4 relative), at
    ``3e-3`` it reaches 1.6e-5, and below that the SCF's own noise divided by
    ``2h`` takes over again (7e-5 at ``1e-3``). What is being measured at the
    optimum is the reference, not the response.

    The volume-changing component is the one with a term the displacement
    perturbation has no counterpart for: ``rho`` is stored as values on a grid
    that does not move and carries a ``1/Omega``, so it responds at ``dpsi = 0``.
    The shear has no such term, which is why both are checked.
    """
    step = 3e-3
    a, b = component
    tangent = strain_tangent(a, b)
    plus = np.asarray(_reconverged("si-electrostriction", step * tangent)[1].density)
    minus = np.asarray(_reconverged("si-electrostriction", -step * tangent)[1].density)
    reference = (plus - minus) / (2 * step)
    ours = np.asarray(_strain("si-electrostriction").drho[a, b])
    error = np.abs(ours - reference).max() / np.abs(reference).max()
    assert error < STRAIN_DENSITY_TOLERANCE


def test_the_strain_response_converges():
    response = _strain("si-electrostriction")
    assert response.converged


@pytest.mark.slow
def test_the_symmetrised_wedge_and_the_closed_grid_give_one_strain_response():
    """The only check ``symmetrize_tensor_density`` has -- ``ph.x`` has no strain.

    The same unshifted 2x2x2 sample two ways: reduced to its wedge with the
    rank-2 average applied, and whole with no average at all. A rotation mixes
    *both* cartesian labels of a strain-labelled response density, so getting
    the average's index order wrong is a different symmetry rather than a worse
    one -- and this is what would catch it.

    Compared through a scalar contraction rather than pointwise, because the two
    runs have different FFT grids: symmetry with fractional translations forces
    the grid to a multiple of their denominators (``fft_fact``) and ``nosym``
    does not, so the closed-grid run is 15^3 and the wedge run 16^3.
    """
    wedge = _strain("si-strain-wedge")
    closed = _strain("si-electrostriction")

    def invariants(response, calculation):
        """``int drho_ab rho_0``: a rank-2 tensor, grid-independent."""
        rho = np.asarray(_converged_density(calculation))
        volume = calculation.system.cell.volume
        cells = np.asarray(response.drho[0, 0]).size
        return np.array([
            [float(np.sum(np.asarray(response.drho[a, b]) * rho)) * volume / cells
             for b in range(3)] for a in range(3)
        ])

    def _converged_density(calculation):
        for case in ("si-strain-wedge", "si-electrostriction"):
            _, _, other, result, _, _ = _converged(case)
            if other is calculation:
                return result.density
        raise AssertionError

    _, _, wedge_calc, _, _, _ = _converged("si-strain-wedge")
    _, _, closed_calc, _, _, _ = _converged("si-electrostriction")
    a = invariants(wedge, wedge_calc)
    b = invariants(closed, closed_calc)
    assert np.allclose(a, b, rtol=2e-3), f"{a}\n{b}"


# -- the variational second-order energy --------------------------------------


def test_the_variational_energy_reproduces_the_dielectric_assembly():
    """``F_ij`` at its stationary point is ``sum w Re <b_i|u_j>`` identically.

    Four terms here -- the band energy, the multiplier, the source and the
    screening -- against ``dielec.f90``'s single overlap. They are equal only if
    ``u`` really is the stationary point, which makes this a check on the
    self-consistent loop, on the sign of ``orthogonalize``, on the screening
    kernel's factor of a half, and on the ``16 pi/Omega`` all at once.

    It is also what forces :func:`refined_states`: without the re-diagonalisation
    this fails by 7e-7 *relative* -- systematically, not as noise, and it does
    not shrink when the response's own thresholds are tightened.
    """
    from pypresso.response.efield import dielectric_tensor

    _, _, calculation, result, eigenvalues, psi = _converged("si-electrostriction")
    density = jnp.asarray(result.density)
    field = dielectric_tensor(
        calculation, psi, eigenvalues, density,
        born_charges=False, keep_internals=True,
    )
    solver = field.internals["solver"]
    b = _project_conduction(solver.psi, jnp.stack(field.internals["bare"]))
    u = _project_conduction(solver.psi, jnp.stack(field.internals["dpsi"]))
    ours = np.asarray(_epsilon_at(
        calculation, jnp.zeros((3, 3)), solver.psi, density, b, u, solver.weights
    ))
    assert np.abs(ours - field.epsilon).max() / np.abs(field.epsilon).max() < 1e-8


# -- the third derivative -----------------------------------------------------


def test_the_susceptibility_derivative_comes_out_cubic():
    """Nothing imposes the crystal class, so the forbidden components measure the code.

    ``d(chi_ij)/dx_kl`` on a cubic crystal has two independent entries in its
    ``3x3`` block and one in its shear block, and the cross blocks vanish. The
    k-grid is closed under the point group and no average is applied anywhere,
    so what is left in a forbidden component is round-off -- unless an index
    convention is wrong somewhere, which is how the multiplier's transposed band
    indices were found: they left 11% of the scale in components the group
    forbids while changing no value at zeroth order.
    """
    tensor = _electrostriction("si-electrostriction").dchi_dstrain
    table = np.array([[tensor[i][j][k][l] for (k, l) in VOIGT] for (i, j) in VOIGT])
    scale = np.abs(table).max()
    forbidden = np.concatenate([table[3:, :3].ravel(), table[:3, 3:].ravel()])
    assert np.abs(forbidden).max() / scale < CUBIC_TOLERANCE
    assert np.ptp(np.diag(table)[:3]) / scale < CUBIC_TOLERANCE
    assert np.ptp(np.diag(table)[3:]) / scale < CUBIC_TOLERANCE
    off = np.array([table[0, 1], table[0, 2], table[1, 2]])
    assert np.ptp(off) / scale < CUBIC_TOLERANCE


@pytest.mark.parametrize("component", [(0, 0), (0, 1)])
def test_the_third_derivative_matches_a_finite_difference(component):
    """``d(eps)/dx`` from one ``jvp`` against a sweep of re-converged cells.

    This is the end-to-end check and the reference is the published method: a
    dielectric constant computed at ``+-h`` and differenced. Both sides freeze
    the plane-wave sphere -- the reference deforms the *same* calculation with
    ``at_strain`` rather than rebuilding a cell -- so what is compared is the
    response, not the basis-set jump.
    """
    from pypresso.response.efield import dielectric_tensor

    step = 3e-3
    a, b = component
    tangent = strain_tangent(a, b)

    def epsilon(strain):
        moved, result = _reconverged("si-electrostriction", strain)
        eigenvalues, psi = refined_states(moved, result)
        field = dielectric_tensor(
            moved, psi, eigenvalues, result.density,
            born_charges=False, keep_internals=True,
        )
        # ``dielec.f90``'s expression **without** ``symmatrix``: the strained
        # crystal is not cubic, and averaging over the undeformed crystal's 48
        # operations would put the answer back.
        solver = field.internals["solver"]
        bare = jnp.stack(field.internals["bare"])
        dpsi = jnp.stack(field.internals["dpsi"])
        volume = moved.system.cell.volume
        out = np.eye(3)
        for i in range(3):
            for j in range(3):
                total = float(jnp.sum(solver.weights * jnp.real(jnp.einsum(
                    "skng,skng->skn", jnp.conj(bare[i]), dpsi[j]))))
                out[i, j] -= 16.0 * np.pi * total / volume
        return out

    reference = (epsilon(step * tangent) - epsilon(-step * tangent)) / (2 * step)
    ours = _electrostriction("si-electrostriction").depsilon_dstrain[:, :, a, b]
    error = np.abs(ours - reference).max() / np.abs(reference).max()
    assert error < THIRD_DERIVATIVE_TOLERANCE


# -- the elastic constants ----------------------------------------------------


@pytest.mark.parametrize("component", [(0, 0), (0, 1)])
def test_the_elastic_constants_match_a_second_difference_of_the_energy(component):
    """``C = (1/Omega) d^2E/dx^2`` against a five-point difference of the SCF energy.

    The energy is the reference with no convention in it at all -- no volume
    factor, no sign, no Voigt -- and both sides freeze the same plane-wave
    sphere, so what is compared is the derivative rather than the basis. Measured:
    **209.38 GPa against 209.38** for ``C_11``.

    It is the check that found the phase's fourth bug. Handing the density in as
    an independent argument, the way ``_force_constants`` does for a phonon,
    makes ``jax.grad`` of the functional a partial derivative at fixed ``rho``
    rather than the *stress*: the two differ by ``(dE/drho).(drho/dx)|_psi``,
    which is zero for a displacement and is not for a strain, because the density
    carries a ``1/Omega``. Written that way this returns 671 GPa.
    """
    step = 4e-3
    a, b = component
    tangent = strain_tangent(a, b)
    _, _, calculation, _, _, _ = _converged("si-electrostriction")

    energies = {
        offset: _reconverged("si-electrostriction", offset * step * tangent)[1].total_energy
        for offset in (-2, -1, 0, 1, 2)
    }
    second = (
        -energies[2] + 16 * energies[1] - 30 * energies[0]
        + 16 * energies[-1] - energies[-2]
    ) / (12 * step**2)
    reference = second / calculation.system.cell.volume

    ours = _electrostriction("si-electrostriction").elastic.tensor[a, b, a, b]
    assert abs(ours - reference) / abs(reference) < ELASTIC_TOLERANCE


# -- the reporting boundary ---------------------------------------------------


def test_the_unit_chain_reproduces_a_published_table():
    """``M = -S : m`` and ``Q = -S : q`` against Tanner et al.'s MgO, no DFT.

    Table I of `arXiv:2012.03841 <https://arxiv.org/abs/2012.03841>`_ tabulates
    all four coefficient families for MgO, so the conversion between them -- the
    ``eps0`` factors, the pm/pN/GN prefixes and Voigt's halves in the compliance
    -- can be checked on published numbers with nothing computed here at all.
    Feeding their ``m`` and ``q`` and textbook elastic constants through this
    module's algebra must return their ``M`` and ``Q``.

    The residual is the elastic constants used, not the algebra: with
    ``C_11 = 300``, ``C_12 = 95`` GPa this gives ``M_11 = 1910`` against their
    1970 and ``Q_12 = -0.075`` against their -0.075.
    """
    voigt = np.zeros((6, 6))
    c11, c12, c44 = 300.0, 95.0, 155.0  # MgO, GPa
    for i in range(3):
        for j in range(3):
            voigt[i, j] = c11 if i == j else c12
    for i in range(3, 6):
        voigt[i, i] = c44
    compliance = _compliance_tensor(np.linalg.inv(voigt) / 1.0e9)

    def cubic(diagonal, off):
        out = np.zeros((3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                out[i, i, j, j] = diagonal if i == j else off
        return out

    m = cubic(-477.7e-12, 16.5e-12)         # N/V^2, their Table I
    q = cubic(-71.6e9, 2.5e9)               # N m^2/C^2
    big_m = -np.einsum("ijmn,mnkl->ijkl", compliance, m) * 1.0e24
    big_q = -np.einsum("ijmn,mnkl->ijkl", compliance, q)
    assert big_m[0, 0, 0, 0] == pytest.approx(1970.0, rel=0.05)
    assert big_m[0, 0, 1, 1] == pytest.approx(-508.0, rel=0.05)
    assert big_q[0, 0, 0, 0] == pytest.approx(0.292, rel=0.05)
    assert big_q[0, 0, 1, 1] == pytest.approx(-0.075, rel=0.05)


def test_the_elasto_optic_tensor_matches_experiment_where_it_should():
    """``p_11`` and ``p_12`` against Biegelsen's measurement, on the real k-set.

    The elasto-optic tensor ``p = -eps^-1 (d eps/dx) eps^-1`` is the one quantity
    this phase produces that a laboratory has measured directly, and the
    symmetry story makes two of its three components a *fair* comparison: in the
    diamond structure no internal displacement is compatible with a tetragonal
    strain, so ``p_11`` and ``p_12`` have no ionic contribution and clamped-ion
    is the whole answer. ``p_44`` carries a Kleinman internal-displacement term
    that is not computed here and is deliberately not asserted.

    **What is asserted is what this k-sample supports, and not more.** The
    closed-grid requirement forces an unshifted Monkhorst-Pack grid and this is
    the small one, which gives ``eps_infinity = 56`` where silicon's is 13.8.
    ``p_11`` survives that with the right sign and within a factor of three of
    the measurement; ``p_12`` does **not** -- it is -0.003 here against +0.017
    measured, sign included. That is the sample, not the derivative: the same
    number comes out of a central difference of ``epsilon`` over re-converged
    strained cells, which shares no machinery with the analytic route.
    `PLAN.md`'s P26 section carries the convergence study, and saying so is the
    honest form of this test.
    """
    out = _electrostriction("si-electrostriction")
    voigt = out.photoelastic_voigt

    # The algebra, which is the regression test: ``p = -eps^-1 (d eps/dx) eps^-1``
    # rebuilt from the two arrays it was derived from.
    inverse = np.linalg.inv(out.epsilon)
    expected = -np.einsum(
        "ia,abkl,bj->ijkl", inverse, out.depsilon_dstrain, inverse
    )
    assert np.allclose(out.photoelastic, expected, atol=1e-14)

    # And the physics, as far as this k-sample supports it -- which is ``p_11``
    # and not ``p_12``. See the docstring.
    assert voigt[0, 0] < 0.0
    assert 0.01 < abs(voigt[0, 0]) < 0.30


def test_a_symmetry_reduced_kset_is_refused():
    """The combination with no average written for it, refused by name.

    A response carrying a field label *and* a strain label needs a rank-3
    average to complete a wedge sum. P24 wrote the rank-1 case and P25 the
    rank-1-plus-atom case; this one is not written, and an unshifted grid needs
    none of them.
    """
    _, _, calculation, result, _, _ = _converged("si-strain-wedge")
    with pytest.raises(NotImplementedError, match="rank-3"):
        electrostriction(calculation, result)
