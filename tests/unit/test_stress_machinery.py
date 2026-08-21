"""P11's fast checks: the strain map, the symmetrisation, and what is refused.

Everything here runs on a two-atom silicon cell without an SCF -- the state
handed to the functional is whatever ``starting_wavefunctions`` produced, which
is not a solution of anything. That is deliberate: the identities checked below
are properties of the *machinery* (a deformation is applied consistently, the
gradient is symmetric, the compiled kernel is invalidated when it must be) and
none of them needs a converged density. The physics is in
``tests/regression/test_stress.py``.
"""

from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pypresso.forces.energy import FrozenState
from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf import Calculation
from pypresso.stress import Stress, compute_stress, format_stress, stress_methods
from pypresso.stress.energy import strained_energy, strained_energy_terms
from pypresso.system import build_system
from pypresso.system.symmetry import symmetrize_matrix
from pypresso.units import RY_TO_KBAR

pytestmark = pytest.mark.unit

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"


@lru_cache(maxsize=None)
def _silicon(pseudo_dir: Path):
    """The displaced two-atom cell, with a state that is not a solution of anything.

    The state is whatever ``starting_wavefunctions`` produced, and the weights
    are the k-point weights: not a solution, which is the point. Every identity
    below is a property of the strain map or of the gradient and holds at any
    state at all.
    """
    system = build_system(read_pw_input(CASES / "si2-nc-stress.in"))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    calculation = Calculation(system, pseudos)

    nbnd = 4
    rho = calculation.starting_density()
    potential = calculation.potential(rho)
    hamiltonians = calculation.hamiltonian(potential.v_scf)
    psi = calculation.starting_wavefunctions(hamiltonians, nbnd)
    nk = system.kpoints.nk
    state = FrozenState(
        wavefunctions=psi,
        weights=jnp.asarray(np.tile(np.asarray(system.kpoints.weights)[:, None],
                                    (1, nbnd))[None]),
        eigenvalues=jnp.zeros((1, nk, nbnd)),
    )
    return calculation, state


def test_zero_strain_leaves_everything_alone(pseudo_dir):
    """``at_strain(0)`` must reproduce the calculation it was called on.

    The arrays are rebuilt from scratch rather than shared, so this is a real
    check that the rebuild path agrees with the original setup -- the radial
    transforms, the projectors, the Ewald sum and the local potential all come
    out of different code than they went in.
    """
    calculation, _ = _silicon(pseudo_dir)
    same = calculation.at_strain(jnp.zeros((3, 3)))

    assert np.asarray(same.system.cell.at) == pytest.approx(
        np.asarray(calculation.system.cell.at), abs=1e-14
    )
    assert np.asarray(same.kinetic) == pytest.approx(np.asarray(calculation.kinetic), abs=1e-12)
    assert np.asarray(same.vltot) == pytest.approx(np.asarray(calculation.vltot), abs=1e-12)
    assert float(same.ewald) == pytest.approx(float(calculation.ewald), abs=1e-12)
    assert np.asarray(same.projectors.vkb) == pytest.approx(
        np.asarray(calculation.projectors.vkb), abs=1e-12
    )


def test_isotropic_strain_scales_the_volume(pseudo_dir):
    """``h -> (1 + e) h`` must give ``Omega (1 + e)^3``, and move the atoms with it."""
    calculation, _ = _silicon(pseudo_dir)
    e = 0.01
    strained = calculation.at_strain(e * jnp.eye(3))

    assert float(strained.system.cell.volume) == pytest.approx(
        float(calculation.system.cell.volume) * (1.0 + e) ** 3, rel=1e-12
    )
    # The atoms are carried in crystal coordinates: their fractional positions
    # do not move, so a uniform dilation is the whole of the change.
    crystal_before = np.asarray(
        calculation.system.structure.positions_crystal(calculation.system.cell)
    )
    crystal_after = np.asarray(
        strained.system.structure.positions_crystal(strained.system.cell)
    )
    assert crystal_after == pytest.approx(crystal_before, abs=1e-12)


def test_shear_transposes_the_way_the_lattice_vectors_do(pseudo_dir):
    """``a_i -> (1 + eps) a_i``, which for row-stored vectors is a right ``D^T``.

    The transpose is invisible for a symmetric strain and wrong for every
    antisymmetric one, so the check uses a strain with a single off-diagonal
    entry.
    """
    calculation, _ = _silicon(pseudo_dir)
    strain = np.zeros((3, 3))
    strain[0, 2] = 0.05
    strained = calculation.at_strain(jnp.asarray(strain))

    at = np.asarray(calculation.system.cell.at)
    expected = np.einsum("ab,ib->ia", np.eye(3) + strain, at)
    assert np.asarray(strained.system.cell.at) == pytest.approx(expected, abs=1e-12)


def test_the_k_points_follow_the_reciprocal_cell(pseudo_dir):
    """A strain moves ``k`` as it moves ``G``: fixed in crystal coordinates.

    ``KPoints.coords`` are cartesian in ``2 pi / alat`` and ``alat`` is static,
    so the natural-looking ``kpoints.cartesian(strained_cell)`` returns the
    *unstrained* k-points. Nothing errors when that happens and it is exactly
    zero at Gamma, so it is checked here on a grid that has no k = 0 direction
    to hide in.
    """
    calculation, _ = _silicon(pseudo_dir)
    strain = jnp.asarray(np.diag([0.03, 0.0, 0.0]))
    strained = calculation.at_strain(strain)

    before = np.asarray(calculation.system.kpoints.crystal(calculation.system.cell))
    # The k-list itself is unchanged -- crystal coordinates are the invariant;
    # what must move is |k+G|^2, and it must move by the amount the new
    # reciprocal cell says.
    kcart = before @ np.asarray(strained.system.cell.bg)
    gcart = np.asarray(calculation.basis.smooth.cartesian(strained.system.cell))
    indices = np.asarray(calculation.basis.planewaves.indices)
    mask = np.asarray(calculation.basis.planewaves.mask)
    expected = np.sum((kcart[:, None, :] + gcart[indices]) ** 2, axis=-1) * mask
    assert np.asarray(strained.kinetic) == pytest.approx(expected, abs=1e-12)
    # ... and it must actually have moved.
    assert np.abs(np.asarray(strained.kinetic) - np.asarray(calculation.kinetic)).max() > 1e-3


def test_the_gradient_is_symmetric(pseudo_dir):
    """Rotational invariance: ``dE/d(eps)`` is symmetric whatever the crystal is.

    A rigid rotation of the cell cannot change the energy, so the antisymmetric
    part of the strain derivative is identically zero -- and unlike the
    point-group argument this holds for a structure with no symmetry and an
    unconverged density, which makes it a check on the gradient itself.
    """
    calculation, state = _silicon(pseudo_dir)
    gradient = np.asarray(
        jax.grad(lambda eps: strained_energy(calculation, eps, state))(jnp.zeros((3, 3)))
    )
    assert np.abs(gradient - gradient.T).max() < 1e-10


def test_the_energy_terms_sum_to_the_energy(pseudo_dir):
    """``strained_energy_terms`` is a decomposition, not a second expression."""
    calculation, state = _silicon(pseudo_dir)
    strain = jnp.asarray(np.diag([0.01, -0.005, 0.0]))
    terms = strained_energy_terms(calculation, strain, state)
    total = strained_energy(calculation, strain, state)
    assert float(sum(terms.values())) == pytest.approx(float(total), abs=1e-10)


def test_finite_difference_of_the_frozen_energy(pseudo_dir):
    """The gradient against a central difference of the very function it differentiates.

    This is the exact check, and it is separate from the one against a
    re-converged SCF: at frozen state there is no stationarity to rely on and no
    Pulay stress to account for, so any disagreement here is a bug in the
    gradient rather than physics.
    """
    calculation, state = _silicon(pseudo_dir)
    gradient = np.asarray(
        jax.grad(lambda eps: strained_energy(calculation, eps, state))(jnp.zeros((3, 3)))
    )

    step = 1e-5
    for a, b in ((0, 0), (1, 2)):
        delta = np.zeros((3, 3))
        delta[a, b] = step
        plus = float(strained_energy(calculation, jnp.asarray(delta), state))
        minus = float(strained_energy(calculation, jnp.asarray(-delta), state))
        assert (plus - minus) / (2.0 * step) == pytest.approx(
            gradient[a, b], rel=1e-5, abs=1e-8
        )


def test_symmetrize_matrix_projects_onto_the_allowed_subspace(pseudo_dir):
    """``symmatrix`` on cubic silicon: any tensor becomes a multiple of the identity."""
    system = build_system(read_pw_input(CASES / "si2-nc-relax.in"))
    import equinox as eqx

    positions = np.array(system.structure.positions)
    positions[1] = np.array([0.25, 0.25, 0.25]) * float(system.cell.alat)
    system = eqx.tree_at(lambda s: s.structure.positions, system, jnp.asarray(positions))
    symmetries = system.symmetry_group()
    assert symmetries.nsym == 48

    rng = np.random.default_rng(0)
    arbitrary = rng.normal(size=(3, 3))
    arbitrary = 0.5 * (arbitrary + arbitrary.T)
    averaged = symmetrize_matrix(arbitrary, system.cell, symmetries)

    assert np.abs(averaged - np.eye(3) * np.trace(averaged) / 3.0).max() < 1e-12
    # The trace is what a point-group average cannot change.
    assert np.trace(averaged) == pytest.approx(np.trace(arbitrary), abs=1e-12)


def test_symmetrize_matrix_is_idempotent(pseudo_dir):
    calculation, _ = _silicon(pseudo_dir)
    rng = np.random.default_rng(1)
    arbitrary = rng.normal(size=(3, 3))
    once = symmetrize_matrix(arbitrary, calculation.system.cell, calculation.symmetries)
    twice = symmetrize_matrix(once, calculation.system.cell, calculation.symmetries)
    assert twice == pytest.approx(once, abs=1e-12)


def test_pressure_and_kbar_are_the_conversion_and_nothing_else():
    tensor = np.diag([1.0, 2.0, 3.0]) * 1e-4
    stress = Stress(tensor=tensor, method="autodiff")
    assert stress.pressure == pytest.approx(2e-4)
    assert stress.pressure_kbar == pytest.approx(2e-4 * RY_TO_KBAR)
    assert stress.kbar == pytest.approx(tensor * RY_TO_KBAR)
    # QE's own number, so that nothing quietly redefines the unit.
    assert RY_TO_KBAR == pytest.approx(147105.07, abs=0.01)


def test_format_stress_prints_qes_layout():
    tensor = np.diag([1.0, 1.0, 1.0]) * -2.054e-4
    text = format_stress(Stress(tensor=tensor, method="autodiff"))
    assert "total   stress  (Ry/bohr**3)" in text
    assert "P=" in text
    assert len(text.splitlines()) == 4


def test_the_registry_has_the_default(pseudo_dir):
    assert "autodiff" in stress_methods()


def test_analytic_total_is_refused_by_name(pseudo_dir):
    """No ``stres_us``, so no analytic total -- and it says which routine is missing."""
    calculation, state = _silicon(pseudo_dir)
    with pytest.raises(NotImplementedError, match="stres_us"):
        compute_stress(calculation, state, method="analytic")


def test_a_spiral_is_refused(pseudo_dir):
    """``spiral_q`` is in lattice coordinates, so a strain turns the spiral too."""
    system = build_system(read_pw_input(CASES / "h-chain-spiral.in"))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    with pytest.raises(NotImplementedError, match="spiral"):
        calculation.at_strain(jnp.zeros((3, 3)))
