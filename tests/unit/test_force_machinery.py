"""Fast checks on the force machinery that need no QE reference.

The numbers against Quantum ESPRESSO are in ``tests/regression/test_forces.py``;
what is here is the structure around them -- the identity that gates the whole
autodiff path, the symmetry a force must have, the registries, and the two
refusals.
"""

from pathlib import Path

import numpy as np
import pytest

from pypresso.forces import compute_forces, force_methods, frozen_energy, state_from_result
from pypresso.forces.registry import get_force_method
from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.relax import get_ion_dynamics, ion_dynamics_schemes
from pypresso.relax.bfgs import BFGS
from pypresso.scf import Calculation, run_scf
from pypresso.system import build_system
from pypresso.system.structure import Structure
from pypresso.system.symmetry import (
    atom_mapping,
    check_symmetry,
    find_symmetries,
    symmetrize_vector,
)

pytestmark = pytest.mark.unit

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"


@pytest.fixture(scope="module")
def silicon(pseudo_dir):
    """The ideal two-atom cell, converged once for the whole module."""
    system = build_system(read_pw_input(CASES / "si2-nc-relax.in"))
    # ...at its *undisplaced* geometry, which is the case with the symmetry.
    import equinox as eqx
    import jax.numpy as jnp

    positions = np.array(system.structure.positions)
    positions[1] = np.array([0.25, 0.25, 0.25]) * float(system.cell.alat)
    system = eqx.tree_at(
        lambda s: s.structure.positions, system, jnp.asarray(positions)
    )
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-10)
    return system, calculation, result


def test_the_frozen_energy_reproduces_the_scf_total(silicon):
    """The gate on the whole autodiff path.

    ``frozen_energy`` is only the force if it is first the *energy*: it has to
    reassemble QE's decomposition out of different pieces (the bare ``D_ij``
    rather than the self-consistent one, the augmentation charge inside the
    density rather than inside ``deeq``) and land on the same number. If it does
    not, its gradient is the derivative of something else.
    """
    system, calculation, result = silicon
    energy = frozen_energy(
        calculation, system.structure.positions, state_from_result(result)
    )
    assert float(energy) == pytest.approx(result.total_energy, abs=1e-9)


@pytest.mark.parametrize("method", ["autodiff", "analytic"])
def test_the_force_on_a_perfect_crystal_vanishes(silicon, method):
    """Symmetry, not convergence: diamond's atoms are at a fixed point of it."""
    _, calculation, result = silicon
    forces = compute_forces(calculation, result, method=method)
    assert np.abs(forces.forces).max() < 1e-10


def test_symmetrize_vector_projects_onto_the_allowed_directions(silicon):
    """``symvector`` keeps what the crystal allows and removes the rest.

    Diamond silicon's site symmetry allows no force at all, so anything at all
    is projected to zero -- which is the strongest form the statement can take,
    and the reason a force on a symmetric crystal is exactly zero rather than
    small.
    """
    system, calculation, _ = silicon
    mapping = atom_mapping(system.cell, system.structure, calculation.symmetries)
    noise = np.array([[0.3, -0.1, 0.2], [0.05, 0.4, -0.7]])
    projected = np.asarray(
        symmetrize_vector(noise, system.cell, calculation.symmetries, mapping)
    )
    assert np.abs(projected).max() < 1e-12


def test_check_symmetry_sees_a_broken_structure(silicon):
    """``checkallsym``: the group is found once and checked afterwards."""
    system, calculation, _ = silicon
    assert check_symmetry(system.cell, system.structure, calculation.symmetries)

    positions = np.array(system.structure.positions)
    positions[1] += np.array([0.1, 0.0, 0.0])
    moved = Structure(
        positions=positions,
        types=system.structure.types,
        species=system.structure.species,
    )
    assert not check_symmetry(system.cell, moved, calculation.symmetries)


def test_if_pos_is_read_and_freezes_a_coordinate(qe_testsuite):
    """The trailing flags of an ``ATOMIC_POSITIONS`` line are not decoration."""
    system = build_system(read_pw_input(qe_testsuite / "pw_relax" / "relax.in"))
    assert system.structure.if_pos == ((1, 1, 1), (0, 0, 0))
    assert np.array_equal(system.structure.free, [[1, 1, 1], [0, 0, 0]])


def test_the_registries_name_what_exists():
    assert set(force_methods()) == {"autodiff", "analytic"}
    assert get_force_method(None) is get_force_method("autodiff")
    with pytest.raises(ValueError, match="unknown force method"):
        get_force_method("hellmann-feynman-by-hand")

    assert "bfgs" in ion_dynamics_schemes()
    assert get_ion_dynamics(None) is BFGS
    with pytest.raises(NotImplementedError, match="ion_dynamics"):
        get_ion_dynamics("verlet")


def test_bfgs_finds_the_minimum_of_a_quadratic():
    """The optimizer on its own, with no electrons in sight.

    A quadratic is what BFGS is exact for: one Hessian update is enough to make
    the second step land on the minimum, so this both checks the algebra and
    pins the iteration count.
    """
    at = np.eye(3) * 10.0
    bfgs = BFGS(at, energy_thr=1e-10, grad_thr=1e-8)
    target = np.array([[0.0, 0.0, 0.0], [2.6, 0.0, 0.0]])
    spring = 0.7

    positions = np.array([[0.3, 0.0, 0.0], [3.0, 0.0, 0.0]])
    for step in range(10):
        displacement = positions - target
        energy = 0.5 * spring * np.sum(displacement**2)
        force = -spring * displacement
        positions, converged = bfgs.step(positions, energy, force)
        if converged:
            break

    assert converged
    assert step <= 3
    assert np.abs(positions - target).max() < 1e-10


def test_bfgs_respects_its_trust_radius():
    """A huge force does not produce a huge step (``trust_radius_ini``)."""
    bfgs = BFGS(np.eye(3) * 10.0)
    positions = np.zeros((1, 3))
    moved, _ = bfgs.step(positions, 0.0, np.array([[100.0, 0.0, 0.0]]))
    assert np.linalg.norm(moved - positions) == pytest.approx(0.5)


def test_a_noncollinear_force_is_refused_rather_than_approximated(pseudo_dir):
    """The spinor terms are not written, so the answer is not offered."""
    from pypresso.forces.energy import FrozenState, frozen_energy as energy_of

    system = build_system(read_pw_input(CASES / "pt-paw-scalar.in"))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    import dataclasses

    system = dataclasses.replace(system, nspin=4, lspinorb=False)
    calculation = object.__new__(Calculation)
    calculation.noncolin = True
    state = FrozenState(
        wavefunctions=np.zeros((1, 1, 1, 1)),
        weights=np.zeros((1, 1, 1)),
        eigenvalues=np.zeros((1, 1, 1)),
    )
    with pytest.raises(NotImplementedError, match="noncollinear"):
        energy_of(calculation, np.zeros((1, 3)), state)
