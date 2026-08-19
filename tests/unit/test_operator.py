"""The Hamiltonian operator, and the two ways of writing it as a matrix.

``Hamiltonian.matrix`` uses matrix-element formulas -- ``V(G - G')`` for the
local part, ``vkb D vkb^dagger`` for the nonlocal one -- which is fast but is a
*second* implementation of the same physics. ``matrix_by_application`` builds the
same matrix by applying the operator to every basis vector, which uses no
formula at all. The point of keeping both is this file: they must agree, and if
they ever stop agreeing one of the two is wrong.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf.driver import Calculation
from pypresso.scf.potential import v_of_rho
from pypresso.system import build_system

pytestmark = pytest.mark.unit

BENCHMARK = Path(__file__).resolve().parents[2] / "benchmarks" / "si-1k.in"


@pytest.fixture(scope="module")
def silicon(pseudo_dir):
    system = build_system(read_pw_input(BENCHMARK))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    potential = v_of_rho(calculation.starting_density(), calculation.basis.dense, system.cell)
    return calculation, calculation.hamiltonian(potential.v_scf)[0]


def test_the_two_matrix_builds_agree(silicon):
    """The formula and the operator must give the same matrix, to round-off."""
    _, hamiltonian = silicon
    direct = np.asarray(hamiltonian.matrix(0))
    applied = np.asarray(hamiltonian.matrix_by_application(0))

    scale = np.abs(applied).max()
    assert np.abs(direct - applied).max() < 1e-10 * scale


def test_the_two_matrix_builds_agree_with_padded_plane_waves(qe_testsuite, pseudo_dir):
    """The same, at every k of a multi-k run -- where the basis is padded.

    Different k-points keep different numbers of plane waves, so all but one are
    padded out to ``npwx``. The padding shares the index of G = 0, which is
    exactly the kind of thing a gather-based build can get wrong while the
    single-k case stays perfect.
    """
    system = build_system(read_pw_input(qe_testsuite / "pw_scf" / "scf.in"))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    potential = v_of_rho(calculation.starting_density(), calculation.basis.dense, system.cell)
    hamiltonian = calculation.hamiltonian(potential.v_scf)[0]

    assert len(set(calculation.basis.planewaves.npw)) > 1, "this case should have padding"
    for ik in range(system.kpoints.nk):
        direct = np.asarray(hamiltonian.matrix(ik))
        applied = np.asarray(hamiltonian.matrix_by_application(ik))
        assert np.abs(direct - applied).max() < 1e-10 * np.abs(applied).max()


def test_the_two_matrix_builds_give_the_same_spectrum(silicon):
    """What actually matters downstream: identical eigenvalues."""
    _, hamiltonian = silicon
    direct = np.linalg.eigvalsh(np.asarray(hamiltonian.matrix(0)))
    applied = np.linalg.eigvalsh(np.asarray(hamiltonian.matrix_by_application(0)))
    assert direct == pytest.approx(applied, abs=1e-10)


def test_the_matrix_reproduces_the_operator(silicon):
    """``H psi`` from the matrix equals ``H psi`` from ``apply``.

    This closes the loop: the matrix is not merely self-consistent between its
    two constructions, it is the operator the SCF actually uses.
    """
    _, hamiltonian = silicon
    npwx = hamiltonian.npwx
    generator = np.random.default_rng(0)
    psi = jnp.asarray(generator.normal(size=npwx) + 1j * generator.normal(size=npwx))
    psi = jnp.where(hamiltonian.mask[0], psi, 0.0)

    by_operator = np.asarray(hamiltonian.apply(psi, 0))
    by_matrix = np.asarray(hamiltonian.matrix(0) @ psi)
    assert by_matrix == pytest.approx(by_operator, abs=1e-10 * np.abs(by_operator).max())


def test_the_fallback_is_taken_when_the_grid_cannot_resolve_differences(silicon):
    """With ``ecutrho < 4 ecutwfc`` the gather would alias, so it is not used."""
    import dataclasses

    _, hamiltonian = silicon
    coarse = dataclasses.replace(hamiltonian, resolves_differences=False)
    assert np.asarray(coarse.matrix(0)) == pytest.approx(
        np.asarray(hamiltonian.matrix_by_application(0)), abs=1e-12
    )
