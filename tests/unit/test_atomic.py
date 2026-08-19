"""Pseudo-atomic orbitals, and what they are for.

They exist to give the eigensolver somewhere sensible to start. So the test that
matters is not that they have some particular value, but that starting from them
is *better* than starting from noise -- which is checkable directly, by
comparing how close each starting guess gets to the exact eigenvalues.
"""

import dataclasses
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.pseudo.radial import simpson_weights
from pypresso.pseudo.atomic import (
    atomic_channels,
    atomic_wavefunctions,
    count_atomic_wavefunctions,
)
from pypresso.scf.driver import Calculation
from pypresso.scf.potential import v_of_rho
from pypresso.solvers import dense_eigensolver_all
from pypresso.solvers.davidson import starting_vectors
from pypresso.solvers.subspace import rayleigh_ritz
from pypresso.system import build_system
from pypresso.system.cell import Cell

pytestmark = pytest.mark.unit

BENCHMARK = Path(__file__).resolve().parents[2] / "benchmarks" / "si-1k.in"
NBND = 4


@pytest.fixture(scope="module")
def silicon(pseudo_dir):
    system = build_system(read_pw_input(BENCHMARK))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    potential = v_of_rho(calculation.starting_density(), calculation.basis.dense, system.cell)
    return system, pseudos, calculation, calculation.hamiltonian(potential.v_scf)[0]


def test_the_channel_count_is_one_per_m_per_orbital(silicon):
    """Silicon's 3s and 3p give 1 + 3 channels per atom, 8 for the two atoms."""
    _, pseudos, _, _ = silicon
    assert [(nb, l) for nb, l, _ in atomic_channels(pseudos[0])] == [
        (0, 0), (1, 1), (1, 1), (1, 1)
    ]
    system, pseudos, _, _ = silicon
    assert count_atomic_wavefunctions(pseudos, system.structure) == 8


def test_the_tabulated_orbitals_are_normalised(silicon):
    """The premise everything else rests on: ``int (r chi)^2 dr = 1``."""
    _, pseudos, _, _ = silicon
    pseudo = pseudos[0]
    weights = np.asarray(simpson_weights(pseudo.rab))
    for orbital in pseudo.orbitals:
        assert float(np.sum(np.asarray(orbital.chi) ** 2 * weights)) == pytest.approx(1.0, abs=1e-5)


def test_the_normalisation_is_right_in_the_isolated_atom_limit():
    """``4 pi / sqrt(Omega)`` checked where the answer is known.

    In the real silicon cell the Bloch sum of an atomic orbital does *not* have
    unit norm -- it is ``1 + sum_R e^{ikR} <chi|chi(.-R)>``, and at alat = 10.2
    bohr the 3s orbitals of neighbouring cells overlap enough to put the norm at
    1.12 while the 3p, whose overlap enters with the opposite sign, sits at 0.92.
    That is physics, not a normalisation error, and the way to tell the two apart
    is to pull the images apart: as the cell grows the overlaps vanish and the
    norm must go to one. If the prefactor were wrong it would converge to
    something else.
    """
    system = build_system(read_pw_input(BENCHMARK))
    pseudo_dir = Path(__file__).parent.parent / "data" / "pseudo"
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    fractional = np.asarray(system.structure.positions) @ np.linalg.inv(
        np.asarray(system.cell.at)
    )

    norms = []
    for alat in (10.2, 24.0):
        cell = Cell.from_ibrav(2, [alat, 0, 0, 0, 0, 0])
        structure = dataclasses.replace(
            system.structure, positions=cell.to_cartesian(fractional)
        )
        stretched = dataclasses.replace(system, cell=cell, structure=structure)
        calculation = Calculation(stretched, pseudos)
        wfc = atomic_wavefunctions(
            pseudos, stretched.structure, stretched.cell, calculation.basis.dense,
            calculation.basis.planewaves, stretched.kpoints,
        )
        norms.append(np.linalg.norm(np.asarray(wfc)[0], axis=1))

    assert np.abs(norms[0] - 1.0).max() > 0.05, "the dense cell should show image overlap"
    assert norms[1] == pytest.approx(np.ones(8), abs=1e-3), "the isolated limit must be 1"


def test_the_atomic_guess_beats_a_random_one(silicon):
    """The point of the whole exercise, measured rather than asserted.

    Both guesses are given the same treatment -- a Rayleigh-Ritz rotation inside
    their own span -- and compared against the exact eigenvalues. The atomic
    orbitals span the occupied states of the isolated atoms, which is most of
    what the crystal's occupied states are.
    """
    system, pseudos, calculation, hamiltonian = silicon
    exact, _ = dense_eigensolver_all(hamiltonian, NBND)
    exact = np.asarray(exact)[0]

    atomic = atomic_wavefunctions(
        pseudos, system.structure, system.cell, calculation.basis.dense,
        calculation.basis.planewaves, system.kpoints,
    )[0]
    random = starting_vectors(
        None, 8, calculation.basis.npwx, calculation.kinetic[0],
        calculation.basis.planewaves.mask[0], atomic.dtype,
    )

    from_atomic, _ = rayleigh_ritz(hamiltonian, 0, atomic, NBND)
    from_random, _ = rayleigh_ritz(hamiltonian, 0, random, NBND)

    atomic_error = np.abs(np.asarray(from_atomic) - exact).max()
    random_error = np.abs(np.asarray(from_random) - exact).max()
    assert atomic_error < random_error
    assert atomic_error < 0.05, "the atomic guess should be within tens of mRy"


def test_rayleigh_ritz_is_variational(silicon):
    """Ritz values sit above the true eigenvalues; a lower one means a bug."""
    _, _, _, hamiltonian = silicon
    exact, vectors = dense_eigensolver_all(hamiltonian, NBND)
    values, _ = rayleigh_ritz(hamiltonian, 0, jnp.asarray(vectors[0]), NBND)
    assert np.all(np.asarray(values) >= np.asarray(exact)[0] - 1e-9)
