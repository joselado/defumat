"""A solve that asks for its step count and one that does not share an executable.

``OPEN.md`` Part III M6. ``return_steps`` was a static argument of the compiled
solver, so ``True`` and ``False`` were two compilations of the whole of it, and a
process that ran an SCF (which asks for the steps) and then a caller that does
not (the residual map, a topology workflow, electrostriction) compiled Davidson
twice at the same shapes. The compiled unit now always carries the two loop
counters and the host wrapper drops them when they were not asked for, so the
two calls hit one cache entry, and the values they return are the same bits.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf.driver import Calculation
from defumat.scf.potential import v_of_rho
from defumat.solvers import davidson
from defumat.system import build_system

pytestmark = pytest.mark.unit

BENCHMARK = Path(__file__).resolve().parents[2] / "benchmarks" / "si-1k.in"
NBND = 8


@pytest.fixture(scope="module")
def hamiltonian(pseudo_dir):
    system = build_system(read_pw_input(BENCHMARK))
    system = dataclasses.replace(system, occupations="smearing",
                                 degauss=0.02, smearing="mv")
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file)
                    for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    potential = v_of_rho(calculation.starting_density(), calculation.basis.dense,
                         system.cell)
    return calculation.hamiltonian(potential.v_scf)[0]


def test_asking_for_the_steps_does_not_compile_the_solver_again(hamiltonian):
    davidson.davidson_eigensolver_all.clear_cache()
    counted = davidson.davidson_eigensolver_all(
        hamiltonian, NBND, None, ethr=1e-8, max_iterations=40, return_steps=True)
    assert davidson._every_k._cache_size() == 1
    plain = davidson.davidson_eigensolver_all(
        hamiltonian, NBND, None, ethr=1e-8, max_iterations=40)

    assert davidson._every_k._cache_size() == 1, (
        "the solve without return_steps compiled a second executable at the "
        "same shapes"
    )
    assert len(counted) == 4 and len(plain) == 2
    for with_steps, without in zip(counted[:2], plain):
        assert np.array_equal(np.asarray(with_steps), np.asarray(without))
