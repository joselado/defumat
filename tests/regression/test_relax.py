"""P15 check: structural relaxation against Quantum ESPRESSO.

Two cases, deliberately different in kind:

* ``si2-nc-relax`` -- silicon with one atom pushed 0.02 alat off its site. The
  answer is known before the calculation runs: diamond's second atom sits at
  ``(1/4, 1/4, 1/4)`` and the relaxation has to put it back. It is also the case
  where the *symmetry* has to survive the trajectory -- the FFT grid and the
  k-point set were chosen for the starting geometry's group, and every step is
  checked against it.
* ``pw_relax/relax.in`` -- QE's own: a CO molecule with the oxygen frozen by
  ``if_pos``, ultrasoft, at Gamma. It tests three things the silicon case does
  not: a constrained coordinate, an augmentation charge that moves, and a
  starting geometry far enough from the minimum that BFGS has to reject a step.

What is compared is the *final* geometry and the *final* energy, not the path:
two BFGS implementations agreeing step for step would be a stronger statement
than either code makes, since the trajectory depends on where each one's SCF
stopped. The relaxed bond length is a property of the potential-energy surface
and is what a relaxation is for.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from defumat.io import read_qe_output
from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.system import build_system
from defumat.workflows.relax import run_relax
from tests.tolerances import FORCE_RY_BOHR, TOTAL_ENERGY_RY

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"

#: How closely the two codes must agree on where an atom ends up, in bohr. Both
#: stop when the largest force is below ``forc_conv_thr = 1e-3 Ry/bohr``, so
#: they stop at slightly different points on the same curve; the bound is what
#: that residual force is worth given silicon's stiffness (~0.5 Ry/bohr^2).
POSITION_BOHR = 5e-3


@lru_cache(maxsize=None)
def _relaxed(path: Path, pseudo_dir: Path):
    system = build_system(read_pw_input(path))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    return run_relax(system, pseudos, conv_thr=1e-10, verbose=False)


def _reference(name: str):
    path = CASES / f"reference.out.{name}"
    if not path.is_file():
        pytest.skip(f"no generated reference for {name}; run tools/generate_reference.py")
    return read_qe_output(path)


def test_displaced_silicon_relaxes_back_onto_its_site(pseudo_dir):
    result = _relaxed(CASES / "si2-nc-relax.in", pseudo_dir)
    reference = _reference("si2-nc-relax")

    assert result.converged
    assert np.abs(result.positions - reference.final_positions).max() < POSITION_BOHR
    assert result.total_energy == pytest.approx(
        reference.final_energy, abs=TOTAL_ENERGY_RY
    )
    assert np.abs(result.forces).max() < FORCE_RY_BOHR * 10


def test_the_relaxed_silicon_is_the_diamond_structure(pseudo_dir):
    """Independent of QE: the separation must come back to (1/4, 1/4, 1/4).

    The pair drifts along x as it relaxes -- both atoms feel equal and opposite
    forces and there is no preferred origin in a periodic crystal -- so what is
    checked is the *separation*, which is the physics.
    """
    result = _relaxed(CASES / "si2-nc-relax.in", pseudo_dir)
    alat = float(result.system.cell.alat)
    separation = (result.positions[1] - result.positions[0]) / alat
    assert np.abs(separation - 0.25).max() < 1e-3


def test_the_relaxation_ends_where_it_should_have(pseudo_dir):
    """Each ionic step lowers the energy, and the last one is the lowest."""
    result = _relaxed(CASES / "si2-nc-relax.in", pseudo_dir)
    energies = [step.total_energy for step in result.steps]
    assert energies[-1] == min(energies)
    assert result.steps[-1].max_force < result.steps[0].max_force


def test_a_frozen_atom_does_not_move(pseudo_dir, qe_testsuite):
    """QE's own CO relaxation, ``if_pos`` and all."""
    result = _relaxed(qe_testsuite / "pw_relax" / "relax.in", pseudo_dir)
    reference = _reference("pw_relax-relax")

    assert result.converged
    # the oxygen carried 0 0 0 in its ATOMIC_POSITIONS line
    assert np.abs(result.positions[1]).max() < 1e-12
    assert np.abs(result.positions - reference.final_positions).max() < POSITION_BOHR
    assert result.total_energy == pytest.approx(
        reference.final_energy, abs=TOTAL_ENERGY_RY
    )
