"""The states at ``k + q`` are S-orthonormal and are the states at the grid point.

The cell where a Davidson solve returned a zero eigenvector under a spurious
lowest root (``tests/unit/test_davidson_zero_root.py`` has the mechanism and
the captured matrices). ``q = (0.2, 0, 0)`` in units of 2 pi/alat is one step
of the 5x5x5 grid, so every ``k + q`` is another grid point and its states are
known twice: from the non-self-consistent solve at ``k + q``, started from
random vectors, and from the SCF at the grid point itself. Two invariants,
neither of which shares the subspace solve with the defect:

* every returned eigenvector has unit norm (norm-conserving, so ``S = 1``),
  which the zero vector failed by exactly 1;
* the ``k + q`` spectrum equals the grid point's, which the shifted spectrum
  failed by the gap to the next band, 0.1 Ry and more.

Which k-point breaks is round-off: 28 on one machine and 65 on another, so this
test is a net with no guarantee that a given machine reproduces the failure.
The unit test is the deterministic one.
"""

from pathlib import Path

import numpy as np
import pytest

from defumat import Calculator
from defumat.response import phononq

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASE = Path(__file__).resolve().parents[1] / "data" / "qe" / "al4-metal-k5-nosym.in"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"
NBND = 10
#: Bands compared against the grid point. Not all ten: at Gamma the SCF's top
#: three empty bands are a triplet at 1.033 Ry where the solve at ``k + q``
#: finds one at 0.990 Ry, so the SCF misses a state there, 0.4 Ry above the
#: Fermi level (0.61 Ry), which is a separate matter from the defect. Seven
#: bands hold every state below it, and a spurious root shifts all of them.
NCOMPARE = 7
#: The SCF converges an empty band only to ``empty_ethr``, 1e-5 Ry in the change
#: of its eigenvalue, so the comparison is held to that; the defect moved every
#: band by at least 0.1 Ry.
TOLERANCE = 1.0e-5


def test_states_at_k_plus_q_are_normalised_and_match_the_grid():
    calc = Calculator.from_file(CASE, pseudo_dir=PSEUDO)
    scf = calc.get_scf()
    calculation = calc.calculation
    v_scf = calculation.potential(scf.density).v_scf
    q = np.array([0.2, 0.0, 0.0]) * calculation.system.cell.tpiba
    # ``ethr = 1e-13``, the response's own and QE's floor, on every band: that
    # is what keeps the top band iterating long enough for the overlap to
    # reach the round-off floor.
    moved, _, eigenvalues, wavefunctions = phononq.states_at_k_plus_q(
        calculation, v_scf, q, nbnd=NBND, ethr=1e-13)

    norms = np.linalg.norm(np.asarray(wavefunctions), axis=-1)
    bad = np.argwhere(np.abs(norms - 1.0) > 1e-8)
    assert bad.size == 0, f"(spin, k, band) with a norm off 1: {bad.tolist()}"

    grid = np.asarray(calculation.system.kpoints.crystal(calculation.system.cell))
    shifted = np.asarray(moved.system.kpoints.crystal(calculation.system.cell))
    reference = np.asarray(scf.eigenvalues)
    worst = 0.0
    for ik, point in enumerate(shifted):
        delta = (grid - point + 0.5) % 1.0 - 0.5
        match = np.flatnonzero(np.all(np.abs(delta) < 1e-8, axis=1))
        assert match.size == 1, f"k + q of k-point {ik} is not on the grid"
        worst = max(worst, float(np.max(np.abs(
            eigenvalues[0, ik, :NCOMPARE] - reference[match[0], :NCOMPARE]))))
    assert worst < TOLERANCE, f"k + q spectrum off the grid point's by {worst:.3e} Ry"
