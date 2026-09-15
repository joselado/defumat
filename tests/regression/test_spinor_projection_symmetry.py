"""``sym_proj_nc``: the group average of a projection when the columns carry a
spin as well as an ``m``.

A noncollinear column without spin-orbit coupling is ``|l m> x |sigma>``, so a
point-group operation turns two things at once -- the real harmonics among
themselves and the spin frame the spinor is written in -- and the operator the
group average runs over is the tensor product ``D^l x conj(U)``. Averaging the
``m`` indices alone leaves the spin frame where it was, which is a smooth,
normalised, plausible projection of the wrong thing.

**What the test is.** The projection is a sum over the Brillouin zone, so a
reduced wedge and the closed grid have to give the same Loewdin charge in every
column, and the wedge only does if the average is over the right operator. The
closed grid is the reference and it shares nothing with the wedge but the
ground state: different k-points, different weights, no symmetrisation at all.

**Two cells, because neither can do the job alone, and that is the point.**

* ``h4-cycloid-90.in`` is four hydrogens with their moments turned 90 degrees
  from one another. Hydrogen carries an ``s`` shell only, so ``D^l`` is the
  number one and the *whole* symmetrisation is the spin factor. Without it the
  wedge misses the closed grid by 3.8e-4 electrons, which is exactly what no
  symmetrisation at all gives; with it, 4.1e-6.
* ``ni-noncol-111.in`` is nickel with its moment along a three-fold axis, and
  it exists because ``ni-ldau-noncol.in`` -- the same crystal with the moment
  along ``z`` -- **cannot tell any spin convention from any other**. Every one
  of that cell's operations turns the spin about ``z``, so its 2x2 matrix is
  diagonal, the phase factors out of a modulus, and the right matrix, its
  transpose, its conjugate and the identity all agree to the last digit. On the
  three-fold cell they do not: ``U``, ``U^T`` and ``U^dagger`` are wrong by
  3.9e-2, 2.1e-2 and 2.1e-2 electrons where ``conj(U)`` is 1.1e-5.

Only ``conj(U)`` passes both, which is what pins it, and it is what the
harmonic factor already does one index over: the contraction is on the *first*
index, so what multiplies the projection is ``D^T = D^{-1}``, the inverse
operation, and the spin factor has to be the inverse of the same operation.

The spin-orbit case stays refused -- its columns are ``|j m_j>`` and
``sym_proj_so`` contracts ``d_matrix_so``'s ``D^j``, a different matrix -- and
the refusal is tested here so that lifting one regime is not read as lifting
both.
"""

from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest

from defumat import Calculator
from defumat.projwfc.projections import atomic_projections, calculation_channels

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """``CLAUDE.md``'s rule for a file that runs several cells: keep the
    results, drop the executables."""
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _converged(stem):
    calc = Calculator.from_file(CASES / stem, pseudo_dir=PSEUDO)
    return calc, calc.get_scf()


def _column_charges(calc, scf, symmetrize):
    """``sum_k sum_n f_nk |<phi_c|S|psi_nk>|^2``, one number per column.

    ``occupations`` already carries the k-point weight -- summing it over ``k``
    and ``n`` gives the electron count -- so the weights are not applied again
    here, which would square them.
    """
    projections = np.asarray(
        atomic_projections(calc.calculation, scf.wavefunctions,
                           symmetrize=symmetrize)
    )
    occupations = np.asarray(scf.occupations)
    if occupations.ndim == 2:
        occupations = occupations[None]
    return np.einsum("skcn,skn->c", projections, occupations)


#: ``(wedge, closed, what the symmetrisation has to reach, what dropping it
#: costs)``. The last two are measured, not chosen: see the module docstring.
PAIRS = (
    ("ni-noncol-111.in", "ni-noncol-111-nosym.in", 1.0e-4, 1.0e-2),
    ("h4-cycloid-90.in", "h4-cycloid-90-nosym.in", 1.0e-5, 1.0e-4),
)


@pytest.mark.parametrize("wedge, closed, tolerance, floor", PAIRS)
def test_the_wedge_reproduces_the_closed_grid(wedge, closed, tolerance, floor):
    calc_w, scf_w = _converged(wedge)
    calc_c, scf_c = _converged(closed)
    assert scf_w.converged and scf_c.converged
    # The two runs have to be the same ground state before their projections can
    # be compared at all.
    assert abs(scf_w.total_energy - scf_c.total_energy) < 1.0e-7

    symmetrised = _column_charges(calc_w, scf_w, symmetrize=True)
    reference = _column_charges(calc_c, scf_c, symmetrize=False)
    assert np.abs(symmetrised - reference).max() < tolerance

    # **And the guard has to fire.** A test that only checks the symmetrised
    # number passes on a cell where the symmetrisation does nothing, which is
    # how a group average over the wrong operator survives: the unsymmetrised
    # wedge must be measurably wrong on the same cell.
    raw = _column_charges(calc_w, scf_w, symmetrize=False)
    assert np.abs(raw - reference).max() > floor


def test_the_operator_is_unitary_per_operation():
    """``d_matrix_nc``'s own check, and it is cheap enough to keep.

    Each operation's ``D^l x conj(U)`` is a product of two unitary matrices, so
    the assembled coefficient block has to be unitary shell by shell. It is the
    one property that catches a mis-shaped gather without running anything.
    """
    from defumat.projwfc.projections import build_projection_symmetry

    calc, _ = _converged("ni-noncol-111.in")
    channels = calculation_channels(calc.calculation)
    symmetry = build_projection_symmetry(
        channels, calc.system.cell, calc.system.structure,
        calc.calculation.symmetries,
    )
    coefficients = np.asarray(symmetry.coefficients)  # (nsym, nwfc, mmax)
    assert np.iscomplexobj(coefficients)

    # One shell at a time -- ``(atom, wfc, l)``, not ``l`` alone, since two
    # shells of the same ``l`` are two gather blocks and stacking them is not a
    # square matrix.
    shells = {(c.atom, c.wfc, c.l) for c in channels}
    for atom, wfc, l in shells:
        columns = [c.index for c in channels
                   if (c.atom, c.wfc, c.l) == (atom, wfc, l)]
        block = coefficients[:, columns, : 2 * (2 * l + 1)]
        identity = np.eye(len(columns))
        for s in range(symmetry.nsym):
            product = block[s] @ block[s].conj().T
            assert np.abs(product - identity).max() < 1.0e-10


def test_spin_orbit_is_still_refused():
    """Lifting the noncollinear regime must not lift the spin-orbit one: the
    matrix ``sym_proj_so`` needs is ``D^j``, which nothing here builds."""
    # The cell has to *reach* the guard, which needs ``use_symmetry`` as well
    # as ``lspinorb``: ``fe-kind1-noncol.in`` and ``i-atom-soc.in`` both carry
    # sixteen operations and both run ``nosym``, so neither of them ever gets
    # there and a test written on one reports a pass for a refusal that was
    # never consulted.
    calc = Calculator.from_file(CASES / "pt2-soc-force.in", pseudo_dir=PSEUDO)
    scf = calc.get_scf()
    with pytest.raises(NotImplementedError, match="spin-orbit"):
        atomic_projections(calc.calculation, scf.wavefunctions, symmetrize=True)
