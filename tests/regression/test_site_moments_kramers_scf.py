"""The Kramers zero of a nonmagnetic spinor site moment, with an SCF behind it.

``OPEN.md`` Part XVIII item 1. At ``nspin = 4`` with no magnetization the
Hamiltonian commutes with time reversal, so the whole-zone site ``<L>`` and
``<S>`` vanish on every atom, and ``_symmetrise_axial`` completes a wedge cut
with ``k -> -k`` to that zero rather than averaging it. The host-side tests in
``tests/unit/test_site_moments_time_reversal.py`` hold the completion on a
stand-in; this file holds it on a converged state, against the whole grid.

Three iodine atoms at generic positions in an orthorhombic cell, so the group
is the identity alone (two identical atoms always have an inversion centre at
their midpoint), on ``I.rel-pbe-nc-dojo``, the one fully-relativistic
norm-conserving dataset with atomic wavefunctions that a nonmagnetic run here
can use. The ``2 2 1 1 1 0`` grid has four points and time reversal halves it
to two; ``nosym`` gives all four, where nothing is completed and the zero is a
measured cancellation between each state and its Kramers partner.

Measured on ``D22-0161``, 2026-10-09, at ``conv_thr = 1e-10``: the raw wedge
sum is 0.239 in ``<L>`` and 0.0162 in ``<S>`` on the worst component, the whole
grid gives 1.1e-9 and 1.0e-9, the two site charges agree to 2.7e-8 electrons
and the two total energies to every printed digit; the two SCFs take 8 s each.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from defumat import Calculator
from defumat.projwfc import angular_momentum

pytestmark = pytest.mark.slow

_P1 = """
&control
  calculation = 'scf'
/
&system
  ibrav = 0, nat = 3, ntyp = 1, ecutwfc = 25.0,
  occupations = 'smearing', smearing = 'mv', degauss = 0.02,
  noncolin = .true., lspinorb = .true.
  {extra}
/
&electrons
  conv_thr = 1e-10
/
ATOMIC_SPECIES
 I 126.9 I.rel-pbe-nc-dojo.UPF
CELL_PARAMETERS bohr
 8.00 0.00 0.00
 0.00 8.40 0.00
 0.00 0.00 8.90
ATOMIC_POSITIONS crystal
 I 0.00 0.00 0.00
 I 0.45 0.16 0.33
 I 0.17 0.60 0.66
K_POINTS automatic
 2 2 1 1 1 0
"""


@lru_cache(maxsize=2)
def _converged(pseudo_dir, extra):
    calculator = Calculator.from_text(_P1.format(extra=extra), pseudo_dir,
                                      announce=False)
    return calculator.calculation, calculator.get_scf()


def _raw(calculation, scf, monkeypatch):
    """The wedge sum with the completion replaced by the identity."""
    with monkeypatch.context() as patch:
        patch.setattr(angular_momentum, "_symmetrise_axial",
                      lambda orbital, spin, charge, calculation: (orbital, spin, charge))
        return angular_momentum.angular_momenta(calculation, scf)


def test_the_wedge_is_completed_to_the_zero_the_whole_grid_measures(
    pseudo_dir, monkeypatch
):
    """Zero on the wedge, a nonzero raw sum beside it, and round-off on the grid.

    The raw arm is the check that the completion fires: without it a zero on
    the wedge could not be told from a cell whose wedge sum was already zero.
    The charges tie the two runs together, since a site charge is invariant
    and needs no completion, so the wedge and the grid must agree on it to the
    SCF's own accuracy.
    """
    calculation, scf = _converged(pseudo_dir, "")
    assert calculation.symmetries.nsym == 1
    assert calculation.system.kpoints.nk == 2, "the premise: time reversal halved it"
    assert calculation.system.nspin_mag == 1

    completed = angular_momentum.angular_momenta(calculation, scf)
    np.testing.assert_array_equal(completed.orbital, 0.0)
    np.testing.assert_array_equal(completed.spin, 0.0)

    raw = _raw(calculation, scf, monkeypatch)
    assert np.abs(raw.orbital).max() > 1e-2
    assert np.abs(raw.spin).max() > 1e-3

    whole_calculation, whole = _converged(pseudo_dir, "nosym = .true.")
    assert whole_calculation.system.kpoints.nk == 4
    measured = angular_momentum.angular_momenta(whole_calculation, whole)
    assert np.abs(measured.orbital).max() < 1e-7
    assert np.abs(measured.spin).max() < 1e-7

    np.testing.assert_allclose(
        [atom.charge for atom in completed.atoms],
        [atom.charge for atom in measured.atoms], rtol=0, atol=1e-6,
    )
    assert scf.total_energy == pytest.approx(whole.total_energy, abs=1e-8)
