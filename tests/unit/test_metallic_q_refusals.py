"""P132: where the metallic phonon at ``q`` is refused, checked without a ground state.

A metal at ``q = 0`` or at a reciprocal lattice vector needs ``ph.x``'s
``ef_shift``, which this route does not carry. The refusal is a function of the
cell and ``q`` alone, so it is checked here at gate speed;
``tests/regression/test_electron_phonon.py`` checks it through the calculator,
and checks that the k-chunked route, which once refused a metal, agrees with the
whole one.
"""

import numpy as np
import pytest

from defumat.response.phononq import _require_a_metallic_q
from defumat.system.cell import Cell

FCC = Cell.from_ibrav(2, [7.5, 0.0, 0.0, 0.0, 0.0, 0.0])


def _cartesian(crystal):
    return np.asarray(FCC.k_to_cartesian(np.asarray(crystal, dtype=float))) * FCC.tpiba


@pytest.mark.parametrize("crystal", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 2.0, 1.0)])
def test_a_metal_at_a_reciprocal_lattice_vector_is_refused(crystal):
    with pytest.raises(NotImplementedError, match="ef_shift"):
        _require_a_metallic_q(FCC, _cartesian(crystal))


@pytest.mark.parametrize("crystal", [(0.0, 0.0, 0.5), (0.25, 0.0, 0.0), (1.0, 0.0, 1e-6)])
def test_a_metal_away_from_the_lattice_runs(crystal):
    _require_a_metallic_q(FCC, _cartesian(crystal))

