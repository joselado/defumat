"""P132: where the metallic phonon at ``q`` is refused, checked without a ground state.

A metal at ``q = 0`` or at a reciprocal lattice vector needs ``ph.x``'s
``ef_shift``, which this route does not carry, and a metal on the k-chunked
route would meet a chunk solver built for an insulator. Both refusals are a
function of the cell and ``q`` alone, so they are checked here at gate speed;
``tests/regression/test_electron_phonon.py`` checks the first through the
calculator.
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
        _require_a_metallic_q(FCC, _cartesian(crystal), streamed=False)


@pytest.mark.parametrize("crystal", [(0.0, 0.0, 0.5), (0.25, 0.0, 0.0), (1.0, 0.0, 1e-6)])
def test_a_metal_away_from_the_lattice_runs(crystal):
    _require_a_metallic_q(FCC, _cartesian(crystal), streamed=False)


def test_a_metal_on_the_chunked_route_is_refused():
    with pytest.raises(NotImplementedError, match="smearing=None"):
        _require_a_metallic_q(FCC, _cartesian((0.25, 0.0, 0.0)), streamed=True)
