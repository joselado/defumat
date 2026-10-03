"""A long band path in memory mode is built a block of points at a time.

``GPU-MEMORY-NEXT.md`` item 6. Moved onto with ``at_kpoints``, a band path held
every point's per-k tables on the device while its streamed solve walked the
points in chunks -- 1.65 MB a point on the NbSe2 monolayer. Where the store
streams and the tables would pass ``KSET_BLOCK_BYTES``,
:func:`~defumat.workflows.bands.run_bands` now builds each block of points as
its own ``at_kpoints``, padded to the whole path's ``(npwx, nsticks)``. Checked
on the CPU with the store forced to stream: the bands are the whole path's, and
every block has one shape -- a per-block width would recompile per block.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from defumat.calculator import Calculator
from defumat.scf.driver import Calculation
from defumat.system.kpoints import KPoints
from defumat.workflows import nscf
from defumat.workflows.bands import run_bands

pytestmark = pytest.mark.unit

SILICON = """
 &control
    calculation = 'scf'
 /
 &system
    ibrav = 2, celldm(1) = 10.2, nat = 2, ntyp = 1,
    ecutwfc = 12.0, nosym = .true.
 /
 &electrons
    conv_thr = 1.0d-10
 /
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS crystal
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS (automatic)
 2 2 2 0 0 0
"""


@pytest.mark.slow  # an SCF and two band paths, about 10 s
def test_a_path_in_blocks_is_the_path_whole(pseudo_dir, monkeypatch):
    monkeypatch.setenv("DEFUMAT_WFC_STORE", "stream")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_text(SILICON, pseudo_dir, announce=False,
                                          memory_mode="memory")
        result = calculator.get_scf()
    calculation = calculator.calculation
    path = KPoints.from_cartesian(
        np.linspace([0.0, 0.0, 0.0], [0.5, 0.5, 0.5], 7), np.full(7, 1.0 / 7))
    options = dict(nbnd=8, conv_thr=1e-10, calculation=calculation)

    monkeypatch.setattr(nscf, "KSET_BLOCK_BYTES", 2**40)
    assert nscf.kset_blocks(calculation, path) is None
    whole = run_bands(calculator.system, calculator.pseudos, result.density,
                      path, **options)

    # Three points a block on seven: the last block is padded with repeats.
    per_k = nscf._per_k_table_bytes(calculation)
    monkeypatch.setattr(nscf, "KSET_BLOCK_BYTES", 3 * per_k + 1)
    blocks, widths = nscf.kset_blocks(calculation, path)
    assert [live for _, live in blocks] == [3, 3, 1]

    shapes = []
    original = Calculation.at_kpoints

    def recording(self, kpoints, widths=None):
        moved = original(self, kpoints, widths=widths)
        shapes.append((moved.basis.planewaves.indices.shape,
                       moved.sticks.columns.shape, moved.hamiltonian_npw))
        return moved

    monkeypatch.setattr(Calculation, "at_kpoints", recording)
    blocked = run_bands(calculator.system, calculator.pseudos, result.density,
                        path, **options)
    assert len(shapes) == 3 and len(set(shapes)) == 1, shapes
    assert shapes[0][0][1] == widths[0] and shapes[0][1][1] == widths[1]
    # The eigensolver's static plane-wave counts too: the whole path's minimum,
    # which is the cap the path taken whole has.
    assert shapes[0][2] == widths[2]
    np.testing.assert_allclose(blocked.eigenvalues, whole.eigenvalues,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(np.asarray(blocked.kpoints.coords),
                               np.asarray(whole.kpoints.coords))
