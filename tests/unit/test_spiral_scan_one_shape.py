"""Every wavevector of a spin-spiral scan is one shape, so the SCF compiles once.

``OPEN.md`` Part XXIII item 9, half (b). Moving a spiral to a new ``q`` rebuilds
both plane-wave spheres, ``k + q/2`` and ``k - q/2``, and each wavevector padded
them to its own widest sphere and its own stick count: on the hydrogen chain the
padded width runs from 1532 to 1544 over eight wavevectors, and every new width
compiled the whole SCF stack again. The scan now pads every point to the widths
of all of them at once, as a band path does block by block.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat import Calculator
from defumat.workflows import spiral

pytestmark = pytest.mark.unit

H_CHAIN = Path(__file__).resolve().parents[1] / "data" / "qe" / "h-chain-spiral.in"
WAVEVECTORS = [(0.0, 0.0, 0.0), (0.0, 0.0, 1 / 16), (0.0, 0.0, 1 / 4)]


def _hamiltonian(calculation):
    potential = jnp.zeros((calculation.nspin_mag,) + tuple(calculation.basis.dense.grid))
    return calculation.hamiltonian(potential)[0]


def test_the_wavevectors_of_a_scan_share_one_shape(pseudo_dir):
    calculation = Calculator.from_file(H_CHAIN, pseudo_dir=pseudo_dir).calculation

    # The control: alone, each wavevector pads to its own sphere.
    alone = [calculation.at_spiral_q(q) for q in WAVEVECTORS]
    assert len({moved.basis.npwx for moved in alone}) > 1

    widths = spiral.scan_widths(calculation, WAVEVECTORS)
    scanned = [calculation.at_spiral_q(q, widths=widths) for q in WAVEVECTORS]
    shapes = {(moved.basis.planewaves.indices.shape, moved.sticks.columns.shape,
               moved.hamiltonian_npw) for moved in scanned}
    nk = 2 * calculation.system.kpoints.nk
    assert shapes == {((nk, widths[0]), (nk, widths[1]), widths[2])}

    # One treedef and one set of leaf shapes for the operator every compiled
    # unit of the SCF takes, which is what "compiles once" rests on.
    hamiltonians = [_hamiltonian(moved) for moved in scanned]
    structures = {jax.tree_util.tree_structure(h) for h in hamiltonians}
    leaf_shapes = {tuple(np.shape(leaf) for leaf in jax.tree_util.tree_leaves(h))
                   for h in hamiltonians}
    assert len(structures) == 1 and len(leaf_shapes) == 1

    # Each sphere is still the one its wavevector asks for, padded and nothing more.
    for own, padded in zip(alone, scanned):
        width = own.basis.npwx
        assert np.array_equal(np.asarray(own.basis.planewaves.indices),
                              np.asarray(padded.basis.planewaves.indices)[:, :width])
        assert not np.asarray(padded.basis.planewaves.mask)[:, width:].any()

    # A later move without widths does not inherit the scan's floor.
    again = scanned[0].at_spiral_q(WAVEVECTORS[1])
    assert again.hamiltonian_npw == alone[1].hamiltonian_npw
    assert again.basis.npwx == alone[1].basis.npwx
