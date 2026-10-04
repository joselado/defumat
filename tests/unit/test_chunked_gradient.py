"""The force and the stress walked a k-chunk at a time.

``GPU-MEMORY-NEXT.md`` item 3 (:mod:`defumat.forces.chunked`). One ``grad`` of
the frozen energy tapes the whole k axis; the chunked route splits the energy
into a separable part and a global part of the whole sums (``becsum``, the
smooth density, ``ns``) and pulls the global gradient back through each chunk.
It is exact, so the standard is round-off against the single pass -- at the
same frozen state, with a short last chunk so the zero-weight padding is
exercised. A streamed state (a host array) must take the route without being
asked, which is what the last test checks.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.calculator import Calculator
from defumat.forces import compute_forces
from defumat.forces.autodiff import _energy_gradient as force_gradient
from defumat.forces.chunked import chunked_gradient, wants_chunks
from defumat.forces.energy import FrozenState, frozen_energy, hoisted, state_from_result
from defumat.stress.autodiff import _energy_gradient as strain_gradient

pytestmark = pytest.mark.unit

PSEUDO = "tests/data/pseudo"

#: Ultrasoft silicon, displaced off its symmetric sites so the force is real,
#: on the whole 2x2x2 grid: eight k-points, chunks of three, a short last one.
CELL = """
 &control
    calculation = 'scf'
 /
 &system
    ibrav = 2, celldm(1) = 10.2, nat = 2, ntyp = 1,
    ecutwfc = 12.0, ecutrho = 96.0, nosym = .true.
 /
 &electrons
    conv_thr = 1.0d-8
 /
ATOMIC_SPECIES
 Si 28.086 Si.pz-n-rrkjus_psl.0.1.UPF
ATOMIC_POSITIONS alat
 Si 0.01 0.00 0.00
 Si 0.26 0.24 0.25
K_POINTS (automatic)
 2 2 2 0 0 0
"""


def _converged(**options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_text(CELL, PSEUDO, announce=False)
        return calculator, calculator.get_scf(**options)


@pytest.mark.slow  # 17 s on the CPU, most of it compiling the two routes
def test_the_chunked_force_and_stress_are_the_single_pass():
    calculator, result = _converged()
    calculation = calculator.calculation
    state = state_from_result(result)
    positions = calculation.system.structure.positions
    zero = jnp.zeros((3, 3))

    compiled, geometry = force_gradient(calculation)
    force = np.asarray(compiled(positions, state, hoisted(calculation), geometry))
    compiled, geometry = strain_gradient(calculation)
    strain = np.asarray(compiled(zero, state, hoisted(calculation), geometry))
    energy, chunked_force = chunked_gradient(calculation, state, "positions",
                                             positions, k_batch=3)
    _, chunked_strain = chunked_gradient(calculation, state, "strain", zero,
                                         k_batch=3)

    assert np.max(np.abs(force)) > 1e-2, "the geometry must carry a force"
    assert float(energy) == pytest.approx(
        float(frozen_energy(calculation, positions, state, spinors=True)), abs=1e-10)
    np.testing.assert_allclose(np.asarray(chunked_force), force, atol=1e-12)
    np.testing.assert_allclose(np.asarray(chunked_strain), strain, atol=1e-12)


def test_a_streamed_state_takes_the_chunked_route_without_being_asked():
    """The state is a host array; it is walked, not moved to the device whole."""
    calculator, result = _converged(wfc_store="stream")
    calculation = calculator.calculation
    streamed = state_from_result(result)
    assert isinstance(streamed.wavefunctions, np.ndarray)
    assert wants_chunks(calculation, streamed)
    on_device = FrozenState(
        wavefunctions=jnp.asarray(streamed.wavefunctions),
        weights=streamed.weights, eigenvalues=streamed.eigenvalues,
        entropy=streamed.entropy)
    assert not wants_chunks(calculation, on_device)
    np.testing.assert_allclose(
        compute_forces(calculation, streamed).unsymmetrized,
        compute_forces(calculation, on_device).unsymmetrized, atol=1e-12)


def test_a_moved_calculation_reuses_the_compiled_force_passes():
    """A relaxation step does not recompile the chunked force or stress.

    ``at_positions`` and ``at_cell`` copy the instance dict and the passes take
    the geometry as arguments (`OPEN.md` Part XXIII item 7), so a moved
    calculation -- the atoms or the cell -- reuses them, and the gradient there
    is still the single pass's at the new geometry.
    """
    calculator, result = _converged()
    calculation = calculator.calculation
    state = state_from_result(result)
    positions = calculation.system.structure.positions
    zero = jnp.zeros((3, 3))
    chunked_gradient(calculation, state, "positions", positions, k_batch=3)
    chunked_gradient(calculation, state, "strain", zero, k_batch=3)
    passes = dict(calculation._chunked_gradient[1])

    nudged = positions + jnp.asarray([[0.01, 0.0, 0.0], [0.0, 0.0, 0.0]])
    moved = calculation.at_positions(nudged)
    _, gradient = chunked_gradient(moved, state, "positions", nudged, k_batch=3)
    assert moved._chunked_gradient[1]["positions"] is passes["positions"]
    compiled, geometry = force_gradient(moved)
    single = compiled(nudged, state, hoisted(moved), geometry)
    np.testing.assert_allclose(np.asarray(gradient), np.asarray(single), atol=1e-12)

    at = np.asarray(calculation.system.cell.at) @ (np.eye(3) + np.diag([-0.01, 0.0, 0.005])).T
    cell = calculation.at_cell(jnp.asarray(at))
    _, strained = chunked_gradient(cell, state, "strain", zero, k_batch=3)
    assert cell._chunked_gradient[1]["strain"] is passes["strain"]
    compiled, geometry = strain_gradient(cell)
    single = compiled(zero, state, hoisted(cell), geometry)
    np.testing.assert_allclose(np.asarray(strained), np.asarray(single), atol=1e-12)
