"""The wavefunction transforms' layout: QE's sticks against one fused box transform.

``DEFUMAT_FFT_LAYOUT=box`` takes the sphere through the whole box with a single 3D
transform instead of QE's ``z``-over-the-sticks then ``xy``. The two are the same
transform summed in a different order, so ``H psi`` and the density must agree to
round-off, here 1e-12 relative, on a scalar and a noncollinear magnetic cell, and an
SCF must reach the same energy. The box layout is ``sticks = None`` on the
calculation, so the movers that slice or rebuild the layout are checked to keep it.
The dial itself is checked for its default and its override.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.batching import FFT_LAYOUTS, resolve_fft_layout
from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf.driver import Calculation, run_scf
from defumat.system import build_system

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]

SILICON = """\
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.2, nat = 2, ntyp = 1, ecutwfc = 12.0{extra}
/
&electrons
  conv_thr = 1.0d-11
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""

MAGNETIC = (",\n  noncolin = .true., nosym = .true., noinv = .true.,\n"
            "  starting_magnetization(1) = 0.5, angle1(1) = 45.0, angle2(1) = 30.0")


def _setup(layout: str, monkeypatch, extra: str = ""):
    monkeypatch.setenv("DEFUMAT_FFT_LAYOUT", layout)
    with tempfile.NamedTemporaryFile("w", suffix=".in", delete=False) as handle:
        handle.write(SILICON.format(extra=extra))
    system = build_system(read_pw_input(Path(handle.name)))
    pseudos = tuple(read_upf(REPO / "tests" / "data" / "pseudo" / s.pseudo_file)
                    for s in system.structure.species)
    return system, pseudos, Calculation(system, pseudos)


def _hamiltonian(calculation: Calculation):
    potential = calculation.potential(calculation.starting_density())
    return calculation.hamiltonian(potential.v_scf)[0]


def test_the_dial_defaults_to_sticks_and_takes_an_override(monkeypatch):
    monkeypatch.delenv("DEFUMAT_FFT_LAYOUT", raising=False)
    assert resolve_fft_layout() == "sticks"
    monkeypatch.setenv("DEFUMAT_FFT_LAYOUT", "BOX")
    assert resolve_fft_layout() == "box"
    assert resolve_fft_layout("sticks") == "sticks"
    assert set(FFT_LAYOUTS) == {"sticks", "box"}
    with pytest.raises(ValueError, match="fft layout"):
        resolve_fft_layout("fused")


@pytest.mark.parametrize("extra", ["", MAGNETIC], ids=["scalar", "spinor"])
def test_h_psi_is_the_same_in_both_layouts(monkeypatch, extra):
    _, _, sticks = _setup("sticks", monkeypatch, extra)
    _, _, box = _setup("box", monkeypatch, extra)
    assert sticks.sticks is not None and box.sticks is None
    h_sticks, h_box = _hamiltonian(sticks), _hamiltonian(box)
    ndim = sticks.system.npol * sticks.basis.planewaves.npwx
    rng = np.random.default_rng(0)
    psi = jnp.asarray(rng.standard_normal((4, ndim)) + 1j * rng.standard_normal((4, ndim)))
    reference = h_sticks.apply(psi, 1)
    result = h_box.apply(psi, 1)
    scale = float(jnp.max(jnp.abs(reference)))
    assert float(jnp.max(jnp.abs(result - reference))) < 1e-12 * scale


@pytest.mark.parametrize("extra", ["", MAGNETIC], ids=["scalar", "spinor"])
def test_the_density_is_the_same_in_both_layouts(monkeypatch, extra):
    _, _, sticks = _setup("sticks", monkeypatch, extra)
    _, _, box = _setup("box", monkeypatch, extra)
    states = sticks.starting_wavefunctions(sticks.hamiltonian(
        sticks.potential(sticks.starting_density()).v_scf), 4)
    rng = np.random.default_rng(1)
    weights = jnp.asarray(rng.uniform(0.1, 1.0, states.shape[:3]))
    reference = sticks.smooth_density(states, weights)
    result = box.smooth_density(states, weights)
    scale = float(jnp.max(jnp.abs(reference)))
    assert float(jnp.max(jnp.abs(result - reference))) < 1e-12 * scale


def test_the_movers_keep_the_box_layout(monkeypatch):
    _, _, box = _setup("box", monkeypatch)
    assert box.fft_layout == "box"
    assert box.at_rows(np.array([0, 1])).sticks is None
    assert box.at_kpoints(box.system.kpoints).sticks is None
    # a mover that rebuilds the sphere takes the layout from the calculation,
    # not from the environment of the moment
    monkeypatch.setenv("DEFUMAT_FFT_LAYOUT", "sticks")
    assert box.at_kpoints(box.system.kpoints).sticks is None


def test_an_scf_reaches_the_same_energy_in_both_layouts(monkeypatch):
    system, pseudos, sticks = _setup("sticks", monkeypatch)
    reference = run_scf(system, pseudos, calculation=sticks, conv_thr=1e-11)
    system, pseudos, box = _setup("box", monkeypatch)
    result = run_scf(system, pseudos, calculation=box, conv_thr=1e-11)
    assert result.converged and reference.converged
    assert result.total_energy == pytest.approx(reference.total_energy, abs=1e-9)
    assert result.iterations == reference.iterations


def test_the_chunked_force_runs_on_the_box_layout(monkeypatch):
    """A streamed state walks the k axis on row-subset calculations, whose ``sticks`` is ``None``."""
    import warnings

    from defumat.calculator import Calculator
    from defumat.forces import compute_forces
    from defumat.forces.chunked import wants_chunks
    from defumat.forces.energy import FrozenState, state_from_result
    from tests.unit.test_chunked_gradient import CELL, PSEUDO

    monkeypatch.setenv("DEFUMAT_FFT_LAYOUT", "box")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_text(CELL, PSEUDO, announce=False)
        result = calculator.get_scf(wfc_store="stream")
    calculation = calculator.calculation
    assert calculation.sticks is None
    streamed = state_from_result(result)
    assert wants_chunks(calculation, streamed)
    on_device = FrozenState(
        wavefunctions=jnp.asarray(streamed.wavefunctions),
        weights=streamed.weights, eigenvalues=streamed.eigenvalues,
        entropy=streamed.entropy)
    chunked = compute_forces(calculation, streamed).unsymmetrized
    assert np.max(np.abs(chunked)) > 1e-2, "the geometry must carry a force"
    np.testing.assert_allclose(
        chunked, compute_forces(calculation, on_device).unsymmetrized, atol=1e-12)
