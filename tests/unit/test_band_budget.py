"""The band batch is resolved once, carried everywhere, and chosen from the card.

``GPU-MEMORY-NEXT.md`` item 9. Memory mode bounds what grows with the k-mesh;
what is left on a large cell is one k-point's working set, and its largest term
is the band loop through the grid. Three things are checked here, all on the
CPU:

* the :class:`~defumat.scf.driver.Calculation` carries the value it resolved
  into every Hamiltonian it builds, and the value does not move the answer;
* the size estimate that chooses it counts what memory mode actually puts on
  the device -- one streamed chunk of the store, not the whole set -- and the
  start, which on a large cell is where the peak was measured
  (`MEMORY-AUDIT.md` D10);
* the choice keeps the whole block wherever it fits, moves only when it does
  not, and yields to an explicit value or ``DEFUMAT_BAND_BATCH``.
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from defumat import batching
from defumat.calculator import Calculator
from defumat.scf.driver import Calculation, resolve_band_batch_for, run_scf
from defumat.sizing import BandBatchChoice, choose_band_batch, estimate_size

pytestmark = pytest.mark.unit


SILICON = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1, ecutwfc = 12.0, nbnd = 8
/
&electrons
  conv_thr = 1e-10
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""


def _calculator(pseudo_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Calculator.from_text(SILICON, pseudo_dir, announce=False)


def test_every_hamiltonian_carries_the_calculations_band_batch(pseudo_dir):
    calculator = _calculator(pseudo_dir)
    calculation = Calculation(calculator.system, calculator.pseudos, band_batch=3)
    assert calculation.band_batch == 3
    v_scf = jnp.zeros((calculation.nspin_mag,) + calculation.basis.dense.grid)
    assert {h.band_batch for h in calculation.hamiltonian(v_scf)} == {3}
    # And the copies a moved geometry or a new k-set makes keep it.
    assert calculation.at_positions(
        jnp.asarray(calculator.system.structure.positions)).band_batch == 3


def test_the_band_batch_moves_the_energy_by_round_off_only(pseudo_dir):
    """``map_bands`` is exact; the density's band sum changes only its order."""
    calculator = _calculator(pseudo_dir)
    energies = []
    for band_batch in (None, 3):
        calculation = Calculation(calculator.system, calculator.pseudos,
                                  band_batch=band_batch)
        result = run_scf(calculator.system, calculator.pseudos,
                         calculation=calculation, conv_thr=1e-10)
        energies.append(result.total_energy)
    assert abs(energies[0] - energies[1]) < 1e-11


def test_a_streamed_store_is_sized_as_one_chunk_and_the_start_is_a_stage(pseudo_dir):
    calculator = _calculator(pseudo_dir)
    device = estimate_size(calculator.system, calculator.pseudos, k_batch=1,
                           band_batch=None, wfc_store="device")
    stream = estimate_size(calculator.system, calculator.pseudos, k_batch=1,
                           band_batch=None, wfc_store="stream")
    whole = device.arrays["wavefunctions (nspin,nk,nbnd,ndim)"]
    chunk = stream.arrays["wavefunctions, one streamed chunk (k,nbnd,ndim)"]
    assert whole == device.nk * chunk
    # ``max(natomwfc, nbnd)``: Si.pz-vbc has an s and a p orbital, 4 per atom.
    assert stream.start_vectors == max(8, stream.nbnd)
    resident = sum(size for name, size in stream.arrays.items()
                   if name not in stream._SUPERSEDED)
    assert stream.peak_bytes == resident + max(
        stream.setup_transient, stream.eigensolver_buffer, stream.start_buffer)
    # The span is whole-k only when the store is not streamed.
    assert device.start_buffer > stream.start_buffer


@pytest.mark.parametrize("band_batch", [1, 3, 4, 5, None])
def test_re_evaluating_at_a_band_batch_is_the_estimate_at_it(pseudo_dir, band_batch):
    calculator = _calculator(pseudo_dir)
    base = estimate_size(calculator.system, calculator.pseudos, band_batch=None)
    direct = estimate_size(calculator.system, calculator.pseudos,
                           band_batch=band_batch)
    moved = base.at_band_batch(band_batch)
    assert abs(moved.eigensolver_buffer - direct.eigensolver_buffer) <= 1
    assert abs(moved.start_buffer - direct.start_buffer) <= 1
    assert moved.band_batch == direct.band_batch


def test_a_batch_that_does_not_divide_the_bands_pays_its_tail(pseudo_dir):
    calculator = _calculator(pseudo_dir)
    base = estimate_size(calculator.system, calculator.pseudos, band_batch=None)
    # 8 bands: 3 is a 3-block and a 2-tail, so five boxes; 4 divides, so four.
    three, four = base.at_band_batch(3), base.at_band_batch(4)
    assert three.eigensolver_buffer - four.eigensolver_buffer == pytest.approx(
        base.band_box_bytes, abs=1)


def test_the_choice_keeps_the_whole_block_whenever_it_fits(pseudo_dir):
    calculator = _calculator(pseudo_dir)
    choice = choose_band_batch(calculator.system, calculator.pseudos,
                               available=10**13)
    assert choice.band_batch is None and choice.fits


def test_the_choice_moves_when_the_block_does_not_fit_and_the_guard_fires(pseudo_dir):
    """Squeeze the budget and check both that it moves and that it refuses."""
    calculator = _calculator(pseudo_dir)
    system, pseudos = calculator.system, calculator.pseudos
    base = estimate_size(system, pseudos, k_batch=1, band_batch=None,
                         projectors="rebuild", wfc_store="stream")
    # Just below what the whole block needs, at the chooser's own headroom.
    squeezed = int(0.999 * base.peak_bytes / 0.6)
    choice = choose_band_batch(system, pseudos, available=squeezed)
    assert choice.fits and choice.band_batch is not None
    assert choice.band_batch <= base.nbnd
    assert choice.estimate <= 0.6 * squeezed
    # Fewest blocks first: nothing with as few blocks fits in fewer boxes.
    assert base.nbnd % choice.band_batch == 0 or choice.band_batch == base.nbnd
    # And the guard: a budget nothing fits in says so rather than passing.
    starved = choose_band_batch(system, pseudos, available=1)
    assert starved.band_batch == 1 and not starved.fits


def test_an_explicit_value_and_the_environment_beat_the_chooser(pseudo_dir, monkeypatch):
    calculator = _calculator(pseudo_dir)
    system, pseudos = calculator.system, calculator.pseudos
    # Pretend to be on a card whose budget forces two bands at a time.
    monkeypatch.setattr(batching, "_backend", lambda: "gpu")
    monkeypatch.setattr(
        "defumat.sizing.choose_band_batch",
        lambda *a, **k: BandBatchChoice(band_batch=2, fits=True, estimate=1,
                                        available=10))
    monkeypatch.delenv("DEFUMAT_BAND_BATCH", raising=False)
    with pytest.warns(RuntimeWarning, match="2 bands at a time"):
        assert resolve_band_batch_for("default", "memory", system, pseudos) == 2
    # speed mode never asks the card about bands.
    assert resolve_band_batch_for("default", "speed", system, pseudos) is None
    monkeypatch.setenv("DEFUMAT_BAND_BATCH", "4")
    assert resolve_band_batch_for("default", "memory", system, pseudos) == 4
    assert resolve_band_batch_for(None, "memory", system, pseudos) is None
    assert resolve_band_batch_for(6, "memory", system, pseudos) == 6


def test_the_size_report_resolves_the_band_batch_the_way_the_run_does(pseudo_dir):
    calculator = _calculator(pseudo_dir)
    assert calculator.estimate().band_batch == calculator.calculation.band_batch


# -- a solve at more bands than the SCF ------------------------------------------


def _a_card_where_more_than(n, monkeypatch):
    """Pretend to be on a card where more than ``n`` bands do not fit whole.

    Memory mode's chooser answers two bands at a time past ``n``, and speed
    mode's check says no past ``n`` -- the guard is tested by feeding it the
    case that must trip it (``CLAUDE.md``), not by finding a cell too large.
    """
    from defumat import sizing

    def chooser(system, pseudos, nbnd=None, **kwargs):
        wide = nbnd is not None and nbnd > n
        return BandBatchChoice(band_batch=2 if wide else None, fits=True,
                               estimate=1, available=10)

    def speed(system, pseudos, nbnd=None, **kwargs):
        wide = nbnd is not None and nbnd > n
        return sizing.SpeedCheck(fits=not wide, estimate=10 * 2**30,
                                 available=4 * 2**30)

    monkeypatch.setattr(batching, "_backend", lambda: "gpu")
    monkeypatch.setattr("defumat.sizing.choose_band_batch", chooser)
    monkeypatch.setattr("defumat.sizing.speed_mode_fits", speed)
    monkeypatch.delenv("DEFUMAT_BAND_BATCH", raising=False)
    monkeypatch.delenv("DEFUMAT_K_BATCH", raising=False)


def test_a_solve_at_more_bands_re_chooses_the_band_batch(pseudo_dir, monkeypatch):
    """The dials were sized for the SCF's eight bands; sixty are re-sized.

    Measured before this existed: ``h40-chain-lsda.in`` in a 2.1 GB pool ran
    its SCF at eight bands at a time and died asking for 1.86 GiB when the same
    calculation was asked for 168 bands, still at eight.
    """
    calculator = _calculator(pseudo_dir)
    system, pseudos = calculator.system, calculator.pseudos
    # On the CPU nothing is re-sized, whatever the count.
    cpu = Calculation(system, pseudos)
    assert cpu.for_bands(60) is cpu

    _a_card_where_more_than(8, monkeypatch)
    calculation = Calculation(system, pseudos, memory_mode="memory")
    assert calculation.band_batch is None
    assert calculation.for_bands(8) is calculation
    with pytest.warns(RuntimeWarning, match="2 bands at a time"):
        wide = calculation.for_bands(60)
    assert wide is not calculation
    assert (wide.memory_mode, wide.k_batch, wide.band_batch) == ("memory", 1, 2)
    # Nothing rebuilt: the copy shares every array.
    assert wide.projector_core is calculation.projector_core
    assert wide.basis is calculation.basis
    # And an explicit value is what runs, at any band count.
    fixed = Calculation(system, pseudos, memory_mode="memory", band_batch=4)
    assert fixed.for_bands(60) is fixed


def test_speed_mode_falls_back_for_a_solve_that_would_not_fit(pseudo_dir, monkeypatch):
    """The SCF fits in speed mode; a sixty-band solve on it does not."""
    calculator = _calculator(pseudo_dir)
    system, pseudos = calculator.system, calculator.pseudos
    _a_card_where_more_than(8, monkeypatch)
    calculation = Calculation(system, pseudos, memory_mode="speed")
    assert (calculation.memory_mode, calculation.k_batch) == ("speed", None)
    with pytest.warns(RuntimeWarning, match="for a solve at 60 bands"):
        with pytest.warns(RuntimeWarning, match="2 bands at a time"):
            wide = calculation.for_bands(60)
    assert (wide.memory_mode, wide.k_batch, wide.band_batch) == ("memory", 1, 2)
    # A chunk the caller chose is not overridden.
    chosen = Calculation(system, pseudos, memory_mode="speed", k_batch=4)
    assert chosen.for_bands(60).k_batch == 4
