"""``k_batch = 'fit'``: the largest k-chunk the card holds, chosen where a Calculation is built.

Memory mode takes one k-point a call, which on a card pays a fixed cost per call
that batching over k amortises: 1.81x at 27 k-points on eight-atom silicon on an
RTX A2000, Davidson steps equal (``PERFORMANCE.md``, "Memory mode on a k-mesh").
``'fit'`` sizes the chunk from :func:`~defumat.sizing.estimate_size`, which matched
the measured peaks there to 3 per cent. Checked here, on the CPU:

* the choice is the whole mesh whenever it fits, otherwise the fewest calls and
  then the least padding, and says so when not even one k-point fits;
* ``'fit'`` is refused by name by a caller that has no system to size, and
  ``DEFUMAT_K_BATCH=fit`` is silent to one;
* a CPU keeps its default, a card takes the chooser, and only from a whole band
  block; the size report resolves the chunk the run takes.
"""

import warnings

import pytest

from defumat import batching
from defumat.calculator import Calculator
from defumat.scf.driver import Calculation, resolve_k_batch_for
from defumat.sizing import BandBatchChoice, KBatchChoice, choose_k_batch, estimate_size

pytestmark = pytest.mark.unit


SILICON = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1, ecutwfc = 12.0, nbnd = 8,
  nosym = .true.
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
 4 4 4 0 0 0
"""


def _calculator(pseudo_dir, **defaults):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Calculator.from_text(SILICON, pseudo_dir, announce=False, **defaults)


def _peak(system, pseudos, chunk):
    return estimate_size(system, pseudos, k_batch=chunk, band_batch=None,
                         projectors="rebuild", wfc_store="stream").peak_bytes


def test_the_chunk_is_the_whole_mesh_whenever_it_fits(pseudo_dir):
    calculator = _calculator(pseudo_dir)
    choice = choose_k_batch(calculator.system, calculator.pseudos, available=10**13)
    assert choice.k_batch is None and choice.fits


def test_the_chunk_takes_the_fewest_calls_and_then_the_least_padding(pseudo_dir):
    """Squeeze the budget between chunks and check the rule, not one number."""
    calculator = _calculator(pseudo_dir)
    system, pseudos = calculator.system, calculator.pseudos
    nk = estimate_size(system, pseudos, k_batch=1).nk
    assert nk == 64
    # A budget that holds 25 k-points a call and not 26, at the chooser's headroom.
    available = int((_peak(system, pseudos, 25) + 1) / 0.6)
    assert _peak(system, pseudos, 26) > 0.6 * available
    choice = choose_k_batch(system, pseudos, available=available)
    assert choice.fits and choice.estimate <= 0.6 * available
    # 25 fits and needs three calls, so does every chunk from 22 to 32, and the
    # one with the least padding among those that fit is 22 (66 solves for 64).
    assert choice.k_batch == 22
    # And the guard: a budget that does not hold one k-point says so.
    starved = choose_k_batch(system, pseudos, available=1)
    assert starved.k_batch == 1 and not starved.fits


def test_fit_is_refused_by_name_where_there_is_no_system_to_size():
    with pytest.raises(ValueError, match="chosen from the card where a Calculation is built"):
        batching.resolve_k_batch("fit")


def test_the_environment_form_is_silent_to_a_caller_without_a_system(monkeypatch):
    monkeypatch.setenv("DEFUMAT_K_BATCH", "fit")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert batching.resolve_k_batch("default", "memory") == batching.memory_preset("memory")["k_batch"]
    assert batching.k_batch_fit_requested("default")
    assert batching.k_batch_fit_requested("fit")
    assert not batching.k_batch_fit_requested(3)
    monkeypatch.delenv("DEFUMAT_K_BATCH")
    assert not batching.k_batch_fit_requested("default")


def test_a_cpu_keeps_its_default(pseudo_dir):
    """A batch over k was measured slower on a CPU, and there is no card to fill."""
    calculator = _calculator(pseudo_dir)
    calculation = Calculation(calculator.system, calculator.pseudos, k_batch="fit")
    assert calculation.k_batch == batching.resolve_k_batch("default", calculation.memory_mode)


def _a_card_that_holds(chunk, monkeypatch, band_batch=None):
    """Pretend to be on a card whose chooser answers ``chunk`` k-points a call.

    The band chooser answers ``band_batch`` at every count, so the rule that a
    split band block keeps one k-point is tested by feeding it that case.
    """
    monkeypatch.setattr(batching, "_backend", lambda: "gpu")
    monkeypatch.setattr(
        "defumat.sizing.choose_k_batch",
        lambda *a, **k: KBatchChoice(k_batch=chunk, fits=True, estimate=1, available=10))
    monkeypatch.setattr(
        "defumat.sizing.choose_band_batch",
        lambda *a, **k: BandBatchChoice(band_batch=band_batch, fits=True, estimate=1,
                                        available=10))
    monkeypatch.delenv("DEFUMAT_K_BATCH", raising=False)
    monkeypatch.delenv("DEFUMAT_BAND_BATCH", raising=False)


def test_a_card_takes_the_chooser_and_only_from_a_whole_band_block(pseudo_dir, monkeypatch):
    calculator = _calculator(pseudo_dir)
    system, pseudos = calculator.system, calculator.pseudos
    _a_card_that_holds(5, monkeypatch)
    assert resolve_k_batch_for("fit", "memory", system, pseudos) == 5
    assert resolve_k_batch_for("fit", "memory", system, pseudos, band_batch=2) == 1
    # an explicit chunk is the dial as it was; "default" in memory mode is 'fit'
    assert resolve_k_batch_for("default", "memory", system, pseudos) == 5
    assert resolve_k_batch_for(3, "memory", system, pseudos) == 3
    calculation = Calculation(system, pseudos, memory_mode="memory", k_batch="fit")
    assert (calculation.k_batch, calculation.band_batch) == (5, None)
    # the environment form reaches the constructor too
    monkeypatch.setenv("DEFUMAT_K_BATCH", "fit")
    assert Calculation(system, pseudos, memory_mode="memory").k_batch == 5


def test_a_split_band_block_keeps_one_k_point_a_call(pseudo_dir, monkeypatch):
    calculator = _calculator(pseudo_dir)
    _a_card_that_holds(5, monkeypatch, band_batch=2)
    with pytest.warns(RuntimeWarning, match="2 bands at a time"):
        calculation = Calculation(calculator.system, calculator.pseudos,
                                  memory_mode="memory", k_batch="fit")
    assert (calculation.k_batch, calculation.band_batch) == (1, 2)


def test_speed_mode_with_fit_keeps_its_own_dials(pseudo_dir, monkeypatch):
    """``'fit'`` is a dial the caller set, so speed mode is not swapped for memory mode."""
    calculator = _calculator(pseudo_dir)
    _a_card_that_holds(7, monkeypatch)
    calculation = Calculation(calculator.system, calculator.pseudos,
                              memory_mode="speed", k_batch="fit")
    assert (calculation.memory_mode, calculation.k_batch) == ("speed", 7)


def test_the_size_report_takes_the_chunk_the_run_takes(pseudo_dir):
    calculator = _calculator(pseudo_dir, k_batch="fit")
    assert calculator.estimate().k_batch == calculator.calculation.k_batch


def test_memory_mode_on_a_card_takes_fit_by_default(pseudo_dir, monkeypatch):
    """The user's decision of 2026-10-02: 184.8 against 31.4 ms an iteration on an H100's mesh."""
    calculator = _calculator(pseudo_dir)
    system, pseudos = calculator.system, calculator.pseudos
    _a_card_that_holds(5, monkeypatch)
    assert Calculation(system, pseudos, memory_mode="memory").k_batch == 5
    # a chunk the caller or the environment names is kept
    assert Calculation(system, pseudos, memory_mode="memory", k_batch=1).k_batch == 1
    monkeypatch.setenv("DEFUMAT_K_BATCH", "1")
    assert Calculation(system, pseudos, memory_mode="memory").k_batch == 1


def test_a_cpu_keeps_one_k_point_by_default(pseudo_dir):
    calculator = _calculator(pseudo_dir)
    calculation = Calculation(calculator.system, calculator.pseudos, memory_mode="memory")
    assert calculation.k_batch == 1


def test_speed_mode_that_does_not_fit_falls_back_to_the_largest_chunk(pseudo_dir, monkeypatch):
    calculator = _calculator(pseudo_dir)
    _a_card_that_holds(5, monkeypatch)

    class Short:
        fits = False

        def describe(self):
            return "estimated 20 GiB against 10"

    monkeypatch.setattr("defumat.sizing.speed_mode_fits", lambda *a, **k: Short())
    with pytest.warns(RuntimeWarning, match="largest k-chunk that fits"):
        calculation = Calculation(calculator.system, calculator.pseudos, memory_mode="speed")
    assert (calculation.memory_mode, calculation.k_batch) == ("memory", 5)


def test_the_store_streams_only_when_the_chunk_is_smaller_than_the_mesh(monkeypatch):
    """With every k-point in flight the stream saved 0.01 to 0.04 GiB and cost 1.9x on an H100."""
    monkeypatch.setattr(batching, "_backend", lambda: "gpu")
    monkeypatch.delenv("DEFUMAT_WFC_STORE", raising=False)
    store = batching.resolve_scf_wfc_store
    assert store("default", "memory", 5, 64) == "stream"
    assert store("default", "memory", None, 64) == "device"
    assert store("default", "memory", 64, 64) == "device"
    assert store("default", "memory", 1, 1) == "device"
    assert store("default", "speed", None, 64) == "device"
    # what the caller or the environment says is kept
    assert store("stream", "memory", None, 64) == "stream"
    assert store("host", "memory", 5, 64) == "host"
    monkeypatch.setenv("DEFUMAT_WFC_STORE", "stream")
    assert store("default", "memory", 1, 1) == "stream"
    monkeypatch.setattr(batching, "_backend", lambda: "cpu")
    monkeypatch.delenv("DEFUMAT_WFC_STORE")
    assert store("default", "memory", 1, 64) == "device"


def test_the_size_report_takes_the_store_the_run_takes(pseudo_dir, monkeypatch):
    for chunk, expected in ((None, "device"), (5, "stream")):
        _a_card_that_holds(chunk, monkeypatch)
        calculator = _calculator(pseudo_dir, memory_mode="memory")
        assert calculator.estimate().wfc_store == expected


# --- one estimate serves the whole bisection ----------------------------------

#: Three cells that put every k-dependent line in play: norm-conserving silicon
#: on 64 k-points, ultrasoft DFT+U nickel at ``nspin = 2`` (projectors in both
#: Davidson lines), and a spinor ultrasoft cobalt (``npol = 2``).
_CELLS = ("silicon", "ni-ldau-ortho.in", "co-tetragonal-anisotropy-soc.in")

#: The dials the chooser is called with: memory mode's own, and their opposite.
_DIALS = ({"band_batch": None, "projectors": "rebuild", "wfc_store": "stream"},
          {"band_batch": 3, "projectors": "store", "wfc_store": "device"})


def _cell(name, pseudo_dir):
    if name == "silicon":
        return _calculator(pseudo_dir)
    from tests.conftest import GENERATED

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Calculator.from_file(GENERATED / name, pseudo_dir, announce=False)


def _bisection(system, pseudos, available, **dials):
    """``choose_k_batch`` as it was: a whole ``estimate_size`` at every step."""
    budget = 0.6 * available

    def peak(chunk):
        return estimate_size(system, pseudos, k_batch=chunk, **dials).peak_bytes

    whole = peak(None)
    if whole <= budget:
        return None, True, int(whole)
    nk = estimate_size(system, pseudos, k_batch=1, **dials).nk
    low, high = 1, max(1, nk - 1)
    if peak(low) > budget:
        return 1, False, int(peak(1))
    while low < high:
        middle = (low + high + 1) // 2
        if peak(middle) <= budget:
            low = middle
        else:
            high = middle - 1
    calls = -(-nk // low)
    chunk = -(-nk // calls)
    return chunk, True, int(peak(chunk))


@pytest.mark.parametrize("name", _CELLS)
def test_one_estimate_is_every_chunk_s_estimate(name, pseudo_dir):
    """``at_k_batch`` is :func:`estimate_size` at that chunk, every field, every line in order."""
    calculator = _cell(name, pseudo_dir)
    system, pseudos = calculator.system, calculator.pseudos
    for dials in _DIALS:
        base = estimate_size(system, pseudos, k_batch=None, **dials)
        other = estimate_size(system, pseudos, k_batch=2, **dials)
        for chunk in (None, 1, 2, 3, 5, 7, base.nk - 1, base.nk, base.nk + 5):
            fresh = estimate_size(system, pseudos, k_batch=chunk, **dials)
            moved = base.at_k_batch(chunk)
            assert moved == fresh, (dials, chunk)
            assert list(moved.arrays.items()) == list(fresh.arrays.items())
            assert moved.peak_bytes == fresh.peak_bytes
            # ... and from an estimate made at another chunk, not only the whole mesh
            assert other.at_k_batch(chunk) == fresh, (dials, chunk)


@pytest.mark.parametrize("name", _CELLS)
def test_the_chooser_takes_the_chunk_the_bisection_took(name, pseudo_dir):
    """Budgets either side of every chunk's peak, the starved one and the whole mesh."""
    calculator = _cell(name, pseudo_dir)
    system, pseudos = calculator.system, calculator.pseudos
    for dials in _DIALS:
        nk = estimate_size(system, pseudos, k_batch=None, **dials).nk
        availables = {1, 10**13}
        for chunk in sorted({1, 2, 3, nk // 3, nk // 2, nk - 1, nk}):
            peak = estimate_size(system, pseudos, k_batch=chunk, **dials).peak_bytes
            for available in (int(peak / 0.6), int((peak + 1) / 0.6), int(peak / 0.6) + 2):
                availables.add(available)
        for available in sorted(availables):
            choice = choose_k_batch(system, pseudos, available=available, **dials)
            assert (choice.k_batch, choice.fits, choice.estimate) == _bisection(
                system, pseudos, available, **dials), (dials, available)
