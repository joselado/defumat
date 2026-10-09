"""The size estimate must be the size, or it is worse than no estimate.

:mod:`defumat.sizing` exists to answer "will this run fit" for an input too
large to build, which means it is never checked against reality by the person
relying on it. So it is checked here instead: every count it reports is
asserted against what a real :class:`~defumat.scf.driver.Calculation` actually
builds, on cells small enough to build both.

The double-grid case is the one that matters most -- ``ngm`` and ``ngms``
coincide for a norm-conserving dataset, so a bug that conflated them would pass
on silicon and be wrong on every ultrasoft run.
"""

import warnings

import numpy as np
import pytest

from defumat.calculator import Calculator
from defumat.scf.driver import default_nbnd
from defumat.sizing import estimate_size
from tests.backend import on_a_cpu

pytestmark = pytest.mark.unit


SILICON = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1, ecutwfc = 12.0
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""

#: A PAW dataset, so ``ecutrho > 4 ecutwfc`` and the two grids come apart.
SILICON_PAW = SILICON.replace(
    "Si 28.086 Si.pz-vbc.UPF", "Si 28.086 Si.pz-n-kjpaw_psl.0.1.UPF"
).replace("ecutwfc = 12.0", "ecutwfc = 20.0, ecutrho = 120.0")

#: A shifted grid, whose k-points hold different numbers of plane waves --
#: ``npwx`` is then a maximum over an unequal set rather than one count.
SILICON_SHIFTED = SILICON.replace("2 2 2 0 0 0", "3 3 3 1 1 1")

#: ``K_POINTS gamma``, which the driver substitutes away.
SILICON_GAMMA = SILICON.replace(
    "K_POINTS automatic\n 2 2 2 0 0 0", "K_POINTS gamma"
)


def _calculator(text, pseudo_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Calculator.from_text(text, pseudo_dir, announce=False)


@pytest.mark.parametrize(
    "text", [SILICON, SILICON_PAW, SILICON_SHIFTED, SILICON_GAMMA],
    ids=["nc", "paw-doublegrid", "shifted", "gamma"],
)
def test_the_estimate_is_what_the_setup_builds(text, pseudo_dir):
    """Every count, against a ``Calculation`` that actually allocated them."""
    calculator = _calculator(text, pseudo_dir)
    estimate = estimate_size(calculator.system, calculator.pseudos)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = calculator.calculation

    assert estimate.ngm == built.basis.dense.ngm
    assert estimate.ngms == built.basis.ngms
    assert estimate.npwx == built.basis.npwx
    assert estimate.dense_grid == built.basis.dense.grid
    assert estimate.smooth_grid == built.basis.smooth.grid
    assert estimate.nkb == built.projectors.nkb
    assert estimate.nelec == pytest.approx(built.nelec)
    assert estimate.nbnd == default_nbnd(
        built.nelec, calculator.system.occupations
    )
    # ``npw`` per k-point, not only its maximum: a shifted grid has an unequal
    # set and a bug that returned the first count would pass on ``npwx`` alone.
    assert estimate.npw == tuple(built.basis.planewaves.npw)


def test_the_augmentation_charge_is_sized_and_is_the_largest_term(pseudo_dir):
    """``Q_ij(G)`` is what decides whether an ultrasoft run starts.

    Setup was outside this module's scope, and on a vacuum-padded slab that was
    not a scope observation but a green light: 34.78 GB reported against a
    117.55 GB measured peak, with each of the setup allocations larger than the
    whole reported total. The augmentation charge is counted exactly here --
    against the array a real ``Calculation`` allocated, not against a formula
    written twice -- and on a PAW cell it is already bigger than every line
    that grows with the cell.
    """
    calculator = _calculator(SILICON_PAW, pseudo_dir)
    estimate = estimate_size(calculator.system, calculator.pseudos)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = calculator.calculation

    assert built.augmentation is not None
    counted = estimate.arrays["augmentation Q_ij(G) (nh,nh,ngm)"]
    assert counted == sum(q.nbytes for q in built.augmentation.qgm)
    assert (
        estimate.arrays["augmentation phases (nat,ngm)"]
        == built.augmentation.phases.nbytes
    )

    # It is the largest line that grows with the cell, and the Bessel transient
    # is larger still. The PAW one-centre tensors are left out of the max: they
    # are ``nh^2 nlm mesh`` per dataset, set by the pseudopotential and not by
    # the cell, so on this two-atom cell they outweigh ``Q_ij(G)`` where on a
    # slab ``Q_ij(G)`` grows with ``ngm`` past them.
    #
    # The symmetry maps are left out too, and counted exactly instead: they are
    # ``nsym ngm`` and silicon's 48 operations make them the larger on this
    # cell (a slab keeps a handful of operations, where ``Q_ij(G)`` has ``nh^2``
    # rows). Since the table went real (``GPU-MEMORY-NEXT.md`` item 19) it is
    # half what it was, which is what put the maps above it here.
    permutations, phases = built._symmetry_maps
    assert (estimate.arrays["symmetry maps (nsym,ngm)"]
            == permutations.nbytes + phases.nbytes)
    largest = max(
        size for name, size in estimate.arrays.items()
        if name not in ("PAW one-centre (nh,nh,nlm,mesh)",
                        "symmetry maps (nsym,ngm)")
    )
    assert counted == largest
    assert estimate.setup_transient > counted


def test_a_tabulated_run_is_sized_as_one_and_not_as_the_stored_array(
    pseudo_dir, monkeypatch
):
    """Above the budget the run holds a table, and the estimate must say so.

    Reporting 65 GB of ``Q_ij(G)`` for a run that will hold 15 MB of table is
    this module's own failure inverted -- a red light for a calculation that
    fits -- so the estimate calls the same budget on the same number the run
    calls it on, and the two move together.
    """
    from defumat.pseudo.augmentation import TabulatedAugmentation

    monkeypatch.setenv("DEFUMAT_AUG_MAX_BYTES", "0")
    calculator = _calculator(SILICON_PAW, pseudo_dir)
    estimate = estimate_size(calculator.system, calculator.pseudos)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = calculator.calculation
    assert isinstance(built.augmentation, TabulatedAugmentation)

    assert "augmentation Q_ij(G) (nh,nh,ngm)" not in estimate.arrays
    counted = estimate.arrays["augmentation table (nbeta,nbeta,nl,nqx)"]
    held = sum(t.nbytes for t in built.augmentation.tables if t is not None)
    # The estimate's nqx uses the run's own cell_factor and dq, so it is the
    # same table; it may reach further than this cell's G set needs.
    assert counted == pytest.approx(held, rel=0.05)

    # The structure factor is built a chunk at a time from the positions, so
    # what is held is the positions and an empty array carrying the dtype
    # (``OPEN.md`` S3), where it was ``(nat, npad)`` complex.
    assert built.augmentation.phases.size == 0
    assert estimate.arrays["augmentation positions (nat,3)"] == (
        built.augmentation.positions.nbytes
    )
    assert "augmentation phases (nat,npad)" not in estimate.arrays
    # And the point: sized for the scheme that runs, not the one that does not.
    stored_would_be = (
        built.projectors.nkb**2 * estimate.ngm * 16 // len(calculator.pseudos)
    )
    assert counted < stored_would_be


def test_the_setup_transient_bounds_the_peak_rather_than_adding_to_it(pseudo_dir):
    """The Bessel intermediate is freed before the eigensolver is asked for one.

    Summing them would report a peak that never exists at any instant, which is
    the opposite of this module's error: an overestimate that refuses a run
    that would have fitted.
    """
    calculator = _calculator(SILICON_PAW, pseudo_dir)
    estimate = estimate_size(calculator.system, calculator.pseudos)

    resident = sum(
        size for name, size in estimate.arrays.items()
        if name not in estimate._SUPERSEDED
    )
    assert estimate.peak_bytes == resident + max(
        estimate.setup_transient, estimate.eigensolver_buffer,
        estimate.start_buffer,
    )
    assert estimate.peak_bytes < (
        resident + estimate.setup_transient + estimate.eigensolver_buffer
        + estimate.start_buffer
    )


def test_a_norm_conserving_run_is_charged_no_augmentation_at_all(pseudo_dir):
    """``Si.pz-vbc`` has no augmentation charge, so neither line may appear."""
    calculator = _calculator(SILICON, pseudo_dir)
    estimate = estimate_size(calculator.system, calculator.pseudos)
    assert not any("augmentation" in name for name in estimate.arrays)
    assert estimate.setup_transient == 0


def test_the_double_grid_is_genuinely_double(pseudo_dir):
    """The PAW case must actually exercise the two-grid path.

    Without this the parametrisation above could pass with ``ngms`` a copy of
    ``ngm`` in every case.
    """
    calculator = _calculator(SILICON_PAW, pseudo_dir)
    estimate = estimate_size(calculator.system, calculator.pseudos)
    assert estimate.doublegrid
    assert estimate.ngms < estimate.ngm
    assert estimate.smooth_grid != estimate.dense_grid


def test_gamma_is_sized_as_the_run_and_not_as_the_request(pseudo_dir):
    """``K_POINTS gamma`` is substituted away, so the estimate must not halve.

    Reporting the half-sphere counts would understate every array by two for a
    run that does not happen. The flag saying it was *asked* for survives, and
    the report says so.
    """
    calculator = _calculator(SILICON_GAMMA, pseudo_dir)
    estimate = estimate_size(calculator.system, calculator.pseudos)
    assert estimate.gamma_requested
    assert not estimate.gamma_only

    plain = estimate_size(*_pair(_calculator(SILICON, pseudo_dir)))
    # The same cell and cutoffs: the full sphere is the full sphere either way.
    assert estimate.ngm == plain.ngm
    assert "FULL sphere" in estimate.report()


def _pair(calculator):
    return calculator.system, calculator.pseudos


def test_nothing_is_allocated_on_the_device(pseudo_dir):
    """The estimate must not build the calculation it is sizing.

    This is the property the module exists for -- an input too large to build
    must still be sizeable -- and it is invisible on a small cell unless
    asserted.
    """
    calculator = _calculator(SILICON, pseudo_dir)
    estimate_size(calculator.system, calculator.pseudos)
    assert calculator._calculation is None


def test_the_spin_channels_do_not_double_the_eigensolver(pseudo_dir, monkeypatch):
    """``diagonalize`` solves the channels one after another.

    The wavefunctions are held for both and the Davidson workspace is not, so a
    collinear run doubles one line of the report and not the other. Getting
    this wrong overstated a real calculation by 90 GB.
    """
    lsda = SILICON.replace(
        "ecutwfc = 12.0",
        "ecutwfc = 12.0, nspin = 2, starting_magnetization(1) = 0.1,\n"
        "  occupations = 'smearing', smearing = 'mv', degauss = 0.02",
    )
    # The device report of a CPU run: a card's memory mode streams the
    # wavefunctions and names that line differently.
    on_a_cpu(monkeypatch)
    # ``nbnd`` is pinned in both: a smeared run defaults to more bands than a
    # fixed-occupation one, and without this the doubling being measured is the
    # band count's rather than the spin axis's.
    one = estimate_size(*_pair(_calculator(SILICON, pseudo_dir)), nbnd=8)
    two = estimate_size(*_pair(_calculator(lsda, pseudo_dir)), nbnd=8)

    assert two.nspin == 2
    assert (two.arrays["Davidson subspace psi+hpsi"]
            == one.arrays["Davidson subspace psi+hpsi"])
    assert two.nbnd == one.nbnd == 8
    assert (two.arrays["wavefunctions (nspin,nk,nbnd,ndim)"]
            == 2 * one.arrays["wavefunctions (nspin,nk,nbnd,ndim)"])


# --- the estimate must size the run the input describes ---------------------
#
# The failure this guards is the module's whole reason for existing turned
# against it: an estimate that silently assumes a library default reports a run
# that does not happen, and nobody relying on it is in a position to notice.
# It was found in the field on `diago_david_ndim` and was worth 35 GB.


_DAVID_2 = SILICON.replace("&electrons\n", "&electrons\n  diago_david_ndim = 2\n")


def test_the_estimate_uses_the_inputs_own_diago_david_ndim(pseudo_dir):
    """An input that sets it must be sized at it, not at ``DAVID_NDIM``.

    The same calculator would *run* at 2, so sizing at 4 describes a different
    calculation -- and the subspace is the largest array in a large run, so the
    discrepancy is not small.
    """
    calculator = _calculator(_DAVID_2, pseudo_dir)
    assert calculator.defaults["david"] == 2

    estimate = calculator.estimate()
    assert estimate.davidson_basis == 2
    # ... and it is genuinely half of what the default would give.
    plain = _calculator(SILICON, pseudo_dir).estimate()
    assert plain.davidson_basis == 4
    assert (estimate.arrays["Davidson subspace psi+hpsi"]
            == plain.arrays["Davidson subspace psi+hpsi"] // 2)


def test_an_explicit_argument_still_overrides_the_input(pseudo_dir):
    """The flags are overrides, not replacements of the reading."""
    calculator = _calculator(_DAVID_2, pseudo_dir)
    assert calculator.estimate(davidson_basis=4).davidson_basis == 4


def test_the_library_default_is_not_a_literal():
    """``DAVID_NDIM`` and the estimate cannot drift apart.

    Writing 4 in two places is how they would.
    """
    from defumat.solvers.davidson import DAVID_NDIM

    calculator_free = estimate_size.__defaults__
    assert DAVID_NDIM == 4
    # The signature default is None -- "ask the constant" -- rather than 4.
    assert None in calculator_free


def test_the_report_says_what_it_assumed(pseudo_dir):
    """Two estimates differing only in an assumption must be tellable apart.

    ``davidson_basis`` and ``k_batch`` are not properties of the input, so a
    reader comparing two reports has to be able to see which one moved.
    """
    report = _calculator(_DAVID_2, pseudo_dir).estimate().report()
    assert "diago_david_ndim = 2" in report
    assert "k in flight" in report


def test_k_batch_is_resolved_the_way_the_run_would_resolve_it(pseudo_dir):
    """The same alignment, one option along.

    ``Calculator`` defaults ``k_batch`` to ``"default"``, which
    :func:`~defumat.batching.resolve_k_batch` turns into QE's one-k-point loop
    on a CPU. An estimate that assumed the whole axis instead would overstate
    the eigensolver on every multi-k cell.
    """
    from defumat.batching import resolve_k_batch

    estimate = _calculator(SILICON, pseudo_dir).estimate()
    expected = resolve_k_batch("default")
    if expected is not None:
        assert estimate.k_batch == min(expected, estimate.nk)


def test_the_peak_never_falls_below_the_floor_it_supersedes(pseudo_dir):
    """The eigensolver's buffer stands in for two Davidson lines, so it has to
    be at least as large as they are -- at every point on the k-batch dial.

    It carries ``k_batch`` because those two do: the buffer was fitted one
    k-point at a time and ``davidson_eigensolver_all`` ``vmap``s that chunk over
    the whole batch. Without the factor the estimate reported a *smaller* peak
    for the end of the dial that actually holds ``nk`` subspaces -- a green
    light for the larger calculation, which is this module's own stated error
    inverted.
    """
    calculator = _calculator(SILICON, pseudo_dir)
    whole_axis = estimate_size(calculator.system, calculator.pseudos, k_batch=None)
    assert whole_axis.k_batch == whole_axis.nk > 1

    superseded = sum(
        whole_axis.arrays[name] for name in whole_axis._SUPERSEDED
    )
    assert whole_axis.eigensolver_buffer >= superseded


def test_the_peak_grows_with_the_k_batch(pseudo_dir):
    """More k-points in flight is more memory, monotonically.

    The dial exists to trade memory for batching, so an estimate that reports
    the trade backwards is worse than no estimate: it says the accelerator end
    is the cheap one.
    """
    calculator = _calculator(SILICON, pseudo_dir)
    one = estimate_size(calculator.system, calculator.pseudos, k_batch=1)
    whole = estimate_size(calculator.system, calculator.pseudos, k_batch=None)
    assert whole.nk > 1
    assert whole.peak_bytes > one.peak_bytes
    # ``int()`` of the scaled sum against ``nk`` times the rounded one, so the
    # comparison is to within the truncation rather than exact.
    assert whole.eigensolver_buffer == pytest.approx(
        whole.nk * one.eigensolver_buffer, abs=whole.nk
    )


def test_the_per_k_tables_and_the_hubbard_projectors_are_counted_exactly(pseudo_dir):
    """``GPU-MEMORY-NEXT.md`` item 24: two resident lines the estimate lacked.

    Held against the arrays a real ``Calculation`` allocated, not against the
    formula a second time: ``|k+G|^2``, the FFT index and the mask, and DFT+U's
    ``wfcU``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_file(
            "tests/data/qe/ni-ldau-nospin.in", pseudo_dir=pseudo_dir,
            announce=False)
        built = calculator.calculation
    estimate = estimate_size(calculator.system, calculator.pseudos)
    tables = (built.kinetic.nbytes + built.fft_index.nbytes
              + np.asarray(built.basis.planewaves.mask).nbytes)
    assert estimate.arrays["per-k basis tables (nk,npwx)"] == tables
    assert (estimate.arrays["Hubbard projectors wfcU (nk,ndim,nwfcU)"]
            == built.wfcU.nbytes)


@pytest.mark.parametrize("element", ["Si", "Al"], ids=["nonsymmorphic", "symmorphic"])
def test_the_symmetry_maps_hold_int32_and_no_phase_a_symmorphic_group_does_not_need(
        element, pseudo_dir):
    """``OPEN.md`` S1: what the run keeps for its symmetrisation, and that it is exact.

    Diamond silicon's group has fractional translations and keeps its phases;
    fcc aluminium's has none, and keeps one column of ones in their place. The
    indices are int32 in both. The average is held to the one the int64
    indices and the whole phase table give, to the bit, and the estimate's
    line to the bytes held.
    """
    from defumat.system.symmetry import apply_symmetry_maps

    text = SILICON
    if element == "Al":
        text = (SILICON.replace("Si 28.086 Si.pz-vbc.UPF", "Al 26.98 Al.pz-vbc.UPF")
                .replace("nat = 2", "nat = 1")
                .replace(" Si 0.00 0.00 0.00\n Si 0.25 0.25 0.25", " Al 0.00 0.00 0.00")
                .replace("celldm(1) = 10.20", "celldm(1) = 7.50"))
    calculator = _calculator(text, pseudo_dir)
    built = calculator.calculation
    permutations, phases = built._symmetry_maps
    nsym, ngm = permutations.shape
    assert permutations.dtype == np.int32
    translations = built.symmetries.translation_array()
    assert np.any(translations) == (element == "Si")
    assert phases.shape == ((nsym, ngm) if element == "Si" else (nsym, 1))

    miller = np.asarray(built.basis.dense.miller)
    whole = np.exp(-2j * np.pi * (miller @ translations.T)).T
    rng = np.random.default_rng(3)
    field = rng.standard_normal(ngm) + 1j * rng.standard_normal(ngm)
    old = np.asarray(apply_symmetry_maps(
        field, np.asarray(permutations).astype(np.int64), whole))
    np.testing.assert_array_equal(
        np.asarray(apply_symmetry_maps(field, permutations, phases)), old)

    estimate = calculator.estimate()
    assert (estimate.arrays["symmetry maps (nsym,ngm)"]
            == permutations.nbytes + phases.nbytes)


def test_the_mixer_history_is_counted_on_the_host(pseudo_dir):
    """``OPEN.md`` S2: Anderson's history, held against a real mixer's lists.

    ``mixing_ndim`` input states and as many residuals, each the packed state
    ``_mix`` hands the mixer -- here the dense density and PAW silicon's
    ``becsum`` -- in host RAM, which the report gave as nothing. Six steps
    through a depth of four, so the lists are full and trimmed, and the line is
    the bytes they hold. It is not part of the device peak.
    """
    from defumat.scf.driver import _mix
    from defumat.scf.mixing import get_mixer

    calculator = _calculator(SILICON_PAW, pseudo_dir)
    built = calculator.calculation
    rho = built.starting_density()
    becsum = built.starting_becsum()
    rng = np.random.default_rng(7)
    mixer = get_mixer("anderson", history=4)
    for _ in range(6):
        noise = rng.standard_normal(np.shape(rho)) * 1.0e-3
        out = tuple(None if b is None else b * (1.0 + rng.uniform(-1e-3, 1e-3))
                    for b in becsum)
        rho, becsum, _, _ = _mix(mixer, rho, rho + noise, becsum, out)
    held = sum(np.asarray(v).nbytes for v in mixer._densities + mixer._residuals)
    assert len(mixer._densities) == 4

    estimate = calculator.estimate(mixing_ndim=4)
    assert estimate.host_arrays == {"mixer history (4 x rho_in, residual)": held}
    assert "Host memory" in estimate.report()
    assert calculator.estimate(mixing_ndim=8).peak_bytes == estimate.peak_bytes
    assert calculator.estimate(mixing_mode="linear").host_arrays == {}
