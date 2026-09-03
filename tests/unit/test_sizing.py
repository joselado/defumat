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

import pytest

from defumat.calculator import Calculator
from defumat.scf.driver import default_nbnd
from defumat.sizing import estimate_size

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


def test_the_spin_channels_do_not_double_the_eigensolver(pseudo_dir):
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
