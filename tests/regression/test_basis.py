"""P2 check: the plane-wave basis matches QE's, counted exactly.

These are integers QE prints -- the number of G-vectors on each grid, the FFT
dimensions, and the number of plane waves at every k-point. They either agree or
they do not; there is no tolerance to hide behind, which makes this the sharpest
check in the project.
"""

from functools import lru_cache
from pathlib import Path

import pytest

from pypresso.basis.builder import build_basis
from pypresso.io import read_qe_output
from pypresso.io.pwin import read_pw_input
from pypresso.system import build_system
from tests.regression.test_geometry import CASES, NEEDS_SPACE_GROUPS

pytestmark = pytest.mark.regression

#: Some committed benchmarks disagree with the current QE on the FFT dimensions
#: while agreeing on every G-vector. pw_scf/scf.in and pw_scf/scf-cg.in are the
#: same system at the same cutoffs -- both report 1459 G-vectors -- but scf.in's
#: benchmark has a 15^3 grid and scf-cg.in's a 16^3 one.
#:
#: The reason is not the machine, as an earlier version of this file guessed. It
#: is that QE now requires the FFT dimensions to be a multiple of the
#: denominators of the crystal's fractional translations (``fft_fact`` in
#: ``PW/src/symm_base.f90``): diamond silicon's are 1/4, so its grids must be a
#: multiple of 4, and 15 is rounded up to 16. That rule postdates the committed
#: references (the suite records REFERENCE_VERSION 6.0). Running the vendored
#: pw.x on pw_scf/scf.in confirms it: QE 7.5 prints 16^3 today.
#:
#: A benchmark is therefore taken at face value only when its grid already
#: satisfies the constraint. When it does not, what is checked is that ours is
#: the benchmark's grid rounded up -- and, as always, that ``ngm`` is identical,
#: which is the part with physical content.


def _fft_factors(system):
    from pypresso.system.symmetry import find_symmetries

    return find_symmetries(system.cell, system.structure).fft_factors()


def _assert_grid(ours, reference, factors):
    from pypresso.basis.fftgrid import good_fft_order

    if all(n % f == 0 for n, f in zip(reference, factors)):
        assert tuple(ours) == tuple(reference)
    else:
        assert tuple(ours) == tuple(
            good_fft_order(n, f) for n, f in zip(reference, factors)
        )


@lru_cache(maxsize=None)
def _cached(input_path: Path, benchmark_path: Path):
    """Build once per input: four tests inspect the same basis, and generating
    G-vectors for the larger cells is the slowest thing in the suite."""
    system = build_system(read_pw_input(input_path))
    return build_basis(system), system, read_qe_output(benchmark_path)


def _basis_and_reference(qe_testsuite, directory, name):
    if name in NEEDS_SPACE_GROUPS:
        pytest.skip("crystal_sg (Wyckoff) positions need space-group expansion (P6)")
    try:
        return _cached(
            qe_testsuite / directory / name,
            qe_testsuite / directory / f"benchmark.out.git.inp={name}",
        )
    except NotImplementedError as refusal:
        # As in ``test_geometry._build``: an input refused by name has no System
        # to build a basis on, and skipping on the refusal keeps the sweep from
        # needing an edit whenever one is added.
        pytest.skip(str(refusal).split(" -- ")[0])


@pytest.mark.parametrize(("directory", "name"), CASES)
def test_dense_grid_matches_reference(qe_testsuite, directory, name):
    basis, system, ref = _basis_and_reference(qe_testsuite, directory, name)

    assert basis.dense.ngm == ref.ngm_dense
    _assert_grid(basis.dense.grid, ref.fft_dense, _fft_factors(system))


@pytest.mark.parametrize(("directory", "name"), CASES)
def test_smooth_grid_matches_reference(qe_testsuite, directory, name):
    basis, system, ref = _basis_and_reference(qe_testsuite, directory, name)

    if ref.ngm_smooth is None:
        # QE prints a smooth grid only when it differs from the dense one, which
        # happens exactly when dual > 4 -- i.e. for ultrasoft and PAW.
        assert system.ecutrho / system.ecutwfc <= 4.0 + 1e-8
        assert not basis.doublegrid
        return

    assert basis.doublegrid
    assert basis.smooth.ngm == ref.ngm_smooth
    _assert_grid(basis.smooth.grid, ref.fft_smooth, _fft_factors(system))
    assert basis.smooth.ngm < basis.dense.ngm


@pytest.mark.parametrize(("directory", "name"), CASES)
def test_plane_wave_counts_match_reference(qe_testsuite, directory, name):
    pwin = read_pw_input(qe_testsuite / directory / name)
    basis, system, ref = _basis_and_reference(qe_testsuite, directory, name)

    if ref.npw is None:
        pytest.skip("QE did not print per-k plane-wave counts for this run")
    card = pwin.card("K_POINTS")
    if card is not None and (card.option or "").lower() == "automatic":
        pytest.skip("automatic grids are symmetry-reduced by QE (P6)")
    if system.nspin == 2:
        pytest.skip("nspin=2 lists each k-point once per spin channel")

    assert list(basis.planewaves.npw) == list(ref.npw)
    assert basis.npwx == max(ref.npw)


@pytest.mark.parametrize(("directory", "name"), CASES)
def test_kinetic_energies_respect_the_cutoff(qe_testsuite, directory, name):
    """Every retained plane wave is inside ecutwfc, and padding contributes none."""
    basis, system, _ = _basis_and_reference(qe_testsuite, directory, name)

    kinetic = basis.planewaves.kinetic(basis.dense, system.kpoints, system.cell)
    mask = basis.planewaves.mask

    assert float(kinetic[mask].max()) <= system.ecutwfc + 1e-8
    if not bool(mask.all()):
        assert float(abs(kinetic[~mask]).max()) == 0.0
