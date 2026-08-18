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

#: One benchmark disagrees on the FFT dimensions while agreeing on everything
#: else, and the disagreement is a property of the machine QE ran on rather than
#: of the physics. pw_scf/scf.in and pw_scf/scf-cg.in are the same system at the
#: same cutoffs -- both report 1459 G-vectors -- but scf.in's benchmark has a
#: 15^3 grid and scf-cg.in's a 16^3 one. 15 = 3*5 is a valid FFT size for FFTW
#: and every other library QE supports, but IBM's ESSL additionally requires a
#: factor of 2, which would push 15 up to 16. The reference outputs predate the
#: current release (the suite records REFERENCE_VERSION 6.0) and were evidently
#: not all produced on the same build. The Miller-index range, and therefore the
#: G-vector set, is identical either way: (15-1)/2 == (16-1)/2 == 7.
KNOWN_FFT_GRID_PROVENANCE = {"scf-cg.in"}


@lru_cache(maxsize=None)
def _cached(input_path: Path, benchmark_path: Path):
    """Build once per input: four tests inspect the same basis, and generating
    G-vectors for the larger cells is the slowest thing in the suite."""
    system = build_system(read_pw_input(input_path))
    return build_basis(system), system, read_qe_output(benchmark_path)


def _basis_and_reference(qe_testsuite, directory, name):
    if name in NEEDS_SPACE_GROUPS:
        pytest.skip("crystal_sg (Wyckoff) positions need space-group expansion (P6)")
    return _cached(
        qe_testsuite / directory / name,
        qe_testsuite / directory / f"benchmark.out.git.inp={name}",
    )


@pytest.mark.parametrize(("directory", "name"), CASES)
def test_dense_grid_matches_reference(qe_testsuite, directory, name):
    basis, _, ref = _basis_and_reference(qe_testsuite, directory, name)

    assert basis.dense.ngm == ref.ngm_dense
    if name in KNOWN_FFT_GRID_PROVENANCE:
        # Same G-sphere, different rounding-up of the box: check the content.
        assert [(n - 1) // 2 for n in basis.dense.grid] == [(n - 1) // 2 for n in ref.fft_dense]
    else:
        assert tuple(basis.dense.grid) == tuple(ref.fft_dense)


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
    assert tuple(basis.smooth.grid) == tuple(ref.fft_smooth)
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
