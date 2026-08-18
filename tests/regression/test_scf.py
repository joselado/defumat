"""P5/P7 check: total energies, eigenvalues and bands against QE.

These are the numbers the whole project exists to reproduce. The energy is
compared **term by term**, not only as a total: the total is variational, so it
is second-order accurate in the density and can look right while a term is
wrong. A per-term comparison localises an error to one physical contribution.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from pypresso.io import read_qe_output
from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf import run_scf
from pypresso.system import build_system
from pypresso.workflows import run_bands
from tests.tolerances import (
    DENSITY_DEPENDENT_TERM_RY,
    EIGENVALUE_EV,
    ENERGY_TERM_RY,
    TOTAL_ENERGY_RY,
)

pytestmark = [pytest.mark.regression, pytest.mark.slow]

#: (directory, input) for the cases a converged SCF is checked against.
#: Chosen to exercise one thing each: the canonical insulator, the same system
#: with an automatic k-grid, with crystal-coordinate k-points, with occupations
#: from input, at gamma only, and a metal with each of QE's smearings.
SCF_CASES = [
    ("pw_scf", "scf.in"),
    ("pw_scf", "scf-kauto.in"),
    ("pw_scf", "scf-kcrys.in"),
    ("pw_scf", "scf-k0.in"),
    ("pw_scf", "scf-occ.in"),
    ("pw_metal", "metal.in"),
    ("pw_metal", "metal-gaussian.in"),
    ("pw_metal", "metal-fermi_dirac.in"),
]


@lru_cache(maxsize=None)
def _converged(input_path: Path, pseudo_dir: Path):
    system = build_system(read_pw_input(input_path))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    return system, pseudos, run_scf(system, pseudos, conv_thr=1e-10, max_iterations=80)


@pytest.mark.parametrize(("directory", "name"), SCF_CASES)
def test_total_energy_matches_reference(qe_testsuite, pseudo_dir, directory, name):
    _, _, result = _converged(qe_testsuite / directory / name, pseudo_dir)
    reference = read_qe_output(qe_testsuite / directory / f"benchmark.out.git.inp={name}")

    assert result.converged
    # QE's own runs stop at conv_thr = 1e-6, so its total is only that well
    # determined; agreement to 1e-6 Ry is the most that can be asked of it.
    assert result.total_energy == pytest.approx(reference.total_energy, abs=TOTAL_ENERGY_RY)


@pytest.mark.parametrize(("directory", "name"), SCF_CASES)
def test_energy_terms_match_reference(qe_testsuite, pseudo_dir, directory, name):
    _, _, result = _converged(qe_testsuite / directory / name, pseudo_dir)
    reference = read_qe_output(qe_testsuite / directory / f"benchmark.out.git.inp={name}")

    assert set(result.energy_terms) == set(reference.energy_terms)
    for term, value in reference.energy_terms.items():
        # The individual terms are first-order sensitive to the density where the
        # total is second-order, so they are compared more loosely than the total.
        # Ewald depends on no density at all and is held to the tight tolerance.
        tolerance = ENERGY_TERM_RY if term == "ewald" else DENSITY_DEPENDENT_TERM_RY
        assert result.energy_terms[term] == pytest.approx(value, abs=tolerance), term

    assert sum(result.energy_terms.values()) == pytest.approx(result.total_energy, abs=1e-10)


@pytest.mark.parametrize(("directory", "name"), SCF_CASES)
def test_eigenvalues_match_reference(qe_testsuite, pseudo_dir, directory, name):
    _, _, result = _converged(qe_testsuite / directory / name, pseudo_dir)
    reference = read_qe_output(qe_testsuite / directory / f"benchmark.out.git.inp={name}")

    if reference.eigenvalues is None:
        pytest.skip("no eigenvalues in this reference")
    ours = result.eigenvalues_ev
    theirs = reference.eigenvalues[0][:, : ours.shape[1]]
    if theirs.shape != ours.shape:
        pytest.skip("QE reduced the k-point set by symmetry; comparison needs P6's IBZ")

    assert ours == pytest.approx(theirs, abs=EIGENVALUE_EV)


def test_ewald_matches_reference_for_every_case(qe_testsuite, pseudo_dir):
    """The Ewald term is independent of the density, so it must be exact."""
    for directory, name in SCF_CASES:
        _, _, result = _converged(qe_testsuite / directory / name, pseudo_dir)
        reference = read_qe_output(qe_testsuite / directory / f"benchmark.out.git.inp={name}")
        assert result.energy_terms["ewald"] == pytest.approx(
            reference.energy_terms["ewald"], abs=ENERGY_TERM_RY
        ), f"{directory}/{name}"


def test_silicon_band_structure(qe_testsuite, pseudo_dir):
    """P7: bands along a path from the converged density of pw_scf/scf.in.

    The three inputs in pw_scf run as a sequence sharing one outdir --
    scf.in, then scf-1.in ('bands'), then scf-2.in ('nscf') -- so scf-1.in's
    reference is the band structure of exactly the density scf.in converges to.
    """
    system, pseudos, scf = _converged(qe_testsuite / "pw_scf" / "scf.in", pseudo_dir)
    band_system = build_system(read_pw_input(qe_testsuite / "pw_scf" / "scf-1.in"))

    bands = run_bands(band_system, pseudos, scf.density)
    reference = read_qe_output(qe_testsuite / "pw_scf" / "benchmark.out.git.inp=scf-1.in")

    assert bands.eigenvalues.shape == reference.eigenvalues[0].shape
    assert bands.eigenvalues_ev == pytest.approx(reference.eigenvalues[0], abs=EIGENVALUE_EV)

    # Diamond silicon is threefold degenerate at the top of the valence band at
    # Gamma; if the density were not symmetrised those three would split.
    at_gamma = bands.eigenvalues_ev[0]
    assert at_gamma[1] == pytest.approx(at_gamma[2], abs=1e-6)
    assert at_gamma[2] == pytest.approx(at_gamma[3], abs=1e-6)

    # LDA underestimates the gap of silicon (experiment ~1.1 eV); the point here
    # is that it is positive and in the right ballpark, not that it is right.
    assert 0.3 < bands.gap(8) < 0.8


def test_nscf_run_reproduces_the_reference(qe_testsuite, pseudo_dir):
    """scf-2.in is an 'nscf' run on a different k-grid, same density."""
    system, pseudos, scf = _converged(qe_testsuite / "pw_scf" / "scf.in", pseudo_dir)
    nscf_system = build_system(read_pw_input(qe_testsuite / "pw_scf" / "scf-2.in"))

    bands = run_bands(nscf_system, pseudos, scf.density)
    reference = read_qe_output(qe_testsuite / "pw_scf" / "benchmark.out.git.inp=scf-2.in")

    assert bands.eigenvalues_ev == pytest.approx(reference.eigenvalues[0], abs=EIGENVALUE_EV)
