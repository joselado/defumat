"""A band side in single precision: the SCF of a norm-conserving cell, and what it refuses.

``Calculation(band_precision='single')`` runs ``H|psi>``, the Davidson work arrays
and the wavefunction store in float32 and keeps the density, the potential, the
mixer and the energies in float64; the small subspace solves run in double. It is a
performance mode and no correctness claim is made in it, so what is checked here is
that it is what it says (32-bit where it should be, 64-bit where it should be), that
it lands within its measured floor of the double answer, that a double run continues
from it to the double answer, and that it refuses what it has not been checked on.
"""

import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.config import DOUBLE, SINGLE, resolve_band_precision
from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf.driver import Calculation, run_scf
from defumat.system import build_system

pytestmark = pytest.mark.unit

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"


def _cell(pseudo_dir, name):
    system = build_system(read_pw_input(BENCHMARKS / name))
    return system, tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)


@pytest.fixture(scope="module")
def silicon(pseudo_dir):
    return _cell(pseudo_dir, "si-1k.in")


def test_the_default_is_double_and_the_environment_moves_it(monkeypatch):
    monkeypatch.delenv("DEFUMAT_BAND_PRECISION", raising=False)
    assert resolve_band_precision() is DOUBLE
    monkeypatch.setenv("DEFUMAT_BAND_PRECISION", "single")
    assert resolve_band_precision() is SINGLE
    assert resolve_band_precision("double") is DOUBLE


def test_the_band_side_is_32_bit_and_the_grid_side_64(silicon):
    system, pseudos = silicon
    calculation = Calculation(system, pseudos, band_precision="single")
    hamiltonian = calculation.hamiltonian(
        jnp.zeros((calculation.nspin_mag,) + calculation.basis.dense.grid))[0]
    assert hamiltonian.dtype == SINGLE.complex
    assert hamiltonian.potential.dtype == hamiltonian.kinetic.dtype == SINGLE.real
    assert hamiltonian.projectors.at_k(0).dtype == SINGLE.complex
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-7)
    assert result.converged
    assert result.wavefunctions.dtype == SINGLE.complex
    assert result.density.dtype == DOUBLE.real
    assert np.asarray(result.eigenvalues).dtype == DOUBLE.real


def test_it_lands_within_its_floor_and_a_double_run_continues_from_it(silicon):
    """The floor measured on this cell is about 1e-6 Ry; the continuation reaches double."""
    system, pseudos = silicon
    exact = run_scf(system, pseudos, conv_thr=1e-12)
    single = run_scf(system, pseudos, conv_thr=1e-7, band_precision="single")
    assert abs(single.total_energy - exact.total_energy) < 1e-5
    assert np.abs(np.asarray(single.eigenvalues) - np.asarray(exact.eigenvalues)).max() < 1e-4
    finished = run_scf(system, pseudos, conv_thr=1e-12, starting_from=single)
    assert finished.wavefunctions.dtype == DOUBLE.complex
    assert abs(finished.total_energy - exact.total_energy) < 1e-10


def test_the_floors_follow_the_precision(silicon):
    system, pseudos = silicon
    double = Calculation(system, pseudos)
    assert (double.ethr_floor, double.conv_thr_floor) == (1e-13, 0.0)
    single = Calculation(system, pseudos, band_precision="single")
    assert single.ethr_floor == pytest.approx(
        Calculation.SINGLE_ETHR_FACTOR * SINGLE.eps * system.ecutwfc)
    with pytest.raises(ValueError, match="below what band_precision = 'single' delivers"):
        run_scf(system, pseudos, calculation=single, conv_thr=1e-10)


def test_what_it_has_not_been_checked_on_is_refused(pseudo_dir, silicon):
    system, pseudos = _cell(pseudo_dir, "si2-us-1k.in")
    with pytest.raises(NotImplementedError, match="collinear norm-conserving"):
        Calculation(system, pseudos, band_precision="single")
    # and every derivative of the energy, through the one gate they share
    from defumat.forces.energy import reject_potential_only

    system, pseudos = silicon
    with pytest.raises(NotImplementedError, match="single precision"):
        reject_potential_only(Calculation(system, pseudos, band_precision="single"))
    reject_potential_only(Calculation(system, pseudos))
