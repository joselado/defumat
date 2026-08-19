"""The stick FFT layout against the whole-box transform.

Two ways of computing the same thing, so the test is that they agree. The stick
path is QE's -- the z transform restricted to the columns the wavefunction
sphere occupies, and the field held with its xy plane contiguous -- and it is
what ``h_psi`` uses; the whole-box transform is the straightforward one and is
what everything on the dense grid still uses.
"""

import dataclasses
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from pypresso.basis.fft import g_to_r, r_to_sticks, sticks_to_r
from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf.driver import Calculation
from pypresso.scf.potential import v_of_rho
from pypresso.system import build_system

pytestmark = pytest.mark.unit

CASES = ["si-1k.in", "si8-1k.in"]


@pytest.fixture(scope="module", params=CASES)
def silicon(request, pseudo_dir):
    path = Path(__file__).resolve().parents[2] / "benchmarks" / request.param
    system = build_system(read_pw_input(path))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    potential = v_of_rho(calculation.starting_density(), calculation.basis.dense, system.cell)
    return system, calculation, calculation.hamiltonian(potential.v_scf)


def _random_wavefunctions(calculation, nbnd=3, seed=0):
    generator = np.random.default_rng(seed)
    shape = (nbnd, calculation.basis.npwx)
    psi = generator.normal(size=shape) + 1j * generator.normal(size=shape)
    return jnp.where(calculation.basis.planewaves.mask[0], jnp.asarray(psi), 0.0)


def test_the_stick_transform_gives_the_same_field(silicon):
    """Same field, only laid out ``(n3, n1, n2)`` instead of ``(n1, n2, n3)``."""
    _, calculation, _ = silicon
    psi = _random_wavefunctions(calculation)
    sticks = calculation.sticks

    whole = np.asarray(g_to_r(psi, calculation.fft_index[0], calculation.basis.dense.grid))
    stick = np.asarray(sticks_to_r(psi, sticks, sticks.columns[0], sticks.index[0]))
    assert np.moveaxis(stick, -3, -1) == pytest.approx(whole, abs=1e-10 * np.abs(whole).max())


def test_the_stick_transform_round_trips(silicon):
    """``r_to_sticks`` undoes ``sticks_to_r`` on the sphere it came from."""
    _, calculation, _ = silicon
    psi = _random_wavefunctions(calculation, seed=1)
    sticks = calculation.sticks

    field = sticks_to_r(psi, sticks, sticks.columns[0], sticks.index[0])
    back = np.asarray(r_to_sticks(field, sticks, sticks.columns[0], sticks.index[0]))
    assert back == pytest.approx(np.asarray(psi), abs=1e-10)


def test_h_psi_is_the_same_either_way(silicon):
    """The operator itself, which is what the choice of layout is for."""
    _, calculation, hamiltonian = silicon
    psi = _random_wavefunctions(calculation, seed=2)

    with_sticks = np.asarray(hamiltonian.apply(psi, 0))
    without = np.asarray(
        dataclasses.replace(hamiltonian, sticks=None).apply(psi, 0)
    )
    assert with_sticks == pytest.approx(without, abs=1e-10 * np.abs(without).max())


def test_every_plane_wave_lands_on_a_stick(silicon):
    """The layout must account for every plane wave, and only occupy real sticks."""
    _, calculation, _ = silicon
    sticks = calculation.sticks
    n1, n2, n3 = sticks.grid

    for ik in range(sticks.nk):
        keep = np.asarray(calculation.basis.planewaves.mask[ik])
        wanted = np.unique(np.asarray(calculation.fft_index[ik])[keep] // n3)
        columns = np.asarray(sticks.columns[ik])
        assert set(wanted) <= set(columns), "a plane wave sits outside every stick"
        assert len(np.unique(columns)) == len(columns), "a column is claimed twice"
        assert len(columns) <= n1 * n2
