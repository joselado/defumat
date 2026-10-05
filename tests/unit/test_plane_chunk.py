"""The local term a chunk of ``z`` planes at a time (``defumat.basis.fft.sticks_local``).

The chunked path issues the same transforms on the same numbers as the
whole-box one, in a different order, so ``H psi`` must agree to round-off --
here to 1e-13 relative, on a scalar and a noncollinear magnetic Hamiltonian,
with a chunk that does not divide ``n3`` so that the padded tail is exercised.
The dial itself is checked for its platform rule and its environment override.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.batching import PLANE_CHUNK_BYTES, resolve_plane_chunk
from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf.driver import Calculation
from defumat.system import build_system
from tests.backend import on_a_cpu

REPO = Path(__file__).resolve().parents[2]

SILICON = """\
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.2, nat = 2, ntyp = 1, ecutwfc = 12.0{extra}
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


def _calculation(extra: str = "") -> Calculation:
    with tempfile.NamedTemporaryFile("w", suffix=".in", delete=False) as handle:
        handle.write(SILICON.format(extra=extra))
    system = build_system(read_pw_input(Path(handle.name)))
    pseudos = tuple(read_upf(REPO / "tests" / "data" / "pseudo" / s.pseudo_file)
                    for s in system.structure.species)
    return Calculation(system, pseudos)


def _compare(calculation: Calculation, chunk: int) -> None:
    rho = calculation.starting_density()
    potential = calculation.potential(rho)
    hamiltonian = calculation.hamiltonian(potential.v_scf)[0]
    whole = dataclasses.replace(hamiltonian, plane_chunk=None)
    chunked = dataclasses.replace(hamiltonian, plane_chunk=chunk)
    n3 = calculation.sticks.grid[2]
    assert n3 % chunk != 0, "the padded tail is what this test is for"
    ndim = calculation.system.npol * calculation.basis.planewaves.npwx
    rng = np.random.default_rng(0)
    psi = jnp.asarray(rng.standard_normal((4, ndim)) + 1j * rng.standard_normal((4, ndim)))
    reference = whole.apply(psi, 1)
    result = chunked.apply(psi, 1)
    scale = float(jnp.max(jnp.abs(reference)))
    assert float(jnp.max(jnp.abs(result - reference))) < 1e-13 * scale


def test_scalar_local_term_is_the_same_in_chunks():
    _compare(_calculation(), chunk=3)


def test_spinor_local_term_is_the_same_in_chunks():
    calculation = _calculation(
        ",\n  noncolin = .true., nosym = .true., noinv = .true.,\n"
        "  starting_magnetization(1) = 0.5, angle1(1) = 45.0, angle2(1) = 30.0")
    assert calculation.noncolin and calculation.nspin_mag == 4
    _compare(calculation, chunk=4)


def test_the_dial_follows_the_platform_and_the_environment(monkeypatch):
    on_a_cpu(monkeypatch)
    grid = (36, 36, 144)
    monkeypatch.delenv("DEFUMAT_PLANE_CHUNK", raising=False)
    # On a CPU: the byte budget's worth of planes, at least one, at most n3,
    # a plane counting its wavefunction components twice and the potential's once.
    assert resolve_plane_chunk(grid, 16) == int(PLANE_CHUNK_BYTES // (36 * 36 * 16 * 2.5))
    assert resolve_plane_chunk(grid, 16, fields=2, potentials=4) == 3
    assert resolve_plane_chunk((400, 400, 10), 16) == 1
    assert resolve_plane_chunk((4, 4, 3), 16) == 3
    monkeypatch.setenv("DEFUMAT_PLANE_CHUNK", "off")
    assert resolve_plane_chunk(grid, 16) is None
    monkeypatch.setenv("DEFUMAT_PLANE_CHUNK", "8")
    assert resolve_plane_chunk(grid, 16) == 8
    assert resolve_plane_chunk(grid, 16, requested=None) is None
    assert resolve_plane_chunk(grid, 16, requested=500) == 144


def _compare_density(calculation: Calculation, chunk: int) -> None:
    """``sum_band`` through plane chunks against the whole-box path, to round-off."""
    potential = calculation.potential(calculation.starting_density())
    hamiltonians = calculation.hamiltonian(potential.v_scf)
    states = calculation.starting_wavefunctions(hamiltonians, 4)
    rng = np.random.default_rng(1)
    weights = jnp.asarray(rng.uniform(0.1, 1.0, states.shape[:3]))
    assert calculation.sticks.grid[2] % chunk != 0
    calculation.plane_chunk = None
    reference = calculation.smooth_density(states, weights)
    calculation.plane_chunk = chunk
    result = calculation.smooth_density(states, weights)
    assert result.shape == reference.shape
    scale = float(jnp.max(jnp.abs(reference)))
    assert float(jnp.max(jnp.abs(result - reference))) < 1e-13 * scale


def test_scalar_density_is_the_same_in_chunks():
    _compare_density(_calculation(), chunk=3)


def test_spinor_density_is_the_same_in_chunks():
    calculation = _calculation(
        ",\n  noncolin = .true., nosym = .true., noinv = .true.,\n"
        "  starting_magnetization(1) = 0.5, angle1(1) = 45.0, angle2(1) = 30.0")
    _compare_density(calculation, chunk=4)
