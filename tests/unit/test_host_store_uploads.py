"""A streamed store reaches the device whole only through ``device_put``, and a chunk as rows.

``OPEN.md`` Part XXIII item 24 left eight files with a ``jnp.asarray`` of a
store that can be a host numpy array: where the store streams, an SCF and
``fixed_density_states(keep_states=True)`` hand the states back that way, and
``jnp.asarray`` of a host array peaks at twice its size on a card where
``jax.device_put`` of a contiguous one peaks at once (``GPU-MEMORY-NEXT.md``,
2026-09-29). Two of the sites take the whole k axis and are reached by a host
store: the band velocities, on a streamed ground state or a streamed NSCF (the
effective mass's included), and the dielectric tensor asked for its internals,
which keeps the whole-k route whatever the store. Three walk the k axis already
and took each chunk through ``jnp.asarray`` of a host copy: the force theorem's
projected band energy, the frozen first-order spin-orbit energy and the
spiral's. What is checked here is that none of the five puts a piece of a host
store through ``jnp.asarray`` any more; the values are held bit for bit where
the call is cheap enough to run whole. The memory is a card's and is not
measured here.

The two whole-k sites are stopped at the first object that receives the
uploaded states, since what follows is a velocity ``jvp`` or a self-consistent
response whose cost says nothing about the upload.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.calculator import Calculator

pytestmark = pytest.mark.unit

PSEUDO = "tests/data/pseudo"

#: Two-atom silicon on the whole unshifted 2x2x2 grid, fixed occupations.
SILICON = """
 &control
    calculation = 'scf'
 /
 &system
    ibrav = 2, celldm(1) = 10.2, nat = 2, ntyp = 1,
    ecutwfc = 10.0, nosym = .true.
 /
 &electrons
    conv_thr = 1.0d-10
 /
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS crystal
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS (automatic)
 2 2 2 0 0 0
"""

#: One iodine atom in a small box with spin-orbit coupling switched off: the
#: cheapest noncollinear cell on a fully-relativistic norm-conserving dataset
#: (``tests/unit/test_spiral_soc.py``'s, without its spiral unless asked).
IODINE = """
 &control
    calculation = 'scf'
 /
 &system
    ibrav = 1, celldm(1) = 8.0, nat = 1, ntyp = 1, ecutwfc = 15.0, nbnd = 8,
    occupations = 'smearing', smearing = 'gaussian', degauss = 0.05,
    noncolin = .true., lspinorb = .true., soc_scale = 0.0, nosym = .true.,
    starting_magnetization(1) = 0.5, angle1(1) = 90.0{spiral}
 /
ATOMIC_SPECIES
 I 126.9 I.rel-pbe-nc-dojo.UPF
ATOMIC_POSITIONS crystal
 I 0.00 0.00 0.00
K_POINTS (automatic)
 1 1 2 0 0 0
"""
SPIRAL = ",\n    spiral_q(1) = 0.0, spiral_q(2) = 0.0, spiral_q(3) = 0.25"


class _Stop(Exception):
    """Raised by a stand-in once it has seen what it was handed."""


def _host_uploads(monkeypatch, store):
    """Record every ``jnp.asarray`` of a numpy array that is a piece of ``store``.

    A piece is anything whose two trailing axes are the store's ``(nbnd,
    ndim)``: the whole store, a chunk of its rows, or one k-point's block.
    """
    trailing = tuple(store.shape[-2:])
    seen = []
    real = jnp.asarray

    def spy(value, *args, **kwargs):
        if (isinstance(value, np.ndarray) and value.ndim >= 2
                and tuple(value.shape[-2:]) == trailing):
            seen.append(value.shape)
        return real(value, *args, **kwargs)

    monkeypatch.setattr(jnp, "asarray", spy)
    return seen


def _silicon(nbnd=8, nocc=4):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculation = Calculator.from_text(SILICON, PSEUDO, k_batch=3,
                                           announce=False).calculation
    density = calculation.starting_density()
    v_scf = calculation.potential(density).v_scf
    states = calculation.starting_wavefunctions(calculation.hamiltonian(v_scf), nbnd)
    nspin, nk = states.shape[:2]
    rng = np.random.default_rng(7)
    below = np.sort(rng.uniform(-0.6, 0.1, (nspin, nk, nocc)), axis=-1)
    above = np.sort(rng.uniform(0.5, 1.6, (nspin, nk, nbnd - nocc)), axis=-1)
    eigenvalues = np.concatenate([below, above], -1)
    return calculation, density, v_scf, np.array(states), eigenvalues


def test_the_band_velocities_take_a_host_store_through_device_put(monkeypatch):
    from defumat.response.velocity import VelocityOperator

    calculation, _, v_scf, host, eigenvalues = _silicon()
    operator = VelocityOperator(calculation, v_scf)
    handed = []

    def both(self, psi, direction):
        handed.append(psi)
        raise _Stop

    seen = _host_uploads(monkeypatch, host)
    monkeypatch.setattr(VelocityOperator, "both", both)
    with pytest.raises(_Stop):
        operator.band_velocities(host, eigenvalues)
    monkeypatch.undo()

    assert not seen, f"jnp.asarray put a host store on the device: {seen}"
    assert isinstance(handed[0], jax.Array)
    assert np.array_equal(np.asarray(handed[0]), host)


def test_the_dielectric_internals_take_a_host_store_through_device_put(monkeypatch):
    """``keep_internals`` keeps the whole-k route, a streamed store included."""
    import defumat.response.efield as efield

    calculation, density, _, host, eigenvalues = _silicon(nbnd=4, nocc=4)
    handed = []

    def solver(calculation, hamiltonians, psi, *args, **kwargs):
        handed.append(psi)
        raise _Stop

    seen = _host_uploads(monkeypatch, host)
    monkeypatch.setattr(efield, "SternheimerSolver", solver)
    with pytest.raises(_Stop):
        efield.dielectric_tensor(calculation, host, eigenvalues, density,
                                 born_charges=False, keep_internals=True)
    monkeypatch.undo()

    assert not seen, f"jnp.asarray put a host store on the device: {seen}"
    assert isinstance(handed[0], jax.Array)
    assert np.array_equal(np.asarray(handed[0]), host)


def _iodine(pseudo_dir: Path, spiral=False):
    from defumat.io.pwin import parse_pw_input
    from defumat.pseudo import read_upf
    from defumat.scf.driver import Calculation
    from defumat.system import build_system

    system = build_system(parse_pw_input(IODINE.format(spiral=SPIRAL if spiral else "")))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file)
                    for s in system.structure.species)
    return system, pseudos, Calculation(system, pseudos)


def _random_store(nk, ndim, nbnd=5, seed=3):
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=(1, nk, nbnd, ndim)) + 1j * rng.normal(size=(1, nk, nbnd, ndim))
    weights = rng.random((1, nk, nbnd))
    return psi, weights


def test_the_spiral_expectation_takes_each_chunk_as_rows(monkeypatch, pseudo_dir):
    from defumat.workflows.anisotropy import _first_order_operator
    from defumat.workflows.spiral_soc import spiral_expectation

    _, _, calculation = _iodine(pseudo_dir, spiral=True)
    delta, _ = _first_order_operator(calculation, None)
    vkb = calculation.projectors.vkb
    host, weights = _random_store(vkb.shape[0] // 2, 2 * vkb.shape[1])

    on_device = spiral_expectation(calculation, jnp.asarray(host), weights, delta)
    seen = _host_uploads(monkeypatch, host)
    walked = spiral_expectation(calculation, host, weights, delta)
    monkeypatch.undo()

    assert not seen, f"jnp.asarray put a host chunk on the device: {seen}"
    for a, b in zip(walked, on_device):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_the_projected_band_energy_takes_each_k_point_through_device_put(
        monkeypatch, pseudo_dir):
    from defumat.workflows.anisotropy import _project_band_energy

    system, _, calculation = _iodine(pseudo_dir)
    nk = calculation.system.kpoints.nk
    ndim = 2 * calculation.projectors.vkb.shape[1]
    host, wg = _random_store(nk, ndim)
    eigenvalues = np.sort(np.random.default_rng(5).uniform(-0.5, 0.5, wg.shape), -1)

    on_device = _project_band_energy(calculation, system, eigenvalues, wg,
                                     jnp.asarray(host), 0.1)
    seen = _host_uploads(monkeypatch, host)
    walked = _project_band_energy(calculation, system, eigenvalues, wg, host, 0.1)
    monkeypatch.undo()

    assert not seen, f"jnp.asarray put a host k-point on the device: {seen}"
    assert np.array_equal(walked.by_orbital, on_device.by_orbital)


def test_the_frozen_expectation_takes_each_chunk_as_rows(monkeypatch, pseudo_dir):
    """:func:`frozen_expectation` on stand-in states, so no NSCF is run."""
    import defumat.workflows.anisotropy as anisotropy

    system, pseudos, calculation = _iodine(pseudo_dir)
    nk = calculation.system.kpoints.nk
    ndim = 2 * calculation.projectors.vkb.shape[1]
    host, _ = _random_store(nk, ndim, nbnd=8)
    eigenvalues = np.sort(np.random.default_rng(5).uniform(-0.5, 0.5, (1, nk, 8)), -1)
    grid = calculation.basis.dense.grid
    rng = np.random.default_rng(11)
    density = np.stack([0.02 + 0.01 * rng.random(grid), 0.01 * rng.random(grid)])
    store = {}

    def states(system, pseudos, density, **_):
        # The calculation already built, turned to the system it is handed, as
        # a scan of the force theorem turns it.
        return calculation.with_texture(system), system, eigenvalues, store["psi"]

    monkeypatch.setattr(anisotropy, "fixed_density_states", states)
    store["psi"] = jnp.asarray(host)
    on_device = anisotropy.frozen_expectation(system, pseudos, density)
    store["psi"] = host
    seen = _host_uploads(monkeypatch, host)
    walked = anisotropy.frozen_expectation(system, pseudos, density)
    monkeypatch.undo()

    assert not seen, f"jnp.asarray put a host chunk on the device: {seen}"
    assert walked == on_device
