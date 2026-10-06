"""Real-time propagation against references that share no machinery with it.

Three comparisons, each an identity rather than an agreement:

* the perturbative orders of the current, by nested ``jvp`` through the
  propagation, against the dense frequency-domain hierarchy with every band on
  the sphere (``defumat.realtime.dense``), on zincblende AlAs, whose lack of an
  inversion centre lets the even orders be nonzero, at Gamma (where the
  projector row at ``k + G = 0`` lives) and at a general point;
* the linear response after a kick against ``optical_conductivity``'s resolvent
  sum fed every eigenstate of the dense ``H(k)``, plus the band curvature the
  sum replaces by its f-sum value;
* the current on the wedge of the field's little group against the whole mesh.

``tools/realtime/orders_vs_dense.py``, ``linear_vs_kubo.py`` and
``little_group_check.py`` are the same comparisons as scripts, with the
measurements recorded in ``PLAN.md`` P134.
"""

import math
import re
from functools import lru_cache
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.realtime.dense import dense_ground_states, dense_hamiltonians, dense_orders
from defumat.realtime.orders import propagate_orders
from defumat.realtime.pulse import Adiabatic, Kick
from defumat.realtime.spectra import conductivity_from_kick
from defumat.scf import Calculation
from defumat.system import build_system
from defumat.system.kpoints import KPoints

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
#: Gamma and one point of no symmetry, weights summing to one.
KSET = (np.array([[0.0, 0.0, 0.0], [0.25, 0.1, -0.05]]), np.array([0.5, 0.5]))


@pytest.fixture(autouse=True)
def _bounded_compilation():
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _cell(pseudo_dir, case, ecut):
    """``(calculation, terms, v_scf)`` at a cutoff chosen small, on :data:`KSET`.

    At the superposition of atomic charges: the identities below hold at any
    frozen potential, so none needs a converged one.
    """
    text = re.sub(r"ecutwfc\s*=\s*[0-9.dD+-]+", f"ecutwfc = {ecut}",
                  (CASES / f"{case}.in").read_text())
    path = Path("/tmp") / f"defumat-realtime-{case}-{ecut}.in"
    path.write_text(text)
    system = build_system(read_pw_input(path))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    system = eqx.tree_at(lambda s: s.kpoints, system,
                         KPoints(coords=KSET[0], weights=KSET[1]))
    calculation = Calculation(system, pseudos)
    v_scf = calculation.potential(calculation.starting_density()).v_scf
    return calculation, calculation.local_terms(v_scf), v_scf


def test_the_orders_are_the_dense_hierarchy(pseudo_dir):
    """``J_(n,m)`` for (1,1), (2,2), (2,0), (3,3), (3,1) on AlAs at 4 Ry.

    ``w = 0.05`` and ``eta = 0.01`` Hartree, the last period of a run of
    ``eta T = 20.1``, two k-points. Measured: **1.4e-5 to 3.0e-5** at 400 steps
    a period and **4.9e-6 to 7.6e-6** at 800, a factor of three to four, which
    is the midpoint rule's ``dt^2``; the same comparison at ``eta T = 6.3``
    reads 0.4 to 4.6 per cent, the start transient the toy model predicts. The
    even orders are nonzero on this cell (``|J_(2,2)| = 9.0e-3`` against
    ``|J_(1,1)| = 7.7e-3``), so they are compared and not a residue.
    """
    calculation, terms, v_scf = _cell(pseudo_dir, "alas-shg", 4.0)
    omega, eta, direction = 0.05, 0.01, np.array([1.0, 0.0, 0.0])
    nocc = int(round(calculation.nelec / 2))
    mask = np.asarray(calculation.basis.planewaves.mask)
    weights = np.asarray(calculation.system.kpoints.weights)
    states = np.zeros((len(weights), nocc, mask.shape[1]), dtype=complex)
    dense = {}
    for ik in range(len(weights)):
        h = dense_hamiltonians(calculation, terms, ik, direction, 4)
        e, u = dense_ground_states(h[0], nocc)
        states[ik][:, np.flatnonzero(mask[ik])] = u
        for key, value in dense_orders(h, e, u, np.full(nocc, weights[ik]),
                                       2 * omega, 2 * eta, nmax=3).items():
            dense[key] = dense.get(key, 0.0) + value
    volume = float(calculation.system.cell.volume)
    period = 2 * math.pi / omega
    length = math.ceil(20.0 / eta / period) * period
    shape = Adiabatic(amplitude=1.0, omega=omega, eta=eta, eta_t=length * eta)
    result = propagate_orders(
        calculation, jnp.asarray(states), np.repeat(weights[:, None], nocc, axis=1),
        v_scf, shape, dt=period / 400, order=3, start=-length, duration=length,
        k_batch=None)
    for n, m in ((1, 1), (2, 2), (2, 0), (3, 3), (3, 1)):
        reference = -dense[(n, m)] / (2.0 * volume)
        value = complex(result.component(n, m, axis=0))
        assert abs(value - reference) < 5e-5 * abs(reference), (n, m, value, reference)
    assert abs(dense[(2, 2)]) > 0.5 * abs(dense[(1, 1)]), "the even order is not a residue"


def test_the_linear_response_is_the_kubo_sum_plus_the_band_curvature(pseudo_dir):
    """``sigma_RT(z) = sigma_Kubo(z) + i D / (Omega z)`` on silicon at 6 Ry.

    The propagation carries the exact diamagnetic term and the Kubo sum the
    f-sum value of it, so on a finite set of k-points the two differ by
    ``D = sum w d^2 eps/dk^2``, taken here by a difference of the dense
    eigenvalues on the frozen sphere. Measured, 0.5 to 16 eV at
    ``eta = 0.02`` Hartree and ``dt = 0.05``: **1.6e-5** of the scale with the
    term, and the whole scale (0.81 against 0.81) without it, so the term is
    the size of the answer on two k-points and not a correction.
    """
    from defumat.response.conductivity import _resolvent_sum

    calculation, terms, v_scf = _cell(pseudo_dir, "si2-nosym", 6.0)
    eta, direction = 0.02, np.array([1.0, 0.0, 0.0])
    nocc = int(round(calculation.nelec / 2))
    mask = np.asarray(calculation.basis.planewaves.mask)
    weights = np.asarray(calculation.system.kpoints.weights)
    volume = float(calculation.system.cell.volume)
    frequencies = np.linspace(0.02, 0.6, 30)
    kcart = np.asarray(calculation.system.kpoints.cartesian(calculation.system.cell))
    states = np.zeros((len(weights), nocc, mask.shape[1]), dtype=complex)
    sigma_kubo = np.zeros(len(frequencies), dtype=complex)
    curvature, step = 0.0, 3e-4
    for ik in range(len(weights)):
        h = dense_hamiltonians(calculation, terms, ik, direction, 1)
        energies, vectors = np.linalg.eigh(h[0])
        states[ik][:, np.flatnonzero(mask[ik])] = vectors[:, :nocc].T
        element = np.zeros((3,) + h[1].shape, dtype=complex)
        element[0] = vectors.conj().T @ h[1] @ vectors
        wg = np.where(np.arange(len(energies)) < nocc, weights[ik], 0.0)
        filling = np.where(np.arange(len(energies)) < nocc, 1.0, 0.0)
        value, _ = _resolvent_sum(jnp.asarray(element), jnp.asarray(energies),
                                  jnp.asarray(wg), jnp.asarray(filling),
                                  jnp.asarray(2.0 * (frequencies + 1j * eta)), 1e-8)
        sigma_kubo += np.asarray(value)[:, 0, 0] / volume
        sums = [np.sum(np.linalg.eigvalsh(dense_hamiltonians(
            calculation, terms, ik, direction, 0,
            kcart=kcart + x * direction[None, :])[0])[:nocc]) for x in (-step, 0.0, step)]
        curvature += weights[ik] * (sums[0] - 2 * sums[1] + sums[2]) / step**2
    z = frequencies + 1j * eta
    reference = sigma_kubo + 1j * curvature / (volume * 2.0 * z)
    result = propagate_orders(
        calculation, jnp.asarray(states), np.repeat(weights[:, None], nocc, axis=1),
        v_scf, Kick(strength=1.0, direction=tuple(direction)), dt=0.05, order=1,
        start=0.0, duration=22.0 / eta, k_batch=None, block_steps=1000)
    sigma = conductivity_from_kick(result.times, result.currents[1], 1.0, direction,
                                   frequencies, eta).sigma[:, 0]
    scale = np.abs(reference).max()
    assert np.abs(sigma - reference).max() < 5e-5 * scale
    assert np.abs(sigma - sigma_kubo).max() > 0.1 * scale, "the curvature term is load-bearing"


def test_the_little_group_of_the_field_gives_the_whole_mesh_current(pseudo_dir):
    """A [100] pulse on silicon's 4x4x4 mesh: 18 points and eight operations against 64.

    Measured: the two currents agree to **6.2e-11** on a scale of 8.8e-4, which
    is the fixed-density solve's own threshold (the states at the members of a
    star are separate solves), and the wedge took 19 s against 63.
    """
    from defumat import Calculator
    from defumat.realtime.pulse import Sin2

    calculator = Calculator.from_file(CASES / "si2-symmetric.in", pseudo_dir=pseudo_dir,
                                      announce=False)
    calculator.get_scf(conv_thr=1e-10)
    pulse = Sin2.from_intensity(5e11, 1.55, 2, (1, 0, 0))
    whole = calculator.get_realtime(pulse, grid=(4, 4, 4), little_group=False, dt=0.2)
    wedge = calculator.get_realtime(pulse, grid=(4, 4, 4), little_group=True, dt=0.2)
    assert wedge.symmetry_operations == 8
    scale = np.abs(whole.current).max()
    assert scale > 1e-4
    assert np.abs(whole.current - wedge.current).max() < 1e-6 * scale
    # the transverse components vanish on the wedge by the symmetrisation, and
    # on the whole mesh by the same symmetry to the solve's threshold
    assert np.abs(wedge.current[:, 1:]).max() < 1e-12
