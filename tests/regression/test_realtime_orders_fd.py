"""The third order of the current by nested ``jvp`` against a finite difference of full runs.

``HARMONICS-NEXT.md``'s ladder item 5: the same propagation run at the field
amplitudes ``lam = +-h, +-2h`` and differenced,

    J^(3) ~ ([J(2h) - J(-2h)] - 2 [J(h) - J(-h)]) / (12 h^3) = J^(3) + 5 h^2 J^(5) + ...,
    J^(1) ~ (8 [J(h) - J(-h)] - [J(2h) - J(-2h)]) / (12 h)   = J^(1) - 4 h^4 J^(5) + ...,

against :func:`~defumat.realtime.orders.propagate_orders`, which takes the
orders by forward differentiation through every step at ``lam = 0``. The two
routes share the propagator, the time grid and the start transient of the
adiabatic switch-on, so the comparison is of the differentiation and nothing
else, and the run can be short. ``tools/realtime/orders_vs_fd.py`` is the same
comparison as a script, with the scan in ``h`` below.
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
from defumat.realtime.dense import dense_ground_states, dense_hamiltonians
from defumat.realtime.orders import fourier_component, propagate_orders
from defumat.realtime.propagate import _prepare, propagate
from defumat.realtime.pulse import Adiabatic
from defumat.scf import Calculation
from defumat.system import build_system
from defumat.system.kpoints import KPoints

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
#: Gamma and one point of no symmetry, weights summing to one, as in
#: ``test_realtime.py``.
KSET = (np.array([[0.0, 0.0, 0.0], [0.25, 0.1, -0.05]]), np.array([0.5, 0.5]))


@pytest.fixture(autouse=True)
def _bounded_compilation():
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _cell(pseudo_dir, case, ecut):
    """``(calculation, terms, v_scf)`` at a cutoff chosen small, on :data:`KSET`.

    At the superposition of atomic charges: the identity below holds at any
    frozen potential.
    """
    text = re.sub(r"ecutwfc\s*=\s*[0-9.dD+-]+", f"ecutwfc = {ecut}",
                  (CASES / f"{case}.in").read_text())
    path = Path("/tmp") / f"defumat-realtime-fd-{case}-{ecut}.in"
    path.write_text(text)
    system = build_system(read_pw_input(path))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    system = eqx.tree_at(lambda s: s.kpoints, system,
                         KPoints(coords=KSET[0], weights=KSET[1]))
    calculation = Calculation(system, pseudos)
    v_scf = calculation.potential(calculation.starting_density()).v_scf
    return calculation, calculation.local_terms(v_scf), v_scf


def test_the_third_order_is_the_finite_difference_of_full_runs(pseudo_dir):
    """``J^(3)(t)`` by nested ``jvp`` against the four-point stencil, on AlAs at 4 Ry.

    ``w = 0.05`` and ``eta = 0.01`` Hartree along ``[100]``, five periods
    (``eta T = 6.3``) at 400 steps a period, two k-points, the dense ground
    states. Measured on D22 (``tools/realtime/orders_vs_fd.py``), the largest
    difference over the whole time series relative to the largest ``|J^(3)|``,
    and the relative difference of ``J_(3,3)`` and ``J_(3,1)``:

    ========  ==========  ==========  ==========  ===========
    h         J^(3)(t)    J_(3,3)     J_(3,1)     J^(1)(t)
    ========  ==========  ==========  ==========  ===========
    2.5e-3    4.1e-4      2.7e-4      1.7e-4      3.3e-8
    1.25e-3   1.0e-4      6.8e-5      4.3e-5      2.1e-9
    6.25e-4   2.5e-5      1.7e-5      1.0e-5      1.3e-10
    3.125e-4  **6.5e-6**  **6.5e-6**  **3.3e-6**  **8.0e-12**
    1.56e-4   3.5e-6      1.0e-5      8.0e-6      4.8e-12
    7.8e-5    2.9e-5      3.4e-5      2.7e-5      1.0e-11
    3.9e-5    2.4e-4      2.1e-3      1.1e-3      1.9e-11
    ========  ==========  ==========  ==========  ===========

    The third order falls by four per halving of ``h`` down to 3.1e-4 and the
    first by sixteen, which are the stencil's ``h^2`` and ``h^4``, so the
    difference there is the stencil's own truncation, ``5 h^2 J^(5)``, and not
    the differentiation; below it the rounding of ``J``, which on this k-set
    is not small at zero field (one k-point of each star), is divided by
    ``12 h^3`` and grows as ``1/h^3``, by 8.3 from 7.8e-5 to 3.9e-5. The test
    runs ``h`` = 3.125e-4 and 6.25e-4 and asserts both the agreement and the
    ratio of four between them, so the agreement is shown to be the
    stencil converging on the derivative rather than a coincidence of one
    ``h``. The centre of the step is checked equal in both routes, since the
    propagator is the same map only if it is.
    """
    calculation, terms, v_scf = _cell(pseudo_dir, "alas-shg", 4.0)
    omega, eta, direction = 0.05, 0.01, np.array([1.0, 0.0, 0.0])
    nocc = int(round(calculation.nelec / 2))
    mask = np.asarray(calculation.basis.planewaves.mask)
    weights = np.asarray(calculation.system.kpoints.weights)
    states = np.zeros((len(weights), nocc, mask.shape[1]), dtype=complex)
    for ik in range(len(weights)):
        h0 = dense_hamiltonians(calculation, terms, ik, direction, 0)[0]
        _, u = dense_ground_states(h0, nocc)
        states[ik][:, np.flatnonzero(mask[ik])] = u
    states = jnp.asarray(states)
    w = np.repeat(weights[:, None], nocc, axis=1)
    period = 2 * math.pi / omega
    length = math.ceil(6.0 / eta / period) * period
    dt = period / 400

    def shape(amplitude):
        return Adiabatic(amplitude=amplitude, omega=omega, eta=eta, eta_t=length * eta,
                         polarization=tuple(direction))

    h = 3.125e-4
    # the centre is clamped by the top of the spectrum, which moves with the
    # largest shift; the two routes step with the same map only when it is not
    centres = [_prepare(calculation, states, w, v_scf, k, dt, "taylor4", None, None).centre
               for k in (0.0, 4 * h)]
    assert abs(centres[0] - centres[1]) < 1e-12

    orders = propagate_orders(calculation, states, w, v_scf, shape(1.0), dt=dt, order=3,
                              start=-length, duration=length, k_batch=None)
    runs = {lam: propagate(calculation, states, w, v_scf, shape(lam), dt=dt,
                           start=-length, duration=length, k_batch=None).current
            for lam in (s * f * h for f in (1, 2, 4) for s in (1, -1))}
    j1, j3 = orders.currents[1], orders.currents[3]
    times = orders.times

    def stencil(step):
        odd1 = runs[step] - runs[-step]
        odd2 = runs[2 * step] - runs[-2 * step]
        return (odd2 - 2.0 * odd1) / (12.0 * step**3), (8.0 * odd1 - odd2) / (12.0 * step)

    errors = {}
    for step in (h, 2 * h):
        fd3, fd1 = stencil(step)
        series = np.abs(fd3 - j3).max() / np.abs(j3).max()
        components = [abs(complex(fourier_component(times, fd3, 3, m, omega, eta)[0])
                          - complex(fourier_component(times, j3, 3, m, omega, eta)[0]))
                      / abs(complex(fourier_component(times, j3, 3, m, omega, eta)[0]))
                      for m in (3, 1)]
        errors[step] = series
        if step == h:
            assert series < 2e-5, series
            assert max(components) < 2e-5, components
            assert np.abs(fd1 - j1).max() / np.abs(j1).max() < 1e-10
    ratio = errors[2 * h] / errors[h]
    assert 3.0 < ratio < 5.0, ratio
    # the third harmonic is not a residue on this cell
    assert abs(complex(orders.component(3, 3, axis=0))) > abs(
        complex(orders.component(1, 1, axis=0)))
