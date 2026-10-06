"""The frequency-domain hierarchy against the dense one and against the propagation.

Two comparisons, each an identity rather than an agreement:

* the iterative hierarchy of :mod:`defumat.realtime.hierarchy` against
  :func:`~defumat.realtime.dense.dense_orders` with every band of the sphere, on
  zincblende AlAs at 4 Ry at Gamma and a general point, all five components of
  orders one to three at three frequencies, and the result unchanged when the
  projector on the computed bands holds half as many;
* the spectrum's second and third orders against the real-time route's at one
  frequency, which shares the table of the projectors and nothing of the solve.

``PLAN.md`` P137 has the measurements.
"""

import math
import re
from functools import lru_cache
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.realtime.dense import dense_ground_states, dense_hamiltonians, dense_orders
from defumat.realtime.hierarchy import HierarchyError, hierarchy_orders
from defumat.scf import Calculation
from defumat.system import build_system
from defumat.system.kpoints import KPoints

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
#: Gamma and one point of no symmetry, weights summing to one, as in ``test_realtime.py``.
KSET = (np.array([[0.0, 0.0, 0.0], [0.25, 0.1, -0.05]]), np.array([0.5, 0.5]))
COMPONENTS = ((1, 1), (2, 2), (2, 0), (3, 3), (3, 1))


@pytest.fixture(autouse=True)
def _bounded_compilation():
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _dense_cell(pseudo_dir, case, ecut, computed):
    """``(calculation, v_scf, hamiltonians, basis, energies)`` on :data:`KSET` at ``ecut``.

    At the superposition of atomic charges, where the identity holds as at any
    frozen potential; ``basis`` is the lowest ``computed`` dense eigenstates of
    each sphere.
    """
    text = re.sub(r"ecutwfc\s*=\s*[0-9.dD+-]+", f"ecutwfc = {ecut}",
                  (CASES / f"{case}.in").read_text())
    path = Path("/tmp") / f"defumat-hierarchy-{case}-{ecut}.in"
    path.write_text(text)
    system = build_system(read_pw_input(path))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    system = eqx.tree_at(lambda s: s.kpoints, system, KPoints(coords=KSET[0], weights=KSET[1]))
    calculation = Calculation(system, pseudos)
    v_scf = calculation.potential(calculation.starting_density()).v_scf
    terms = calculation.local_terms(v_scf)
    mask = np.asarray(calculation.basis.planewaves.mask)
    nk, npwx = mask.shape
    hamiltonians, basis = [], np.zeros((nk, computed, npwx), dtype=complex)
    energies = np.zeros((nk, computed))
    for ik in range(nk):
        h = dense_hamiltonians(calculation, terms, ik, np.array([1.0, 0.0, 0.0]), 4)
        values, vectors = np.linalg.eigh(h[0])
        basis[ik][:, np.flatnonzero(mask[ik])] = vectors[:, :computed].T
        energies[ik] = values[:computed]
        hamiltonians.append(h)
    return calculation, v_scf, hamiltonians, basis, energies


def test_the_iterative_hierarchy_is_the_dense_hierarchy(pseudo_dir):
    """All five components at 0.02, 0.05 and 0.11 Ha, ``eta = 0.01`` Ha, on AlAs at 4 Ry.

    Measured with twelve computed bands in the projector: **1e-13 to 8e-12**
    relative at a BiCGStab tolerance of 1e-10, in 18 to 19 iterations; 2e-11 to
    1.6e-10 with six bands (18 to 39 iterations), 1e-11 to 1.3e-9 at a tolerance
    of 1e-8 and 5e-13 to 3.5e-12 at 1e-12. The frequencies put ``3w`` below,
    across and above the gap of this cell, and the even orders are nonzero
    here (no inversion), so every component is compared and none is a residue.
    """
    calculation, v_scf, hamiltonians, basis, energies = _dense_cell(
        pseudo_dir, "alas-shg", 4.0, 12)
    nocc = int(round(calculation.nelec / 2))
    weights = np.asarray(calculation.system.kpoints.weights)
    volume = float(calculation.system.cell.volume)
    omegas, eta = (0.02, 0.05, 0.11), 0.01
    w = np.repeat(weights[:, None], nocc, axis=1)
    results = {}
    for computed in (12, 6):
        results[computed] = hierarchy_orders(
            calculation, basis[:, :computed], energies[:, :computed], w, v_scf,
            omegas=omegas, eta=eta, direction=(1.0, 0.0, 0.0), order=3, tolerance=1e-10)
    for iw, omega in enumerate(omegas):
        dense = {}
        for ik, h in enumerate(hamiltonians):
            e, u = dense_ground_states(h[0], nocc)
            for key, value in dense_orders(h, e, u, np.full(nocc, weights[ik]), 2 * omega,
                                           2 * eta, nmax=3).items():
                dense[key] = dense.get(key, 0.0) + value
        for key in COMPONENTS:
            reference = -dense[key] / (2.0 * volume)
            for computed, tolerance in ((12, 1e-10), (6, 1e-9)):
                value = results[computed]["components"][key][iw, 0]
                assert abs(value - reference) < tolerance * abs(reference), (
                    computed, omega, key, value, reference)
    assert results[12]["iterations"].max() < 40


def test_a_component_that_does_not_converge_is_refused(pseudo_dir):
    calculation, v_scf, _, basis, energies = _dense_cell(pseudo_dir, "alas-shg", 4.0, 12)
    nocc = int(round(calculation.nelec / 2))
    weights = np.repeat(np.asarray(calculation.system.kpoints.weights)[:, None], nocc, axis=1)
    with pytest.raises(HierarchyError, match="did not converge"):
        hierarchy_orders(calculation, basis, energies, weights, v_scf, omegas=(0.05,), eta=0.01,
                         direction=(1.0, 0.0, 0.0), order=1, max_iterations=2)


def test_the_spectrum_is_the_propagation_at_one_frequency(pseudo_dir, tmp_path):
    """The hierarchy's ``J_(n,m)`` against ``get_harmonic_orders`` on AlAs at 6 Ry, 2x2x2, [111].

    1.5 eV and ``eta = 0.3`` eV, ``eta_t = 12``, the field along [111], whose
    little group keeps six operations; the propagation's error is its ``dt^2``
    and its start transient, ``exp(-eta_t)`` times a resonance factor. Measured:
    XXMEASUREDXX.
    """
    from defumat import Calculator

    text = (CASES / "alas-shg.in").read_text()
    text = re.sub(r"ecutwfc\s*=\s*[0-9.dD+-]+", "ecutwfc = 6.0", text)
    text = re.sub(r"^\s*no(sym|inv)\s*=.*\n", "", text, flags=re.M)
    path = tmp_path / "alas-shg-6.in"
    path.write_text(text)
    calculator = Calculator.from_file(path, pseudo_dir=pseudo_dir, announce=False)
    calculator.get_scf(conv_thr=1e-12)
    direction = tuple(np.ones(3) / math.sqrt(3.0))
    options = dict(broadening=0.3, direction=direction, grid=(2, 2, 2), conv_thr=1e-12)
    spectrum = calculator.get_nonlinear_spectrum([1.5], order=3, **options)
    orders = calculator.get_harmonic_orders(1.5, order=3, eta_t=12.0, steps_per_period=400,
                                            **options)
    for key in COMPONENTS:
        value = spectrum.component(*key, axis=0)[0]
        reference = complex(orders.component(*key, axis=0))
        assert abs(value - reference) < 1e-3 * abs(reference), (key, value, reference)
