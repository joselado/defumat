"""chi^(2)(-2w; w, w) of zincblende AlAs from the real-time second order, against get_shg.

The second order of the current under the adiabatic field, ``J_(2,2)``, is turned
into ``chi^(2)`` by :func:`~defumat.workflows.realtime.chi2_from_orders`, and the
reference is the sum over states of :mod:`defumat.response.shg` on the same whole
unshifted mesh and cutoff. AlAs is the cell because silicon's second order is zero by
inversion, where an agreement would be about a residue. ``alas-shg.in`` is run with
its symmetry kept and its density on its own 4x4x4 grid, and the field is along
[111], whose little group C3v keeps 20 of the 64 points and whose current is
``(2/sqrt 3) chi_xyz``.

What the comparison found, at 12 Ry on the whole 4x4x4 mesh with ``eta = 0.2`` eV
unless said otherwise. The real-time side of the mesh and cutoff series is the dense
hierarchy of :mod:`defumat.realtime.dense` with every band of the sphere, which the
propagation itself reproduces: on this mesh, 3.0e-4 at 0.7 eV (``eta_t = 12``, 1250
steps a period) and 6.1e-5 at 1.5 eV (600 steps a period), 124.918 + 20.117i and
215.161 + 321.547i pm/V.

* **The conventions differ, and the relation is** ``chi_RT = -conj(chi_shg)``, at every
  frequency (0.4 to 1.7 eV), broadening (0.1, 0.2 and 0.3 eV), mesh (4^3, 6^3, 8^3) and
  cutoff (8, 12, 16 Ry) measured, in the sense that ``|chi_RT + conj(chi_shg)|`` is the
  few per cent of the next item while ``|chi_RT - conj(chi_shg)|`` and
  ``|chi_RT - chi_shg|`` are of the size of ``chi`` itself. The conjugate is the time
  convention, ``second_harmonic`` evaluating at ``w - i eta``
  (``response/shg.py:493-494``, as Elk's ``nonlinopt.f90:203`` does). The minus is the
  sign of the charge, which ``chi^(2)`` is odd in: the real-time route is
  an electron of charge -1 (``H(k + kappa)`` with ``kappa = A/c``, ``E = -dkappa/dt``,
  ``J = -(1/Omega) sum dH/dk``), and the sum over states carries no charge at all
  (``shg.py:658``, Elk's ``t0 = wkptnr/omega``), which is Hughes and Sipe's ``e^3`` set
  to +1. So the real-time route's static ``chi_xyz`` is positive on this cell where
  ``get_shg``'s is negative.
* **With every band of the sphere the magnitudes differ by 5 to 9 per cent, and the
  difference is the projectors.** At 0.4, 0.7 and 1.5 eV it is 7.5, 6.8 and 5.0 per
  cent here, 8.3, 7.7 and 5.3 on 6x6x6 and 8.6, 8.2 and 5.9 on 8x8x8, so it is not the
  mesh; it is 8.2, 7.8 and 5.1 at 8 Ry and 6.8, 5.9 and 6.0 at 16 Ry. With ``D_ij = 0``
  and the local potential tripled to keep a gap, the same comparison closes to 6.2e-4
  to 6.8e-4 at 12 Ry and 2.2e-5 to 4.3e-5 at 8 Ry. That is the term ``photocurrent.py``'s
  docstring names: the sum rule of Hughes and Sipe puts the free-electron
  ``delta_ab`` where ``<n|d^2H/dk_a dk_b|m>`` belongs, which is exact for a local
  Hamiltonian and leaves out the projectors' curvature with a nonlocal one, while the
  real-time route differentiates ``H(k + kappa)`` itself.
* **The truncated band sum moves the other way**, so a truncated ``get_shg`` agrees
  better than a complete one: against the propagation, 1.35, 0.68 and 1.20 per cent at
  23, 40 and 80 bands at 0.7 eV and 1.44, 1.37 and 1.87 at 1.5 eV, against 6.9 and 5.0
  with every band. Against the hierarchy at ``eta = 0.1`` eV and 0.4 eV the sum is 2.2
  per cent short at 23 bands, 1.4 over at 181 (the smallest sphere on the mesh) and 6.6
  over with every band of every sphere, so the states at the edge of a 12 Ry sphere
  carry 5 per cent.
* **Dropping the pairs closer than the broadening** (``shg.py:56``, ``:289``) is worth
  0.2 to 1 per cent of ``chi`` here, measured as ``degeneracy_tol = 1e-6`` Ry against
  the default.

The step has to keep the driver's centre on the occupied states, which is tighter than
the stability bound it refuses: at 12 Ry a ``dt`` of 0.305 is inside the bound of 0.341,
puts the centre 9.8 Ry above the bands, damps each occupied state by a factor 0.93 to
0.95 a step, and the first-order current fell from 1e-8 to 1e-20 in 800 steps; 0.2 and
below keep the centre at 0.15 Ry. At 8 Ry, 0.38 is inside the bound of 0.46 and leaves
2e-5 of an occupied state's amplitude after 1000 steps, 0.3 leaves 0.997, and 0.25 and
below put the centre on the bands.
"""

import math
import re
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defumat import Calculator
from defumat.realtime.dense import dense_hamiltonians, dense_orders
from defumat.response.shg import CHI2_AU_TO_PM_PER_V, _chi_at_k
from defumat.scf import Calculation
from defumat.system.kpoints import KPoints
from defumat.units import HARTREE_TO_EV, RY_TO_EV
from defumat.workflows.realtime import chi2_from_orders

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
#: The field along [111], whose little group in -43m is C3v: 20 of the 64 points.
DIRECTION = np.ones(3) / math.sqrt(3.0)
#: In zincblende every chi^abc with three distinct labels is chi_xyz, so the
#: current along a [111] field is sum_abc e_a e_b e_c chi^abc = (2/sqrt 3) chi_xyz.
TO_XYZ = math.sqrt(3.0) / 2.0
GRID = (4, 4, 4)


@pytest.fixture(autouse=True)
def _bounded_compilation():
    yield
    jax.clear_caches()


def _calculator(pseudo_dir, tmp_path, ecut):
    """``(calculator, scf)`` for ``alas-shg.in`` at ``ecut``, symmetry kept, on its 4x4x4 grid.

    ``nosym`` is dropped so that the field's little group can reduce the mesh
    (under ``nosym`` it is the identity alone); the density is the symmetric
    one, and both routes below run on it.
    """
    text = (CASES / "alas-shg.in").read_text()
    text = re.sub(r"ecutwfc\s*=\s*[0-9.dD+-]+", f"ecutwfc = {ecut}", text)
    text = re.sub(r"^\s*no(sym|inv)\s*=.*\n", "", text, flags=re.M)
    path = tmp_path / f"alas-shg-{ecut}.in"
    path.write_text(text)
    calculator = Calculator.from_file(path, pseudo_dir=pseudo_dir, announce=False)
    return calculator, calculator.get_scf(conv_thr=1e-12)


def _discriminates(value, reference, tolerance):
    """``value = -conj(reference)`` to ``tolerance``, and neither ``conj`` nor the bare value."""
    assert abs(reference.imag) > 0.3 * abs(reference), "Im must be large enough to decide"
    assert abs(value + np.conj(reference)) < tolerance * abs(value)
    assert abs(value - np.conj(reference)) > 1.5 * abs(value)
    assert abs(value - reference) > 0.5 * abs(value)


def test_the_sum_over_states_is_the_velocity_gauge_when_the_hamiltonian_is_local(
        pseudo_dir, tmp_path):
    """The dense hierarchy against every band of get_shg's assembly, projectors removed.

    ``D_ij = 0`` and the local potential tripled, which keeps a gap of 2.72 eV on
    the mesh at 8 Ry, so that ``d^2 H/dk^2`` is the free-electron one exactly and
    the sum rule the assembly is written in holds. At 1.5 eV and ``eta = 0.2`` eV,
    ``chi_xyz`` reads -32.5294 - 48.5278i from the hierarchy and -32.5272 - 48.5266i
    as minus the conjugate of the sum over states, **4.3e-5** apart (2.2e-5 and
    2.5e-5 at 0.4 and 0.7 eV), against 5.0 to 8.6 per cent with the projectors in.
    Measured 46 s with its SCF on two D22 cores.
    """
    calculator, scf = _calculator(pseudo_dir, tmp_path, 8.0)
    system = eqx.tree_at(lambda s: s.kpoints, calculator.system,
                         KPoints.automatic(GRID, (0, 0, 0), calculator.system.cell))
    calc = Calculation(system, calculator.pseudos)
    v_scf = calc.potential(jnp.asarray(scf.density)).v_scf
    terms = calc.local_terms(v_scf)
    dij = calc.projectors.dij
    terms = eqx.tree_at(lambda t: t.deeq, terms,
                        jnp.zeros((calc.nspin,) + dij.shape, dij.dtype),
                        is_leaf=lambda x: x is None)
    terms = eqx.tree_at(lambda t: (t.potentials, t.waves), terms,
                        (tuple(3.0 * p for p in terms.potentials),
                         tuple(3.0 * p for p in terms.waves)))
    nocc = int(round(calc.nelec / 2))
    volume = float(calc.system.cell.volume)
    weights = np.asarray(calc.system.kpoints.weights)
    npwx = int(np.asarray(calc.basis.planewaves.mask).shape[1])
    frequency, broadening = 1.5 / HARTREE_TO_EV, 0.2 / HARTREE_TO_EV
    sos = jax.jit(lambda e, v, f, w: _chi_at_k(
        e, e, v, f, w, jnp.asarray([2.0 * frequency]), 2.0 * broadening,
        2.0 * broadening, 1e-8).sum(axis=0)[0, 0, 1, 2])
    current, chi_sos, gap = 0.0, 0.0, np.inf
    for ik in range(len(weights)):
        h = dense_hamiltonians(calc, terms, ik, DIRECTION, 3)
        e, u = np.linalg.eigh(h[0])
        gap = min(gap, e[nocc] - e[nocc - 1])
        current += -dense_orders(h, e[:nocc], u[:, :nocc].T, np.full(nocc, weights[ik]),
                                 2.0 * frequency, 2.0 * broadening, nmax=2)[(2, 2)]
        # every band of the sphere, padded to npwx with inert bands far above
        m = len(e)
        velocity = np.zeros((3, npwx, npwx), dtype=complex)
        for a in range(3):
            velocity[a, :m, :m] = u.conj().T @ dense_hamiltonians(
                calc, terms, ik, np.eye(3)[a], 1)[1] @ u
        energies = np.concatenate([e, 1.0e4 + np.arange(npwx - m)])
        filling = (np.arange(npwx) < nocc).astype(float)
        chi_sos += complex(sos(jnp.asarray(energies), jnp.asarray(velocity),
                               jnp.asarray(filling), float(weights[ik])))
    assert gap * RY_TO_EV > 1.0, "the local model must keep a gap at every k-point"
    z = frequency + 1j * broadening
    dense = -2j * (current / (2.0 * volume)) / z**3 * TO_XYZ
    _discriminates(dense, chi_sos * 4.0 / volume, 1e-4)


def test_the_second_order_current_is_get_shg_up_to_its_conventions(pseudo_dir, tmp_path):
    """``chi_RT = -conj(chi_shg)`` to 2 per cent at 23 bands, near the two-photon resonance.

    8 Ry, the whole 4x4x4 mesh (20 points of the field's little group for the
    propagation), 1.5 eV against a direct gap of 3.40 eV on the mesh, ``eta = 0.3``
    eV, ``eta_t = 12``, 480 steps a period (``dt = 0.2375``, 4800 steps). Measured:
    the propagation gives ``chi_xyz`` = 145.1967 + 119.0447i pm/V, the dense
    hierarchy 145.2015 + 119.0707i (1.4e-4), and ``get_shg`` at 23 bands
    -141.9804 + 117.3862i, so **1.93e-2** for minus its conjugate, 1.98 for its
    conjugate and 1.53 for the value itself. Every band of the sphere gives
    155.3600 + 122.4520i and 5.7 per cent, which is the projectors' term of the
    module docstring; the 23-band agreement is the truncation cancelling part of it.
    Measured 388 s on two D22 cores, 1.8 GiB resident for the file.
    """
    calculator, _ = _calculator(pseudo_dir, tmp_path, 8.0)
    frequency, broadening = 1.5, 0.3  # eV
    orders = calculator.get_harmonic_orders(
        frequency, broadening=broadening, direction=tuple(DIRECTION), order=2,
        eta_t=12.0, steps_per_period=480, grid=GRID, conv_thr=1e-12)
    along = chi2_from_orders(orders, axis=DIRECTION) * TO_XYZ * CHI2_AU_TO_PM_PER_V
    across = chi2_from_orders(orders, axis=0) * 1.5 * CHI2_AU_TO_PM_PER_V
    shg = calculator.get_shg(
        kpoints=KPoints.automatic(GRID, (0, 0, 0), calculator.system.cell),
        nbnd=23, frequencies=np.array([frequency / RY_TO_EV]),
        broadening=broadening / RY_TO_EV, conv_thr=1e-12)
    # the x component of the same current gives the same tensor element: the
    # little group's average has made the current parallel to [111]
    assert abs(across - along) < 1e-8 * abs(along)
    _discriminates(along, complex(shg.chi[0, 0, 1, 2]), 4e-2)
