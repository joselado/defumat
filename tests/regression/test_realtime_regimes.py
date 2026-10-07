"""The harmonic routes on collinear, spinor, ultrasoft and PAW ground states.

Each comparison is an identity rather than an agreement:

* **the first order against the Kubo sum** with the generalised velocity of
  ``PLAN.md`` P99 and every eigenstate of the dense ``H(k)``, ``S(k)``, plus the
  band curvature the sum replaces by its f-sum value (the norm-conserving
  check of ``test_realtime.py``), on ultrasoft AlAs, ultrasoft AlAs with
  spin-orbit coupling and trigonal selenium with it: an independent code, which
  sees the augmented equation of motion and its current, since every operator
  of the nonlinear problem enters at first order as a function of k;
* **the frequency-domain hierarchy against the real-time orders** at one
  frequency, orders one to three, on a collinear magnet, ultrasoft and PAW AlAs
  and selenium: two solvers, the hierarchy's expansion of the equation of
  motion in the field against the equation itself;
* **a collinear magnet written as spinors** against itself in channels: the
  frozen hierarchy and the first order with the induced potential, Hartree and
  adiabatic exchange-correlation, where the spinor route carries the
  magnetization as a 2x2 field and the collinear one as two channels.

``PLAN.md`` P139 to P141 have the measurements.
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
import scipy.linalg as sla

from defumat import Calculator
from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.realtime.orders import propagate_orders
from defumat.realtime.pulse import Kick
from defumat.realtime.spectra import conductivity_from_kick
from defumat.scf import Calculation
from defumat.system import build_system
from defumat.system.kpoints import KPoints, for_spin

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
#: Gamma and one point of no symmetry.
KSET = (np.array([[0.0, 0.0, 0.0], [0.25, 0.1, -0.05]]), np.array([0.5, 0.5]))
COMPONENTS = ((1, 1), (2, 2), (2, 0), (3, 3), (3, 1))


@pytest.fixture(autouse=True)
def _bounded_compilation():
    yield
    jax.clear_caches()


def _input(tmp_path, case, ecut, ecutrho=None, extra="", drop=(), grid=None):
    text = (CASES / f"{case}.in").read_text()
    text = re.sub(r"ecutwfc\s*=\s*[0-9.dD+-]+", f"ecutwfc = {ecut}", text)
    if ecutrho is not None:
        text = re.sub(r"ecutrho\s*=\s*[0-9.dD+-]+", f"ecutrho = {ecutrho}", text)
    for key in drop:
        text = re.sub(rf"^\s*{key}\s*=.*$", "", text, flags=re.M)
    if extra:
        text = re.sub(r"&system", "&system\n    " + extra, text, count=1)
    if grid is not None:
        text = re.sub(r"K_POINTS.*\n\s*[0-9 ]+\n", f"K_POINTS (automatic)\n {grid}\n", text)
    path = tmp_path / f"{case}-{ecut}-{abs(hash(extra)) % 1000}.in"
    path.write_text(text)
    return path


@pytest.mark.parametrize("case, ecut, ecutrho, tolerance", [
    ("alas-epsilon-us", 10.0, 40.0, 5e-5),
    ("alas-epsilon-us-soc", 10.0, 40.0, 5e-5),
    ("se-trigonal-soc", 12.0, None, 5e-5),
])
def test_the_first_order_is_the_kubo_sum_plus_the_band_curvature(pseudo_dir, tmp_path, case,
                                                                 ecut, ecutrho, tolerance):
    """``sigma_RT(z) = sigma_Kubo(z) + i D/(Omega z)``, the generalised velocity in the Kubo sum.

    At the superposition of atomic charges on :data:`KSET`, ``eta = 0.02``
    Hartree, ``dt = 0.05``, the first order of a kick through the step in
    ``kappa`` the augmented states cross. The dense generalised eigenproblem
    gives every state of the sphere to the resolvent sum, whose velocity is
    :meth:`~defumat.response.velocity.VelocityOperator.generalised_matrix_elements`,
    and ``D`` is a second difference of the occupied generalised eigenvalues.
    """
    from defumat.response.conductivity import _resolvent_sum
    from defumat.response.velocity import VelocityOperator

    system = build_system(read_pw_input(_input(tmp_path, case, ecut, ecutrho)))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    system = eqx.tree_at(lambda s: s.kpoints, system,
                         for_spin(KPoints(coords=KSET[0], weights=KSET[1]), system.nspin))
    calc = Calculation(system, pseudos)
    v_scf = calc.potential(calc.starting_density()).v_scf
    terms = calc.local_terms(v_scf)
    weights = np.asarray(calc.system.kpoints.weights)
    nk, volume = len(weights), float(calc.system.cell.volume)
    nocc = int(round(calc.nelec / (1 if calc.noncolin else 2)))
    eta, direction = 0.02, np.array([1.0, 0.0, 0.0])
    frequencies = np.linspace(0.02, 0.6, 30)
    kcart = np.asarray(calc.system.kpoints.cartesian(calc.system.cell))

    def dense(kc):
        ham = calc.at_kcart(jnp.asarray(kc)).hamiltonian_from(terms)[0]
        return [(np.asarray(ham.matrix(ik)), np.asarray(ham.overlap_matrix(ik)))
                for ik in range(nk)], np.asarray(ham.state_mask)

    mats, mask = dense(kcart)
    nb = min(int(m.sum()) for m in mask)
    psi = np.zeros((nk, nb, mask.shape[1]), dtype=complex)
    eps = np.zeros((nk, nb))
    for ik in range(nk):
        keep = np.flatnonzero(mask[ik])
        h, s = mats[ik]
        e, v = sla.eigh(h[np.ix_(keep, keep)], s[np.ix_(keep, keep)])
        psi[ik][:, keep] = v[:, :nb].T
        eps[ik] = e[:nb]
    elements = np.asarray(VelocityOperator(calc, v_scf).generalised_matrix_elements(
        jnp.asarray(psi)[None], jnp.asarray(eps)[None]))[:, 0]
    sigma_kubo = np.zeros(len(frequencies), dtype=complex)
    curvature, step = 0.0, 3e-4
    for ik in range(nk):
        wg = np.where(np.arange(nb) < nocc, weights[ik], 0.0)
        filling = np.where(np.arange(nb) < nocc, 1.0, 0.0)
        value, _ = _resolvent_sum(jnp.asarray(elements[:, ik]), jnp.asarray(eps[ik]),
                                  jnp.asarray(wg), jnp.asarray(filling),
                                  jnp.asarray(2.0 * (frequencies + 1j * eta)), 1e-8)
        sigma_kubo += np.asarray(value)[:, 0, 0] / volume
        sums = []
        for x in (-step, 0.0, step):
            moved, _ = dense(kcart + x * direction[None, :])
            keep = np.flatnonzero(mask[ik])
            h, s = moved[ik]
            sums.append(np.sum(sla.eigh(h[np.ix_(keep, keep)], s[np.ix_(keep, keep)],
                                        eigvals_only=True)[:nocc]))
        curvature += weights[ik] * (sums[0] - 2 * sums[1] + sums[2]) / step**2
    z = frequencies + 1j * eta
    reference = sigma_kubo + 1j * curvature / (volume * 2.0 * z)
    result = propagate_orders(
        calc, psi[:, :nocc], np.repeat(weights[:, None], nocc, axis=1), v_scf,
        Kick(strength=1.0, direction=tuple(direction)), dt=0.05, order=1, start=0.0,
        duration=22.0 / eta, k_batch=None, block_steps=1000)
    sigma = conductivity_from_kick(result.times, result.currents[1], 1.0, direction,
                                   frequencies, eta).sigma[:, 0]
    scale = np.abs(reference).max()
    assert np.abs(sigma - reference).max() < tolerance * scale


@pytest.mark.parametrize("variant", ["mag", "us", "paw", "se"])
def test_the_spectrum_is_the_propagation_at_one_frequency(pseudo_dir, tmp_path, variant):
    """The hierarchy's ``J_(n,m)`` against ``get_harmonic_orders``, 1.5 eV, ``eta = 0.3`` eV, [111].

    ``test_realtime_hierarchy.py``'s comparison, which measured 3.5e-5 to 1.9e-4
    on norm-conserving AlAs at 400 steps a period, on a collinear magnet (AlAs
    at ``tot_magnetization = 2``), ultrasoft and PAW AlAs, and trigonal
    selenium with spin-orbit coupling.
    """
    cells = {
        "mag": ("alas-shg", 6.0, None, "nspin = 2, tot_magnetization = 2, "
                "starting_magnetization(1) = 0.3,", (), "2 2 2 0 0 0"),
        "us": ("alas-epsilon-us", 10.0, 40.0, "", (), None),
        "paw": ("alas-piezo-tiny-paw", 10.0, 44.0, "", ("nosym",), None),
        "se": ("se-trigonal-soc", 12.0, None, "", (), None),
    }
    case, ecut, ecutrho, extra, drop, grid = cells[variant]
    # an augmented dataset's generalised spectrum reaches 23 Ry at 10 Ry, and the
    # step at 400 a period is just past the propagator's bound there
    steps = 500 if variant in ("us", "paw") else 400
    path = _input(tmp_path, case, ecut, ecutrho, extra, drop, grid)
    calculator = Calculator.from_file(path, pseudo_dir=pseudo_dir, announce=False)
    calculator.get_scf(conv_thr=1e-12)
    options = dict(broadening=0.3, direction=tuple(np.ones(3) / math.sqrt(3.0)),
                   grid=(2, 2, 2), conv_thr=1e-12)
    spectrum = calculator.get_nonlinear_spectrum([1.5], order=3, **options)
    orders = calculator.get_harmonic_orders(1.5, order=3, eta_t=12.0, steps_per_period=steps,
                                            **options)
    references = {key: complex(orders.component(*key, axis=0)) for key in COMPONENTS}
    for key in COMPONENTS:
        value = spectrum.component(*key, axis=0)[0]
        scale = max(abs(v) for k, v in references.items() if k[0] == key[0])
        assert abs(value - references[key]) < 5e-4 * scale, (variant, key, value,
                                                             references[key])


def test_a_collinear_magnet_written_as_spinors_is_itself(pseudo_dir, tmp_path):
    """The spinor routes against the collinear ones on AlAs with ``tot_magnetization = 2``.

    Five electrons up and three down at 6 Ry on the whole 2x2x2 mesh; the
    collinear states of each channel are made the spinors ``(u, 0)`` and
    ``(0, u)``, and the noncollinear potential is built from the density
    ``(n, 0, 0, m)``, whose LSDA ``v0 +- B_z`` are the channels' to 7e-16. At
    1.5 eV, ``eta = 0.3`` eV, [111]: the frozen hierarchy, every component of
    orders one to three, measured 1.1e-12 to 6.1e-12, and its first order with
    the induced potential 3.2e-12 (Hartree) and 2.5e-12 (adiabatic LSDA), where
    the update moves the current by 2.4 and 3.4 per cent.
    """
    from defumat.realtime.hierarchy import hierarchy_linear_self_consistent, hierarchy_orders
    from defumat.realtime.pulse import EV_TO_HA
    from defumat.workflows.realtime import _kset, _solved_states

    collinear = Calculator.from_file(
        _input(tmp_path, "alas-shg", 6.0, extra="nspin = 2, tot_magnetization = 2, "
               "starting_magnetization(1) = 0.3,", grid="2 2 2 0 0 0"),
        pseudo_dir=pseudo_dir, announce=False)
    rho = np.asarray(collinear.get_scf(conv_thr=1e-12).density)
    direction = np.ones(3) / math.sqrt(3.0)
    kset, _, _ = _kset(collinear.system, Kick(strength=1.0, direction=tuple(direction)), None,
                       (2, 2, 2), False)
    calc_c, _, w_c, v_c, bands, energies, _ = _solved_states(
        collinear.system, collinear.pseudos, rho, kset, nbnd=16, conv_thr=1e-12,
        k_batch=None, calculation=None)
    spinor = Calculator.from_file(
        _input(tmp_path, "alas-shg", 6.0, extra="noncolin = .true., "
               "starting_magnetization(1) = 0.3, angle1(1) = 0.0, angle2(1) = 0.0,",
               grid="2 2 2 0 0 0"), pseudo_dir=pseudo_dir, announce=False)
    calc_n = Calculation(eqx.tree_at(lambda s: s.kpoints, spinor.system,
                                     for_spin(kset, spinor.system.nspin)), spinor.pseudos)
    rho_n = np.zeros((4,) + rho.shape[1:])
    rho_n[0], rho_n[3] = rho[0] + rho[1], rho[0] - rho[1]
    v_n = np.asarray(calc_n.potential(jnp.asarray(rho_n)).v_scf)
    v_c = np.asarray(v_c)
    assert np.abs(v_n[0] + v_n[3] - v_c[0]).max() < 1e-12
    assert np.abs(v_n[0] - v_n[3] - v_c[1]).max() < 1e-12

    def spinors(up, down):
        return np.concatenate([np.concatenate([up, np.zeros_like(up)], axis=-1),
                               np.concatenate([np.zeros_like(down), down], axis=-1)], axis=1)

    up, down = (int(np.flatnonzero(np.any(w_c[c] > 0, axis=0))[-1]) + 1 for c in (0, 1))
    computed = 12
    basis_n = np.concatenate([spinors(bands[0][:, :up], bands[1][:, :down]),
                              spinors(bands[0][:, up:computed], bands[1][:, down:computed])],
                             axis=1)
    energies_n = np.concatenate([energies[0][:, :up], energies[1][:, :down],
                                 energies[0][:, up:computed], energies[1][:, down:computed]],
                                axis=1)
    w_n = np.concatenate([w_c[0][:, :up], w_c[1][:, :down]], axis=1)
    common = dict(omegas=(1.5 * EV_TO_HA,), eta=0.3 * EV_TO_HA, direction=direction)
    a = hierarchy_orders(calc_c, bands[:, :, :computed], energies[:, :, :computed], w_c, v_c,
                         order=3, **common)
    b = hierarchy_orders(calc_n, basis_n, energies_n, w_n, jnp.asarray(v_n), order=3, **common)
    for key in COMPONENTS:
        x, y = a["components"][key][0], b["components"][key][0]
        assert np.abs(x - y).max() < 1e-9 * np.abs(x).max(), (key, x, y)
    for potential in ("hartree", "hxc"):
        a = hierarchy_linear_self_consistent(calc_c, bands[:, :, :computed],
                                             energies[:, :, :computed], w_c, v_c,
                                             potential=potential, **common)
        b = hierarchy_linear_self_consistent(calc_n, basis_n, energies_n, w_n,
                                             jnp.asarray(v_n), potential=potential, **common)
        x, y = a["components"][(1, 1)][0], b["components"][(1, 1)][0]
        frozen = a["frozen"][(1, 1)][0]
        assert np.abs(x - y).max() < 1e-9 * np.abs(x).max(), (potential, x, y)
        assert np.abs(x - frozen).max() > 1e-2 * np.abs(x).max(), "the update is load-bearing"
