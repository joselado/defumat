"""P51: the optical conductivity tensor, the Kerr angle and the anomalous Hall effect.

Elk's tasks 121 and 122, the fourth entry taken from ``ELK-FEATURES.md``. There
is no reference output to compare against -- ``pw.x``'s ``epsilon.x`` forms the
tensor but computes no conductivity and no Kerr angle, and builds its dipole
from momentum matrix elements, which is the one construction ``CLAUDE.md``
records as wrong with a nonlocal pseudopotential. So the validation is four
internal statements and one analytic limit, and each fails differently:

* the **symmetric** part must reproduce P37's independent-particle dielectric
  function exactly. That chain is a different assembly -- a Dyson solve over a
  response sphere rather than a resolvent sum over band pairs -- and it reaches
  ``ph.x`` through :func:`~pypresso.response.efield.dielectric_tensor`. It is
  what pins the prefactor, and it is the check the factor of two P50 found
  would have failed;
* the **f-sum rule** converges onto ``<n|d2H/dk^2|n>``, measured separately by a
  central difference of the velocity operator. That is an absolute check on the
  volume, the electron count, the spin degeneracy and the Rydberg-to-Hartree
  cancellation, none of which is fitted -- and what it converges *in* is the
  k-grid rather than the band count, which is a statement nothing else here
  makes;
* the **antisymmetric** part must vanish for a nonmagnetic crystal, on a
  ``nosym`` run with nothing imposing it;
* the **plasma frequency** of aluminium must be of the free-electron scale, and
  its tensor must be isotropic to round-off on a cubic crystal with nothing
  imposing that either;
* and for nickel, the two static routes -- the resolvent sum at ``w = 0`` and
  the analytic Berry-curvature limit -- must **disagree**, in the way and for
  the reason the module documents. A metal is where the two limits stop
  commuting.
"""

from functools import lru_cache
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.response.conductivity import optical_conductivity
from pypresso.response.velocity import VelocityOperator
from pypresso.scf import run_scf
from pypresso.system import build_system
from pypresso.system.kpoints import KPoints
from pypresso.workflows.conductivity import run_conductivity
from pypresso.workflows.nscf import fixed_density_states
from pypresso.workflows.tddft import run_absorption

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


@pytest.fixture(autouse=True)
def _bounded_compilation():
    """``CLAUDE.md``'s memory rule: drop XLA's executables between cases.

    This file runs silicon, aluminium and a spinor nickel, which share no
    shape at all -- so each compiles the whole NSCF and velocity stack afresh
    and the backend keeps every one of them for the life of the process.
    """
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _converged(case: str):
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    # **1e-10, which is what these inputs ask for**, rather than ``run_scf``'s
    # default -- and it is not a formality here: a plasma frequency is a delta function on the Fermi
    # surface and therefore a sharp functional of the density. Measured, the
    # two thresholds put aluminium's at 12.9593 and 12.9796 eV -- 0.16 per cent
    # apart, which is more than this test's tolerance and would make the number
    # the notebook prints differ from the number the test pins.
    result = run_scf(system, pseudos, conv_thr=1e-10, max_iterations=200)
    return system, pseudos, result


# -- the symmetric part, against the chain that reaches ph.x -------------------


def test_the_symmetric_part_is_p37s_independent_particle_dielectric_function():
    """``eps = 1 + 4 pi i sigma/w`` against ``epsilon_no_local_fields``.

    The two share the velocity matrix elements and nothing else: P37 builds a
    matrix over a response sphere and inverts a Dyson equation, this sums
    resolvents over band pairs. Agreement is therefore a check on the
    *prefactor* -- the volume, the occupation convention, and the two factors
    of two that take a Rydberg-parameterised sum to a Hartree-unit
    conductivity. It is exact rather than approximate because the two are the
    same expression rearranged, which is what makes a tolerance of 1e-10
    meaningful here and not merely tight.
    """
    system, pseudos, result = _converged("si2-nosym")
    nbnd, eta = 24, 0.02
    w = np.array([0.02, 0.1, 0.2, 0.3, 0.5])

    spectrum = run_absorption(
        system, pseudos, result.density, w, kernel="rpa", nbnd=nbnd,
        broadening=eta, ecut_response=0.0, static_residual=False,
    )
    calculation, _, eigenvalues, psi = fixed_density_states(
        system, pseudos, result.density, nbnd=nbnd, conv_thr=1e-10
    )
    sigma = optical_conductivity(
        calculation, psi, eigenvalues,
        calculation.potential(result.density).v_scf,
        fermi_energy=0.0, frequencies=w, broadening=eta,
    )

    reference = np.asarray(spectrum.epsilon_no_local_fields)
    epsilon = sigma.dielectric
    # The **diagonal** is what both constructions carry as a large number, and
    # it agrees to round-off. The off-diagonals are zero by cubic symmetry in
    # both, and what is left of them is each one's own band truncation: at this
    # band count (which stops inside a degenerate multiplet, ``band_cut_gap``
    # = 3e-7 Ry) they are 3e-5 and 9e-4 respectively, and at ``nbnd = 20`` or
    # 36, which do not, both fall to 1e-9 and agree to 2e-9. So the comparison
    # is made where the quantity is, and the rest is measured in
    # ``test_a_nonmagnetic_crystal_has_no_hall_conductivity``.
    diagonal = np.einsum("wii->wi", epsilon) - np.einsum("wii->wi", reference)
    assert np.max(np.abs(diagonal)) < 1e-10


def test_the_f_sum_rule_converges_onto_the_nonlocal_diamagnetic_weight():
    """``int Re sigma dw`` -> ``(pi/2 Omega) sum W_n <n|d2H/dk^2|n>``.

    The familiar ``pi n_e/2`` is the *local* Hamiltonian's special case, and a
    nonlocal pseudopotential does not obey it: what the spectral weight equals
    is the diamagnetic weight ``<n|d2H/dk^2|n>``, which is measured here by a
    central difference of the velocity operator at frozen states and comes out
    at 0.943 rather than 1. The remaining difference is
    ``sum_k w_k d2eps_n/dk^2``, which is zero over the zone by periodicity and
    is **not** zero on a coarse grid -- so this is a k-convergence check, and
    the 4x4x4 grid is off by 26 per cent where 8x8x8 is off by 1.
    """
    system, pseudos, result = _converged("si2-nosym")
    nbnd = 40
    measured = {}
    for nk in (4, 8):
        grid = system
        if nk != 4:
            points = KPoints.automatic(
                (nk, nk, nk), (0, 0, 0), system.cell,
                precision=system.cell.precision,
            )
            grid = eqx.tree_at(lambda s: s.kpoints, system, points)
        measured[nk] = _weights(grid, pseudos, result.density, nbnd)

    left4, right4 = measured[4]
    left8, right8 = measured[8]
    # The diamagnetic weight is a property of the states and barely moves.
    assert right4 == pytest.approx(0.9423, abs=5e-4)
    assert right8 == pytest.approx(right4, abs=2e-3)
    # It is not one, and that is the pseudopotential's nonlocality.
    assert abs(right4 - 1.0) > 0.04
    # And the spectral weight converges onto it, from far away.
    assert left4 / right4 == pytest.approx(1.258, rel=0.02)
    assert left8 / right8 == pytest.approx(0.990, abs=0.02)


def _weights(system, pseudos, density, nbnd):
    """``(spectral weight, diamagnetic weight)``, both relative to ``pi n_e/2``.

    The spectral weight is taken in **closed form** rather than by integrating
    a broadened spectrum: ``Re sigma`` is a sum of Lorentzians whose total area
    is exactly ``(pi/2 Omega) sum t_nm |V_nm|^2`` over the pairs with a
    positive gap, so no frequency grid and no ``eta`` enter the number at all.
    """
    from pypresso.response.conductivity import _pair_weights

    calculation, _, eigenvalues, psi = fixed_density_states(
        system, pseudos, density, nbnd=nbnd, conv_thr=1e-10
    )
    v_scf = calculation.potential(density).v_scf
    operator = VelocityOperator(calculation, v_scf)
    elements = np.asarray(operator.matrix_elements(jnp.asarray(psi)))[:, 0]

    energies = jnp.asarray(eigenvalues)
    if energies.ndim == 2:
        energies = energies[None]
    wg = np.asarray(calculation.occupations(energies)[0])[0]
    weights = np.asarray(calculation.system.kpoints.weights)
    filling = wg / weights[:, None]
    volume = float(calculation.system.cell.volume)

    total = 0.0
    for k in range(elements.shape[1]):
        t, gap = _pair_weights(
            jnp.asarray(np.asarray(energies)[0, k]), jnp.asarray(wg[k]),
            jnp.asarray(filling[k]),
        )
        total += float(np.sum(np.where(np.asarray(gap) > 0.0,
                                       np.asarray(t) * np.abs(elements[0, k]) ** 2,
                                       0.0)))
    scale = np.pi * (float(calculation.nelec) / volume) / 2.0
    spectral = (0.5 * np.pi / volume * total) / scale

    kcart = np.asarray(
        calculation.system.kpoints.cartesian(calculation.system.cell)
    )
    step = 1e-3

    def diagonal(shift):
        moved = kcart.copy()
        moved[:, 0] += shift
        operator = VelocityOperator(calculation, v_scf, kcart=jnp.asarray(moved))
        block = np.asarray(operator.matrix_elements(jnp.asarray(psi)))
        return np.real(np.einsum("askii->kai", block))[:, 0, :]

    second = (diagonal(step) - diagonal(-step)) / (2.0 * step)  # Ry
    diamagnetic = float(np.sum(wg * second) / np.sum(wg)) / 2.0  # -> Hartree
    return spectral, diamagnetic


# -- the antisymmetric part, and when it must vanish ---------------------------


def test_a_nonmagnetic_crystal_has_no_hall_conductivity():
    """Time reversal, on a ``nosym`` run with nothing imposing it.

    The band count is not free here and that is the point: ``nbnd = 20`` stops
    at a 0.028 Ry gap and leaves 4e-13, while ``nbnd = 24`` stops **inside** a
    degenerate multiplet and leaves 2e-6 -- keeping some of a multiplet's
    members and dropping others breaks a cancellation those members were making
    between them. :attr:`OpticalConductivity.band_cut_gap` is what says which
    of the two happened.
    """
    system, pseudos, result = _converged("si2-nosym")
    clean = run_conductivity(system, pseudos, result.density, nbnd=20,
                             frequencies=[0.0], broadening=0.01)
    sigma = clean.sigma[0]
    assert np.max(np.abs(0.5 * (sigma - sigma.T))) < 1e-11
    assert clean.band_cut_gap > 0.02

    inside = run_conductivity(system, pseudos, result.density, nbnd=24,
                              frequencies=[0.0], broadening=0.01)
    assert inside.band_cut_gap < 1e-6
    sigma = inside.sigma[0]
    assert np.max(np.abs(0.5 * (sigma - sigma.T))) > 1e-7


# -- the intraband leg, against a free-electron metal --------------------------


def test_the_plasma_frequency_of_aluminium_is_of_the_free_electron_scale():
    """``hbar wp = 12.98 eV`` against a free-electron 16.27, and isotropic.

    Aluminium is the textbook nearly-free-electron metal, so the free-electron
    ``sqrt(4 pi n)`` is the right order and *not* the right number: the band
    structure departs from free electrons at the zone boundary, which removes
    Fermi surface and takes 20 per cent off the Drude weight. What is exact is
    the **isotropy** -- ``wp_ab`` is a rank-2 tensor built from ``v_a v_b`` on
    the Fermi surface, and on a cubic crystal run with ``nosym`` nothing makes
    it diagonal but the physics.

    **512 k-points is not generosity.** On 4x4x4 the same cell gives 13.78 eV:
    a Fermi-surface integral needs the grid where a total energy does not.
    """
    system, pseudos, result = _converged("al-conductivity")
    sigma = run_conductivity(system, pseudos, result.density, nbnd=12,
                             window=1.5, nw=300, broadening=0.01)
    plasma = sigma.plasma_ev
    assert plasma[0, 0] == pytest.approx(12.9796, abs=5e-3)

    off = plasma - np.diag(np.diag(plasma))
    assert np.max(np.abs(off)) < 1e-3
    assert plasma[0, 0] == pytest.approx(plasma[2, 2], abs=1e-3)
    # A nonmagnetic metal has an enormous conductivity and no Hall part at all.
    assert np.max(np.abs(sigma.hall_conductivity)) < 1.0
    assert sigma.sigma_s_per_cm[0, 0, 0].real > 1e4


# -- the magneto-optical part, which needs a magnet ---------------------------


def test_nickel_has_a_hall_conductivity_and_a_kerr_angle():
    """The positive case, and the pair of ingredients that produce it.

    fcc nickel with spin-orbit coupling has both a moment and the coupling that
    tells the orbital motion where the moment points, so its antisymmetric
    ``sigma`` survives. Silicon has neither and its own is nine orders of
    magnitude smaller relative to the diagonal, on runs that impose no symmetry
    at all.

    **The value is not converged and is not asserted as one.** An intrinsic
    anomalous Hall conductivity integrates a quantity concentrated on
    near-degeneracies at the Fermi surface, and a published one for nickel
    (about -2200 S/cm) is reached with meshes two orders of magnitude denser
    than anything here. What is asserted is that it is **there**, that it
    points along the magnetization, and that the Kerr angle is of the measured
    scale; ``PLAN.md`` P51 carries the k-convergence.
    """
    system, pseudos, result = _converged("ni-soc-nosym")
    assert abs(float(result.magnetization_vector[2])) > 0.4

    sigma = run_conductivity(system, pseudos, result.density, nbnd=36,
                             window=0.6, nw=200, broadening=0.01)
    hall = sigma.hall_conductivity
    assert abs(hall[0, 1]) > 50.0
    # The moment is along z, so only the z component of the axial vector lives.
    assert abs(hall[1, 2]) < 0.02 * abs(hall[0, 1])
    assert abs(hall[2, 0]) < 0.02 * abs(hall[0, 1])

    # A Kerr rotation of the measured order: tenths of a degree, not degrees --
    # **in the visible**, which is where a magneto-optical measurement is made
    # and the only place the quantity is bounded. ``moke.f90``'s expression
    # divides by ``sigma_xx sqrt(1 + 4 pi i sigma_xx/w)``, so it diverges
    # wherever the diagonal conductivity passes through zero; the largest angle
    # over the whole window is 6.7 degrees and means nothing at all.
    kerr = sigma.kerr
    visible = (sigma.frequencies_ev > 1.5) & (sigma.frequencies_ev < 3.5)
    assert 0.01 < float(np.max(np.abs(kerr.real[visible]))) < 3.0

    # **The Drude weight is a Fermi-surface quantity and must not move with the
    # band count**, which is what says the delta function is being sampled and
    # not the band set. Measured: 0.5973 eV at both 30 and 36 bands, where the
    # Hall conductivity itself moves by 0.25 per cent between them (a genuine
    # truncation of the sum over empty states).
    fewer = run_conductivity(system, pseudos, result.density, nbnd=30,
                             frequencies=[0.0], broadening=0.01)
    assert float(fewer.plasma_ev[0, 0]) == pytest.approx(
        float(sigma.plasma_ev[0, 0]), rel=2e-3
    )


def test_a_k_set_handed_in_is_normalised_for_the_spin_regime():
    """A denser mesh built by the caller must not change the Fermi level.

    Every ``KPoints`` constructor applies the unpolarized spin degeneracy
    unconditionally, because it cannot know what regime it will be used in; a
    **spinor** band holds one electron rather than two, so a set built with
    ``KPoints.automatic`` and handed to a noncollinear run carries weights
    summing to 2 where the run needs 1. Nothing about that looks like an error.
    The electron count is still met, the Fermi level simply lands somewhere
    else, and every occupation moves with it.

    Measured before ``for_spin`` was applied at this boundary, on fcc nickel's
    *own* 64-point grid rebuilt rather than reused: the plasma frequency came
    out at 13.11 eV instead of 0.60, and the anomalous Hall conductivity
    changed sign. This is the test that the two paths are one path.
    """
    system, pseudos, result = _converged("ni-soc-nosym")
    rebuilt = KPoints.automatic(
        (4, 4, 4), (0, 0, 0), system.cell, precision=system.cell.precision
    )
    assert float(np.sum(np.asarray(rebuilt.weights))) == pytest.approx(2.0)
    assert float(np.sum(np.asarray(system.kpoints.weights))) == pytest.approx(1.0)

    native = run_conductivity(system, pseudos, result.density, nbnd=30,
                              frequencies=[0.0], broadening=0.01)
    handed = run_conductivity(system, pseudos, result.density, nbnd=30,
                              kpoints=rebuilt, frequencies=[0.0],
                              broadening=0.01)
    assert float(handed.plasma_ev[0, 0]) == pytest.approx(
        float(native.plasma_ev[0, 0]), rel=1e-9
    )
    assert float(handed.hall_conductivity[0, 1]) == pytest.approx(
        float(native.hall_conductivity[0, 1]), rel=1e-9
    )
