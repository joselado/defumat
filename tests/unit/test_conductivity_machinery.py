"""The parts of the optical conductivity that need no self-consistent field.

Three kinds of thing live here, and each is silent when wrong. The **unit
conversion**, because a conductivity assembled in Rydberg quantities comes out
in *Hartree* atomic units and the cancellation that makes that true is a
coincidence of the expression's homogeneity rather than a rule (``PLAN.md``
P50's factor of two is the warning). The **post-processing** -- the dielectric
tensor and the Kerr angle -- which are closed-form functions of ``sigma`` and
can therefore be checked against the closed form. And the **refusals**, which
are the promise that a run which starts is a run whose physics is there.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pypresso.io.pwin import read_pw_input
from pypresso.response.conductivity import (
    OpticalConductivity,
    require_a_conductivity_regime,
)
from pypresso.system import build_system
from pypresso.units import AU_TO_S_PER_CM, FPI

pytestmark = [pytest.mark.unit]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"


def _tensor(sigma, frequencies, *, broadening=0.01, nelec=8.0, volume=270.0):
    """An :class:`OpticalConductivity` built by hand, to test what it derives."""
    sigma = np.asarray(sigma, dtype=complex)
    return OpticalConductivity(
        frequencies=np.asarray(frequencies, dtype=float),
        sigma=sigma,
        interband=sigma,
        intraband=np.zeros_like(sigma),
        plasma=np.zeros((3, 3)),
        volume=volume,
        broadening=broadening,
        relaxation=broadening,
        nbnd=16,
        nelec=nelec,
    )


# -- the unit that a conductivity is in ---------------------------------------


def test_the_atomic_unit_of_conductivity_is_e2_over_hbar_bohr():
    """4.6e6 S/m, and it is what takes an anomalous Hall constant to S/cm.

    The number is fixed by the assembly rather than chosen: the Kubo sum is
    ``1/Omega`` times a squared velocity over a squared energy, which in
    Rydberg units is ``bohr^2 / bohr^3`` and therefore ``1/bohr`` -- the
    atomic unit ``e^2/(hbar a_0)``. An anomalous Hall conductivity of a
    ferromagnet is of order 1000 S/cm, which is 0.02 of it.
    """
    assert AU_TO_S_PER_CM == pytest.approx(45998.48, rel=1.0e-6)


def test_the_hall_conductivity_is_the_antisymmetric_part():
    """A symmetric ``sigma`` has no Hall conductivity however large it is.

    The symmetric part of a metal's static conductivity is its Drude weight and
    is orders of magnitude larger than the Hall part, so reading ``sigma[0, 1]``
    rather than antisymmetrising is a leak that grows with the metal.
    """
    sigma = np.zeros((1, 3, 3), dtype=complex)
    sigma[0] = [[10.0, 3.0, 0.0], [3.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    assert np.max(np.abs(_tensor(sigma, [0.0]).hall_conductivity)) == 0.0

    sigma[0, 0, 1], sigma[0, 1, 0] = 3.0 + 1.0, 3.0 - 1.0
    hall = _tensor(sigma, [0.0]).hall_conductivity
    assert hall[0, 1] == pytest.approx(AU_TO_S_PER_CM, rel=1.0e-12)
    assert hall[1, 0] == pytest.approx(-AU_TO_S_PER_CM, rel=1.0e-12)


def test_the_dielectric_tensor_is_the_hartree_frequency_not_the_rydberg_one():
    """``eps = 1 + 4 pi i sigma / w`` with ``w`` in **Hartree**, twice over.

    ``sigma`` is in Hartree atomic units, so the ``w`` it is divided by has to
    be too -- and the frequency axis this module carries is in Ry, as every
    energy in the package is. Getting it wrong is a clean factor of two in
    every dielectric function, with nothing else changing.
    """
    w = np.array([0.4])  # Ry
    sigma = np.zeros((1, 3, 3), dtype=complex)
    sigma[0] = np.eye(3) * (0.0 + 0.5j)
    eps = _tensor(sigma, w, broadening=0.0).dielectric
    assert eps[0, 0, 0] == pytest.approx(1.0 + FPI * 1j * 0.5j / 0.2)


def test_the_kerr_angle_is_moke_f90s_expression():
    """``-sigma_xy / (sigma_xx sqrt(1 + 4 pi i sigma_xx / w))``, in degrees.

    ``moke.f90`` is 87 lines and this is the only one that computes anything;
    the rest reads two files. It is zero at ``w = 0`` by construction there,
    which is a definition rather than a limit.
    """
    w = np.array([0.0, 0.4])
    sigma = np.zeros((2, 3, 3), dtype=complex)
    for i in range(2):
        sigma[i, 0, 0] = 0.3 + 0.1j
        sigma[i, 0, 1], sigma[i, 1, 0] = 0.02 - 0.01j, -0.02 + 0.01j
    kerr = _tensor(sigma, w).kerr

    assert kerr[0] == 0.0
    sxx, sxy, omega = 0.3 + 0.1j, 0.02 - 0.01j, 0.2
    expected = -sxy / (sxx * np.sqrt(1.0 + FPI * 1j * sxx / omega))
    assert kerr[1] == pytest.approx(expected * 180.0 / np.pi)


def test_the_fsum_is_the_spectral_weight_over_pi_n_over_two():
    """A flat ``Re sigma`` of known area comes back as its own ratio.

    The f-sum rule's normalisation is ``pi n_e / 2`` with ``n_e`` the electron
    density, so the *volume* and the *electron count* both enter it -- which is
    what makes it a check on the assembly's prefactor and not only on its band
    truncation.
    """
    w = np.linspace(0.0, 2.0, 2001)  # Ry
    sigma = np.zeros((w.size, 3, 3), dtype=complex)
    sigma[:, 0, 0] = 1.0
    nelec, volume = 8.0, 270.0
    # The integral runs over w in Hartree, so a flat unit Re sigma over 2 Ry
    # has area 1.0.
    expected = 1.0 / (np.pi * (nelec / volume) / 2.0)
    assert _tensor(sigma, w, nelec=nelec, volume=volume).fsum == pytest.approx(
        expected, rel=1.0e-9
    )


# -- what it refuses -----------------------------------------------------------


def _calculation(case: str, **overrides):
    """A stand-in carrying only what the guard reads."""
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    fields = {"is_ultrasoft": False, "is_paw": False, "spiral": False,
              "system": system}
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_an_ultrasoft_dataset_is_refused_for_the_overlaps_velocity():
    """The ``dS/dk`` term is written and unvalidated, exactly as in P47."""
    with pytest.raises(NotImplementedError, match="dS/dk"):
        require_a_conductivity_regime(_calculation("si2-nosym", is_ultrasoft=True))
    with pytest.raises(NotImplementedError, match="dS/dk"):
        require_a_conductivity_regime(_calculation("si2-nosym", is_paw=True))


def test_a_spin_spiral_is_refused_for_its_two_spheres():
    with pytest.raises(NotImplementedError, match="two spinor components"):
        require_a_conductivity_regime(_calculation("si2-nosym", spiral=True))


def test_a_symmetry_reduced_k_set_is_refused_because_sigma_xy_is_axial():
    """The wedge refusal, and the ``nosym`` run that passes it.

    ``si-epsilon.in`` is silicon on a reduced set; ``si2-nosym.in`` is the
    same cell on the whole grid. The antisymmetric part of ``sigma`` is the
    Berry-curvature integral in the static limit, so it is an axial vector and
    a wedge does not sum to the cell's.
    """
    with pytest.raises(NotImplementedError, match="axial vector"):
        require_a_conductivity_regime(_calculation("si-epsilon"))
    require_a_conductivity_regime(_calculation("si2-nosym"))
