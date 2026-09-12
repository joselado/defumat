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

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.response.conductivity import (
    OpticalConductivity,
    _curvature_sum,
    _pair_weights,
    _resolvent_sum,
    require_a_conductivity_regime,
)
from defumat.solvers.davidson import EMPTY_ETHR_FLOOR
from defumat.system import build_system
from defumat.units import AU_TO_S_PER_CM, FPI

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


# -- the degeneracy guard, and which route actually needs it -------------------

def test_a_degenerate_pair_cancels_on_the_frequency_route_and_diverges_on_the_other():
    """``OPEN.md`` B1's mechanism, and the correction to it.

    The sum divides by ``e_mn``, so a pair degenerate by symmetry that the
    eigensolver returns split by round-off looks like a catastrophe. On the
    **frequency** route with a smeared occupation it is not one, and not by
    luck. The two orderings carry ``t_nm = -t_mn`` and ``z_mn = conj(z_nm)``,
    so what survives is ``w_k[f(e_n) - f(e_m)]``, itself linear in the gap
    whenever ``f`` is a smooth function of energy: the singularity cancels
    analytically and the answer is flat as the splitting goes to zero.

    The **curvature** route divides by ``e_mn`` twice. The same numerator
    difference kills one power and leaves ``1/g``, so that route -- which is
    the intrinsic anomalous Hall conductivity -- is what the guard is really
    protecting. Asserting both together is the point: a test of the guard on
    the frequency route alone would be measuring something that cannot move.
    """
    rng = np.random.default_rng(0)
    v = rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2))
    element = jnp.asarray(v + np.conj(np.swapaxes(v, 1, 2)))  # Hermitian, as dH/dk is
    zomega = jnp.asarray([0.0 + 0.01j])

    frequency, curvature = [], []
    for gap in (1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4):
        energies = jnp.asarray([0.0, gap])
        # One occupation *function* of energy, which is what a smearing is.
        filling = 1.0 / (1.0 + jnp.exp(energies / 0.02))
        # The guard off, so what is measured is the expression and not the mask.
        f, _ = _resolvent_sum(element, energies, filling, filling, zomega, 0.0)
        c, _ = _curvature_sum(element, energies, filling, filling, 0.0)
        frequency.append(float(jnp.max(jnp.abs(f))))
        curvature.append(float(jnp.max(jnp.abs(c))))

    # Flat over eight decades of splitting: 1.9237e4 at every one of them.
    assert max(frequency) / min(frequency) < 1.001

    # And exactly ``1/g`` on the other route -- each step of a hundred in the
    # gap is a step of a hundred in the answer.
    for coarse, fine in zip(curvature[:-1], curvature[1:]):
        assert coarse / fine == pytest.approx(100.0, rel=1e-3)


def test_fixed_occupations_across_a_degeneracy_diverge_on_both_routes():
    """The other place nothing cancels, and it has a name already.

    A band set cut *inside* a degenerate multiplet puts one member full and
    its partner empty at the same eigenvalue, so ``f`` is not a function of
    energy at all and the cancellation above has nothing to work with:
    ``1/g`` on the frequency route and ``1/g^2`` on the curvature one. That is
    :attr:`~defumat.response.conductivity.OpticalConductivity.band_cut_gap`'s
    pathology, measured here from the expression rather than from a run.
    """
    rng = np.random.default_rng(0)
    v = rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2))
    element = jnp.asarray(v + np.conj(np.swapaxes(v, 1, 2)))
    zomega = jnp.asarray([0.0 + 0.01j])
    filling = jnp.asarray([1.0, 0.0])   # full and empty, at the same energy

    frequency, curvature = [], []
    for gap in (1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4):
        energies = jnp.asarray([0.0, gap])
        f, _ = _resolvent_sum(element, energies, filling, filling, zomega, 0.0)
        c, _ = _curvature_sum(element, energies, filling, filling, 0.0)
        frequency.append(float(jnp.max(jnp.abs(f))))
        curvature.append(float(jnp.max(jnp.abs(c))))

    for coarse, fine in zip(frequency[:-1], frequency[1:]):
        assert coarse / fine == pytest.approx(100.0, rel=1e-3)
    for coarse, fine in zip(curvature[:-1], curvature[1:]):
        assert coarse / fine == pytest.approx(1.0e4, rel=1e-3)


def test_the_guard_covers_the_round_off_an_scf_leaves_on_an_empty_band():
    """Why the constant is 1e-5 and not ``dielectric.f90``'s 1e-8.

    :func:`optical_conductivity` takes an array of eigenvalues and cannot see
    where they came from, and its own docstring's advice is not a guarantee:
    hand it an ``SCFResult``'s and the *empty* bands carry
    ``max(5 ethr, 1e-5)`` Ry, since that is all
    :func:`~defumat.scf.driver.band_thresholds` asks of them. A symmetry
    degeneracy split by 3e-7 of that residue is the case, and on the curvature
    route the old 1e-8 let it through into a ``1/g^2`` weight.
    """
    rng = np.random.default_rng(1)
    v = rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2))
    element = jnp.asarray(v + np.conj(np.swapaxes(v, 1, 2)))
    energies = jnp.asarray([0.0, 3.0e-7])
    # The occupation has to be the *function*, not two equal numbers. Two
    # states given exactly the same filling cancel on both routes -- the
    # numerator difference is identically zero -- and the divergence the guard
    # is for lives in ``f(e_n) - f(e_m)``, which a smearing makes small and
    # nonzero rather than absent.
    filling = 1.0 / (1.0 + jnp.exp(energies / 0.02))

    kept, survived = _curvature_sum(element, energies, filling, filling, 1.0e-8)
    dropped, removed = _curvature_sum(element, energies, filling, filling,
                                      EMPTY_ETHR_FLOOR)
    assert int(survived) == 0            # the old constant saw nothing
    assert int(removed) == 2             # the new one sees the pair
    # Measured 1.593e8 against a curvature whose honest value here is nothing
    # at all: the pair is degenerate and the number is its round-off inverted.
    assert float(jnp.max(jnp.abs(kept))) > 1.0e8
    assert float(jnp.max(jnp.abs(dropped))) == 0.0


def test_the_guard_does_not_count_the_diagonal_it_always_masks():
    """The count has to discriminate, and ``e_nn = 0`` is the way it would not.

    Every partially filled band of every metal has ``W_n (1 - f_n) != 0`` on
    the diagonal, where the gap is zero exactly. Counting those would report a
    large number on every metal ever run and the same number whether or not
    anything was wrong, which is ``CLAUDE.md``'s "a check whose null result
    cannot be told from a pass".
    """
    energies = jnp.asarray([0.0, 0.1, 0.2, 0.3])
    wg = jnp.asarray([1.0, 0.5, 0.5, 0.0])
    filling = jnp.asarray([1.0, 0.5, 0.5, 0.0])
    _, _, dropped = _pair_weights(energies, wg, filling, EMPTY_ETHR_FLOOR)
    assert int(dropped) == 0


def test_the_two_halves_of_the_tensor_partition_the_pairs_at_the_tolerance():
    """A gap of exactly ``tol`` belongs to one half, not to neither.

    The interband sum keeps ``|gap| > tol`` and the Drude multiplet block
    keeps ``|gap| <= tol``, so every pair is in exactly one of them. With both
    written ``>`` and ``<`` a pair sitting on the boundary contributed to no
    part of ``sigma`` at all.
    """
    tol = 1.0e-6
    energies = jnp.asarray([0.0, tol])
    wg = jnp.asarray([0.5, 0.5])
    filling = jnp.asarray([0.5, 0.5])
    t, gap, dropped = _pair_weights(energies, wg, filling, tol)
    # Excluded from the interband half ...
    assert float(jnp.max(jnp.abs(t))) == 0.0
    assert int(dropped) == 2
    # ... and therefore inside the multiplet block, which is ``<=``.
    off = ~np.eye(2, dtype=bool)
    assert bool(np.all(np.abs(np.asarray(gap))[off] <= tol))
