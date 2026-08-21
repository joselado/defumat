"""P23 checks: continuing an SCF across a change of spin regime.

The expensive part -- that a continued run reaches the same self-consistent
solution as a fresh one -- is in ``tests/regression/test_continuation.py``. What
is checked here is everything that can be checked without an SCF: that the
promotion conserves charge, that it puts the magnetization where the target's
input says it goes, that it refuses the cases it cannot do rather than
approximating them, and that :meth:`System.with_spin` rebuilds the k-points
instead of merely relabelling them.
"""

from functools import lru_cache

import numpy as np
import pytest

from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf import Calculation, SCFResult
from pypresso.scf.continuation import (
    ContinuedState,
    continued_state,
    from_spin_components,
    promote_ns,
    promote_wavefunctions,
    spin_components,
)
from pypresso.system import build_system
from tests.conftest import QE_ROOT

pytestmark = pytest.mark.unit

SILICON = QE_ROOT / "test-suite" / "pw_scf" / "scf.in"
PSEUDO = "tests/data/pseudo/Si.pz-vbc.UPF"


def _silicon(**spin):
    if not SILICON.is_file():
        pytest.skip("QE reference tree not present")
    import dataclasses

    system = build_system(read_pw_input(SILICON))
    # Two channels need an occupation scheme that can fill them unequally.
    system = dataclasses.replace(
        system, occupations="smearing", smearing="gaussian", degauss=0.02,
        tstress=False,
    )
    return system.with_spin(**spin) if spin else system


@lru_cache(maxsize=None)
def _calculation(nspin: int, magnetization: tuple = (0.0,), angles: tuple = ()) -> Calculation:
    kwargs = {"nspin": nspin, "starting_magnetization": magnetization}
    if angles:
        kwargs["angle1"], kwargs["angle2"] = angles
    system = _silicon(**kwargs)
    return Calculation(system, (read_upf(PSEUDO),))


def _result(density, nspin, nspin_mag=None, **extra) -> SCFResult:
    """An :class:`SCFResult` carrying only what a continuation reads."""
    density = np.asarray(density, dtype=float)
    zeros = np.zeros((1, 1))
    return SCFResult(
        converged=True, iterations=1, total_energy=0.0, energy_terms={},
        eigenvalues=zeros, occupations=zeros,
        wavefunctions=extra.pop("wavefunctions", None),
        density=density, potential=density,
        nspin=nspin,
        nspin_mag=nspin if nspin_mag is None else nspin_mag,
        **extra,
    )


def _random_density(shape, nspin_mag, seed=0):
    """A positive charge with a magnetization smaller than it, on ``shape``."""
    rng = np.random.default_rng(seed)
    charge = 1.0 + rng.random(shape)
    moment = 0.3 * (rng.random((3,) + shape) - 0.5)
    return np.asarray(from_spin_components(charge, moment, nspin_mag))


# --------------------------------------------------------------------------
# The representation itself


@pytest.mark.parametrize("nspin_mag", [1, 2, 4])
def test_spin_components_round_trip(nspin_mag):
    values = _random_density((4, 4, 4), nspin_mag, seed=nspin_mag)
    charge, moment = spin_components(values, nspin_mag)
    again = from_spin_components(charge, moment, nspin_mag)
    assert np.allclose(np.asarray(again), values)


def test_collinear_magnetization_is_on_z():
    values = _random_density((3, 3, 3), 2)
    _, moment = spin_components(values, 2)
    assert np.allclose(moment[0], 0.0) and np.allclose(moment[1], 0.0)
    assert np.allclose(moment[2], values[0] - values[1])


# --------------------------------------------------------------------------
# The density


def test_unpolarized_to_collinear_conserves_charge_and_seeds_a_moment():
    calculation = _calculation(2, (0.4,))
    grid = tuple(calculation.basis.dense.grid)
    source = _random_density(grid, 1)
    state = continued_state(_result(source, 1), calculation, wavefunctions=False)

    assert state.regimes == (1, 2)
    assert state.seeded
    charge, moment = spin_components(state.density, 2)
    assert np.allclose(np.asarray(charge), source[0])
    # Nothing else in the SCF breaks spin symmetry, so an unseeded promotion
    # would converge straight back to the unpolarized solution.
    assert float(np.max(np.abs(np.asarray(moment)))) > 1.0e-6


def test_a_target_with_no_starting_magnetization_starts_unpolarized():
    calculation = _calculation(2, (0.0,))
    grid = tuple(calculation.basis.dense.grid)
    source = _random_density(grid, 1)
    state = continued_state(_result(source, 1), calculation, wavefunctions=False)
    assert np.allclose(np.asarray(state.density[0]), np.asarray(state.density[1]))


def test_collinear_to_noncollinear_rotates_onto_the_targets_axis():
    # angle1 = 90 points the moment along x, which is QE's own pw_noncolin case.
    calculation = _calculation(4, (0.5,), ((90.0,), (0.0,)))
    assert calculation.nspin_mag == 4
    grid = tuple(calculation.basis.dense.grid)
    source = _random_density(grid, 2)
    state = continued_state(_result(source, 2), calculation, wavefunctions=False)

    assert not state.seeded
    charge, moment = spin_components(state.density, 4)
    assert np.allclose(np.asarray(charge), source[0] + source[1])
    assert np.allclose(np.asarray(moment[0]), source[0] - source[1])
    assert np.allclose(np.asarray(moment[1]), 0.0)
    assert np.allclose(np.asarray(moment[2]), 0.0)


def test_demotion_to_one_channel_keeps_the_total_charge():
    calculation = _calculation(1)
    grid = tuple(calculation.basis.dense.grid)
    source = _random_density(grid, 4)
    state = continued_state(_result(source, 4, nspin_mag=4), calculation,
                            wavefunctions=False)
    assert state.density.shape == (1,) + grid
    assert np.allclose(np.asarray(state.density[0]), source[0])
    assert "dropped" in state.description


def test_noncollinear_along_x_comes_back_down_onto_z():
    """The inverse of the rotation, and an antiferromagnet keeps its signs."""
    calculation = _calculation(2, (0.4,))
    grid = tuple(calculation.basis.dense.grid)
    rng = np.random.default_rng(11)
    charge = 1.0 + rng.random(grid)
    # A *sign-changing* scalar magnetization, all of it along x: the case whose
    # signed integral is zero and whose axis a mean would fail to find.
    scalar = rng.normal(size=grid)
    zero = np.zeros_like(scalar)
    source = np.stack([charge, scalar, zero, zero])

    state = continued_state(_result(source, 4, nspin_mag=4), calculation,
                            wavefunctions=False)
    _, moment = spin_components(state.density, 2)
    assert np.allclose(np.asarray(state.density[0] + state.density[1]), charge)
    assert np.allclose(np.abs(np.asarray(moment[2])), np.abs(scalar))
    # ... and the same magnetization, not its absolute value: the sign
    # structure is what makes it an antiferromagnet rather than a ferromagnet.
    assert np.allclose(np.asarray(moment[2]), scalar) or np.allclose(
        np.asarray(moment[2]), -scalar
    )


def test_a_genuinely_noncollinear_source_cannot_become_collinear():
    calculation = _calculation(2, (0.4,))
    grid = tuple(calculation.basis.dense.grid)
    source = _random_density(grid, 4, seed=5)
    with pytest.raises(ValueError, match="genuinely noncollinear"):
        continued_state(_result(source, 4, nspin_mag=4), calculation,
                        wavefunctions=False)


def test_carry_refuses_a_source_with_no_magnetization():
    calculation = _calculation(2, (0.4,))
    grid = tuple(calculation.basis.dense.grid)
    source = _random_density(grid, 1)
    with pytest.raises(ValueError, match="none to carry"):
        continued_state(_result(source, 1), calculation, magnetization="carry")


def test_seed_overrides_a_source_that_has_a_magnetization():
    calculation = _calculation(2, (0.4,))
    grid = tuple(calculation.basis.dense.grid)
    source = _random_density(grid, 2)
    state = continued_state(_result(source, 2), calculation,
                            magnetization="seed", wavefunctions=False)
    assert state.seeded
    _, seeded = spin_components(state.density, 2)
    _, carried = spin_components(source, 2)
    assert not np.allclose(np.asarray(seeded), np.asarray(carried))


def test_none_starts_the_target_unpolarized():
    calculation = _calculation(2, (0.4,))
    grid = tuple(calculation.basis.dense.grid)
    source = _random_density(grid, 2)
    state = continued_state(_result(source, 2), calculation,
                            magnetization="none", wavefunctions=False)
    assert np.allclose(np.asarray(state.density[0]), np.asarray(state.density[1]))


def test_a_grid_mismatch_is_refused():
    calculation = _calculation(2, (0.4,))
    source = _random_density((3, 3, 3), 1)
    with pytest.raises(ValueError, match="grid"):
        continued_state(_result(source, 1), calculation)


def test_species_pointing_different_ways_are_refused_with_the_escape_hatch():
    import dataclasses

    calculation = _calculation(4, (0.5,), ((90.0,), (0.0,)))
    # One species in this cell, so the disagreement has to be built by hand:
    # two types, one along x and one along z.
    system = dataclasses.replace(
        calculation.system, starting_magnetization=(0.5, 0.5),
        angle1=(90.0, 0.0), angle2=(0.0, 0.0),
    )
    stand_in = _Stand(system, calculation)
    grid = tuple(calculation.basis.dense.grid)
    with pytest.raises(ValueError, match="magnetization='seed'"):
        continued_state(_result(_random_density(grid, 2), 2), stand_in,
                        wavefunctions=False)


class _Stand:
    """A calculation with a different ``System`` bolted on.

    Building a real two-species silicon would mean a second pseudopotential and
    a second FFT grid for the sake of one refusal; what the refusal reads is the
    target's ``starting_magnetization`` and its angles, and those are on the
    system.
    """

    def __init__(self, system, calculation):
        self.system = system
        self._calculation = calculation

    def __getattr__(self, name):
        if name == "starting_magnetization":
            return np.asarray(self.system.starting_magnetization)
        if name == "magnetization_directions":
            theta = np.radians(np.asarray(self.system.angle1))
            phi = np.radians(np.asarray(self.system.angle2))
            return np.stack([np.sin(theta) * np.cos(phi),
                             np.sin(theta) * np.sin(phi),
                             np.cos(theta)], axis=1)
        return getattr(self._calculation, name)


# --------------------------------------------------------------------------
# becsum and ns


@lru_cache(maxsize=None)
def _ultrasoft(nspin: int, magnetization: tuple) -> Calculation:
    """The same cell with an ultrasoft dataset, which is what has a ``becsum``."""
    system = _silicon(nspin=nspin, starting_magnetization=magnetization)
    return Calculation(
        system, (read_upf("tests/data/pseudo/Si.pz-n-rrkjus_psl.0.1.UPF"),)
    )


def test_becsum_is_promoted_channel_by_channel_like_the_density():
    source_run = _ultrasoft(1, (0.0,))
    target = _ultrasoft(2, (0.4,))
    atomic = source_run.starting_becsum()
    grid = tuple(target.basis.dense.grid)
    state = continued_state(
        _result(_random_density(grid, 1), 1, becsum=atomic), target,
        wavefunctions=False,
    )
    assert len(state.becsum) == len(atomic)
    for promoted, source in zip(state.becsum, atomic):
        assert promoted.shape[0] == 2
        # The charge is conserved species by species, exactly as it is for the
        # density -- the two halves of the mixed state promoted by one rule.
        assert np.allclose(np.asarray(promoted[0] + promoted[1]),
                           np.asarray(source[0]))


def test_becsum_of_a_different_pseudopotential_is_dropped_not_reshaped():
    target = _ultrasoft(2, (0.4,))
    wrong = tuple(
        None if block is None else np.zeros((1, block.shape[1], 3, 3))
        for block in target.starting_becsum()
    )
    grid = tuple(target.basis.dense.grid)
    with pytest.warns(RuntimeWarning, match="different pseudopotential"):
        state = continued_state(
            _result(_random_density(grid, 1), 1, becsum=wrong), target,
            wavefunctions=False,
        )
    atomic = target.starting_becsum()
    for promoted, seed in zip(state.becsum, atomic):
        assert np.allclose(np.asarray(promoted), np.asarray(seed))


def test_norm_conserving_becsum_is_empty():
    calculation = _calculation(2, (0.4,))
    grid = tuple(calculation.basis.dense.grid)
    state = continued_state(_result(_random_density(grid, 1), 1), calculation,
                            wavefunctions=False)
    assert state.becsum == ()


def test_ns_is_copied_into_both_channels_and_averaged_back():
    setup = _Setup()
    calculation = _Hubbard(setup, nspin=2)
    ns = np.arange(1 * 2 * 3 * 3, dtype=float).reshape(1, 2, 3, 3)
    promoted = np.asarray(promote_ns(_result(np.zeros((1, 2, 2, 2)), 1, ns=ns),
                                     calculation))
    assert promoted.shape == (2, 2, 3, 3)
    # ``ns`` is per channel for every nspin (``new_ns`` halves the unpolarized
    # one), so this is a copy and not a halving.
    assert np.allclose(promoted[0], ns[0]) and np.allclose(promoted[1], ns[0])

    back = np.asarray(promote_ns(
        _result(np.zeros((1, 2, 2, 2)), 2, ns=promoted), _Hubbard(setup, nspin=1)
    ))
    assert back.shape == (1, 2, 3, 3)
    assert np.allclose(back[0], ns[0])


def test_ns_in_a_noncollinear_target_is_refused_by_name():
    calculation = _Hubbard(_Setup(), nspin=4)
    ns = np.zeros((2, 2, 3, 3))
    with pytest.raises(NotImplementedError, match="ns_nc"):
        promote_ns(_result(np.zeros((1, 2, 2, 2)), 2, ns=ns), calculation)


class _Setup:
    nslot = 2
    ldmx = 3


class _Hubbard:
    """The three attributes :func:`promote_ns` reads off a calculation."""

    def __init__(self, setup, nspin):
        self.hubbard = setup
        self.nspin = nspin
        self.is_hubbard = True


# --------------------------------------------------------------------------
# The wavefunctions


def test_wavefunctions_seed_both_channels_unchanged():
    calculation = _calculation(2, (0.4,))
    nk, npwx = calculation.system.kpoints.nk, calculation.basis.npwx
    psi = np.zeros((1, nk, 4, npwx), dtype=complex)
    psi[0, :, 0, 0] = 1.0
    span = promote_wavefunctions(
        _result(_random_density(tuple(calculation.basis.dense.grid), 1), 1,
                wavefunctions=psi, system=calculation.system),
        calculation,
    )
    assert span.shape == (nk, 4, npwx)
    assert np.allclose(np.asarray(span), psi[0])


def test_two_channels_become_orthonormal_spinors():
    source = _calculation(2, (0.4,))
    # A *nonmagnetic* noncollinear target, which is the spin-orbit case: the
    # k-set is the collinear one, so the states can be carried at all.
    target = _calculation(4, (0.0,))
    nk, npwx = source.system.kpoints.nk, source.basis.npwx
    assert target.system.kpoints.nk == nk and target.npol == 2
    rng = np.random.default_rng(3)
    psi = rng.normal(size=(2, nk, 3, npwx)) + 1j * rng.normal(size=(2, nk, 3, npwx))
    # Orthonormal within each channel, which is what an eigensolver returns.
    for spin in range(2):
        for k in range(nk):
            psi[spin, k] = np.linalg.qr(psi[spin, k].T)[0].T

    span = np.asarray(promote_wavefunctions(
        _result(_random_density(tuple(source.basis.dense.grid), 2), 2,
                wavefunctions=psi, system=source.system),
        target,
    ))
    assert span.shape == (nk, 6, 2 * npwx)
    # The up block occupies the first component and the down block the second,
    # so the two halves are orthogonal by construction and the whole set is
    # orthonormal -- which is what makes it a usable span.
    overlap = span[0] @ span[0].conj().T
    assert np.allclose(overlap, np.eye(6), atol=1.0e-10)
    assert np.allclose(span[0, :3, npwx:], 0.0)
    assert np.allclose(span[0, 3:, :npwx], 0.0)


def test_a_spinor_is_not_split_back_into_channels():
    target = _calculation(2, (0.4,))
    source = _calculation(4, (0.0,))
    nk, npwx = source.system.kpoints.nk, source.basis.npwx
    psi = np.zeros((1, nk, 4, 2 * npwx), dtype=complex)
    result = _result(_random_density(tuple(source.basis.dense.grid), 1), 4,
                     nspin_mag=1, wavefunctions=psi, system=source.system)
    with pytest.warns(RuntimeWarning, match="not being carried over"):
        assert promote_wavefunctions(result, target) is None


def test_a_different_k_set_drops_the_wavefunctions():
    calculation = _calculation(2, (0.4,))
    nk, npwx = calculation.system.kpoints.nk, calculation.basis.npwx
    psi = np.zeros((1, nk + 1, 4, npwx), dtype=complex)
    result = _result(_random_density(tuple(calculation.basis.dense.grid), 1), 1,
                     wavefunctions=psi, system=calculation.system)
    with pytest.warns(RuntimeWarning, match="k-points"):
        assert promote_wavefunctions(result, calculation) is None


# --------------------------------------------------------------------------
# System.with_spin


def test_with_spin_applies_the_degspin_convention_once():
    unpolarized = _silicon()
    assert float(np.sum(unpolarized.kpoints.weights)) == pytest.approx(2.0)
    for nspin in (2, 4):
        target = unpolarized.with_spin(nspin, starting_magnetization=(0.3,))
        assert float(np.sum(target.kpoints.weights)) == pytest.approx(1.0)
    # ... and coming back restores it, rather than doubling what was halved.
    back = unpolarized.with_spin(2, starting_magnetization=(0.3,)).with_spin(
        1, starting_magnetization=(0.0,)
    )
    assert float(np.sum(back.kpoints.weights)) == pytest.approx(2.0)


def test_with_spin_expands_the_k_set_for_a_magnetic_noncollinear_run():
    unpolarized = _silicon()
    plain = unpolarized.with_spin(2, starting_magnetization=(0.3,))
    magnetic = unpolarized.with_spin(4, starting_magnetization=(0.3,))
    assert plain.kpoints.nk == unpolarized.kpoints.nk
    # The magnetic group is smaller and has no -k = k, so irreducible_BZ hands
    # the noncollinear run k-points the collinear one never had.
    assert magnetic.kpoints.nk > unpolarized.kpoints.nk
    assert float(np.sum(magnetic.kpoints.weights)) == pytest.approx(1.0)


def test_with_spin_doubles_nbnd_into_a_spinor_calculation():
    import dataclasses

    system = dataclasses.replace(_silicon(), nbnd=8)
    assert system.with_spin(4, starting_magnetization=(0.3,)).nbnd == 16
    assert system.with_spin(4, starting_magnetization=(0.3,)).with_spin(
        2, starting_magnetization=(0.3,)
    ).nbnd == 8


def test_with_spin_refuses_what_the_input_reader_refuses():
    system = _silicon()
    with pytest.raises(ValueError, match="lspinorb"):
        system.with_spin(2, lspinorb=True)
    with pytest.raises(ValueError, match="nspin = 1"):
        system.with_spin(1, starting_magnetization=(0.5,))
    with pytest.raises(ValueError, match="expected 1, 2 or 4"):
        system.with_spin(3)


def test_continued_state_can_be_handed_to_run_scf_only_on_its_own():
    from pypresso.scf import run_scf

    with pytest.raises(ValueError, match="two states at once"):
        run_scf(_silicon(), (read_upf(PSEUDO),),
                starting_from=ContinuedState(density=np.zeros((1, 2, 2, 2))),
                starting_density=np.zeros((1, 2, 2, 2)))
