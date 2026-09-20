"""P23 checks: continuing an SCF across a change of spin regime.

The expensive part -- that a continued run reaches the same self-consistent
solution as a fresh one -- is in ``tests/regression/test_continuation.py``. What
is checked here is everything that can be checked without an SCF: that the
promotion conserves charge, that it puts the magnetization where the target's
input says it goes, that it refuses the cases it cannot do rather than
approximating them, and that :meth:`System.with_spin` rebuilds the k-points
instead of merely relabelling them.
"""

import types
import pathlib
from functools import lru_cache

import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf import Calculation, SCFResult
from defumat.scf.continuation import (
    ContinuedState,
    continued_state,
    from_spin_components,
    promote_ns,
    promote_wavefunctions,
    spin_components,
)
from defumat.system import build_system
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


def _random_density(shape, nspin_mag, seed=0, calculation=None):
    """A positive charge with a magnetization smaller than it, on ``shape``.

    ``calculation`` normalises the charge to that run's ``nelec``, which is not
    decoration: a continuation refuses a source density whose own electron count
    is not the target's, because that is what a swapped dataset looks like, and
    a random charge is a dataset nobody has. It is left out where the density
    never reaches :func:`~defumat.scf.continuation.continued_state` -- a
    ``promote_wavefunctions`` call, or a shape that is refused for its grid
    before anything counts electrons.
    """
    rng = np.random.default_rng(seed)
    charge = 1.0 + rng.random(shape)
    if calculation is not None:
        volume = float(calculation.system.cell.volume)
        charge = charge * (float(calculation.nelec) / (charge.mean() * volume))
    moment = 0.3 * (rng.random((3,) + shape) - 0.5) * charge.mean()
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
    source = _random_density(grid, 1, calculation=calculation)
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
    source = _random_density(grid, 1, calculation=calculation)
    state = continued_state(_result(source, 1), calculation, wavefunctions=False)
    assert np.allclose(np.asarray(state.density[0]), np.asarray(state.density[1]))


def test_collinear_to_noncollinear_rotates_onto_the_targets_axis():
    # angle1 = 90 points the moment along x, which is QE's own pw_noncolin case.
    calculation = _calculation(4, (0.5,), ((90.0,), (0.0,)))
    assert calculation.nspin_mag == 4
    grid = tuple(calculation.basis.dense.grid)
    source = _random_density(grid, 2, calculation=calculation)
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
    source = _random_density(grid, 4, calculation=calculation)
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
    charge *= calculation.nelec / (charge.mean()
                                  * float(calculation.system.cell.volume))
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
    source = _random_density(grid, 4, seed=5, calculation=calculation)
    with pytest.raises(ValueError, match="genuinely noncollinear"):
        continued_state(_result(source, 4, nspin_mag=4), calculation,
                        wavefunctions=False)


def test_carry_refuses_a_source_with_no_magnetization():
    calculation = _calculation(2, (0.4,))
    grid = tuple(calculation.basis.dense.grid)
    source = _random_density(grid, 1, calculation=calculation)
    with pytest.raises(ValueError, match="none to carry"):
        continued_state(_result(source, 1), calculation, magnetization="carry")


def test_seed_overrides_a_source_that_has_a_magnetization():
    calculation = _calculation(2, (0.4,))
    grid = tuple(calculation.basis.dense.grid)
    source = _random_density(grid, 2, calculation=calculation)
    state = continued_state(_result(source, 2), calculation,
                            magnetization="seed", wavefunctions=False)
    assert state.seeded
    _, seeded = spin_components(state.density, 2)
    _, carried = spin_components(source, 2)
    assert not np.allclose(np.asarray(seeded), np.asarray(carried))


def test_none_starts_the_target_unpolarized():
    calculation = _calculation(2, (0.4,))
    grid = tuple(calculation.basis.dense.grid)
    source = _random_density(grid, 2, calculation=calculation)
    state = continued_state(_result(source, 2), calculation,
                            magnetization="none", wavefunctions=False)
    assert np.allclose(np.asarray(state.density[0]), np.asarray(state.density[1]))


def test_a_grid_mismatch_is_refused():
    calculation = _calculation(2, (0.4,))
    source = _random_density((3, 3, 3), 1)
    with pytest.raises(ValueError, match="grid"):
        continued_state(_result(source, 1), calculation)


def test_atoms_pointing_different_ways_are_refused_with_the_escape_hatch():
    """One scalar cannot point two ways, and the message names the way out.

    The disagreement is stated per *atom*, with a ``STARTING_MOMENTS`` card, in
    place of the two hand-built species this test used to carry: the direction
    now comes from :attr:`System.local_moments`, so a card is what a real
    disagreement looks like and no stand-in is needed for it.
    """
    calculation = _textured(((0.4, 0.0, 0.0), (0.0, 0.4, 0.0)))
    grid = tuple(calculation.basis.dense.grid)
    source = _random_density(grid, 2, calculation=calculation)
    with pytest.raises(ValueError, match="magnetization='seed'"):
        continued_state(_result(source, 2, system=_calculation(2, (0.4,)).system),
                        calculation, wavefunctions=False)


def _textured(per_atom) -> Calculation:
    """A spinor silicon whose texture is in the card and *not* in the angles.

    ``angle1``/``angle2`` stay at zero, which is the case the direction used to
    be read from: a card user's target said "along z" to anything that asked
    the angles.
    """
    import dataclasses

    calculation = _calculation(4, (0.4,))
    return Calculation(
        dataclasses.replace(calculation.system, angle1=(0.0,), angle2=(0.0,),
                            starting_moments=tuple(tuple(float(x) for x in row)
                                                   for row in per_atom)),
        (read_upf(PSEUDO),),
    )


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
        _result(_random_density(grid, 1, calculation=target), 1, becsum=atomic), target,
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
            _result(_random_density(grid, 1, calculation=target), 1, becsum=wrong), target,
            wavefunctions=False,
        )
    atomic = target.starting_becsum()
    for promoted, seed in zip(state.becsum, atomic):
        assert np.allclose(np.asarray(promoted), np.asarray(seed))


def test_norm_conserving_becsum_is_empty():
    calculation = _calculation(2, (0.4,))
    grid = tuple(calculation.basis.dense.grid)
    state = continued_state(
        _result(_random_density(grid, 1, calculation=calculation), 1), calculation,
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


def test_ns_promotes_into_the_two_diagonal_spin_blocks_of_a_spinor():
    """2 -> 4 is the density's own rule one axis out: decompose, decide, recompose.

    A spinor ``ns`` is ``(4, nslot, ldmx, ldmx)`` complex, the four entries
    being ``(uu, ud, du, dd)``, so a collinear pair goes into the two diagonal
    blocks with the off-diagonal ones zero -- the same shape
    ``initial_ns_noncollinear`` builds for a moment along z.

    **This used to raise for every ``nspin = 4`` target**, naming a blocker
    P62b removed: ``ns_nc`` is implemented and measured at 1.2e-7 Ry on
    relativistic BN. Converging a hard magnet is staged -- get a collinear
    antiferromagnet, then promote it and let the moments cant -- and that route
    was closed for anything with a HUBBARD card.
    """
    setup = _Setup()
    ns = np.arange(2 * 2 * 3 * 3, dtype=float).reshape(2, 2, 3, 3)
    promoted = np.asarray(promote_ns(
        _result(np.zeros((1, 2, 2, 2)), 2, ns=ns), _Hubbard(setup, nspin=4)))
    assert promoted.shape == (4, 2, 3, 3)
    assert np.iscomplexobj(promoted)
    assert np.allclose(promoted[0], ns[0])       # uu
    assert np.allclose(promoted[3], ns[1])       # dd
    assert np.allclose(promoted[1], 0.0)         # ud
    assert np.allclose(promoted[2], 0.0)         # du

    # 1 -> 4 puts the same block in both diagonal slots, as 1 -> 2 does.
    single = ns[:1]
    both = np.asarray(promote_ns(
        _result(np.zeros((1, 2, 2, 2)), 1, ns=single), _Hubbard(setup, nspin=4)))
    assert np.allclose(both[0], single[0]) and np.allclose(both[3], single[0])


def test_a_spinor_ns_passes_through_unchanged_which_is_the_checkpoint_resume():
    """4 -> 4 is the case that broke the P76 restart.

    The old refusal was gated on the *target's* ``nspin`` alone, so it caught a
    noncollinear run resuming from its own checkpoint: ``checkpoint.py`` rebuilds
    an ``SCFResult`` carrying ``ns`` and the driver sends it through
    ``continued_state``. A wall-clock-killed noncollinear DFT+U run could not
    restart from its own ``checkpoint_dir``, which is exactly the long run the
    checkpointing was written for.
    """
    setup = _Setup()
    rng = np.random.default_rng(11)
    ns = rng.normal(size=(4, 2, 3, 3)) + 1j * rng.normal(size=(4, 2, 3, 3))
    same = promote_ns(_result(np.zeros((1, 2, 2, 2)), 4, ns=ns),
                      _Hubbard(setup, nspin=4))
    assert np.allclose(np.asarray(same), ns)


def test_demoting_a_spinor_ns_keeps_the_diagonal_and_refuses_a_canted_shell():
    """4 -> 2 is exact only while the shell is polarised along z.

    Taking the diagonal of a canted occupation matrix drops the transverse spin
    blocks, which is a different state rather than a coarser one -- so it is
    refused by name instead.
    """
    setup = _Setup()
    ns = np.zeros((4, 2, 3, 3), dtype=complex)
    ns[0] = np.eye(3) * 0.8
    ns[3] = np.eye(3) * 0.2
    back = np.asarray(promote_ns(_result(np.zeros((1, 2, 2, 2)), 4, ns=ns),
                                 _Hubbard(setup, nspin=2)))
    assert back.shape == (2, 2, 3, 3)
    assert not np.iscomplexobj(back)
    assert np.allclose(back[0], np.real(ns[0])) and np.allclose(back[1], np.real(ns[3]))

    ns[1] = ns[2] = np.eye(3) * 0.3
    with pytest.raises(NotImplementedError, match="off the axis z"):
        promote_ns(_result(np.zeros((1, 2, 2, 2)), 4, ns=ns),
                   _Hubbard(setup, nspin=2))


def test_a_promoted_ns_points_where_the_target_asks_and_not_along_z():
    """The promotion is a rotation, not only a reshape, and it was not.

    ``promote_density`` and ``promote_becsum`` both turn the source's
    magnetization onto the axis ``angle1``/``angle2`` name, and ``promote_ns``
    dropped the two collinear channels into the two diagonal spin blocks
    regardless -- a moment along ``z`` on a shell whose density had just been
    laid along ``x``. Measured on fcc nickel (``U = 4.0`` eV, the converged
    collinear ferromagnet carried into a noncollinear run with ``angle1 = 90``):
    the density crossed with **0.491 mu_B along x** and the shell arrived with
    **0.383 along z**, worth **30.7 mRy** of Hubbard splitting on the wrong
    axis, 8 iterations against 2 to turn it -- and with ``mixing_fixed_ns = 10``
    the run converged and reported success with the shell still on ``z``,
    **4.11 mRy** above the right answer on a cell whose anisotropy is exactly
    zero.

    The cross-check is ``initial_ns_noncollinear``, which is what a *fresh* run
    of the same target builds, on a direction with ``angle2 = 90`` so that
    ``m_y`` is the whole of the transverse moment: ``sigma_x`` is symmetric, so
    an axis in the ``xz`` plane cannot tell a transposed pack from a correct
    one, and this is the object where that transpose has already been the bug
    once (``test_the_seeded_occupation_matrix_is_qes_own_init_ns_nc``).
    """
    from defumat.hubbard.occupations import initial_ns, initial_ns_noncollinear
    from defumat.scf.continuation import _SpinTransfer

    setup = types.SimpleNamespace(
        nslot=2, types=(0, 0), atoms=(0, 1), ldmx=5, noncolin=False,
        species=(types.SimpleNamespace(ldim=5, occupation=8.0),),
    )
    theta, phi = np.deg2rad(90.0), np.deg2rad(90.0)
    axis = (float(np.sin(theta) * np.cos(phi)), float(np.sin(theta) * np.sin(phi)),
            float(np.cos(theta)))
    collinear = np.asarray(initial_ns(setup, 2, [0.5]))
    target = _Hubbard(setup, nspin=4)
    promoted = np.asarray(promote_ns(
        _result(np.zeros((1, 2, 2, 2)), 2, ns=collinear), target,
        _SpinTransfer(source=2, target=4, mode="carry", direction=axis)))
    fresh = np.asarray(initial_ns_noncollinear(
        setup, starting_magnetization=[0.5],
        angle1=[90.0], angle2=[90.0]))

    # Hund's rule gives the same two fillings on both routes, so the promotion
    # of a fresh collinear start *is* the fresh spinor start -- which is the
    # statement that the continuation is no longer the odd path out.
    np.testing.assert_allclose(promoted, fresh, atol=1e-15)
    assert abs(promoted[1, 0, 0, 0].imag) > 0.1, "m_y is the whole moment here"

    # ...and along z it is still the two diagonal blocks, to a round-off that
    # the decomposition costs and nothing else does.
    flat = np.asarray(promote_ns(
        _result(np.zeros((1, 2, 2, 2)), 2, ns=collinear), target,
        _SpinTransfer(source=2, target=4, mode="carry", direction=None)))
    assert np.abs(flat[0] - collinear[0]).max() < 3.0e-16
    assert np.abs(flat[3] - collinear[1]).max() < 3.0e-16
    assert np.abs(flat[1]).max() == 0.0 and np.abs(flat[2]).max() == 0.0


def test_demoting_a_shell_collinear_off_z_follows_the_density_axis():
    """The same gap read backwards, and it was a refusal rather than a number.

    ``_collinear_axis`` accepts a density collinear along any axis and rotates
    it onto ``z``, while the occupation matrix's own test measured the
    transverse pair in the *laboratory* frame -- so a DFT+U demotion from a
    state collinear along ``x`` was refused where the identical non-Hubbard one
    succeeded. The axis is the density's, sign included, so the shell's "up" is
    the density's up.
    """
    from defumat.hubbard.occupations import spinor_ns_from_components
    from defumat.scf.continuation import _SpinTransfer

    setup = _Setup()
    up = np.diag([0.9, 0.8, 0.7])[None] * np.ones((2, 1, 1))
    down = np.diag([0.3, 0.2, 0.1])[None] * np.ones((2, 1, 1))
    axis = (1.0, 0.0, 0.0)
    ns = np.asarray(spinor_ns_from_components(
        up + down, (up - down) * np.asarray(axis)[:, None, None, None]))
    transfer = _SpinTransfer(source=4, target=2, mode="carry", project=axis)

    back = np.asarray(promote_ns(_result(np.zeros((1, 2, 2, 2)), 4, ns=ns),
                                 _Hubbard(setup, nspin=2), transfer))
    assert not np.iscomplexobj(back)
    np.testing.assert_allclose(back[0], up, atol=1e-15)
    np.testing.assert_allclose(back[1], down, atol=1e-15)

    # Without the axis it is the state this whole entry is about -- a shell
    # lying off z -- and it is refused by name, as it was before.
    with pytest.raises(NotImplementedError, match="off the axis z"):
        promote_ns(_result(np.zeros((1, 2, 2, 2)), 4, ns=ns),
                   _Hubbard(setup, nspin=2))

    # And a genuinely canted shell is refused even with the axis, naming it.
    canted = np.asarray(spinor_ns_from_components(
        up + down, (up - down) * np.asarray([0.6, 0.0, 0.8])[:, None, None, None]))
    with pytest.raises(NotImplementedError, match=r"off the axis \(1.0000"):
        promote_ns(_result(np.zeros((1, 2, 2, 2)), 4, ns=canted),
                   _Hubbard(setup, nspin=2), transfer)


def test_a_demotion_that_drops_the_moment_does_not_care_where_it_pointed():
    """The same asymmetry one face further in, and it had no test at all.

    A one-channel target has nowhere to put a magnetization and
    ``magnetization='none'`` has just decided not to keep one, so both throw the
    whole thing away -- and the density's own ``_SpinTransfer.apply`` does it
    without a word on either path. The occupation matrix refused instead,
    measuring an axis on a moment that was about to be discarded, so a DFT+U
    demotion of a canted shell into ``nspin = 1`` raised where the identical
    non-Hubbard one ran. There was no ``4 -> 1`` ``ns`` test in the file, which
    is why.

    What has to survive is the *charge*, which is the part worth carrying.
    """
    from defumat.hubbard.occupations import spinor_ns_from_components
    from defumat.scf.continuation import _SpinTransfer

    setup = _Setup()
    up = np.diag([0.9, 0.8, 0.7])[None] * np.ones((2, 1, 1))
    down = np.diag([0.3, 0.2, 0.1])[None] * np.ones((2, 1, 1))
    canted = np.asarray(spinor_ns_from_components(
        up + down, (up - down) * np.asarray([0.6, 0.0, 0.8])[:, None, None, None]))
    result = _result(np.zeros((1, 2, 2, 2)), 4, ns=canted)

    single = np.asarray(promote_ns(result, _Hubbard(setup, nspin=1),
                                   _SpinTransfer(4, 1, mode="none")))
    assert single.shape == (1, 2, 3, 3) and not np.iscomplexobj(single)
    np.testing.assert_allclose(single[0], 0.5 * (up + down), atol=1e-15)

    flattened = np.asarray(promote_ns(result, _Hubbard(setup, nspin=2),
                                      _SpinTransfer(4, 2, mode="none")))
    assert flattened.shape == (2, 2, 3, 3)
    np.testing.assert_allclose(flattened[0], flattened[1], atol=0.0)
    np.testing.assert_allclose(flattened[0], 0.5 * (up + down), atol=1e-15)

    # A z-polarised shell goes the same way it always did, to the last bit:
    # ``(Re uu + Re dd)/2`` and ``Re(uu + dd)/2`` are the same number.
    straight = np.asarray(spinor_ns_from_components(
        up + down, (up - down) * np.asarray([0.0, 0.0, 1.0])[:, None, None, None]))
    old_route = np.mean(np.real(np.stack([straight[0], straight[3]])), axis=0,
                        keepdims=True)
    now = np.asarray(promote_ns(
        _result(np.zeros((1, 2, 2, 2)), 4, ns=straight), _Hubbard(setup, nspin=1),
        _SpinTransfer(4, 1, mode="none")))
    assert np.array_equal(now, old_route)


class _Setup:
    nslot = 2
    ldmx = 3
    noncolin = False


class _Hubbard:
    """The three attributes :func:`promote_ns` reads off a calculation."""

    def __init__(self, setup, nspin):
        import copy
        # ``ns_components`` reads ``noncolin`` off the setup, and a spinor
        # target is a spinor setup -- the two cannot disagree in a real run.
        self.hubbard = copy.copy(setup)
        self.hubbard.noncolin = nspin == 4
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
    from defumat.scf import run_scf

    with pytest.raises(ValueError, match="two states at once"):
        run_scf(_silicon(), (read_upf(PSEUDO),),
                starting_from=ContinuedState(density=np.zeros((1, 2, 2, 2))),
                starting_density=np.zeros((1, 2, 2, 2)))


# --------------------------------------------------------------------------
# The electron count


def test_a_density_carrying_the_wrong_number_of_electrons_is_refused():
    """The check the structure comparison beside it cannot make.

    That one sums the *target's* ``z_valence`` over the *source's* atom types,
    which is ``nelec`` again whenever the atoms are the same -- so it sees a
    changed structure and never a changed dataset. A dataset is what actually
    changes here: switching spin-orbit coupling on means a fully-relativistic
    file, and a file with a semicore shell in it carries a different
    ``z_valence`` on the same atoms.
    """
    calculation = _calculation(2, (0.4,))
    grid = tuple(calculation.basis.dense.grid)
    source = _random_density(grid, 1, calculation=calculation)
    # Two electrons more, which is the smallest a swapped dataset ever differs
    # by and is eight orders above what a converged density's own count drifts.
    source = source * ((calculation.nelec + 2.0) / calculation.nelec)
    with pytest.raises(ValueError, match="integrates to"):
        continued_state(_result(source, 1), calculation, wavefunctions=False)


def test_a_converged_density_passes_the_electron_count_comfortably():
    """The guard has to not fire on the case it will see every time.

    Measured on real runs rather than on this synthetic one: ``int n(r) dr``
    comes out at 40.00000000000003 against 40 on ten-atom norm-conserving
    silicon and 10.000000000000002 against 10 on platinum PAW, augmentation
    charge included, so the tolerance sits ten orders above the noise.
    """
    calculation = _calculation(2, (0.4,))
    grid = tuple(calculation.basis.dense.grid)
    source = _random_density(grid, 1, calculation=calculation)
    state = continued_state(_result(source, 1), calculation, wavefunctions=False)
    assert state.density.shape[0] == 2


# --------------------------------------------------------------------------
# The spin spiral, which is a change of *frame* and not of regime


def _spiral(q=(0.0, 0.0, 0.5), magnetization=(0.5,), angle1=(90.0,)) -> Calculation:
    """A silicon spin spiral: ``nspin = 4``, no symmetry, moments in-plane.

    ``angle1 = 90`` is the planar spiral; ``angle1 = 0`` is the case the guard
    refuses, a moment on the axis the spiral turns about.
    """
    import dataclasses

    system = _silicon(nspin=4, starting_magnetization=magnetization,
                      angle1=angle1, angle2=(0.0,))
    system = dataclasses.replace(system, spiral_q=q, nosym=True)
    return Calculation(system, (read_upf(PSEUDO),))


def test_a_vector_magnetization_does_not_cross_into_a_spirals_frame():
    """Its transverse pair is measured in a frame that turns, and ours does not."""
    spiral = _spiral()
    plain = _calculation(4, (0.5,), ((90.0,), (0.0,)))
    grid = tuple(spiral.basis.dense.grid)
    source = _random_density(grid, 4, calculation=spiral)
    with pytest.raises(NotImplementedError, match="turns with q"):
        continued_state(_result(source, 4, nspin_mag=4, system=plain.system),
                        spiral, wavefunctions=False)


def test_a_spirals_magnetization_does_not_cross_out_of_its_frame_either():
    """The same refusal read the other way round, which is the demotion."""
    spiral = _spiral()
    plain = _calculation(4, (0.5,), ((90.0,), (0.0,)))
    grid = tuple(plain.basis.dense.grid)
    source = _random_density(grid, 4, calculation=plain)
    with pytest.raises(NotImplementedError, match="turns with q"):
        continued_state(_result(source, 4, nspin_mag=4, system=spiral.system),
                        plain, wavefunctions=False)


def test_a_moment_on_the_spiral_axis_is_refused_as_the_ferromagnet_it_is():
    """``z`` is the axis the spiral turns about, so a moment on it does not turn.

    The state is then stationary at every ``q``: the run converges, reports a
    moment, and has computed the ferromagnet. **The refusal is at the door**,
    where it catches a run started from scratch as well as a continued one, so
    what is checked here is that the spiral cannot be built at all -- a moment
    on the axis, and no moment, which is the same stationary point.
    """
    for label, moments in (("on the axis", (0.0,)), ("no moment", None)):
        with pytest.raises(NotImplementedError, match="rotation axis"):
            if moments is None:
                _spiral(magnetization=(0.0,), angle1=(90.0,))
            else:
                _spiral(angle1=moments)


def test_the_planar_spiral_the_guard_exists_to_allow_is_built():
    """The case that must keep working, and the one every spiral input uses."""
    assert _spiral(angle1=(90.0,)).spiral


def test_a_texture_card_is_what_a_scalar_magnetization_is_laid_along():
    """The direction comes from the atoms, not from the per-species angles.

    A ``STARTING_MOMENTS`` card overrides ``angle1``/``angle2``, which are then
    both zero -- so reading the angles said "along z" for a target whose card
    says otherwise, and a carried scalar went in along an axis the run was
    built not to use.
    """
    from defumat.scf.continuation import _common_direction

    along_x = _textured(((0.4, 0.0, 0.0), (0.4, 0.0, 0.0)))
    assert _common_direction(along_x) == pytest.approx((1.0, 0.0, 0.0))


def test_a_collinear_moment_crosses_into_a_spiral_when_the_angles_are_in_plane():
    """The case that must keep working, and it is the useful one.

    A collinear source has no transverse pair to misread: what crosses is one
    scalar field, laid along the target's own ``angle1``, which for a planar
    spiral is the seed a spiral run wants.
    """
    spiral = _spiral(angle1=(90.0,))
    collinear = _calculation(2, (0.4,))
    grid = tuple(spiral.basis.dense.grid)
    source = _random_density(grid, 2, calculation=spiral)
    state = continued_state(_result(source, 2, system=collinear.system), spiral,
                            wavefunctions=False)
    _, moment = spin_components(state.density, 4)
    assert not np.allclose(np.asarray(moment[0]), 0.0)
    assert np.allclose(np.asarray(moment[2]), 0.0)


def test_one_spiral_continues_into_another_at_a_different_wavevector():
    """Both sides in the same rotated frame, which is what a ``q`` sweep is."""
    source_run = _spiral(q=(0.0, 0.0, 0.5))
    target = _spiral(q=(0.0, 0.0, 0.25))
    grid = tuple(target.basis.dense.grid)
    source = _random_density(grid, 4, calculation=target)
    state = continued_state(
        _result(source, 4, nspin_mag=4, system=source_run.system), target,
        wavefunctions=False,
    )
    assert state.magnetization == "carry"


def test_a_source_that_cannot_say_which_frame_it_used_says_so():
    """Silence is not agreement: the frame question has no answer here."""
    spiral = _spiral()
    grid = tuple(spiral.basis.dense.grid)
    source = _random_density(grid, 4, calculation=spiral)
    with pytest.warns(RuntimeWarning, match="rotated frame cannot"):
        continued_state(_result(source, 4, nspin_mag=4), spiral,
                        wavefunctions=False)


# --------------------------------------------------------------------------
# A magnetization that was held by something


def _held(calculation, **fields) -> Calculation:
    import dataclasses

    return dataclasses.replace(calculation.system, **fields)


def test_a_constrained_source_says_so_when_its_moment_is_released():
    """The workflow is right and it is the silence that is not.

    Hold the moment, converge, let go: the moment being carried is then the
    constraint's answer rather than the functional's, and the run starts on a
    state it will converge away from.
    """
    calculation = _calculation(2, (0.4,))
    grid = tuple(calculation.basis.dense.grid)
    source = _random_density(grid, 2, calculation=calculation)
    held = _held(calculation, constrained_magnetization="atomic")
    with pytest.warns(RuntimeWarning, match="rather than the functional's"):
        continued_state(_result(source, 2, system=held), calculation,
                        wavefunctions=False)


def test_a_target_under_the_same_constraint_is_not_warned_about():
    """Nothing is being released, so there is nothing to say."""
    import warnings as _warnings

    calculation = _calculation(2, (0.4,))
    grid = tuple(calculation.basis.dense.grid)
    source = _random_density(grid, 2, calculation=calculation)
    held = _held(calculation, constrained_magnetization="atomic")
    stand_in = _Stand(held, calculation)
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        continued_state(_result(source, 2, system=held), stand_in,
                        wavefunctions=False)


def test_a_field_that_decays_to_nothing_is_not_a_held_magnetization():
    """``reducebf`` multiplies the field away, so the density it leaves is free.

    The response stack refuses such a state for a different reason, that it
    rebuilds its potential from the *input* field; here there is nothing to
    release and a warning would be noise on every symmetry-broken start.
    """
    import warnings as _warnings

    calculation = _calculation(2, (0.4,))
    grid = tuple(calculation.basis.dense.grid)
    source = _random_density(grid, 2, calculation=calculation)
    decayed = _held(calculation, b_field=(0.0, 0.0, 0.01), reducebf=0.5)
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        continued_state(_result(source, 2, system=decayed), calculation,
                        wavefunctions=False)


# --------------------------------------------------------------------------
# ``magnetization`` reaching the front door


def test_magnetization_without_a_seed_is_refused_rather_than_ignored():
    """It says how a continuation crosses, and there is no continuation."""
    from defumat.scf import run_scf

    calculation = _calculation(2, (0.4,))
    with pytest.raises(ValueError, match="no continuation here"):
        run_scf(calculation.system, (read_upf(PSEUDO),),
                calculation=calculation, magnetization="seed")


def test_magnetization_beside_a_continued_state_is_refused_as_a_second_answer():
    """The state has already resolved the question; two answers is one too many."""
    from defumat.scf import run_scf

    calculation = _calculation(2, (0.4,))
    grid = tuple(calculation.basis.dense.grid)
    state = continued_state(
        _result(_random_density(grid, 1, calculation=calculation), 1),
        calculation, wavefunctions=False,
    )
    assert isinstance(state, ContinuedState)
    with pytest.raises(ValueError, match="already resolved"):
        run_scf(calculation.system, (read_upf(PSEUDO),), calculation=calculation,
                starting_from=state, magnetization="seed")


def test_with_moments_seeds_where_with_spin_carries():
    """The default is the whole difference between the two methods.

    A new texture is a new ``STARTING_MOMENTS`` card, and carrying a
    noncollinear source's moment across applies no rotation -- so ``"auto"``,
    which resolves to carry whenever the source has a moment, would start the
    run on the configuration the new card was written to leave.
    """
    from defumat.calculator import Calculator

    calculation = _calculation(2, (0.4,))
    calculator = Calculator(calculation.system, (read_upf(PSEUDO),),
                            announce=False)
    calculator._scf = _result(
        _random_density(tuple(calculation.basis.dense.grid), 2,
                        calculation=calculation),
        2, system=calculation.system,
    )

    assert calculator.with_spin(4)._seed_magnetization == "auto"
    assert calculator.with_moments(
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]))._seed_magnetization == "seed"
    assert calculator.with_spin(4, magnetization="none")._seed_magnetization == "none"


def test_a_deformed_cell_is_refused_by_the_cell_and_not_by_the_count():
    """A strain leaves the grid alone and changes what the density integrates to.

    The two are worth separating because the electron count cannot: a per-cent
    strain usually leaves the FFT dimensions unchanged, so the shape check
    passes, and the integral would then read the wrong number of electrons and
    blame a pseudopotential that did not change.
    """
    import dataclasses

    calculation = _calculation(2, (0.4,))
    grid = tuple(calculation.basis.dense.grid)
    source = _random_density(grid, 1, calculation=calculation)
    # 0.5 per cent, which this cell keeps at a 16^3 grid where 1 per cent moves
    # it to 20^3 and would be caught by the shape check instead.
    strained = calculation.system.with_cell(
        np.asarray(calculation.system.cell.at) * 1.005)
    assert tuple(Calculation(strained, (read_upf(PSEUDO),)).basis.dense.grid) == grid
    with pytest.raises(ValueError, match="different cells"):
        continued_state(_result(source, 1, system=strained), calculation,
                        wavefunctions=False)


def _seeded_calculator(calculation):
    """A derived calculator holding a converged parent state as its seed."""
    from defumat.calculator import Calculator

    parent = Calculator(calculation.system, (read_upf(PSEUDO),), announce=False)
    parent._scf = _result(
        _random_density(tuple(calculation.basis.dense.grid), 1,
                        calculation=calculation),
        1, system=calculation.system,
    )
    return parent


def _record_run_scf(monkeypatch, calculation):
    """Replace ``run_scf`` where the calculator looks it up, and record calls."""
    calls = []

    def fake_run_scf(system, pseudos, **options):
        calls.append(options)
        return _result(
            _random_density(tuple(calculation.basis.dense.grid), 1,
                            calculation=calculation),
            1, system=calculation.system,
        )

    monkeypatch.setattr("defumat.calculator.run_scf", fake_run_scf)
    return calls


def test_a_seeded_calculator_reads_the_checkpoint_it_has_been_writing(
        monkeypatch, tmp_path):
    """A checkpoint is later state than the seed, and used to be invisible.

    ``run_scf`` reads its own ``checkpoint_dir`` only when nothing was passed as
    ``starting_from``, and a derived calculator inserts the parent's state there
    on the caller's behalf -- so a job killed at its wall clock and resubmitted
    restarted from the seed every time and never read the file it had written.
    """
    from defumat.scf.driver import SCF_CHECKPOINT

    calculation = _calculation(1)
    calculator = _seeded_calculator(calculation).with_spin(2)
    calls = _record_run_scf(monkeypatch, calculation)

    calculator.get_scf(checkpoint_dir=tmp_path)
    assert calls[-1]["starting_from"] is not None, (
        "an empty directory has nothing to continue, so the seed still goes in")

    (tmp_path / SCF_CHECKPOINT).write_bytes(b"")
    calculator._scf = None
    calculator.get_scf(checkpoint_dir=tmp_path)
    assert "starting_from" not in calls[-1]
    assert "magnetization" not in calls[-1], (
        "both keys are withheld together: a resume refuses a magnetization")


def test_the_seed_and_its_magnetization_are_withheld_together(
        monkeypatch, tmp_path):
    """``with_moments`` defaults to ``'seed'``, which a resume refuses by name.

    Withholding ``starting_from`` alone would turn a resubmitted texture run
    from one that silently restarts into one that raises.
    """
    from defumat.scf.driver import SCF_CHECKPOINT

    calculation = _calculation(1)
    calculator = _seeded_calculator(calculation).with_moments(
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]))
    assert calculator._seed_magnetization == "seed"
    calls = _record_run_scf(monkeypatch, calculation)

    (tmp_path / SCF_CHECKPOINT).write_bytes(b"")
    calculator.get_scf(checkpoint_dir=tmp_path)
    assert "magnetization" not in calls[-1]

    # And a caller who asks for one anyway still reaches ``run_scf``'s refusal,
    # which is where that argument is decided.
    calculator._scf = None
    calculator.get_scf(checkpoint_dir=tmp_path, magnetization="seed")
    assert calls[-1]["magnetization"] == "seed"


def test_a_checkpoint_left_behind_does_not_cost_the_cache(monkeypatch, tmp_path):
    """The cache key is the options, and the seed is not one of them.

    A converged run does not delete its last checkpoint, so whether the seed is
    inserted depends on a file that appears halfway through the calculator's
    life. Keying on it would make the second ``get_scf`` miss and rerun the
    whole SCF, against a mid-run state at that.
    """
    from defumat.scf.driver import SCF_CHECKPOINT

    calculation = _calculation(1)
    calculator = _seeded_calculator(calculation).with_spin(2)
    calls = _record_run_scf(monkeypatch, calculation)

    first = calculator.get_scf(checkpoint_dir=tmp_path)
    (tmp_path / SCF_CHECKPOINT).write_bytes(b"")
    assert calculator.get_scf(checkpoint_dir=tmp_path) is first
    assert len(calls) == 1


# --- the Hubbard occupation matrix follows the density's decision ---------------


LDAU = "tests/data/qe/ni-ldau-j0.in"
LDAU_PSEUDO = "tests/data/pseudo/Ni.pz-nd-rrkjus.UPF"


def _hubbard_ns(calculation, source_split: float):
    """A ``(2, nslot, ldmx, ldmx)`` matrix with a known spin splitting."""
    hubbard = calculation.hubbard
    shape = (hubbard.nslot, hubbard.ldmx, hubbard.ldmx)
    up = np.broadcast_to(np.eye(hubbard.ldmx) * 0.9, shape)
    down = np.broadcast_to(np.eye(hubbard.ldmx) * (0.9 - source_split), shape)
    return np.stack([up, down])


@lru_cache(maxsize=1)
def _ldau_calculation() -> Calculation:
    """A real ``nspin = 2`` DFT+U calculation -- built, never converged."""
    if not pathlib.Path(LDAU_PSEUDO).is_file():
        pytest.skip("the Ni ultrasoft dataset is not present")
    system = build_system(read_pw_input(pathlib.Path(LDAU)))
    return Calculation(system, (read_upf(LDAU_PSEUDO),))


def _result_with_ns(ns) -> SCFResult:
    """An :class:`SCFResult` carrying an ``ns`` and the shape beside it."""
    return _result(np.zeros((2, 2, 2, 2)), 2, ns=np.asarray(ns))


@pytest.mark.parametrize("mode,expected", [
    ("carry", "kept"), ("none", "flat"), ("seed", None),
])
def test_ns_follows_the_magnetization_decision(mode, expected):
    """It was the one member of the mixed triple not handed the transfer.

    ``promote_density`` and ``promote_becsum`` both take the ``_SpinTransfer``
    and ``promote_ns`` did not, so ``magnetization='none'`` and
    ``magnetization='seed'`` reseeded or zeroed the density and left ``ns``
    carrying the source's full spin polarization. Iteration 1 then
    diagonalised a Hamiltonian with a Hubbard splitting of order ``U`` on a
    density that was exactly spin-degenerate or the new texture's, and
    ``mixing_fixed_ns`` freezes ``ns_state`` there for as many iterations as it
    is set to. Measured on ``ni-ldau-j0.in`` (``U = 3.0`` eV, converged
    ferromagnetic at 0.69 mu_B): the matrix that crossed had
    ``max|ns_up - ns_dn| = 0.129291``, worth **28.5 mRy** of splitting in the
    Hubbard potential, against zero after.

    ``'seed'`` returns ``None`` rather than an average, because there is no way
    to lay a density texture onto per-site occupations here that ``init_ns``
    does not do better from the *target's* own ``starting_magnetization`` --
    which is exactly what ``run_scf`` does with a ``None``.
    """
    from defumat.scf.continuation import _SpinTransfer

    calculation = _ldau_calculation()
    ns = _hubbard_ns(calculation, 0.4)
    transfer = _SpinTransfer(source=2, target=2, mode=mode)
    out = promote_ns(_result_with_ns(ns), calculation, transfer)

    if expected is None:
        assert out is None
        return
    out = np.asarray(out)
    splitting = np.abs(out[0] - out[1]).max()
    if expected == "kept":
        assert splitting == pytest.approx(0.4)
    else:
        assert splitting == pytest.approx(0.0, abs=1e-14)
        # the converged *charge* is what is worth carrying, and it survives
        assert np.trace(out[0, 0]) + np.trace(out[1, 0]) == pytest.approx(
            np.trace(ns[0, 0]) + np.trace(ns[1, 0]))


def test_ns_with_no_transfer_is_unchanged():
    """``transfer=None`` is the old signature and must still mean 'carry'."""
    calculation = _ldau_calculation()
    ns = _hubbard_ns(calculation, 0.4)
    out = np.asarray(promote_ns(_result_with_ns(ns), calculation))
    np.testing.assert_allclose(out, ns)


# --- ...and so does the kinetic energy density -------------------------------


@pytest.mark.parametrize("nspin_mag,expected", [(1, "kept"), (2, "flat"), (4, "trace")])
def test_tau_is_depolarized_in_its_own_storage(nspin_mag, expected):
    """``tau`` is **not** stored the way the density is, which is the trap here.

    At ``nspin_mag = 2`` it is ``(up, down)`` -- ``sum_band.f90`` converts
    ``rho`` to ``(total, magnetization)`` at the end and leaves ``kin_r`` alone,
    and ``potinit.f90`` says so in a comment -- while at ``nspin_mag = 4`` it
    *is* on the Pauli basis, ``(tau, tau_x, tau_y, tau_z)``. So sending it
    through ``_SpinTransfer.apply``, which reads the density's convention,
    would be wrong at two channels and right at four.

    What it is worth: on ``h-fcc-magnon.in`` under ``input_dft = 'tb09'`` at
    the input's own mixing, the converged-to-0.53-mu_B state carries
    ``max|tau_up - tau_dn| = 0.0223`` against a ``tau`` whose maximum is
    ``0.0340`` -- **65 per cent** -- and that used to cross a
    ``magnetization='none'`` continuation onto a density made exactly
    spin-degenerate. (That cell does not fully converge under ``tb09`` at the
    default iteration count, which does not change the size of the splitting.)
    A potential-only meta-GGA reads ``tau`` straight into ``v_x`` rather than
    through an energy, so the first Hamiltonian is split by a magnetization the
    density does not have.
    """
    from defumat.scf.continuation import depolarize_tau

    grid = (4, 4, 4)
    rng = np.random.default_rng(7)
    tau = rng.uniform(0.1, 1.0, (nspin_mag,) + grid)
    out = np.asarray(depolarize_tau(tau))
    source = np.asarray(tau)

    assert out.shape == source.shape
    if expected == "kept":
        np.testing.assert_array_equal(out, source)
    elif expected == "flat":
        np.testing.assert_allclose(out[0], out[1], atol=1e-15)
        # the total is what survives, which is the part worth carrying
        assert out.sum() == pytest.approx(source.sum(), rel=1e-12)
    else:
        np.testing.assert_allclose(out[0], source[0], atol=1e-15)  # the trace
        assert np.abs(out[1:]).max() == 0.0                        # the vector


def test_tau_crosses_untouched_when_the_magnetization_does():
    """``'carry'`` must not lose the guess the shape test exists to keep."""
    from defumat.scf.continuation import depolarize_tau

    tau = np.random.default_rng(3).uniform(0.1, 1.0, (2, 4, 4, 4))
    # the driver only calls it for 'none'; this pins that the identity branch
    # is the one channel and not a silent flattening of two.
    np.testing.assert_array_equal(
        np.asarray(depolarize_tau(tau[:1])), np.asarray(tau[:1]))


# --- with_positions and the FFT grid -------------------------------------------


def _silicon_calculator(text=None):
    from defumat.calculator import Calculator

    if not SILICON.is_file():
        pytest.skip("QE reference tree not present")
    if text is None:
        text = SILICON.read_text()
    return Calculator.from_text(text, "tests/data/pseudo", announce=False)


def test_with_positions_freezes_the_fft_grid():
    """A 0.02 bohr displacement used to change the grid under the seed.

    The grid is a function of the **symmetry**, not only of the cutoffs:
    ``symm_base.f90`` requires its dimensions to be a multiple of the
    fractional translations' denominators, implemented here as
    ``fft_factors``. On the canonical silicon cell a displacement of 0.02 bohr
    takes ``nsym`` from 48 to 4, ``fft_factors`` from ``(4, 4, 4)`` to
    ``(1, 1, 1)`` and the dense grid from ``(16, 16, 16)`` to
    ``(15, 15, 15)``.

    Rebuilding was wrong in both branches. With a converged parent the seed
    this method promises to carry was refused by ``_check_grid`` -- *"the
    source density is on a (16, 16, 16) grid and this run uses (15, 15, 15)"*
    -- for a displacement that changed neither the cell nor either cutoff.
    Without one it ran silently at the displaced grid, so the undisplaced
    reference and the displaced run were on different grids.
    """
    calculator = _silicon_calculator()
    grid = calculator.calculation.basis.dense.grid
    positions = np.asarray(calculator.system.structure.positions).copy()
    positions[1, 0] += 0.02
    moved = calculator.with_positions(positions)
    assert len(moved.system.symmetry_group().rotation_array()) < 48
    assert moved.calculation.basis.dense.grid == grid


def test_with_positions_rebuilds_where_a_frozen_grid_would_be_unsound():
    """...and freezing is conditional, in the direction the method is not for.

    Moving an atom **onto** a more symmetric site gives the target a group
    whose fractional translations the parent's grid may not be able to
    represent -- a calculator built directly on a displaced silicon has
    ``nsym = 4`` and a ``(15, 15, 15)`` grid, and the ideal site needs
    ``fft_factors = (4, 4, 4)``, which 15 is not a multiple of. Freezing there
    would hand ``sym_rho`` operations the grid cannot carry, so that case
    rebuilds and the seed check then raises honestly.
    """
    text = SILICON.read_text().replace("Si 0.25 0.25 0.25", "Si 0.30 0.25 0.25")
    calculator = _silicon_calculator(text)
    assert calculator.calculation.basis.dense.grid == (15, 15, 15)

    ideal = np.asarray(calculator.system.structure.positions).copy()
    ideal[1] = np.array([0.25, 0.25, 0.25]) * float(calculator.system.cell.alat)
    back = calculator.with_positions(ideal)
    assert back.system.symmetry_group(back.system.nosym).fft_factors() == (4, 4, 4)
    assert back.calculation.basis.dense.grid == (16, 16, 16)
