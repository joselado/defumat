"""The parts of the piezoelectric tensor that need no self-consistent field.

The Voigt convention and the polar-crystal guard, both of which are decisions
rather than computations -- and both of which are silent when wrong. A factor
of two in the first makes every published ``e_14`` disagree by two with no
symmetry saying so; the second is the difference between the improper mixed
derivative this phase computes and the proper piezoelectric response a
measurement sees.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.response.piezo import (
    VOIGT,
    polar_direction,
    require_a_nonpolar_crystal,
    require_a_piezoelectric_tensor,
    to_voigt,
)
from defumat.scf import Calculation
from defumat.system import build_system

pytestmark = [pytest.mark.unit]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


def _crystal(case: str):
    """A stand-in for a calculation: the guard reads only the crystal.

    :func:`~defumat.response.piezo.polar_direction` searches the symmetries of
    the structure rather than reading the run's -- deliberately, since a
    response is usually run with ``nosym`` and that list would call every
    crystal polar -- so nothing here needs pseudopotentials or a converged
    state.
    """
    return SimpleNamespace(system=build_system(read_pw_input(CASES / f"{case}.in")))


def test_voigt_carries_no_factor_of_two():
    """``e_iJ = e_(i)jk``: the engineering two is on the strain, not on this.

    ``P_i = sum_jk e_(i)jk eps_jk`` runs over all nine pairs, and Voigt's
    ``eps_4 = 2 eps_23`` absorbs the doubling of the two equal shear terms --
    so the coefficient is untouched. It is
    :class:`~defumat.response.elastic.ElasticConstants`' convention one rank
    down, and the check is that a contraction gives the same answer in both.
    """
    rng = np.random.default_rng(0)
    e = rng.normal(size=(3, 3, 3))
    e = 0.5 * (e + e.transpose(0, 2, 1))
    strain = rng.normal(size=(3, 3))
    strain = 0.5 * (strain + strain.T)

    full = np.einsum("kij,ij->k", e, strain)
    voigt_strain = np.array([
        strain[i, j] if i == j else 2 * strain[i, j] for i, j in VOIGT
    ])
    assert np.allclose(to_voigt(e) @ voigt_strain, full)


def test_a_cubic_crystal_admits_no_spontaneous_polarization():
    """``-43m`` and ``m-3m`` both average their rotations to zero.

    A polarization has to be invariant under every operation of the point
    group, so the group average is the projector onto the directions one may
    point along. Zincblende AlAs and diamond silicon both give zero, which is
    what makes the improper-to-proper correction vanish for them.
    """
    for case in ("alas-raman", "si-electrostriction"):
        assert np.abs(polar_direction(_crystal(case))).max() < 1e-10
        require_a_nonpolar_crystal(_crystal(case))


def _input(case: str):
    """The ``System`` and its pseudopotentials: all the guard chain reads."""
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(
        read_upf(PSEUDO / sp.pseudo_file) for sp in system.structure.species
    )
    return system, pseudos


def _calculation(case: str):
    """A real ``Calculation``, for the guards that are still asked of one."""
    return Calculation(*_input(case))


@pytest.fixture
def no_calculation(monkeypatch):
    """Make building a ``Calculation`` fail the test outright.

    ``pytest.fail`` raises an outcome rather than an ``Exception``, so neither
    a ``pytest.raises(NotImplementedError)`` nor a broad ``except`` inside the
    package can swallow it: a refusal that reaches for a calculation is caught
    here even when it would otherwise have gone on to raise the right message.
    """
    def refuse(*args, **kwargs):
        pytest.fail("a Calculation was built to refuse a regime the input decides")

    monkeypatch.setattr(Calculation, "__init__", refuse)


@pytest.mark.parametrize("case, message", [
    ("si2-us", "ultrasoft"),
    ("al-metal", "metal"),
    ("o-atom-fixed-lsda", "nspin = 2"),
    # A one-atom hydrogen cell, not the germanene slab this once used: the
    # guard refuses on ``noncolin`` alone, so the cell is incidental to what is
    # asserted. The slab was replaced while each case still built a
    # ``Calculation`` (3.6 GB and 11 s against 189 MB and 1.3 s); with the
    # input alone the cost of the cell no longer enters.
    ("h-atom-noncolin", "noncollinear"),
])
def test_the_regimes_this_was_never_run_in_are_refused(case, message,
                                                       no_calculation):
    """Every one of these would return a number, and none of them is measured.

    The guard chain is deliberately made of the *bare* forms: the linear
    response solver runs for a metal and for two spin channels, and this
    assembly on top of it has been run with neither, so the flags that would
    say otherwise are not passed. The ultrasoft case is no longer refused for
    being ultrasoft -- that half was lifted once the ladder measured it against
    a Berry-phase value -- and is refused here for its **mesh**, this cell
    carrying a grid below :data:`~defumat.response.piezo.ULTRASOFT_MESH`;
    :func:`test_an_ultrasoft_dataset_is_refused_by_its_mesh_and_paw_outright`
    is where that distinction is asserted rather than incidental.

    **Asked of the input, with building a calculation made to fail.** Each
    case used to construct a whole ``Calculation`` -- the G sphere, both FFT
    grids, ``vkb`` and, for the ultrasoft cell, ``Q_ij(G)`` -- so that the
    guard could read four flags and a symmetry off it (``OPEN.md`` Part III
    X3). What the guard reads is a property of the ``System`` and the
    pseudopotentials, and ``no_calculation`` is what says it no longer reaches
    for anything else.
    """
    system, pseudos = _input(case)
    with pytest.raises(NotImplementedError, match=message):
        require_a_piezoelectric_tensor(system, pseudos=pseudos)


def test_the_input_reads_the_way_the_calculation_does():
    """The guard's view of a ``System`` is the ``Calculation``'s, flag by flag.

    The refusals above are asked of the input, and they are only the same
    refusals if every attribute the chain reads comes out the same as the
    constructor's. This is the one test that builds a calculation to say so,
    on the committed cell where most of those flags are *not* at their
    default -- a PAW dataset, so ``is_paw`` and ``is_ultrasoft`` both hold,
    ``nspin = 2``, ``nosym``, and ``K_POINTS gamma``, which the constructor
    substitutes away for an augmented dataset and the view deliberately does
    not. A parity check on a cell where everything is ``False`` on both sides
    would pass whatever either side computed.
    """
    from defumat.response.piezo import _Regime

    system, pseudos = _input("o2-paw-afm")
    view = _Regime.of(system, pseudos)
    calculation = Calculation(system, pseudos)

    assert view.is_paw and view.is_ultrasoft and view.nspin == 2
    assert system.kpoints.gamma_only and not view.gamma_only
    for name in ("nspin", "noncolin", "spiral", "gamma_only",
                 "two_fermi_energies", "is_ultrasoft", "is_paw", "is_hubbard"):
        assert getattr(view, name) == getattr(calculation, name), name
    assert (view.magnetic_field is None) == (calculation.magnetic_field is None)
    assert view.functional.name == calculation.functional.name
    assert view.functional.is_meta == calculation.functional.is_meta
    assert view.system.nosym == calculation.system.nosym
    assert np.array_equal(view.symmetries.rotation_array(),
                          calculation.symmetries.rotation_array())
    # What the guards read off the k-set is untouched by the substitution the
    # view skips, which is the whole argument for skipping it.
    for field in ("grid", "shift"):
        assert getattr(view.system.kpoints, field) == getattr(
            calculation.system.kpoints, field
        ), field
    assert view.system.kpoints.nk == calculation.system.kpoints.nk


def test_a_system_without_its_pseudopotentials_is_not_guessed_at():
    """Half the chain reads the datasets, so a bare ``System`` is refused loudly.

    And a ``Calculation`` passed *with* pseudopotentials is refused too, since
    it already carries the ones it was built with and two sources for one
    answer is how the two come apart.
    """
    system, pseudos = _input("alas-raman")
    with pytest.raises(TypeError, match="pseudopotentials"):
        require_a_piezoelectric_tensor(system)
    with pytest.raises(TypeError, match="not both"):
        require_a_piezoelectric_tensor(
            SimpleNamespace(system=system), pseudos=pseudos
        )


def test_the_calculator_refuses_before_its_implicit_ground_state(no_calculation):
    """The facade asks the input first, so a metal costs no SCF to be refused.

    ``get_piezoelectric_tensor`` runs a ground state when none is cached, and
    the refusal used to live only inside the entry point it then called: an
    aluminium crystal ran its whole self-consistent field and was told
    afterwards that the tensor of a metal was never measured. With building a
    calculation made to fail, the old order fails this test and the new one
    raises the refusal before anything is built.
    """
    from defumat import Calculator

    system, pseudos = _input("al-metal")
    calculator = Calculator(system, pseudos, announce=False)
    with pytest.raises(NotImplementedError, match="metal"):
        calculator.get_piezoelectric_tensor()


def test_the_ladder_refuses_before_its_first_rung(no_calculation):
    """The same for the k-mesh ladder, which is a ground state per rung.

    Asked of the first rung rather than of the input's own k-set, because
    every rung is unshifted whatever the input asked for.
    """
    from defumat.workflows.piezo_ladder import piezoelectric_kmesh_ladder

    system, pseudos = _input("al-metal")
    with pytest.raises(NotImplementedError, match="metal"):
        piezoelectric_kmesh_ladder(system, pseudos, meshes=(2,))


def test_an_unknown_route_is_refused_before_anything_is_solved():
    """``method`` names a route and a typo must not quietly give the default.

    The two routes are the same number at very different cost -- the
    transcribed one carries no tape at all, which is what makes it the one to
    reach for on a dense mesh -- so a caller who asked for it and silently got
    the other would get the right answer and a peak they had chosen against.
    The check is before the field response rather than after it, because the
    response is the expensive part and there is nothing to learn from solving
    it first.
    """
    from defumat.response.piezo import PIEZOELECTRIC_METHODS, piezoelectric_tensor

    assert PIEZOELECTRIC_METHODS == ("autodiff", "zstar_eu")
    with pytest.raises(ValueError, match="unknown piezoelectric method"):
        piezoelectric_tensor(_calculation("alas-raman"), None, method="zstar-eu")


def test_the_transcribed_route_refuses_paw_and_no_longer_refuses_ultrasoft():
    """``zstar_eu`` refuses **PAW**, and *separately* from the tensor's own guard.

    The two refusals are about different things and the test is that lifting
    one does not lift the other: :func:`require_a_measured_dataset` says the
    quantity has never been measured against an independent reference on an
    augmented dataset, and
    :func:`require_a_norm_conserving_transcription` says whether this assembly
    is the one to measure it with.

    **Ultrasoft used to be refused here and is not any more.**
    ``zstar_eu.f90`` hands an augmented dataset to ``zstar_eu_us.f90``, and what
    those three hundred lines are worth in the strain coordinate turned out to
    be two contractions, because only one leg of this derivative moves ``S``.
    With both in, the two routes agree to 2.6e-09 C/m^2 on ultrasoft AlAs at
    ``ecutwfc = 12`` and 1.7e-07 at the ``ecutwfc = 10`` the test cell is
    committed at, where they were 1.6 per cent apart at both. Those numbers are
    ``test_piezoelectric_augmented.py``'s. What is still refused is PAW, whose one-centre
    energy is a function of ``becsum`` directly, so the cross term the grid
    integral misses is on no grid at all.
    """
    from defumat.response.piezo import (
        require_a_measured_dataset,
        require_a_norm_conserving_transcription,
    )

    ultrasoft = _calculation("si2-us")
    # Runs, where it used to raise.
    require_a_norm_conserving_transcription(ultrasoft)
    require_a_norm_conserving_transcription(_calculation("alas-raman"))
    with pytest.raises(NotImplementedError, match="PAW"):
        require_a_norm_conserving_transcription(_calculation("o2-paw-afm"))
    # ... and the dataset refusal is untouched by any of that: it still refuses
    # the ultrasoft cell this route now accepts, which is the whole reason the
    # two are separate functions.
    with pytest.raises(NotImplementedError, match="ultrasoft"):
        require_a_measured_dataset(ultrasoft)


def test_the_kmesh_guard_fires_and_says_what_the_mesh_was():
    """The guard for the one parameter no check inside this quantity can see.

    **Testing that it fires rather than that it is quiet**, which is
    ``CLAUDE.md``'s rule and is what a guard of this kind needs: the whole
    reason it exists is that a coarse mesh looks exactly like a converged one
    from inside -- the three routes share a field response, the symmetry
    statements hold on any mesh, and the ``Z*`` anchor is the same assembly in
    another coordinate -- so nothing downstream could tell a silent guard from
    a working one.

    Three states, and the middle one is the one worth having: no ladder at all
    warns and quotes AlAs's curve; a ladder whose last step is above
    :data:`~defumat.response.piezo.KMESH_STEP` warns and quotes **that** number
    instead; and a ladder below it says nothing.
    """
    from defumat.response.piezo import KMESH_STEP, _warn_about_the_kmesh

    calculation = _calculation("alas-raman")
    nk = calculation.system.kpoints.nk

    with pytest.warns(RuntimeWarning, match="k-convergence has not been measured"):
        _warn_about_the_kmesh(calculation, None)
    with pytest.warns(RuntimeWarning, match=f"integrated over {nk} k-points"):
        _warn_about_the_kmesh(calculation, None)
    with pytest.warns(RuntimeWarning, match="moved the tensor by 5.0 per cent"):
        _warn_about_the_kmesh(calculation, 0.05)

    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        _warn_about_the_kmesh(calculation, KMESH_STEP / 2)


def test_the_kmesh_the_result_reports_is_the_one_that_was_integrated():
    """``nk`` and ``grid`` come off the k-set rather than off the input file.

    A run may be handed a set the input never mentioned -- a ladder rung, a
    substituted grid, an explicit list -- and what the number was integrated
    over is then the only honest thing to report. ``grid`` is ``None`` for a
    list, which is not a failure: an explicit set has no divisions.
    """
    from defumat.response.piezo import _kmesh_of

    calculation = _calculation("alas-raman")
    nk, grid = _kmesh_of(calculation)
    assert nk == calculation.system.kpoints.nk
    assert grid is None or len(grid) == 3


def test_the_ladder_refuses_an_order_that_would_read_as_a_drift():
    """Coarsest first, because the last step *is* the result.

    A ladder given its meshes the other way round would report the step from
    the dense rung to the coarse one and call it the drift, which is a number
    with the right magnitude and the wrong meaning -- the kind of thing nothing
    downstream can catch. Refused at the door, with the repeats refused beside
    it, since a repeated mesh makes a step of exactly zero and would read as
    perfect convergence.
    """
    from defumat.workflows.piezo_ladder import piezoelectric_kmesh_ladder

    system = build_system(read_pw_input(CASES / "alas-raman.in"))
    pseudos = tuple(
        read_upf(PSEUDO / sp.pseudo_file) for sp in system.structure.species
    )
    with pytest.raises(ValueError, match="coarsest first"):
        piezoelectric_kmesh_ladder(system, pseudos, meshes=(6, 4))
    with pytest.raises(ValueError, match="repeat"):
        piezoelectric_kmesh_ladder(system, pseudos, meshes=(4, 4))
    with pytest.raises(ValueError, match="at least one mesh"):
        piezoelectric_kmesh_ladder(system, pseudos, meshes=())


def test_an_ultrasoft_dataset_is_refused_by_its_mesh_and_paw_outright():
    """The refusal after the ladder: PAW always, ultrasoft below a measured mesh.

    **What was measured and what it licenses.** The ladder on zincblende AlAs
    puts an ultrasoft ``e_14`` 1.26 per cent from a Berry-phase finite
    difference at ``8 8 8`` and 0.57 at ``10 10 10``, against the
    norm-conserving calibration's 1.62 and 1.19 on the same meshes -- so from
    eight divisions on, an augmented dataset is nearer an independent reference
    than the route this package validates against ``ph.x``. Below that the
    dataset and the mesh are not separable: the same cell reads 15.6 per cent
    out at ``4 4 4``, almost all of it k-convergence, which is why a coarse
    ultrasoft run is refused rather than warned about while a coarse
    norm-conserving one is warned about rather than refused.

    PAW keeps the whole refusal, and the difference is the kind of evidence
    that exists for each: the PAW wedge completion is measured against **its own
    closed grid**, which is an internal identity, and no PAW crystal has been
    compared with anything outside this code.

    The stand-in is a namespace rather than a ``Calculation`` because what the
    guard reads is three flags and a grid, and building a real 8x8x8 calculation
    to assert a refusal costs 512 k-points' worth of basis for nothing.
    """
    from defumat.response.piezo import (
        KMESH_STEP,
        ULTRASOFT_MESH,
        require_a_measured_dataset,
    )

    def stand_in(grid, paw=False):
        nk = 1 if grid is None else grid[0] * grid[1] * grid[2]
        return SimpleNamespace(
            is_ultrasoft=True, is_paw=paw,
            system=SimpleNamespace(kpoints=SimpleNamespace(nk=nk, grid=grid)),
        )

    with pytest.raises(NotImplementedError, match="denser"):
        require_a_measured_dataset(stand_in((4, 4, 4)))
    with pytest.raises(NotImplementedError, match="explicit k-point list"):
        require_a_measured_dataset(stand_in(None))
    with pytest.raises(NotImplementedError, match="PAW"):
        require_a_measured_dataset(stand_in((10, 10, 10), paw=True))

    # And the three ways through: a dense enough grid, a measured ladder below
    # the step threshold, and the ladder's own exemption for its coarse rungs.
    require_a_measured_dataset(stand_in((ULTRASOFT_MESH,) * 3))
    require_a_measured_dataset(stand_in((4, 4, 4)), KMESH_STEP / 2)
    require_a_measured_dataset(stand_in((4, 4, 4)), allow_a_coarse_mesh=True)

    # A drift *above* the threshold is not evidence and does not open the door.
    with pytest.raises(NotImplementedError, match="denser"):
        require_a_measured_dataset(stand_in((4, 4, 4)), 10 * KMESH_STEP)

    # A norm-conserving dataset passes whatever its mesh, which is what makes
    # this a dataset refusal with a mesh condition rather than a mesh refusal.
    require_a_measured_dataset(SimpleNamespace(
        is_ultrasoft=False, is_paw=False,
        system=SimpleNamespace(kpoints=SimpleNamespace(nk=1, grid=(1, 1, 1))),
    ))
