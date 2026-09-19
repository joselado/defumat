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


def _calculation(case: str):
    """A real ``Calculation``, which is what the guard chain reads."""
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(
        read_upf(PSEUDO / sp.pseudo_file) for sp in system.structure.species
    )
    return Calculation(system, pseudos)


@pytest.mark.parametrize("case, message", [
    ("si2-us", "ultrasoft"),
    ("al-metal", "metal"),
    ("o-atom-fixed-lsda", "nspin = 2"),
    # A one-atom hydrogen cell, not the germanene slab this used to build:
    # the guard reads a ``Calculation`` and refuses on ``noncolin`` alone, so
    # the cell is incidental to what is asserted -- and constructing the slab
    # cost 3.6 GB and 11 s for a refusal that fires identically here at 189 MB
    # and 1.3 s. Both raise the same message, checked rather than assumed.
    ("h-atom-noncolin", "noncollinear"),
])
def test_the_regimes_this_was_never_run_in_are_refused(case, message):
    """Every one of these would return a number, and none of them is measured.

    The guard chain is deliberately made of the *bare* forms: the linear
    response solver runs for a metal and for two spin channels, and this
    assembly on top of it has been run with neither, so the flags that would
    say otherwise are not passed. An ultrasoft dataset is the interesting one --
    nothing in the assembly is norm-conserving, and what is missing is a
    non-centrosymmetric ultrasoft crystal to measure it on, since a
    centrosymmetric one agrees with zero whatever is wrong.
    """
    with pytest.raises(NotImplementedError, match=message):
        require_a_piezoelectric_tensor(_calculation(case))


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
