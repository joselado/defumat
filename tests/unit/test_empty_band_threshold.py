"""QE's per-band diagonalisation threshold, and the step counter beside it.

``cegterg`` does not converge every band to the same accuracy. A band whose
fractional occupation has fallen below 0.01 is "empty" for the eigensolver's
purposes (``PW/src/sum_band.f90:118-128``) and is tested against
``empty_ethr = max(5 ethr, 1e-5)`` instead of against ``ethr``
(``KS_Solvers/Davidson/cegterg.f90:129,556-563``). The physics that licenses it
is that an empty state carries no charge: the density, the total energy and
everything derived from them are blind to it, so converging it as hard as an
occupied state is work spent on a number nothing reads.

That is also what these tests have to show. Fewer steps is easy to get by
loosening something that matters; the claim being made is the narrow one -- the
*occupied* eigenvalues, and the energy they build, are where they were. So the
check is against :func:`tests.exact_reference.exact_eigenpairs`, which forms
``H`` and calls ``eigh``: it splits "the Hamiltonian is right" from "the
eigensolver converged" in one step, where comparing the solver against itself
cannot.
"""

import dataclasses
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf.driver import (
    EMPTY_BAND_OCCUPATION,
    ETHR_INIT,
    Calculation,
    band_thresholds,
    run_scf,
)
from defumat.scf.potential import v_of_rho
from defumat.solvers.davidson import (
    DAVID_NDIM,
    EMPTY_ETHR_FLOOR,
    _every_k,
    davidson_eigensolver_all,
    empty_band_threshold,
)
from defumat.system import build_system
from tests.exact_reference import exact_eigenpairs_all

pytestmark = pytest.mark.unit

BENCHMARK = Path(__file__).resolve().parents[2] / "benchmarks" / "si-1k.in"

#: Four filled bands and six empty ones, which is what puts the two thresholds
#: in the same run. Smearing rather than fixed occupations, because that is the
#: case where a band's occupation is a number rather than 0 or 1.
NBND = 10
FILLED = 4


@pytest.fixture(scope="module")
def silicon(pseudo_dir):
    """The metallic silicon cell, its pseudopotentials, and one Hamiltonian."""
    system = build_system(read_pw_input(BENCHMARK))
    system = dataclasses.replace(system, occupations="smearing",
                                 degauss=0.02, smearing="mv")
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file)
                    for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    potential = v_of_rho(calculation.starting_density(), calculation.basis.dense,
                         system.cell)
    return system, pseudos, calculation.hamiltonian(potential.v_scf)[0]


# --------------------------------------------------------------------------
# The rule itself: sum_band.f90's btype and cegterg.f90's empty_ethr.
# --------------------------------------------------------------------------

def test_the_first_scf_iteration_holds_every_band_to_ethr():
    """``btype`` comes out of ``init_run.f90:149`` all ones, and stays so.

    Nothing has been diagonalised yet, so there are no occupations to call a
    band empty with. This is why the first iteration is the control experiment
    for the whole feature: it must be identical with the switch either way.
    """
    thresholds = band_thresholds(3e-4, None, None, shape=(2, 5, 8))
    assert np.asarray(thresholds) == pytest.approx(np.full((2, 5, 8), 3e-4))


def test_diago_full_acc_restores_the_single_threshold_exactly():
    """``sum_band.f90:123``'s ``IF ( .NOT. diago_full_acc )``, and nothing else.

    Every band is empty in this ``wg``, so with the switch off the whole array
    would be ``empty_ethr``; with it on it is ``ethr`` to the last bit, which is
    the behaviour this package had before the vector existed.
    """
    weights = jnp.asarray([0.5, 0.5])
    wg = jnp.zeros((1, 2, 3))
    on = band_thresholds(1e-9, wg, weights, diago_full_acc=True)
    off = band_thresholds(1e-9, wg, weights, diago_full_acc=False)
    assert np.array_equal(np.asarray(on), np.full((1, 2, 3), 1e-9))
    assert np.array_equal(np.asarray(off), np.full((1, 2, 3), EMPTY_ETHR_FLOOR))


def test_an_empty_band_is_loosened_and_an_occupied_one_is_not():
    """The 0.01 threshold is on ``wg / wk``, not on ``wg``.

    QE's k-point weight carries the spin degeneracy, so the ratio is the
    fractional occupation and the literal 0.01 means the same thing in every
    spin regime. Here ``wk`` is 0.25, so an occupancy of 0.5 is a ``wg`` of
    0.125 -- smaller than 0.01 and still an occupied band.
    """
    weights = jnp.asarray([0.25, 0.25])
    occupancies = jnp.asarray([1.0, 0.5, EMPTY_BAND_OCCUPATION, 0.009, 0.0])
    wg = weights[None, :, None] * occupancies[None, None, :]

    ethr = 1e-9
    thresholds = np.asarray(band_thresholds(ethr, wg, weights))
    loose = empty_band_threshold(ethr)
    assert thresholds[0, 0].tolist() == [ethr, ethr, ethr, loose, loose]
    # The comparison is strict ``<``, so a band sitting exactly on 0.01 is
    # occupied (``cegterg`` inherits ``sum_band``'s ``< 0.01D0``).
    assert thresholds[0, 0, 2] == ethr


def test_a_bare_per_k_array_is_refused_rather_than_sliced_wrong(silicon):
    """``diagonalize`` slices the leading axis, so the rank has to be the one
    it promises.

    A ``(nk, nbnd)`` array looks plausible and would be indexed on its k axis
    as though it were a spin axis -- one k-point's thresholds applied to every
    k-point, with no error and no way to see it in the answer.
    """
    system, pseudos, hamiltonian = silicon
    calculation = Calculation(system, pseudos)
    with pytest.raises(ValueError, match="wrong axis"):
        calculation.diagonalize(
            [hamiltonian], NBND, None, jnp.full((hamiltonian.nk, NBND), 1e-6)
        )


def test_a_zero_weight_kpoint_keeps_full_accuracy():
    """``FORALL( ik = 1:nks, wk(ik) > 0.D0 )`` -- ``sum_band.f90:125``.

    An ``nscf`` run can carry k-points of zero weight whose bands are the whole
    point of it (``non_scf.f90:107-110``). Their ``wg`` is zero, so an unmasked
    ratio would call every one of them empty -- and the division itself must not
    happen, since a ``NaN`` that compares false is not what the Fortran says.
    """
    weights = jnp.asarray([0.5, 0.0])
    wg = jnp.zeros((1, 2, 3))
    thresholds = np.asarray(band_thresholds(1e-9, wg, weights))
    assert np.isfinite(thresholds).all()
    assert thresholds[0, 0].tolist() == [EMPTY_ETHR_FLOOR] * 3
    assert thresholds[0, 1].tolist() == [1e-9] * 3


def test_empty_ethr_has_a_floor_and_the_floor_is_what_bites():
    """``empty_ethr = MAX( ( ethr * 5.D0 ), 1.D-5 )`` -- ``cegterg.f90:129``.

    Five times a loose ``ethr`` is still loose, so early in an SCF the two
    thresholds barely differ. It is late, once ``ethr`` has fallen below 2e-6,
    that the constant takes over -- and at QE's ``ethr`` floor of 1e-13 the two
    differ by eight orders of magnitude. Writing ``5 ethr`` alone gives a
    plausible small speedup and misses most of the effect.
    """
    assert empty_band_threshold(1e-2) == pytest.approx(5e-2)
    assert empty_band_threshold(2e-6) == pytest.approx(1e-5)
    assert empty_band_threshold(1e-13) == EMPTY_ETHR_FLOOR
    assert empty_band_threshold(1e-13) / 1e-13 == pytest.approx(1e8)


# --------------------------------------------------------------------------
# The vector reaching the solver.
# --------------------------------------------------------------------------

def test_a_constant_vector_is_the_scalar_bit_for_bit(silicon):
    """The threshold is data on the traced ``ethr`` slot, not a new algorithm.

    Three routes to the same answer, and they must agree to the last bit,
    because a run whose bands are all occupied -- every insulator at
    ``nbnd = nelec/2`` -- takes the third and must reproduce what it always
    produced.

    The middle one is the measurement rather than the argument. A scalar handed
    to :func:`davidson_eigensolver_all` is broadcast on the host before it
    reaches the compiled unit, so comparing that against an array is comparing
    the new path with itself; calling the compiled unit with a genuine ``()``
    aval takes the ``jnp.ndim(ethr) < 2`` branch, which *is* the code as it was.
    """
    _, _, hamiltonian = silicon
    scalar, _ = davidson_eigensolver_all(hamiltonian, NBND, None, ethr=1e-6,
                                         max_iterations=60)
    unbroadcast, _ = _every_k(hamiltonian, NBND, None, 1e-6, None, DAVID_NDIM,
                              60, "default", False)
    vector, _ = davidson_eigensolver_all(
        hamiltonian, NBND, None, ethr=jnp.full((hamiltonian.nk, NBND), 1e-6),
        max_iterations=60,
    )
    assert np.array_equal(np.asarray(scalar), np.asarray(unbroadcast))
    assert np.array_equal(np.asarray(scalar), np.asarray(vector))


def test_a_loosened_band_is_the_only_one_that_loses_accuracy(silicon):
    """One band held to 1e-2 while the rest are held to 1e-13.

    This is the per-band test doing per-band work: the loosened root stops
    being expanded while its neighbours keep going, and the error lands on it
    alone. Measured against the dense solve, not against another Davidson run.
    """
    _, _, hamiltonian = silicon
    exact, _ = exact_eigenpairs_all(hamiltonian, NBND)

    thresholds = np.full((hamiltonian.nk, NBND), 1e-13)
    thresholds[:, NBND - 1] = 1e-2
    values, _ = davidson_eigensolver_all(hamiltonian, NBND, None,
                                         ethr=jnp.asarray(thresholds),
                                         max_iterations=60)
    error = np.abs(np.asarray(values) - np.asarray(exact))
    assert error[:, : NBND - 1].max() < 1e-9
    assert error[:, NBND - 1].max() > 1e3 * error[:, : NBND - 1].max()


def test_the_solver_reports_its_step_count_and_what_it_left_behind(silicon):
    """``final[14]`` and ``final[13]``, which the loop already carried.

    They have to be read together. A count at the budget with a small
    ``notcnv`` is one straggler; with a large one it is a stall; and a *short*
    count can also mean the eigenvalues stopped being finite. Reporting only
    the count invites the wrong reading of all three.
    """
    _, _, hamiltonian = silicon
    plain = davidson_eigensolver_all(hamiltonian, NBND, None, ethr=1e-8,
                                     max_iterations=60)
    assert len(plain) == 2, "the two-value form must be what it always was"

    values, vectors, steps, unsettled = davidson_eigensolver_all(
        hamiltonian, NBND, None, ethr=1e-8, max_iterations=60, return_steps=True
    )
    steps, unsettled = np.asarray(steps), np.asarray(unsettled)
    assert steps.shape == unsettled.shape == (hamiltonian.nk,)
    assert np.array_equal(np.asarray(values), np.asarray(plain[0]))
    assert (steps > 0).all() and (steps < 60).all()
    assert (unsettled == 0).all(), "it converged, so nothing is left unsettled"

    # A tighter threshold is more steps, which is what says the number counts
    # the loop rather than something incidental.
    _, _, tighter, _ = davidson_eigensolver_all(
        hamiltonian, NBND, None, ethr=1e-13, max_iterations=60, return_steps=True
    )
    assert np.asarray(tighter).sum() > steps.sum()

    # Seeded with the answer, the loop makes at most one pass -- the budget is
    # a ceiling on a cheap solve, not a cost it always pays.
    exact, exact_vectors = exact_eigenpairs_all(hamiltonian, NBND)
    _, _, seeded, _ = davidson_eigensolver_all(
        hamiltonian, NBND, exact_vectors, ethr=1e-8, max_iterations=60,
        return_steps=True,
    )
    assert np.asarray(seeded).max() <= 1


# --------------------------------------------------------------------------
# The correctness claim: the density must not move.
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_runs(silicon):
    """The same metallic cell with the empty-band threshold on and off."""
    system, pseudos, _ = silicon
    return {
        flag: run_scf(system, pseudos, nbnd=NBND, conv_thr=1e-10,
                      diago_full_acc=flag)
        for flag in (False, True)
    }


def test_the_occupied_eigenvalues_are_the_exact_ones_with_the_feature_on(
        silicon, two_runs):
    """The claim that matters: a loose empty band must not move the density.

    Diagonalising the converged potential densely is the independent check --
    it shares the Hamiltonian with the SCF and none of the eigensolver. The two
    tolerances are deliberately different and both are QE's own logic: the
    occupied bands are what the density is built from and are held to ``ethr``,
    while the empty ones are held to ``empty_ethr`` **on the change between
    consecutive estimates**, which bounds their error only weakly -- so they are
    checked for being roughly right, not for being converged.
    """
    system, pseudos, _ = silicon
    result = two_runs[False]
    calculation = Calculation(system, pseudos)
    potential = v_of_rho(result.density, calculation.basis.dense, system.cell)
    hamiltonian = calculation.hamiltonian(potential.v_scf)[0]
    exact, _ = exact_eigenpairs_all(hamiltonian, NBND)

    values = np.asarray(result.eigenvalues)
    exact = np.asarray(exact)
    assert values[..., :FILLED] == pytest.approx(exact[..., :FILLED], abs=1e-6)
    assert values[..., FILLED:] == pytest.approx(exact[..., FILLED:], abs=1e-3)


def test_the_total_energy_does_not_notice(two_runs):
    """``INPUT_PW.txt:2375``: "should not affect ... ground-state properties"."""
    loose, full = two_runs[False], two_runs[True]
    assert loose.converged and full.converged
    assert float(loose.total_energy) == pytest.approx(
        float(full.total_energy), abs=1e-9
    )
    assert np.asarray(loose.eigenvalues)[..., :FILLED] == pytest.approx(
        np.asarray(full.eigenvalues)[..., :FILLED], abs=1e-6
    )


def test_the_first_iteration_is_the_control_and_the_rest_is_the_saving(two_runs):
    """``pw.x``'s own experiment, on a cell that fits in a unit test.

    On the 157-atom slab ``PERFORMANCE.md`` reports iteration 1 bit-identical
    between the two settings and iteration 2 going 8.5 steps to 12.5 when the
    threshold is switched off -- a factor of 1.47. The same two statements are
    what is asserted here: the control, because the first iteration has no
    occupations either way, and then fewer steps overall.
    """
    loose, full = two_runs[False], two_runs[True]

    first_loose, first_full = loose.history[0], full.history[0]
    # ...and the control is only a control if the ``ethr``-too-large retry did
    # not fire, which would give iteration 1 a second pass with real
    # occupations (``electrons.f90:890-908``). It leaves ``ethr`` moved off
    # ``ETHR_INIT``, so checking the starting value is checking the
    # precondition rather than assuming it.
    assert first_loose["ethr"] == ETHR_INIT
    assert first_full["ethr"] == ETHR_INIT
    assert first_loose["davidson_iterations"] == first_full["davidson_iterations"]
    assert first_loose["total_energy"] == pytest.approx(
        first_full["total_energy"], abs=1e-10
    )

    total_loose = sum(h["davidson_iterations"] for h in loose.history)
    total_full = sum(h["davidson_iterations"] for h in full.history)
    assert total_loose < total_full


def test_the_scf_history_carries_the_step_count(two_runs):
    """Per iteration, in ``pw.x``'s form: ``avg_iter / nkstot``.

    ``c_bands.f90:159`` divides by the number of k-points *times* spin
    channels, because QE stores a collinear run as a k-list of twice the
    length. The array here is ``(nspin, nk)`` and the mean over both axes is
    the same quantity.
    """
    for entry in two_runs[False].history:
        assert entry["davidson_iterations"] > 0
        assert isinstance(entry["davidson_unconverged"], int)
    assert two_runs[False].history[-1]["davidson_unconverged"] == 0


def test_a_fixed_density_run_converges_every_band_tightly(silicon, two_runs):
    """``non_scf.f90:63`` diagonalises before any ``sum_band``, so ``btype`` is
    all ones -- and it must stay so here.

    This is the calculation whose *purpose* is the empty bands. Deriving the
    threshold from occupations everywhere would loosen exactly the states a band
    structure is asked for, and a comparison against a QE reference would then
    fail in the conduction bands only, which reads as a physics bug rather than
    a solver setting. So the conduction bands of a fixed-density run are held to
    the same tolerance as the valence ones -- against the dense solve, on the
    same potential.
    """
    from defumat.workflows.nscf import fixed_density_states

    system, pseudos, _ = silicon
    result = two_runs[False]
    calculation, _, eigenvalues, _ = fixed_density_states(
        system, pseudos, result.density, nbnd=NBND, conv_thr=1e-10,
    )
    potential = v_of_rho(result.density, calculation.basis.dense, system.cell)
    exact, _ = exact_eigenpairs_all(
        calculation.hamiltonian(potential.v_scf)[0], NBND
    )
    assert np.asarray(eigenvalues)[0] == pytest.approx(np.asarray(exact), abs=1e-7)


# --------------------------------------------------------------------------
# The input variable.
# --------------------------------------------------------------------------

def test_the_input_file_switch_is_read_as_a_logical(tmp_path):
    """``diago_full_acc`` is a Fortran logical in ``&electrons``.

    It arrives from the generic namelist parser already a Python ``bool``, and
    the conversion table has to say so: without an arm of its own it falls into
    the string branch and becomes ``"True"`` -- truthy either way, and the wrong
    type to hand to :func:`~defumat.scf.driver.run_scf`.
    """
    import inspect

    from defumat.calculator import (
        SCF_ONLY_OPTIONS,
        SHARED_OPTIONS,
        electrons_defaults,
    )

    text = BENCHMARK.read_text().replace(
        "&electrons", "&electrons\n    diago_full_acc = .true.", 1
    )
    path = tmp_path / "full-acc.in"
    path.write_text(text)

    adopted = electrons_defaults(read_pw_input(path))
    assert adopted["diago_full_acc"] is True

    # ...and it reaches the SCF, and stops there: a band-structure or response
    # run has no occupations, so an option that loosens empty bands would name
    # something that does nothing.
    assert "diago_full_acc" in SHARED_OPTIONS
    assert "diago_full_acc" in SCF_ONLY_OPTIONS
    assert "diago_full_acc" in inspect.signature(run_scf).parameters
