"""The magnetocrystalline anisotropy from total energies, not band sums.

:func:`~defumat.workflows.anisotropy.run_anisotropy` freezes the density
converged without spin-orbit coupling and diagonalises once per direction, so
every term of the total energy except the band sum cancels between two
directions by construction. :func:`~defumat.workflows.anisotropy.
run_relaxed_anisotropy` does not cancel anything: it converges a whole
noncollinear run per direction and subtracts two total energies of order 100 Ry
in their eighth decimal.

The two answer slightly different questions. The relaxed one lets the density
respond to the spin-orbit field, which is variational and so lowers both
directions; the anisotropy is the difference of two such gains. **That is why
this is worth its two SCF runs**: nothing bounds the difference between the two
routes in general, and the regime the frozen one cannot reach at all -- PAW,
whose handoff would have to carry a ``becsum`` belonging to a run with a
different pseudopotential file -- is exactly where a production magnet lives.
"""

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

from defumat import Calculator
from defumat.units import RY_TO_EV

pytestmark = [pytest.mark.regression]

PSEUDO = "tests/data/pseudo"
XZ = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))


def _calculator(name):
    return Calculator.from_file(f"tests/data/qe/{name}", pseudo_dir=PSEUDO,
                                announce=False)


@pytest.mark.slow
def test_without_spin_orbit_coupling_every_direction_has_the_same_energy():
    """The identity that pins the whole route, and it is not a weak one.

    Without spin-orbit coupling the Hamiltonian commutes with a global spin
    rotation, so the total energy cannot depend on where the moment points --
    **exactly**, not approximately. Any dependence is machinery: the k-set
    moving between directions, or the gradient-corrected functional's
    quantization axis left behind while the moment is turned, which is the bug
    this same identity found on the frozen route and was worth 36.8 meV there.
    """
    from defumat.workflows.anisotropy import run_relaxed_anisotropy

    calculator = _calculator("co-tetragonal-relaxed-mae.in")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_relaxed_anisotropy(
            calculator.system, calculator.pseudos, directions=XZ,
            soc_scale=0.0, require_spin_orbit=False,
            conv_thr=1.0e-12, max_iterations=400,
        )
    assert result.converged
    spread = abs(result.difference(0, 1)) * RY_TO_EV * 1000.0
    # Measured at 1.9e-10 meV, the scalar-relativistic partner's order (the
    # test below). Until `PLAN.md` P115 it was a floor of 1.1e-2 meV that no
    # ``conv_thr`` removed: the reduction built ``becsum`` with the full
    # ``fcoef`` sandwich, which ties the spin to the orbital index.
    assert spread < 1.0e-6, (
        f"no spin-orbit coupling, so the two directions must have the same "
        f"total energy; they differ by {spread:.3e} meV"
    )
    # **The total is the guard the spread cannot be.** The reduction that left
    # that floor was not the derivative of its own energy either, and it put
    # the total at -125.698 Ry: 51 Ry below the coupled run, with 8.97 of the
    # nine electrons on the atom. Switching the coupling off costs 0.44 mRy on
    # this cell (-74.405364571 Ry against pw.x's coupled -74.40580247), so a
    # mRy is room for the coupling energy and none for that.
    for total in result.total_energies:
        assert total == pytest.approx(-74.40580247, abs=1.0e-3)


@pytest.mark.slow
def test_the_same_identity_is_exact_on_the_scalar_relativistic_partner():
    """Where the identity above really bites, and what it localises.

    The test above runs a **fully-relativistic** dataset with
    ``soc_scale = 0``, so it asserts two things at once: that the route does
    not depend on the moment direction, and that switching the coupling off in
    every ``fcoef`` sandwich leaves nothing behind. Only the first is the
    route's, and separating them is what says which one is imperfect.

    ``Co.pbe-nd-rrkjus`` is the matched scalar-relativistic partner of
    ``Co.rel-pbe-nd-rrkjus``: same element, same functional, same generation,
    and **no** ``dvan_so``, ``qq_so`` or ``fcoef`` to reduce. On it the identity
    is exact to **3.5e-09 meV**. The relativistic dataset stopped on a floor of
    **1.1e-2 meV** at every ``conv_thr`` from 1e-12 down until `PLAN.md` P115,
    and this test is what localised that floor to the reduction; it now reads
    1.9e-10 there as well.
    """
    from defumat.workflows.anisotropy import run_relaxed_anisotropy

    source = Path("tests/data/qe/co-tetragonal-relaxed-mae.in").read_text()
    scalar = (source.replace("Co.rel-pbe-nd-rrkjus.UPF", "Co.pbe-nd-rrkjus.UPF")
                    .replace("lspinorb = .true.,", "lspinorb = .false.,"))
    written = Path(tempfile.mkdtemp()) / "co-tetragonal-scalar.in"
    written.write_text(scalar)

    calculator = Calculator.from_file(written, pseudo_dir=PSEUDO, announce=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_relaxed_anisotropy(
            calculator.system, calculator.pseudos, directions=XZ,
            soc_scale=None, require_spin_orbit=False,
            conv_thr=1.0e-12, max_iterations=400,
        )
    assert result.converged
    spread = abs(result.difference(0, 1)) * RY_TO_EV * 1000.0
    assert spread < 1.0e-6, (
        f"with no spin-orbit machinery to reduce, the two directions must "
        f"agree exactly; they differ by {spread:.3e} meV"
    )


def test_paw_is_refused_by_the_frozen_route_and_allowed_by_the_relaxed_one():
    """The asymmetry is the reason this route exists, so it is asserted.

    The force theorem's handoff is a density and nothing else; a PAW
    Hamiltonian needs ``ddd_paw``, built from a ``becsum`` that belongs to the
    *other* leg's wavefunctions, with a different pseudopotential file and a
    different projector count. A relaxed run hands nothing over, so the refusal
    does not apply -- and this test fails if either half of that ever changes
    silently.
    """
    from defumat.workflows.anisotropy import _refuse_relaxed, _refuse_system

    calculator = _calculator("ni-tetragonal-relaxed-mae-paw.in")
    assert any(p.is_paw for p in calculator.pseudos), "the cell must be PAW"

    with pytest.raises(NotImplementedError, match="PAW"):
        _refuse_system(calculator.system, calculator.pseudos)

    # ... and the relaxed route accepts the identical system.
    _refuse_relaxed(calculator.system, calculator.pseudos)


def test_the_relaxed_route_refuses_what_would_make_its_total_meaningless():
    """Three refusals that are *not* inherited from the frozen route.

    Each is the same argument arriving at the same answer for its own reason,
    which `CLAUDE.md`'s "inherit a refusal only after checking which machine it
    belongs to" is about: a field's energy is outside the reported total, so two
    directions differ by a Zeeman term neither total accounts for; a spiral has
    no spin-orbit coupling to be anisotropic about; and a collinear run has no
    direction to point.
    """
    from defumat.workflows.anisotropy import _refuse_relaxed

    spiral = _calculator("h-fcc-spiral-scan.in")
    with pytest.raises(NotImplementedError, match="spiral"):
        _refuse_relaxed(spiral.system, spiral.pseudos)

    collinear = _calculator("co-tetragonal-anisotropy-sr.in")
    with pytest.raises(ValueError, match="noncolin"):
        _refuse_relaxed(collinear.system, collinear.pseudos)


def test_a_drift_is_reported_rather_than_absorbed():
    """Nothing holds the moment, so the result has to say where it ended.

    A relaxed direction that is not stationary by symmetry can converge with its
    moment somewhere else entirely, and the total energy then belongs to a state
    other than the one asked for. This checks the *reporting*, on synthetic
    results, because the physics of when a drift happens is a property of the
    crystal and not of this code.
    """
    from defumat.workflows.anisotropy import (RELAXED_DRIFT_TOL,
                                              RelaxedAnisotropy,
                                              RelaxedDirection)

    straight = RelaxedDirection(
        direction=(0.0, 0.0, 1.0), total_energy=-100.0, converged=True,
        iterations=12, accuracy=1e-11, moment=(0.0, 0.0, 1.6), drift=0.0,
        moment_length=1.6,
    )
    wandered = RelaxedDirection(
        direction=(1.0, 0.0, 0.0), total_energy=-100.001, converged=True,
        iterations=40, accuracy=1e-11, moment=(0.7, 0.0, 1.4),
        drift=63.4, moment_length=1.565,
    )
    result = RelaxedAnisotropy(
        directions=(straight.direction, wandered.direction),
        results=(straight, wandered),
    )
    assert result.drifts.tolist() == [0.0, 63.4]
    assert result.drifts[1] > RELAXED_DRIFT_TOL
    assert result.converged
    assert np.isclose(result.anisotropy, 0.001)


def test_the_drift_warning_actually_fires():
    """**Test that the guard fires**, which is this project's own rule.

    A diagnostic that returns a clean zero across the whole family it is meant
    to discriminate reads as agreement rather than as silence, and nothing
    downstream can tell the difference (``CLAUDE.md``, "a check whose null
    result cannot be told from a pass" -- five instances in one production run).
    So the drift warning is fed a case that must trip it, rather than checked
    only on cases that must not.

    The results are synthetic because the *warning* is what is under test, not
    the physics of when a moment drifts -- that is a property of the crystal.
    Monkeypatching the per-direction worker is what lets one assertion cover the
    warning, its threshold and its text.
    """
    import defumat.workflows.anisotropy as aniso
    from defumat.workflows.anisotropy import RelaxedDirection

    def fake(system, pseudos, direction=None, **kwargs):
        drift = 0.0 if direction[2] else 41.5
        return RelaxedDirection(
            direction=tuple(float(x) for x in direction),
            total_energy=-100.0 - 1e-5 * drift, converged=True, iterations=11,
            accuracy=1e-11, moment=tuple(float(x) for x in direction),
            drift=drift, moment_length=1.6,
        )

    saved = aniso.run_relaxed_direction
    try:
        aniso.run_relaxed_direction = fake
        with pytest.warns(RuntimeWarning, match="drifted more than"):
            drifted = aniso.run_relaxed_anisotropy(None, None, directions=XZ)
        # ... and it must stay quiet when nothing drifts, or it discriminates
        # nothing: a warning that always fires is the same defect as one that
        # never does.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            aniso.run_relaxed_anisotropy(
                None, None, directions=((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)))
        assert not [w for w in caught if "drifted more than" in str(w.message)]
    finally:
        aniso.run_relaxed_direction = saved

    assert drifted.drifts.max() == pytest.approx(41.5)


def test_the_unconverged_warning_actually_fires():
    """The companion guard, for the same reason.

    A relaxed anisotropy is a difference of total energies in their eighth
    decimal, so an unconverged direction does not give a slightly wrong
    anisotropy -- it gives one dominated by where the SCF happened to stop.
    """
    import defumat.workflows.anisotropy as aniso
    from defumat.workflows.anisotropy import RelaxedDirection

    def fake(system, pseudos, direction=None, **kwargs):
        return RelaxedDirection(
            direction=tuple(float(x) for x in direction), total_energy=-100.0,
            converged=False, iterations=300, accuracy=4.2e-7,
            moment=tuple(float(x) for x in direction), drift=0.0,
            moment_length=1.6,
        )

    saved = aniso.run_relaxed_direction
    try:
        aniso.run_relaxed_direction = fake
        with pytest.warns(RuntimeWarning, match="did not converge"):
            result = aniso.run_relaxed_anisotropy(None, None, directions=XZ)
    finally:
        aniso.run_relaxed_direction = saved

    assert not result.converged


@pytest.mark.slow
def test_the_relaxed_anisotropy_agrees_with_pw_x():
    """The external number the relaxed route did not have until 2026-09-23.

    Two serial ``pw.x`` 7.5 runs of the same cell, one per cardinal axis, at the
    ``conv_thr = 1e-12`` P87's number was taken at, committed as
    ``co-tetragonal-relaxed-mae-{x,z}.in`` with their outputs beside them. The
    two cardinal axes of a tetragonal crystal are stationary directions of the
    anisotropy energy, so ``pw.x`` converges each one where it was put with
    nothing holding it -- which is what makes a hand-differenced pair the
    like-for-like reference (there is no QE routine for it, README note 19).

    Measured: both totals agree to the eight decimals ``pw.x`` prints
    (-74.4057695967 against -74.40576959, -74.4058024728 against -74.40580247),
    and the anisotropy is 0.447302 meV against 0.4474. The tolerances are the
    printed precision and nothing looser: 1e-8 Ry per total, rounded, and twice
    that on the difference.
    """
    from defumat.io import read_qe_output

    references = [read_qe_output(Path(f"tests/data/qe/reference.out.co-tetragonal-relaxed-mae-{d}"))
                  for d in "xz"]
    calculator = _calculator("co-tetragonal-relaxed-mae.in")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = calculator.get_relaxed_anisotropy(
            directions=XZ, conv_thr=1.0e-12, max_iterations=400)
    assert result.converged
    for ours, theirs in zip(result.total_energies, references):
        assert ours == pytest.approx(theirs.total_energy, abs=1.0e-8)
    difference = result.difference(0, 1)
    reference = references[0].total_energy - references[1].total_energy
    assert difference == pytest.approx(reference, abs=2.0e-8)
    # The easy axis is c on both sides, which the sign carries.
    assert difference > 0.0
