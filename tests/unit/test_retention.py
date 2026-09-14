"""Nothing large may be alive while its replacement is being built.

These are guards on *reference lifetimes*, not on physics. Every one of them
protects a one-line drop that a later edit would undo without noticing, because
nothing about the code reads wrong without it: ``self._scf = run_scf(...)`` looks
like it replaces the cache, and it does -- but only once ``run_scf`` has returned,
so the previous run's wavefunctions sit under the whole of the new SCF including
its Davidson peak. The sizes are in ``PERFORMANCE.md`` (MEMORY-AUDIT A6, A7);
what is asserted here is the lifetime, which is what the size follows from.

Each test is checked to **fail** with its drop removed. They are cheap because
none of them needs the real computation: the thing being replaced is stubbed, and
what the stub asserts is what was still bound when it was entered.
"""

import numpy as np
import pytest

import defumat.calculator as calculator_module
import defumat.response.efield as efield
import defumat.scf.driver as driver
from defumat.calculator import Calculator

pytestmark = pytest.mark.unit


SILICON = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1, ecutwfc = 12.0
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""


@pytest.fixture
def silicon(pseudo_dir):
    return Calculator.from_text(SILICON, pseudo_dir, announce=False)


def test_the_cached_scf_is_dropped_before_the_new_one_runs(silicon, monkeypatch):
    """A6(ii). The cache slot holds the largest arrays in the process."""
    silicon.get_scf(max_iterations=1)
    assert silicon.scf_result is not None

    seen = {}
    original = calculator_module.run_scf

    def watched(*args, **kwargs):
        seen["scf"] = silicon._scf
        seen["strain"] = silicon._strain_response
        return original(*args, **kwargs)

    monkeypatch.setattr(calculator_module, "run_scf", watched)
    silicon.get_scf(max_iterations=2)
    assert seen["scf"] is None, (
        "the previous run's wavefunctions were live under the new SCF"
    )
    # A6(i)'s other half: the strain response was cleared *after* the new run,
    # which is the one place clearing it cannot help.
    assert seen["strain"] is None


def test_the_cached_relaxation_is_dropped_before_the_new_one_runs(silicon,
                                                                 monkeypatch):
    """A6(ii). A relaxation holds every ionic step's converged state."""
    import defumat.workflows.relax as relax_module

    seen = {}
    original = relax_module.run_relax

    def watched(*args, **kwargs):
        seen["relax"] = silicon._relax
        return "a relaxation"

    monkeypatch.setattr(relax_module, "run_relax", watched)
    silicon._relax = "the previous one"
    silicon.get_relax()
    assert seen["relax"] is None


def test_the_cached_strain_response_is_dropped_before_the_recompute(silicon,
                                                                   monkeypatch):
    """A6(i). ``options`` recomputes, and that is the double-live path."""
    import defumat.response.strain as strain_module

    seen = {}

    def watched(*args, **kwargs):
        seen["strain"] = silicon._strain_response
        return "a response"

    monkeypatch.setattr(strain_module, "strain_response", watched)
    silicon.get_scf()
    silicon._strain_response = "the previous one"
    silicon.get_strain_response(verbose=False)
    assert seen["strain"] is None


def test_the_continued_span_is_released_once_it_has_been_read(pseudo_dir,
                                                              monkeypatch):
    """A6(iv). A 1 -> 4 promotion allocates a span the size of the run's own
    wavefunctions, and the ``if wavefunctions is None`` guard opens once.

    Two halves: the span reaches the one call that reads it, and it is gone from
    ``run_scf``'s frame by the time the first diagonalisation runs -- which is
    already inside the same iteration, so every later one inherits it.
    """
    calc = Calculator.from_text(SILICON, pseudo_dir, announce=False)
    scalar = calc.get_scf(conv_thr=1e-6)

    spinor = calc.system.with_spin(4, starting_magnetization=(0.0,))
    handed, spans = [], []
    build = driver.Calculation.starting_wavefunctions
    diagonalize = driver.Calculation.diagonalize

    def watched_build(self, *args, span=None, **kwargs):
        handed.append(span)
        return build(self, *args, span=span, **kwargs)

    def watched_diagonalize(self, *args, **kwargs):
        spans.append(_run_scf_frame().f_locals["starting_wavefunctions"])
        return diagonalize(self, *args, **kwargs)

    monkeypatch.setattr(driver.Calculation, "starting_wavefunctions", watched_build)
    monkeypatch.setattr(driver.Calculation, "diagonalize", watched_diagonalize)
    driver.run_scf(spinor, calc.pseudos, conv_thr=1e-6, max_iterations=3,
                   starting_from=scalar)

    assert len(handed) == 1, "the guard opened more than once"
    assert handed[0] is not None, "the promotion did not reach the only reader"
    assert len(spans) >= 2, "needs at least two diagonalisations to say anything"
    assert all(span is None for span in spans), (
        "the promoted span was still bound after the only call that reads it"
    )


def _run_scf_frame():
    """The innermost ``run_scf`` frame on the stack."""
    import inspect

    for record in inspect.stack():
        if record.function == "run_scf":
            return record.frame
    raise AssertionError("run_scf is not on the stack")


@pytest.mark.slow
def test_the_field_keeps_its_commutators_only_for_a_reader(monkeypatch):
    """A7. Both lists are read after the loop and by nothing inside it.

    Behavioural rather than a source grep: the loop is stopped at its first
    Sternheimer solve and the lists are read out of ``dielectric_tensor``'s own
    frame, so a reformat cannot break this and a revert cannot pass it.
    """
    filled = {}

    class Stop(Exception):
        pass

    # Stopped at the top of the first self-consistent iteration, which is past
    # the loop that fills both lists and before any of the work.
    def watched(*args, **kwargs):
        frame = _frame_of("dielectric_tensor")
        filled["commutators"] = len(frame.f_locals["commutators"])
        filled["projector_velocities"] = len(
            frame.f_locals["projector_velocities"])
        # The loop's own names, which outlive it: the last axis's ``derivative``
        # is a whole ``(nk, npwx, nkb)`` block and ``overlap`` and
        # ``commutator`` a band block each.
        filled["dangling"] = sorted(
            name for name in ("commutator", "derivative", "overlap", "position")
            if frame.f_locals.get(name) is not None
        )
        raise Stop

    monkeypatch.setattr(efield, "_require_a_finite_kernel", watched)

    for born, keep, expected in (
        (False, False, (0, 0)),
        (True, False, (3, 3)),
        (False, True, (3, 0)),
    ):
        filled.clear()
        with pytest.raises(Stop):
            _run_a_field(born_charges=born, keep_internals=keep)
        assert (filled["commutators"], filled["projector_velocities"]) == expected, (
            f"born_charges={born} keep_internals={keep} kept "
            f"{filled['commutators']} commutators and "
            f"{filled['projector_velocities']} projector velocities"
        )
        assert filled["dangling"] == [], (
            f"born_charges={born} keep_internals={keep}: the loop's own "
            f"{filled['dangling']} are still bound under the whole of it"
        )


def _run_a_field(**options):
    """The electric-field response of the committed ultrasoft dielectric case.

    Ultrasoft, because that is where the augmentation dipole splits
    ``commutators`` from ``bare`` and where ``projector_velocities`` is built at
    all -- on a norm-conserving dataset both are free and there is nothing to
    assert.
    """
    from pathlib import Path

    from defumat.io.pwin import read_pw_input
    from defumat.pseudo import read_upf
    from defumat.scf import Calculation, run_scf
    from defumat.system import build_system

    root = Path(__file__).resolve().parents[1]
    system = build_system(read_pw_input(root / "data" / "qe" / "si-epsilon-us.in"))
    pseudos = tuple(read_upf(root / "data" / "pseudo" / s.pseudo_file)
                    for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    scf = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-6)
    return efield.dielectric_tensor(
        calculation, scf.wavefunctions, scf.eigenvalues, scf.density, scf.becsum,
        **options,
    )


def _frame_of(name):
    """The innermost frame of the named function on the stack."""
    import inspect

    for record in inspect.stack():
        if record.function == name:
            return record.frame
    raise AssertionError(f"{name} is not on the stack")
