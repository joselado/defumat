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


def test_the_field_keeps_its_commutators_only_for_a_reader(monkeypatch):
    """A7. Both lists are read after the loop and by nothing inside it.

    Driven through the assembly rather than through a real response: what the
    test is about is which lists were filled, and the loop that fills them is
    above the first Sternheimer solve.
    """
    source = _dielectric_source()
    assert "if keep_commutators:" in source, (
        "commutators are retained unconditionally again"
    )
    assert "if born_charges:\n                projector_velocities.append" in source, (
        "projector_velocities are retained unconditionally again"
    )
    assert "commutator = derivative = overlap = position = None" in source, (
        "the loop's own names outlive it again -- the last axis's derivative is "
        "a whole (nk, npwx, nkb) block"
    )


def _dielectric_source():
    import inspect

    return inspect.getsource(efield.dielectric_tensor)
