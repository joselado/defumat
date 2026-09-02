"""The magnetoelectric tensor, on the two AlAs cells that differ only by spin-orbit.

The sharp statement here is the **null**, and it is a symmetry statement rather
than a tolerance: without spin-orbit coupling the spin and the lattice decouple,
so a global spin rotation maps ``B`` to ``-B`` and leaves every charge observable
fixed. Then ``P(+B) = P(-B)``, and ``alpha`` vanishes *identically* -- no
plausible bug in the assembly survives it, because every stage of the
calculation still runs and still produces a polarization.

``alas-magnetoelectric.in`` and ``alas-magnetoelectric-nosoc.in`` are the same
cell, cutoff, field, k-grid and ultrasoft PBE family at the same valence; the
only difference is that the second uses scalar-relativistic datasets with
``lspinorb = .false.``. The datasets have to change with the switch and that is a
refusal rather than a choice: a fully-relativistic dataset with ``lspinorb``
off is refused here rather than having its two ``j`` channels averaged.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.pseudo.upf import read_upf
from defumat.response.magnetoelectric import magnetoelectric_tensor
from defumat.system.builder import build_system

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: One column of the tensor: two self-consistent runs and six polarizations.
#: The whole tensor is three times this and says nothing more on a cubic cell.
#: ``conv_thr`` here is the *polarization's* threshold, and it is tight for the
#: reason the null's docstring gives: it, and not the self-consistent field's,
#: is what floors the residue. ``chain`` is left at its default, which is off.
OPTIONS = dict(
    nppstr=6, transverse=(2, 2), directions=(2,),
    scf_options=dict(conv_thr=1.0e-10, max_iterations=120), conv_thr=1.0e-11,
)


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    yield
    jax.clear_caches()


@lru_cache(maxsize=4)
def tensor(name: str, delta: float):
    system = build_system(read_pw_input(CASES / name))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return magnetoelectric_tensor(system, pseudos, delta=delta, **OPTIONS)


def test_the_tensor_is_spin_orbit_coupling_and_almost_nothing_else():
    """The null against its control, which is the pair that means something.

    Without spin-orbit coupling nothing ties the spin the field acts on to the
    lattice the polarization is measured along, so a global spin rotation maps
    ``B`` to ``-B`` and leaves every charge observable fixed: the tensor vanishes
    *identically* in exact arithmetic.

    It does not vanish to machine precision, and **what stopped it from doing so
    was the loop rather than the physics**. Two causes, and neither is visible
    without the other: ``chain`` seeded the ``-delta/2`` run from the
    ``+delta/2`` one, which is exactly the ``B -> -B`` asymmetry the null rests
    on; and the *polarization* threshold -- not the self-consistent field's --
    set the floor underneath it. Chained at a tight polarization threshold gives
    6.9e-8 and unchained at a loose one 5.3e-8, against 3.9e-8 for the original
    pairing; both together give **3.4e-9**. ``chain`` is off by default now.

    It is still asserted as a *ratio* rather than an absolute, because the two
    ends are separately converged runs and the residue is a fixed phase offset
    divided by ``delta``. The noise floor of two runs at the *same* field is
    exactly zero -- they are bit-identical -- and two runs at different fields
    are not that comparison.

    A null on its own would not be evidence either: an assembly that returned
    zero always would pass it. The control is the identical calculation on the
    identical crystal with ``lspinorb`` on.
    """
    null = tensor("alas-magnetoelectric-nosoc.in", 0.02)
    signal = tensor("alas-magnetoelectric.in", 0.02)
    assert signal.magnitude > 1.0e-6, "the control produced no response to test against"
    assert signal.magnitude / null.magnitude > 300.0, (
        f"spin-orbit coupling should account for the tensor: signal "
        f"{signal.magnitude:.3e} against null {null.magnitude:.3e}"
    )
    assert signal.largest_step < 0.25       # the branch guard is not doing work


def test_the_response_is_linear_in_the_field_step():
    """``delta = 0.02`` is inside the linear regime and ``0.04`` is not.

    Measured with ``chain`` off and a tight polarization threshold: 4.322 and
    4.438 (x 1e-6) at ``delta`` of 0.010 and 0.020, against 5.098 at 0.040. The step has to be large enough that the
    phase difference clears each run's convergence residue and small enough to
    stay linear, and this is where that window sits for this cell.
    """
    small = tensor("alas-magnetoelectric.in", 0.01)
    large = tensor("alas-magnetoelectric.in", 0.02)
    assert abs(large.magnitude - small.magnitude) / small.magnitude < 0.05


def test_the_facade_and_the_functional_entry_point_agree():
    """``Calculator.get_magnetoelectric_tensor`` must be a delegation and nothing else.

    It was not. The input file's ``conv_thr`` is the *self-consistent field's*
    number, and ``_defaults_for`` matches by parameter name, so it landed in the
    **polarization's** threshold instead while the six ground states kept their
    defaults. The facade then reported a spin-orbit null ratio of 4 where this
    entry point gives 1293 -- a plausible small number rather than an error, on
    the same crystal and the same call.

    The two thresholds are named apart now (``scf_conv_thr`` and
    ``polarization_conv_thr``) so the collision cannot recur, and this is the
    check none of the other tests here made: every one of them went through the
    functional path and none through the facade.
    """
    from defumat import Calculator

    options = dict(delta=0.02, directions=(2,), nppstr=6, transverse=(2, 2))
    facade = Calculator.from_file(
        CASES / "alas-magnetoelectric.in", PSEUDO, announce=False
    ).get_magnetoelectric_tensor(**options)

    system = build_system(read_pw_input(CASES / "alas-magnetoelectric.in"))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    direct = magnetoelectric_tensor(system, pseudos, **options)

    assert facade.magnitude == pytest.approx(direct.magnitude, rel=1e-10)
    assert np.allclose(facade.alpha, direct.alpha, rtol=1e-10)
