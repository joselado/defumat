"""The Berry-phase polarization against ``pw.x``'s own ``lberry`` run.

The reference is ``tests/data/qe/reference.out.alas-berry``, generated with the
vendored ``pw.x`` from ``tests/data/qe/alas-berry.in`` -- the ``pw_berry``
test-suite cases could not be used directly, because both are PbTiO3 with
Vanderbilt ultrasoft datasets in UPF v1, which this package's reader refuses by
name.

The comparison is per string as well as on the totals, which is what makes it a
check of the assembly rather than of one number: four strings, four phases, and
the ionic table atom by atom.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest

from pypresso.io.pwin import read_pw_input
from pypresso.pseudo.upf import read_upf
from pypresso.scf.driver import run_scf
from pypresso.system.builder import build_system
from pypresso.workflows.polarization import run_polarization

pytestmark = pytest.mark.regression

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """Keep the peak bounded: several distinct cells compile the SCF stack each.

    The results stay cached; only XLA's executables are dropped, which is the
    trade ``CLAUDE.md`` prescribes for any file running more than about three
    cells.
    """
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def converged(name: str):
    system = build_system(read_pw_input(CASES / name))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return system, pseudos, run_scf(system, pseudos, conv_thr=1.0e-12).density


@lru_cache(maxsize=2)
def alas_polarization():
    """AlAs along ``b_3``, exactly the settings of ``alas-berry.in``."""
    system, pseudos, density = converged("alas-raman.in")
    return run_polarization(system, pseudos, density, gdir=2, nppstr=6,
                            transverse=(2, 2), conv_thr=1.0e-12)


# --- the numbers pw.x prints ------------------------------------------------

#: ``reference.out.alas-berry``, the ELECTRONIC POLARIZATION table. pw.x prints
#: five decimals, which is the resolution every comparison below is held to.
QE_STRINGS = (0.02777, 0.00252, 0.00252, 0.00224)
QE_IONIC = -0.25000
QE_ELECTRONIC = 0.00876
QE_TOTAL = -0.24124
QE_P_E_OMEGA_BOHR = -1.8038727
QE_QUANTUM = 7.4776542


def test_the_string_phases_match_pw_x():
    """Four strings, four phases, to the five decimals pw.x prints.

    Comparing the strings rather than only their average is what makes this a
    check of the string product itself: an error in the wrap, the alignment by
    Miller index or the branch would move the individual phases while leaving
    a plausible mean.
    """
    result = alas_polarization()
    assert len(result.strings) == 4
    assert np.allclose(result.strings.phases, QE_STRINGS, atol=1.0e-5)


def test_the_ionic_phase_matches_pw_x_atom_by_atom():
    result = alas_polarization()
    assert np.allclose(result.ion_phases, [0.0, -0.25], atol=1.0e-5)
    assert result.ionic_phase == pytest.approx(QE_IONIC, abs=1.0e-5)


def test_the_electronic_and_total_phases_match_pw_x():
    result = alas_polarization()
    assert result.electronic_phase == pytest.approx(QE_ELECTRONIC, abs=1.0e-5)
    assert result.total_phase == pytest.approx(QE_TOTAL, abs=1.0e-5)
    assert result.quantum == pytest.approx(1.0)


def test_the_polarization_in_physical_units_matches_pw_x():
    """And the quantum with it, which is the lattice vector's own length."""
    result = alas_polarization()
    assert result.polarization == pytest.approx(QE_P_E_OMEGA_BOHR, abs=1.0e-4)
    assert result.polarization_quantum_e_omega_bohr == pytest.approx(
        QE_QUANTUM, abs=1.0e-5
    )
    assert np.allclose(result.direction, [-0.70710678, 0.70710678, 0.0], atol=1e-6)


# --- statements that need no reference at all -------------------------------

def test_silicon_matches_pw_x_and_carries_the_larger_quantum():
    """``reference.out.si-berry``: the case AlAs cannot be.

    Both species have an **even** valence, so the quantum is 2 rather than 1 and
    the ionic half is reduced mod 2 where AlAs's is reduced mod 1. `pw.x` gives
    an ionic phase of 1.00000 (mod 2), an electronic phase of exactly zero and a
    total of 1.00000 (mod 2).
    """
    system, pseudos, density = converged("si2-nosym.in")
    result = run_polarization(system, pseudos, density, gdir=2, nppstr=8,
                              transverse=(4, 4), conv_thr=1.0e-10)
    assert result.quantum == pytest.approx(2.0)
    assert abs(result.electronic_phase) < 1.0e-6
    assert abs(abs(result.ionic_phase) - 1.0) < 1.0e-6
    assert abs(abs(result.total_phase) - 1.0) < 1.0e-6


def test_a_centrosymmetric_crystal_is_pinned_to_zero_or_half_a_quantum():
    """Silicon's phase cannot be anything else, and nothing here imposes it.

    Inversion maps ``P`` to ``-P``, and a polarization defined modulo a quantum
    can only satisfy that at ``0`` or at half a quantum. It is the cheapest
    statement in this file and the only one that would catch a sign convention
    reversed between the ionic and the electronic halves -- which would leave
    AlAs's *magnitude* right. Silicon lands on the **half**, which is the more
    informative of the two answers: a bug that returned zero would also pass a
    test written as "0 or a half".
    """
    system, pseudos, density = converged("si2-nosym.in")
    result = run_polarization(system, pseudos, density, gdir=2, nppstr=8,
                              transverse=(4, 4), conv_thr=1.0e-10)
    residue = result.total_phase / result.quantum
    distance = min(abs(residue), abs(abs(residue) - 0.5))
    assert distance < 1.0e-6, f"total phase {result.total_phase} is not pinned"
    assert abs(abs(residue) - 0.5) < 1.0e-6, "silicon should be on the half"


def test_the_phase_is_invariant_under_a_shift_of_the_string_mesh():
    """Which transverse points the strings start from cannot matter much.

    Each string is a closed loop and the average over them is an integral over
    the transverse plane, so a half-step offset of that plane is a different
    quadrature of the same quantity -- it changes the answer by the sampling
    error and not by a quantum, which is what this bounds.
    """
    system, pseudos, density = converged("alas-raman.in")
    plain = run_polarization(system, pseudos, density, gdir=2, nppstr=8,
                             transverse=(4, 4), conv_thr=1.0e-10)
    shifted = run_polarization(system, pseudos, density, gdir=2, nppstr=8,
                               transverse=(4, 4), shift=(1, 1, 0),
                               conv_thr=1.0e-10)
    assert shifted.total_phase == pytest.approx(plain.total_phase, abs=5.0e-3)
