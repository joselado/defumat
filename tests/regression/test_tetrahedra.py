"""P8 check: the three tetrahedron variants against QE, on fcc aluminium.

``test-suite/pw_metal/`` ships one benchmark per variant, and the mapping from
``occupations`` to algorithm is *not* the one the file names suggest -- it comes
from ``PW/src/set_occupations.f90``:

* ``metal-tetrahedra.in``   -- ``tetrahedra-opt``, 4x4x4 shifted, an SCF run, so
  it is the only one of the three with a total energy to compare;
* ``metal-tetrahedra-1.in`` -- ``tetrahedra`` (Bloechl), 6x6x6 shifted, ``nbnd=4``;
* ``metal-tetrahedra-2.in`` -- ``tetrahedra-lin``, same grid.

The last two are ``calculation='nscf'`` and QE's ``jobconfig`` runs them after
``metal-tetrahedra.in`` in the same ``outdir``, so the density they diagonalise
against is exactly the one the first run converges to. That is reproduced here:
one SCF, then two fixed-density runs on the denser grid.

Only the Fermi level is printed for an NSCF run, and it is the right thing to
compare anyway: it is the one number that depends on the *whole* tetrahedron
machinery -- the decomposition, the ``equiv`` map and the integrated DOS -- and
the three variants disagree about it by 40 meV, which is twenty times the
tolerance.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from defumat.io import read_qe_output
from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf import run_scf
from defumat.scf.driver import Calculation
from defumat.scf.occupations import tetrahedra_for, tetrahedron_occupations
from defumat.system import build_system
from defumat.units import RY_TO_EV
from defumat.workflows import run_bands
from tests.conftest import reference_output
from tests.tolerances import (
    DENSITY_DEPENDENT_TERM_RY,
    ENERGY_TERM_RY,
    FERMI_EV,
    TOTAL_ENERGY_RY,
)

pytestmark = [pytest.mark.regression, pytest.mark.slow]

SCF_INPUT = "metal-tetrahedra.in"

#: The two NSCF follow-ons, with the algorithm each one exercises.
NSCF_CASES = [("metal-tetrahedra-1.in", "bloechl"), ("metal-tetrahedra-2.in", "linear")]


@lru_cache(maxsize=None)
def _converged(testsuite: Path, pseudo_dir: Path):
    system = build_system(read_pw_input(testsuite / "pw_metal" / SCF_INPUT))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    return system, pseudos, run_scf(system, pseudos, conv_thr=1e-10, max_iterations=80)


def _tetrahedron_fermi(system, pseudos, density):
    """The Fermi level an NSCF run reports: diagonalise, then integrate."""
    bands = run_bands(system, pseudos, density, conv_thr=1e-10)
    calculation = Calculation(system, pseudos)
    tetrahedra = tetrahedra_for(
        system.occupations, system.kpoints, calculation.symmetries, system.cell
    )
    _, ef = tetrahedron_occupations(
        tetrahedra, bands.eigenvalues, system.kpoints.weights, calculation.nelec
    )
    return bands, tetrahedra, float(ef) * RY_TO_EV


def test_optimized_tetrahedra_total_energy(qe_testsuite, pseudo_dir):
    _, _, result = _converged(qe_testsuite, pseudo_dir)
    reference = read_qe_output(reference_output("pw_metal", SCF_INPUT, qe_testsuite))

    assert result.converged
    assert result.total_energy == pytest.approx(reference.total_energy, abs=TOTAL_ENERGY_RY)


def test_optimized_tetrahedra_energy_terms(qe_testsuite, pseudo_dir):
    """No ``smearing`` term: the tetrahedron method has no entropy to subtract."""
    _, _, result = _converged(qe_testsuite, pseudo_dir)
    reference = read_qe_output(reference_output("pw_metal", SCF_INPUT, qe_testsuite))

    assert set(result.energy_terms) == set(reference.energy_terms)
    assert "smearing" not in result.energy_terms
    for term, value in reference.energy_terms.items():
        tolerance = ENERGY_TERM_RY if term == "ewald" else DENSITY_DEPENDENT_TERM_RY
        assert result.energy_terms[term] == pytest.approx(value, abs=tolerance), term


def test_optimized_tetrahedra_fermi_energy(qe_testsuite, pseudo_dir):
    _, _, result = _converged(qe_testsuite, pseudo_dir)
    reference = read_qe_output(reference_output("pw_metal", SCF_INPUT, qe_testsuite))

    assert result.fermi_energy * RY_TO_EV == pytest.approx(reference.fermi_energy, abs=FERMI_EV)


def test_occupation_weights_sum_to_the_electron_count(qe_testsuite, pseudo_dir):
    """The tetrahedra carry the Brillouin-zone measure; ``wk`` only the spin."""
    _, _, result = _converged(qe_testsuite, pseudo_dir)
    assert float(np.sum(result.occupations)) == pytest.approx(3.0, abs=1e-9)


@pytest.mark.parametrize(("name", "kind"), NSCF_CASES)
def test_nscf_tetrahedra_fermi_energy(qe_testsuite, pseudo_dir, name, kind):
    _, pseudos, scf = _converged(qe_testsuite, pseudo_dir)
    system = build_system(read_pw_input(qe_testsuite / "pw_metal" / name))
    reference = read_qe_output(qe_testsuite / "pw_metal" / f"benchmark.out.git.inp={name}")

    bands, tetrahedra, fermi_ev = _tetrahedron_fermi(system, pseudos, scf.density)

    assert tetrahedra.kind == kind
    assert tetrahedra.ntetra == 6 * 6 * 6 * 6
    assert fermi_ev == pytest.approx(reference.fermi_energy, abs=FERMI_EV)
    assert bands.eigenvalues.shape == reference.eigenvalues[0].shape


def test_the_three_variants_disagree_by_more_than_the_tolerance(qe_testsuite, pseudo_dir):
    """Otherwise the three-way comparison above would prove nothing.

    Bloechl's correction and the shortest-diagonal decomposition each move the
    Fermi level by tens of meV on this cell; if a variant were silently running
    another's algorithm, the numbers would coincide rather than differ.
    """
    _, pseudos, scf = _converged(qe_testsuite, pseudo_dir)
    levels = [
        _tetrahedron_fermi(
            build_system(read_pw_input(qe_testsuite / "pw_metal" / name)), pseudos, scf.density
        )[2]
        for name, _ in NSCF_CASES
    ]
    assert abs(levels[0] - levels[1]) > 20.0 * FERMI_EV


def test_nosym_and_the_wedge_reach_the_same_fermi_level(pseudo_dir):
    """A ``nosym`` run is the same physics on the complete grid, and it was not.

    The tetrahedra are built on the full grid and every corner is looked up in
    the *reduced* list, so the group that reduced the k-set and the group the
    corners are folded with have to be the same one. ``build_tetrahedra`` was
    handed the rotations alone, with ``time_reversal`` defaulting to ``True``:
    under ``nosym`` the run diagonalises all 64 points of aluminium's grid and
    the corners were folded onto 32 of them, so half the list never contributed
    and the other half was counted twice. Every index is valid, so nothing
    raised.

    **It reaches the SCF and not only the density of states**, because
    ``Calculation.occupations`` builds its own tetrahedra the same way. Measured
    on ``al-tetrahedra.in`` with ``nosym`` added, against the same input without
    it:

        before   E = -4.22533421 Ry   E_F = 0.683911 Ry
        after    E = -4.19790161 Ry   E_F = 0.626627 Ry
        wedge    E = -4.19789860 Ry   E_F = 0.626626 Ry

    -- 27.4 mRy on the total and **57.3 mRy, 0.78 eV, on the Fermi level**, on a
    quantity that has to agree with the symmetric run exactly because it is the
    same crystal sampled on the same points.
    """
    from pathlib import Path

    from defumat import Calculator

    text = (Path(__file__).resolve().parents[1] / "data" / "qe"
            / "al-tetrahedra.in").read_text()
    wedge = Calculator.from_text(text, pseudo_dir, announce=False)
    whole = Calculator.from_text(
        text.replace("&system", "&system\n    nosym = .true.,"), pseudo_dir,
        announce=False)

    reduced = wedge.get_scf(conv_thr=1e-10)
    complete = whole.get_scf(conv_thr=1e-10)
    assert reduced.converged and complete.converged
    assert len(np.asarray(whole.system.kpoints.weights)) == 64, (
        "nosym has to reach the complete grid or this compares nothing")

    assert complete.fermi_energy == pytest.approx(reduced.fermi_energy, abs=1e-5)
    assert complete.total_energy == pytest.approx(reduced.total_energy, abs=1e-5)

    # The density of states on top of it, compared with the wedge's rather than
    # with ``nelec``: the integral to ``E_F`` is a trapezoid on a finite energy
    # grid and misses the electron count by 0.035 either way, so the statement
    # that means something is that the two runs agree.
    def filled(calculator, scf):
        dos = calculator.get_dos()
        energies, values = np.asarray(dos.energies), np.asarray(dos.dos)
        below = energies <= scf.fermi_energy
        return float(np.trapezoid(values[below], energies[below]))

    assert filled(whole, complete) == pytest.approx(filled(wedge, reduced), abs=1e-3)
