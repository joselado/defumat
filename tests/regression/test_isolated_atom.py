"""An isolated atom at Gamma with an ultrasoft pseudopotential.

``pw_atom/atom.in``: one oxygen in a 10-bohr cube, ``K_POINTS gamma``,
``occupations='from_input'``, ``O.pz-rrkjus.UPF``. Every piece of it existed
already -- the Gamma path, the ultrasoft path, occupations from input -- but the
*combination* was covered by nothing, and it turned out to be broken in three
independent ways, all of which this file now pins:

1. **The Gamma-only half-sphere storage was generated but never consumed.**
   ``ggen`` keeps one G of each ``(G, -G)`` pair at k = 0, and P2 implemented
   that selection, but nothing downstream ever learned to read it: ``h_psi``
   would have needed QE's ``vloc_psi_gamma`` packing, the eigensolver its real
   ``regterg`` overlaps, and the augmentation charge its ``fact = 2``. Silicon
   at ``K_POINTS gamma`` did not even reach that -- it failed in the symmetry
   maps, since rotating a stored half-sphere G lands on one that is not stored.
   :func:`defumat.scf.driver._without_gamma_storage` now substitutes an
   explicit k = 0 with the full sphere, which is the same physics at twice the
   storage, and says so.
2. **The old Vanderbilt augmentation format was refused.** Every ``rrkjus``
   file in the test set stores one ``PP_QIJ`` per projector pair rather than one
   ``PP_QIJL`` per ``(pair, L)``; expanding the first into the second is what
   ``set_upf_q`` exists for, and without it no ultrasoft oxygen or nickel
   dataset could be read at all.
3. **The local functional was switched off where the density is negative.**
   A truncated plane-wave density goes slightly negative in vacuum -- QE prints
   how much every iteration -- and ``xc_lda`` evaluates the functional at
   ``|rho|`` there rather than treating the point as empty. Clamping to the
   vacuum threshold instead is invisible in a bulk crystal and moves this cell's
   total energy by 1.3e-7 Ry and each of its energy terms by ~1e-5.

An isolated atom is the right shape of test for all three: it is mostly vacuum,
it is Gamma-only because a molecule in a box has no dispersion, and the
augmentation charge is a large fraction of an oxygen 2p.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from defumat.io import read_qe_output
from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf import run_scf
from defumat.system import build_system
from tests.conftest import reference_output
from tests.tolerances import (
    EIGENVALUE_EV,
    ENERGY_TERM_RY,
    TOTAL_ENERGY_RY,
)

pytestmark = [pytest.mark.regression, pytest.mark.slow]

#: The occupied bands of the oxygen atom: 2s and the three 2p. The two empty
#: bands above them are diffuse vacuum states of the periodic box, and QE
#: interpolates its local potential from a ``dq = 0.01`` table where this code
#: integrates directly -- a difference that averages out over a bound state's
#: many plane waves and does not over a state spread across the cell. They carry
#: no weight in the density, so nothing else depends on them.
OCCUPIED = 4

#: Looser than ``USPP_TERM_RY`` for this cell alone, and the reason is the
#: Cauchy-Schwarz bound on what ``conv_thr`` actually promises. ``dr2`` is the
#: Hartree energy of the density residual, so a term that is *linear* in the
#: residual can be as far out as ``sqrt(4 E_H dr2)``: with ``E_H = 17 Ry`` and
#: ``dr2 = 1e-10`` that is 8e-5 Ry, against 2e-5 for silicon's ``E_H = 1 Ry``.
#: Measured rather than argued: tightening this code's ``conv_thr`` from 1e-10 to
#: 1e-13 moves the Hartree term by 3e-5 Ry while moving the total by 3e-11. The
#: total is second-order in the residual and is compared at ``TOTAL_ENERGY_RY``,
#: where the agreement is 4e-9.
ISOLATED_ATOM_TERM_RY = 5e-5


@lru_cache(maxsize=None)
def _converged(qe_testsuite: Path, pseudo_dir: Path):
    system = build_system(read_pw_input(qe_testsuite / "pw_atom" / "atom.in"))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    with pytest.warns(UserWarning, match="gamma"):
        result = run_scf(
            system, pseudos, conv_thr=1e-10, max_iterations=100, mixing_beta=0.25
        )
    return system, pseudos, result


def _reference(qe_testsuite: Path):
    return read_qe_output(reference_output("pw_atom", "atom.in", qe_testsuite))


def test_total_energy_matches_reference(qe_testsuite, pseudo_dir):
    _, _, result = _converged(qe_testsuite, pseudo_dir)
    assert result.converged
    assert result.total_energy == pytest.approx(
        _reference(qe_testsuite).total_energy, abs=TOTAL_ENERGY_RY
    )


def test_energy_terms_match_reference(qe_testsuite, pseudo_dir):
    _, _, result = _converged(qe_testsuite, pseudo_dir)
    reference = _reference(qe_testsuite)

    assert set(result.energy_terms) == set(reference.energy_terms)
    for term, value in reference.energy_terms.items():
        tolerance = ENERGY_TERM_RY if term == "ewald" else ISOLATED_ATOM_TERM_RY
        assert result.energy_terms[term] == pytest.approx(value, abs=tolerance), term


def test_occupied_eigenvalues_match_reference(qe_testsuite, pseudo_dir):
    _, _, result = _converged(qe_testsuite, pseudo_dir)
    reference = _reference(qe_testsuite)

    ours = result.eigenvalues_ev[:, :OCCUPIED]
    theirs = reference.eigenvalues[0][:, :OCCUPIED]
    assert ours == pytest.approx(theirs, abs=EIGENVALUE_EV)

    # The three 2p levels are degenerate by the cubic box's symmetry; a split
    # here would mean the density was not symmetrised.
    p_levels = result.eigenvalues_ev[0, 1:4]
    assert p_levels.max() - p_levels.min() < 1e-6


def test_augmented_density_integrates_to_the_electron_count(qe_testsuite, pseudo_dir):
    """``int rho = nelec`` identically, which is what says ``q_ij`` and
    ``Q_ij(G=0)`` are the same numbers -- here through the expanded ``PP_QIJ``
    rather than a tabulated ``PP_QIJL``."""
    system, pseudos, result = _converged(qe_testsuite, pseudo_dir)
    nelec = sum(pseudos[t].z_valence for t in system.structure.types)

    density = np.asarray(result.density)
    charge = float(np.sum(density)) * float(system.cell.volume) / density.size
    assert charge == pytest.approx(nelec, abs=1e-9)


def test_gamma_storage_is_substituted_rather_than_used(qe_testsuite, pseudo_dir):
    """The k-set the calculation actually ran on, and its weight.

    The substitution has to keep the weight it was given: rebuilding the k-point
    through :meth:`KPoints.from_cartesian` would renormalise it and reapply the
    spin degeneracy, which is exactly the factor an ``nspin = 2`` run has already
    halved.
    """
    from defumat.scf.driver import Calculation

    system = build_system(read_pw_input(qe_testsuite / "pw_atom" / "atom.in"))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)

    assert system.kpoints.gamma_only
    with pytest.warns(UserWarning, match="gamma"):
        calculation = Calculation(system, pseudos)

    assert not calculation.system.kpoints.gamma_only
    assert not calculation.basis.dense.gamma_only
    assert float(calculation.system.kpoints.weights.sum()) == pytest.approx(
        float(system.kpoints.weights.sum())
    )
    # The full sphere is exactly twice the half sphere less the G = 0 it shares.
    from defumat.basis.builder import build_basis

    assert calculation.basis.dense.ngm == 2 * build_basis(system).dense.ngm - 1
