"""GGA check: PBE silicon against Quantum ESPRESSO, on every pseudopotential kind.

The references are generated once with the vendored ``pw.x``
(``tools/generate_reference.py``) and stored beside the inputs, at
``conv_thr = 1e-10`` on both sides so that the two codes stop at the same fixed
point and the energy terms can be compared rather than only the variational
total.

The cases are the LDA ones with the functional changed, deliberately: what
differs from :mod:`tests.regression.test_uspp` is one term in the potential, so
a failure that appears here and not there is the gradient correction.

* ``si2-nc-pbe`` -- norm-conserving, with a dataset actually generated for PBE.
  The plainest possible gradient-corrected run, and the one whose functional
  comes from the pseudopotential's own header rather than from the input.
* ``si2-nc-revpbe`` / ``si2-nc-pbesol`` -- the other two members of the family,
  requested through ``input_dft``. revPBE changes only the exchange enhancement's
  ``kappa``; PBEsol changes ``mu`` in exchange *and* ``beta`` in correlation, so
  between them the three constants are pinned separately.
* ``si2-us-pbe`` / ``si8-us-pbe`` -- ultrasoft. The gradient correction is
  evaluated on the dense grid, where the augmentation charge lives, so these
  check that it sees the augmented density and not the smooth one.
* ``si2-paw-pbe`` / ``si8-paw-pbe`` -- PAW, which needs the whole one-centre
  gradient machinery on top: a radial derivative, an angular gradient from the
  harmonics' own derivatives, and a divergence on the sphere
  (:mod:`pypresso.paw.gradient`). QE prints the one-centre energy separately, so
  it is checked directly rather than only through the total.
* ``si2-nc-pbe-bands`` -- a band structure on the converged PBE density. It is
  the potential-rebuilding path that a non-self-consistent run takes, and the
  failure it guards against is that path quietly using a different functional
  from the SCF that produced the density.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from pypresso.io import read_qe_output
from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf import run_scf
from pypresso.system import build_system
from pypresso.workflows import run_bands
from tests.tolerances import (
    EIGENVALUE_EV,
    ENERGY_TERM_RY,
    TOTAL_ENERGY_RY,
    USPP_TERM_RY,
)

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"

NORM_CONSERVING = ["si2-nc-pbe", "si2-nc-revpbe", "si2-nc-pbesol"]
ULTRASOFT = ["si2-us-pbe", "si8-us-pbe"]
PAW = ["si2-paw-pbe", "si8-paw-pbe"]
ALL = NORM_CONSERVING + ULTRASOFT + PAW


@lru_cache(maxsize=None)
def _converged(case: str, pseudo_dir: Path):
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    return system, pseudos, run_scf(system, pseudos, conv_thr=1e-10, max_iterations=80)


def _reference(case: str):
    path = CASES / f"reference.out.{case}"
    if not path.is_file():
        pytest.skip(f"no generated reference for {case}; run tools/generate_reference.py")
    return read_qe_output(path)


@pytest.mark.parametrize("case", ALL)
def test_total_energy_matches_reference(pseudo_dir, case):
    _, _, result = _converged(case, pseudo_dir)
    reference = _reference(case)

    assert result.converged
    assert result.total_energy == pytest.approx(reference.total_energy, abs=TOTAL_ENERGY_RY)


@pytest.mark.parametrize("case", ALL)
def test_energy_terms_match_reference(pseudo_dir, case):
    _, _, result = _converged(case, pseudo_dir)
    reference = _reference(case)

    assert set(result.energy_terms) == set(reference.energy_terms)
    for term, value in reference.energy_terms.items():
        # Same bound as the ultrasoft cases and for the same reason: both codes
        # are at the same fixed point to ~1e-11 in dr2, but the terms are
        # first-order sensitive to where exactly each mixer stops.
        tolerance = ENERGY_TERM_RY if term == "ewald" else USPP_TERM_RY
        assert result.energy_terms[term] == pytest.approx(value, abs=tolerance), term

    assert sum(result.energy_terms.values()) == pytest.approx(result.total_energy, abs=1e-10)


@pytest.mark.parametrize("case", ALL)
def test_eigenvalues_match_reference(pseudo_dir, case):
    _, _, result = _converged(case, pseudo_dir)
    reference = _reference(case)

    ours = result.eigenvalues_ev
    theirs = reference.eigenvalues[0][:, : ours.shape[1]]
    assert ours == pytest.approx(theirs, abs=EIGENVALUE_EV)


def test_the_functional_comes_from_the_pseudopotential(pseudo_dir):
    """``Si.pbe-rrkj`` names PBE in its header and nothing in the input does.

    This is the case that used to run silently under LDA -- the header was
    parsed and ignored -- so it is worth asserting on the resolved functional
    and not only on the energy it produces.
    """
    from pypresso.scf.driver import Calculation

    system, pseudos, _ = _converged("si2-nc-pbe", pseudo_dir)
    assert system.input_dft is None
    assert Calculation(system, pseudos).functional.name == "PBE"


def test_a_gradient_functional_changes_the_answer(pseudo_dir):
    """The same dataset under PBE and under LDA must not agree.

    A gradient correction that was computed and then dropped -- masked out
    everywhere, or added to an array nobody reads -- would leave every other
    test in this file passing, since each compares against a reference produced
    by the same code path. This one does not depend on any reference: the two
    functionals differ, so the two energies must.
    """
    _, pseudos, pbe = _converged("si2-nc-pbe", pseudo_dir)

    pwin = read_pw_input(CASES / "si2-nc-pbe.in")
    pwin.namelists["system"]["input_dft"] = "pz"
    with pytest.warns(UserWarning, match="input_dft asks for PZ"):
        lda = run_scf(build_system(pwin), pseudos, conv_thr=1e-8, max_iterations=80)

    # 0.019 Ry apart in practice -- seven orders of magnitude above the 1e-9 the
    # comparisons above hold to, and small only because the dataset itself was
    # generated for PBE, which pulls the two answers towards each other.
    assert abs(pbe.total_energy - lda.total_energy) > 1e-3


def test_band_structure_on_a_gradient_corrected_density(pseudo_dir):
    """P7's path with a GGA: the density from si2-nc-pbe, bands along scf-1's path."""
    _, pseudos, scf = _converged("si2-nc-pbe", pseudo_dir)
    band_system = build_system(read_pw_input(CASES / "si2-nc-pbe-bands.in"))
    reference = _reference("si2-nc-pbe-bands")

    bands = run_bands(band_system, pseudos, scf.density)

    assert bands.eigenvalues.shape == reference.eigenvalues[0].shape
    assert bands.eigenvalues_ev == pytest.approx(reference.eigenvalues[0], abs=EIGENVALUE_EV)

    # Degeneracies at Gamma survive, as they do in the LDA band structure: the
    # gradient correction is symmetrised along with the density it is built from.
    at_gamma = bands.eigenvalues_ev[0]
    assert at_gamma[1] == pytest.approx(at_gamma[2], abs=1e-6)
    assert at_gamma[2] == pytest.approx(at_gamma[3], abs=1e-6)

    # PBE underestimates silicon's gap much as LDA does; the point is that it is
    # positive and in the right region, not that it is right.
    assert 0.3 < bands.gap(8) < 0.9


def test_ultrasoft_and_paw_agree_on_the_gradient_corrected_energy(pseudo_dir):
    """Two datasets for the same physics must give nearly the same answer.

    They are not comparable term by term -- PAW carries its one-centre energy
    where ultrasoft does not -- but the cohesive physics is the same, so the
    two eight-atom runs must agree on the *shape* of the band structure. A
    one-centre gradient term with the wrong sign or a missing factor would move
    the PAW eigenvalues and leave the ultrasoft ones alone.
    """
    _, _, us = _converged("si8-us-pbe", pseudo_dir)
    _, _, paw = _converged("si8-paw-pbe", pseudo_dir)
    occupied = 16
    us_width = np.ptp(us.eigenvalues_ev[:, :occupied])
    paw_width = np.ptp(paw.eigenvalues_ev[:, :occupied])
    assert us_width == pytest.approx(paw_width, abs=0.05)
