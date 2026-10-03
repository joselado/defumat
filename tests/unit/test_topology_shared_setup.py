"""A fixed-density state source builds its k-independent terms once per workflow.

``OPEN.md`` Part XXIII, item 12. A Berry-phase polarization asks
:class:`~defumat.workflows.topology.DFTSource` for its states once per string, a
streamed Chern number once per column and a Wilson loop once per pumping step.
What each call needs and the k-points do not touch -- the frozen potential, the
ultrasoft ``newd`` integral and the smooth-grid copy of the potential, and the
augmentation charge between neighbouring k-points ``q_ij(b)`` -- used to be built
again on every call, the last because its cache was keyed on the per-call copy
of the calculation.

Each count is taken where the old code's count is larger than one, so the
assertion can fail: two strings, which gave two of each.

The cell is ultrasoft silicon at a low cutoff with the atomic starting density.
The counts do not depend on the density being converged, and an ultrasoft dataset
is what gives ``newd`` and ``q_ij(b)`` anything to do.
"""

import pytest

from defumat.io.pwin import parse_pw_input
from defumat.pseudo import read_upf
from defumat.scf.driver import Calculation
from defumat.system.builder import build_system
from defumat.topology import augmentation as topology_augmentation
from defumat.workflows.polarization import run_polarization

pytestmark = pytest.mark.unit


_SILICON = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.2, nat = 2, ntyp = 1,
  ecutwfc = 12.0, ecutrho = 96.0, nosym = .true., noinv = .true.
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-n-rrkjus_psl.0.1.UPF
ATOMIC_POSITIONS crystal
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""


@pytest.fixture(scope="module")
def silicon(pseudo_dir):
    system = build_system(parse_pw_input(_SILICON))
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    density = Calculation(system, pseudos).starting_density()
    return system, pseudos, density


def _counting(monkeypatch, owner, name):
    calls = []
    original = getattr(owner, name)

    def counted(*args, **kwargs):
        calls.append(name)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, counted)
    return calls


def test_a_polarization_builds_its_k_independent_terms_once(silicon, monkeypatch):
    """Two strings, one potential, one ``newd`` and one ``q_ij(b)``.

    Before, each was built once per string: two of each here, and sixteen on
    zincblende AlAs's default four-by-four set of strings.
    """
    system, pseudos, density = silicon
    potentials = _counting(monkeypatch, Calculation, "potential")
    integrals = _counting(monkeypatch, Calculation, "coefficients")
    factors = _counting(monkeypatch, topology_augmentation, "augmentation_at_q")

    result = run_polarization(system, pseudos, density, gdir=2, nppstr=4,
                              transverse=(2, 1), nocc=4)

    assert len(result.strings.phases) == 2, "the counts need more than one string"
    assert len(potentials) == 1
    assert len(integrals) == 1
    assert len(factors) == 1

