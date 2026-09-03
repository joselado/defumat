"""P64: the orbital magnetization against ``pw.x``'s ``lorbm``, and its nulls.

The case is an **iodine atom in a twelve-bohr box** (``i-atom-soc.in``), which
is the smallest system that has an orbital magnetization at all: neutral iodine
is ``5s2 5p5``, one hole in the p shell, so Hund's rules give ``S = 1/2``,
``L = 1`` and -- the shell being more than half full -- ``J = 3/2`` with the
orbital moment parallel to the spin. Spin-orbit coupling is what locks them
together and a fully relativistic dataset is what carries it; fixed occupations
leave a 0.16 eV gap above the seventh spinor band, which is what makes the
manifold isolated.

Three statements, in the order of how much they can go wrong silently:

* the two Kubo terms **separately** reproduce ``pw.x``'s
  (``reference.out.i-atom-soc-orbm``), which is a stronger comparison than the
  total: the split is QE's rather than the papers', so agreeing on both halves
  says the assembly is the same one and not merely a compensating pair;
* the moment is of **atomic size and opposite to the spin's direction**, which
  is the physics rather than the transcription: the moment of an orbital
  angular momentum is ``-mu_B <L>``, so ``<L_z> > 0`` from Hund's third rule
  gives ``M_z < 0``. ``<L>`` on the same run (P48) is the independent number,
  and the *difference* between the two is what a projection onto atomic
  orbitals does not see -- the current outside those orbitals, and the nonlocal
  pseudopotential's own contribution to the velocity, since ``<L>`` is
  ``r x p`` where this contracts ``H``;
* switching the coupling off inside the same relativistic dataset
  (``soc_scale = 0``) leaves **nothing**. That is the sharpest null available,
  because everything else about the run -- the magnetization, the dataset, the
  cell, the mesh -- is unchanged.
"""

from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf import run_scf
from defumat.system import build_system
from defumat.workflows import run_orbital_magnetization

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: ``reference.out.i-atom-soc-orbm``, the ORBITAL MAGNETIZATION block, in Bohr
#: magnetons per cell. ``pw.x`` prints the two terms and not their sum.
QE_LC = -0.59863696744183559
QE_IC = -0.57579873587164021
QE_TOTAL = QE_LC + QE_IC

#: How far the two codes may differ. Both evaluate the *same* discrete
#: expression on the same mesh, so what is left is the difference between two
#: separately converged densities -- measured by re-running this case at
#: ``conv_thr`` 1e-6, 1e-8 and 1e-12, where ``M_LC`` moves by 5e-6, 2e-7 and
#: 2e-6 against ``pw.x``. A tighter tolerance would be pinning that noise.
TOLERANCE = 1.0e-5


@pytest.fixture(autouse=True)
def _bounded_compilation():
    """``CLAUDE.md``'s memory rule: drop XLA's executables between cases."""
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _converged(case: str, soc_scale: float | None = None):
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    if soc_scale is not None:
        system = system.with_soc_scale(soc_scale)
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    return system, pseudos, run_scf(system, pseudos, conv_thr=1e-10,
                                    max_iterations=200)


@lru_cache(maxsize=2)
def _iodine(soc_scale: float | None = None):
    """``M_orb`` on the 3x3x3 grid ``i-atom-soc-orbm.in`` asks ``pw.x`` for."""
    system, pseudos, result = _converged("i-atom-soc", soc_scale)
    return run_orbital_magnetization(
        system, pseudos, result.density, divisions=(3, 3, 3), nocc=7,
        conv_thr=1e-10,
    )


def test_both_kubo_terms_match_pw_x():
    """``M_LC`` and ``M_IC`` separately, on the grid ``lorbm`` was run on."""
    result = _iodine()
    assert result.lc[2] == pytest.approx(QE_LC, abs=TOLERANCE)
    assert result.ic[2] == pytest.approx(QE_IC, abs=TOLERANCE)
    assert result.total[2] == pytest.approx(QE_TOTAL, abs=2 * TOLERANCE)


def test_the_transverse_components_vanish():
    """A moment driven along ``z`` in a cubic box has no ``x`` or ``y`` part.

    Nothing imposes it -- the run is ``nosym``, the mesh is the whole zone and
    the assembly never sees a symmetry operation. ``pw.x``'s own transverse
    components are 1.6e-4 and 5.4e-4 on this cell, which is its density's
    residual asymmetry rather than a different answer; the bound here is loose
    enough to cover both codes' and tight enough to catch a swapped Cartesian
    index, which would put the whole 1.17 in the wrong slot.
    """
    result = _iodine()
    assert np.abs(result.total[:2]).max() < 1.0e-3
    assert abs(result.total[2]) > 1.0


def test_the_orbital_moment_is_atomic_and_opposite_to_the_spin():
    """``|M| ~ 1 mu_B``, negative where the spin density points along ``+z``.

    The size is Hund's third rule for a ``p^5`` hole -- ``L = 1``, locked
    parallel to ``S`` -- and the sign is that the moment of an angular momentum
    is ``-mu_B <L>``. Neither is a fit: the calculation is handed a starting
    magnetization and a dataset, and nothing tells it what ``L`` should be.
    """
    system, pseudos, scf = _converged("i-atom-soc")
    result = _iodine()
    assert float(np.asarray(scf.magnetization_vector)[2]) > 0.9
    assert result.total[2] < 0.0
    assert 0.9 < abs(result.total[2]) < 1.5


def test_without_spin_orbit_coupling_there_is_none():
    """``soc_scale = 0`` in the same relativistic dataset: nothing is left.

    The magnet is still a magnet -- the SCF converges to the same 1.0 mu_B of
    spin -- so what this isolates is the coupling and nothing else. It is the
    same statement P48 makes as ``<L> = 1.7e-16``, one quantity further on.
    """
    result = _iodine(0.0)
    assert np.abs(result.total).max() < 1.0e-6
    assert np.abs(result.lc).max() < 1.0e-6
    assert np.abs(result.ic).max() < 1.0e-6


def test_the_manifold_is_trivial_and_the_chemical_potential_does_nothing():
    """No Chern vector, so ``mu`` is not a knob on this crystal.

    Worth asserting rather than assuming: ``dM/dmu`` is the one term ``pw.x``
    drops, and if it were *not* small here the comparison above would be
    comparing two conventions rather than two calculations.
    """
    result = _iodine()
    assert np.abs(result.chern).max() < 1.0e-3
    assert np.abs(result.dm_dmu).max() < 1.0e-2
    assert result.smallest_determinant > 0.5
