"""P23 check: a continued run reaches the same solution as a fresh one.

The claim being tested is an *identity*, not a comparison against Quantum
ESPRESSO: a run started from another regime's converged state must converge to
the same self-consistent solution as one started from the atoms, because the
starting guess is a guess and nothing else. The fresh runs in each pair are
already validated against QE elsewhere (``test_lsda.py``,
``test_noncollinear_magnetism.py``, ``test_spinorbit.py``), so what these cases
add is that the continuation does not move the answer.

The second number each case reports is the iteration count, and it is the point
of the phase rather than a bonus. It is *not* asserted -- an iteration count
depends on the mixer's history and on the ``ethr`` schedule, so pinning it would
be a test of the mixer -- but it is printed, and ``PLAN.md`` P23 records what it
was:

* **silicon, 1 -> 2 -> 4** -- the charge is the whole answer, so the
  continuation is nearly free (5 -> 4 and 5 -> 1), and 2 -> 2 with more bands
  covers the *per channel* span and its top-up with random vectors.
* **bcc iron, 2 -> 4** -- the collinear ground state rotated onto ``x``, which
  is QE's own ``pw_noncolin`` benchmark: 25 iterations from the atoms, **1**
  from the collinear state.
* **bcc iron, 1 -> 2** -- the case where the saving is small (30 -> 27) and the
  reason is worth stating: the *magnetization* is the slow variable here and it
  is exactly what the non-magnetic run has none of. What the pair does give is
  the magnetic stabilisation energy, from two runs of the same cell.
* **platinum, scalar PAW -> fully-relativistic PAW** -- switching spin-orbit
  coupling on. The datasets differ, so ``becsum`` is re-seeded and only the
  density carries; 13 iterations from the atoms, 7 from the scalar run.
"""

import dataclasses
from functools import lru_cache

import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf import run_scf
from defumat.system import build_system
from tests.conftest import GENERATED, QE_ROOT
from tests.tolerances import MAGNETIZATION_BOHRMAG

pytestmark = [pytest.mark.regression, pytest.mark.slow]

#: The two runs of a pair are the same fixed point reached from different
#: starting guesses, so they agree to about the convergence threshold rather
#: than to QE's printing precision.
SAME_SOLUTION_RY = 1e-7

PSEUDO = "tests/data/pseudo"


def _input(path):
    if not path.is_file():
        pytest.skip(f"reference input not present at {path}")
    return dataclasses.replace(build_system(read_pw_input(path)), tstress=False)


@lru_cache(maxsize=None)
def _silicon():
    system = _input(QE_ROOT / "test-suite" / "pw_scf" / "scf.in")
    # Two channels need an occupation scheme that can fill them unequally.
    return dataclasses.replace(
        system, occupations="smearing", smearing="gaussian", degauss=0.02,
    ), (read_upf(f"{PSEUDO}/Si.pz-vbc.UPF"),)


@lru_cache(maxsize=None)
def _iron():
    system = _input(
        QE_ROOT / "test-suite" / "pw_noncolin" / "noncolin.in"
    )
    return system, (read_upf(f"{PSEUDO}/Fe.pz-nd-rrkjus.UPF"),)


def _report(name, fresh, continued):
    print(
        f"\n{name}: fresh {fresh.total_energy:.9f} Ry in {fresh.iterations} "
        f"iterations, continued {continued.total_energy:.9f} Ry in "
        f"{continued.iterations}"
    )


def test_silicon_unpolarized_to_collinear_to_noncollinear():
    system, pseudos = _silicon()
    unpolarized = run_scf(system, pseudos, conv_thr=1e-8)

    collinear = system.with_spin(2, starting_magnetization=(0.3,))
    fresh = run_scf(collinear, pseudos, conv_thr=1e-8)
    continued = run_scf(collinear, pseudos, conv_thr=1e-8, starting_from=unpolarized)
    _report("Si 1 -> 2", fresh, continued)
    assert continued.total_energy == pytest.approx(fresh.total_energy, abs=SAME_SOLUTION_RY)
    # Silicon is not magnetic: the seed decays and both runs end unpolarized,
    # which is the answer and not a failure of the seed.
    assert abs(continued.magnetization) < MAGNETIZATION_BOHRMAG

    # The same regime with more bands, which is the *per channel* span: one set
    # of vectors per Hamiltonian rather than one shared, topped up with random
    # vectors to reach the new nbnd. It has to reach the same answer too.
    wider = run_scf(collinear, pseudos, nbnd=12, conv_thr=1e-8, starting_from=continued)
    _report("Si 2 -> 2, nbnd 12", fresh, wider)
    assert wider.total_energy == pytest.approx(fresh.total_energy, abs=SAME_SOLUTION_RY)

    # ... and on into a noncollinear (nonmagnetic, so nspin_mag = 1) run, which
    # is the spin-orbit shape without a relativistic dataset to put in it.
    spinor = collinear.with_spin(4, starting_magnetization=(0.0,))
    assert spinor.nspin_mag == 1
    fresh4 = run_scf(spinor, pseudos, conv_thr=1e-8)
    continued4 = run_scf(spinor, pseudos, conv_thr=1e-8, starting_from=continued)
    _report("Si 2 -> 4", fresh4, continued4)
    assert continued4.total_energy == pytest.approx(fresh4.total_energy, abs=SAME_SOLUTION_RY)
    # The same electrons however they are written down.
    assert continued4.total_energy == pytest.approx(
        unpolarized.total_energy, abs=SAME_SOLUTION_RY
    )


def test_iron_collinear_to_noncollinear_rotates_the_moment():
    """QE's ``pw_noncolin`` benchmark, reached from its own collinear run.

    ``angle1 = 90`` points bcc iron's moment along ``x``. The collinear run
    knows only ``|m|``; what the continuation has to do is put that number on
    the axis the target's input asks for, and then not move.
    """
    spinor, pseudos = _iron()
    collinear = spinor.with_spin(2)
    converged = run_scf(collinear, pseudos, conv_thr=1e-8, mixing_beta=0.2)

    fresh = run_scf(spinor, pseudos, conv_thr=1e-8, mixing_beta=0.2)
    continued = run_scf(spinor, pseudos, conv_thr=1e-8, mixing_beta=0.2,
                        starting_from=converged)
    _report("Fe 2 -> 4", fresh, continued)
    assert continued.total_energy == pytest.approx(fresh.total_energy, abs=SAME_SOLUTION_RY)

    moment = np.asarray(continued.magnetization_vector)
    assert moment[0] == pytest.approx(converged.magnetization, abs=MAGNETIZATION_BOHRMAG)
    assert abs(moment[1]) < 1e-6 and abs(moment[2]) < 1e-6
    # The collinear state is already the answer, so the continued run is one
    # iteration -- it converges on the state it was handed.
    assert continued.iterations < fresh.iterations

    # ... and back down again, which is what makes the pair reversible. The
    # magnetization is along x, so the demotion has to *find* that axis and lay
    # it back on z rather than reading m_z, which is zero. The wavefunctions are
    # deliberately not carried: a spinor does not split into two channels.
    with pytest.warns(RuntimeWarning, match="not being carried over"):
        back = run_scf(collinear, pseudos, conv_thr=1e-8, mixing_beta=0.2,
                       starting_from=continued)
    _report("Fe 4 -> 2", converged, back)
    assert back.total_energy == pytest.approx(converged.total_energy, abs=SAME_SOLUTION_RY)
    assert back.magnetization == pytest.approx(
        converged.magnetization, abs=MAGNETIZATION_BOHRMAG
    )
    assert back.iterations < converged.iterations


def test_iron_nonmagnetic_to_ferromagnetic_finds_the_magnetic_state():
    """The case the seed exists for, and the stabilisation energy it gives.

    The non-magnetic solution is a stationary point of the polarized functional
    too, so a promotion that carried only the converged charge would converge
    straight back to it. What breaks the symmetry is the target's
    ``starting_magnetization``, seeded onto the converged charge -- and the
    difference between the two total energies is the magnetic stabilisation
    energy of bcc iron, from two runs of the same cell.
    """
    spinor, pseudos = _iron()
    nonmagnetic = spinor.with_spin(1, starting_magnetization=(0.0,))
    collinear = spinor.with_spin(2)

    unpolarized = run_scf(nonmagnetic, pseudos, conv_thr=1e-8, mixing_beta=0.2)
    fresh = run_scf(collinear, pseudos, conv_thr=1e-8, mixing_beta=0.2)
    continued = run_scf(collinear, pseudos, conv_thr=1e-8, mixing_beta=0.2,
                        starting_from=unpolarized)
    _report("Fe 1 -> 2", fresh, continued)

    assert continued.total_energy == pytest.approx(fresh.total_energy, abs=SAME_SOLUTION_RY)
    assert continued.magnetization == pytest.approx(
        fresh.magnetization, abs=MAGNETIZATION_BOHRMAG
    )
    stabilisation = fresh.total_energy - unpolarized.total_energy
    print(f"  magnetic stabilisation energy: {stabilisation:.6f} Ry")
    assert stabilisation < -0.01


def test_platinum_switches_spin_orbit_coupling_on():
    """Scalar-relativistic PAW -> fully-relativistic PAW with ``lspinorb``.

    The two runs use *different datasets* -- which is what switching spin-orbit
    coupling on means for an ultrasoft or PAW pseudopotential, since QE's
    ``average_pp`` refuses to j-average one (``PW/src/average_pp.f90``) and this
    code refuses the same combination. So the projector counts differ, and the
    continuation carries the density while re-seeding ``becsum`` from the
    target's own dataset. It still saves half the iterations, because the charge
    is what they were spent on.
    """
    scalar = _input(GENERATED / "pt-paw-scalar.in")
    spinor = _input(QE_ROOT / "test-suite" / "pw_spinorbit" / "spinorbit-paw.in")
    sr = (read_upf(f"{PSEUDO}/Pt.pbe-n-kjpaw_psl.0.1.UPF"),)
    fr = (read_upf(f"{PSEUDO}/Pt.rel-pbe-n-kjpaw_psl.0.1.UPF"),)

    converged = run_scf(scalar, sr, conv_thr=1e-9)
    fresh = run_scf(spinor, fr, conv_thr=1e-9)
    with pytest.warns(RuntimeWarning, match="different pseudopotential"):
        continued = run_scf(spinor, fr, conv_thr=1e-9, starting_from=converged)
    _report("Pt scalar -> spin-orbit", fresh, continued)
    assert continued.total_energy == pytest.approx(fresh.total_energy, abs=SAME_SOLUTION_RY)
    assert continued.iterations < fresh.iterations
