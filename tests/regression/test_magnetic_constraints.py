"""P18: constrained moments and external fields, against QE.

QE's suite ships three constrained runs on bcc iron, one per scheme it
implements for a noncollinear calculation: the moment on each atom
(``i_cons = 1``), the *direction* of that moment (``i_cons = 2``), and the total
moment of the cell (``i_cons = 3``). All three are here, and with them the
identity that a constraint machinery which is switched off must change nothing.

**The references are regenerated rather than committed**, and for a reason
specific to this phase: ``noncolin-constrain_atomic.in`` carries a commented-out
``lambda = 1`` above the ``lambda = 0.005`` it actually sets, and the committed
2017 output prints a constraint energy of 8.022 Ry at the starting density,
which is the *unscaled* sum of squares -- what ``lambda = 1`` gives. The
committed output therefore does not belong to the committed input.
``tools/generate_reference.py`` re-runs all three with the vendored pw.x.

**A constrained total energy is a softer number than an unconstrained one.** The
penalty holds the moment away from where the functional wants it, so the energy
is first-order sensitive to exactly where the moment ends up, and QE's own last
two iterations of the atomic case still move the constraint energy by 1.3e-6 Ry.
That is why the tolerance here is looser than the 1e-8 an ordinary total energy
gets, and it is a property of the state rather than of either code.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from pypresso.io import read_qe_output
from pypresso.io.pwin import parse_pw_input, read_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf import run_scf
from pypresso.system import build_system

pytestmark = [pytest.mark.regression, pytest.mark.slow]

#: A constrained run's total energy, for the reason in the module docstring. At
#: ``conv_thr = 1e-10`` -- what the unconstrained cases use, and what QE's
#: reference was generated at -- the same comparison is 1.5e-6: the penalty makes
#: the energy first-order sensitive to the moment, so both codes have to be run
#: further than usual before they are comparing the same state.
CONSTRAINED_RY = 3e-7


@lru_cache(maxsize=None)
def _run(path: Path, pseudo_dir: Path, conv_thr: float = 1e-11, max_iterations: int = 200):
    pwin = read_pw_input(path)
    system = build_system(pwin)
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    # The input's own ``mixing_beta``, which for these runs is not a detail: a
    # constraint on the *total* moment is a uniform field proportional to the
    # moment's error, which is a stiff global feedback. QE's input asks for 0.3
    # and oscillates by several Bohr magnetons per iteration even so; at the
    # 0.7 default neither code converges in 150 iterations.
    beta = float(pwin.get("electrons", "mixing_beta") or 0.7)
    return system, run_scf(
        system, pseudos, conv_thr=conv_thr, max_iterations=max_iterations,
        mixing_beta=beta,
    )


CASES = [
    ("noncolin-constrain_atomic.in", "atomic"),
    ("noncolin-constrain_angle.in", "atomic direction"),
    ("noncolin-constrain_total.in", "total"),
]


@pytest.mark.parametrize(("name", "scheme"), CASES)
def test_constrained_total_energy(name, scheme, qe_testsuite, pseudo_dir, benchmark):
    reference = read_qe_output(benchmark("pw_noncolin", name))
    system, result = _run(qe_testsuite / "pw_noncolin" / name, pseudo_dir)

    assert system.constrained_magnetization == scheme
    assert result.converged
    assert result.total_energy == pytest.approx(
        reference.total_energy, abs=CONSTRAINED_RY
    )

    # The constrained moment itself, to the two decimals QE prints.
    assert np.asarray(result.magnetization_vector) == pytest.approx(
        np.asarray(reference.magnetization_vector), abs=6e-3
    )


def test_constraint_energy_matches_qe(qe_testsuite, pseudo_dir, benchmark):
    """``etcon``, which QE prints at every iteration and never adds to the total.

    Checked at *both* ends: at the starting density, where it is decided by the
    penalty expression and the integration spheres alone and agrees to eight
    decimals, and at convergence, where it also depends on where each code's SCF
    stopped.
    """
    name = "noncolin-constrain_atomic.in"
    reference_text = Path(benchmark("pw_noncolin", name)).read_text()
    printed = [
        float(line.split("=")[1])
        for line in reference_text.splitlines()
        if "constraint energy" in line
    ]
    if not printed:
        pytest.skip("this reference does not print the constraint energy")

    system, result = _run(qe_testsuite / "pw_noncolin" / name, pseudo_dir)
    from pypresso.scf.driver import Calculation

    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    starting = calculation.potential(calculation.starting_density())
    assert float(starting.e_constraint) == pytest.approx(printed[0], abs=1e-7)

    assert result.constraint_energy == pytest.approx(printed[-1], abs=1e-6)
    # And it is *not* in the total energy: QE prints it and never returns it.
    assert result.total_energy == pytest.approx(
        read_qe_output(benchmark("pw_noncolin", name)).total_energy, abs=CONSTRAINED_RY
    )


def test_no_constraint_is_no_change(qe_testsuite, pseudo_dir, tmp_path_factory):
    """``lambda`` at zero and no field: the same run as with no machinery at all.

    The field enters through ``v_scf``, so a zero field is a zero *added array*
    rather than a skipped code path -- which is exactly the kind of thing that
    is right to within round-off and worth pinning to it.
    """
    text = (qe_testsuite / "pw_noncolin" / "noncolin.in").read_text()
    directory = tmp_path_factory.mktemp("zero-field")
    with_field = directory / "zero-field.in"
    marker = text.lower().index("&system") + len("&system")
    with_field.write_text(
        text[:marker]
        + "\n    constrained_magnetization = 'atomic', lambda = 0.0\n"
        + text[marker:]
    )

    _, plain = _run(qe_testsuite / "pw_noncolin" / "noncolin.in", pseudo_dir)
    system, constrained = _run(with_field, pseudo_dir)

    assert system.constrained_magnetization == "atomic"
    assert constrained.constraint_energy == pytest.approx(0.0, abs=1e-14)
    assert constrained.total_energy == pytest.approx(plain.total_energy, abs=1e-9)


def test_a_local_field_breaks_the_symmetry_it_should(pseudo_dir, tmp_path_factory):
    """Elk's ``bfcmt`` and ``reducebf``: a field that starts a state and then leaves.

    The use case, exactly: **a run with no starting magnetization cannot become
    magnetic.** Nothing in a spin-symmetric functional breaks the symmetry, so
    the SCF sits in the nonmagnetic solution however far from the ground state
    it is -- for a hydrogen atom, 52 mRy above it. A field breaks the symmetry;
    ``reducebf`` then multiplies the field down to nothing, and what is left is
    the magnetic solution of the *field-free* problem.

    Three runs make that a test rather than an anecdote: the nonmagnetic one to
    show there is something to break out of, the properly-started magnetic one
    as the answer, and a field held fixed to show that the field itself distorts
    the state it is used to find.
    """
    from tests.conftest import GENERATED

    text = (GENERATED / "h-atom-lsda.in").read_text()
    directory = tmp_path_factory.mktemp("reducebf")
    marker = text.lower().index("&system") + len("&system")
    unmagnetised = text.replace(
        "starting_magnetization(1) = 0.6", "starting_magnetization(1) = 0.0"
    )
    card = "LOCAL_MAGNETIC_FIELDS\n 0.0 0.0 0.10\n"

    def variant(source: str, extra: str, name: str) -> Path:
        path = directory / name
        path.write_text(source[:marker] + extra + source[marker:] + card)
        return path

    nonmagnetic = directory / "nonmagnetic.in"
    nonmagnetic.write_text(unmagnetised)
    fading = variant(unmagnetised, "\n    reducebf = 0.5\n", "fading.in")
    held = variant(unmagnetised, "\n", "held.in")

    _, magnetic = _run(GENERATED / "h-atom-lsda.in", pseudo_dir)
    _, stuck = _run(nonmagnetic, pseudo_dir)
    system, reduced = _run(fading, pseudo_dir)
    _, fixed = _run(held, pseudo_dir)

    assert system.atomic_b_field == ((0.0, 0.0, 0.1),)
    # Without the field the run never finds the magnetic state at all.
    assert stuck.magnetization == pytest.approx(0.0, abs=1e-12)
    assert stuck.total_energy - magnetic.total_energy > 0.05

    # With it, and with reducebf, it finds it *and* ends up field-free.
    assert reduced.magnetization == pytest.approx(magnetic.magnetization, abs=1e-8)
    assert reduced.total_energy == pytest.approx(magnetic.total_energy, abs=1e-9)

    # A field left switched on finds the state too, and shifts it.
    assert fixed.magnetization == pytest.approx(magnetic.magnetization, abs=1e-8)
    assert abs(fixed.total_energy - magnetic.total_energy) > 1e-5


def _fsm(pseudo_dir, scheme, max_iterations):
    """``fe-fsm.in`` run with one of the two update rules."""
    from tests.conftest import GENERATED

    text = (GENERATED / "fe-fsm.in").read_text()
    marker = text.index("&system") + len("&system")
    source = text[:marker] + f"\n    fsm_update = '{scheme}'\n" + text[marker:]
    system = build_system(parse_pw_input(source))
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    return system, run_scf(
        system, pseudos, conv_thr=1e-8, max_iterations=max_iterations,
        mixing_beta=float(parse_pw_input(source).get("electrons", "mixing_beta") or 0.7),
    )


def test_fixed_spin_moment_holds_the_moment(pseudo_dir):
    """Elk's ``fsmtype``: a field driven by the moment's error, not a penalty.

    The difference from QE's schemes is why Elk has both. A penalty adds
    ``lambda (m - m_fix)^2`` to the energy and leaves a residual force on the
    moment -- the constrained state sits wherever the penalty balances the
    functional, which for iron's ``lambda = 0.005`` is 1.68 mu_B against a
    target of 0.5. The feedback scheme instead *searches* for the field at which
    the unconstrained functional puts the moment where it was asked, so what
    converges is a genuine stationary point under that field -- and the field it
    found is a result worth reading, which is why Elk prints it and
    :attr:`SCFResult.magnetic_field` carries it.

    bcc iron held at 2.0 mu_B, where it wants 3.18. The moment is part of the
    *convergence test*: the field is outside the density, so ``dr2`` falls below
    ``conv_thr`` long before the moment arrives.

    The budget is a **performance guard**, not a tolerance. The default
    ``secant`` update takes 74 iterations on this case; the interleaved rule it
    replaced took 1380, and if a change puts this back into the hundreds it is
    the controller that broke, not the physics.
    """
    system, result = _fsm(pseudo_dir, "secant", 300)

    assert system.constrained_magnetization == "fsm"
    assert system.fsm_update == "secant"
    assert result.converged
    assert result.iterations < 200
    assert result.magnetization == pytest.approx(2.0, abs=1e-3)
    # No penalty in the energy -- the constraint is entirely in the field.
    assert result.constraint_energy == pytest.approx(0.0, abs=1e-14)
    # ... and the field is one the run found for itself.
    assert abs(float(np.asarray(result.magnetic_field.uniform)[0])) > 1e-3


def test_the_two_fixed_spin_moment_rules_find_the_same_field(pseudo_dir):
    """``secant`` and ``elk`` are the same answer at different cost.

    The transcription is kept and checked against the scheme that replaced it,
    the way every pluggable piece here is: what may differ is the path, never
    the fixed point. Both stop as soon as ``|m - 2|`` is inside 1e-3 and they
    approach from opposite sides, so the residual difference in the field is
    that tolerance divided by the susceptibility -- 1.1e-3 mu_B over the
    45 mu_B/Ry measured on this case, which is the 4e-5 Ry asserted below.

    **Why the interleaved rule is slow, since the number invites the question.**
    Elk updates the field after *every* SCF iteration, so the controller reads a
    moment that has not finished responding to the last nudge. Instrumented on
    this case, the susceptibility it appears to see swings between ``+2591`` and
    ``-1252`` mu_B/Ry between consecutive iterations. The gain is not what is
    wrong: Elk's ``tau = 0.02`` against a measured ``1/chi`` of 0.022 is already
    a Newton step. At converged density ``m(B)`` is smooth -- 2.499, 2.274,
    2.036, 1.837 mu_B at ``B = 0``, -0.005, -0.010, -0.020 Ry -- which is what
    the secant rule steps on.

    **How long the ringing takes is itself chaotic, and the assertion below says
    only what survives that.** Measured at 1380 iterations once and at 288
    another time, and what separated the two runs was ``|psi|^2`` being evaluated
    as ``Re(conj(psi) psi)`` rather than ``abs(psi)**2`` -- the same number to
    **3.5 eps** (:func:`pypresso.scf.density.band_density`). A marginally damped
    controller has no well-defined damping time at that resolution, so an earlier
    ``secant.iterations * 5 < elk.iterations`` was asserting a number that does
    not exist. Every *physics* assertion above is unaffected: both rules reach
    the same field, the same energy and the same moment, which is what the test
    is for.
    """
    _, secant = _fsm(pseudo_dir, "secant", 300)
    _, elk = _fsm(pseudo_dir, "elk", 2000)

    assert secant.converged and elk.converged
    field_secant = float(np.asarray(secant.magnetic_field.uniform)[0])
    field_elk = float(np.asarray(elk.magnetic_field.uniform)[0])
    assert field_secant == pytest.approx(field_elk, abs=1e-4)
    assert secant.total_energy == pytest.approx(elk.total_energy, abs=1e-4)
    assert secant.magnetization == pytest.approx(2.0, abs=1e-3)
    assert elk.magnetization == pytest.approx(2.0, abs=1e-3)
    # The point of the exercise: the secant rule is the cheaper path to the same
    # fixed point. By how much is not assertable -- see the docstring.
    assert secant.iterations < elk.iterations
