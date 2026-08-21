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
from pypresso.io.pwin import read_pw_input
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


def test_fixed_spin_moment_holds_the_moment(pseudo_dir):
    """Elk's ``fsmtype``: a field driven by the moment's error, not a penalty.

    The difference from QE's schemes is why Elk has both. A penalty adds
    ``lambda (m - m_fix)^2`` to the energy and leaves a residual force on the
    moment -- the constrained state sits wherever the penalty balances the
    functional, which for iron's ``lambda = 0.005`` is 1.68 mu_B against a
    target of 0.5. The feedback scheme instead *searches* for the field at which
    the unconstrained functional puts the moment where it was asked
    (``bfieldfsm.f90``), so what converges is a genuine stationary point under
    that field -- and the field it found is a result worth reading, which is why
    Elk prints it and :attr:`SCFResult.magnetic_field` carries it.

    bcc iron held at 2.0 mu_B, where it wants 3.18. **This takes several hundred
    iterations**, which is the scheme rather than this implementation: the
    feedback is only stable for a small ``tau`` (at 0.1 the moment flips and the
    run saturates at 8 mu_B), and Elk's own default is 0.01. It is also why the
    moment is part of the *convergence test* here -- the field is outside the
    density, so ``dr2`` falls below ``conv_thr`` long before the moment arrives.

    **How many hundred is not a stable number.** A proportional controller rings
    before it settles, and where it starts ringing from depends on the starting
    wavefunctions: this case took ~350 iterations until P20 made the atomic
    orbitals go through QE's ``upf_check_atwfc_norm`` renormalisation
    (:func:`pypresso.pseudo.upf._renormalize_orbitals`), which changed nothing
    but the eigensolver's seed and moved it to 746. The moment it converges to
    is the same, and so is the field it finds. The budget below is set with room
    for that, and the number itself is not a claim about anything.
    """
    from tests.conftest import GENERATED

    system, result = _run(
        GENERATED / "fe-fsm.in", pseudo_dir, conv_thr=1e-8, max_iterations=1200
    )

    assert system.constrained_magnetization == "fsm"
    assert result.converged
    assert result.magnetization == pytest.approx(2.0, abs=1e-3)
    # No penalty in the energy -- the constraint is entirely in the field.
    assert result.constraint_energy == pytest.approx(0.0, abs=1e-14)
    # ... and the field is one the run found for itself.
    assert abs(float(np.asarray(result.magnetic_field.uniform)[0])) > 1e-3
