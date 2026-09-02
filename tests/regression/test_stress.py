"""P11 check: the stress tensor against Quantum ESPRESSO, four ways.

**The QE comparison** runs over two families. The first is QE's own test suite,
wherever an input already carries ``tstress = .true.`` -- silicon with explicit,
crystal-coordinate and Gamma-only k-points, with occupations from input, a
metal, an ultrasoft molecule, and an LSDA metal. The second is five inputs
generated here, each adding one thing to the one before:

* ``si2-nc-stress``     -- norm-conserving LDA on a *displaced* cell, so the
  tensor is anisotropic and a shear component that should be nonzero is.
* ``si2-nc-sheared``    -- the same dataset in a cell with ``a3`` tilted, which
  leaves the crystal with two symmetry operations and every entry of ``sigma``
  free. It is the case a transposed deformation or a sign error in
  ``at_strain`` cannot survive, and the only one whose reference is compared
  almost unsymmetrised.
* ``si2-us-stress``     -- ultrasoft, which adds the augmentation charge's own
  strain derivative (QE's ``addusstress``) and the first nonzero ``stres_cc``.
* ``si2-paw-stress``    -- PAW, whose one-centre terms reach the stress through
  ``becsum`` and nothing else.
* ``si2-us-pbe-stress`` -- PBE, which adds ``stres_gradcorr``.
* ``ni-ldau-stress``    -- DFT+U, where the Hubbard energy is measured through
  projectors that are atomic orbitals at ``k + G`` and therefore move with the
  cell. That dependence *is* QE's ``stres_hub.f90`` -- 2291 lines -- and it
  arrives here by differentiating through the projectors, exactly as
  ``force_hub`` did (P20). It is also the case that catches this phase's
  k-point trap, since a Hubbard projector built at the unstrained k-points is
  wrong without being an error.

**The term-by-term comparison** is the sharper of the two, and it is available
because these references are generated with ``verbosity = 'high'``, where
``stress.f90`` prints each contribution in kbar. Every term is compared: a total
can be right with two terms wrong in opposite directions, and this project has
had that happen once already, in the forces.

**The two implementations against each other.** The autodiff stress and the
transcription of ``stres_knl``/``stres_har``/``stres_loc``/``stres_cc``/
``stres_gradcorr``/``stres_ewa`` share no machinery -- one differentiates the
energy, the other evaluates six hand-derived expressions -- and unlike the two
force methods they have *nothing* separating them, since neither has a
``force_corr``-like correction. They must therefore agree to round-off, and they
do, at 5e-16.

**The finite differences** are two, and they check different things. One is a
central difference of the frozen-state energy itself, which is exact and
isolates a bug in the gradient; it is in ``tests/unit/test_stress_machinery.py``
because it needs no SCF. The other re-converges the SCF at strained cells with
the plane-wave sphere *rebuilt*, and it does not agree exactly -- the difference
is the **Pulay stress** of a finite basis, measured here against the cutoff.
"""

import dataclasses
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from defumat.forces.energy import state_from_result
from defumat.io import read_qe_output
from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf import Calculation, run_scf
from defumat.stress import compute_stress
from defumat.stress.analytic import analytic_terms
from defumat.stress.autodiff import autodiff_stress_terms
from defumat.system import build_system
from defumat.system.symmetry import symmetrize_matrix
from tests.conftest import reference_output
from defumat.units import RY_TO_KBAR
from tests.tolerances import STRESS_RY_BOHR3, TOTAL_ENERGY_RY

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"

#: The five generated cases, in the order they add capability.
GENERATED = [
    "si2-nc-stress",
    "si2-nc-sheared",
    "si2-us-stress",
    "si2-paw-stress",
    "si2-us-pbe-stress",
    "ni-ldau-stress",
]

#: QE test-suite inputs that carry ``tstress = .true.`` and whose
#: pseudopotentials are committed here.
BORROWED = [
    ("pw_scf", "scf.in"),
    ("pw_scf", "scf-occ.in"),
    ("pw_scf", "scf-kcrys.in"),
    ("pw_scf", "scf-k0.in"),
    ("pw_uspp", "uspp2.in"),
    ("pw_lsda", "lsda.in"),
    ("pw_metal", "metal.in"),
]

#: QE's committed 6.x benchmarks stop at ``conv_thr = 1e-6`` and print the
#: tensor to 8 decimals; where a regenerated reference exists both codes sit at
#: 1e-10 and agree three orders of magnitude better than this. The tolerance is
#: `PLAN.md`'s and the measured numbers are in its P11 section.
STRESS = STRESS_RY_BOHR3

#: The two implementations have nothing between them, unlike the two force
#: methods, so round-off is the whole of their disagreement.
METHOD_AGREEMENT = 1e-12


@lru_cache(maxsize=None)
def _converged(input_path: Path, pseudo_dir: Path):
    system = build_system(read_pw_input(input_path))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-10,
                     max_iterations=100)
    return system, calculation, result


def _symmetrised(calculation, tensor):
    """What ``compute_stress`` does to a raw tensor, applied to QE's as well."""
    tensor = 0.5 * (np.asarray(tensor) + np.asarray(tensor).T)
    if calculation.system.nosym or calculation.symmetries.nsym <= 1:
        return tensor
    return symmetrize_matrix(tensor, calculation.system.cell, calculation.symmetries)


# --- against Quantum ESPRESSO ------------------------------------------------
@pytest.mark.parametrize("case", GENERATED)
def test_generated_case_matches_qe(pseudo_dir, case):
    reference = read_qe_output(CASES / f"reference.out.{case}")
    _, calculation, result = _converged(CASES / f"{case}.in", pseudo_dir)

    assert result.converged
    assert result.total_energy == pytest.approx(reference.total_energy, abs=TOTAL_ENERGY_RY)

    stress = compute_stress(calculation, result)
    assert stress.tensor == pytest.approx(np.asarray(reference.stress), abs=STRESS)
    # The pressure is the tensor's own trace, so it is held to the *same*
    # tolerance and not to an invented second one: comparing it against QE's
    # printed ``P=`` at 0.01 kbar is a bound of 7e-8 Ry/bohr^3, tighter than the
    # tensor is asked to meet and tighter than the DFT+U case's own floor
    # (PLAN.md P11 trap 5). What it does check is that ``Stress.pressure``
    # really is ``tr sigma / 3`` in the unit it claims.
    assert stress.pressure == pytest.approx(
        float(np.trace(np.asarray(reference.stress)) / 3.0), abs=STRESS
    )
    assert stress.pressure_kbar == pytest.approx(reference.pressure, abs=1.0)


@pytest.mark.parametrize(("directory", "name"), BORROWED)
def test_testsuite_case_matches_qe(qe_testsuite, pseudo_dir, directory, name):
    path = reference_output(directory, name, qe_testsuite)
    if path is None:
        pytest.skip(f"no benchmark for {directory}/{name}")
    reference = read_qe_output(path)
    if reference.stress is None:
        pytest.skip("this reference carries no stress")

    _, calculation, result = _converged(qe_testsuite / directory / name, pseudo_dir)
    stress = compute_stress(calculation, result)
    assert stress.tensor == pytest.approx(np.asarray(reference.stress), abs=STRESS)


def test_the_rotational_residue_is_round_off(pseudo_dir):
    """``dE/d(eps)`` is symmetric because the energy is rotationally invariant.

    That argument uses no symmetry of the crystal, so the antisymmetric part is
    a check on the gradient that survives on the sheared cell, where the point
    group has two operations and averages almost nothing away.
    """
    _, calculation, result = _converged(CASES / "si2-nc-sheared.in", pseudo_dir)
    stress = compute_stress(calculation, result)
    assert stress.rotational_residue < 1e-12


# --- term by term ------------------------------------------------------------
#: How QE's printed contributions map onto the energy's, which is not one to one.
#: ``exc-cor`` already carries the gradient correction and the core charge is a
#: row of its own; the augmentation charge has no row here at all, which is why
#: this comparison runs on the norm-conserving cases.
_TERM_MAP = {
    "kinetic": ("kinetic",),
    "local": ("local",),
    "hartree": ("hartree",),
    "ewald": ("ewald",),
    "nonlocal": ("nonlocal",),
    "xc": ("xc",),
}

#: QE prints the per-term table in kbar with two decimals, so 0.01 kbar --
#: 6.8e-8 Ry/bohr^3 -- is the floor on this comparison. It is met.
TERM_PRINTING_FLOOR = 1e-7


@pytest.mark.parametrize("case", ["si2-nc-stress", "si2-nc-sheared"])
def test_every_term_matches_qes_own_breakdown(pseudo_dir, case):
    reference = read_qe_output(CASES / f"reference.out.{case}")
    assert reference.stress_terms, "the reference was not run with verbosity='high'"

    _, calculation, result = _converged(CASES / f"{case}.in", pseudo_dir)
    ours = autodiff_stress_terms(calculation, state_from_result(result))

    for qe_name, names in _TERM_MAP.items():
        theirs = _symmetrised(calculation, reference.stress_terms[qe_name])
        mine = _symmetrised(calculation, sum(np.asarray(ours[n]) for n in names))
        assert mine == pytest.approx(theirs, abs=TERM_PRINTING_FLOOR), qe_name

    # ... and they must add up to the total, which is not implied by each of
    # them matching: QE prints a row this code has no counterpart for whenever
    # one is nonzero, and a silently dropped contribution would show here.
    total = _symmetrised(calculation, sum(np.asarray(v) for v in ours.values()))
    assert total == pytest.approx(np.asarray(reference.stress), abs=STRESS)


@pytest.mark.parametrize("case", ["si2-nc-stress", "si2-nc-sheared"])
def test_the_two_implementations_agree(pseudo_dir, case):
    """Autodiff against QE's transcribed expressions, term by term.

    Nothing separates these two -- no ``force_corr``, no convergence
    correction -- so anything above round-off is a bug in one of them. The
    exchange-correlation family is compared as a group because the split is
    different on the two sides: one ``etxc`` differentiated here, a diagonal
    plus ``stres_cc`` plus ``stres_gradcorr`` there.
    """
    _, calculation, result = _converged(CASES / f"{case}.in", pseudo_dir)
    state = state_from_result(result)
    ours = {k: np.asarray(v) for k, v in autodiff_stress_terms(calculation, state).items()}
    theirs = analytic_terms(calculation, state)

    for name in ("kinetic", "hartree", "local", "ewald"):
        assert 0.5 * (ours[name] + ours[name].T) == pytest.approx(
            0.5 * (theirs[name] + theirs[name].T), abs=METHOD_AGREEMENT
        ), name

    xc = theirs["xc"] + theirs["core"] + theirs["gradcorr"]
    assert 0.5 * (ours["xc"] + ours["xc"].T) == pytest.approx(
        0.5 * (xc + xc.T), abs=METHOD_AGREEMENT
    )


def test_the_transcription_reaches_the_gradient_correction(pseudo_dir):
    """The PBE case, where ``stres_gradcorr`` is the term that has to be right.

    On an LDA run the gradient correction is identically zero and the
    comparison above passes without touching it.
    """
    _, calculation, result = _converged(CASES / "si2-us-pbe-stress.in", pseudo_dir)
    state = state_from_result(result)
    theirs = analytic_terms(calculation, state)
    assert np.abs(theirs["gradcorr"]).max() > 1e-5
    assert np.abs(theirs["core"]).max() > 1e-5


# --- the pressure identity and the Pulay stress ------------------------------
def _energy_at_scale(pseudo_dir, scale: float, ecutwfc: float):
    """The converged total energy of the ideal silicon cell at ``alat * scale``.

    **The lattice parameter is what is scaled**, through the input, rather than
    ``cell.at`` afterwards. Scaling the vectors alone leaves the k-points where
    they were in *cartesian* space -- ``KPoints.coords`` are in units of
    ``2 pi / alat`` and ``alat`` is a static field -- so the finite difference
    would be taken along a path on which the Brillouin-zone sampling changes.
    That is this phase's trap in its most expensive form: the first version of
    this test measured a "Pulay stress" of 47 kbar that did not fall with the
    cutoff, because it was not a Pulay stress.

    The cell is rebuilt from scratch, so the plane-wave sphere is *reselected*
    at the new lattice constant, which is what makes this an independent check
    rather than a re-derivation of the frozen-basis gradient -- and what makes
    it disagree by the Pulay stress that is really there.
    """
    import equinox as eqx
    import jax.numpy as jnp

    pwin = read_pw_input(CASES / "si2-nc-stress.in")
    pwin.namelists["system"]["celldm"] = {(1,): 10.20 * scale}
    pwin.namelists["system"]["ecutwfc"] = ecutwfc
    pwin.namelists["system"]["ecutrho"] = 4.0 * ecutwfc
    system = build_system(pwin)

    # ...on the ideal sites, so that what is measured is the equation of state
    # and not a displaced atom relaxing along with it.
    positions = np.array(system.structure.positions)
    positions[1] = np.array([0.25, 0.25, 0.25]) * float(system.cell.alat)
    system = eqx.tree_at(lambda s: s.structure.positions, system, jnp.asarray(positions))

    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-11,
                     max_iterations=120)
    return system, calculation, result


#: Where the finite difference is taken. The test suite's ``ecutwfc = 12`` is
#: chosen for speed and is nowhere near basis-set convergence, so its Pulay
#: stress (91 kbar, larger than the pressure and of the opposite sign) swamps
#: the comparison; by 40 Ry it is under 2 kbar. The sweep is in `PLAN.md`'s P11
#: section.
FD_ECUTWFC = 40.0

#: What the Pulay stress leaves behind at that cutoff, in kbar. Measured, not
#: guessed -- see the sweep.
PULAY_KBAR = 3.0


def test_pressure_is_minus_de_by_dv(pseudo_dir):
    """``tr sigma / 3 = -dE/dV``, against an SCF re-converged at two volumes.

    The sharpest statement the stress makes, because it needs no reference at
    all: the pressure is a *thermodynamic* derivative, and reproducing it from a
    strain gradient taken at one volume is what says the gradient is the
    derivative of the energy the SCF actually minimises.

    The two do not agree exactly and the residue has a name. The plane-wave
    sphere is frozen while differentiating and reselected at each volume of the
    finite difference, so what is left over is the **Pulay stress** of a finite
    basis -- which is why the comparison is made at a cutoff where that is
    small, and why the number it is held to is one that was measured.
    """
    step = 0.005  # in the linear scale, so a volume step of ~1.5%
    _, calculation, result = _energy_at_scale(pseudo_dir, 1.0, FD_ECUTWFC)
    stress = compute_stress(calculation, result)

    energies, volumes = [], []
    for scale in (1.0 - step, 1.0 + step):
        _, moved, moved_result = _energy_at_scale(pseudo_dir, scale, FD_ECUTWFC)
        energies.append(moved_result.total_energy)
        volumes.append(float(moved.system.cell.volume))

    numerical = -(energies[1] - energies[0]) / (volumes[1] - volumes[0])
    residue = abs(stress.pressure - numerical) * RY_TO_KBAR
    assert residue < PULAY_KBAR, (
        f"tr sigma/3 = {stress.pressure_kbar:.2f} kbar against "
        f"-dE/dV = {numerical * RY_TO_KBAR:.2f} kbar"
    )
    # ... and the pressure itself must be a real number rather than a small one
    # that happens to sit inside the tolerance: the two agree to a few per cent
    # of the *kinetic* term, which is the scale the cancellations happen on.
    assert abs(stress.pressure) > 1e-6


def test_the_frozen_energy_at_zero_strain_is_the_scf_total(pseudo_dir):
    """The gate on the whole path, as it is for the forces.

    ``strained_energy(0)`` has to *be* the total energy before its derivative
    can be the stress: it reassembles QE's decomposition out of different pieces
    and must land on the same number.
    """
    import jax.numpy as jnp

    from defumat.stress.energy import strained_energy

    for case in ("si2-nc-sheared", "si2-us-stress", "si2-paw-stress"):
        _, calculation, result = _converged(CASES / f"{case}.in", pseudo_dir)
        energy = float(strained_energy(
            calculation, jnp.zeros((3, 3)), state_from_result(result)
        ))
        assert energy == pytest.approx(result.total_energy, abs=1e-8), case


def test_tstress_puts_the_tensor_on_the_result(pseudo_dir):
    """``tstress`` is QE's ``&control`` switch, and all three of its states work.

    The input decides by default, an explicit argument overrides it either way.
    The first version of this test asked whether a run of ``si2-nc-stress.in``
    with no argument had *no* stress -- which was right before ``run_scf``
    started reading the input and wrong afterwards, and is exactly the thing
    worth pinning down now.
    """
    system = build_system(read_pw_input(CASES / "si2-nc-stress.in"))
    assert system.tstress
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    reference = read_qe_output(CASES / "reference.out.si2-nc-stress")

    # The input says yes and nothing overrides it.
    from_input = run_scf(system, pseudos, conv_thr=1e-8)
    assert from_input.stress is not None
    assert from_input.stress.tensor == pytest.approx(
        np.asarray(reference.stress), abs=STRESS
    )

    # An explicit False wins over the input...
    assert run_scf(system, pseudos, conv_thr=1e-8, tstress=False).stress is None

    # ...and an explicit True wins over an input that does not ask.
    quiet = dataclasses.replace(system, tstress=False)
    assert run_scf(quiet, pseudos, conv_thr=1e-8).stress is None
    assert run_scf(quiet, pseudos, conv_thr=1e-8, tstress=True).stress is not None


def test_an_input_asking_for_an_impossible_stress_warns_rather_than_raising(pseudo_dir):
    """``tstress = .true.`` on a regime P11 does not cover must not fail the SCF.

    Three of QE's own spin-orbit benchmarks carry ``tstress = .true.``, and
    before this was handled every one of them ended in a ``NotImplementedError``
    from a *stress* nobody in those tests wanted -- an optional diagnostic
    failing the calculation that produced it. QE's own convention is the one
    followed here: ``input.f90`` switches ``tstress`` off again for combinations
    it cannot do, and ``stress()`` opens with an ``infomsg`` and a bare
    ``RETURN`` for the rest.

    Asked for **by hand** it still raises, and that distinction is the whole
    point: a flag left in a file is not a request for a number.
    """
    system = build_system(read_pw_input(CASES / "h-chain-90deg.in"))
    assert not system.tstress
    system = dataclasses.replace(system, tstress=True)
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)

    with pytest.warns(RuntimeWarning, match="tstress"):
        result = run_scf(system, pseudos, conv_thr=1e-6, max_iterations=40)
    assert result.stress is None
    assert result.total_energy is not None  # the SCF itself was unaffected

    with pytest.raises(NotImplementedError, match="noncollinear"):
        run_scf(system, pseudos, conv_thr=1e-6, max_iterations=40, tstress=True)


def test_a_noncollinear_run_is_refused_rather_than_approximated(pseudo_dir):
    """``pw_noncolin/noncolin.in`` is the rung this phase did not reach."""
    import jax.numpy as jnp

    from defumat.stress.energy import strained_energy

    system = build_system(read_pw_input(CASES / "h-chain-90deg.in"))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    with pytest.raises(NotImplementedError, match="noncollinear"):
        strained_energy(calculation, jnp.zeros((3, 3)), None)
