"""P9 check: collinear spin polarization against Quantum ESPRESSO.

**Every LSDA benchmark in QE's test-suite is ultrasoft or PAW.** There is no
norm-conserving spin-polarized case anywhere in it, so per-spin ``becsum``,
per-spin ``D_ij`` and an augmentation charge in each channel were part of this
phase from the first stage rather than something to add later.

The cases build on each other and each isolates one thing:

* ``atom-lsda`` -- an oxygen atom at Gamma with ``occupations='from_input'``
  given per channel. There is no Fermi search at all, so what it tests is the
  whole of the plumbing and nothing else: the ``(nspin, ...)`` density, a
  Hartree term built from the **total** density and copied to both channels, an
  exchange-correlation potential that is not, per-spin ``becsum``/``D_ij``, and
  mixing over both channels.
* ``lsda`` -- fcc nickel with Marzari-Vanderbilt smearing on a 4x4x4 grid: one
  Fermi level shared by the channels, ``DEGSPIN`` dropped to 1 so the weights
  sum to one *per channel*, and a magnetization that is an output rather than an
  input. Its ``conv_thr`` is 1e-10, so the comparison is as tight as any here.
* ``lsda-tot_magnetization`` / ``lsda-nelup+neldw`` -- the same cell with the
  magnetization constrained, which means **two** independent Fermi levels and a
  ``-TS`` summed over both. The two inputs differ only in how they spell the
  constraint and must give the same answer.
* ``atom-sigmapbe`` -- spin-polarized PBE. Exchange by the spin-scaling
  relation, correlation by the PW92 spin interpolation, and a gradient
  correction whose cross term ``v2c_ud`` exists because correlation depends on
  the *total* density's gradient.
* ``o-paw-spin`` / ``o-paw-spin-pbe`` -- PAW one-centre terms per channel, LDA
  and PBE. ``PAW_h_potential`` sums over spin before solving the radial Poisson
  equation and copies the one answer into both channels; the exchange-correlation
  pass runs per channel on the spherical quadrature, and with a GGA so does the
  vector field whose divergence is subtracted.
* ``paw-atom_spin_lda`` -- the LDA half of the pair QE *ships*. It is the only
  case here with ``nosym``, and it needs it: its minority channel occupies one of
  three p orbitals, which is not a state the crystal's symmetry admits. That also
  makes it nearly degenerate under *which* orbital, which is why the two cases
  above exist -- see ``TOTAL_RY``.

References are regenerated with the vendored ``pw.x`` at ``conv_thr = 1e-10``
(``tools/generate_reference.py``). The committed benchmarks are QE 6.1 runs
stopped at 1e-6, where the printed energy *terms* are only good to about 1e-4 Ry
and comparing against them would measure QE's stopping point.
"""

import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from pypresso.io import read_qe_output
from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf import run_scf
from pypresso.system import build_system
from pypresso.units import RY_TO_EV
from tests.conftest import reference_output
from tests.tolerances import (
    EIGENVALUE_EV,
    ENERGY_TERM_RY,
    FERMI_EV,
    MAGNETIZATION_BOHRMAG,
    TOTAL_ENERGY_RY,
)

pytestmark = [pytest.mark.regression, pytest.mark.slow]

#: ``(directory, input, occupied bands per channel)``. The band count is needed
#: because an isolated atom's *empty* levels are diffuse states of the periodic
#: box, and QE interpolates its local potential from a ``dq = 0.01`` table where
#: this code integrates directly -- a difference that averages out over a bound
#: state's many plane waves and does not over a state spread across the cell.
#: ``None`` means every computed band is compared, which is right for a metal
#: where nothing is diffuse.
#: ``directory = None`` means an input committed under ``tests/data/qe`` rather
#: than borrowed from QE's test-suite.
CASES = [
    ("pw_atom", "atom-lsda.in", 4),
    ("pw_lsda", "lsda.in", None),
    ("pw_lsda", "lsda-tot_magnetization.in", None),
    ("pw_lsda", "lsda-nelup+neldw.in", None),
    ("pw_atom", "atom-sigmapbe.in", 4),
    (None, "o-paw-spin.in", 4),
    (None, "o-paw-spin-pbe.in", 4),
    ("pw_pawatom", "paw-atom_spin_lda.in", 4),
    # ``pw_pawatom/paw-atom_spin.in`` -- the PBE half of the pair QE ships -- is
    # deliberately absent. Its landscape is flat enough that neither code
    # converges it quickly (QE takes 71 iterations on the LDA version and fails
    # outright at mixing_beta = 0.3), and a PBE PAW iteration on its 64^3 grid is
    # expensive enough that the case alone would dominate the suite's runtime.
    # ``o-paw-spin-pbe`` above exercises the identical code path -- the
    # per-channel vector field on the sphere, its theta component divided by
    # sin(theta), and correlation's cross term -- on a state that is actually
    # well conditioned, which is the better test as well as the cheaper one.
]

IDS = [f"{directory or 'local'}/{name}" for directory, name, _ in CASES]

GENERATED = Path(__file__).resolve().parents[1] / "data" / "qe"

#: Energy *terms* are first-order in the density residual where the total is
#: second-order, so what ``conv_thr`` promises them is the Cauchy-Schwarz bound
#: ``sqrt(4 E_H dr2)`` -- 8e-5 Ry for an isolated atom's ``E_H = 17..21 Ry`` at
#: ``dr2 = 1e-10``, against 2e-5 for a small crystal's. Measured, not argued:
#: tightening this code's own ``conv_thr`` from 1e-10 to 1e-13 moves the Hartree
#: term of ``atom-lsda`` by 3e-5 Ry while moving its total by 3e-11.
TERM_RY = {
    "pw_lsda": 2e-5,
    "pw_atom": 5e-5,
    # A *polarized* isolated atom in a 25-bohr box is the loosest thing here: the
    # observed term differences are 5e-4 for the well-conditioned pair and 1e-3
    # for the two QE ships. That is not this code's arithmetic -- see the control
    # below -- it is where two different mixers stop on a very flat landscape.
    None: 1e-3,
    "pw_pawatom": 2e-3,
}

#: And once more for the eigenvalues. The two millielectronvolts of
#: ``EIGENVALUE_EV`` are what every other case here meets; the occupied levels of
#: ``paw-atom_spin_lda`` come out 4 meV from QE's for the same reason its terms
#: do -- two mixers stopping in different places on a landscape flat enough that
#: QE needs 71 iterations to cross it. The *degeneracy structure* is identical
#: (a 2 + 1 split of each channel's p manifold, the pair degenerate to 6e-5 eV),
#: which is what says the two codes found the same state and not two different
#: ones.
EIGENVALUE_EV_BY_DIRECTORY = {"pw_pawatom": 1e-2}

#: The same story for the total, which is second-order in the residual where the
#: terms are first-order. Achieved: <= 5e-9 Ry for the five non-PAW cases,
#: 2.0e-7 for the well-conditioned PAW pair, 4.1e-6 for the two QE ships.
#:
#: **The looseness is the cases', not the code's**, and there is a control that
#: says so: run the *unpolarized* PAW oxygen (``pw_pawatom/paw-atom_lda``, which
#: this code reproduces to 2.2e-9 Ry) with ``nspin = 2`` and the two channels
#: equal. Every spin path executes -- two Hamiltonians, per-channel ``becsum``
#: and ``D_ij``, the polarized functional at ``zeta = 0``, the one-centre terms
#: per channel, the magnetization term in ``dr2`` -- and the answer moves by
#: **2.0e-12 Ry**. What is left is physics: QE needs 71 iterations on
#: ``paw-atom_spin_lda``, fails to converge at all at ``mixing_beta = 0.3``, and
#: moves its own answer by 5.8e-7 Ry between ``mixing_ndim = 8`` and ``4``.
TOTAL_RY = {"pw_pawatom": 2e-5}


def _input_path(directory, name: str, qe_testsuite: Path) -> Path:
    return GENERATED / name if directory is None else qe_testsuite / directory / name


def _reference_path(directory, name: str, qe_testsuite: Path) -> Path:
    if directory is None:
        return GENERATED / f"reference.out.{Path(name).stem}"
    return reference_output(directory, name, qe_testsuite)


@lru_cache(maxsize=None)
def _converged(directory, name: str, qe_testsuite: Path, pseudo_dir: Path):
    path = _input_path(directory, name, qe_testsuite)
    pwin = read_pw_input(path)
    system = build_system(pwin)
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    # The input's own mixing beta: the isolated atoms ask for 0.25 and oscillate
    # at the 0.7 default, which is why QE's inputs set it.
    beta = float(pwin.get("electrons", "mixing_beta", 0.7))
    with warnings.catch_warnings():
        # The Gamma cases warn that the half-sphere storage is substituted for;
        # that substitution has its own test in test_isolated_atom.
        warnings.simplefilter("ignore", UserWarning)
        result = run_scf(
            system, pseudos, conv_thr=1e-10, max_iterations=300, mixing_beta=beta
        )
    return system, pseudos, result


@pytest.fixture(scope="module")
def case(qe_testsuite, pseudo_dir):
    def _run(directory: str, name: str):
        result = _converged(directory, name, qe_testsuite, pseudo_dir)
        reference = read_qe_output(_reference_path(directory, name, qe_testsuite))
        return result, reference

    return _run


@pytest.mark.parametrize(("directory", "name", "occupied"), CASES, ids=IDS)
def test_total_energy_matches_reference(case, directory, name, occupied):
    (_, _, result), reference = case(directory, name)
    assert result.converged
    assert result.nspin == 2
    assert result.total_energy == pytest.approx(
        reference.total_energy, abs=TOTAL_RY.get(directory, TOTAL_ENERGY_RY)
    )


@pytest.mark.parametrize(("directory", "name", "occupied"), CASES, ids=IDS)
def test_energy_terms_match_reference(case, directory, name, occupied):
    (_, _, result), reference = case(directory, name)

    assert set(result.energy_terms) == set(reference.energy_terms)
    for term, value in reference.energy_terms.items():
        # Ewald depends on no density at all and is held to the tight bound.
        tolerance = ENERGY_TERM_RY if term == "ewald" else TERM_RY[directory]
        assert result.energy_terms[term] == pytest.approx(value, abs=tolerance), term

    assert sum(result.energy_terms.values()) == pytest.approx(result.total_energy, abs=1e-10)


@pytest.mark.parametrize(("directory", "name", "occupied"), CASES, ids=IDS)
def test_magnetization_matches_reference(case, directory, name, occupied):
    """The number LSDA exists to produce.

    Both are compared: an antiferromagnet has zero total moment and a large
    absolute one, so checking only the first would pass on a calculation that had
    lost the magnetism entirely.
    """
    (_, _, result), reference = case(directory, name)
    assert result.magnetization == pytest.approx(
        reference.magnetization, abs=MAGNETIZATION_BOHRMAG
    )
    assert result.absolute_magnetization == pytest.approx(
        reference.absolute_magnetization, abs=MAGNETIZATION_BOHRMAG
    )


@pytest.mark.parametrize(("directory", "name", "occupied"), CASES, ids=IDS)
def test_eigenvalues_match_reference(case, directory, name, occupied):
    (_, _, result), reference = case(directory, name)

    # pw.x prints eigenvalues in eV; everything internal is in Ry.
    ours = result.eigenvalues_by_spin * RY_TO_EV
    theirs = reference.eigenvalues
    assert ours.shape[0] == theirs.shape[0] == 2
    stop = occupied or ours.shape[-1]
    tolerance = EIGENVALUE_EV_BY_DIRECTORY.get(directory, EIGENVALUE_EV)
    assert ours[:, :, :stop] == pytest.approx(theirs[:, :, :stop], abs=tolerance)


def test_shared_fermi_level_matches_reference(case):
    """Nickel: one Fermi level for both channels, as an unconstrained run has."""
    (_, _, result), reference = case("pw_lsda", "lsda.in")
    assert result.fermi_energy_up is None
    assert result.fermi_energy * RY_TO_EV == pytest.approx(
        reference.fermi_energy, abs=FERMI_EV
    )


@pytest.mark.parametrize("name", ["lsda-tot_magnetization.in", "lsda-nelup+neldw.in"])
def test_two_fermi_levels_match_reference(case, name):
    """Constrained magnetization: one Fermi level per channel.

    This is the case that found the trap in the Fermi search. Cold and
    Methfessel-Paxton occupations *overshoot* -- a cold-smeared level reaches
    1.07 before settling at 1 -- so the electron count is not monotonic and
    ``N(E_F) = nelec`` has several roots. Nickel's majority channel with six
    electrons in it is nearly full, the count is nearly flat over a whole
    electron-volt, and a plain bisection lands on a root 0.74 eV away from QE's.
    The occupations are the same to 1e-5 either way and the density never
    notices; ``-TS`` is out by 3e-4 Ry, and so is the total energy.
    """
    (_, _, result), reference = case("pw_lsda", name)
    assert result.fermi_energy_up is not None
    assert result.fermi_energy_up * RY_TO_EV == pytest.approx(
        reference.fermi_energy_up, abs=FERMI_EV
    )
    assert result.fermi_energy_down * RY_TO_EV == pytest.approx(
        reference.fermi_energy_down, abs=FERMI_EV
    )


def test_the_two_spellings_of_the_constraint_agree(case):
    """``tot_magnetization = 2`` and ``= 2.0`` are the same physics.

    ``pw_lsda`` ships both because ``set_nelup_neldw`` treats an integer
    magnetization differently from a fractional one -- it truncates the electron
    count before splitting it -- and 2 and 2.0 have to come out on the same side
    of that test.
    """
    (_, _, one), _ = case("pw_lsda", "lsda-tot_magnetization.in")
    (_, _, two), _ = case("pw_lsda", "lsda-nelup+neldw.in")
    assert one.total_energy == pytest.approx(two.total_energy, abs=1e-10)


@pytest.mark.parametrize(("directory", "name", "occupied"), CASES, ids=IDS)
def test_the_density_integrates_to_the_occupations(case, directory, name, occupied):
    """``int (rho_up + rho_dw) = sum(wg)``, identically.

    With an augmentation charge in each channel this is the cheapest check that
    the per-spin ``becsum`` and the per-spin ``addusdens`` agree with each other:
    an ultrasoft state is normalised as ``<psi|S|psi> = 1`` and the part of the
    norm ``S`` supplies is exactly what ``Q_ij(G=0)`` puts back.

    Against the *occupation weights*, not against ``nelec``, and the difference
    is not pedantry. With ``tot_magnetization`` set, ``efermig`` refines each
    channel's Fermi level with Newton's method and accepts the result when the
    electron count is within ``eps_cold_MP = 1e-2`` -- which on nickel's nearly
    full majority channel it exploits, stopping 2.2e-5 electrons out. That is
    QE's answer and this code reproduces it, so the identity that has to hold
    exactly is the one between the density and the weights it was built from.
    """
    (system, pseudos, result), _ = case(directory, name)
    nelec = sum(pseudos[t].z_valence for t in system.structure.types)

    density = np.asarray(result.total_density)
    charge = float(np.sum(density)) * float(system.cell.volume) / density.size
    assert charge == pytest.approx(float(np.sum(result.occupations)), abs=1e-9)

    # And the weights themselves account for every electron -- exactly where one
    # Fermi level is solved for, to QE's own tolerance where two are.
    tolerance = 1e-2 if result.fermi_energy_up is not None else 1e-9
    assert float(np.sum(result.occupations)) == pytest.approx(nelec, abs=tolerance)


def test_an_unimplemented_spin_correlation_is_refused():
    """The P13 convention, applied to the spin slots.

    A correlation term with no polarized parameterisation must stop the run
    rather than fall back to the unpolarized fit, which would converge to a
    plausible number that matches nothing.
    """
    import dataclasses

    from pypresso.xc.functional import get_functional

    functional = get_functional("PZ")
    assert functional.supports_spin
    functional.require_spin()

    crippled = dataclasses.replace(functional, correlation_spin=None)
    assert not crippled.supports_spin
    with pytest.raises(NotImplementedError, match="spin-polarized"):
        crippled.require_spin()


def test_two_equal_channels_reproduce_the_unpolarized_answer(qe_testsuite, pseudo_dir, tmp_path):
    """The control that separates the spin *code* from the spin *physics*.

    ``pw_pawatom/paw-atom_lda`` with ``nspin = 2`` and the two channels occupied
    equally is the same calculation as with ``nspin = 1``, run through every
    spin-specific path there is: two Hamiltonians, ``becsum`` and ``D_ij`` per
    channel, the polarized functional evaluated at ``zeta = 0``, the one-centre
    terms per channel, and ``dr2``'s magnetization term. If any of them were
    wrong by the ~1e-7 Ry the polarized PAW cases sit at, it would show here --
    and nothing else would be in the way, because the unpolarized answer is
    already known to 2.2e-9 Ry.

    It is also the cheapest possible check that the ``(nspin, ...)`` migration
    did not quietly change what an unpolarized calculation computes.
    """
    source = (qe_testsuite / "pw_pawatom" / "paw-atom_lda.in").read_text()
    polarized = tmp_path / "equal-channels.in"
    polarized.write_text(
        source.replace("    nbnd = 6\n", "    nbnd = 6\n    nspin = 2\n").replace(
            "2. 1.333333333333 1.333333333333 1.333333333333   0. 0.",
            "1. 0.666666666667 0.666666666667 0.666666666667   0. 0.\n"
            "1. 0.666666666667 0.666666666667 0.666666666667   0. 0.",
        )
    )

    energies = {}
    for label, path in (("nspin=1", qe_testsuite / "pw_pawatom" / "paw-atom_lda.in"),
                        ("nspin=2", polarized)):
        system = build_system(read_pw_input(path))
        pseudos = tuple(
            read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = run_scf(system, pseudos, conv_thr=1e-10, max_iterations=200)
        energies[label] = result.total_energy
        if label == "nspin=2":
            assert result.nspin == 2
            assert result.magnetization == pytest.approx(0.0, abs=1e-12)

    assert energies["nspin=2"] == pytest.approx(energies["nspin=1"], abs=1e-10)
