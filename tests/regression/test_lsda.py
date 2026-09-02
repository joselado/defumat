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
* ``o-atom-fixed-lsda`` -- the same oxygen atom at ``occupations = 'fixed'``
  with ``tot_magnetization = 2``, which is the *only* shape fixed occupations
  have under LSDA: ``input.f90:784-800`` refuses the combination without a
  ``tot_magnetization`` and requires an integer one, so there is no shared-Fermi
  fixed branch to test. Each channel is then filled by ``iweights_only`` with
  ``degspin = 1`` -- four bands up and two down -- and the answer is the atom's
  Hund's-rule ground state, the same configuration ``atom-lsda`` reaches by
  writing the occupations out by hand.
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

from defumat.io import read_qe_output
from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf import run_scf
from defumat.system import build_system
from defumat.units import RY_TO_EV
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
    (None, "o-atom-fixed-lsda.in", 4),
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


def test_fixed_occupations_fill_each_channel_to_its_own_count(case):
    """``iweights_only`` per channel: four bands up, two down, and no Fermi search.

    The check that the *band counts* are right and not only the total. Six
    valence electrons with ``tot_magnetization = 2`` give ``nelup = 4`` and
    ``neldw = 2``, so the occupation weights are the k-point weight on the first
    four bands of channel 0 and the first two of channel 1, exactly -- there is
    no level to solve for and nothing fractional anywhere.

    The two reported levels are QE's own convention and they **coincide** here,
    which is the case's second point: the majority channel's highest occupied
    level is -9.64 eV and the minority channel's is -6.63, so the HOMO over both
    channels is -6.63 -- and the minority 2p shell is threefold degenerate with
    one electron in it, so the LUMO is that same -6.63. A fixed occupation that
    cuts a degenerate multiplet is what this combination is *for*, and it is also
    why the residual solver cannot take it (``test_scf_solvers``).
    """
    (_, _, result), reference = case(None, "o-atom-fixed-lsda.in")
    weights = np.asarray(result.occupations)
    assert weights.shape[0] == 2
    per_channel = (weights > 0).sum(axis=-1)
    assert list(per_channel[0]) == [4]
    assert list(per_channel[1]) == [2]
    # Full bands, not fractions: every nonzero weight is the k-point's own.
    nonzero = weights[weights > 0]
    assert np.allclose(nonzero, nonzero[0])

    assert result.fermi_energy is None
    assert result.homo * RY_TO_EV == pytest.approx(reference.homo, abs=FERMI_EV)
    assert result.lumo * RY_TO_EV == pytest.approx(reference.lumo, abs=FERMI_EV)
    # ...and each channel's own highest occupied level is carried beside them,
    # which is what ``iweights`` returns as ef_up and ef_dw.
    assert result.fermi_energy_up * RY_TO_EV == pytest.approx(-9.6440, abs=1e-3)
    assert result.fermi_energy_down * RY_TO_EV == pytest.approx(-6.6339, abs=1e-3)


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

    from defumat.xc.functional import get_functional

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


# --------------------------------------------------------------------------
# LSDA meets P8: the NSCF grid run and the density of states.
# --------------------------------------------------------------------------


def test_nscf_on_a_denser_grid_matches_reference(qe_testsuite, pseudo_dir):
    """``pw_lsda/lsda-2.in``: nickel's Fermi level on an 8x8x8 grid.

    The second half of the pair QE ships -- ``lsda.in`` converges the density on
    a 4x4x4 grid and ``lsda-2.in`` is a ``calculation='nscf'`` restarting from it
    on a grid eight times denser. It is the one case here that exercises the
    fixed-density path with two spin channels, and the quantity it pins is the
    one an NSCF exists to produce: a Fermi level converged with respect to the
    k-point sampling rather than to the density.

    It also pins a trap that only appears when spin meets a *second* k-set. Every
    ``KPoints`` constructor applies the spin degeneracy unconditionally, and
    ``build_system`` divides it out again for ``nspin = 2``; a grid built later
    -- which is exactly what a denser NSCF grid is -- never passed through that
    step and counted every electron twice. The failure is silent: the density of
    states still integrates to ten electrons, at a Fermi level 2.3 eV too low.
    :func:`defumat.system.kpoints.for_spin` is now the single place that knows
    the rule, and both callers go through it.
    """
    from defumat.workflows.nscf import run_nscf

    _, _, scf = _converged("pw_lsda", "lsda.in", qe_testsuite, pseudo_dir)

    dense_input = qe_testsuite / "pw_lsda" / "lsda-2.in"
    system = build_system(read_pw_input(dense_input))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    nscf = run_nscf(system, pseudos, scf.density, conv_thr=1e-10)

    reference = read_qe_output(reference_output("pw_lsda", "lsda-2.in", qe_testsuite))
    assert nscf.nspin == 2
    assert nscf.kpoints.nk == reference.nk
    assert nscf.eigenvalues.shape == (2, reference.nk, reference.nbnd)
    assert nscf.fermi_energy * RY_TO_EV == pytest.approx(
        reference.fermi_energy, abs=FERMI_EV
    )


def test_the_spin_resolved_density_of_states(qe_testsuite, pseudo_dir):
    """Nickel's DOS, one curve per channel, on the same denser grid.

    Two things are checked and they are different in kind. The **sum rule** --
    ``N(E_F) = nelec`` summed over the channels -- is a statement about the
    integration being right, and it holds to the accuracy of the energy grid.
    The **exchange splitting** is a statement about the physics: the majority
    curve is shifted below the minority one, so the two integrate to different
    numbers of electrons at the same Fermi level, and the difference is the
    magnetization. Getting the weight convention wrong (see the test above)
    leaves the first intact and destroys the second, which is why both are here.
    """
    from defumat.workflows.dos import run_dos

    system, pseudos, scf = _converged("pw_lsda", "lsda.in", qe_testsuite, pseudo_dir)
    nelec = sum(pseudos[t].z_valence for t in system.structure.types)

    dos, nscf = run_dos(system, pseudos, scf.density, grid=(8, 8, 8), conv_thr=1e-10)

    assert dos.nspin == 2
    assert dos.dos.shape == (2, len(dos.energies))
    assert dos.states_below(dos.fermi_energy) == pytest.approx(nelec, abs=1e-3)

    # The channels are not the same curve: the majority one is fuller at E_F.
    up, down = (
        float(np.interp(dos.fermi_energy, dos.energies, dos.integrated[spin]))
        for spin in range(2)
    )
    assert up + down == pytest.approx(nelec, abs=1e-3)
    assert up - down > 0.4, "nickel's majority channel must hold more electrons"

    # ... and the total is exactly the two channels added, which is what every
    # unpolarized-shaped consumer reads.
    assert dos.total_dos.shape == dos.energies.shape
    assert dos.total_dos == pytest.approx(dos.dos[0] + dos.dos[1])

    # Not asserted, and worth saying why: this D(E) goes *negative* in places.
    # Marzari-Vanderbilt occupations overshoot 1 before settling -- the same
    # property that makes the Fermi search non-monotonic (see
    # test_two_fermi_levels_match_reference) -- so their derivative, which is
    # what a smeared DOS is, has negative lobes. QE's dos_g does the same at
    # ngauss = -1. A tetrahedron DOS is the one that cannot be negative.


def test_the_dos_file_gets_two_columns_when_polarized(qe_testsuite, pseudo_dir):
    """``dos.f90``'s LSDA format: ``dosup``, ``dosdw``, and one ``Int dos``."""
    from defumat.io.output import format_dos
    from defumat.workflows.dos import compute_dos, energy_grid

    system, pseudos, scf = _converged("pw_lsda", "lsda.in", qe_testsuite, pseudo_dir)
    eigenvalues = scf.eigenvalues_by_spin
    energies = energy_grid(eigenvalues, delta_e=0.02, degauss=system.degauss)
    dos = compute_dos(
        eigenvalues,
        system.kpoints.weights,
        energies,
        system.smearing,
        degauss=system.degauss,
        fermi_energy=scf.fermi_energy,
    )

    lines = format_dos(dos).splitlines()
    assert lines[0].startswith("#  E (eV)   dosup(E)     dosdw(E)   Int dos(E)")
    assert "EFermi" in lines[0]
    # f8.3 plus three e12.4 fields.
    assert len(lines[1]) == 8 + 3 * 12
    assert len(lines) == len(energies) + 1
