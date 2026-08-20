"""P14: noncollinear spinors and spin-orbit coupling, against QE.

Two kinds of check, and they test different things.

**The identity.** A noncollinear calculation with *no* spin-orbit coupling and no
magnetization is the same physics as an unpolarized collinear one, done in a
space twice as large. So it must reproduce the collinear total energy term by
term to machine precision, and every eigenvalue must come out exactly twice.
Nothing about QE is involved -- the comparison is against this code's own
collinear answer, which is already validated -- so the check is sharp: it fails
on any error in the spinor Hamiltonian, the doubled eigensolver, the spinor
``sum_band``, the projector occupations or the k-point weights, and it is
insensitive to none of them. It runs on all three pseudopotential kinds.

**The physics.** With ``lspinorb`` the ``j``-resolved projectors enter and the
answer changes, so there is nothing to compare against but QE. Platinum is the
case QE's own test suite uses: one heavy atom, fcc, where the spin-orbit
splitting of the 5d states is tenths of an eV. The references are regenerated
with the vendored pw.x 7.5 (``tools/generate_reference.py``), because the
committed benchmarks are 2017 runs stopped at ``conv_thr = 1e-8``.

Fcc platinum has inversion symmetry, so Kramers degeneracy survives the
spin-orbit coupling and every level stays doubly degenerate. That is asserted
too: it is a property of the *symmetry* rather than of the numbers, so it holds
to machine precision or not at all.
"""

from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from pypresso.io import read_qe_output
from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.scf import run_scf
from pypresso.system import build_system
from tests.conftest import GENERATED
from tests.tolerances import (
    EIGENVALUE_EV,
    ENERGY_TERM_RY,
    FERMI_EV,
    TOTAL_ENERGY_RY,
)

pytestmark = [pytest.mark.regression, pytest.mark.slow]

RY_TO_EV = 13.605693122994

#: Energy *terms* for a spin-orbit run. Looser than the total by five orders of
#: magnitude for the reason ``USPP_TERM_RY`` documents -- the terms are
#: first-order sensitive to where each mixer stops and the total is not -- and
#: looser than ``USPP_TERM_RY`` itself because these cells are metals with
#: smearing, where the density at the fixed point is decided by a Fermi level
#: that both codes locate independently.
SPINORBIT_TERM_RY = 1e-4

#: Collinear inputs to re-run with ``noncolin = .true.`` added and nothing else
#: changed: norm-conserving, ultrasoft and PAW silicon. Smearing is added along
#: with it because a noncollinear band holds one electron rather than two, so
#: ``occupations = 'fixed'`` would be filling a different number of bands in the
#: two runs and the comparison would not be like for like.
DOUBLING_CASES = [
    ("norm-conserving", "pw_scf/scf.in"),
    ("ultrasoft", "si2-us.in"),
    ("paw", "si2-paw.in"),
]

#: The spin-orbit cases: (input, regenerated reference stem).
SPINORBIT_CASES = [
    ("spinorbit.in", "pw_spinorbit-spinorbit"),
    ("spinorbit-pbe.in", "pw_spinorbit-spinorbit-pbe"),
    ("spinorbit-paw.in", "pw_spinorbit-spinorbit-paw"),
]


def _input_text(name: str, qe_testsuite: Path) -> str:
    if "/" in name:
        return (qe_testsuite / name).read_text()
    return (GENERATED / name).read_text()


def _with_namelist_lines(text: str, lines: str) -> str:
    """Insert extra ``&system`` variables into a pw.x input, in place."""
    marker = text.lower().index("&system") + len("&system")
    return text[:marker] + "\n" + lines + text[marker:]


@lru_cache(maxsize=None)
def _run(path: Path, pseudo_dir: Path):
    system = build_system(read_pw_input(path))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    return system, run_scf(system, pseudos, conv_thr=1e-10, max_iterations=100)


@pytest.mark.parametrize(("kind", "name"), DOUBLING_CASES)
def test_spinors_reproduce_the_collinear_answer(
    kind, name, qe_testsuite, pseudo_dir, tmp_path_factory
):
    """``noncolin`` with no magnetization and no spin-orbit changes nothing.

    The same input twice, differing by one flag. The total energy must agree to
    round-off -- not to a tolerance borrowed from a QE comparison, because both
    numbers come from this code and any difference between them is a bug.
    """
    text = _input_text(name, qe_testsuite)
    smeared = _with_namelist_lines(
        text, "    occupations='smearing', smearing='gaussian', degauss=0.02,"
    )
    directory = tmp_path_factory.mktemp(f"doubling-{kind}")
    collinear = directory / "collinear.in"
    collinear.write_text(smeared)
    noncollinear = directory / "noncollinear.in"
    noncollinear.write_text(_with_namelist_lines(smeared, "    noncolin = .true.,"))

    reference_system, reference = _run(collinear, pseudo_dir)
    system, result = _run(noncollinear, pseudo_dir)

    assert system.nspin == 4 and system.npol == 2
    # No starting magnetization, so there is nothing for the density to carry:
    # one component, exactly as an unpolarized run has.
    assert system.nspin_mag == 1
    assert reference.converged and result.converged

    assert result.total_energy == pytest.approx(reference.total_energy, abs=1e-9)
    for term, value in reference.energy_terms.items():
        assert result.energy_terms[term] == pytest.approx(value, abs=1e-9), term

    # Twice as many bands, in degenerate pairs, matching the collinear ones.
    doubled = np.repeat(reference.eigenvalues, 2, axis=1)
    assert result.eigenvalues.shape[1] >= doubled.shape[1]
    assert np.abs(result.eigenvalues[:, : doubled.shape[1]] - doubled).max() < 1e-10


@pytest.mark.parametrize(("name", "stem"), SPINORBIT_CASES)
def test_spin_orbit_total_energy(name, stem, qe_testsuite, pseudo_dir):
    reference_path = GENERATED / f"reference.out.{stem}"
    if not reference_path.is_file():
        pytest.skip(f"{reference_path.name} not generated")
    reference = read_qe_output(reference_path)
    system, result = _run(qe_testsuite / "pw_spinorbit" / name, pseudo_dir)

    assert system.lspinorb and system.nspin == 4
    assert result.converged
    assert result.total_energy == pytest.approx(reference.total_energy, abs=TOTAL_ENERGY_RY)

    for term, value in reference.energy_terms.items():
        if term not in result.energy_terms:
            continue
        # Ewald depends on the geometry alone, so it must match exactly; the
        # rest are density-dependent.
        tolerance = ENERGY_TERM_RY if term == "ewald" else SPINORBIT_TERM_RY
        assert result.energy_terms[term] == pytest.approx(value, abs=tolerance), term


@pytest.mark.parametrize(("name", "stem"), SPINORBIT_CASES)
def test_spin_orbit_eigenvalues(name, stem, qe_testsuite, pseudo_dir):
    reference_path = GENERATED / f"reference.out.{stem}"
    if not reference_path.is_file():
        pytest.skip(f"{reference_path.name} not generated")
    reference = read_qe_output(reference_path)
    _, result = _run(qe_testsuite / "pw_spinorbit" / name, pseudo_dir)

    expected = np.squeeze(np.asarray(reference.eigenvalues))
    got = np.asarray(result.eigenvalues) * RY_TO_EV
    assert got.shape == expected.shape
    assert np.abs(got - expected).max() < EIGENVALUE_EV
    assert result.fermi_energy * RY_TO_EV == pytest.approx(
        reference.fermi_energy, abs=FERMI_EV
    )


@pytest.mark.parametrize(("name", "stem"), SPINORBIT_CASES)
def test_kramers_degeneracy_survives_spin_orbit(name, stem, qe_testsuite, pseudo_dir):
    """Every level of fcc platinum stays doubly degenerate.

    Time reversal alone pairs ``|k, up>`` with ``|-k, down>``; it takes
    inversion as well to pair two states at the *same* k, and fcc has it. The
    bands therefore stay doubly degenerate however strong the spin-orbit
    coupling is, and the splitting this measures is round-off rather than
    physics. It is worth asserting because the opposite failure -- a spurious
    splitting from a non-Hermitian ``D`` or a mispaired spin block -- looks like
    a physical result and is exactly what a spin-orbit implementation gets
    wrong.
    """
    _, result = _run(qe_testsuite / "pw_spinorbit" / name, pseudo_dir)
    levels = np.asarray(result.eigenvalues)
    splitting = np.abs(levels[:, 0::2] - levels[:, 1::2]).max() * RY_TO_EV
    assert splitting < 1e-6


def test_spin_orbit_lifts_the_degeneracy_a_scalar_run_keeps(qe_testsuite, pseudo_dir):
    """The 5d manifold splits into ``j = 3/2`` and ``j = 5/2``.

    Without spin-orbit coupling the five d levels at a general k-point are
    degenerate only by accident; what is *not* accidental is that with it the
    manifold acquires a splitting of order tenths of an eV. This is the
    "something actually happened" check that sits under the numerical ones --
    it would pass trivially if ``dvan_so`` were spin-diagonal, so it is paired
    with the doubling test above, which would then fail.
    """
    _, result = _run(qe_testsuite / "pw_spinorbit" / "spinorbit.in", pseudo_dir)
    # The six lowest Kramers pairs at the first k-point are the 5d manifold.
    levels = np.asarray(result.eigenvalues)[0, 0:12:2] * RY_TO_EV
    spread = levels.max() - levels.min()
    assert spread > 1.0, f"the 5d manifold spans only {spread:.3f} eV"


def test_spin_orbit_without_noncolin_is_refused(qe_testsuite, tmp_path):
    text = (qe_testsuite / "pw_spinorbit" / "spinorbit.in").read_text()
    text = text.replace("noncolin=.true.,", "")
    path = tmp_path / "bad.in"
    path.write_text(text)
    with pytest.raises(ValueError, match="lspinorb"):
        build_system(read_pw_input(path))


def test_a_relativistic_dataset_without_lspinorb_is_refused(qe_testsuite, pseudo_dir, tmp_path):
    """QE would j-average the projectors (``average_pp``); this refuses instead."""
    from pypresso.scf.driver import Calculation

    text = (qe_testsuite / "pw_spinorbit" / "spinorbit.in").read_text()
    text = text.replace("lspinorb=.true.,", "")
    path = tmp_path / "averaged.in"
    path.write_text(text)
    system = build_system(read_pw_input(path))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    with pytest.raises(NotImplementedError, match="average_pp"):
        Calculation(system, pseudos)


# --- bismuthene ---------------------------------------------------------------
#
# The system the feature exists for. A flat honeycomb layer of bismuth is a
# near-semimetal: without spin-orbit coupling its frontier bands come within a
# tenth of an eV of each other along M-K. Bismuth's spin-orbit coupling is
# enormous, and it opens a gap of half an electronvolt there -- which is what
# makes bismuthene a two-dimensional topological insulator. So the quantity to
# check is the gap, and the fact that it is *made* of the coupling.
#
# These are the test-sized cells (20 Ry, 6x6x1). The converged ones -- 35 Ry,
# 12x12x1 -- are committed beside them with their own references, and are what
# notebook 08 and PLAN.md quote; they are far too slow for a test.
#
# **On the total-energy tolerance.** These two agree with QE to ~3e-5 Ry rather
# than the ~1e-8 every other case here reaches, and the cause is measured rather
# than assumed: the *same* cell run under LDA (``bismuthene-soc-small-lda``,
# same dataset, same grids, same k-points, same spinor path, ``input_dft =
# 'PZ'``) agrees to 7e-9 Ry -- see the test at the end of this file. What is
# left is the
# gradient correction evaluated over the two thirds of this cell that are
# vacuum, where XClib's thresholds decide whether a point contributes at all.
# It is a property of P13 and of the geometry, not of spin-orbit coupling: the
# collinear ``nosoc`` run shows the identical offset with the identical sign
# pattern across the terms, and the *difference* between the two runs -- which is
# the physical claim -- agrees with QE to 1.6e-6 Ry. Every PBE case validated
# before this one was a dense bulk crystal, which is why no earlier test could
# reach it.

#: ``(tag, occupied bands)``. A spinor band holds one electron, a spin-degenerate
#: one holds two, and bismuthene has 30 valence electrons in the cell.
BISMUTHENE = [("soc", 30), ("nosoc", 15)]

#: See the note above. Not a claim about this code's accuracy in general.
VACUUM_GGA_TOTAL_RY = 1e-4

#: How many of the topmost bands neither code converges, per case. Measured
#: rather than guessed -- the per-band disagreement with QE over the whole path
#: is at most 0.46 meV below these and jumps by a factor of thirty inside them:
#: 10.7 meV for the top Kramers pair of the ``soc`` run, and 14.6 / 15.3 / 16.2
#: meV for the top three of ``nosoc``, which has no Kramers doubling to make the
#: unconverged states come in twos.
UNCONVERGED_TOP_BANDS = {"soc": 2, "nosoc": 3}


@lru_cache(maxsize=None)
def _bismuthene(tag: str, pseudo_dir: Path):
    scf_system = build_system(read_pw_input(GENERATED / f"bismuthene-{tag}-small.in"))
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in scf_system.structure.species
    )
    scf = run_scf(scf_system, pseudos, conv_thr=1e-10, max_iterations=100)

    from pypresso.workflows import run_bands

    path_system = build_system(read_pw_input(GENERATED / f"bismuthene-{tag}-small-bands.in"))
    bands = run_bands(
        path_system, pseudos, scf.density, nbnd=path_system.nbnd,
        fermi_energy=scf.fermi_energy,
    )
    return scf_system, scf, path_system, bands


def _gap(tag: str, occupied: int, pseudo_dir: Path) -> float:
    """The fundamental gap in eV: lowest conduction minus highest valence."""
    _, _, _, bands = _bismuthene(tag, pseudo_dir)
    levels = bands.eigenvalues_ev
    return float(levels[:, occupied].min() - levels[:, occupied - 1].max())


def _reference_gap(tag: str, occupied: int) -> float:
    reference = read_qe_output(GENERATED / f"reference.out.bismuthene-{tag}-small-bands")
    levels = np.squeeze(np.asarray(reference.eigenvalues))
    return float(levels[:, occupied].min() - levels[:, occupied - 1].max())


def _skip_without_references():
    for tag in ("soc", "nosoc"):
        for suffix in ("", "-bands"):
            if not (GENERATED / f"reference.out.bismuthene-{tag}-small{suffix}").is_file():
                pytest.skip("bismuthene references not generated")


@pytest.mark.parametrize(("tag", "occupied"), BISMUTHENE)
def test_bismuthene_total_energy(tag, occupied, pseudo_dir):
    _skip_without_references()
    reference = read_qe_output(GENERATED / f"reference.out.bismuthene-{tag}-small")
    system, scf, _, _ = _bismuthene(tag, pseudo_dir)

    assert scf.converged
    assert (system.nspin == 4) == (tag == "soc")
    assert scf.total_energy == pytest.approx(reference.total_energy, abs=VACUUM_GGA_TOTAL_RY)
    # Geometry alone, so this one has no excuse.
    assert scf.energy_terms["ewald"] == pytest.approx(
        reference.energy_terms["ewald"], abs=ENERGY_TERM_RY
    )


def test_the_energy_spin_orbit_costs_matches_reference(pseudo_dir):
    """The *difference* the coupling makes, where the vacuum offset cancels.

    Both runs carry the same systematic error from the gradient correction in
    the vacuum (see the note above), and it is the same to five figures, so the
    difference between them is a far sharper comparison than either total.
    """
    _skip_without_references()
    mine, theirs = [], []
    for tag, _ in BISMUTHENE:
        _, scf, _, _ = _bismuthene(tag, pseudo_dir)
        mine.append(scf.total_energy)
        theirs.append(read_qe_output(GENERATED / f"reference.out.bismuthene-{tag}-small").total_energy)
    assert (mine[0] - mine[1]) == pytest.approx(theirs[0] - theirs[1], abs=1e-5)


@pytest.mark.parametrize(("tag", "occupied"), BISMUTHENE)
def test_bismuthene_bands_match_reference(tag, occupied, pseudo_dir):
    """...and the ``crystal_b`` path expands to exactly QE's k-list.

    Checked explicitly rather than assumed: this is the first band path in the
    suite given in crystal coordinates, and a path differing from QE's by a
    reciprocal-lattice vector would give band values that look plausible
    everywhere while being compared at the wrong k.

    The **topmost** bands are excluded -- see ``UNCONVERGED_TOP_BANDS``. They
    are the last states an iterative solver converges: they sit at the edge of
    the subspace in both codes, with nothing above them to mix with, and neither
    code holds them to the threshold the occupied states meet. Comparing them
    would measure the two eigensolvers' stopping points rather than the physics.
    Everything below agrees to 0.46 meV.
    """
    _skip_without_references()
    reference = read_qe_output(GENERATED / f"reference.out.bismuthene-{tag}-small-bands")
    _, _, path_system, bands = _bismuthene(tag, pseudo_dir)

    mine = np.asarray(path_system.kpoints.cartesian(path_system.cell))
    mine = mine * path_system.cell.alat / (2.0 * np.pi)  # QE prints 2pi/alat
    assert np.abs(mine - np.asarray(reference.kpoints)).max() < 1e-5

    got = bands.eigenvalues_ev
    want = np.squeeze(np.asarray(reference.eigenvalues))
    n = min(got.shape[1], want.shape[1]) - UNCONVERGED_TOP_BANDS[tag]
    assert np.abs(got[:, :n] - want[:, :n]).max() < EIGENVALUE_EV


@pytest.mark.parametrize(("tag", "occupied"), BISMUTHENE)
def test_bismuthene_gap_matches_reference(tag, occupied, pseudo_dir):
    _skip_without_references()
    assert _gap(tag, occupied, pseudo_dir) == pytest.approx(
        _reference_gap(tag, occupied), abs=EIGENVALUE_EV
    )


def test_spin_orbit_opens_the_gap(pseudo_dir):
    """The whole point: the gap is made of the spin-orbit coupling.

    Both runs use the same cell, cutoffs, k-points and functional; they differ
    only in whether the pseudopotential kept its ``j`` channels apart. So the
    difference between the two gaps is the spin-orbit coupling and nothing else.
    """
    _skip_without_references()
    gaps = {tag: _gap(tag, occupied, pseudo_dir) for tag, occupied in BISMUTHENE}
    assert gaps["nosoc"] < 0.2, f"without spin-orbit the gap is already {gaps['nosoc']:.3f} eV"
    assert gaps["soc"] > 0.4, f"spin-orbit opened only {gaps['soc']:.3f} eV"
    assert gaps["soc"] > 3.0 * gaps["nosoc"]


def test_kramers_degeneracy_on_the_bismuthene_path(pseudo_dir):
    """Planar bismuthene has inversion, so the spin-orbit bands stay paired.

    Buckle the layer or put it on a substrate and inversion goes, and the same
    coupling splits them -- that is the Rashba effect. Here it must not.
    """
    _skip_without_references()
    _, _, _, bands = _bismuthene("soc", pseudo_dir)
    levels = bands.eigenvalues_ev
    assert np.abs(levels[:, 0::2] - levels[:, 1::2]).max() < 1e-6


def test_the_same_cell_under_lda_has_no_such_offset(pseudo_dir):
    """The control that localises bismuthene's 3e-5 Ry to the functional.

    Everything is held fixed except the gradient correction: the same
    fully-relativistic dataset, the same 20 Ry / dual 8 grids, the same 6x6x1
    k-grid, the same noncollinear spinor path with ``lspinorb``. Only
    ``input_dft = 'PZ'`` differs, and the agreement with QE improves by four
    orders of magnitude. That is what makes the offset a property of P13's
    thresholds over this cell's vacuum rather than of P14.

    Running a PBE dataset under LDA is not a physical calculation, and both
    codes are asked for the same unphysical thing -- which is the point: the
    comparison is code against code, not either against nature.
    """
    reference_path = GENERATED / "reference.out.bismuthene-soc-small-lda"
    if not reference_path.is_file():
        pytest.skip("bismuthene LDA control reference not generated")
    system = build_system(read_pw_input(GENERATED / "bismuthene-soc-small-lda.in"))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    scf = run_scf(system, pseudos, conv_thr=1e-10, max_iterations=100)

    reference = read_qe_output(reference_path)
    assert scf.converged
    assert scf.total_energy == pytest.approx(reference.total_energy, abs=TOTAL_ENERGY_RY)
    # ...and it is four orders better than the PBE pair above, not merely inside
    # a loose tolerance.
    assert abs(scf.total_energy - reference.total_energy) < 0.01 * VACUUM_GGA_TOTAL_RY
