"""P36: Raman and infrared spectra, against ``dynmat.x`` and against symmetry.

**This phase has a working reference and P35 did not**, which is the whole
reason it is a separate phase rather than four lines inside that one. QE reaches
the mode-resolved activities through ``dynmat.x``, whose ``RamanIR``
(``LR_Modules/dynmat_sub.f90``) reads ``dchi_dtau``, ``zstar`` and ``eps0`` off
a dynamical-matrix file and contracts them with the eigendisplacements. It
solves nothing and shares nothing with the ``lraman`` branch that ``PLAN.md``
P35 establishes has regressed in the vendored 7.5 build -- so writing that file
from *our* tensors (:func:`~pypresso.io.dynmat.write_dynamical_matrix`) and
running the vendored binary on it is a genuine transcription check.

The rest are checks that need no Fortran at all, and the sharpest of them is
silicon: one Raman-active triplet and **no infrared activity whatever**, because
in diamond the two atoms carry the same ``Z*`` and the optical mode moves them
against each other. That is a symmetry statement, so it holds at any cutoff and
on any grid, and an assembly with the mode-dipole contraction wrong would not
reproduce it.
"""

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from pypresso.io.dynmat import write_dynamical_matrix
from pypresso.io.pwin import read_pw_input
from pypresso.pseudo import read_upf
from pypresso.response.electrostriction import refined_states
from pypresso.response.nonlinear import raman_tensors
from pypresso.response.phonon import dynamical_matrix
from pypresso.response.spectra import vibrational_spectrum
from pypresso.scf import Calculation, run_scf
from pypresso.system import build_system

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"
QE_BIN = (Path(__file__).resolve().parents[2] / "quantum_espresso"
          / "qe-7.5-ReleasePack" / "qe-7.5" / "bin" / "dynmat.x")

#: ``dynmat.x`` prints frequencies to two decimals and the activities to four,
#: so what the comparison can resolve is the last digit it writes. This is the
#: same convention the ``projwfc.x`` tests use.
PRINTED = 5e-5

#: What the vendored ``ph.x`` prints for AlAs's Born charges on this exact
#: input (``reference.out.ph-alas-raman``), and it is the **first polar** ``Z*``
#: checked here -- every earlier case is silicon or carbon, where ``Z*`` is a
#: residue near zero. Note that the two do **not** sum to zero: at
#: ``ecutwfc = 10`` charge neutrality is violated by -1.256, and ``ph.x`` says
#: so too and prints an ASR-corrected +-2.5528 beside it. Reproducing the
#: *uncorrected* pair is the check; reproducing the violation is part of it.
QE_ALAS_BORN = {"Al": 1.92461, "As": -3.18098}
BORN_TOLERANCE = 3e-4


@lru_cache(maxsize=None)
def _converged(case: str):
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                     max_iterations=100)
    return system, pseudos, calculation, result


@lru_cache(maxsize=None)
def _pieces(case: str):
    """The Raman tensors, the phonons and the spectrum, solved once per case.

    The displacement response is threaded from the Raman tensors into the
    dynamical matrix, which is what makes this one solve rather than two --
    :attr:`~pypresso.response.nonlinear.RamanTensors.displacement`.
    """
    _, _, calculation, result = _converged(case)
    raman = raman_tensors(
        calculation, result, born_charges=True, keep_internals=True
    )
    eigenvalues, psi = refined_states(calculation, result)
    phonons = dynamical_matrix(
        calculation, psi, eigenvalues, jnp.asarray(result.density),
        response=raman.displacement,
    )
    spectrum = vibrational_spectrum(
        calculation, result, raman=raman, phonons=phonons
    )
    return calculation, raman, phonons, spectrum


def _run_dynmat(case: str, tmp_path: Path) -> str:
    """Write the dynamical-matrix file and run the vendored ``dynmat.x`` on it."""
    if not QE_BIN.exists():
        pytest.skip(f"vendored dynmat.x not built at {QE_BIN}")
    system, _, _, _ = _converged(case)
    _, raman, phonons, _ = _pieces(case)
    fildyn = tmp_path / f"{case}.dynG"
    write_dynamical_matrix(
        fildyn, system.cell, system.structure, phonons.matrix,
        epsilon=raman.epsilon, born=np.asarray(raman.field.born_charges),
        raman=raman.raman, title=f"pypresso {case}",
    )
    # ``asr = 'no'`` and ``q`` left at zero on purpose. The first is what
    # ``ph.x`` prints its charges without, and the second keeps the
    # **non-analytic** LO-TO term out (``rigid.f90``'s ``nonanal``): the
    # ``Gamma`` matrix P25 computes is the analytic one, so a comparison that
    # let ``dynmat.x`` add a term this code does not have would be measuring
    # the term rather than the assembly.
    (tmp_path / "dynmat.in").write_text(
        f" &input\n   fildyn = '{fildyn.name}'\n   filout = 'dynmat.out'\n"
        f"   asr = 'no'\n /\n"
    )
    finished = subprocess.run(
        [str(QE_BIN)], stdin=(tmp_path / "dynmat.in").open(),
        capture_output=True, text=True, cwd=tmp_path, timeout=300,
    )
    assert "JOB DONE" in finished.stdout, finished.stdout[-2000:]
    return finished.stdout


def _parse_modes(output: str) -> np.ndarray:
    """The ``# mode`` table ``RamanIR`` prints, as ``(3 nat, 5)``."""
    lines = output.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("# mode"))
    rows = []
    for line in lines[start + 1:]:
        fields = line.split()
        if len(fields) != 6:
            break
        rows.append([float(value) for value in fields[1:]])
    return np.array(rows)


# -- the transcription check --------------------------------------------------


@pytest.mark.parametrize("case", ["alas-raman-wedge", "si-epsilon-unshifted"])
def test_the_mode_table_matches_dynmat_x(case, tmp_path):
    """``RamanIR``'s table against this code's own assembly, column by column.

    The two share the tensors and share nothing else: ``dynmat.x`` re-reads them
    from a text file, re-diagonalises the dynamical matrix with its own
    ``cdiagh2``, and contracts them in Fortran. Frequencies, infrared activities
    and Raman activities all come out the same to every digit either code
    prints.

    **The activities are compared per degenerate multiplet and not per mode**,
    which is the module's own rule and is not a hedge -- running this test is
    what demonstrated it. The two eigensolvers leave *different bases* inside
    silicon's acoustic triplet, and the per-mode depolarisation ratios there are
    0.3544/0.7163/0.4065 here against 0.5873/0.2446/0.7264 from ``dynmat.x``,
    on modes whose Raman activity both codes print as **0.0000**. Neither set of
    numbers means anything: the ratio is ``0/0`` amplified out of a sum-rule
    residue, in a manifold whose basis is arbitrary. The quantities that survive
    the mixing -- the frequencies, and the sums of both activities over the
    multiplet -- agree exactly.
    """
    _, _, _, spectrum = _pieces(case)
    reference = _parse_modes(_run_dynmat(case, tmp_path))
    assert reference.shape[0] == spectrum.frequencies.size

    for mode in range(reference.shape[0]):
        freq, thz, _, _, _ = reference[mode]
        assert spectrum.frequencies[mode] == pytest.approx(freq, abs=5e-3)
        assert spectrum.frequencies_thz[mode] == pytest.approx(thz, abs=PRINTED)

    for group in range(int(spectrum.manifold.max()) + 1):
        members = spectrum.manifold == group
        assert spectrum.infrared[members].sum() == pytest.approx(
            reference[members, 2].sum(), abs=PRINTED * members.sum()
        )
        assert spectrum.raman_activity[members].sum() == pytest.approx(
            reference[members, 3].sum(), abs=5e-4 * members.sum()
        )

    # The depolarisation ratio only where there is an intensity to depolarise.
    # On both crystals that is the optical triplet, where it is 3/4 and where
    # both codes agree to the digit -- because zincblende's one allowed
    # component is off-diagonal, so ``alpha`` vanishes for every member however
    # the multiplet is rotated.
    active = spectrum.raman_activity > 1e-3 * spectrum.raman_activity.max()
    assert active.sum() == 3
    assert spectrum.depolarisation[active] == pytest.approx(
        reference[active, 4], abs=PRINTED
    )


def test_the_polarizability_block_matches_dynmat_x(tmp_path):
    """The table above the modes: ``Omega chi/(4 pi)`` in ``A^3``, and ``cmfac``.

    Small, and the reason it is here is that it is the only part of the output
    that checks the *volume and unit* conversion independently of the mode
    projection -- everything in the mode table carries the eigendisplacements
    as well.
    """
    output = _run_dynmat("alas-raman-wedge", tmp_path)
    _, _, _, spectrum = _pieces("alas-raman-wedge")

    lines = output.splitlines()
    start = next(i for i, line in enumerate(lines) if "Polarizability" in line)
    factor = float(lines[start + 1].split("multiply by")[1].split("for")[0])
    rows = np.array([
        [float(value) for value in lines[start + 2 + i].split()] for i in range(3)
    ])
    assert np.abs(rows - spectrum.polarizability).max() < 1e-5
    assert spectrum.clausius_mossotti == pytest.approx(factor, abs=1e-6)


# -- what the crystals say ----------------------------------------------------


def test_silicon_has_one_raman_active_triplet_and_no_infrared_activity():
    """Diamond's ``T_2g`` at ~520 cm^-1, and the silence that goes with it.

    Two independent statements and neither depends on how converged the run is:

    * the optical triplet is **Raman active**, and its depolarisation ratio is
      exactly 3/4 because the only component ``-43m`` allows is off-diagonal, so
      the isotropic invariant vanishes identically;
    * it is **infrared silent**, because an operation of the group carries one
      silicon onto the other and therefore gives them the same ``Z*``, so the
      antiphase motion of the optical mode has no dipole at all. Silicon is
      transparent in the infrared for this reason.

    The measured frequency here is **519.2 cm^-1** against an experimental 520,
    which is not the check -- the check is the structure of the table.
    """
    _, _, _, spectrum = _pieces("si-epsilon-unshifted")
    optical = slice(3, 6)

    assert np.allclose(spectrum.frequencies[optical], 519.2, atol=0.1)
    assert spectrum.raman_activity[optical].min() > 1e3
    assert np.allclose(spectrum.depolarisation[optical], 0.75, atol=1e-6)
    assert np.abs(spectrum.infrared[optical]).max() < 1e-8
    # The acoustic modes are Raman silent by the translational sum rule, to the
    # residue P35 reports rather than exactly.
    assert spectrum.raman_activity[:3].max() < 1e-3 * spectrum.raman_activity.max()


def test_alas_is_active_in_both_channels():
    """A polar crystal, where silicon's infrared silence is the special case."""
    _, _, _, spectrum = _pieces("alas-raman-wedge")
    optical = slice(3, 6)
    assert np.allclose(spectrum.frequencies[optical], 353.25, atol=0.1)
    assert spectrum.raman_activity[optical].min() > 1.0
    assert spectrum.infrared[optical].min() > 1.0


def test_the_born_charges_of_alas_match_ph_x():
    """The first **polar** ``Z*`` checked here, and it is not a small residue.

    Every other Born-charge case in this suite is silicon or carbon, where the
    answer is a cancellation down to ~0.05 and agreeing is a statement about the
    machinery's precision. AlAs's are 1.92 and -3.18, so agreeing is a statement
    about the machinery being right.

    **They do not sum to zero and that is not a failure of this code**: at
    ``ecutwfc = 10`` the vendored ``ph.x`` gets the same violation, -1.256, on
    the same input, and prints an ASR-corrected pair beside it. Charge
    neutrality is exact only in a complete basis.
    """
    _, raman, _, _ = _pieces("alas-raman-wedge")
    charges = np.asarray(raman.field.born_charges)
    for atom, expected in enumerate(QE_ALAS_BORN.values()):
        assert charges[atom, 0, 0] == pytest.approx(expected, abs=BORN_TOLERANCE)
        # Cubic, and nothing imposes it.
        assert np.abs(charges[atom] - np.eye(3) * charges[atom, 0, 0]).max() < 1e-8
    assert charges.sum(axis=0)[0, 0] == pytest.approx(-1.25637, abs=BORN_TOLERANCE)


# -- the machinery ------------------------------------------------------------


def test_reusing_the_displacement_response_gives_the_same_dynamical_matrix():
    """The threading is an optimisation, so it has to change nothing.

    :func:`~pypresso.response.nonlinear.raman_tensors` and
    :func:`~pypresso.response.phonon.dynamical_matrix` need the *same*
    ``solve_linter`` output, and solving it twice is the dominant cost of a
    spectrum. This is the check that handing it across is exact rather than
    nearly so -- the matrix is compared entry by entry against one built from a
    solve of its own.
    """
    case = "si-epsilon-unshifted"
    _, _, pieces_phonons, _ = _pieces(case)
    _, _, calculation, result = _converged(case)
    eigenvalues, psi = refined_states(calculation, result)
    fresh = dynamical_matrix(
        calculation, psi, eigenvalues, jnp.asarray(result.density)
    )
    assert np.abs(fresh.matrix - pieces_phonons.matrix).max() < 1e-12
    assert np.abs(fresh.frequencies - pieces_phonons.frequencies).max() < 1e-8


def test_a_degenerate_multiplet_is_reported_as_one():
    """``by_manifold`` groups the triplets, which is the comparable form."""
    _, _, _, spectrum = _pieces("si-epsilon-unshifted")
    grouped = spectrum.by_manifold()
    assert len(grouped) == 2
    assert list(spectrum.manifold) == [0, 0, 0, 1, 1, 1]
    assert grouped[1][1] == pytest.approx(spectrum.raman_activity[3:].sum())
