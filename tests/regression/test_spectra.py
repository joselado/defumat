"""P36: Raman and infrared spectra, against ``dynmat.x`` and against symmetry.

**This phase has a working reference and P35 did not**, which is the whole
reason it is a separate phase rather than four lines inside that one. QE reaches
the mode-resolved activities through ``dynmat.x``, whose ``RamanIR``
(``LR_Modules/dynmat_sub.f90``) reads ``dchi_dtau``, ``zstar`` and ``eps0`` off
a dynamical-matrix file and contracts them with the eigendisplacements. It
solves nothing and shares nothing with the ``lraman`` branch that ``PLAN.md``
P35 establishes has regressed in the vendored 7.5 build -- so writing that file
from *our* tensors (:func:`~defumat.io.dynmat.write_dynamical_matrix`) and
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

from defumat.io.dynmat import write_dynamical_matrix
from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.response.electrostriction import refined_states
from defumat.response.nonlinear import raman_tensors
from defumat.response.phonon import dynamical_matrix
from defumat.response.spectra import loto_modes, vibrational_spectrum
from defumat.scf import Calculation, run_scf
from defumat.system import build_system

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
    :attr:`~defumat.response.nonlinear.RamanTensors.displacement`.
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


@lru_cache(maxsize=4)
def _loto(case: str, direction, neutralize: bool = False):
    """The spectrum with the long-range field added along ``direction``.

    The expensive halves are handed in, so this costs one ``3 nat x 3 nat``
    diagonalisation on top of :func:`_pieces`.
    """
    _, _, calculation, result = _converged(case)
    _, raman, phonons, _ = _pieces(case)
    return vibrational_spectrum(
        calculation, result, raman=raman, phonons=phonons,
        loto_direction=direction, neutralize=neutralize,
    )


@lru_cache(maxsize=4)
def _neutralized(case: str):
    """The analytic spectrum with ``sum_a Z* = 0`` imposed."""
    _, _, calculation, result = _converged(case)
    _, raman, phonons, _ = _pieces(case)
    return vibrational_spectrum(
        calculation, result, raman=raman, phonons=phonons, neutralize=True,
    )


def _run_dynmat(case: str, tmp_path: Path, direction=None,
                permittivity: bool = False) -> str:
    """Write the dynamical-matrix file and run the vendored ``dynmat.x`` on it.

    ``direction`` is ``dynmat.x``'s ``q``: given one, it adds the non-analytic
    term (``rigid.f90``'s ``nonanal``) before diagonalising, so the frequencies
    it prints are the LO ones. ``permittivity`` turns on ``lplasma``, which
    prints the mode effective charges and the static dielectric tensor and
    implies ``lperm``.
    """
    if not QE_BIN.exists():
        pytest.skip(f"vendored dynmat.x not built at {QE_BIN}")
    system, _, _, _ = _converged(case)
    _, raman, phonons, _ = _pieces(case)
    fildyn = tmp_path / f"{case}.dynG"
    write_dynamical_matrix(
        fildyn, system.cell, system.structure, phonons.matrix,
        epsilon=raman.epsilon, born=np.asarray(raman.field.born_charges),
        raman=raman.raman, title=f"defumat {case}",
    )
    # ``asr = 'no'`` is what ``ph.x`` prints its charges without. ``q`` is left
    # at zero unless a caller asks for a direction: the analytic matrix and the
    # non-analytic one are two different comparisons and both are made below.
    namelist = [
        " &input",
        f"   fildyn = '{fildyn.name}'",
        "   filout = 'dynmat.out'",
        "   asr = 'no'",
    ]
    if direction is not None:
        namelist += [f"   q({i + 1}) = {value}"
                     for i, value in enumerate(direction)]
    if permittivity:
        namelist.append("   lplasma = .true.")
    (tmp_path / "dynmat.in").write_text("\n".join(namelist) + "\n /\n")
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


def _parse_plasma(output: str) -> np.ndarray:
    """The ``lplasma`` table: ``(3 nat, 6)`` of freq, ``Z~*``, ``W_eff``, ``deps``."""
    lines = output.splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.startswith("# mode") and "Z~*_x" in line)
    rows = []
    for line in lines[start + 2:]:
        fields = line.split()
        if len(fields) != 7:
            break
        rows.append([float(value) for value in fields[1:]])
    return np.array(rows)


def _parse_permittivity(output: str) -> np.ndarray:
    """The ``... with zone-center polar mode contributions`` 3x3 block."""
    lines = output.splitlines()
    start = next(i for i, line in enumerate(lines)
                 if "zone-center polar mode contributions" in line)
    return np.array([
        [float(value) for value in lines[start + 1 + i].split()] for i in range(3)
    ])


def test_the_lo_modes_match_dynmat_x(tmp_path):
    """The non-analytic term, against the Fortran that has it: ``nonanal``.

    ``dynmat.x`` adds the long-range field itself, from the same ``Z*`` and
    ``eps`` written to the file, so pointing it at a direction and comparing the
    frequencies checks this code's rank-one term against QE's -- the sign, the
    ``4 pi e^2``, the volume, and which index of ``Z*`` is contracted with
    ``q``. The last of those is the one worth naming: ``Z*`` is not symmetric in
    a general crystal, and contracting the wrong index of it is invisible in
    zincblende, where it is.
    """
    _, raman, phonons, analytic = _pieces("alas-raman-wedge")
    spectrum = _loto("alas-raman-wedge", (1.0, 0.0, 0.0))
    reference = _parse_modes(
        _run_dynmat("alas-raman-wedge", tmp_path, direction=(1.0, 0.0, 0.0))
    )

    for mode in range(reference.shape[0]):
        freq, thz, _, _, _ = reference[mode]
        assert spectrum.frequencies[mode] == pytest.approx(freq, abs=5e-3)
        assert spectrum.frequencies_thz[mode] == pytest.approx(thz, abs=PRINTED)

    # ... and it is a splitting rather than a shift: two of the three optical
    # modes stay where they were and one is raised. AlAs's is 41 cm^-1 in
    # experiment; at ``ecutwfc = 10`` this cell gives tens of cm^-1 and the
    # comparison above is what pins the number.
    optical = np.sort(analytic.frequencies)[-3:]
    split = np.sort(spectrum.frequencies)[-3:]
    assert split[0] == pytest.approx(optical[0], abs=5e-3)
    assert split[1] == pytest.approx(optical[1], abs=5e-3)
    assert split[2] > optical[2] + 20.0


def test_an_asymmetric_born_charge_contracts_on_the_right_index(tmp_path):
    """The index of ``Z*`` that ``q`` is contracted with, on a crystal that shows it.

    ``zag(i) = sum_alpha q_alpha Z*_(alpha i)`` sums over the **field** label,
    and every crystal committed here has a symmetric ``Z*`` -- zincblende's is
    a multiple of the identity -- so transposing it changes nothing. That is
    the shape of the bug P54 found in an occupation factor: a transposed index
    pair passes every physical check there is.

    So the charges are *made up* rather than computed. The dynamical matrix,
    the cell and the masses are AlAs's; ``Z*`` is asymmetric and neutral by
    construction, ``q`` is low-symmetry, and ``dynmat.x`` is asked what it makes
    of the same file. Nothing here is a physical crystal and nothing needs to
    be: what is under test is one contraction.

    **Asymmetric is not enough and the first attempt here was not.** What the
    splitting depends on is ``|Z*^T q|^2`` against ``|Z* q|^2``, which are
    ``q^T Z Z^T q`` and ``q^T Z^T Z q`` -- equal for any **normal** matrix, and
    a circulant one is normal, so a perfectly asymmetric cyclic ``Z*`` gives the
    two contractions bit for bit. The matrix below is not normal.
    """
    if not QE_BIN.exists():
        pytest.skip(f"vendored dynmat.x not built at {QE_BIN}")
    system, _, calculation, _ = _converged("alas-raman-wedge")
    _, raman, phonons, _ = _pieces("alas-raman-wedge")

    charge = np.array([[1.4, 2.8, -0.3], [0.1, 1.9, 2.2], [-0.6, 0.2, 1.1]])
    asymmetric = np.stack([charge, -charge])
    direction = (0.3, -0.7, 0.5)
    epsilon = np.asarray(raman.epsilon)
    volume = float(calculation.system.cell.volume)
    masses = calculation.system.structure.masses

    fildyn = tmp_path / "asymmetric.dynG"
    write_dynamical_matrix(
        fildyn, system.cell, system.structure, phonons.matrix,
        epsilon=epsilon, born=asymmetric, raman=raman.raman,
        title="asymmetric Z*",
    )
    namelist = [" &input", f"   fildyn = '{fildyn.name}'",
                "   filout = 'dynmat.out'", "   asr = 'no'"]
    namelist += [f"   q({i + 1}) = {value}"
                 for i, value in enumerate(direction)]
    (tmp_path / "dynmat.in").write_text("\n".join(namelist) + "\n /\n")
    finished = subprocess.run(
        [str(QE_BIN)], stdin=(tmp_path / "dynmat.in").open(),
        capture_output=True, text=True, cwd=tmp_path, timeout=300,
    )
    assert "JOB DONE" in finished.stdout, finished.stdout[-2000:]

    ours, _ = loto_modes(phonons.matrix, masses, asymmetric, epsilon,
                         direction, volume)
    theirs = _parse_modes(finished.stdout)[:, 0]
    assert np.abs(np.sort(ours) - np.sort(theirs)).max() < 5e-3

    # ... and the transpose is a different answer, which is what makes the
    # comparison above worth making.
    transposed, _ = loto_modes(
        phonons.matrix, masses, np.swapaxes(asymmetric, 1, 2), epsilon,
        direction, volume,
    )
    # 11.6 cm^-1 apart, against the 5e-3 the right contraction agrees to.
    assert np.abs(np.sort(transposed) - np.sort(theirs)).max() > 5.0


def test_the_static_permittivity_matches_dynmat_x(tmp_path):
    """``polar_mode_permittivity``, against ``lplasma``'s two printed blocks.

    The mode effective charges are compared **summed over each multiplet**, for
    the reason the mode table is: ``Z~*`` is a vector attached to one member of
    a degenerate triplet and the eigensolver's basis inside it is arbitrary,
    where ``sum_nu |Z~*_nu|^2`` is invariant. The dielectric tensor is a sum
    over the whole multiplet already and needs no such care.
    """
    spectrum = _pieces("alas-raman-wedge")[3]
    output = _run_dynmat("alas-raman-wedge", tmp_path, permittivity=True)
    reference = _parse_plasma(output)

    for group in range(int(spectrum.manifold.max()) + 1):
        members = spectrum.manifold == group
        ours = np.sum(spectrum.mode_effective_charges[members] ** 2)
        theirs = np.sum(reference[members, 1:4] ** 2)
        assert ours == pytest.approx(theirs, rel=1e-4)

    assert np.abs(
        _parse_permittivity(output) - spectrum.static_permittivity
    ).max() < PRINTED


def test_lyddane_sachs_teller_holds_for_alas():
    """``eps_0/eps_inf = (omega_LO/omega_TO)^2`` on a crystal rather than a model.

    Both sides are computed here and they share only ``Z*`` and ``eps``: one is
    a ratio of two diagonalisations of the dynamical matrix and the other a sum
    of mode dipoles over ``omega^2``. For a diatomic cubic crystal the relation
    is an identity, so what it measures is the two assemblies against each other
    -- and it is the check that a *dielectric* constant is tied to a
    *frequency*, which no comparison against a printed table can see, both codes
    reading the same tensors off the same file.

    **It needs the charge-neutral ``Z*`` and that is the finding.** At
    ``ecutwfc = 10`` this cell's Born charges miss ``sum_a Z*_a = 0`` by -1.257,
    which charges the crystal: :func:`~defumat.response.spectra.nonanal` then
    lifts a **longitudinal acoustic** mode from 1.8 to 33.8 cm^-1 and the LO
    frequency lands 7.7 cm^-1 low. Neither is visible against ``dynmat.x``,
    which is handed the same charges and reproduces the same wrong number. LST
    sees it: 1.6e-3 raw against 5.0e-11 neutralised.
    """
    analytic = _neutralized("alas-raman-wedge")
    longitudinal = _loto("alas-raman-wedge", (1.0, 0.0, 0.0), True)

    ratio = analytic.static_permittivity[0, 0] / analytic.epsilon[0, 0]
    splitting = (
        np.max(longitudinal.frequencies) / np.max(analytic.frequencies)
    ) ** 2
    assert ratio == pytest.approx(splitting, rel=1e-8)

    # ... and the acoustic modes are where they were, which is the half of the
    # statement that the ratio above cannot see.
    raw = _pieces("alas-raman-wedge")[3]
    assert np.abs(np.sort(longitudinal.frequencies)[:3]
                  - np.sort(raw.frequencies)[:3]).max() < 1e-4


def test_the_neutrality_violation_is_what_breaks_it():
    """The same relation with the raw charges, measured rather than avoided.

    Committed as a *number* because it is the size of a basis-set error showing
    up in a place nothing else here looks: the acoustic branch of a polar
    crystal under its own macroscopic field.
    """
    analytic = _pieces("alas-raman-wedge")[3]
    longitudinal = _loto("alas-raman-wedge", (1.0, 0.0, 0.0), False)

    ratio = analytic.static_permittivity[0, 0] / analytic.epsilon[0, 0]
    splitting = (
        np.max(longitudinal.frequencies) / np.max(analytic.frequencies)
    ) ** 2
    assert abs(ratio - splitting) == pytest.approx(1.64e-3, rel=0.05)
    # The spurious longitudinal acoustic mode, which is the mechanism.
    assert np.sort(longitudinal.frequencies)[2] == pytest.approx(33.8, abs=0.5)


def test_silicon_is_not_split_and_has_no_ionic_permittivity():
    """A non-polar crystal: with ``Z* = 0`` both quantities vanish identically.

    Silicon's two atoms are equivalent, so their Born charges are *equal* rather
    than opposite and the whole of what this cell reports (-1.196 each) is the
    neutrality violation -- imposing the sum rule leaves exactly zero, and with
    it no splitting and no ionic screening. The two ways of being zero are
    different, one a rank-one term with a zero factor and the other a mode
    dipole with nothing in it, and neither is imposed by hand.
    """
    analytic = _neutralized("si-epsilon-unshifted")
    longitudinal = _loto("si-epsilon-unshifted", (1.0, 1.0, 1.0), True)

    assert np.abs(longitudinal.frequencies - analytic.frequencies).max() < 1e-9
    assert np.abs(analytic.ionic_permittivity).max() < 1e-12
    # The raw charges are a pure violation and the optical mode is silent
    # either way: equal charges on the two atoms cancel in the mode dipole.
    raw = _pieces("si-epsilon-unshifted")[3]
    assert np.abs(raw.ionic_permittivity).max() < 1e-12


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

    :func:`~defumat.response.nonlinear.raman_tensors` and
    :func:`~defumat.response.phonon.dynamical_matrix` need the *same*
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
