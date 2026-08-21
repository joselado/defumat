"""P8 (projwfc) check: the projected DOS against ``projwfc.x``, case by case.

QE's test-suite has **no** ``projwfc`` case -- no input, no committed
``filpdos`` file anywhere in the tree -- so the references here were generated
once with the vendored Fortran (``make pp``; ``tools/generate_reference.py``,
its ``PROJWFC`` table) and committed beside the inputs. Everything asserted
below reads one of those files; nothing was transcribed by hand.

Three independent things are compared, in increasing distance from the raw
projection:

1. **The projections themselves**, band by band and k-point by k-point, against
   ``print_proj``'s listing. That listing is rounded to three decimals and drops
   anything below 0.001, so it pins the projection to 1e-3 and no further --
   which is still the sharpest check here, because it involves no integration.
2. **The Löwdin charges** and the spilling parameter: the same projections
   integrated against the occupations. ``print_lowdin`` prints ``f8.4``.
3. **The projected density of states** on ``partialdos``'s own energy grid.
   ``filpdos`` is written in ``e11.3`` -- three significant digits -- so the
   comparison there is relative rather than absolute, and it is additionally
   limited by the eigenvalues: the curves are deltas centred on levels the two
   codes agree on only to ~2e-4 eV, and a 0.05 eV Gaussian has a slope of
   200 states/eV^2 at its steepest.

The cases cover every path through the projection: norm-conserving silicon with
fixed occupations, the same cell ultrasoft and PAW (where ``S`` is not the
identity, which is the trap the projection shares with DFT+U), spin-polarized
nickel with a smearing, and aluminium with both tetrahedron families.
"""

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from pypresso.io import read_pdos_file, read_projwfc_output
from pypresso.io.pwin import read_pw_input
from pypresso.projwfc.channels import L_LABELS, M_LABELS
from pypresso.projwfc.projections import atomic_projections
from pypresso.pseudo import read_upf
from pypresso.scf import run_scf
from pypresso.scf.driver import Calculation
from pypresso.system import build_system
from pypresso.units import RY_TO_EV
from pypresso.workflows.pdos import run_pdos
from tests.tolerances import LOWDIN_CHARGE, PDOS_RELATIVE, PROJECTION

pytestmark = [pytest.mark.regression, pytest.mark.slow]

GENERATED = Path(__file__).resolve().parents[1] / "data" / "qe"

#: ``stem -> input``, the same table ``tools/generate_reference.py`` generated
#: the references from: a path under ``test-suite/``, or a bare stem for one of
#: the dedicated inputs in ``tests/data/qe``.
CASES = {
    "pw_scf-scf": "pw_scf/scf.in",
    "si2-us": "si2-us",
    "si2-us-dense": "si2-us-dense",
    "si2-paw": "si2-paw",
    "pw_lsda-lsda": "pw_lsda/lsda.in",
    "pw_metal-metal-tetrahedra": "pw_metal/metal-tetrahedra.in",
    "al-tetrahedra": "al-tetrahedra",
}

#: ``projwfc.x``'s ``DeltaE`` for those runs, in eV.
DELTA_E = 0.05

#: Cases whose reference asked ``projwfc.x`` for a broadening of its own, in Ry
#: (``degauss``'s unit there, unlike ``DeltaE``'s). Kept in step with
#: ``tools/generate_reference.py``'s ``PROJWFC_DEGAUSS``: with 29 k-points and no
#: broadening beyond the energy step, a density of states is a comb of spikes.
DEGAUSS = {"si2-us-dense": 0.0147}



def _converged_top(states) -> float:
    """The energy above which the two codes' bands are not the same bands, in eV.

    **The highest band a Davidson run computes is the least converged one**, in
    both codes and for the same reason: ``cegterg`` and this solver both stop on
    the accuracy of the states the density needs, and an empty band at the top of
    the window is carried along rather than converged. On ``si2-us-dense`` the
    eighth band at one k-point differs by 0.19 eV between the two codes where the
    lower seven agree everywhere to 1e-4 -- so a comparison that includes it is
    measuring where each eigensolver happened to stop.

    Everything below the *lowest* energy of that top band is untouched by it,
    which is where the comparisons are made.
    """
    bands = np.atleast_3d(states.eigenvalues_by_spin)
    return float(np.min(bands[..., -1])) * RY_TO_EV


def _input_path(source: str, testsuite: Path) -> Path:
    return testsuite / source if "/" in source else GENERATED / f"{source}.in"


@lru_cache(maxsize=None)
def _run(stem: str, testsuite: Path, pseudo_dir: Path):
    """SCF, then project its own states, on the grid ``projwfc.x`` used.

    The energy window is taken from the reference file rather than from this
    run's own band extremes. ``partialdos`` sizes its grid from *QE's*
    eigenvalues, which differ from these in the fourth decimal of an eV, and
    without pinning the window the two curves would be sampled at points
    0.0005 eV apart -- a difference in where they are evaluated, not in what
    they are.
    """
    system = build_system(read_pw_input(_input_path(CASES[stem], testsuite)))
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    scf = run_scf(system, pseudos, conv_thr=1e-10, max_iterations=80)
    energies_ev = read_pdos_file(GENERATED / f"reference.{stem}.pdos_tot")[0]
    pdos, states = run_pdos(
        system,
        pseudos,
        scf,
        delta_e=DELTA_E / RY_TO_EV,
        degauss=DEGAUSS.get(stem),
        emin=energies_ev[0] / RY_TO_EV,
        emax=energies_ev[-1] / RY_TO_EV,
    )
    # The raw projections as well: the density of states is an integral of them
    # and ``print_proj`` lists them directly, which is the closer comparison.
    projections = atomic_projections(Calculation(system, pseudos), scf.wavefunctions)
    return system, scf, pdos, states, projections


@pytest.fixture(scope="module")
def case(qe_testsuite, pseudo_dir):
    def _get(stem: str):
        reference = GENERATED / f"reference.projwfc.{stem}"
        if not reference.is_file():
            pytest.skip(f"no projwfc reference for {stem}")
        return (read_projwfc_output(reference),) + _run(stem, qe_testsuite, pseudo_dir)

    return _get


# --------------------------------------------------------------------------
# 1. The projection columns and the projections themselves
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stem", list(CASES))
def test_the_projection_columns_are_qes(case, stem):
    """``fill_nlmchi``: the same atoms, shells, ``l`` and ``m``, in that order."""
    reference, _, _, pdos, _, _ = case(stem)
    assert len(pdos.channels) == len(reference.states) == reference.natomwfc
    for mine, theirs in zip(pdos.channels, reference.states):
        assert (mine.atom + 1, mine.wfc, mine.l, mine.m + 1) == (
            theirs["atom"], theirs["wfc"], theirs["l"], theirs["m"]
        )
        assert mine.species == theirs["species"]


@pytest.mark.parametrize("stem", list(CASES))
def test_projections_match_band_by_band(case, stem):
    """``|<phi|S|psi>|^2`` at every k-point and band, against ``print_proj``.

    QE prints only the entries above 0.001, so the comparison is made where it
    printed something and the rest is checked to be *below* what it would have
    printed -- which is the whole of the information in that listing.
    """
    reference, _, _, _, _, projections = case(stem)
    # QE stacks the spin channels along its own k axis, up first (``isk``), and
    # carries the columns as ``proj(nwfc, ibnd, ik)``.
    stacked = np.concatenate(list(np.transpose(projections, (0, 1, 3, 2))), axis=0)
    assert stacked.shape == reference.projections.shape
    # The topmost band is dropped from both sides: see :func:`_converged_top`.
    stacked = stacked[:, :-1]
    theirs = reference.projections[:, :-1]
    psi2 = reference.psi2[:, :-1]

    printed = theirs > 0.0
    assert np.abs(stacked - theirs)[printed].max() < PROJECTION
    assert stacked[~printed].max() < PROJECTION
    # ``|psi|^2``, the sum over every column, is printed for every band.
    assert np.abs(stacked.sum(axis=-1) - psi2).max() < PROJECTION


# --------------------------------------------------------------------------
# 2. Löwdin charges
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stem", list(CASES))
def test_lowdin_charges_match(case, stem):
    """Charge per atom, per ``l`` and per ``m``, and the spilling parameter."""
    reference, _, _, pdos, _, _ = case(stem)
    charges = pdos.charges
    by_spin = charges.charges_by_spin
    lm_by_spin = charges.charges_lm_by_spin

    for atom, printed in reference.charges.items():
        assert charges.total[atom - 1] == pytest.approx(
            printed["total"], abs=LOWDIN_CHARGE
        )
        for l, letter in enumerate(L_LABELS[: by_spin.shape[2]]):
            if letter in printed:
                assert by_spin[:, atom - 1, l].sum() == pytest.approx(
                    printed[letter], abs=LOWDIN_CHARGE
                )
        # The per-``m`` decomposition, which is what the symmetrisation moves
        # around while leaving every total above it unchanged. An unpolarized
        # run prints it on the atom's own lines; a polarized one prints it per
        # channel and prints the sums on the atom's line instead.
        blocks = (
            [(spin, printed[key]) for spin, key in enumerate(("up", "down"))]
            if "up" in printed else [(0, printed)]
        )
        for spin, block in blocks:
            for l in range(by_spin.shape[2]):
                if "up" in printed:
                    assert by_spin[spin, atom - 1, l] == pytest.approx(
                        block.get(L_LABELS[l], 0.0), abs=LOWDIN_CHARGE
                    )
                for m, suffix in enumerate(M_LABELS[l][: 2 * l + 1]):
                    label = f"{L_LABELS[l]}{suffix}"
                    if label in block:
                        assert lm_by_spin[spin, atom - 1, l, m] == pytest.approx(
                            block[label], abs=LOWDIN_CHARGE
                        )

    assert charges.spilling == pytest.approx(reference.spilling, abs=LOWDIN_CHARGE)


# --------------------------------------------------------------------------
# 3. The projected density of states
# --------------------------------------------------------------------------


def _compare_curve(mine, theirs, label, window=None) -> float:
    """Relative agreement of two curves in states/eV, sampled on one grid."""
    mine, theirs = mine[: theirs.size], theirs
    if window is not None:
        mine, theirs = mine[window], theirs[window]
    scale = max(theirs.max(), 1.0e-8)
    error = np.abs(mine - theirs).max() / scale
    assert error < PDOS_RELATIVE, f"{label}: {error:.4f} of a peak of {scale:.3f}"
    return error


@pytest.mark.parametrize("stem", list(CASES))
def test_total_dos_and_summed_pdos_match(case, stem):
    """``pdos_tot``: the plain DOS and the sum of every channel, both columns."""
    _, _, _, pdos, states, _ = case(stem)
    energies, columns, _ = read_pdos_file(GENERATED / f"reference.{stem}.pdos_tot")
    assert np.abs(pdos.energies_ev[: energies.size] - energies).max() < 1e-6
    window = energies <= _converged_top(states)

    nspin = pdos.nspin
    dos = np.atleast_2d(pdos.total.dos)[:, : energies.size] / RY_TO_EV
    summed = np.atleast_2d(pdos.summed)[:, : energies.size] / RY_TO_EV
    for spin in range(nspin):
        _compare_curve(dos[spin], columns[spin], f"{stem} dos[{spin}]", window)
        _compare_curve(
            summed[spin], columns[nspin + spin], f"{stem} pdostot[{spin}]", window
        )


@pytest.mark.parametrize("stem", list(CASES))
def test_per_shell_pdos_files_match(case, stem):
    """Every ``pdos_atm#N(X)_wfc#n(l)`` file: its ``ldos`` and every ``m``."""
    _, _, _, pdos, states, _ = case(stem)
    nspin = pdos.nspin
    top = _converged_top(states)
    files = sorted(GENERATED.glob(f"reference.{stem}.pdos_atm*"))
    assert files, f"no per-shell reference files for {stem}"

    for path in files:
        atom, wfc = _atom_and_wfc(path.name)
        energies, columns, _ = read_pdos_file(path)
        window = energies <= top
        ldos = np.atleast_2d(pdos.select(atom=atom - 1, wfc=wfc))
        for spin in range(nspin):
            _compare_curve(
                ldos[spin] / RY_TO_EV, columns[spin],
                f"{path.name} ldos[{spin}]", window,
            )
        # Then one pdos column per m, the spin channels interleaved as
        # ``partialdos`` writes them.
        for m in range((columns.shape[0] - nspin) // nspin):
            mine = np.atleast_2d(pdos.select(atom=atom - 1, wfc=wfc, m=m))
            for spin in range(nspin):
                _compare_curve(
                    mine[spin] / RY_TO_EV,
                    columns[nspin + m * nspin + spin],
                    f"{path.name} m={m}[{spin}]",
                    window,
                )


def _atom_and_wfc(name: str) -> tuple[int, int]:
    """``reference.si2-us.pdos_atm#2(Si)_wfc#1(s)`` -> ``(2, 1)``."""
    match = re.search(r"pdos_atm#(\d+)\([^)]*\)_wfc#(\d+)", name)
    assert match is not None, name
    return int(match.group(1)), int(match.group(2))


# --------------------------------------------------------------------------
# The sum rules, which need no reference at all
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stem", list(CASES))
def test_the_projections_obey_bessels_inequality(case, stem):
    """No band puts more than all of itself on the atomic basis.

    ``sum_p |<phi_p|S|psi>|^2 <= <psi|S|psi> = 1`` for a set orthonormal in the
    generalised metric, which is exactly what the Löwdin transform makes the
    orbitals -- so this checks the orthogonalisation, the ``S`` in the
    projection and the symmetrisation at once, and it needs no reference.
    QE's own ``|psi|^2`` line never exceeds 1.000 either.

    It is a statement about the *projections*, not about the density of states:
    Marzari-Vanderbilt's delta goes negative in its tails, so ``sum_p D_p`` may
    legitimately exceed ``D`` at an energy where ``D`` has been pushed down.
    """
    _, _, _, _, _, projections = case(stem)
    assert projections.sum(axis=2).max() < 1.0 + 1e-10


@pytest.mark.parametrize("stem", list(CASES))
def test_the_projected_states_integrate_to_the_projection_sum(case, stem):
    """``sum_p N_p`` above every band counts each band's projection once.

    ``N_p(E) -> sum_kb w_k proj[k, b, p]`` when ``E`` is above the whole
    spectrum, whatever the scheme -- the integration carries the k-point
    weights and the spin degeneracy and nothing else. It is the sum rule with
    the incompleteness of the atomic basis divided out, so unlike a comparison
    against the total density of states it is exact.

    The optimised tetrahedron method reaches it only to 1e-3, and for the reason
    `PLAN.md` P8 (trap 5) already records: its corner energies are a stencil
    with negative weights, so some of them fall outside the range of the
    eigenvalues the grid was sized from.
    """
    _, _, _, pdos, states, projections = case(stem)
    weights = np.asarray(states.kpoints.weights)

    expected = float(np.einsum("skpb,k->", projections, weights))
    counted = float(np.sum(pdos.integrated_by_spin[..., -1]))
    tolerance = 2e-3 if pdos.scheme.replace("_", "-") == "tetrahedra-opt" else 1e-6
    assert counted == pytest.approx(expected, abs=tolerance * max(expected, 1.0))


@pytest.mark.parametrize("stem", list(CASES))
def test_the_lowdin_charges_nearly_span_the_occupied_states(case, stem):
    """The spilling is small and positive: the atomic basis nearly spans them.

    Sanchez-Portal's parameter is ``1 - sum n / nelec``; a negative one would
    mean the projections overcounted (a broken orthogonalisation), and a large
    one that the dataset's own orbitals do not describe its own valence states.
    Both are failure modes rather than tolerances, which is why the bound is
    loose and two-sided.
    """
    _, _, _, pdos, _, _ = case(stem)
    charges = pdos.charges
    assert 0.0 < charges.spilling < 0.05
    assert charges.charges_by_spin.sum() == pytest.approx(
        charges.nelec * (1.0 - charges.spilling), rel=1e-10
    )


def test_a_denser_grid_goes_through_the_nscf_route(case, pseudo_dir):
    """``grid=`` re-solves the bands before projecting, as a ``nscf`` run would.

    There is no ``projwfc.x`` reference for this, because producing one means a
    three-step ``scf`` -> ``nscf`` -> ``projwfc`` sequence sharing an outdir and
    the committed cases are all single runs. What is checked is that the route
    works and that the projections it produces still obey Bessel's inequality --
    the wavefunctions are new, the projectors are rebuilt on a new k-set, and
    both have to line up for that to hold.
    """
    _, system, scf, coarse, _, _ = case("si2-us")
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    pdos, states = run_pdos(
        system, pseudos, scf, grid=(4, 4, 4), nbnd=8, delta_e=DELTA_E / RY_TO_EV
    )

    assert states.kpoints.nk == 8  # the irreducible wedge of a 4x4x4 grid
    assert np.max(np.atleast_2d(pdos.summed) - np.atleast_2d(pdos.total.dos)) < 1e-10
    # The Lowdin charges are a property of the k-set, so they move -- but not by
    # much, and the spilling is a property of the *basis* and barely at all.
    assert pdos.charges.total == pytest.approx(coarse.charges.total, abs=0.02)
    assert pdos.charges.spilling == pytest.approx(coarse.charges.spilling, abs=2e-3)
