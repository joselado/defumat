"""P20 check: DFT+U against Quantum ESPRESSO.

The simplified rotationally-invariant functional (QE's ``lda_plus_u_kind = 0``)
on the seven cases the committed pseudopotentials can reach. Each isolates one
thing:

* ``lda+U-noU`` -- antiferromagnetic FeO with ``U = 1e-8 eV``. The **null test**:
  every line of the Hubbard machinery runs -- the projectors are built, the
  occupation matrix is measured, symmetrised and mixed, the term is in the
  Hamiltonian -- and the answer must be the plain LSDA one. A sign error or a
  factor of two in the projection is invisible here and shows up in every case
  below, which is what makes running it first worth the time.
* ``lda+U`` -- the same cell with ``U = 4.3 eV`` on both iron sublattices.
  Ultrasoft, ``nspin = 2``, ``atomic`` projectors, symmetry-reduced k-points, so
  the occupation matrix has to be symmetrised.
* ``lda+U-user_ns`` -- the same again with ``starting_ns_eigenvalue`` steering
  the occupation matrix. It converges to a **different** self-consistent state
  (-174.5374 Ry against -174.4716), which is the point: a magnetic insulator has
  more than one, and ``ns_adj`` is how a particular one is asked for. Getting
  ``ns_adj``'s timing wrong -- QE runs it after the *first* iteration, not on the
  starting matrix -- lands on the other solution.
* ``lda+U_force`` -- the same, displaced, with forces (see ``test_forces_*``).
* ``ni-ldau-nospin`` -- fcc nickel, ``nspin = 1``. **The only case that tests the
  factor of two**: ``ns`` is halved when there is one channel and the energy is
  doubled to compensate, while the potential is not. Every ``nspin = 2`` case
  passes with or without that rule.
* ``ni-ldau-ortho`` -- the same cell with ``ortho-atomic`` projectors, which is
  what most published DFT+U uses. It is also what exposed the atomic-orbital
  renormalisation (``upf_check_atwfc_norm``): the Löwdin orthogonalisation runs
  over *all* the atomic orbitals, so nickel's 4s enters the transform that
  produces the 3d projectors, and the file's 4s is not normalised.
* ``ni-ldau-j0`` -- the same with ``J0 = 1 eV``, which is the only validation the
  ``J0`` and ``beta`` terms have (QE's own ``lda+U+J0.in`` needs a
  pseudopotential that is not committed here).

References are regenerated with the vendored ``pw.x`` at ``conv_thr = 1e-10``
(``tools/generate_reference.py``), for the reason the LSDA suite gives: the
committed benchmarks stop at 1e-6 and their printed *terms* are worth about
1e-4 Ry.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from defumat.forces import compute_forces
from defumat.io import read_qe_output
from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf import run_scf
from defumat.scf.driver import Calculation
from defumat.system import build_system
from tests.tolerances import (
    FORCE_RY_BOHR,
    MAGNETIZATION_BOHRMAG,
    TOTAL_ENERGY_RY,
)

pytestmark = [pytest.mark.regression, pytest.mark.slow]

GENERATED = Path(__file__).resolve().parents[1] / "data" / "qe"

#: ``(test-suite directory or None, input name)``. ``None`` means an input
#: committed under ``tests/data/qe`` rather than borrowed from QE's suite.
CASES = [
    ("pw_lda+U", "lda+U-noU.in"),
    ("pw_lda+U", "lda+U.in"),
    ("pw_lda+U", "lda+U-user_ns.in"),
    ("pw_lda+U", "lda+U_force.in"),
    (None, "ni-ldau-nospin.in"),
    (None, "ni-ldau-ortho.in"),
    (None, "ni-ldau-j0.in"),
]

IDS = [f"{directory or 'local'}/{name}" for directory, name in CASES]

#: Energy *terms* are first-order in the density residual where the total is
#: second-order, so two codes at the same fixed point to 1e-11 in ``dr2`` still
#: differ here by more than they do in the total. Achieved: <= 1.3e-5 Ry on the
#: FeO cases and <= 6.1e-5 on ortho-atomic nickel, against totals that agree to
#: <= 6.7e-9 Ry. The Hubbard term itself is far tighter than this bound -- see
#: :data:`HUBBARD_TERM_RY`, which is the number this phase is actually about.
TERM_RY = 1e-4

#: The Hubbard energy specifically, which is what P20 adds. It is a functional
#: of ``ns`` alone, and ``ns`` converges with the density rather than being
#: differentiated from it, so it tracks the total rather than the other terms.
#: Achieved: <= 4.6e-7 Ry over the seven cases.
HUBBARD_TERM_RY = 2e-6

#: ``Tr[ns]`` per atom and channel, as ``write_ns`` prints it with five
#: decimals -- so the last printed digit is what the comparison can ask for.
#: Achieved: 1e-5, i.e. exactly that digit.
TRACE = 1e-4

#: The eigenvalues of ``ns``, which ``write_ns`` prints with three decimals --
#: so half of the last printed digit is all the comparison can ask for.
EIGENVALUE = 1e-3


def _input_path(directory, name, qe_testsuite):
    return GENERATED / name if directory is None else qe_testsuite / directory / name


def _reference_path(directory, name):
    stem = Path(name).stem
    return GENERATED / (
        f"reference.out.{stem}" if directory is None
        else f"reference.out.{directory}-{stem}"
    )


@lru_cache(maxsize=None)
def _converged(directory, name, qe_testsuite, pseudo_dir):
    pwin = read_pw_input(_input_path(directory, name, qe_testsuite))
    system = build_system(pwin)
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    result = run_scf(
        system, pseudos, calculation=calculation, conv_thr=1e-10,
        max_iterations=250,
        mixing_beta=float(pwin.get("electrons", "mixing_beta", 0.7)),
        mixing_fixed_ns=int(pwin.get("electrons", "mixing_fixed_ns", 0)),
    )
    return system, calculation, result


@pytest.fixture(scope="module", params=CASES, ids=IDS)
def case(request, qe_testsuite, pseudo_dir):
    directory, name = request.param
    path = _reference_path(directory, name)
    if not path.is_file():
        pytest.skip(f"no generated reference for {name}; run tools/generate_reference.py")
    system, calculation, result = _converged(directory, name, qe_testsuite, pseudo_dir)
    return directory, name, system, calculation, result, read_qe_output(path)


def test_converged(case):
    _, name, _, _, result, _ = case
    assert result.converged, f"{name} did not reach conv_thr"


def test_total_energy(case):
    _, _, _, _, result, reference = case
    assert result.total_energy == pytest.approx(
        reference.total_energy, abs=TOTAL_ENERGY_RY
    )


def test_hubbard_energy(case):
    """``eth``, the term QE prints as ``Hubbard energy``."""
    _, _, _, _, result, reference = case
    assert "hubbard" in result.energy_terms
    assert result.energy_terms["hubbard"] == pytest.approx(
        reference.energy_terms["hubbard"], abs=HUBBARD_TERM_RY
    )


def test_energy_terms(case):
    _, _, _, _, result, reference = case
    tolerance = TERM_RY
    for term, value in result.energy_terms.items():
        expected = reference.energy_terms.get(term)
        if expected is None:
            continue
        assert value == pytest.approx(expected, abs=tolerance), term


def test_occupation_matrix_traces(case):
    """``Tr[ns]`` per atom and channel, against QE's ``write_ns`` output."""
    _, _, _, _, result, reference = case
    expected = _reference_traces(reference)
    computed = result.hubbard_occupations
    assert set(computed) == set(expected)
    for atom, (up, down, total) in computed.items():
        assert (up, down, total) == pytest.approx(expected[atom], abs=TRACE)


def test_occupation_matrix_is_symmetric(case):
    """``ns`` is real symmetric by construction; the group average keeps it so."""
    _, _, _, _, result, _ = case
    ns = np.asarray(result.ns)
    assert ns == pytest.approx(ns.transpose(0, 1, 3, 2), abs=1e-12)


def test_occupation_matrix_eigenvalues(case):
    """The eigenvalues of ``ns``, against the ones ``write_ns`` prints.

    A far stronger statement than the traces: five numbers per atom per channel,
    and they are invariant under the unitary freedom a degenerate manifold has,
    so they compare the *matrix* without comparing a basis choice.

    They are **not** confined to ``[0, 1]``. Non-orthogonal projectors --
    ``atomic``, which the FeO cases use -- can put more than one electron in an
    orbital, and QE's own output for ``lda+U.in`` prints 1.003; it is the
    documented reason ``norm-atomic`` and ``ortho-atomic`` exist. Asserting the
    bound would be asserting a property of the physics that the chosen
    projectors do not have.
    """
    _, _, _, _, result, reference = case
    expected = _reference_eigenvalues(reference)
    ns = np.asarray(result.ns)
    for slot, atom in enumerate(result.hubbard_setup.atoms):
        for spin in range(ns.shape[0]):
            values = np.linalg.eigvalsh(ns[spin, slot])
            assert values == pytest.approx(
                expected[atom][spin], abs=EIGENVALUE
            ), f"atom {atom + 1}, spin {spin + 1}"


def test_magnetization(case):
    _, _, system, _, result, reference = case
    if system.nspin != 2 or reference.absolute_magnetization is None:
        pytest.skip("unpolarized")
    assert result.absolute_magnetization == pytest.approx(
        reference.absolute_magnetization, abs=MAGNETIZATION_BOHRMAG
    )


def test_forces(qe_testsuite, pseudo_dir):
    """``force_hub``, from differentiating the energy through the projectors.

    ``pw_lda+U/lda+U_force.in`` displaces both irons along the body diagonal, so
    the Hubbard contribution to the force is nonzero and QE prints the total.
    Nothing here transcribes ``force_hub.f90``: the occupation matrix is
    measured through projectors that move with the atoms, and ``jax.grad`` of
    the energy carries that dependence.
    """
    directory, name = "pw_lda+U", "lda+U_force.in"
    path = _reference_path(directory, name)
    if not path.is_file():
        pytest.skip("no generated reference; run tools/generate_reference.py")
    _, calculation, result = _converged(directory, name, qe_testsuite, pseudo_dir)
    reference = read_qe_output(path)

    forces = compute_forces(calculation, result, method="autodiff")
    assert np.abs(np.asarray(forces.forces) - reference.forces).max() < FORCE_RY_BOHR
    # ...and the term is not negligible on this geometry, so the agreement above
    # is a statement about ``force_hub`` and not only about the other five terms.
    assert np.abs(reference.forces).max() > 1e-3


def test_analytic_forces_are_refused(qe_testsuite, pseudo_dir):
    """The transcribed expressions have no ``force_hub`` and say so."""
    _, calculation, result = _converged(
        "pw_lda+U", "lda+U_force.in", qe_testsuite, pseudo_dir
    )
    with pytest.raises(NotImplementedError, match="force_hub"):
        compute_forces(calculation, result, method="analytic")


def _reference_traces(reference) -> dict:
    """``{atom: (up, down, total)}`` from QE's ``write_ns`` block.

    ``write_ns`` prints three numbers for a polarized run and one -- the whole
    manifold's occupation -- for an unpolarized one, where the two channels are
    identical by construction. Both are expanded to the triple that
    :attr:`SCFResult.hubbard_occupations` reports.
    """
    text = Path(reference.path).read_text()
    traces = {}
    for line in text.splitlines():
        if "Tr[ns(" not in line:
            continue
        head, _, values = line.partition("=")
        atom = int(head[head.index("(") + 1 : head.index(")")])
        numbers = [float(v) for v in values.split()]
        if len(numbers) == 1:
            numbers = [0.5 * numbers[0], 0.5 * numbers[0], numbers[0]]
        # The last block printed is the converged one.
        traces[atom - 1] = tuple(numbers[:3])
    return traces


def _reference_eigenvalues(reference) -> dict:
    """``{atom: [eigenvalues per channel]}`` from QE's ``write_ns`` block.

    The last block printed is the converged one, and an unpolarized run prints
    one channel where a polarized one prints two.
    """
    lines = Path(reference.path).read_text().splitlines()
    values, atom = {}, None
    for i, line in enumerate(lines):
        if "ATOM" in line and line.strip().startswith("---"):
            atom = int(line.replace("-", " ").split()[1]) - 1
            values[atom] = []
        elif atom is not None and line.strip() == "eigenvalues:":
            values[atom].append([float(v) for v in lines[i + 1].split()])
    # ``write_ns`` runs at the first iteration too; keep only the last block.
    counts = {a: len(v) for a, v in values.items()}
    per_atom = min(counts.values())
    channels = 1 if reference.absolute_magnetization in (None, 0.0) else 2
    channels = min(channels, per_atom)
    return {a: v[-channels:] for a, v in values.items()}
