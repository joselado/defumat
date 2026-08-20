"""P12 check: ultrasoft and PAW silicon against Quantum ESPRESSO.

No committed QE benchmark covers these pseudopotentials, so the references here
are generated once with the vendored ``pw.x`` (``tools/generate_reference.py``)
and stored beside the inputs. They are run at ``conv_thr = 1e-10`` on both sides
so that the two codes stop at the same fixed point and the energy terms can be
compared, not only the variational total.

The cases build on each other, and each isolates one thing:

* ``si2-nc-dual8`` -- **norm-conserving** at ``dual = 8``. No augmentation
  charge at all; what it exercises is the smooth/dense grid split, which the
  norm-conserving path never needed. A failure here is a plumbing failure and
  nothing else.
* ``si2-us`` / ``si8-us`` -- ultrasoft, two atoms and eight. The eight-atom cell
  is the one that catches anything scaling with the number of atoms, and its
  cubic cell is a *supercell*, which is where QE's rule about fractional
  translations bites (see ``system.symmetry.is_supercell``).
* ``si2-paw`` / ``si8-paw`` -- PAW, which adds the one-centre terms on top of
  everything ultrasoft does. QE prints its one-centre contribution as a separate
  energy term, so it is checked directly rather than only through the total.
* ``si2-paw-fullk`` -- the same two-atom cell on the *unreduced* k-grid. With
  every point of every star present, ``becsum`` is symmetric before anything
  symmetrises it, so this case validates the one-centre machinery with
  ``PAW_symmetrize`` factored out. It was worth having: the one-centre terms
  were right to 2e-7 relative here while the reduced-k case was still 3e-5 out,
  which is what localised the remaining error to the symmetrisation.
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
from tests.tolerances import (
    EIGENVALUE_EV,
    ENERGY_TERM_RY,
    TOTAL_ENERGY_RY,
    USPP_TERM_RY,
)

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"

NORM_CONSERVING = ["si2-nc-dual8"]
ULTRASOFT = ["si2-us", "si8-us"]
PAW = ["si2-paw", "si2-paw-fullk", "si8-paw"]
ALL = NORM_CONSERVING + ULTRASOFT + PAW


@lru_cache(maxsize=None)
def _converged(case: str, pseudo_dir: Path):
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    return system, pseudos, run_scf(system, pseudos, conv_thr=1e-10, max_iterations=80)


def _reference(case: str):
    path = CASES / f"reference.out.{case}"
    if not path.is_file():
        pytest.skip(f"no generated reference for {case}; run tools/generate_reference.py")
    return read_qe_output(path)


@pytest.mark.parametrize("case", ALL)
def test_total_energy_matches_reference(pseudo_dir, case):
    _, _, result = _converged(case, pseudo_dir)
    reference = _reference(case)

    assert result.converged
    assert result.total_energy == pytest.approx(reference.total_energy, abs=TOTAL_ENERGY_RY)


@pytest.mark.parametrize("case", ALL)
def test_energy_terms_match_reference(pseudo_dir, case):
    _, _, result = _converged(case, pseudo_dir)
    reference = _reference(case)

    assert set(result.energy_terms) == set(reference.energy_terms)
    for term, value in reference.energy_terms.items():
        tolerance = ENERGY_TERM_RY if term == "ewald" else USPP_TERM_RY
        assert result.energy_terms[term] == pytest.approx(value, abs=tolerance), term

    assert sum(result.energy_terms.values()) == pytest.approx(result.total_energy, abs=1e-10)


@pytest.mark.parametrize("case", ALL)
def test_eigenvalues_match_reference(pseudo_dir, case):
    _, _, result = _converged(case, pseudo_dir)
    reference = _reference(case)

    ours = result.eigenvalues_ev
    theirs = reference.eigenvalues[0][:, : ours.shape[1]]
    assert ours == pytest.approx(theirs, abs=EIGENVALUE_EV)


@pytest.mark.parametrize("case", ULTRASOFT + PAW)
def test_augmented_density_integrates_to_the_electron_count(pseudo_dir, case):
    """The identity that says ``becsum`` and ``q_ij`` agree with each other.

    An ultrasoft state is normalised as ``<psi|S|psi> = 1``, not
    ``<psi|psi> = 1``, and the part of the norm that ``S`` supplies is exactly
    the augmentation charge's integral. So ``int rho = nelec`` holds *identically*
    -- to round-off, not to a tolerance -- as long as the ``q_ij`` used in ``S``
    and the ``Q_ij(G=0)`` used in ``addusdens`` are the same numbers. It is the
    cheapest check there is that they are, and it fails loudly if either the
    projector bookkeeping or the ``becsum`` packing is wrong.
    """
    system, pseudos, result = _converged(case, pseudo_dir)
    nelec = sum(pseudos[t].z_valence for t in system.structure.types)

    density = np.asarray(result.density)
    charge = float(np.sum(density)) * float(system.cell.volume) / density.size
    assert charge == pytest.approx(nelec, abs=1e-9)


@pytest.mark.parametrize(
    ("case", "pseudo_file"),
    [
        # q_with_l = T: Q^L_ij is tabulated per (pair, L).
        ("si2-us", None),
        # q_with_l = F: one Q_ij per pair, expanded onto the L grid by
        # ``upf._expand_qij``. Checking it here rather than only through a total
        # energy is what makes the expansion falsifiable on its own.
        ("si2-us", "O.pz-rrkjus.UPF"),
        ("si2-us", "Ni.pz-nd-rrkjus.UPF"),
    ],
)
def test_the_augmentation_charge_reproduces_the_files_own_q(pseudo_dir, case, pseudo_file):
    """``Omega * Q_ij(G=0)`` must be the ``PP_Q`` block the UPF file tabulates.

    An independent check on the whole radial-to-reciprocal chain: ``PP_Q`` is
    written by the pseudopotential generator, not derived from ``PP_QIJL``, so
    agreeing with it exercises the mesh (``kkbeta``, not the 10-bohr one), the
    Simpson weights, the ``4 pi / Omega`` normalisation and the ``L = 0``
    coupling coefficient at once. The bound is the file's own consistency: the
    two tabulations agree with each other to about 1e-6 relative.

    The cell is silicon's in every case -- only ``Q(G = 0)`` is being read, and
    it depends on the cell through ``1/Omega`` alone -- so a pseudopotential from
    another element can be dropped into it to exercise its own storage format.
    """
    from pypresso.pseudo.augmentation import build_augmentation
    from pypresso.pseudo.projectors import projector_channels

    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudo = read_upf(
        pseudo_dir / (pseudo_file or system.structure.species[0].pseudo_file)
    )
    from pypresso.basis.builder import build_basis

    basis = build_basis(system)
    augmentation = build_augmentation((pseudo,), system.structure, system.cell, basis.dense)

    channels = projector_channels(pseudo)
    expected = np.zeros((len(channels), len(channels)))
    for i, (nb_i, _, lm_i) in enumerate(channels):
        for j, (nb_j, _, lm_j) in enumerate(channels):
            if lm_i == lm_j:
                expected[i, j] = pseudo.augmentation.q[nb_i, nb_j]

    assert np.asarray(augmentation.qq[0]) == pytest.approx(expected, abs=1e-6)
