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

import jax.numpy as jnp
import numpy as np
import pytest

from defumat.io import read_qe_output
from defumat.io.pwin import read_pw_input
from defumat.pseudo import read_upf
from defumat.scf import run_scf
from defumat.system import build_system
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
    from defumat.pseudo.augmentation import build_augmentation
    from defumat.pseudo.projectors import projector_channels

    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudo = read_upf(
        pseudo_dir / (pseudo_file or system.structure.species[0].pseudo_file)
    )
    from defumat.basis.builder import build_basis

    basis = build_basis(system)
    augmentation = build_augmentation((pseudo,), system.structure, system.cell, basis.dense)

    channels = projector_channels(pseudo)
    expected = np.zeros((len(channels), len(channels)))
    for i, (nb_i, _, lm_i) in enumerate(channels):
        for j, (nb_j, _, lm_j) in enumerate(channels):
            if lm_i == lm_j:
                expected[i, j] = pseudo.augmentation.q[nb_i, nb_j]

    assert np.asarray(augmentation.qq[0]) == pytest.approx(expected, abs=1e-6)


def test_two_species_naming_one_dataset_share_one_augmentation_charge(pseudo_dir):
    """One species per magnetic site must not build ``Q_ij(G)`` twice.

    ``angle1``/``angle2`` are per *species*, so the standard way to write a
    noncollinear texture is one species per site, all of them naming the same
    UPF file. ``Q_ij(G)`` is a property of the dataset and of the G set, so
    every one of those species wants the identical ``(nh, nh, ngm)`` array --
    which on a 45-atom NiBr2 slab is 65 GB each. They share one array here, and
    the check is object identity rather than equality: two arrays that agree
    numerically still cost twice the memory, which is the whole point.
    """
    import dataclasses

    from defumat.pseudo.augmentation import build_augmentation
    from defumat.basis.builder import build_basis
    from defumat.system.structure import Structure

    system = build_system(read_pw_input(CASES / "si2-us.in"))
    basis = build_basis(system)

    # The same file read twice, as ``Calculator`` reads it: two distinct
    # objects, so nothing can be deduplicated by identity.
    path = pseudo_dir / system.structure.species[0].pseudo_file
    first, second = read_upf(path), read_upf(path)
    assert first is not second

    base = system.structure
    split = Structure(
        positions=base.positions,
        types=(0, 1),
        species=(base.species[0], dataclasses.replace(base.species[0], name="Si1")),
        precision=base.precision,
    )

    augmentation = build_augmentation((first, second), split, system.cell, basis.dense)
    assert augmentation.qgm[0] is augmentation.qgm[1]
    assert augmentation.qq[0] is augmentation.qq[1]
    assert augmentation.species_atoms == ((0,), (1,))


def _augmentation_pair(pseudo_dir, case="si2-us"):
    """The same cell's augmentation charge under both storage schemes."""
    from defumat.basis.builder import build_basis
    from defumat.pseudo.augmentation import build_augmentation

    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    dense = build_basis(system).dense
    stored = build_augmentation(pseudos, system.structure, system.cell, dense)
    tabulated = build_augmentation(
        pseudos, system.structure, system.cell, dense, max_bytes=0
    )
    return system, dense, stored, tabulated


def test_the_tabulated_augmentation_charge_agrees_with_the_stored_one(pseudo_dir):
    """QE's table against evaluating the transform at every G, on one cell.

    The two are different answers to the same question and the whole switch
    between them rests on the gap being below anything this project claims.
    ``Q(G = 0)`` is *exact* rather than close, because at ``q = 0`` the four
    Lagrange weights are ``(1, 0, 0, 0)`` and the interpolation reads the first
    knot straight out -- which is also what makes the ``PP_Q`` check above
    reach the table.
    """
    from defumat.pseudo.augmentation import TabulatedAugmentation

    system, dense, stored, tabulated = _augmentation_pair(pseudo_dir)
    assert isinstance(tabulated, TabulatedAugmentation)

    assert np.asarray(stored.qq[0]) == pytest.approx(np.asarray(tabulated.qq[0]), abs=0)

    rng = np.random.default_rng(0)
    nh = stored.qgm[0].shape[0]
    nat = len(stored.species_atoms[0])
    becsum = rng.normal(size=(nat, nh, nh))
    becsum = jnp.asarray(0.5 * (becsum + becsum.transpose(0, 2, 1)))
    charge = np.asarray(stored.charge((becsum,)))
    assert np.asarray(tabulated.charge((becsum,))) == pytest.approx(
        charge, rel=0, abs=1e-9 * np.abs(charge).max()
    )

    potential = jnp.asarray(
        rng.normal(size=dense.ngm) + 1j * rng.normal(size=dense.ngm)
    )
    integrals = np.asarray(stored.integrals(potential)[0])
    assert np.asarray(tabulated.integrals(potential)[0]) == pytest.approx(
        integrals, rel=0, abs=1e-9 * np.abs(integrals).max()
    )

    # And the point of the exercise. The ratio is ngm/nqx and so grows with
    # the cell: 9.7x on this two-atom one, 38x on benchmarks/si8-us-1k.in, and
    # 9100x on the 45-atom NiBr2 slab this was written for.
    held_stored = sum(q.nbytes for q in stored.qgm)
    held_table = sum(t.nbytes for t in tabulated.tables if t is not None)
    assert held_table * 5 < held_stored


@pytest.mark.parametrize("chunk", [1024, 5000], ids=["divides-npad", "does-not"])
def test_the_augmentation_chunk_is_invisible(pseudo_dir, monkeypatch, chunk):
    """The block size is a loop bound over an exact sum, so it changes nothing.

    ``ngm`` is not a multiple of any round number, so the scan pads; a padded
    G is the **origin**, where ``Q_ij(G)`` is far from zero, and forgetting to
    mask it adds a smooth spurious term rather than raising anything.
    """
    from defumat.basis.builder import build_basis
    from defumat.pseudo.augmentation import build_augmentation

    system = build_system(read_pw_input(CASES / "si2-us.in"))
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    dense = build_basis(system).dense
    reference = build_augmentation(system_pseudos := pseudos, system.structure,
                                   system.cell, dense, max_bytes=0)

    monkeypatch.setenv("DEFUMAT_AUG_CHUNK", str(chunk))
    chunked = build_augmentation(
        system_pseudos, system.structure, system.cell, dense, max_bytes=0
    )
    assert chunked.chunk == min(chunk, dense.ngm)

    rng = np.random.default_rng(1)
    nh = reference.beta_of[0].shape[0]
    nat = len(reference.species_atoms[0])
    becsum = rng.normal(size=(nat, nh, nh))
    becsum = jnp.asarray(0.5 * (becsum + becsum.transpose(0, 2, 1)))
    expected = np.asarray(reference.charge((becsum,)))
    got = np.asarray(chunked.charge((becsum,)))
    assert not np.isnan(got).any()
    assert got == pytest.approx(expected, rel=0, abs=1e-12 * np.abs(expected).max())


def test_past_the_end_of_the_table_is_nan_and_not_a_clamp(pseudo_dir):
    """A gather clamps its indices silently; a clamped Q^L(q) is plausible.

    The failure this guards is a *stress* under a strain large enough to move
    ``|G|`` past ``qmax``: the extrapolation would be smooth, of the right
    order and wrong, and no identity in the suite is sensitive to it. NaN is
    the only report available from inside a traced function.
    """
    from defumat.pseudo.augmentation import AUG_DQ, _interpolate_qrad, _qrad_table

    system = build_system(read_pw_input(CASES / "si2-us.in"))
    pseudo = read_upf(pseudo_dir / system.structure.species[0].pseudo_file)
    table = _qrad_table(pseudo, 5.0, float(system.cell.volume), 3)
    top = (table.shape[-1] - 4) * AUG_DQ

    inside = np.asarray(_interpolate_qrad(table, jnp.asarray([0.0, top - 1e-6])))
    assert np.isfinite(inside).all()
    outside = np.asarray(_interpolate_qrad(table, jnp.asarray([top + 0.05, top + 10.0])))
    assert np.isnan(outside).all()


def test_the_storage_scheme_is_chosen_by_size(pseudo_dir):
    """The switch is the array's size, and a generous budget keeps the old path."""
    from defumat.basis.builder import build_basis
    from defumat.pseudo.augmentation import (
        AugmentationCharge, TabulatedAugmentation, build_augmentation,
    )

    system = build_system(read_pw_input(CASES / "si2-us.in"))
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    dense = build_basis(system).dense
    roomy = build_augmentation(pseudos, system.structure, system.cell, dense,
                               max_bytes=1 << 40)
    assert type(roomy) is AugmentationCharge
    tight = build_augmentation(pseudos, system.structure, system.cell, dense,
                               max_bytes=0)
    assert isinstance(tight, TabulatedAugmentation)


@pytest.mark.slow
def test_an_scf_through_the_table_reaches_the_same_total_energy(pseudo_dir, monkeypatch):
    """The whole loop, not just the two contractions.

    ``newd`` rebuilds ``D_ij`` from ``integrals`` and ``addusdens`` the density
    from ``charge``, both every iteration, so an error in either compounds
    through the self-consistency rather than staying where it was made.
    """
    system = build_system(read_pw_input(CASES / "si2-us.in"))
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    monkeypatch.setenv("DEFUMAT_AUG_MAX_BYTES", "off")
    stored = run_scf(system, pseudos)
    monkeypatch.setenv("DEFUMAT_AUG_MAX_BYTES", "0")
    tabulated = run_scf(system, pseudos)

    assert len(tabulated.history) == len(stored.history)
    assert float(tabulated.total_energy) == pytest.approx(
        float(stored.total_energy), abs=1e-8
    )


@pytest.mark.parametrize(
    "text,expected",
    [("off", 1 << 62), ("2G", 2 * 1024**3), ("512M", 512 * 1024**2),
     ("64K", 65536), ("1048576", 1048576), ("0", 0)],
)
def test_the_augmentation_budget_reads_every_form_the_guide_prints(
    monkeypatch, text, expected
):
    """``docs/features.tex`` prints ``2G``; ``float("2G")`` raises.

    The suffixes are not decoration: ``DEFUMAT_TEST_MEM_MAX`` already takes
    them, so anyone who has capped a test run will write this one the same way
    and get a traceback out of setup rather than a memory budget.
    """
    from defumat.pseudo.augmentation import _aug_max_bytes

    monkeypatch.setenv("DEFUMAT_AUG_MAX_BYTES", text)
    assert _aug_max_bytes() == expected


def test_the_augmentation_budget_defaults_when_unset(monkeypatch):
    from defumat.pseudo.augmentation import AUG_MAX_BYTES, _aug_max_bytes

    monkeypatch.delenv("DEFUMAT_AUG_MAX_BYTES", raising=False)
    assert _aug_max_bytes() == AUG_MAX_BYTES


def _converged_both_schemes(case, pseudo_dir, monkeypatch):
    """The same cell converged under each augmentation storage scheme."""
    from defumat.calculator import Calculator

    out = []
    for budget in ("off", "0"):
        monkeypatch.setenv("DEFUMAT_AUG_MAX_BYTES", budget)
        calculator = Calculator.from_file(CASES / f"{case}.in", pseudo_dir,
                                          announce=False)
        out.append((calculator.calculation, calculator.get_scf()))
    return out


@pytest.mark.slow
def test_a_force_through_the_table_matches_the_stored_path(pseudo_dir, monkeypatch):
    """The derivative, which is the only thing the table's two guards exist for.

    Nothing else here reaches it: every committed cell is under the budget, so
    the whole force and stress suite runs the stored path and would pass with
    ``grad`` through the interpolation broken. The agreement should be tight
    rather than merely close, because ``jax.grad`` of the four Lagrange weights
    *is* ``dqvan2``'s ``work1`` -- QE differentiates the same stencil by hand
    and gets the same expression.
    """
    from defumat.forces import compute_forces

    (calc_a, res_a), (calc_b, res_b) = _converged_both_schemes(
        "si2-us-force", pseudo_dir, monkeypatch
    )
    stored = np.asarray(compute_forces(calc_a, res_a, method="autodiff").forces)
    tabulated = np.asarray(compute_forces(calc_b, res_b, method="autodiff").forces)
    assert np.isfinite(tabulated).all()
    assert np.abs(tabulated - stored).max() < 1e-8


@pytest.mark.slow
def test_a_stress_through_the_table_matches_the_stored_path(pseudo_dir, monkeypatch):
    """The strain derivative is where the table's range guard actually lives.

    ``Calculation.at_strain`` rebuilds the augmentation charge on a **traced**
    cell, so the interpolation is inside the stress tape and ``|G|`` is the
    thing being differentiated. A clamped extrapolation past ``qmax`` would
    show up here and nowhere else, which is why off-table is NaN and why the
    table reaches ``2 sqrt(ecutrho)``.
    """
    from defumat.stress import compute_stress

    (calc_a, res_a), (calc_b, res_b) = _converged_both_schemes(
        "si2-us-stress", pseudo_dir, monkeypatch
    )
    stored = np.asarray(compute_stress(calc_a, res_a).tensor)
    tabulated = np.asarray(compute_stress(calc_b, res_b).tensor)
    assert np.isfinite(tabulated).all()
    assert np.abs(tabulated - stored).max() < 1e-8
