"""Vertical tunnelling transport, on converged cells.

``PLAN.md`` P66. **Neither ``pw.x`` nor Elk computes this quantity**, so there
is no reference output anywhere in this repository to compare against and the
whole of the validation closes inside the package. That makes the choice of
checks the important part, and they are four:

* the **Tersoff-Hamann limit**, which is exact rather than approximate: widen
  the substrate from a plane to the whole cell and the exit-plane Gram matrix
  becomes the identity by orthonormality, so the transmission becomes P65's
  tunnelling density of states -- the same number :func:`run_stm` returns, with
  no factor between them. It shares no line of code with the plane path and it
  checks the k-weights, the spin degeneracy, the normalisation and the tip
  sampler at once;
* the **three spin regimes**, which must agree on a cell with no magnetization
  -- the check that catches P51's ``for_spin`` factor of two, invisible in any
  ratio;
* the **spin partition**: a substrate polarized along ``n`` plus one along
  ``-n`` is a substrate that takes both, exactly;
* the **physics**, which is the phase's reason for existing: monolayer graphene's
  Dirac pair is degenerate, so the substrate cannot tell its two members apart and
  its transmission *is* the STM image, while an AB bilayer's current has to cross
  both layers and its is not.

The algebra -- the Gram matrix, the contraction, the amplitude weights, the
sampler -- is in ``tests/unit/test_transport_machinery.py`` and needs no SCF.
"""

from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest

from defumat import Calculator
from defumat.workflows.stm import run_stm
from defumat.workflows.transport import run_vertical_transport

pytestmark = pytest.mark.slow

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: Where the two planes go on ``h-sheet.in``, whose atom sits at 0.5.
SHEET = dict(exit_height=0.15, height=0.85, broadening=0.05)


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """``CLAUDE.md``'s rule for a file that sweeps cells: keep the results and
    drop the executables. XLA holds every one it builds for the life of the
    process, and this file runs a dozen cells that share no shape."""
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _converged(case: str):
    calculator = Calculator.from_file(CASES / f"{case}.in", pseudo_dir=PSEUDO)
    calculator.get_scf()
    return calculator


def _variant(tmp_path, case: str, insert: str = "", replacements=()):
    """``case`` with a line added to ``&system`` and some text swapped."""
    text = (CASES / f"{case}.in").read_text()
    for old, new in replacements:
        assert old in text, old
        text = text.replace(old, new)
    if insert:
        text = text.replace("    nat = 1, ntyp = 1,\n",
                            "    nat = 1, ntyp = 1,\n" + insert)
    path = tmp_path / "variant.in"
    path.write_text(text)
    calculator = Calculator.from_file(path, pseudo_dir=PSEUDO)
    calculator.get_scf()
    return calculator


# --------------------------------------------------------------------------
# the Tersoff-Hamann limit: the check that shares no code with what it checks
# --------------------------------------------------------------------------


def test_the_whole_cell_exit_region_is_the_tunnelling_density_of_states():
    """``S_k -> delta_nn'`` by orthonormality, and P65's image comes back.

    Exact rather than approximate, and with **no factor** between the two: the
    on-shell amplitude is normalised as ``sqrt(delta(E - e)/eta)`` precisely so
    that its square is ``tunnelling_weights``. A factor here would be a factor
    everywhere, and nothing else in the phase could see it -- every other check
    is a ratio or a partition.
    """
    calculator = _converged("h-sheet")
    geometry = dict(height=0.80, shape=(6, 6))
    ours = run_vertical_transport(
        calculator.system, calculator.pseudos, calculator.get_scf(),
        exit_height=0.20, broadening=0.02, exit_region="volume", **geometry)
    reference = np.asarray(run_stm(
        calculator.system, calculator.pseudos, calculator.get_scf(),
        width=0.02, **geometry).values)
    assert np.abs(ours.image - reference).max() / reference.max() < 1.0e-11


def test_the_exit_plane_gram_matrix_is_positive_and_hermitian_on_a_real_cell():
    """The structural guarantee, on wavefunctions rather than random vectors.

    ``S_k`` being a Gram matrix is what makes the transmission non-negative with
    nothing clipping it -- which a tunnelling density built from a smeared delta
    is not (``PLAN.md`` P52, P65).
    """
    calculator = _converged("h-sheet")
    image = run_vertical_transport(
        calculator.system, calculator.pseudos, calculator.get_scf(),
        shape=(4, 4), **SHEET)
    assert image.notes["hermiticity"] < 1.0e-14
    assert image.least_eigenvalue > -1.0e-12
    assert image.image.min() > 0.0


def test_the_band_count_converges():
    """The phase's one convergence parameter, and it is a mild one.

    The on-shell amplitude carries a Gaussian, so a state far from the tip
    energy is suppressed as a Gaussian rather than as ``1/(E - e)``: the map
    settles to a part in 10^4 by twenty bands. That is the whole reason the
    amplitude is the on-shell one -- the literal Landauer denominator does not
    converge at *any* band count (:mod:`defumat.transport.green` measures it).
    """
    calculator = _converged("graphene-monolayer")
    maps = [run_vertical_transport(
        calculator.system, calculator.pseudos, calculator.get_scf(),
        exit_height=0.38, height=0.62, shape=(6, 6), broadening=0.02,
        grid=(6, 6, 1), nbnd=nbnd).image for nbnd in (12, 30)]
    assert abs(maps[1].mean() / maps[0].mean() - 1.0) < 1.0e-4
    assert np.abs(maps[1] / maps[1].mean()
                  - maps[0] / maps[0].mean()).max() < 1.0e-4


# --------------------------------------------------------------------------
# the spin regimes and the polarized substrate
# --------------------------------------------------------------------------


def test_the_three_spin_regimes_agree_where_there_is_no_magnetization(tmp_path):
    """The check that catches a factor of two, and the only one that can.

    A cell with no moment run as ``nspin = 1``, as ``nspin = 2`` and as a spinor
    is the same physics three ways, and the k-weights carry the spin degeneracy
    differently in each (2, then 1 per channel, then 1 with two components).
    Neither the contrast nor the partition can see that factor, both being
    ratios -- this is P51's ``for_spin`` trap in the form P52 records it.
    """
    options = dict(shape=(4, 4), energies=-0.40, **SHEET)
    scalar = _converged("h-sheet")
    a = run_vertical_transport(scalar.system, scalar.pseudos,
                               scalar.get_scf(), **options).image

    collinear = _variant(tmp_path, "h-sheet",
                         "    nspin = 2, starting_magnetization(1) = 0.0,\n")
    b = run_vertical_transport(collinear.system, collinear.pseudos,
                               collinear.get_scf(), **options).image
    assert abs(b.mean() / a.mean() - 1.0) < 1.0e-10

    spinor = _variant(
        tmp_path, "h-sheet",
        "    noncolin = .true., starting_magnetization(1) = 0.0,\n",
        [("nbnd = 8", "nbnd = 16")])
    c = run_vertical_transport(spinor.system, spinor.pseudos,
                               spinor.get_scf(), **options).image
    assert abs(c.mean() / a.mean() - 1.0) < 1.0e-5


@pytest.mark.parametrize("regime,insert,nbnd,pair", [
    ("collinear", "    nspin = 2, starting_magnetization(1) = 0.8,\n", 8,
     ("up", "down")),
    ("spinor", "    noncolin = .true., starting_magnetization(1) = 0.8,"
               " angle1(1) = 90.0,\n", 16, ("x", "-x")),
])
def test_a_polarized_substrate_partitions_the_transmission(
        tmp_path, regime, insert, nbnd, pair):
    """``n`` plus ``-n`` is a substrate that takes both: a partition, exactly.

    One calculation split two ways rather than two calculations, so what it is
    held to is round-off and not physics.
    """
    calculator = _variant(tmp_path, "h-sheet", insert,
                          [("nbnd = 8", f"nbnd = {nbnd}"),
                           ("celldm(1) = 5.0", "celldm(1) = 8.0")])
    options = dict(shape=(3, 3), **SHEET)
    result = calculator.get_scf()
    total = run_vertical_transport(calculator.system, calculator.pseudos,
                                   result, **options).image
    parts = [run_vertical_transport(calculator.system, calculator.pseudos,
                                    result, spin=s, **options).image
             for s in pair]
    assert np.abs(parts[0] + parts[1] - total).max() / total.max() < 1.0e-14
    # and the cell is genuinely magnetic, so the split is not two halves
    assert abs(parts[0].mean() - parts[1].mean()) > 0.1 * total.mean()


def test_a_substrate_across_the_moment_has_no_preference(tmp_path):
    """A magnet with its moment in the plane, and a substrate along ``z``.

    Exactly half the total, because ``n.m = 0``. It is the statement that the
    2x2 acceptance is a projector on a *direction* rather than a channel label,
    and it cannot be said at all without a noncollinear run.
    """
    calculator = _variant(
        tmp_path, "h-sheet",
        "    noncolin = .true., starting_magnetization(1) = 0.8,"
        " angle1(1) = 90.0,\n",
        [("nbnd = 8", "nbnd = 16"), ("celldm(1) = 5.0", "celldm(1) = 8.0")])
    options = dict(shape=(3, 3), **SHEET)
    result = calculator.get_scf()
    total = run_vertical_transport(calculator.system, calculator.pseudos,
                                   result, **options).image
    across = run_vertical_transport(calculator.system, calculator.pseudos,
                                    result, spin="z", **options).image
    assert np.abs(across - 0.5 * total).max() / total.max() < 1.0e-5


# --------------------------------------------------------------------------
# the physics
# --------------------------------------------------------------------------


def test_graphene_transmits_through_one_band_and_is_the_stm_image():
    """The claim the phase was started on, made quantitative.

    At the Fermi level graphene's states are the two Dirac states at ``K``, and
    they are *degenerate partners*: the little group acts irreducibly on the
    pair, so by Schur's lemma the exit-plane overlap restricted to it is a
    multiple of the identity (measured ``diag(0.05405086, 0.05405084)``) and
    there is nothing off its diagonal for the current to interfere through. The
    transmission is then proportional to the local density of states at the tip.
    """
    calculator = _converged("graphene-monolayer")
    result = calculator.get_scf()
    shared = dict(height=0.62, shape=(10, 10), grid=(6, 6, 1), nbnd=20)
    image = run_vertical_transport(calculator.system, calculator.pseudos,
                                   result, exit_height=0.38, broadening=0.02,
                                   **shared)
    stm = run_stm(calculator.system, calculator.pseudos, result,
                  width=0.02, **shared)
    correlation = np.corrcoef(image.image.ravel(),
                              np.asarray(stm.values).ravel())[0, 1]
    assert correlation > 0.9999
    assert np.abs(image.interference).max() / image.image.max() < 1.0e-3


def test_the_bilayer_does_not_reduce_to_its_surface_density_of_states():
    """The other half of the claim, and the reason the quantity exists.

    In an AB bilayer the current has to cross *both* sheets, and the low-energy
    bands are layer-polarized -- a state that is large on the top layer is small
    on the bottom. So the amplitudes that reach the substrate interfere, the map
    stops tracking the surface density of states, and the sublattice contrast an
    STM sees is not the contrast a vertical current sees. Nothing in a
    Tersoff-Hamann image can express that.
    """
    calculator = _converged("graphene-bilayer")
    result = calculator.get_scf()
    shared = dict(height=0.86, shape=(10, 10), grid=(6, 6, 1), nbnd=24)
    image = run_vertical_transport(calculator.system, calculator.pseudos,
                                   result, exit_height=0.14, broadening=0.02,
                                   **shared)
    stm = run_stm(calculator.system, calculator.pseudos, result,
                  width=0.02, **shared)
    correlation = np.corrcoef(image.image.ravel(),
                              np.asarray(stm.values).ravel())[0, 1]
    assert correlation < 0.5
    # and the interference is destructive: the coherent map sits far below the
    # sum of the bands taken one at a time, which is what layer polarization
    # plus a coherent path through both layers does
    assert image.incoherent.mean() > 10.0 * image.image.mean()
    assert image.image.min() > 0.0


# --------------------------------------------------------------------------
# ultrasoft and PAW
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pseudo", ["C.pz-rrkjus.UPF", "Si.pz-n-kjpaw_psl.0.1.UPF"])
def test_an_ultrasoft_or_paw_dataset_works_with_both_planes_in_the_vacuum(
        tmp_path, pseudo):
    """Nothing extra is needed, and where the planes are is the reason.

    In the vacuum a pseudo-wavefunction *is* the true one, so the exit-plane
    overlap wants no augmentation charge -- and both planes of a tunnelling
    geometry are in the vacuum by construction. Inside a sphere they differ,
    which is guarded rather than approximated.
    """
    calculator = _variant(tmp_path, "h-sheet", "", [
        ("H  1.008  H.pz-vbc.UPF", f"X  12.0  {pseudo}"),
        (" H 0.0 0.0 0.5", " X 0.0 0.0 0.5"),
        ("ecutwfc = 15.0", "ecutwfc = 25.0, ecutrho = 200.0"),
        ("celldm(1) = 5.0", "celldm(1) = 5.5"),
    ])
    result = calculator.get_scf()
    image = run_vertical_transport(calculator.system, calculator.pseudos,
                                   result, shape=(4, 4), **SHEET)
    assert image.image.min() > 0.0
    assert image.notes["hermiticity"] < 1.0e-13
    assert image.least_eigenvalue > -1.0e-12

    with pytest.raises(NotImplementedError, match="augmentation sphere"):
        run_vertical_transport(calculator.system, calculator.pseudos, result,
                               exit_height=0.50, height=0.85, shape=(2, 2),
                               broadening=0.05)


# --------------------------------------------------------------------------
# what it refuses
# --------------------------------------------------------------------------


def test_a_k_set_with_two_divisions_along_the_normal_is_refused():
    """Lateral momentum is conserved exactly and the momentum along the normal
    is not: two ``k_perp`` at the same ``k_par`` interfere with a phase that
    depends on where the exit plane sits, and the count of lateral cells changes
    with them. A two-dimensional material is a slab with one k-point along its
    normal, so this is a statement about the input and not a missing term.
    """
    calculator = _converged("h-sheet")
    other = Calculator.from_file(CASES / "h-chain-afm.in", pseudo_dir=PSEUDO)
    with pytest.raises(NotImplementedError, match="divisions along the stacking"):
        run_vertical_transport(other.system, other.pseudos,
                               calculator.get_scf(), exit_height=0.1, height=0.9)


def test_a_symmetry_reduced_k_set_is_refused():
    """A wedge sums to the map symmetrised over the *whole* point group, and
    only the subgroup that leaves the exit plane where it is belongs to this
    geometry: a mirror through the slab exchanges the tip side with the
    substrate side, which is not a symmetry of a tip above a substrate.
    """
    calculator = _converged("h-sheet")
    other = Calculator.from_file(CASES / "graphene-bilayer.in", pseudo_dir=PSEUDO)
    with pytest.raises(NotImplementedError, match="symmetry-reduced"):
        run_vertical_transport(other.system, other.pseudos,
                               calculator.get_scf(), exit_height=0.1, height=0.9)


def test_the_dense_grid_is_built_whole_rather_than_reduced():
    """Which is what makes ``grid=`` usable at all, a wedge being refused."""
    calculator = _converged("graphene-monolayer")
    image = run_vertical_transport(
        calculator.system, calculator.pseudos, calculator.get_scf(),
        exit_height=0.38, height=0.62, shape=(2, 2), broadening=0.02,
        grid=(4, 4, 1), nbnd=12)
    assert image.grid == (4, 4, 1)


def test_a_spin_selective_substrate_needs_something_to_select():
    calculator = _converged("h-sheet")
    with pytest.raises(NotImplementedError, match="needs a magnetization"):
        run_vertical_transport(calculator.system, calculator.pseudos,
                               calculator.get_scf(), shape=(2, 2),
                               spin="up", **SHEET)


def test_the_atoms_have_to_lie_between_the_two_planes():
    """A cell is periodic, so "above" and "below" are relative to the atoms.

    With both planes on the same side the electron tunnels through the vacuum
    and around the periodic image, which is a real number and not this one.
    """
    calculator = _converged("h-sheet")
    with pytest.warns(UserWarning, match="do not lie between"):
        run_vertical_transport(calculator.system, calculator.pseudos,
                               calculator.get_scf(), exit_height=0.60,
                               height=0.80, shape=(2, 2))


def test_the_literal_landauer_denominator_warns_that_it_does_not_converge():
    """It is the exact expression and a band sum cannot evaluate it.

    Kept reachable so that the statement can be measured rather than asserted;
    :mod:`defumat.transport.green` carries the measurement -- a factor of 349
    of cancellation on a cell small enough to diagonalise completely.
    """
    calculator = _converged("h-sheet")
    with pytest.warns(UserWarning, match="cannot evaluate it"):
        run_vertical_transport(calculator.system, calculator.pseudos,
                               calculator.get_scf(), shape=(2, 2),
                               method="resolvent", **SHEET)
