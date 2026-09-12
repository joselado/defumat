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
* the **spin partition**: a lead polarized along ``n`` plus one along ``-n`` is
  a lead that takes both, exactly -- for the substrate and for the tip alike;
* the **physics**, which is the phase's reason for existing: monolayer graphene's
  Dirac pair is degenerate, so the substrate cannot tell its two members apart and
  its transmission *is* the STM image, while an AB bilayer's current has to cross
  both layers and its is not.

The algebra -- the Gram matrix, the contraction, the amplitude weights, the
sampler -- is in ``tests/unit/test_transport_machinery.py`` and needs no SCF.
"""

import tempfile
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

#: The magnetic hydrogen sheet as a collinear run; the spinor one is committed
#: as ``h-sheet-noncolin.in``, whose moment is put in a **generic** direction
#: rather than along an axis. With the moment along ``x`` every ``m_y`` in the
#: answer is zero, and the transposed tip-spin contraction -- which is the
#: answer for a tip at ``(n_x, -n_y, n_z)`` -- would agree with the right one
#: exactly. A test that cannot fail is not a test.
MAGNETS = {
    "collinear": ("    nspin = 2, starting_magnetization(1) = 0.8,\n", 8),
}


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


@lru_cache(maxsize=2)
def _magnet(kind: str):
    """The magnetic hydrogen sheet, converged once and shared.

    ``lru_cache(maxsize=2)`` rather than ``None``: two is what a comparison
    between the two spin regimes needs and is the largest that is not a leak
    (``CLAUDE.md``, memory).
    """
    if kind == "spinor":
        # committed, because the notebook needs it too and because the generic
        # moment direction is load-bearing rather than incidental
        calculator = Calculator.from_file(CASES / "h-sheet-noncolin.in",
                                          pseudo_dir=PSEUDO)
        calculator.get_scf()
        return calculator
    insert, nbnd = MAGNETS[kind]
    return _variant(Path(tempfile.mkdtemp()), "h-sheet", insert,
                    [("nbnd = 8", f"nbnd = {nbnd}"),
                     ("celldm(1) = 5.0", "celldm(1) = 8.0")])


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


def test_the_tip_amplitudes_are_bounded_by_the_k_dial():
    """The assembly's host array is ``(npol, nk, nbnd, npoints)`` and the dial
    has to reach it.

    This workflow's largest allocation is not on the device and is not the
    wavefunctions: it is the tip amplitudes, one complex number per spinor
    component, k-point, band and *pixel*. An image is thousands of pixels, so
    the array grows with the product of two things a user turns up for a better
    picture, and it sat outside every dial.

    Two assertions, because a chunked sum has to be both smaller and the same
    answer: the largest array the assembly asks numpy for scales with the chunk
    rather than with ``nk``, and the map is unchanged by it. The sum over k
    ends in ``kweights @ term`` in every branch, so a chunk moves nothing but
    the order the contributions are added in.
    """
    calculator = _converged("h-sheet")
    nk = len(calculator.system.kpoints.weights)
    assert nk > 1, "a single k-point cannot show a chunking"

    def run(k_batch):
        biggest = 0
        empty = np.empty

        def spy(shape, *args, **kwargs):
            nonlocal biggest
            size = int(np.prod(shape)) if isinstance(shape, tuple) else int(shape)
            biggest = max(biggest, size)
            return empty(shape, *args, **kwargs)

        np.empty = spy
        try:
            image = run_vertical_transport(
                calculator.system, calculator.pseudos, calculator.get_scf(),
                shape=(8, 8), k_batch=k_batch, **SHEET)
        finally:
            np.empty = empty
        return image, biggest

    one, small = run(1)
    whole, large = run(None)

    # 8 bands x 64 pixels a k-point: the whole axis is nk times that, and the
    # chunked run never asks for more than one k-point's worth.
    assert small < 2 * 8 * 8 * 8
    assert large > (nk // 2) * 8 * 8 * 8
    assert np.abs(one.image - whole.image).max() / whole.image.max() < 1.0e-13
    assert np.abs(one.incoherent - whole.incoherent).max() \
        / whole.incoherent.max() < 1.0e-13
    assert one.least_eigenvalue == pytest.approx(whole.least_eigenvalue, rel=1e-12)
    assert one.notes["channels"] == pytest.approx(whole.notes["channels"], rel=1e-12)
    assert one.notes["band_edge_weight"] == pytest.approx(
        whole.notes["band_edge_weight"], rel=1e-10)


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
# the polarized tip
# --------------------------------------------------------------------------


def test_a_magnetic_tip_in_the_whole_cell_limit_is_the_spin_polarized_stm_image():
    """The strongest check the phase has, now with both spins kept coherently.

    Widen the substrate from a plane to the whole cell and the Gram matrix is
    the identity by orthonormality, so ``T(r) = sum_kn w_k delta(E - e)
    v_n^dagger P_t v_n`` with ``v_n`` the spinor amplitude -- which is
    ``[rho + P n.m]/2``, the spin-polarized tunnelling density of states
    :func:`run_stm` returns. Exactly, with no factor, through a completely
    different code path: this one contracts a 2x2 matrix built from sampled
    amplitudes, the other builds a four-component density on the FFT grid and
    projects it.

    **The tip direction and the sample's moment both have a ``y`` component on
    purpose.** The transposed index order returns the answer for a tip at
    ``(n_x, -n_y, n_z)``, which here differs by a factor of two, and on a moment
    along an axis would not differ at all.
    """
    calculator = _magnet("spinor")
    result = calculator.get_scf()
    geometry = dict(height=0.80, shape=(6, 6))
    direction, p = (1.0, 1.0, 1.0), 0.85
    ours = run_vertical_transport(
        calculator.system, calculator.pseudos, result, exit_height=0.20,
        broadening=0.02, exit_region="volume", tip_spin=direction,
        tip_polarization=p, **geometry)
    reference = np.asarray(run_stm(
        calculator.system, calculator.pseudos, result, width=0.02,
        spin=direction, polarization=p, **geometry).values)
    assert np.abs(ours.image - reference).max() / np.abs(reference).max() < 1e-11

    # and the mirrored tip -- what a transposed contraction would have given --
    # is a different image entirely, so the check above has something to catch
    mirrored = np.asarray(run_stm(
        calculator.system, calculator.pseudos, result, width=0.02,
        spin=(1.0, -1.0, 1.0), polarization=p, **geometry).values)
    assert np.abs(mirrored - reference).max() / np.abs(reference).max() > 1.0


@pytest.mark.parametrize("kind,pair", [
    ("collinear", ("up", "down")),
    ("spinor", ((0.3, 0.8, 0.5), (-0.3, -0.8, -0.5))),
])
def test_a_polarized_tip_partitions_the_transmission(kind, pair):
    """``P_t(n, P) + P_t(-n, P) = 1``, so the two tips add back to the map a
    nonmagnetic one draws. One calculation split two ways, so what it is held to
    is round-off. The substrate is polarized at the same time, along a third
    direction: the two polarizers are independent and the identity is the tip's
    alone.
    """
    calculator = _magnet(kind)
    result = calculator.get_scf()
    substrate = dict(spin="up" if kind == "collinear" else (0.0, 0.0, 1.0),
                     polarization=0.6)
    options = dict(shape=(3, 3), **SHEET, **substrate)
    total = run_vertical_transport(calculator.system, calculator.pseudos,
                                   result, **options).image
    parts = [run_vertical_transport(calculator.system, calculator.pseudos,
                                    result, tip_spin=n, tip_polarization=0.9,
                                    **options).image for n in pair]
    assert np.abs(parts[0] + parts[1] - total).max() / total.max() < 1.0e-14
    # a nonmagnetic tip is exactly half, which is P65's convention
    half = run_vertical_transport(calculator.system, calculator.pseudos, result,
                                  tip_spin=pair[0], tip_polarization=0.0,
                                  **options).image
    assert np.abs(half - 0.5 * total).max() / total.max() < 1.0e-14
    # and the split is a real preference rather than two halves
    assert abs(parts[0].mean() - parts[1].mean()) > 0.1 * total.mean()
    assert parts[0].min() > 0.0 and parts[1].min() > 0.0


def test_the_two_moments_together_are_a_tunnelling_magnetoresistance_map():
    """The capability that needed both polarizers: the angle between them.

    A magnetic tip over a magnetic sheet on a magnetic substrate is a spin
    valve, and what it measures is the relative orientation of the two
    electrodes. Parallel and antiparallel must differ, and a tip at ninety
    degrees to the substrate must sit between them -- which is the statement
    that the answer depends on ``cos`` of the angle and not on the two
    directions separately.
    """
    calculator = _magnet("spinor")
    result = calculator.get_scf()
    options = dict(shape=(3, 3), **SHEET, spin=(1.0, 0.0, 0.0),
                   polarization=0.9)

    def at(direction):
        return run_vertical_transport(
            calculator.system, calculator.pseudos, result, tip_spin=direction,
            tip_polarization=0.9, **options).image.mean()

    parallel = at((1.0, 0.0, 0.0))
    antiparallel = at((-1.0, 0.0, 0.0))
    perpendicular = at((0.0, 0.0, 1.0))
    ratio = parallel / antiparallel
    assert ratio > 1.05 or ratio < 0.95, ratio
    assert min(parallel, antiparallel) < perpendicular < max(parallel,
                                                             antiparallel)
    # the perpendicular tip is the mean of the two, exactly: the acceptance is
    # linear in n and the two parallel/antiparallel projectors average to it
    assert abs(perpendicular - 0.5 * (parallel + antiparallel)) < 1e-4 * parallel


def test_a_collinear_tip_and_substrate_multiply():
    """Two spin filters in series, and on a collinear run that is exact.

    Spin is conserved through the junction there, so the two channels are two
    independent calculations and each polarizer is a weight ``(1 +- P)/2`` on
    them. The product is not an approximation to a 2x2 contraction; it is what
    the 2x2 contraction becomes when everything is diagonal.
    """
    calculator = _magnet("collinear")
    result = calculator.get_scf()
    options = dict(shape=(3, 3), **SHEET)

    def at(**extra):
        return run_vertical_transport(calculator.system, calculator.pseudos,
                                      result, **options, **extra).image

    up_up = at(spin="up", tip_spin="up")
    # a fully polarized tip takes one channel and a fully polarized substrate
    # takes one channel: with the two opposed, nothing gets through -- and the
    # zero says so by name rather than blaming the k-set for it
    with pytest.warns(UserWarning, match="leave no channel open"):
        up_down = at(spin="up", tip_spin="down")
    assert np.abs(up_down).max() == 0.0
    assert up_up.min() > 0.0
    # and the channel the two agree on is the substrate's own answer
    assert np.abs(up_up - at(spin="up")).max() / up_up.max() < 1.0e-14


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


def test_a_polarized_tip_needs_something_to_couple_to():
    """The same statement about the other lead: with no magnetization every
    direction takes half of everything, which is the charge map again."""
    calculator = _converged("h-sheet")
    with pytest.raises(NotImplementedError, match="tip needs a magnetization"):
        run_vertical_transport(calculator.system, calculator.pseudos,
                               calculator.get_scf(), shape=(2, 2),
                               tip_spin="up", **SHEET)


def test_a_transverse_tip_on_a_collinear_run_is_refused():
    """``m_x`` and ``m_y`` are absent there rather than zero, so projecting on
    a direction off the ``z`` axis would be a statement the calculation cannot
    make -- P65's reasoning, applied to the tip of this geometry."""
    calculator = _magnet("collinear")
    with pytest.raises(NotImplementedError, match="transverse component"):
        run_vertical_transport(calculator.system, calculator.pseudos,
                               calculator.get_scf(), shape=(2, 2),
                               tip_spin=(1.0, 0.0, 0.0), **SHEET)


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
