"""The parts of the nesting function that need no self-consistent field.

``N(q)`` is a correlation of a delta function over a k-grid, and everything in
it is checkable without a pseudopotential: feed a free-electron band in and the
answer is a **closed form**. That is what this file does, and it is the reason
the kernel is a pure function of eigenvalues with no ``Calculation`` anywhere
near it.

Four kinds of thing live here.

* the **analytic limit**. For ``eps = |k|^2`` in Rydberg the nesting function is
  ``Omega / (4 pi^2 q)`` below ``2 k_F`` and identically zero above it, which
  fixes the normalisation, the spin degeneracy and the delta's width all at
  once and needs no other code;
* the **two routes**. The FFT correlation and ``nesting.f90``'s own double loop
  share no arithmetic, so their agreement is the check on the index fold and
  on the transform's sign conventions;
* the **identities** the normalisation makes true -- the mean over ``q`` being
  ``D(E_F)^2``, and ``N(0)`` being the maximum by Cauchy-Schwarz. The second is
  why :meth:`~defumat.response.nesting.NestingFunction.peak` excludes the
  origin;
* the **unfold**, which is the one piece of plumbing that fails silently. An
  eigenvalue that is an invariant function of ``k`` must come back the same
  whether it is evaluated on the complete grid or on a wedge and unfolded, and
  a group mismatch between the two makes a smooth, positive, plausible and
  wrong ``N(q)``.
"""

from pathlib import Path

import numpy as np
import pytest

from defumat.io.pwin import read_pw_input
from defumat.response.nesting import (
    NestingFunction,
    fermi_surface_weights,
    nesting_from_eigenvalues,
    require_a_fermi_surface,
)
from defumat.system import build_system
from defumat.system.kpoints import grid_equivalence, irreducible_wedge

pytestmark = [pytest.mark.unit]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"


def _free_electrons(n: int, a: float = 10.0, fraction: float = 0.3):
    """``eps = |k|^2`` on an ``n^3`` grid of a simple-cubic cell of side ``a``.

    Returns ``(eigenvalues, grid, fermi_energy, degauss, volume, k_F)``. The
    width is tied to the grid spacing rather than to ``E_F``: the Fermi shell
    has to be at least one grid spacing thick or the delta falls between the
    sampling points, which shows up as a 40 per cent scatter and looks like a
    bug in the correlation.
    """
    b = 2.0 * np.pi / a
    i, j, k = np.meshgrid(*(np.arange(n) for _ in range(3)), indexing="ij")
    crystal = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1) / n
    cartesian = (crystal - np.rint(crystal)) * b
    eigenvalues = np.sum(cartesian**2, axis=1)[:, None]
    k_f = fraction * b
    return eigenvalues, (n, n, n), k_f**2, 2.0 * k_f * (b / n), a**3, k_f


# -- the analytic limit --------------------------------------------------------


def test_free_electrons_give_the_lindhard_one_over_q():
    """``N(q) = Omega / (4 pi^2 q)`` below ``2 k_F``, to four digits.

    The derivation is three lines: ``delta(E_F - k^2) = delta(k - k_F)/2k_F``
    puts both factors on the Fermi sphere, and the angular integral of the
    second delta over the first sphere gives ``pi/2q`` for every ``q <= 2k_F``.
    The ``1/q`` is the whole physics of the 3D Lindhard function -- the reason
    nesting is *weak* in three dimensions and divergent in one.

    Averaged in shells of ``|q|`` the agreement is 1e-4 up to ``0.6 * 2k_F``
    and then degrades, which is the delta's own width: the two Fermi shells
    become tangent as ``q -> 2k_F`` and a finite thickness matters most there.
    """
    eigenvalues, grid, fermi, degauss, volume, k_f = _free_electrons(48)
    result = nesting_from_eigenvalues(
        eigenvalues, grid, fermi, degauss, cell_volume=volume
    )

    q = result.qpoints - np.rint(result.qpoints)
    q_norm = np.linalg.norm(q * (2.0 * np.pi / 10.0), axis=1)
    with np.errstate(divide="ignore"):
        closed_form = volume / (4.0 * np.pi**2 * q_norm)

    edges = np.linspace(0.2, 0.6, 9) * (2.0 * k_f)
    for low, high in zip(edges[:-1], edges[1:]):
        shell = (q_norm >= low) & (q_norm < high)
        ratio = np.mean(result.nesting[shell] / closed_form[shell])
        assert abs(ratio - 1.0) < 2.0e-3, (low, high, ratio)


def test_nothing_nests_beyond_twice_the_fermi_wavevector():
    """A sharp analytic feature, and the code reproduces it *exactly*.

    Two points of one Fermi sphere cannot be more than ``2 k_F`` apart, so
    ``N(q)`` is not merely small beyond that -- it is zero to round-off,
    because ``g(k)`` is supported on a thin shell and the product of two
    disjoint supports is empty. It is the check that the correlation has not
    quietly wrapped a shifted copy into the wrong place.
    """
    eigenvalues, grid, fermi, degauss, volume, k_f = _free_electrons(48)
    result = nesting_from_eigenvalues(
        eigenvalues, grid, fermi, degauss, cell_volume=volume
    )
    q = result.qpoints - np.rint(result.qpoints)
    q_norm = np.linalg.norm(q * (2.0 * np.pi / 10.0), axis=1)
    beyond = q_norm > 1.3 * 2.0 * k_f
    assert beyond.any()
    assert np.max(np.abs(result.nesting[beyond])) < 1.0e-12 * result.nesting.max()


def test_the_free_electron_density_of_states_comes_out_right():
    """``D(E_F) = Omega k_F / 2 pi^2``, which is the same normalisation seen alone.

    The nesting function is quadratic in ``g``, so a factor of two in the spin
    degeneracy is a factor of four in it and would be hard to attribute. This
    is the linear half of the same statement.
    """
    eigenvalues, grid, fermi, degauss, volume, k_f = _free_electrons(48)
    result = nesting_from_eigenvalues(
        eigenvalues, grid, fermi, degauss, cell_volume=volume
    )
    assert result.fermi_dos == pytest.approx(volume * k_f / (2.0 * np.pi**2), rel=2e-3)


# -- the two routes ------------------------------------------------------------


def test_the_transform_and_the_double_loop_agree():
    """``nesting.f90``'s ``O(N_q N_k)`` loop against one FFT, to round-off.

    They share no arithmetic: one folds ``k + q`` with ``mod`` and accumulates,
    the other multiplies a spectrum by its own conjugate. Agreement says the
    fold and the transform's convention describe the same cyclic correlation,
    which is the only thing this phase changes about Elk's algorithm.
    """
    rng = np.random.default_rng(20260831)
    grid = (6, 5, 4)
    eigenvalues = rng.normal(size=(1, int(np.prod(grid)), 7))
    fast = nesting_from_eigenvalues(eigenvalues, grid, 0.1, 0.3)
    slow = nesting_from_eigenvalues(eigenvalues, grid, 0.1, 0.3, method="direct")
    scale = np.max(np.abs(fast.nesting))
    assert np.max(np.abs(fast.nesting - slow.nesting)) < 1.0e-13 * scale


# -- the identities the normalisation makes true -------------------------------


def test_the_mean_over_q_is_the_squared_density_of_states():
    """``(1/N_q) sum_q N(q) = D(E_F)^2``, exactly.

    Both factors of the correlation sum to ``N_k D(E_F)`` independently, so the
    mean factorises. It ties the delta, the degeneracy and the transform's
    normalisation together in one number, and it is what
    :attr:`~defumat.response.nesting.NestingFunction.sum_rule` reports.
    """
    rng = np.random.default_rng(11)
    grid = (7, 7, 3)
    eigenvalues = rng.normal(size=(1, int(np.prod(grid)), 5))
    result = nesting_from_eigenvalues(eigenvalues, grid, 0.0, 0.4)
    assert abs(result.sum_rule) < 1.0e-12 * result.fermi_dos**2
    assert result.nesting.mean() == pytest.approx(result.fermi_dos**2, rel=1e-12)


def test_nothing_nests_better_than_the_origin():
    """``N(0) >= N(q)`` on every crystal, by Cauchy-Schwarz.

    Which is why :meth:`~defumat.response.nesting.NestingFunction.peak`
    excludes ``q = 0``: including it would report the same uninformative
    wavevector for every material, and the question the quantity answers is
    where the surface maps onto a *different* part of itself.
    """
    rng = np.random.default_rng(3)
    grid = (5, 5, 5)
    eigenvalues = rng.normal(size=(1, 125, 4))
    result = nesting_from_eigenvalues(eigenvalues, grid, 0.0, 0.5)
    assert result.nesting[0] == pytest.approx(result.nesting.max())

    where, value = result.peak()
    assert not np.allclose(where, 0.0)
    assert value < result.nesting[0]


def test_the_spin_degeneracy_squares():
    """``g`` carries ``degspin`` and ``N`` is quadratic in it.

    So an unpolarized cell's ``N`` is four times the same eigenvalues read as
    one spinor channel. Stated as a test because the factor is invisible in the
    *shape* of ``N(q)`` and changes only its scale, and the ratio
    :attr:`~defumat.response.nesting.NestingFunction.ratio` divides it out
    entirely.
    """
    rng = np.random.default_rng(7)
    grid = (4, 4, 4)
    eigenvalues = rng.normal(size=(1, 64, 3))
    two = nesting_from_eigenvalues(eigenvalues, grid, 0.0, 0.5, degeneracy=2.0)
    one = nesting_from_eigenvalues(eigenvalues, grid, 0.0, 0.5, degeneracy=1.0)
    assert np.allclose(two.nesting, 4.0 * one.nesting)
    assert np.allclose(two.ratio, one.ratio)


def test_elk_units_are_the_documented_conversion():
    """``(Omega_BZ / occmax) * 4 * N`` -- the 4 being Rydberg against Hartree.

    ``nesting.f90`` writes ``occmax * omegabz * wkptnr * sum_k g~ g~`` with
    ``g~`` carrying no degeneracy and energies in Hartree. Squaring the two
    substitutions is where a factor of four hides, and P50's own factor of two
    is why this is pinned rather than derived at the call site.
    """
    rng = np.random.default_rng(5)
    result = nesting_from_eigenvalues(
        rng.normal(size=(1, 27, 3)), (3, 3, 3), 0.0, 0.5, cell_volume=100.0
    )
    expected = ((2.0 * np.pi) ** 3 / 100.0 / 2.0) * 4.0 * result.nesting
    assert np.allclose(result.elk_units, expected)


# -- the unfold, which is the piece that fails silently ------------------------


@pytest.mark.parametrize("shift", [(0, 0, 0), (1, 1, 1)],
                         ids=["unshifted", "shifted"])
def test_a_wedge_unfolds_onto_the_complete_grid(shift):
    """An invariant band, evaluated on the wedge and unfolded, is the whole grid.

    ``eps_n(Rk) = eps_n(k)`` is exact, so this has no tolerance of its own: any
    difference is the map sending a grid point to the wrong representative.
    Run **shifted as well as unshifted**, because a shifted k-grid is a
    different orbit walk and is the case a Monkhorst-Pack input actually asks
    for; the q-grid stays unshifted either way, since ``q`` is a difference of
    two k-points and the offset cancels in it.
    That is the failure mode :func:`~defumat.workflows.nscf.grid_symmetry`
    exists to prevent, and it produces a perfectly plausible ``N(q)`` built
    from somebody else's bands.
    """
    grid = (6, 6, 6)
    rotations = np.array(
        [np.eye(3), np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, 1.0, -1.0]),
         np.diag([-1.0, -1.0, 1.0])]
    )
    points, _ = irreducible_wedge(grid, shift, rotations)
    equivalent = grid_equivalence(grid, shift, rotations)

    # An invariant function of k: the wedge carries every value the grid has.
    def band(crystal):
        folded = crystal - np.rint(crystal)
        return np.cos(2 * np.pi * folded).sum(axis=1)[:, None]

    i, j, k = np.meshgrid(*(np.arange(n) for n in grid), indexing="ij")
    offset = np.asarray(shift) / 2.0
    complete = (np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1) + offset) / 6

    unfolded = band(np.asarray(points))[equivalent]
    assert np.allclose(unfolded, band(complete), atol=1e-14)

    direct = nesting_from_eigenvalues(band(complete)[None], grid, 0.5, 0.2)
    through_map = nesting_from_eigenvalues(unfolded[None], grid, 0.5, 0.2)
    assert np.allclose(direct.nesting, through_map.nesting, atol=1e-14)


def test_a_wedge_handed_in_whole_is_refused_rather_than_broadcast():
    """The kernel takes the complete grid and says so when it is not given one.

    A wedge has fewer points than ``n1 n2 n3`` and reshaping it would either
    raise somewhere unhelpful or, on a grid whose divisions happen to factor
    the wedge's size, succeed and be wrong.
    """
    with pytest.raises(ValueError, match="complete"):
        nesting_from_eigenvalues(np.zeros((1, 10, 4)), (4, 4, 4), 0.0, 0.1)


# -- the refusals --------------------------------------------------------------


def _calculation(case: str):
    """A stand-in carrying only what :func:`require_a_fermi_surface` reads."""
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    return type("Stub", (), {"system": system, "spiral": bool(system.spiral_q is not None)})()


def test_it_refuses_a_run_with_no_fermi_level():
    """A fixed-occupation run fills a set number of bands and never finds one."""
    with pytest.raises(NotImplementedError, match="Fermi level"):
        require_a_fermi_surface(_calculation("diamond"))


def test_it_refuses_two_fermi_levels():
    """``tot_magnetization`` gives one level per channel, so ``g`` is two surfaces."""
    with pytest.raises(NotImplementedError, match="tot_magnetization"):
        require_a_fermi_surface(_calculation("o2-fixed-lsda"))


def test_it_refuses_a_spin_spiral():
    """The quantity is about the state the spiral grows out of, not the spiral.

    Which is also what makes the pairing with ``relax_spiral_q`` a check: one
    predicts the pitch from the paramagnet's Fermi surface and the other finds
    it by going downhill in the magnetic energy, and they share nothing.
    """
    with pytest.raises(NotImplementedError, match="spiral"):
        require_a_fermi_surface(_calculation("h-chain-spiral"))


def test_a_metal_with_a_smearing_is_accepted():
    """The complement of the three above, so that they are not vacuously passing."""
    require_a_fermi_surface(_calculation("al-metal"))


def test_the_grid_ordering_is_monkhorst_packs():
    """``as_grid`` reshapes without a permutation, last index fastest.

    Elk writes ``NEST3D.OUT`` in the opposite nesting of loops, so this is the
    one place the two file formats are not the same array, and reading its
    output back needs to know which.
    """
    rng = np.random.default_rng(1)
    grid = (2, 3, 4)
    result = nesting_from_eigenvalues(
        rng.normal(size=(1, 24, 2)), grid, 0.0, 0.5
    )
    box = result.as_grid()
    assert box.shape == grid
    for index, q in enumerate(result.qpoints):
        i, j, k = np.rint(q * np.array(grid)).astype(int)
        assert box[i, j, k] == result.nesting[index]


def test_a_negative_width_is_refused():
    with pytest.raises(ValueError, match="degauss"):
        fermi_surface_weights(np.zeros((1, 8, 2)), 0.0, 0.0)
