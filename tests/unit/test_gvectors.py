"""Structural properties of the G-vector set and the plane-wave basis."""

import jax
import numpy as np
import pytest

from pypresso.basis.gvectors import generate_gvectors
from pypresso.basis.planewaves import build_plane_wave_basis
from pypresso.system.cell import Cell
from pypresso.system.kpoints import KPoints

pytestmark = pytest.mark.unit

SILICON = Cell.from_ibrav(2, [10.2, 0, 0, 0, 0, 0])


@pytest.fixture(scope="module")
def gvectors():
    return generate_gvectors(SILICON, 48.0)


def test_gamma_is_first_and_shells_are_ordered(gvectors):
    """G = 0 first is relied on wherever a G = 0 term is treated specially
    (the Hartree divergence, the average potential)."""
    assert np.asarray(gvectors.miller[0]).tolist() == [0, 0, 0]
    g2 = np.asarray(gvectors.g2(SILICON))
    assert g2[0] == 0.0
    assert np.all(np.diff(g2) >= -1e-12)


def test_every_g_is_inside_the_cutoff_and_none_is_missing(gvectors):
    """The set is exactly {G : |G|^2 <= gcut} over the representable range."""
    from pypresso.basis.fftgrid import gcut_from_ecut

    gcut = gcut_from_ecut(48.0, SILICON.alat)
    bg = np.asarray(SILICON.bg_2pi_alat)

    assert np.all(np.asarray(gvectors.g2(SILICON)) <= gcut + 1e-12)

    half = [(n - 1) // 2 for n in gvectors.grid]
    ranges = [np.arange(-h, h + 1) for h in half]
    i, j, k = np.meshgrid(*ranges, indexing="ij")
    candidates = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1)
    expected = candidates[np.sum((candidates @ bg) ** 2, axis=1) <= gcut]

    assert gvectors.ngm == len(expected)
    ours = {tuple(m) for m in np.asarray(gvectors.miller)}
    assert ours == {tuple(m) for m in expected}


def test_set_is_closed_under_inversion(gvectors):
    """Without the Gamma trick both G and -G are present, which is what makes a
    real-valued density expressible on this set."""
    ours = {tuple(m) for m in np.asarray(gvectors.miller)}
    assert all((-m[0], -m[1], -m[2]) in ours for m in ours)


def test_fft_indices_are_unique_and_in_range(gvectors):
    index = np.asarray(gvectors.fft_index)
    n1, n2, n3 = gvectors.grid
    assert len(np.unique(index)) == gvectors.ngm
    assert index.min() >= 0 and index.max() < n1 * n2 * n3
    assert index[0] == 0  # G = 0 sits at the box origin


def test_gamma_only_keeps_one_of_each_pair(gvectors):
    half = generate_gvectors(SILICON, 48.0, gamma_only=True)

    assert half.ngm == (gvectors.ngm + 1) // 2
    kept = {tuple(m) for m in np.asarray(half.miller)}
    assert (0, 0, 0) in kept
    for m in kept:
        if m != (0, 0, 0):
            assert (-m[0], -m[1], -m[2]) not in kept, f"both {m} and its inverse kept"
    # Together with their inverses they cover the full sphere.
    assert kept | {(-a, -b, -c) for a, b, c in kept} == {
        tuple(m) for m in np.asarray(gvectors.miller)
    }


def test_cartesian_g_is_differentiable_with_respect_to_the_cell(gvectors):
    """Miller indices are fixed integers; the physical G depends on the cell, so
    a strain derivative has something to flow through (rule D2)."""

    def total(at):
        return float(0.0) + jax.numpy.sum(
            gvectors.kinetic(Cell(at=at, alat=SILICON.alat))[:10]
        )

    gradient = jax.grad(total)(SILICON.at)
    assert gradient.shape == (3, 3)
    assert float(jax.numpy.abs(gradient).max()) > 0.0


def test_kinetic_units_are_rydberg(gvectors):
    """|G|^2 in 1/bohr^2 equals |G|^2 in tpiba^2 times tpiba^2."""
    reduced = np.asarray(gvectors.g2(SILICON))
    rydberg = np.asarray(gvectors.kinetic(SILICON))
    assert rydberg == pytest.approx(reduced * SILICON.tpiba**2)


def test_plane_wave_selection_and_padding():
    kpoints = KPoints.from_cartesian([[0.25, 0.25, 0.25], [0.25, 0.25, 0.75]], [1.0, 3.0])
    gvectors = generate_gvectors(SILICON, 48.0)
    basis = build_plane_wave_basis(gvectors, kpoints, SILICON, 12.0)

    assert basis.npw == (180, 186)
    assert basis.npwx == 186
    assert basis.mask.sum() == sum(basis.npw)
    # Padding must point at a valid index so a gather never goes out of bounds.
    assert int(np.asarray(basis.indices).max()) < gvectors.ngm
    assert np.all(np.asarray(basis.indices[0, 180:]) == 0)

    kinetic = np.asarray(basis.kinetic(gvectors, kpoints, SILICON))
    assert kinetic[np.asarray(basis.mask)].max() <= 12.0 + 1e-8
    assert np.all(kinetic[~np.asarray(basis.mask)] == 0.0)
    # Ordered by |k+G|^2 within each k-point, as gk_sort does.
    for ik, npw in enumerate(basis.npw):
        assert np.all(np.diff(kinetic[ik, :npw]) >= -1e-10)


def test_absurd_cutoff_is_reported():
    kpoints = KPoints.from_cartesian([[0.5, 0.5, 0.5]], [1.0])
    gvectors = generate_gvectors(SILICON, 48.0)
    with pytest.raises(ValueError, match="ecutwfc is far too small"):
        build_plane_wave_basis(gvectors, kpoints, SILICON, 1e-12)
