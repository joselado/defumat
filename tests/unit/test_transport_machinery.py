"""The vertical-transport machinery, on objects rather than on converged cells.

``PLAN.md`` P66. What needs an SCF is in ``tests/regression/test_transport.py``;
what is here is the algebra: the exit-plane Gram matrix, the contraction, the
amplitude weights, the tip sampler, and the refusals.

The three statements worth reading first, because they are what the whole
construction rests on:

* ``S_k`` is a **Gram matrix** -- Hermitian and positive semi-definite -- so the
  transmission built from it cannot be negative;
* summed over the exit plane's own coordinate it is the **identity**, which is
  orthonormality and is the only check on the closed-form ``h3`` collapse that
  does not go through a wavefunction;
* the contraction is **invariant under a rotation inside a degenerate
  multiplet**, which is rule D4 satisfied by construction rather than by
  handling degeneracies.
"""

import numpy as np
import pytest

from defumat.basis.sample import sample_wavefunctions
from defumat.transport.green import (
    amplitude_weights,
    channel_basis,
    transmission,
)
from defumat.transport.substrate import (
    exit_overlap,
    spin_projector,
    surface_area,
    volume_overlap,
)


class _Cell:
    """The two things :mod:`defumat.transport.substrate` asks a cell for."""

    def __init__(self, at):
        self.at = np.asarray(at, dtype=float)
        self.volume = float(abs(np.linalg.det(self.at)))


def _orthonormal_bands(miller, nbnd, seed=0):
    """``nbnd`` orthonormal coefficient vectors on a sphere of Miller indices."""
    rng = np.random.default_rng(seed)
    npw = miller.shape[0]
    raw = rng.normal(size=(nbnd, npw)) + 1.0j * rng.normal(size=(nbnd, npw))
    q, _ = np.linalg.qr(raw.T)
    return q.T[:nbnd]


def _sphere(n=3):
    grid = np.arange(-n, n + 1)
    h = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1)
    return h.reshape((-1, 3))


CUBIC = _Cell(np.diag([4.0, 5.0, 9.0]))
HEXAGONAL = _Cell([[4.0, 0.0, 0.0], [-2.0, 3.4641016151377544, 0.0], [0.0, 0.0, 12.0]])


# --------------------------------------------------------------------------
# the exit plane
# --------------------------------------------------------------------------


def test_the_surface_area_is_the_cross_product_and_not_the_volume_over_a_length():
    """``|a1 x a2|``, which for a hexagonal cell is not ``Omega / |a3|``."""
    assert surface_area(CUBIC, 2) == pytest.approx(20.0)
    assert surface_area(HEXAGONAL, 2) == pytest.approx(13.856406460551018)
    # the trap: dividing the volume by the wrong length agrees for a cubic cell
    assert surface_area(CUBIC, 2) == pytest.approx(CUBIC.volume / 9.0)


@pytest.mark.parametrize("cell", [CUBIC, HEXAGONAL])
def test_the_exit_overlap_is_hermitian_and_positive_semidefinite(cell):
    """It is a Gram matrix, and that is what makes the transmission positive."""
    miller = _sphere(2)
    bands = _orthonormal_bands(miller, 6)
    matrix = exit_overlap(bands, miller, 0.31, 2, cell)
    assert np.abs(matrix - matrix.conj().T).max() < 1.0e-14
    assert np.linalg.eigvalsh(matrix).min() > -1.0e-14


def test_the_exit_overlap_integrates_to_the_identity_along_the_normal():
    """Sweep the plane through the cell and orthonormality comes back.

    The one check on the closed-form ``sum_h3 c e^{2 pi i h3 s3}`` collapse that
    involves no wavefunction and no sampling: the ``s3`` integral of
    ``e^{2 pi i (h3' - h3) s3}`` is ``delta_{h3 h3'}``, which restores the full
    G-sum. A midpoint rule on ``4 max|h3| + 4`` points is *exact* for it, the
    integrand being a trigonometric polynomial.
    """
    miller = _sphere(3)
    bands = _orthonormal_bands(miller, 5, seed=1)
    steps = 4 * int(np.abs(miller[:, 2]).max()) + 4
    swept = sum(exit_overlap(bands, miller, s3, 2, HEXAGONAL)
                for s3 in np.arange(steps) / steps) / steps
    scale = HEXAGONAL.volume / surface_area(HEXAGONAL, 2)
    assert np.abs(scale * swept - np.eye(5)).max() < 1.0e-13


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_the_sum_rule_holds_on_every_axis(axis):
    """A stacking axis is an argument, not an assumption about which one it is."""
    miller = _sphere(2)
    bands = _orthonormal_bands(miller, 4, seed=2)
    steps = 4 * int(np.abs(miller[:, axis]).max()) + 4
    swept = sum(exit_overlap(bands, miller, s, axis, HEXAGONAL)
                for s in np.arange(steps) / steps) / steps
    scale = HEXAGONAL.volume / surface_area(HEXAGONAL, axis)
    assert np.abs(scale * swept - np.eye(4)).max() < 1.0e-13


def test_the_volume_overlap_is_the_identity_for_orthonormal_bands():
    """Which is the whole content of the Tersoff-Hamann diagnostic."""
    miller = _sphere(2)
    bands = _orthonormal_bands(miller, 7, seed=3)
    assert np.abs(volume_overlap(bands) - np.eye(7)).max() < 1.0e-13


def test_the_volume_overlap_applies_a_given_metric():
    """An ultrasoft dataset's orthonormality is ``<psi|S|psi>``, not ``sum c* c``.

    Without it the diagnostic is short of the augmentation charge -- 2 per cent
    on an ultrasoft carbon sheet -- which reads exactly like an assembly error.
    """
    miller = _sphere(1)
    bands = _orthonormal_bands(miller, 3, seed=4)
    metric = np.diag(np.linspace(1.0, 2.0, miller.shape[0]))
    matrix = volume_overlap(bands, overlap=lambda p: p @ metric.T)
    expected = bands.conj() @ metric.T @ bands.T
    assert np.abs(matrix - expected).max() < 1.0e-13
    assert np.abs(matrix - np.eye(3)).max() > 0.1  # the metric actually did something


def test_a_padded_plane_wave_is_dropped_rather_than_trusted_to_be_zero():
    """Padding points at ``G = 0``, so a nonzero coefficient there would alias."""
    miller = _sphere(1)
    bands = _orthonormal_bands(miller, 3, seed=5)
    mask = np.ones(miller.shape[0], dtype=bool)
    mask[-4:] = False
    poisoned = bands.copy()
    poisoned[:, -4:] = 7.0 + 3.0j
    assert np.abs(exit_overlap(poisoned, miller, 0.2, 2, CUBIC, mask=mask)
                  - exit_overlap(bands, miller, 0.2, 2, CUBIC, mask=mask)).max() < 1e-14


def test_the_spin_projector_is_the_stm_tip_one_level_down():
    """``(1 + P n.sigma)/2``: a projector at ``P = 1``, the mean at ``P = 0``."""
    up = spin_projector("z", 1.0)
    assert np.abs(up - np.array([[1.0, 0.0], [0.0, 0.0]])).max() < 1.0e-14
    assert np.abs(up @ up - up).max() < 1.0e-14
    assert np.abs(spin_projector("x", 1.0) + spin_projector("-x", 1.0)
                  - np.eye(2)).max() < 1.0e-14
    assert np.abs(spin_projector("y", 0.0) - 0.5 * np.eye(2)).max() < 1.0e-14
    with pytest.raises(ValueError, match=r"polarization must be in"):
        spin_projector("z", 1.5)


# --------------------------------------------------------------------------
# the amplitude weights
# --------------------------------------------------------------------------


def test_the_on_shell_amplitude_squares_to_the_stm_weight():
    """``|a|^2 = delta(E - e)/eta``, which is P65's tunnelling weight exactly.

    It is what makes the whole-cell exit region reproduce ``run_stm`` with no
    factor between them rather than up to one.
    """
    from defumat.stm.image import smeared_delta

    eps = np.array([[-0.2, 0.0, 0.15, 1.0]])
    a = amplitude_weights(eps, 0.0, 0.05, "spectral", "gaussian")
    expected = smeared_delta(-eps / 0.05, "gaussian") / 0.05
    assert np.abs(np.abs(a) ** 2 - expected).max() < 1.0e-14
    assert np.abs(a.imag).max() == 0.0


def test_a_delta_that_goes_negative_has_no_square_root():
    """P52's objection, one order sharper: an *amplitude* is being taken."""
    eps = np.linspace(-1.0, 1.0, 40)[None]
    with pytest.raises(ValueError, match="negative on its wings"):
        amplitude_weights(eps, 0.0, 0.05, "spectral", "marzari-vanderbilt")
    amplitude_weights(eps, 0.0, 0.05, "spectral", "fermi-dirac")  # positive: fine


def test_the_resolvent_is_the_literal_denominator():
    eps = np.array([[0.1, 0.4]])
    a = amplitude_weights(eps, 0.2, 0.01, "resolvent")
    assert a[0, 0] == pytest.approx(1.0 / (0.1 + 0.01j))


def test_an_unknown_method_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown method"):
        amplitude_weights(np.zeros((1, 2)), 0.0, 0.1, "bardeen")


# --------------------------------------------------------------------------
# the contraction
# --------------------------------------------------------------------------


def _random_case(nk=3, nbnd=4, npts=5, seed=7):
    rng = np.random.default_rng(seed)
    amplitudes = (rng.normal(size=(nk, nbnd, npts))
                  + 1.0j * rng.normal(size=(nk, nbnd, npts)))
    raw = rng.normal(size=(nk, nbnd, nbnd)) + 1.0j * rng.normal(size=(nk, nbnd, nbnd))
    overlaps = np.einsum("kij,klj->kil", raw, raw.conj())  # Gram, so PSD
    kweights = rng.uniform(0.1, 1.0, size=nk)
    weights = rng.uniform(0.1, 1.0, size=(nk, nbnd)).astype(complex)
    return amplitudes, overlaps, kweights, weights


def test_the_transmission_is_never_negative():
    """Structural: every k-term is ``a^dagger S a`` with ``S`` positive."""
    amplitudes, overlaps, kweights, weights = _random_case()
    assert transmission(amplitudes, overlaps, kweights, weights).min() >= 0.0


def test_the_transmission_is_the_plane_integral_of_the_squared_amplitude():
    """The contraction written the other way round: ``int |sum_n a_n psi*_n|^2``.

    Two index orders differ by a transpose of ``S`` and by nothing else, and
    since ``S`` is Hermitian the wrong one is real, positive and plausible.
    This pins which one it is.
    """
    amplitudes, overlaps, kweights, weights = _random_case(nk=1, nbnd=3, npts=2)
    a = amplitudes[0] * weights[0][:, None]
    brute = np.real(np.einsum("np,mp,nm->p", a, a.conj(), overlaps[0]))
    assert np.abs(transmission(amplitudes, overlaps, kweights, weights)
                  - kweights[0] * brute).max() < 1.0e-12


def test_the_contraction_is_blind_to_a_rotation_inside_a_degenerate_multiplet():
    """Rule D4, satisfied by construction rather than by handling degeneracies.

    A degenerate multiplet's eigenvectors are arbitrary up to a unitary mixing.
    Here ``a`` and ``S`` are both covariant under it and the weight is the same
    number for every member, so the quadratic form cannot see it -- which is
    what the diagonal-of-an-operator constructions of P51 and P54 could not say.
    """
    rng = np.random.default_rng(11)
    amplitudes, overlaps, kweights, weights = _random_case(nk=2, nbnd=4, npts=3)
    weights = np.ones_like(weights)  # one degenerate multiplet: equal weights
    raw = rng.normal(size=(4, 4)) + 1.0j * rng.normal(size=(4, 4))
    u, _ = np.linalg.qr(raw)
    # The mixed bands are ``psi'_i = sum_n U_ni psi_n``, so the amplitudes take
    # ``U^T`` and the overlap takes ``U^dagger . U``. Transforming the two
    # inconsistently is the easiest way to write a test that fails on correct
    # code, and it did.
    mixed_a = np.einsum("ni,knp->kip", u, amplitudes)
    mixed_s = np.einsum("ni,knm,mj->kij", u.conj(), overlaps, u)
    assert np.abs(transmission(mixed_a, mixed_s, kweights, weights)
                  - transmission(amplitudes, overlaps, kweights, weights)).max() < 1e-12


def test_the_incoherent_map_is_blind_to_a_degenerate_rotation_too():
    """The same invariance as the coherent map, and it needs work to get.

    ``T_coh`` is a quadratic form and is invariant for free. ``T_incoh`` is a
    **diagonal**, which is exactly what is not invariant under the rotation a
    degenerate eigensolver is free in -- rule D4 arriving in a diagnostic
    rather than in an answer. Diagonalising ``S_k`` inside each multiplet first
    is what restores it, and without ``eigenvalues=`` the same call moves.
    """
    rng = np.random.default_rng(31)
    amplitudes, overlaps, kweights, weights = _random_case(nk=2, nbnd=4, npts=3)
    weights = np.ones_like(weights)
    eigenvalues = np.tile(np.array([0.0, 0.0, 1.0, 2.0]), (2, 1))  # one pair
    raw = rng.normal(size=(2, 2)) + 1.0j * rng.normal(size=(2, 2))
    u, _ = np.linalg.qr(raw)
    full = np.eye(4, dtype=complex)
    full[:2, :2] = u
    mixed_a = np.einsum("ni,knp->kip", full, amplitudes)
    mixed_s = np.einsum("ni,knm,mj->kij", full.conj(), overlaps, full)

    fixed = dict(coherent=False, eigenvalues=eigenvalues)
    before = transmission(amplitudes, overlaps, kweights, weights, **fixed)
    after = transmission(mixed_a, mixed_s, kweights, weights, **fixed)
    assert np.abs(after - before).max() / before.max() < 1.0e-12

    # and without the multiplet basis it genuinely moves, which is the point
    naive_before = transmission(amplitudes, overlaps, kweights, weights,
                                coherent=False)
    naive_after = transmission(mixed_a, mixed_s, kweights, weights,
                               coherent=False)
    assert np.abs(naive_after - naive_before).max() / naive_before.max() > 1e-3


def test_a_multiplet_the_substrate_cannot_tell_apart_needs_no_rotation():
    """Schur's lemma, which is why this correction is usually invisible.

    At a symmetry point the little group acts irreducibly on a multiplet, so
    any invariant operator restricted to it is a multiple of the identity and
    every basis is already a channel basis. Graphene's Dirac pair measures
    ``diag(0.05405086, 0.05405084)`` with off-diagonals at 1e-8.
    """
    amplitudes, overlaps, kweights, weights = _random_case(nk=1, nbnd=3, npts=2)
    overlaps = np.zeros((1, 3, 3), dtype=complex)
    overlaps[0] = np.diag([0.4, 0.4, 1.1])  # scalar on the degenerate pair
    eigenvalues = np.array([[0.0, 0.0, 1.0]])
    with_basis = transmission(amplitudes, overlaps, kweights, weights,
                              coherent=False, eigenvalues=eigenvalues)
    without = transmission(amplitudes, overlaps, kweights, weights,
                           coherent=False)
    assert np.abs(with_basis - without).max() < 1.0e-14


def test_the_channel_basis_leaves_a_nondegenerate_spectrum_alone():
    amplitudes, overlaps, kweights, weights = _random_case(nk=2, nbnd=4)
    eigenvalues = np.tile(np.array([0.0, 0.3, 0.7, 1.2]), (2, 1))
    assert np.abs(channel_basis(overlaps, eigenvalues)
                  - np.eye(4)[None]).max() < 1.0e-14


def test_the_incoherent_map_drops_exactly_the_off_diagonal():
    amplitudes, overlaps, kweights, weights = _random_case()
    diagonal = np.zeros_like(overlaps)
    idx = np.arange(overlaps.shape[1])
    diagonal[:, idx, idx] = overlaps[:, idx, idx]
    assert np.abs(transmission(amplitudes, overlaps, kweights, weights,
                               coherent=False)
                  - transmission(amplitudes, diagonal, kweights, weights)).max() < 1e-12


def test_mismatched_shapes_are_refused_rather_than_broadcast():
    amplitudes, overlaps, kweights, weights = _random_case()
    with pytest.raises(ValueError, match="overlaps are"):
        transmission(amplitudes, overlaps[:, :2, :2], kweights, weights)
    with pytest.raises(ValueError, match="state weights"):
        transmission(amplitudes, overlaps, kweights, weights[:, :2])


# --------------------------------------------------------------------------
# the tip sampler
# --------------------------------------------------------------------------


def test_the_sampler_reproduces_a_plane_wave_it_was_built_from():
    """One coefficient, one plane wave: ``Omega^{-1/2} e^{2 pi i (k + h).s}``."""
    miller = _sphere(1)
    which = 5
    coefficients = np.zeros((1, miller.shape[0]), dtype=complex)
    coefficients[0, which] = 1.0
    k = np.array([0.25, -0.5, 0.0])
    points = np.array([[0.1, 0.2, 0.3], [0.7, -0.4, 1.9]])
    got = sample_wavefunctions(coefficients, miller, k, points, CUBIC.volume)
    phase = np.exp(2.0j * np.pi * (points @ (miller[which] + k)))
    assert np.abs(got[0] - phase / np.sqrt(CUBIC.volume)).max() < 1.0e-14


def test_the_sampler_carries_the_bloch_phase_outside_the_cell():
    """``psi(s + 1) = e^{2 pi i k} psi(s)``: a tip above a slab is outside it."""
    miller = _sphere(2)
    bands = _orthonormal_bands(miller, 2, seed=9)
    k = np.array([0.0, 0.0, 0.375])
    inside = np.array([[0.2, 0.3, 0.4]])
    outside = inside + np.array([0.0, 0.0, 1.0])
    a = sample_wavefunctions(bands, miller, k, inside, CUBIC.volume)
    b = sample_wavefunctions(bands, miller, k, outside, CUBIC.volume)
    assert np.abs(b - a * np.exp(2.0j * np.pi * k[2])).max() < 1.0e-13


def test_the_sampler_is_chunked_without_changing_its_answer():
    miller = _sphere(2)
    bands = _orthonormal_bands(miller, 3, seed=10)
    k = np.array([0.1, 0.2, 0.3])
    points = np.random.default_rng(0).uniform(-1.0, 2.0, size=(37, 3))
    whole = sample_wavefunctions(bands, miller, k, points, CUBIC.volume)
    pieces = sample_wavefunctions(bands, miller, k, points, CUBIC.volume, chunk=5)
    assert np.abs(whole - pieces).max() < 1.0e-15


def test_the_sampler_refuses_a_sphere_that_is_not_its_own():
    miller = _sphere(1)
    bands = _orthonormal_bands(miller, 2, seed=12)
    with pytest.raises(ValueError, match="not the same k-point"):
        sample_wavefunctions(bands, _sphere(2), np.zeros(3),
                             np.zeros((1, 3)), CUBIC.volume)


# --------------------------------------------------------------------------
# the whole chain against its own definition
# --------------------------------------------------------------------------


def test_the_whole_contraction_is_a_real_space_integral_of_the_green_function():
    """``T(r) = int_plane |sum_n a_n(r) psi*_n(r')|^2 d^2 r'``, done literally.

    The closing check, and the only one that sees every piece at once: build
    wavefunctions out of plane waves, evaluate ``G(r, r')`` on a real-space grid
    covering the exit plane, square it and integrate with a quadrature -- then
    compare against the closed-form Gram matrix and the contraction. Nothing in
    the fast path evaluates ``G`` anywhere, so this shares only the sampler with
    what it checks, and it is what pins the index order that
    :func:`test_the_transmission_is_the_plane_integral_of_the_squared_amplitude`
    checks in the abstract.

    The quadrature is *exact*, not approximate: ``psi*_n psi_m`` is a
    trigonometric polynomial of in-plane degree ``2 max|h|``, so a grid past
    that integrates it with no error at all.
    """
    cell, height, axis = HEXAGONAL, 0.23, 2
    miller = _sphere(2)
    bands = _orthonormal_bands(miller, 4, seed=21)
    k = np.array([0.25, 0.5, 0.0])
    tips = np.array([[0.11, 0.42, 0.77], [0.6, 0.1, 0.9]])
    weights = np.array([[0.9, 0.4, 0.25, 0.7]], dtype=complex)

    # the fast path
    overlaps = exit_overlap(bands, miller, height, axis, cell)[None]
    amplitudes = sample_wavefunctions(bands, miller, k, tips, cell.volume)[None]
    fast = transmission(amplitudes, overlaps, np.array([1.0]), weights)

    # the definition
    n = 2 * int(np.abs(miller[:, :2]).max()) + 3
    u, v = np.meshgrid(np.arange(n) / n, np.arange(n) / n, indexing="ij")
    plane = np.stack([u.ravel(), v.ravel(), np.full(u.size, height)], axis=1)
    on_plane = sample_wavefunctions(bands, miller, k, plane, cell.volume)
    a = amplitudes[0] * weights[0][:, None]
    green = np.einsum("np,nq->pq", a, on_plane.conj())
    slow = (surface_area(cell, axis) / plane.shape[0]) * (np.abs(green) ** 2).sum(axis=1)

    assert np.abs(fast - slow).max() / slow.max() < 1.0e-12
