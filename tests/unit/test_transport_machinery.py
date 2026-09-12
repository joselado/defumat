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

A **magnetic tip** adds a fourth: the two spinor components are contracted
through ``P_t = (1 + P n.sigma)/2`` rather than added, and the resulting
``T = Tr[P_t M]`` is still non-negative (both factors are positive
semi-definite), still blind to a degenerate rotation, partitions exactly
between ``+n`` and ``-n``, and gives half the unpolarized map at ``P = 0``.
"""

import numpy as np
import pytest

from defumat.basis.sample import sample_wavefunctions
from defumat.transport.green import (
    amplitude_weights,
    channel_basis,
    spin_transmission,
    transmission,
)
from defumat.transport.substrate import (
    exit_overlap,
    spin_projector,
    surface_area,
    volume_overlap,
)


class _Cell:
    """The three things :mod:`defumat.transport` asks a cell for."""

    def __init__(self, at):
        self.at = np.asarray(at, dtype=float)
        self.volume = float(abs(np.linalg.det(self.at)))
        # ``Cell.bg``'s convention: rows b1, b2, b3, with a_i . b_j = 2 pi.
        self.bg = 2.0 * np.pi * np.linalg.inv(self.at).T


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
# the magnetic tip
# --------------------------------------------------------------------------


def _spinor_case(nk=3, nbnd=4, npts=5, seed=13):
    """The same random Gram data as :func:`_random_case`, with two components."""
    rng = np.random.default_rng(seed)
    amplitudes = (rng.normal(size=(2, nk, nbnd, npts))
                  + 1.0j * rng.normal(size=(2, nk, nbnd, npts)))
    raw = rng.normal(size=(nk, nbnd, nbnd)) + 1.0j * rng.normal(size=(nk, nbnd, nbnd))
    overlaps = np.einsum("kij,klj->kil", raw, raw.conj())
    kweights = rng.uniform(0.1, 1.0, size=nk)
    weights = rng.uniform(0.1, 1.0, size=(nk, nbnd)).astype(complex)
    return amplitudes, overlaps, kweights, weights


def _unpolarized(amplitudes, overlaps, kweights, weights, **kwargs):
    """What the traced-over-spin tip gives: the two components added."""
    return sum(transmission(amplitudes[c], overlaps, kweights, weights, **kwargs)
               for c in range(2))


@pytest.mark.parametrize("coherent", [True, False])
def test_an_unpolarized_tip_projector_is_the_traced_sum(coherent):
    """``P_t = 1`` is the tip that takes both spins, which is what was there.

    The whole extension is one 2x2 contraction, and the identity matrix has to
    put it back exactly where it started -- otherwise every existing number in
    the phase moves.
    """
    case = _spinor_case()
    fixed = ({} if coherent else
             dict(coherent=False,
                  eigenvalues=np.tile(np.array([0.0, 0.0, 1.0, 2.0]),
                                      (case[0].shape[1], 1))))
    ours = spin_transmission(*case, np.eye(2, dtype=complex), **fixed)
    assert np.abs(ours - _unpolarized(*case, **fixed)).max() < 1.0e-13


@pytest.mark.parametrize("coherent", [True, False])
def test_the_two_tip_directions_partition_the_unpolarized_map(coherent):
    """``P_t(n, P) + P_t(-n, P) = 1`` for every ``n`` and every ``P``.

    An identity that shares no machinery with the contraction: it is a
    statement about the projector alone, and it holds for a generic direction
    and a partial polarization, not only for the axes.
    """
    case = _spinor_case()
    rng = np.random.default_rng(5)
    fixed = ({} if coherent else
             dict(coherent=False,
                  eigenvalues=np.tile(np.array([0.0, 0.0, 1.0, 2.0]),
                                      (case[0].shape[1], 1))))
    total = _unpolarized(*case, **fixed)
    for _ in range(3):
        direction = rng.normal(size=3)
        p = float(rng.uniform(-1.0, 1.0))
        plus = spin_transmission(*case, spin_projector(direction, p), **fixed)
        minus = spin_transmission(*case, spin_projector(-direction, p), **fixed)
        assert np.abs(plus + minus - total).max() / total.max() < 1.0e-14


def test_a_nonmagnetic_tip_gives_exactly_half():
    """``P = 0`` is ``P_t = 1/2``, which is P65's convention for a tip with no
    moment: half the charge, the average of the two channels."""
    case = _spinor_case()
    half = spin_transmission(*case, spin_projector((0.3, -0.7, 0.2), 0.0))
    total = _unpolarized(*case)
    assert np.abs(half - 0.5 * total).max() / total.max() < 1.0e-14


@pytest.mark.parametrize("coherent", [True, False])
def test_a_polarized_tip_cannot_make_the_transmission_negative(coherent):
    """The structural guarantee survives the extension, and it needs both
    factors: ``M = A^T S A^*`` is a Gram matrix in tip-spin space and ``P_t`` is
    a projector, so ``Tr[P_t M] = Tr[P_t^{1/2} M P_t^{1/2}] >= 0``."""
    amplitudes, overlaps, kweights, weights = _spinor_case()
    eigenvalues = np.tile(np.array([0.0, 0.0, 1.0, 2.0]), (overlaps.shape[0], 1))
    fixed = ({} if coherent
             else dict(coherent=False, eigenvalues=eigenvalues))
    rng = np.random.default_rng(17)
    for _ in range(5):
        projector = spin_projector(rng.normal(size=3), 1.0)
        assert spin_transmission(amplitudes, overlaps, kweights, weights,
                                 projector, **fixed).min() >= 0.0
    # and the 2x2 matrix it contracts is itself positive semi-definite
    a = amplitudes * weights[None, :, :, None]
    sa = np.einsum("kij,tkjp->tkip", overlaps, a.conj())
    m = np.einsum("sknp,tknp->stkp", a, sa)
    spectrum = np.linalg.eigvalsh(np.moveaxis(m, (0, 1), (-2, -1)))
    assert spectrum.min() > -1.0e-10


def test_the_tip_spin_index_order_is_the_definition_and_not_its_transpose():
    """``M[s,s'] = a_s^T S a_{s'}^*``, with the *un-conjugated* component first.

    The transposed version is ``M^T = M^*`` because ``M`` is Hermitian, so
    contracting it with ``P_t`` returns ``Tr[P_t^* M]`` -- which is exactly the
    answer for a tip along ``(n_x, -n_y, n_z)``. Real, non-negative, identical
    for any tip in the ``xz`` plane, and wrong. It is P54's and P66's transposed
    index one level up, in tip-spin space, and only a check against the
    definition sees it.
    """
    amplitudes, overlaps, kweights, weights = _spinor_case()
    direction = np.array([0.4, 0.8, -0.3])
    projector = spin_projector(direction, 0.9)

    a = amplitudes * weights[None, :, :, None]
    sa = np.einsum("kij,tkjp->tkip", overlaps, a.conj())
    m = np.einsum("sknp,tknp->stkp", a, sa)
    wrong = kweights @ np.real(np.einsum("ts,stkp->kp", projector,
                                         np.swapaxes(m, 0, 1)))
    right = spin_transmission(amplitudes, overlaps, kweights, weights, projector)

    mirrored = spin_projector((direction[0], -direction[1], direction[2]), 0.9)
    assert np.abs(wrong - spin_transmission(amplitudes, overlaps, kweights,
                                            weights, mirrored)).max() < 1.0e-12
    assert wrong.min() > 0.0                       # plausible
    assert np.abs(wrong - right).max() / right.max() > 1.0e-2   # and different


def test_a_polarized_tip_is_blind_to_a_rotation_inside_a_degenerate_multiplet():
    """Rule D4, still satisfied by construction: ``M`` is bilinear-covariant in
    ``a`` and ``a^*``, so mixing a degenerate multiplet cannot move it. The
    incoherent branch is a diagonal and still needs ``eigenvalues=``."""
    rng = np.random.default_rng(23)
    amplitudes, overlaps, kweights, weights = _spinor_case(nk=2, nbnd=4, npts=3)
    weights = np.ones_like(weights)
    eigenvalues = np.tile(np.array([0.0, 0.0, 1.0, 2.0]), (2, 1))
    raw = rng.normal(size=(2, 2)) + 1.0j * rng.normal(size=(2, 2))
    u, _ = np.linalg.qr(raw)
    full = np.eye(4, dtype=complex)
    full[:2, :2] = u
    mixed_a = np.einsum("ni,sknp->skip", full, amplitudes)
    mixed_s = np.einsum("ni,knm,mj->kij", full.conj(), overlaps, full)
    projector = spin_projector((0.2, 0.5, -0.8), 0.7)

    before = spin_transmission(amplitudes, overlaps, kweights, weights, projector)
    after = spin_transmission(mixed_a, mixed_s, kweights, weights, projector)
    assert np.abs(after - before).max() / before.max() < 1.0e-12

    fixed = dict(coherent=False, eigenvalues=eigenvalues)
    incoherent_before = spin_transmission(amplitudes, overlaps, kweights,
                                          weights, projector, **fixed)
    incoherent_after = spin_transmission(mixed_a, mixed_s, kweights, weights,
                                         projector, **fixed)
    assert (np.abs(incoherent_after - incoherent_before).max()
            / incoherent_before.max()) < 1.0e-12


def test_the_polarized_tip_refuses_shapes_it_cannot_read():
    amplitudes, overlaps, kweights, weights = _spinor_case()
    identity = np.eye(2, dtype=complex)
    with pytest.raises(ValueError, match="both spinor components"):
        spin_transmission(amplitudes[0], overlaps, kweights, weights, identity)
    with pytest.raises(ValueError, match="projector is 2x2"):
        spin_transmission(amplitudes, overlaps, kweights, weights, np.eye(3))
    with pytest.raises(ValueError, match="overlaps are"):
        spin_transmission(amplitudes, overlaps[:, :2, :2], kweights, weights,
                          identity)
    with pytest.raises(ValueError, match="state weights"):
        spin_transmission(amplitudes, overlaps, kweights, weights[:, :2],
                          identity)


def test_the_tip_acceptance_refuses_what_it_has_nothing_to_couple_to():
    """The two refusals, on the pure function that makes them.

    A run with no magnetization has nothing for the tip's moment to couple to,
    and a collinear run carries no transverse magnetization -- ``m_x`` and
    ``m_y`` there are absent rather than zero, which is P65's reasoning applied
    to the other lead.
    """
    from defumat.workflows.transport import _tip_acceptance

    assert _tip_acceptance(None, 1.0, 1, 1) == (None, None)
    with pytest.raises(NotImplementedError, match="tip needs a magnetization"):
        _tip_acceptance("up", 1.0, 1, 1)
    with pytest.raises(NotImplementedError, match="transverse component"):
        _tip_acceptance((1.0, 0.0, 0.0), 1.0, 1, 2)
    with pytest.raises(ValueError, match="tip polarization must be in"):
        _tip_acceptance("up", 1.5, 1, 2)
    with pytest.raises(ValueError, match="tip polarization must be in"):
        _tip_acceptance("x", -2.0, 2, 1)
    # a collinear tip is the same (1 +- P)/2 weight the substrate gets
    assert _tip_acceptance("up", 0.6, 1, 2)[1] == pytest.approx((0.8, 0.2))
    assert _tip_acceptance("down", 0.6, 1, 2)[1] == pytest.approx((0.2, 0.8))


def test_the_spinor_contraction_is_a_real_space_integral_of_the_green_function():
    """The magnetic tip against its definition, with a quadrature in between.

    ``T(r) = int_plane dr' Tr_spin[P_t G(r,r') Gamma_s G^dag(r',r)]`` written
    out literally: build the 2x2 spinor Green's function on a grid covering the
    exit plane, contract it with both leads' acceptances, integrate. It shares
    only the sampler with the fast path, and it is what pins the index order of
    ``M`` on data where a transpose is visible -- the substrate's own projector
    is off-diagonal here, and so is the tip's.
    """
    cell, height, axis = HEXAGONAL, 0.23, 2
    miller = _sphere(2)
    nbnd, npwx = 4, miller.shape[0]
    # the spinor's two components are the two halves of a row, which is how the
    # whole package stores them; nothing here needs them orthonormal
    spinors = _orthonormal_bands(np.zeros((2 * npwx, 3), dtype=int), nbnd, seed=41)
    k = np.array([0.25, 0.5, 0.0])
    tips = np.array([[0.11, 0.42, 0.77], [0.6, 0.1, 0.9]])
    weights = np.array([[0.9, 0.4, 0.25, 0.7]], dtype=complex)
    tip_projector = spin_projector((0.3, 0.6, -0.5), 0.8)
    substrate = spin_projector((-0.2, 0.7, 0.4), 0.9)

    # the fast path
    overlaps = exit_overlap(spinors, miller, height, axis, cell, npol=2,
                            projector=substrate)[None]
    sampled = sample_wavefunctions(spinors.reshape((nbnd, 2, npwx)), miller, k,
                                   tips, cell.volume)
    amplitudes = np.moveaxis(sampled, 1, 0)[:, None]     # (2, 1, nbnd, npts)
    fast = spin_transmission(amplitudes, overlaps, np.array([1.0]), weights,
                             tip_projector)

    # the definition
    n = 2 * int(np.abs(miller[:, :2]).max()) + 3
    u, v = np.meshgrid(np.arange(n) / n, np.arange(n) / n, indexing="ij")
    plane = np.stack([u.ravel(), v.ravel(), np.full(u.size, height)], axis=1)
    on_plane = sample_wavefunctions(spinors.reshape((nbnd, 2, npwx)), miller, k,
                                    plane, cell.volume)                # (n,2,q)
    a = amplitudes[:, 0] * weights[0][None, :, None]                   # (2,n,p)
    # G_{ss'}(r, r') = sum_n a_{n,s}(r) psi*_{n,s'}(r')
    green = np.einsum("snp,ntq->stpq", a, on_plane.conj())
    # Tr_spin[P_t G Gamma_s G^dag] at each (tip point p, exit point q), with
    # G^dag(r', r)[v, s] = conj(G(r, r')[s, v]) -- the exit variable is the one
    # that carries the conjugate, which is the whole index-order question
    slow = np.real(np.einsum("st,tupq,uv,svpq->p", tip_projector, green,
                             substrate, green.conj(), optimize=True))
    slow *= surface_area(cell, axis) / plane.shape[0]

    assert np.abs(fast - slow).max() / np.abs(slow).max() < 1.0e-12


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


# --- the band-count diagnostic is a diagonal too -----------------------------


def _degenerate_top(nk=2, nbnd=4, npts=3, seed=11):
    """A case whose **topmost** two bands are degenerate.

    That is the configuration the band-count diagnostic is read in: the
    truncation cuts at ``nbnd``, and if the top band has a partner the cut goes
    through a multiplet.
    """
    rng = np.random.default_rng(seed)
    amplitudes = (rng.normal(size=(nk, nbnd, npts))
                  + 1.0j * rng.normal(size=(nk, nbnd, npts)))
    raw = rng.normal(size=(nk, nbnd, nbnd)) + 1.0j * rng.normal(size=(nk, nbnd, nbnd))
    overlaps = np.einsum("kij,klj->kil", raw, raw.conj())
    kweights = rng.uniform(0.1, 1.0, size=nk)
    weights = rng.uniform(0.1, 1.0, size=(nk, nbnd)).astype(complex)
    eigenvalues = np.tile(np.array([0.0, 0.4, 1.0, 1.0]), (nk, 1))
    return amplitudes, overlaps, kweights, weights, eigenvalues


def test_the_top_multiplet_mask_is_the_whole_degenerate_block():
    from defumat.workflows.transport import _top_multiplet_mask

    _, _, _, _, eigenvalues = _degenerate_top()
    mask = _top_multiplet_mask(eigenvalues)
    assert mask == pytest.approx(np.tile([0.0, 0.0, 1.0, 1.0], (2, 1)))

    # A nondegenerate spectrum keeps the single topmost band, which is what the
    # diagnostic always meant.
    plain = np.tile(np.array([0.0, 0.4, 1.0, 1.7]), (2, 1))
    assert _top_multiplet_mask(plain) == pytest.approx(
        np.tile([0.0, 0.0, 0.0, 1.0], (2, 1))
    )


def test_the_band_edge_weight_is_blind_to_a_degenerate_rotation():
    """Rule D4 on the diagnostic rather than on the answer.

    The numerator was the raw diagonal of the *single* topmost band and the
    denominator had already been rotated into the substrate's channels, so the
    ratio that certifies the band-count truncation moved with a basis nobody
    chose. Taking the whole multiplet, in the same basis, fixes both halves.
    """
    from defumat.workflows.transport import _top_multiplet_mask

    amplitudes, overlaps, kweights, weights, eigenvalues = _degenerate_top()
    # ``amplitude_weights`` is a function of the eigenvalue alone, so two
    # members of a multiplet carry the *same* weight. A random per-band weight
    # would break the invariance on its own and would be testing the fixture.
    weights = np.asarray([[0.3, 0.5, 0.9, 0.9]] * len(kweights), dtype=complex)
    mask = _top_multiplet_mask(eigenvalues)

    def edge(a, s):
        top = transmission(a, s, kweights, weights * mask, coherent=False,
                           eigenvalues=eigenvalues)
        total = transmission(a, s, kweights, weights, coherent=False,
                             eigenvalues=eigenvalues)
        return top.sum() / total.sum()

    # An arbitrary unitary inside the degenerate top pair: exactly the freedom
    # the eigensolver has and no symmetry check sees.
    rng = np.random.default_rng(31)
    raw = rng.normal(size=(2, 2)) + 1.0j * rng.normal(size=(2, 2))
    u, _ = np.linalg.qr(raw)
    rotation = np.eye(4, dtype=complex)
    rotation[2:, 2:] = u
    mixed_a = np.einsum("ni,knp->kip", rotation, amplitudes)
    mixed_s = np.einsum("ni,knm,mj->kij", rotation.conj(), overlaps, rotation)

    assert abs(edge(mixed_a, mixed_s) - edge(amplitudes, overlaps)) < 1e-12

    # ... and the form that stood there genuinely moves, which is the point.
    def naive(a, s):
        top = transmission(a[:, -1:], s[:, -1:, -1:], kweights,
                           weights[:, -1:], coherent=False)
        total = transmission(a, s, kweights, weights, coherent=False,
                             eigenvalues=eigenvalues)
        return top.sum() / total.sum()

    assert abs(naive(mixed_a, mixed_s) - naive(amplitudes, overlaps)) > 1e-3


# --------------------------------------------------------------------------
# the momentum-resolved weight: a plane tip instead of a point one
# --------------------------------------------------------------------------


def _two_planes(seed=5, nbnd=4, nk=2):
    """A case with two *distinct* exit planes, which is what separates the forms.

    Both Gram matrices have to have off-diagonals, and they have to be
    different from each other: the transposed contraction the module docstring
    warns about agrees with the right one whenever either is diagonal, so a case
    built from one plane cannot tell them apart.
    """
    cell, axis = HEXAGONAL, 2
    miller = _sphere(2)
    exit_gram, tip_gram = [], []
    for ik in range(nk):
        bands = _orthonormal_bands(miller, nbnd, seed=seed + ik)
        exit_gram.append(exit_overlap(bands, miller, 0.13, axis, cell))
        tip_gram.append(exit_overlap(bands, miller, 0.71, axis, cell))
    rng = np.random.default_rng(seed)
    kweights = rng.uniform(0.3, 1.0, size=nk)
    weights = rng.uniform(0.2, 1.0, size=(nk, nbnd))
    return np.array(exit_gram), np.array(tip_gram), kweights, weights


def test_the_momentum_weight_is_the_plane_integral_of_the_point_tip_map():
    """The theorem the whole module rests on, evaluated literally on both sides.

    ``sum_k W(k)`` must equal the point-tip map of
    :func:`~defumat.transport.green.transmission` integrated over one cell's
    worth of the tip plane. The two routes share only ``exit_overlap``: one
    samples ``psi`` on a real-space quadrature of the tip plane and squares a
    Green's function, the other never leaves reciprocal space. The quadrature is
    *exact* rather than approximate -- ``psi*_n psi_m`` is a trigonometric
    polynomial of in-plane degree ``2 max|h|``, so a grid past that integrates
    it with no error -- which is why this can be asserted at round-off.
    """
    from defumat.transport.momentum import momentum_weights

    cell, axis = HEXAGONAL, 2
    exit_height, tip_height = 0.13, 0.71
    miller = _sphere(2)
    k = np.array([0.25, 0.5, 0.0])
    nbnd, nk = 4, 2

    bands = [_orthonormal_bands(miller, nbnd, seed=5 + ik) for ik in range(nk)]
    exit_gram = np.array([exit_overlap(b, miller, exit_height, axis, cell)
                          for b in bands])
    tip_gram = np.array([exit_overlap(b, miller, tip_height, axis, cell)
                         for b in bands])
    rng = np.random.default_rng(5)
    kweights = rng.uniform(0.3, 1.0, size=nk)
    weights = rng.uniform(0.2, 1.0, size=(nk, nbnd))

    fast = momentum_weights(exit_gram, tip_gram, kweights, weights)["weight"]

    # The point-tip map, on an exact quadrature of the tip plane.
    n = 2 * int(np.abs(miller[:, :2]).max()) + 3
    u, v = np.meshgrid(np.arange(n) / n, np.arange(n) / n, indexing="ij")
    plane = np.stack([u.ravel(), v.ravel(), np.full(u.size, tip_height)], axis=1)
    amplitudes = np.array([
        sample_wavefunctions(b, miller, k, plane, cell.volume) for b in bands])
    slow_map = transmission(amplitudes, exit_gram, kweights,
                            weights.astype(complex))
    slow = surface_area(cell, axis) / plane.shape[0] * slow_map.sum()

    assert abs(fast.sum() - slow) / abs(slow) < 1.0e-12


def test_the_transposed_contraction_is_a_different_number_on_two_planes():
    """The trap, made visible: it needs two *distinct* planes to show at all.

    ``Tr[D S^exit D S^tip]`` against the elementwise ``sum_nm ... S^exit[n,m]
    S^tip[n,m]``, which is ``Tr[D S^exit D (S^tip)^T]``. Both are real, both are
    non-negative, and they coincide the moment either matrix is diagonal -- so
    the Tersoff-Hamann limit, a one-band metal and every single-plane check
    agree. On random orthonormal bands they differ by about **3%**, which is
    the size worth knowing: far above round-off and far below anything a plot
    would show, so the wrong form is not something a picture catches. The
    previous test is what says which of the two is the plane integral.
    """
    from defumat.transport.momentum import momentum_weights

    exit_gram, tip_gram, kweights, weights = _two_planes()
    right = momentum_weights(exit_gram, tip_gram, kweights, weights)["weight"]

    scaled = exit_gram * weights[:, :, None] * weights[:, None, :]
    transposed = kweights * np.real(np.einsum("kij,kij->k", scaled, tip_gram))

    assert np.all(right > 0.0) and np.all(transposed > 0.0)
    difference = np.abs(right - transposed).max() / right.max()
    assert 1.0e-3 < difference < 0.5


def test_the_momentum_weight_is_blind_to_a_rotation_inside_a_multiplet():
    """Rule D4, satisfied by construction: ``Tr[U'XU U'YU] = Tr[XY]``.

    A degenerate eigensolver may return any basis inside a multiplet. Both Gram
    matrices rotate the same way, and a trace of their product does not move --
    which is why ``weight`` needs no degeneracy handling at all, where the
    ``incoherent`` column below does.
    """
    from defumat.transport.momentum import momentum_weights

    exit_gram, tip_gram, kweights, weights = _two_planes()
    weights[:, 1:3] = weights[:, 1:3].mean()  # a degenerate pair, equal g_n
    before = momentum_weights(exit_gram, tip_gram, kweights, weights)["weight"]

    rng = np.random.default_rng(3)
    raw = rng.normal(size=(2, 2)) + 1.0j * rng.normal(size=(2, 2))
    u = np.linalg.qr(raw)[0]
    mix = np.eye(exit_gram.shape[1], dtype=complex)
    mix[1:3, 1:3] = u
    rotate = lambda g: np.einsum("ni,knm,mj->kij", mix.conj(), g, mix)
    after = momentum_weights(rotate(exit_gram), rotate(tip_gram), kweights,
                             weights)["weight"]

    assert np.abs(after - before).max() / before.max() < 1.0e-12


def test_the_tersoff_hamann_column_is_the_structureless_substrate_limit():
    """``S^exit -> 1`` and the transmission *is* the tip plane's local DOS.

    Not approximately: widening the substrate to the whole cell makes its Gram
    matrix the identity by orthonormality, and then the trace collapses onto the
    tip's diagonal. Both further limits are here too -- both matrices the
    identity leaves the plain Fermi-surface weight, which is what ``bare`` is.
    """
    from defumat.transport.momentum import momentum_weights

    exit_gram, tip_gram, kweights, weights = _two_planes()
    identity = np.broadcast_to(np.eye(exit_gram.shape[1], dtype=complex),
                               exit_gram.shape).copy()

    columns = momentum_weights(identity, tip_gram, kweights, weights)
    assert columns["weight"] == pytest.approx(columns["tersoff_hamann"],
                                              rel=1.0e-13)
    both = momentum_weights(identity, identity, kweights, weights)
    assert both["weight"] == pytest.approx(both["bare"], rel=1.0e-13)
    assert both["bare"] == pytest.approx(kweights * (weights**2).sum(axis=1))


def test_the_momentum_weight_cannot_go_negative():
    """Two positive semi-definite matrices, so ``Tr[XY] >= 0``. No sign to have."""
    from defumat.transport.momentum import momentum_weights

    exit_gram, tip_gram, kweights, weights = _two_planes(seed=17, nbnd=6, nk=4)
    for column in momentum_weights(exit_gram, tip_gram, kweights, weights,
                                   eigenvalues=np.zeros((4, 6))).values():
        assert np.all(column >= 0.0)


def test_the_incoherent_column_is_blind_to_a_degenerate_rotation():
    """A diagonal is not invariant, so it is taken in the substrate's own basis.

    This is the one column that needs the degeneracy handling ``weight`` does
    not, and it is the same :func:`~defumat.transport.green.channel_basis` the
    real-space map uses.
    """
    from defumat.transport.momentum import momentum_weights

    exit_gram, tip_gram, kweights, weights = _two_planes()
    eigenvalues = np.tile(np.array([0.0, 0.5, 0.5, 1.2]), (exit_gram.shape[0], 1))
    weights[:, 1:3] = weights[:, 1:3].mean()
    before = momentum_weights(exit_gram, tip_gram, kweights, weights,
                              eigenvalues=eigenvalues)["incoherent"]

    rng = np.random.default_rng(8)
    u = np.linalg.qr(rng.normal(size=(2, 2))
                     + 1.0j * rng.normal(size=(2, 2)))[0]
    mix = np.eye(exit_gram.shape[1], dtype=complex)
    mix[1:3, 1:3] = u
    rotate = lambda g: np.einsum("ni,knm,mj->kij", mix.conj(), g, mix)
    after = momentum_weights(rotate(exit_gram), rotate(tip_gram), kweights,
                             weights, eigenvalues=eigenvalues)["incoherent"]

    assert np.abs(after - before).max() / before.max() < 1.0e-10
    assert np.all(before <= momentum_weights(
        exit_gram, tip_gram, kweights, weights)["weight"] * 1e6)


def test_momentum_weights_refuse_shapes_they_cannot_read():
    from defumat.transport.momentum import momentum_weights

    exit_gram, tip_gram, kweights, weights = _two_planes()
    with pytest.raises(ValueError, match="same shape"):
        momentum_weights(exit_gram, tip_gram[:, :2, :2], kweights, weights)
    with pytest.raises(ValueError, match="do not match"):
        momentum_weights(exit_gram, tip_gram, kweights, weights[:, :2])


# --- the zone partition and the vacuum decay ---------------------------------


def test_the_pocket_partition_covers_the_zone_exactly_once():
    """Closer to a corner than to the centre: a partition, not a contour mask.

    Every k-point belongs to exactly one side, so the two shares add to one and
    no threshold enters -- masking on the weight instead would make the shares
    depend on where the threshold was put.

    **On a hexagonal zone the corner share is exactly 2/3**, which is what fixes
    the geometry rather than an assumption about which region is bigger. The six
    perpendicular bisectors between ``Gamma`` and the corners bound a hexagon of
    inradius ``|K|/2``, of area ``sqrt(3) |K|^2 / 2``; the zone itself has
    inradius ``|M| = (sqrt(3)/2)|K|`` and area ``3 sqrt(3) |K|^2 / 2``. The ratio
    is ``1/3``, so the corners take the other two-thirds -- and it is worth
    knowing before reading any share, because a Fermi surface that carried its
    weight uniformly would already report 67% at ``K``.
    """
    from defumat.transport.momentum import pocket_mask

    n = 30
    grid = np.stack(np.meshgrid(np.arange(n) / n, np.arange(n) / n, [0.0],
                                indexing="ij"), axis=-1).reshape((-1, 3))
    cartesian = grid @ np.asarray(HEXAGONAL.bg)
    mask = pocket_mask(cartesian, HEXAGONAL)
    assert mask.dtype == bool and mask.shape == (n * n,)
    assert mask.mean() == pytest.approx(2.0 / 3.0, abs=0.01)
    # Gamma is never a corner point and a corner is always one.
    assert not mask[0]
    corner = np.array([[1 / 3, 1 / 3, 0.0]]) @ np.asarray(HEXAGONAL.bg)
    assert pocket_mask(corner, HEXAGONAL)[0]


def test_the_decay_constant_is_the_amplitude_one_and_not_twice_it():
    """``W ~ exp(-2 kappa z)`` for a tip sweep and ``exp(-4 kappa z)`` for both.

    Planted rather than measured: the factor is the whole content of the
    function, and getting it wrong makes :func:`decay_identity` come out four
    times too large -- which is the failure mode the identity exists to catch.
    """
    from defumat.transport.momentum import decay_constants

    heights = np.linspace(2.0, 8.0, 7)
    kappa = np.array([0.31, 0.77])
    tip = np.exp(-2.0 * np.outer(heights, kappa))
    assert decay_constants(heights, tip) == pytest.approx(kappa, rel=1.0e-10)
    both = np.exp(-4.0 * np.outer(heights, kappa))
    assert decay_constants(heights, both, planes="both") == pytest.approx(
        kappa, rel=1.0e-10)
    with pytest.raises(ValueError, match="planes must be"):
        decay_constants(heights, tip, planes="exit")


def test_the_decay_identity_closes_on_a_planted_free_electron_tail():
    """``kappa^2 - kappa'^2 = |k|^2 - |k'|^2``, and a factor of two reads as four.

    Two pockets given the *same* barrier height and different lateral momenta:
    the identity is then exact by construction, so what it tests is the
    bookkeeping. Feeding it twice a decay constant, which is what fitting
    ``|psi|^2`` without halving the slope would give, puts the residual at 3 --
    a factor of four on the measured side.
    """
    from defumat.transport.momentum import decay_identity

    barrier, k_corner, k_centre = 0.55, np.array([0.42, 0.0, 0.0]), np.zeros(3)
    kappa = [np.sqrt(barrier**2 + float(k @ k)) for k in (k_corner, k_centre)]
    closed = decay_identity(kappa[0], kappa[1], k_corner, k_centre)
    assert closed["relative_residual"] < 1.0e-12

    doubled = decay_identity(2 * kappa[0], 2 * kappa[1], k_corner, k_centre)
    assert doubled["relative_residual"] == pytest.approx(3.0, rel=1.0e-12)
