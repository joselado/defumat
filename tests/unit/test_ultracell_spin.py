"""P88 stage 3b: the spinor ultracell's spin algebra, with no SCF in sight.

The physics of a noncollinear ultracell is entirely in the frozen states and in
``dV``; what stage 3b adds to the *code* is a component axis and a 2x2 multiply.
That is a small piece of algebra with a large blast radius, and it has the
property this project keeps paying for: **a sign in it is invisible in every
energy and every symmetry check**. Flipping ``m_y`` gives the state of the
opposite chirality, which is exactly degenerate with the right one whenever
spin-orbit coupling is off, so an SCF converges, the total energy is right, the
moments have the right lengths, and the helix turns the wrong way.

So the algebra is pinned here, by three identities that hold **exactly** on
arbitrary inputs rather than to a convergence threshold:

1. **A spinor whose down component is zero is a collinear state**, and the
   matrix built for it must be the collinear matrix, to round-off. Likewise for
   the up component. This is the whole of ``npol = 2`` reduced to a case whose
   answer is already validated against a supercell (stage 1 and 3a).
2. **A rigid spin rotation is not a physical change.** Turning every spinor by
   ``U = exp(-i theta n.sigma/2)`` and the field by the matching ``R(theta, n)``
   leaves the Hamiltonian matrix *unchanged* -- not unitarily equivalent,
   unchanged -- because ``U^dagger (R B . sigma) U = B . sigma``. Rotating about
   **x** is what makes this a test: it mixes ``y`` with ``z``, where a rotation
   about ``z`` would leave a wrong ``m_y`` sign untouched.
3. **The density rotates with it.** The same rotation of the same states must
   leave the charge alone and rotate the magnetization, which is the output side
   of the same convention.

None of the three needs a converged anything, so they are cheap and they are in
the push gate -- which is where a convention check belongs, because it is the
kind that goes dead silently.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from defumat.config import DOUBLE
from defumat.hamiltonian.noncollinear import spin_multiply
from defumat.ultracell.density import spinor_ultracell_density, ultracell_density
from defumat.ultracell.hamiltonian import ultracell_matrix

pytestmark = pytest.mark.unit

#: Small enough to run in milliseconds, large enough that the box index is not
#: a permutation of something trivial and that ``Q`` values genuinely differ.
GRID = (6, 4, 4)
CELLS = 2
NBND = 3
NPWX = 5


def _inputs(seed: int):
    """Random coefficients, a random box index and random eigenvalues.

    Nothing here is orthonormal and nothing is an eigenstate: the matrix build
    is bilinear in the coefficients and linear in ``dV``, so every identity
    below is an identity of the *expression* and holds on arbitrary input. That
    is deliberate -- a test that first has to converge something is a test whose
    failures are ambiguous.
    """
    rng = np.random.default_rng(seed)
    points = int(np.prod(GRID))
    shape = (CELLS, NBND, NPWX)

    def draw():
        return jnp.asarray(
            rng.normal(size=shape) + 1j * rng.normal(size=shape),
            dtype=DOUBLE.complex,
        )

    box_index = jnp.asarray(
        rng.choice(points, size=(CELLS, NPWX), replace=False).reshape(CELLS, NPWX)
    )
    eigenvalues = jnp.asarray(rng.normal(size=(CELLS, NBND)), dtype=DOUBLE.real)
    return draw(), draw(), box_index, eigenvalues, rng


def _potential(rng, components: int):
    """A real ``(components, *GRID)`` potential in the ``(v_0, B)`` layout."""
    return jnp.asarray(
        rng.normal(size=(components,) + GRID), dtype=DOUBLE.real
    )


def _pauli_rotation(angle: float, axis) -> np.ndarray:
    """``U = exp(-i theta n.sigma/2)``, the 2x2 spinor rotation."""
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    dotted = n[0] * sx + n[1] * sy + n[2] * sz
    return np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * dotted


def _cartesian_rotation(angle: float, axis) -> np.ndarray:
    """Rodrigues' ``R(theta, n)``, written down independently of ``U``.

    Independently on purpose: deriving one from the other would make the test
    check that two spellings of the same convention agree, which is exactly the
    tautology the ``m_y`` sign would hide behind.
    """
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    cross = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return (
        np.cos(angle) * np.eye(3)
        + np.sin(angle) * cross
        + (1 - np.cos(angle)) * np.outer(n, n)
    )


def _as_spinor(up, down):
    """``(N, nbnd, 2 npwx)`` from two ``(N, nbnd, npwx)`` halves."""
    return jnp.concatenate([up, down], axis=-1)


# -- 1. a spinor with one component is a collinear state ---------------------


@pytest.mark.parametrize("channel", [0, 1])
def test_a_one_component_spinor_is_the_collinear_matrix(channel):
    """``npol = 2`` on ``(c, 0)`` reproduces the collinear build exactly.

    The collinear channel ``s`` feels ``v_s``, and in the ``(v_0, B)`` layout
    that is ``v_0 + v_z`` for up and ``v_0 - v_z`` for down. A state with no
    down component can be acted on by the off-diagonal Pauli terms -- they
    *produce* a down component -- but the bra has none either, so those terms
    cannot come back into the matrix element. That is why the identity is exact
    rather than approximate, and it is what makes the ``npol = 2`` build
    inherit stage 1's and stage 3a's supercell numbers on the diagonal.
    """
    up, down, box_index, eigenvalues, rng = _inputs(seed=11 + channel)
    zero = jnp.zeros_like(up)
    states = _as_spinor(up, zero) if channel == 0 else _as_spinor(zero, down)
    single = up if channel == 0 else down

    potential = _potential(rng, 4)
    v0, bz = potential[0], potential[3]
    # The transverse components are deliberately non-zero: nothing in this
    # state can feel them, and a build that accidentally let them through --
    # by contracting the wrong component, say -- would fail here.
    collinear = v0 + bz if channel == 0 else v0 - bz

    spinor = ultracell_matrix(
        states, eigenvalues, box_index, potential, GRID, batch=1, npol=2
    )
    reference = ultracell_matrix(
        single, eigenvalues, box_index, collinear, GRID, batch=1
    )
    assert np.abs(np.asarray(spinor - reference)).max() < 1e-12


def _both_channels(rng, seed=23):
    """A spinor set holding every up state and every down state, and its map.

    The basis index is ``(Q, n)`` flattened, so putting the up states first and
    the down states second *within each* ``Q`` does not give a block-diagonal
    matrix in the flattened order -- it gives a permutation of one. Building
    that permutation explicitly is the point: it is the same index arithmetic
    the density unpacks with, and getting it wrong is how a state would end up
    attributed to the wrong cell of the ultracell.
    """
    up, down, box_index, eigenvalues, _ = _inputs(seed=seed)
    zero = jnp.zeros_like(up)
    states = jnp.concatenate(
        [_as_spinor(up, zero), _as_spinor(zero, down)], axis=1
    )
    levels = jnp.concatenate([eigenvalues, eigenvalues + 0.25], axis=1)
    rows = [
        np.array([q * 2 * NBND + channel * NBND + n
                  for q in range(CELLS) for n in range(NBND)])
        for channel in (0, 1)
    ]
    return (up, down), states, (eigenvalues, eigenvalues + 0.25), levels, \
        box_index, rows


def test_a_collinear_potential_gives_the_two_collinear_matrices():
    """Both channels at once, under a potential along ``z``: two blocks.

    **The block structure belongs to the potential and not to the basis**, which
    is the sentence the whole regime turns on. A potential with ``B_x = B_y = 0``
    is diagonal in spin, so a spinor Hamiltonian built in a basis of pure up and
    pure down states *is* the two collinear matrices -- and that is what makes
    ``nspin = 2`` a legitimate shortcut rather than a different physics. The
    companion test below is the other half: put the transverse components back
    and the blocks are gone.
    """
    (up, down), states, (e_up, e_dw), levels, box_index, rows = _both_channels(
        np.random.default_rng(0)
    )
    rng = np.random.default_rng(23)
    potential = _potential(rng, 4).at[1:3].set(0.0)
    v_up, v_dw = potential[0] + potential[3], potential[0] - potential[3]

    spinor = np.asarray(
        ultracell_matrix(states, levels, box_index, potential, GRID,
                         batch=1, npol=2)
    )
    reference = np.zeros_like(spinor)
    for index, (c, e, v) in zip(rows, ((up, e_up, v_up), (down, e_dw, v_dw))):
        block = np.asarray(ultracell_matrix(c, e, box_index, v, GRID, batch=1))
        reference[np.ix_(index, index)] = block
    assert np.abs(spinor - reference).max() < 1e-12


def test_a_transverse_field_couples_the_two_channels():
    """The guard fires: with ``B_x``, ``B_y`` on, the blocks are not blocks.

    A test that only ever asserts a matrix *is* block diagonal cannot tell a
    correct spin-diagonal potential from a build that dropped the off-diagonal
    Pauli terms altogether -- and dropping them is exactly the bug that would
    make a helix impossible while every collinear number stayed right. So the
    same comparison is run with the transverse field restored and the
    off-diagonal blocks required to be *large*.
    """
    _, states, _, levels, box_index, rows = _both_channels(
        np.random.default_rng(0)
    )
    rng = np.random.default_rng(23)
    potential = _potential(rng, 4)
    spinor = np.asarray(
        ultracell_matrix(states, levels, box_index, potential, GRID,
                         batch=1, npol=2)
    )
    coupling = spinor[np.ix_(rows[0], rows[1])]
    assert np.abs(coupling).max() > 1e-2 * np.abs(spinor).max()


# -- 2. a rigid spin rotation changes nothing --------------------------------


@pytest.mark.parametrize("axis", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                  (1.0, 1.0, 1.0)])
def test_a_rigid_spin_rotation_leaves_the_matrix_alone(axis):
    """``H(U psi, R B) == H(psi, B)``, exactly, for a proper rotation.

    **Turn the states and the field together and nothing has happened**, which
    is what "no spin-orbit coupling" means: without it the Hamiltonian knows the
    spin frame only through ``B``, so a global rotation of both is a relabelling.
    The identity underneath is ``U^dagger (sigma . B) U = sigma . (R^T B)``,
    read off numerically from ``U^dagger sigma_a U = R_ab sigma_b`` rather than
    written from intuition -- ``CLAUDE.md``'s "index order in a transposed pair
    reads as a sign", and the first draft of this test had *both* of its
    transposes the wrong way round.

    The **x** axis is the one that earns its place in the list: it mixes ``y``
    with ``z``, so a wrong sign on ``m_y`` breaks it, where a rotation about
    ``z`` would leave the transverse pair's relative sign untested and pass
    either way. A wrong sign is not a rotation of anything -- it is a
    reflection, determinant -1 -- so no choice of ``R`` rescues it at a generic
    angle, which is what the companion test below shows.
    """
    angle = 0.7
    up, down, box_index, eigenvalues, rng = _inputs(seed=37)
    states = _as_spinor(up, down)
    potential = _potential(rng, 4)

    U = _pauli_rotation(angle, axis)
    R = _cartesian_rotation(angle, axis)

    components = states.reshape(CELLS, NBND, 2, NPWX)
    turned = jnp.einsum(
        "ab,qnbp->qnap", jnp.asarray(U, dtype=states.dtype), components
    ).reshape(CELLS, NBND, 2 * NPWX)
    rotated_field = jnp.einsum(
        "ab,bxyz->axyz", jnp.asarray(R, dtype=potential.dtype), potential[1:]
    )
    rotated = jnp.concatenate([potential[:1], rotated_field])

    plain = ultracell_matrix(states, eigenvalues, box_index, potential, GRID,
                             batch=1, npol=2)
    spun = ultracell_matrix(turned, eigenvalues, box_index, rotated, GRID,
                            batch=1, npol=2)
    scale = float(np.abs(np.asarray(plain)).max())
    assert np.abs(np.asarray(spun - plain)).max() < 1e-11 * scale


def _wrong_multiply(field, potential):
    """``spin_multiply`` with ``m_y``'s sign flipped, and nothing else.

    The plausible bug, written down so that the check above can be shown to
    catch it. It is a perfectly self-consistent operator -- Hermitian, real
    spectrum, correct collinear limit -- describing the state of the opposite
    chirality.
    """
    v0, mx, my, mz = potential[0], potential[1], potential[2], potential[3]
    up, down = field[..., 0, :, :, :], field[..., 1, :, :, :]
    return jnp.stack(
        [
            up * (v0 + mz) + down * (mx + 1j * my),
            down * (v0 - mz) + up * (mx - 1j * my),
        ],
        axis=-4,
    )


def test_the_rotation_identity_rejects_a_flipped_transverse_sign():
    """The guard fires, and it is checked against a *wrong implementation*.

    ``CLAUDE.md``'s "a check whose null result cannot be told from a pass". The
    rotation test asserts that a difference is zero, and the first version of
    this guard tried to trip it by negating ``B_y`` in the *input* -- which
    proves nothing, because the identity holds for every field, a flipped one
    included, and the guard passed at 1.8e-15. What has to be perturbed is the
    operator, so the wrong Pauli algebra is written out above and required to
    fail the same identity that the right one satisfies.

    The identity is taken at the pointwise level, where it is exact and needs no
    ultracell at all: ``V(v_0, R B) U psi = U V(v_0, B) psi``.
    """
    angle, axis = 0.7, (1.0, 0.0, 0.0)
    rng = np.random.default_rng(41)
    potential = _potential(rng, 4)
    field = jnp.asarray(
        rng.normal(size=(2,) + GRID) + 1j * rng.normal(size=(2,) + GRID),
        dtype=DOUBLE.complex,
    )
    U = jnp.asarray(_pauli_rotation(angle, axis), dtype=DOUBLE.complex)
    R = _cartesian_rotation(angle, axis)
    rotated = jnp.concatenate([
        potential[:1],
        jnp.einsum("ab,bxyz->axyz", jnp.asarray(R, dtype=potential.dtype),
                   potential[1:]),
    ])
    turned = jnp.einsum("ab,bxyz->axyz", U, field)

    def residual(multiply):
        left = multiply(turned, rotated)
        right = jnp.einsum("ab,bxyz->axyz", U, multiply(field, potential))
        return float(jnp.max(jnp.abs(left - right)))

    scale = float(jnp.max(jnp.abs(spin_multiply(field, potential))))
    assert residual(spin_multiply) < 1e-12 * scale
    assert residual(_wrong_multiply) > 1e-2 * scale


# -- 3. the density side of the same convention ------------------------------


def test_a_one_component_spinor_density_is_the_collinear_one():
    """``(c, 0)`` gives the collinear channel's density in ``n`` and ``m_z``.

    A fully up-polarized state has ``n = |c|^2``, ``m_z = +|c|^2`` and no
    transverse magnetization at all, so all four components are pinned by the
    one scalar :func:`ultracell_density` already produces.
    """
    up, _, box_index, _, rng = _inputs(seed=51)
    zero = jnp.zeros_like(up)
    vectors = jnp.asarray(
        rng.normal(size=(CELLS * NBND, 4)) + 1j * rng.normal(size=(CELLS * NBND, 4)),
        dtype=DOUBLE.complex,
    )
    weights = jnp.asarray(rng.uniform(size=4), dtype=DOUBLE.real)

    spinor = np.asarray(spinor_ultracell_density(
        _as_spinor(up, zero), vectors, weights, box_index, GRID, 1.0, batch=1
    ))
    scalar = np.asarray(ultracell_density(
        up, vectors, weights, box_index, GRID, 1.0, batch=1
    ))
    peak = np.abs(scalar).max()
    assert np.abs(spinor[0] - scalar).max() < 1e-12 * peak
    assert np.abs(spinor[3] - scalar).max() < 1e-12 * peak
    assert np.abs(spinor[1]).max() < 1e-12 * peak
    assert np.abs(spinor[2]).max() < 1e-12 * peak


@pytest.mark.parametrize("axis", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
def test_the_density_rotates_with_the_states(axis):
    """Turn every spinor by ``U``: the charge is unchanged and ``m`` turns by ``R``.

    ``<U psi|sigma_a|U psi> = <psi|U^dagger sigma_a U|psi> = R_ab m_b``, from
    the same defining relation the matrix test uses -- so the magnetization
    rotates by ``R`` and the field that leaves the Hamiltonian alone rotates by
    ``R`` as well. The two look like they should be transposes of each other and
    are not, which is why both were taken from the algebra rather than guessed.
    """
    angle = 0.9
    up, down, box_index, _, rng = _inputs(seed=67)
    states = _as_spinor(up, down)
    vectors = jnp.asarray(
        rng.normal(size=(CELLS * NBND, 3)) + 1j * rng.normal(size=(CELLS * NBND, 3)),
        dtype=DOUBLE.complex,
    )
    weights = jnp.asarray(rng.uniform(size=3), dtype=DOUBLE.real)

    U = _pauli_rotation(angle, axis)
    R = _cartesian_rotation(angle, axis)
    components = states.reshape(CELLS, NBND, 2, NPWX)
    turned = jnp.einsum(
        "ab,qnbp->qnap", jnp.asarray(U, dtype=states.dtype), components
    ).reshape(CELLS, NBND, 2 * NPWX)

    plain = np.asarray(spinor_ultracell_density(
        states, vectors, weights, box_index, GRID, 1.0, batch=1))
    spun = np.asarray(spinor_ultracell_density(
        turned, vectors, weights, box_index, GRID, 1.0, batch=1))

    peak = np.abs(plain[0]).max()
    assert np.abs(spun[0] - plain[0]).max() < 1e-12 * peak
    expected = np.einsum("ab,bxyz->axyz", R, plain[1:])
    assert np.abs(spun[1:] - expected).max() < 1e-11 * peak
    # ...and the guard fires: the transverse components actually moved, so a
    # rotation that did nothing would not pass by default.
    assert np.abs(spun[1:] - plain[1:]).max() > 1e-3 * peak


def test_a_nonmagnetic_spinor_potential_is_a_scalar_multiplication():
    """``nspin_mag = 1``: a spin-orbit run with no magnetization.

    The density has one component, the potential has one component, and the
    matrix must be what a scalar potential acting on both spinor halves gives.
    This is the regime an ``lspinorb`` insulator is in, and the whole point of
    it is that nothing above the matrix build has to know: ``spin_multiply``'s
    one-component branch is ``vloc_psi_nc``'s ``.NOT. domag``.
    """
    up, down, box_index, eigenvalues, rng = _inputs(seed=83)
    states = _as_spinor(up, down)
    scalar = _potential(rng, 1)

    matrix = np.asarray(ultracell_matrix(
        states, eigenvalues, box_index, scalar, GRID, batch=1, npol=2))
    padded = jnp.concatenate([scalar, jnp.zeros((3,) + GRID, dtype=scalar.dtype)])
    reference = np.asarray(ultracell_matrix(
        states, eigenvalues, box_index, padded, GRID, batch=1, npol=2))
    assert np.abs(matrix - reference).max() < 1e-12

    # The two spinor halves are independent under a scalar potential, so the
    # matrix is the sum of the two collinear ones built on the same ``dV``.
    halves = sum(
        np.asarray(ultracell_matrix(c, 0.0 * eigenvalues, box_index, scalar[0],
                                    GRID, batch=1))
        for c in (up, down)
    )
    diagonal = np.diag(np.asarray(eigenvalues).reshape(-1))
    assert np.abs(matrix - halves - diagonal).max() < 1e-12


def test_spin_multiply_is_hermitian_pointwise():
    """The 2x2 matrix at each point is Hermitian, so the ultracell one is too.

    Written as a property of :func:`spin_multiply` rather than checked on the
    assembled matrix, because ``ultracell_matrix`` symmetrises its result and
    would hide a non-Hermitian potential behind that.
    """
    rng = np.random.default_rng(97)
    potential = _potential(rng, 4)
    field = jnp.asarray(
        rng.normal(size=(2,) + GRID) + 1j * rng.normal(size=(2,) + GRID),
        dtype=DOUBLE.complex,
    )
    other = jnp.asarray(
        rng.normal(size=(2,) + GRID) + 1j * rng.normal(size=(2,) + GRID),
        dtype=DOUBLE.complex,
    )
    left = jnp.sum(jnp.conj(other) * spin_multiply(field, potential))
    right = jnp.sum(jnp.conj(spin_multiply(other, potential)) * field)
    assert abs(complex(left - right)) < 1e-10 * abs(complex(left))
