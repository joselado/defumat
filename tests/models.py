"""Tight-binding models with known topological invariants.

The invariants in :mod:`defumat.topology` are integers, and an integer is only
checked by a case whose answer is known independently. A DFT calculation big
enough to be topologically interesting is minutes of one core; these models are
milliseconds, and their answers come from the literature rather than from this
code. They are the same systems the reference implementations use -- ``pyqula``
(``examples/2d/z2_transition``, ``examples/2d/quantum_geometric_tensor``) and
``elkpy`` (``tests/test_wilson_gauge_invariance.py``'s doubled Qi-Wu-Zhang
model) -- so the numbers here are comparable with theirs.

They live in ``tests/`` and not in the package on purpose: defumat is a
plane-wave DFT code, and a library of model Hamiltonians is not part of it.
What *is* part of it is :class:`defumat.topology.ModelStates`, the state set
that any ``H(k)`` plugs into.

Every Hamiltonian here takes a k-point in **crystal** coordinates and returns a
Hermitian matrix, built in the periodic (cell) gauge -- ``H(k + b) = H(k)`` for
any reciprocal lattice vector -- so that the wrap needs no orbital phases.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

TWOPI = 2.0 * np.pi

PAULI_X = jnp.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = jnp.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = jnp.array([[1, 0], [0, -1]], dtype=complex)

#: Lattice-vector parts of the three nearest-neighbour bonds of a honeycomb
#: lattice with sublattice A at the origin and B at (1/3, 1/3).
HONEYCOMB_NN = np.array([[0, 0, 0], [-1, 0, 0], [0, -1, 0]], dtype=float)
#: The three next-nearest-neighbour vectors circulating one way around a
#: sublattice: the *differences* of the nearest-neighbour bonds, which is what
#: makes the Haldane gap ``3 sqrt(3) t2 |sin phi|`` rather than a third of it.
#: Taking any three lattice vectors that happen to be next-nearest neighbours
#: gives a model that is still a Chern insulator and whose transition sits
#: somewhere else -- the failure that looks like a wrong invariant.
HONEYCOMB_NNN = np.array([[1, 0, 0], [-1, 1, 0], [0, -1, 0]], dtype=float)


def haldane(t: float = 1.0, t2: float = 0.2, phi: float = np.pi / 2, mass: float = 0.0):
    """Haldane's honeycomb model (PRL 61, 2015 (1988)).

    A Chern insulator: the lower band carries ``|C| = 1`` while
    ``|mass| < 3 sqrt(3) t2 |sin phi|`` and ``C = 0`` beyond it.
    """
    def hamiltonian(k):
        k = jnp.asarray(k)
        f = jnp.sum(jnp.exp(1j * TWOPI * (jnp.asarray(HONEYCOMB_NN) @ k)))
        g = jnp.sum(jnp.exp(1j * TWOPI * (jnp.asarray(HONEYCOMB_NNN) @ k)))
        haa = mass + t2 * (jnp.exp(1j * phi) * g + jnp.exp(-1j * phi) * jnp.conj(g))
        hbb = -mass + t2 * (jnp.exp(-1j * phi) * g + jnp.exp(1j * phi) * jnp.conj(g))
        return jnp.array([[haa, t * f], [t * jnp.conj(f), hbb]])

    return hamiltonian


def kane_mele(t: float = 1.0, soc: float = 0.05, mass: float = 0.0):
    """Kane and Mele's honeycomb model (PRL 95, 226801 (2005)).

    Two time-reversed Haldane copies, one per spin. Z2-nontrivial while
    ``|mass| < 3 sqrt(3) soc`` -- the critical mass ``pyqula``'s
    ``examples/2d/z2_transition`` scans across -- and trivial beyond it.
    A sublattice imbalance breaks inversion, so this model has no parity
    invariant and the Wilson loop is the only route to its Z2.
    """
    up = haldane(t=t, t2=soc, phi=np.pi / 2, mass=mass)
    down = haldane(t=t, t2=soc, phi=-np.pi / 2, mass=mass)

    def hamiltonian(k):
        block = jnp.zeros((4, 4), dtype=complex)
        block = block.at[:2, :2].set(up(k))
        return block.at[2:, 2:].set(down(k))

    return hamiltonian


def kane_mele_critical_mass(soc: float) -> float:
    """``3 sqrt(3) soc``, where the Kane-Mele gap closes at K and K'."""
    return 3.0 * np.sqrt(3.0) * soc


def qwz(mass: float):
    """The Qi-Wu-Zhang two-band Chern insulator (PRB 74, 085308 (2006))."""
    def hamiltonian(k):
        k1, k2 = TWOPI * k[0], TWOPI * k[1]
        return (
            jnp.sin(k1) * PAULI_X
            + jnp.sin(k2) * PAULI_Y
            + (mass - jnp.cos(k1) - jnp.cos(k2)) * PAULI_Z
        )

    return hamiltonian


def doubled_qwz(mass: float):
    """Two time-reversed copies of :func:`qwz` -- an Sz-conserving 2D Z2 model.

    ``elkpy``'s ``test_wilson_gauge_invariance.py`` uses exactly this to tie the
    Z2 to a spin Chern number: ``nu = C_up mod 2``. It has an inversion centre,
    ``diag(1, -1)`` in each sector, so it is also the cheapest case on which the
    Wilson-loop and Fu-Kane parity routes can be made to disagree if either is
    wrong.
    """
    up = qwz(mass)

    def hamiltonian(k):
        block = jnp.zeros((4, 4), dtype=complex)
        block = block.at[:2, :2].set(up(k))
        return block.at[2:, 2:].set(jnp.conj(up(k)))

    return hamiltonian


#: The inversion representation of :func:`doubled_qwz` on its four-dimensional
#: basis: ``sigma_z`` in each spin sector, since
#: ``sigma_z h(k) sigma_z = h(-k)``.
DOUBLED_QWZ_INVERSION = np.diag([1.0, -1.0, 1.0, -1.0]).astype(complex)


def wilson_fermion_3d(mass: float):
    """The three-dimensional lattice Dirac (Wilson-fermion) topological insulator.

    Four bands on a simple cubic lattice,

        H(k) = sum_i sin(k_i) Gamma_i + (m + sum_i cos(k_i)) Gamma_4,

    with ``Gamma_i = tau_x (x) sigma_i`` and ``Gamma_4 = tau_z (x) 1`` -- the
    standard regularisation of a 3D Dirac Hamiltonian (Qi, Hughes and Zhang,
    PRB 78, 195424 (2008)). It is the cheapest system with all four Z2 indices,
    an inversion centre (``P = Gamma_4``) and time-reversal symmetry, which is
    what makes it the case where the Wilson sweep over six planes and the
    eight-point parity product can be compared.

    Its invariants are known in closed form, which is why it is here rather than
    a more realistic model. At a TRIM every ``sin k_i`` vanishes, so
    ``H = epsilon Gamma_4`` with ``epsilon = m + sum_i cos k_i``, the occupied
    pair has parity ``-sign(epsilon)``, and

        delta(k) = -sign(m + sum_i cos k_i).

    Multiplying those out: ``(0; 000)`` for ``|m| > 3``, ``(1; 000)`` for
    ``-3 < m < -1``, ``(0; 111)`` for ``-1 < m < 1`` and ``(1; 111)`` for
    ``1 < m < 3``, with the gap closing at ``m = -3, -1, 1, 3``.
    """
    gamma = [
        np.kron([[0, 1], [1, 0]], np.asarray(PAULI_X)),
        np.kron([[0, 1], [1, 0]], np.asarray(PAULI_Y)),
        np.kron([[0, 1], [1, 0]], np.asarray(PAULI_Z)),
    ]
    gamma4 = np.kron([[1, 0], [0, -1]], np.eye(2))
    gamma = jnp.asarray(np.stack(gamma))
    gamma4 = jnp.asarray(gamma4)

    def hamiltonian(k):
        angles = TWOPI * jnp.asarray(k)
        return (
            jnp.einsum("i,iab->ab", jnp.sin(angles), gamma)
            + (mass + jnp.sum(jnp.cos(angles))) * gamma4
        )

    return hamiltonian


#: Inversion for :func:`wilson_fermion_3d`: ``Gamma_4``, which anticommutes with
#: the three ``Gamma_i`` and so sends ``H(k)`` to ``H(-k)``.
WILSON_FERMION_INVERSION = np.kron([[1, 0], [0, -1]], np.eye(2)).astype(complex)


def wilson_fermion_indices(mass: float) -> tuple[int, tuple[int, int, int]]:
    """``(nu0, (nu1, nu2, nu3))`` of :func:`wilson_fermion_3d`, in closed form."""
    deltas = {}
    for a in (0.0, 0.5):
        for b in (0.0, 0.5):
            for c in (0.0, 0.5):
                total = mass + sum(np.cos(TWOPI * x) for x in (a, b, c))
                deltas[(a, b, c)] = -int(np.sign(total))
    nu0 = 0 if int(np.prod(list(deltas.values()))) == 1 else 1
    weak = []
    for axis in range(3):
        product = int(np.prod([v for k, v in deltas.items() if k[axis] == 0.5]))
        weak.append(0 if product == 1 else 1)
    return nu0, tuple(weak)


def random_gauge(key_seed: int, shape, unitary: bool = False):
    """Random per-k gauge transformations, for the invariance tests.

    ``unitary=False`` gives a diagonal phase per state -- the freedom an
    eigensolver has for a nondegenerate band. ``unitary=True`` gives a full
    unitary mixing of the manifold, which is the freedom it has inside a
    degenerate one, and which only a determinant-based construction survives.
    """
    rng = np.random.default_rng(key_seed)
    nk, nbnd = shape
    if not unitary:
        return np.exp(2j * np.pi * rng.random((nk, nbnd)))[:, :, None]
    out = np.zeros((nk, nbnd, nbnd), dtype=complex)
    for i in range(nk):
        matrix = rng.normal(size=(nbnd, nbnd)) + 1j * rng.normal(size=(nbnd, nbnd))
        q, r = np.linalg.qr(matrix)
        out[i] = q * (np.diag(r) / np.abs(np.diag(r)))
    return out


def nonorthogonal(hamiltonian, s: float = 0.25, tau=None):
    """An orthonormal model rewritten in a **non-orthogonal** basis of its own.

    Returns ``(h, overlap, frame)``, three functions of a crystal k-point. With
    ``A(k)`` any smooth invertible matrix, setting ``H = A^dagger H_0 A`` and
    ``S = A^dagger A`` gives a generalised eigenproblem ``H c = e S c`` whose
    eigenvalues are ``H_0``'s and whose eigenvectors are ``c = A^{-1} v``. So the
    *physical* states are unchanged and their Berry curvature is exactly
    ``H_0``'s -- computable at the same k-points, with no mesh error and no sum
    to truncate.

    **This is the model an ultrasoft dataset is**, which is the point of it.
    There ``S = T^dagger T`` with ``T`` carrying ``beta^k``, the true states are
    ``T c``, and the Berry connection is

        <Psi_n|d Psi_m> = c_n^dagger S d c_m + c_n^dagger T^dagger (d T) c_m,

    the second piece being the augmentation dipole. ``A`` plays ``T`` here, so a
    curvature built from ``dH`` and ``dS`` alone must miss ``H_0``'s by exactly
    that second piece and nothing else -- which is what makes the omission
    measurable rather than arguable.

    ``frame`` returns ``A(k)`` itself, because **the correction is a property of
    ``A`` and not of ``S``**: replacing ``A`` by ``U(k) A`` with ``U`` unitary
    leaves ``S = A^dagger A`` untouched and moves the physical states. That is
    the same statement as an ultrasoft code needing ``dpqq`` rather than only
    ``dS/dk``.

    Args:
        hamiltonian: the orthonormal model, ``H_0(k)``.
        s: how far from orthonormal. ``A = 1 + s M(k)`` with ``M`` Hermitian and
            bounded by 1 in norm, so anything below 1 keeps ``A`` invertible;
            0.25 makes ``dS/dk`` the same order as ``dH/dk`` without pushing the
            conditioning anywhere interesting.
        tau: the phase the k-dependence of ``M`` is built from, a 3-vector in
            crystal coordinates. Defaults to a generic one, on purpose: a
            high-symmetry choice makes ``M`` commute with ``H_0`` at special
            k-points and the missing term vanish exactly there.
    """
    tau = np.array([0.31, 0.17, 0.0]) if tau is None else np.asarray(tau, dtype=float)

    def frame(k):
        k = jnp.asarray(k)
        phase = jnp.exp(1j * TWOPI * jnp.dot(jnp.asarray(tau), k))
        size = hamiltonian(k).shape[0]
        # Hermitian, periodic in k, and not diagonal -- a diagonal M commutes
        # with nothing in particular but leaves the off-diagonal block of
        # ``A^dagger (dA)`` zero, which is the block the missing term lives in.
        off = jnp.zeros((size, size), dtype=complex)
        rows = jnp.arange(size - 1)
        off = off.at[rows, rows + 1].set(phase)
        off = off.at[rows + 1, rows].set(jnp.conj(phase))
        diagonal = jnp.diag(jnp.cos(TWOPI * k[0]) * jnp.ones(size))
        return jnp.eye(size, dtype=complex) + s * (off + diagonal)

    def h(k):
        a = frame(k)
        return a.conj().T @ hamiltonian(k) @ a

    def overlap(k):
        a = frame(k)
        return a.conj().T @ a

    return h, overlap, frame
