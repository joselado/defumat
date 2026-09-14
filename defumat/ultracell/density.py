"""The ultracell density: one transform per occupied ultracell state.

``PLAN.md`` P88. An ultracell state at ``k0`` is

    Psi_j(r) = sum_{Q,n} a^j_{Q,n} psi_{k0+Q,n}(r),

and in a plane-wave basis that sum is a **scatter**: ``psi_{k0+Q,n}`` is a list
of coefficients on the ``k0+Q`` sphere, and putting each of them at
``J = n G + q`` in the ultracell box (:mod:`defumat.ultracell.grid`) assembles
the whole state as one ultracell plane-wave vector. One inverse transform then
gives ``Psi_j`` on the ultracell real-space grid, and the density is the
occupation-weighted square, accumulated over states.

That is the whole routine, and it is worth saying what it replaces. Elk's
``rhomaguk`` cannot do this, because in LAPW there is no single coefficient
vector to scatter: it Fourier transforms the ``Q``-amplitudes to get the
envelope cell by cell, and then rebuilds the wavefunction in *every* cell ``R``
out of the central k-point's states. That is the same physical picture --
``rhomaguk`` is where the envelope becomes visible as the Fourier transform of
the ``Q`` amplitudes -- with an extra approximation in it and, on a plane-wave
basis, more work rather than less.

**Normalisation, which the ``N = 1`` null cannot check.** The basis functions
are normalised over the unit cell, so an ultracell state built with
``sum |a|^2 = 1`` is normalised over the unit cell too, and its contribution to
the density is ``w |Psi|^2 / Omega_cell`` with ``w = w_k0 f_j / N`` -- the same
volume the ordinary :func:`~defumat.scf.density.band_density` divides by, and
**not** the ultracell's. The density that comes out is then per unit cell at
every point, so it integrates to ``N`` times the electron count over the
ultracell, and it equals the unit cell's own density exactly when nothing is
modulated. Dividing by ``Omega_u`` instead is wrong by ``N`` and is invisible at
``N = 1``: what catches it is asserting the *tiled* density at ``N > 1``.
"""

from __future__ import annotations

import jax.numpy as jnp

from defumat.batching import sum_k

__all__ = ["ultracell_density", "spinor_ultracell_density"]


def ultracell_density(
    coefficients: jnp.ndarray,
    vectors: jnp.ndarray,
    weights: jnp.ndarray,
    box_index: jnp.ndarray,
    grid: tuple[int, int, int],
    volume: float,
    batch: int | None = 1,
) -> jnp.ndarray:
    """The density on the ultracell box from one ``k0``'s ultracell states.

    Args:
        coefficients: ``(N, nbnd, npwx)`` frozen states at ``k0 + Q``.
        vectors: ``(N nbnd, nstate)`` ultracell eigenvectors, column ``j`` the
            amplitudes ``a^j_{Q,n}`` in the ``(Q, n)`` C-order
            :func:`~defumat.ultracell.hamiltonian.ultracell_matrix` uses.
        weights: ``(nstate,)`` ``w_k0 f_j / N``.
        box_index: ``(N, npwx)`` flat ultracell box index.
        grid: the ultracell box shape.
        volume: the **unit cell** volume in bohr^3.
        batch: ultracell states in flight at once; one by default, because each
            holds a whole ultracell box.

    Returns ``(*grid)`` real.
    """
    cells, nbnd, _ = coefficients.shape
    points = int(grid[0]) * int(grid[1]) * int(grid[2])
    flat_index = box_index.reshape(-1)

    def one_state(state):
        amplitudes = state["a"].reshape(cells, nbnd)
        # Collapse the band index first: the Q-th sphere's contribution is one
        # coefficient vector, so the scatter below moves N npwx numbers rather
        # than N nbnd npwx of them.
        mixed = jnp.einsum("qn,qnp->qp", amplitudes, coefficients)
        scattered = jnp.zeros(points, dtype=mixed.dtype)
        scattered = scattered.at[flat_index].add(mixed.reshape(-1))
        field = jnp.fft.ifftn(scattered.reshape(grid)) * points
        # ``Re(conj f f)`` and not ``abs(f)**2``: a wavefunction has nodes on
        # grid points by symmetry and ``abs`` has no derivative at zero, which
        # is :func:`defumat.scf.density.band_density`'s trap in a second place.
        return state["w"] * jnp.real(jnp.conj(field) * field)

    total = sum_k(one_state, {"a": vectors.T, "w": weights}, batch=batch)
    return total / volume


def spinor_ultracell_density(
    coefficients: jnp.ndarray,
    vectors: jnp.ndarray,
    weights: jnp.ndarray,
    box_index: jnp.ndarray,
    grid: tuple[int, int, int],
    volume: float,
    nspin_mag: int = 4,
    batch: int | None = 1,
) -> jnp.ndarray:
    """``(nspin_mag, *box)`` from one ``k0``'s **spinor** ultracell states.

    :func:`ultracell_density` with a component axis, and it is a separate
    function for the same reason
    :func:`defumat.scf.density.spinor_band_density` is separate from
    :func:`defumat.scf.density.band_density`: a collinear run builds *one*
    channel's density at a time and the caller knows which, while a spinor state
    produces all four components of ``(n, m_x, m_y, m_z)`` at once, because they
    are four bilinears in the same pair of transformed components.

    Args:
        coefficients: ``(N, nbnd, 2 npwx)`` frozen spinor states at ``k0 + Q``.
        vectors: ``(N nbnd, nstate)`` ultracell eigenvectors, column ``j`` the
            amplitudes in the ``(Q, n)`` C-order
            :func:`~defumat.ultracell.hamiltonian.ultracell_matrix` uses.
        weights: ``(nstate,)`` ``w_k0 f_j / N``. A spinor state holds **one**
            electron, which is in the k-point weights rather than here --
            :func:`~defumat.system.kpoints.for_spin`, and it is the trap
            ``CLAUDE.md`` lists for every caller-built k-set.
        nspin_mag: 4 for a magnetic run, 1 for a spin-orbit run that carries no
            magnetization -- where the charge is the only component and every
            routine above this one runs as though the calculation were scalar.

    The Pauli convention is ``spinor_band_density``'s and is not restated here:
    ``m_x = 2 Re(conj(u) d)``, ``m_y = 2 Im(conj(u) d)``, ``m_z = |u|^2 -
    |d|^2``. A sign on ``m_y`` is a state of the opposite chirality and is
    degenerate with the right one wherever spin-orbit coupling is off, so
    nothing in an energy or a symmetry check can see it -- which is why the
    convention is taken from the one place that already has a ``pw.x`` number
    behind it rather than rewritten.
    """
    cells, nbnd, width = coefficients.shape
    npwx = width // 2
    points = int(grid[0]) * int(grid[1]) * int(grid[2])
    flat_index = box_index.reshape(-1)
    parts = coefficients.reshape(cells, nbnd, 2, npwx)

    def one_state(state):
        amplitudes = state["a"].reshape(cells, nbnd)
        mixed = jnp.einsum("qn,qnap->qap", amplitudes, parts)
        scattered = jnp.zeros((2, points), dtype=mixed.dtype)
        scattered = scattered.at[:, flat_index].add(
            mixed.transpose(1, 0, 2).reshape(2, -1)
        )
        field = jnp.fft.ifftn(
            scattered.reshape((2,) + tuple(grid)), axes=(-3, -2, -1)
        ) * points
        up, down = field[0], field[1]
        # ``Re(conj(z) z)`` rather than ``abs(z)**2``, the trap
        # :func:`ultracell_density` names and the same one a second time.
        up_density = jnp.real(jnp.conj(up) * up)
        down_density = jnp.real(jnp.conj(down) * down)
        charge = up_density + down_density
        if nspin_mag == 1:
            return state["w"] * charge[None]
        cross = jnp.conj(up) * down
        return state["w"] * jnp.stack([
            charge,
            2.0 * jnp.real(cross),
            2.0 * jnp.imag(cross),
            up_density - down_density,
        ])

    total = sum_k(one_state, {"a": vectors.T, "w": weights}, batch=batch)
    return total / volume
