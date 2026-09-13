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

__all__ = ["ultracell_density"]


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
