"""A frozen basis closed under time reversal: the reference's states and their Kramers partners.

``PLAN.md`` P121, ``OPEN.md`` Part XX. The ultracell is expanded in the frozen
states of *one* reference, and a magnetic reference has a direction: without
spin-orbit coupling its spinors are eigenstates of ``sigma . e_0``, majority
orbitals with spin along ``+e_0`` and minority ones against it. Turning a
moment off ``e_0`` needs a majority orbital with the *other* spinor, which that
set does not contain, so a truncated solve prefers ``+e_0`` and leans toward
it. On four cells of hydrogen at ``nbnd = 16`` that is a cone of 10.64 degrees
with the reference along the helix axis, and a distorted helix (steps of 94,
74 and 84 degrees where 90 was seeded, and a uniform moment along the
reference) with the reference in the helix plane; in NiBr2 it is a uniform
in-plane moment of 0.32 of the helix on three cells at ``nbnd = 40`` and 0.144
on fifteen at 96, where Elk's supercell has 9e-5.

**The fix is to add each state's Kramers partner**, which removes the
preference between ``+e_0`` and ``-e_0`` because time reversal maps one onto
the other. The partners of the states at ``k`` are the time-reversed states at
``-k``, and those are the eigenstates at ``k`` of the reference with its
magnetization reversed, since ``Theta H[m] Theta^-1 = H[-m]`` with spin-orbit
coupling or without it. So the partners come from a second fixed-density solve
at the reversed density, and the reversed ``becsum`` on a PAW or ultrasoft
dataset; nothing has to be mapped from ``-k`` to ``k``. Without spin-orbit
coupling the closed span is the orbitals times both spinors, which is invariant
under every global spin rotation, so no frame is preferred at all.

**The two sets are not orthogonal, and they are nearly dependent.** A band
with little exchange splitting has nearly the same orbital in both channels, so
its partner nearly duplicates a state already there: on the hydrogen cell the
union's overlap has smallest eigenvalue 1.6e-7 at ``nbnd = 16`` and 1.1e-9 at
64, and half of its eigenvalues are below 1e-2. The new directions are the
small differences between the majority and the minority orbitals, and they are
what the fix consists of: dropping everything below an overlap of 1e-2 keeps
the closure (the cone stays zero) and loses the accuracy, 1.96e-4 Ry per cell
above the supercell where a floor of 1e-8 gives 4e-7. So the floor
(:data:`OVERLAP_FLOOR`) exists for arithmetic and not for physics.

**The union is therefore handed to the loop already diagonalised.** At each
folded ``k`` the reference Hamiltonian and the overlap are built on the union
with ``apply`` and ``apply_s``, the overlap's directions below the floor are
dropped (canonical orthogonalisation), and the reference Hamiltonian is
diagonalised in what is left. What comes back is an orthonormal set in which
the reference Hamiltonian is diagonal, which is the only property
:func:`~defumat.ultracell.hamiltonian.ultracell_matrix` asks of its basis, so
nothing downstream changes. It does not ask that the partners be eigenstates of
anything: the matrix is the exact reference Hamiltonian projected onto the
span, and the reversed solve only decides what the span is. The reference's
own states are in it, so its lowest Ritz values are the reference's eigenvalues
and the tiled null still holds.

**A folded k-point that drops directions keeps its slot**: the missing ones
are zero vectors at :data:`SENTINEL_SHIFT` above the highest Ritz value of the
whole set, so every folded k-point carries the same count, which the static
shapes of the loop need. A zero vector couples to nothing, so its level is
exactly the sentinel and it is never occupied; :attr:`KramersBasis.ranks` says
how many are real.

**What it costs.** A second frozen solve, which is the expensive step of the
method, and a basis of up to ``2 nbnd`` per folded k-point: twice the
coefficients and the projections, twice the ket transforms per iteration, four
times the matrix and up to eight times its dense solve. While the union is
being built the reference's states, the partners and the result are all alive,
``4 nk nbnd npol npwx`` complex numbers against the ``nk nbnd npol npwx`` of
the old basis, and after it the loop holds ``2 nk nbnd npol npwx``; on the
hydrogen cell the process peak stayed at 1.6 to 1.7 GB either way, which is the
compiled kernels rather than the states. Measured at equal
basis size it is the better use of the budget by far: 1.3e-6 Ry per cell above
the supercell at ``nbnd = 8`` closed, against 4.37e-4 at ``nbnd = 16`` and
1.56e-4 at ``nbnd = 32`` on the old basis.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

__all__ = [
    "OVERLAP_FLOOR",
    "SENTINEL_SHIFT",
    "KramersBasis",
    "time_reversed",
    "kramers_closed_basis",
]

#: Overlap eigenvalues of the union below this are dropped. Nothing on the
#: hydrogen cell reaches it at ``nbnd = 16`` (smallest 1.6e-7) and the accuracy
#: falls as it is raised -- 4e-7, 1.1e-5 and 1.96e-4 Ry per cell above the
#: supercell at 1e-8, 1e-4 and 1e-2 -- so it guards the arithmetic of
#: ``s^{-1/2}`` and is not a knob.
OVERLAP_FLOOR = 1.0e-8

#: Ry above the highest Ritz value where a dropped direction's zero vector sits.
SENTINEL_SHIFT = 10.0


def time_reversed(density, becsum, nspin_mag: int) -> tuple:
    """The reference with its magnetization reversed: ``(density, becsum)``.

    The charge component is kept and the three magnetization components change
    sign, on the grid and in every species' ``becsum``, which is what time
    reversal does to a density and what makes the reversed Hamiltonian
    ``Theta H Theta^-1``. ``becsum`` is the unit cell's, ``(nspin_mag, nat_t,
    nh, nh)`` per species or ``None``.
    """
    if int(nspin_mag) != 4:
        raise ValueError(
            f"a Kramers partner is a noncollinear object and nspin_mag is "
            f"{nspin_mag}"
        )
    density = jnp.asarray(density)
    reversed_density = density.at[1:].multiply(-1.0)
    reversed_becsum = tuple(
        None if values is None else jnp.asarray(values).at[1:].multiply(-1.0)
        for values in becsum
    )
    return reversed_density, reversed_becsum


@dataclasses.dataclass(frozen=True)
class KramersBasis:
    """The union's Ritz pairs at every folded k-point, padded to one count."""

    #: ``(nk, M)`` Ritz values in Ry, the padded slots at the sentinel.
    eigenvalues: np.ndarray
    #: ``(nk, M, npol npwx)`` orthonormal vectors, the padded slots zero.
    wavefunctions: jnp.ndarray
    #: ``(nk,)`` how many directions each folded k-point kept.
    ranks: np.ndarray
    #: The smallest overlap eigenvalue kept anywhere, the conditioning the
    #: ``s^{-1/2}`` ran at.
    smallest_overlap: float
    #: ``(nk,)`` how many directions each folded k-point dropped.
    dropped: np.ndarray


def kramers_closed_basis(hamiltonian, states, partners, mask) -> KramersBasis:
    """Rayleigh-Ritz of the reference Hamiltonian on ``states`` and ``partners``.

    Args:
        hamiltonian: the reference's
            :class:`~defumat.hamiltonian.noncollinear.SpinorHamiltonian` on the
            folded k-set, the one ``states`` are eigenstates of.
        states: ``(nk, nbnd, 2 npwx)`` the reference's frozen states.
        partners: ``(nk, nbnd, 2 npwx)`` the reversed reference's, on the same
            k-set.
        mask: ``(nk, 2 npwx)`` the plane-wave mask of a spinor, applied to both
            sets before anything is built from them.

    Returns a :class:`KramersBasis`. Done once, before the loop, and on the
    host: it is two applications of the reference Hamiltonian per folded
    k-point, and the dense algebra is ``(2 nbnd)^3``.
    """
    states = np.asarray(states)
    partners = np.asarray(partners)
    mask = np.asarray(mask)
    nk, nbnd, width = states.shape
    ritz, vectors, ranks, smallest = [], [], [], np.inf
    for ik in range(nk):
        union = np.concatenate([states[ik], partners[ik]], axis=0)
        union = np.where(mask[ik][None], union, 0.0)
        trial = jnp.asarray(union)
        s = np.asarray(trial.conj() @ hamiltonian.apply_s(trial, ik).T)
        h = np.asarray(trial.conj() @ hamiltonian.apply(trial, ik).T)
        s = 0.5 * (s + s.conj().T)
        h = 0.5 * (h + h.conj().T)
        values, directions = np.linalg.eigh(s)
        keep = values > OVERLAP_FLOOR
        smallest = min(smallest, float(values[keep].min()))
        transform = directions[:, keep] / np.sqrt(values[keep])
        energies, rotation = np.linalg.eigh(transform.conj().T @ h @ transform)
        ritz.append(energies)
        vectors.append((transform @ rotation).T @ union)
        ranks.append(int(keep.sum()))

    ranks = np.asarray(ranks)
    count = int(ranks.max())
    sentinel = float(max(float(np.max(e)) for e in ritz)) + SENTINEL_SHIFT
    eigenvalues = np.full((nk, count), sentinel)
    wavefunctions = np.zeros((nk, count, width), dtype=states.dtype)
    for ik in range(nk):
        eigenvalues[ik, : ranks[ik]] = ritz[ik]
        wavefunctions[ik, : ranks[ik]] = vectors[ik]
    return KramersBasis(
        eigenvalues=eigenvalues,
        wavefunctions=jnp.asarray(wavefunctions),
        ranks=ranks,
        smallest_overlap=float(smallest),
        dropped=2 * nbnd - ranks,
    )
