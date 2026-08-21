"""The Hubbard occupation matrix ``ns``, and where it comes from.

``PW/src/new_ns.f90``:

    ns^{I s}_{m1 m2} = sum_{k,v} f_{kv} <phi^I_{m1}|psi_{kvs}> <psi_{kvs}|phi^I_{m2}>

with ``phi`` the projectors of :mod:`pypresso.hubbard.projectors`. Three things
about that expression are not obvious from it:

* **``ns`` is per spin channel.** With ``nspin = 1`` the band sum runs over
  doubly-occupied states, so QE halves the result: ``ns`` then means the
  occupation of *one* of the two identical channels, and the energy is doubled
  instead (:mod:`pypresso.hubbard.energy`).
* **It has to be symmetrised**, and for the same reason ``becsum`` does
  (:mod:`pypresso.paw.symmetry`): the band sum runs over the irreducible wedge,
  so what comes out has the symmetry of the wedge rather than of the crystal.
  The average is over the group with each index rotated by the matrices that
  mix real spherical harmonics of the manifold's ``l`` -- the same
  :func:`~pypresso.paw.symmetry.harmonic_rotations` PAW uses, deliberately, so
  that one convention is validated once.
* **It is real and symmetric.** ``Re(p_{m2} conj(p_{m1}))`` is symmetric in
  ``m1, m2`` by construction, so the matrix built directly needs none of the
  hermiticity repair ``new_ns`` performs on its half-filled array.

The layout is ``(nspin, nslot, ldmx, ldmx)``: spin leading (rule R6), one slot
per correlated atom rather than per atom, and manifolds of different ``l``
padded with zeros up to the largest. QE carries ``ns(ldmx,ldmx,nspin,nat)`` --
same content, reversed index order, and every atom present.

``init_ns`` and ``ns_adj`` are here too: the starting occupation from Hund's
rule (``PW/src/init_ns.f90``) and the adjustment of its *eigenvalues* to what
``starting_ns_eigenvalue`` asks for (``PW/src/ns_adj.f90``). The second is not a
convenience -- an antiferromagnet has more than one self-consistent ``ns``, and
it is how a run is steered to the one that is wanted.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from pypresso.batching import map_k

__all__ = [
    "NsSymmetry",
    "adjust_ns",
    "build_ns_symmetry",
    "initial_ns",
    "occupation_matrix",
    "projections",
]


def projections(wfcU: jnp.ndarray, psi: jnp.ndarray, k_batch=1) -> jnp.ndarray:
    """``<wfcU|psi>``: ``(nk, nbnd, nwfcU)`` from ``(nk, nbnd, npwx)`` states."""
    def one(arrays):
        columns, states = arrays
        return jnp.einsum("gi,bg->bi", jnp.conj(columns), states)

    return map_k(one, (wfcU, psi), batch=k_batch)


def occupation_matrix(
    wfcU: jnp.ndarray,
    wavefunctions: jnp.ndarray,
    weights: jnp.ndarray,
    columns: jnp.ndarray,
    mask: jnp.ndarray,
    k_batch=1,
) -> jnp.ndarray:
    """``nr``: the unsymmetrised occupation matrix, ``(nspin, nslot, ldmx, ldmx)``.

    ``columns[slot, m]`` is the position of orbital ``m`` of that slot in
    ``wfcU``, and ``mask[slot, m]`` says whether it is a real orbital or padding
    -- the two together are what let manifolds of different ``l`` share one
    rectangular array. ``weights`` are QE's ``wg``, the k-point weight times the
    occupation.
    """
    nspin = wavefunctions.shape[0]
    blocks = []
    for spin in range(nspin):
        proj = projections(wfcU, wavefunctions[spin], k_batch)  # (nk, nbnd, nwfcU)
        block = proj[..., columns]  # (nk, nbnd, nslot, ldmx)
        block = jnp.where(mask, block, 0.0)
        blocks.append(
            jnp.einsum("kb,kbna,kbnc->nac", weights[spin], jnp.conj(block), block).real
        )
    ns = jnp.stack(blocks)
    # ``IF (nspin == 1) nr = 0.5d0 * nr``: with one channel the band sum already
    # counted both spins.
    return 0.5 * ns if nspin == 1 else ns


class NsSymmetry:
    """The group average over ``ns``, precomputed per species.

    Structurally :class:`pypresso.paw.symmetry.BecsumSymmetry` restricted to a
    single ``(n, l)`` block: for each operation, a rank-four tensor rotating the
    two ``m`` indices and a permutation saying which atom the operation's image
    comes from.
    """

    __slots__ = ("operators", "sources", "slots", "ldims", "nsym", "shape")

    def __init__(self, operators, sources, slots, ldims, nsym, shape):
        self.operators = operators
        self.sources = sources
        self.slots = slots
        self.ldims = ldims
        self.nsym = nsym
        self.shape = shape

    def apply(self, ns: jnp.ndarray) -> jnp.ndarray:
        """The symmetrised occupation matrix, same shape as the input."""
        if self.nsym <= 1:
            return ns
        out = jnp.zeros_like(ns)
        for operator, sources, slots, ldim in zip(
            self.operators, self.sources, self.slots, self.ldims
        ):
            block = ns[:, slots, :ldim, :ldim]  # (nspin, ngroup, ldim, ldim)
            gathered = block[:, sources]  # (nspin, nsym, ngroup, ldim, ldim)
            averaged = jnp.einsum(
                "sikjl,zsnkl->znij", operator, gathered
            ) / self.nsym
            out = out.at[:, slots, :ldim, :ldim].set(averaged)
        return out


def build_ns_symmetry(setup, cell, structure, symmetries) -> NsSymmetry | None:
    """Precompute the average. ``None`` when the group is trivial.

    **Collinear time reversal is not handled.** ``new_ns`` flips the spin index
    of an operation that is a symmetry only together with time reversal
    (``colin_mag == 2``, ``t_rev(isym) == 1``). No such operation can appear in
    the benchmarks this is validated against -- their two magnetic sublattices
    are different *species*, so nothing maps one to the other -- and building
    the branch without a case that exercises it would be writing untested code.
    A run whose symmetry group carries ``t_rev`` is refused where the setup is
    built rather than silently symmetrised without the flip.
    """
    from pypresso.paw.symmetry import harmonic_rotations
    from pypresso.system.symmetry import atom_mapping

    if setup is None or symmetries is None or symmetries.nsym <= 1:
        return None

    lmax = max(setup.species[t].l for t in setup.types)
    rotations = harmonic_rotations(cell, symmetries, lmax)
    mapping = atom_mapping(cell, structure, symmetries)
    slot_of_atom = {atom: slot for slot, atom in enumerate(setup.atoms)}

    operators, sources, groups, ldims = [], [], [], []
    for t in sorted(set(setup.types)):
        slots = [slot for slot, kind in enumerate(setup.types) if kind == t]
        l = setup.species[t].l
        ldim = 2 * l + 1
        # ``D[s, i, k]`` in this project's convention is QE's ``d(k, i, s)`` --
        # source index first in the Fortran, target first here. PAW's
        # symmetrisation fixes that convention and is validated against QE, so
        # reusing it is what keeps the two from disagreeing.
        d = np.asarray(rotations[l])
        operators.append(jnp.asarray(np.einsum("sik,sjl->sikjl", d, d)))
        position = {slot: n for n, slot in enumerate(slots)}
        sources.append(jnp.asarray([
            [position[slot_of_atom[int(mapping[s, setup.atoms[slot]])]]
             for slot in slots]
            for s in range(symmetries.nsym)
        ]))
        groups.append(jnp.asarray(slots))
        ldims.append(ldim)

    return NsSymmetry(
        operators=tuple(operators),
        sources=tuple(sources),
        slots=tuple(groups),
        ldims=tuple(ldims),
        nsym=symmetries.nsym,
        shape=(setup.nslot, setup.ldmx),
    )


def initial_ns(setup, nspin: int, starting_magnetization) -> jnp.ndarray:
    """``init_ns``: the starting occupation matrix, diagonal, from Hund's rule.

    Majority-spin levels are filled first and the remainder is spread equally
    over the minority ones; a species with no starting magnetization gets half
    the reference occupation in each channel. The result is what the first
    Hubbard potential is built from, and for a magnetic insulator it is what
    decides which of several self-consistent solutions the run finds.
    """
    ns = np.zeros((nspin, setup.nslot, setup.ldmx, setup.ldmx))
    magnetization = np.asarray(starting_magnetization, dtype=float)
    for slot, t in enumerate(setup.types):
        item = setup.species[t]
        ldim, total = item.ldim, item.occupation
        moment = magnetization[t] if t < len(magnetization) else 0.0
        if nspin == 2 and moment != 0.0:
            major, minor = (0, 1) if moment > 0.0 else (1, 0)
            if total > ldim:
                for m in range(ldim):
                    ns[major, slot, m, m] = 1.0
                    ns[minor, slot, m, m] = (total - ldim) / ldim
            else:
                for m in range(ldim):
                    ns[major, slot, m, m] = total / ldim
        else:
            for spin in range(nspin):
                for m in range(ldim):
                    ns[spin, slot, m, m] = total / 2.0 / ldim
    return jnp.asarray(ns)


def adjust_ns(ns: jnp.ndarray, setup) -> jnp.ndarray:
    """``ns_adj``: replace the eigenvalues of ``ns`` by the ones asked for.

    ``starting_ns_eigenvalue(m, ispin, ityp)`` names one eigenvalue of one
    channel of one species. The matrix is diagonalised, the named eigenvalues
    are overwritten, and it is rebuilt from the *original* eigenvectors -- so
    the orbital that the level belongs to is whatever the starting matrix says
    it is, which for the diagonal starting matrix of :func:`initial_ns` is one
    of the ``m`` states in the file's order.
    """
    if not setup.starting_ns:
        return ns
    values = np.array(ns)
    for slot, t in enumerate(setup.types):
        ldim = setup.ldims[slot]
        for spin in range(values.shape[0]):
            requested = {
                m: value for (kind, ispin, m), value in setup.starting_ns.items()
                if kind == t and ispin == spin and m < ldim
            }
            if not requested:
                continue
            block = values[spin, slot, :ldim, :ldim]
            eigenvalues, vectors = np.linalg.eigh(block)
            for m, value in requested.items():
                if value >= 0.0:
                    eigenvalues[m] = value
            values[spin, slot, :ldim, :ldim] = (
                vectors * eigenvalues
            ) @ vectors.conj().T
    return jnp.asarray(values)
