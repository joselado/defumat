"""Crystal symmetry: finding the operations and using them to symmetrise.

Two distinct jobs, and the first is easier than it looks:

* **The point group of the lattice** is every integer matrix that preserves the
  metric ``a_i . a_j``. Rather than testing a fixed catalogue of 48 rotations in
  a canonical frame (QE's ``symm_base.f90``), this searches for lattice vectors
  of the right lengths and mutual angles, which works for any cell without
  needing it in a standard orientation.
* **The space group of the crystal** keeps only those operations that also map
  atoms onto atoms of the same species, possibly after a fractional
  translation. Diamond silicon is the canonical example of why the translation
  is needed: its two atoms are related by an operation with ``f = (1/4,1/4,1/4)``,
  and a code that only looks for symmorphic operations finds half the group.

**Why this is needed before the SCF can match QE.** With a symmetry-reduced
k-point set -- which is what almost every QE input uses, including the two
special points in ``pw_scf/scf.in`` -- the density built from those k-points
alone is *not* symmetric. QE restores the symmetry explicitly (``symme.f90``).
Without that step the calculation converges happily to a density of the wrong
symmetry, and degenerate levels split by a few tens of meV.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from pypresso.basis.gvectors import GVectors
from pypresso.system.cell import Cell
from pypresso.system.structure import Structure

__all__ = ["Symmetries", "lattice_point_group", "find_symmetries", "symmetrize_density"]

_TOLERANCE = 1.0e-6


class Symmetries(eqx.Module):
    """The space group of a crystal, in crystal coordinates.

    ``rotations[s]`` is the integer matrix ``M`` defined by its action on the
    basis vectors, ``S a_i = sum_j M_ij a_j``. Two consequences follow from that
    definition and are easy to get backwards:

    * a position in crystal coordinates transforms as ``c' = c M`` (row vector);
    * Miller indices transform as ``m' = M m``, i.e. ``m' = m M^T`` as rows.

    ``translations[s]`` is the fractional translation, also in crystal
    coordinates. Both are stored as nested tuples so they are hashable and can
    be static fields: the symmetry group is a property of the structure, decided
    once, and changing it must retrace.
    """

    rotations: tuple = eqx.field(static=True)
    translations: tuple = eqx.field(static=True)

    @property
    def nsym(self) -> int:
        return len(self.rotations)

    @property
    def symmorphic(self) -> bool:
        return bool(np.allclose(self.translation_array(), 0.0, atol=_TOLERANCE))

    def rotation_array(self) -> np.ndarray:
        return np.array(self.rotations, dtype=int)

    def translation_array(self) -> np.ndarray:
        return np.array(self.translations, dtype=float)

    def __len__(self) -> int:
        return self.nsym


def lattice_point_group(at: np.ndarray) -> list[np.ndarray]:
    """Every integer matrix that maps the lattice onto itself preserving lengths.

    ``M`` is returned in the direct-lattice basis: ``S a_i = sum_j M_ij a_j``.
    The search looks for lattice vectors with the same length as each ``a_i``
    and keeps the triples whose mutual dot products reproduce the metric, which
    is exactly the condition for the linear map taking ``a_i`` to them to be an
    isometry.
    """
    at = np.asarray(at, dtype=float)
    metric = at @ at.T

    # Candidate images of each basis vector: lattice vectors of the same length.
    ranges = np.arange(-3, 4)
    i, j, k = np.meshgrid(ranges, ranges, ranges, indexing="ij")
    integers = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1)
    vectors = integers @ at
    lengths = np.linalg.norm(vectors, axis=1)

    candidates = [
        integers[np.abs(lengths - np.linalg.norm(at[axis])) < _TOLERANCE] for axis in range(3)
    ]

    operations = []
    for first in candidates[0]:
        for second in candidates[1]:
            if abs(first @ metric @ second - metric[0, 1]) > _TOLERANCE:
                continue
            for third in candidates[2]:
                if abs(first @ metric @ third - metric[0, 2]) > _TOLERANCE:
                    continue
                if abs(second @ metric @ third - metric[1, 2]) > _TOLERANCE:
                    continue
                rotation = np.array([first, second, third], dtype=int)
                if abs(abs(round(np.linalg.det(rotation))) - 1) > _TOLERANCE:
                    continue
                operations.append(rotation)
    return operations


def find_symmetries(cell: Cell, structure: Structure) -> Symmetries:
    """The space group operations of the crystal.

    An operation survives if some fractional translation maps every atom onto an
    atom of the same species. Candidate translations are the differences between
    the image of the first atom and every atom of its species -- if any
    translation works, one of those does.
    """
    at = np.asarray(cell.at)
    positions = np.asarray(structure.positions_crystal(cell)) % 1.0
    types = np.asarray(structure.types)

    rotations, translations = [], []
    for rotation in lattice_point_group(at):
        # With M defined by S a_i = sum_j M_ij a_j, a position in crystal
        # coordinates transforms as c' = c M. Transposing here silently keeps
        # only the operations that happen to be symmetric, which for diamond
        # silicon is 12 of the 48.
        rotated = positions @ rotation

        for candidate in _candidate_translations(rotated, positions, types):
            if _maps_structure(rotated + candidate, positions, types):
                rotations.append(rotation)
                translations.append(np.round(candidate, 10) % 1.0)
                break

    return Symmetries(
        # Plain Python ints and floats, not NumPy scalars: these are static
        # fields, and equinox rightly objects to array-like values there.
        rotations=tuple(tuple(tuple(int(v) for v in row) for row in r) for r in rotations),
        translations=tuple(tuple(float(v) for v in t) for t in translations),
    )


def _candidate_translations(rotated, positions, types):
    """Translations that at least map the first atom onto one of its own kind."""
    same_species = np.flatnonzero(types == types[0])
    return [positions[target] - rotated[0] for target in same_species]


def _maps_structure(rotated, positions, types) -> bool:
    """Whether every rotated position coincides with an atom of the same species."""
    difference = rotated[:, None, :] - positions[None, :, :]
    difference -= np.rint(difference)  # modulo a lattice translation
    matches = np.all(np.abs(difference) < _TOLERANCE, axis=-1)
    matches &= types[:, None] == types[None, :]
    return bool(np.all(matches.any(axis=1)))


def symmetrize_density(
    rho_g: jnp.ndarray,
    gvectors: GVectors,
    symmetries: Symmetries,
    maps=None,
) -> jnp.ndarray:
    """Average the density over the space group, in G space.

    Under ``r -> S r + f`` the transform picks up a phase:

        rho_sym(G) = (1/N) sum_S e^{-i G . f_S} rho(S^T G)

    and in Miller indices ``S^T G`` is ``R^T m``, an integer operation -- so the
    whole thing is a permutation of the G list with phases, precomputed once.
    """
    if symmetries.nsym <= 1:
        return rho_g

    permutations, phases = maps if maps is not None else symmetry_maps(gvectors, symmetries)
    # One batched gather rather than a Python loop over operations: with 48
    # operations the loop dispatches ~150 tiny kernels per call, which costs
    # far more than the arithmetic on a 1459-element array.
    return jnp.mean(phases * rho_g[permutations], axis=0)


def symmetry_maps(gvectors: GVectors, symmetries: Symmetries):
    """For each operation, the G-index permutation and the translation phases.

    Returns ``(nsym, ngm)`` arrays. Built once with NumPy -- integer bookkeeping
    over a fixed G list, the definition of setup work. Callers that symmetrise
    repeatedly should hold on to the result rather than rebuilding it, which is
    what :class:`pypresso.scf.driver.Calculation` does.
    """
    miller = np.asarray(gvectors.miller)
    lookup = {tuple(m): index for index, m in enumerate(miller)}

    permutations, phases = [], []
    for rotation, translation in zip(symmetries.rotation_array(), symmetries.translation_array()):
        rotated = miller @ rotation.T  # m' = M m, written for row vectors
        try:
            permutation = np.array([lookup[tuple(m)] for m in rotated])
        except KeyError as error:  # pragma: no cover - would mean a non-symmetry
            raise ValueError(
                "a symmetry operation maps a G-vector outside the cutoff sphere; "
                "the operation is not a symmetry of the reciprocal lattice"
            ) from error
        permutations.append(permutation)
        phases.append(np.exp(-2j * np.pi * (miller @ translation)))
    return jnp.asarray(np.array(permutations)), jnp.asarray(np.array(phases))
