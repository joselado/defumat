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

from typing import TYPE_CHECKING

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from pypresso.system.cell import Cell
from pypresso.system.structure import Structure

if TYPE_CHECKING:  # only for annotations: importing it eagerly makes a cycle,
    from pypresso.basis.gvectors import GVectors  # basis -> system -> basis

__all__ = [
    "is_supercell",
    "atom_mapping",
    "cartesian_rotations",
    "harmonic_rotations","Symmetries", "lattice_point_group", "find_symmetries", "symmetrize_density",
           "symmetry_maps", "apply_symmetry_maps"]

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

    def fft_factors(self) -> tuple[int, int, int]:
        """The FFT dimensions must be a multiple of these (``symm_base.f90``).

        A fractional translation of ``1/n`` along an axis maps a grid onto
        itself only if the grid has a multiple of ``n`` points along that axis.
        QE takes the least common multiple over every non-symmorphic operation
        and forces the FFT dimensions to be divisible by it -- which is why
        diamond silicon's grids come out at 16 and 32 rather than the 15 and 30
        the cutoff alone would ask for.

        It is not optional decoration. It changes ``etxc``, which is evaluated
        pointwise on the grid, in the sixth decimal; matching QE's grid is the
        difference between agreeing with it to 1e-6 Ry and to 1e-9.
        """
        factors = [1, 1, 1]
        for translation in self.translations:
            for axis, value in enumerate(translation):
                # ft is stored folded into [0, 1); a translation of 3/4 is the
                # same grid constraint as one of 1/4.
                value = min(abs(value), abs(1.0 - value))
                if value <= _TOLERANCE:
                    continue
                n = int(round(1.0 / value))
                # QE only accepts translations that are a simple fraction; a
                # rotation needing anything else is discarded before this point.
                if n and abs(1.0 / value - n) < 1.0e-4:
                    factors[axis] = _lcm(factors[axis], n)
        return tuple(factors)

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


def is_supercell(cell: Cell, structure: Structure) -> bool:
    """Whether the cell is a supercell of a smaller one (``symm_base.f90``).

    The test is QE's: if the *identity* rotation combined with some non-lattice
    translation maps the structure onto itself, the cell contains more than one
    formula unit of a smaller cell. QE then **disables fractional translations
    entirely** -- every non-symmorphic operation is discarded, and the FFT grid
    loses its divisibility constraint with them.

    That looks like throwing away real symmetry, and it is; the justification in
    the Fortran is that a supercell's non-symmorphic operations are artefacts of
    the choice of cell rather than of the crystal. It has to be reproduced
    regardless, because it changes what QE computes: the eight-atom cubic cell
    of diamond silicon keeps 24 operations rather than 48 and gets a 45^3 grid
    rather than 48^3, and with a k-point set that is not itself symmetric the
    two symmetrisations give densities differing in the fifth decimal of the
    energy.
    """
    positions = np.asarray(structure.positions_crystal(cell)) % 1.0
    types = np.asarray(structure.types)
    if len(positions) < 2:
        return False

    for candidate in positions[1:] - positions[0]:
        candidate = candidate - np.rint(candidate)
        if np.all(np.abs(candidate) < _TOLERANCE):
            continue
        if _maps_structure(positions + candidate, positions, types):
            return True
    return False


def find_symmetries(cell: Cell, structure: Structure) -> Symmetries:
    """The space group operations of the crystal.

    An operation survives if some fractional translation maps every atom onto an
    atom of the same species. Candidate translations are the differences between
    the image of the first atom and every atom of its species -- if any
    translation works, one of those does.

    Fractional translations are dropped altogether when the cell turns out to be
    a supercell; see :func:`is_supercell`.
    """
    at = np.asarray(cell.at)
    positions = np.asarray(structure.positions_crystal(cell)) % 1.0
    types = np.asarray(structure.types)
    symmorphic_only = is_supercell(cell, structure)

    rotations, translations = [], []
    for rotation in lattice_point_group(at):
        # With M defined by S a_i = sum_j M_ij a_j, a position in crystal
        # coordinates transforms as c' = c M. Transposing here silently keeps
        # only the operations that happen to be symmetric, which for diamond
        # silicon is 12 of the 48.
        rotated = positions @ rotation

        for candidate in _candidate_translations(rotated, positions, types):
            if symmorphic_only and np.any(
                np.abs(candidate - np.rint(candidate)) > _TOLERANCE
            ):
                continue
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
    gvectors: "GVectors",
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
    return apply_symmetry_maps(rho_g, permutations, phases)


def apply_symmetry_maps(rho_g: jnp.ndarray, permutations, phases) -> jnp.ndarray:
    """The averaging itself, given precomputed maps -- pure JAX, safe to ``jit``.

    One batched gather rather than a Python loop over operations: with 48
    operations the loop dispatches ~150 tiny kernels per call, which costs far
    more than the arithmetic on a 1459-element array. Split out from
    :func:`symmetrize_density` because that function's arguments include the
    :class:`Symmetries` object, whose rotations are static NumPy arrays and so
    cannot cross a ``jit`` boundary.
    """
    return jnp.mean(phases * rho_g[permutations], axis=0)


def symmetry_maps(gvectors: "GVectors", symmetries: Symmetries):
    """For each operation, the G-index permutation and the translation phases.

    Returns ``(nsym, ngm)`` arrays. Built once with NumPy -- integer bookkeeping
    over a fixed G list, the definition of setup work. Callers that symmetrise
    repeatedly should hold on to the result rather than rebuilding it, which is
    what :class:`pypresso.scf.driver.Calculation` does.
    """
    miller = np.asarray(gvectors.miller)
    grid = np.asarray(gvectors.grid)

    # Look the rotated indices up through the FFT box rather than through a
    # Python dict: every G in the sphere has a distinct residue modulo the grid
    # (that is what makes the box big enough to hold the sphere), so wrapping the
    # Miller index into the box is an injective key and one array indexing
    # replaces 48 x ngm dictionary probes.
    lookup = np.full(tuple(grid), -1, dtype=np.int64)
    wrapped = tuple((miller % grid).T)
    lookup[wrapped] = np.arange(len(miller))

    rotations = symmetries.rotation_array()
    translations = symmetries.translation_array()

    rotated = np.einsum("gc,sdc->sgd", miller, rotations)  # m' = M m, row vectors
    permutations = lookup[tuple((rotated % grid).transpose(2, 0, 1))]
    if np.any(permutations < 0):  # pragma: no cover - would mean a non-symmetry
        raise ValueError(
            "a symmetry operation maps a G-vector outside the cutoff sphere; "
            "the operation is not a symmetry of the reciprocal lattice"
        )
    phases = np.exp(-2j * np.pi * (miller @ translations.T)).T
    return jnp.asarray(permutations), jnp.asarray(phases)


def _lcm(a: int, b: int) -> int:
    """QE's ``mcm``, with 0 meaning "no constraint"."""
    if a == 0 or b == 0:
        return max(a, b)
    return abs(a * b) // math.gcd(a, b)


def atom_mapping(cell: Cell, structure: Structure, symmetries: Symmetries) -> np.ndarray:
    """``irt[s, a]``: the atom that operation ``s`` sends atom ``a`` to.

    ``sgam_at`` in ``symm_base.f90`` builds the same table while it is testing
    the operations. Everything that acts on a per-atom quantity -- ``becsum``
    for PAW, forces, the density in real space -- needs it, because an operation
    is only a symmetry of the *crystal*, not of any individual atom.
    """
    positions = np.asarray(structure.positions_crystal(cell)) % 1.0
    types = np.asarray(structure.types)
    rotations = symmetries.rotation_array()
    translations = symmetries.translation_array()

    mapping = np.zeros((len(rotations), len(positions)), dtype=int)
    for s, (rotation, translation) in enumerate(zip(rotations, translations)):
        images = positions @ rotation + translation
        for a, image in enumerate(images):
            difference = image[None, :] - positions
            difference -= np.rint(difference)
            matches = np.flatnonzero(
                (np.abs(difference) < _TOLERANCE).all(axis=1) & (types == types[a])
            )
            if matches.size != 1:
                raise AssertionError(
                    f"symmetry operation {s} does not map atom {a} onto exactly one atom"
                )
            mapping[s, a] = matches[0]
    return mapping


def cartesian_rotations(cell: Cell, symmetries: Symmetries) -> np.ndarray:
    """The symmetry rotations as cartesian matrices acting on column vectors.

    With ``S a_i = sum_j M_ij a_j`` and the lattice vectors as the rows of
    ``A``, that reads ``M A = A R^T``, so ``R = (A^-1 M A)^T``.
    """
    at = np.asarray(cell.at, dtype=float)
    inverse = np.linalg.inv(at)
    return np.array([(inverse @ m @ at).T for m in symmetries.rotation_array()])
