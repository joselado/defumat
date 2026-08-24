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

__all__ = ["Symmetries", "lattice_point_group", "find_symmetries", "is_supercell",
           "symmetrize_density", "symmetry_maps", "apply_symmetry_maps",
           "atom_mapping", "cartesian_rotations",
           "symmetrize_vector", "symmetrize_matrix", "symmetrize_atom_tensor",
           "check_symmetry", "check_lattice_symmetry",
           "magnetic_symmetries", "magnetization_signs",
           "symmetrize_magnetization", "symmetrize_vector_density",
           "symmetrize_tensor_density"]

_TOLERANCE = 1.0e-6
#: QE compares magnetizations with ``eps2 = 1e-5`` (``sgam_at_mag``), looser than
#: the position tolerance because ``m_loc`` is a product of input numbers.
_MAGNETIC_TOLERANCE = 1.0e-5


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
    #: ``t_rev``: 1 where the operation is a symmetry only when followed by time
    #: reversal, 0 otherwise. Empty means "none of them", which is the case for
    #: every nonmagnetic run; see :func:`magnetic_symmetries`.
    time_reversed: tuple = eqx.field(static=True, default=())

    @property
    def nsym(self) -> int:
        return len(self.rotations)

    @property
    def magnetic(self) -> bool:
        """Whether any operation needs time reversal to be one."""
        return any(self.time_reversed)

    def t_rev_array(self) -> np.ndarray:
        """``(nsym,)`` of 0/1, zeros when the group is an ordinary one."""
        if not self.time_reversed:
            return np.zeros(self.nsym, dtype=int)
        return np.array(self.time_reversed, dtype=int)

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

    # **The comparison is scale-free, and QE's is too.** ``symm_base.f90``
    # tests its candidate rotations against ``at``, which is in units of
    # ``alat``, with an absolute ``eps1 = 1e-6``; doing the same arithmetic in
    # bohr makes the *same crystal* lose operations as its lattice constant
    # grows, because the residue being tested against a fixed number is a
    # length or a length squared. On the rhombohedral arsenic of QE's
    # ``pw_vc-relax/vc-relax4.in``, whose cell is written to eight decimals,
    # the metric's off-diagonals are spread by 1.7e-7 alat^2 -- inside QE's
    # threshold -- and by **8.5e-6 bohr^2**, outside a bare 1e-6: eight of the
    # twelve operations were dropped and the k-set came out twice too large.
    # A variable-cell relaxation makes this worse than a fixed setting, since
    # the cell it applies to changes size during the run.
    length_scale = float(np.linalg.norm(at, axis=1).max())
    length_tolerance = _TOLERANCE * length_scale
    metric_tolerance = _TOLERANCE * length_scale**2

    # Candidate images of each basis vector: lattice vectors of the same length.
    #
    # How far the search has to reach is a property of the cell and is bounded
    # exactly rather than guessed. An image ``v`` of ``a_i`` has ``|v| = |a_i|``
    # and integer coordinates ``n_j = v . b_j`` in the reciprocal basis
    # (``a_i . b_j = delta_ij``, no 2 pi), so ``|n_j| <= max_i |a_i| |b_j|``.
    # A fixed window instead of this is wrong the moment the cell is a
    # **supercell**: five primitive cells stacked along one axis need
    # coefficients of five, and a window of three silently drops the three-fold
    # axis of a 10-atom silicon supercell -- pypresso found 2 operations where
    # QE found 6, and the two densities were symmetrised differently enough to
    # move the total energy by 3e-6 Ry. The cost of the honest bound is a
    # slightly larger candidate list built once per run on the host.
    reciprocal = np.linalg.inv(at).T
    extent = np.linalg.norm(at, axis=1).max() * np.linalg.norm(reciprocal, axis=1)
    ranges = [np.arange(-n, n + 1) for n in np.floor(extent + _TOLERANCE).astype(int)]
    i, j, k = np.meshgrid(*ranges, indexing="ij")
    integers = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1)
    vectors = integers @ at
    lengths = np.linalg.norm(vectors, axis=1)

    candidates = [
        integers[np.abs(lengths - np.linalg.norm(at[axis])) < length_tolerance]
        for axis in range(3)
    ]

    operations = []
    for first in candidates[0]:
        for second in candidates[1]:
            if abs(first @ metric @ second - metric[0, 1]) > metric_tolerance:
                continue
            for third in candidates[2]:
                if abs(first @ metric @ third - metric[0, 2]) > metric_tolerance:
                    continue
                if abs(second @ metric @ third - metric[1, 2]) > metric_tolerance:
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
            if not _crystallographic_translation(candidate):
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


def magnetic_symmetries(
    cell: Cell, structure: Structure, symmetries: Symmetries, moments: np.ndarray
) -> Symmetries:
    """Keep only the operations that are symmetries of the magnetization too.

    ``sgam_at_mag`` in ``symm_base.f90``. A magnetic noncollinear run has a
    *vector* on every atom, and an operation of the space group is a symmetry of
    the crystal only if it also maps that vector field onto itself. Two things
    make this different from testing the positions:

    * the magnetization is an **axial** vector, so the rotated moment carries a
      factor ``det(R)`` -- an inversion leaves it alone where it reverses a
      position (QE spells this ``sname(1:3) == 'inv'``);
    * an operation that sends every moment to *minus* its image is still a
      symmetry, of the crystal followed by **time reversal**. It is kept, with
      ``t_rev = 1``, and everything downstream that acts on a magnetization owes
      it a further sign.

    Only operations that do one or the other for *every* atom survive; the rest
    are dropped. Skipping this filter is not a missed optimisation -- it
    symmetrises with operations that reverse the magnetization without recording
    it, which averages the moment to zero and converges to the nonmagnetic
    solution.

    Args:
        moments: ``(nat, 3)`` cartesian starting moments (``m_loc`` in
            ``setup.f90``).
    """
    if symmetries.nsym <= 1:
        return symmetries
    moments = np.asarray(moments, dtype=float)
    mapping = atom_mapping(cell, structure, symmetries)
    rotations = cartesian_rotations(cell, symmetries)

    kept, translations, t_rev = [], [], []
    for s, rotation in enumerate(rotations):
        determinant = np.sign(np.linalg.det(rotation))
        rotated = determinant * (moments @ rotation.T)
        images = moments[mapping[s]]
        same = np.all(np.abs(rotated - images) < _MAGNETIC_TOLERANCE)
        opposite = np.all(np.abs(rotated + images) < _MAGNETIC_TOLERANCE)
        if not (same or opposite):
            continue
        kept.append(symmetries.rotations[s])
        translations.append(symmetries.translations[s])
        # ``t1`` wins when both hold, which happens only for a zero moment.
        t_rev.append(0 if same else 1)

    return Symmetries(
        rotations=tuple(kept),
        translations=tuple(translations),
        time_reversed=tuple(t_rev),
    )


def magnetization_signs(cell: Cell, symmetries: Symmetries) -> np.ndarray:
    """``det(R) * (-1)^t_rev`` per operation -- the sign an axial vector picks up.

    Split out because three places need the same rule: symmetrising the
    magnetization density, symmetrising a per-atom moment, and the magnetic
    filter above.
    """
    rotations = cartesian_rotations(cell, symmetries)
    determinants = np.sign(np.linalg.det(rotations))
    return determinants * np.where(symmetries.t_rev_array() == 1, -1.0, 1.0)


def symmetrize_magnetization(
    mag_g: jnp.ndarray, permutations, phases, rotations: jnp.ndarray
) -> jnp.ndarray:
    """Average a magnetization density over the group -- ``sym_rho``'s ``nspin = 4``.

    ``mag_g`` is ``(3, ngm)`` in **cartesian** components. The scalar rule

        rho_sym(G) = (1/N) sum_S e^{-i G . f_S} rho(S^T G)

    gains one factor: the three components rotate into each other as well, so

        m_sym(G) = (1/N) sum_S d_S R_S . e^{-i G . f_S} m(S^T G)

    with ``R_S`` the cartesian rotation and ``d_S = det(R_S) (-1)^{t_rev}`` the
    axial-vector sign of :func:`magnetization_signs`. The permutation and the
    phases are the *same* ones the charge uses -- the rotation matrix is the
    whole difference — which is why this shares :func:`symmetry_maps`.

    Args:
        rotations: ``(nsym, 3, 3)`` already multiplied by the signs.
    """
    return symmetrize_vector_density(mag_g, permutations, phases, rotations)


def symmetrize_vector_density(
    field_g: jnp.ndarray, permutations, phases, rotations: jnp.ndarray
) -> jnp.ndarray:
    """Average a three-component density over the group, components and all.

        f_sym(G) = (1/N) sum_S R_S . e^{-i G . f_S} f(S^T G)

    The scalar rule with the three components rotated into each other as well.
    Two callers, differing only in what they hand in as ``rotations``:
    :func:`symmetrize_magnetization`, whose magnetization is **axial** and
    carries ``det(R)`` and a time-reversal sign, and the electric field's
    response density (:mod:`pypresso.response.efield`), which is **polar** and
    carries the plain rotation. Getting that distinction wrong is not a worse
    average but a different symmetry.

    Args:
        field_g: ``(3, ngm)`` in cartesian components.
        rotations: ``(nsym, 3, 3)``, signs already folded in if there are any.
    """
    gathered = phases[:, None, :] * field_g[:, permutations].transpose(1, 0, 2)
    return jnp.mean(jnp.einsum("sij,sjg->sig", rotations, gathered), axis=0)


def symmetrize_atom_displacement_density(
    field_g: jnp.ndarray, permutations, phases, rotations: jnp.ndarray, mapping
) -> jnp.ndarray:
    """Average ``3 nat`` response densities over the group -- ``symdvscf`` at ``q = 0``.

    :func:`symmetrize_vector_density` carries one more index. A perturbation
    that displaces **atom a along direction i** is not labelled by a direction
    alone: an operation rotates the direction *and* carries the perturbation to
    the atom it maps onto, so the average is

        drho_{a,i}(r) <- (1/N) sum_S R_ij drho_{S^-1(a),j}({S|f}^-1 r).

    That is :func:`symmetrize_vector`'s atom permutation -- ``irt``, the same
    table the forces use -- on top of the polar-vector rotation the electric
    field's response already needed. **Both indices have to move together**: a
    displacement of one atom is not a symmetry-adapted object, and averaging its
    three directions while leaving it on its own atom is an average over a group
    the perturbation does not have.

    **The atom index is the INVERSE permutation, and that is forced rather than
    chosen.** ``{S|f}`` carries a displacement of atom ``a`` along ``i`` into a
    displacement of atom ``irt[s,a]`` along ``R i``, so labelling the result by
    the atom it lands *on* puts ``S^-1`` under the sum. Written with ``irt``
    itself the average runs over a set that is not this object's group action,
    and the result is **not even a projector** -- which is how it is testable
    without a reference. The error is invisible wherever every operation's
    permutation is an involution: one atom in the cell (identity), and diamond
    silicon and two-atom aluminium, where atoms only ever swap in pairs. That
    was every cell this was checked on until the **four-atom conventional cell
    of fcc aluminium**, whose 48 operations contain 3-cycles on the three
    face-centring atoms; there it is worth **0.33** on a field that is invariant
    by construction (``PLAN.md`` P28).

    :func:`symmetrize_atom_pair_tensor` is the companion and does **not** share
    the direction: it carries two atom labels and no spatial argument, so
    ``irt`` is right there. The two are checked separately, because the thing
    that fixes the direction is whether an atom label travels with a spatial
    one.

    On a crystal with one atom in the cell ``mapping`` is the identity and this
    reduces to :func:`symmetrize_vector_density` exactly. On diamond silicon it
    does not: the operations that exchange the two sublattices are half the
    group, and they are why the two atoms' response densities are not
    independent quantities.

    Args:
        field_g: ``(nat, 3, ngm)`` in cartesian components.
        rotations: ``(nsym, 3, 3)`` cartesian, polar (no ``det(R)`` sign).
        mapping: ``(nsym, nat)`` from :func:`atom_mapping`.
    """
    # ``argsort`` of a permutation is its inverse: ``inverse[s, irt[s,a]] = a``.
    # Inverted here rather than at the call site so that callers keep passing
    # :func:`atom_mapping`'s table unchanged and the reason sits by the
    # derivation above.
    inverse = jnp.asarray(np.argsort(np.asarray(mapping), axis=1))
    # (nsym, nat, 3, ngm) by broadcasting three index arrays against each other:
    # ``gathered[s, a, j, g] = phase[s, g] * field_g[irt^-1[s, a], j, S^T G_g]``.
    # The atom gather and the G-vector gather are independent, so they are one
    # indexing expression rather than two passes over the array.
    gathered = phases[:, None, None, :] * field_g[
        inverse[:, :, None, None],
        jnp.arange(field_g.shape[1])[None, None, :, None],
        permutations[:, None, None, :],
    ]
    return jnp.mean(jnp.einsum("sij,sajg->saig", rotations, gathered), axis=0)


def symmetrize_tensor_density(
    field_g: jnp.ndarray, permutations, phases, rotations: jnp.ndarray
) -> jnp.ndarray:
    """Average nine response densities labelled by a **rank-2** perturbation.

    :func:`symmetrize_vector_density` with one more cartesian index, and the
    perturbation it exists for is a homogeneous **strain**
    (:mod:`pypresso.response.strain`). A strain is not three directions but a
    tensor, so an operation rotates both of its indices:

        drho_ab(r) <- (1/N) sum_S R_ai R_bj drho_ij({S|f}^-1 r).

    The reasoning is the same one line as for the vector case, written out
    because getting it backwards is a different symmetry rather than a worse
    average. The response to a general strain ``eps`` is
    ``drho[eps](r) = sum_ab G_ab(r) eps_ab``; a symmetry operation of the
    crystal satisfies ``drho[R eps R^T](R r) = drho[eps](r)``, and matching
    coefficients of ``eps_ab`` gives exactly the average above.

    **The convention is :func:`symmetrize_vector_density`'s, index for index**
    -- ``out_i = R_ij f_j`` there becomes ``out_ij = R_ik R_jl f_kl`` here --
    and it is *checked* rather than asserted: an unshifted Monkhorst-Pack grid
    is closed under the point group, so the same response can be computed on the
    reduced wedge with this average and on the whole grid without it, and the
    two must agree (``tests/regression/test_electrostriction.py``).

    Args:
        field_g: ``(3, 3, ngm)`` in cartesian components.
        rotations: ``(nsym, 3, 3)`` cartesian, polar (no ``det(R)`` sign): a
            strain is built from two polar vectors, so it carries no sign of its
            own even under an improper operation.
    """
    flat = field_g.reshape((9,) + field_g.shape[2:])
    gathered = (phases[:, None, :] * flat[:, permutations].transpose(1, 0, 2))
    gathered = gathered.reshape((-1, 3, 3) + field_g.shape[2:])
    return jnp.mean(
        jnp.einsum("sik,sjl,sklg->sijg", rotations, rotations, gathered), axis=0
    )


#: Denominators QE accepts in a fractional translation (``symm_base.f90``,
#: ``sgam_at``: "ft_ is in crystal axis and is a valid fractional translation
#: only if ft_(i)=0 or ft_(i)=1/n, with n=2,3,4,6"). Those are the orders a
#: screw axis or a glide plane can have in three dimensions.
_CRYSTALLOGRAPHIC_DENOMINATORS = (2, 3, 4, 6)


def _crystallographic_translation(candidate) -> bool:
    """QE's filter on a fractional translation, transcribed.

    **It rejects operations that really are symmetries**, and that is the point
    of transcribing it rather than improving on it. The test is on the
    *components* of the translation in crystal axes, so it depends on where the
    origin sits: a mirror plane at ``z = 2/5`` of a five-layer cell is written
    with ``ft = (0, 0, 4/5)`` when the origin is at a layer, and 5 is not one of
    the orders a screw or a glide can have -- so QE drops it, keeping 6
    operations of the 12 that map five-layer graphite onto itself.

    Keeping the other six is not more correct, it is a **different
    calculation**: the extra operations carry a translation with a denominator
    of five, ``fft_fact`` then forces the FFT dimensions to be multiples of
    five, and ``c10-graphite-d2`` gets a 20x20x**135** grid where ``pw.x``
    chooses 20x20x**128**. The exchange-correlation energy is evaluated
    pointwise on that grid, so the two totals differ by **1.7e-4 Ry** -- an
    order of magnitude more than any tolerance here -- and neither code is
    wrong. The rule is `CLAUDE.md`'s: what QE computes is the target, and a
    divergence in the symmetry group is a divergence in everything downstream
    of it.
    """
    for component in np.asarray(candidate):
        residue = component - np.rint(component)
        if abs(residue) < _TOLERANCE:
            continue
        order = int(np.rint(1.0 / abs(residue)))
        if abs(1.0 / abs(residue) - order) > _TOLERANCE:
            return False
        if order not in _CRYSTALLOGRAPHIC_DENOMINATORS:
            return False
    return True


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


def symmetrize_vector(
    vectors: jnp.ndarray, cell: Cell, symmetries: Symmetries, mapping: np.ndarray
) -> jnp.ndarray:
    """Impose the crystal symmetry on a per-atom vector -- ``symvector``.

    ``symme.f90``. The forces are the case this exists for. Averaging over the
    group is not cosmetic there: a force computed from a symmetry-reduced
    k-point set is a *vector* built from a sum that is only exact for scalars,
    so its component along a direction the crystal's symmetry forbids is a
    residue of the reduction rather than physics. Symmetrising projects it onto
    the subspace the symmetry allows, and on an undistorted crystal it is what
    makes the force come out identically zero.

    The vector is taken to crystal axes first, because that is where the
    rotations are integers -- QE does the same, for the same reason -- and
    ``mapping`` is ``irt``, which atom each operation sends an atom to
    (:func:`atom_mapping`).

    Args:
        vectors: ``(nat, 3)`` cartesian.
        mapping: ``(nsym, nat)`` from :func:`atom_mapping`.
    """
    if symmetries.nsym <= 1:
        return vectors
    at = np.asarray(cell.at, dtype=float)
    rotations = symmetries.rotation_array().astype(float)

    # Covariant components: work_j = a_j . v, which is what the integer
    # rotations act on.
    work = vectors @ at.T
    rotated = jnp.einsum("sij,snj->ni", rotations, work[mapping])
    return (rotated / symmetries.nsym) @ np.linalg.inv(at).T


def symmetrize_matrix(
    matrix: np.ndarray, cell: Cell, symmetries: Symmetries
) -> np.ndarray:
    """Impose the crystal symmetry on a cartesian rank-2 tensor -- ``symmatrix``.

    ``symme.f90``. The stress is the case this exists for, and it is
    :func:`symmetrize_vector`'s argument one rank up: a tensor computed from a
    Brillouin-zone sum over the irreducible wedge is only exact for a scalar, so
    its components along directions the crystal's symmetry forbids are a residue
    of the reduction. On cubic silicon the off-diagonal entries come out at 1e-6
    Ry/bohr^3 before this and identically zero after, and the diagonal's three
    entries are averaged into agreement -- which is why QE symmetrises **every
    term and then the total again** (``stress.f90``: ``symmatrix(sigma)`` at the
    end, on top of the per-term calls in ``stres_knl`` and elsewhere).

    The tensor is taken to crystal axes first, because that is where the
    rotations are integers -- QE does the same, for the same reason. The two
    conversions are ``cart_to_crys`` (``M -> A M A^T``, rows of ``A`` the
    lattice vectors) and ``crys_to_cart`` (``M -> B^T M B``, rows of ``B`` the
    reciprocal ones), which are inverses because ``A B^T = 1``.

    Args:
        matrix: ``(3, 3)`` cartesian.
    """
    matrix = np.asarray(matrix, dtype=float)
    if symmetries.nsym <= 1:
        return matrix
    at = np.asarray(cell.at_alat, dtype=float)
    bg = np.asarray(cell.bg_2pi_alat, dtype=float)
    rotations = symmetries.rotation_array().astype(float)

    crystal = at @ matrix @ at.T
    averaged = np.einsum("sik,sjl,kl->ij", rotations, rotations, crystal)
    return bg.T @ (averaged / symmetries.nsym) @ bg


def symmetrize_atom_tensor(
    tensors: np.ndarray, cell: Cell, symmetries: Symmetries, mapping: np.ndarray
) -> np.ndarray:
    """Impose the crystal symmetry on a per-atom rank-2 tensor -- ``symtensor``.

    ``symme.f90``. :func:`symmetrize_matrix` with :func:`symmetrize_vector`'s
    atom permutation: an operation both rotates the two cartesian indices and
    carries the tensor from an atom to the atom it maps onto. The Born effective
    charges are the case this exists for, and on silicon they are a difference
    of large numbers -- ``Z_ion = 4`` against an electronic part near ``4.076``
    -- so the residue the reduction leaves is not small relative to the answer.

    Args:
        tensors: ``(nat, 3, 3)`` cartesian.
        mapping: ``(nsym, nat)`` from :func:`atom_mapping`.
    """
    tensors = np.asarray(tensors, dtype=float)
    if symmetries.nsym <= 1:
        return tensors
    at = np.asarray(cell.at_alat, dtype=float)
    bg = np.asarray(cell.bg_2pi_alat, dtype=float)
    rotations = symmetries.rotation_array().astype(float)

    crystal = np.einsum("ik,nkl,jl->nij", at, tensors, at)
    averaged = np.einsum(
        "sik,sjl,snkl->nij", rotations, rotations, crystal[mapping]
    ) / symmetries.nsym
    return np.einsum("ki,nkl,lj->nij", bg, averaged, bg)


def symmetrize_atom_pair_tensor(
    tensors: np.ndarray, cell: Cell, symmetries: Symmetries, mapping: np.ndarray
) -> np.ndarray:
    """Impose the crystal symmetry on a force-constant matrix -- ``symdynph_gq``.

    ``PHonon/PH/symdynph_gq.f90`` at ``q = 0``. :func:`symmetrize_atom_tensor`
    with **two** atom indices instead of one: an operation rotates both
    cartesian indices and carries the pair ``(a, b)`` to the pair
    ``(S(a), S(b))``, so

        D_(a i)(b j) <- (1/N) sum_S R_ik R_jl D_(S(a) k)(S(b) l).

    It is needed for exactly the reason :func:`symmetrize_vector` is needed for
    the forces and :func:`symmetrize_matrix` for the stress, one rank further
    up: a Brillouin-zone sum over the irreducible wedge is exact for a scalar
    and not for anything with a free index, so the components the crystal
    forbids are a residue of the reduction rather than physics. **Symmetrising
    the response density inside the self-consistent loop does not do this job**
    -- that fixes the screening each perturbation sees, and this fixes the wedge
    sum in the assembled matrix.

    Args:
        tensors: ``(nat, 3, nat, 3)`` cartesian, in Ry/bohr^2.
        mapping: ``(nsym, nat)`` from :func:`atom_mapping`.
    """
    tensors = np.asarray(tensors, dtype=float)
    if symmetries.nsym <= 1:
        return tensors
    at = np.asarray(cell.at_alat, dtype=float)
    bg = np.asarray(cell.bg_2pi_alat, dtype=float)
    rotations = symmetries.rotation_array().astype(float)

    # To crystal axes, where the rotations are integers -- the same conversion
    # pair as ``symmatrix``, applied to each of the two cartesian indices.
    crystal = np.einsum("ik,akbl,jl->aibj", at, tensors, at)
    gathered = crystal[mapping[:, :, None], :, mapping[:, None, :], :]
    averaged = np.einsum(
        "sik,sjl,sabkl->aibj", rotations, rotations, gathered
    ) / symmetries.nsym
    return np.einsum("ki,akbl,lj->aibj", bg, averaged, bg)


def check_lattice_symmetry(
    cell: Cell, symmetries: Symmetries, tolerance: float = 1.0e-6
) -> bool:
    """Whether ``cell`` still admits every rotation of ``symmetries``.

    :func:`check_symmetry`'s missing half, and it is missing for a reason that
    stops holding the moment the cell can move. That check works in *crystal*
    coordinates, so a deformation of the cell leaves every one of its numbers
    untouched -- a cubic crystal stretched into a tetragonal one passes it
    unchanged, with four of its rotations no longer symmetries of anything. A
    rotation is stored as an integer matrix acting on crystal row vectors, so
    what it has to preserve is the *metric*:

        R g R^T = g,   g_ij = a_i . a_j,

    which is ``checkallsym``'s lattice half (``symm_base.f90`` finds the group
    from the metric in the first place). A variable-cell relaxation needs both
    checks and needs them for the same reason as a fixed-cell one: the FFT grid
    and the k-point set were chosen for a group, and a step that leaves that
    group has invalidated them. A stress symmetrised over the group cannot do
    this in exact arithmetic, exactly as a symmetrised force cannot move an
    atom off its site, so a failure here is a bug rather than a physical event.
    """
    metric = np.asarray(cell.at) @ np.asarray(cell.at).T
    scale = max(float(np.abs(metric).max()), 1.0)
    for rotation in symmetries.rotation_array():
        rotation = np.asarray(rotation, dtype=float)
        if np.abs(rotation @ metric @ rotation.T - metric).max() > tolerance * scale:
            return False
    return True


def check_symmetry(cell: Cell, structure: Structure, symmetries: Symmetries) -> bool:
    """Whether ``structure`` still has every operation of ``symmetries``.

    ``checkallsym`` in ``PW/src/checkallsym.f90``. QE finds the symmetry group
    once, at setup, and from then on only checks it -- because the FFT grid and
    the k-point set were chosen for that group and cannot change underneath a
    relaxation. This is that check: after an ionic step the moved structure must
    still be invariant under the operations the run was set up with.
    """
    positions = np.asarray(structure.positions_crystal(cell)) % 1.0
    types = np.asarray(structure.types)
    for rotation, translation in zip(
        symmetries.rotation_array(), symmetries.translation_array()
    ):
        if not _maps_structure(positions @ rotation + translation, positions, types):
            return False
    return True
