"""Symmetrising the projector occupations.

The density on the FFT grid is symmetrised in G space, which takes care of
everything the plane-wave part of an ultrasoft calculation produces -- including
the augmentation charge, since that is added to the density before the
symmetrisation runs. ``becsum`` itself is not covered by that: PAW feeds it to
the one-centre terms directly, on each atom's own radial mesh, where the
crystal's symmetry has no grid to act on.

It has to be imposed explicitly, and it is not optional for the same reason the
density's symmetrisation is not (`PLAN.md` P5): a symmetry-reduced k-point set
gives an unsymmetric ``becsum``, and diamond silicon's three ``p`` channels come
out with occupations of 1.003, 1.268, 1.268 where symmetry says they must be
equal. The one-centre energy of that is wrong in the fifth decimal.

``PW/src/paw_symmetry.f90``:

    becsym^a_ij = 1/nsym sum_S D^{l_i}_{m_i m}(S) D^{l_j}_{m_j m'}(S)
                              becsum^{S(a)}_{(n_i m)(n_j m')}

-- an average over the group, with each pair of channels rotated by the matrices
that mix real spherical harmonics of the same ``l``, and the atom index following
where the operation sends the atom. The radial index ``n`` is untouched: a
rotation cannot mix projectors of different shape.

The same average applies to ``ddd_paw`` (QE's ``PAW_symmetrize_ddd``), which is
a derivative with respect to ``becsum`` and so transforms the same way.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from pypresso.pseudo.harmonics import real_spherical_harmonics
from pypresso.pseudo.projectors import projector_channels
from pypresso.system.symmetry import (
    Symmetries,
    atom_mapping,
    cartesian_rotations,
    magnetization_signs,
)

__all__ = ["BecsumSymmetry", "build_becsum_symmetry", "harmonic_rotations"]


def harmonic_rotations(cell, symmetries: Symmetries, lmax: int) -> list:
    """``D^l[s]``: how each operation mixes the ``2l+1`` real harmonics.

    Defined by ``Y_lm(R r) = sum_m' D^l[m, m'] Y_lm'(r)``, and obtained the way
    ``PW/src/d_matrix.f90`` obtains it -- evaluate both sides at a set of
    directions and invert -- for the same reason
    :mod:`pypresso.pseudo.coupling` does it that way: the matrices are defined
    *by* this project's harmonics, so deriving them from it is the only way they
    cannot disagree with it. They come out orthogonal, which is the check that
    the harmonics are properly normalised, and a test asserts it.

    Returns one ``(nsym, 2l+1, 2l+1)`` array per ``l`` from 0 to ``lmax``.
    """
    rotations = cartesian_rotations(cell, symmetries)
    directions = _spread_directions(max(4 * lmax + 2, 8))
    reference = np.asarray(real_spherical_harmonics(jnp.asarray(directions), lmax))

    matrices = []
    for l in range(lmax + 1):
        block = slice(l * l, (l + 1) ** 2)
        # More directions than unknowns, resolved in the least-squares sense:
        # the fit is exact (the harmonics span themselves), and the extra rows
        # only make the inversion better conditioned than a square draw would be.
        pseudo_inverse = np.linalg.pinv(reference[:, block])  # (2l+1, ndir)
        matrices.append(
            np.array([
                np.asarray(real_spherical_harmonics(
                    jnp.asarray(directions @ rotation.T), lmax
                ))[:, block].T @ pseudo_inverse.T
                for rotation in rotations
            ])
        )
    return matrices


def _spread_directions(n: int) -> np.ndarray:
    """``n`` well-separated unit vectors, deterministically."""
    index = np.arange(n) + 0.5
    z = 1.0 - 2.0 * index / n
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = np.pi * (1.0 + 5.0**0.5) * index
    return np.stack([radius * np.cos(phi), radius * np.sin(phi), z], axis=1)


class BecsumSymmetry:
    """The group average, precomputed as one tensor per species.

    ``operator[t]`` is ``(nsym, nh, nh, nh, nh)`` -- for each operation, the
    matrix taking a source atom's ``becsum`` to its contribution to the image's.
    Building it once turns the symmetrisation into a single contraction per
    iteration, which is what keeps it inside ``jit`` and off the host.

    ``nh`` is a few for every element that exists, so the tensor is small; the
    number of operations, not the size, is what makes the naive loop expensive.
    """

    __slots__ = ("operators", "mapping", "species_atoms", "nsym", "rotations")

    def __init__(self, operators, mapping, species_atoms, nsym, rotations=None):
        self.operators = operators
        self.mapping = mapping
        self.species_atoms = species_atoms
        self.nsym = nsym
        #: ``(nsym, 3, 3)`` signed cartesian rotations, or ``None`` when the
        #: spin channel is a spectator. Present only for ``nspin_mag = 4``.
        self.rotations = rotations

    def apply(self, becsum: tuple) -> tuple:
        """The symmetrised ``becsum``, in the same per-species layout."""
        if self.nsym <= 1:
            return becsum
        out = []
        for values, operator, sources in zip(
            becsum, self.operators, self.mapping
        ):
            if values is None or operator is None:
                out.append(values)
                continue
            # sources[s, n] is the index, within this species' atoms, of the
            # atom that operation s sends *onto* atom n -- the inverse
            # permutation, which is what pairs the harmonic rotation of s with
            # the atom QE's ``D(isym)^T becsum(irt(isym,ia)) D(isym)`` pairs it
            # with, once that sum is relabelled S -> S^-1. In every regime but
            # the noncollinear magnetic one the spin channel is a spectator: an
            # operation permutes atoms and rotates harmonics, and does neither
            # to the spin index -- QE's PAW_symmetrize loops over ``is`` outside
            # everything else.
            gathered = values[:, sources]  # (nspin, nsym, nat_t, nh, nh)
            if self.rotations is None:
                out.append(
                    jnp.einsum("sijkl,zsnkl->znij", operator, gathered) / self.nsym
                )
                continue
            # ``nspin_mag = 4``: the charge component is a spectator and the
            # three magnetization components are an axial vector, rotated by
            # the same signed cartesian matrices the density's symmetrisation
            # uses (``paw_symmetry.f90``'s ``s(kpol,is,invs(isym)) * segno``).
            # Doing them as three scalars is a different symmetry, not a
            # coarser one.
            charge = jnp.einsum("sijkl,snkl->nij", operator, gathered[0])
            magnetization = jnp.einsum(
                "sijkl,scd,dsnkl->cnij", operator, self.rotations, gathered[1:]
            )
            out.append(
                jnp.concatenate([charge[None], magnetization], axis=0) / self.nsym
            )
        return tuple(out)

    def apply_directional(self, becsum, rotations) -> tuple:
        """``PAW_dusymmetrize``: three ``becsum`` responses that form a vector.

        The counterpart of :meth:`~pypresso.scf.driver.Calculation.
        symmetrize_directional` one level down. A linear response to a
        perturbation along a *direction* is not three independent ``becsum``
        responses: an operation rotates the directions into each other as well as
        permuting the atoms and rotating the harmonics, so the wedge average is

            dbecsum_a <- (1/N) sum_S R_ab (operator_S . dbecsum_b[S^-1 atom]).

        It is the magnetization branch of :meth:`apply` with the **plain**
        cartesian rotation in place of the signed one -- an induced charge is a
        polar vector where a magnetization is axial. Getting that wrong is worth
        1.6e-2 on the dielectric constant of PAW silicon, against the 5e-5 the
        rest of the machinery reaches.

        Args:
            becsum: per species, ``(3, nspin, nat_t, nh, nh)`` -- the cartesian
                direction leading.
            rotations: ``(nsym, 3, 3)`` cartesian, unsigned
                (:func:`~pypresso.system.symmetry.cartesian_rotations`).
        """
        if self.nsym <= 1:
            return becsum
        rotations = jnp.asarray(rotations)
        out = []
        for values, operator, sources in zip(becsum, self.operators, self.mapping):
            if values is None or operator is None:
                out.append(values)
                continue
            gathered = values[:, :, sources]  # (3, nspin, nsym, nat_t, nh, nh)
            out.append(
                jnp.einsum(
                    "sijkl,scd,dzsnkl->cznij", operator, rotations, gathered
                ) / self.nsym
            )
        return tuple(out)

    def apply_strain(self, becsum, rotations) -> tuple:
        """``PAW_dusymmetrize`` for a **rank-2** perturbation: a strain.

        :meth:`apply_directional` with one more cartesian index, and the same
        rule ``symmetrize_strain_response`` uses on the grid one level up: a
        homogeneous strain is not two directions but a tensor, so an operation
        rotates *both* of its indices:

            dbecsum_ab <- (1/N) sum_S R_ac R_bd (op_S . dbecsum_cd[S^-1 n]).

        **It is not two applications of the vector case.** Averaging over the
        group twice would apply the harmonic operator and the atom permutation
        twice as well, which is a different projector and not a finer one.

        Args:
            becsum: per species, ``(3, 3, nspin, nat_t, nh, nh)``.
            rotations: ``(nsym, 3, 3)`` cartesian, unsigned -- a strain is built
                from two polar vectors and carries no sign of its own.
        """
        if self.nsym <= 1:
            return becsum
        rotations = jnp.asarray(rotations)
        out = []
        for values, operator, sources in zip(becsum, self.operators, self.mapping):
            if values is None or operator is None:
                out.append(values)
                continue
            gathered = values[:, :, :, sources]  # (3, 3, nspin, nsym, nat_t, nh, nh)
            out.append(
                jnp.einsum(
                    "sijkl,sac,sbd,cdzsnkl->abznij",
                    operator, rotations, rotations, gathered
                ) / self.nsym
            )
        return tuple(out)

    def apply_atom_displacement(self, becsum, rotations, mapping) -> tuple:
        """``PAW_dusymmetrize`` for the ``3 nat`` **displacement** patterns.

        :meth:`apply_directional` carries one more index, and it is the same
        index :meth:`~pypresso.scf.driver.Calculation.symmetrize_atom_displacement`
        adds to ``symdvscf``'s electric-field version: a perturbation that moves
        **atom a along direction i** is not labelled by a direction alone, so an
        operation rotates the direction *and* carries the perturbation to the
        atom it maps onto:

            dbecsum_{a,i} <- (1/N) sum_S R_ij (op_S . dbecsum_{S^-1(a),j}[S^-1 n]).

        **Two atom labels move here and they are different labels.** ``a`` is
        the atom that was *displaced* and ``n`` is the atom whose ``becsum``
        block this is; both are permuted, both by the inverse of ``irt``, and
        they are permuted independently because the two are unrelated -- moving
        atom 0 changes the one-centre occupations of atom 1.

        Args:
            becsum: per species, ``(nat, 3, nspin, nat_t, nh, nh)`` -- the
                displaced atom leading, then its cartesian direction.
            rotations: ``(nsym, 3, 3)`` cartesian, unsigned. An induced
                ``becsum`` is a polar object, exactly as in
                :meth:`apply_directional`.
            mapping: ``(nsym, nat)`` from
                :func:`~pypresso.system.symmetry.atom_mapping` -- ``irt``
                itself, inverted here so that call sites keep passing it
                unchanged.
        """
        if self.nsym <= 1:
            return becsum
        rotations = jnp.asarray(rotations)
        # ``argsort`` of a permutation is its inverse. The same inversion
        # :func:`~pypresso.system.symmetry.symmetrize_atom_displacement_density`
        # makes, and for the same reason: labelling the average by the atom the
        # perturbation lands *on* puts ``S^-1`` under the sum. It is invisible
        # wherever every operation's permutation is an involution, which is
        # every cell in this test suite but the four-atom aluminium one.
        inverse = np.argsort(np.asarray(mapping), axis=1)
        out = []
        for values, operator, sources in zip(becsum, self.operators, self.mapping):
            if values is None or operator is None:
                out.append(values)
                continue
            sources = np.asarray(sources)
            # One gather per operation rather than one fancy-indexing
            # expression over both atom axes: ``nsym`` is at most 48 and this
            # runs once per response iteration, where the readable form is
            # worth more than the fused one.
            gathered = jnp.stack([
                values[jnp.asarray(inverse[s])][:, :, :, jnp.asarray(sources[s])]
                for s in range(self.nsym)
            ])  # (nsym, nat, 3, nspin, nat_t, nh, nh)
            out.append(
                jnp.einsum(
                    "sijkl,scd,sadznkl->acznij", operator, rotations, gathered
                ) / self.nsym
            )
        return tuple(out)


def build_becsum_symmetry(
    pseudos, structure, cell, symmetries: Symmetries, nspin_mag: int = 1
) -> BecsumSymmetry | None:
    """Precompute the group average. ``None`` when there is nothing to average."""
    if symmetries.nsym <= 1:
        return None

    lmax = max((p.lmax for p in pseudos), default=-1)
    if lmax < 0:
        return None
    rotations = harmonic_rotations(cell, symmetries, lmax)
    mapping = atom_mapping(cell, structure, symmetries)

    types = np.asarray(structure.types)
    operators, sources_per_species, species_atoms = [], [], []
    for t, pseudo in enumerate(pseudos):
        atoms = np.flatnonzero(types == t)
        species_atoms.append(tuple(int(a) for a in atoms))
        channels = projector_channels(pseudo)
        if not pseudo.is_paw or not len(atoms) or not channels:
            operators.append(None)
            sources_per_species.append(None)
            continue

        nh = len(channels)
        # D acts within an (n, l) block; a channel is (n, l, m), so the operator
        # is block diagonal in n and l and mixes only m.
        single = np.zeros((symmetries.nsym, nh, nh))
        for i, (nb_i, l_i, lm_i) in enumerate(channels):
            for k, (nb_k, l_k, lm_k) in enumerate(channels):
                if nb_i != nb_k:
                    continue
                single[:, i, k] = rotations[l_i][:, lm_i - l_i**2, lm_k - l_k**2]
        operators.append(jnp.asarray(np.einsum("sik,sjl->sijkl", single, single)))

        # Which of this species' atoms is sent *onto* each of them, as a
        # position in the species' own atom list -- the inverse of ``irt``.
        # ``PAW_symmetrize`` contracts ``D(isym)`` on the *source* channel of
        # ``becsum(irt(isym,ia))``; this code contracts it on the target one, so
        # the two are the same sum with S relabelled S^-1, and the atom has to
        # be relabelled with it. They coincide whenever every orbit permutation
        # is its own inverse, which is every cell validated so far, and come
        # apart on the first three-fold orbit.
        position = {int(a): n for n, a in enumerate(atoms)}
        inverse = np.empty((symmetries.nsym, len(atoms)), dtype=int)
        for s in range(symmetries.nsym):
            for n, a in enumerate(atoms):
                inverse[s, position[int(mapping[s, a])]] = n
        sources_per_species.append(jnp.asarray(inverse))

    return BecsumSymmetry(
        operators=tuple(operators),
        mapping=tuple(sources_per_species),
        species_atoms=tuple(species_atoms),
        nsym=symmetries.nsym,
        rotations=(
            jnp.asarray(
                magnetization_signs(cell, symmetries)[:, None, None]
                * cartesian_rotations(cell, symmetries)
            )
            if nspin_mag == 4 else None
        ),
    )
