"""Pseudo-atomic orbitals in the plane-wave basis: the starting wavefunctions.

An SCF has to start its eigensolver somewhere. Starting from random vectors
works and is what this code did, but it wastes the first diagonalisation
discovering that electrons sit near atoms -- something the pseudopotential file
already knows, because it carries the orbitals of the isolated atom in
``PP_PSWFC``. QE starts from a superposition of those (``wfcinit`` ->
``atomic_wfc``), and its first SCF iteration costs two Davidson steps where a
random start costs eight.

The expression is the projectors' expression with a different radial function,

    <k+G| chi_lm^a> = 4 pi / sqrt(Omega) i^l Y_lm(k+G) chi_l(|k+G|) e^{-i(k+G).tau_a}

following ``Modules/atomic_wfc_mod.f90``. **The phase is** ``i^l``, not the
``(-i)^l`` of the projectors, and the Fortran says why in a comment: it is what
makes the k = 0 wavefunctions real in real space. Getting it wrong does not
fail loudly -- it produces a starting guess that is merely worse.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from pypresso.basis.gvectors import GVectors
from pypresso.basis.planewaves import PlaneWaveBasis
from pypresso.pseudo.formfactors import atomic_form_factors
from pypresso.pseudo.projectors import _angular_part, _assemble, _radial_table
from pypresso.pseudo.upf import Pseudopotential
from pypresso.system.cell import Cell
from pypresso.system.kpoints import KPoints
from pypresso.system.structure import Structure

__all__ = ["atomic_channels", "atomic_wavefunctions", "count_atomic_wavefunctions"]


def atomic_channels(pseudo: Pseudopotential) -> list[tuple[int, int, int]]:
    """``(radial index, l, lm column)`` for every orbital channel of a species.

    Orbitals with negative occupation are skipped, as QE skips them, and the
    radial index counts only the kept ones so that it indexes
    :func:`~pypresso.pseudo.formfactors.atomic_form_factors` directly.
    """
    channels, kept = [], 0
    for orbital in pseudo.orbitals:
        if orbital.occupation < 0.0:
            continue
        for m in range(2 * orbital.l + 1):
            channels.append((kept, orbital.l, orbital.l * orbital.l + m))
        kept += 1
    return channels


def count_atomic_wavefunctions(
    pseudos: tuple[Pseudopotential, ...], structure: Structure
) -> int:
    """QE's ``natomwfc``: how many atomic orbitals the crystal has in total."""
    return sum(len(atomic_channels(pseudos[t])) for t in structure.types)


def atomic_wavefunctions(
    pseudos: tuple[Pseudopotential, ...],
    structure: Structure,
    cell: Cell,
    gvectors: GVectors,
    planewaves: PlaneWaveBasis,
    kpoints: KPoints,
) -> jnp.ndarray:
    """``(nk, natomwfc, npwx)`` pseudo-atomic orbitals at every k-point.

    Bands-first, matching how wavefunctions are carried everywhere else, so the
    result can be handed straight to an eigensolver as a starting guess.
    """
    channels_by_species = [atomic_channels(p) for p in pseudos]
    natomwfc = sum(len(channels_by_species[t]) for t in structure.types)
    if natomwfc == 0:
        return jnp.zeros((kpoints.nk, 0, planewaves.npwx), dtype=cell.precision.complex)

    lmax = max((l for channels in channels_by_species for _, l, _ in channels), default=0)
    kg, kg_norm, ylm = _angular_part(
        gvectors.cartesian(cell), planewaves.indices, kpoints.cartesian(cell), lmax
    )

    shape = kg_norm.shape
    flat = kg_norm.reshape(-1)
    form_factors = tuple(
        atomic_form_factors(p, flat, float(cell.volume)) for p in pseudos
    )
    radial = _radial_table(form_factors, shape)
    offset = np.cumsum([0] + [f.shape[0] for f in form_factors])

    chi_of, lm_of, l_of, atom_of = [], [], [], []
    for atom, species in enumerate(structure.types):
        for nb, l, lm in channels_by_species[species]:
            chi_of.append(offset[species] + nb)
            lm_of.append(lm)
            l_of.append(l)
            atom_of.append(atom)

    # The same assembly as the projectors, with i^l in place of (-i)^l.
    wfc = _assemble(
        kg,
        ylm,
        radial,
        structure.positions,
        planewaves.mask,
        jnp.asarray(chi_of),
        jnp.asarray(lm_of),
        jnp.asarray(atom_of),
        jnp.asarray((1j) ** np.asarray(l_of)),
    )
    return jnp.transpose(wfc, (0, 2, 1)).astype(cell.precision.complex)
