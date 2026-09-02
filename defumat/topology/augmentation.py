"""The augmentation charge at an arbitrary wavevector: ultrasoft's ``S`` between
two *different* k-points.

Inside a single k-point the overlap operator is
``S = 1 + sum_ij |beta_i> q_ij <beta_j|`` with ``q_ij = int Q_ij(r) dr``, and
:meth:`defumat.hamiltonian.operator.Hamiltonian.apply_s` applies exactly that.
A Berry phase does not stay inside one k-point. What it needs is

    <u_mk|S|u_n k'> = <u_mk|u_nk'> + sum_a sum_ij q^a_ij(b) <psi_mk|beta_i^k>
                                                            <beta_j^k'|psi_nk'>

with ``b = k' - k`` and

    q^a_ij(b) = int Q_ij(r - tau_a) e^{-i b . r} dr
              = Omega Q_ij(b) e^{-i b . tau_a},

the augmentation charge evaluated at ``b`` rather than at ``G = 0``. This is
Vanderbilt's ultrasoft Berry phase (R. D. King-Smith and D. Vanderbilt,
PRB 47, 1651 (1993); D. Vanderbilt and R. D. King-Smith, PRB 48, 4442 (1993)),
and it is what ``PW/src/bp_c_phase.f90`` builds in its ``q_g`` array. Dropping it
does not fail: it gives an overlap matrix that is not unitary and a Chern number
that misses integrality by a few per cent on an ultrasoft dataset and by nothing
at all on a norm-conserving one -- the failure that looks like a mesh
convergence problem and is not.

**Why this module rather than a call into** :mod:`defumat.pseudo.augmentation`.
That module tabulates ``Q_ij(G)`` on the dense G-vector set, because that is
where the density needs it. ``b`` is *not* a G-vector: it is a fraction of one,
``b_d / n_d`` for a mesh of ``n_d`` points along direction ``d``. So the radial
transforms are evaluated afresh at ``|b|`` -- which costs one Bessel integral per
channel per mesh direction, twice for a whole plane, and is why the arbitrary-q
form is written as a function rather than cached as a table.

**Spin-orbit.** A fully relativistic dataset does not use ``q_ij`` but the
``j``-resolved ``qq_so[i, j, s1, s2]``, and the map between them is the ``fcoef``
sandwich of ``upflib/upf_spinorb.f90`` (``transform_qq_so``). That map is a
linear, k-independent contraction on the channel indices, so it applies to
``q_ij(b)`` unchanged -- which is the whole point of routing this through
:meth:`defumat.pseudo.spinorbit.SpinOrbitCoupling.qq_so` rather than writing the
b-dependence into a second copy of it. Using the scalar ``q_ij(b)`` in a spinor
run is the silent-wrong failure CLAUDE.md's spin-orbit row warns about: the
overlap stays plausible and the invariant is meaningless.

**The check that this is right** is ``b -> 0``: the result must reproduce the
``qq`` that :class:`~defumat.pseudo.projectors.Projectors` already carries (and
``qq_so`` in a spinor run) to round-off. ``tests/unit/test_topology_overlap.py``
pins it.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from defumat.pseudo.augmentation import (
    _assemble_qgm,
    radial_augmentation_transforms,
)
from defumat.pseudo.coupling import harmonic_products
from defumat.pseudo.harmonics import real_spherical_harmonics
from defumat.pseudo.projectors import projector_channels

__all__ = ["augmentation_at_q"]


def augmentation_at_q(calculation, qcart) -> jnp.ndarray | None:
    """``q_ij(b)`` as the block matrix the projections contract against.

    Args:
        calculation: the :class:`~defumat.scf.driver.Calculation` whose
            pseudopotentials, structure and channel layout the matrix follows.
        qcart: ``(3,)`` cartesian wavevector in 1/bohr -- ``k' - k``, the
            *unwrapped* geometric difference. Using the wrapped one (the
            difference of two mesh points that are neighbours only through a
            reciprocal lattice vector) is wrong by a whole reciprocal lattice
            vector in both ``Q_ij(b)`` and the structure factor.

    Returns ``(nkb, nkb)`` complex for a scalar calculation, ``(2, 2, nkb, nkb)``
    for a noncollinear one, and ``None`` when there is no augmentation charge at
    all -- which is a norm-conserving run, where ``S`` is the identity and the
    caller adds nothing.
    """
    augmentation = calculation.augmentation
    if augmentation is None:
        return None

    pseudos = calculation.pseudos
    structure = calculation.system.structure
    cell = calculation.system.cell
    qcart = np.asarray(qcart, dtype=float).reshape(1, 3)

    lmax = max(p.lmax for p in pseudos)
    ap = harmonic_products(lmax)
    nl = 2 * lmax + 1
    volume = float(cell.volume)

    qmod = jnp.asarray(np.sqrt(np.sum(qcart**2, axis=1)))
    ylm = real_spherical_harmonics(jnp.asarray(qcart), 2 * lmax)  # (1, (2lmax+1)^2)

    per_species: list[np.ndarray | None] = []
    for pseudo in pseudos:
        channels = projector_channels(pseudo)
        if not pseudo.is_ultrasoft or not channels:
            per_species.append(None)
            continue
        nl_species = min(nl, pseudo.augmentation.nqlc) if pseudo.augmentation else nl
        radial = radial_augmentation_transforms(pseudo, qmod, volume, nl_species)
        beta_of = np.array([nb for nb, _, _ in channels])
        lm_of = np.array([lm for _, _, lm in channels])
        coefficients = jnp.asarray(ap[:, lm_of[:, None], lm_of[None, :]])
        values = _assemble_qgm(
            coefficients, ylm, radial, jnp.asarray(beta_of), nl_species
        )
        # ``qgm`` carries a 1/Omega from its own prefactor; ``qq`` is
        # ``Omega * Q(G = 0)``, so the same factor restores the integral here.
        per_species.append(volume * np.asarray(values[:, :, 0]))

    positions = np.asarray(structure.positions)  # cartesian, bohr
    phases = np.exp(-1j * (positions @ qcart[0]))  # e^{-i b . tau_a}

    if calculation.noncolin:
        return _spinor_blocks(calculation, per_species, phases)
    return _scalar_blocks(calculation, per_species, phases)


def _scalar_blocks(calculation, per_species, phases) -> jnp.ndarray:
    """Assemble the per-species values into the ``(nkb, nkb)`` block matrix."""
    augmentation = calculation.augmentation
    blocks = []
    for values, atoms in zip(per_species, augmentation.species_atoms):
        if values is None or not atoms:
            nh = 0 if values is None else values.shape[0]
            blocks.append(jnp.zeros((len(atoms), nh, nh), dtype=complex))
            continue
        blocks.append(
            jnp.asarray(values[None] * np.asarray(phases)[list(atoms)][:, None, None])
        )
    return augmentation.block_matrix(tuple(blocks))


def _spinor_blocks(calculation, per_species, phases) -> jnp.ndarray:
    """The same, through ``transform_qq_so``, as ``(2, 2, nkb, nkb)``.

    The ``fcoef`` sandwich is linear, so the atom's structure factor multiplies
    the transformed block exactly as it multiplied the untransformed one.
    """
    from defumat.scf.driver import _spin_block_diagonal

    types = calculation.system.structure.types
    per_atom = []
    for atom, t in enumerate(types):
        nh = calculation.pseudos[t].nh
        values = per_species[t]
        matrix = np.zeros((nh, nh), dtype=complex) if values is None else values
        transformed = calculation.spin_orbit[t].qq_so(matrix)
        per_atom.append(transformed * phases[atom])
    return jnp.asarray(_spin_block_diagonal(per_atom))
