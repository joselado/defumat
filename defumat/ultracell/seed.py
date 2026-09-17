"""Starting an ultracell in the texture it is meant to converge to.

``PLAN.md`` P88 stage 8. Nothing in a self-consistent loop breaks spin symmetry
on its own, so an ultracell built on a ferromagnetic unit cell and tiled *stays*
ferromagnetic: the tiled state is an exact fixed point of the loop, for every
``N``. A modulated magnetization therefore has to come from somewhere, and until
this the only door was an applied ``magnetic_field`` -- which is an honest
quantity, the ``Q``-resolved spin susceptibility, and is a *response* rather than
an ordered state. This is the other door: the loop is handed the texture as its
**initial condition** and keeps it.

**Why there is nothing to fade, which is the one design decision here.** Elk
seeds a *random field* (``rndbfcu``) and takes it away geometrically
(``reducebf``), and it has to, because its self-consistency runs on the
**potential**: a seed put into the potential is a term in the functional being
minimised, so it has to leave before the converged state means anything, and
what is left is a fixed point of a field that is going to zero. Here the mixed
quantity is the **density** (:mod:`defumat.ultracell.driver`), so the seed is an
initial condition and not a term: the functional is the field-free one from the
first iteration onwards, the convergence test is the ordinary ``dr2`` with
nothing subtracted from it, and ``reducebf``'s awkward criterion never arises.
The cost of that choice is the other half of the same sentence -- a seed cannot
*hold* a texture the way a field can, so a state whose basin the seed misses
decays back to the tiled one, and the loop says so by converging to it.

**What a seed is: the reference's own moment, turned and scaled.** The caller
does not know the magnetization profile inside a unit cell and should not have
to: what they know is the texture, one direction (and length) per cell. So the
seed is a field ``s(r)`` that multiplies the *converged unit cell's* own
magnetization, exactly as ``starting_magnetization`` multiplies an *atom's*,

* collinear, ``nspin_mag = 2``: ``s`` is a number and ``m(r) -> s(r) m_0(r)``,
  so ``s = cos(2 pi x / n)`` is an amplitude wave and ``s = -1`` in every other
  cell is an antiferromagnet;
* noncollinear, ``nspin_mag = 4``: ``s`` is a vector and
  ``m(r) -> |s(r)| R(e_0 -> s(r)/|s(r)|) m_0(r)``, the whole cell's
  magnetization rigidly rotated from the reference's own direction ``e_0`` onto
  the seed's, so a seed whose direction turns by ``2 pi / n`` from cell to cell
  is a helix of pitch ``n`` cells.

``s = 1`` and ``s = e_0`` are the **identity**, to round-off in the collinear
case (which goes through the charge-and-moment representation and back, one
addition and one halving) and exactly in the noncollinear one (which is already
stored in it). That is what makes the tiled null a test of this code rather than
of the seed. ``|s| <= 1`` is the range: ``s`` scales a moment that is already
there, and ``|s| > 1`` asks for more magnetization than the cell has, which is a
negative channel density.

**The seed is evaluated pointwise**, so a helix turns continuously across a unit
cell rather than in steps, which is what a spin spiral does and what makes the
*differences* between neighbouring cells exactly the pitch it was given. Two
consequences worth knowing. A caller who wants each cell rotated rigidly instead
writes ``floor(x)`` into the callable. And a smooth seed is not a per-cell sign:
``cos(pi x)`` changes sign halfway through each cell, so the *cell* moment it
leaves is small where the *atoms* are cleanly antiferromagnetic -- which is the
physical state, and is why the check that matters is against a supercell rather
than against :meth:`~defumat.ultracell.driver.UltracellResult.cell_moments`.

**Which axis to turn about, which is the one piece of advice here.** The
truncated problem closes exactly one rotation sector and its axis is the
reference's own magnetization: without spin-orbit coupling the unit cell's
spinors are eigenstates of ``sigma . e_0``, so a rotation about ``e_0`` is a
phase on each of them and a rotation about anything else is not. A seed outside
that sector is not wrong -- a global spin rotation costs nothing, so what the
loop does is traverse a flat manifold to the frame its own basis prefers -- but
it cost **290 iterations against 14** on four cells of hydrogen, and it arrived
turning about ``e_0`` rather than about the axis asked for.
:func:`warn_if_the_seed_leaves_the_closed_sector` measures it and says so.

**The charge is not touched.** A spin density wave's charge modulation is a
second-order consequence of the spin one and the loop generates it; seeding it
would be seeding an answer.

**One field, two consumers.** An augmented dataset carries a second mixed
variable, ``becsum``, and most of a transition metal's moment is inside the
projector spheres rather than on the grid -- so seeding the density and leaving
``becsum`` tiled would seed almost nothing on a PAW magnet and would make the
first iteration contradict itself, which is the requirement
:mod:`defumat.scf.continuation` states for an atomic start ("the two starting
guesses have to agree about how polarized the atom is"). Both are therefore
seeded from the **same** evaluated field: ``becsum`` on copy ``R``, atom ``a``
reads ``s`` at that atom's own grid point of the box, so the two cannot disagree
about the coordinate convention or about the period.

**The copy index is a lattice translation in C-order over the triple**, which is
:func:`~defumat.ultracell.augmentation.becsum_per_copy`'s convention (an
``ifftn`` over the ``Q``-grid, reshaped) and
:func:`~defumat.ultracell.grid.Ultracell.q_triples`'s order. A seed landing on
the wrong copy is a converged, plausible and wrong modulation, and no tiled null
can see it because the tiled seed is the same on every copy -- which is why
``tests/unit/test_ultracell_seed.py`` checks a non-uniform seed against the
copies by hand.
"""

from __future__ import annotations

import warnings

import numpy as np
import jax.numpy as jnp

from defumat.scf.continuation import (
    MAGNETIZATION_TOL,
    from_spin_components,
    spin_components,
)

__all__ = [
    "SEED_CONE_TOL",
    "SEED_NET_TOL",
    "refuse_an_unmagnetized_reference",
    "seeded_becsum",
    "seeded_density",
    "reference_axis",
    "warn_if_the_seed_leaves_the_closed_sector",
]

#: How much of ``int |m| dr`` the *net* moment ``|int m dr|`` must carry for the
#: reference's direction to be a direction at all. A ferromagnetic cell gives
#: nearly one; a compensated one gives zero and its net is a residue of
#: cancellation whose direction is round-off, so a rigid rotation "from ``e_0``"
#: would rotate from noise. Only the noncollinear seed reads it -- a collinear
#: one scales along ``z`` and needs no axis.
SEED_NET_TOL = 1.0e-2

#: How far a seed's directions may drift off a cone about ``e_0`` before the
#: warning below fires, measured as the spread of ``|s-hat . e_0|`` over the box.
#: Zero for a texture that turns about the reference's own direction, which is
#: the case the truncated basis closes; about 0.25 for a 90 degree helix seeded
#: about an axis 54.7 degrees away from it, which took 290 iterations against 14.
#:
#: **The absolute value is the whole of the invariant and it was nearly left
#: out.** The cone's *half-angle* is what a rotation about ``e_0`` preserves, and
#: it is ``arccos|s-hat . e_0|`` rather than ``arccos(s-hat . e_0)``: a staggered
#: seed of ``+e_0`` and ``-e_0`` in alternate cells has cosines of ``+1`` and
#: ``-1``, so the signed spread is **1.0** and would warn -- on a state that is
#: collinear along ``e_0``, lies inside the up/down span with no rotation at all,
#: and converges in seven iterations (the collinear ladder is that same physics
#: one regime down). A guard that fires on an antiferromagnet is worse than no
#: guard, because the first thing anyone tries after a helix is a staggered seed.
SEED_CONE_TOL = 1.0e-3

#: Where the minimal rotation from ``e_0`` to ``s`` stops being defined:
#: ``1 + e_0 . s`` below this is the antipodal case, which a period-2 seed hits
#: in every other cell, so it is a branch rather than an edge case.
_ANTIPODAL = 1.0e-7


def refuse_an_unmagnetized_reference(magnetization, volume: float) -> float:
    """The seed scales a moment, so there has to be one. Returns ``int |m| dr``.

    **This is the guard whose absence would be silent.** A seed multiplies the
    reference's own magnetization, so on a cell that converged to zero moment it
    multiplies zero: the run is accepted, every iteration is a no-op, and the
    answer that comes back is the unmagnetized one the caller was trying to
    leave. That is exactly the "a check whose null result cannot be told from a
    pass" shape ``CLAUDE.md`` names, one level out -- the *calculation* returns a
    null that reads as a converged answer. The signed integral is the wrong test
    (an antiferromagnetic reference integrates to zero and is as magnetic as a
    state gets), so this is QE's *absolute* magnetization against
    :data:`~defumat.scf.continuation.MAGNETIZATION_TOL`.
    """
    m = np.asarray(magnetization, dtype=float)
    absolute = float(
        np.sum(np.sqrt(np.sum(m**2, axis=0))) * volume / m[0].size
    )
    if absolute <= MAGNETIZATION_TOL:
        raise ValueError(
            f"a magnetization seed scales the converged unit cell's own "
            f"moment, and this one has none: its absolute magnetization is "
            f"{absolute:.3e} mu_B, which is the unpolarized solution. Scaling "
            f"it would leave every iteration a no-op and return the "
            f"unpolarized state as a converged answer. Give the unit cell a "
            f"starting_magnetization and converge it magnetic first, or drive "
            f"the modulation with magnetic_field=, which needs no moment to "
            f"start from"
        )
    return absolute


def reference_axis(magnetization, volume: float) -> np.ndarray:
    """The unit vector the reference's own magnetization points along.

    ``magnetization`` is ``(3, ...)`` on the box and ``volume`` the *unit
    cell's*, which only sets the scale the two integrals are compared at and
    cancels in the ratio. Raises when the net is a residue of cancellation --
    see :data:`SEED_NET_TOL`.
    """
    m = np.asarray(magnetization, dtype=float)
    absolute = refuse_an_unmagnetized_reference(m, volume)
    net = m.reshape(3, -1).sum(axis=1) * volume / m[0].size
    length = float(np.linalg.norm(net))
    if length <= SEED_NET_TOL * absolute:
        raise NotImplementedError(
            f"a noncollinear seed turns the unit cell's own magnetization "
            f"rigidly, from the direction it points along onto the seed's, and "
            f"this reference has no such direction: its net moment is "
            f"{length:.3e} mu_B against an absolute magnetization of "
            f"{absolute:.3e}, so the net is a residue of cancellation and its "
            f"direction is round-off. A compensated unit cell -- an "
            f"antiferromagnet, whose texture would have to be rotated "
            f"sublattice by sublattice -- is not written: seed a modulation of "
            f"a cell whose moments are parallel, or drive this one with "
            f"magnetic_field= instead, which needs no reference direction"
        )
    return net / length


def warn_if_the_seed_leaves_the_closed_sector(seed, axis: np.ndarray) -> float:
    """Whether the seeded texture turns about the reference's own direction.

    **The one piece of advice this module has, and it is measured.** A helix of
    pitch ``N`` cells is invariant under a translation by one cell followed by a
    spin rotation of ``360/N`` about its own axis, and the self-consistent map
    commutes with both -- so that sector is closed and a run started in it stays
    in it. In the *truncated* problem it is closed only if the basis is, and the
    basis is the unit cell's own spinors: without spin-orbit coupling they are
    eigenstates of ``sigma . e_0``, so a rotation about ``e_0`` is a phase on
    each of them and a rotation about any other axis is not. There is therefore
    **one** closed sector and its axis is ``e_0``.

    A seed lies in it exactly when its directions sit on a cone about ``e_0``,
    which is ``|s-hat . e_0|`` being the same at every point -- what this
    measures, the absolute value being the cone's half-angle and the reason a
    staggered ``+-e_0`` seed is silent (see :data:`SEED_CONE_TOL`).
    A run outside it is not wrong: a global spin rotation costs nothing without
    spin-orbit coupling, so what it does is traverse that flat manifold to reach
    the frame its own basis prefers. What it costs was measured on four cells of
    hydrogen at ``nbnd = 16``: **290 iterations against 14**, arriving at the
    same state -- the two agree to 0.45 per cent after one global rotation and
    to 3.5e-9 Ry -- but turning about ``e_0`` rather than about the axis that
    was asked for.

    With ``lspinorb`` no axis closes the sector at all, since a spin-orbit
    Hamiltonian's eigenstates are eigenstates of no ``sigma . n``. The warning
    is then advice about a cheaper starting point and not about a protected
    quantity, which is why it is a warning and not a refusal. Returns the
    spread, for a caller that would rather test than catch.
    """
    values = np.asarray(seed, dtype=float)
    length = np.sqrt(np.sum(values**2, axis=0))
    live = length > 0.0
    if not np.any(live):
        return 0.0
    projection = np.tensordot(np.asarray(axis, dtype=float), values,
                              axes=(0, 0))[live] / length[live]
    spread = float(np.std(np.abs(projection)))
    if spread > SEED_CONE_TOL:
        warnings.warn(
            f"this seed does not turn about the reference's own magnetization: "
            f"the angle between the seed and that direction varies by "
            f"{spread:.3f} in |cos| over the box, where a texture turning about "
            f"it holds it constant. The run is not wrong -- a global spin rotation costs "
            f"nothing without spin-orbit coupling -- but the truncated basis "
            f"closes only the sector whose axis is the reference's direction, "
            f"so the loop has to traverse a flat manifold to the frame that "
            f"basis prefers, and it arrives turning about that direction "
            f"rather than the one asked for. On four cells of hydrogen this was "
            f"290 iterations against 14 for the same state. Converge the unit "
            f"cell with its moment along the axis the texture is to turn about, "
            f"and write the seed about that axis. With spin-orbit coupling no "
            f"axis closes the sector, so this is advice about a cheaper start "
            f"rather than about a protected pitch",
            stacklevel=3,
        )
    return spread


def _turned(moment, seed, axis: np.ndarray):
    """``|s| R(axis -> s/|s|) m``, pointwise over the trailing axes.

    ``moment`` is ``(3, *rest)`` and ``seed`` ``(3, *rest)`` broadcastable
    against it; ``axis`` is the one constant unit vector the rotation starts
    from. Rodrigues' formula in the form that takes the *pair* of vectors rather
    than an axis and an angle, ``R v = c v + w x v + (w . v) w / (1 + c)`` with
    ``c = a . b`` and ``w = a x b``, so there is no arccos and no normalisation
    of a cross product that vanishes.

    **The antipodal branch is a real case and not a guard.** At ``b = -a`` the
    rotation taking ``a`` to ``b`` is not unique, and a period-2 seed is exactly
    that in every other cell, so the branch runs on the physics rather than on
    an accident: there it is the half turn about a fixed axis perpendicular to
    ``a``. For a reference whose magnetization is collinear along ``a`` -- which
    is what a cell with no spin-orbit canting has -- every such half turn gives
    the same ``-m``, so the choice is invisible; on a canted reference it turns
    the perpendicular components and the choice is a convention, stated here.
    """
    moment = jnp.asarray(moment)
    seed = jnp.asarray(seed)
    a = jnp.asarray(axis, dtype=moment.dtype)
    length = jnp.sqrt(jnp.sum(seed**2, axis=0))
    safe = jnp.where(length > 0.0, length, 1.0)
    b = seed / safe

    def dot(u, v):
        return jnp.sum(u * v, axis=0)

    def cross(u, v):
        return jnp.stack([u[1] * v[2] - u[2] * v[1],
                          u[2] * v[0] - u[0] * v[2],
                          u[0] * v[1] - u[1] * v[0]])

    shape = jnp.broadcast_shapes(moment.shape, b.shape)
    a = jnp.broadcast_to(a.reshape((3,) + (1,) * (len(shape) - 1)), shape)
    moment = jnp.broadcast_to(moment, shape)
    b = jnp.broadcast_to(b, shape)

    c = dot(a, b)
    w = cross(a, b)
    # The denominator is kept finite where the branch below replaces the value,
    # because ``0 * inf`` is a NaN that ``where`` does not undo.
    near = (1.0 + c) < _ANTIPODAL
    denominator = jnp.where(near, 1.0, 1.0 + c)
    turned = (c * moment + cross(w, moment)
              + dot(w, moment) * w / denominator)
    perpendicular = _perpendicular(np.asarray(axis, dtype=float))
    p = jnp.broadcast_to(
        jnp.asarray(perpendicular, dtype=moment.dtype).reshape(
            (3,) + (1,) * (len(shape) - 1)), shape)
    flipped = 2.0 * dot(p, moment) * p - moment
    # A zero-length seed is zero magnetization, not a rotation of it.
    turned = jnp.where(near, flipped, turned)
    return jnp.where(length > 0.0, length * turned, jnp.zeros_like(turned))


def _perpendicular(axis: np.ndarray) -> np.ndarray:
    """One unit vector perpendicular to ``axis``, chosen deterministically.

    The cross product with whichever cartesian direction ``axis`` has least of,
    which is the choice that is furthest from degenerate.
    """
    other = np.zeros(3)
    other[int(np.argmin(np.abs(axis)))] = 1.0
    perpendicular = np.cross(axis, other)
    return perpendicular / np.linalg.norm(perpendicular)


def _refuse_a_seed_over_one(seed, nspin_mag: int) -> None:
    """``|s| <= 1``, because ``s`` scales a moment that is already there.

    A collinear channel density is ``(n +- m)/2`` and the reference's own
    ``|m| <= n`` pointwise, so ``|s| <= 1`` is exactly the condition that the
    seeded density stays positive; above it the loop starts from a state that is
    not a density at all. The noncollinear case is the same statement about
    ``|m| <= n``, which is what makes a spin-density matrix positive
    semi-definite.
    """
    values = np.asarray(seed, dtype=float)
    size = np.abs(values) if nspin_mag == 2 else np.sqrt(
        np.sum(values**2, axis=0))
    worst = float(np.max(size)) if size.size else 0.0
    if worst > 1.0 + 1.0e-12:
        raise ValueError(
            f"a magnetization seed multiplies the converged unit cell's own "
            f"moment, so |s| <= 1 -- this one reaches {worst:.4f}. It is "
            f"starting_magnetization's range and for the same reason: the "
            f"moment is bounded by the charge pointwise, so a factor above one "
            f"asks for more magnetization than the cell has and makes a "
            f"channel density negative. Scale the texture, not the moment: a "
            f"larger moment is a property of the unit cell and is set there"
        )


def seeded_density(density, seed, nspin_mag: int, volume: float, axis=None):
    """The tiled density with its magnetization turned and scaled by ``seed``.

    Args:
        density: ``(nspin_mag, *box)``, the tiled reference -- what the loop
            would otherwise start from.
        seed: ``(*box)`` for a collinear run and ``(3, *box)`` for a
            noncollinear one, already evaluated on the box.
        nspin_mag: 2 or 4. There is nothing for a seed to do at 1 and the caller
            refuses it before here, where the message can say which of the two
            reasons applies.
        volume: the **unit cell's** volume, which the refusal below needs to
            read the reference's magnetization in Bohr magnetons. It is an
            argument rather than a caller's responsibility because the guard
            belongs with the operation: a second caller that forgot it would be
            the "a refusal one caller has and its sibling does not" defect.
        axis: the reference direction the rotation starts from, which
            :func:`reference_axis` reads off this same magnetization.
            Noncollinear only, and it is an argument rather than something
            recomputed here so that the density and the spheres are rotated from
            one axis and not from two.

    Returns the seeded density, same shape and dtype.
    """
    _refuse_a_seed_over_one(seed, nspin_mag)
    charge, moment = spin_components(density, nspin_mag)
    refuse_an_unmagnetized_reference(moment, volume)
    if nspin_mag == 2:
        # ``spin_components`` puts a collinear moment on ``z`` and
        # ``from_spin_components`` reads it back off ``z``, so the whole
        # collinear case is one multiplication in that representation.
        moment = moment * jnp.asarray(seed)[None]
    else:
        warn_if_the_seed_leaves_the_closed_sector(seed, _required(axis))
        moment = _turned(moment, seed, axis)
    seeded = from_spin_components(charge, moment, nspin_mag)
    return jnp.asarray(seeded, dtype=jnp.asarray(density).dtype)


def _required(axis):
    """The reference axis, refused rather than invented when it is missing."""
    if axis is None:
        raise ValueError(
            "a noncollinear seed needs the reference axis it turns the "
            "magnetization from -- reference_axis() off the same density -- so "
            "that the grid and the projector spheres are rotated from one axis"
        )
    return axis


def seeded_becsum(becsum, seed, ultracell, positions, species_atoms,
                  nspin_mag: int, axis=None) -> tuple:
    """The tiled ``becsum`` with every copy's moment turned by the same field.

    Args:
        becsum: the tiled per-copy occupations, one ``(N, nspin_mag, nat_t, nh,
            nh)`` array per species or ``None`` for a species that has none.
        seed: the same evaluated field
            :func:`seeded_density` was given.
        ultracell: the :class:`~defumat.ultracell.grid.Ultracell`.
        positions: ``(nat, 3)`` crystal coordinates of the unit cell's atoms.
        species_atoms: which atom indices belong to each species, in the order
            ``becsum``'s own ``nat_t`` axis runs -- the augmentation table's
            ``species_atoms``.
        nspin_mag: 2 or 4.
        axis: the reference direction, as in :func:`seeded_density` and the same
            one, so that the two halves of a seeded state cannot be rotated from
            two different axes.

    Returns a tuple in ``becsum``'s own shape.
    """
    values = np.asarray(seed, dtype=float)
    grid = tuple(ultracell.grid)
    cell_grid = tuple(ultracell.cell_grid)
    triples = np.asarray(ultracell.q_triples)
    out = []
    for atoms, entry in zip(species_atoms, becsum):
        if entry is None or not len(atoms):
            out.append(entry)
            continue
        entry = jnp.asarray(entry)
        # **The atom's own grid point of the box**, cell by cell: the seed is
        # one field and this is where the sphere reads it. The position is
        # rounded to the unit cell's own grid, which a seed can afford --
        # it varies on the scale of a cell and this is an initial condition --
        # and it is what makes the sphere and the grid agree by construction.
        inside = np.stack([
            np.round(np.asarray(positions)[a] * np.asarray(cell_grid))
            % np.asarray(cell_grid) for a in atoms
        ]).astype(np.int64)
        index = (triples[:, None, :] * np.asarray(cell_grid) + inside[None])
        index = index % np.asarray(grid)
        picked = values[..., index[..., 0], index[..., 1], index[..., 2]]
        # ``(N, nat_t)`` for a collinear seed and ``(3, N, nat_t)`` for a
        # noncollinear one, which is the shape the two branches below want.
        charge, moment = spin_components(jnp.moveaxis(entry, 0, 1), nspin_mag)
        trailing = (None,) * (charge.ndim - 2)
        if nspin_mag == 2:
            moment = moment * jnp.asarray(picked)[(None, ..., *trailing)]
        else:
            moment = _turned(
                moment, jnp.asarray(picked)[(slice(None), ..., *trailing)],
                _required(axis))
        seeded = from_spin_components(charge, moment, nspin_mag)
        out.append(jnp.asarray(jnp.moveaxis(seeded, 0, 1), dtype=entry.dtype))
    return tuple(out)
