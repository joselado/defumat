"""Atomic species and positions.

Positions are cartesian in bohr and live on the traced side of the code: forces
are ``grad`` of the total energy with respect to them (rule D5), so they must be
a JAX array, not setup metadata. Species identity, in contrast, is static -- it
selects which pseudopotential tables are used and changing it must retrace.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from pypresso.config import DEFAULT_PRECISION, Precision
from pypresso.system.cell import Cell
from pypresso.units import ANGSTROM_TO_BOHR

__all__ = ["Species", "Structure"]


class Species(eqx.Module):
    """One atomic type as declared in ``ATOMIC_SPECIES``.

    ``mass`` is in atomic mass units, as written in the input file, and it is
    what a *dynamical* property needs: nothing in the ground state, a force or a
    stress depends on it -- the Born-Oppenheimer energy surface is a function of
    where the nuclei are and not of what they weigh -- so it went unread until
    the phonons (:mod:`pypresso.response.phonon`), which divide the force
    constants by it. :attr:`Structure.masses` is the per-atom view.
    """

    name: str = eqx.field(static=True)
    mass: float = eqx.field(static=True)
    pseudo_file: str = eqx.field(static=True)


class Structure(eqx.Module):
    """Atoms in a cell: cartesian positions in bohr plus their species.

    ``types[i]`` indexes into ``species`` for atom ``i`` -- the equivalent of
    QE's ``ityp``, but 0-based.
    """

    positions: jnp.ndarray  # (nat, 3) cartesian, bohr -- traced
    types: tuple[int, ...] = eqx.field(static=True)
    species: tuple[Species, ...] = eqx.field(static=True)
    precision: Precision = eqx.field(static=True, default=DEFAULT_PRECISION)
    #: ``if_pos``: the optional trailing ``0``/``1`` flags of an
    #: ``ATOMIC_POSITIONS`` line, one per cartesian component of each atom. A
    #: zero freezes that component during a relaxation -- QE multiplies the
    #: force by it, so a frozen coordinate feels no force and never moves.
    #: ``()`` means every coordinate is free. Static: it is an input flag, not a
    #: quantity anything differentiates.
    if_pos: tuple[tuple[int, int, int], ...] = eqx.field(static=True, default=())

    def __check_init__(self):
        if self.positions.shape[1:] != (3,):
            raise ValueError(f"positions must be (nat, 3), got {self.positions.shape}")
        if len(self.types) != self.positions.shape[0]:
            raise ValueError(f"{len(self.types)} types for {self.positions.shape[0]} atoms")
        if self.types and max(self.types) >= len(self.species):
            raise ValueError("a type index points past the end of the species list")
        if self.if_pos and len(self.if_pos) != self.positions.shape[0]:
            raise ValueError(
                f"{len(self.if_pos)} if_pos rows for {self.positions.shape[0]} atoms"
            )

    @property
    def free(self) -> np.ndarray:
        """``if_pos`` as an ``(nat, 3)`` array of ones and zeros."""
        if not self.if_pos:
            return np.ones((self.nat, 3))
        return np.asarray(self.if_pos, dtype=float)

    @property
    def nat(self) -> int:
        return len(self.types)

    @property
    def ntyp(self) -> int:
        return len(self.species)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self.species[t].name for t in self.types)

    @property
    def masses(self) -> np.ndarray:
        """``(nat,)`` atomic masses in **amu**, the unit the input file uses.

        QE's ``amass`` is per *type* and is indexed through ``ityp`` at every
        use (``dyndia``: ``amass(ityp(na))``); this is that indexing done once.
        The conversion to Rydberg units is
        :data:`~pypresso.units.AMU_TO_RY` and belongs where the mass meets an
        energy, not here -- ``ph.x``'s ``amass`` is in amu too, and keeping the
        same unit is what makes an input's ``amass(1) = 28.086`` comparable
        without arithmetic.
        """
        return np.array([self.species[t].mass for t in self.types], dtype=float)

    def positions_crystal(self, cell: Cell) -> jnp.ndarray:
        return cell.to_crystal(self.positions)

    def positions_alat(self, cell: Cell) -> jnp.ndarray:
        return self.positions / cell.alat

    @classmethod
    def from_card_units(
        cls,
        positions,
        types,
        species,
        units: str,
        cell: Cell,
        precision: Precision = DEFAULT_PRECISION,
        if_pos=(),
    ) -> "Structure":
        """Build from an ``ATOMIC_POSITIONS`` card in any of QE's unit systems.

        QE accepts ``alat`` (units of ``celldm(1)``), ``bohr``, ``angstrom`` and
        ``crystal`` (fractional). ``crystal_sg`` (Wyckoff positions) needs the
        space group machinery and is deferred to the symmetry phase.
        """
        positions = np.asarray(positions, dtype=float)
        units = (units or "alat").lower()

        if units == "alat":
            cartesian = positions * cell.alat
        elif units == "bohr":
            cartesian = positions
        elif units == "angstrom":
            cartesian = positions * ANGSTROM_TO_BOHR
        elif units == "crystal":
            cartesian = np.asarray(cell.to_cartesian(positions))
        elif units == "crystal_sg":
            raise NotImplementedError(
                "ATOMIC_POSITIONS crystal_sg needs space-group expansion (symmetry phase)"
            )
        else:
            raise ValueError(f"unknown ATOMIC_POSITIONS units {units!r}")

        return cls(
            positions=precision.as_real(cartesian),
            types=tuple(int(t) for t in types),
            species=tuple(species),
            precision=precision,
            if_pos=tuple(tuple(int(v) for v in row) for row in if_pos),
        )
