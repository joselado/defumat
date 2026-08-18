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

    ``mass`` is in atomic mass units, as written in the input file; it is unused
    until molecular dynamics and is carried only so nothing is lost in parsing.
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

    def __check_init__(self):
        if self.positions.shape[1:] != (3,):
            raise ValueError(f"positions must be (nat, 3), got {self.positions.shape}")
        if len(self.types) != self.positions.shape[0]:
            raise ValueError(f"{len(self.types)} types for {self.positions.shape[0]} atoms")
        if self.types and max(self.types) >= len(self.species):
            raise ValueError("a type index points past the end of the species list")

    @property
    def nat(self) -> int:
        return len(self.types)

    @property
    def ntyp(self) -> int:
        return len(self.species)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self.species[t].name for t in self.types)

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
        )
