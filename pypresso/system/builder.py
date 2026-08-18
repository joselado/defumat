"""Turn a parsed ``pw.x`` input into the objects the rest of the code uses.

This is where QE's input conventions are interpreted -- which cell parameters
win when several are given, what a card's units mean, which defaults apply --
and it is deliberately the *only* place that knows them. Everything downstream
takes a :class:`System` and never looks at an input file again.

Defaults come from ``Modules/input_parameters.f90``; the rules for combining
``ibrav``/``celldm``/``A,B,C``/``CELL_PARAMETERS`` come from ``PW/src/input.f90``.
"""

from __future__ import annotations

import equinox as eqx
import numpy as np

from pypresso.config import DEFAULT_PRECISION, Precision
from pypresso.io.pwin import PwInput, read_pw_input
from pypresso.system.cell import Cell, celldm_from_abc
from pypresso.system.kpoints import KPoints
from pypresso.system.structure import Species, Structure
from pypresso.system.symmetry import find_symmetries
from pypresso.units import ANGSTROM_TO_BOHR

__all__ = ["System", "build_system", "system_from_file"]


class System(eqx.Module):
    """A cell, its atoms, and the k-points to sample -- the output of setup.

    Everything a phase beyond P1 needs about *what* is being calculated, with
    none of the input-file syntax left in it.
    """

    cell: Cell
    structure: Structure
    kpoints: KPoints
    ecutwfc: float = eqx.field(static=True)
    ecutrho: float = eqx.field(static=True)
    nspin: int = eqx.field(static=True, default=1)
    calculation: str = eqx.field(static=True, default="scf")
    nbnd: int | None = eqx.field(static=True, default=None)
    occupations: str = eqx.field(static=True, default="fixed")
    smearing: str = eqx.field(static=True, default="gaussian")
    degauss: float = eqx.field(static=True, default=0.0)
    #: Occupations read from an OCCUPATIONS card, for occupations='from_input'.
    input_occupations: tuple[float, ...] | None = eqx.field(static=True, default=None)


def system_from_file(path, precision: Precision = DEFAULT_PRECISION) -> System:
    return build_system(read_pw_input(path), precision=precision)


def build_system(pwin: PwInput, precision: Precision = DEFAULT_PRECISION) -> System:
    cell = _build_cell(pwin, precision)
    structure = _build_structure(pwin, cell, precision)

    # An automatic k-grid is reduced to its irreducible wedge here, which is
    # where QE does it too (``setup.f90``, after the symmetry analysis and
    # before anything is sized from the k-point count). It needs the crystal's
    # symmetries, hence the ordering: cell, then structure, then symmetry, then
    # k-points. Explicit k-point lists are taken as given, as QE takes them.
    symmetries = find_symmetries(cell, structure)
    kpoints = _build_kpoints(pwin, cell, precision, symmetries.rotation_array())

    ecutwfc = pwin.get("system", "ecutwfc")
    if ecutwfc is None:
        raise ValueError(f"{pwin.path or 'input'}: ecutwfc is required")
    # QE's default: the density cutoff is 4*ecutwfc, exact for norm-conserving
    # pseudopotentials (the density has twice the wavefunction's G range).
    ecutrho = pwin.get("system", "ecutrho") or 4.0 * float(ecutwfc)

    return System(
        cell=cell,
        structure=structure,
        kpoints=kpoints,
        ecutwfc=float(ecutwfc),
        ecutrho=float(ecutrho),
        nspin=int(pwin.get("system", "nspin", 1)),
        calculation=str(pwin.get("control", "calculation", "scf")).lower(),
        nbnd=pwin.get("system", "nbnd"),
        occupations=str(pwin.get("system", "occupations", "fixed")).lower(),
        smearing=str(pwin.get("system", "smearing", "gaussian")).lower(),
        degauss=float(pwin.get("system", "degauss", 0.0)),
        input_occupations=_input_occupations(pwin),
    )


def _input_occupations(pwin: PwInput) -> tuple[float, ...] | None:
    """The OCCUPATIONS card, flattened. Present only for occupations='from_input'."""
    card = pwin.card("OCCUPATIONS")
    if card is None:
        return None
    return tuple(value for row in card.floats() for value in row)


def _build_cell(pwin: PwInput, precision: Precision) -> Cell:
    ibrav = pwin.get("system", "ibrav")
    if ibrav is None:
        raise ValueError(f"{pwin.path or 'input'}: ibrav is required")
    ibrav = int(ibrav)

    celldm = np.array(pwin.indexed("system", "celldm", 6))
    a = pwin.get("system", "a")
    if celldm[0] == 0.0 and a is not None:
        # The crystallographic alternative to celldm: A,B,C in angstrom plus cosines.
        celldm = celldm_from_abc(
            ibrav,
            float(a),
            float(pwin.get("system", "b", 0.0)),
            float(pwin.get("system", "c", 0.0)),
            float(pwin.get("system", "cosab", 0.0)),
            float(pwin.get("system", "cosac", 0.0)),
            float(pwin.get("system", "cosbc", 0.0)),
        )

    if ibrav != 0:
        return Cell.from_ibrav(ibrav, celldm, precision=precision)

    card = pwin.require_card("CELL_PARAMETERS")
    vectors = np.array(card.floats(), dtype=float)
    if vectors.shape != (3, 3):
        raise ValueError("CELL_PARAMETERS must give three vectors of three components")

    units = card.option
    if units is None:
        # Historical default, kept because old inputs rely on it: alat when a
        # lattice parameter was supplied, bohr otherwise.
        units = "alat" if celldm[0] != 0.0 else "bohr"

    if units == "alat":
        if celldm[0] == 0.0:
            raise ValueError("CELL_PARAMETERS alat needs celldm(1) or A")
        return Cell.from_vectors(vectors * celldm[0], alat=float(celldm[0]), precision=precision)
    if units == "bohr":
        return Cell.from_vectors(vectors, alat=celldm[0] or None, precision=precision)
    if units == "angstrom":
        vectors = vectors * ANGSTROM_TO_BOHR
        return Cell.from_vectors(vectors, alat=celldm[0] or None, precision=precision)
    raise ValueError(f"unknown CELL_PARAMETERS units {units!r}")


def _build_structure(pwin: PwInput, cell: Cell, precision: Precision) -> Structure:
    species_card = pwin.require_card("ATOMIC_SPECIES")
    species, index_of = [], {}
    for line in species_card.lines:
        name, mass, pseudo = line.split()[:3]
        index_of[name] = len(species)
        species.append(Species(name=name, mass=float(mass), pseudo_file=pseudo))

    positions_card = pwin.require_card("ATOMIC_POSITIONS")
    names, coordinates = [], []
    for line in positions_card.lines:
        tokens = line.split()
        names.append(tokens[0])
        # Trailing 0/1 flags (if_pos, which freezes a coordinate during
        # relaxation) may follow the three coordinates; they are not geometry.
        coordinates.append([float(t) for t in tokens[1:4]])

    unknown = set(names) - set(index_of)
    if unknown:
        raise ValueError(f"ATOMIC_POSITIONS names a species not in ATOMIC_SPECIES: {sorted(unknown)}")

    nat = pwin.get("system", "nat")
    if nat is not None and int(nat) != len(names):
        raise ValueError(f"nat={nat} but ATOMIC_POSITIONS lists {len(names)} atoms")

    return Structure.from_card_units(
        positions=coordinates,
        types=[index_of[name] for name in names],
        species=species,
        units=positions_card.option,
        cell=cell,
        precision=precision,
    )


def _build_kpoints(
    pwin: PwInput, cell: Cell, precision: Precision, rotations=None
) -> KPoints:
    card = pwin.card("K_POINTS")
    if card is None:
        return KPoints.gamma(precision=precision)

    option = (card.option or "tpiba").lower()

    if option == "gamma":
        return KPoints.gamma(precision=precision)

    if option == "automatic":
        values = [int(v) for v in card.lines[0].split()[:6]]
        return KPoints.automatic(
            tuple(values[:3]), tuple(values[3:6]), cell,
            precision=precision, rotations=rotations,
        )

    # All remaining forms start with a count, then one line per k-point.
    rows = np.array([[float(t) for t in line.split()[:4]] for line in card.lines[1:]])
    declared = int(card.lines[0].split()[0])
    if len(rows) != declared:
        raise ValueError(f"K_POINTS declares {declared} points but lists {len(rows)}")

    points, fourth = rows[:, :3], rows[:, 3]

    if option in ("tpiba", "crystal"):
        if option == "crystal":
            return KPoints.from_crystal(points, fourth, cell, precision=precision)
        return KPoints.from_cartesian(points, fourth, precision=precision)

    if option in ("tpiba_b", "crystal_b"):
        return KPoints.band_path(
            points, fourth, cell, crystal=option.startswith("crystal"), precision=precision
        )

    if option in ("tpiba_c", "crystal_c"):
        raise NotImplementedError(
            "K_POINTS *_c (three points defining a plane) is not needed before the band phase"
        )

    raise ValueError(f"unknown K_POINTS option {option!r}")
