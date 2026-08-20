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
from pypresso.system.kpoints import KPoints, for_spin as kpoints_for_spin
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
    #: 1 for an unpolarized calculation, 2 for collinear LSDA, 4 for
    #: noncollinear. Static, because it is an array *rank* everywhere
    #: downstream, not a value.
    #:
    #: ``nspin`` alone does not fix any array shape once it is 4: QE keeps
    #: *three* numbers and so does this code (``set_spin_vars`` in
    #: ``Modules/noncol.f90``). :attr:`npol` is how many spinor components a
    #: wavefunction has, :attr:`nspin_mag` how many components the density and
    #: the potential have, and ``nspin`` only says which of the three regimes is
    #: in force. Collapsing them is the mistake that makes a nonmagnetic
    #: spin-orbit run allocate -- and symmetrise -- a magnetization it does not
    #: have.
    nspin: int = eqx.field(static=True, default=1)
    calculation: str = eqx.field(static=True, default="scf")
    nbnd: int | None = eqx.field(static=True, default=None)
    occupations: str = eqx.field(static=True, default="fixed")
    #: ``input_dft``: an exchange-correlation functional that overrides the one
    #: the pseudopotentials were generated with. ``None`` -- the normal case --
    #: means the pseudopotentials decide.
    input_dft: str | None = eqx.field(static=True, default=None)
    smearing: str = eqx.field(static=True, default="gaussian")
    degauss: float = eqx.field(static=True, default=0.0)
    #: Occupations read from an OCCUPATIONS card, for occupations='from_input'.
    #: One row per spin channel when nspin = 2 (``f_inp(:, isk(ik))``).
    input_occupations: tuple[float, ...] | None = eqx.field(static=True, default=None)
    #: ``starting_magnetization(i)``, per species, in [-1, 1]. It splits the
    #: superposition of atomic charges the SCF starts from -- and it is the only
    #: thing that does, so an LSDA run left at zero converges to the unpolarized
    #: solution whenever that is a stationary point, which for a symmetric
    #: crystal it always is.
    starting_magnetization: tuple[float, ...] = eqx.field(static=True, default=())
    #: ``tot_magnetization``: constrain ``N_up - N_dw`` instead of letting the
    #: two channels share one Fermi level. ``None`` -- QE's -10000 sentinel --
    #: means unconstrained.
    tot_magnetization: float | None = eqx.field(static=True, default=None)
    #: ``nosym``: use no symmetry at all. Not an optimisation switch -- an input
    #: whose occupations break the crystal's symmetry (an atom with one of its
    #: three p channels filled) needs it, and symmetrising anyway converges to a
    #: different state.
    nosym: bool = eqx.field(static=True, default=False)
    #: ``lspinorb``: use the ``j``-resolved projectors of a fully-relativistic
    #: pseudopotential, which is what puts spin-orbit coupling in the
    #: Hamiltonian. Requires ``noncolin`` -- QE refuses the combination too,
    #: because the spin-orbit term does not commute with ``S_z`` and there is no
    #: collinear Hamiltonian for it to enter.
    lspinorb: bool = eqx.field(static=True, default=False)
    #: ``angle1``/``angle2`` in degrees, per species: the polar and azimuthal
    #: angles of that species' starting magnetization. Only meaningful when
    #: ``noncolin`` -- a collinear run has nothing to point.
    angle1: tuple[float, ...] = eqx.field(static=True, default=())
    angle2: tuple[float, ...] = eqx.field(static=True, default=())

    @property
    def lsda(self) -> bool:
        return self.nspin == 2

    @property
    def noncolin(self) -> bool:
        return self.nspin == 4

    @property
    def npol(self) -> int:
        """How many spinor components a wavefunction has: 2 noncollinear, 1 not.

        This is an array *dimension* of every wavefunction-shaped quantity, and
        the reason a noncollinear Hamiltonian is one operator on a space of
        ``npol * npwx`` rather than ``nspin`` operators on ``npwx``.
        """
        return 2 if self.noncolin else 1

    @property
    def domag(self) -> bool:
        """Whether the run carries a magnetization at all (``setup.f90``).

        For a noncollinear calculation this is decided by ``starting_magnetization``
        being nonzero *somewhere* and by nothing else: a spin-orbit run on a
        nonmagnetic crystal has spinor wavefunctions and a scalar density, and
        QE says so in the comment above the assignment -- "set the domag
        variable to make a spin-orbit calculation with zero magnetization".

        It is a property of the input rather than of the converged state, which
        is what makes it static: the magnetization cannot appear during the SCF
        if nothing in the starting guess breaks the symmetry.
        """
        if not self.noncolin:
            return False
        return any(abs(m) > 1.0e-6 for m in self.starting_magnetization)

    @property
    def nspin_mag(self) -> int:
        """Components of the density and the potential: 1, 2 or 4.

        4 only for a *magnetic* noncollinear run -- ``(n, m_x, m_y, m_z)``. A
        nonmagnetic spin-orbit run has one, exactly as an unpolarized run does,
        and every routine that builds, mixes, symmetrises or integrates a
        density then runs unchanged.
        """
        if self.noncolin:
            return 4 if self.domag else 1
        return self.nspin


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
    nosym = _logical(pwin.get("system", "nosym", False))
    symmetries = find_symmetries(cell, structure)
    rotations = None if nosym else symmetries.rotation_array()
    kpoints = _build_kpoints(pwin, cell, precision, rotations)

    # ``noncolin`` and ``nspin`` say the same thing twice in a pw.x input, and
    # ``input.f90`` resolves it in one direction: noncolin wins and sets
    # nspin = 4. An input that says both consistently is common; one that says
    # nspin = 2 *and* noncolin is not, and is refused rather than silently
    # resolved, since which one the author meant is not recoverable.
    noncolin = _logical(pwin.get("system", "noncolin", False))
    lspinorb = _logical(pwin.get("system", "lspinorb", False))
    nspin = int(pwin.get("system", "nspin", 1))
    if noncolin:
        if nspin == 2:
            raise ValueError(
                "noncolin = .true. together with nspin = 2: a noncollinear "
                "calculation is nspin = 4; drop one of the two"
            )
        nspin = 4
    if nspin not in (1, 2, 4):
        raise ValueError(f"nspin = {nspin}: expected 1, 2 or 4")
    if lspinorb and nspin != 4:
        # QE's own check (``input.f90``). Spin-orbit does not commute with S_z,
        # so there is no collinear Hamiltonian for it to enter -- asking for it
        # without noncolin is an input error, not a request to be approximated.
        raise ValueError(
            "lspinorb = .true. needs noncolin = .true.: the spin-orbit term "
            "couples the two spin channels, so it has no collinear form"
        )
    # ``setup.f90``'s ``degspin``, applied in the one place that knows the rule
    # -- a k-set built later, for a denser DOS grid, has to go through the same
    # function or it counts every electron twice.
    kpoints = kpoints_for_spin(kpoints, nspin)

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
        nspin=nspin,
        calculation=str(pwin.get("control", "calculation", "scf")).lower(),
        nbnd=pwin.get("system", "nbnd"),
        occupations=str(pwin.get("system", "occupations", "fixed")).lower(),
        input_dft=pwin.get("system", "input_dft"),
        smearing=str(pwin.get("system", "smearing", "gaussian")).lower(),
        degauss=float(pwin.get("system", "degauss", 0.0)),
        input_occupations=_input_occupations(pwin),
        starting_magnetization=tuple(
            pwin.indexed("system", "starting_magnetization", structure.ntyp)
        ),
        tot_magnetization=_tot_magnetization(pwin),
        nosym=nosym,
        lspinorb=lspinorb,
        angle1=tuple(pwin.indexed("system", "angle1", structure.ntyp)),
        angle2=tuple(pwin.indexed("system", "angle2", structure.ntyp)),
    )


def _logical(value) -> bool:
    """A Fortran logical that may already have been parsed to a bool."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in (".true.", ".t.", "true", "t")


def _tot_magnetization(pwin: PwInput) -> float | None:
    """``tot_magnetization``, with QE's sentinel turned into ``None``.

    ``input_parameters.f90`` defaults it to -10000 and ``set_nelup_neldw`` tests
    ``< -9999`` -- the flag for "not given" is the *value*, and the switch it
    controls (``two_fermi_energies``) changes the physics, so the sentinel is
    resolved once here rather than compared for again downstream.
    """
    value = pwin.get("system", "tot_magnetization")
    if value is None or float(value) < -9999.0:
        return None
    return float(value)


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
    names, coordinates, if_pos = [], [], []
    for line in positions_card.lines:
        tokens = line.split()
        names.append(tokens[0])
        coordinates.append([float(t) for t in tokens[1:4]])
        # Trailing 0/1 flags -- ``if_pos``, which freezes a coordinate during a
        # relaxation. Absent means free.
        flags = tokens[4:7]
        if_pos.append([int(float(f)) for f in flags] if len(flags) == 3 else [1, 1, 1])

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
        if_pos=if_pos,
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
