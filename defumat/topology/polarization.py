"""The Berry-phase electronic polarization: King-Smith and Vanderbilt's phase.

The polarization of a crystal is not the dipole of its unit cell -- that integral
depends on where the cell is cut and is not a property of the crystal at all.
What *is* a property is a phase: the Berry phase accumulated by the occupied
manifold as ``k`` is carried once across the Brillouin zone along one reciprocal
lattice vector (King-Smith and Vanderbilt, PRB 47, 1651 (1993); Resta, Rev. Mod.
Phys. 66, 899 (1994)). It is defined **modulo a quantum**, and every physical
statement made with it is a *difference* between two geometries, never a raw
value.

Structurally this is the same object :mod:`defumat.topology.wilson` already
computes and for the same reason: a determinant of overlaps between neighbouring
k-points is blind to the unitary mixing a degenerate eigensolver leaves (rule
D4), so the string product is gauge-invariant even though no individual state is.
The difference from a Wilson loop is what is kept -- a Wilson loop needs the
*eigenvalues* of the matrix product, so each factor is unitarised and the matrix
structure survives; a polarization needs only the total phase, so each factor is
reduced to its determinant immediately and the product is a product of scalars.

Three things here are conventions rather than physics, and all three are QE's
(``PW/src/bp_c_phase.f90``), transcribed so that the two codes' output can be
compared line by line:

* **the branch.** The string phases are averaged as complex numbers first, the
  average's argument becomes a reference angle, and each string's phase is then
  placed around it. Averaging the raw angles instead would break wherever a
  string sits near the branch cut, which is a function of nothing physical;
* **the quantum**, which is 2 for an unpolarized run with all-even valences and
  1 otherwise. It is 2 because a spin-degenerate band holds two electrons, so
  moving one filled band by one lattice vector moves the phase by 2;
* **the ionic phase**, ``Z_v`` times the atom's crystal coordinate along the
  string direction, reduced mod 1 for an odd valence and mod 2 for an even one.

The electronic phase is **doubled for** ``nspin = 1``, which is the same
``degspin`` trap :mod:`defumat.response.conductivity` records: the strings hold
one spin channel and an unpolarized band holds two electrons.

Refused by name: a metal (a Berry phase is a property of a gapped manifold and
there is none), a spin spiral (the two spinor components live on different
spheres, so a string overlap is not a single gather), and ``nspin = 2`` (two
channels give two independent string sets, which is a layout this does not have
rather than a missing term).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from defumat.topology.links import link_phase

__all__ = [
    "StringPhases",
    "Polarization",
    "string_phase",
    "combine_string_phases",
    "ionic_phase",
    "polarization_quantum",
]

#: Elementary charge in coulomb and the bohr radius in metre, as QE's
#: ``Modules/constants.f90`` carries them -- the C/m^2 conversion is the only
#: place a non-atomic unit appears in this module.
ELECTRON_SI = 1.602176634e-19
BOHR_RADIUS_SI = 0.529177210903e-10


def string_phase(states, k_batch: int | None | str = "default",
                 closing_shift=None) -> float:
    """The Berry phase of one string of k-points, in radians.

    ``states`` holds the occupied manifold at the ``N`` points of the string
    **in order and without repeating the endpoint**; the string is closed from
    the last point back to the first, displaced by ``closing_shift`` -- the
    reciprocal lattice vector along the string direction.

    That is one point fewer than QE stores. ``kp_strings`` lays out ``nppstr``
    points spanning the *whole* reciprocal vector, so its last point is the
    first one's periodic image and is diagonalised a second time; the wrap is
    then applied to the coefficients anyway (``bp_c_phase.f90``'s ``map_g``,
    ``gtr = g - gpar``). Dropping the repeat leaves the same ``nppstr - 1``
    links and the same product, and saves one diagonalisation per string.

    The phase is ``arg`` of the product of the links' determinants, which is
    ``AIMAG(LOG(zeta))`` in the Fortran and carries **its** sign.
    """
    n = states.nk
    if n < 2:
        raise ValueError("a polarization string needs at least two k-points")
    zero = np.zeros(3, dtype=int)
    shift = zero if closing_shift is None else np.asarray(closing_shift, dtype=int)

    interior = [(i, i + 1, zero) for i in range(n - 1)]
    links = list(states.overlaps(interior, k_batch=k_batch))
    links.append(states.overlaps([(n - 1, 0, shift)], k_batch=1)[0])

    # The determinant of each link, normalised -- ``link_phase`` is
    # ``det M / |det M|``, whose argument is the determinant's argument and
    # which cannot underflow the way a raw determinant of a large manifold does.
    total = np.complex128(1.0)
    for matrix in links:
        total *= np.complex128(link_phase(matrix))
    return float(np.angle(total))


def combine_string_phases(phases, weights, nspin: int = 1) -> "StringPhases":
    """QE's branch-consistent average over the strings of one spin channel.

    ``phases`` are the raw per-string phases in radians as :func:`string_phase`
    returns them, ``weights`` their (normalised) weights in the transverse
    plane. Returns the per-string phases in units of ``2 pi`` and the channel
    average, both after the branch fixing.

    **The averaging is the part that must not be simplified.** Averaging angles
    directly is wrong wherever a string sits near the cut, so
    ``bp_c_phase.f90`` averages the unit complex numbers, takes the argument of
    that average as a reference ``theta0``, and re-expresses each string's phase
    relative to it. A second pass then moves any string still a whole turn away
    from the *first* string onto the same branch. Both passes are transcribed
    because a string set that never needs them and one that does look identical
    until the answer is wrong by a quantum.
    """
    phases = np.asarray(phases, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if phases.shape != weights.shape:
        raise ValueError(
            f"{phases.size} string phases and {weights.size} weights"
        )
    two_pi = 2.0 * np.pi

    # --- average as complex numbers, and take the reference angle from it
    unit = np.exp(1j * phases)
    average = np.sum(weights * unit)
    theta0 = float(np.angle(average))

    # --- place every phase around theta0, then fold to (-pi, pi]
    placed = theta0 + np.angle(unit / average)
    placed = placed - two_pi * np.rint(placed / two_pi)

    # --- and put any string a whole turn from the first one back onto its branch
    reference = placed[0] / two_pi
    scaled = placed / two_pi
    placed = np.where(
        np.abs(scaled + 1.0 - reference) < np.abs(scaled - reference),
        placed + two_pi,
        placed,
    )
    scaled = placed / two_pi
    placed = np.where(
        np.abs(scaled - 1.0 - reference) < np.abs(scaled - reference),
        placed - two_pi,
        placed,
    )

    per_string = placed / two_pi
    # **Fold the single channel first, then double**, which is the order
    # ``bp_c_phase.f90`` uses (``pdl_elec_up = phiup/tpi - nint(...)``, and only
    # then ``pdl_elec_tot = pdl_elec_up + pdl_elec_dw``). Doubling first and
    # folding the sum onto ``[-1/2, 1/2)`` is a different number whenever the
    # doubled phase leaves that interval, and it is wrong rather than merely
    # rebranched: an all-even cell's quantum is **2**, so shifting the total by
    # 1 moves it by *half* a quantum. Neither AlAs (odd valence, quantum 1) nor
    # a centrosymmetric cell (whose phase is 0 or exactly half a quantum, a set
    # the shift maps onto itself) can see the difference, which is why the
    # order is transcribed rather than inferred from a comparison.
    channel = _fold(float(np.sum(weights * placed) / two_pi), 1.0)
    if nspin == 1:
        # One string set, two electrons per band. The same ``degspin`` factor
        # that a caller-built k-set has to apply by hand elsewhere in this
        # package -- and the one that is invisible in the shape of the answer.
        per_string = per_string * 2.0
        channel = channel * 2.0
    return StringPhases(phases=per_string, weights=weights, total=channel)


def _fold(value: float, quantum: float) -> float:
    """``value`` reduced onto ``[-quantum/2, quantum/2)``, QE's ``- nint(...)``."""
    return float(value - quantum * np.rint(value / quantum))


def ionic_phase(positions_crystal, valences, gdir: int):
    """The ions' contribution: ``Z_v`` times the crystal coordinate, reduced.

    ``positions_crystal`` is ``(nat, 3)`` in crystal coordinates and
    ``valences`` the ``(nat,)`` pseudopotential valence charges. Returns
    ``(per-atom phases, total, quantum)``, all in units of ``2 pi``.

    QE writes this as ``Z_v tau . gpar`` with ``tau`` in units of ``alat`` and
    ``gpar`` the reciprocal vector in ``2 pi / alat``, which is exactly ``Z_v``
    times the atom's crystal coordinate along ``gdir``. Each atom is reduced mod
    1 if its valence is odd and mod 2 if it is even, and the *total* mod 1 if
    any valence is odd -- so a cell of even-valence species keeps the larger
    quantum and one odd species collapses it for the whole crystal.
    """
    positions = np.asarray(positions_crystal, dtype=float)
    charges = np.asarray(valences, dtype=float)
    if positions.shape[0] != charges.shape[0]:
        raise ValueError(
            f"{positions.shape[0]} positions and {charges.shape[0]} valences"
        )
    odd = np.rint(charges).astype(int) % 2 == 1
    raw = charges * positions[:, gdir]
    quanta = np.where(odd, 1.0, 2.0)
    per_atom = raw - quanta * np.rint(raw / quanta)

    total_quantum = 1.0 if bool(np.any(odd)) else 2.0
    total = _fold(float(np.sum(per_atom)), total_quantum)
    return per_atom, total, total_quantum


def polarization_quantum(valences, nspin: int = 1) -> float:
    """The quantum the total phase is defined modulo: 2 or 1.

    Two only when nothing forces it down -- an unpolarized run whose species all
    carry an even valence. An odd valence anywhere, or a run that resolves the
    two spin channels, makes it 1.
    """
    charges = np.rint(np.asarray(valences, dtype=float)).astype(int)
    return 2.0 if (nspin == 1 and not bool(np.any(charges % 2 == 1))) else 1.0


@dataclass(frozen=True)
class StringPhases:
    """The per-string Berry phases of one spin channel, in units of ``2 pi``."""

    phases: np.ndarray
    weights: np.ndarray
    total: float

    def __len__(self) -> int:
        return int(self.phases.size)


@dataclass(frozen=True)
class Polarization:
    """The Berry-phase polarization along one reciprocal lattice direction.

    Every phase is in units of ``2 pi`` and every one of them is defined only
    modulo its quantum, which is carried beside it rather than left implicit.
    :attr:`total_phase` is what a *difference* between two geometries should be
    taken of; the values in physical units are that phase times the length of
    the lattice vector, and inherit the same ambiguity.
    """

    #: Which reciprocal lattice vector the string runs along, 0-based.
    gdir: int
    #: The ions' phase and the electrons' phase, in units of ``2 pi``.
    ionic_phase: float
    electronic_phase: float
    #: Their sum, reduced onto ``[-quantum/2, quantum/2)``.
    total_phase: float
    #: The quantum the total is defined modulo, in the same units.
    quantum: float
    #: Per-atom ionic phases, in the order the structure carries the atoms.
    ion_phases: np.ndarray
    #: The strings' own phases, before the weighted sum.
    strings: StringPhases
    #: ``|a_gdir|`` in bohr, and the cell volume in bohr^3.
    lattice_length: float
    volume: float
    #: The unit vector along ``a_gdir``, which is the direction the phase
    #: measures a polarization *along* -- not the direction of ``P`` itself
    #: unless the crystal's symmetry makes them the same.
    direction: np.ndarray
    #: How many k-points each string carried, and how many strings there were.
    points_per_string: int = 0

    @property
    def polarization(self) -> float:
        """``P`` in ``(e/Omega).bohr`` -- QE's first line."""
        return self.lattice_length * self.total_phase

    @property
    def polarization_quantum_e_omega_bohr(self) -> float:
        return self.lattice_length * self.quantum

    @property
    def polarization_e_bohr2(self) -> float:
        """``P`` in ``e/bohr^2``."""
        return self.lattice_length * self.total_phase / self.volume

    @property
    def polarization_si(self) -> float:
        """``P`` in ``C/m^2``."""
        factor = ELECTRON_SI / BOHR_RADIUS_SI**2
        return self.polarization_e_bohr2 * factor

    @property
    def quantum_si(self) -> float:
        factor = ELECTRON_SI / BOHR_RADIUS_SI**2
        return self.lattice_length * self.quantum / self.volume * factor
