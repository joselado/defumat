"""The Berry-phase polarization, end to end from a converged density.

One entry point, :func:`run_polarization`, which is ``pw.x``'s ``lberry`` run:
a fixed-density diagonalisation on strings of k-points along one reciprocal
lattice vector, the occupied manifold's Berry phase accumulated along each
string, and the ions' phase added to it.

The strings are walked **one at a time**. That is the memory decision
:func:`~pypresso.topology.wilson.wannier_centers` already makes and it matters
more here, because a polarization mesh is the whole transverse plane rather
than half a zone: the resident set is one string's occupied manifold,
``npoints * nbnd * npwx`` complex, and the number of strings never enters it.
"""

from __future__ import annotations

import numpy as np

import jax.numpy as jnp

from pypresso.system.builder import System
from pypresso.topology.mesh import string_mesh
from pypresso.topology.polarization import (
    Polarization,
    combine_string_phases,
    ionic_phase,
    polarization_quantum,
    string_phase,
)
from pypresso.workflows.topology import _source

__all__ = ["run_polarization"]


def run_polarization(
    system: System,
    pseudos,
    density: jnp.ndarray,
    gdir: int = 2,
    nppstr: int = 7,
    transverse: tuple[int, int] = (4, 4),
    shift: tuple[int, int, int] = (0, 0, 0),
    nocc: int | None = None,
    nbnd: int | None = None,
    conv_thr: float = 1.0e-8,
    k_batch: int | None | str = "default",
    becsum: tuple = (),
    ns: jnp.ndarray | None = None,
) -> Polarization:
    """The Berry-phase polarization along reciprocal lattice vector ``gdir``.

    Args:
        gdir: which reciprocal lattice vector the strings run along, **0-based**
            here where ``pw.x``'s input variable of the same name is 1-based.
        nppstr: k-points per string in QE's counting, which repeats the
            endpoint; the mesh built here holds ``nppstr - 1`` distinct points
            and closes the string with the reciprocal lattice vector instead.
            Passing the number from a ``pw.x`` input therefore gives the same
            calculation.
        transverse: how many strings along the two crystal directions other
            than ``gdir``.
        shift: QE's ``k1, k2, k3`` half-step offsets of the transverse grid.
        nocc: the occupied band count, defaulting to the electron count over
            one or two as a band is a spinor or not.

    The polarization is defined **modulo a quantum** and the result carries it.
    A single value is not a physical statement on its own: what is meaningful is
    a *difference* between two geometries taken on the same branch, which is
    what a Born effective charge and a piezoelectric constant are made of.
    """
    _refuse_ungapped(system)
    if system.nspin == 2:
        raise NotImplementedError(
            "a Berry-phase polarization with nspin = 2 is not implemented: the "
            "two channels give two independent string sets and one phase each, "
            "which is a layout this does not have rather than a missing term "
            "(pw.x carries them as nspin_lsda and sums the two at the end)"
        )
    if getattr(system, "spiral_q", None) is not None:
        raise NotImplementedError(
            "a Berry-phase polarization of a spin spiral is not implemented: "
            "the two spinor components live on spheres centred at k +- q/2, so "
            "an overlap between neighbouring k-points is not a single gather"
        )

    source = _source(system, pseudos, density, nocc, nbnd, conv_thr, k_batch,
                     becsum=becsum, ns=ns)
    axis = int(gdir) % 3
    mesh = string_mesh(transverse, int(nppstr) - 1, gdir=axis, shift=shift)
    nstring, npoints = mesh.shape

    # One string at a time: the states of the whole mesh are never resident.
    phases = np.empty(nstring)
    for index in range(nstring):
        states = source.states(mesh.points[index])
        phases[index] = string_phase(states, k_batch=k_batch,
                                     closing_shift=mesh.span2)
        del states

    weights = np.full(nstring, 1.0 / nstring)
    strings = combine_string_phases(phases, weights, nspin=int(system.nspin))

    valences = np.array(
        [float(pseudos[t].z_valence) for t in np.asarray(system.structure.types)]
    )
    positions = np.asarray(system.structure.positions_crystal(system.cell))
    ion_phases, ionic, _ = ionic_phase(positions, valences, axis)

    quantum = polarization_quantum(valences, nspin=int(system.nspin))
    # Not reduced again. ``bp_c_phase.f90`` adds the two contributions and
    # reports the sum against ``mod_tot`` without folding it, and each half has
    # already been reduced by *its own* quantum -- which is not always the
    # total's. Folding here would put a value on a different branch from the one
    # pw.x prints, for no gain: everything physical is a difference.
    total = float(strings.total + ionic)

    cell = system.cell
    at = np.asarray(cell.to_cartesian(np.eye(3)))
    lattice_vector = at[axis]
    length = float(np.linalg.norm(lattice_vector))

    return Polarization(
        gdir=axis,
        ionic_phase=float(ionic),
        electronic_phase=float(strings.total),
        total_phase=total,
        quantum=float(quantum),
        ion_phases=ion_phases,
        strings=strings,
        lattice_length=length,
        volume=float(cell.volume),
        direction=lattice_vector / length,
        points_per_string=int(npoints),
    )


def _refuse_ungapped(system: System) -> None:
    """A Berry phase is a property of a gapped manifold; a metal has none.

    The same refusal every invariant in :mod:`pypresso.topology` makes, and it
    is worth making at the workflow rather than letting the gap check inside
    :class:`~pypresso.workflows.topology.DFTSource` report it per k-point: a
    smeared occupation does not say which bands are in the manifold, and a
    string phase taken over an arbitrary count is a confident number with no
    meaning.
    """
    occupations = str(getattr(system, "occupations", "fixed")).lower()
    if occupations not in ("fixed", "from_input"):
        raise NotImplementedError(
            "a Berry-phase polarization of a metal is not defined: the phase is "
            f"a property of an isolated occupied manifold and {occupations} "
            "smears the occupation across the Fermi level, so which bands the "
            "string carries is not determined. pw.x refuses the same "
            "combination (bp_c_phase.f90 needs a fixed band count)"
        )
