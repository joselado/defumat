"""Projection of the Kohn-Sham states onto pseudo-atomic orbitals.

``PP/src/projwfc.f90``'s ``projwave``: what the ``natomwfc`` projection columns
are (:mod:`~pypresso.projwfc.channels`) and the projection itself
(:mod:`~pypresso.projwfc.projections`).

What is *done* with a projection -- the projected density of states, the Löwdin
charges, the spilling parameter -- lives in :mod:`pypresso.workflows.pdos`
instead, and not for tidiness: the projected DOS is the plain DOS with a weight
in front of the delta, so it goes through
:data:`pypresso.workflows.dos.DOS_SCHEMES`, and rule R3 has this layer sitting
*below* the workflows rather than importing from them.

:mod:`~pypresso.projwfc.angular_momentum` is the one thing here that is not a
transcription of ``projwave``: the same projection contracted with the angular
momentum operators instead of squared, which gives the site-resolved ``<L>``,
``<S>`` and ``<J>`` of Elk's tasks 15/16 (P48). ``pw.x`` has the *cell's*
orbital magnetization (``orbm_kubo``) and nothing per atom.
"""

from pypresso.projwfc.angular_momentum import (
    AtomicMoments,
    SiteAngularMomentum,
    angular_momenta,
    orbital_matrices,
)
from pypresso.projwfc.channels import (
    AtomicChannel,
    L_LABELS,
    M_LABELS,
    channel_table,
    projection_channels,
)
from pypresso.projwfc.projections import (
    PROJECTION_KINDS,
    ProjectionSymmetry,
    atomic_projections,
    build_projection_symmetry,
)

__all__ = [
    "AtomicChannel",
    "AtomicMoments",
    "L_LABELS",
    "M_LABELS",
    "PROJECTION_KINDS",
    "ProjectionSymmetry",
    "SiteAngularMomentum",
    "angular_momenta",
    "atomic_projections",
    "build_projection_symmetry",
    "channel_table",
    "orbital_matrices",
    "projection_channels",
]
