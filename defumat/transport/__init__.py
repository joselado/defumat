"""Vertical tunnelling transport through a two-dimensional material.

``PLAN.md`` P66. An electron enters at a point (the tip) and leaves through a
plane below (the substrate), so what is computed is the nonlocal Green's
function between the two -- which reduces to a Tersoff-Hamann image exactly
when the material has one band to tunnel through, and departs from it by an
interference when it has more. Neither ``pw.x`` nor Elk computes it.

The entry point is :func:`defumat.workflows.transport.run_vertical_transport`,
or :meth:`defumat.calculator.Calculator.get_vertical_transport`.
"""

from defumat.transport.green import VerticalTransport, transmission
from defumat.transport.substrate import (
    exit_overlap,
    spin_projector,
    surface_area,
    volume_overlap,
)

__all__ = ["VerticalTransport", "transmission", "exit_overlap",
           "volume_overlap", "surface_area", "spin_projector"]
