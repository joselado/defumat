"""Moving the ions: optimizers for a ``calculation = 'relax'`` run.

The optimizer is host-side and knows nothing about the electrons -- it is handed
an energy and a force and returns positions. What produces those is
:mod:`pypresso.forces`, and what drives the two together is
:mod:`pypresso.workflows.relax`.
"""

from __future__ import annotations

from pypresso.relax.bfgs import BFGS, BFGSSettings
from pypresso.relax.registry import (
    DEFAULT_ION_DYNAMICS,
    get_ion_dynamics,
    ion_dynamics_schemes,
    register_ion_dynamics,
)

__all__ = ["BFGS", "BFGSSettings", "get_ion_dynamics", "register_ion_dynamics",
           "ion_dynamics_schemes", "DEFAULT_ION_DYNAMICS"]

register_ion_dynamics("bfgs", BFGS)
