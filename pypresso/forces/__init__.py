"""Forces on the atoms, and what every method has to do to them afterwards.

``PW/src/forces.f90`` is the driver being mirrored: it sums the contributions,
imposes that the total force on the crystal is zero, symmetrises the result, and
only then applies ``if_pos``. Those last three steps belong to the *force*, not
to whichever expression produced it, so they live here and both methods go
through them.

Why each of them is not optional:

* **The total force must vanish.** Nothing in the energy expression knows that
  translating the whole crystal cannot change it; the residue that survives is
  the error the FFT grid and the G-vector cutoff make, and QE subtracts its
  average for the same reason (``sumfor``). It is a useful diagnostic before it
  is removed -- a large one means something is under-converged -- so it is kept
  on the result.
* **The force must be symmetrised.** The Brillouin-zone sum over the irreducible
  wedge is exact for a scalar and not for a vector, so a force computed from a
  reduced k-set has components along directions the crystal's symmetry forbids.
  ``symvector`` projects them out; on an ideal crystal it is what makes the
  force identically zero rather than 1e-4.
* **``if_pos`` is applied last**, after the forces are reported, exactly as QE
  does it: a frozen coordinate still *has* a force, it is simply not allowed to
  move along it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pypresso.forces.analytic import analytic_forces
from pypresso.forces.autodiff import autodiff_forces
from pypresso.forces.energy import (
    FrozenState,
    frozen_energy,
    reject_magnetic_field,
    reject_potential_only,
    state_from_result,
)
from pypresso.forces.spiral import (
    SpiralGradient,
    compute_spiral_gradient,
    spiral_energy,
)
from pypresso.forces.registry import (
    DEFAULT_FORCE_METHOD,
    force_methods,
    get_force_method,
    register_force_method,
)
from pypresso.system.symmetry import atom_mapping, symmetrize_vector

__all__ = ["Forces", "compute_forces", "FrozenState", "frozen_energy",
           "state_from_result", "force_methods", "register_force_method",
           "SpiralGradient", "compute_spiral_gradient", "spiral_energy"]

register_force_method("autodiff", autodiff_forces)
register_force_method("analytic", analytic_forces)


@dataclass
class Forces:
    """The forces on the atoms, in Ry/bohr, cartesian."""

    #: ``(nat, 3)``: what QE prints -- summed, translation-corrected and
    #: symmetrised, but *not* yet masked by ``if_pos``.
    forces: np.ndarray
    #: Which expression they came from.
    method: str
    #: ``sum_a F_a`` before it was subtracted, one number per cartesian
    #: component. QE's ``sumfor``; large values mean an under-converged run.
    total_before_correction: np.ndarray
    #: The individual contributions, when the method computes them separately.
    terms: dict = field(default_factory=dict)

    #: ``if_pos`` as ones and zeros, or ``None`` when every coordinate is free.
    _free: np.ndarray = field(default=None, repr=False)

    @property
    def constrained(self) -> np.ndarray:
        """The forces an optimizer may move along: ``forces * if_pos``."""
        return self.forces if self._free is None else self.forces * self._free

    @property
    def max_force(self) -> float:
        """``max |F|`` over the components that are free to move."""
        return float(np.abs(self.constrained).max()) if self.forces.size else 0.0


def compute_forces(calculation, result_or_state, method: str | None = None) -> Forces:
    """The forces on the atoms of ``calculation`` in its converged state.

    Args:
        calculation: the :class:`~pypresso.scf.driver.Calculation` the state
            belongs to -- its positions are where the force is evaluated.
        result_or_state: an :class:`~pypresso.scf.driver.SCFResult`, or the
            :class:`~pypresso.forces.energy.FrozenState` taken from one.
        method: ``'autodiff'`` (default) or ``'analytic'``.
    """
    state = (
        result_or_state
        if isinstance(result_or_state, FrozenState)
        else state_from_result(result_or_state)
    )
    name = (method or DEFAULT_FORCE_METHOD).lower()
    # **Before dispatch, so that every method is covered by one check.** The
    # autodiff route reaches these through :func:`~pypresso.forces.energy.
    # energy_at`; the analytic route is a transcription of QE's six expressions
    # and shares nothing with it, so a refusal written into the functional did
    # not reach ``method='analytic'`` at all -- a Tran-Blaha run came back with
    # a force, and a run under a magnetic field came back with one from either.
    reject_potential_only(calculation)
    reject_magnetic_field(calculation)
    raw = get_force_method(name)(calculation, state)

    terms = {}
    if isinstance(raw, tuple):
        raw, terms = raw
    forces = np.asarray(raw, dtype=float)

    total = forces.sum(axis=0)
    forces = forces - total / forces.shape[0]

    structure = calculation.system.structure
    if calculation.symmetries.nsym > 1 and not calculation.system.nosym:
        mapping = atom_mapping(
            calculation.system.cell, structure, calculation.symmetries
        )
        forces = np.asarray(
            symmetrize_vector(
                forces, calculation.system.cell, calculation.symmetries, mapping
            )
        )

    return Forces(
        forces=forces,
        method=name,
        total_before_correction=total,
        terms={key: np.asarray(value) for key, value in terms.items()},
        _free=structure.free,
    )
