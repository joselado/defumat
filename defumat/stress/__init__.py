"""The stress tensor, and what every method has to do to it afterwards.

``PW/src/stress.f90`` is the driver being mirrored: it sums the contributions,
symmetrises the result and prints it in Ry/bohr^3 and in kbar. The
symmetrisation belongs to the *stress*, not to whichever expression produced it,
so it lives here and every method goes through it.

    sigma_ab = -(1/Omega) dE/d(epsilon_ab)

is the definition, with ``epsilon`` the symmetric strain of
:mod:`defumat.stress.energy`; the sign is QE's, so that ``tr sigma / 3`` is the
pressure and a cell under compression reports a positive one.

**Two properties of the raw tensor are diagnostics before they are removed**,
and both are kept on the result:

* **The antisymmetric part must vanish.** The total energy is invariant under a
  rotation of the cell, so ``dE/d(epsilon)`` is symmetric for a reason that has
  nothing to do with the crystal -- which makes
  :attr:`Stress.rotational_residue` a free check on the whole gradient, valid on
  a structure with no symmetry at all. It comes out at 1e-17 Ry/bohr^3 when the
  gradient is right -- **except on PAW (2.6e-7) and DFT+U (6.2e-6)**, where the
  one-centre angular quadrature and the orbital rotation matrices are not
  *exactly* rotationally invariant, and where the residue is therefore the
  accuracy floor of that dataset rather than a bug. `PLAN.md`'s P11 trap 5 has
  the measurements.
* **The symmetry-forbidden components must vanish.** They do not, before
  :func:`~defumat.system.symmetry.symmetrize_matrix`, because the
  Brillouin-zone sum ran over the irreducible wedge;
  :attr:`Stress.symmetry_residue` is how large that residue was, and a big one
  means an under-converged run rather than a bug.

**Which pressure QE prints.** ``ry_kbar`` is applied at the printing boundary and
nowhere else (rule R6): :attr:`Stress.tensor` is Ry/bohr^3 throughout and
:attr:`Stress.kbar` is the conversion, so nothing downstream can pick up a
factor of 147105 by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from defumat.forces.energy import FrozenState, state_from_result
from defumat.stress.analytic import ANALYTIC_TERMS, analytic_terms
from defumat.stress.autodiff import autodiff_stress, autodiff_stress_terms
from defumat.stress.energy import (
    require_a_differentiable_cell,
    strained_energy,
    strained_energy_terms,
)
from defumat.stress.registry import (
    DEFAULT_STRESS_METHOD,
    get_stress_method,
    register_stress_method,
    stress_methods,
)
from defumat.system.symmetry import symmetrize_matrix
from defumat.units import RY_TO_KBAR

__all__ = ["Stress", "compute_stress", "stress_methods", "register_stress_method",
           "strained_energy", "strained_energy_terms", "analytic_terms",
           "autodiff_stress_terms", "ANALYTIC_TERMS", "format_stress"]

register_stress_method("autodiff", autodiff_stress)


@dataclass
class Stress:
    """The stress tensor, in Ry/bohr^3, cartesian."""

    #: ``(3, 3)``: what QE prints -- symmetrised over the crystal's point group.
    tensor: np.ndarray
    #: Which expression it came from.
    method: str
    #: The individual contributions, when the method computed them separately.
    #: Each is already symmetrised, so they sum to :attr:`tensor`.
    terms: dict = field(default_factory=dict)
    #: ``max |sigma - sigma^T| / 2`` of the *raw* gradient. Rotational
    #: invariance of the energy makes this zero independently of the crystal, so
    #: it is a check on the gradient rather than on the symmetry group.
    rotational_residue: float = 0.0
    #: ``max |sigma_sym - sigma_raw|``: how much the point-group average moved
    #: the tensor. Large means an under-converged k-set or density.
    symmetry_residue: float = 0.0

    @property
    def kbar(self) -> np.ndarray:
        """The tensor in kbar, which is the second block QE prints."""
        return self.tensor * RY_TO_KBAR

    @property
    def pressure(self) -> float:
        """``tr sigma / 3`` in Ry/bohr^3 -- equal to ``-dE/dV``."""
        return float(np.trace(self.tensor) / 3.0)

    @property
    def pressure_kbar(self) -> float:
        """The ``P=`` on QE's stress line."""
        return self.pressure * RY_TO_KBAR

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Stress(P={self.pressure_kbar:.2f} kbar, method={self.method!r})"


def compute_stress(
    calculation, result_or_state, method: str | None = None, terms: bool = False
) -> Stress:
    """The stress of ``calculation`` in its converged state.

    Args:
        calculation: the :class:`~defumat.scf.driver.Calculation` the state
            belongs to -- its cell is where the derivative is evaluated.
        result_or_state: an :class:`~defumat.scf.driver.SCFResult`, or the
            :class:`~defumat.forces.energy.FrozenState` taken from one.
        method: ``'autodiff'`` (default). ``'analytic'`` is refused by name --
            see :mod:`defumat.stress.registry` for why there is no analytic
            total -- and its terms are reached through :func:`analytic_terms`.
        terms: also compute the decomposition, which costs a forward-mode
            Jacobian (nine passes) instead of one reverse pass.
    """
    name = (method or DEFAULT_STRESS_METHOD).lower()
    if name == "analytic":
        raise NotImplementedError(
            "there is no analytic stress total: stres_us (the projectors' own "
            "strain derivative) and addusstress are not transcribed, so a sum "
            "of the terms that are would be missing the whole nonlocal "
            "pseudopotential. Use method='autodiff', which has every term, and "
            "defumat.stress.analytic_terms for the term-by-term cross-check"
        )
    require_a_differentiable_cell(calculation)
    state = (
        result_or_state
        if isinstance(result_or_state, FrozenState)
        else state_from_result(result_or_state)
    )

    raw = np.asarray(get_stress_method(name)(calculation, state), dtype=float)
    rotational = float(np.abs(raw - raw.T).max() / 2.0)
    # The antisymmetric part is round-off in an exact gradient, so it is
    # measured and then removed rather than left to be averaged away by a
    # symmetry group the crystal may not have.
    raw = 0.5 * (raw + raw.T)
    tensor = _symmetrised(calculation, raw)

    breakdown = {}
    if terms:
        breakdown = {
            key: _symmetrised(calculation, 0.5 * (np.asarray(value) + np.asarray(value).T))
            for key, value in autodiff_stress_terms(calculation, state).items()
        }

    return Stress(
        tensor=tensor,
        method=name,
        terms=breakdown,
        rotational_residue=rotational,
        symmetry_residue=float(np.abs(tensor - raw).max()),
    )


def _symmetrised(calculation, tensor: np.ndarray) -> np.ndarray:
    """``symmatrix`` with the calculation's own group, or a no-op under ``nosym``."""
    if calculation.system.nosym or calculation.symmetries.nsym <= 1:
        return tensor
    return symmetrize_matrix(tensor, calculation.system.cell, calculation.symmetries)


def format_stress(stress: Stress) -> str:
    """The stress block ``stress.f90`` writes, in QE's layout.

    Ry/bohr^3 on the left and kbar on the right, with the pressure on the header
    line -- the same three-line table, so an output can be read next to
    ``pw.x``'s without conversion.
    """
    lines = [
        f"          total   stress  (Ry/bohr**3)                   (kbar)"
        f"     P={stress.pressure_kbar:12.2f}"
    ]
    for row in range(3):
        left = "".join(f"{value:13.8f}" for value in stress.tensor[row])
        right = "".join(f"{value:12.2f}" for value in stress.kbar[row])
        lines.append(f"{left}    {right}")
    return "\n".join(lines)
