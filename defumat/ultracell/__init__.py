"""Ultra long-range calculations: a modulation over many unit cells.

``PLAN.md`` P88. See :mod:`defumat.ultracell.grid` for the one index map the
subpackage rests on and :mod:`defumat.ultracell.driver` for the loop.
"""

from defumat.ultracell.density import ultracell_density
from defumat.ultracell.energy import ultracell_energy, ultracell_entropy
from defumat.ultracell.driver import (
    UltracellResult,
    require_an_ultracell_regime,
    run_ultracell,
)
from defumat.ultracell.grid import Ultracell, folded_kpoints
from defumat.ultracell.hamiltonian import multiplet_cut, ultracell_matrix
from defumat.ultracell.states import UltracellStates, ultracell_band_density
from defumat.ultracell.seed import (
    reference_axis,
    seeded_becsum,
    seeded_density,
    warn_if_the_seed_leaves_the_closed_sector,
)
from defumat.ultracell.potential import (
    delta_potential,
    ultracell_potential,
    with_external_potential,
)

__all__ = [
    "Ultracell",
    "UltracellStates",
    "ultracell_band_density",
    "UltracellResult",
    "folded_kpoints",
    "run_ultracell",
    "ultracell_matrix",
    "ultracell_density",
    "ultracell_energy",
    "ultracell_entropy",
    "ultracell_potential",
    "delta_potential",
    "with_external_potential",
    "multiplet_cut",
    "require_an_ultracell_regime",
    "seeded_density",
    "seeded_becsum",
    "reference_axis",
    "warn_if_the_seed_leaves_the_closed_sector",
]
