"""Van der Waals corrections to a semilocal functional (``vdw_corr``).

One correction is implemented -- Grimme's D2 -- and it is the pair-potential
kind: a function of where the nuclei are and of nothing else, added to the total
energy, the force and the stress after the SCF has run. The three that are
*density* functionals (Tkatchenko-Scheffler, MBD, XDM) would enter ``v_of_rho``
instead and are refused by name in :mod:`defumat.vdw.registry`, as is D3, whose
coefficients depend on the coordination numbers.
"""

from defumat.vdw.grimme import (
    D2_BETA,
    D2_COEFFICIENTS,
    GrimmeD2,
    build_grimme_d2,
)
from defumat.vdw.registry import (
    VDW_CORRECTIONS,
    build_vdw_correction,
    canonical_vdw_corr,
    register_vdw,
)

__all__ = [
    "D2_BETA",
    "D2_COEFFICIENTS",
    "GrimmeD2",
    "VDW_CORRECTIONS",
    "build_grimme_d2",
    "build_vdw_correction",
    "canonical_vdw_corr",
    "register_vdw",
]
