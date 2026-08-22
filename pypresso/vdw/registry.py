"""The name registry for van der Waals corrections (``vdw_corr``).

QE offers five and selects between them with one string
(``Modules/set_vdw_corr.f90``); this is the same switch, written as the
project's standard registry so that adding D3 or Tkatchenko-Scheffler is a new
file plus a registration rather than an edit to a branch in the driver.

**The unimplemented ones are refused by name.** ``set_vdw_corr`` warns and
carries on with no correction at all for a name it does not recognise, which for
an input asking for ``grimme-d3`` means silently getting plain PBE -- a 30 meV
error on a layered crystal and no message in the output that survives a grep for
"Error". Here an unimplemented correction stops the run and says what would have
to be written.
"""

from __future__ import annotations

from typing import Callable

__all__ = [
    "VDW_CORRECTIONS",
    "register_vdw",
    "canonical_vdw_corr",
    "build_vdw_correction",
    "vdw_options",
]

#: name -> builder ``(cell, structure, **options) -> correction``.
VDW_CORRECTIONS: dict[str, Callable] = {}

#: ``set_vdw_corr``'s spellings, mapped onto the canonical name. QE accepts
#: several per correction and an input in the wild uses all of them.
_ALIASES = {
    "grimme-d2": "grimme-d2", "dft-d": "grimme-d2",
    "grimme-d3": "grimme-d3", "dft-d3": "grimme-d3",
    "ts": "ts", "ts-vdw": "ts", "tkatchenko-scheffler": "ts",
    "mbd": "mbd", "many-body-dispersion": "mbd", "mbd_vdw": "mbd",
    "xdm": "xdm",
    "none": "none", "": "none",
}

#: What each unimplemented correction would need, so the refusal is a statement
#: about the work rather than a shrug.
_NOT_IMPLEMENTED = {
    "grimme-d3": (
        "Grimme-D3: the coefficients depend on the coordination number of each "
        "atom, so C6 is a function of the geometry rather than a table lookup "
        "and its derivative is part of the force (Modules/dftd3/)"
    ),
    "ts": (
        "Tkatchenko-Scheffler: the coefficients are rescaled by each atom's "
        "Hirshfeld volume, which is a functional of the self-consistent density, "
        "so the correction enters v_of_rho and is not a pair potential over the "
        "nuclei (Modules/tsvdw.f90)"
    ),
    "mbd": (
        "many-body dispersion: a coupled-oscillator eigenproblem over the atoms "
        "on top of Tkatchenko-Scheffler's density dependence (MBD/)"
    ),
    "xdm": (
        "exchange-hole dipole moment: the coefficients come from the exchange "
        "hole of the converged density, so it is density-dependent like TS "
        "(Modules/xdm_dispersion.f90)"
    ),
}


def register_vdw(name: str):
    """Register a builder under ``name``. Used as a decorator."""

    def decorate(builder):
        VDW_CORRECTIONS[name] = builder
        return builder

    return decorate


def canonical_vdw_corr(name: str | None) -> str:
    """QE's spelling of a ``vdw_corr`` value reduced to the canonical one.

    Unlike ``set_vdw_corr``, an unknown name is an error rather than a warning
    followed by no correction: a typo in ``vdw_corr`` should not silently give a
    different functional.
    """
    if name is None:
        return "none"
    key = name.strip().lower()
    if key not in _ALIASES:
        known = ", ".join(sorted(set(_ALIASES.values()) - {"none"}))
        raise ValueError(f"unknown vdw_corr = {name!r}; pw.x has {known}")
    return _ALIASES[key]


def build_vdw_correction(name: str | None, cell, structure, **options):
    """The correction ``vdw_corr`` names, or ``None`` for ``'none'``."""
    canonical = canonical_vdw_corr(name)
    if canonical == "none":
        return None
    if canonical not in VDW_CORRECTIONS:
        raise NotImplementedError(
            f"vdw_corr = {name!r} is not implemented -- {_NOT_IMPLEMENTED[canonical]}"
        )
    return VDW_CORRECTIONS[canonical](cell, structure, **options)


def vdw_options(system) -> dict:
    """The builder arguments a :class:`~pypresso.system.builder.System` implies.

    One place translates input variables into a correction's parameters, so the
    driver does not grow a branch per correction and a new one declares its own
    inputs here beside its builder.
    """
    if canonical_vdw_corr(system.vdw_corr) != "grimme-d2":
        return {}
    return {
        "s6": system.london_s6,
        "rcut": system.london_rcut,
        # ``()`` means the input said nothing; QE's -1 sentinel inside a
        # populated tuple means the same thing for one species and is handled by
        # the builder.
        "c6": system.london_c6 or None,
        "rvdw": system.london_rvdw or None,
    }
