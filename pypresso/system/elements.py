"""Element symbols and atomic numbers.

The one place a *label* becomes a **Z**. Nothing in the ground state needs it --
a pseudopotential carries its own valence charge and its own radial grid, and
which element it was generated for is metadata the SCF never reads. What needs
it is a correction parameterised by element rather than by pseudopotential:
Grimme's D2 dispersion coefficients (:mod:`pypresso.vdw.grimme`) are tabulated
against Z, so a run has to say which element each species *is*.

:func:`atomic_number` follows ``upflib/atomic_number.f90``'s rules for reading a
species label rather than requiring a bare symbol, because QE's inputs do not
supply one: two inequivalent atoms of the same element are written ``C1`` and
``C2``, or ``Fe_up`` and ``Fe_dw``, and both must resolve to the element. The
rule is that a trailing digit, ``-`` or ``_`` ends the symbol.
"""

from __future__ import annotations

__all__ = ["ELEMENTS", "atomic_number"]

#: Element symbols in order of atomic number, ``ELEMENTS[Z - 1]``. The list is
#: ``atomic_number.f90``'s, up to Z = 118.
ELEMENTS: tuple[str, ...] = (
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)

_BY_SYMBOL = {symbol.lower(): z for z, symbol in enumerate(ELEMENTS, start=1)}


def element_symbol(label: str) -> str:
    """The element symbol a species label names, e.g. ``Fe_up -> Fe``.

    ``atomic_number.f90``'s cases, in its order: a one-character label is the
    symbol; a label whose second character is a digit, ``-`` or ``_`` is a
    one-character symbol with a tag after it; anything else is a two-character
    symbol. Capitalisation is normalised, so ``FE``, ``fe`` and ``Fe`` agree.
    """
    text = label.strip()
    if not text:
        raise ValueError("an empty species label names no element")
    if len(text) == 1 or text[1] in "0123456789-_":
        return text[0].upper()
    return text[0].upper() + text[1].lower()


def atomic_number(label: str) -> int:
    """``Z`` of the element a species label names.

    Raises rather than returning a sentinel: a label that names no element is an
    input error, and the alternative -- QE's ``atomic_number`` returning 0 and
    the caller checking -- is how a silently missing dispersion coefficient
    would get in.
    """
    symbol = element_symbol(label)
    try:
        return _BY_SYMBOL[symbol.lower()]
    except KeyError:
        raise ValueError(
            f"species label {label!r} does not name an element (read as {symbol!r})"
        ) from None
