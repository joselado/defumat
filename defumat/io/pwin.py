"""Parser for ``pw.x`` input files: Fortran namelists plus QE's cards.

The grammar is small but has several traps that are cheaper to handle once here
than to rediscover later: Fortran logicals (``.true.``/``.t.``), ``d``-exponents
(``1.d-8``), indexed variables (``celldm(1)``), card options written bare, in
braces or in parentheses (``K_POINTS {crystal}``), and comments introduced by
either ``!`` or ``#``.

This layer only tokenises. It does not know what an ``ibrav`` means -- turning an
input into a cell, a structure and a k-point set is
:mod:`defumat.system.builder`'s job, which keeps the parser reusable for the
other QE codes' inputs later.

The reference for what is accepted is ``Doc-QE-7.5/Doc-7.5/INPUT_PW.txt``, and
the defaults are declared in ``Modules/input_parameters.f90``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Card", "PwInput", "read_pw_input", "parse_pw_input"]

#: Card names pw.x understands. Anything else on a bare line is an error rather
#: than silently ignored data.
CARD_NAMES = frozenset(
    {
        "ATOMIC_SPECIES",
        "ATOMIC_POSITIONS",
        "ATOMIC_VELOCITIES",
        "ATOMIC_FORCES",
        "K_POINTS",
        "CELL_PARAMETERS",
        "CONSTRAINTS",
        "OCCUPATIONS",
        "HUBBARD",
        "ADDITIONAL_K_POINTS",
        "SOLVENTS",
        "TOTAL_CHARGE",
        # A defumat extension, and the only card here that pw.x does not have:
        # Elk's per-atom external field (``bfcmt`` in its atoms block), for which
        # QE has no input at all even though ``add_bfield`` can apply one. One
        # line per atom, three cartesian components in Ry.
        "LOCAL_MAGNETIC_FIELDS",
    }
)

#: One ``name = value`` assignment inside a namelist body.
#:
#: The bare-token alternative is *tempered*: it consumes anything that is not a
#: comma, a newline or a comment, **except** where the next thing is another
#: assignment. Fortran's value separator is "a comma or one or more blanks"
#: (F2018 13.11.3), so ``ecutrho = 100.0  nbnd = 8`` is two assignments and QE's
#: own test suite writes it that way (``pw_uspp/uspp-hyb-k.in``). Without the
#: lookahead the first value swallows the second assignment whole, and the
#: failure is *silent* wherever the swallowed text still converts: ``nosym =
#: .true. noinv = .true.`` gave ``nosym`` the string ``'.true. noinv = .true.'``,
#: which is not one of the spellings ``_logical`` accepts, so it read as **False**
#: and the run symmetrised a calculation that had asked it not to.
_ENTRY = re.compile(
    r"""(?P<key>[A-Za-z_]\w*)              # variable name
        (?:\s*\(\s*(?P<index>[\d,\s]+)\s*\))?   # optional (i) or (i,j)
        \s*=\s*
        (?P<value>
            '[^']*'|"[^"]*"                # a quoted string, spaces and all
          | (?:                            # or a bare token, blank-separated
                (?!\s+[A-Za-z_]\w*\s*(?:\(\s*[\d,\s]+\s*\))?\s*=)
                [^,\n!#]
            )+
        )
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Card:
    """One QE card: its name, its option, and its data lines (comments removed)."""

    name: str
    option: str | None
    lines: tuple[str, ...]

    def floats(self) -> list[list[float]]:
        """Every data line as a list of floats. Convenient for numeric cards."""
        return [[fortran_float(tok) for tok in line.split()] for line in self.lines]


@dataclass(frozen=True)
class PwInput:
    """A parsed ``pw.x`` input.

    Namelist names and variable names are lower-cased; card names are
    upper-cased, matching how QE writes them. Indexed variables are stored as
    dicts keyed by their (1-based) index tuple, e.g.
    ``namelists['system']['celldm'] == {(1,): 10.2, (3,): 1.5}``.
    """

    namelists: dict[str, dict[str, Any]] = field(default_factory=dict)
    cards: dict[str, Card] = field(default_factory=dict)
    path: Path | None = None

    def get(self, namelist: str, key: str, default: Any = None) -> Any:
        """Look up a variable, returning ``default`` when it was not given.

        Defaults are the caller's business precisely because QE's defaults are
        context-dependent (``ecutrho`` is ``4*ecutwfc``, ``nbnd`` depends on the
        electron count); silently inventing one here would hide that.
        """
        return self.namelists.get(namelist, {}).get(key, default)

    def indexed(self, namelist: str, key: str, size: int, default: float = 0.0) -> list[float]:
        """An indexed variable as a dense 1-based list of length ``size``."""
        raw = self.get(namelist, key)
        values = [default] * size
        if isinstance(raw, dict):
            for index, value in raw.items():
                position = index[0] - 1
                if not 0 <= position < size:
                    raise ValueError(f"{key}{index} out of range 1..{size}")
                values[position] = float(value)
        elif raw is not None:
            values[0] = float(raw)
        return values

    def card(self, name: str) -> Card | None:
        return self.cards.get(name.upper())

    def require_card(self, name: str) -> Card:
        card = self.card(name)
        if card is None:
            raise ValueError(f"{self.path or 'input'}: missing required card {name}")
        return card


def read_pw_input(path: str | Path) -> PwInput:
    path = Path(path)
    return parse_pw_input(path.read_text(errors="replace"), path=path)


def parse_pw_input(text: str, path: Path | None = None) -> PwInput:
    """Parse the text of a ``pw.x`` input file."""
    namelists: dict[str, dict[str, Any]] = {}
    cards: dict[str, Card] = {}

    lines = [_strip_comment(line) for line in text.splitlines()]

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if line.startswith("&"):
            name, body, i = _collect_namelist(lines, i)
            namelists[name] = _parse_namelist_body(body)
            continue

        card_name = line.split()[0].strip(",").upper()
        if card_name in CARD_NAMES:
            card, i = _collect_card(lines, i, card_name)
            cards[card_name] = card
            continue

        raise ValueError(f"{path or 'input'}: line {i + 1}: unrecognised input {line!r}")

    return PwInput(namelists=namelists, cards=cards, path=path)


def _strip_comment(line: str) -> str:
    """Remove a trailing ``!`` or ``#`` comment, respecting quoted strings."""
    out, quote = [], None
    for char in line:
        if quote:
            out.append(char)
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
            out.append(char)
        elif char in "!#":
            break
        else:
            out.append(char)
    return "".join(out)


def _collect_namelist(lines: list[str], start: int) -> tuple[str, str, int]:
    """Gather a ``&name ... /`` block. The name may be glued to the first entry."""
    head = lines[start].strip()
    match = re.match(r"&\s*(\w+)", head)
    name = match.group(1).lower()
    # Slice by the match, not by searching for the lowered name: inputs write
    # namelists in any case (&SYSTEM, &System), and searching would miss.
    body = [head[match.end() :]]

    i = start
    while i < len(lines):
        stripped = lines[i].strip()
        if i != start:
            body.append(lines[i])
        # A lone '/' closes the namelist; it may also terminate a data line.
        if (i != start and stripped == "/") or (stripped.endswith("/") and stripped != "&" + name):
            body[-1] = body[-1].rstrip().rstrip("/")
            return name, "\n".join(body), i + 1
        i += 1
    raise ValueError(f"namelist &{name} is not terminated by '/'")


def _collect_card(lines: list[str], start: int, name: str) -> tuple[Card, int]:
    """Gather a card's data lines: everything up to the next card or namelist."""
    header = lines[start].strip()
    option = _card_option(header, name)

    data = []
    i = start + 1
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        first = stripped.split()[0].strip(",").upper()
        if first in CARD_NAMES or stripped.startswith("&"):
            break
        data.append(stripped)
        i += 1
    return Card(name=name, option=option, lines=tuple(data)), i


#: The option keywords ``read_cards.f90`` looks for, longest first so that
#: ``crystal_b`` is not read as ``crystal``.
_CARD_OPTIONS = (
    "crystal_sg",
    "crystal_b",
    "crystal_c",
    "tpiba_b",
    "tpiba_c",
    "automatic",
    "angstrom",
    "crystal",
    "tpiba",
    "gamma",
    "bohr",
    "alat",
)


def _card_option(header: str, name: str) -> str | None:
    """``K_POINTS {crystal}``, ``K_POINTS (crystal)`` and ``K_POINTS crystal``."""
    rest = header[len(name) :].strip().strip(",").strip()
    rest = rest.strip("{}()").strip().lower()
    if not rest:
        return None
    if not rest.isidentifier():
        # ``CELL_PARAMETERS (alat=  7.01033620)``, which is how ``pw.x`` writes
        # the relaxed cell it expects to read back. ``read_cards.f90`` picks the
        # option with ``matches()`` -- a *substring* test -- and ignores
        # everything else on the line, including that number.
        for keyword in _CARD_OPTIONS:
            if keyword in rest:
                return keyword
    return rest


def _parse_namelist_body(body: str) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for match in _ENTRY.finditer(body):
        key = match.group("key").lower()
        value = _convert(match.group("value").strip())
        if match.group("index") is None:
            entries[key] = value
        else:
            index = tuple(int(part) for part in match.group("index").replace(" ", "").split(","))
            entries.setdefault(key, {})[index] = value
    return entries


_LOGICAL_TRUE = {".true.", ".t.", "true", ".TRUE."}
_LOGICAL_FALSE = {".false.", ".f.", "false", ".FALSE."}


def fortran_float(token: str) -> float:
    """A Fortran real literal as a Python float: ``1.d-8``, ``2.5D+3``, ``4.3``.

    The cards need it as much as the namelists do -- ``HUBBARD`` writes
    ``U Fe1-3d 1.d-8`` -- and ``float()`` does not accept the ``d`` exponent.
    """
    return float(re.sub(r"[dD]([-+]?\d)", r"e\1", token.strip()))


def _convert(token: str) -> Any:
    """Fortran literal -> Python value."""
    if token[:1] in "'\"" and token[:1] == token[-1:]:
        return token[1:-1]
    lowered = token.lower()
    if lowered in _LOGICAL_TRUE:
        return True
    if lowered in _LOGICAL_FALSE:
        return False
    # Fortran double-precision exponents: 1.d-8, 2.5D+3
    numeric = re.sub(r"[dD]([-+]?\d)", r"e\1", token)
    try:
        return int(numeric)
    except ValueError:
        pass
    try:
        return float(numeric)
    except ValueError:
        return token
